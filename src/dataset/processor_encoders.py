from typing import Dict

import torch
import transformers

from src.constants import IGNORE_INDEX, LLAVA_IMAGE_TOKEN

from .internvl_utils import (
    INTERNVL_IMG_CONTEXT_TOKEN,
    INTERNVL_IMG_END_TOKEN,
    INTERNVL_IMG_START_TOKEN,
    INTERNVL_NUM_IMAGE_TOKEN,
    build_internvl_pixel_values,
)

EOS_TOKEN = "<end_of_utterance>"

IDEFICS3_PROCESSOR = getattr(transformers, "Idefics3Processor", None)
SMOLVLM_PROCESSOR = getattr(transformers, "SmolVLMProcessor", None)
GEMMA3_PROCESSOR = getattr(transformers, "Gemma3Processor", None)
QWEN2_VL_PROCESSOR = getattr(transformers, "Qwen2VLProcessor", None)
QWEN2_5_VL_PROCESSOR = getattr(transformers, "Qwen2_5_VLProcessor", None)


def is_internvl_teacher_model_id(model_id: str | None) -> bool:
    return isinstance(model_id, str) and "internvl" in model_id.lower()


def ignore_mask(length: int) -> torch.Tensor:
    return torch.full((length,), IGNORE_INDEX, dtype=torch.long)


def ones_like_ids(input_ids: torch.Tensor) -> torch.Tensor:
    return torch.ones_like(input_ids, dtype=torch.long)


def build_image_text_content(user_text: str, turn_images: list) -> list:
    clean_text = user_text.replace(LLAVA_IMAGE_TOKEN, "").strip()
    content = [{"type": "image", "image": image} for image in turn_images]
    if clean_text:
        content.append({"type": "text", "text": clean_text})
    return content


def encode_turn_with_template(
    processor: transformers.ProcessorMixin,
    user_content: list,
    response_value: str,
) -> tuple:
    """Shared per-turn encoding for apply_chat_template-based processors."""
    prompt_messages = [{"role": "user", "content": user_content}]
    full_messages = prompt_messages + [
        {"role": "assistant", "content": [{"type": "text", "text": response_value}]}
    ]
    prompt_enc = processor.apply_chat_template(
        prompt_messages, tokenize=True, return_dict=True, return_tensors="pt", add_generation_prompt=True
    )
    full_enc = processor.apply_chat_template(
        full_messages, tokenize=True, return_dict=True, return_tensors="pt", add_generation_prompt=False
    )
    prompt_ids = prompt_enc["input_ids"]
    full_ids = full_enc["input_ids"]
    if prompt_ids.size(1) > full_ids.size(1):
        raise ValueError("Prompt encoding is longer than full conversation encoding.")
    response_ids = full_ids[:, prompt_ids.size(1):]
    input_ids = full_ids.squeeze(0).to(torch.long)
    attention_mask = full_enc.get("attention_mask", ones_like_ids(full_ids)).squeeze(0).to(torch.long)
    labels = torch.cat([ignore_mask(prompt_ids.size(1)), response_ids.squeeze(0).to(torch.long)], dim=0)
    return input_ids, attention_mask, labels, prompt_enc, full_enc


def build_smolvlm_user_content(user_text: str, turn_images) -> list[dict]:
    pieces = user_text.split(LLAVA_IMAGE_TOKEN)
    expected_image_count = len(pieces) - 1
    if expected_image_count == 0 or len(turn_images) != expected_image_count:
        raise ValueError(
            "SmolVLM training samples must include one loaded image per image token. "
            f"image_tokens={expected_image_count}, loaded_images={len(turn_images)}"
        )

    content = []
    for piece_index, text_piece in enumerate(pieces):
        text = text_piece.strip()
        if text:
            content.append({"type": "text", "text": text})
        if piece_index < expected_image_count:
            content.append({"type": "image", "image": turn_images[piece_index]})
    return content


def smolvlm_encode_conversation(
    sources,
    images,
    processor: transformers.ProcessorMixin,
) -> Dict[str, torch.Tensor]:
    all_input_ids = []
    all_attention_masks = []
    all_labels = []
    pixel_values = None
    pixel_attention_mask = None
    image_idx = 0
    previous_full_length = 0
    messages = []

    for j in range(0, len(sources), 2):
        user_input = sources[j]
        gpt_response = sources[j + 1]

        user_text = user_input["value"]
        if LLAVA_IMAGE_TOKEN not in user_text or images is None:
            raise ValueError("SmolVLM training samples must include image tokens and loaded images.")
        image_count = user_text.count(LLAVA_IMAGE_TOKEN)
        turn_images = images[image_idx : image_idx + image_count]
        image_idx += image_count

        user_message = {
            "role": "user",
            "content": build_smolvlm_user_content(user_text, turn_images),
        }
        assistant_message = {
            "role": "assistant",
            "content": [{"type": "text", "text": gpt_response["value"]}],
        }
        prompt_messages = messages + [user_message]
        full_messages = prompt_messages + [assistant_message]

        prompt_enc = processor.apply_chat_template(
            prompt_messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        )
        full_enc = processor.apply_chat_template(
            full_messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=False,
        )

        prompt_ids = prompt_enc["input_ids"]
        full_ids = full_enc["input_ids"]
        if prompt_ids.size(1) > full_ids.size(1):
            raise ValueError("SmolVLM prompt encoding is longer than full conversation encoding.")
        if previous_full_length > prompt_ids.size(1):
            raise ValueError("SmolVLM conversation encoding shortened between turns.")

        prompt_delta = prompt_ids[:, previous_full_length:]
        response_ids = full_ids[:, prompt_ids.size(1) :]
        prompt_attention_mask = prompt_enc.get("attention_mask", ones_like_ids(prompt_ids))
        full_attention_mask = full_enc.get("attention_mask", ones_like_ids(full_ids))
        prompt_delta_attention_mask = prompt_attention_mask[:, previous_full_length:]
        response_attention_mask = full_attention_mask[:, prompt_ids.size(1) :]

        if prompt_delta.numel() > 0:
            all_input_ids.append(prompt_delta.squeeze(0).to(torch.long))
            all_attention_masks.append(prompt_delta_attention_mask.squeeze(0).to(torch.long))
            all_labels.append(
                torch.full(
                    (prompt_delta.size(1),),
                    IGNORE_INDEX,
                    dtype=torch.long,
                )
            )
        if response_ids.numel() > 0:
            all_input_ids.append(response_ids.squeeze(0).to(torch.long))
            all_attention_masks.append(response_attention_mask.squeeze(0).to(torch.long))
            all_labels.append(response_ids.squeeze(0).to(torch.long))

        previous_full_length = full_ids.size(1)
        messages = full_messages
        pixel_values = full_enc.get("pixel_values", prompt_enc.get("pixel_values"))
        pixel_attention_mask = full_enc.get(
            "pixel_attention_mask",
            prompt_enc.get("pixel_attention_mask"),
        )

    input_ids = torch.cat(all_input_ids, dim=0).to(torch.long)
    attention_mask = torch.cat(all_attention_masks, dim=0).to(torch.long)
    labels = torch.cat(all_labels, dim=0).to(torch.long)

    return dict(
        input_ids=input_ids,
        labels=labels,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        pixel_attention_mask=pixel_attention_mask,
    )


def qwen_encode_conversation(
    sources,
    images,
    processor: transformers.ProcessorMixin,
) -> Dict[str, torch.Tensor]:
    all_input_ids = []
    all_attention_masks = []
    all_labels = []

    pixel_values = None
    image_grid_thw = None
    image_idx = 0

    for idx, j in enumerate(range(0, len(sources), 2)):
        user_input = sources[j]
        gpt_response = sources[j + 1]
        is_last_turn = idx == (len(sources) // 2 - 1)

        user_text = user_input["value"]
        if LLAVA_IMAGE_TOKEN not in user_text or images is None:
            raise ValueError("Qwen-VL training samples must include image tokens and loaded images.")
        n_images = user_text.count(LLAVA_IMAGE_TOKEN)
        clean_text = user_text.replace(LLAVA_IMAGE_TOKEN, "").strip()

        turn_images = images[image_idx : image_idx + n_images]
        image_idx += n_images
        user_content = [{"type": "image"}] * n_images + [{"type": "text", "text": clean_text}]

        prompt_text = processor.apply_chat_template(
            [{"role": "user", "content": user_content}],
            tokenize=False,
            add_generation_prompt=True,
        )
        enc = processor(text=[prompt_text], images=turn_images, return_tensors="pt")
        prompt_ids = enc["input_ids"]
        prompt_attention_mask = enc.get("attention_mask", ones_like_ids(prompt_ids))
        pixel_values = enc.get("pixel_values", None)
        image_grid_thw = enc.get("image_grid_thw", None)

        suffix = "" if is_last_turn else "\n"
        response_text = gpt_response["value"] + "<|im_end|>" + suffix
        response_enc = processor.tokenizer(
            response_text,
            add_special_tokens=False,
            return_tensors="pt",
        )
        response_ids = response_enc["input_ids"]
        response_attention_mask = response_enc.get("attention_mask", ones_like_ids(response_ids))

        input_ids = torch.cat([prompt_ids, response_ids], dim=1).squeeze(0)
        attention_mask = torch.cat([prompt_attention_mask, response_attention_mask], dim=1).squeeze(0)
        labels = torch.cat([ignore_mask(prompt_ids.size(1)), response_ids.squeeze(0).to(torch.long)], dim=0)
        all_input_ids.append(input_ids)
        all_attention_masks.append(attention_mask)
        all_labels.append(labels)

    input_ids = torch.cat(all_input_ids, dim=0).to(torch.long)
    attention_mask = torch.cat(all_attention_masks, dim=0).to(torch.long)
    labels = torch.cat(all_labels, dim=0).to(torch.long)

    return dict(
        input_ids=input_ids,
        labels=labels,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        pixel_attention_mask=None,
        image_grid_thw=image_grid_thw,
    )


def gemma3_encode_conversation(
    sources,
    images,
    processor: transformers.ProcessorMixin,
) -> Dict[str, torch.Tensor]:
    """Encode one conversation with Gemma 3 by masking prompt tokens out of the full chat encoding."""
    all_input_ids, all_attention_masks, all_labels = [], [], []
    pixel_values = None
    image_idx = 0

    for j in range(0, len(sources), 2):
        user_input = sources[j]
        gpt_response = sources[j + 1]
        user_text = user_input["value"]
        if LLAVA_IMAGE_TOKEN not in user_text or images is None:
            raise ValueError("Gemma 3 training samples must include image tokens and loaded images.")
        n_images = user_text.count(LLAVA_IMAGE_TOKEN)
        turn_images = images[image_idx : image_idx + n_images]
        image_idx += n_images
        user_content = build_image_text_content(user_text, turn_images)
        input_ids, attention_mask, labels, prompt_enc, full_enc = encode_turn_with_template(
            processor, user_content, gpt_response["value"]
        )
        all_input_ids.append(input_ids)
        all_attention_masks.append(attention_mask)
        all_labels.append(labels)
        pixel_values = full_enc.get("pixel_values", prompt_enc.get("pixel_values"))

    input_ids = torch.cat(all_input_ids, dim=0)
    attention_mask = torch.cat(all_attention_masks, dim=0)
    labels = torch.cat(all_labels, dim=0)
    return dict(
        input_ids=input_ids,
        labels=labels,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        pixel_attention_mask=None,
    )


def internvl3_encode_conversation(
    sources,
    images,
    teacher_processor: dict,
) -> Dict[str, torch.Tensor]:
    tokenizer = teacher_processor["tokenizer"]
    num_image_token = teacher_processor.get("num_image_token", INTERNVL_NUM_IMAGE_TOKEN)
    img_start_token = teacher_processor.get("img_start_token", INTERNVL_IMG_START_TOKEN)
    img_end_token = teacher_processor.get("img_end_token", INTERNVL_IMG_END_TOKEN)
    img_context_token = teacher_processor.get("img_context_token", INTERNVL_IMG_CONTEXT_TOKEN)

    all_input_ids = [torch.tensor([tokenizer.bos_token_id or 1])]
    all_labels = [torch.tensor([IGNORE_INDEX])]
    pixel_values = None
    image_flags = None
    image_idx = 0

    for idx, j in enumerate(range(0, len(sources), 2)):
        user_input = sources[j]
        gpt_response = sources[j + 1]
        is_last_turn = idx == (len(sources) // 2 - 1)

        if user_input["value"].startswith(LLAVA_IMAGE_TOKEN):
            user_prompt = f"User:{user_input['value']}{EOS_TOKEN}\nAssistant: "
        else:
            user_prompt = f"User: {user_input['value']}{EOS_TOKEN}\nAssistant: "

        gpt_prompt = f"{gpt_response['value']}{EOS_TOKEN}" if is_last_turn else f"{gpt_response['value']}{EOS_TOKEN}\n"

        if LLAVA_IMAGE_TOKEN not in user_prompt or images is None:
            raise ValueError("InternVL training samples must include image tokens and loaded images.")

        turn_images = images[image_idx : image_idx + user_input["value"].count(LLAVA_IMAGE_TOKEN)]
        image_idx += len(turn_images)
        pixel_value_chunks = []
        for turn_image in turn_images:
            turn_pixel_values = build_internvl_pixel_values(turn_image, teacher_processor)
            pixel_value_chunks.append(turn_pixel_values)
            image_tokens = (
                img_start_token + img_context_token * (num_image_token * turn_pixel_values.shape[0]) + img_end_token
            )
            user_prompt = user_prompt.replace(LLAVA_IMAGE_TOKEN, image_tokens, 1)
        pixel_values = torch.cat(pixel_value_chunks, dim=0)
        image_flags = torch.ones((pixel_values.shape[0], 1), dtype=torch.long)

        prompt_input_ids = tokenizer(
            user_prompt,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"]
        response_input_ids = tokenizer(
            gpt_prompt,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"]

        input_ids = torch.cat([prompt_input_ids, response_input_ids], dim=1).squeeze(0)
        labels = torch.cat([ignore_mask(prompt_input_ids.size(1)), response_input_ids.squeeze(0).to(torch.long)], dim=0)
        all_input_ids.append(input_ids)
        all_labels.append(labels)

    return dict(
        input_ids=torch.cat(all_input_ids, dim=0).to(torch.long),
        labels=torch.cat(all_labels, dim=0).to(torch.long),
        attention_mask=None,
        pixel_values=pixel_values,
        pixel_attention_mask=None,
        image_flags=image_flags,
    )


PROCESSOR_ENCODERS = {
    processor_type: encoder
    for processor_type, encoder in (
        (IDEFICS3_PROCESSOR, smolvlm_encode_conversation),
        (SMOLVLM_PROCESSOR, smolvlm_encode_conversation),
        (GEMMA3_PROCESSOR, gemma3_encode_conversation),
        (QWEN2_VL_PROCESSOR, qwen_encode_conversation),
        (QWEN2_5_VL_PROCESSOR, qwen_encode_conversation),
    )
    if processor_type is not None
}

__all__ = [
    "PROCESSOR_ENCODERS",
    "gemma3_encode_conversation",
    "internvl3_encode_conversation",
    "is_internvl_teacher_model_id",
    "qwen_encode_conversation",
    "smolvlm_encode_conversation",
]
