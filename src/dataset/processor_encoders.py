from typing import Dict

import torch
import transformers

from src.constants import IGNORE_INDEX, LLAVA_IMAGE_TOKEN

from .internvl_utils import _build_internvl_pixel_values

EOS_TOKEN = "<end_of_utterance>"
INTERNVL3_1B_MODEL_ID = "OpenGVLab/InternVL3-1B"
INTERNVL3_DUMMY_IMAGE_FLAGS = (1, 1)

IDEFICS3_PROCESSOR = getattr(transformers, "Idefics3Processor", None)
SMOLVLM_PROCESSOR = getattr(transformers, "SmolVLMProcessor", None)
GEMMA3_PROCESSOR = getattr(transformers, "Gemma3Processor", None)
QWEN2_VL_PROCESSOR = getattr(transformers, "Qwen2VLProcessor", None)
QWEN2_5_VL_PROCESSOR = getattr(transformers, "Qwen2_5_VLProcessor", None)
QWEN3_VL_PROCESSOR = getattr(transformers, "Qwen3VLProcessor", None)


def is_internvl_teacher_model_id(model_id: str | None) -> bool:
    return isinstance(model_id, str) and "internvl" in model_id.lower()


def smolvlm_encode_conversation(
    sources,
    images,
    processor: transformers.ProcessorMixin,
) -> Dict[str, torch.Tensor]:
    all_input_ids = [torch.tensor([1])]
    all_labels = [torch.tensor([-100])]

    pixel_values = None
    pixel_attention_mask = None

    for idx, j in enumerate(range(0, len(sources), 2)):
        user_input = sources[j]
        gpt_response = sources[j + 1]
        is_last_turn = idx == (len(sources) // 2 - 1)

        if user_input["content"].startswith(LLAVA_IMAGE_TOKEN):
            user_prompt = f"User:{user_input['content']}{EOS_TOKEN}\nAssistant: "
        else:
            user_prompt = f"User: {user_input['content']}{EOS_TOKEN}\nAssistant: "

        gpt_prompt = (
            f"{gpt_response['content']}{EOS_TOKEN}"
            if is_last_turn
            else f"{gpt_response['content']}{EOS_TOKEN}\n"
        )

        if LLAVA_IMAGE_TOKEN in user_prompt:
            enc = processor(text=user_prompt, images=images, return_tensors="pt")
            prompt_input_ids = enc["input_ids"]
            pixel_values = enc.get("pixel_values", None)
            pixel_attention_mask = enc.get("pixel_attention_mask", None)
        else:
            prompt_input_ids = processor.tokenizer(
                user_prompt,
                add_special_tokens=False,
                return_tensors="pt",
            )["input_ids"]

        response_input_ids = processor.tokenizer(
            gpt_prompt,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"]

        input_ids = torch.cat([prompt_input_ids, response_input_ids], dim=1).squeeze(0)
        labels = torch.cat(
            [
                torch.tensor([IGNORE_INDEX] * len(prompt_input_ids[0])),
                response_input_ids.squeeze(0),
            ],
            dim=0,
        )
        all_input_ids.append(input_ids)
        all_labels.append(labels)

    input_ids = torch.cat(all_input_ids, dim=0).to(torch.long)
    labels = torch.cat(all_labels, dim=0).to(torch.long)
    attention_mask = torch.ones_like(input_ids)

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
    all_labels = []

    pixel_values = None
    image_grid_thw = None
    image_idx = 0

    for idx, j in enumerate(range(0, len(sources), 2)):
        user_input = sources[j]
        gpt_response = sources[j + 1]
        is_last_turn = idx == (len(sources) // 2 - 1)

        user_text = user_input["content"]
        has_image = LLAVA_IMAGE_TOKEN in user_text and images is not None
        n_images = user_text.count(LLAVA_IMAGE_TOKEN)
        clean_text = user_text.replace(LLAVA_IMAGE_TOKEN, "").strip()

        if has_image:
            turn_images = images[image_idx:image_idx + n_images]
            image_idx += n_images
            user_content = [{"type": "image"}] * n_images + [{"type": "text", "text": clean_text}]
        else:
            turn_images = None
            user_content = clean_text

        prompt_text = processor.apply_chat_template(
            [{"role": "user", "content": user_content}],
            tokenize=False,
            add_generation_prompt=True,
        )
        if has_image:
            enc = processor(text=[prompt_text], images=turn_images, return_tensors="pt")
            prompt_ids = enc["input_ids"]
            pixel_values = enc.get("pixel_values", None)
            image_grid_thw = enc.get("image_grid_thw", None)
        else:
            prompt_ids = processor.tokenizer(
                prompt_text,
                add_special_tokens=False,
                return_tensors="pt",
            )["input_ids"]

        suffix = "" if is_last_turn else "\n"
        response_text = gpt_response["content"] + "<|im_end|>" + suffix
        response_ids = processor.tokenizer(
            response_text,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"]

        input_ids = torch.cat([prompt_ids, response_ids], dim=1).squeeze(0)
        labels = torch.cat(
            [
                torch.tensor([IGNORE_INDEX] * len(prompt_ids[0])),
                response_ids.squeeze(0),
            ],
            dim=0,
        )
        all_input_ids.append(input_ids)
        all_labels.append(labels)

    input_ids = torch.cat(all_input_ids, dim=0).to(torch.long)
    labels = torch.cat(all_labels, dim=0).to(torch.long)
    attention_mask = torch.ones_like(input_ids)

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
    all_input_ids = []
    all_labels = []

    pixel_values = None
    image_idx = 0

    for j in range(0, len(sources), 2):
        user_input = sources[j]
        gpt_response = sources[j + 1]

        user_text = user_input["content"]
        has_image = LLAVA_IMAGE_TOKEN in user_text and images is not None
        n_images = user_text.count(LLAVA_IMAGE_TOKEN)
        clean_text = user_text.replace(LLAVA_IMAGE_TOKEN, "").strip()

        user_content = []
        if has_image:
            turn_images = images[image_idx:image_idx + n_images]
            image_idx += n_images
            user_content.extend({"type": "image", "image": image} for image in turn_images)

        if clean_text:
            user_content.append({"type": "text", "text": clean_text})
        if not user_content:
            user_content = [{"type": "text", "text": ""}]

        prompt_messages = [{"role": "user", "content": user_content}]
        full_messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": [{"type": "text", "text": gpt_response["content"]}]},
        ]

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
            raise ValueError("Gemma 3 prompt encoding is longer than full conversation encoding.")

        response_ids = full_ids[:, prompt_ids.size(1):]
        input_ids = full_ids.squeeze(0)
        labels = torch.cat(
            [
                torch.full((prompt_ids.size(1),), IGNORE_INDEX, dtype=torch.long),
                response_ids.squeeze(0).to(torch.long),
            ],
            dim=0,
        )

        all_input_ids.append(input_ids.to(torch.long))
        all_labels.append(labels)

        if has_image:
            pixel_values = full_enc.get("pixel_values", prompt_enc.get("pixel_values"))

    input_ids = torch.cat(all_input_ids, dim=0).to(torch.long)
    labels = torch.cat(all_labels, dim=0).to(torch.long)
    attention_mask = torch.ones_like(input_ids)

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
    num_image_token = teacher_processor.get("num_image_token", 256)
    img_start_token = teacher_processor.get("img_start_token", "<img>")
    img_end_token = teacher_processor.get("img_end_token", "</img>")
    img_context_token = teacher_processor.get("img_context_token", "<IMG_CONTEXT>")

    all_input_ids = [torch.tensor([tokenizer.bos_token_id or 1])]
    all_labels = [torch.tensor([IGNORE_INDEX])]
    pixel_values = None
    image_flags = None
    image_idx = 0

    for idx, j in enumerate(range(0, len(sources), 2)):
        user_input = sources[j]
        gpt_response = sources[j + 1]
        is_last_turn = idx == (len(sources) // 2 - 1)

        if user_input["content"].startswith(LLAVA_IMAGE_TOKEN):
            user_prompt = f"User:{user_input['content']}{EOS_TOKEN}\nAssistant: "
        else:
            user_prompt = f"User: {user_input['content']}{EOS_TOKEN}\nAssistant: "

        gpt_prompt = (
            f"{gpt_response['content']}{EOS_TOKEN}"
            if is_last_turn
            else f"{gpt_response['content']}{EOS_TOKEN}\n"
        )

        if LLAVA_IMAGE_TOKEN in user_prompt and images is not None:
            turn_images = images[image_idx:image_idx + user_input["content"].count(LLAVA_IMAGE_TOKEN)]
            image_idx += len(turn_images)
            pixel_value_chunks = []
            for turn_image in turn_images:
                turn_pixel_values = _build_internvl_pixel_values(turn_image, teacher_processor)
                pixel_value_chunks.append(turn_pixel_values)
                image_tokens = (
                    img_start_token
                    + img_context_token * (num_image_token * turn_pixel_values.shape[0])
                    + img_end_token
                )
                user_prompt = user_prompt.replace(LLAVA_IMAGE_TOKEN, image_tokens, 1)
            if pixel_value_chunks:
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
        labels = torch.cat(
            [
                torch.tensor([IGNORE_INDEX] * len(prompt_input_ids[0])),
                response_input_ids.squeeze(0),
            ],
            dim=0,
        )
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
        (QWEN3_VL_PROCESSOR, qwen_encode_conversation),
    )
    if processor_type is not None
}

QWEN_PROCESSORS = {
    processor_type
    for processor_type in (
        QWEN2_VL_PROCESSOR,
        QWEN2_5_VL_PROCESSOR,
        QWEN3_VL_PROCESSOR,
    )
    if processor_type is not None
}

DICT_TEACHER_ENCODERS = {
    INTERNVL3_1B_MODEL_ID: internvl3_encode_conversation,
}


__all__ = [
    "DICT_TEACHER_ENCODERS",
    "INTERNVL3_1B_MODEL_ID",
    "INTERNVL3_DUMMY_IMAGE_FLAGS",
    "PROCESSOR_ENCODERS",
    "QWEN_PROCESSORS",
    "gemma3_encode_conversation",
    "internvl3_encode_conversation",
    "is_internvl_teacher_model_id",
    "qwen_encode_conversation",
    "smolvlm_encode_conversation",
]
