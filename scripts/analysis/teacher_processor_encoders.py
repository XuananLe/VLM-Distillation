from __future__ import annotations

from typing import Any

import torch
import transformers

from src.constants import IGNORE_INDEX, LLAVA_IMAGE_TOKEN
from src.dataset.internvl_utils import (
    INTERNVL_IMG_CONTEXT_TOKEN,
    INTERNVL_IMG_END_TOKEN,
    INTERNVL_IMG_START_TOKEN,
    INTERNVL_NUM_IMAGE_TOKEN,
    build_internvl_pixel_values,
)

EOS_TOKEN = "<end_of_utterance>"


GEMMA3_PROCESSOR = getattr(transformers, "Gemma3Processor", None)
QWEN2_VL_PROCESSOR = getattr(transformers, "Qwen2VLProcessor", None)
QWEN2_5_VL_PROCESSOR = getattr(transformers, "Qwen2_5_VLProcessor", None)


def ignore_mask(length: int) -> torch.Tensor:
    return torch.full((length,), IGNORE_INDEX, dtype=torch.long)


def ones_like_ids(input_ids: torch.Tensor) -> torch.Tensor:
    return torch.ones_like(input_ids, dtype=torch.long)


def qwen_encode_conversation(
    sources,
    images,
    processor: transformers.ProcessorMixin,
) -> dict[str, torch.Tensor]:
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
            raise ValueError("Qwen-VL teacher samples must include image tokens and loaded images.")
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

    return dict(
        input_ids=torch.cat(all_input_ids, dim=0).to(torch.long),
        labels=torch.cat(all_labels, dim=0).to(torch.long),
        attention_mask=torch.cat(all_attention_masks, dim=0).to(torch.long),
        pixel_values=pixel_values,
        pixel_attention_mask=None,
        image_grid_thw=image_grid_thw,
    )


def gemma3_encode_conversation(
    sources,
    images,
    processor: transformers.ProcessorMixin,
) -> dict[str, torch.Tensor]:
    all_input_ids, all_attention_masks, all_labels = [], [], []
    pixel_values = None
    image_idx = 0

    for j in range(0, len(sources), 2):
        user_input = sources[j]
        gpt_response = sources[j + 1]
        user_text = user_input["value"]
        if LLAVA_IMAGE_TOKEN not in user_text or images is None:
            raise ValueError("Gemma 3 teacher samples must include image tokens and loaded images.")
        n_images = user_text.count(LLAVA_IMAGE_TOKEN)
        turn_images = images[image_idx : image_idx + n_images]
        image_idx += n_images
        clean_text = user_text.replace(LLAVA_IMAGE_TOKEN, "").strip()
        user_content = [{"type": "image", "image": image} for image in turn_images]
        if clean_text:
            user_content.append({"type": "text", "text": clean_text})

        prompt_messages = [{"role": "user", "content": user_content}]
        full_messages = prompt_messages + [
            {"role": "assistant", "content": [{"type": "text", "text": gpt_response["value"]}]}
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

        response_ids = full_ids[:, prompt_ids.size(1) :]
        all_input_ids.append(full_ids.squeeze(0).to(torch.long))
        all_attention_masks.append(full_enc.get("attention_mask", ones_like_ids(full_ids)).squeeze(0).to(torch.long))
        all_labels.append(torch.cat([ignore_mask(prompt_ids.size(1)), response_ids.squeeze(0).to(torch.long)], dim=0))
        pixel_values = full_enc.get("pixel_values", prompt_enc.get("pixel_values"))

    return dict(
        input_ids=torch.cat(all_input_ids, dim=0),
        labels=torch.cat(all_labels, dim=0),
        attention_mask=torch.cat(all_attention_masks, dim=0),
        pixel_values=pixel_values,
        pixel_attention_mask=None,
    )


def internvl3_encode_conversation(
    sources,
    images,
    teacher_processor: dict[str, Any],
) -> dict[str, torch.Tensor]:
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
            raise ValueError("InternVL teacher samples must include image tokens and loaded images.")

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


TEACHER_PROCESSOR_ENCODERS = {
    processor_type: encoder
    for processor_type, encoder in (
        (GEMMA3_PROCESSOR, gemma3_encode_conversation),
        (QWEN2_VL_PROCESSOR, qwen_encode_conversation),
        (QWEN2_5_VL_PROCESSOR, qwen_encode_conversation),
    )
    if processor_type is not None
}


def encode_teacher_data(sources, images, teacher_processor) -> dict[str, torch.Tensor]:
    teacher_model_id = teacher_processor.get("model_id") if isinstance(teacher_processor, dict) else None
    if isinstance(teacher_processor, dict):
        if not isinstance(teacher_model_id, str) or "internvl" not in teacher_model_id.lower():
            raise ValueError(f"Unsupported dict teacher processor for {teacher_model_id!r}.")
        teacher_data = internvl3_encode_conversation(sources, images, teacher_processor)
    else:
        encoder = TEACHER_PROCESSOR_ENCODERS.get(type(teacher_processor))
        if encoder is None:
            raise ValueError(f"Unsupported teacher processor type {type(teacher_processor).__name__!r}.")
        teacher_data = encoder(sources, images, teacher_processor)

    if teacher_data["attention_mask"] is None:
        teacher_data["attention_mask"] = torch.ones_like(teacher_data["input_ids"])
    if teacher_data["pixel_values"] is None:
        raise ValueError(f"Teacher encoder did not produce image tensors. teacher_model_id={teacher_model_id!r}")
    return teacher_data


__all__ = [
    "TEACHER_PROCESSOR_ENCODERS",
    "encode_teacher_data",
    "gemma3_encode_conversation",
    "internvl3_encode_conversation",
    "qwen_encode_conversation",
]
