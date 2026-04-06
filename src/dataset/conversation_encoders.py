from typing import Callable, Dict, Optional

import torch
import transformers

from .processor_encoders import (
    DICT_TEACHER_ENCODERS,
    INTERNVL3_DUMMY_IMAGE_FLAGS,
    PROCESSOR_ENCODERS,
    QWEN_PROCESSORS,
    internvl3_encode_conversation,
    is_internvl_teacher_model_id,
)


def encode_with_processor(
    sources,
    images,
    processor: transformers.ProcessorMixin,
    role: str,
) -> Dict[str, torch.Tensor]:
    processor_type = type(processor)
    encoder = PROCESSOR_ENCODERS.get(processor_type)
    if encoder is None:
        raise ValueError(f"Unsupported {role} processor type {processor_type.__name__!r}.")
    return encoder(sources, images, processor)


def encode_student_data(
    sources,
    images,
    processor: transformers.ProcessorMixin,
) -> Dict[str, torch.Tensor]:
    return encode_with_processor(sources, images, processor, role="student")


def _finalize_teacher_data(
    teacher_data: Dict[str, torch.Tensor],
    teacher_processor,
    teacher_model_id: Optional[str],
    dummy_pixel_tensors: Callable[[], tuple[torch.Tensor, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    if teacher_data["attention_mask"] is None:
        teacher_data["attention_mask"] = torch.ones_like(teacher_data["input_ids"])

    if teacher_data["pixel_values"] is not None:
        return teacher_data

    if is_internvl_teacher_model_id(teacher_model_id):
        image_size = teacher_processor.get("image_size", 448)
        teacher_data["pixel_values"] = torch.zeros((1, 3, image_size, image_size))
        teacher_data["image_flags"] = torch.zeros(INTERNVL3_DUMMY_IMAGE_FLAGS, dtype=torch.long)
        return teacher_data

    if type(teacher_processor) not in QWEN_PROCESSORS:
        pixel_values, pixel_attention_mask = dummy_pixel_tensors()
        teacher_data["pixel_values"] = pixel_values
        teacher_data["pixel_attention_mask"] = pixel_attention_mask

    return teacher_data


def encode_teacher_data(
    sources,
    images,
    teacher_processor,
    dummy_pixel_tensors: Callable[[], tuple[torch.Tensor, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    teacher_model_id = teacher_processor.get("model_id") if isinstance(teacher_processor, dict) else None
    if isinstance(teacher_processor, dict):
        encoder = (
            internvl3_encode_conversation
            if is_internvl_teacher_model_id(teacher_model_id)
            else DICT_TEACHER_ENCODERS.get(teacher_model_id)
        )
        if encoder is None:
            raise ValueError(f"Unsupported dict teacher processor for {teacher_model_id!r}.")
        teacher_data = encoder(sources, images, teacher_processor)
    else:
        teacher_data = encode_with_processor(sources, images, teacher_processor, role="teacher")

    return _finalize_teacher_data(
        teacher_data,
        teacher_processor,
        teacher_model_id,
        dummy_pixel_tensors,
    )


__all__ = [
    "encode_student_data",
    "encode_teacher_data",
]
