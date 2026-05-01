from typing import Dict, Optional

import torch
import transformers

from .processor_encoders import (
    PROCESSOR_ENCODERS,
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


def finalize_teacher_data(
    teacher_data: Dict[str, torch.Tensor],
    teacher_model_id: Optional[str],
) -> Dict[str, torch.Tensor]:
    """Fill teacher-side defaults so every teacher batch exposes the expected multimodal fields."""
    if teacher_data["attention_mask"] is None:
        teacher_data["attention_mask"] = torch.ones_like(teacher_data["input_ids"])

    if teacher_data["pixel_values"] is not None:
        return teacher_data

    raise ValueError(
        "Teacher encoder did not produce image tensors. "
        f"teacher_model_id={teacher_model_id!r}"
    )


def encode_teacher_data(
    sources,
    images,
    teacher_processor,
) -> Dict[str, torch.Tensor]:
    """Encode one sample for a live teacher and normalize its optional multimodal fields."""
    teacher_model_id = teacher_processor.get("model_id") if isinstance(teacher_processor, dict) else None
    if isinstance(teacher_processor, dict):
        if not is_internvl_teacher_model_id(teacher_model_id):
            raise ValueError(f"Unsupported dict teacher processor for {teacher_model_id!r}.")
        teacher_data = internvl3_encode_conversation(sources, images, teacher_processor)
    else:
        teacher_data = encode_with_processor(sources, images, teacher_processor, role="teacher")

    return finalize_teacher_data(
        teacher_data,
        teacher_model_id,
    )


__all__ = [
    "encode_with_processor",
    "encode_teacher_data",
]
