import os
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import transformers
import ujson as json
from PIL import Image
from torch.utils.data import Dataset

from src.params import DataArguments

from .data_collator import DataCollatorForSupervisedDataset
from .smolvlm_encoder import smolvlm_encode_conversation
from .teacher_logits_cache import TeacherLogitsCache

# One encoded sample contains processor-owned multimodal tensors plus optional
# cached teacher logits used by offline distillation.
# {
#     "input_ids": tensor(...),
#     "labels": tensor(...),
#     "attention_mask": tensor(...),
#     "pixel_values": tensor(...),
#     "pixel_attention_mask": tensor(...),  # when returned by the processor
#     "teacher_0_cached_logits": tensor(...),
#     "teacher_0_cached_labels": tensor(...),
# }


class SupervisedDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        processor: transformers.ProcessorMixin,
        data_args: DataArguments,
        teacher_logits_cache_dir: Optional[str] = None,
        teacher_model_ids: Optional[list[str]] = None,
    ):
        super(SupervisedDataset, self).__init__()
        with open(data_path, "r") as data_file:
            self.training_records = json.load(data_file)

        self.processor = processor
        self.data_args = data_args
        self.teacher_logits_cache = None
        if teacher_logits_cache_dir is not None:
            if not teacher_model_ids:
                raise ValueError("teacher_model_ids must be provided when using cached teacher logits.")
            self.teacher_logits_cache = TeacherLogitsCache(
                cache_dir=teacher_logits_cache_dir,
                teacher_model_ids=teacher_model_ids,
            )

        self.teacher_count = self.teacher_logits_cache.teacher_count if self.teacher_logits_cache is not None else 0

    def __len__(self):
        return len(self.training_records)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.training_records[i]
        image_files = sources["image"]
        image_folder = self.data_args.image_folder
        if isinstance(image_files, str):
            image_files = [image_files]

        images = []
        for image_file in image_files:
            resolved_path = image_file
            if not os.path.exists(resolved_path):
                resolved_path = os.path.join(image_folder, image_file)
            images.append(Image.open(resolved_path).convert("RGB"))

        sources = sources["conversations"]

        encoded_sample = smolvlm_encode_conversation(
            sources,
            images,
            self.processor,
        )
        if encoded_sample["pixel_values"] is None:
            raise ValueError("Student encoder did not produce image tensors for an image-only sample.")

        if self.teacher_logits_cache is not None:
            for teacher_index in range(self.teacher_count):
                cache_sample = self.teacher_logits_cache.load_sample(teacher_index, i)
                prefix = f"teacher_{teacher_index}"
                encoded_sample[f"{prefix}_cached_logits"] = cache_sample["logits"]
                encoded_sample[f"{prefix}_cached_labels"] = cache_sample["labels"]

        return encoded_sample


def make_supervised_data_module(
    processor,
    data_args,
    teacher_model_ids: Optional[list[str]] = None,
    teacher_logits_cache_dir: Optional[str] = None,
):
    supervised_dataset = SupervisedDataset(
        data_path=data_args.data_path,
        processor=processor,
        data_args=data_args,
        teacher_logits_cache_dir=teacher_logits_cache_dir,
        teacher_model_ids=teacher_model_ids,
    )
    data_collator = DataCollatorForSupervisedDataset(
        pad_token_id=processor.tokenizer.pad_token_id,
    )

    return dict(train_dataset=supervised_dataset, data_collator=data_collator)
