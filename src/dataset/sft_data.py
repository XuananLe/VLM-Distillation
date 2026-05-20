import os
from dataclasses import replace
from typing import Dict, Optional

import torch
import transformers
import ujson as json
from PIL import Image
from torch.utils.data import Dataset

from src.params import DataArguments

from .conversation_encoders import (
    encode_teacher_data,
    encode_with_processor,
)
from .data_collator import DataCollatorForSupervisedDataset
from .teacher_logits_cache import TeacherLogitsCache

# This return a sample
# {
#       "input_ids":tensor([...]                                                    ),
#       "labels": tensor([...]),
#       "attention_mask": tensor([...]),
#       "pixel_values": tensor(...),
#       "pixel_attention_mask": tensor(...),
#       "teacher_0_cached_logits": tensor(...),
#       "teacher_0_cached_labels": tensor(...),
# }


class SupervisedDataset(Dataset):
    def __init__(
        self,
        data_path: str | list,
        processor: transformers.ProcessorMixin,
        data_args: DataArguments,
        teacher_processors: Optional[list[transformers.ProcessorMixin]] = None,
        teacher_logits_cache_dir: Optional[str] = None,
        teacher_model_ids: Optional[list[str]] = None,
    ):
        super(SupervisedDataset, self).__init__()
        if isinstance(data_path, str):
            training_records = json.load(open(data_path, "r"))
        else:
            training_records = data_path

        self.processor = processor
        self.teacher_processors = list(teacher_processors or [])
        self.training_records = training_records
        self.data_args = data_args
        self.teacher_logits_cache = None
        if teacher_logits_cache_dir is not None:
            if not teacher_model_ids:
                raise ValueError("teacher_model_ids must be provided when using cached teacher logits.")
            self.teacher_logits_cache = TeacherLogitsCache(
                cache_dir=teacher_logits_cache_dir,
                teacher_model_ids=teacher_model_ids,
                expected_num_samples=len(self.training_records),
            )

        processor_teacher_count = len(self.teacher_processors)
        cache_teacher_count = self.teacher_logits_cache.teacher_count if self.teacher_logits_cache is not None else 0
        if processor_teacher_count and cache_teacher_count and processor_teacher_count != cache_teacher_count:
            raise ValueError(
                "Teacher processor count does not match the teacher-logits cache count. "
                f"processors={processor_teacher_count}, cache={cache_teacher_count}"
            )
        self.teacher_count = max(processor_teacher_count, cache_teacher_count)

    def __len__(self):
        return len(self.training_records)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.training_records[i]
        images = None

        if "image" in sources:
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

        encoded_sample = encode_with_processor(
            sources,
            images,
            self.processor,
            role="student",
        )
        if encoded_sample["pixel_values"] is None:
            raise ValueError("Student encoder did not produce image tensors for an image-only sample.")

        if self.teacher_logits_cache is not None:
            for teacher_index in range(self.teacher_count):
                cache_sample = self.teacher_logits_cache.load_sample(teacher_index, i)
                prefix = "teacher" if self.teacher_count == 1 else f"teacher_{teacher_index}"
                encoded_sample[f"{prefix}_cached_logits"] = cache_sample["logits"]
                encoded_sample[f"{prefix}_cached_labels"] = cache_sample["labels"]

        if not self.teacher_processors:
            return encoded_sample

        teacher_count = self.teacher_count
        for teacher_index, teacher_processor in enumerate(self.teacher_processors):
            teacher_data = encode_teacher_data(sources, images, teacher_processor)
            prefix = "teacher" if teacher_count == 1 else f"teacher_{teacher_index}"

            encoded_sample[f"{prefix}_input_ids"] = teacher_data["input_ids"]
            encoded_sample[f"{prefix}_labels"] = teacher_data["labels"]
            encoded_sample[f"{prefix}_attention_mask"] = teacher_data["attention_mask"]
            encoded_sample[f"{prefix}_pixel_values"] = teacher_data["pixel_values"]
            encoded_sample[f"{prefix}_pixel_attention_mask"] = teacher_data["pixel_attention_mask"]
            if teacher_data.get("image_sizes") is not None:
                encoded_sample[f"{prefix}_image_sizes"] = teacher_data["image_sizes"]
            if teacher_data.get("image_grid_thw") is not None:
                encoded_sample[f"{prefix}_image_grid_thw"] = teacher_data["image_grid_thw"]
            if teacher_data.get("image_flags") is not None:
                encoded_sample[f"{prefix}_image_flags"] = teacher_data["image_flags"]

        return encoded_sample


def make_supervised_data_module(
    processor,
    data_args,
    teacher_processors: Optional[list[transformers.ProcessorMixin]] = None,
    teacher_model_ids: Optional[list[str]] = None,
    teacher_logits_cache_dir: Optional[str] = None,
):
    normalized_teacher_processors = list(teacher_processors or [])
    sft_dataset = SupervisedDataset(
        data_path=data_args.data_path,
        processor=processor,
        data_args=data_args,
        teacher_processors=normalized_teacher_processors,
        teacher_logits_cache_dir=teacher_logits_cache_dir,
        teacher_model_ids=teacher_model_ids,
    )
    eval_dataset = None
    if data_args.eval_data_path:
        eval_dataset = SupervisedDataset(
            data_path=data_args.eval_data_path,
            processor=processor,
            data_args=replace(data_args, data_path=data_args.eval_data_path),
            teacher_processors=normalized_teacher_processors,
            teacher_model_ids=teacher_model_ids,
        )
    teacher_pad = None
    if len(normalized_teacher_processors) == 1:
        if isinstance(normalized_teacher_processors[0], dict):
            teacher_pad = normalized_teacher_processors[0]["tokenizer"].pad_token_id
        elif hasattr(normalized_teacher_processors[0], "tokenizer"):
            teacher_pad = normalized_teacher_processors[0].tokenizer.pad_token_id
        else:
            teacher_pad = normalized_teacher_processors[0].pad_token_id
    teacher_pad_ids = []
    for teacher_processor in normalized_teacher_processors:
        if isinstance(teacher_processor, dict):
            teacher_pad_ids.append(teacher_processor["tokenizer"].pad_token_id)
        elif hasattr(teacher_processor, "tokenizer"):
            teacher_pad_ids.append(teacher_processor.tokenizer.pad_token_id)
        else:
            teacher_pad_ids.append(teacher_processor.pad_token_id)
    data_collator = DataCollatorForSupervisedDataset(
        pad_token_id=processor.tokenizer.pad_token_id,
        teacher_pad_token_id=teacher_pad,
        teacher_pad_token_ids=teacher_pad_ids,
    )

    return dict(train_dataset=sft_dataset, eval_dataset=eval_dataset, data_collator=data_collator)
