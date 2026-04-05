import os
from typing import Optional

import torch
import ujson as json


class TeacherLogitsCache:
    def __init__(
        self,
        *,
        cache_dir: str,
        teacher_model_ids: Optional[list[str]],
        expected_num_samples: int,
    ):
        metadata_path = os.path.join(cache_dir, "metadata.json")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(
                f"Teacher-logits cache metadata not found: {metadata_path}"
            )

        metadata = json.load(open(metadata_path, "r"))
        cached_teacher_ids = list(metadata.get("teacher_model_ids") or [])
        if teacher_model_ids and cached_teacher_ids and list(teacher_model_ids) != cached_teacher_ids:
            raise ValueError(
                "Teacher-logits cache teacher_model_ids do not match the current training "
                f"teachers.\ncache={cached_teacher_ids}\ntrain={teacher_model_ids}"
            )

        cached_num_samples = metadata.get("num_samples")
        if cached_num_samples is not None and int(cached_num_samples) != expected_num_samples:
            raise ValueError(
                "Teacher-logits cache sample count does not match the current dataset. "
                f"cache={cached_num_samples}, dataset={expected_num_samples}"
            )

        self.cache_dir = cache_dir
        self.file_name_template = metadata.get("file_name_template", "{dataset_index}.pt")
        self.teacher_model_ids = cached_teacher_ids or list(teacher_model_ids or [])
        self.teacher_slugs = [
            teacher_model_id.split("/")[-1] for teacher_model_id in self.teacher_model_ids
        ]

    @property
    def teacher_count(self) -> int:
        return len(self.teacher_slugs)

    def load_sample(self, teacher_index: int, dataset_index: int) -> dict[str, torch.Tensor]:
        teacher_dir = os.path.join(self.cache_dir, self.teacher_slugs[teacher_index])
        sample_path = os.path.join(
            teacher_dir,
            self.file_name_template.format(dataset_index=dataset_index),
        )
        if not os.path.exists(sample_path):
            raise FileNotFoundError(
                f"Cached teacher logits not found for sample {dataset_index}: {sample_path}"
            )
        return torch.load(sample_path, map_location="cpu")


__all__ = ["TeacherLogitsCache"]
