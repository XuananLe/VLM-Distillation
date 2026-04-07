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
        self.cache_dir = cache_dir
        shared_metadata = self._load_metadata(cache_dir)
        if shared_metadata is not None:
            self._init_from_shared_root(
                metadata=shared_metadata,
                teacher_model_ids=teacher_model_ids,
                expected_num_samples=expected_num_samples,
            )
            return

        self._init_from_parent_root(
            teacher_model_ids=teacher_model_ids,
            expected_num_samples=expected_num_samples,
        )

    @property
    def teacher_count(self) -> int:
        return len(self.teacher_slugs)

    def load_sample(self, teacher_index: int, dataset_index: int) -> dict[str, torch.Tensor]:
        teacher_dir = os.path.join(
            self.teacher_cache_roots[teacher_index],
            self.teacher_slugs[teacher_index],
        )
        sample_path = os.path.join(
            teacher_dir,
            self.file_name_templates[teacher_index].format(dataset_index=dataset_index),
        )
        if not os.path.exists(sample_path):
            raise FileNotFoundError(
                f"Cached teacher logits not found for sample {dataset_index}: {sample_path}"
            )
        return torch.load(sample_path, map_location="cpu")

    @staticmethod
    def _load_metadata(cache_root: str) -> Optional[dict]:
        metadata_path = os.path.join(cache_root, "metadata.json")
        if not os.path.exists(metadata_path):
            return None
        with open(metadata_path, "r") as handle:
            return json.load(handle)

    @staticmethod
    def _validate_num_samples(
        metadata: dict,
        *,
        expected_num_samples: int,
        cache_root: str,
    ) -> None:
        cached_num_samples = metadata.get("num_samples")
        if cached_num_samples is not None and int(cached_num_samples) != expected_num_samples:
            raise ValueError(
                "Teacher-logits cache sample count does not match the current dataset. "
                f"cache_root={cache_root}, cache={cached_num_samples}, "
                f"dataset={expected_num_samples}"
            )

    def _init_from_shared_root(
        self,
        *,
        metadata: dict,
        teacher_model_ids: Optional[list[str]],
        expected_num_samples: int,
    ) -> None:
        cached_teacher_ids = list(metadata.get("teacher_model_ids") or [])
        if teacher_model_ids and cached_teacher_ids and list(teacher_model_ids) != cached_teacher_ids:
            raise ValueError(
                "Teacher-logits cache teacher_model_ids do not match the current training "
                f"teachers.\ncache={cached_teacher_ids}\ntrain={teacher_model_ids}"
            )
        self._validate_num_samples(
            metadata,
            expected_num_samples=expected_num_samples,
            cache_root=self.cache_dir,
        )

        self.teacher_model_ids = cached_teacher_ids or list(teacher_model_ids or [])
        self.teacher_slugs = [
            teacher_model_id.split("/")[-1] for teacher_model_id in self.teacher_model_ids
        ]
        shared_template = metadata.get("file_name_template", "{dataset_index}.pt")
        self.file_name_template = shared_template
        self.file_name_templates = [shared_template for _ in self.teacher_slugs]
        self.teacher_cache_roots = [self.cache_dir for _ in self.teacher_slugs]

    def _init_from_parent_root(
        self,
        *,
        teacher_model_ids: Optional[list[str]],
        expected_num_samples: int,
    ) -> None:
        if not teacher_model_ids:
            raise FileNotFoundError(
                "Teacher-logits cache metadata not found at the cache root and no "
                "teacher_model_ids were provided to resolve child cache directories. "
                f"cache_dir={self.cache_dir}"
            )

        teacher_entries: dict[str, tuple[str, str]] = {}
        for entry in os.scandir(self.cache_dir):
            if not entry.is_dir():
                continue
            metadata = self._load_metadata(entry.path)
            if metadata is None:
                continue
            self._validate_num_samples(
                metadata,
                expected_num_samples=expected_num_samples,
                cache_root=entry.path,
            )
            file_name_template = metadata.get("file_name_template", "{dataset_index}.pt")
            for cached_teacher_id in list(metadata.get("teacher_model_ids") or []):
                if cached_teacher_id in teacher_entries:
                    raise ValueError(
                        "Multiple teacher-logits cache roots provide the same teacher. "
                        f"teacher={cached_teacher_id}"
                    )
                teacher_entries[cached_teacher_id] = (entry.path, file_name_template)

        missing_teacher_ids = [
            teacher_model_id
            for teacher_model_id in teacher_model_ids
            if teacher_model_id not in teacher_entries
        ]
        if missing_teacher_ids:
            available_teacher_ids = sorted(teacher_entries)
            raise FileNotFoundError(
                "Could not resolve cached teacher logits for the requested teachers from "
                f"cache_dir={self.cache_dir}.\nmissing={missing_teacher_ids}\n"
                f"available={available_teacher_ids}"
            )

        self.teacher_model_ids = list(teacher_model_ids)
        self.teacher_slugs = [
            teacher_model_id.split("/")[-1] for teacher_model_id in self.teacher_model_ids
        ]
        self.teacher_cache_roots = [
            teacher_entries[teacher_model_id][0] for teacher_model_id in self.teacher_model_ids
        ]
        self.file_name_templates = [
            teacher_entries[teacher_model_id][1] for teacher_model_id in self.teacher_model_ids
        ]
        self.file_name_template = self.file_name_templates[0]


__all__ = ["TeacherLogitsCache"]
