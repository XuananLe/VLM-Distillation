import json
from pathlib import Path, PurePosixPath

import torch
from safetensors.torch import load_file

DEFAULT_FILE_NAME_TEMPLATE = "{dataset_index}.pt"


class TeacherLogitsCache:
    def __init__(
        self,
        *,
        cache_dir: str,
        teacher_model_ids: list[str],
        expected_num_samples: int,
    ):
        cache_root = Path(cache_dir)
        root_metadata_path = cache_root / "metadata.json"
        if root_metadata_path.is_file():
            root_metadata = json.loads(root_metadata_path.read_text(encoding="utf-8"))
            cached_teacher_ids = list(root_metadata.get("teacher_model_ids") or [])
            if cached_teacher_ids and cached_teacher_ids != teacher_model_ids:
                raise ValueError(
                    "Teacher-logits cache teacher_model_ids do not match the current training "
                    f"teachers.\ncache={cached_teacher_ids}\ntrain={teacher_model_ids}"
                )
            cached_num_samples = root_metadata.get("num_samples")
            if cached_num_samples is not None and int(cached_num_samples) != expected_num_samples:
                raise ValueError(
                    "Teacher-logits cache sample count does not match the current dataset. "
                    f"cache_root={cache_root}, cache={cached_num_samples}, "
                    f"dataset={expected_num_samples}"
                )
            file_name_template = str(root_metadata.get("file_name_template") or DEFAULT_FILE_NAME_TEMPLATE)
            self.entries = [
                (cache_root, teacher_model_id, file_name_template) for teacher_model_id in teacher_model_ids
            ]
            return

        requested_teacher_ids = set(teacher_model_ids)
        entries_by_teacher: dict[str, tuple[Path, str, str]] = {}
        for metadata_path in cache_root.glob("*/metadata.json"):
            child_root = metadata_path.parent
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("num_samples") is not None and int(metadata["num_samples"]) != expected_num_samples:
                continue

            for teacher_model_id in metadata.get("teacher_model_ids") or []:
                if teacher_model_id not in requested_teacher_ids:
                    continue
                if teacher_model_id in entries_by_teacher:
                    raise ValueError(f"Multiple teacher-logits cache roots provide the same teacher: {teacher_model_id}")
                entries_by_teacher[teacher_model_id] = (
                    child_root,
                    teacher_model_id,
                    str(metadata.get("file_name_template") or DEFAULT_FILE_NAME_TEMPLATE),
                )
        missing_teacher_ids = [
            teacher_model_id for teacher_model_id in teacher_model_ids if teacher_model_id not in entries_by_teacher
        ]
        if missing_teacher_ids:
            raise FileNotFoundError(
                "Could not resolve cached teacher logits for the requested teachers from "
                f"cache_dir={cache_root}.\nmissing={missing_teacher_ids}\n"
                f"available={sorted(entries_by_teacher)}"
            )
        self.entries = [entries_by_teacher[teacher_model_id] for teacher_model_id in teacher_model_ids]

    @property
    def teacher_count(self) -> int:
        return len(self.entries)

    def load_sample(self, teacher_index: int, dataset_index: int) -> dict[str, torch.Tensor]:
        root, teacher_model_id, file_name_template = self.entries[teacher_index]
        sample_path = (
            root
            / PurePosixPath(teacher_model_id).name
            / file_name_template.format(dataset_index=dataset_index)
        )
        if not sample_path.exists():
            raise FileNotFoundError(f"Cached teacher logits not found for sample {dataset_index}: {sample_path}")
        if sample_path.suffix == ".safetensors":
            return load_file(sample_path, device="cpu")
        if sample_path.suffix == ".pt":
            return torch.load(sample_path, map_location="cpu")
        raise ValueError(f"Unsupported teacher-logits cache file extension: {sample_path}")


__all__ = ["TeacherLogitsCache"]
