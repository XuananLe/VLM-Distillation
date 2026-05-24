import json
from pathlib import Path, PurePosixPath

import torch

DEFAULT_FILE_NAME_TEMPLATE = "{dataset_index}.pt"


class TeacherLogitsCache:
    def __init__(
        self,
        *,
        cache_dir: str,
        teacher_model_ids: list[str],
    ):
        cache_root = Path(cache_dir)
        requested_teacher_ids = set(teacher_model_ids)
        entries_by_teacher: dict[str, tuple[Path, str, str]] = {}
        for metadata_path in cache_root.glob("*/metadata.json"):
            child_root = metadata_path.parent
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            for teacher_model_id in metadata.get("teacher_model_ids") or []:
                if teacher_model_id not in requested_teacher_ids:
                    continue
                entries_by_teacher[teacher_model_id] = (
                    child_root,
                    teacher_model_id,
                    str(metadata.get("file_name_template") or DEFAULT_FILE_NAME_TEMPLATE),
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
        return torch.load(sample_path, map_location="cpu")


__all__ = ["TeacherLogitsCache"]
