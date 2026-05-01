from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from pydantic import BaseModel, ConfigDict, Field
from safetensors.torch import load_file
import torch


class TeacherCacheMetadata(BaseModel):
    model_config = ConfigDict(extra="ignore")

    teacher_model_ids: list[str] = Field(default_factory=list)
    num_samples: int | None = None
    file_name_template: str = "{dataset_index}.pt"


@dataclass(slots=True, frozen=True)
class TeacherCacheEntry:
    teacher_model_id: str
    root: Path
    file_name_template: str

    @property
    def sample_dir(self) -> Path:
        return self.root / PurePosixPath(self.teacher_model_id).name


class TeacherLogitsCache:
    def __init__(
        self,
        *,
        cache_dir: str,
        teacher_model_ids: list[str],
        expected_num_samples: int,
    ):
        cache_root = Path(cache_dir)
        metadata = load_metadata(cache_root)
        if metadata is not None:
            cached_teacher_ids = metadata.teacher_model_ids
            if cached_teacher_ids and cached_teacher_ids != teacher_model_ids:
                raise ValueError(
                    "Teacher-logits cache teacher_model_ids do not match the current training "
                    f"teachers.\ncache={cached_teacher_ids}\ntrain={teacher_model_ids}"
                )
            validate_num_samples(
                metadata,
                expected_num_samples=expected_num_samples,
                cache_root=cache_root,
            )
            self.entries = [
                TeacherCacheEntry(
                    teacher_model_id=teacher_model_id,
                    root=cache_root,
                    file_name_template=metadata.file_name_template,
                )
                for teacher_model_id in teacher_model_ids
            ]
            return

        entries_by_teacher: dict[str, TeacherCacheEntry] = {}
        for metadata_path in cache_root.glob("*/metadata.json"):
            child_root = metadata_path.parent
            metadata = TeacherCacheMetadata.model_validate_json(metadata_path.read_text())
            validate_num_samples(
                metadata,
                expected_num_samples=expected_num_samples,
                cache_root=child_root,
            )
            for teacher_model_id in metadata.teacher_model_ids:
                if teacher_model_id in entries_by_teacher:
                    raise ValueError(
                        "Multiple teacher-logits cache roots provide the same teacher. "
                        f"teacher={teacher_model_id}"
                    )
                entries_by_teacher[teacher_model_id] = TeacherCacheEntry(
                    teacher_model_id=teacher_model_id,
                    root=child_root,
                    file_name_template=metadata.file_name_template,
                )

        missing_teacher_ids = [
            teacher_model_id
            for teacher_model_id in teacher_model_ids
            if teacher_model_id not in entries_by_teacher
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
        entry = self.entries[teacher_index]
        sample_path = entry.sample_dir / entry.file_name_template.format(dataset_index=dataset_index)
        if not sample_path.exists():
            raise FileNotFoundError(
                f"Cached teacher logits not found for sample {dataset_index}: {sample_path}"
            )
        if sample_path.suffix == ".safetensors":
            return load_file(sample_path, device="cpu")
        return torch.load(sample_path, map_location="cpu")


def load_metadata(cache_root: Path) -> TeacherCacheMetadata | None:
    metadata_path = cache_root / "metadata.json"
    if not metadata_path.is_file():
        return None
    return TeacherCacheMetadata.model_validate_json(metadata_path.read_text())


def validate_num_samples(
    metadata: TeacherCacheMetadata,
    *,
    expected_num_samples: int,
    cache_root: Path,
) -> None:
    cached_num_samples = metadata.num_samples
    if cached_num_samples is not None and int(cached_num_samples) != expected_num_samples:
        raise ValueError(
            "Teacher-logits cache sample count does not match the current dataset. "
            f"cache_root={cache_root}, cache={cached_num_samples}, "
            f"dataset={expected_num_samples}"
        )


__all__ = ["TeacherLogitsCache"]
