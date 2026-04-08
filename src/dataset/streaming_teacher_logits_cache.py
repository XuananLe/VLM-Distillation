import hashlib
import os
import re
import time
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlparse

import torch
import s3fs
from fsspec.implementations.cached import WholeFileCacheFileSystem
from .teacher_logits_cache import TeacherLogitsCache

DEFAULT_STREAMING_CACHE_LIMIT = "100gb"
DEFAULT_LOCAL_CACHE_ROOT = "/tmp/teacher-logits-streaming"
KNOWN_TEACHER_CACHE_ALIASES = {
    "gemma-3-4b-it": "gemma3_4b",
    "internvl2-1b": "internvl2_1b",
    "internvl2-2b": "internvl2_2b",
    "qwen2-vl-2b-instruct": "qwen2vl_2b",
    "qwen2.5-vl-3b-instruct": "qwen25vl_3b",
}
def join_remote_uri(root: str, *parts: str) -> str:
    value = root.rstrip("/")
    for part in parts:
        value = f"{value}/{part.strip('/')}"
    return value


def normalize_teacher_cache_alias(teacher_model_id: str) -> str:
    teacher_slug = teacher_model_id.split("/")[-1].lower()
    cached_alias = KNOWN_TEACHER_CACHE_ALIASES.get(teacher_slug)
    if cached_alias is not None:
        return cached_alias

    normalized = teacher_slug
    for suffix in ("-instruct", "-chat", "-it"):
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)]
    normalized = normalized.replace(".", "")
    normalized = normalized.replace("-", "_")
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized


def parse_size_limit_bytes(value: str | int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value

    normalized = str(value).strip().lower()
    match = re.fullmatch(r"(\d+)([kmgt]?b)?", normalized)
    if match is None:
        raise ValueError(
            "cache_limit must be an integer byte count or a string like 100gb, 512mb, 64kb."
        )

    amount = int(match.group(1))
    suffix = match.group(2) or "b"
    multipliers = {
        "b": 1,
        "kb": 1024,
        "mb": 1024 ** 2,
        "gb": 1024 ** 3,
        "tb": 1024 ** 4,
    }
    return amount * multipliers[suffix]


class StreamingTeacherLogitsCache:
    def __init__(
        self,
        *,
        cache_dir: str | None,
        teacher_model_ids: list[str],
        expected_num_samples: int,
        dataset_name: str | None,
        remote_uri: str | None = None,
        local_cache_dir: str | None = None,
        cache_limit: str | int | None = None,
        predownload: int | None = None,
    ):
        del local_cache_dir, predownload
        if not teacher_model_ids:
            raise ValueError("teacher_model_ids must be provided for teacher logits.")

        self._local_cache = None
        if remote_uri is None:
            if cache_dir is None:
                raise ValueError(
                    "teacher_logits_cache_dir must be provided when reading teacher logits "
                    "from a local cache root."
                )
            self._local_cache = TeacherLogitsCache(
                cache_dir=cache_dir,
                teacher_model_ids=teacher_model_ids,
                expected_num_samples=expected_num_samples,
            )
            self.teacher_model_ids = self._local_cache.teacher_model_ids
            self.teacher_slugs = self._local_cache.teacher_slugs
            return

        del cache_dir
        if not dataset_name:
            raise ValueError("dataset_name must be provided when using remote teacher logits.")

        self.remote_uri = remote_uri
        self.cache_limit_bytes = parse_size_limit_bytes(
            cache_limit or DEFAULT_STREAMING_CACHE_LIMIT
        )
        self.teacher_model_ids = list(teacher_model_ids)
        self.teacher_slugs = [
            teacher_model_id.split("/")[-1] for teacher_model_id in self.teacher_model_ids
        ]
        self.teacher_remote_dataset_names = [
            f"{dataset_name}_teacher_logits_{normalize_teacher_cache_alias(teacher_model_id)}"
            for teacher_model_id in self.teacher_model_ids
        ]
        self.remote_teacher_dataset_roots = [
            join_remote_uri(self.remote_uri, remote_dataset_name, teacher_slug)
            for remote_dataset_name, teacher_slug in zip(
                self.teacher_remote_dataset_names,
                self.teacher_slugs,
            )
        ]
        self.local_teacher_dataset_roots = [
            self._local_cache_path(DEFAULT_LOCAL_CACHE_ROOT, remote_root)
            for remote_root in self.remote_teacher_dataset_roots
        ]
        self._cache_filesystems = [
            self._build_cache_filesystem(local_root)
            for local_root in self.local_teacher_dataset_roots
        ]

        for local_root in self.local_teacher_dataset_roots:
            os.makedirs(local_root, exist_ok=True)

    @property
    def teacher_count(self) -> int:
        if self._local_cache is not None:
            return self._local_cache.teacher_count
        return len(self.teacher_slugs)

    def load_sample(self, teacher_index: int, dataset_index: int) -> dict[str, torch.Tensor]:
        if self._local_cache is not None:
            return self._local_cache.load_sample(teacher_index, dataset_index)

        local_root = self.local_teacher_dataset_roots[teacher_index]
        cache_fs = self._cache_filesystems[teacher_index]
        remote_path = join_remote_uri(
            self.remote_teacher_dataset_roots[teacher_index],
            f"{dataset_index}.pt",
        )

        with self._teacher_lock(local_root):
            file_obj = cache_fs.open(remote_path, "rb")
            self._touch_cached_entry(cache_fs, remote_path)
            self._evict_if_needed(cache_fs, keep_remote_path=remote_path)

        with file_obj:
            sample = torch.load(file_obj, map_location="cpu")

        loaded_index = int(sample.get("dataset_index", dataset_index))
        if loaded_index != dataset_index:
            raise ValueError(
                "Streaming teacher cache sample order mismatch. "
                f"expected={dataset_index}, loaded={loaded_index}"
            )
        return {
            "logits": sample["logits"],
            "labels": sample["labels"],
        }

    def _touch_cached_entry(
        self,
        cache_fs: WholeFileCacheFileSystem,
        remote_path: str,
    ) -> None:
        normalized_path = cache_fs._strip_protocol(remote_path)
        entry = cache_fs._metadata.cached_files[-1].get(normalized_path)
        if entry is None:
            return
        entry["time"] = time.time()
        cached_detail = cache_fs._metadata.check_file(normalized_path, None)
        if cached_detail:
            _, local_path = cached_detail
            if os.path.exists(local_path):
                os.utime(local_path, None)
        cache_fs.save_cache()

    def _evict_if_needed(
        self,
        cache_fs: WholeFileCacheFileSystem,
        keep_remote_path: str,
    ) -> None:
        if self.cache_limit_bytes is None:
            return

        total_size = cache_fs.cache_size()
        if total_size <= self.cache_limit_bytes:
            return

        keep_path = cache_fs._strip_protocol(keep_remote_path)
        eviction_candidates = sorted(
            (
                (remote_path, detail)
                for remote_path, detail in cache_fs._metadata.cached_files[-1].items()
                if remote_path != keep_path
            ),
            key=lambda item: item[1].get("time", 0),
        )
        for candidate_path, _ in eviction_candidates:
            if total_size <= self.cache_limit_bytes:
                break

            cached_detail = cache_fs._metadata.check_file(candidate_path, None)
            if not cached_detail:
                continue
            _, local_path = cached_detail
            candidate_size = os.path.getsize(local_path) if os.path.exists(local_path) else 0
            cache_fs.pop_from_cache(candidate_path)
            total_size -= candidate_size

    def _build_cache_filesystem(self, local_root: str) -> WholeFileCacheFileSystem:
        s3_filesystem = s3fs.S3FileSystem(**self._build_s3_options())
        return WholeFileCacheFileSystem(
            fs=s3_filesystem,
            cache_storage=local_root,
            check_files=False,
            expiry_time=0,
            same_names=True,
        )

    def _build_s3_options(self) -> dict[str, object]:
        endpoint_url = (
            os.environ.get("S3_ENDPOINT_URL")
            or os.environ.get("AWS_ENDPOINT_URL_S3")
            or os.environ.get("AWS_S3_ENDPOINT_URL")
        )
        region_name = os.environ.get("AWS_DEFAULT_REGION") or os.environ.get("AWS_REGION")
        if endpoint_url and "cloudflarestorage.com" in endpoint_url and not region_name:
            region_name = "auto"
        kwargs: dict[str, object] = {}
        if endpoint_url:
            kwargs["endpoint_url"] = endpoint_url
        if region_name:
            kwargs["client_kwargs"] = {"region_name": region_name}
        return kwargs

    @contextmanager
    def _teacher_lock(self, local_root: str):
        lock_path = os.path.join(local_root, ".cache.lock")
        with open(lock_path, "a+", encoding="utf-8") as lock_file:
            try:
                import fcntl

                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                yield
            finally:
                try:
                    import fcntl

                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                except Exception:
                    pass

    @staticmethod
    def _local_cache_path(base_local_root: str, remote_root: str) -> str:
        parsed = urlparse(remote_root)
        root_name = Path(parsed.path.rstrip("/")).name or "teacher"
        digest = hashlib.sha1(remote_root.encode("utf-8")).hexdigest()[:12]
        return os.path.join(base_local_root, f"{root_name}-{digest}")


__all__ = ["StreamingTeacherLogitsCache"]
