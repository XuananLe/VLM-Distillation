import argparse
import io
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
from streaming import MDSWriter

from src.train.arg_utils import parse_model_id_list


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repack per-sample teacher-logits caches into sharded MDS datasets."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Parent directory containing one per-teacher cache directory named by teacher slug.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Parent directory where repacked MDS caches will be written.",
    )
    parser.add_argument(
        "--teacher-model-ids",
        required=True,
        help="Python list literal or comma-separated teacher model IDs.",
    )
    parser.add_argument(
        "--remote-output-uri",
        default=None,
        help="Optional s3:// parent URI where repacked MDS caches will also be uploaded.",
    )
    parser.add_argument(
        "--compression",
        default=None,
        help="Optional MDS shard compression, e.g. `zstd:7`.",
    )
    parser.add_argument(
        "--size-limit",
        default="256mb",
        help="Shard size limit passed to MDSWriter.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of samples to convert per teacher.",
    )
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="Overwrite existing output directories if they already exist.",
    )
    parser.add_argument(
        "--keep-local",
        action="store_true",
        help="When also uploading remotely, keep the local shard files after upload.",
    )
    return parser.parse_args()


def _join_remote_uri(root: str, *parts: str) -> str:
    value = root.rstrip("/")
    for part in parts:
        value = f"{value}/{part.strip('/')}"
    return value


def _serialize_tensor(tensor: torch.Tensor) -> bytes:
    buffer = io.BytesIO()
    torch.save(tensor.detach().cpu(), buffer)
    return buffer.getvalue()


def _list_sample_files(input_teacher_dir: Path) -> list[Path]:
    sample_files = [
        path for path in input_teacher_dir.iterdir()
        if path.is_file() and path.suffix == ".pt" and path.stem.isdigit()
    ]
    if not sample_files:
        raise FileNotFoundError(f"No sample .pt files found in {input_teacher_dir}")
    return sorted(sample_files, key=lambda path: int(path.stem))


def _write_teacher_dataset(
    *,
    input_root: Path,
    output_root: Path,
    remote_output_root: str | None,
    teacher_model_id: str,
    limit: int | None,
    compression: str | None,
    size_limit: str,
    exist_ok: bool,
    keep_local: bool,
) -> None:
    teacher_slug = teacher_model_id.split("/")[-1]
    input_teacher_dir = input_root / teacher_slug
    if not input_teacher_dir.is_dir():
        raise FileNotFoundError(f"Expected teacher directory not found: {input_teacher_dir}")

    output_teacher_dir = output_root / teacher_slug
    output_teacher_dir.mkdir(parents=True, exist_ok=True)
    remote_teacher_dir = (
        _join_remote_uri(remote_output_root, teacher_slug) if remote_output_root else None
    )
    writer_out = (
        (str(output_teacher_dir), remote_teacher_dir)
        if remote_teacher_dir
        else str(output_teacher_dir)
    )

    columns = {
        "dataset_index": "int",
        "logits": "bytes",
        "labels": "bytes",
    }
    sample_files = _list_sample_files(input_teacher_dir)
    if limit is not None:
        sample_files = sample_files[:limit]

    with MDSWriter(
        columns=columns,
        out=writer_out,
        compression=compression,
        size_limit=size_limit,
        keep_local=keep_local,
        exist_ok=exist_ok,
    ) as writer:
        for sample_path in sample_files:
            sample = torch.load(sample_path, map_location="cpu")
            dataset_index = int(sample.get("dataset_index", int(sample_path.stem)))
            writer.write(
                {
                    "dataset_index": dataset_index,
                    "logits": _serialize_tensor(sample["logits"]),
                    "labels": _serialize_tensor(sample["labels"]),
                }
            )


def main() -> None:
    args = parse_args()
    teacher_model_ids = parse_model_id_list(args.teacher_model_ids, arg_name="--teacher-model-ids")
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for teacher_model_id in teacher_model_ids:
        _write_teacher_dataset(
            input_root=input_dir,
            output_root=output_dir,
            remote_output_root=args.remote_output_uri,
            teacher_model_id=teacher_model_id,
            limit=args.limit,
            compression=args.compression,
            size_limit=args.size_limit,
            exist_ok=args.exist_ok,
            keep_local=args.keep_local,
        )
        print(
            f"converted {teacher_model_id} -> "
            f"{output_dir / teacher_model_id.split('/')[-1]}"
        )


if __name__ == "__main__":
    main()
