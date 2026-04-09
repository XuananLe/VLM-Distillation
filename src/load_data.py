from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset.vqa_loading import (
    canonical_dataset_name,
    extract_image_as_pil,
    infer_schema,
    load_dataset_split,
    pick_first_text,
)

SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert VQA datasets to LLaVA JSON")
    parser.add_argument(
        "--dataset",
        type=str,
        default="textvqa",
        help="Dataset alias/name: textvqa, docvqa, chartqa (or known HF ids).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to convert (e.g. train/validation/test/val).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data"),
        help="Root output directory when output paths are not explicitly set.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional output JSON path. Default: data/<dataset>/<split>_llava.json",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=None,
        help="Optional image directory. Default: data/<dataset>/images",
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        default=min(8, os.cpu_count()),
        help="Number of worker processes for dataset conversion. Use 1 to disable multiprocessing.",
    )
    return parser.parse_args()


def _default_output_paths(dataset_name: str, split: str, output_root: Path) -> tuple[Path, Path]:
    base = output_root / dataset_name
    return base / f"{split}_llava.json", base / "images"


def _print_schema_info(dataset: Any, resolved_schema: dict[str, str | None]) -> None:
    print("Pulled schema from dataset.features:")
    for name, feature in dataset.features.items():
        print(f"  - {name}: {feature}")
    print("Resolved fields:")
    for key in ("image_field", "question_field", "answer_field", "id_field", "image_name_field"):
        print(f"  - {key}: {resolved_schema.get(key)}")


def _sanitize_for_filename(raw_value: Any) -> str:
    return SAFE_FILENAME_RE.sub("_", str(raw_value))


def _pick_first_answer(answer_value: Any) -> str | None:
    values = answer_value if isinstance(answer_value, (list, tuple)) else (answer_value,)
    return next((text for text in (pick_first_text(value) for value in values) if text), None)


def _chartqa_split_label(sample: dict[str, Any]) -> str | None:
    raw_value = sample.get("human_or_machine")
    if raw_value is None:
        return None
    normalized = str(raw_value).strip().lower()
    if normalized in {"0", "human"}:
        return "human"
    if normalized in {"1", "machine", "augmented"}:
        return "augmented"
    return normalized


def _write_image_if_missing(image_value: Any, output_path: Path) -> None:
    if output_path.exists():
        return
    temp_path = output_path.parent / f"{output_path.name}.tmp.{os.getpid()}"
    extract_image_as_pil(image_value).save(temp_path, format="JPEG", quality=95)
    if output_path.exists():
        temp_path.unlink(missing_ok=True)
        return
    temp_path.replace(output_path)


def _convert_sample_to_llava_row(
    task: tuple[int, dict[str, Any], dict[str, str | None], str, str]
) -> tuple[str, dict[str, Any] | None, str | None, str | None]:
    row_idx, sample, schema, dataset_name, image_dir_str = task

    question = pick_first_text(sample.get(schema["question_field"])) if schema["question_field"] else None
    if not question:
        return "skipped_no_question", None, None, None

    answer = _pick_first_answer(sample.get(schema["answer_field"])) if schema["answer_field"] else None
    if not answer:
        return "skipped_no_answer", None, None, None

    sample_id = sample.get(schema["id_field"]) if schema["id_field"] else None
    image_name_source = sample.get(schema["image_name_field"]) if schema["image_name_field"] else None
    image_name_source = image_name_source or sample_id or f"row_{row_idx:08d}"
    image_id = _sanitize_for_filename(image_name_source)
    image_filename = f"{image_id}.jpg"

    try:
        _write_image_if_missing(sample.get(schema["image_field"]), Path(image_dir_str) / image_filename)
    except Exception as exc:
        return "skipped_image_error", None, None, f"  Warning: skipping row {row_idx} — image error: {exc}"

    llava_row = {
        "id": str(sample_id if sample_id is not None else row_idx),
        "image": image_filename,
        "conversations": [
            {"from": "human", "value": f"<image>\n{question}"},
            {"from": "gpt", "value": answer},
        ],
    }
    if dataset_name == "chartqa":
        chartqa_split = _chartqa_split_label(sample)
        if chartqa_split is not None:
            llava_row["chartqa_split"] = chartqa_split
    return "written", llava_row, image_id, None


def convert_dataset_to_llava(
    dataset_name: str,
    split: str,
    output_json: Path,
    image_dir: Path,
    num_proc: int,
) -> dict[str, int]:
    dataset, loaded_from = load_dataset_split(dataset_name, split, log_fallback=True)
    print(f"Loaded dataset: {loaded_from} (split={split})")
    if num_proc < 1:
        raise ValueError("--num-proc must be at least 1.")

    schema = infer_schema(dataset, require_answer_field=True)
    _print_schema_info(dataset, schema)

    image_dir.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    image_id_to_filename: dict[str, str] = {}
    llava_rows: list[dict[str, Any]] = []
    stats = Counter(
        {
            "skipped_no_question": 0,
            "skipped_no_answer": 0,
            "skipped_image_error": 0,
        }
    )

    tasks = (
        (row_idx, sample, schema, dataset_name, str(image_dir))
        for row_idx, sample in enumerate(dataset)
    )
    progress = tqdm(
        total=len(dataset),
        desc=f"Converting {dataset_name}:{split}",
        unit="sample",
    )
    if num_proc <= 1:
        results = map(_convert_sample_to_llava_row, tasks)
    else:
        with ProcessPoolExecutor(max_workers=num_proc) as executor:
            results = executor.map(_convert_sample_to_llava_row, tasks, chunksize=32)
            for status, llava_row, image_id, warning in results:
                progress.update(1)
                if warning:
                    print(warning)
                if status != "written":
                    stats[status] += 1
                    continue
                assert llava_row is not None and image_id is not None
                image_id_to_filename.setdefault(image_id, llava_row["image"])
                llava_rows.append(llava_row)
        progress.close()
        with output_json.open("w", encoding="utf-8") as f:
            json.dump(llava_rows, f, ensure_ascii=False, indent=2)

        stats["written_samples"] = len(llava_rows)
        stats["unique_images"] = len(image_id_to_filename)
        return stats

    for status, llava_row, image_id, warning in results:
        progress.update(1)
        if warning:
            print(warning)
        if status != "written":
            stats[status] += 1
            continue
        assert llava_row is not None and image_id is not None
        image_id_to_filename.setdefault(image_id, llava_row["image"])
        llava_rows.append(llava_row)
    progress.close()

    with output_json.open("w", encoding="utf-8") as f:
        json.dump(llava_rows, f, ensure_ascii=False, indent=2)

    stats["written_samples"] = len(llava_rows)
    stats["unique_images"] = len(image_id_to_filename)
    return stats


def main() -> None:
    args = parse_args()

    dataset_name = canonical_dataset_name(args.dataset)
    default_json, default_image_dir = _default_output_paths(dataset_name, args.split, args.output_root)
    output_json = args.output_json or default_json
    image_dir = args.image_dir or default_image_dir

    stats = convert_dataset_to_llava(
        dataset_name=dataset_name,
        split=args.split,
        output_json=output_json,
        image_dir=image_dir,
        num_proc=args.num_proc,
    )

    print("Conversion complete.")
    print(f"  Dataset: {dataset_name}")
    print(f"  Output JSON: {output_json}")
    print(f"  Image dir:   {image_dir}")
    for key, value in (
        ("Samples", stats["written_samples"]),
        ("Images", stats["unique_images"]),
        ("Skipped (no question)", stats["skipped_no_question"]),
        ("Skipped (no answer)", stats["skipped_no_answer"]),
        ("Skipped (image error)", stats["skipped_image_error"]),
    ):
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
