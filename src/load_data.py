from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
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


def convert_dataset_to_llava(
    dataset_name: str,
    split: str,
    output_json: Path,
    image_dir: Path,
) -> dict[str, int]:
    dataset, loaded_from = load_dataset_split(dataset_name, split, log_fallback=True)
    print(f"Loaded dataset: {loaded_from} (split={split})")

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

    for row_idx, sample in enumerate(
        tqdm(
            dataset,
            total=len(dataset),
            desc=f"Converting {dataset_name}:{split}",
            unit="sample",
        )
    ):
        question = pick_first_text(sample.get(schema["question_field"])) if schema["question_field"] else None
        if not question:
            stats["skipped_no_question"] += 1
            continue

        answer = _pick_first_answer(sample.get(schema["answer_field"])) if schema["answer_field"] else None
        if not answer:
            stats["skipped_no_answer"] += 1
            continue

        sample_id = sample.get(schema["id_field"]) if schema["id_field"] else None
        image_name_source = sample.get(schema["image_name_field"]) if schema["image_name_field"] else None
        image_name_source = image_name_source or sample_id or f"row_{row_idx:08d}"

        image_id = _sanitize_for_filename(image_name_source)
        image_filename = image_id_to_filename.get(image_id)
        if image_filename is None:
            try:
                image_filename = f"{image_id}.jpg"
                extract_image_as_pil(sample.get(schema["image_field"])).save(
                    image_dir / image_filename,
                    format="JPEG",
                    quality=95,
                )
                image_id_to_filename[image_id] = image_filename
            except Exception as e:
                print(f"  Warning: skipping row {row_idx} — image error: {e}")
                stats["skipped_image_error"] += 1
                continue

        llava_rows.append(
            {
                "id": str(sample_id if sample_id is not None else row_idx),
                "image": image_filename,
                "conversations": [
                    {"from": "human", "value": f"<image>\n{question}"},
                    {"from": "gpt", "value": answer},
                ],
            }
        )

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
