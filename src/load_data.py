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
    """Parse CLI flags for converting a VQA dataset split into LLaVA-style JSON."""
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


def convert_dataset_to_llava(
    dataset_name: str,
    split: str,
    output_json: Path,
    image_dir: Path,
) -> dict[str, int]:
    """Convert one VQA dataset split into saved JPEG images plus LLaVA-style chat JSON."""
    dataset, loaded_from = load_dataset_split(dataset_name, split, log_fallback=True)
    print(f"Loaded dataset: {loaded_from} (split={split})")

    schema = infer_schema(dataset_name, dataset, require_answer_field=True)
    print("Pulled schema from dataset.features:")
    for name, feature in dataset.features.items():
        print(f"  - {name}: {feature}")
    print("Resolved fields:")
    for key in ("image_field", "question_field", "answer_field", "id_field", "image_name_field"):
        print(f"  - {key}: {schema.get(key)}")

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

        if schema["answer_field"]:
            answer_value = sample.get(schema["answer_field"])
            answer_values = answer_value if isinstance(answer_value, (list, tuple)) else (answer_value,)
            answer = next((text for text in (pick_first_text(value) for value in answer_values) if text), None)
        else:
            answer = None
        if not answer:
            stats["skipped_no_answer"] += 1
            continue

        sample_id = sample.get(schema["id_field"]) if schema["id_field"] else None
        image_name_source = sample.get(schema["image_name_field"]) if schema["image_name_field"] else None
        image_name_source = image_name_source or sample_id or f"row_{row_idx:08d}"

        image_id = SAFE_FILENAME_RE.sub("_", str(image_name_source))
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
            except (OSError, TypeError, ValueError) as e:
                print(f"  Warning: skipping row {row_idx} — image error: {e}")
                stats["skipped_image_error"] += 1
                continue

        llava_row = {
            "id": str(sample_id if sample_id is not None else row_idx),
            "image": image_filename,
            "conversations": [
                {"from": "human", "value": f"<image>\n{question}"},
                {"from": "gpt", "value": answer},
            ],
        }
        if dataset_name == "chartqa":
            raw_chartqa_split = sample.get("human_or_machine")
            chartqa_split = None
            if raw_chartqa_split is not None:
                chartqa_split = str(raw_chartqa_split).strip().lower()
                if chartqa_split in {"0", "human"}:
                    chartqa_split = "human"
                elif chartqa_split in {"1", "machine", "augmented"}:
                    chartqa_split = "augmented"
            if chartqa_split is not None:
                llava_row["chartqa_split"] = chartqa_split
        llava_rows.append(llava_row)

    with output_json.open("w", encoding="utf-8") as f:
        json.dump(llava_rows, f, ensure_ascii=False, indent=2)

    stats["written_samples"] = len(llava_rows)
    stats["unique_images"] = len(image_id_to_filename)
    return stats


def main() -> None:
    """Run the dataset-conversion CLI end to end."""
    args = parse_args()

    dataset_name = canonical_dataset_name(args.dataset)
    default_output_dir = args.output_root / dataset_name
    output_json = args.output_json or default_output_dir / f"{args.split}_llava.json"
    image_dir = args.image_dir or default_output_dir / "images"

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
