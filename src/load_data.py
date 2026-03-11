from __future__ import annotations

import argparse
import io
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

DATASET_SOURCES = {
    "textvqa": {
        "hub": "facebook/textvqa",
        "config": "textvqa",
        "fallback_hub": "lmms-lab/textvqa",
        "fallback_config": None,
    },
    "docvqa": {
        "hub": "HuggingFaceM4/DocumentVQA",
        "config": None,
        "fallback_hub": "lmms-lab/DocVQA",
        "fallback_config": None,
    },
    "chartqa": {
        "hub": "HuggingFaceM4/ChartQA",
        "config": None,
        "fallback_hub": "lmms-lab/ChartQA",
        "fallback_config": None,
    },
}

DATASET_ALIASES = {
    "textvqa": "textvqa",
    "facebook/textvqa": "textvqa",
    "lmms-lab/textvqa": "textvqa",
    "docvqa": "docvqa",
    "documentvqa": "docvqa",
    "huggingfacem4/documentvqa": "docvqa",
    "lmms-lab/docvqa": "docvqa",
    "chartqa": "chartqa",
    "chart qa": "chartqa",
    "huggingfacem4/chartqa": "chartqa",
    "lmms-lab/chartqa": "chartqa",
}

QUESTION_NAME_PRIORITY = ("question", "query", "prompt")
ANSWER_NAME_PRIORITY = ("answers", "answer", "label", "target")
ID_NAME_PRIORITY = ("question_id", "questionid", "id", "docid", "image_id")
IMAGE_NAME_PRIORITY = ("image_id", "docid")
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


def _canonical_dataset_name(dataset_name: str) -> str:
    key = dataset_name.strip().lower()
    canonical = DATASET_ALIASES.get(key)
    if canonical is None:
        supported = ", ".join(sorted(DATASET_SOURCES.keys()))
        raise ValueError(f"Unsupported dataset '{dataset_name}'. Supported: {supported}")
    return canonical


def _default_output_paths(dataset_name: str, split: str, output_root: Path) -> tuple[Path, Path]:
    base = output_root / dataset_name
    return base / f"{split}_llava.json", base / "images"


def _hf_load(dataset_id: str, config: str | None, split: str):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("Missing dependency `datasets`. Install it with: pip install datasets") from exc

    if config is None:
        return load_dataset(dataset_id, split=split)
    return load_dataset(dataset_id, config, split=split)


def _load_dataset_split(dataset_name: str, split: str):
    source = DATASET_SOURCES[dataset_name]
    try:
        return _hf_load(source["hub"], source["config"], split), source["hub"]
    except RuntimeError as exc:
        fallback_hub = source["fallback_hub"]
        if fallback_hub and "Dataset scripts are no longer supported" in str(exc):
            print(
                f"Dataset '{source['hub']}' uses a dataset script. "
                f"Falling back to '{fallback_hub}'."
            )
            return _hf_load(fallback_hub, source["fallback_config"], split), fallback_hub
        raise


def _normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _is_string_feature(feature: Any) -> bool:
    return getattr(feature, "dtype", None) == "string"


def _is_sequence_of_strings_feature(feature: Any) -> bool:
    if feature.__class__.__name__ not in {"Sequence", "List", "LargeList"}:
        return False
    inner_feature = getattr(feature, "feature", None)
    if inner_feature is None:
        return False
    return _is_string_feature(inner_feature)


def _is_image_feature(feature: Any) -> bool:
    return feature.__class__.__name__ == "Image" or getattr(feature, "dtype", None) == "image"


def _pick_exact_name(fields: list[str], priority: tuple[str, ...]) -> str | None:
    normalized_fields = {_normalize_name(f): f for f in fields}
    return next((normalized_fields.get(_normalize_name(target)) for target in priority if normalized_fields.get(_normalize_name(target))), None)


def _pick_contains_name(fields: list[str], tokens: tuple[str, ...]) -> str | None:
    normalized_tokens = tuple(_normalize_name(token) for token in tokens)
    return next(
        (
            field
            for field in fields
            if any(token in _normalize_name(field) for token in normalized_tokens)
        ),
        None,
    )


def _infer_schema(dataset: Any) -> dict[str, str | None]:
    features = dataset.features
    all_fields = list(features.keys())
    image_fields = [name for name, feat in features.items() if _is_image_feature(feat)]
    string_fields = [name for name, feat in features.items() if _is_string_feature(feat)]
    string_sequence_fields = [name for name, feat in features.items() if _is_sequence_of_strings_feature(feat)]

    image_field = image_fields[0] if image_fields else ("image" if "image" in all_fields else None)
    if image_field is None:
        raise ValueError(f"Unable to infer image field from schema: {all_fields}")

    question_field = (
        _pick_exact_name(string_fields, QUESTION_NAME_PRIORITY)
        or _pick_contains_name(string_fields, ("question", "query", "prompt"))
    )
    if question_field is None:
        raise ValueError(f"Unable to infer question field from schema: {all_fields}")

    answer_field = (
        _pick_exact_name(string_sequence_fields, ANSWER_NAME_PRIORITY)
        or _pick_exact_name(string_fields, ANSWER_NAME_PRIORITY)
        or _pick_contains_name(string_sequence_fields, ("answer", "label", "target"))
        or _pick_contains_name(string_fields, ("answer", "label", "target"))
    )
    if answer_field is None:
        raise ValueError(f"Unable to infer answer field from schema: {all_fields}")

    id_field = _pick_exact_name(all_fields, ID_NAME_PRIORITY) or _pick_contains_name(all_fields, ("id",))
    image_name_field = (
        _pick_exact_name(all_fields, IMAGE_NAME_PRIORITY)
        or _pick_contains_name(all_fields, ("imageid",))
        or id_field
    )

    return {
        "image_field": image_field,
        "question_field": question_field,
        "answer_field": answer_field,
        "id_field": id_field,
        "image_name_field": image_name_field,
    }


def _print_schema_info(dataset: Any, resolved_schema: dict[str, str | None]) -> None:
    print("Pulled schema from dataset.features:")
    for name, feature in dataset.features.items():
        print(f"  - {name}: {feature}")
    print("Resolved fields:")
    for key in ("image_field", "question_field", "answer_field", "id_field", "image_name_field"):
        print(f"  - {key}: {resolved_schema.get(key)}")


def _sanitize_for_filename(raw_value: Any) -> str:
    return SAFE_FILENAME_RE.sub("_", str(raw_value))


def _extract_image_as_pil(image_value: Any) -> Any:
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Missing dependency `Pillow`. Install it with: pip install Pillow") from exc

    if isinstance(image_value, Image.Image):
        return image_value.convert("RGB")

    if isinstance(image_value, dict) and image_value.get("bytes") is not None:
        return Image.open(io.BytesIO(image_value["bytes"])).convert("RGB")

    if isinstance(image_value, dict) and image_value.get("path"):
        return Image.open(image_value["path"]).convert("RGB")

    raise TypeError("Unsupported image field type. Expected PIL image or dict with `bytes`/`path`.")


def _pick_first_text(value: Any) -> str | None:
    if value is None or isinstance(value, (list, tuple, dict)):
        return None
    return str(value).strip() or None


def _pick_first_answer(answer_value: Any) -> str | None:
    values = answer_value if isinstance(answer_value, (list, tuple)) else (answer_value,)
    return next((text for text in (_pick_first_text(value) for value in values) if text), None)


def convert_dataset_to_llava(
    dataset_name: str,
    split: str,
    output_json: Path,
    image_dir: Path,
) -> dict[str, int]:
    dataset, loaded_from = _load_dataset_split(dataset_name, split)
    print(f"Loaded dataset: {loaded_from} (split={split})")

    schema = _infer_schema(dataset)
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
        question = _pick_first_text(sample.get(schema["question_field"])) if schema["question_field"] else None
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
                _extract_image_as_pil(sample.get(schema["image_field"])).save(
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

    dataset_name = _canonical_dataset_name(args.dataset)
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
