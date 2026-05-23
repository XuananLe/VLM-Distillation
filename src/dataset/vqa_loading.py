import io
from typing import Any

from datasets import load_dataset
from PIL import Image

DATASETS = {
    "textvqa": {
        "aliases": ("textvqa", "facebook/textvqa", "lmms-lab/textvqa"),
        "hub": "lmms-lab/textvqa",
        "schema": {
            "image_field": "image",
            "question_field": "question",
            "answer_field": "answers",
            "id_field": "question_id",
            "image_name_field": "image_id",
        },
    },
    "docvqa": {
        "aliases": ("docvqa", "documentvqa", "huggingfacem4/documentvqa", "lmms-lab/docvqa"),
        "hub": "HuggingFaceM4/DocumentVQA",
        "schema": {
            "image_field": "image",
            "question_field": "question",
            "answer_field": "answers",
            "id_field": "questionId",
            "image_name_field": "docId",
        },
    },
    "chartqa": {
        "aliases": ("chartqa", "chart qa", "huggingfacem4/chartqa", "lmms-lab/chartqa"),
        "hub": "HuggingFaceM4/ChartQA",
        "schema": {
            "image_field": "image",
            "question_field": "query",
            "answer_field": "label",
            "id_field": None,
            "image_name_field": None,
        },
    },
}

DATASET_ALIASES = {alias: name for name, config in DATASETS.items() for alias in config["aliases"]}


def canonical_dataset_name(dataset_name: str) -> str:
    key = dataset_name.strip().lower()
    canonical = DATASET_ALIASES.get(key)
    if canonical is None:
        supported = ", ".join(sorted(DATASETS))
        raise ValueError(f"Unsupported dataset '{dataset_name}'. Supported: {supported}")
    return canonical


def load_dataset_split(
    dataset_name: str,
    split: str,
):
    dataset_name = canonical_dataset_name(dataset_name)
    source = DATASETS[dataset_name]
    schema = source["schema"].copy()
    dataset = load_dataset(source["hub"], split=split)

    all_fields = set(dataset.features)
    missing_fields = sorted(field for field in schema.values() if field is not None and field not in all_fields)
    if missing_fields:
        raise ValueError(
            f"Dataset '{dataset_name}' schema changed. Missing fields: {missing_fields}. "
            f"Available fields: {sorted(all_fields)}"
        )
    return dataset, source["hub"], schema


def pick_first_text(value: Any) -> str | None:
    """Return the first non-empty scalar text value or None for non-text containers."""
    if value is None or isinstance(value, (list, tuple, dict)):
        return None
    text = str(value).strip()
    return text or None


def extract_image_as_pil(image_value: Any) -> Any:
    """Convert a dataset image field into an RGB PIL image."""
    if isinstance(image_value, Image.Image):
        return image_value.convert("RGB")
    if isinstance(image_value, dict) and image_value.get("bytes") is not None:
        return Image.open(io.BytesIO(image_value["bytes"])).convert("RGB")
    if isinstance(image_value, dict) and image_value.get("path"):
        return Image.open(image_value["path"]).convert("RGB")
    raise TypeError("Unsupported image field type. Expected PIL image or dict with `bytes`/`path`.")


__all__ = [
    "canonical_dataset_name",
    "extract_image_as_pil",
    "load_dataset_split",
    "pick_first_text",
]
