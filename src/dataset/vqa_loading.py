import io
from typing import Any

from datasets import load_dataset
from PIL import Image

DATASET_SOURCES = {
    "textvqa": {
        "hub": "lmms-lab/textvqa",
        "config": None,
        "fallback_hub": "facebook/textvqa",
        "fallback_config": "textvqa",
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

DATASET_SCHEMAS = {
    "textvqa": {
        "image_field": "image",
        "question_field": "question",
        "answer_field": "answers",
        "id_field": "question_id",
        "image_name_field": "image_id",
    },
    "docvqa": {
        "image_field": "image",
        "question_field": "question",
        "answer_field": "answers",
        "id_field": "questionId",
        "image_name_field": "docId",
    },
    "chartqa": {
        "image_field": "image",
        "question_field": "query",
        "answer_field": "label",
        "id_field": None,
        "image_name_field": None,
    },
}


def canonical_dataset_name(dataset_name: str) -> str:
    """Normalize one user dataset name or alias to the repo's canonical dataset key."""
    key = dataset_name.strip().lower()
    canonical = DATASET_ALIASES.get(key)
    if canonical is None:
        supported = ", ".join(sorted(DATASET_SOURCES))
        raise ValueError(f"Unsupported dataset '{dataset_name}'. Supported: {supported}")
    return canonical


def load_dataset_split(
    dataset_name: str,
    split: str,
    *,
    log_fallback: bool = False,
):
    """Load one VQA dataset split and fall back when the primary hub entry requires a script."""
    source = DATASET_SOURCES[dataset_name]
    try:
        if source["config"] is None:
            return load_dataset(source["hub"], split=split), source["hub"]
        return load_dataset(source["hub"], source["config"], split=split), source["hub"]
    except RuntimeError as exc:
        fallback_hub = source["fallback_hub"]
        if fallback_hub and "Dataset scripts are no longer supported" in str(exc):
            if log_fallback:
                print(f"Dataset '{source['hub']}' uses a dataset script. Falling back to '{fallback_hub}'.")
            if source["fallback_config"] is None:
                return load_dataset(fallback_hub, split=split), fallback_hub
            return load_dataset(fallback_hub, source["fallback_config"], split=split), fallback_hub
        raise


def infer_schema(dataset_name: str, dataset: Any, *, require_answer_field: bool) -> dict[str, str | None]:
    """Return the fixed schema for one supported VQA dataset and validate it against the loaded split."""
    dataset_name = canonical_dataset_name(dataset_name)
    schema = DATASET_SCHEMAS[dataset_name].copy()
    if require_answer_field and schema["answer_field"] is None:
        raise ValueError(f"Dataset '{dataset_name}' does not define an answer field.")

    all_fields = set(dataset.features)
    missing_fields = sorted(field for field in schema.values() if field is not None and field not in all_fields)
    if missing_fields:
        raise ValueError(
            f"Dataset '{dataset_name}' schema changed. Missing fields: {missing_fields}. "
            f"Available fields: {sorted(all_fields)}"
        )
    return schema


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
    "DATASET_ALIASES",
    "DATASET_SCHEMAS",
    "DATASET_SOURCES",
    "canonical_dataset_name",
    "extract_image_as_pil",
    "infer_schema",
    "load_dataset_split",
    "pick_first_text",
]
