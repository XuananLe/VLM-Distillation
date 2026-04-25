from __future__ import annotations

import io
import re
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

QUESTION_NAME_PRIORITY = ("question", "query", "prompt")
ANSWER_NAME_PRIORITY = ("answers", "answer", "label", "target")
ID_NAME_PRIORITY = ("question_id", "questionid", "id", "docid", "image_id")
IMAGE_NAME_PRIORITY = ("image_id", "docid")


def canonical_dataset_name(dataset_name: str) -> str:
    """Normalize one user dataset name or alias to the repo's canonical dataset key."""
    key = dataset_name.strip().lower()
    canonical = DATASET_ALIASES.get(key)
    if canonical is None:
        supported = ", ".join(sorted(DATASET_SOURCES))
        raise ValueError(f"Unsupported dataset '{dataset_name}'. Supported: {supported}")
    return canonical


def load_hf_dataset(dataset_id: str, config: str | None, split: str):
    """Load one Hugging Face dataset split with an optional config name."""
    if config is None:
        return load_dataset(dataset_id, split=split)
    return load_dataset(dataset_id, config, split=split)


def load_dataset_split(
    dataset_name: str,
    split: str,
    *,
    log_fallback: bool = False,
):
    """Load one VQA dataset split and fall back when the primary hub entry requires a script."""
    source = DATASET_SOURCES[dataset_name]
    try:
        return load_hf_dataset(source["hub"], source["config"], split), source["hub"]
    except RuntimeError as exc:
        fallback_hub = source["fallback_hub"]
        if fallback_hub and "Dataset scripts are no longer supported" in str(exc):
            if log_fallback:
                print(
                    f"Dataset '{source['hub']}' uses a dataset script. "
                    f"Falling back to '{fallback_hub}'."
                )
            return load_hf_dataset(fallback_hub, source["fallback_config"], split), fallback_hub
        raise


def normalize_name(name: str) -> str:
    """Normalize a schema field name for fuzzy field matching."""
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def is_string_feature(feature: Any) -> bool:
    """Return whether a datasets feature behaves like a scalar string field."""
    return getattr(feature, "dtype", None) == "string"


def is_sequence_of_strings_feature(feature: Any) -> bool:
    """Return whether a datasets feature is a list-like container of strings."""
    if feature.__class__.__name__ not in {"Sequence", "List", "LargeList"}:
        return False
    inner_feature = getattr(feature, "feature", None)
    if inner_feature is None:
        return False
    return is_string_feature(inner_feature)


def is_image_feature(feature: Any) -> bool:
    """Return whether a datasets feature is an image field."""
    return feature.__class__.__name__ == "Image" or getattr(feature, "dtype", None) == "image"


def pick_exact_name(fields: list[str], priority: tuple[str, ...]) -> str | None:
    """Pick the first exact field-name match from a priority list after normalization."""
    normalized_fields = {normalize_name(field): field for field in fields}
    for target in priority:
        matched = normalized_fields.get(normalize_name(target))
        if matched is not None:
            return matched
    return None


def pick_contains_name(fields: list[str], tokens: tuple[str, ...]) -> str | None:
    """Pick the first field whose normalized name contains one of the target tokens."""
    normalized_tokens = tuple(normalize_name(token) for token in tokens)
    for field in fields:
        normalized_field = normalize_name(field)
        if any(token in normalized_field for token in normalized_tokens):
            return field
    return None


def infer_schema(dataset: Any, *, require_answer_field: bool) -> dict[str, str | None]:
    """Infer image, question, answer, and id fields from a VQA-style dataset schema."""
    features = dataset.features
    all_fields = list(features.keys())
    image_fields = [name for name, feature in features.items() if is_image_feature(feature)]
    string_fields = [name for name, feature in features.items() if is_string_feature(feature)]
    string_sequence_fields = [
        name for name, feature in features.items() if is_sequence_of_strings_feature(feature)
    ]

    image_field = image_fields[0] if image_fields else ("image" if "image" in all_fields else None)
    if image_field is None:
        raise ValueError(f"Unable to infer image field from schema: {all_fields}")

    question_field = (
        pick_exact_name(string_fields, QUESTION_NAME_PRIORITY)
        or pick_contains_name(string_fields, QUESTION_NAME_PRIORITY)
    )
    if question_field is None:
        raise ValueError(f"Unable to infer question field from schema: {all_fields}")

    answer_field = (
        pick_exact_name(string_sequence_fields, ANSWER_NAME_PRIORITY)
        or pick_exact_name(string_fields, ANSWER_NAME_PRIORITY)
        or pick_contains_name(string_sequence_fields, ANSWER_NAME_PRIORITY)
        or pick_contains_name(string_fields, ANSWER_NAME_PRIORITY)
    )
    if require_answer_field and answer_field is None:
        raise ValueError(f"Unable to infer answer field from schema: {all_fields}")

    id_field = pick_exact_name(all_fields, ID_NAME_PRIORITY) or pick_contains_name(all_fields, ("id",))
    image_name_field = (
        pick_exact_name(all_fields, IMAGE_NAME_PRIORITY)
        or pick_contains_name(all_fields, ("imageid",))
        or id_field
    )

    return {
        "image_field": image_field,
        "question_field": question_field,
        "answer_field": answer_field,
        "id_field": id_field,
        "image_name_field": image_name_field,
    }


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
    "ANSWER_NAME_PRIORITY",
    "DATASET_ALIASES",
    "DATASET_SOURCES",
    "ID_NAME_PRIORITY",
    "IMAGE_NAME_PRIORITY",
    "QUESTION_NAME_PRIORITY",
    "canonical_dataset_name",
    "extract_image_as_pil",
    "infer_schema",
    "load_dataset_split",
    "load_hf_dataset",
    "pick_contains_name",
    "pick_exact_name",
    "pick_first_text",
]
