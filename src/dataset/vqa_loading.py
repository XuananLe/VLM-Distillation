from typing import Any

from datasets import load_dataset
from PIL import Image

DATASETS = {
    "textvqa": {
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


def load_dataset_split(
    dataset_name: str,
    split: str,
):
    if dataset_name not in DATASETS:
        supported = ", ".join(sorted(DATASETS))
        raise ValueError(f"Unsupported dataset '{dataset_name}'. Supported: {supported}")
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
    if value is None or isinstance(value, (list, tuple, dict)):
        return None
    text = str(value).strip()
    return text or None


def extract_image_as_pil(image_value: Image.Image) -> Image.Image:
    return image_value.convert("RGB")


__all__ = [
    "extract_image_as_pil",
    "load_dataset_split",
    "pick_first_text",
]
