import json
import argparse
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm


DEFAULT_DATASET_NAMES = {
    "chartqa": "HuggingFaceM4/ChartQA",
    "documentvqa": "HuggingFaceM4/DocumentVQA",
}
DEFAULT_OUTPUT_JSON_NAME = "train_llava.json"

DATASET_SPLIT_ALIASES = {
    "chartqa": {"validation": "val"},
    "documentvqa": {"val": "validation"},
}

VALID_SPLITS = {
    "chartqa": {"train", "val", "test"},
    "documentvqa": {"train", "validation", "test"},
}


def _normalize_split(dataset_type: str, split: str) -> str:
    split = split.strip().lower()
    split_aliases = DATASET_SPLIT_ALIASES.get(dataset_type, {})
    split = split_aliases.get(split, split)

    if split not in VALID_SPLITS[dataset_type]:
        supported = ", ".join(sorted(VALID_SPLITS[dataset_type]))
        raise ValueError(
            f"Unsupported split '{split}' for dataset '{dataset_type}'. "
            f"Supported splits: {supported}"
        )
    return split


def _format_answers(answers, use_first_answer_only: bool = False) -> str:
    if isinstance(answers, list):
        cleaned_answers = [str(answer).strip() for answer in answers if str(answer).strip()]
        if not cleaned_answers:
            return ""
        if use_first_answer_only:
            return cleaned_answers[0]
        return " or ".join(cleaned_answers) if len(cleaned_answers) > 1 else cleaned_answers[0]

    if answers is None:
        return ""
    return str(answers).strip()


def _save_image(image, images_dir: Path, split: str, idx: int, image_format: str) -> str:
    image_format = image_format.lower()
    image_filename = f"{split}_{idx}.{'jpg' if image_format == 'jpg' else 'png'}"

    if image_format == "jpg" and image.mode in ("RGBA", "LA", "P"):
        image = image.convert("RGB")

    image_path = images_dir / image_filename
    image.save(image_path)

    return f"images/{image_filename}"


def convert_chartqa_to_llava(
    out_root: str,
    split: str = "train",
    dataset_name: str = "HuggingFaceM4/ChartQA",
    use_first_answer_only: bool = False,
    image_format: str = "jpg"
):
    """
    Convert ChartQA dataset to LLaVA JSON format.

    Args:
        out_root: Root output directory (will create images/ subdir and output.json)
        split: Dataset split to use ("train", "val", or "test")
        dataset_name: HuggingFace dataset identifier
        use_first_answer_only: If True, use only first answer; else join all answers
        image_format: Image format to save (jpg or png)
    """
    split = _normalize_split("chartqa", split)

    print(f"Loading ChartQA dataset (split: {split})")
    dataset = load_dataset(dataset_name, split=split)

    print(f"Loaded {len(dataset)} samples from {split} split")

    # Create output directories
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    images_dir = out_root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    llava_data = []

    for idx, sample in enumerate(tqdm(dataset, desc="Converting")):
        # Generate unique ID
        sample_id = f"chartqa_{split}_{idx}"

        # Extract answer
        answer_text = _format_answers(sample.get("label"), use_first_answer_only=use_first_answer_only)

        # Save image
        image_relative_path = _save_image(
            image=sample["image"],
            images_dir=images_dir,
            split=split,
            idx=idx,
            image_format=image_format,
        )

        # Create LLaVA format entry
        llava_entry = {
            "id": sample_id,
            "image": image_relative_path,
            "conversations": [
                {
                    "from": "human",
                    "value": f"<image>\n{sample['query']}"
                },
                {
                    "from": "gpt",
                    "value": answer_text
                }
            ]
        }

        # Add metadata
        llava_entry['source'] = 'chartqa'
        llava_entry['human_or_machine'] = 'human' if sample.get('human_or_machine') == 0 else 'machine'

        llava_data.append(llava_entry)

    # Save to JSON as train_llava.json
    output_json = out_root / DEFAULT_OUTPUT_JSON_NAME
    print(f"\nSaving {len(llava_data)} examples to {output_json}")

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(llava_data, f, indent=2, ensure_ascii=False)

    print(f"✓ Conversion complete!")
    print(f"  Output directory: {out_root}")
    print(f"  JSON: {output_json}")
    print(f"  Images: {images_dir}")
    print(f"  Total samples: {len(llava_data)}")

    # Print statistics
    if 'human_or_machine' in dataset.features:
        human_count = sum(1 for s in dataset if s.get('human_or_machine') == 0)
        machine_count = len(dataset) - human_count
        print(f"\nData source distribution:")
        print(f"  Human-generated: {human_count}")
        print(f"  Machine-generated: {machine_count}")


def convert_documentvqa_to_llava(
    out_root: str,
    split: str = "train",
    dataset_name: str = "HuggingFaceM4/DocumentVQA",
    use_first_answer_only: bool = False,
    image_format: str = "jpg"
):
    """
    Convert DocumentVQA dataset to LLaVA JSON format.

    Args:
        out_root: Root output directory (will create images/ subdir and output.json)
        split: Dataset split to use ("train", "validation", or "test")
        dataset_name: HuggingFace dataset identifier
        use_first_answer_only: If True, use only first answer; else join all answers
        image_format: Image format to save (jpg or png)
    """
    split = _normalize_split("documentvqa", split)

    print(f"Loading DocumentVQA dataset (split: {split})")
    dataset = load_dataset(dataset_name, split=split)

    print(f"Loaded {len(dataset)} samples from {split} split")

    # Create output directories
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    images_dir = out_root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    llava_data = []

    for idx, sample in enumerate(tqdm(dataset, desc="Converting")):
        # Generate unique ID
        question_id = sample.get("questionId")
        sample_id = f"documentvqa_{split}_{question_id if question_id is not None else idx}"

        # Extract answer
        answer_text = _format_answers(sample.get("answers"), use_first_answer_only=use_first_answer_only)

        # Save image
        image_relative_path = _save_image(
            image=sample["image"],
            images_dir=images_dir,
            split=split,
            idx=idx,
            image_format=image_format,
        )

        # Create LLaVA format entry
        llava_entry = {
            "id": sample_id,
            "image": image_relative_path,
            "conversations": [
                {
                    "from": "human",
                    "value": f"<image>\n{sample['question']}"
                },
                {
                    "from": "gpt",
                    "value": answer_text
                }
            ],
            "source": "documentvqa",
            "question_id": question_id,
            "doc_id": sample.get("docId"),
            "ucsf_document_id": sample.get("ucsf_document_id"),
            "ucsf_document_page_no": sample.get("ucsf_document_page_no"),
            "question_types": sample.get("question_types", []),
        }

        llava_data.append(llava_entry)

    # Save to JSON as train_llava.json
    output_json = out_root / DEFAULT_OUTPUT_JSON_NAME
    print(f"\nSaving {len(llava_data)} examples to {output_json}")

    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(llava_data, f, indent=2, ensure_ascii=False)

    print(f"✓ Conversion complete!")
    print(f"  Output directory: {out_root}")
    print(f"  JSON: {output_json}")
    print(f"  Images: {images_dir}")
    print(f"  Total samples: {len(llava_data)}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace datasets to LLaVA JSON format"
    )
    parser.add_argument(
        "out_root",
        type=str,
        help="Root output directory (will create images/ subdir and {split}_llava.json)"
    )
    parser.add_argument(
        "--dataset-type",
        type=str,
        default="chartqa",
        choices=["chartqa", "documentvqa"],
        help="Dataset conversion format (default: chartqa)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "validation", "test"],
        help="Dataset split to use (aliases: val <-> validation)"
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="HuggingFace dataset identifier (default depends on dataset-type)"
    )
    parser.add_argument(
        "--first-answer-only",
        action="store_true",
        help="Use only the first answer instead of joining multiple answers"
    )
    parser.add_argument(
        "--image-format",
        type=str,
        default="jpg",
        choices=["jpg", "png"],
        help="Image format to save (default: jpg)"
    )

    args = parser.parse_args()
    dataset_name = args.dataset_name or DEFAULT_DATASET_NAMES[args.dataset_type]

    if args.dataset_type == "chartqa":
        convert_chartqa_to_llava(
            out_root=args.out_root,
            split=args.split,
            dataset_name=dataset_name,
            use_first_answer_only=args.first_answer_only,
            image_format=args.image_format
        )
    elif args.dataset_type == "documentvqa":
        convert_documentvqa_to_llava(
            out_root=args.out_root,
            split=args.split,
            dataset_name=dataset_name,
            use_first_answer_only=args.first_answer_only,
            image_format=args.image_format
        )


if __name__ == "__main__":
    main()
