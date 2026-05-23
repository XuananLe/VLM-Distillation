import argparse
import json
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
from PIL import Image
from safetensors.torch import save_file
from teacher_processor_encoders import encode_teacher_data
from torch.utils.data import Dataset

from src.constants import IGNORE_INDEX
from src.dataset.data_utils import pad_frames, pad_sequence
from src.dataset.supervised_data import make_supervised_data_module
from src.params import DataArguments
from src.train.model_setup import load_vlm_bundle

REQUIRED_TEACHER_INPUTS = ("input_ids", "attention_mask", "pixel_values")
OPTIONAL_TEACHER_INPUTS = ("pixel_attention_mask", "image_grid_thw", "image_flags", "image_sizes")


def get_processor_pad_token_id(processor) -> int:
    if isinstance(processor, dict):
        return int(processor["tokenizer"].pad_token_id)
    if hasattr(processor, "tokenizer") and processor.tokenizer.pad_token_id is not None:
        return int(processor.tokenizer.pad_token_id)
    if getattr(processor, "pad_token_id", None) is not None:
        return int(processor.pad_token_id)
    raise ValueError(f"Could not resolve pad_token_id for processor type {type(processor).__name__}.")


def load_record_images(record: dict, image_folder: str) -> list[Image.Image] | None:
    if "image" not in record:
        return None
    image_files = record["image"]
    if isinstance(image_files, str):
        image_files = [image_files]

    images = []
    for image_file in image_files:
        resolved_path = image_file
        if not os.path.exists(resolved_path):
            resolved_path = os.path.join(image_folder, image_file)
        images.append(Image.open(resolved_path).convert("RGB"))
    return images


def add_teacher_inputs(
    *,
    sample: dict[str, torch.Tensor],
    sources,
    images,
    teacher_processors,
) -> None:
    teacher_count = len(teacher_processors)
    for teacher_index, teacher_processor in enumerate(teacher_processors):
        teacher_data = encode_teacher_data(sources, images, teacher_processor)
        prefix = "teacher" if teacher_count == 1 else f"teacher_{teacher_index}"

        sample[f"{prefix}_input_ids"] = teacher_data["input_ids"]
        sample[f"{prefix}_labels"] = teacher_data["labels"]
        sample[f"{prefix}_attention_mask"] = teacher_data["attention_mask"]
        sample[f"{prefix}_pixel_values"] = teacher_data["pixel_values"]
        sample[f"{prefix}_pixel_attention_mask"] = teacher_data["pixel_attention_mask"]
        if teacher_data.get("image_sizes") is not None:
            sample[f"{prefix}_image_sizes"] = teacher_data["image_sizes"]
        if teacher_data.get("image_grid_thw") is not None:
            sample[f"{prefix}_image_grid_thw"] = teacher_data["image_grid_thw"]
        if teacher_data.get("image_flags") is not None:
            sample[f"{prefix}_image_flags"] = teacher_data["image_flags"]


class TeacherInputDataset(Dataset):
    def __init__(self, base_dataset: Dataset, teacher_processors, image_folder: str, limit: int | None = None):
        self.base_dataset = base_dataset
        self.teacher_processors = list(teacher_processors)
        self.image_folder = image_folder
        self.limit = len(base_dataset) if limit is None else min(limit, len(base_dataset))

    def __len__(self) -> int:
        return self.limit

    def __getitem__(self, index: int):
        item = self.base_dataset[index]
        record = self.base_dataset.training_records[index]
        add_teacher_inputs(
            sample=item,
            sources=record["conversations"],
            images=load_record_images(record, self.image_folder),
            teacher_processors=self.teacher_processors,
        )
        item["dataset_index"] = index
        return item


class CacheTeacherDataCollator:
    def __init__(self, student_collator, teacher_processors):
        self.student_collator = student_collator
        self.teacher_pad_token_ids = [get_processor_pad_token_id(processor) for processor in teacher_processors]

    def collate_teacher_batch(self, examples, batch_dict, prefix: str) -> None:
        teacher_pad = (
            self.teacher_pad_token_ids[0]
            if prefix == "teacher"
            else self.teacher_pad_token_ids[int(prefix.split("_")[1])]
        )
        teacher_input_ids = pad_sequence(
            [example[f"{prefix}_input_ids"] for example in examples],
            padding_side="right",
            padding_value=teacher_pad,
        )
        teacher_labels = pad_sequence(
            [example[f"{prefix}_labels"] for example in examples],
            padding_side="right",
            padding_value=IGNORE_INDEX,
        )
        teacher_attention_mask = pad_sequence(
            [example[f"{prefix}_attention_mask"] for example in examples],
            padding_side="right",
            padding_value=0,
        )
        batch_dict.update(
            {
                f"{prefix}_input_ids": teacher_input_ids,
                f"{prefix}_labels": teacher_labels,
                f"{prefix}_attention_mask": teacher_attention_mask,
            }
        )

        pixel_key = f"{prefix}_pixel_values"
        teacher_pixel_values = [example[pixel_key] for example in examples]
        if teacher_pixel_values[0].dim() == 5:
            batch_dict[pixel_key] = pad_frames(teacher_pixel_values, pad_value=0.0)
        else:
            batch_dict[pixel_key] = torch.cat(teacher_pixel_values, dim=0)

        pixel_attention_key = f"{prefix}_pixel_attention_mask"
        teacher_pixel_attention_masks = [example.get(pixel_attention_key) for example in examples]
        if teacher_pixel_attention_masks[0] is not None:
            batch_dict[pixel_attention_key] = pad_frames(teacher_pixel_attention_masks, pad_value=0)

        for suffix in ("image_grid_thw", "image_sizes", "image_flags"):
            key = f"{prefix}_{suffix}"
            if key in examples[0]:
                batch_dict[key] = torch.cat([example[key] for example in examples], dim=0)

    def __call__(self, examples):
        batch = self.student_collator(examples)
        if "teacher_input_ids" in examples[0]:
            teacher_prefixes = ["teacher"]
        else:
            teacher_prefixes = []
            for key in examples[0]:
                if key.startswith("teacher_") and key.endswith("_input_ids"):
                    teacher_prefixes.append(key.removesuffix("_input_ids"))
            teacher_prefixes.sort(key=lambda prefix: int(prefix.split("_")[1]))

        for prefix in teacher_prefixes:
            self.collate_teacher_batch(examples, batch, prefix)
        return batch


def normalize_teacher_models(teacher_models):
    """Freeze live teachers used only for offline logit-cache generation."""
    for model in teacher_models:
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
    return teacher_models


def build_live_teacher_batches(inputs, num_teachers: int):
    """Collect live-teacher input batches for one cache-generation step."""
    prefixes = []
    if "teacher_input_ids" in inputs:
        prefixes.append("teacher")
    prefixes.extend(f"teacher_{index}" for index in range(num_teachers) if f"teacher_{index}_input_ids" in inputs)

    if not prefixes:
        raise ValueError("No live teacher inputs were found in the batch.")

    batches = []
    for prefix in prefixes:
        teacher_inputs = {}
        for suffix in REQUIRED_TEACHER_INPUTS:
            teacher_inputs[suffix] = inputs[f"{prefix}_{suffix}"]
        for suffix in OPTIONAL_TEACHER_INPUTS:
            key = f"{prefix}_{suffix}"
            if key in inputs:
                teacher_inputs[suffix] = inputs[key]
        batches.append((teacher_inputs, inputs[f"{prefix}_labels"]))
    return batches


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache supervised-token teacher logits for offline distillation.")
    parser.add_argument(
        "--student-model-id",
        default="HuggingFaceTB/SmolVLM-500M-Instruct",
        help="Student processor used to build the dataset module.",
    )
    parser.add_argument(
        "--teacher-model-ids",
        required=True,
        nargs="+",
        help="Teacher model IDs passed as repeated values.",
    )
    parser.add_argument("--data-path", required=True, help="Path to training LLaVA JSON.")
    parser.add_argument("--image-folder", required=True, help="Image root used by the dataset.")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where teacher-specific shard files will be written.",
    )
    parser.add_argument(
        "--file-name-template",
        default="{dataset_index}.safetensors",
        help="Output filename template for each sample. Available field: dataset_index.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional sample limit for smoke tests.",
    )
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
        help="Storage dtype for cached logits.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device used for teacher forward passes.",
    )
    return parser.parse_args()


def build_cache_dataset_state(
    args: argparse.Namespace,
    *,
    teacher_ids: list[str],
    teacher_processors,
):
    _, student_processor, _, _ = load_vlm_bundle(
        model_id=args.student_model_id,
        padding_side="right",
        load_model=False,
    )
    if student_processor is None:
        raise ValueError(
            "Teacher-logit caching requires a processor/tokenizer bundle for the student dataset path, "
            f"but processor loading failed for {args.student_model_id!r}."
        )

    data_args = DataArguments(
        data_path=args.data_path,
        image_folder=args.image_folder,
    )
    data_module = make_supervised_data_module(
        processor=student_processor,
        data_args=data_args,
    )
    dataset = TeacherInputDataset(
        data_module["train_dataset"],
        teacher_processors=teacher_processors,
        image_folder=args.image_folder,
        limit=args.limit,
    )
    return teacher_ids, dataset, CacheTeacherDataCollator(data_module["data_collator"], teacher_processors)


def load_teacher_bundles(teacher_ids: list[str], device: str):
    dtype_map = {
        "cuda": torch.bfloat16,
        "cpu": torch.float32,
    }
    torch_dtype = dtype_map["cuda"] if device.startswith("cuda") else dtype_map["cpu"]
    teacher_models = []
    teacher_processors = []
    for teacher_id in teacher_ids:
        teacher_model, teacher_processor, _, _ = load_vlm_bundle(
            model_id=teacher_id,
            cache_dir=None,
            device=device,
            compute_dtype=torch_dtype,
            disable_flash_attn2=not device.startswith("cuda"),
        )
        teacher_models.append(teacher_model)
        teacher_processors.append(teacher_processor)
    return normalize_teacher_models(teacher_models), teacher_processors


def prepare_storage_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def save_sample(
    *,
    output_root: Path,
    teacher_slug: str,
    teacher_model_id: str,
    storage_dtype: torch.dtype,
    file_name_template: str,
    item: dict,
) -> None:
    teacher_dir = output_root / teacher_slug
    teacher_dir.mkdir(parents=True, exist_ok=True)
    file_name = file_name_template.format(dataset_index=item["dataset_index"])
    sample_path = teacher_dir / file_name
    metadata = {
        "teacher_model_id": teacher_model_id,
        "storage_dtype": str(storage_dtype).replace("torch.", ""),
        "dataset_index": str(item["dataset_index"]),
        "num_supervised_tokens": str(item["num_supervised_tokens"]),
    }
    if sample_path.suffix == ".safetensors":
        save_file(
            {
                "logits": item["logits"],
                "labels": item["labels"],
            },
            sample_path,
            metadata=metadata,
        )
        return
    if sample_path.suffix == ".pt":
        torch.save(
            {
                **metadata,
                "dataset_index": item["dataset_index"],
                "num_supervised_tokens": item["num_supervised_tokens"],
                "logits": item["logits"],
                "labels": item["labels"],
            },
            sample_path,
        )
        return
    raise ValueError(f"Unsupported cache sample extension. Use .safetensors or .pt: {sample_path}")


def main() -> None:
    args = parse_args()
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is not available.")

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    storage_dtype = prepare_storage_dtype(args.dtype)

    teacher_ids = list(args.teacher_model_ids)
    teacher_models, teacher_processors = load_teacher_bundles(teacher_ids, device=device)
    teacher_ids, dataset, data_collator = build_cache_dataset_state(
        args,
        teacher_ids=teacher_ids,
        teacher_processors=teacher_processors,
    )

    teacher_slugs = [teacher_id.split("/")[-1] for teacher_id in teacher_ids]
    total_samples = 0
    total_supervised_tokens = [0 for _ in teacher_ids]

    for dataset_index in range(len(dataset)):
        example = dataset[dataset_index]
        original_dataset_index = int(example.pop("dataset_index"))
        batch = data_collator([example])
        live_teacher_batches = build_live_teacher_batches(batch, len(teacher_models))

        for teacher_idx, (teacher_model, (teacher_inputs, teacher_labels)) in enumerate(
            zip(teacher_models, live_teacher_batches)
        ):
            model_param = next(teacher_model.parameters())
            model_dtype = model_param.dtype if model_param.is_floating_point() else None
            prepared_inputs = {}
            for key, value in teacher_inputs.items():
                if torch.is_tensor(value):
                    target_dtype = model_dtype if model_dtype is not None and value.is_floating_point() else value.dtype
                    prepared_inputs[key] = value.to(
                        device=model_param.device,
                        dtype=target_dtype,
                    )
                else:
                    prepared_inputs[key] = value
            prepared_labels = teacher_labels.to(model_param.device)

            with torch.no_grad():
                teacher_outputs = teacher_model(
                    **prepared_inputs,
                    return_dict=True,
                    output_hidden_states=False,
                )
            teacher_logits = teacher_outputs.logits.detach()
            supervised_labels = prepared_labels

            sample_mask = supervised_labels[0].ne(-100)
            sample_logits = (
                teacher_logits[0][sample_mask]
                .to(
                    dtype=storage_dtype,
                    device="cpu",
                )
                .contiguous()
            )
            sample_labels = supervised_labels[0][sample_mask].to(device="cpu").contiguous()
            total_supervised_tokens[teacher_idx] += int(sample_mask.sum().item())
            save_sample(
                output_root=output_root,
                teacher_slug=teacher_slugs[teacher_idx],
                teacher_model_id=teacher_ids[teacher_idx],
                storage_dtype=storage_dtype,
                file_name_template=args.file_name_template,
                item={
                    "dataset_index": original_dataset_index,
                    "num_supervised_tokens": int(sample_logits.shape[0]),
                    "logits": sample_logits,
                    "labels": sample_labels,
                },
            )

            teacher_outputs = None
            teacher_logits = None
            supervised_labels = None

        total_samples += 1
        print(
            json.dumps(
                {
                    "processed_samples": total_samples,
                    "limit": len(dataset),
                    "teacher_supervised_tokens": total_supervised_tokens,
                }
            ),
            flush=True,
        )

    metadata = {
        "student_model_id": args.student_model_id,
        "teacher_model_ids": teacher_ids,
        "data_path": args.data_path,
        "image_folder": args.image_folder,
        "num_samples": len(dataset),
        "storage_dtype": args.dtype,
        "processing_mode": "single_sample",
        "file_name_template": args.file_name_template,
        "teacher_supervised_tokens": {
            teacher_ids[idx]: total_supervised_tokens[idx] for idx in range(len(teacher_ids))
        },
    }
    (output_root / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
