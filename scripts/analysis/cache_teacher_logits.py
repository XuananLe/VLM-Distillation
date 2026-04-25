import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
from torch.utils.data import Dataset

from src.dataset.sft_data import make_supervised_data_module
from src.params import DataArguments
from src.trainer.distillation_utils import (
    build_teacher_batches,
)
from src.trainer.setup_utils import normalize_teacher_models
from src.train.train_utils import (
    load_model_and_processor,
    load_processor_and_tokenizer,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cache supervised-token teacher logits for offline distillation."
    )
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
        default="{dataset_index}.pt",
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


class IndexedDataset(Dataset):
    def __init__(self, base_dataset: Dataset, limit: int | None = None):
        self.base_dataset = base_dataset
        self.limit = len(base_dataset) if limit is None else min(limit, len(base_dataset))

    def __len__(self) -> int:
        return self.limit

    def __getitem__(self, index: int):
        item = self.base_dataset[index]
        item["dataset_index"] = index
        return item


def make_dataset_and_collator(
    args: argparse.Namespace,
    *,
    teacher_ids: list[str],
    teacher_processors,
):
    student_processor, _, _ = load_processor_and_tokenizer(
        args.student_model_id,
        padding_side="right",
    )
    if student_processor is None:
        raise ValueError(
            "Teacher-logit caching requires an AutoProcessor for the student dataset path, "
            f"but processor loading failed for {args.student_model_id!r}."
        )

    data_args = DataArguments(
        data_path=args.data_path,
        image_folder=args.image_folder,
        lazy_preprocess=True,
    )
    data_module = make_supervised_data_module(
        processor=student_processor,
        data_args=data_args,
        teacher_processors=teacher_processors,
    )
    indexed_dataset = IndexedDataset(data_module["train_dataset"], limit=args.limit)
    return teacher_ids, indexed_dataset, data_module["data_collator"]


def load_teachers_and_processors(teacher_ids: list[str], device: str):
    dtype_map = {
        "cuda": torch.bfloat16,
        "cpu": torch.float32,
    }
    torch_dtype = dtype_map["cuda"] if device.startswith("cuda") else dtype_map["cpu"]
    teacher_models = []
    teacher_processors = []
    for teacher_id in teacher_ids:
        teacher_model, teacher_processor, _, _ = load_model_and_processor(
            model_id=teacher_id,
            cache_dir=None,
            device=device,
            compute_dtype=torch_dtype,
            disable_flash_attn2=not device.startswith("cuda"),
        )
        teacher_models.append(teacher_model)
        teacher_processors.append(teacher_processor)
    teacher_models, _ = normalize_teacher_models(teacher_models, len(teacher_models))
    return teacher_models, teacher_processors


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
    torch.save(
        {
            "teacher_model_id": teacher_model_id,
            "storage_dtype": str(storage_dtype).replace("torch.", ""),
            "dataset_index": item["dataset_index"],
            "num_supervised_tokens": item["num_supervised_tokens"],
            "logits": item["logits"],
            "labels": item["labels"],
        },
        sample_path,
    )


def main() -> None:
    args = parse_args()
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is not available.")

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    storage_dtype = prepare_storage_dtype(args.dtype)

    teacher_ids = list(args.teacher_model_ids)
    teacher_models, teacher_processors = load_teachers_and_processors(teacher_ids, device=device)
    teacher_ids, dataset, data_collator = make_dataset_and_collator(
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
        teacher_batches = build_teacher_batches(batch, len(teacher_models))

        for teacher_idx, (teacher_model, (teacher_inputs, teacher_labels)) in enumerate(
            zip(teacher_models, teacher_batches)
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
            effective_labels = prepared_labels

            sample_mask = effective_labels[0].ne(-100)
            sample_logits = teacher_logits[0][sample_mask].to(
                dtype=storage_dtype,
                device="cpu",
            ).contiguous()
            sample_labels = effective_labels[0][sample_mask].to(device="cpu").contiguous()
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

            del teacher_outputs
            del teacher_logits
            del effective_labels

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
