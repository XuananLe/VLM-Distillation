from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
import torch.nn.functional as F
from teacher_processor_encoders import encode_teacher_data
from torch.utils.data import DataLoader, Dataset

from src.components.loss import uld_loss
from src.constants import IGNORE_INDEX
from src.dataset.data_collator import DataCollatorForSupervisedDataset
from src.dataset.data_utils import pad_frames, pad_sequence
from src.dataset.smolvlm_encoder import smolvlm_encode_conversation
from src.dataset.vqa_loading import (
    canonical_dataset_name,
    extract_image_as_pil,
    load_dataset_split,
    pick_first_text,
)
from src.train.model_setup import (
    load_vlm_bundle,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute CE vs KD gradient agreement on a VQA subset.")
    parser.add_argument("--student-model-id", required=True, help="Student model id/path.")
    parser.add_argument("--teacher-model-id", required=True, help="Teacher model id/path.")
    parser.add_argument("--dataset", default="docvqa", help="Dataset alias/name.")
    parser.add_argument("--split", default="train", help="Dataset split.")
    parser.add_argument("--subset-size", type=int, default=1000, help="Number of valid samples to evaluate.")
    parser.add_argument("--offset", type=int, default=0, help="Starting raw row offset before filtering.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for the agreement run.")
    parser.add_argument("--student-temperature", type=float, default=2.0, help="Student KD temperature.")
    parser.add_argument("--teacher-temperature", type=float, default=2.0, help="Teacher KD temperature.")
    parser.add_argument(
        "--loss-function",
        default="uld_loss",
        help="Compatibility option. This script only supports `uld_loss`.",
    )
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        default="bfloat16",
        help="Model compute dtype.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
    )
    parser.add_argument(
        "--attn-implementation",
        default="flash_attention_2",
        help="Transformers attention implementation.",
    )
    parser.add_argument("--cache-dir", default=None, help="Optional HF cache dir.")
    parser.add_argument(
        "--output-json",
        default="artifacts/gradient_agreement/docvqa/qwen2_vl_2b_vs_smolvlm500m_1000samples/report.json",
        help="Where to write the JSON report.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Print progress every N batches.",
    )
    return parser.parse_args()


def resolve_dtype(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping[name]


def normalize_device(device: str) -> str:
    if device == "cuda":
        return "cuda:0"
    return device


def get_processor_pad_token_id(processor: Any) -> int:
    if hasattr(processor, "tokenizer") and processor.tokenizer.pad_token_id is not None:
        return int(processor.tokenizer.pad_token_id)
    if getattr(processor, "pad_token_id", None) is not None:
        return int(processor.pad_token_id)
    raise ValueError(f"Could not resolve pad_token_id for processor type {type(processor).__name__}.")


class GradientAgreementCollator:
    def __init__(self, *, student_pad_token_id: int, teacher_pad_token_id: int):
        self.student_collator = DataCollatorForSupervisedDataset(pad_token_id=student_pad_token_id)
        self.teacher_pad_token_id = teacher_pad_token_id

    def __call__(self, examples):
        batch = self.student_collator(examples)
        batch["teacher_input_ids"] = pad_sequence(
            [example["teacher_input_ids"] for example in examples],
            padding_side="right",
            padding_value=self.teacher_pad_token_id,
        )
        batch["teacher_labels"] = pad_sequence(
            [example["teacher_labels"] for example in examples],
            padding_side="right",
            padding_value=IGNORE_INDEX,
        )
        batch["teacher_attention_mask"] = pad_sequence(
            [example["teacher_attention_mask"] for example in examples],
            padding_side="right",
            padding_value=0,
        )

        teacher_pixel_values = [example["teacher_pixel_values"] for example in examples]
        if teacher_pixel_values[0].dim() == 5:
            batch["teacher_pixel_values"] = pad_frames(teacher_pixel_values, pad_value=0.0)
        else:
            batch["teacher_pixel_values"] = torch.cat(teacher_pixel_values, dim=0)

        teacher_pixel_attention_masks = [example.get("teacher_pixel_attention_mask") for example in examples]
        if teacher_pixel_attention_masks[0] is not None:
            batch["teacher_pixel_attention_mask"] = pad_frames(teacher_pixel_attention_masks, pad_value=0)

        for suffix in ("image_grid_thw", "image_sizes", "image_flags"):
            key = f"teacher_{suffix}"
            if key in examples[0]:
                batch[key] = torch.cat([example[key] for example in examples], dim=0)
        return batch


def pick_first_answer(answer_value: Any) -> str | None:
    values = answer_value if isinstance(answer_value, (list, tuple)) else (answer_value,)
    for value in values:
        text = pick_first_text(value)
        if text:
            return text
    return None


@dataclass
class SelectedSample:
    row_index: int
    question: str
    answer: str


class DocVQAGradientAgreementDataset(Dataset):
    def __init__(
        self,
        hf_dataset,
        selected_samples: list[SelectedSample],
        schema: dict[str, str | None],
        student_processor,
        teacher_processor,
    ):
        self.hf_dataset = hf_dataset
        self.selected_samples = selected_samples
        self.schema = schema
        self.student_processor = student_processor
        self.teacher_processor = teacher_processor

    def __len__(self) -> int:
        return len(self.selected_samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        selected = self.selected_samples[index]
        sample = self.hf_dataset[selected.row_index]
        image = extract_image_as_pil(sample[self.schema["image_field"]])
        sources = [
            {"from": "human", "value": f"<image>\n{selected.question}"},
            {"from": "gpt", "value": selected.answer},
        ]

        encoded_sample = smolvlm_encode_conversation(
            sources,
            [image],
            self.student_processor,
        )
        if encoded_sample["pixel_values"] is None:
            raise ValueError("Student encoder did not produce image tensors for an image-only sample.")

        teacher_data = encode_teacher_data(sources, [image], self.teacher_processor)
        encoded_sample["teacher_input_ids"] = teacher_data["input_ids"]
        encoded_sample["teacher_labels"] = teacher_data["labels"]
        encoded_sample["teacher_attention_mask"] = teacher_data["attention_mask"]
        encoded_sample["teacher_pixel_values"] = teacher_data["pixel_values"]
        encoded_sample["teacher_pixel_attention_mask"] = teacher_data["pixel_attention_mask"]
        if teacher_data.get("image_sizes") is not None:
            encoded_sample["teacher_image_sizes"] = teacher_data["image_sizes"]
        if teacher_data.get("image_grid_thw") is not None:
            encoded_sample["teacher_image_grid_thw"] = teacher_data["image_grid_thw"]
        if teacher_data.get("image_flags") is not None:
            encoded_sample["teacher_image_flags"] = teacher_data["image_flags"]
        return encoded_sample


def select_docvqa_subset(dataset_name: str, split: str, subset_size: int, offset: int):
    hf_dataset, loaded_from, schema = load_dataset_split(dataset_name, split)
    selected_samples: list[SelectedSample] = []
    for row_index in range(offset, len(hf_dataset)):
        sample = hf_dataset[row_index]
        question = pick_first_text(sample.get(schema["question_field"])) if schema["question_field"] else None
        answer = pick_first_answer(sample.get(schema["answer_field"])) if schema["answer_field"] else None
        if not question or not answer:
            continue
        selected_samples.append(
            SelectedSample(
                row_index=row_index,
                question=question,
                answer=answer,
            )
        )
        if len(selected_samples) >= subset_size:
            break
    if len(selected_samples) < subset_size:
        raise ValueError(
            f"Requested {subset_size} valid samples from {dataset_name}:{split} starting at offset {offset}, "
            f"but only found {len(selected_samples)}."
        )
    return hf_dataset, loaded_from, schema, selected_samples


def load_vlm_runtime(
    model_id: str,
    *,
    device: str,
    dtype: torch.dtype,
    attn_implementation: str,
    cache_dir: str | None,
):
    model, processor, _, _ = load_vlm_bundle(
        model_id,
        cache_dir=cache_dir,
        compute_dtype=dtype,
        device=device,
        disable_flash_attn2=attn_implementation != "flash_attention_2",
        padding_side="right",
        attn_implementation=attn_implementation,
        model_kwargs={
            "low_cpu_mem_usage": True,
        },
    )
    model.eval()
    return model, processor


def move_batch_to_model_device(model, batch: dict[str, Any]) -> dict[str, Any]:
    parameter = next(model.parameters())
    model_device = parameter.device
    model_dtype = parameter.dtype
    moved = {}
    for key, value in batch.items():
        if value is None:
            continue
        if not isinstance(value, torch.Tensor):
            moved[key] = value
            continue
        tensor = value.to(model_device)
        if torch.is_floating_point(tensor):
            tensor = tensor.to(model_dtype)
        moved[key] = tensor
    return moved


def split_teacher_batch(batch: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], torch.Tensor]:
    student_inputs = {key: value for key, value in batch.items() if not key.startswith("teacher")}
    teacher_inputs = {
        "input_ids": batch["teacher_input_ids"],
        "attention_mask": batch["teacher_attention_mask"],
        "pixel_values": batch["teacher_pixel_values"],
    }
    if "teacher_pixel_attention_mask" in batch:
        teacher_inputs["pixel_attention_mask"] = batch["teacher_pixel_attention_mask"]
    if "teacher_image_grid_thw" in batch:
        teacher_inputs["image_grid_thw"] = batch["teacher_image_grid_thw"]
    if "teacher_image_flags" in batch:
        teacher_inputs["image_flags"] = batch["teacher_image_flags"]
    return student_inputs, teacher_inputs, batch["teacher_labels"]


def resample_sequence_to_length(sequence_logits: torch.Tensor, target_length: int) -> torch.Tensor:
    current_length = int(sequence_logits.size(0))
    if current_length == target_length:
        return sequence_logits
    if current_length == 0:
        raise ValueError("Cannot resample an empty sequence.")
    if target_length == 0:
        raise ValueError("Cannot resample to an empty sequence.")
    if current_length == 1:
        return sequence_logits.expand(target_length, -1)

    # Align teacher answer progress to the student answer span without assuming
    # token-by-token semantic correspondence across tokenizers.
    sequence = sequence_logits.transpose(0, 1).unsqueeze(0).float()
    resized = F.interpolate(
        sequence,
        size=target_length,
        mode="linear",
        align_corners=True,
    )
    return resized.squeeze(0).transpose(0, 1).to(sequence_logits.dtype)


def compute_single_teacher_uld_loss(
    *,
    student_logits: torch.Tensor,
    student_labels: torch.Tensor,
    teacher_logits: torch.Tensor,
    teacher_labels: torch.Tensor,
    student_temperature: float,
    teacher_temperature: float,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    student_mask = student_labels != -100
    teacher_mask = teacher_labels != -100

    sample_losses = []
    student_answer_positions = []
    for sample_index in range(student_logits.size(0)):
        student_positions = student_mask[sample_index].nonzero(as_tuple=False).flatten()
        teacher_positions = teacher_mask[sample_index].nonzero(as_tuple=False).flatten()
        student_answer_positions.append(student_positions)

        if student_positions.numel() == 0 or teacher_positions.numel() == 0:
            raise ValueError(f"Student and teacher labels have no supervised answer tokens for sample {sample_index}.")

        student_slice = student_logits[sample_index, student_positions]
        teacher_slice = teacher_logits[sample_index, teacher_positions]
        teacher_slice = resample_sequence_to_length(teacher_slice, student_slice.size(0))
        sample_losses.append(
            uld_loss(
                student_logits=student_slice,
                teacher_logits=teacher_slice,
                student_temperature=student_temperature,
                teacher_temperature=teacher_temperature,
            )
        )
    if not sample_losses:
        raise ValueError("Cannot compute ULD loss for an empty student batch.")
    return torch.stack(sample_losses).mean(), student_answer_positions


def compute_sample_cosines(
    g_ce: torch.Tensor,
    g_kd: torch.Tensor,
    student_answer_positions: list[torch.Tensor],
) -> list[float]:
    cosines = []
    for sample_index, positions in enumerate(student_answer_positions):
        if positions.numel() == 0:
            continue
        ce_vector = g_ce[sample_index, positions].float().reshape(-1)
        kd_vector = g_kd[sample_index, positions].float().reshape(-1)
        ce_norm = ce_vector.norm()
        kd_norm = kd_vector.norm()
        if ce_norm.item() == 0.0 or kd_norm.item() == 0.0:
            continue
        cosine = F.cosine_similarity(
            ce_vector.unsqueeze(0),
            kd_vector.unsqueeze(0),
            dim=1,
            eps=1e-8,
        )[0]
        cosines.append(float(cosine.item()))
    return cosines


def build_report(
    *,
    args: argparse.Namespace,
    loaded_from: str,
    num_batches: int,
    num_selected_samples: int,
    num_cosine_samples: int,
    sample_cosines: list[float],
    batch_cosines: list[float],
    ce_losses: list[float],
    kd_losses: list[float],
    elapsed_seconds: float,
) -> dict[str, Any]:
    if not sample_cosines:
        raise RuntimeError("No valid gradient cosine measurements were produced.")

    def summarize(values: list[float]) -> dict[str, float]:
        if not values:
            return {}
        return {
            "mean": float(statistics.fmean(values)),
            "std": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
            "median": float(statistics.median(values)),
            "min": float(min(values)),
            "max": float(max(values)),
        }

    report = {
        "student_model_id": args.student_model_id,
        "teacher_model_id": args.teacher_model_id,
        "dataset": args.dataset,
        "split": args.split,
        "loaded_from": loaded_from,
        "subset_size_requested": args.subset_size,
        "subset_size_selected": num_selected_samples,
        "subset_size_with_cosine": num_cosine_samples,
        "offset": args.offset,
        "batch_size": args.batch_size,
        "loss_function": args.loss_function,
        "temperature": args.temperature,
        "dtype": args.dtype,
        "device": args.device,
        "attn_implementation": args.attn_implementation,
        "alignment_strategy": (
            "teacher supervised answer span is resampled to the student supervised answer length; "
            "cosine is computed on the full student supervised answer region"
        ),
        "num_batches": num_batches,
        "elapsed_seconds": elapsed_seconds,
        "sample_cosine": summarize(sample_cosines),
        "batch_cosine": summarize(batch_cosines),
        "ce_loss": summarize(ce_losses),
        "kd_loss": summarize(kd_losses),
        "positive_sample_fraction": float(sum(value > 0 for value in sample_cosines) / len(sample_cosines)),
        "negative_sample_fraction": float(sum(value < 0 for value in sample_cosines) / len(sample_cosines)),
        "non_negative_sample_fraction": float(sum(value >= 0 for value in sample_cosines) / len(sample_cosines)),
    }
    return report


def main() -> None:
    args = parse_args()
    os.makedirs(Path(args.output_json).parent, exist_ok=True)
    args.device = normalize_device(args.device)

    dataset_name = canonical_dataset_name(args.dataset)
    dtype = resolve_dtype(args.dtype)
    if args.loss_function != "uld_loss":
        raise ValueError(f"This script is hard-wired to `uld_loss`. Received --loss-function={args.loss_function!r}.")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1.")
    if args.subset_size < 1:
        raise ValueError("--subset-size must be >= 1.")

    torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = True

    print(f"Loading dataset {dataset_name}:{args.split} ...", flush=True)
    hf_dataset, loaded_from, schema, selected_samples = select_docvqa_subset(
        dataset_name=dataset_name,
        split=args.split,
        subset_size=args.subset_size,
        offset=args.offset,
    )
    print(f"Loaded dataset from {loaded_from}. Using {len(selected_samples)} valid samples.", flush=True)

    print("Loading student model ...", flush=True)
    student_model, student_processor = load_vlm_runtime(
        args.student_model_id,
        device=args.device,
        dtype=dtype,
        attn_implementation=args.attn_implementation,
        cache_dir=args.cache_dir,
    )
    print("Loading teacher model ...", flush=True)
    teacher_model, teacher_processor = load_vlm_runtime(
        args.teacher_model_id,
        device=args.device,
        dtype=dtype,
        attn_implementation=args.attn_implementation,
        cache_dir=args.cache_dir,
    )
    for parameter in teacher_model.parameters():
        parameter.requires_grad_(False)

    collator = GradientAgreementCollator(
        student_pad_token_id=get_processor_pad_token_id(student_processor),
        teacher_pad_token_id=get_processor_pad_token_id(teacher_processor),
    )
    analysis_dataset = DocVQAGradientAgreementDataset(
        hf_dataset=hf_dataset,
        selected_samples=selected_samples,
        schema=schema,
        student_processor=student_processor,
        teacher_processor=teacher_processor,
    )
    dataloader = DataLoader(
        analysis_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collator,
    )

    print("Starting gradient agreement run ...", flush=True)
    start_time = time.time()
    sample_cosines: list[float] = []
    batch_cosines: list[float] = []
    ce_losses: list[float] = []
    kd_losses: list[float] = []

    for batch_index, batch in enumerate(dataloader, start=1):
        student_model.zero_grad(set_to_none=True)
        student_inputs, teacher_inputs, teacher_labels = split_teacher_batch(batch)
        student_inputs = move_batch_to_model_device(student_model, student_inputs)
        teacher_inputs = move_batch_to_model_device(teacher_model, teacher_inputs)
        teacher_labels = move_batch_to_model_device(teacher_model, {"labels": teacher_labels})["labels"]

        student_outputs = student_model(**student_inputs, return_dict=True)
        student_logits = student_outputs.logits

        with torch.no_grad():
            teacher_outputs = teacher_model(**teacher_inputs, return_dict=True)
        teacher_logits = teacher_outputs.logits.detach()

        ce_loss = student_outputs.loss
        kd_loss, student_answer_positions = compute_single_teacher_uld_loss(
            student_logits=student_logits,
            student_labels=student_inputs["labels"],
            teacher_logits=teacher_logits,
            teacher_labels=teacher_labels,
            student_temperature=args.student_temperature,
            teacher_temperature=args.teacher_temperature,
        )

        g_ce = torch.autograd.grad(ce_loss, student_logits, retain_graph=True)[0]
        g_kd = torch.autograd.grad(kd_loss, student_logits, retain_graph=False)[0]

        current_sample_cosines = compute_sample_cosines(g_ce, g_kd, student_answer_positions)
        if current_sample_cosines:
            sample_cosines.extend(current_sample_cosines)
            batch_cosines.append(float(statistics.fmean(current_sample_cosines)))

        ce_losses.append(float(ce_loss.detach().float().item()))
        kd_losses.append(float(kd_loss.detach().float().item()))

        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

        if batch_index == 1 or batch_index % args.progress_every == 0:
            running_mean = statistics.fmean(sample_cosines) if sample_cosines else float("nan")
            print(
                f"[batch {batch_index}/{math.ceil(len(analysis_dataset) / args.batch_size)}] "
                f"ce={ce_losses[-1]:.4f} kd={kd_losses[-1]:.4f} running_sample_cos={running_mean:.4f}",
                flush=True,
            )

    elapsed_seconds = time.time() - start_time
    report = build_report(
        args=args,
        loaded_from=loaded_from,
        num_batches=len(dataloader),
        num_selected_samples=len(selected_samples),
        num_cosine_samples=len(sample_cosines),
        sample_cosines=sample_cosines,
        batch_cosines=batch_cosines,
        ce_losses=ce_losses,
        kd_losses=kd_losses,
        elapsed_seconds=elapsed_seconds,
    )

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print("Final report:", flush=True)
    print(json.dumps(report, indent=2), flush=True)
    print(f"Saved report to {output_path}", flush=True)


if __name__ == "__main__":
    main()
