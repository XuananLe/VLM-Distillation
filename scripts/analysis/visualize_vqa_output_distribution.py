from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[2]
DEXAR_ROOT = ROOT / "demo" / "DEX-AR"
for path in (ROOT, ROOT / "src", DEXAR_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dexar.backends import DexarBackend
from run_docvqa_subset import build_vqa_prompt, pick_first_answer

from src.dataset.vqa_loading import (
    extract_image_as_pil,
    load_dataset_split,
    pick_first_text,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize autoregressive next-token output distributions for a VQA sample."
    )
    parser.add_argument("--model-name", default="HuggingFaceTB/SmolVLM-500M-Instruct")
    parser.add_argument("--dataset", default="docvqa", help="Dataset name: textvqa, docvqa, or chartqa.")
    parser.add_argument("--split", default="train")
    parser.add_argument("--row-index", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def candidate_images(backend: DexarBackend, image: Image.Image):
    rgb_image = image.convert("RGB")
    yielded: set[tuple[str, tuple[int, int]]] = set()

    for prepared_name, prepared_image in (("original", rgb_image),):
        yielded_key = (prepared_name, prepared_image.size)
        if yielded_key not in yielded:
            yielded.add(yielded_key)
            yield prepared_name, prepared_image

    contained = ImageOps.contain(
        rgb_image,
        (backend.recommended_image_size, backend.recommended_image_size),
    )
    contained_key = (f"contained_{backend.recommended_image_size}", contained.size)
    if contained_key not in yielded:
        yielded.add(contained_key)
        yield contained_key[0], contained

    if backend.family != "qwen2vl":
        square = rgb_image.resize(
            (backend.recommended_image_size, backend.recommended_image_size)
        )
        square_key = (f"square_{backend.recommended_image_size}", square.size)
        if square_key not in yielded:
            yield square_key[0], square


def decode_token(tokenizer: Any, token_id: int) -> str:
    text = tokenizer.decode([int(token_id)], skip_special_tokens=False)
    if text == "":
        text = tokenizer.convert_ids_to_tokens(int(token_id))
    return text


def label_token(token: str, max_chars: int = 28) -> str:
    label = token.replace("\n", "\\n").replace("\t", "\\t")
    if label == " ":
        label = "<space>"
    elif label.startswith(" "):
        label = "_" + label[1:]
    if len(label) > max_chars:
        label = label[: max_chars - 1] + "..."
    return label


def save_step_plot(step: dict[str, Any], output_dir: Path) -> None:
    labels = [label_token(item["token"]) for item in step["top_tokens"]]
    probs = [float(item["probability"]) for item in step["top_tokens"]]
    y_positions = list(range(len(labels)))

    fig_height = max(4.0, 0.28 * len(labels) + 1.4)
    fig, ax = plt.subplots(figsize=(8.5, fig_height))
    ax.barh(y_positions, probs, color="#3867d6")
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Probability")
    ax.set_title(
        f"Step {step['step']} next-token distribution | greedy: "
        f"{label_token(step['chosen_token'])}"
    )
    ax.set_xlim(0, max(0.01, max(probs) * 1.12))
    for y_pos, prob in zip(y_positions, probs):
        ax.text(prob, y_pos, f" {prob:.3f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / f"step_{int(step['step']):02d}_top_tokens.png", dpi=160)
    plt.close(fig)


def save_grid_plot(steps: list[dict[str, Any]], output_dir: Path, per_step_k: int = 10) -> None:
    if not steps:
        return
    cols = 2
    rows = math.ceil(len(steps) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(14, max(4, rows * 3.2)))
    flat_axes = list(axes.flat) if hasattr(axes, "flat") else [axes]

    for ax, step in zip(flat_axes, steps):
        top_tokens = step["top_tokens"][:per_step_k]
        labels = [label_token(item["token"], max_chars=18) for item in top_tokens]
        probs = [float(item["probability"]) for item in top_tokens]
        y_positions = list(range(len(labels)))
        ax.barh(y_positions, probs, color="#2b8a3e")
        ax.set_yticks(y_positions)
        ax.set_yticklabels(labels, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlim(0, max(0.01, max(probs) * 1.15))
        ax.set_title(f"Step {step['step']}: {label_token(step['chosen_token'], 18)}")
        ax.tick_params(axis="x", labelsize=7)

    for ax in flat_axes[len(steps):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / "top_tokens_grid.png", dpi=160)
    plt.close(fig)


def save_dynamics_plot(steps: list[dict[str, Any]], output_dir: Path) -> None:
    if not steps:
        return
    x_values = [int(step["step"]) for step in steps]
    max_probs = [float(step["chosen_probability"]) for step in steps]
    entropies = [float(step["entropy"]) for step in steps]
    normalized_entropies = [float(step["normalized_entropy"]) for step in steps]
    top5_mass = [float(step["top_mass"]["5"]) for step in steps]
    top20_mass = [float(step["top_mass"]["20"]) for step in steps]

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(x_values, max_probs, marker="o", label="greedy token prob")
    axes[0].plot(x_values, top5_mass, marker="o", label="top-5 mass")
    axes[0].plot(x_values, top20_mass, marker="o", label="top-20 mass")
    axes[0].set_ylabel("Probability mass")
    axes[0].set_ylim(0, 1.02)
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(x_values, entropies, marker="o", label="entropy")
    axes[1].plot(x_values, normalized_entropies, marker="o", label="normalized entropy")
    axes[1].set_xlabel("Generation step")
    axes[1].set_ylabel("Entropy")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_dir / "distribution_dynamics.png", dpi=160)
    plt.close(fig)


def extend_generation_inputs(
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    model_extra_inputs: dict[str, torch.Tensor],
    next_token_id: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=-1)
    attention_mask = torch.cat(
        [
            attention_mask,
            torch.ones((1, 1), device=attention_mask.device, dtype=attention_mask.dtype),
        ],
        dim=1,
    )
    if "token_type_ids" in model_extra_inputs:
        model_extra_inputs = dict(model_extra_inputs)
        model_extra_inputs["token_type_ids"] = torch.cat(
            [
                model_extra_inputs["token_type_ids"],
                torch.zeros(
                    (1, 1),
                    device=model_extra_inputs["token_type_ids"].device,
                    dtype=model_extra_inputs["token_type_ids"].dtype,
                ),
            ],
            dim=1,
        )
    return input_ids, attention_mask, model_extra_inputs


def collect_distribution(
    *,
    backend: DexarBackend,
    image: Image.Image,
    prompt: str,
    max_new_tokens: int,
    top_k: int,
    device: str,
) -> tuple[str, list[dict[str, Any]]]:
    encoded_prompt = backend.encode_prompt(prompt=prompt, image=image, device=torch.device(device))
    input_ids = encoded_prompt.model_inputs["input_ids"]
    attention_mask = encoded_prompt.model_inputs.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, device=input_ids.device)
    model_extra_inputs = {
        key: value
        for key, value in encoded_prompt.model_inputs.items()
        if key not in {"input_ids", "attention_mask"}
    }

    tokenizer = backend.processor.tokenizer
    eos_ids = tokenizer.eos_token_id
    if eos_ids is None:
        eos_id_set: set[int] = set()
    elif isinstance(eos_ids, list):
        eos_id_set = {int(token_id) for token_id in eos_ids}
    else:
        eos_id_set = {int(eos_ids)}

    steps: list[dict[str, Any]] = []
    generated_ids: list[int] = []
    backend.model.eval()

    with torch.inference_mode():
        for step_index in range(max_new_tokens):
            outputs = backend.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
                use_cache=False,
                **model_extra_inputs,
            )
            logits = torch.nan_to_num(outputs.logits[:, -1, :].float(), nan=-1e9)
            probs = torch.softmax(logits, dim=-1)[0]
            step_top_k = min(int(top_k), int(probs.shape[-1]))
            top_probs, top_ids = torch.topk(probs, k=step_top_k)
            next_token_id = top_ids[:1]
            next_id = int(next_token_id.item())
            generated_ids.append(next_id)

            entropy = -torch.sum(probs * torch.clamp(probs, min=1e-45).log()).item()
            vocab_size = int(probs.shape[-1])
            top_mass = {
                str(k): float(top_probs[: min(k, step_top_k)].sum().item())
                for k in (1, 5, 10, 20)
            }
            chosen_token = decode_token(tokenizer, next_id)
            steps.append(
                {
                    "step": step_index,
                    "chosen_token_id": next_id,
                    "chosen_token": chosen_token,
                    "chosen_probability": float(top_probs[0].item()),
                    "entropy": entropy,
                    "normalized_entropy": entropy / math.log(vocab_size),
                    "top_mass": top_mass,
                    "top_tokens": [
                        {
                            "rank": rank + 1,
                            "token_id": int(token_id),
                            "token": decode_token(tokenizer, int(token_id)),
                            "probability": float(probability),
                        }
                        for rank, (token_id, probability) in enumerate(
                            zip(top_ids.detach().cpu().tolist(), top_probs.detach().cpu().tolist())
                        )
                    ],
                }
            )

            input_ids, attention_mask, model_extra_inputs = extend_generation_inputs(
                input_ids=input_ids,
                attention_mask=attention_mask,
                model_extra_inputs=model_extra_inputs,
                next_token_id=next_token_id,
            )
            if next_id in eos_id_set:
                break

    generated_answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return generated_answer, steps


def main() -> None:
    args = parse_args()
    dataset_name = args.dataset
    device = resolve_device(args.device)
    args.output_root.mkdir(parents=True, exist_ok=True)

    backend = DexarBackend.from_pretrained(args.model_name, device)
    dataset, loaded_from, schema = load_dataset_split(dataset_name, args.split)
    sample = dataset[args.row_index]
    question = pick_first_text(sample.get(schema["question_field"])) if schema["question_field"] else None
    ground_truth = pick_first_answer(sample.get(schema["answer_field"])) if schema["answer_field"] else None
    if not question:
        raise ValueError(f"Row {args.row_index} does not contain a usable question.")

    sample_id = sample.get(schema["id_field"]) if schema["id_field"] else args.row_index
    source_image = extract_image_as_pil(sample.get(schema["image_field"]))
    source_image.save(args.output_root / "original_image.png")
    prompt = build_vqa_prompt(dataset_name, backend.family, question)
    (args.output_root / "prompt.txt").write_text(prompt, encoding="utf-8")

    last_error: Exception | None = None
    for image_preparation, prepared_image in candidate_images(backend, source_image):
        try:
            generated_answer, steps = collect_distribution(
                backend=backend,
                image=prepared_image,
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                top_k=args.top_k,
                device=device,
            )
            prepared_image.save(args.output_root / "input_image.png")
            break
        except (RuntimeError, ValueError, OSError, TypeError) as exc:
            last_error = exc
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    else:
        assert last_error is not None
        raise last_error

    for step in steps:
        save_step_plot(step, args.output_root)
    save_grid_plot(steps, args.output_root, per_step_k=min(10, args.top_k))
    save_dynamics_plot(steps, args.output_root)

    metadata = {
        "model_name": args.model_name,
        "model_family": backend.family,
        "dataset": dataset_name,
        "loaded_from": loaded_from,
        "split": args.split,
        "row_index": args.row_index,
        "sample_id": str(sample_id),
        "question": question,
        "ground_truth_answer": ground_truth,
        "generated_answer": generated_answer,
        "image_preparation": image_preparation,
        "original_image_size": list(source_image.size),
        "input_image_size": list(prepared_image.size),
        "max_new_tokens": args.max_new_tokens,
        "top_k": args.top_k,
    }
    (args.output_root / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (args.output_root / "distribution_steps.json").write_text(
        json.dumps({"metadata": metadata, "steps": steps}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "output_root": str(args.output_root),
                "generated_answer": generated_answer,
                "question": question,
                "ground_truth_answer": ground_truth,
                "num_steps": len(steps),
                "first_step_top_tokens": steps[0]["top_tokens"][:5] if steps else [],
            },
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
