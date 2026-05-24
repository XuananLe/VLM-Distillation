from __future__ import annotations

import argparse
import gc
import json
import math
import re
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
for path in (ROOT, ROOT / "src", DEXAR_ROOT, ROOT / "scripts" / "analysis"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dexar.backends import DexarBackend
from run_docvqa_subset import build_vqa_prompt
from visualize_vqa_output_distribution import (
    collect_distribution,
    save_dynamics_plot,
    save_grid_plot,
    save_step_plot,
)

from src.dataset.vqa_loading import (
    extract_image_as_pil,
    load_dataset_split,
    pick_first_text,
)

SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect next-token output-distribution visualizations for VQA samples."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=[
            "HuggingFaceTB/SmolVLM-500M-Instruct",
            "Qwen/Qwen2.5-VL-3B-Instruct",
            "LiquidAI/LFM2.5-VL-450M",
        ],
    )
    parser.add_argument("--dataset", default="docvqa", help="Dataset name: textvqa, docvqa, or chartqa.")
    parser.add_argument("--split", default="train")
    parser.add_argument("--subset-size", type=int, default=50)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-consecutive-failures", type=int, default=30)
    parser.add_argument(
        "--prompt-style",
        choices=("current", "official_docvqa"),
        default="current",
        help="Prompt template to use for VQA generation.",
    )
    parser.add_argument(
        "--save-step-plots",
        action="store_true",
        help="Also save one top-token bar chart per generation step for each sample.",
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Recompute samples even if metadata.json already exists.",
    )
    return parser.parse_args()


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def sanitize_for_filename(raw_value: Any) -> str:
    sanitized = SAFE_FILENAME_RE.sub("_", str(raw_value)).strip("._")
    return sanitized or "sample"


def slugify_model(model_name: str) -> str:
    return sanitize_for_filename(model_name.replace("/", "__"))


def normalize_answer(value: Any) -> str:
    text = str(value or "").lower().strip()
    text = re.sub(r"[^a-z0-9.]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_anls_text(value: Any) -> str:
    text = str(value or "").lower().strip()
    return re.sub(r"\s+", " ", text)


def levenshtein_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_char in enumerate(right, start=1):
            current.append(
                min(
                    previous[right_index] + 1,
                    current[-1] + 1,
                    previous[right_index - 1] + int(left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def anls_score(prediction: Any, target: Any) -> float:
    pred = normalize_anls_text(prediction)
    gold = normalize_anls_text(target)
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    distance = levenshtein_distance(pred, gold)
    similarity = 1.0 - distance / max(len(pred), len(gold))
    return similarity if similarity >= 0.5 else 0.0


def best_anls_score(prediction: Any, targets: list[str]) -> float:
    if not targets:
        return 0.0
    return max(anls_score(prediction, target) for target in targets)


def pick_all_answers(answer_value: Any) -> list[str]:
    values = answer_value if isinstance(answer_value, (list, tuple)) else (answer_value,)
    answers = []
    for value in values:
        if value is None or isinstance(value, dict):
            continue
        text = str(value).strip()
        if text:
            answers.append(text)
    return answers


def parse_answer(raw_answer: str) -> str:
    text = str(raw_answer or "").strip()
    matches = list(re.finditer(r"answer\s*:\s*", text, flags=re.IGNORECASE))
    if matches:
        text = text[matches[-1].end() :]
    return text.strip().strip("\"'`")


def extract_numbers(value: Any) -> list[str]:
    return re.findall(r"[-+]?\d*\.?\d+", str(value or "").replace(",", ""))


def exact_match(prediction: Any, targets: Any) -> bool:
    if isinstance(targets, list):
        return any(normalize_answer(prediction) == normalize_answer(target) for target in targets)
    return normalize_answer(prediction) == normalize_answer(targets)


def loose_match_one(prediction: Any, target: Any) -> bool:
    pred = normalize_answer(prediction)
    gold = normalize_answer(target)
    if not pred or not gold:
        return False
    if pred == gold or gold in pred or pred in gold:
        return True
    pred_numbers = extract_numbers(pred)
    gold_numbers = extract_numbers(gold)
    return bool(pred_numbers and gold_numbers and gold_numbers[0] in pred_numbers)


def loose_match(prediction: Any, targets: Any) -> bool:
    if isinstance(targets, list):
        return any(loose_match_one(prediction, target) for target in targets)
    return loose_match_one(prediction, targets)


def build_prompt(
    *,
    dataset_name: str,
    model_family: str,
    question: str,
    prompt_style: str,
) -> str:
    if prompt_style != "official_docvqa":
        return build_vqa_prompt(dataset_name, model_family, question)

    if dataset_name != "docvqa":
        raise ValueError("--prompt-style official_docvqa is only valid with --dataset docvqa.")

    instruction = (
        "Answer the question according to the image using a single word or phrase.\n"
        f"{question}\n"
        'The last line of your response should be of the form "ANSWER: [ANSWER]" '
        "without quotes where [ANSWER] is the answer to the question."
    )
    if model_family == "smolvlm":
        return f"<|im_start|>User:<image>{instruction}<end_of_utterance>\nAssistant:"
    if model_family == "llava":
        return f"USER: <image>\n{instruction}\nASSISTANT:"
    if model_family == "internvl":
        return f"<image>\n{instruction}"
    return f"<image>\n{instruction}"


def candidate_images(backend: DexarBackend, image: Image.Image):
    rgb_image = image.convert("RGB")
    yielded: set[tuple[str, tuple[int, int]]] = set()

    yielded.add(("original", rgb_image.size))
    yield "original", rgb_image

    contained_sizes = [backend.recommended_image_size]
    if backend.family == "qwen2vl":
        contained_sizes.extend([384, 336, 280, 224, 168])

    for contained_size in contained_sizes:
        contained = ImageOps.contain(rgb_image, (contained_size, contained_size))
        key = (f"contained_{contained_size}", contained.size)
        if key in yielded:
            continue
        yielded.add(key)
        yield key[0], contained

    if backend.family != "qwen2vl":
        square = rgb_image.resize((backend.recommended_image_size, backend.recommended_image_size))
        key = (f"square_{backend.recommended_image_size}", square.size)
        if key not in yielded:
            yield key[0], square


def summarize_steps(steps: list[dict[str, Any]]) -> dict[str, Any]:
    if not steps:
        return {
            "num_steps": 0,
            "mean_chosen_probability": None,
            "mean_entropy": None,
            "mean_normalized_entropy": None,
            "first_step_chosen_probability": None,
            "first_step_entropy": None,
            "first_step_top_tokens": [],
        }
    return {
        "num_steps": len(steps),
        "mean_chosen_probability": sum(float(step["chosen_probability"]) for step in steps) / len(steps),
        "mean_entropy": sum(float(step["entropy"]) for step in steps) / len(steps),
        "mean_normalized_entropy": sum(float(step["normalized_entropy"]) for step in steps) / len(steps),
        "mean_top5_mass": sum(float(step["top_mass"]["5"]) for step in steps) / len(steps),
        "mean_top20_mass": sum(float(step["top_mass"]["20"]) for step in steps) / len(steps),
        "first_step_chosen_probability": float(steps[0]["chosen_probability"]),
        "first_step_entropy": float(steps[0]["entropy"]),
        "first_step_normalized_entropy": float(steps[0]["normalized_entropy"]),
        "first_step_top_tokens": steps[0]["top_tokens"][:5],
    }


def save_sample_outputs(
    *,
    output_dir: Path,
    source_image: Image.Image,
    prepared_image: Image.Image,
    prompt: str,
    metadata: dict[str, Any],
    steps: list[dict[str, Any]],
    save_step_plots: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    source_image.save(output_dir / "original_image.png")
    prepared_image.save(output_dir / "input_image.png")
    (output_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
    if save_step_plots:
        for step in steps:
            save_step_plot(step, output_dir)
    save_grid_plot(steps, output_dir, per_step_k=min(10, int(metadata["top_k"])))
    save_dynamics_plot(steps, output_dir)
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "distribution_steps.json").write_text(
        json.dumps({"metadata": metadata, "steps": steps}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def aggregate_by_step(records: list[dict[str, Any]]) -> dict[int, dict[str, float]]:
    buckets: dict[int, dict[str, list[float]]] = {}
    for record in records:
        for step in record.get("steps", []):
            step_index = int(step.get("step_index", step.get("step")))
            bucket = buckets.setdefault(
                step_index,
                {
                    "chosen_probability": [],
                    "entropy": [],
                    "normalized_entropy": [],
                    "top5_mass": [],
                    "top20_mass": [],
                },
            )
            bucket["chosen_probability"].append(float(step["chosen_probability"]))
            bucket["entropy"].append(float(step["entropy"]))
            bucket["normalized_entropy"].append(float(step["normalized_entropy"]))
            bucket["top5_mass"].append(float(step["top_mass"]["5"]))
            bucket["top20_mass"].append(float(step["top_mass"]["20"]))

    return {
        step_index: {
            key: sum(values) / len(values)
            for key, values in bucket.items()
            if values
        }
        for step_index, bucket in sorted(buckets.items())
    }


def save_model_aggregate_plots(
    *,
    output_dir: Path,
    model_label: str,
    records: list[dict[str, Any]],
) -> None:
    by_step = aggregate_by_step(records)
    if by_step:
        step_indices = list(by_step)
        fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        axes[0].plot(
            step_indices,
            [by_step[index]["chosen_probability"] for index in step_indices],
            marker="o",
            label="greedy token prob",
        )
        axes[0].plot(
            step_indices,
            [by_step[index]["top5_mass"] for index in step_indices],
            marker="o",
            label="top-5 mass",
        )
        axes[0].plot(
            step_indices,
            [by_step[index]["top20_mass"] for index in step_indices],
            marker="o",
            label="top-20 mass",
        )
        axes[0].set_ylim(0, 1.02)
        axes[0].set_ylabel("Mean probability mass")
        axes[0].legend()
        axes[0].grid(alpha=0.25)

        axes[1].plot(
            step_indices,
            [by_step[index]["entropy"] for index in step_indices],
            marker="o",
            label="entropy",
        )
        axes[1].plot(
            step_indices,
            [by_step[index]["normalized_entropy"] for index in step_indices],
            marker="o",
            label="normalized entropy",
        )
        axes[1].set_xlabel("Generation step")
        axes[1].set_ylabel("Mean entropy")
        axes[1].legend()
        axes[1].grid(alpha=0.25)

        fig.suptitle(f"{model_label}: mean output-distribution dynamics")
        fig.tight_layout()
        fig.savefig(output_dir / "aggregate_distribution_dynamics.png", dpi=160)
        plt.close(fig)

    first_step_probs = [
        float(record["step_summary"]["first_step_chosen_probability"])
        for record in records
        if record.get("step_summary", {}).get("first_step_chosen_probability") is not None
    ]
    if first_step_probs:
        fig, ax = plt.subplots(figsize=(8, 4.8))
        ax.hist(first_step_probs, bins=12, color="#3867d6", edgecolor="white")
        ax.set_xlabel("First-step greedy token probability")
        ax.set_ylabel("Samples")
        ax.set_title(f"{model_label}: first-token confidence over samples")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / "first_step_confidence_hist.png", dpi=160)
        plt.close(fig)


def summarize_model(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {
            "num_samples": 0,
            "exact": 0,
            "loose": 0,
        }

    exact = sum(1 for record in records if record.get("exact_match"))
    loose = sum(1 for record in records if record.get("loose_match"))
    anls_values = [float(record.get("anls", 0.0)) for record in records]

    def mean_metric(name: str) -> float | None:
        values = [
            record["step_summary"].get(name)
            for record in records
            if record.get("step_summary", {}).get(name) is not None
        ]
        if not values:
            return None
        return float(sum(float(value) for value in values) / len(values))

    return {
        "num_samples": len(records),
        "exact": exact,
        "exact_rate": exact / len(records),
        "loose": loose,
        "loose_rate": loose / len(records),
        "mean_anls": sum(anls_values) / len(anls_values),
        "mean_num_steps": mean_metric("num_steps"),
        "mean_chosen_probability": mean_metric("mean_chosen_probability"),
        "mean_entropy": mean_metric("mean_entropy"),
        "mean_normalized_entropy": mean_metric("mean_normalized_entropy"),
        "mean_top5_mass": mean_metric("mean_top5_mass"),
        "mean_top20_mass": mean_metric("mean_top20_mass"),
        "mean_first_step_chosen_probability": mean_metric("first_step_chosen_probability"),
        "mean_first_step_entropy": mean_metric("first_step_entropy"),
        "mean_first_step_normalized_entropy": mean_metric("first_step_normalized_entropy"),
    }


def write_model_summary(
    model_root: Path,
    model_name: str,
    model_slug: str,
    dataset_name: str,
    records: list[dict[str, Any]],
    failures: list[dict[str, Any]],
) -> dict[str, Any]:
    summary = {
        "model_name": model_name,
        "model_slug": model_slug,
        "model_root": str(model_root),
        "dataset": dataset_name,
        "records": [record["metadata"] for record in records],
        "failures": failures,
        "summary": summarize_model(records),
    }
    (model_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def process_model(
    *,
    model_name: str,
    dataset_name: str,
    loaded_from: str,
    dataset: Any,
    schema: dict[str, str | None],
    args: argparse.Namespace,
    device: str,
) -> dict[str, Any]:
    model_slug = slugify_model(model_name)
    model_root = args.output_root / model_slug
    samples_root = model_root / dataset_name
    samples_root.mkdir(parents=True, exist_ok=True)
    print(f"[model] loading {model_name}", flush=True)
    backend = DexarBackend.from_pretrained(model_name, device)

    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    consecutive_failures = 0
    row_index = int(args.offset)

    try:
        while len(records) < int(args.subset_size) and row_index < len(dataset):
            sample = dataset[row_index]
            question = (
                pick_first_text(sample.get(schema["question_field"]))
                if schema["question_field"]
                else None
            )
            ground_truth_answers = (
                pick_all_answers(sample.get(schema["answer_field"]))
                if schema["answer_field"]
                else []
            )
            ground_truth = ground_truth_answers[0] if ground_truth_answers else None
            sample_id = sample.get(schema["id_field"]) if schema["id_field"] else row_index
            safe_sample_id = sanitize_for_filename(sample_id)
            output_dir = samples_root / f"sample_{len(records):02d}_row_{row_index:06d}_{safe_sample_id}"

            print(
                f"[{model_slug}] sample {len(records) + 1}/{args.subset_size} row={row_index} id={sample_id}",
                flush=True,
            )
            if not question:
                failures.append({"row_index": row_index, "error": "missing question"})
                consecutive_failures += 1
                row_index += 1
                continue

            if (
                not args.no_skip_existing
                and (output_dir / "metadata.json").exists()
                and (output_dir / "distribution_steps.json").exists()
            ):
                metadata = json.loads((output_dir / "metadata.json").read_text(encoding="utf-8"))
                steps_payload = json.loads((output_dir / "distribution_steps.json").read_text(encoding="utf-8"))
                records.append(
                    {
                        "metadata": metadata,
                        "steps": steps_payload.get("steps", []),
                        "step_summary": metadata.get("step_summary", {}),
                        "exact_match": bool(metadata.get("exact_match")),
                        "loose_match": bool(metadata.get("loose_match")),
                        "anls": float(metadata.get("anls", 0.0)),
                    }
                )
                row_index += 1
                consecutive_failures = 0
                continue

            try:
                source_image = extract_image_as_pil(sample.get(schema["image_field"]))
                prompt = build_prompt(
                    dataset_name=dataset_name,
                    model_family=backend.family,
                    question=question,
                    prompt_style=args.prompt_style,
                )
                last_error: Exception | None = None
                for image_preparation, prepared_image in candidate_images(backend, source_image):
                    try:
                        generated_answer, steps = collect_distribution(
                            backend=backend,
                            image=prepared_image,
                            prompt=prompt,
                            max_new_tokens=int(args.max_new_tokens),
                            top_k=int(args.top_k),
                            device=device,
                        )
                        break
                    except (RuntimeError, ValueError, OSError, TypeError) as exc:
                        last_error = exc
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                else:
                    assert last_error is not None
                    raise last_error

                step_summary = summarize_steps(steps)
                parsed_answer = parse_answer(generated_answer)
                sample_anls = best_anls_score(parsed_answer, ground_truth_answers)
                metadata = {
                    "model_name": model_name,
                    "model_family": backend.family,
                    "dataset": dataset_name,
                    "loaded_from": loaded_from,
                    "split": args.split,
                    "row_index": row_index,
                    "sample_id": str(sample_id),
                    "question": question,
                    "ground_truth_answer": ground_truth,
                    "ground_truth_answers": ground_truth_answers,
                    "generated_answer": generated_answer,
                    "parsed_answer": parsed_answer,
                    "exact_match": exact_match(parsed_answer, ground_truth_answers),
                    "loose_match": loose_match(parsed_answer, ground_truth_answers),
                    "anls": sample_anls,
                    "prompt_style": args.prompt_style,
                    "image_preparation": image_preparation,
                    "original_image_size": list(source_image.size),
                    "input_image_size": list(prepared_image.size),
                    "max_new_tokens": int(args.max_new_tokens),
                    "top_k": int(args.top_k),
                    "step_summary": step_summary,
                }
                save_sample_outputs(
                    output_dir=output_dir,
                    source_image=source_image,
                    prepared_image=prepared_image,
                    prompt=prompt,
                    metadata=metadata,
                    steps=steps,
                    save_step_plots=bool(args.save_step_plots),
                )
                records.append(
                    {
                        "metadata": metadata,
                        "steps": steps,
                        "step_summary": step_summary,
                        "exact_match": bool(metadata["exact_match"]),
                        "loose_match": bool(metadata["loose_match"]),
                        "anls": float(metadata["anls"]),
                    }
                )
                consecutive_failures = 0
                row_index += 1

                write_model_summary(model_root, model_name, model_slug, dataset_name, records, failures)

            except Exception as exc:
                failures.append(
                    {
                        "row_index": row_index,
                        "sample_id": str(sample_id),
                        "error": repr(exc),
                    }
                )
                consecutive_failures += 1
                row_index += 1
                print(f"[{model_slug}] skipped row after error: {exc!r}", flush=True)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if consecutive_failures >= int(args.max_consecutive_failures):
                raise RuntimeError(
                    f"Aborting {model_name}: {consecutive_failures} consecutive failures."
                )

    finally:
        try:
            backend.model.to("cpu")
        except Exception:
            pass
        del backend
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    save_model_aggregate_plots(output_dir=model_root, model_label=model_name, records=records)
    model_summary = write_model_summary(model_root, model_name, model_slug, dataset_name, records, failures)
    print(json.dumps(model_summary["summary"], indent=2), flush=True)
    return model_summary


def save_suite_comparison(output_root: Path, model_summaries: list[dict[str, Any]]) -> None:
    if not model_summaries:
        return
    labels = [summary["model_name"].split("/")[-1] for summary in model_summaries]
    exact_rates = [float(summary["summary"].get("exact_rate") or 0.0) for summary in model_summaries]
    loose_rates = [float(summary["summary"].get("loose_rate") or 0.0) for summary in model_summaries]
    anls_scores = [float(summary["summary"].get("mean_anls") or 0.0) for summary in model_summaries]
    first_probs = [
        float(summary["summary"].get("mean_first_step_chosen_probability") or 0.0)
        for summary in model_summaries
    ]
    entropies = [
        float(summary["summary"].get("mean_first_step_normalized_entropy") or 0.0)
        for summary in model_summaries
    ]

    x_positions = list(range(len(labels)))
    fig, axes = plt.subplots(2, 1, figsize=(11, 8))
    width = 0.35
    axes[0].bar([x - width for x in x_positions], exact_rates, width, label="exact")
    axes[0].bar(x_positions, loose_rates, width, label="loose")
    axes[0].bar([x + width for x in x_positions], anls_scores, width, label="ANLS")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0, max(0.05, max(loose_rates + exact_rates + anls_scores) * 1.25))
    axes[0].set_xticks(x_positions)
    axes[0].set_xticklabels(labels, rotation=12, ha="right")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar([x - width / 2 for x in x_positions], first_probs, width, label="first-token prob")
    axes[1].bar([x + width / 2 for x in x_positions], entropies, width, label="first-token norm entropy")
    axes[1].set_ylabel("Mean value")
    axes[1].set_xticks(x_positions)
    axes[1].set_xticklabels(labels, rotation=12, ha="right")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.25)

    fig.suptitle("Output-distribution comparison")
    fig.tight_layout()
    fig.savefig(output_root / "suite_comparison.png", dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    dataset_name = args.dataset
    device = resolve_device(args.device)
    args.output_root.mkdir(parents=True, exist_ok=True)

    dataset, loaded_from, schema = load_dataset_split(dataset_name, args.split)
    model_summaries: list[dict[str, Any]] = []
    for model_name in args.models:
        model_summaries.append(
            process_model(
                model_name=model_name,
                dataset_name=dataset_name,
                loaded_from=loaded_from,
                dataset=dataset,
                schema=schema,
                args=args,
                device=device,
            )
        )

    suite_summary = {
        "dataset": dataset_name,
        "split": args.split,
        "loaded_from": loaded_from,
        "subset_size": args.subset_size,
        "max_new_tokens": args.max_new_tokens,
        "top_k": args.top_k,
        "prompt_style": args.prompt_style,
        "models": model_summaries,
    }
    (args.output_root / "suite_summary.json").write_text(
        json.dumps(suite_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    save_suite_comparison(args.output_root, model_summaries)
    print(json.dumps({k: suite_summary[k] for k in ("dataset", "split", "subset_size")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
