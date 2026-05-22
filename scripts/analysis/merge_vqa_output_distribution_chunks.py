#!/usr/bin/env python3
"""Merge chunked VQA output-distribution runs into one summary.

This is intentionally lightweight: it keeps per-sample artifacts in their
original chunk folders, and writes merged JSON plus aggregate plots.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunk-roots", nargs="+", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-slug", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--dataset", default="docvqa")
    parser.add_argument("--split", default="train")
    parser.add_argument("--loaded-from", default="HuggingFaceM4/DocumentVQA")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--prompt-style", default="official_docvqa")
    return parser.parse_args()


def mean(values: list[float | int | None]) -> float | None:
    valid = [float(value) for value in values if value is not None]
    return sum(valid) / len(valid) if valid else None


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"num_samples": 0, "exact": 0, "loose": 0}

    exact = sum(1 for record in records if record.get("exact_match"))
    loose = sum(1 for record in records if record.get("loose_match"))

    def step_metric(name: str) -> float | None:
        return mean(
            [
                record.get("step_summary", {}).get(name)
                for record in records
                if record.get("step_summary", {}).get(name) is not None
            ]
        )

    return {
        "num_samples": len(records),
        "exact": exact,
        "exact_rate": exact / len(records),
        "loose": loose,
        "loose_rate": loose / len(records),
        "mean_anls": mean([record.get("anls", 0.0) for record in records]),
        "mean_num_steps": step_metric("num_steps"),
        "mean_chosen_probability": step_metric("mean_chosen_probability"),
        "mean_entropy": step_metric("mean_entropy"),
        "mean_normalized_entropy": step_metric("mean_normalized_entropy"),
        "mean_top5_mass": step_metric("mean_top5_mass"),
        "mean_top20_mass": step_metric("mean_top20_mass"),
        "mean_first_step_chosen_probability": step_metric("first_step_chosen_probability"),
        "mean_first_step_entropy": step_metric("first_step_entropy"),
        "mean_first_step_normalized_entropy": step_metric("first_step_normalized_entropy"),
    }


def sample_dir_for_record(chunk_root: Path, model_slug: str, index: int, record: dict[str, Any]) -> Path:
    row_index = int(record["row_index"])
    sample_id = str(record["sample_id"])
    expected = chunk_root / model_slug / "docvqa" / f"sample_{index:02d}_row_{row_index:06d}_{sample_id}"
    if expected.exists():
        return expected
    matches = list((chunk_root / model_slug / "docvqa").glob(f"sample_*_row_{row_index:06d}_*"))
    if not matches:
        raise FileNotFoundError(expected)
    return matches[0]


def load_chunks(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    records_with_steps: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    sample_sources: list[str] = []

    for chunk_root in args.chunk_roots:
        summary_path = chunk_root / args.model_slug / "summary.json"
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        failures.extend(payload.get("failures", []))
        for index, metadata in enumerate(payload["records"]):
            sample_dir = sample_dir_for_record(chunk_root, args.model_slug, index, metadata)
            steps_path = sample_dir / "distribution_steps.json"
            steps = (
                json.loads(steps_path.read_text(encoding="utf-8")).get("steps", [])
                if steps_path.exists()
                else []
            )
            sample_sources.append(str(sample_dir))
            records_with_steps.append(
                {
                    "metadata": metadata,
                    "steps": steps,
                    "step_summary": metadata.get("step_summary", {}),
                    "exact_match": bool(metadata.get("exact_match")),
                    "loose_match": bool(metadata.get("loose_match")),
                    "anls": float(metadata.get("anls", 0.0)),
                }
            )

    records_with_steps.sort(key=lambda item: int(item["metadata"]["row_index"]))
    return records_with_steps, failures, sample_sources


def aggregate_by_step(records_with_steps: list[dict[str, Any]]) -> dict[int, dict[str, float]]:
    buckets: dict[int, dict[str, list[float]]] = {}
    for record in records_with_steps:
        for step in record.get("steps", []):
            index = int(step.get("step_index", step.get("step")))
            bucket = buckets.setdefault(
                index,
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
        index: {metric: float(mean(values)) for metric, values in bucket.items()}
        for index, bucket in sorted(buckets.items())
    }


def save_aggregate_plots(output_dir: Path, model_name: str, records_with_steps: list[dict[str, Any]]) -> None:
    by_step = aggregate_by_step(records_with_steps)
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
        fig.suptitle(f"{model_name}: mean output-distribution dynamics")
        fig.tight_layout()
        fig.savefig(output_dir / "aggregate_distribution_dynamics.png", dpi=160)
        plt.close(fig)

    first_step_probs = [
        float(record["step_summary"]["first_step_chosen_probability"])
        for record in records_with_steps
        if record.get("step_summary", {}).get("first_step_chosen_probability") is not None
    ]
    if first_step_probs:
        fig, ax = plt.subplots(figsize=(8, 4.8))
        ax.hist(first_step_probs, bins=12, color="#3867d6", edgecolor="white")
        ax.set_xlabel("First-step greedy token probability")
        ax.set_ylabel("Samples")
        ax.set_title(f"{model_name}: first-token confidence over samples")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / "first_step_confidence_hist.png", dpi=160)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    model_root = args.output_root / args.model_slug
    model_root.mkdir(parents=True, exist_ok=True)

    records_with_steps, failures, sample_sources = load_chunks(args)
    records = [record["metadata"] for record in records_with_steps]
    model_summary = {
        "model_name": args.model_name,
        "model_slug": args.model_slug,
        "model_root": str(model_root),
        "dataset": args.dataset,
        "records": records,
        "failures": failures,
        "summary": summarize(records),
        "merged_from": [str(path) for path in args.chunk_roots],
        "sample_sources": sample_sources,
    }
    (model_root / "summary.json").write_text(
        json.dumps(model_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    suite_summary = {
        "dataset": args.dataset,
        "split": args.split,
        "loaded_from": args.loaded_from,
        "subset_size": len(records),
        "max_new_tokens": args.max_new_tokens,
        "top_k": args.top_k,
        "prompt_style": args.prompt_style,
        "models": [model_summary],
    }
    (args.output_root / "suite_summary.json").write_text(
        json.dumps(suite_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (model_root / "sample_sources.txt").write_text(
        "\n".join(sample_sources) + "\n",
        encoding="utf-8",
    )
    save_aggregate_plots(model_root, args.model_name, records_with_steps)
    print(json.dumps(model_summary["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
