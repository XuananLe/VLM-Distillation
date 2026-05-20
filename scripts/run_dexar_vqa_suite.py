from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

DEFAULT_MODELS = (
    "HuggingFaceTB/SmolVLM-256M-Instruct",
    "HuggingFaceTB/SmolVLM-500M-Instruct",
    "google/gemma-3-4b-it",
    "OpenGVLab/InternVL2-1B",
    "Qwen/Qwen2.5-VL-3B-Instruct",
    "Qwen/Qwen2-VL-2B-Instruct",
)
DEFAULT_DATASETS = ("textvqa", "docvqa", "chartqa")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DEX-AR VQA subsets for a suite of models and rank clear outputs.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--subset-size", type=int, default=30)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--split", default="train")
    parser.add_argument("--layer-index", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument(
        "--target-mode",
        choices=("generated", "ground_truth"),
        default="generated",
    )
    parser.add_argument(
        "--prompt-style",
        choices=("vqa", "paper"),
        default="vqa",
    )
    parser.add_argument("--models", nargs="*", default=list(DEFAULT_MODELS))
    parser.add_argument("--datasets", nargs="*", default=list(DEFAULT_DATASETS))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--sheet-k", type=int, default=3)
    parser.add_argument("--max-consecutive-failures", type=int, default=30)
    parser.add_argument("--no-skip-existing", action="store_true")
    return parser.parse_args()


def slugify_model(model_name: str) -> str:
    return model_name.replace("/", "__").replace(".", "_").replace("-", "_").replace(" ", "_")


def summary_is_complete(summary_path: Path, subset_size: int) -> bool:
    if not summary_path.exists():
        return False
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return int(summary.get("subset_size_completed", 0)) >= subset_size


def run_command(command: list[str], *, env: dict[str, str]) -> int:
    print("[suite] " + " ".join(command), flush=True)
    process = subprocess.run(command, env=env)
    return int(process.returncode)


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("DEXAR_SHOW_PLOTS", "0")
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("TRANSFORMERS_VERBOSITY", "error")

    results: list[dict[str, object]] = []

    for model_name in args.models:
        model_slug = slugify_model(model_name)
        model_root = args.output_root / model_slug
        model_root.mkdir(parents=True, exist_ok=True)

        model_status = {
            "model_name": model_name,
            "model_slug": model_slug,
            "model_root": str(model_root),
            "datasets": [],
        }
        print(f"[suite] model={model_name} root={model_root}", flush=True)

        for dataset in args.datasets:
            dataset_root = model_root / dataset
            summary_path = dataset_root / "summary.json"
            if not args.no_skip_existing and summary_is_complete(summary_path, args.subset_size):
                print(f"[suite] skip complete {model_name} {dataset}", flush=True)
                dataset_status = {
                    "dataset": dataset,
                    "status": "skipped_complete",
                    "output_root": str(dataset_root),
                }
                model_status["datasets"].append(dataset_status)
                continue

            command = [
                sys.executable,
                "demo/DEX-AR/run_docvqa_subset.py",
                "--model-name",
                model_name,
                "--dataset",
                dataset,
                "--split",
                args.split,
                "--subset-size",
                str(args.subset_size),
                "--offset",
                str(args.offset),
                "--layer-index",
                str(args.layer_index),
                "--device",
                args.device,
                "--max-new-tokens",
                str(args.max_new_tokens),
                "--target-mode",
                args.target_mode,
                "--prompt-style",
                args.prompt_style,
                "--output-root",
                str(dataset_root),
                "--max-consecutive-failures",
                str(args.max_consecutive_failures),
            ]
            returncode = run_command(command, env=env)
            dataset_status = {
                "dataset": dataset,
                "status": "completed" if returncode == 0 else "failed",
                "returncode": returncode,
                "output_root": str(dataset_root),
            }
            model_status["datasets"].append(dataset_status)

        completed = all(
            summary_is_complete(model_root / dataset / "summary.json", args.subset_size) for dataset in args.datasets
        )
        if completed:
            analysis_command = [
                sys.executable,
                "scripts/analysis/select_dexar_clear_samples.py",
                "--root",
                str(model_root),
                "--top-k",
                str(args.top_k),
                "--sheet-k",
                str(args.sheet_k),
                "--datasets",
                *args.datasets,
            ]
            analysis_returncode = run_command(analysis_command, env=env)
            model_status["analysis_status"] = "completed" if analysis_returncode == 0 else "failed"
            model_status["analysis_returncode"] = analysis_returncode
        else:
            model_status["analysis_status"] = "skipped_incomplete"

        results.append(model_status)
        (args.output_root / "suite_summary.json").write_text(
            json.dumps({"results": results}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    print(json.dumps({"results": results}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
