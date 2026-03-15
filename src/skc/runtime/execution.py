import gc
import itertools
from contextlib import contextmanager

import numpy as np
import torch
from tqdm import tqdm

from src.components.skc import (
    compute_skc_from_matrices,
    extract_sample_representations,
    skc_score,
)
from src.components.skc_cluster import run_pipeline

from ..data.loading import build_probe_samples, load_probe_dataset
from ..data.probe import build_loader
from ..vlm.api import load_vlm
from ..vlm.processors import load_vlm_processor


def extract_representations(model, loader, layer_index: int, name: str):
    reps = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"  {name}", leave=False):
            reps.append(
                extract_sample_representations(model, batch, layer_index).float().cpu()
            )
    return torch.cat(reps, dim=0)


def print_skc_matrix(skc_matrix: np.ndarray, names: list[str]) -> None:
    short_names = [name.split("/")[-1][:14] for name in names]
    col_width = max(len(name) for name in short_names) + 2
    header = " " * col_width + "".join(f"{name:>{col_width}}" for name in short_names)
    print(header)
    for row_index, row_name in enumerate(short_names):
        row = f"{row_name:<{col_width}}"
        for col_index in range(len(names)):
            if row_index == col_index:
                row += f"{'—':>{col_width}}"
            else:
                row += f"{skc_matrix[row_index, col_index]:>{col_width}.4f}"
        print(row)


def select_dtype() -> torch.dtype:
    if not torch.cuda.is_available():
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def cleanup_inference_objects(model=None, processor=None, loader=None) -> None:
    if loader is not None:
        del loader
    if processor is not None:
        del processor
    if model is not None:
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@contextmanager
def loaded_model_bundle(model_name: str, dtype: torch.dtype, probe_samples):
    model = processor = loader = None
    try:
        model, family = load_vlm(model_name, dtype)
        processor = load_vlm_processor(model_name, family, model)
        loader = build_loader(processor, probe_samples, family, model=model)
        yield model, family, loader
    finally:
        cleanup_inference_objects(model=model, processor=processor, loader=loader)


def run_pairwise_mode(args, dtype: torch.dtype, probe_samples) -> None:
    print("Loading models...")
    name_a, name_b = args.models

    with loaded_model_bundle(name_a, dtype, probe_samples) as (model_a, family_a, loader_a):
        print(f"  Loaded {name_a} ({family_a})")
        with loaded_model_bundle(name_b, dtype, probe_samples) as (model_b, family_b, loader_b):
            print(f"  Loaded {name_b} ({family_b})")

            score = skc_score(
                model_a,
                model_b,
                dataloader_a=loader_a,
                dataloader_b=loader_b,
                layer_index=args.layer_index,
            )

    print(f"\n{'-' * 50}")
    print(f"  SKC({name_a.split('/')[-1]}  ↔  {name_b.split('/')[-1]}) = {score:.4f}")
    print(f"{'-' * 50}")


def extract_all_representations(args, dtype: torch.dtype, probe_samples):
    print("\nExtracting representations...")
    all_representations = {}
    model_names = []
    failed_models = []

    for name in args.models:
        try:
            with loaded_model_bundle(name, dtype, probe_samples) as (model, family, loader):
                print(f"  Loaded {name} ({family})")
                all_representations[name] = extract_representations(
                    model,
                    loader,
                    args.layer_index,
                    name,
                )
                model_names.append(name)
        except Exception as exc:
            failed_models.append((name, str(exc)))
            print(f"  FAILED {name}: {exc}")

    return all_representations, model_names, failed_models


def print_failed_models(failed_models) -> None:
    if not failed_models:
        return
    print("\nSkipped models (failed to load/extract):")
    for name, reason in failed_models:
        print(f"  - {name}: {reason}")


def compute_skc_matrix(model_names, all_representations):
    count = len(model_names)
    skc_matrix = np.zeros((count, count))
    for i, j in tqdm(list(itertools.combinations(range(count), 2)), desc="  Pairs"):
        skc, *_ = compute_skc_from_matrices(
            all_representations[model_names[i]],
            all_representations[model_names[j]],
        )
        skc_matrix[i, j] = skc
        skc_matrix[j, i] = skc
    return skc_matrix


def run_matrix_mode(args, dtype: torch.dtype, probe_samples) -> None:
    all_representations, model_names, failed_models = extract_all_representations(
        args,
        dtype,
        probe_samples,
    )
    print_failed_models(failed_models)

    if len(model_names) < 2:
        raise RuntimeError(
            "Need at least two successfully loaded models to compute SKC matrix."
        )

    print("\nComputing pairwise SKC...")
    skc_matrix = compute_skc_matrix(model_names, all_representations)

    print("\n--- SKC Matrix ---")
    print_skc_matrix(skc_matrix, model_names)

    print("\n--- Pipeline ---")
    n_teachers = min(args.n_teachers, len(model_names) - 1)
    selected = run_pipeline(
        skc_matrix,
        model_names,
        n_teachers=n_teachers,
        redundancy_threshold=args.redundancy_threshold,
    )

    print(f"\n{'=' * 50}")
    print(f"  Recommended teachers ({n_teachers}):")
    for name in selected:
        print(f"    {name}")
    print(f"{'=' * 50}")


def run(args) -> None:
    if len(args.models) < 2:
        raise ValueError("--models requires at least two model IDs.")

    dtype = select_dtype()
    dataset, schema, loaded_from = load_probe_dataset(args.dataset, args.split, args.config)
    probe_samples = build_probe_samples(dataset, schema, args.n)

    print(f"Models  : {args.models}")
    print(f"Dataset : {loaded_from} (split={args.split})")
    print(f"Probe n : {len(probe_samples)}  |  vision_layer={args.layer_index}")
    print(f"DType   : {dtype}\n")

    if len(args.models) == 2:
        run_pairwise_mode(args, dtype, probe_samples)
        return

    run_matrix_mode(args, dtype, probe_samples)
