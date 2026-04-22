#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.components.cka import compute_skc_from_matrices
from src.components.forward_utils import (
    forward_with_kwarg_retry,
    infer_batch_size,
    prepare_forward_inputs,
    unwrap_tensor,
)
from src.components.vision_forward import pool_vision_features
from src.dataset.vqa_loading import extract_image_as_pil, pick_first_text
from src.skc.data.loading import load_probe_dataset
from src.skc.data.probe import build_loader
from src.skc.runtime.execution import cleanup_inference_objects, select_dtype
from src.skc.vlm.api import load_vlm
from src.skc.vlm.processors import load_vlm_processor
from src.utils import find_vision_layer_indices, get_specific_layer

DEFAULT_MODEL_A = "Qwen/Qwen2-VL-2B-Instruct"
DEFAULT_MODEL_B = "HuggingFaceTB/SmolVLM-256M-Instruct"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot a layer-to-layer CKA heatmap between two VLM vision towers."
    )
    parser.add_argument("--model-a", default=DEFAULT_MODEL_A, help="First model ID.")
    parser.add_argument("--model-b", default=DEFAULT_MODEL_B, help="Second model ID.")
    parser.add_argument("--dataset", default="textvqa", help="Dataset alias or HF dataset id.")
    parser.add_argument("--split", default="train", help="Dataset split to sample from.")
    parser.add_argument("--config", default=None, help="Optional HF dataset config.")
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Skip the first N valid dataset samples before collecting the probe set.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=100,
        help="Number of valid image-question samples to include in the CKA computation.",
    )
    parser.add_argument(
        "--question-ids",
        default=None,
        help=(
            "Optional comma-separated list of question IDs to use instead of --sample-index/--n-samples. "
            "Matches the dataset ID field when present, otherwise falls back to the 0-based valid-example index."
        ),
    )
    parser.add_argument(
        "--last-n-layers",
        type=int,
        default=10,
        help="Compare the final N layers from each model.",
    )
    parser.add_argument(
        "--layer-source",
        choices=("model", "vision"),
        default="model",
        help="Use full-model hidden states or vision-tower layers.",
    )
    parser.add_argument(
        "--token-scope",
        choices=("pooled", "answer"),
        default="pooled",
        help="For model layers, pool all visible tokens or only answer tokens.",
    )
    parser.add_argument(
        "--output-dir",
        default="/output/cka_plots/qwen2vl2b_vs_smolvlm256m_textvqa",
        help="Directory for heatmap/image/json outputs.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="Saved figure DPI.",
    )
    return parser


def preview_answer(row: dict, answer_field: str | None):
    if not answer_field:
        return None
    answer_value = row.get(answer_field)
    if isinstance(answer_value, (list, tuple)):
        return [str(item) for item in answer_value[:3]]
    if answer_value is None:
        return None
    return str(answer_value)


def pick_first_answer_text(row: dict, answer_field: str | None):
    if not answer_field:
        return None
    answer_value = row.get(answer_field)
    if isinstance(answer_value, (list, tuple)):
        for item in answer_value:
            text = pick_first_text(item)
            if text:
                return text
        return None
    return pick_first_text(answer_value)


def parse_question_ids(raw_value: str | None) -> list[str] | None:
    if raw_value is None:
        return None

    raw_value = raw_value.strip()
    if not raw_value:
        return None

    if raw_value.startswith("["):
        parsed = json.loads(raw_value)
        if not isinstance(parsed, list):
            raise ValueError("--question-ids JSON input must decode to a list.")
        return [str(item).strip() for item in parsed if str(item).strip()]

    return [item.strip() for item in raw_value.split(",") if item.strip()]


def pick_probe_samples(
    dataset_name: str,
    split: str,
    config: str | None,
    sample_index: int,
    n_samples: int,
    token_scope: str,
    question_ids: list[str] | None = None,
):
    if question_ids is None and n_samples < 1:
        raise ValueError("--n-samples must be at least 1.")
    if question_ids is not None and len(question_ids) == 0:
        raise ValueError("--question-ids must contain at least one ID when set.")

    dataset, schema, loaded_from = load_probe_dataset(dataset_name, split, config)
    valid_index = 0
    probe_samples = []
    row_indices = []
    question_preview = []
    answer_preview = []
    sample_id_preview = []
    selected_actual_ids = []
    requested_question_ids = question_ids or []
    requested_question_id_set = set(requested_question_ids)
    selected_by_question_id = {}
    id_field = schema.get("id_field")

    for row_index, row in enumerate(dataset):
        question = pick_first_text(row.get(schema["question_field"])) if schema["question_field"] else None
        if not question:
            continue

        try:
            image = extract_image_as_pil(row.get(schema["image_field"]))
        except Exception:
            continue

        dataset_question_id = row.get(id_field) if id_field else None
        candidate_question_ids = []
        if dataset_question_id is not None:
            candidate_question_ids.append(str(dataset_question_id))
        candidate_question_ids.append(str(valid_index))

        if requested_question_ids:
            matched_question_id = next(
                (
                    candidate_question_id
                    for candidate_question_id in candidate_question_ids
                    if candidate_question_id in requested_question_id_set
                    and candidate_question_id not in selected_by_question_id
                ),
                None,
            )
            if matched_question_id is None:
                valid_index += 1
                continue

        if not requested_question_ids and valid_index < sample_index:
            valid_index += 1
            continue

        probe_sample = {"question": question, "image": image}
        if token_scope == "answer":
            answer_text = pick_first_answer_text(row, schema["answer_field"])
            if not answer_text:
                valid_index += 1
                continue
            probe_sample["answer"] = answer_text

        if requested_question_ids:
            selected_by_question_id[matched_question_id] = {
                "probe_sample": probe_sample,
                "row_index": row_index,
                "question": question,
                "answer_preview": preview_answer(row, schema["answer_field"]),
                "actual_id": str(dataset_question_id) if dataset_question_id is not None else str(valid_index),
            }
            valid_index += 1
            if len(selected_by_question_id) >= len(requested_question_ids):
                break
            continue

        probe_samples.append(probe_sample)
        row_indices.append(row_index)
        if len(question_preview) < 3:
            question_preview.append(question)
        answer_item = preview_answer(row, schema["answer_field"])
        if answer_item is not None and len(answer_preview) < 3:
            answer_preview.append(answer_item)
        if len(sample_id_preview) < 10:
            sample_id_preview.append(
                str(dataset_question_id) if dataset_question_id is not None else str(valid_index)
            )
        selected_actual_ids.append(
            str(dataset_question_id) if dataset_question_id is not None else str(valid_index)
        )

        valid_index += 1
        if len(probe_samples) >= n_samples:
            break

    if requested_question_ids:
        missing_question_ids = [
            question_id for question_id in requested_question_ids if question_id not in selected_by_question_id
        ]
        if missing_question_ids:
            raise KeyError(
                "Could not find all requested question IDs in the valid probe pool. "
                f"Missing: {missing_question_ids[:20]}"
            )

        ordered_records = [selected_by_question_id[question_id] for question_id in requested_question_ids]
        probe_samples = [record["probe_sample"] for record in ordered_records]
        row_indices = [record["row_index"] for record in ordered_records]
        question_preview = [record["question"] for record in ordered_records[:3]]
        answer_preview = [
            record["answer_preview"]
            for record in ordered_records
            if record["answer_preview"] is not None
        ][:3]
        sample_id_preview = [record["actual_id"] for record in ordered_records[:10]]
        selected_actual_ids = [record["actual_id"] for record in ordered_records]
        n_samples = len(probe_samples)

    if len(probe_samples) < n_samples:
        raise IndexError(
            f"Requested {n_samples} valid samples starting at index {sample_index}, "
            f"but only collected {len(probe_samples)} from dataset={dataset_name!r}, split={split!r}."
        )

    metadata = {
        "dataset": loaded_from,
        "split": split,
        "config": config,
        "schema": schema,
        "sample_offset": None if requested_question_ids else sample_index,
        "num_samples": len(probe_samples),
        "first_row_index": row_indices[0],
        "last_row_index": row_indices[-1],
        "question_preview": question_preview,
        "first_question": question_preview[0] if question_preview else None,
        "answer_preview": answer_preview,
        "sample_id_preview": sample_id_preview,
        "selected_question_ids": requested_question_ids or selected_actual_ids,
        "image_name_preview": None,
        "image_size": list(probe_samples[0]["image"].size),
    }
    return probe_samples, metadata


def select_final_indices(total_layers: int, last_n_layers: int):
    keep = min(last_n_layers, total_layers)
    start = total_layers - keep
    return list(range(start, total_layers))


def select_final_layer_indices(model, last_n_layers: int):
    vision_info = find_vision_layer_indices(model)
    indices = select_final_indices(vision_info["total_layers"], last_n_layers)
    return vision_info, indices


def get_hidden_states_from_outputs(outputs):
    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states is not None:
        return tuple(hidden_states)

    if isinstance(outputs, dict):
        hidden_states = outputs.get("hidden_states")
        if hidden_states is not None:
            return tuple(hidden_states)

    nested_attr_candidates = (
        "language_model_outputs",
        "language_model_output",
        "text_model_output",
        "text_outputs",
        "model_outputs",
    )
    for attr_name in nested_attr_candidates:
        nested = getattr(outputs, attr_name, None)
        if nested is None and isinstance(outputs, dict):
            nested = outputs.get(attr_name)
        nested_hidden_states = getattr(nested, "hidden_states", None) if nested is not None else None
        if nested_hidden_states is not None:
            return tuple(nested_hidden_states)

    if isinstance(outputs, (tuple, list)):
        for item in outputs:
            if item is None:
                continue
            nested_hidden_states = getattr(item, "hidden_states", None)
            if nested_hidden_states is not None:
                return tuple(nested_hidden_states)

    raise RuntimeError("Model forward output did not expose `hidden_states`.")


def get_decoder_hidden_states(outputs):
    hidden_states = get_hidden_states_from_outputs(outputs)
    if not hidden_states:
        raise RuntimeError("Model returned an empty hidden_states tuple.")
    if len(hidden_states) == 1:
        return list(hidden_states)
    return list(hidden_states[1:])


def pool_model_hidden_states(
    hidden_states: torch.Tensor,
    attention_mask,
    token_mask=None,
) -> torch.Tensor:
    tensor = hidden_states.detach().float().cpu()

    if tensor.ndim == 1:
        return tensor.unsqueeze(0)

    if tensor.ndim == 2:
        return tensor.mean(dim=0, keepdim=True)

    if tensor.ndim == 3:
        if token_mask is not None and token_mask.ndim == 2 and token_mask.shape == tensor.shape[:2]:
            mask = token_mask.detach().float().cpu().unsqueeze(-1)
            denom = mask.sum(dim=1).clamp_min(1.0)
            return (tensor * mask).sum(dim=1) / denom
        if attention_mask is not None and attention_mask.ndim == 2 and attention_mask.shape == tensor.shape[:2]:
            mask = attention_mask.detach().float().cpu().unsqueeze(-1)
            denom = mask.sum(dim=1).clamp_min(1.0)
            return (tensor * mask).sum(dim=1) / denom
        return tensor.mean(dim=1)

    raise ValueError(f"Unsupported hidden-state shape for model-layer pooling: {tuple(tensor.shape)}")


def collect_layer_representations(model, loader, layer_indices: list[int], model_name: str):
    raw_outputs = {}
    handles = []
    layer_metadata = {}
    representations = {layer_index: [] for layer_index in layer_indices}
    total_batches = len(loader)

    for layer_index in layer_indices:
        layer, layer_name = get_specific_layer(model, layer_index)

        def make_hook(index: int, name: str):
            def hook(module, hook_inputs, output):
                del module, hook_inputs
                tensor = unwrap_tensor(output)
                if tensor is None:
                    return
                raw_outputs[index] = {
                    "layer_name": name,
                    "shape": list(tensor.shape),
                    "tensor": tensor.detach(),
                }

            return hook

        handles.append(layer.register_forward_hook(make_hook(layer_index, layer_name)))

    try:
        for batch_index, batch in enumerate(loader, start=1):
            raw_outputs.clear()
            inputs = prepare_forward_inputs(model, batch)
            batch_size = infer_batch_size(inputs)
            with torch.inference_mode():
                forward_with_kwarg_retry(model, inputs)

            for layer_index in layer_indices:
                payload = raw_outputs.get(layer_index)
                if payload is None:
                    raise RuntimeError(
                        f"Layer {layer_index} did not emit hook output for batch {batch_index}."
                    )

                pooled = pool_vision_features(payload["tensor"], batch_size).float().cpu()
                representations[layer_index].append(pooled)
                layer_metadata[layer_index] = {
                    "layer_name": payload["layer_name"],
                    "hook_shape": payload["shape"],
                    "hidden_size": int(pooled.shape[-1]),
                }

            if batch_index == 1 or batch_index == total_batches or batch_index % 10 == 0:
                print(f"  {short_model_name(model_name)}: processed {batch_index}/{total_batches} samples")
    finally:
        for handle in handles:
            handle.remove()

    layer_records = []
    for layer_index in layer_indices:
        layer_info = layer_metadata[layer_index]
        representation_matrix = torch.cat(representations[layer_index], dim=0)
        layer_records.append(
            {
                "layer_index": layer_index,
                "layer_name": layer_info["layer_name"],
                "hook_shape": layer_info["hook_shape"],
                "num_samples": int(representation_matrix.shape[0]),
                "hidden_size": int(representation_matrix.shape[1]),
                "representations": representation_matrix,
            }
        )

    return layer_records


def collect_model_layer_representations(
    model,
    loader,
    last_n_layers: int,
    model_name: str,
    token_scope: str,
):
    representations = None
    layer_metadata = {}
    selected_indices = None
    total_layers = None
    total_batches = len(loader)

    for batch_index, batch in enumerate(loader, start=1):
        inputs = prepare_forward_inputs(model, batch)
        attention_mask = inputs.get("attention_mask")
        answer_token_mask = batch.get("answer_token_mask") if token_scope == "answer" else None
        inputs["output_hidden_states"] = True
        inputs["return_dict"] = True
        inputs["use_cache"] = False

        with torch.inference_mode():
            outputs = forward_with_kwarg_retry(model, inputs)

        hidden_states = get_decoder_hidden_states(outputs)
        if total_layers is None:
            total_layers = len(hidden_states)
            selected_indices = select_final_indices(total_layers, last_n_layers)
            representations = {layer_index: [] for layer_index in selected_indices}

        if len(hidden_states) != total_layers:
            raise RuntimeError(
                f"Model hidden-state count changed across batches: expected {total_layers}, got {len(hidden_states)}."
            )

        for layer_index in selected_indices:
            pooled = pool_model_hidden_states(
                hidden_states[layer_index],
                attention_mask,
                token_mask=answer_token_mask,
            )
            representations[layer_index].append(pooled)
            layer_metadata[layer_index] = {
                "layer_name": f"hidden_states.{layer_index}",
                "hook_shape": list(hidden_states[layer_index].shape),
                "hidden_size": int(pooled.shape[-1]),
            }

        del outputs

        if batch_index == 1 or batch_index == total_batches or batch_index % 10 == 0:
            print(f"  {short_model_name(model_name)}: processed {batch_index}/{total_batches} samples")

    if representations is None or selected_indices is None:
        raise RuntimeError("No model-layer representations were collected.")

    layer_records = []
    for layer_index in selected_indices:
        layer_info = layer_metadata[layer_index]
        representation_matrix = torch.cat(representations[layer_index], dim=0)
        layer_records.append(
            {
                "layer_index": layer_index,
                "layer_name": layer_info["layer_name"],
                "hook_shape": layer_info["hook_shape"],
                "num_samples": int(representation_matrix.shape[0]),
                "hidden_size": int(representation_matrix.shape[1]),
                "representations": representation_matrix,
            }
        )

    return layer_records, total_layers


def extract_model_layers(
    model_name: str,
    dtype: torch.dtype,
    probe_samples,
    last_n_layers: int,
    layer_source: str,
    token_scope: str,
):
    model = processor = loader = None
    try:
        model, family = load_vlm(model_name, dtype)
        processor = load_vlm_processor(model_name, family, model)
        loader = build_loader(processor, probe_samples, family, model=model)
        metadata = {
            "model_name": model_name,
            "family": family,
            "layer_source": layer_source,
        }

        if layer_source == "vision":
            if token_scope != "pooled":
                raise ValueError("--token-scope answer is only supported with --layer-source model.")
            vision_info, layer_indices = select_final_layer_indices(model, last_n_layers)
            print(
                f"Collecting {len(probe_samples)} pooled representations from "
                f"{short_model_name(model_name)} across {len(layer_indices)} vision layers"
            )
            layer_records = collect_layer_representations(model, loader, layer_indices, model_name)
            metadata.update(
                {
                    "layer_total": vision_info["total_layers"],
                    "layer_stack_path": vision_info["encoder_path"],
                    "selected_layer_indices": layer_indices,
                }
            )
        else:
            scope_label = "answer-token" if token_scope == "answer" else "pooled"
            print(
                f"Collecting {len(probe_samples)} {scope_label} representations from "
                f"{short_model_name(model_name)} across the last {last_n_layers} full-model layers"
            )
            layer_records, total_layers = collect_model_layer_representations(
                model,
                loader,
                last_n_layers,
                model_name,
                token_scope,
            )
            metadata.update(
                {
                    "layer_total": total_layers,
                    "layer_stack_path": "model.hidden_states",
                    "selected_layer_indices": [record["layer_index"] for record in layer_records],
                }
            )

        metadata = {
            **metadata,
            "selected_layers": [
                {
                    "layer_index": record["layer_index"],
                    "layer_name": record["layer_name"],
                    "hook_shape": record["hook_shape"],
                    "num_samples": record["num_samples"],
                    "hidden_size": record["hidden_size"],
                }
                for record in layer_records
            ],
        }
        return layer_records, metadata
    finally:
        cleanup_inference_objects(model=model, processor=processor, loader=loader)


def build_cka_matrix(records_a, records_b):
    matrix = np.zeros((len(records_a), len(records_b)), dtype=np.float32)
    for row_index, record_a in enumerate(records_a):
        for col_index, record_b in enumerate(records_b):
            cka, *_ = compute_skc_from_matrices(
                record_a["representations"],
                record_b["representations"],
            )
            matrix[row_index, col_index] = cka
    return matrix


def short_model_name(model_name: str) -> str:
    return model_name.split("/")[-1]


def save_heatmap(
    matrix: np.ndarray,
    records_a,
    records_b,
    sample_metadata: dict,
    model_a: str,
    model_b: str,
    layer_source: str,
    output_path: Path,
    dpi: int,
):
    labels_a = [str(record["layer_index"]) for record in records_a]
    labels_b = [str(record["layer_index"]) for record in records_b]
    question_preview = " | ".join(sample_metadata["question_preview"])
    wrapped_question_preview = textwrap.fill(f"Preview: {question_preview}", width=80)
    sample_id_preview = sample_metadata.get("sample_id_preview") or []
    sample_id_text = ", ".join(sample_id_preview[:10])
    wrapped_sample_id_preview = (
        textwrap.fill(f"Question IDs: {sample_id_text}", width=80)
        if sample_id_text
        else None
    )

    fig_width = max(9.0, 5.5 + len(labels_b) * 0.4)
    fig_height = max(8.0, 4.8 + len(labels_a) * 0.35)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    image = ax.imshow(
        matrix,
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        aspect="auto",
        origin="lower",
    )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Linear CKA")

    ax.set_xticks(np.arange(len(labels_b)))
    ax.set_xticklabels(labels_b, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(np.arange(len(labels_a)))
    ax.set_yticklabels(labels_a, fontsize=9)
    layer_label = "model hidden layer index" if layer_source == "model" else "vision layer index"
    ax.set_xlabel(f"{short_model_name(model_b)} {layer_label}")
    ax.set_ylabel(f"{short_model_name(model_a)} {layer_label}")

    title_prefix = "Full-Model Layer CKA" if layer_source == "model" else "Vision-Layer CKA"

    title = (
        f"{title_prefix}: {short_model_name(model_a)} vs {short_model_name(model_b)}\n"
        f"{sample_metadata['dataset']} split={sample_metadata['split']} | "
        f"samples={sample_metadata['num_samples']} | offset={sample_metadata['sample_offset']}\n"
        f"{wrapped_question_preview}"
    )
    if wrapped_sample_id_preview:
        title = f"{title}\n{wrapped_sample_id_preview}"
    ax.set_title(title, fontsize=11)

    annotation_fontsize = max(6, min(10, int(14 - 0.35 * max(matrix.shape))))
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            value = float(matrix[row_index, col_index])
            text_color = "white" if value < 0.55 else "black"
            ax.text(
                col_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=annotation_fontsize,
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.layer_source == "vision" and args.token_scope != "pooled":
        raise ValueError("--token-scope answer is only supported with --layer-source model.")
    question_ids = parse_question_ids(args.question_ids)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dtype = select_dtype()
    probe_samples, sample_metadata = pick_probe_samples(
        dataset_name=args.dataset,
        split=args.split,
        config=args.config,
        sample_index=args.sample_index,
        n_samples=args.n_samples,
        token_scope=args.token_scope,
        question_ids=question_ids,
    )

    sample_image_path = output_dir / "sample_image.png"
    probe_samples[0]["image"].save(sample_image_path)

    print(f"Model A      : {args.model_a}")
    print(f"Model B      : {args.model_b}")
    print(f"Dataset      : {sample_metadata['dataset']} (split={args.split})")
    print(f"Layer source : {args.layer_source}")
    print(f"Token scope  : {args.token_scope}")
    print(
        f"Samples      : {sample_metadata['num_samples']}  |  "
        f"offset={sample_metadata['sample_offset']}"
    )
    print(
        f"Row span     : {sample_metadata['first_row_index']} -> "
        f"{sample_metadata['last_row_index']}"
    )
    print(f"DType        : {dtype}")
    print(f"Output dir   : {output_dir}")
    print()
    print(f"Q preview    : {sample_metadata['question_preview']}")
    if sample_metadata["sample_id_preview"]:
        print(f"QID preview  : {sample_metadata['sample_id_preview']}")
    if sample_metadata["answer_preview"]:
        print(f"Ans preview  : {sample_metadata['answer_preview']}")
    print()

    records_a, metadata_a = extract_model_layers(
        model_name=args.model_a,
        dtype=dtype,
        probe_samples=probe_samples,
        last_n_layers=args.last_n_layers,
        layer_source=args.layer_source,
        token_scope=args.token_scope,
    )
    records_b, metadata_b = extract_model_layers(
        model_name=args.model_b,
        dtype=dtype,
        probe_samples=probe_samples,
        last_n_layers=args.last_n_layers,
        layer_source=args.layer_source,
        token_scope=args.token_scope,
    )

    cka_matrix = build_cka_matrix(records_a, records_b)

    heatmap_path = output_dir / "final_layers_cka_heatmap.png"
    matrix_path = output_dir / "final_layers_cka_matrix.json"
    save_heatmap(
        matrix=cka_matrix,
        records_a=records_a,
        records_b=records_b,
        sample_metadata=sample_metadata,
        model_a=args.model_a,
        model_b=args.model_b,
        layer_source=args.layer_source,
        output_path=heatmap_path,
        dpi=args.dpi,
    )

    payload = {
        "sample": sample_metadata,
        "dtype": str(dtype).replace("torch.", ""),
        "n_samples": args.n_samples,
        "layer_source": args.layer_source,
        "token_scope": args.token_scope,
        "model_a": metadata_a,
        "model_b": metadata_b,
        "cka_matrix": cka_matrix.tolist(),
        "outputs": {
            "heatmap_png": str(heatmap_path),
            "sample_image_png": str(sample_image_path),
        },
    }
    matrix_path.write_text(json.dumps(payload, indent=2))

    print("Artifacts")
    print(f"  Heatmap : {heatmap_path}")
    print(f"  Matrix  : {matrix_path}")
    print(f"  Image   : {sample_image_path}")
    print(f"  Samples : {sample_metadata['num_samples']}")
    print(
        "  Range   : "
        f"min={float(cka_matrix.min()):.4f}, "
        f"mean={float(cka_matrix.mean()):.4f}, "
        f"max={float(cka_matrix.max()):.4f}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
