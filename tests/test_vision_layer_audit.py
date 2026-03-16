import argparse
import json
import os
import shlex
import sys
import traceback
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.components.skc import extract_sample_representations
from src.components.vision_forward import (
    forward_with_kwarg_retry,
    infer_batch_size,
    pool_vision_features,
    prepare_forward_inputs,
    unwrap_tensor,
)
from src.skc.config.vision import FAMILY_VISION_SPECS, MODEL_PRESETS
from src.skc.data.loading import build_probe_samples, load_probe_dataset
from src.skc.data.probe import build_loader
from src.skc.runtime.execution import cleanup_inference_objects, select_dtype
from src.skc.vlm.api import load_vlm
from src.skc.vlm.families import resolve_model_family
from src.skc.vlm.processors import load_vlm_processor
from src.utils import find_vision_layer_indices, get_specific_layer, resolve_module_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit whether SKC is extracting from the true last vision encoder layer."
    )
    parser.add_argument("--models", nargs="*", help="Explicit model IDs to audit.")
    parser.add_argument(
        "--preset",
        default="vision_benchmark_current_working_set",
        choices=sorted(MODEL_PRESETS),
        help="Preset model list to audit when --models is not supplied.",
    )
    parser.add_argument("--dataset", default="textvqa", help="Probe dataset alias or HF dataset id.")
    parser.add_argument("--split", default="train", help="Probe dataset split.")
    parser.add_argument("--config", default=None, help="Optional dataset config.")
    parser.add_argument("--n", type=int, default=1, help="Number of probe samples to use per model.")
    parser.add_argument(
        "--json_out",
        default=str(ROOT / "vision_layer_audit_report.json"),
        help="Where to save the JSON audit report.",
    )
    parser.add_argument(
        "--fail_on_review",
        action="store_true",
        help="Exit non-zero when any model is marked manual_review.",
    )
    return parser


def parse_args(argv=None):
    return build_parser().parse_args(argv)


def get_model_list(args) -> list[str]:
    if args.models:
        return list(args.models)
    return list(MODEL_PRESETS[args.preset])


def resolve_path_if_present(root, path: str):
    try:
        return resolve_module_path(root, path)
    except (AttributeError, IndexError, KeyError, TypeError):
        return None


def get_config_value(config, path: str):
    current = config
    for part in path.split("."):
        if current is None:
            return None
        if isinstance(current, dict):
            current = current.get(part)
        else:
            current = getattr(current, part, None)
    return current


def normalize_stack(module_stack):
    if module_stack is None:
        return None
    if isinstance(module_stack, (list, tuple)):
        return list(module_stack)
    if hasattr(module_stack, "__len__") and hasattr(module_stack, "__getitem__"):
        try:
            return [module_stack[index] for index in range(len(module_stack))]
        except Exception:
            return None
    return None


def infer_expected_layer_count(config, spec):
    if spec is None:
        return None, None
    for path in spec.depth_config_hints:
        value = get_config_value(config, path)
        if isinstance(value, int) and value > 0:
            return value, path
    return None, None


def collect_documented_encoder_paths(model, discovered_encoder_path: str, spec):
    encoder_paths = []
    seen = set()
    hint_paths = ()
    if spec is not None:
        hint_paths = spec.encoder_path_hints

    for path in list(hint_paths) + [discovered_encoder_path]:
        if not path or path in seen:
            continue
        module = resolve_path_if_present(model, path)
        if module is None:
            continue
        seen.add(path)
        encoder_paths.append((path, module))
    return encoder_paths


def collect_documented_stacks(model, encoder_paths, spec):
    if spec is None:
        return []

    stacks = []
    seen_stack_ids = set()

    for encoder_path, encoder_module in encoder_paths:
        for stack_hint in spec.layer_stack_hints:
            stack_module = resolve_path_if_present(encoder_module, stack_hint)
            stack_list = normalize_stack(stack_module)
            if not stack_list:
                continue
            stack_ids = tuple(id(layer) for layer in stack_list)
            if stack_ids in seen_stack_ids:
                continue
            seen_stack_ids.add(stack_ids)
            full_path = f"{encoder_path}.{stack_hint}"
            stacks.append(
                {
                    "encoder_path": encoder_path,
                    "stack_path": full_path,
                    "layers": stack_list,
                }
            )

    return stacks


def collect_hook_outputs(model, batch, total_layers: int):
    inputs = prepare_forward_inputs(model, batch)
    batch_size = infer_batch_size(inputs)
    outputs = {}
    handles = []
    layers = []

    for index in range(total_layers):
        layer, layer_name = get_specific_layer(model, index)
        layers.append((index, layer_name, layer))

        def make_hook(name):
            def hook(module, hook_inputs, output):
                del module, hook_inputs
                tensor = unwrap_tensor(output)
                if tensor is None:
                    return
                outputs[name] = tensor.detach().float().cpu()

            return hook

        handles.append(layer.register_forward_hook(make_hook(layer_name)))

    try:
        with torch.no_grad():
            forward_with_kwarg_retry(model, inputs)
    finally:
        for handle in handles:
            handle.remove()

    layer_summaries = []
    for index, layer_name, _layer in layers:
        tensor = outputs.get(layer_name)
        layer_summaries.append(
            {
                "index": index,
                "name": layer_name,
                "shape": list(tensor.shape) if tensor is not None else None,
            }
        )

    return outputs, layer_summaries, batch_size


def build_probe_batch(processor, family, model, probe_samples):
    loader = build_loader(processor, probe_samples, family, model=model)
    try:
        return next(iter(loader))
    finally:
        del loader


def get_report_path(args) -> Path:
    return Path(args.json_out).resolve()


def audit_one_model(model_name: str, probe_samples, dtype: torch.dtype):
    model = processor = None
    report = {"model": model_name}
    _cfg, inferred_family = resolve_model_family(model_name)
    spec = FAMILY_VISION_SPECS.get(inferred_family)
    report["family"] = inferred_family

    try:
        model, family = load_vlm(model_name, dtype)
        processor = load_vlm_processor(model_name, family, model)
        batch = build_probe_batch(processor, family, model, probe_samples)
        spec = FAMILY_VISION_SPECS.get(family)
        vision_info = find_vision_layer_indices(model)
        discovered_last_layer, discovered_last_name = get_specific_layer(model, -1)
        expected_count, expected_count_path = infer_expected_layer_count(model.config, spec)
        encoder_candidates = collect_documented_encoder_paths(
            model,
            vision_info["encoder_path"],
            spec,
        )
        documented_stacks = collect_documented_stacks(model, encoder_candidates, spec)
        manual_last_matches = []

        for stack in documented_stacks:
            stack_layers = stack["layers"]
            manual_last_matches.append(
                {
                    "stack_path": stack["stack_path"],
                    "length": len(stack_layers),
                    "matches_discovered_last": bool(stack_layers[-1] is discovered_last_layer),
                }
            )

        outputs, layer_summaries, batch_size = collect_hook_outputs(
            model,
            batch,
            vision_info["total_layers"],
        )
        extracted = extract_sample_representations(model, batch, -1).float().cpu()
        last_hook = outputs.get(discovered_last_name)
        pooled_last_hook = None
        max_abs_diff = None
        extractor_matches_last_hook = False

        if last_hook is not None:
            pooled_last_hook = pool_vision_features(last_hook, batch_size)
            max_abs_diff = float((pooled_last_hook - extracted).abs().max().item())
            extractor_matches_last_hook = torch.allclose(
                pooled_last_hook,
                extracted,
                atol=1e-4,
                rtol=1e-3,
            )

        all_hooks_fired = all(summary["shape"] is not None for summary in layer_summaries)
        documented_match = any(item["matches_discovered_last"] for item in manual_last_matches)

        status = "passed"
        reasons = []
        if expected_count is not None and expected_count != vision_info["total_layers"]:
            status = "failed"
            reasons.append(
                f"discovered layer count {vision_info['total_layers']} != config {expected_count_path}={expected_count}"
            )
        if not all_hooks_fired:
            status = "failed"
            reasons.append("not all discovered vision layer hooks fired in the forward pass")
        if documented_stacks and not documented_match:
            if status != "failed":
                status = "manual_review"
            reasons.append("documented final stack layer does not match discovered last layer")
        if not documented_stacks:
            status = "manual_review"
            reasons.append("no documented vision stack hint resolved for this family")
        if not extractor_matches_last_hook:
            reasons.append("advisory: pooled last hook output does not exactly match extract_sample_representations(..., -1)")
        if spec is not None and spec.requires_manual_review:
            if status == "passed":
                status = "manual_review"
            reasons.append(spec.notes)
        if len(documented_stacks) > 1:
            if status == "passed":
                status = "manual_review"
            reasons.append("multiple documented vision stacks resolved; selection is architecture-dependent")

        report.update(
            {
                "family": family,
                "status": status,
                "reasons": reasons,
                "docs": list(spec.docs) if spec is not None else [],
                "notes": spec.notes if spec is not None else "",
                "vision_info": {
                    "encoder_type": vision_info["encoder_type"],
                    "encoder_path": vision_info["encoder_path"],
                    "total_layers": vision_info["total_layers"],
                    "layer_names": vision_info["layer_names"],
                    "selected_last_layer": discovered_last_name,
                },
                "expected_count_from_config": expected_count,
                "expected_count_path": expected_count_path,
                "documented_encoder_candidates": [path for path, _module in encoder_candidates],
                "documented_stacks": [
                    {
                        "encoder_path": stack["encoder_path"],
                        "stack_path": stack["stack_path"],
                        "length": len(stack["layers"]),
                    }
                    for stack in documented_stacks
                ],
                "documented_last_matches": manual_last_matches,
                "hook_summary": {
                    "all_hooks_fired": all_hooks_fired,
                    "layer_shapes": layer_summaries,
                    "extractor_matches_last_hook": extractor_matches_last_hook,
                    "max_abs_diff_vs_extractor": max_abs_diff,
                    "last_hook_shape": list(last_hook.shape) if last_hook is not None else None,
                    "pooled_last_hook_shape": (
                        list(pooled_last_hook.shape) if pooled_last_hook is not None else None
                    ),
                    "extractor_shape": list(extracted.shape),
                },
            }
        )
        return report
    except Exception as exc:
        if spec is not None and spec.requires_manual_review:
            report.update(
                {
                    "status": "manual_review",
                    "docs": list(spec.docs),
                    "notes": spec.notes,
                    "reasons": [
                        spec.notes,
                        f"runtime audit skipped or blocked: {type(exc).__name__}: {exc}",
                    ],
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            return report
        report.update(
            {
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        return report
    finally:
        cleanup_inference_objects(model=model, processor=processor)


def summarize_report(report):
    selected_layer = None
    vision_info = report.get("vision_info")
    if isinstance(vision_info, dict):
        selected_layer = vision_info.get("selected_last_layer")
    layer_suffix = f" | layer={selected_layer}" if selected_layer else ""

    if report["status"] == "passed":
        return (
            f"[PASS] {report['model']} | family={report['family']} | "
            f"layers={report['vision_info']['total_layers']} | "
            f"selected={report['vision_info']['selected_last_layer']}"
        )
    if report["status"] == "manual_review":
        return (
            f"[REVIEW] {report['model']} | family={report.get('family')} | "
            f"{'; '.join(report.get('reasons', []))}{layer_suffix}"
        )
    reason = report.get("error") or "; ".join(report.get("reasons", []))
    return f"[FAIL] {report['model']} | {reason}{layer_suffix}"


def main(argv=None):
    args = parse_args(argv)
    models = get_model_list(args)
    dtype = select_dtype()
    dataset, schema, loaded_from = load_probe_dataset(args.dataset, args.split, args.config)
    probe_samples = build_probe_samples(dataset, schema, args.n)

    print(f"Models     : {models}")
    print(f"Dataset    : {loaded_from} (split={args.split})")
    print(f"Probe n    : {len(probe_samples)}")
    print(f"DType      : {dtype}")
    print()

    reports = []
    for model_name in models:
        report = audit_one_model(model_name, probe_samples, dtype)
        reports.append(report)
        print(summarize_report(report))

    output_path = get_report_path(args)
    output_path.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print()
    print(f"Saved JSON report to {output_path}")

    failed = [report for report in reports if report["status"] == "failed"]
    reviews = [report for report in reports if report["status"] == "manual_review"]
    passed = [report for report in reports if report["status"] == "passed"]

    print(
        f"Summary: {len(passed)} passed, {len(reviews)} manual_review, {len(failed)} failed."
    )

    if failed or (args.fail_on_review and reviews):
        return 1
    return 0


class VisionLayerAuditTest(unittest.TestCase):
    def test_vision_layer_audit(self):
        if os.environ.get("RUN_VISION_LAYER_AUDIT") != "1":
            self.skipTest("Set RUN_VISION_LAYER_AUDIT=1 to run the vision-layer audit test.")

        audit_args = shlex.split(os.environ.get("VISION_LAYER_AUDIT_ARGS", ""))
        self.assertEqual(main(audit_args), 0)


if __name__ == "__main__":
    raise SystemExit(main())
