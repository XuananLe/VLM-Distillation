import contextlib
import gc
import re

import torch

from src.components.forward_utils import unwrap_tensor
from src.components.vision_forward import infer_vision_group_counts, pool_vision_features
from src.components.matching import topk_soft_match_student_teacher
from src.components.forward_utils import (
    forward_with_kwarg_retry,
    get_decoder_hidden_states,
    infer_batch_size,
    pool_model_hidden_states,
)
from src.components.skc import linear_cka_loss
from src.utils import find_vision_layer_indices, get_specific_layer, resolve_module_path


REQUIRED_TEACHER_INPUTS = ("input_ids", "attention_mask", "pixel_values")
OPTIONAL_TEACHER_INPUTS = ("pixel_attention_mask", "image_grid_thw", "image_flags")


def release_eval_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()


def is_layer_distillation_enabled(
    *,
    layer_distill_source: str,
    student_layer_indices: list[int] | None,
    layer_match_json_path: str | None,
) -> bool:
    if layer_distill_source not in {"vision", "model"}:
        return False

    has_layer_matching = bool(student_layer_indices) or bool(layer_match_json_path)
    if not has_layer_matching:
        return False

    return True


def get_base_model(model):
    """Extract the underlying model, handling wrappers like DeepSpeed and PEFT."""
    current = model
    while hasattr(current, "module"):
        current = current.module
    while hasattr(current, "get_base_model"):
        next_model = current.get_base_model()
        if next_model is current:
            break
        current = next_model
    return current


def resolve_layer_indices(total_layers: int, layer_indices: list[int], label: str) -> list[int]:
    resolved = []
    for layer_index in layer_indices:
        normalized = total_layers + layer_index if layer_index < 0 else layer_index
        if normalized < 0 or normalized >= total_layers:
            raise IndexError(
                f"{label} layer index {layer_index} resolved to {normalized}, "
                f"but valid range is 0-{total_layers - 1}."
            )
        resolved.append(int(normalized))
    return resolved


def prepare_vision_layer_distillation(
    student_model,
    teacher_models,
    student_layer_indices: list[int],
    teacher_layer_indices: list[int],
) -> tuple[list[int], list[list[tuple[int, int]]]]:
    student_vision_info = find_vision_layer_indices(get_base_model(student_model))
    resolved_student_layer_indices = resolve_layer_indices(
        student_vision_info["total_layers"],
        student_layer_indices,
        "Student vision",
    )

    teacher_layer_pairs = []
    for teacher_index, teacher_model in enumerate(teacher_models):
        teacher_vision_info = find_vision_layer_indices(get_base_model(teacher_model))
        resolved_teacher_layer_indices = resolve_layer_indices(
            teacher_vision_info["total_layers"],
            teacher_layer_indices,
            f"Teacher {teacher_index} vision",
        )
        teacher_layer_pairs.append(list(zip(resolved_student_layer_indices, resolved_teacher_layer_indices)))

    return resolved_student_layer_indices, teacher_layer_pairs

@contextlib.contextmanager
def capture_layer_outputs(model, layer_indices: list[int]):
    raw_outputs = {}
    layer_model = get_base_model(model)

    with contextlib.ExitStack() as stack:
        for layer_index in layer_indices:
            layer, _ = get_specific_layer(layer_model, layer_index)

            def make_hook(index: int):
                def hook(module, hook_inputs, output):
                    del module, hook_inputs
                    raw_outputs[index] = unwrap_tensor(output)
                return hook

            handle = layer.register_forward_hook(make_hook(layer_index))
            stack.callback(handle.remove)
        yield raw_outputs


def pool_vision_representations(
    raw_outputs: dict[int, torch.Tensor],
    layer_indices: list[int],
    batch_size: int,
    model_inputs,
) -> dict[int, torch.Tensor]:
    group_counts = infer_vision_group_counts(model_inputs, batch_size)
    return {
        layer_index: pool_vision_features(
            raw_outputs[layer_index],
            batch_size,
            group_counts=group_counts,
        )
        for layer_index in layer_indices
    }


def build_teacher_batches(inputs, student_inputs, num_teachers: int):
    prefixes = []
    if "teacher_input_ids" in inputs:
        prefixes.append("teacher")
    prefixes.extend(
        f"teacher_{index}"
        for index in range(num_teachers)
        if f"teacher_{index}_input_ids" in inputs
    )

    if not prefixes:
        return [
            (
                {key: value for key, value in student_inputs.items() if key != "labels"},
                student_inputs.get("labels"),
            )
        ]

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


def resolve_gradnorm_reference_params(
    model,
    *,
    layer_distill_source: str,
    student_layer_indices: list[int],
) -> tuple[list[torch.nn.Parameter], str]:
    base_model = get_base_model(model)

    if layer_distill_source == "vision" and student_layer_indices:
        layer_index = max(student_layer_indices)
        layer, layer_name = get_specific_layer(base_model, layer_index)
        params = [param for param in layer.parameters() if param.requires_grad]
        if params:
            return params, f"vision layer {layer_index} ({layer_name})"

    if layer_distill_source == "model" and student_layer_indices:
        layer_index = max(student_layer_indices)
        for path in (
            "model.text_model.model.layers",
            "model.text_model.layers",
            "text_model.model.layers",
            "text_model.layers",
            "model.language_model.model.layers",
            "model.language_model.layers",
            "language_model.model.layers",
            "language_model.layers",
            "model.decoder.layers",
            "decoder.layers",
            "model.layers",
            "layers",
        ):
            try:
                layers = resolve_module_path(base_model, path)
            except (AttributeError, IndexError, KeyError, TypeError):
                continue
            if layer_index >= len(layers):
                continue
            params = [param for param in layers[layer_index].parameters() if param.requires_grad]
            if params:
                return params, f"model layer {layer_index} ({path})"

        layer_param_map: dict[int, list[torch.nn.Parameter]] = {}
        for name, param in base_model.named_parameters():
            if not param.requires_grad or "vision" in name:
                continue
            match = re.search(r"\.layers\.(\d+)\.", name)
            if match:
                layer_param_map.setdefault(int(match.group(1)), []).append(param)
        params = layer_param_map.get(layer_index, [])
        if params:
            return params, f"model layer {layer_index} (named-parameter fallback)"

    for path in ("model.connector", "connector", "model.text_model", "text_model", "model.vision_model", "vision_model"):
        try:
            module = resolve_module_path(base_model, path)
        except (AttributeError, IndexError, KeyError, TypeError):
            continue
        params = [param for param in module.parameters() if param.requires_grad]
        if params:
            return params, path

    params = [param for param in base_model.parameters() if param.requires_grad]
    if not params:
        return [], "no trainable parameters"
    return params, "all trainable parameters"


def setup_layer_matching(
    model,
    teacher_models,
    layer_match_json_path,
    layer_match_topk,
    layer_distill_source,
    student_layer_indices,
    teacher_layer_indices,
):
    """
    Setup layer matching configuration for distillation.
    Returns: (student_layer_indices, teacher_layer_soft_matches)
    """
    teacher_layer_soft_matches = []

    if layer_match_json_path:
        if len(teacher_models) != 1:
            raise ValueError("layer_match_json_path currently supports only single-teacher distillation.")
        matches, summary = topk_soft_match_student_teacher(
            layer_match_json_path,
            student_key="model_b",
            teacher_key="model_a",
            topk=layer_match_topk,
        )
        student_layer_indices = [match["student_layer_index"] for match in matches]
        teacher_layer_soft_matches = [matches]

        print(f"  - Layer match JSON: {layer_match_json_path}")
        print(f"  - Layer match top-k: {summary['topk']}")

    elif layer_distill_source == "vision":
        (
            student_layer_indices,
            teacher_layer_pairs,
        ) = prepare_vision_layer_distillation(
            model,
            teacher_models,
            student_layer_indices,
            teacher_layer_indices,
        )
        teacher_layer_soft_matches = [
            [
                {
                    "student_layer_index": student_layer_index,
                    "teacher_layer_indices": [teacher_layer_index],
                    "teacher_layer_weights": [1.0],
                }
                for student_layer_index, teacher_layer_index in layer_pairs
            ]
            for layer_pairs in teacher_layer_pairs
        ]
    else:
        student_layer_indices = list(student_layer_indices)
        teacher_layer_soft_matches = [
            [
                {
                    "student_layer_index": student_layer_index,
                    "teacher_layer_indices": [teacher_layer_index],
                    "teacher_layer_weights": [1.0],
                }
                for student_layer_index, teacher_layer_index in zip(student_layer_indices, teacher_layer_indices)
            ]
            for _ in teacher_models
        ]

    return student_layer_indices, teacher_layer_soft_matches


def compute_student_representations(
    layer_distill_source,
    student_layer_indices,
    student_inputs,
    student_layer_outputs,
    student_outputs,
):
    """Extract and pool student model representations for layer distillation."""
    if layer_distill_source == "vision":
        student_batch_size = infer_batch_size(student_inputs)
        return pool_vision_representations(
            student_layer_outputs,
            student_layer_indices,
            student_batch_size,
            student_inputs,
        )
    else:
        student_hidden_states = get_decoder_hidden_states(student_outputs)
        student_attention_mask = student_inputs.get("attention_mask")
        return {
            layer_index: pool_model_hidden_states(student_hidden_states[layer_index], student_attention_mask)
            for layer_index in student_layer_indices
        }


def compute_teacher_forward_and_layer_distillation(
    teacher_model,
    teacher_inputs,
    teacher_layer_soft_matches,
    layer_distill_source,
    student_layer_representations,
    output_hidden_states,
    suppress_stdout=False,
):
    """Forward pass through teacher and compute layer distillation representations and loss."""
    import io
    from contextlib import nullcontext, redirect_stdout

    teacher_hook_context = nullcontext(None)
    teacher_layer_indices = sorted(
        {
            teacher_layer_index
            for match in teacher_layer_soft_matches
            for teacher_layer_index in match["teacher_layer_indices"]
        }
    )

    if layer_distill_source == "vision":
        teacher_hook_context = capture_layer_outputs(teacher_model, teacher_layer_indices)

    with teacher_hook_context as teacher_layer_outputs:
        with torch.no_grad():
            stdout_context = redirect_stdout(io.StringIO()) if suppress_stdout else nullcontext()
            with stdout_context:
                teacher_outputs = forward_with_kwarg_retry(
                    teacher_model,
                    {**teacher_inputs, "return_dict": True, "output_hidden_states": output_hidden_states},
                )

    # Compute layer distillation loss
    layer_loss = None
    if student_layer_representations is not None:
        if layer_distill_source == "vision":
            teacher_batch_size = infer_batch_size(teacher_inputs)
            teacher_layer_representations = pool_vision_representations(
                teacher_layer_outputs,
                teacher_layer_indices,
                teacher_batch_size,
                teacher_inputs,
            )
        else:
            teacher_hidden_states = get_decoder_hidden_states(teacher_outputs)
            teacher_attention_mask = teacher_inputs.get("attention_mask")
            teacher_layer_representations = {
                layer_index: pool_model_hidden_states(teacher_hidden_states[layer_index], teacher_attention_mask)
                for layer_index in teacher_layer_indices
            }

        soft_match_losses = []
        for match in teacher_layer_soft_matches:
            weighted_losses = [
                weight * linear_cka_loss(
                    student_layer_representations[match["student_layer_index"]],
                    teacher_layer_representations[teacher_layer_index],
                )
                for teacher_layer_index, weight in zip(
                    match["teacher_layer_indices"],
                    match["teacher_layer_weights"],
                )
            ]
            if weighted_losses:
                soft_match_losses.append(torch.stack(weighted_losses).sum())
        if soft_match_losses:
            layer_loss = torch.stack(soft_match_losses).mean()

    return teacher_outputs, layer_loss
