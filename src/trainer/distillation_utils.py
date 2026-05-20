import contextlib
from pathlib import Path

import torch
from einops import reduce

from src.components.cka import linear_cka_loss
from src.components.forward_utils import (
    get_decoder_hidden_states,
    infer_batch_size,
    pool_model_hidden_states,
    unwrap_tensor,
)
from src.components.matching import load_cka_json, topk_soft_match_student_teacher
from src.components.vision_forward import infer_vision_group_counts, pool_vision_features
from src.utils import find_vision_layer_indices, get_specific_layer

REQUIRED_TEACHER_INPUTS = ("input_ids", "attention_mask", "pixel_values")
OPTIONAL_TEACHER_INPUTS = ("pixel_attention_mask", "image_grid_thw", "image_flags", "image_sizes")


def get_base_model(model):
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
                f"{label} layer index {layer_index} resolved to {normalized}, but valid range is 0-{total_layers - 1}."
            )
        resolved.append(int(normalized))
    return resolved


def prepare_vision_layer_distillation(
    student_model,
    teacher_models,
    student_layer_indices: list[int],
    teacher_layer_indices: list[int],
) -> tuple[list[int], list[list[tuple[int, int]]]]:
    """Resolve student/teacher vision-layer pairs for direct vision-layer distillation."""
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
                    """Store one hooked layer output after unwrapping nested tensors."""
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
    """Pool captured vision-layer outputs into one vector per sample and layer."""
    group_counts = infer_vision_group_counts(model_inputs, batch_size)
    return {
        layer_index: pool_vision_features(
            raw_outputs[layer_index],
            batch_size,
            group_counts=group_counts,
        )
        for layer_index in layer_indices
    }


def build_live_teacher_batches(inputs, num_teachers: int):
    """Collect live-teacher input batches from a collated batch dict."""
    prefixes = []
    if "teacher_input_ids" in inputs:
        prefixes.append("teacher")
    prefixes.extend(f"teacher_{index}" for index in range(num_teachers) if f"teacher_{index}_input_ids" in inputs)

    if not prefixes:
        raise ValueError("No live teacher inputs were found in the batch.")

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


def build_cached_teacher_target_batches(inputs, num_teachers: int):
    """Collect cached teacher-logit batches from a collated batch dict."""
    prefixes = []
    if "teacher_cached_logits" in inputs:
        prefixes.append("teacher")
    prefixes.extend(f"teacher_{index}" for index in range(num_teachers) if f"teacher_{index}_cached_logits" in inputs)

    if not prefixes:
        return None

    return [
        (
            inputs[f"{prefix}_cached_logits"],
            inputs[f"{prefix}_cached_labels"],
        )
        for prefix in prefixes
    ]


def resolve_model_name_for_layer_matching(model) -> str:
    model_config = getattr(model, "config", None)
    for value in (
        getattr(model_config, "_name_or_path", None),
        getattr(model_config, "name_or_path", None),
        getattr(model, "name_or_path", None),
    ):
        if isinstance(value, str) and value:
            return value.rstrip("/")
    raise ValueError("Layer-match artifact lookup requires models loaded from named checkpoints.")


def find_layer_match_json(
    *,
    layer_match_dir: Path,
    student_model_name: str,
    teacher_model_name: str,
) -> Path:
    for candidate_path in sorted(layer_match_dir.rglob("final_layers_cka_matrix.json")):
        payload = load_cka_json(str(candidate_path))
        if payload["model_a_name"] == teacher_model_name and payload["model_b_name"] == student_model_name:
            return candidate_path
    raise ValueError(
        "No layer-match artifact found for teacher/student pair: "
        f"teacher={teacher_model_name!r}, student={student_model_name!r}, "
        f"root={str(layer_match_dir)!r}."
    )


def setup_layer_matching(
    model,
    teacher_models,
    layer_match_json_path,
    layer_match_topk,
    layer_distill_source,
    student_layer_indices,
    teacher_layer_indices,
):
    """Resolve the student-to-teacher layer matches used by auxiliary layer distillation."""
    teacher_layer_soft_matches = []

    if layer_match_json_path:
        layer_match_path = Path(layer_match_json_path)
        if layer_match_path.is_dir():
            student_model_name = resolve_model_name_for_layer_matching(model)
            layer_match_paths = [
                find_layer_match_json(
                    layer_match_dir=layer_match_path,
                    student_model_name=student_model_name,
                    teacher_model_name=resolve_model_name_for_layer_matching(teacher_model),
                )
                for teacher_model in teacher_models
            ]
        else:
            if len(teacher_models) != 1:
                raise ValueError(
                    "A single layer-match JSON supports only single-teacher distillation. "
                    "Pass a directory containing one CKA JSON per teacher for multi-teacher runs."
                )
            layer_match_paths = [layer_match_path]

        resolved_student_layer_indices = None
        for layer_match_path in layer_match_paths:
            matches, _summary = topk_soft_match_student_teacher(
                str(layer_match_path),
                student_key="model_b",
                teacher_key="model_a",
                topk=layer_match_topk,
            )
            match_student_indices = [match["student_layer_index"] for match in matches]
            if resolved_student_layer_indices is None:
                resolved_student_layer_indices = match_student_indices
            elif match_student_indices != resolved_student_layer_indices:
                raise ValueError(
                    "Layer-match artifacts disagree on student layer indices: "
                    f"{match_student_indices} != {resolved_student_layer_indices}."
                )
            teacher_layer_soft_matches.append(matches)
        student_layer_indices = list(resolved_student_layer_indices or [])
    elif layer_distill_source == "vision":
        student_layer_indices, teacher_layer_pairs = prepare_vision_layer_distillation(
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
    elif layer_distill_source == "model":
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
    else:
        raise ValueError(f"Unsupported layer distillation source: {layer_distill_source!r}")

    return student_layer_indices, teacher_layer_soft_matches


def compute_student_representations(
    layer_distill_source,
    student_layer_indices,
    student_inputs,
    student_layer_outputs,
    student_outputs,
):
    """Build pooled student representations for the configured layer-distillation source."""
    if layer_distill_source == "vision":
        student_batch_size = infer_batch_size(student_inputs)
        return pool_vision_representations(
            student_layer_outputs,
            student_layer_indices,
            student_batch_size,
            student_inputs,
        )
    if layer_distill_source == "model":
        student_hidden_states = get_decoder_hidden_states(student_outputs)
        student_attention_mask = student_inputs.get("attention_mask")
        return {
            layer_index: pool_model_hidden_states(student_hidden_states[layer_index], student_attention_mask)
            for layer_index in student_layer_indices
        }
    raise ValueError(f"Unsupported layer distillation source: {layer_distill_source!r}")


def compute_teacher_layer_distillation_loss(
    teacher_model,
    teacher_inputs,
    teacher_layer_soft_matches,
    layer_distill_source,
    student_layer_representations,
    output_hidden_states,
):
    """Run one live teacher forward and optionally compute its auxiliary layer-matching loss."""
    teacher_hook_context = contextlib.nullcontext(None)
    teacher_layer_indices = sorted(
        {
            teacher_layer_index
            for match in teacher_layer_soft_matches
            for teacher_layer_index in match["teacher_layer_indices"]
        }
    )

    if layer_distill_source == "vision":
        teacher_hook_context = capture_layer_outputs(teacher_model, teacher_layer_indices)
    elif layer_distill_source != "model":
        raise ValueError(f"Unsupported layer distillation source: {layer_distill_source!r}")

    with teacher_hook_context as teacher_layer_outputs:
        model_param = next(teacher_model.parameters())
        model_dtype = model_param.dtype if model_param.is_floating_point() else None
        prepared_teacher_inputs = dict(teacher_inputs)
        for key, value in prepared_teacher_inputs.items():
            if torch.is_tensor(value):
                target_dtype = model_dtype if model_dtype is not None and value.is_floating_point() else value.dtype
                prepared_teacher_inputs[key] = value.to(device=model_param.device, dtype=target_dtype)
        with torch.no_grad():
            teacher_outputs = teacher_model(
                **prepared_teacher_inputs,
                return_dict=True,
                output_hidden_states=output_hidden_states,
                # Layer distillation uses hidden states, not teacher logits. Keeping
                # only one logit position avoids a huge sequence-by-vocab allocation.
                logits_to_keep=1,
            )

    if student_layer_representations is None:
        raise ValueError("Layer distillation requires student layer representations.")
    if layer_distill_source == "vision":
        teacher_batch_size = infer_batch_size(teacher_inputs)
        teacher_layer_representations = pool_vision_representations(
            teacher_layer_outputs,
            teacher_layer_indices,
            teacher_batch_size,
            teacher_inputs,
        )
    elif layer_distill_source == "model":
        teacher_hidden_states = get_decoder_hidden_states(teacher_outputs)
        teacher_attention_mask = teacher_inputs.get("attention_mask")
        teacher_layer_representations = {
            layer_index: pool_model_hidden_states(teacher_hidden_states[layer_index], teacher_attention_mask)
            for layer_index in teacher_layer_indices
        }
    else:
        raise ValueError(f"Unsupported layer distillation source: {layer_distill_source!r}")

    soft_match_losses = []
    for match in teacher_layer_soft_matches:
        weighted_losses = [
            weight
            * linear_cka_loss(
                student_layer_representations[match["student_layer_index"]],
                teacher_layer_representations[teacher_layer_index],
            )
            for teacher_layer_index, weight in zip(
                match["teacher_layer_indices"],
                match["teacher_layer_weights"],
                strict=True,
            )
        ]
        if not weighted_losses:
            raise ValueError("Layer match entry has no teacher layers.")
        soft_match_losses.append(reduce(torch.stack(weighted_losses), "t ->", "sum"))
    if not soft_match_losses:
        raise ValueError("Layer distillation has no soft layer matches.")
    layer_loss = reduce(torch.stack(soft_match_losses), "t ->", "mean")

    return layer_loss
