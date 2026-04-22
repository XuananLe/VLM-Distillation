import gc
from types import SimpleNamespace

import torch
from einops import rearrange

from src.components.forward_utils import forward_with_kwarg_retry


REQUIRED_TEACHER_INPUTS = ("input_ids", "attention_mask", "pixel_values")
OPTIONAL_TEACHER_INPUTS = ("pixel_attention_mask", "image_grid_thw", "image_flags", "image_sizes")


def release_eval_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()


def _infer_model_device_and_dtype(teacher_model):
    try:
        param = next(teacher_model.parameters())
    except StopIteration:
        return getattr(teacher_model, "device", None), None
    dtype = param.dtype if param.is_floating_point() else None
    return param.device, dtype


def prepare_teacher_model_inputs(teacher_model, teacher_inputs):
    model_device, model_dtype = _infer_model_device_and_dtype(teacher_model)
    prepared_inputs = {}
    for key, value in teacher_inputs.items():
        if not torch.is_tensor(value):
            prepared_inputs[key] = value
            continue

        target_dtype = model_dtype if model_dtype is not None and value.is_floating_point() else value.dtype
        if model_device is None:
            prepared_inputs[key] = value.to(dtype=target_dtype)
        else:
            prepared_inputs[key] = value.to(device=model_device, dtype=target_dtype)
    return prepared_inputs


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


def build_cached_teacher_batches(inputs, num_teachers: int):
    prefixes = []
    if "teacher_cached_logits" in inputs:
        prefixes.append("teacher")
    prefixes.extend(
        f"teacher_{index}"
        for index in range(num_teachers)
        if f"teacher_{index}_cached_logits" in inputs
    )

    if not prefixes:
        return None

    return [
        (
            inputs[f"{prefix}_cached_logits"],
            inputs[f"{prefix}_cached_labels"],
        )
        for prefix in prefixes
    ]


def select_supervised_logit_positions(labels: torch.Tensor) -> torch.Tensor | None:
    if labels.ndim != 2:
        return None

    positions = labels.ne(-100).any(dim=0).nonzero(as_tuple=False).squeeze(-1)
    if positions.numel() == 0 or positions.numel() == labels.size(1):
        return None
    return positions


def select_labels_at_positions(
    labels: torch.Tensor,
    positions: torch.Tensor | None,
) -> torch.Tensor:
    if positions is None:
        return labels
    return labels.index_select(dim=1, index=positions.to(device=labels.device))


def _slice_hidden_states_for_logits(
    hidden_states: torch.Tensor,
    logits_to_keep: int | torch.Tensor,
) -> torch.Tensor:
    if isinstance(logits_to_keep, int):
        if logits_to_keep == 0:
            return hidden_states
        return hidden_states[:, -logits_to_keep:, :]

    if not isinstance(logits_to_keep, torch.Tensor):
        raise TypeError(f"Unsupported logits_to_keep type: {type(logits_to_keep)!r}")

    positions = rearrange(logits_to_keep, "... -> (...)").to(
        device=hidden_states.device,
        dtype=torch.long,
    )
    return hidden_states.index_select(dim=1, index=positions)


def _compute_teacher_forward_with_manual_logit_slice(
    teacher_model,
    teacher_inputs,
    *,
    output_hidden_states: bool,
    logits_to_keep: int | torch.Tensor,
):
    model_type = getattr(getattr(teacher_model, "config", None), "model_type", None)
    if model_type != "qwen2_vl":
        return None

    backbone_inputs = {
        **teacher_inputs,
        "return_dict": True,
        "output_hidden_states": output_hidden_states,
    }
    backbone_outputs = forward_with_kwarg_retry(teacher_model.model, backbone_inputs)
    hidden_states = backbone_outputs[0]
    sliced_hidden_states = _slice_hidden_states_for_logits(hidden_states, logits_to_keep)
    logits = teacher_model.lm_head(sliced_hidden_states)
    return SimpleNamespace(logits=logits, hidden_states=backbone_outputs.hidden_states)


def compute_teacher_forward(
    teacher_model,
    teacher_inputs,
    *,
    output_hidden_states: bool = False,
    suppress_stdout: bool = False,
    logits_to_keep: int | torch.Tensor | None = None,
):
    import io
    from contextlib import nullcontext, redirect_stdout

    prepared_teacher_inputs = prepare_teacher_model_inputs(teacher_model, teacher_inputs)

    call_inputs = {
        **prepared_teacher_inputs,
        "return_dict": True,
        "output_hidden_states": output_hidden_states,
    }
    if logits_to_keep is not None:
        call_inputs["logits_to_keep"] = logits_to_keep

    with torch.no_grad():
        stdout_context = redirect_stdout(io.StringIO()) if suppress_stdout else nullcontext()
        with stdout_context:
            manual_outputs = None
            if logits_to_keep is not None:
                manual_outputs = _compute_teacher_forward_with_manual_logit_slice(
                    teacher_model,
                    prepared_teacher_inputs,
                    output_hidden_states=output_hidden_states,
                    logits_to_keep=logits_to_keep,
                )
            if manual_outputs is not None:
                return manual_outputs
            return forward_with_kwarg_retry(teacher_model, call_inputs)
