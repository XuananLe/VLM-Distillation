import gc

import torch

from src.components.forward_utils import forward_with_kwarg_retry


REQUIRED_TEACHER_INPUTS = ("input_ids", "attention_mask", "pixel_values")
OPTIONAL_TEACHER_INPUTS = ("pixel_attention_mask", "image_grid_thw", "image_flags")


def release_eval_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()


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

    call_inputs = {
        **teacher_inputs,
        "return_dict": True,
        "output_hidden_states": output_hidden_states,
    }
    if logits_to_keep is not None:
        call_inputs["logits_to_keep"] = logits_to_keep

    with torch.no_grad():
        stdout_context = redirect_stdout(io.StringIO()) if suppress_stdout else nullcontext()
        with stdout_context:
            return forward_with_kwarg_retry(teacher_model, call_inputs)
