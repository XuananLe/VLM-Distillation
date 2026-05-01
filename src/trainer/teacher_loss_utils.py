from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch

from src.trainer.gradient_utils import (
    compute_parameter_grads,
    parameter_gradient_cosine,
    trainable_parameters,
)
from src.trainer.kd_sequence_utils import compute_single_teacher_loss


@dataclass(slots=True)
class TeacherTargetBatch:
    logits: torch.Tensor
    labels: torch.Tensor


def resolve_teacher_target_batches(
    *,
    student_logits: torch.Tensor,
    teacher_models: Sequence | None,
    live_teacher_batches,
    cached_teacher_target_batches,
    prepare_input_fn: Callable,
) -> list[TeacherTargetBatch]:
    if cached_teacher_target_batches is not None:
        return [
            TeacherTargetBatch(
                logits=prepare_input_fn(cached_teacher_logits).to(
                    dtype=student_logits.dtype
                ),
                labels=prepare_input_fn(cached_teacher_labels),
            )
            for cached_teacher_logits, cached_teacher_labels in cached_teacher_target_batches
        ]

    if not teacher_models or live_teacher_batches is None:
        raise ValueError("Teacher KD requires cached logits or live teacher batches.")
    if len(teacher_models) != len(live_teacher_batches):
        raise ValueError(
            "Teacher model count does not match live teacher batch count: "
            f"{len(teacher_models)} != {len(live_teacher_batches)}."
        )

    target_batches = []
    for teacher_index, (teacher_model, (teacher_inputs, teacher_labels)) in enumerate(
        zip(teacher_models, live_teacher_batches, strict=True)
    ):
        prepared_teacher_labels = prepare_input_fn(teacher_labels)
        if not prepared_teacher_labels.ne(-100).any():
            raise ValueError(f"Teacher {teacher_index} labels contain no supervised answer tokens.")

        prepared_teacher_inputs = prepare_input_fn(teacher_inputs)
        model_param = next(teacher_model.parameters())
        model_dtype = model_param.dtype if model_param.is_floating_point() else None
        for key, value in prepared_teacher_inputs.items():
            if torch.is_tensor(value):
                target_dtype = model_dtype if model_dtype is not None and value.is_floating_point() else value.dtype
                prepared_teacher_inputs[key] = value.to(device=model_param.device, dtype=target_dtype)

        with torch.no_grad():
            teacher_outputs = teacher_model(
                **prepared_teacher_inputs,
                return_dict=True,
                output_hidden_states=False,
            )
        target_batches.append(
            TeacherTargetBatch(
                logits=teacher_outputs.logits.detach(),
                labels=prepared_teacher_labels,
            )
        )
        del teacher_outputs

    return target_batches


def compute_teacher_loss_matrix(
    *,
    student_logits: torch.Tensor,
    student_labels: torch.Tensor,
    model,
    ce_loss: torch.Tensor,
    teacher_target_batches: Sequence[TeacherTargetBatch],
    collect_grace_tensors: bool,
    collect_teacher_targets_for_selection: bool,
    grace_threshold: float,
    distillation_prepare_batch_fn: Callable,
    distillation_loss_fn: Callable,
    student_temperature: float,
    teacher_temperature: float,
    skip_student_eos: bool,
    skip_teacher_eos: bool,
) -> tuple[
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    list[torch.Tensor] | None,
    list[torch.Tensor] | None,
]:
    """Compute per-teacher KD losses and optional parameter-space GRACE signals."""
    if not teacher_target_batches:
        raise ValueError("Teacher KD requires at least one teacher target batch.")

    teacher_losses = []
    grace_scores = []
    grace_active_masks = []
    selection_teacher_logits = [] if collect_teacher_targets_for_selection else None
    selection_teacher_labels = [] if collect_teacher_targets_for_selection else None

    grace_parameters = None
    ce_parameter_grads = None
    if collect_grace_tensors:
        grace_parameters = trainable_parameters(model)
        ce_parameter_grads = compute_parameter_grads(
            loss=ce_loss,
            parameters=grace_parameters,
        )

    def append_parameter_grace_score(teacher_loss: torch.Tensor) -> None:
        if grace_parameters is None or ce_parameter_grads is None:
            return
        kd_parameter_grads = compute_parameter_grads(
            loss=teacher_loss.mean(),
            parameters=grace_parameters,
        )
        agreement = parameter_gradient_cosine(
            ce_parameter_grads,
            kd_parameter_grads,
            loss=teacher_loss,
        ).to(device=student_logits.device)
        per_sample_agreement = agreement.expand(student_logits.size(0))
        grace_scores.append(per_sample_agreement)
        grace_active_masks.append(per_sample_agreement > grace_threshold)

    for teacher_index, teacher_target_batch in enumerate(teacher_target_batches):
        teacher_logits = teacher_target_batch.logits
        teacher_labels = teacher_target_batch.labels
        if collect_teacher_targets_for_selection:
            selection_teacher_logits.append(teacher_logits)
            selection_teacher_labels.append(teacher_labels)
        distillation_prepare_batch_fn(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            teacher_labels=teacher_labels,
            teacher_index=teacher_index,
        )
        teacher_loss = compute_single_teacher_loss(
            student_logits=student_logits,
            student_labels=student_labels,
            teacher_logits=teacher_logits,
            teacher_labels=teacher_labels,
            distillation_loss_fn=distillation_loss_fn,
            student_temperature=student_temperature,
            teacher_temperature=teacher_temperature,
            skip_student_eos=skip_student_eos,
            skip_teacher_eos=skip_teacher_eos,
            teacher_index=teacher_index,
        )
        teacher_losses.append(teacher_loss)
        append_parameter_grace_score(teacher_loss)

    teacher_loss_matrix = torch.stack(teacher_losses, dim=-1)
    if not grace_scores:
        return (
            teacher_loss_matrix,
            None,
            None,
            selection_teacher_logits,
            selection_teacher_labels,
        )
    return (
        teacher_loss_matrix,
        torch.stack(grace_scores, dim=-1),
        torch.stack(grace_active_masks, dim=-1),
        selection_teacher_logits,
        selection_teacher_labels,
    )


__all__ = [
    "TeacherTargetBatch",
    "compute_teacher_loss_matrix",
    "resolve_teacher_target_batches",
]
