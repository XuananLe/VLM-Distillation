from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from src.constants import IGNORE_INDEX
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


def compute_per_sample_ce_losses(
    *,
    student_logits: torch.Tensor,
    student_labels: torch.Tensor,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    if student_logits.ndim != 3:
        raise ValueError(f"student_logits must have shape [batch, seq, vocab], got {tuple(student_logits.shape)}")
    if student_labels.ndim != 2:
        raise ValueError(f"student_labels must have shape [batch, seq], got {tuple(student_labels.shape)}")
    if student_logits.shape[:2] != student_labels.shape:
        raise ValueError(
            "student logits and labels must share batch/sequence shape, got "
            f"{tuple(student_logits.shape[:2])} and {tuple(student_labels.shape)}."
        )
    if student_logits.size(1) < 2:
        raise ValueError("Cannot compute per-sample CE with sequence length below 2.")

    shifted_logits = student_logits[:, :-1, :].float().contiguous()
    shifted_labels = student_labels[:, 1:].contiguous()
    token_losses = F.cross_entropy(
        shifted_logits.view(-1, shifted_logits.size(-1)),
        shifted_labels.view(-1),
        ignore_index=ignore_index,
        reduction="none",
    ).view(student_logits.size(0), -1)
    supervised_mask = shifted_labels.ne(ignore_index)
    missing_supervision = ~supervised_mask.any(dim=1)
    if missing_supervision.any():
        bad_indices = missing_supervision.nonzero(as_tuple=True)[0].tolist()
        raise ValueError(f"Cannot compute per-sample GRACE CE losses without supervised tokens: {bad_indices}.")
    supervised_counts = supervised_mask.sum(dim=1).to(dtype=token_losses.dtype)
    return token_losses.sum(dim=1) / supervised_counts


def resolve_teacher_target_batches(
    *,
    student_logits: torch.Tensor,
    cached_teacher_target_batches,
    prepare_input_fn: Callable,
) -> list[TeacherTargetBatch]:
    if cached_teacher_target_batches is None:
        raise ValueError("Teacher KD requires cached teacher logits.")
    return [
        TeacherTargetBatch(
            logits=prepare_input_fn(cached_teacher_logits).to(dtype=student_logits.dtype),
            labels=prepare_input_fn(cached_teacher_labels),
        )
        for cached_teacher_logits, cached_teacher_labels in cached_teacher_target_batches
    ]


def compute_teacher_loss_matrix(
    *,
    student_logits: torch.Tensor,
    student_labels: torch.Tensor,
    model,
    teacher_target_batches: Sequence[TeacherTargetBatch],
    collect_grace_tensors: bool,
    grace_threshold: float,
    distillation_prepare_batch_fn: Callable,
    distillation_loss_fn: Callable,
    student_temperature: float,
    teacher_temperature: float,
) -> tuple[
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    """Compute per-teacher KD losses and optional parameter-space GRACE signals."""
    if not teacher_target_batches:
        raise ValueError("Teacher KD requires at least one teacher target batch.")

    teacher_losses = []
    grace_scores = []
    grace_active_masks = []

    grace_parameters = None
    ce_parameter_grads_by_sample = None
    if collect_grace_tensors:
        grace_parameters = trainable_parameters(model)
        per_sample_ce_losses = compute_per_sample_ce_losses(
            student_logits=student_logits,
            student_labels=student_labels,
        )
        # GRACE is meant to decide teacher usefulness for each example; averaging
        # CE here would collapse it back into one batch-level teacher score.
        ce_parameter_grads_by_sample = [
            compute_parameter_grads(
                loss=sample_ce_loss,
                parameters=grace_parameters,
            )
            for sample_ce_loss in per_sample_ce_losses
        ]

    def append_parameter_grace_score(teacher_loss: torch.Tensor) -> None:
        if grace_parameters is None or ce_parameter_grads_by_sample is None:
            return
        per_sample_agreements = []
        for sample_index, sample_teacher_loss in enumerate(teacher_loss):
            # Keep the KD direction sample-local so teacher agreement can differ
            # across examples in the same mini-batch.
            kd_parameter_grads = compute_parameter_grads(
                loss=sample_teacher_loss,
                parameters=grace_parameters,
            )
            per_sample_agreements.append(
                parameter_gradient_cosine(
                    ce_parameter_grads_by_sample[sample_index],
                    kd_parameter_grads,
                    loss=sample_teacher_loss,
                ).to(device=student_logits.device)
            )
        per_sample_agreement = torch.stack(per_sample_agreements)
        grace_scores.append(per_sample_agreement)
        grace_active_masks.append(per_sample_agreement > grace_threshold)

    for teacher_index, teacher_target_batch in enumerate(teacher_target_batches):
        teacher_logits = teacher_target_batch.logits
        teacher_labels = teacher_target_batch.labels
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
        )
    return (
        teacher_loss_matrix,
        torch.stack(grace_scores, dim=-1),
        torch.stack(grace_active_masks, dim=-1),
    )


__all__ = [
    "TeacherTargetBatch",
    "compute_teacher_loss_matrix",
    "compute_per_sample_ce_losses",
    "resolve_teacher_target_batches",
]
