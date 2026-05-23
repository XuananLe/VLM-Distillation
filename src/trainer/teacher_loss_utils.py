from collections.abc import Callable, Sequence

import torch
import torch.nn.functional as F

from src.constants import IGNORE_INDEX
from src.trainer.gradient_utils import (
    compute_parameter_grads,
    parameter_gradient_cosine,
    trainable_parameters,
)


def compute_teacher_loss_matrix(
    *,
    student_logits: torch.Tensor,
    student_labels: torch.Tensor,
    model,
    teacher_target_batches: Sequence[tuple[torch.Tensor, torch.Tensor]],
    collect_grace_tensors: bool,
    collect_teacher_targets_for_selection: bool,
    grace_threshold: float,
    distillation_prepare_batch_fn: Callable,
    distillation_loss_fn: Callable,
    student_temperature: float,
    teacher_temperature: float,
) -> tuple[
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    list[torch.Tensor] | None,
    list[torch.Tensor] | None,
]:
    if not teacher_target_batches:
        raise ValueError("Teacher KD requires at least one teacher target batch.")

    teacher_losses = []
    grace_scores = []
    grace_active_masks = []
    selection_teacher_logits = [] if collect_teacher_targets_for_selection else None
    selection_teacher_labels = [] if collect_teacher_targets_for_selection else None

    grace_parameters = None
    ce_parameter_grads_by_sample = None
    if collect_grace_tensors:
        grace_parameters = trainable_parameters(model)
        shifted_logits = student_logits[:, :-1, :].float().contiguous()
        shifted_labels = student_labels[:, 1:].contiguous()
        token_losses = F.cross_entropy(
            shifted_logits.view(-1, shifted_logits.size(-1)),
            shifted_labels.view(-1),
            ignore_index=IGNORE_INDEX,
            reduction="none",
        ).view(student_logits.size(0), -1)
        supervised_mask = shifted_labels.ne(IGNORE_INDEX)
        missing_supervision = ~supervised_mask.any(dim=1)
        if missing_supervision.any():
            bad_indices = missing_supervision.nonzero(as_tuple=True)[0].tolist()
            raise ValueError(f"Cannot compute per-sample GRACE CE losses without supervised tokens: {bad_indices}.")
        per_sample_ce_losses = token_losses.sum(dim=1) / supervised_mask.sum(dim=1).to(
            dtype=token_losses.dtype,
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

    for teacher_index, (teacher_logits, teacher_labels) in enumerate(teacher_target_batches):
        if collect_teacher_targets_for_selection:
            selection_teacher_logits.append(teacher_logits)
            selection_teacher_labels.append(teacher_labels)

        distillation_prepare_batch_fn(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            teacher_labels=teacher_labels,
            teacher_index=teacher_index,
        )
        sample_losses = []
        for sample_index in range(student_logits.size(0)):
            student_logits_masked = student_logits[sample_index][student_labels[sample_index] != IGNORE_INDEX]
            teacher_logits_masked = teacher_logits[sample_index][teacher_labels[sample_index] != IGNORE_INDEX]

            if student_logits_masked.size(0) > 0:
                student_logits_masked = student_logits_masked[:-1]
            if teacher_logits_masked.size(0) > 0:
                teacher_logits_masked = teacher_logits_masked[:-1]

            min_len = min(student_logits_masked.size(0), teacher_logits_masked.size(0))
            if min_len == 0:
                raise ValueError("Student and teacher labels have no aligned supervised answer tokens after masking.")

            sample_losses.append(
                distillation_loss_fn(
                    student_logits=student_logits_masked[:min_len],
                    teacher_logits=teacher_logits_masked[:min_len],
                    student_temperature=student_temperature,
                    teacher_temperature=teacher_temperature,
                    teacher_index=teacher_index,
                )
            )
        teacher_loss = torch.stack(sample_losses)
        teacher_losses.append(teacher_loss)

        if grace_parameters is not None and ce_parameter_grads_by_sample is not None:
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
    "compute_teacher_loss_matrix",
]
