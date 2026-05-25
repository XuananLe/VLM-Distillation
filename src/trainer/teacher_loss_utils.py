from collections.abc import Callable, Sequence

import torch

from src.constants import IGNORE_INDEX


def compute_teacher_loss_matrix(
    *,
    student_logits: torch.Tensor,
    student_labels: torch.Tensor,
    teacher_target_batches: Sequence[tuple[torch.Tensor, torch.Tensor]],
    collect_teacher_targets_for_selection: bool,
    distillation_prepare_batch_fn: Callable,
    distillation_loss_fn: Callable,
    student_temperature: float,
    teacher_temperature: float,
) -> tuple[
    torch.Tensor,               # teacher_loss_matrix [batch, num_teachers]
    list[torch.Tensor] | None,  # selection_teacher_logits
    list[torch.Tensor] | None,  # selection_teacher_labels
]:
    if not teacher_target_batches:
        raise ValueError("Teacher KD requires at least one teacher target batch.")

    teacher_losses = []
    selection_teacher_logits = [] if collect_teacher_targets_for_selection else None
    selection_teacher_labels = [] if collect_teacher_targets_for_selection else None

    for teacher_index, (teacher_logits, teacher_labels) in enumerate(teacher_target_batches):
        if collect_teacher_targets_for_selection:
            selection_teacher_logits.append(teacher_logits)
            selection_teacher_labels.append(teacher_labels)

        # Trie loss reuses its prebuilt trie and updates runtime vocab state if needed.
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

            # Drop the final supervised token so KD aligns with teacher predictions,
            # not teacher EOS continuation.
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

        teacher_losses.append(torch.stack(sample_losses, dim=0))

    return (
        torch.stack(teacher_losses, dim=-1),
        selection_teacher_logits,
        selection_teacher_labels,
    )


__all__ = [
    "compute_teacher_loss_matrix",
]
