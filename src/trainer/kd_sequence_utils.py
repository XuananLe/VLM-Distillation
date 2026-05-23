from collections.abc import Callable

import torch


def prepare_distillation_sequences(
    *,
    student_logits: torch.Tensor,
    student_labels: torch.Tensor,
    teacher_logits: torch.Tensor,
    teacher_labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Distillation only compares answer positions. Prompt/user tokens stay masked
    # out through the standard `labels == -100` convention.
    student_logits_masked = student_logits[student_labels != -100]
    teacher_logits_masked = teacher_logits[teacher_labels != -100]

    if student_logits_masked.size(0) > 0:
        student_logits_masked = student_logits_masked[:-1]
    if teacher_logits_masked.size(0) > 0:
        teacher_logits_masked = teacher_logits_masked[:-1]

    # Student and teacher may expose different supervised lengths after masking, so
    # KD uses the shared prefix only instead of trying to do token-level realignment.
    min_len = min(student_logits_masked.size(0), teacher_logits_masked.size(0))

    if min_len == 0:
        raise ValueError("Student and teacher labels have no aligned supervised answer tokens after masking.")

    return (
        student_logits_masked[:min_len],
        teacher_logits_masked[:min_len],
    )


def get_supervised_positions(
    labels: torch.Tensor,
    *,
    skip_last: bool = False,
) -> torch.Tensor:
    """Return the supervised token positions, optionally dropping the last one as EOS."""
    positions = labels.ne(-100).nonzero(as_tuple=False).squeeze(-1)
    if skip_last and positions.numel() > 0:
        positions = positions[:-1]
    return positions


def compute_single_teacher_loss(
    *,
    student_logits: torch.Tensor,
    student_labels: torch.Tensor,
    teacher_logits: torch.Tensor,
    teacher_labels: torch.Tensor,
    distillation_loss_fn: Callable,
    student_temperature: float,
    teacher_temperature: float,
    teacher_index: int | None = None,
) -> torch.Tensor:
    """Compute per-sample KD losses for one teacher batch after supervised-token alignment."""
    sample_losses = []
    for sample_index in range(student_logits.size(0)):
        # Sequence alignment is done per sample because each example can have a
        # different number of supervised answer tokens after masking / EOS drop.
        student_logits_masked, teacher_logits_masked = prepare_distillation_sequences(
            student_logits=student_logits[sample_index],
            student_labels=student_labels[sample_index],
            teacher_logits=teacher_logits[sample_index],
            teacher_labels=teacher_labels[sample_index],
        )
        sample_losses.append(
            distillation_loss_fn(
                student_logits=student_logits_masked,
                teacher_logits=teacher_logits_masked,
                student_temperature=student_temperature,
                teacher_temperature=teacher_temperature,
                teacher_index=teacher_index,
            )
        )
    if not sample_losses:
        raise ValueError("Cannot compute teacher loss for an empty student batch.")
    return torch.stack(sample_losses)


__all__ = [
    "compute_single_teacher_loss",
    "get_supervised_positions",
    "prepare_distillation_sequences",
]
