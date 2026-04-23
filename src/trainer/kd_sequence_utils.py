from collections.abc import Callable

import torch


def prepare_distillation_sequences(
    *,
    student_logits: torch.Tensor,
    student_labels: torch.Tensor,
    teacher_logits: torch.Tensor,
    teacher_labels: torch.Tensor,
    skip_student_eos: bool,
    skip_teacher_eos: bool,
    ce_grad: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Align one student/teacher token sequence pair onto their shared supervised prefix."""
    # Distillation only compares answer positions. Prompt/user tokens stay masked
    # out through the standard `labels == -100` convention.
    student_logits_masked = student_logits[student_labels != -100]
    teacher_logits_masked = teacher_logits[teacher_labels != -100]
    ce_grad_masked = ce_grad[student_labels != -100] if ce_grad is not None else None

    if skip_student_eos and student_logits_masked.size(0) > 0:
        student_logits_masked = student_logits_masked[:-1]
        if ce_grad_masked is not None:
            ce_grad_masked = ce_grad_masked[:-1]
    if skip_teacher_eos and teacher_logits_masked.size(0) > 0:
        teacher_logits_masked = teacher_logits_masked[:-1]

    # Student and teacher may expose different supervised lengths after masking, so
    # KD uses the shared prefix only instead of trying to do token-level realignment.
    min_len = min(student_logits_masked.size(0), teacher_logits_masked.size(0))
    if ce_grad_masked is not None:
        min_len = min(min_len, ce_grad_masked.size(0))
 
    if min_len == 0:
        return (
            student_logits.new_zeros((0, student_logits.size(-1))),
            teacher_logits.new_zeros((0, teacher_logits.size(-1))),
            None if ce_grad_masked is None else ce_grad.new_zeros((0, ce_grad.size(-1))),
        )

    return (
        student_logits_masked[:min_len],
        teacher_logits_masked[:min_len],
        None if ce_grad_masked is None else ce_grad_masked[:min_len],
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
    skip_student_eos: bool,
    skip_teacher_eos: bool,
    teacher_index: int | None = None,
) -> torch.Tensor:
    """Compute per-sample KD losses for one teacher batch after supervised-token alignment."""
    sample_losses = []
    for sample_index in range(student_logits.size(0)):
        # Sequence alignment is done per sample because each example can have a
        # different number of supervised answer tokens after masking / EOS drop.
        student_logits_masked, teacher_logits_masked, _ = prepare_distillation_sequences(
            student_logits=student_logits[sample_index],
            student_labels=student_labels[sample_index],
            teacher_logits=teacher_logits[sample_index],
            teacher_labels=teacher_labels[sample_index],
            skip_student_eos=skip_student_eos,
            skip_teacher_eos=skip_teacher_eos,
        )
        if student_logits_masked.size(0) > 0:
            sample_losses.append(
                distillation_loss_fn(
                    student_logits=student_logits_masked,
                    teacher_logits=teacher_logits_masked,
                    student_temperature=student_temperature,
                    teacher_temperature=teacher_temperature,
                    teacher_index=teacher_index,
                )
            )
        else:
            sample_losses.append(student_logits.new_zeros(()))
    if not sample_losses:
        return student_logits.new_zeros((student_logits.size(0),))
    return torch.stack(sample_losses)


__all__ = [
    "compute_single_teacher_loss",
    "get_supervised_positions",
    "prepare_distillation_sequences",
]
