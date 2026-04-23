from collections.abc import Callable

import torch
import torch.nn.functional as F

from src.trainer.kd_sequence_utils import get_supervised_positions


def compute_pooled_ce_grace_grad(
    *,
    student_logits: torch.Tensor,
    student_labels: torch.Tensor,
) -> torch.Tensor:
    """Pool CE logit gradients into one vocab-space direction per sample for GRACE."""
    # GRACE compares one pooled logit-space direction per sample, not full
    # parameter gradients. The pooled vector stays in vocab space: [batch, vocab].
    pooled_grads = []
    vocab_size = student_logits.size(-1)
    zero_grad = student_logits.new_zeros((vocab_size,), dtype=torch.float32)

    for sample_index in range(student_logits.size(0)):
        positions = get_supervised_positions(student_labels[sample_index])
        if positions.numel() == 0:
            pooled_grads.append(zero_grad)
            continue

        sample_labels = student_labels[sample_index, positions]
        # For CE, dL/dlogits = p - y. GRACE only needs one direction per sample, so
        # the per-token gradients are averaged over supervised answer positions.
        sample_grad = F.softmax(student_logits[sample_index, positions].float(), dim=-1)
        sample_grad[torch.arange(sample_labels.numel(), device=sample_labels.device), sample_labels] -= 1.0
        pooled_grads.append(sample_grad.sum(dim=0) / positions.numel())

    if not pooled_grads:
        return student_logits.new_zeros((0, vocab_size), dtype=torch.float32)
    return torch.stack(pooled_grads, dim=0)


def compute_pooled_kd_grace_grad(
    *,
    student_logits: torch.Tensor,
    student_labels: torch.Tensor,
    teacher_logits: torch.Tensor,
    teacher_labels: torch.Tensor,
    distillation_logit_grad_fn: Callable,
    loss_function: str,
    student_temperature: float,
    teacher_temperature: float,
    skip_student_eos: bool,
    skip_teacher_eos: bool,
    teacher_index: int | None = None,
) -> torch.Tensor:
    """Pool KD logit gradients into one vocab-space direction per sample for GRACE."""
    pooled_grads = []
    vocab_size = student_logits.size(-1)
    zero_grad = student_logits.new_zeros((vocab_size,), dtype=torch.float32)

    for sample_index in range(student_logits.size(0)):
        # Use the original student supervised-token count as the normalization
        # anchor so KD and CE pooled gradients stay on a comparable scale.
        supervised_student_count = int(student_labels[sample_index].ne(-100).sum().item())
        if supervised_student_count == 0:
            pooled_grads.append(zero_grad)
            continue

        student_positions = get_supervised_positions(
            student_labels[sample_index],
            skip_last=skip_student_eos,
        )
        teacher_positions = get_supervised_positions(
            teacher_labels[sample_index],
            skip_last=skip_teacher_eos,
        )

        matched_tokens = min(student_positions.numel(), teacher_positions.numel())
        if matched_tokens == 0:
            pooled_grads.append(zero_grad)
            continue

        # KD alignment here is intentionally simple: compare only the shared prefix
        # of supervised positions after optional EOS dropping on each side.
        student_positions = student_positions[:matched_tokens]
        teacher_positions = teacher_positions[:matched_tokens]
        sample_kd_grad = distillation_logit_grad_fn(
            student_logits=student_logits[sample_index, student_positions],
            teacher_logits=teacher_logits[sample_index, teacher_positions].to(
                device=student_logits.device,
                dtype=student_logits.dtype,
            ),
            student_temperature=student_temperature,
            teacher_temperature=teacher_temperature,
            teacher_index=teacher_index,
            loss_function=loss_function,
        )
        # Keep the KD gradient on the same scale as the CE reference by normalizing
        # with the student supervised-token count, not just the matched prefix length.
        pooled_grads.append(sample_kd_grad.sum(dim=0) / supervised_student_count)

    if not pooled_grads:
        return student_logits.new_zeros((0, vocab_size), dtype=torch.float32)
    return torch.stack(pooled_grads, dim=0)


__all__ = [
    "compute_pooled_ce_grace_grad",
    "compute_pooled_kd_grace_grad",
]
