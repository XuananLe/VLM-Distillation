from collections.abc import Callable

import torch
import torch.nn.functional as F

from src.trainer.kd_sequence_utils import get_supervised_positions


def compute_pooled_ce_alignment_grad(
    *,
    student_logits: torch.Tensor,
    student_labels: torch.Tensor,
) -> torch.Tensor:
    pooled_grads = []
    vocab_size = student_logits.size(-1)
    zero_grad = student_logits.new_zeros((vocab_size,), dtype=torch.float32)

    for sample_index in range(student_logits.size(0)):
        positions = get_supervised_positions(student_labels[sample_index])
        if positions.numel() == 0:
            pooled_grads.append(zero_grad)
            continue

        sample_labels = student_labels[sample_index, positions]
        sample_grad = F.softmax(student_logits[sample_index, positions].float(), dim=-1)
        sample_grad[torch.arange(sample_labels.numel(), device=sample_labels.device), sample_labels] -= 1.0
        pooled_grads.append(sample_grad.sum(dim=0) / positions.numel())

    if not pooled_grads:
        return student_logits.new_zeros((0, vocab_size), dtype=torch.float32)
    return torch.stack(pooled_grads, dim=0)


def compute_pooled_kd_alignment_grad(
    *,
    student_logits: torch.Tensor,
    student_labels: torch.Tensor,
    teacher_logits: torch.Tensor,
    teacher_labels: torch.Tensor,
    distillation_logit_grad_fn: Callable,
    loss_function: str,
    temperature: float,
    student_temperature: float,
    teacher_temperature: float,
    skip_student_eos: bool,
    skip_teacher_eos: bool,
) -> torch.Tensor:
    pooled_grads = []
    vocab_size = student_logits.size(-1)
    zero_grad = student_logits.new_zeros((vocab_size,), dtype=torch.float32)

    for sample_index in range(student_logits.size(0)):
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

        student_positions = student_positions[:matched_tokens]
        teacher_positions = teacher_positions[:matched_tokens]
        sample_kd_grad = distillation_logit_grad_fn(
            loss_function,
            student_logits[sample_index, student_positions],
            teacher_logits[sample_index, teacher_positions].to(
                device=student_logits.device,
                dtype=student_logits.dtype,
            ),
            temperature=temperature,
            student_temperature=student_temperature,
            teacher_temperature=teacher_temperature,
        )
        pooled_grads.append(sample_kd_grad.sum(dim=0) / supervised_student_count)

    if not pooled_grads:
        return student_logits.new_zeros((0, vocab_size), dtype=torch.float32)
    return torch.stack(pooled_grads, dim=0)


def apply_gradient_alignment_routing(
    *,
    teacher_gate_weights: torch.Tensor | None,
    teacher_alignment_scores: torch.Tensor | None,
    teacher_alignment_active: torch.Tensor | None,
    prev_alignment_score_ema: torch.Tensor | None,
    gradient_alignment_ema_decay: float,
    gradient_alignment_softmax_beta: float,
    gradient_alignment_router_blend_lambda: float,
    gradient_alignment_epsilon: float,
) -> tuple[
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    if (
        teacher_gate_weights is None
        or teacher_alignment_scores is None
        or teacher_alignment_active is None
    ):
        return (
            teacher_gate_weights,
            teacher_alignment_scores,
            teacher_alignment_active,
            None,
            None,
            prev_alignment_score_ema,
        )

    available_mask = teacher_gate_weights > 0
    if prev_alignment_score_ema is None or prev_alignment_score_ema.numel() != teacher_alignment_scores.size(-1):
        smoothed_alignment_scores = teacher_alignment_scores
    else:
        prev_alignment_score_ema = prev_alignment_score_ema.to(
            device=teacher_alignment_scores.device,
            dtype=teacher_alignment_scores.dtype,
        )
        smoothed_alignment_scores = (
            gradient_alignment_ema_decay * prev_alignment_score_ema.unsqueeze(0)
            + (1.0 - gradient_alignment_ema_decay) * teacher_alignment_scores
        )

    batch_alignment_mean = teacher_alignment_scores.detach().mean(dim=0).float()
    if prev_alignment_score_ema is None or prev_alignment_score_ema.numel() != batch_alignment_mean.numel():
        next_alignment_score_ema = batch_alignment_mean
    else:
        next_alignment_score_ema = (
            gradient_alignment_ema_decay * prev_alignment_score_ema.float()
            + (1.0 - gradient_alignment_ema_decay) * batch_alignment_mean
        )

    masked_smoothed_scores = smoothed_alignment_scores.masked_fill(~available_mask, float("-inf"))
    gradient_weights = torch.softmax(
        gradient_alignment_softmax_beta * masked_smoothed_scores,
        dim=-1,
    )
    gradient_weights = gradient_weights * available_mask.to(dtype=teacher_gate_weights.dtype)
    gradient_weights = gradient_weights / gradient_weights.sum(dim=-1, keepdim=True).clamp(
        min=torch.finfo(gradient_weights.dtype).eps
    )

    router_weights = teacher_gate_weights.clamp(min=torch.finfo(teacher_gate_weights.dtype).eps)
    blended_weights = (
        router_weights.pow(gradient_alignment_router_blend_lambda)
        * gradient_weights.clamp(min=torch.finfo(gradient_weights.dtype).eps).pow(
            1.0 - gradient_alignment_router_blend_lambda
        )
    )
    blended_weights = blended_weights * available_mask.to(dtype=teacher_gate_weights.dtype)
    blended_weights = blended_weights / blended_weights.sum(dim=-1, keepdim=True).clamp(
        min=torch.finfo(blended_weights.dtype).eps
    )

    available_counts = available_mask.sum(dim=-1, keepdim=True).clamp(min=1)
    uniform_weights = available_mask.to(dtype=teacher_gate_weights.dtype) / available_counts
    score_spread = (
        masked_smoothed_scores.max(dim=-1).values
        - smoothed_alignment_scores.masked_fill(~available_mask, float("inf")).min(dim=-1).values
    )
    use_uniform = score_spread < gradient_alignment_epsilon
    effective_weights = torch.where(use_uniform.unsqueeze(-1), uniform_weights, blended_weights)
    fallback_rate = use_uniform.to(dtype=teacher_gate_weights.dtype).mean()
    return (
        effective_weights,
        teacher_alignment_scores,
        teacher_alignment_active,
        gradient_weights,
        fallback_rate,
        next_alignment_score_ema,
    )


__all__ = [
    "apply_gradient_alignment_routing",
    "compute_pooled_ce_alignment_grad",
    "compute_pooled_kd_alignment_grad",
]
