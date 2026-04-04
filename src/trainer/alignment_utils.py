from collections.abc import Callable, Iterable

import torch
import torch.nn.functional as F

from src.trainer.distillation_utils import (
    compute_teacher_forward,
    select_labels_at_positions,
    select_supervised_logit_positions,
)


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
    student_logits_masked = student_logits[student_labels != -100]
    teacher_logits_masked = teacher_logits[teacher_labels != -100]
    ce_grad_masked = ce_grad[student_labels != -100] if ce_grad is not None else None

    if skip_student_eos and student_logits_masked.size(0) > 0:
        student_logits_masked = student_logits_masked[:-1]
        if ce_grad_masked is not None:
            ce_grad_masked = ce_grad_masked[:-1]
    if skip_teacher_eos and teacher_logits_masked.size(0) > 0:
        teacher_logits_masked = teacher_logits_masked[:-1]

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
    temperature: float,
    student_temperature: float,
    teacher_temperature: float,
    skip_student_eos: bool,
    skip_teacher_eos: bool,
) -> torch.Tensor:
    sample_losses = []
    for sample_index in range(student_logits.size(0)):
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
                    temperature=temperature,
                    student_temperature=student_temperature,
                    teacher_temperature=teacher_temperature,
                )
            )
        else:
            sample_losses.append(student_logits.new_zeros(()))
    if not sample_losses:
        return student_logits.new_zeros((student_logits.size(0),))
    return torch.stack(sample_losses)


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


def compute_teacher_loss_matrix(
    *,
    student_logits: torch.Tensor,
    student_labels: torch.Tensor,
    teacher_models: Iterable,
    teacher_batches,
    prepare_input_fn: Callable,
    collect_alignment_tensors: bool,
    gradient_alignment_threshold: float,
    distillation_loss_fn: Callable,
    distillation_logit_grad_fn: Callable,
    loss_function: str,
    temperature: float,
    student_temperature: float,
    teacher_temperature: float,
    skip_student_eos: bool,
    skip_teacher_eos: bool,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    teacher_losses = []
    alignment_scores = []
    alignment_active = []

    pooled_ce_grad = None
    if collect_alignment_tensors:
        with torch.no_grad():
            pooled_ce_grad = compute_pooled_ce_alignment_grad(
                student_logits=student_logits.detach(),
                student_labels=student_labels,
            )

    for teacher_model, (teacher_inputs, teacher_labels) in zip(teacher_models, teacher_batches):
        prepared_teacher_labels = prepare_input_fn(teacher_labels)
        teacher_logit_positions = select_supervised_logit_positions(prepared_teacher_labels)
        if teacher_logit_positions is None and not prepared_teacher_labels.ne(-100).any():
            zero_losses = student_logits.new_zeros((student_logits.size(0),))
            teacher_losses.append(zero_losses)
            if pooled_ce_grad is not None:
                zero_alignment = pooled_ce_grad.new_zeros((student_logits.size(0),))
                alignment_scores.append(zero_alignment)
                alignment_active.append(zero_alignment > gradient_alignment_threshold)
            continue

        teacher_outputs = compute_teacher_forward(
            teacher_model,
            prepare_input_fn(teacher_inputs),
            output_hidden_states=False,
            suppress_stdout=getattr(teacher_model, "_suppress_forward_stdout", False),
            logits_to_keep=teacher_logit_positions,
        )
        teacher_logits = teacher_outputs.logits.detach()
        del teacher_outputs

        selected_teacher_labels = select_labels_at_positions(
            prepared_teacher_labels,
            teacher_logit_positions,
        )
        effective_teacher_labels = (
            selected_teacher_labels
            if selected_teacher_labels.size(1) == teacher_logits.size(1)
            else prepared_teacher_labels
        )
        teacher_losses.append(
            compute_single_teacher_loss(
                student_logits=student_logits,
                student_labels=student_labels,
                teacher_logits=teacher_logits,
                teacher_labels=effective_teacher_labels,
                distillation_loss_fn=distillation_loss_fn,
                temperature=temperature,
                student_temperature=student_temperature,
                teacher_temperature=teacher_temperature,
                skip_student_eos=skip_student_eos,
                skip_teacher_eos=skip_teacher_eos,
            )
        )
        if pooled_ce_grad is not None:
            with torch.no_grad():
                pooled_kd_grad = compute_pooled_kd_alignment_grad(
                    student_logits=student_logits.detach(),
                    student_labels=student_labels,
                    teacher_logits=teacher_logits,
                    teacher_labels=effective_teacher_labels,
                    distillation_logit_grad_fn=distillation_logit_grad_fn,
                    loss_function=loss_function,
                    temperature=temperature,
                    student_temperature=student_temperature,
                    teacher_temperature=teacher_temperature,
                    skip_student_eos=skip_student_eos,
                    skip_teacher_eos=skip_teacher_eos,
                )
                agreement = F.cosine_similarity(pooled_ce_grad, pooled_kd_grad, dim=-1, eps=1e-8)
            alignment_scores.append(agreement)
            alignment_active.append(agreement > gradient_alignment_threshold)
            del pooled_kd_grad
        del teacher_logits

    teacher_loss_matrix = torch.stack(teacher_losses, dim=-1)
    if not alignment_scores:
        return teacher_loss_matrix, None, None
    return (
        teacher_loss_matrix,
        torch.stack(alignment_scores, dim=-1),
        torch.stack(alignment_active, dim=-1),
    )


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
