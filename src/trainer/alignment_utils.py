from collections.abc import Callable, Iterable

import torch
import torch.nn.functional as F

from src.trainer.gradient_utils import (
    compute_pooled_ce_alignment_grad,
    compute_pooled_kd_alignment_grad,
)
from src.trainer.kd_sequence_utils import compute_single_teacher_loss
from src.trainer.distillation_utils import (
    compute_teacher_forward,
    select_labels_at_positions,
    select_supervised_logit_positions,
)


def compute_teacher_loss_matrix(
    *,
    student_logits: torch.Tensor,
    student_labels: torch.Tensor,
    teacher_models: Iterable | None,
    teacher_batches,
    cached_teacher_batches=None,
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

    if cached_teacher_batches is not None:
        for cached_teacher_logits, cached_teacher_labels in cached_teacher_batches:
            prepared_teacher_logits = prepare_input_fn(cached_teacher_logits).to(
                dtype=student_logits.dtype
            )
            prepared_teacher_labels = prepare_input_fn(cached_teacher_labels)
            teacher_losses.append(
                compute_single_teacher_loss(
                    student_logits=student_logits,
                    student_labels=student_labels,
                    teacher_logits=prepared_teacher_logits,
                    teacher_labels=prepared_teacher_labels,
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
                        teacher_logits=prepared_teacher_logits,
                        teacher_labels=prepared_teacher_labels,
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

        teacher_loss_matrix = torch.stack(teacher_losses, dim=-1)
        if not alignment_scores:
            return teacher_loss_matrix, None, None
        return (
            teacher_loss_matrix,
            torch.stack(alignment_scores, dim=-1),
            torch.stack(alignment_active, dim=-1),
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
