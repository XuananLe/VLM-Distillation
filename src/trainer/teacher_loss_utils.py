from collections.abc import Callable, Iterable

import torch

from src.trainer.gradient_utils import (
    compute_parameter_grads,
    parameter_gradient_cosine,
    trainable_parameters,
)
from src.trainer.kd_sequence_utils import compute_single_teacher_loss


def compute_teacher_loss_matrix(
    *,
    student_logits: torch.Tensor,
    student_labels: torch.Tensor,
    model,
    ce_loss: torch.Tensor,
    teacher_models: Iterable | None,
    teacher_batches,
    cached_teacher_batches=None,
    prepare_input_fn: Callable,
    collect_grace_tensors: bool,
    collect_teacher_target_batches: bool,
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
    """Compute per-teacher KD losses and optional GRACE signals for one student batch."""
    teacher_losses = []
    grace_scores = []
    grace_active = []
    teacher_logit_batches = [] if collect_teacher_target_batches else None
    teacher_label_batches = [] if collect_teacher_target_batches else None

    grace_parameters = None
    ce_parameter_grads = None
    if collect_grace_tensors:
        grace_parameters = trainable_parameters(model)
        ce_parameter_grads = compute_parameter_grads(
            loss=ce_loss,
            parameters=grace_parameters,
        )

    def append_parameter_grace_score(teacher_loss: torch.Tensor) -> None:
        """Compare full student-parameter gradients for CE and this teacher's KD loss."""
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
        grace_active.append(per_sample_agreement > grace_threshold)

    if cached_teacher_batches is not None:
        # Cached and live-teacher paths intentionally return the same tensor contract
        # so routing, GRACE, and reinforced selection can ignore the teacher source.
        for teacher_index, (cached_teacher_logits, cached_teacher_labels) in enumerate(cached_teacher_batches):
            prepared_teacher_logits = prepare_input_fn(cached_teacher_logits).to(
                dtype=student_logits.dtype
            )
            prepared_teacher_labels = prepare_input_fn(cached_teacher_labels)
            if collect_teacher_target_batches:
                teacher_logit_batches.append(prepared_teacher_logits)
                teacher_label_batches.append(prepared_teacher_labels)
            distillation_prepare_batch_fn(
                student_logits=student_logits,
                teacher_logits=prepared_teacher_logits,
                teacher_labels=prepared_teacher_labels,
                teacher_index=teacher_index,
            )
            teacher_loss = compute_single_teacher_loss(
                student_logits=student_logits,
                student_labels=student_labels,
                teacher_logits=prepared_teacher_logits,
                teacher_labels=prepared_teacher_labels,
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
                teacher_logit_batches,
                teacher_label_batches,
            )
        return (
            teacher_loss_matrix,
            torch.stack(grace_scores, dim=-1),
            torch.stack(grace_active, dim=-1),
            teacher_logit_batches,
            teacher_label_batches,
        )

    for teacher_index, (teacher_model, (teacher_inputs, teacher_labels)) in enumerate(zip(teacher_models, teacher_batches)):
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
        teacher_logits = teacher_outputs.logits.detach()
        del teacher_outputs
        if collect_teacher_target_batches:
            teacher_logit_batches.append(teacher_logits)
            teacher_label_batches.append(prepared_teacher_labels)
        distillation_prepare_batch_fn(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            teacher_labels=prepared_teacher_labels,
            teacher_index=teacher_index,
        )
        teacher_loss = compute_single_teacher_loss(
            student_logits=student_logits,
            student_labels=student_labels,
            teacher_logits=teacher_logits,
            teacher_labels=prepared_teacher_labels,
            distillation_loss_fn=distillation_loss_fn,
            student_temperature=student_temperature,
            teacher_temperature=teacher_temperature,
            skip_student_eos=skip_student_eos,
            skip_teacher_eos=skip_teacher_eos,
            teacher_index=teacher_index,
        )
        teacher_losses.append(teacher_loss)
        append_parameter_grace_score(teacher_loss)
        del teacher_logits

    teacher_loss_matrix = torch.stack(teacher_losses, dim=-1)
    if not grace_scores:
        return (
            teacher_loss_matrix,
            None,
            None,
            teacher_logit_batches,
            teacher_label_batches,
        )
    return (
        teacher_loss_matrix,
        torch.stack(grace_scores, dim=-1),
        torch.stack(grace_active, dim=-1),
        teacher_logit_batches,
        teacher_label_batches,
    )
