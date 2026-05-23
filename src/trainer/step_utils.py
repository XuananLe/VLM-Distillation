from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypedDict

import torch
from einops import einsum

from src.components.grace import apply_grace_routing
from src.trainer.distillation_utils import (
    build_cached_teacher_target_batches,
)
from src.trainer.routing_utils import (
    apply_teacher_gate_topk,
    compute_teacher_gate_entropy_loss,
    compute_teacher_gate_z_loss,
)
from src.trainer.teacher_loss_utils import (
    TeacherTargetBatch,
    compute_teacher_loss_matrix,
)


@dataclass(slots=True)
class TeacherBatchSources:
    cached_teacher_target_batches: list[tuple[torch.Tensor, torch.Tensor]]


class StudentForwardState(TypedDict):
    student_outputs: Any
    student_logits: torch.Tensor
    teacher_router_logits: torch.Tensor | None
    teacher_router_weights: torch.Tensor | None


class TeacherGateState(TypedDict):
    teacher_gate_entropy_loss: torch.Tensor | None
    teacher_gate_z_loss: torch.Tensor | None
    routed_teacher_weights: torch.Tensor | None


class TeacherLossState(TypedDict):
    teacher_loss_matrix: torch.Tensor
    teacher_grace_scores: torch.Tensor | None
    teacher_grace_active_mask: torch.Tensor | None


class TeacherWeightingState(TypedDict):
    teacher_mix_weights: torch.Tensor | None
    teacher_grace_scores: torch.Tensor | None
    teacher_grace_active_mask: torch.Tensor | None
    teacher_grace_weights: torch.Tensor | None
    teacher_grace_fallback_rate: torch.Tensor | None
    distillation_loss: torch.Tensor


def prepare_teacher_batch_sources(*, inputs: Mapping[str, torch.Tensor], num_teachers: int) -> TeacherBatchSources:
    cached_teacher_target_batches = build_cached_teacher_target_batches(inputs, num_teachers)
    if cached_teacher_target_batches is not None and len(cached_teacher_target_batches) != num_teachers:
        raise ValueError(
            "Cached teacher-logit batch count does not match the configured teacher count. "
            f"cached={len(cached_teacher_target_batches)}, configured={num_teachers}"
        )
    if cached_teacher_target_batches is None:
        raise ValueError("No cached teacher logits were found in the batch.")
    return TeacherBatchSources(
        cached_teacher_target_batches=cached_teacher_target_batches,
    )


def build_student_forward_state(
    *,
    trainer: Any,
    model: torch.nn.Module,
    student_inputs: Mapping[str, torch.Tensor],
) -> StudentForwardState:
    """
    {
      "student_outputs": CausalLMOutputWithPast(
          loss=tensor(0.72),
          logits=tensor shape [2, 6, 49280],
      ),

      "student_logits": tensor shape [2, 6, 49280],

      "teacher_router_logits": tensor shape [2, 4],

      "teacher_router_weights": tensor shape [2, 4],
    }
    """
    # https://huggingface.co/docs/transformers/model_doc/smolvlm
    # https://github.com/huggingface/transformers/blob/v5.1.0/src/transformers/models/smolvlm/modeling_smolvlm.py#L572
    student_outputs = model(
        **student_inputs,
        return_dict=True,
    )
    student_logits = student_outputs.logits

    teacher_router_logits = (
        trainer.teacher_gate.compute_router_logits(
            student_labels=student_inputs["labels"],
        )
        if trainer.teacher_gate is not None
        else None
    )
    teacher_router_weights = torch.softmax(teacher_router_logits, dim=-1) if teacher_router_logits is not None else None

    return {
        "student_outputs": student_outputs,
        "student_logits": student_logits,
        "teacher_router_logits": teacher_router_logits,
        "teacher_router_weights": teacher_router_weights,
    }


def resolve_teacher_gate_state(
    *,
    trainer: Any,
    teacher_router_logits: torch.Tensor | None,
    teacher_router_weights: torch.Tensor | None,
) -> TeacherGateState:
    teacher_gate_state: TeacherGateState = {
        "teacher_gate_entropy_loss": None,
        "teacher_gate_z_loss": None,
        "routed_teacher_weights": teacher_router_weights,
    }

    if teacher_router_weights is not None:
        teacher_gate_state["teacher_gate_z_loss"] = compute_teacher_gate_z_loss(teacher_router_logits)
        teacher_gate_state["teacher_gate_entropy_loss"] = compute_teacher_gate_entropy_loss(teacher_router_weights)
        teacher_gate_state["routed_teacher_weights"] = apply_teacher_gate_topk(
            teacher_router_logits,
            teacher_router_weights,
            trainer.teacher_gate_top_k,
        )
    return teacher_gate_state


def build_teacher_loss_state(
    *,
    trainer: Any,
    model: torch.nn.Module,
    student_logits: torch.Tensor,
    student_labels: torch.Tensor,
    teacher_target_batches: Sequence[TeacherTargetBatch],
) -> TeacherLossState:
    (
        teacher_loss_matrix,
        teacher_grace_scores,
        teacher_grace_active_mask,
    ) = compute_teacher_loss_matrix(
        student_logits=student_logits,
        student_labels=student_labels,
        model=model,
        teacher_target_batches=teacher_target_batches,
        collect_grace_tensors=trainer.should_apply_grace_routing(),
        grace_threshold=trainer.grace_threshold,
        distillation_prepare_batch_fn=trainer.distillation_prepare_batch_fn,
        distillation_loss_fn=trainer.distillation_loss_fn,
        student_temperature=trainer.student_temperature,
        teacher_temperature=trainer.teacher_temperature,
    )
    return {
        "teacher_loss_matrix": teacher_loss_matrix,
        "teacher_grace_scores": teacher_grace_scores,
        "teacher_grace_active_mask": teacher_grace_active_mask,
    }


def resolve_teacher_weighting_state(
    *,
    trainer: Any,
    routed_teacher_weights: torch.Tensor | None,
    teacher_grace_scores: torch.Tensor | None,
    teacher_grace_active_mask: torch.Tensor | None,
    teacher_loss_matrix: torch.Tensor,
) -> TeacherWeightingState:
    teacher_grace_weights: torch.Tensor | None = None
    teacher_grace_fallback_rate: torch.Tensor | None = None
    teacher_mix_weights: torch.Tensor | None = None

    if not trainer.should_apply_grace_routing():
        if routed_teacher_weights is not None:
            routed_teacher_mask = routed_teacher_weights.gt(0)
            missing_rows = ~routed_teacher_mask.any(dim=-1)
            if missing_rows.any():
                bad_indices = missing_rows.nonzero(as_tuple=True)[0].tolist()
                raise ValueError(f"Router produced no available teacher assignments for samples {bad_indices}.")
            teacher_mix_weights = routed_teacher_weights.to(dtype=teacher_loss_matrix.dtype)
            teacher_mix_weights = teacher_mix_weights / (
                teacher_mix_weights.sum(dim=-1, keepdim=True).clamp(min=torch.finfo(teacher_mix_weights.dtype).eps)
            )
    else:
        (
            teacher_mix_weights,
            teacher_grace_scores,
            teacher_grace_active_mask,
            teacher_grace_weights,
            teacher_grace_fallback_rate,
            trainer.teacher_grace_score_ema,
        ) = apply_grace_routing(
            routed_teacher_weights=routed_teacher_weights,
            teacher_grace_scores=teacher_grace_scores,
            teacher_grace_active_mask=teacher_grace_active_mask,
            prev_grace_score_ema=trainer.teacher_grace_score_ema,
            grace_ema_decay=trainer.grace_ema_decay,
            grace_softmax_beta=trainer.grace_softmax_beta,
            grace_router_blend_lambda=trainer.grace_router_blend_lambda,
            grace_epsilon=trainer.grace_epsilon,
        )
    distillation_loss = teacher_loss_matrix.mean()
    if teacher_mix_weights is not None:
        distillation_loss = einsum(
            teacher_loss_matrix,
            teacher_mix_weights,
            "batch teacher, batch teacher -> batch",
        ).mean()
    elif routed_teacher_weights is not None:
        distillation_loss = einsum(
            teacher_loss_matrix,
            routed_teacher_weights,
            "batch teacher, batch teacher -> batch",
        ).mean()

    return {
        "teacher_mix_weights": teacher_mix_weights,
        "teacher_grace_scores": teacher_grace_scores,
        "teacher_grace_active_mask": teacher_grace_active_mask,
        "teacher_grace_weights": teacher_grace_weights,
        "teacher_grace_fallback_rate": teacher_grace_fallback_rate,
        "distillation_loss": distillation_loss,
    }


def compute_total_loss(
    *,
    trainer: Any,
    ce_loss: torch.Tensor,
    distillation_loss: torch.Tensor,
    teacher_gate_entropy_loss: torch.Tensor | None,
    teacher_gate_z_loss: torch.Tensor | None,
) -> torch.Tensor:
    base_kd_loss = trainer.alpha * distillation_loss
    loss = ce_loss + base_kd_loss
    if teacher_gate_entropy_loss is not None:
        loss = loss + teacher_gate_entropy_loss * trainer.teacher_gate_entropy_alpha
    if teacher_gate_z_loss is not None:
        loss = loss + teacher_gate_z_loss * trainer.teacher_gate_router_z_loss_alpha
    return loss


__all__ = [
    "build_student_forward_state",
    "build_teacher_loss_state",
    "compute_total_loss",
    "prepare_teacher_batch_sources",
    "resolve_teacher_gate_state",
    "resolve_teacher_weighting_state",
    "TeacherBatchSources",
]
