from dataclasses import dataclass

import torch
from einops import einsum

from src.components.grace import apply_grace_routing
from src.components.reinforced_teacher_selection import compute_reinforced_selection_state
from src.trainer.teacher_loss_utils import (
    compute_teacher_loss_matrix,
    resolve_teacher_target_batches,
)
from src.trainer.routing_utils import (
    apply_teacher_gate_topk,
    compute_teacher_gate_entropy_loss,
    compute_teacher_gate_z_loss,
)
from src.trainer.distillation_utils import (
    build_cached_teacher_target_batches,
    build_live_teacher_batches,
)


@dataclass(slots=True)
class TeacherBatchSources:
    live_teacher_batches: list | None
    cached_teacher_target_batches: list | None


def prepare_teacher_batch_sources(*, inputs, num_teachers: int, teacher_models):
    cached_teacher_target_batches = build_cached_teacher_target_batches(inputs, num_teachers)
    if cached_teacher_target_batches is not None and len(cached_teacher_target_batches) != num_teachers:
        raise ValueError(
            "Cached teacher-logit batch count does not match the configured teacher count. "
            f"cached={len(cached_teacher_target_batches)}, configured={num_teachers}"
        )
    live_teacher_batches = None
    if teacher_models:
        live_teacher_batches = build_live_teacher_batches(
            inputs,
            num_teachers,
        )
    elif cached_teacher_target_batches is None:
        raise ValueError(
            "No teacher inputs were found in the batch. Provide teacher models or cached teacher logits."
        )
    return TeacherBatchSources(
        live_teacher_batches=live_teacher_batches,
        cached_teacher_target_batches=cached_teacher_target_batches,
    )


def build_student_forward_state(*, trainer, model, student_inputs):
    """
    {
      "student_outputs": CausalLMOutputWithPast(
          loss=tensor(0.72),
          logits=tensor shape [2, 6, 49280],
          hidden_states=...
      ),

      "student_logits": tensor shape [2, 6, 49280],

      "student_layer_representations": {
          3: tensor shape [2, 960],
          7: tensor shape [2, 960],
      },

      "teacher_router_logits": tensor shape [2, 4],

      "teacher_router_weights": tensor shape [2, 4],
    }
    """
    if trainer.teacher_gate is not None:
        trainer.teacher_gate.reset()
    if trainer.reinforced_teacher_selector is not None:
        trainer.reinforced_teacher_selector.reset()

    with trainer.layer_distiller.capture_student(model) as student_layer_outputs:
        # https://huggingface.co/docs/transformers/model_doc/smolvlm
        # https://github.com/huggingface/transformers/blob/v5.1.0/src/transformers/models/smolvlm/modeling_smolvlm.py#L572
        student_outputs = model(
            **student_inputs,
            return_dict=True,
            **trainer.layer_distiller.student_forward_kwargs(),
        )
    student_logits = student_outputs.logits
    student_layer_representations = trainer.layer_distiller.student_representations(
        student_inputs=student_inputs,
        student_outputs=student_outputs,
        captured_outputs=student_layer_outputs,
    )

    teacher_router_logits = (
        trainer.teacher_gate.compute_router_logits(
            student_labels=student_inputs["labels"],
            student_attention_mask=student_inputs.get("attention_mask"),
        )
        if trainer.teacher_gate is not None
        else None
    )
    teacher_router_weights = (
        torch.softmax(teacher_router_logits, dim=-1)
        if teacher_router_logits is not None
        else None
    )

    return {
        "student_outputs": student_outputs,
        "student_logits": student_logits,
        "student_layer_representations": student_layer_representations,
        "teacher_router_logits": teacher_router_logits,
        "teacher_router_weights": teacher_router_weights,
    }


def build_layer_distillation_state(
    *,
    trainer,
    live_teacher_batches,
    student_layer_representations,
):
    """Compute the auxiliary live-teacher layer-distillation loss when that path is enabled."""
    layer_distillation_loss = trainer.layer_distiller.compute_loss(
        teacher_models=trainer.teacher_models,
        live_teacher_batches=live_teacher_batches,
        student_layer_representations=student_layer_representations,
    )
    return {
        "layer_distillation_loss": layer_distillation_loss,
    }


def resolve_teacher_gate_state(*, trainer, teacher_router_logits, teacher_router_weights):
    teacher_gate_state = {
        "teacher_gate_entropy_loss": None,
        "teacher_gate_z_loss": None,
        "teacher_gate_assignment_rate": None,
        "routed_teacher_weights": teacher_router_weights,
    }

    if teacher_router_weights is not None:
        teacher_gate_state["teacher_gate_z_loss"] = compute_teacher_gate_z_loss(teacher_router_logits)
        teacher_gate_state["teacher_gate_entropy_loss"] = compute_teacher_gate_entropy_loss(teacher_router_weights)
        (
            teacher_gate_state["routed_teacher_weights"],
            teacher_gate_state["teacher_gate_assignment_rate"],
        ) = apply_teacher_gate_topk(
            teacher_router_logits,
            teacher_router_weights,
            trainer.teacher_gate_top_k,
        )
    return teacher_gate_state


def build_teacher_loss_state(
    *,
    trainer,
    model,
    student_logits,
    student_labels,
    ce_loss,
    teacher_target_batches,
):
    """Compute per-teacher KD losses and optional GRACE tensors for the current batch."""
    (
        teacher_loss_matrix,
        teacher_grace_scores,
        teacher_grace_active_mask,
        selection_teacher_logits,
        selection_teacher_labels,
    ) = compute_teacher_loss_matrix(
        student_logits=student_logits,
        student_labels=student_labels,
        model=model,
        ce_loss=ce_loss,
        teacher_target_batches=teacher_target_batches,
        collect_grace_tensors=trainer.should_apply_grace_routing(),
        collect_teacher_targets_for_selection=trainer.teacher_weighting_strategy == "reinforced_selection",
        grace_threshold=trainer.grace_threshold,
        distillation_prepare_batch_fn=trainer.distillation_prepare_batch_fn,
        distillation_loss_fn=trainer.distillation_loss_fn,
        student_temperature=trainer.student_temperature,
        teacher_temperature=trainer.teacher_temperature,
        skip_student_eos=trainer.skip_student_eos,
        skip_teacher_eos=trainer.skip_teacher_eos,
    )
    return {
        "teacher_loss_matrix": teacher_loss_matrix,
        "teacher_grace_scores": teacher_grace_scores,
        "teacher_grace_active_mask": teacher_grace_active_mask,
        "selection_teacher_logits": selection_teacher_logits,
        "selection_teacher_labels": selection_teacher_labels,
    }


def build_teacher_target_batches(
    *,
    trainer,
    student_logits,
    teacher_batch_sources: TeacherBatchSources,
):
    return resolve_teacher_target_batches(
        student_logits=student_logits,
        teacher_models=trainer.teacher_models,
        live_teacher_batches=teacher_batch_sources.live_teacher_batches,
        cached_teacher_target_batches=teacher_batch_sources.cached_teacher_target_batches,
        prepare_input_fn=trainer._prepare_input,
    )


def resolve_teacher_weighting_state(
    *,
    trainer,
    routed_teacher_weights,
    teacher_grace_scores,
    teacher_grace_active_mask,
    teacher_loss_matrix,
    selection_teacher_logits,
    selection_teacher_labels,
    student_labels,
    student_attention_mask,
    student_ce_loss,
):
    teacher_grace_weights = None
    teacher_grace_fallback_rate = None
    teacher_mix_weights = None
    reinforced_selection_metrics = None
    teacher_selection_policy_loss = None

    if trainer.teacher_weighting_strategy == "reinforced_selection":
        # Reinforced selection bypasses gate/GRACE blending and turns the per-teacher
        # KD losses into a Bernoulli policy problem over teacher subsets.
        reinforced_state = compute_reinforced_selection_state(
            selector=trainer.reinforced_teacher_selector,
            teacher_loss_matrix=teacher_loss_matrix,
            selection_teacher_logits=selection_teacher_logits or [],
            selection_teacher_labels=selection_teacher_labels or [],
            student_labels=student_labels,
            student_attention_mask=student_attention_mask,
            student_ce_loss=student_ce_loss,
            teacher_temperature=trainer.teacher_temperature,
            skip_teacher_eos=trainer.skip_teacher_eos,
            warmup_active=trainer.reinforced_selection_warmup_active(),
            reward_type=trainer.reinforced_selection_reward_type,
            prev_reward_baseline=trainer.reinforced_selection_reward_baseline,
            reward_ema_decay=trainer.reinforced_selection_reward_ema_decay,
        )
        trainer.reinforced_selection_reward_baseline = reinforced_state["next_reward_baseline"]
        teacher_mix_weights = reinforced_state["selection_weights"]
        teacher_selection_policy_loss = reinforced_state["policy_loss"]
        reinforced_selection_metrics = reinforced_state["metrics"]
        return {
            "teacher_mix_weights": teacher_mix_weights,
            "teacher_grace_scores": teacher_grace_scores,
            "teacher_grace_active_mask": teacher_grace_active_mask,
            "teacher_grace_weights": None,
            "teacher_grace_fallback_rate": None,
            "distillation_loss": reinforced_state["distillation_loss"],
            "teacher_selection_policy_loss": teacher_selection_policy_loss,
            "reinforced_selection_metrics": reinforced_selection_metrics,
        }

    if not trainer.should_apply_grace_routing():
        if routed_teacher_weights is not None:
            routed_teacher_mask = routed_teacher_weights.gt(0)
            missing_rows = ~routed_teacher_mask.any(dim=-1)
            if missing_rows.any():
                bad_indices = missing_rows.nonzero(as_tuple=True)[0].tolist()
                raise ValueError(
                    "Router produced no available teacher assignments for samples "
                    f"{bad_indices}."
                )
            teacher_mix_weights = routed_teacher_weights.to(
                dtype=teacher_loss_matrix.dtype
            )
            teacher_mix_weights = teacher_mix_weights / (
                teacher_mix_weights.sum(dim=-1, keepdim=True).clamp(
                    min=torch.finfo(teacher_mix_weights.dtype).eps
                )
            )
    else:
        # Full GRACE blends router availability with gradient agreement scores.
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
        "teacher_selection_policy_loss": teacher_selection_policy_loss,
        "reinforced_selection_metrics": reinforced_selection_metrics,
    }


def compute_total_loss(
    *,
    trainer,
    ce_loss,
    distillation_loss,
    layer_distillation_loss,
    teacher_gate_entropy_loss,
    teacher_gate_z_loss,
    teacher_selection_policy_loss=None,
):
    base_kd_loss = trainer.alpha * distillation_loss
    loss = ce_loss + base_kd_loss
    if layer_distillation_loss is not None:
        loss = loss + layer_distillation_loss * trainer.layer_distill_weight
    if teacher_gate_entropy_loss is not None:
        loss = loss + teacher_gate_entropy_loss * trainer.teacher_gate_entropy_alpha
    if teacher_gate_z_loss is not None:
        loss = loss + teacher_gate_z_loss * trainer.teacher_gate_router_z_loss_alpha
    if teacher_selection_policy_loss is not None:
        loss = loss + teacher_selection_policy_loss * trainer.reinforced_selection_policy_alpha
    return loss


__all__ = [
    "build_layer_distillation_state",
    "build_student_forward_state",
    "build_teacher_loss_state",
    "build_teacher_target_batches",
    "compute_total_loss",
    "prepare_teacher_batch_sources",
    "resolve_teacher_gate_state",
    "resolve_teacher_weighting_state",
    "TeacherBatchSources",
]
