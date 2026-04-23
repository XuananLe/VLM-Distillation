import time
from contextlib import nullcontext

import torch
from einops import einsum

from src.components.grace import apply_grace_routing
from src.components.reinforced_teacher_selection import compute_reinforced_selection_state
from src.components.forward_utils import forward_with_kwarg_retry
from src.trainer.alignment_utils import compute_teacher_loss_matrix
from src.trainer.routing_utils import (
    apply_teacher_gate_constraints,
    compute_teacher_gate_balance_loss,
    compute_teacher_gate_entropy_loss,
    compute_teacher_gate_z_loss,
)
from src.trainer.distillation_utils import (
    build_cached_teacher_batches,
    build_teacher_batches,
    capture_layer_outputs,
    compute_student_representations,
    compute_teacher_forward_and_layer_distillation,
)


def move_teacher_models_to_device(teacher_models, target_device) -> None:
    """Move live teacher models onto the student's current device before a training step."""
    for index, teacher_model in enumerate(teacher_models):
        teacher_device = next(teacher_model.parameters()).device
        if teacher_device != target_device:
            teacher_models[index] = teacher_model.to(target_device)


def prepare_teacher_batches(*, inputs, student_inputs, num_teachers: int, teacher_models):
    """Resolve live-teacher batches and cached-teacher batches from the current trainer inputs."""
    cached_teacher_batches = build_cached_teacher_batches(inputs, num_teachers)
    if cached_teacher_batches is not None and len(cached_teacher_batches) != num_teachers:
        raise ValueError(
            "Cached teacher-logit batch count does not match the configured teacher count. "
            f"cached={len(cached_teacher_batches)}, configured={num_teachers}"
        )
    teacher_batches = None
    if teacher_models:
        teacher_batches = build_teacher_batches(
            inputs,
            student_inputs,
            num_teachers,
            fallback_to_student_inputs=cached_teacher_batches is None,
        )
    elif cached_teacher_batches is None:
        raise ValueError(
            "No teacher inputs were found in the batch. Provide teacher models or cached teacher logits."
        )
    return teacher_batches, cached_teacher_batches


def run_student_forward_and_teacher_gate(*, trainer, model, student_inputs):
    """Run the student forward pass and, when enabled, compute raw teacher-gate routing tensors."""
    if trainer.teacher_gate is not None:
        trainer.teacher_gate.reset()
    if trainer.reinforced_teacher_selector is not None:
        trainer.reinforced_teacher_selector.reset()

    output_hidden_states = (
        trainer.layer_distillation_enabled and trainer.layer_distill_source == "model"
    )
    student_hook_context = (
        capture_layer_outputs(model, trainer.student_layer_indices)
        if trainer.layer_distillation_enabled and trainer.layer_distill_source == "vision"
        else nullcontext(None)
    )
    student_forward_start_time = time.perf_counter()
    with student_hook_context as student_layer_outputs:
        student_outputs = forward_with_kwarg_retry(
            model,
            {
                **student_inputs,
                "return_dict": True,
                "output_hidden_states": output_hidden_states,
            },
        )
    student_forward_time = time.perf_counter() - student_forward_start_time
    student_logits = student_outputs.logits
    student_layer_representations = None
    if trainer.layer_distillation_enabled:
        student_layer_representations = compute_student_representations(
            trainer.layer_distill_source,
            trainer.student_layer_indices,
            student_inputs,
            student_layer_outputs,
            student_outputs,
        )

    teacher_gate_start_time = time.perf_counter()
    teacher_gate_logits = (
        trainer.teacher_gate.compute_router_logits(
            labels=student_inputs["labels"],
            attention_mask=student_inputs.get("attention_mask"),
        )
        if trainer.teacher_gate is not None
        else None
    )
    teacher_gate_time = time.perf_counter() - teacher_gate_start_time
    teacher_gate_routing_scores = (
        trainer.teacher_gate.prepare_routing_scores(teacher_gate_logits)
        if teacher_gate_logits is not None
        else None
    )
    teacher_gate_weights = (
        torch.softmax(teacher_gate_routing_scores, dim=-1)
        if teacher_gate_logits is not None
        else None
    )

    return {
        "student_outputs": student_outputs,
        "student_logits": student_logits,
        "student_layer_representations": student_layer_representations,
        "student_forward_time": student_forward_time,
        "teacher_gate_time": teacher_gate_time,
        "teacher_gate_logits": teacher_gate_logits,
        "teacher_gate_weights": teacher_gate_weights,
        "teacher_gate_routing_scores": teacher_gate_routing_scores,
    }


def compute_layer_distillation_loss(
    *,
    trainer,
    teacher_batches,
    student_layer_representations,
):
    """Compute the auxiliary live-teacher layer-distillation loss when that path is enabled."""
    if not trainer.layer_distillation_enabled:
        return {
            "layer_distillation_loss": None,
            "layer_distillation_time": None,
        }
    if teacher_batches is None:
        raise ValueError(
            "Layer distillation requires live teacher inputs. "
            "Provide teacher processors alongside the cached teacher logits."
        )

    layer_distillation_start_time = time.perf_counter()
    layer_distillation_losses = []
    output_hidden_states = trainer.layer_distill_source == "model"
    for teacher_index, (teacher_model, (teacher_inputs, _teacher_labels)) in enumerate(
        zip(trainer.teacher_models, teacher_batches)
    ):
        _teacher_outputs, layer_loss = compute_teacher_forward_and_layer_distillation(
            teacher_model=teacher_model,
            teacher_inputs=teacher_inputs,
            teacher_layer_soft_matches=trainer.teacher_layer_soft_matches[teacher_index],
            layer_distill_source=trainer.layer_distill_source,
            student_layer_representations=student_layer_representations,
            output_hidden_states=output_hidden_states,
            suppress_stdout=getattr(teacher_model, "_suppress_forward_stdout", False),
        )
        if layer_loss is not None:
            layer_distillation_losses.append(layer_loss)

    if layer_distillation_losses:
        layer_distillation_loss = torch.stack(layer_distillation_losses).mean()
    else:
        reference_tensor = next(iter(student_layer_representations.values()))
        layer_distillation_loss = reference_tensor.new_zeros(())

    return {
        "layer_distillation_loss": layer_distillation_loss,
        "layer_distillation_time": time.perf_counter() - layer_distillation_start_time,
    }


def apply_teacher_gate_routing(*, trainer, model, teacher_gate_logits, teacher_gate_weights, teacher_gate_routing_scores):
    """Apply gate constraints and auxiliary routing losses to the current teacher-gate outputs."""
    state = {
        "teacher_gate_balance_loss": None,
        "teacher_gate_entropy_loss": None,
        "teacher_gate_z_loss": None,
        "teacher_gate_soft_load": None,
        "teacher_gate_hard_load": None,
        "teacher_gate_capacity": None,
        "teacher_gate_assignment_rate": None,
        "teacher_gate_expert_load": None,
        "teacher_gate_routing_fallback_rate": None,
        "teacher_gate_assignment_mask": None,
        "routed_teacher_gate_weights": teacher_gate_weights,
    }

    routing_constraint_start_time = time.perf_counter()
    if teacher_gate_weights is not None:
        state["teacher_gate_z_loss"] = compute_teacher_gate_z_loss(teacher_gate_logits)
        state["teacher_gate_entropy_loss"] = compute_teacher_gate_entropy_loss(teacher_gate_weights)
        (
            state["routed_teacher_gate_weights"],
            state["teacher_gate_capacity"],
            state["teacher_gate_assignment_rate"],
            state["teacher_gate_expert_load"],
            state["teacher_gate_routing_fallback_rate"],
            state["teacher_gate_assignment_mask"],
        ) = apply_teacher_gate_constraints(
            teacher_gate_routing_scores,
            teacher_gate_weights,
            trainer.current_teacher_gate_top_k(),
            trainer.teacher_gate_capacity_factor,
        )
        (
            state["teacher_gate_balance_loss"],
            state["teacher_gate_soft_load"],
            state["teacher_gate_hard_load"],
        ) = compute_teacher_gate_balance_loss(
            teacher_gate_weights,
            trainer.current_teacher_gate_top_k(),
            routing_scores=teacher_gate_routing_scores,
            assignment_mask=state["teacher_gate_assignment_mask"],
        )
        if model.training:
            trainer.teacher_gate.update_expert_bias(state["teacher_gate_expert_load"].detach())

    state["routing_constraint_time"] = time.perf_counter() - routing_constraint_start_time
    return state


def compute_teacher_losses_and_grace(
    *,
    trainer,
    student_logits,
    student_labels,
    teacher_batches,
    cached_teacher_batches,
):
    """Compute per-teacher KD losses and optional GRACE tensors for the current batch."""
    teacher_loss_matrix_start_time = time.perf_counter()
    (
        teacher_loss_matrix,
        teacher_grace_scores,
        teacher_grace_active,
        teacher_logit_batches,
        teacher_label_batches,
    ) = compute_teacher_loss_matrix(
        student_logits=student_logits,
        student_labels=student_labels,
        teacher_models=trainer.teacher_models,
        teacher_batches=teacher_batches,
        cached_teacher_batches=cached_teacher_batches,
        prepare_input_fn=trainer._prepare_input,
        collect_grace_tensors=trainer.should_apply_grace_routing(),
        collect_teacher_target_batches=trainer.teacher_weighting_strategy == "reinforced_selection",
        grace_threshold=trainer.grace_threshold,
        distillation_prepare_batch_fn=trainer.distillation_prepare_batch_fn,
        distillation_loss_fn=trainer.distillation_loss_fn,
        distillation_logit_grad_fn=trainer.distillation_logit_grad_fn,
        loss_function=trainer.loss_function,
        temperature=trainer.temperature,
        student_temperature=trainer.student_temperature,
        teacher_temperature=trainer.teacher_temperature,
        skip_student_eos=trainer.skip_student_eos,
        skip_teacher_eos=trainer.skip_teacher_eos,
    )
    return {
        "teacher_loss_matrix": teacher_loss_matrix,
        "teacher_grace_scores": teacher_grace_scores,
        "teacher_grace_active": teacher_grace_active,
        "teacher_logit_batches": teacher_logit_batches,
        "teacher_label_batches": teacher_label_batches,
        "teacher_loss_matrix_time": time.perf_counter() - teacher_loss_matrix_start_time,
    }


def apply_grace_and_compute_distillation_loss(
    *,
    trainer,
    routed_teacher_gate_weights,
    teacher_grace_scores,
    teacher_grace_active,
    teacher_loss_matrix,
    teacher_logits_batches,
    teacher_label_batches,
    labels,
    attention_mask,
    ce_loss,
):
    """Turn teacher losses plus routing state into the final KD loss and GRACE metrics."""
    grace_routing_start_time = time.perf_counter()
    teacher_grace_weights = None
    teacher_grace_fallback_rate = None
    effective_teacher_gate_weights = None
    reinforced_selection_metrics = None
    teacher_selection_policy_loss = None

    if trainer.teacher_weighting_strategy == "reinforced_selection":
        # Reinforced selection bypasses gate/GRACE blending and turns the per-teacher
        # KD losses into a Bernoulli policy problem over teacher subsets.
        reinforced_state = compute_reinforced_selection_state(
            selector=trainer.reinforced_teacher_selector,
            teacher_loss_matrix=teacher_loss_matrix,
            teacher_logits_batches=teacher_logits_batches or [],
            teacher_label_batches=teacher_label_batches or [],
            labels=labels,
            attention_mask=attention_mask,
            ce_loss=ce_loss,
            teacher_temperature=trainer.teacher_temperature,
            skip_teacher_eos=trainer.skip_teacher_eos,
            warmup_active=trainer.reinforced_selection_warmup_active(),
            reward_type=trainer.reinforced_selection_reward_type,
            prev_reward_baseline=trainer.reinforced_selection_reward_baseline,
            reward_ema_decay=trainer.reinforced_selection_reward_ema_decay,
        )
        trainer.reinforced_selection_reward_baseline = reinforced_state["next_reward_baseline"]
        effective_teacher_gate_weights = reinforced_state["selection_weights"]
        teacher_selection_policy_loss = reinforced_state["policy_loss"]
        reinforced_selection_metrics = reinforced_state["metrics"]
        return {
            "effective_teacher_gate_weights": effective_teacher_gate_weights,
            "teacher_grace_scores": teacher_grace_scores,
            "teacher_grace_active": teacher_grace_active,
            "teacher_grace_weights": None,
            "teacher_grace_fallback_rate": None,
            "distillation_loss": reinforced_state["distillation_loss"],
            "teacher_selection_policy_loss": teacher_selection_policy_loss,
            "reinforced_selection_metrics": reinforced_selection_metrics,
            "grace_routing_time": time.perf_counter() - grace_routing_start_time,
        }

    if not trainer.should_apply_grace_routing():
        # Router-only mode treats every routed teacher as equally available; GRACE is
        # the component that turns those routed slots into non-uniform final weights.
        if routed_teacher_gate_weights is not None:
            available_teacher_mask = routed_teacher_gate_weights.gt(0)
            if not available_teacher_mask.any():
                available_teacher_mask = torch.ones_like(
                    routed_teacher_gate_weights,
                    dtype=torch.bool,
                )
            effective_teacher_gate_weights = available_teacher_mask.to(
                dtype=teacher_loss_matrix.dtype,
            )
            effective_teacher_gate_weights = effective_teacher_gate_weights / (
                effective_teacher_gate_weights.sum(dim=-1, keepdim=True).clamp(
                    min=torch.finfo(effective_teacher_gate_weights.dtype).eps
                )
            )
    else:
        # Full GRACE blends router availability with gradient agreement scores.
        (
            effective_teacher_gate_weights,
            teacher_grace_scores,
            teacher_grace_active,
            teacher_grace_weights,
            teacher_grace_fallback_rate,
            trainer.teacher_grace_score_ema,
        ) = apply_grace_routing(
            teacher_gate_weights=routed_teacher_gate_weights,
            teacher_grace_scores=teacher_grace_scores,
            teacher_grace_active=teacher_grace_active,
            prev_grace_score_ema=trainer.teacher_grace_score_ema,
            grace_ema_decay=trainer.grace_ema_decay,
            grace_softmax_beta=trainer.grace_softmax_beta,
            grace_router_blend_lambda=trainer.grace_router_blend_lambda,
            grace_epsilon=trainer.grace_epsilon,
        )
    grace_routing_time = time.perf_counter() - grace_routing_start_time

    distillation_loss = teacher_loss_matrix.mean()
    if effective_teacher_gate_weights is not None:
        distillation_loss = einsum(
            teacher_loss_matrix,
            effective_teacher_gate_weights,
            "batch teacher, batch teacher -> batch",
        ).mean()
    elif routed_teacher_gate_weights is not None:
        distillation_loss = einsum(
            teacher_loss_matrix,
            routed_teacher_gate_weights,
            "batch teacher, batch teacher -> batch",
        ).mean()

    return {
        "effective_teacher_gate_weights": effective_teacher_gate_weights,
        "teacher_grace_scores": teacher_grace_scores,
        "teacher_grace_active": teacher_grace_active,
        "teacher_grace_weights": teacher_grace_weights,
        "teacher_grace_fallback_rate": teacher_grace_fallback_rate,
        "distillation_loss": distillation_loss,
        "teacher_selection_policy_loss": teacher_selection_policy_loss,
        "reinforced_selection_metrics": reinforced_selection_metrics,
        "grace_routing_time": grace_routing_time,
    }


def compute_total_loss(
    *,
    trainer,
    ce_loss,
    distillation_loss,
    layer_distillation_loss,
    teacher_gate_balance_loss,
    teacher_gate_entropy_loss,
    teacher_gate_z_loss,
    teacher_selection_policy_loss=None,
):
    """Combine CE, KD, routing, policy, and layer-distillation terms into one scalar loss."""
    base_kd_loss = trainer.alpha * distillation_loss
    loss = ce_loss + base_kd_loss
    if layer_distillation_loss is not None:
        loss = loss + layer_distillation_loss * trainer.layer_distill_weight
    if teacher_gate_balance_loss is not None:
        loss = loss + teacher_gate_balance_loss * trainer.teacher_gate_balance_alpha
    if teacher_gate_entropy_loss is not None:
        loss = loss + teacher_gate_entropy_loss * trainer.teacher_gate_entropy_alpha
    if teacher_gate_z_loss is not None:
        loss = loss + teacher_gate_z_loss * trainer.teacher_gate_router_z_loss_alpha
    if teacher_selection_policy_loss is not None:
        loss = loss + teacher_selection_policy_loss * trainer.reinforced_selection_policy_alpha
    return loss


__all__ = [
    "apply_grace_and_compute_distillation_loss",
    "apply_teacher_gate_routing",
    "compute_layer_distillation_loss",
    "compute_teacher_losses_and_grace",
    "compute_total_loss",
    "move_teacher_models_to_device",
    "prepare_teacher_batches",
    "run_student_forward_and_teacher_gate",
]
