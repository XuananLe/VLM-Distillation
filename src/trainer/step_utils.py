import time

import torch

from src.components.grace import apply_grace_routing
from src.components.objective_conflict import resolve_objective_conflict_weights
from src.components.reinforced_teacher_selection import compute_reinforced_selection_state
from src.components.teacher_weighting import solve_gradient_weight_vector
from src.components.forward_utils import forward_with_kwarg_retry, infer_batch_size
from src.trainer.alignment_utils import compute_teacher_loss_matrix
from src.trainer.gradient_utils import (
    compute_pooled_ce_grace_grad,
)
from src.trainer.routing_utils import (
    apply_teacher_gate_constraints,
    compute_teacher_gate_balance_loss,
    compute_teacher_gate_entropy_loss,
    compute_teacher_gate_z_loss,
)
from src.trainer.distillation_utils import build_cached_teacher_batches, build_teacher_batches


def move_teacher_models_to_device(teacher_models, target_device) -> None:
    for index, teacher_model in enumerate(teacher_models):
        teacher_device = next(teacher_model.parameters()).device
        if teacher_device != target_device:
            teacher_models[index] = teacher_model.to(target_device)


def prepare_teacher_batches(*, inputs, student_inputs, num_teachers: int, teacher_models):
    cached_teacher_batches = build_cached_teacher_batches(inputs, num_teachers)
    teacher_batches = None
    if cached_teacher_batches is None and teacher_models:
        teacher_batches = build_teacher_batches(inputs, student_inputs, num_teachers)
    elif cached_teacher_batches is None:
        raise ValueError(
            "No teacher inputs were found in the batch. Provide teacher models or cached teacher logits."
        )
    elif len(cached_teacher_batches) != num_teachers:
        raise ValueError(
            "Cached teacher-logit batch count does not match the configured teacher count. "
            f"cached={len(cached_teacher_batches)}, configured={num_teachers}"
        )
    return teacher_batches, cached_teacher_batches


def run_student_forward_and_teacher_gate(*, trainer, model, student_inputs):
    if trainer.teacher_gate is not None:
        trainer.teacher_gate.reset()
    if trainer.reinforced_teacher_selector is not None:
        trainer.reinforced_teacher_selector.reset()

    student_forward_start_time = time.perf_counter()
    student_outputs = forward_with_kwarg_retry(
        model,
        {**student_inputs, "return_dict": True, "output_hidden_states": False},
    )
    student_forward_time = time.perf_counter() - student_forward_start_time
    student_logits = student_outputs.logits

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
        "student_forward_time": student_forward_time,
        "teacher_gate_time": teacher_gate_time,
        "teacher_gate_logits": teacher_gate_logits,
        "teacher_gate_weights": teacher_gate_weights,
        "teacher_gate_routing_scores": teacher_gate_routing_scores,
    }


def apply_teacher_gate_routing(*, trainer, model, teacher_gate_logits, teacher_gate_weights, teacher_gate_routing_scores):
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
    teacher_loss_matrix_start_time = time.perf_counter()
    (
        teacher_loss_matrix,
        teacher_grace_scores,
        teacher_grace_active,
        teacher_gradient_vectors,
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
        collect_teacher_gradient_vectors=(
            trainer.teacher_weighting_strategy == "gradient_optimal"
            or trainer.objective_conflict_strategy != "fixed"
        ),
        collect_teacher_target_batches=trainer.teacher_weighting_strategy == "reinforced_selection",
        grace_threshold=trainer.grace_threshold,
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
        "teacher_gradient_vectors": teacher_gradient_vectors,
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
    teacher_gradient_vectors,
    teacher_loss_matrix,
    teacher_logits_batches,
    teacher_label_batches,
    labels,
    attention_mask,
    ce_loss,
):
    grace_routing_start_time = time.perf_counter()
    teacher_grace_weights = None
    teacher_grace_fallback_rate = None
    effective_teacher_gate_weights = None
    reinforced_selection_metrics = None
    teacher_selection_policy_loss = None

    if trainer.teacher_weighting_strategy == "reinforced_selection":
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

    if trainer.teacher_weighting_strategy == "gradient_optimal":
        if teacher_gradient_vectors is not None:
            solved_weights = solve_gradient_weight_vector(
                teacher_gradient_vectors,
                weight_cap=trainer.gradient_weight_cap,
                max_steps=trainer.gradient_weight_steps,
            ).to(
                device=teacher_loss_matrix.device,
                dtype=teacher_loss_matrix.dtype,
            )
            effective_teacher_gate_weights = solved_weights.unsqueeze(0).expand(
                teacher_loss_matrix.size(0),
                -1,
            )
    else:
        if not trainer.should_apply_grace_routing():
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
        distillation_loss = (teacher_loss_matrix * effective_teacher_gate_weights).sum(dim=-1).mean()
    elif routed_teacher_gate_weights is not None:
        distillation_loss = (teacher_loss_matrix * routed_teacher_gate_weights).sum(dim=-1).mean()

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


def compute_objective_conflict_state(
    *,
    trainer,
    student_logits,
    student_labels,
    distillation_loss,
    effective_teacher_gate_weights,
    routed_teacher_gate_weights,
    teacher_gradient_vectors,
):
    if trainer.objective_conflict_strategy == "fixed":
        return {
            "objective_ce_weight": None,
            "objective_kd_weight": None,
            "objective_gradient_cosine": None,
        }

    pooled_ce_grads = compute_pooled_ce_grace_grad(
        student_logits=student_logits.detach(),
        student_labels=student_labels,
    )
    ce_grad_vector = pooled_ce_grads.mean(dim=0)

    if teacher_gradient_vectors is None:
        kd_grad_vector = ce_grad_vector.new_zeros(ce_grad_vector.shape)
    else:
        if effective_teacher_gate_weights is not None:
            teacher_mix = effective_teacher_gate_weights.mean(dim=0).to(
                dtype=teacher_gradient_vectors.dtype,
                device=teacher_gradient_vectors.device,
            )
        elif routed_teacher_gate_weights is not None:
            teacher_mix = routed_teacher_gate_weights.mean(dim=0).to(
                dtype=teacher_gradient_vectors.dtype,
                device=teacher_gradient_vectors.device,
            )
        else:
            teacher_mix = teacher_gradient_vectors.new_full(
                (teacher_gradient_vectors.size(0),),
                1.0 / teacher_gradient_vectors.size(0),
            )
        kd_grad_vector = (teacher_gradient_vectors * teacher_mix.unsqueeze(-1)).sum(dim=0)

    base_kd_scale = max(1.0 - trainer.alpha, 0.0)
    objective_weights, gradient_cosine = resolve_objective_conflict_weights(
        strategy=trainer.objective_conflict_strategy,
        ce_grad=ce_grad_vector,
        kd_grad=base_kd_scale * kd_grad_vector,
        cagrad_c=trainer.objective_conflict_cagrad_c,
        cagrad_grid_steps=trainer.objective_conflict_cagrad_grid_steps,
    )
    return {
        "objective_ce_weight": objective_weights[0].to(dtype=distillation_loss.dtype),
        "objective_kd_weight": objective_weights[1].to(dtype=distillation_loss.dtype),
        "objective_gradient_cosine": gradient_cosine.to(dtype=distillation_loss.dtype),
    }


def update_eval_ce_stats(*, trainer, student_inputs, ce_loss) -> None:
    if trainer.model.training:
        return
    batch_size = infer_batch_size(student_inputs)
    trainer.eval_ce_loss_sum += ce_loss.detach().float().item() * batch_size
    trainer.eval_ce_loss_count += batch_size


def compute_total_loss(
    *,
    trainer,
    ce_loss,
    distillation_loss,
    teacher_gate_balance_loss,
    teacher_gate_entropy_loss,
    teacher_gate_z_loss,
    teacher_selection_policy_loss=None,
    objective_ce_weight=None,
    objective_kd_weight=None,
):
    base_kd_loss = (1.0 - trainer.alpha) * distillation_loss
    if objective_ce_weight is None or objective_kd_weight is None:
        loss = ce_loss + base_kd_loss
    else:
        loss = objective_ce_weight * ce_loss + objective_kd_weight * base_kd_loss
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
    "compute_objective_conflict_state",
    "compute_teacher_losses_and_grace",
    "compute_total_loss",
    "move_teacher_models_to_device",
    "prepare_teacher_batches",
    "run_student_forward_and_teacher_gate",
    "update_eval_ce_stats",
]
