import torch

from src.trainer.routing_utils import mean_categorical_entropy, summarize_teacher_vector


def build_distillation_train_metrics(
    *,
    loss: torch.Tensor,
    distillation_loss: torch.Tensor,
    ce_loss: torch.Tensor,
    layer_distillation_loss: torch.Tensor | None,
    layer_distillation_time: float | None,
    layer_distill_source: str | None,
    compute_loss_time: float,
    student_forward_time: float,
    teacher_gate_time: float,
    routing_constraint_time: float,
    teacher_loss_matrix_time: float,
    grace_routing_time: float,
    outside_compute_loss_time: float | None,
    teacher_loss_matrix: torch.Tensor,
    routed_teacher_gate_weights: torch.Tensor | None,
    effective_teacher_gate_weights: torch.Tensor | None,
    teacher_gate_weights: torch.Tensor | None,
    teacher_gate_logits: torch.Tensor | None,
    teacher_gate_routing_scores: torch.Tensor | None,
    teacher_gate_balance_loss: torch.Tensor | None,
    teacher_gate_entropy_loss: torch.Tensor | None,
    teacher_gate_z_loss: torch.Tensor | None,
    teacher_gate_capacity,
    teacher_gate_routing_fallback_rate: torch.Tensor | None,
    teacher_gate_soft_load: torch.Tensor | None,
    teacher_gate_hard_load: torch.Tensor | None,
    teacher_gate_assignment_rate: torch.Tensor | None,
    teacher_grace_scores: torch.Tensor | None,
    teacher_grace_active: torch.Tensor | None,
    teacher_grace_score_ema: torch.Tensor | None,
    teacher_grace_weights: torch.Tensor | None,
    teacher_grace_fallback_rate: torch.Tensor | None,
    reinforced_selection_metrics: dict[str, float] | None,
    grace_warmup_active: bool,
) -> dict[str, float]:
    """Assemble the scalar metrics logged for one distillation training step."""
    metrics = {
        "loss": loss.item(),
        "distillation_loss": distillation_loss.item(),
        "ce_loss": ce_loss.item(),
        "perf_compute_loss_s": compute_loss_time,
        "perf_student_forward_s": student_forward_time,
        "perf_teacher_gate_s": teacher_gate_time,
        "perf_routing_constraint_s": routing_constraint_time,
        "perf_teacher_loss_matrix_s": teacher_loss_matrix_time,
        "perf_grace_routing_s": grace_routing_time,
    }
    if layer_distillation_loss is not None and layer_distill_source is not None:
        metrics[f"{layer_distill_source}_layer_distill_loss"] = layer_distillation_loss.item()
    if layer_distillation_time is not None:
        metrics["perf_layer_distillation_s"] = layer_distillation_time
    if outside_compute_loss_time is not None:
        metrics["perf_outside_compute_loss_s"] = outside_compute_loss_time
    metrics.update(summarize_teacher_vector("teacher_kd_loss", teacher_loss_matrix))
    if reinforced_selection_metrics is not None:
        metrics.update(reinforced_selection_metrics)

    logged_weights = (
        effective_teacher_gate_weights
        if effective_teacher_gate_weights is not None
        else routed_teacher_gate_weights
    )
    if logged_weights is not None:
        metrics.update(summarize_teacher_vector("teacher_mix_w", logged_weights))
        metrics["teacher_mix_entropy"] = mean_categorical_entropy(logged_weights).item()

    if routed_teacher_gate_weights is None:
        return metrics

    metrics.update(summarize_teacher_vector("teacher_gate_w", logged_weights))
    metrics.update(summarize_teacher_vector("teacher_gate_router_w", teacher_gate_weights))
    metrics.update(summarize_teacher_vector("teacher_gate_routed_w", routed_teacher_gate_weights))
    metrics.update(summarize_teacher_vector("teacher_gate_logit", teacher_gate_logits))
    metrics.update(summarize_teacher_vector("teacher_gate_score", teacher_gate_routing_scores))
    metrics["teacher_gate_balance_loss"] = teacher_gate_balance_loss.item()
    metrics["teacher_gate_entropy_loss"] = (
        teacher_gate_entropy_loss.item() if teacher_gate_entropy_loss is not None else 0.0
    )
    metrics["teacher_gate_router_z_loss"] = (
        teacher_gate_z_loss.item() if teacher_gate_z_loss is not None else 0.0
    )
    metrics["teacher_gate_capacity"] = float(teacher_gate_capacity)
    metrics["teacher_gate_router_entropy"] = mean_categorical_entropy(
        teacher_gate_weights
    ).item()
    metrics["teacher_gate_routed_entropy"] = mean_categorical_entropy(
        routed_teacher_gate_weights
    ).item()
    metrics["teacher_gate_final_entropy"] = mean_categorical_entropy(logged_weights).item()
    metrics["teacher_gate_active_teachers"] = (
        routed_teacher_gate_weights > 0
    ).to(dtype=logged_weights.dtype).sum(dim=-1).mean().item()
    metrics["teacher_gate_routing_fallback_rate"] = teacher_gate_routing_fallback_rate.item()
    metrics["teacher_grace_warmup_active"] = float(grace_warmup_active)
    metrics.update(
        {
            f"teacher_gate_soft_load_{teacher_index}": load.item()
            for teacher_index, load in enumerate(teacher_gate_soft_load.detach())
        }
    )
    metrics.update(
        {
            f"teacher_gate_hard_load_{teacher_index}": load.item()
            for teacher_index, load in enumerate(teacher_gate_hard_load.detach())
        }
    )
    metrics.update(
        {
            f"teacher_gate_assignment_rate_{teacher_index}": rate.item()
            for teacher_index, rate in enumerate(teacher_gate_assignment_rate.detach())
        }
    )
    if teacher_grace_scores is not None and teacher_grace_active is not None:
        metrics.update(
            summarize_teacher_vector("teacher_grace_score", teacher_grace_scores)
        )
        metrics.update(
            {
                f"teacher_grace_active_{teacher_index}": active.item()
                for teacher_index, active in enumerate(
                    teacher_grace_active.detach().to(dtype=logged_weights.dtype).mean(dim=0)
                )
            }
        )
        metrics["teacher_grace_active_teachers"] = (
            teacher_grace_active.detach().to(dtype=logged_weights.dtype).sum(dim=-1).mean().item()
        )
        if teacher_grace_score_ema is not None:
            metrics.update(
                {
                    f"teacher_grace_score_ema_{teacher_index}": score.item()
                    for teacher_index, score in enumerate(teacher_grace_score_ema.detach())
                }
            )

    if teacher_grace_weights is not None:
        metrics.update(
            summarize_teacher_vector("teacher_grace_weight", teacher_grace_weights)
        )
        normalized_grace_weights = teacher_grace_weights / teacher_grace_weights.sum(
            dim=-1,
            keepdim=True,
        ).clamp(min=torch.finfo(teacher_grace_weights.dtype).eps)
        metrics["teacher_grace_entropy"] = mean_categorical_entropy(
            normalized_grace_weights
        ).item()
        metrics["teacher_grace_fallback_rate"] = teacher_grace_fallback_rate.item()
        metrics["teacher_grace_uniform_rate"] = teacher_grace_fallback_rate.item()

    return metrics


__all__ = ["build_distillation_train_metrics"]
