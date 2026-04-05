import torch

from src.trainer.routing_utils import mean_categorical_entropy, summarize_teacher_vector


def build_distillation_train_metrics(
    *,
    loss: torch.Tensor,
    distillation_loss: torch.Tensor,
    ce_loss: torch.Tensor,
    compute_loss_time: float,
    student_forward_time: float,
    teacher_gate_time: float,
    routing_constraint_time: float,
    teacher_loss_matrix_time: float,
    alignment_routing_time: float,
    outside_compute_loss_time: float | None,
    teacher_loss_matrix: torch.Tensor,
    routed_teacher_gate_weights: torch.Tensor | None,
    effective_teacher_gate_weights: torch.Tensor | None,
    teacher_gate_weights: torch.Tensor | None,
    teacher_gate_logits: torch.Tensor | None,
    teacher_gate_routing_scores: torch.Tensor | None,
    teacher_gate_balance_loss: torch.Tensor | None,
    teacher_gate_z_loss: torch.Tensor | None,
    teacher_gate_capacity,
    teacher_gate_routing_fallback_rate: torch.Tensor | None,
    teacher_gate_soft_load: torch.Tensor | None,
    teacher_gate_hard_load: torch.Tensor | None,
    teacher_gate_assignment_rate: torch.Tensor | None,
    teacher_gate_bias: torch.Tensor | None,
    teacher_alignment_scores: torch.Tensor | None,
    teacher_alignment_active: torch.Tensor | None,
    teacher_alignment_score_ema: torch.Tensor | None,
    teacher_alignment_weights: torch.Tensor | None,
    teacher_alignment_fallback_rate: torch.Tensor | None,
    alignment_warmup_active: bool,
) -> dict[str, float]:
    metrics = {
        "loss": loss.item(),
        "distillation_loss": distillation_loss.item(),
        "ce_loss": ce_loss.item(),
        "perf_compute_loss_s": compute_loss_time,
        "perf_student_forward_s": student_forward_time,
        "perf_teacher_gate_s": teacher_gate_time,
        "perf_routing_constraint_s": routing_constraint_time,
        "perf_teacher_loss_matrix_s": teacher_loss_matrix_time,
        "perf_alignment_routing_s": alignment_routing_time,
    }
    if outside_compute_loss_time is not None:
        metrics["perf_outside_compute_loss_s"] = outside_compute_loss_time
    metrics.update(summarize_teacher_vector("teacher_kd_loss", teacher_loss_matrix))

    if routed_teacher_gate_weights is None:
        return metrics

    logged_weights = (
        effective_teacher_gate_weights
        if effective_teacher_gate_weights is not None
        else routed_teacher_gate_weights
    )
    metrics.update(summarize_teacher_vector("teacher_gate_w", logged_weights))
    metrics.update(summarize_teacher_vector("teacher_gate_router_w", teacher_gate_weights))
    metrics.update(summarize_teacher_vector("teacher_gate_routed_w", routed_teacher_gate_weights))
    metrics.update(summarize_teacher_vector("teacher_gate_logit", teacher_gate_logits))
    metrics.update(summarize_teacher_vector("teacher_gate_score", teacher_gate_routing_scores))
    metrics["teacher_gate_balance_loss"] = teacher_gate_balance_loss.item()
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
    metrics["teacher_alignment_warmup_active"] = float(alignment_warmup_active)
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
    metrics.update(
        {
            f"teacher_gate_bias_{teacher_index}": bias.item()
            for teacher_index, bias in enumerate(teacher_gate_bias.detach())
        }
    )

    if teacher_alignment_scores is not None and teacher_alignment_active is not None:
        metrics.update(
            summarize_teacher_vector("teacher_alignment_score", teacher_alignment_scores)
        )
        metrics.update(
            {
                f"teacher_alignment_active_{teacher_index}": active.item()
                for teacher_index, active in enumerate(
                    teacher_alignment_active.detach().to(dtype=logged_weights.dtype).mean(dim=0)
                )
            }
        )
        metrics["teacher_alignment_active_teachers"] = (
            teacher_alignment_active.detach().to(dtype=logged_weights.dtype).sum(dim=-1).mean().item()
        )
        if teacher_alignment_score_ema is not None:
            metrics.update(
                {
                    f"teacher_alignment_score_ema_{teacher_index}": score.item()
                    for teacher_index, score in enumerate(teacher_alignment_score_ema.detach())
                }
            )

    if teacher_alignment_weights is not None:
        metrics.update(
            summarize_teacher_vector("teacher_alignment_weight", teacher_alignment_weights)
        )
        normalized_alignment_weights = teacher_alignment_weights / teacher_alignment_weights.sum(
            dim=-1,
            keepdim=True,
        ).clamp(min=torch.finfo(teacher_alignment_weights.dtype).eps)
        metrics["teacher_alignment_entropy"] = mean_categorical_entropy(
            normalized_alignment_weights
        ).item()
        metrics["teacher_alignment_fallback_rate"] = teacher_alignment_fallback_rate.item()
        metrics["teacher_alignment_uniform_rate"] = teacher_alignment_fallback_rate.item()

    return metrics


__all__ = ["build_distillation_train_metrics"]
