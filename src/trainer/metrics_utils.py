import torch

from src.trainer.routing_utils import mean_categorical_entropy


def _indexed_metrics(prefix: str, tensor: torch.Tensor) -> dict:
    return {f"{prefix}_{i}": v.item() for i, v in enumerate(tensor.detach())}


def build_distillation_train_metrics(
    *,
    loss: torch.Tensor,
    distillation_loss: torch.Tensor,
    ce_loss: torch.Tensor,
    routed_teacher_weights: torch.Tensor | None,
    teacher_mix_weights: torch.Tensor | None,
    teacher_router_weights: torch.Tensor | None,
    teacher_gate_entropy_loss: torch.Tensor | None,
    teacher_gate_z_loss: torch.Tensor | None,
    teacher_grace_active_mask: torch.Tensor | None,
    teacher_grace_score_ema: torch.Tensor | None,
    teacher_grace_weights: torch.Tensor | None,
    teacher_grace_fallback_rate: torch.Tensor | None,
    reinforced_selection_metrics: dict[str, float] | None,
    grace_routing_active: bool,
) -> dict[str, float]:
    metrics = {
        "loss": loss.item(),
        "distillation_loss": distillation_loss.item(),
        "ce_loss": ce_loss.item(),
    }
    if reinforced_selection_metrics is not None:
        metrics.update(reinforced_selection_metrics)

    logged_teacher_mix_weights = teacher_mix_weights if teacher_mix_weights is not None else routed_teacher_weights
    if logged_teacher_mix_weights is not None:
        metrics["teacher_mix_entropy"] = mean_categorical_entropy(logged_teacher_mix_weights).item()

    if routed_teacher_weights is None:
        return metrics

    metrics["teacher_gate_entropy_loss"] = (
        teacher_gate_entropy_loss.item() if teacher_gate_entropy_loss is not None else 0.0
    )
    metrics["teacher_gate_router_z_loss"] = teacher_gate_z_loss.item() if teacher_gate_z_loss is not None else 0.0
    metrics["teacher_gate_router_entropy"] = mean_categorical_entropy(teacher_router_weights).item()
    metrics["teacher_gate_routed_entropy"] = mean_categorical_entropy(routed_teacher_weights).item()
    metrics["teacher_gate_mix_entropy"] = mean_categorical_entropy(logged_teacher_mix_weights).item()
    metrics["teacher_gate_active_teachers"] = (
        (routed_teacher_weights > 0).to(dtype=logged_teacher_mix_weights.dtype).sum(dim=-1).mean().item()
    )
    metrics["teacher_grace_routing_active"] = float(grace_routing_active)
    if teacher_grace_active_mask is not None:
        metrics.update(_indexed_metrics(
            "teacher_grace_active",
            teacher_grace_active_mask.detach().to(dtype=logged_teacher_mix_weights.dtype).mean(dim=0),
        ))
        metrics["teacher_grace_active_teachers"] = (
            teacher_grace_active_mask.detach().to(dtype=logged_teacher_mix_weights.dtype).sum(dim=-1).mean().item()
        )
        if teacher_grace_score_ema is not None:
            metrics.update(_indexed_metrics("teacher_grace_score_ema", teacher_grace_score_ema))

    if teacher_grace_weights is not None:
        normalized_grace_weights = teacher_grace_weights / teacher_grace_weights.sum(
            dim=-1,
            keepdim=True,
        ).clamp(min=torch.finfo(teacher_grace_weights.dtype).eps)
        metrics["teacher_grace_entropy"] = mean_categorical_entropy(normalized_grace_weights).item()
        metrics["teacher_grace_fallback_rate"] = teacher_grace_fallback_rate.item()
        metrics["teacher_grace_uniform_rate"] = teacher_grace_fallback_rate.item()

    return metrics


__all__ = ["build_distillation_train_metrics"]
