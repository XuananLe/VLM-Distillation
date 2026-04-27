import torch


def apply_grace_routing(
    *,
    teacher_gate_weights: torch.Tensor | None,
    teacher_grace_scores: torch.Tensor | None,
    teacher_grace_active: torch.Tensor | None,
    prev_grace_score_ema: torch.Tensor | None,
    grace_ema_decay: float,
    grace_softmax_beta: float,
    grace_router_blend_lambda: float,
    grace_epsilon: float,
) -> tuple[
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    """Blend router weights with GRACE agreement scores and return the effective routing state."""
    if (
        teacher_gate_weights is None
        or teacher_grace_scores is None
        or teacher_grace_active is None
    ):
        return (
            teacher_gate_weights,
            teacher_grace_scores,
            teacher_grace_active,
            None,
            None,
            prev_grace_score_ema,
        )

    router_available_mask = teacher_gate_weights > 0
    if prev_grace_score_ema is None or prev_grace_score_ema.numel() != teacher_grace_scores.size(-1):
        smoothed_grace_scores = teacher_grace_scores
    else:
        prev_grace_score_ema = prev_grace_score_ema.to(
            device=teacher_grace_scores.device,
            dtype=teacher_grace_scores.dtype,
        )
        # Per-sample scores are noisy, so smooth them against the running teacher-level EMA.
        smoothed_grace_scores = (
            grace_ema_decay * prev_grace_score_ema.unsqueeze(0)
            + (1.0 - grace_ema_decay) * teacher_grace_scores
        )

    batch_grace_mean = teacher_grace_scores.detach().mean(dim=0).float()
    if prev_grace_score_ema is None or prev_grace_score_ema.numel() != batch_grace_mean.numel():
        next_grace_score_ema = batch_grace_mean
    else:
        next_grace_score_ema = (
            grace_ema_decay * prev_grace_score_ema.float()
            + (1.0 - grace_ema_decay) * batch_grace_mean
        )

    active_mask = router_available_mask & teacher_grace_active.to(dtype=torch.bool)
    has_active = active_mask.any(dim=-1, keepdim=True)

    router_weights = teacher_gate_weights * router_available_mask.to(dtype=teacher_gate_weights.dtype)
    router_weights = router_weights / router_weights.sum(dim=-1, keepdim=True).clamp(
        min=torch.finfo(router_weights.dtype).eps
    )

    # If no teacher passes the GRACE threshold, use the router as-is. Otherwise,
    # restrict GRACE weighting to teachers that are both routed and active.
    grace_mask = torch.where(has_active, active_mask, router_available_mask)

    masked_smoothed_scores = smoothed_grace_scores.masked_fill(~grace_mask, float("-inf"))
    gradient_weights = torch.softmax(
        grace_softmax_beta * masked_smoothed_scores,
        dim=-1,
    )
    gradient_weights = gradient_weights * grace_mask.to(dtype=teacher_gate_weights.dtype)
    gradient_weights = gradient_weights / gradient_weights.sum(dim=-1, keepdim=True).clamp(
        min=torch.finfo(gradient_weights.dtype).eps
    )

    router_weights_clamped = router_weights.clamp(min=torch.finfo(router_weights.dtype).eps)
    # Blend in log-space / geometric space:
    # w_blend ∝ w_router^lambda * w_grad^(1-lambda).
    blended_weights = (
        router_weights_clamped.pow(grace_router_blend_lambda)
        * gradient_weights.clamp(min=torch.finfo(gradient_weights.dtype).eps).pow(
            1.0 - grace_router_blend_lambda
        )
    )
    blended_weights = blended_weights * grace_mask.to(dtype=teacher_gate_weights.dtype)
    blended_weights = blended_weights / blended_weights.sum(dim=-1, keepdim=True).clamp(
        min=torch.finfo(blended_weights.dtype).eps
    )

    active_counts = grace_mask.sum(dim=-1, keepdim=True).clamp(min=1)
    uniform_weights = grace_mask.to(dtype=teacher_gate_weights.dtype) / active_counts
    score_spread = (
        masked_smoothed_scores.max(dim=-1).values
        - smoothed_grace_scores.masked_fill(~grace_mask, float("inf")).min(dim=-1).values
    )
    # If all active teachers look almost identical to GRACE, fall back to uniform
    # over the active set instead of overfitting to tiny score differences.
    use_uniform = has_active.squeeze(-1) & (score_spread < grace_epsilon)
    grace_effective_weights = torch.where(use_uniform.unsqueeze(-1), uniform_weights, blended_weights)
    effective_weights = torch.where(has_active, grace_effective_weights, router_weights)
    fallback_rate = ((~has_active.squeeze(-1)) | use_uniform).to(dtype=teacher_gate_weights.dtype).mean()
    gradient_weights = torch.where(has_active, gradient_weights, torch.zeros_like(gradient_weights))
    return (
        effective_weights,
        teacher_grace_scores,
        teacher_grace_active,
        gradient_weights,
        fallback_rate,
        next_grace_score_ema,
    )

__all__ = [
    "apply_grace_routing",
]
