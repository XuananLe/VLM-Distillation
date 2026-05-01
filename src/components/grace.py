import torch


def apply_grace_routing(
    *,
    routed_teacher_weights: torch.Tensor | None,
    teacher_grace_scores: torch.Tensor | None,
    teacher_grace_active_mask: torch.Tensor | None,
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
    if (
        routed_teacher_weights is None
        or teacher_grace_scores is None
        or teacher_grace_active_mask is None
    ):
        return (
            routed_teacher_weights,
            teacher_grace_scores,
            teacher_grace_active_mask,
            None,
            None,
            prev_grace_score_ema,
        )

    routed_teacher_mask = routed_teacher_weights > 0
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
    # ema = beta * old_ema + (1 - beta) * new_value
    batch_grace_mean = teacher_grace_scores.detach().mean(dim=0).float()
    if prev_grace_score_ema is None or prev_grace_score_ema.numel() != batch_grace_mean.numel():
        next_grace_score_ema = batch_grace_mean
    else:
        next_grace_score_ema = (
            grace_ema_decay * prev_grace_score_ema.float()
            + (1.0 - grace_ema_decay) * batch_grace_mean
        )

    grace_active_routed_mask = routed_teacher_mask & teacher_grace_active_mask.to(dtype=torch.bool)
    has_grace_active_teacher = grace_active_routed_mask.any(dim=-1, keepdim=True)

    normalized_router_weights = routed_teacher_weights * routed_teacher_mask.to(dtype=routed_teacher_weights.dtype)
    normalized_router_weights = normalized_router_weights / normalized_router_weights.sum(dim=-1, keepdim=True).clamp(
        min=torch.finfo(normalized_router_weights.dtype).eps
    )

    # If no teacher passes the GRACE threshold, use the router as-is. Otherwise,
    # restrict GRACE weighting to teachers that are both routed and active.
    grace_teacher_mask = torch.where(
        has_grace_active_teacher,
        grace_active_routed_mask,
        routed_teacher_mask,
    )

    masked_smoothed_scores = smoothed_grace_scores.masked_fill(~grace_teacher_mask, float("-inf"))
    grace_agreement_weights = torch.softmax(
        grace_softmax_beta * masked_smoothed_scores,
        dim=-1,
    )
    grace_agreement_weights = grace_agreement_weights * grace_teacher_mask.to(dtype=routed_teacher_weights.dtype)
    grace_agreement_weights = grace_agreement_weights / grace_agreement_weights.sum(dim=-1, keepdim=True).clamp(
        min=torch.finfo(grace_agreement_weights.dtype).eps
    )

    router_weights_clamped = normalized_router_weights.clamp(min=torch.finfo(normalized_router_weights.dtype).eps)
    # w_blend = w_router^lambda * w_grad^(1-lambda).
    router_grace_blend_weights = (
        router_weights_clamped.pow(grace_router_blend_lambda)
        * grace_agreement_weights.clamp(min=torch.finfo(grace_agreement_weights.dtype).eps).pow(
            1.0 - grace_router_blend_lambda
        )
    )
    router_grace_blend_weights = router_grace_blend_weights * grace_teacher_mask.to(dtype=routed_teacher_weights.dtype)
    router_grace_blend_weights = router_grace_blend_weights / router_grace_blend_weights.sum(dim=-1, keepdim=True).clamp(
        min=torch.finfo(router_grace_blend_weights.dtype).eps
    )

    grace_teacher_counts = grace_teacher_mask.sum(dim=-1, keepdim=True).clamp(min=1)
    uniform_grace_teacher_weights = grace_teacher_mask.to(dtype=routed_teacher_weights.dtype) / grace_teacher_counts
    grace_score_spread = (
        masked_smoothed_scores.max(dim=-1).values
        - smoothed_grace_scores.masked_fill(~grace_teacher_mask, float("inf")).min(dim=-1).values
    )
    # If all active teachers look almost identical to GRACE, fall back to uniform
    # over the active set instead of overfitting to tiny score differences.
    use_uniform_grace_weights = has_grace_active_teacher.squeeze(-1) & (grace_score_spread < grace_epsilon)
    grace_mix_weights = torch.where(
        use_uniform_grace_weights.unsqueeze(-1),
        uniform_grace_teacher_weights,
        router_grace_blend_weights,
    )
    teacher_mix_weights = torch.where(
        has_grace_active_teacher,
        grace_mix_weights,
        normalized_router_weights,
    )
    grace_fallback_rate = (
        (~has_grace_active_teacher.squeeze(-1)) | use_uniform_grace_weights
    ).to(dtype=routed_teacher_weights.dtype).mean()
    grace_agreement_weights = torch.where(
        has_grace_active_teacher,
        grace_agreement_weights,
        torch.zeros_like(grace_agreement_weights),
    )
    return (
        teacher_mix_weights,
        teacher_grace_scores,
        teacher_grace_active_mask,
        grace_agreement_weights,
        grace_fallback_rate,
        next_grace_score_ema,
    )

__all__ = [
    "apply_grace_routing",
]
