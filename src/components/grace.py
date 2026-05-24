import torch


def apply_grace_routing(
    *,
    routed_teacher_weights: torch.Tensor,
    teacher_grace_scores: torch.Tensor,
    teacher_grace_active_mask: torch.Tensor,
    prev_grace_score_ema: torch.Tensor | None,
    grace_ema_decay: float,
    grace_softmax_beta: float,
    grace_router_blend_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    routed_teacher_mask = routed_teacher_weights > 0
    has_prev_ema = prev_grace_score_ema is not None and prev_grace_score_ema.numel() == teacher_grace_scores.size(-1)
    if has_prev_ema:
        prev_grace_score_ema = prev_grace_score_ema.to(
            device=teacher_grace_scores.device,
            dtype=teacher_grace_scores.dtype,
        )
        smoothed_grace_scores = (
            grace_ema_decay * prev_grace_score_ema.unsqueeze(0) + (1.0 - grace_ema_decay) * teacher_grace_scores
        )
    else:
        smoothed_grace_scores = teacher_grace_scores

    batch_grace_mean = teacher_grace_scores.detach().mean(dim=0).float()
    if has_prev_ema:
        next_grace_score_ema = (
            grace_ema_decay * prev_grace_score_ema.float() + (1.0 - grace_ema_decay) * batch_grace_mean
        )
    else:
        next_grace_score_ema = batch_grace_mean

    grace_active_routed_mask = routed_teacher_mask & teacher_grace_active_mask
    has_grace_active_teacher = grace_active_routed_mask.any(dim=-1, keepdim=True)

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

    router_weights_clamped = routed_teacher_weights.clamp(min=torch.finfo(routed_teacher_weights.dtype).eps)
    # w_blend = w_router^lambda * w_grad^(1-lambda).
    router_grace_blend_weights = router_weights_clamped.pow(grace_router_blend_lambda) * grace_agreement_weights.clamp(
        min=torch.finfo(grace_agreement_weights.dtype).eps
    ).pow(1.0 - grace_router_blend_lambda)
    router_grace_blend_weights = router_grace_blend_weights * grace_teacher_mask.to(dtype=routed_teacher_weights.dtype)
    router_grace_blend_weights = router_grace_blend_weights / router_grace_blend_weights.sum(
        dim=-1, keepdim=True
    ).clamp(min=torch.finfo(router_grace_blend_weights.dtype).eps)

    teacher_mix_weights = torch.where(
        has_grace_active_teacher,
        router_grace_blend_weights,
        routed_teacher_weights,
    )
    return teacher_mix_weights, next_grace_score_ema


__all__ = [
    "apply_grace_routing",
]
