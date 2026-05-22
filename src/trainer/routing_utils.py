import torch


def mean_categorical_entropy(weights: torch.Tensor) -> torch.Tensor:
    safe_weights = weights.clamp(min=torch.finfo(weights.dtype).eps)
    return (-(safe_weights * safe_weights.log()).sum(dim=-1)).mean()


def compute_teacher_gate_z_loss(router_logits: torch.Tensor) -> torch.Tensor:
    return torch.logsumexp(router_logits.float(), dim=-1).square().mean().to(dtype=router_logits.dtype)


def compute_teacher_gate_entropy_loss(teacher_router_weights: torch.Tensor) -> torch.Tensor:
    return -mean_categorical_entropy(teacher_router_weights).to(dtype=teacher_router_weights.dtype)


def apply_teacher_gate_topk(
    teacher_router_logits: torch.Tensor,
    teacher_router_weights: torch.Tensor,
    teacher_gate_top_k: int,
) -> torch.Tensor:
    num_teachers = teacher_router_weights.shape[1]
    top_k = min(teacher_gate_top_k, num_teachers)
    topk_indices = teacher_router_logits.topk(top_k, dim=-1).indices
    topk_mask = (
        torch.nn.functional.one_hot(
            topk_indices,
            num_classes=num_teachers,
        )
        .sum(dim=1)
        .to(dtype=torch.bool)
    )

    routed_weights = torch.where(
        topk_mask,
        teacher_router_weights,
        torch.zeros_like(teacher_router_weights),
    )

    routed_weights = routed_weights / routed_weights.sum(dim=-1, keepdim=True).clamp(
        min=torch.finfo(routed_weights.dtype).eps
    )
    return routed_weights
