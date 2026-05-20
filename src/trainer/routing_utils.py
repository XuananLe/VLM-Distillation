import torch


def summarize_teacher_vector(prefix: str, values: torch.Tensor) -> dict[str, float]:
    """Average a `[batch, teacher]` tensor over the batch and emit per-teacher scalar metrics."""
    return {
        f"{prefix}_{teacher_index}": value.item() for teacher_index, value in enumerate(values.detach().mean(dim=0))
    }


def mean_categorical_entropy(weights: torch.Tensor) -> torch.Tensor:
    """Return the mean categorical entropy of a batch of teacher-weight vectors."""
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
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, num_teachers = teacher_router_weights.shape
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
    assignment_rate = topk_mask.float().sum(dim=0) / max(batch_size, 1)
    return routed_weights, assignment_rate
