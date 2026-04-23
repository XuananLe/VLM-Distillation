import math

import torch
from einops import einsum


def summarize_teacher_vector(prefix: str, values: torch.Tensor) -> dict[str, float]:
    """Average a `[batch, teacher]` tensor over the batch and emit per-teacher scalar metrics."""
    return {
        f"{prefix}_{teacher_index}": value.item()
        for teacher_index, value in enumerate(values.detach().mean(dim=0))
    }


def mean_categorical_entropy(weights: torch.Tensor) -> torch.Tensor:
    """Return the mean categorical entropy of a batch of teacher-weight vectors."""
    safe_weights = weights.clamp(min=torch.finfo(weights.dtype).eps)
    return (-(safe_weights * safe_weights.log()).sum(dim=-1)).mean()


def compute_teacher_gate_balance_loss(
    teacher_gate_weights: torch.Tensor,
    teacher_gate_top_k: int,
    routing_scores: torch.Tensor | None = None,
    assignment_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the MoE-style gate balance loss and its soft/hard load summaries."""
    _, num_teachers = teacher_gate_weights.shape
    soft_load = teacher_gate_weights.mean(dim=0)
    if assignment_mask is None:
        assignment_mask = build_topk_assignment_mask(
            routing_source=teacher_gate_weights if routing_scores is None else routing_scores,
            teacher_gate_top_k=teacher_gate_top_k,
            num_teachers=num_teachers,
        ).sum(dim=1).to(dtype=torch.bool)
    hard_counts = assignment_mask.to(dtype=teacher_gate_weights.dtype).sum(dim=0)
    hard_load = hard_counts / hard_counts.sum().clamp(min=1.0)
    # Same MoE-style idea as Switch/auxiliary load balancing: align the soft routing
    # mass with the hard assignment frequencies, scaled by num_teachers to keep O(1).
    balance_loss = num_teachers * einsum(
        soft_load,
        hard_load,
        "teacher, teacher ->",
    )
    return balance_loss, soft_load, hard_load


def compute_teacher_gate_z_loss(router_logits: torch.Tensor) -> torch.Tensor:
    """Return the router z-loss used to keep gate logits numerically bounded."""
    return torch.logsumexp(router_logits.float(), dim=-1).square().mean().to(dtype=router_logits.dtype)


def compute_teacher_gate_entropy_loss(teacher_gate_weights: torch.Tensor) -> torch.Tensor:
    """Return the negative-entropy regularizer applied to teacher-gate weights."""
    return -mean_categorical_entropy(teacher_gate_weights).to(dtype=teacher_gate_weights.dtype)


def apply_teacher_gate_constraints(
    routing_scores: torch.Tensor,
    teacher_gate_weights: torch.Tensor,
    teacher_gate_top_k: int,
    teacher_gate_capacity_factor: float,
) -> tuple[torch.Tensor, int, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply top-k and capacity constraints to router weights and return routed assignment stats."""
    batch_size, num_teachers = teacher_gate_weights.shape
    top_k = min(teacher_gate_top_k, num_teachers)
    # Capacity is a per-expert cap after top-k routing. Without it, one expert can
    # absorb the whole batch even when other experts are nominally in the top-k set.
    capacity = max(
        1,
        math.ceil(teacher_gate_capacity_factor * batch_size * top_k / num_teachers),
    )

    topk_scores, topk_indices = routing_scores.topk(top_k, dim=-1)
    assignment_mask = torch.zeros_like(teacher_gate_weights, dtype=torch.bool)
    for teacher_index in range(num_teachers):
        candidate_mask = topk_indices == teacher_index
        if not candidate_mask.any():
            continue
        sample_indices, topk_slots = candidate_mask.nonzero(as_tuple=True)
        candidate_scores = topk_scores[sample_indices, topk_slots]
        if candidate_scores.numel() > capacity:
            keep_indices = candidate_scores.topk(capacity, largest=True, sorted=False).indices
            sample_indices = sample_indices[keep_indices]
        assignment_mask[sample_indices, teacher_index] = True

    missing_indices = (~assignment_mask.any(dim=-1)).nonzero(as_tuple=True)[0]
    fallback_rate = teacher_gate_weights.new_tensor(missing_indices.numel() / max(batch_size, 1))
    if missing_indices.numel() > 0:
        # Capacity pruning can leave a sample with no surviving expert; fall back to
        # plain argmax so downstream weighting never sees an all-zero teacher row.
        fallback_indices = routing_scores.argmax(dim=-1)
        assignment_mask[missing_indices, fallback_indices[missing_indices]] = True

    routed_weights = torch.where(
        assignment_mask,
        teacher_gate_weights,
        torch.zeros_like(teacher_gate_weights),
    )

    routed_weights = routed_weights / routed_weights.sum(dim=-1, keepdim=True).clamp(
        min=torch.finfo(routed_weights.dtype).eps
    )
    expert_load = assignment_mask.float().sum(dim=0)
    assignment_rate = expert_load / max(batch_size, 1)
    return routed_weights, capacity, assignment_rate, expert_load, fallback_rate, assignment_mask


def build_topk_assignment_mask(
    *,
    routing_source: torch.Tensor,
    teacher_gate_top_k: int,
    num_teachers: int,
) -> torch.Tensor:
    """Build the raw top-k assignment mask before any capacity pruning."""
    top_k = min(teacher_gate_top_k, num_teachers)
    topk_teachers = routing_source.topk(top_k, dim=-1).indices
    return torch.nn.functional.one_hot(
        topk_teachers,
        num_classes=num_teachers,
    )


def build_capacity_limited_assignment_mask(
    *,
    topk_scores: torch.Tensor,
    topk_indices: torch.Tensor,
    num_teachers: int,
    capacity: int,
    reference_weights: torch.Tensor,
) -> torch.Tensor:
    """Prune top-k teacher assignments down to the configured per-teacher capacity."""
    assignment_mask = torch.zeros_like(reference_weights, dtype=torch.bool)
    for teacher_index in range(num_teachers):
        candidate_mask = topk_indices == teacher_index
        if not candidate_mask.any():
            continue
        sample_indices, topk_slots = candidate_mask.nonzero(as_tuple=True)
        candidate_scores = topk_scores[sample_indices, topk_slots]
        if candidate_scores.numel() > capacity:
            keep_indices = candidate_scores.topk(capacity, largest=True, sorted=False).indices
            sample_indices = sample_indices[keep_indices]
        assignment_mask[sample_indices, teacher_index] = True
    return assignment_mask
