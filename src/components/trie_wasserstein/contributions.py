from __future__ import annotations

from dataclasses import dataclass

import torch

from .diagnostics import PrefixTailStats


@dataclass(slots=True)
class EdgeContributionResult:
    keys: torch.Tensor
    masses: torch.Tensor
    stats: PrefixTailStats


def build_signed_edge_contributions(
    *,
    scaled_logits: torch.Tensor,
    path_flat: torch.Tensor,
    path_offsets: torch.Tensor,
    ignored_mask: torch.Tensor,
    prefix_bucket_ids: torch.Tensor,
    edge_count: int,
    topk: int,
    valid_count: int,
    prefix_tail_path_flat: torch.Tensor,
    prefix_tail_path_offsets: torch.Tensor,
    num_prefix_tail_buckets: int,
    sign: float,
) -> EdgeContributionResult:
    """Expand token distributions into signed row-edge masses one row at a time."""
    dtype = scaled_logits.dtype

    if valid_count == 0:
        raise ValueError("Trie Wasserstein loss has no valid token paths for this tokenizer.")
    if num_prefix_tail_buckets == 0:
        raise ValueError("Trie Wasserstein loss has no prefix-tail buckets.")

    k = min(topk, valid_count)
    all_keys: list[torch.Tensor] = []
    all_masses: list[torch.Tensor] = []
    exact_mass_values: list[torch.Tensor] = []
    tail_mass_values: list[torch.Tensor] = []
    tail_entropy_values: list[torch.Tensor] = []
    tail_top_mass_values: list[torch.Tensor] = []

    # Keep this row-wise: the prefix-tail bookkeeping is easier to audit than
    # the previous vectorized version, and this loss is dominated by sparse trie
    # accounting rather than dense GPU matmuls.
    for row_index, row_logits in enumerate(scaled_logits):
        # Fold the row id into the edge key so identical trie edges from
        # different samples do not cancel before the per-sample loss is formed.
        row_base = row_logits.new_tensor(row_index * edge_count, dtype=torch.long)

        # Special and extended tokens must not appear in the normalizer; otherwise
        # they steal probability mass even though no semantic trie path can use it.
        masked_logits = row_logits.masked_fill(ignored_mask, float("-inf"))
        kept_logits, kept_token_ids = torch.topk(masked_logits, k=k, dim=-1, sorted=False)

        log_z = torch.logsumexp(masked_logits, dim=-1)
        kept_masses = (kept_logits - log_z).exp()

        probs = (masked_logits - log_z).exp()
        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        bucket_masses = probs.new_zeros(num_prefix_tail_buckets)
        bucket_masses.scatter_add_(0, prefix_bucket_ids, probs)
        kept_bucket_ids = prefix_bucket_ids[kept_token_ids]

        # Top-k tokens travel on exact token paths, so their mass must be removed
        # from the coarser prefix-tail bucket to avoid charging it twice.
        bucket_masses.scatter_add_(0, kept_bucket_ids, -kept_masses)
        bucket_masses = bucket_masses.clamp_min(0.0)

        tail_mass = bucket_masses.sum()
        bucket_distribution = bucket_masses / tail_mass.clamp(min=torch.finfo(dtype).eps)
        bucket_entropy = -(
            bucket_distribution
            * bucket_distribution.clamp(min=torch.finfo(dtype).eps).log()
        ).sum()

        exact_mass_values.append(kept_masses.sum().detach())
        tail_mass_values.append(tail_mass.detach())
        tail_entropy_values.append(bucket_entropy.detach())
        tail_top_mass_values.append(bucket_masses.max().detach())

        for token_id, mass in zip(kept_token_ids, kept_masses):
            start = path_offsets[token_id]
            end = path_offsets[token_id + 1]
            edge_ids = path_flat[start:end]
            if edge_ids.numel() == 0:
                continue
            all_keys.append(row_base + edge_ids)
            all_masses.append(mass.expand(edge_ids.numel()) * float(sign))

        active_bucket_ids = bucket_masses.nonzero(as_tuple=True)[0]
        for bucket_id in active_bucket_ids:
            start = prefix_tail_path_offsets[bucket_id]
            end = prefix_tail_path_offsets[bucket_id + 1]
            edge_ids = prefix_tail_path_flat[start:end]
            if edge_ids.numel() == 0:
                continue
            mass = bucket_masses[bucket_id]
            all_keys.append(row_base + edge_ids)
            all_masses.append(mass.expand(edge_ids.numel()) * float(sign))

    if not all_keys:
        raise ValueError("Trie Wasserstein loss produced no edge contributions.")

    stats = PrefixTailStats(
        exact_mass_mean=torch.stack(exact_mass_values).mean(),
        tail_mass_mean=torch.stack(tail_mass_values).mean(),
        tail_bucket_entropy=torch.stack(tail_entropy_values).mean(),
        tail_top_bucket_mass=torch.stack(tail_top_mass_values).mean(),
        num_tail_buckets=float(num_prefix_tail_buckets),
    )
    return EdgeContributionResult(
        keys=torch.cat(all_keys, dim=0),
        masses=torch.cat(all_masses, dim=0),
        stats=stats,
    )


def reduce_signed_edge_contributions_to_tree_loss(
    *,
    student_result: EdgeContributionResult,
    teacher_result: EdgeContributionResult,
    edge_weights: torch.Tensor,
    edge_count: int,
    num_rows: int,
) -> torch.Tensor:
    """Reduce signed row-edge masses to the mean tree-Wasserstein loss."""
    all_keys = torch.cat([student_result.keys, teacher_result.keys], dim=0)
    signed_masses = torch.cat([student_result.masses, teacher_result.masses], dim=0)

    # This sparse reduce replaces a dense [row, edge] balance matrix, which would
    # be wasteful because each token touches only a short trie path.
    active_keys, inverse = torch.unique(all_keys, return_inverse=True)
    signed_edge_balance = signed_masses.new_zeros(active_keys.numel())
    signed_edge_balance.index_add_(0, inverse, signed_masses)
    active_edges = active_keys.remainder(edge_count)
    total_loss = (
        edge_weights.index_select(0, active_edges) * signed_edge_balance.abs()
    ).sum()
    return total_loss / num_rows
