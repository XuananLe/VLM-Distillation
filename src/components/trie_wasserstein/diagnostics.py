from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class PrefixTailStats:
    exact_mass_mean: torch.Tensor
    tail_mass_mean: torch.Tensor
    tail_bucket_entropy: torch.Tensor
    tail_top_bucket_mass: torch.Tensor
    num_tail_buckets: float


def summarize_prefix_tail_stats(stats: PrefixTailStats) -> dict[str, float]:
    """Convert one prefix-tail stat object into scalar logging values."""
    return {
        "trie_exact_mass_mean": float(stats.exact_mass_mean.item()),
        "trie_tail_mass_mean": float(stats.tail_mass_mean.item()),
        "trie_tail_bucket_entropy": float(stats.tail_bucket_entropy.item()),
        "trie_tail_top_bucket_mass": float(stats.tail_top_bucket_mass.item()),
        "trie_num_tail_buckets": float(stats.num_tail_buckets),
    }
