from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class TrieMassStats:
    exact_mass_mean: torch.Tensor
    tail_mass_mean: torch.Tensor


def summarize_trie_mass_stats(stats: TrieMassStats) -> dict[str, float]:
    """Convert one trie mass stat object into scalar logging values."""
    return {
        "trie_exact_mass_mean": float(stats.exact_mass_mean.item()),
        "trie_tail_mass_mean": float(stats.tail_mass_mean.item()),
    }
