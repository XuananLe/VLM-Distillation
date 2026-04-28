from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass(slots=True)
class TrieNode:
    """One byte-trie node storing outgoing children and the incoming edge id."""

    children: dict[int, "TrieNode"] = field(default_factory=dict)
    edge_id: int | None = None


@dataclass(slots=True)
class TrieBuildResult:
    """Packed trie-path tables for one tokenizer."""

    path_flat: torch.Tensor
    path_offsets: torch.Tensor
    ignored_mask: torch.Tensor
    prefix_bucket_ids: torch.Tensor


@dataclass(slots=True)
class TrieRuntimeState:
    """Static CPU trie state shared by every forward pass for one tokenizer pair."""

    edge_weights: torch.Tensor
    student_path_flat: torch.Tensor
    student_path_offsets: torch.Tensor
    student_ignored_mask: torch.Tensor
    student_prefix_bucket_ids: torch.Tensor
    teacher_path_flat: torch.Tensor
    teacher_path_offsets: torch.Tensor
    teacher_ignored_mask: torch.Tensor
    teacher_prefix_bucket_ids: torch.Tensor
    prefix_tail_path_flat: torch.Tensor
    prefix_tail_path_offsets: torch.Tensor
    num_edges: int
    num_prefix_tail_buckets: int
    student_valid_count: int
    teacher_valid_count: int
