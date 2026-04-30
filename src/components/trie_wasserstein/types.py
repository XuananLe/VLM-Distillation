from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass(slots=True)
class TrieNode:
    children: dict[int, "TrieNode"] = field(default_factory=dict)
    edge_id: int | None = None


@dataclass(slots=True)
class TrieBuildResult:
    path_flat: torch.Tensor
    path_offsets: torch.Tensor
    ignored_mask: torch.Tensor


@dataclass(slots=True)
class TrieRuntimeState:
    edge_weights: torch.Tensor
    student_path_flat: torch.Tensor
    student_path_offsets: torch.Tensor
    student_ignored_mask: torch.Tensor
    teacher_path_flat: torch.Tensor
    teacher_path_offsets: torch.Tensor
    teacher_ignored_mask: torch.Tensor
    num_edges: int
    tail_edge_id: int
    student_valid_count: int
    teacher_valid_count: int
