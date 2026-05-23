from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch

BoundaryKind = Literal["none", "space", "continuation", "end_word"]


@dataclass(slots=True)
class TrieNode:
    children: dict[int, "TrieNode"] = field(default_factory=dict)
    edge_id: int = -1


@dataclass(slots=True)
class TrieRuntimeState:
    edge_weights: torch.Tensor
    student_token_paths: list[list[int]]
    student_ignored_mask: torch.Tensor
    student_non_text_mask: torch.Tensor
    teacher_token_paths: list[list[int]]
    teacher_ignored_mask: torch.Tensor
    teacher_non_text_mask: torch.Tensor
    tail_edge_id: int
