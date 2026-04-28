from __future__ import annotations

import torch
from torch import nn

from .contributions import (
    build_signed_edge_contributions,
    reduce_signed_edge_contributions_to_tree_loss,
)
from .diagnostics import PrefixTailStats
from .runtime_state import extend_vocab_state_with_ignored_tokens
from .trie_build import build_trie_state_from_tokenizers
from src.tokenizer_utils import (
    default_ignored_token_ids,
    normalize_model_id,
    resolve_vocab_size,
)


class TrieWassersteinLoss(nn.Module):
    """
    Tree-Wasserstein distillation on a shared byte trie.

    This module restores token identity across mismatched vocabularies by
    placing student and teacher token pieces on a shared UTF-8 byte trie and
    computing the weighted subtree-mass imbalance. The implementation uses a
    sparse top-k approximation with prefix-tail buckets.
    """

    def __init__(
        self,
        student_tokenizer,
        teacher_tokenizer,
        rho: float = 0.7,
        topk: int = 64,
        tail_depth: int = 1,
        tail_weight: float = 0.5,
        ignored_student_token_ids: set[int] | None = None,
        ignored_teacher_token_ids: set[int] | None = None,
    ):
        """Precompute shared byte-trie state for one student/teacher tokenizer pair."""
        super().__init__()
        if not 0.0 < float(rho) < 1.0:
            raise ValueError(f"rho must be in (0, 1), got {rho}")
        if int(topk) < 1:
            raise ValueError(f"topk must be >= 1, got {topk}")
        if int(tail_depth) < 1:
            raise ValueError(f"tail_depth must be >= 1, got {tail_depth}")
        if float(tail_weight) <= 0.0:
            raise ValueError(f"tail_weight must be > 0, got {tail_weight}")

        self.student_tokenizer = getattr(student_tokenizer, "tokenizer", None) or student_tokenizer
        self.teacher_tokenizer = getattr(teacher_tokenizer, "tokenizer", None) or teacher_tokenizer
        self.student_model_id = normalize_model_id(getattr(self.student_tokenizer, "name_or_path", None))
        self.teacher_model_id = normalize_model_id(getattr(self.teacher_tokenizer, "name_or_path", None))
        self.student_tokenizer_vocab_size = resolve_vocab_size(self.student_tokenizer)
        self.teacher_tokenizer_vocab_size = resolve_vocab_size(self.teacher_tokenizer)
        self.student_vocab_size = self.student_tokenizer_vocab_size
        self.teacher_vocab_size = self.teacher_tokenizer_vocab_size
        self.rho = float(rho)
        self.topk = int(topk)
        self.tail_depth = int(tail_depth)
        self.tail_weight = float(tail_weight)

        ignored_student = (
            default_ignored_token_ids(self.student_tokenizer)
            if ignored_student_token_ids is None
            else {int(token_id) for token_id in ignored_student_token_ids}
        )
        ignored_teacher = (
            default_ignored_token_ids(self.teacher_tokenizer)
            if ignored_teacher_token_ids is None
            else {int(token_id) for token_id in ignored_teacher_token_ids}
        )

        trie_state = build_trie_state_from_tokenizers(
            student_vocab_size=self.student_vocab_size,
            teacher_vocab_size=self.teacher_vocab_size,
            student_ignored_token_ids=tuple(sorted(ignored_student)),
            teacher_ignored_token_ids=tuple(sorted(ignored_teacher)),
            rho=self.rho,
            tail_depth=self.tail_depth,
            tail_weight=self.tail_weight,
            student_tokenizer=self.student_tokenizer,
            teacher_tokenizer=self.teacher_tokenizer,
        )

        self.num_edges = trie_state.num_edges
        self.num_prefix_tail_buckets = trie_state.num_prefix_tail_buckets
        self.student_valid_count = trie_state.student_valid_count
        self.teacher_valid_count = trie_state.teacher_valid_count

        self.register_buffer("edge_weights", trie_state.edge_weights, persistent=True)
        self.register_buffer("student_path_flat", trie_state.student_path_flat, persistent=True)
        self.register_buffer(
            "student_path_offsets",
            trie_state.student_path_offsets,
            persistent=True,
        )
        self.register_buffer(
            "student_ignored_mask",
            trie_state.student_ignored_mask,
            persistent=True,
        )
        self.register_buffer(
            "student_prefix_bucket_ids",
            trie_state.student_prefix_bucket_ids,
            persistent=True,
        )
        self.register_buffer("teacher_path_flat", trie_state.teacher_path_flat, persistent=True)
        self.register_buffer(
            "teacher_path_offsets",
            trie_state.teacher_path_offsets,
            persistent=True,
        )
        self.register_buffer(
            "teacher_ignored_mask",
            trie_state.teacher_ignored_mask,
            persistent=True,
        )
        self.register_buffer(
            "teacher_prefix_bucket_ids",
            trie_state.teacher_prefix_bucket_ids,
            persistent=True,
        )
        self.register_buffer(
            "prefix_tail_path_flat",
            trie_state.prefix_tail_path_flat,
            persistent=True,
        )
        self.register_buffer(
            "prefix_tail_path_offsets",
            trie_state.prefix_tail_path_offsets,
            persistent=True,
        )

        self.last_prefix_tail_stats: dict[str, PrefixTailStats] = {}

    def extend_vocab_state_with_ignored_tokens(
        self,
        *,
        side: str,
        target_vocab_size: int,
    ) -> None:
        """Extend one trie side with ignored extra tokens."""
        extend_vocab_state_with_ignored_tokens(
            module=self,
            side=side,
            target_vocab_size=target_vocab_size,
        )

    def prepare_runtime_state(
        self,
        *,
        student_vocab_size: int,
        teacher_vocab_size: int,
        teacher_labels: torch.Tensor | None = None,
    ) -> None:
        """Validate or extend trie runtime state for one batch."""
        if student_vocab_size > self.student_vocab_size:
            self.extend_vocab_state_with_ignored_tokens(
                side="student",
                target_vocab_size=student_vocab_size,
            )
        elif student_vocab_size < self.student_vocab_size:
            raise ValueError(
                "student logits vocab size does not match the trie state: "
                f"{student_vocab_size} != {self.student_vocab_size}"
            )

        if teacher_vocab_size > self.teacher_vocab_size:
            self.extend_vocab_state_with_ignored_tokens(
                side="teacher",
                target_vocab_size=teacher_vocab_size,
            )
        elif teacher_vocab_size < self.teacher_vocab_size:
            raise ValueError(
                "teacher logits vocab size does not match the trie state: "
                f"{teacher_vocab_size} != {self.teacher_vocab_size}"
            )

        if teacher_labels is None:
            return

        valid_labels = teacher_labels[teacher_labels != -100]
        if valid_labels.numel() == 0:
            raise ValueError("Teacher labels contain no supervised answer tokens for trie OT.")

        max_label = int(valid_labels.max().item())
        if max_label >= self.teacher_tokenizer_vocab_size:
            raise ValueError(
                "teacher labels contain ids outside tokenizer space: "
                f"max label {max_label} >= tokenizer vocab {self.teacher_tokenizer_vocab_size}"
            )

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        student_temperature: float = 1.0,
        teacher_temperature: float = 1.0,
    ) -> torch.Tensor:
        """Compute mean trie-Wasserstein KD over aligned positions."""
        if student_logits.ndim != 2:
            raise ValueError(
                f"student_logits must have shape (N, V_s), got {tuple(student_logits.shape)}"
            )
        if teacher_logits.ndim != 2:
            raise ValueError(
                f"teacher_logits must have shape (N, V_t), got {tuple(teacher_logits.shape)}"
            )
        if student_logits.size(0) != teacher_logits.size(0):
            raise ValueError(
                "student and teacher must have the same token dimension, got "
                f"{student_logits.size(0)} and {teacher_logits.size(0)}"
            )
        if student_logits.size(0) == 0:
            raise ValueError("Trie Wasserstein loss received no aligned supervised token positions.")
        if student_logits.size(-1) != self.student_vocab_size:
            raise ValueError(
                "student logits vocab size does not match the trie state: "
                f"{student_logits.size(-1)} != {self.student_vocab_size}"
            )
        if teacher_logits.size(-1) != self.teacher_vocab_size:
            raise ValueError(
                "teacher logits vocab size does not match the trie state: "
                f"{teacher_logits.size(-1)} != {self.teacher_vocab_size}"
            )

        # These loss modules live in Python closures, not as trainer/model
        # submodules, so they must move themselves to the logits device.
        self.to(student_logits.device)
        student_scaled_logits = student_logits.float() / float(student_temperature)
        teacher_scaled_logits = teacher_logits.detach().float() / float(teacher_temperature)

        student_result = build_signed_edge_contributions(
            scaled_logits=student_scaled_logits,
            path_flat=self.student_path_flat,
            path_offsets=self.student_path_offsets,
            ignored_mask=self.student_ignored_mask,
            prefix_bucket_ids=self.student_prefix_bucket_ids,
            edge_count=self.num_edges,
            topk=self.topk,
            valid_count=self.student_valid_count,
            prefix_tail_path_flat=self.prefix_tail_path_flat,
            prefix_tail_path_offsets=self.prefix_tail_path_offsets,
            num_prefix_tail_buckets=self.num_prefix_tail_buckets,
            sign=1.0,
        )
        teacher_result = build_signed_edge_contributions(
            scaled_logits=teacher_scaled_logits,
            path_flat=self.teacher_path_flat,
            path_offsets=self.teacher_path_offsets,
            ignored_mask=self.teacher_ignored_mask,
            prefix_bucket_ids=self.teacher_prefix_bucket_ids,
            edge_count=self.num_edges,
            topk=self.topk,
            valid_count=self.teacher_valid_count,
            prefix_tail_path_flat=self.prefix_tail_path_flat,
            prefix_tail_path_offsets=self.prefix_tail_path_offsets,
            num_prefix_tail_buckets=self.num_prefix_tail_buckets,
            sign=-1.0,
        )
        self.last_prefix_tail_stats = {
            "student": student_result.stats,
            "teacher": teacher_result.stats,
        }

        return reduce_signed_edge_contributions_to_tree_loss(
            student_result=student_result,
            teacher_result=teacher_result,
            edge_weights=self.edge_weights,
            edge_count=self.num_edges,
            num_rows=student_logits.size(0),
        )
