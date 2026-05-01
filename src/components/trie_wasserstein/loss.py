from __future__ import annotations

import torch
from torch import nn

from .contributions import (
    build_signed_edge_contributions,
    reduce_signed_edge_contributions_to_tree_loss,
)
from .runtime_state import extend_vocab_state_with_ignored_tokens
from .trie_build import build_trie_state_from_tokenizers
from src.tokenizer_utils import (
    default_ignored_token_ids,
    resolve_vocab_size,
)


class TrieWassersteinLoss(nn.Module):
    def __init__(
        self,
        student_tokenizer,
        teacher_tokenizer,
        rho: float = 0.7,
        topk: int = 64,
        ignored_student_token_ids: set[int] | None = None,
        ignored_teacher_token_ids: set[int] | None = None,
    ) -> None:
        super().__init__()
        if not 0.0 < float(rho) < 1.0:
            raise ValueError(f"rho must be in (0, 1), got {rho}")
        if int(topk) < 1:
            raise ValueError(f"topk must be >= 1, got {topk}")

        self.student_tokenizer = getattr(student_tokenizer, "tokenizer", None) or student_tokenizer
        self.teacher_tokenizer = getattr(teacher_tokenizer, "tokenizer", None) or teacher_tokenizer
        self.student_tokenizer_vocab_size = resolve_vocab_size(self.student_tokenizer)
        self.teacher_tokenizer_vocab_size = resolve_vocab_size(self.teacher_tokenizer)
        self.student_vocab_size = self.student_tokenizer_vocab_size
        self.teacher_vocab_size = self.teacher_tokenizer_vocab_size
        self.rho = float(rho)
        self.topk = int(topk)

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
            student_tokenizer=self.student_tokenizer,
            teacher_tokenizer=self.teacher_tokenizer,
        )

        self.tail_edge_id = trie_state.tail_edge_id

        self.register_buffer("edge_weights", trie_state.edge_weights, persistent=True)
        self.student_token_paths = trie_state.student_token_paths
        self.register_buffer(
            "student_ignored_mask",
            trie_state.student_ignored_mask,
            persistent=True,
        )
        self.teacher_token_paths = trie_state.teacher_token_paths
        self.register_buffer(
            "teacher_ignored_mask",
            trie_state.teacher_ignored_mask,
            persistent=True,
        )

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
        if student_vocab_size < self.student_vocab_size or teacher_vocab_size < self.teacher_vocab_size:
            raise ValueError(
                "Error: invalid trie vocab sizes. "
                f"student_vocab={student_vocab_size}, "
                f"teacher_vocab={teacher_vocab_size}, "
                f"expected_student_vocab_at_least={self.student_vocab_size}, "
                f"expected_teacher_vocab_at_least={self.teacher_vocab_size}"
            )

        if student_vocab_size > self.student_vocab_size:
            self.extend_vocab_state_with_ignored_tokens(
                side="student",
                target_vocab_size=student_vocab_size,
            )

        if teacher_vocab_size > self.teacher_vocab_size:
            self.extend_vocab_state_with_ignored_tokens(
                side="teacher",
                target_vocab_size=teacher_vocab_size,
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
        invalid_logits = (
            student_logits.ndim != 2
            or teacher_logits.ndim != 2
            or student_logits.size(0) != teacher_logits.size(0)
            or student_logits.size(0) == 0
            or student_logits.size(-1) != self.student_vocab_size
            or teacher_logits.size(-1) != self.teacher_vocab_size
        )
        if invalid_logits:
            raise ValueError(
                "Error: invalid trie"
                f"student_shape={tuple(student_logits.shape)}, "
                f"teacher_shape={tuple(teacher_logits.shape)}, "
                f"expected_student_vocab={self.student_vocab_size}, "
                f"expected_teacher_vocab={self.teacher_vocab_size}"
            )

        # These loss modules live in Python closures, not as trainer/model
        # submodules, so they must move themselves to the logits device.
        self.to(student_logits.device)
        student_scaled_logits = student_logits.float() / float(student_temperature)
        teacher_scaled_logits = teacher_logits.detach().float() / float(teacher_temperature)

        student_edge_masses = build_signed_edge_contributions(
            scaled_logits=student_scaled_logits,
            token_paths=self.student_token_paths,
            ignored_mask=self.student_ignored_mask,
            tail_edge_id=self.tail_edge_id,
            topk=self.topk,
            sign=1.0,
        )
        teacher_edge_masses = build_signed_edge_contributions(
            scaled_logits=teacher_scaled_logits,
            token_paths=self.teacher_token_paths,
            ignored_mask=self.teacher_ignored_mask,
            tail_edge_id=self.tail_edge_id,
            topk=self.topk,
            sign=-1.0,
        )
        return reduce_signed_edge_contributions_to_tree_loss(
            student_edge_masses=student_edge_masses,
            teacher_edge_masses=teacher_edge_masses,
            edge_weights=self.edge_weights,
        )
