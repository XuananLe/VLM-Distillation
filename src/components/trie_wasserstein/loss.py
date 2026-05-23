from __future__ import annotations

import torch
from torch import nn

from src.tokenizer_utils import (
    collect_non_text_token_ids,
    default_ignored_token_ids,
    resolve_vocab_size,
)

from .canonicalization import resolve_underscore_boundary_marker
from .contributions import (
    build_signed_edge_contributions,
    reduce_signed_edge_contributions_to_tree_loss,
)
from .runtime_state import extend_vocab_state_with_unmapped_tokens
from .trie_build import build_trie_state_from_tokenizers


class TrieWassersteinLoss(nn.Module):
    def __init__(
        self,
        student_tokenizer,
        teacher_tokenizer,
        rho: float = 0.7,
        topk: int = 64,
        boundary_weight: float = 0.05,
        non_text_token_weight: float = 1.0,
        student_underscore_is_boundary_marker: bool | None = None,
        teacher_underscore_is_boundary_marker: bool | None = None,
        ignored_student_token_ids: set[int] | None = None,
        ignored_teacher_token_ids: set[int] | None = None,
    ) -> None:
        super().__init__()
        if not 0.0 < float(rho) < 1.0:
            raise ValueError(f"rho must be in (0, 1), got {rho}")
        if int(topk) < 1:
            raise ValueError(f"topk must be >= 1, got {topk}")
        if float(boundary_weight) < 0.0:
            raise ValueError(f"boundary_weight must be >= 0, got {boundary_weight}")
        if float(non_text_token_weight) < 0.0:
            raise ValueError(f"non_text_token_weight must be >= 0, got {non_text_token_weight}")

        self.student_tokenizer = getattr(student_tokenizer, "tokenizer", None) or student_tokenizer
        self.teacher_tokenizer = getattr(teacher_tokenizer, "tokenizer", None) or teacher_tokenizer
        self.student_tokenizer_vocab_size = resolve_vocab_size(self.student_tokenizer)
        self.teacher_tokenizer_vocab_size = resolve_vocab_size(self.teacher_tokenizer)
        self.student_vocab_size = self.student_tokenizer_vocab_size
        self.teacher_vocab_size = self.teacher_tokenizer_vocab_size
        self.rho = float(rho)
        self.topk = int(topk)
        self.boundary_weight = float(boundary_weight)
        self.non_text_token_weight = float(non_text_token_weight)
        self.student_underscore_is_boundary_marker = resolve_underscore_boundary_marker(
            self.student_tokenizer,
            student_underscore_is_boundary_marker,
        )
        self.teacher_underscore_is_boundary_marker = resolve_underscore_boundary_marker(
            self.teacher_tokenizer,
            teacher_underscore_is_boundary_marker,
        )

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
        non_text_student = collect_non_text_token_ids(self.student_tokenizer) - ignored_student
        non_text_teacher = collect_non_text_token_ids(self.teacher_tokenizer) - ignored_teacher

        trie_state = build_trie_state_from_tokenizers(
            student_vocab_size=self.student_vocab_size,
            teacher_vocab_size=self.teacher_vocab_size,
            student_ignored_token_ids=tuple(sorted(ignored_student)),
            teacher_ignored_token_ids=tuple(sorted(ignored_teacher)),
            student_non_text_token_ids=tuple(sorted(non_text_student)),
            teacher_non_text_token_ids=tuple(sorted(non_text_teacher)),
            rho=self.rho,
            student_tokenizer=self.student_tokenizer,
            teacher_tokenizer=self.teacher_tokenizer,
            boundary_weight=self.boundary_weight,
            student_underscore_is_boundary_marker=self.student_underscore_is_boundary_marker,
            teacher_underscore_is_boundary_marker=self.teacher_underscore_is_boundary_marker,
        )

        self.tail_edge_id = trie_state.tail_edge_id

        self.register_buffer("edge_weights", trie_state.edge_weights, persistent=True)
        self.student_token_paths = trie_state.student_token_paths
        self.register_buffer(
            "student_ignored_mask",
            trie_state.student_ignored_mask,
            persistent=True,
        )
        self.register_buffer(
            "student_non_text_mask",
            trie_state.student_non_text_mask,
            persistent=True,
        )
        self.teacher_token_paths = trie_state.teacher_token_paths
        self.register_buffer(
            "teacher_ignored_mask",
            trie_state.teacher_ignored_mask,
            persistent=True,
        )
        self.register_buffer(
            "teacher_non_text_mask",
            trie_state.teacher_non_text_mask,
            persistent=True,
        )

    def extend_vocab_state_with_unmapped_tokens(
        self,
        *,
        side: str,
        target_vocab_size: int,
    ) -> None:
        extend_vocab_state_with_unmapped_tokens(
            module=self,
            side=side,
            target_vocab_size=target_vocab_size,
        )

    def prepare_runtime_state(
        self,
        *,
        student_vocab_size: int,
        teacher_vocab_size: int,
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
            self.extend_vocab_state_with_unmapped_tokens(
                side="student",
                target_vocab_size=student_vocab_size,
            )

        if teacher_vocab_size > self.teacher_vocab_size:
            self.extend_vocab_state_with_unmapped_tokens(
                side="teacher",
                target_vocab_size=teacher_vocab_size,
            )

    def forward(
        self,
        student_logits: torch.Tensor, # [N, V_student]
        teacher_logits: torch.Tensor, # [N, V_teacher]
        student_temperature: float = 1.0,
        teacher_temperature: float = 1.0,
    ) -> torch.Tensor:
        if student_temperature <= 0.0 or teacher_temperature <= 0.0:
            raise ValueError("student_temperature and teacher_temperature must be positive")
        if student_logits.device != teacher_logits.device:
            raise ValueError(
                "student_logits and teacher_logits must be on the same device, got "
                f"{student_logits.device} and {teacher_logits.device}"
            )

        invalid_logits = (
            student_logits.ndim != 2
            or teacher_logits.ndim != 2
            or student_logits.size(0) != teacher_logits.size(0)
            or student_logits.size(0) == 0
        )
        if invalid_logits:
            raise ValueError(
                "Error: invalid trie logits. "
                f"student_shape={tuple(student_logits.shape)}, "
                f"teacher_shape={tuple(teacher_logits.shape)}"
            )

        self.prepare_runtime_state(
            student_vocab_size=student_logits.size(-1),
            teacher_vocab_size=teacher_logits.size(-1),
        )

        if student_logits.size(-1) != self.student_vocab_size or teacher_logits.size(-1) != self.teacher_vocab_size:
            raise ValueError(
                "Error: invalid trie vocab sizes. "
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

        student_edge_masses, student_non_text_masses = build_signed_edge_contributions(
            scaled_logits=student_scaled_logits,
            token_paths=self.student_token_paths,
            ignored_mask=self.student_ignored_mask,
            non_text_mask=self.student_non_text_mask,
            tail_edge_id=self.tail_edge_id,
            topk=self.topk,
            sign=1.0,
        )
        teacher_edge_masses, _ = build_signed_edge_contributions(
            scaled_logits=teacher_scaled_logits,
            token_paths=self.teacher_token_paths,
            ignored_mask=self.teacher_ignored_mask,
            non_text_mask=self.teacher_non_text_mask,
            tail_edge_id=self.tail_edge_id,
            topk=self.topk,
            sign=-1.0,
        )
        trie_loss = reduce_signed_edge_contributions_to_tree_loss(
            student_edge_masses=student_edge_masses,
            teacher_edge_masses=teacher_edge_masses,
            edge_weights=self.edge_weights,
        )
        return trie_loss + self.non_text_token_weight * student_non_text_masses.mean()
