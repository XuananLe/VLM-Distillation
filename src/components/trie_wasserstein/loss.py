from __future__ import annotations

import torch
from torch import nn

from src.tokenizer_utils import (
    collect_non_text_token_ids,
    default_ignored_token_ids,
    resolve_vocab_size,
)

from .contributions import (
    build_signed_edge_contributions,
    reduce_signed_edge_contributions_to_tree_loss,
)
from .trie_build import build_trie_state_from_tokenizers


class TrieWassersteinLoss(nn.Module):
    def __init__(
        self,
        student_tokenizer,
        teacher_tokenizer,
        rho: float = 0.7,
        topk: int = 64,
    ) -> None:
        super().__init__()
        if not 0.0 < float(rho) < 1.0:
            raise ValueError(f"rho must be in (0, 1), got {rho}")
        if int(topk) < 1:
            raise ValueError(f"topk must be >= 1, got {topk}")

        student_tokenizer = getattr(student_tokenizer, "tokenizer", None) or student_tokenizer
        teacher_tokenizer = getattr(teacher_tokenizer, "tokenizer", None) or teacher_tokenizer
        self.student_vocab_size = resolve_vocab_size(student_tokenizer)
        self.teacher_vocab_size = resolve_vocab_size(teacher_tokenizer)
        self.topk = int(topk)

        student_ignored_ids = default_ignored_token_ids(student_tokenizer)
        teacher_ignored_ids = default_ignored_token_ids(teacher_tokenizer)

        trie_state = build_trie_state_from_tokenizers(
            student_vocab_size=self.student_vocab_size,
            teacher_vocab_size=self.teacher_vocab_size,
            student_ignored_token_ids=student_ignored_ids,
            teacher_ignored_token_ids=teacher_ignored_ids,
            student_non_text_token_ids=collect_non_text_token_ids(student_tokenizer) - student_ignored_ids,
            teacher_non_text_token_ids=collect_non_text_token_ids(teacher_tokenizer) - teacher_ignored_ids,
            rho=float(rho),
            student_tokenizer=student_tokenizer,
            teacher_tokenizer=teacher_tokenizer,
        )

        self.tail_edge_id = trie_state.tail_edge_id

        for name in (
            "edge_weights",
            "student_ignored_mask",
            "student_non_text_mask",
            "teacher_ignored_mask",
            "teacher_non_text_mask",
        ):
            self.register_buffer(name, getattr(trie_state, name), persistent=True)
        self.student_token_paths = trie_state.student_token_paths
        self.teacher_token_paths = trie_state.teacher_token_paths

    @staticmethod
    def extend_mask(mask: torch.Tensor, extra_tokens: int, value: bool) -> torch.Tensor:
        return torch.cat(
            [
                mask,
                torch.full((extra_tokens,), value, dtype=torch.bool, device=mask.device),
            ],
            dim=0,
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
            extra_tokens = student_vocab_size - self.student_vocab_size
            self.student_token_paths = self.student_token_paths + [[] for _ in range(extra_tokens)]
            self.student_ignored_mask = self.extend_mask(self.student_ignored_mask, extra_tokens, False)
            self.student_non_text_mask = self.extend_mask(self.student_non_text_mask, extra_tokens, True)
            self.student_vocab_size = student_vocab_size

        if teacher_vocab_size > self.teacher_vocab_size:
            extra_tokens = teacher_vocab_size - self.teacher_vocab_size
            self.teacher_token_paths = self.teacher_token_paths + [[] for _ in range(extra_tokens)]
            self.teacher_ignored_mask = self.extend_mask(self.teacher_ignored_mask, extra_tokens, False)
            self.teacher_non_text_mask = self.extend_mask(self.teacher_non_text_mask, extra_tokens, True)
            self.teacher_vocab_size = teacher_vocab_size

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        student_temperature: float = 1.0,
        teacher_temperature: float = 1.0,
    ) -> torch.Tensor:
        self.prepare_runtime_state(
            student_vocab_size=student_logits.size(-1),
            teacher_vocab_size=teacher_logits.size(-1),
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
        return trie_loss + student_non_text_masses.mean()
