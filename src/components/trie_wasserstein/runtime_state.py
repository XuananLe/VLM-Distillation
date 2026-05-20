from __future__ import annotations

import torch


def extend_vocab_state_with_unmapped_tokens(
    *,
    module,
    side: str,
    target_vocab_size: int,
) -> None:
    if side == "student":
        current_vocab_size = module.student_vocab_size
        if target_vocab_size <= current_vocab_size:
            return
        token_paths = module.student_token_paths
        ignored_mask = module.student_ignored_mask
        non_text_mask = module.student_non_text_mask
    elif side == "teacher":
        current_vocab_size = module.teacher_vocab_size
        if target_vocab_size <= current_vocab_size:
            return
        token_paths = module.teacher_token_paths
        ignored_mask = module.teacher_ignored_mask
        non_text_mask = module.teacher_non_text_mask
    else:
        raise ValueError(f"side must be 'student' or 'teacher', got {side!r}")

    extra_tokens = target_vocab_size - current_vocab_size

    # Runtime vocab growth usually comes from model-added IDs outside the
    # tokenizer byte vocabulary; keep their mass visible instead of hiding it.
    extended_paths = token_paths + [[] for _ in range(extra_tokens)]
    extended_ignored_mask = torch.cat(
        [
            ignored_mask,
            torch.zeros(extra_tokens, dtype=torch.bool, device=ignored_mask.device),
        ],
        dim=0,
    )
    extended_non_text_mask = torch.cat(
        [
            non_text_mask,
            torch.ones(extra_tokens, dtype=torch.bool, device=non_text_mask.device),
        ],
        dim=0,
    )

    if side == "student":
        module.student_vocab_size = target_vocab_size
        module.student_token_paths = extended_paths
        module.student_ignored_mask = extended_ignored_mask
        module.student_non_text_mask = extended_non_text_mask
    else:
        module.teacher_vocab_size = target_vocab_size
        module.teacher_token_paths = extended_paths
        module.teacher_ignored_mask = extended_ignored_mask
        module.teacher_non_text_mask = extended_non_text_mask
