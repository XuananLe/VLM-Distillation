import torch


def extend_vocab_state_with_ignored_tokens(
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
    elif side == "teacher":
        current_vocab_size = module.teacher_vocab_size
        if target_vocab_size <= current_vocab_size:
            return
        token_paths = module.teacher_token_paths
        ignored_mask = module.teacher_ignored_mask
    else:
        raise ValueError(f"Unknown trie side: {side!r}")

    extra_tokens = target_vocab_size - current_vocab_size

    # Runtime vocab growth usually comes from model-added tokens outside the
    # tokenizer's byte vocabulary; ignoring them is safer than inventing paths.
    extended_paths = token_paths + [[] for _ in range(extra_tokens)]
    extended_ignored_mask = torch.cat(
        [
            ignored_mask,
            torch.ones(extra_tokens, dtype=torch.bool, device=ignored_mask.device),
        ],
        dim=0,
    )

    if side == "student":
        module.student_vocab_size = target_vocab_size
        module.student_token_paths = extended_paths
        module.student_ignored_mask = extended_ignored_mask
    else:
        module.teacher_vocab_size = target_vocab_size
        module.teacher_token_paths = extended_paths
        module.teacher_ignored_mask = extended_ignored_mask
