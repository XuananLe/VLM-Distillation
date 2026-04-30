import torch


def extend_vocab_state_with_ignored_tokens(
    *,
    module,
    side: str,
    target_vocab_size: int,
) -> None:
    """Extend one trie side with ignored extra tokens without rebuilding the trie."""
    if side == "student":
        current_vocab_size = module.student_vocab_size
        if target_vocab_size <= current_vocab_size:
            return
        path_flat = module.student_path_flat
        path_offsets = module.student_path_offsets
        ignored_mask = module.student_ignored_mask
    elif side == "teacher":
        current_vocab_size = module.teacher_vocab_size
        if target_vocab_size <= current_vocab_size:
            return
        path_flat = module.teacher_path_flat
        path_offsets = module.teacher_path_offsets
        ignored_mask = module.teacher_ignored_mask
    else:
        raise ValueError(f"Unknown trie side: {side!r}")

    extra_tokens = target_vocab_size - current_vocab_size

    # Runtime vocab growth usually comes from model-added tokens outside the
    # tokenizer's byte vocabulary; ignoring them is safer than inventing paths.
    repeated_offset = path_offsets[-1].repeat(extra_tokens)
    extended_offsets = torch.cat([path_offsets[:-1], repeated_offset, path_offsets[-1:]], dim=0)
    extended_ignored_mask = torch.cat(
        [ignored_mask, torch.ones(extra_tokens, dtype=torch.bool)],
        dim=0,
    )

    if side == "student":
        module.student_vocab_size = target_vocab_size
        module.student_path_flat = path_flat
        module.student_path_offsets = extended_offsets
        module.student_ignored_mask = extended_ignored_mask
    else:
        module.teacher_vocab_size = target_vocab_size
        module.teacher_path_flat = path_flat
        module.teacher_path_offsets = extended_offsets
        module.teacher_ignored_mask = extended_ignored_mask
