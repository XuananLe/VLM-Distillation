from __future__ import annotations

from src.constants import TOKENIZER_VOCAB_SIZES


SMOLVLM_IMAGE_SPECIAL_TOKENS = (
    "<image>",
    "<fake_token_around_image>",
    "<global-img>",
    *(f"<row_{row}_col_{col}>" for row in range(1, 7) for col in range(1, 7)),
)


def normalize_model_id(model_id) -> str | None:
    """Return a canonical model id string when the tokenizer exposes one."""
    if not isinstance(model_id, str) or not model_id:
        return None
    return model_id.rstrip("/")


def resolve_vocab_size(tokenizer) -> int:
    model_id = normalize_model_id(getattr(tokenizer, "name_or_path", None))
    try:
        return TOKENIZER_VOCAB_SIZES[model_id]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported tokenizer for trie OT: {model_id!r}. "
            f"Supported models: {sorted(TOKENIZER_VOCAB_SIZES)}"
        ) from exc


def token_piece_to_bytes(tokenizer, token_id: int) -> bytes:
    token_id = int(token_id)
    if hasattr(tokenizer, "decode"):
        try:
            text = tokenizer.decode(
                [token_id],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        except TypeError:
            text = tokenizer.decode([token_id])
    else:
        token = tokenizer.convert_ids_to_tokens(token_id)
        if hasattr(tokenizer, "convert_tokens_to_string"):
            text = tokenizer.convert_tokens_to_string([token])
        else:
            text = token

    if text is None:
        text = ""
    if not isinstance(text, str):
        text = str(text)
    return text.encode("utf-8")


def default_ignored_token_ids(tokenizer) -> set[int]:
    """Return special token ids ignored by token-distribution losses."""
    ignored = set(getattr(tokenizer, "all_special_ids", []) or [])
    for attr_name in (
        "pad_token_id",
        "bos_token_id",
        "eos_token_id",
        "unk_token_id",
    ):
        token_id = getattr(tokenizer, attr_name, None)
        if token_id is not None:
            ignored.add(int(token_id))

    unknown_token_id = getattr(tokenizer, "unk_token_id", None)
    for token in SMOLVLM_IMAGE_SPECIAL_TOKENS:
        if not hasattr(tokenizer, "convert_tokens_to_ids"):
            break
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is None or token_id == unknown_token_id:
            continue
        ignored.add(int(token_id))
    return ignored
