from __future__ import annotations

import unicodedata

from .types import BoundaryKind


def canonicalize_token_piece(
    tokenizer,
    token_id: int,
) -> tuple[bytes, BoundaryKind]:
    tokenizer_type = type(tokenizer)
    class_name = f"{tokenizer_type.__module__}.{tokenizer_type.__qualname__}".lower()
    backend = getattr(tokenizer, "backend_tokenizer", None)
    backend_model = getattr(backend, "model", None)
    backend_model_name = (
        "" if backend_model is None else f"{type(backend_model).__module__}.{type(backend_model).__qualname__}".lower()
    )
    hash_is_continuation_marker = "wordpiece" in class_name or "wordpiece" in backend_model_name

    if hasattr(tokenizer, "convert_ids_to_tokens"):
        token = tokenizer.convert_ids_to_tokens(int(token_id))
        raw_token = str(token) if token is not None else None
    else:
        raw_token = None
    if raw_token is None and hasattr(tokenizer, "id_to_token"):
        token = tokenizer.id_to_token(int(token_id))
        raw_token = str(token) if token is not None else None
    if raw_token is None:
        raw_token = str(
            tokenizer.decode(
                [int(token_id)],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        )

    boundary_kind: BoundaryKind
    if len(raw_token) > 1 and raw_token.startswith("Ġ"):
        boundary_kind, raw_content = "space", raw_token[1:]
    elif len(raw_token) > 1 and raw_token.startswith("▁"):
        boundary_kind, raw_content = "space", raw_token[1:]
    elif hash_is_continuation_marker and len(raw_token) > 2 and raw_token.startswith("##"):
        boundary_kind, raw_content = "continuation", raw_token[2:]
    elif len(raw_token) > 4 and raw_token.endswith("</w>"):
        boundary_kind, raw_content = "end_word", raw_token[:-4]
    else:
        boundary_kind, raw_content = "none", raw_token

    backend = getattr(tokenizer, "backend_tokenizer", None)
    decoder = getattr(backend, "decoder", None)
    if decoder is not None:
        try:
            decoded = str(decoder.decode([raw_content]))
        except (ValueError, TypeError, AttributeError, UnicodeDecodeError):
            decoded = None
    else:
        decoded = None
    if decoded is None and hasattr(tokenizer, "convert_tokens_to_string"):
        try:
            decoded = str(tokenizer.convert_tokens_to_string([raw_content]))
        except (ValueError, TypeError, AttributeError, UnicodeDecodeError):
            decoded = None
    if decoded is None:
        decoded = raw_content

    # Some decoders expose leading whitespace directly instead of a visible
    # marker; moving that signal to metadata keeps content edges comparable.
    if boundary_kind == "none":
        stripped = decoded.lstrip(" \t\r\n")
        if stripped and stripped != decoded:
            boundary_kind = "space"
            decoded = stripped
    else:
        stripped = decoded.lstrip(" \t\r\n")
        if stripped:
            decoded = stripped

    # Marker-only pieces should remain real pieces; otherwise they would vanish
    # from the trie and incorrectly become indistinguishable from empty content.
    if decoded == "":
        decoded = raw_content if raw_content else raw_token

    return unicodedata.normalize("NFC", decoded).encode("utf-8"), boundary_kind


__all__ = [
    "canonicalize_token_piece",
]
