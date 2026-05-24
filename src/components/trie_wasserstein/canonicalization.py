from __future__ import annotations

import unicodedata

from .types import BoundaryKind


def canonicalize_token_piece(
    tokenizer,
    token_id: int,
) -> tuple[bytes, BoundaryKind]:
    raw_token = str(tokenizer.convert_ids_to_tokens(int(token_id)))

    if len(raw_token) > 1 and raw_token[0] in {"Ġ", "▁"}:
        boundary_kind, raw_content = "space", raw_token[1:]
    else:
        boundary_kind, raw_content = "none", raw_token

    decoder = getattr(getattr(tokenizer, "backend_tokenizer", None), "decoder", None)
    if decoder is not None:
        decoded = str(decoder.decode([raw_content]))
    elif hasattr(tokenizer, "convert_tokens_to_string"):
        decoded = str(tokenizer.convert_tokens_to_string([raw_content]))
    else:
        decoded = raw_content

    stripped = decoded.lstrip(" \t\r\n")
    if stripped:
        if boundary_kind == "none" and stripped != decoded:
            boundary_kind = "space"
        decoded = stripped

    if decoded == "":
        decoded = raw_content if raw_content else raw_token

    return unicodedata.normalize("NFC", decoded).encode("utf-8"), boundary_kind


__all__ = [
    "canonicalize_token_piece",
]
