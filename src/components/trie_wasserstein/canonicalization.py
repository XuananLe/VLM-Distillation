from __future__ import annotations

import unicodedata

from .types import BoundaryKind, CanonicalTokenPiece


SMOLVLM_MODEL_ID_PREFIX = "huggingfacetb/smolvlm"


def tokenizer_model_id(tokenizer) -> str:
    model_id = getattr(tokenizer, "name_or_path", None)
    return model_id if isinstance(model_id, str) else ""


def tokenizer_class_name(tokenizer) -> str:
    tokenizer_type = type(tokenizer)
    return f"{tokenizer_type.__module__}.{tokenizer_type.__qualname__}"


def tokenizer_backend_model_name(tokenizer) -> str:
    backend = getattr(tokenizer, "backend_tokenizer", None)
    model = getattr(backend, "model", None)
    if model is None:
        return ""
    return f"{type(model).__module__}.{type(model).__qualname__}"


def resolve_underscore_boundary_marker(
    tokenizer,
    explicit_setting: bool | None,
) -> bool:
    if explicit_setting is not None:
        return bool(explicit_setting)

    model_id = tokenizer_model_id(tokenizer).lower()
    class_name = tokenizer_class_name(tokenizer).lower()

    # SmolVLM-500M declares a GPT2Tokenizer with add_prefix_space=false in its
    # tokenizer config. GPT-2 byte-level BPE uses visible space markers such as
    # "Ġ"; a plain underscore remains literal content for code/text tokens.
    if model_id.startswith(SMOLVLM_MODEL_ID_PREFIX) or "gpt2tokenizer" in class_name:
        return False

    return False


def resolve_hash_continuation_marker(tokenizer) -> bool:
    class_name = tokenizer_class_name(tokenizer).lower()
    backend_model_name = tokenizer_backend_model_name(tokenizer).lower()
    return "wordpiece" in class_name or "wordpiece" in backend_model_name


def convert_id_to_token(tokenizer, token_id: int) -> str:
    if hasattr(tokenizer, "convert_ids_to_tokens"):
        token = tokenizer.convert_ids_to_tokens(int(token_id))
        if token is not None:
            return str(token)

    if hasattr(tokenizer, "id_to_token"):
        token = tokenizer.id_to_token(int(token_id))
        if token is not None:
            return str(token)

    return str(
        tokenizer.decode(
            [int(token_id)],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    )


def decode_piece_text(tokenizer, piece_text: str) -> str:
    backend = getattr(tokenizer, "backend_tokenizer", None)
    decoder = getattr(backend, "decoder", None)

    if decoder is not None:
        try:
            return str(decoder.decode([piece_text]))
        except Exception:
            pass

    if hasattr(tokenizer, "convert_tokens_to_string"):
        try:
            return str(tokenizer.convert_tokens_to_string([piece_text]))
        except Exception:
            pass

    return piece_text


def split_marker(
    raw_token: str,
    *,
    underscore_is_boundary_marker: bool,
    hash_is_continuation_marker: bool,
) -> tuple[BoundaryKind, str]:
    if len(raw_token) > 1 and raw_token.startswith("Ġ"):
        return "space", raw_token[1:]

    if len(raw_token) > 1 and raw_token.startswith("▁"):
        return "space", raw_token[1:]

    if (
        underscore_is_boundary_marker
        and len(raw_token) > 1
        and raw_token.startswith("_")
    ):
        return "space", raw_token[1:]

    if (
        hash_is_continuation_marker
        and len(raw_token) > 2
        and raw_token.startswith("##")
    ):
        return "continuation", raw_token[2:]

    if len(raw_token) > 4 and raw_token.endswith("</w>"):
        return "end_word", raw_token[:-4]

    return "none", raw_token


def canonicalize_token_piece(
    tokenizer,
    token_id: int,
    *,
    underscore_is_boundary_marker: bool = False,
    hash_is_continuation_marker: bool | None = None,
    normalize_unicode: bool = True,
) -> CanonicalTokenPiece:
    raw_token = convert_id_to_token(tokenizer, int(token_id))
    boundary_kind, raw_content = split_marker(
        raw_token,
        underscore_is_boundary_marker=underscore_is_boundary_marker,
        hash_is_continuation_marker=(
            resolve_hash_continuation_marker(tokenizer)
            if hash_is_continuation_marker is None
            else bool(hash_is_continuation_marker)
        ),
    )

    decoded = decode_piece_text(tokenizer, raw_content)

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

    if normalize_unicode:
        decoded = unicodedata.normalize("NFC", decoded)

    return CanonicalTokenPiece(
        content_bytes=decoded.encode("utf-8"),
        boundary_kind=boundary_kind,
    )


__all__ = [
    "canonicalize_token_piece",
    "convert_id_to_token",
    "decode_piece_text",
    "resolve_hash_continuation_marker",
    "resolve_underscore_boundary_marker",
    "split_marker",
    "tokenizer_backend_model_name",
    "tokenizer_class_name",
    "tokenizer_model_id",
]
