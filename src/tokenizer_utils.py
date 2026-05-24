from __future__ import annotations

import re
from collections.abc import Iterable

NON_TEXT_TOKEN_RE = re.compile(
    r"image|vision|patch|pixel|img|(^|[^a-z0-9])(?:boi|eoi|vis|quad|box|ref)([^a-z0-9]|$)"
)
NON_TEXT_TOKEN_ATTR_NAMES = (
    "image",
    "global_image",
    "fake_image",
    "boi",
    "eoi",
    "img_context",
    "img_start",
    "img_end",
    "vision_start",
    "vision_end",
    "vision_pad",
    "image_pad",
)
NON_TEXT_TOKEN_ID_ATTR_NAMES = NON_TEXT_TOKEN_ATTR_NAMES + ("image_start", "image_end")

# Sources:
# https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct/blob/main/tokenizer_config.json
# https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct/blob/main/tokenizer_config.json
SMOLVLM_NON_TEXT_TOKEN_STRINGS = (
    "<image>",
    "<fake_token_around_image>",
    "<global-img>",
    *(f"<row_{row}_col_{col}>" for row in range(1, 7) for col in range(1, 7)),
)

# Sources:
# https://huggingface.co/docs/transformers/v4.52.2/en/model_doc/gemma3
# https://huggingface.co/google/gemma-3-4b-it/blob/main/tokenizer_config.json
GEMMA_NON_TEXT_TOKEN_STRINGS = (
    "<start_of_image>",
    "<end_of_image>",
    "<image_soft_token>",
)

# Sources:
# https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/blob/main/tokenizer_config.json
# https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct/blob/main/tokenizer_config.json
QWEN_VL_NON_TEXT_TOKEN_STRINGS = (
    "<|vision_start|>",
    "<|vision_end|>",
    "<|vision_pad|>",
    "<|image_pad|>",
    "<|box_start|>",
    "<|box_end|>",
    "<|quad_start|>",
    "<|quad_end|>",
    "<|object_ref_start|>",
    "<|object_ref_end|>",
)

# Source:
# https://huggingface.co/OpenGVLab/InternVL2-1B/blob/main/tokenizer_config.json
INTERNVL_NON_TEXT_TOKEN_STRINGS = (
    "<img>",
    "</img>",
    "<IMG_CONTEXT>",
    "<quad>",
    "</quad>",
    "<ref>",
    "</ref>",
    "<box>",
    "</box>",
)


SUPPORTED_MODEL_NON_TEXT_TOKEN_STRINGS = {
    "HuggingFaceTB/SmolVLM-256M-Instruct": SMOLVLM_NON_TEXT_TOKEN_STRINGS,
    "HuggingFaceTB/SmolVLM-500M-Instruct": SMOLVLM_NON_TEXT_TOKEN_STRINGS,
    "HuggingFaceTB/SmolVLM-Instruct": SMOLVLM_NON_TEXT_TOKEN_STRINGS,
    "HuggingFaceTB/SmolVLM2-2.2B-Instruct": SMOLVLM_NON_TEXT_TOKEN_STRINGS,
    "OpenGVLab/InternVL2-1B": INTERNVL_NON_TEXT_TOKEN_STRINGS,
    "Qwen/Qwen2.5-VL-3B-Instruct": QWEN_VL_NON_TEXT_TOKEN_STRINGS,
    "Qwen/Qwen2-VL-2B-Instruct": QWEN_VL_NON_TEXT_TOKEN_STRINGS,
    "google/gemma-3-4b-it": GEMMA_NON_TEXT_TOKEN_STRINGS,
}

VLM_NON_TEXT_TOKEN_STRINGS = tuple(
    dict.fromkeys(token for tokens in SUPPORTED_MODEL_NON_TEXT_TOKEN_STRINGS.values() for token in tokens)
)


VLM_CONTROL_SPECIAL_TOKEN_STRINGS = ("<end_of_utterance>",)


def token_string_looks_non_text(token: str) -> bool:
    return bool(NON_TEXT_TOKEN_RE.search(token.lower()))


def token_strings_to_ids(tokenizer, tokens: Iterable[str]) -> set[int]:
    if not hasattr(tokenizer, "convert_tokens_to_ids"):
        return set()

    token_list = list(tokens)
    try:
        converted_ids = tokenizer.convert_tokens_to_ids(token_list)
    except (TypeError, ValueError):
        converted_ids = None
    if converted_ids is None:
        converted_ids = [tokenizer.convert_tokens_to_ids(token) for token in token_list]
    elif isinstance(converted_ids, int):
        converted_ids = [converted_ids]

    token_ids: set[int] = set()
    unknown_token_id = getattr(tokenizer, "unk_token_id", None)
    for token_id in converted_ids:
        if token_id is None or token_id == unknown_token_id:
            continue
        token_ids.add(int(token_id))
    return token_ids


def iter_token_values(value):
    if value is None:
        return
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        yield from value
    else:
        yield value


def add_non_text_tokens(tokens: set[str], candidates) -> None:
    for token in candidates:
        token_text = str(token)
        if token_string_looks_non_text(token_text):
            tokens.add(token_text)


def collect_non_text_token_ids(tokenizer) -> set[int]:
    tokens = set(VLM_NON_TEXT_TOKEN_STRINGS)

    special_tokens_map = getattr(tokenizer, "special_tokens_map_extended", None)
    if not special_tokens_map:
        special_tokens_map = getattr(tokenizer, "special_tokens_map", {}) or {}
    for token_value in special_tokens_map.values():
        add_non_text_tokens(tokens, iter_token_values(token_value))

    add_non_text_tokens(tokens, getattr(tokenizer, "additional_special_tokens", ()) or ())

    token_ids: set[int] = set()
    added_vocab = getattr(tokenizer, "added_tokens_encoder", None)
    if hasattr(tokenizer, "get_added_vocab"):
        added_vocab = added_vocab or tokenizer.get_added_vocab()
    for token, token_id in (added_vocab or {}).items():
        if token_string_looks_non_text(str(token)):
            token_ids.add(int(token_id))

    # Visual placeholders carry model-side image semantics, not text boundary
    # semantics, so trie OT should mask them instead of inventing boundary kinds.
    for name in NON_TEXT_TOKEN_ATTR_NAMES:
        attr_name = f"{name}_token"
        attr_value = getattr(tokenizer, attr_name, None)
        if isinstance(attr_value, str):
            tokens.add(attr_value)

    token_ids.update(token_strings_to_ids(tokenizer, tokens))
    for name in NON_TEXT_TOKEN_ID_ATTR_NAMES:
        attr_name = f"{name}_token_id"
        token_id = getattr(tokenizer, attr_name, None)
        if token_id is not None:
            token_ids.add(int(token_id))
    return token_ids


def resolve_vocab_size(tokenizer) -> int:
    try:
        return int(len(tokenizer))
    except (AttributeError, NotImplementedError, TypeError):
        pass

    vocab_size = getattr(tokenizer, "vocab_size", None)
    if vocab_size is not None:
        return int(vocab_size)
    raise ValueError(f"Tokenizer {tokenizer!r} does not expose len(tokenizer) or vocab_size.")


def default_ignored_token_ids(tokenizer) -> set[int]:
    ignored = set(getattr(tokenizer, "all_special_ids", []) or [])
    ignored.update(token_strings_to_ids(tokenizer, VLM_CONTROL_SPECIAL_TOKEN_STRINGS))
    ignored.difference_update(collect_non_text_token_ids(tokenizer))
    return ignored
