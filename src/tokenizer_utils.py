from __future__ import annotations

import re

from src.constants import TOKENIZER_VOCAB_SIZES

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
    "<|video_pad|>",
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
    normalized_token = token.lower()
    if any(marker in normalized_token for marker in ("image", "vision", "patch", "pixel", "img", "video")):
        return True
    return any(
        re.search(rf"(^|[^a-z0-9]){re.escape(marker)}([^a-z0-9]|$)", normalized_token)
        for marker in ("boi", "eoi", "vis", "quad", "box", "ref")
    )


def collect_non_text_token_strings(tokenizer) -> set[str]:
    tokens = set(VLM_NON_TEXT_TOKEN_STRINGS)
    model_id = getattr(tokenizer, "name_or_path", None)
    if isinstance(model_id, str):
        tokens.update(SUPPORTED_MODEL_NON_TEXT_TOKEN_STRINGS.get(model_id.rstrip("/"), ()))

    for token in getattr(tokenizer, "additional_special_tokens", []) or []:
        token_text = str(token)
        if token_string_looks_non_text(token_text):
            tokens.add(token_text)

    special_tokens_map = getattr(tokenizer, "special_tokens_map", {}) or {}
    for token_value in special_tokens_map.values():
        if isinstance(token_value, (list, tuple)):
            candidate_tokens = token_value
        else:
            candidate_tokens = (token_value,)
        for token in candidate_tokens:
            token_text = str(token)
            if token_string_looks_non_text(token_text):
                tokens.add(token_text)

    model_specific_tokens = getattr(tokenizer, "model_specific_special_tokens", {}) or {}
    for token in model_specific_tokens.values():
        token_text = str(token)
        if token_string_looks_non_text(token_text):
            tokens.add(token_text)

    if hasattr(tokenizer, "get_added_vocab"):
        for token in tokenizer.get_added_vocab():
            token_text = str(token)
            if token_string_looks_non_text(token_text):
                tokens.add(token_text)

    # Visual placeholders carry model-side image semantics, not text boundary
    # semantics, so trie OT should mask them instead of inventing boundary kinds.
    for attr_name in (
        "image_token",
        "global_image_token",
        "fake_image_token",
        "boi_token",
        "eoi_token",
        "img_context_token",
        "img_start_token",
        "img_end_token",
        "vision_start_token",
        "vision_end_token",
        "vision_pad_token",
        "image_pad_token",
        "video_pad_token",
    ):
        attr_value = getattr(tokenizer, attr_name, None)
        if isinstance(attr_value, str):
            tokens.add(attr_value)

    return tokens


def collect_control_token_strings(tokenizer) -> set[str]:
    del tokenizer
    return set(VLM_CONTROL_SPECIAL_TOKEN_STRINGS)


def token_strings_to_ids(tokenizer, tokens: set[str]) -> set[int]:
    token_ids: set[int] = set()
    unknown_token_id = getattr(tokenizer, "unk_token_id", None)
    for token in tokens:
        if not hasattr(tokenizer, "convert_tokens_to_ids"):
            break
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is None or token_id == unknown_token_id:
            continue
        token_ids.add(int(token_id))
    return token_ids


def collect_non_text_token_ids(tokenizer) -> set[int]:
    token_ids = token_strings_to_ids(tokenizer, collect_non_text_token_strings(tokenizer))
    for attr_name in (
        "image_token_id",
        "global_image_token_id",
        "fake_image_token_id",
        "boi_token_id",
        "eoi_token_id",
        "img_context_token_id",
        "img_start_token_id",
        "img_end_token_id",
        "image_start_token_id",
        "image_end_token_id",
        "vision_start_token_id",
        "vision_end_token_id",
        "vision_pad_token_id",
        "image_pad_token_id",
        "video_token_id",
        "video_pad_token_id",
    ):
        token_id = getattr(tokenizer, attr_name, None)
        if token_id is not None:
            token_ids.add(int(token_id))
    return token_ids


def resolve_vocab_size(tokenizer) -> int:
    model_id = getattr(tokenizer, "name_or_path", None)
    model_id = model_id.rstrip("/") if isinstance(model_id, str) and model_id else None
    try:
        return TOKENIZER_VOCAB_SIZES[model_id]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported tokenizer for trie OT: {model_id!r}. Supported models: {sorted(TOKENIZER_VOCAB_SIZES)}"
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

    ignored.update(token_strings_to_ids(tokenizer, collect_control_token_strings(tokenizer)))
    ignored.difference_update(collect_non_text_token_ids(tokenizer))
    return ignored
