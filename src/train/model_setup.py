import importlib

import torch
from PIL import Image, ImageFile
from transformers import (
    AutoConfig,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
    Gemma3ForConditionalGeneration,
)

from src.dataset.internvl_utils import (
    INTERNVL_IMAGE_SIZE,
    INTERNVL_IMG_CONTEXT_TOKEN,
    INTERNVL_MAX_NUM_TILES,
    INTERNVL_NUM_IMAGE_TOKEN,
)
from src.train.internvl_compat import load_internvl_model

importlib.import_module("pillow_avif")
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

SMOLVLM_MODEL_TYPES = {"smolvlm", "smolvlm2", "idefics3"}
SUPPORTED_AUTO_MODEL_TYPES = {
    *SMOLVLM_MODEL_TYPES,
    "qwen2_vl",
    "qwen2_5_vl",
    "gemma3",
}


def resolve_model_type(model_id: str) -> str | None:
    # https://huggingface.co/docs/transformers/v5.1.0/en/model_doc/auto

    config = AutoConfig.from_pretrained(
        model_id,
        trust_remote_code=True,
    )
    return getattr(config, "model_type", None)


def load_processor_bundle(
    model_id: str,
    *,
    padding_side: str | None = None,
    cache_dir: str | None = None,
):
    """Load the processor and tokenizer interface for one supported multimodal model."""
    if "internvl" in model_id.lower():
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            padding_side=padding_side or "right",
            trust_remote_code=True,
            use_fast=True,
        )
        processor = {
            "model_id": model_id,
            "tokenizer": tokenizer,
            "image_size": INTERNVL_IMAGE_SIZE,
            "normalize_type": "imagenet",
            "max_num_tiles": INTERNVL_MAX_NUM_TILES,
            "num_image_token": INTERNVL_NUM_IMAGE_TOKEN,
        }
        return processor, tokenizer, "internvl"

    model_type = resolve_model_type(model_id)
    if model_type not in SUPPORTED_AUTO_MODEL_TYPES:
        raise ValueError("We only support SmolVLM, Qwen-VL, Gemma 3, and InternVL.")
    processor_kwargs = {
        "trust_remote_code": True,
        "use_fast": False,
    }
    if padding_side is not None:
        processor_kwargs["padding_side"] = padding_side
    if cache_dir is not None:
        processor_kwargs["cache_dir"] = cache_dir

    processor = AutoProcessor.from_pretrained(
        model_id,
        **processor_kwargs,
    )
    if hasattr(processor, "image_processor"):
        processor.image_processor.image_format = "AVIF"
        processor.image_processor.do_convert_rgb = True
    tokenizer = getattr(processor, "tokenizer", None)
    return processor, (tokenizer if tokenizer is not None else processor), model_type


def load_vlm_components(
    *,
    model_id: str,
    cache_dir: str | None,
    device,
    compute_dtype: torch.dtype,
    disable_flash_attn2: bool,
    padding_side: str = "right",
    model_kwargs: dict | None = None,
    attn_implementation: str | None = None,
):
    resolved_attn_implementation = attn_implementation or ("flash_attention_2" if not disable_flash_attn2 else "eager")
    if "internvl" in model_id.lower():
        model_type = "internvl"
        model = load_internvl_model(
            model_id=model_id,
            cache_dir=cache_dir,
            device=device,
            compute_dtype=compute_dtype,
            use_flash_attn=resolved_attn_implementation == "flash_attention_2",
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            padding_side=padding_side,
            trust_remote_code=True,
            use_fast=True,
        )
        model.img_context_token_id = tokenizer.convert_tokens_to_ids(INTERNVL_IMG_CONTEXT_TOKEN)
        vision_config = getattr(model.config, "vision_config", None)
        processor = {
            "model_id": model_id,
            "tokenizer": tokenizer,
            "image_size": getattr(model.config, "force_image_size", None)
            or getattr(vision_config, "image_size", INTERNVL_IMAGE_SIZE),
            "normalize_type": (
                "siglip" if getattr(vision_config, "model_type", None) == "siglip_vision_model" else "imagenet"
            ),
            "max_num_tiles": INTERNVL_MAX_NUM_TILES,
            "num_image_token": getattr(model, "num_image_token", INTERNVL_NUM_IMAGE_TOKEN),
        }
    else:
        processor, tokenizer, model_type = load_processor_bundle(
            model_id,
            padding_side=padding_side,
            cache_dir=cache_dir,
        )
        loader_kwargs = {"device_map": {"": device}}
        loader_kwargs.update(model_kwargs or {})
        if model_type == "gemma3":
            loader_cls = Gemma3ForConditionalGeneration
        else:
            loader_cls = AutoModelForImageTextToText
        model = loader_cls.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            attn_implementation=resolved_attn_implementation,
            dtype=compute_dtype,
            trust_remote_code=True,
            **loader_kwargs,
        )

    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    return model, processor, tokenizer, model_type


__all__ = [
    "load_processor_bundle",
    "load_vlm_components",
    "resolve_model_type",
]
