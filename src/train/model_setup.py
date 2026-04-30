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
    "qwen3_vl",
    "gemma3",
    "llava_next",
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
    model_type = resolve_model_type(model_id)
    if model_type not in SUPPORTED_AUTO_MODEL_TYPES:
        raise ValueError(
            "We only support generic families are SmolVLM, Qwen-VL, Gemma 3, and Granite Vision."
        )
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


def load_model(
    *,
    model_id: str,
    model_type: str | None,
    cache_dir: str | None,
    attn_implementation: str,
    compute_dtype: torch.dtype,
    trust_remote_code: bool = True,
    model_kwargs: dict | None = None,
):
    if model_type not in SUPPORTED_AUTO_MODEL_TYPES:
        raise ValueError(
            "Unsupported model family for the generic model loader: "
            f"model_id={model_id!r}, model_type={model_type!r}. "
            "Supported generic families are SmolVLM, Qwen-VL, Gemma 3, and Granite Vision."
        )

    loader_kwargs = dict(model_kwargs or {})
    if model_type == "gemma3":
        loader_cls = Gemma3ForConditionalGeneration
    else:
        loader_cls = AutoModelForImageTextToText
    return loader_cls.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        attn_implementation=attn_implementation,
        dtype=compute_dtype,
        trust_remote_code=trust_remote_code,
        **loader_kwargs,
    )


def load_vlm_bundle(
    *,
    model_id: str,
    cache_dir: str | None,
    device,
    compute_dtype: torch.dtype,
    disable_flash_attn2: bool,
    padding_side: str = "right",
    model_kwargs: dict | None = None,
):
    attn_implementation = "flash_attention_2" if not disable_flash_attn2 else "eager"
    if "internvl" in model_id.lower():
        model_type = "internvl"
        model = load_internvl_model(
            model_id=model_id,
            cache_dir=cache_dir,
            device=device,
            compute_dtype=compute_dtype,
            use_flash_attn=not disable_flash_attn2,
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
                "siglip"
                if getattr(vision_config, "model_type", None) == "siglip_vision_model"
                else "imagenet"
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
        model = load_model(
            model_id=model_id,
            model_type=model_type,
            cache_dir=cache_dir,
            attn_implementation=attn_implementation,
            compute_dtype=compute_dtype,
            model_kwargs=loader_kwargs,
        )

    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    return model, processor, tokenizer, model_type

__all__ = [
    "load_model",
    "load_processor_bundle",
    "load_vlm_bundle",
    "resolve_model_type",
]
