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
INTERNVL_IMAGE_SIZE = 448
INTERNVL_MAX_NUM_TILES = 12
INTERNVL_NUM_IMAGE_TOKEN = 256
INTERNVL_IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"


def load_vlm_components(
    *,
    model_id: str,
    cache_dir: str | None = None,
    device=None,
    compute_dtype: torch.dtype | None = None,
    disable_flash_attn2: bool = False,
    padding_side: str = "right",
    model_kwargs: dict | None = None,
    attn_implementation: str | None = None,
    load_model: bool = True,
):
    if attn_implementation is None:
        attn_implementation = "eager" if disable_flash_attn2 else "flash_attention_2"

    if "internvl" in model_id.lower():
        model_type = "internvl"
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            padding_side=padding_side,
            trust_remote_code=True,
            use_fast=True,
        )
        model = None
        vision_config = None
        if load_model:
            if device is None or compute_dtype is None:
                raise ValueError("load_vlm_components requires `device` and `compute_dtype` when load_model=True.")
            model = load_internvl_model(
                model_id=model_id,
                cache_dir=cache_dir,
                device=device,
                compute_dtype=compute_dtype,
                use_flash_attn=attn_implementation == "flash_attention_2",
            )
            model.img_context_token_id = tokenizer.convert_tokens_to_ids(INTERNVL_IMG_CONTEXT_TOKEN)
            vision_config = getattr(model.config, "vision_config", None)
        processor = {
            "model_id": model_id,
            "tokenizer": tokenizer,
            "image_size": (
                getattr(model.config, "force_image_size", None)
                or getattr(vision_config, "image_size", INTERNVL_IMAGE_SIZE)
                if model is not None
                else INTERNVL_IMAGE_SIZE
            ),
            "normalize_type": (
                "siglip" if getattr(vision_config, "model_type", None) == "siglip_vision_model" else "imagenet"
            ),
            "max_num_tiles": INTERNVL_MAX_NUM_TILES,
            "num_image_token": getattr(model, "num_image_token", INTERNVL_NUM_IMAGE_TOKEN)
            if model is not None
            else INTERNVL_NUM_IMAGE_TOKEN,
        }
    else:
        config = AutoConfig.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            trust_remote_code=True,
        )
        model_type = getattr(config, "model_type", None)
        if model_type not in SUPPORTED_AUTO_MODEL_TYPES:
            raise ValueError("We only support SmolVLM, Qwen-VL, Gemma 3, and InternVL.")
        processor = AutoProcessor.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            padding_side=padding_side,
            trust_remote_code=True,
            use_fast=False,
        )
        if hasattr(processor, "image_processor"):
            processor.image_processor.image_format = "AVIF"
            processor.image_processor.do_convert_rgb = True
        tokenizer = getattr(processor, "tokenizer", None) or processor
        model = None
        if not load_model:
            return model, processor, tokenizer, model_type
        if device is None or compute_dtype is None:
            raise ValueError("load_vlm_components requires `device` and `compute_dtype` when load_model=True.")
        loader_kwargs = {"device_map": {"": device}}
        loader_kwargs.update(model_kwargs or {})
        if model_type == "gemma3":
            loader_cls = Gemma3ForConditionalGeneration
        else:
            loader_cls = AutoModelForImageTextToText
        model = loader_cls.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            attn_implementation=attn_implementation,
            dtype=compute_dtype,
            trust_remote_code=True,
            **loader_kwargs,
        )

    if model is not None and hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    return model, processor, tokenizer, model_type


__all__ = [
    "load_vlm_components",
]
