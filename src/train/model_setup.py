import importlib

import torch
from PIL import Image, ImageFile
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
    Gemma3ForConditionalGeneration,
)

importlib.import_module("pillow_avif")
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

try:
    from transformers import AutoModelForVision2Seq
except ImportError:
    AutoModelForVision2Seq = None


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
    """Load a config and return its model_type so later loaders can branch by family."""
    config = AutoConfig.from_pretrained(
        model_id,
        trust_remote_code=True,
    )
    return getattr(config, "model_type", None)


def load_processor_and_tokenizer(
    model_id: str,
    *,
    padding_side: str | None = None,
    cache_dir: str | None = None,
):
    """Load the processor and tokenizer interface for one supported multimodal model."""
    model_type = resolve_model_type(model_id)
    if model_type not in SUPPORTED_AUTO_MODEL_TYPES:
        raise ValueError(
            "Unsupported model family for the generic processor loader: "
            f"model_id={model_id!r}, model_type={model_type!r}. "
            "Supported generic families are SmolVLM, Qwen-VL, Gemma 3, and Granite Vision "
            "(loaded in Transformers as `llava_next`). InternVL uses its dedicated loading path."
        )
    processor_kwargs = {"trust_remote_code": True}
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
    """Load one supported model with the configured dtype and attention backend."""
    if model_type not in SUPPORTED_AUTO_MODEL_TYPES:
        raise ValueError(
            "Unsupported model family for the generic model loader: "
            f"model_id={model_id!r}, model_type={model_type!r}. "
            "Supported generic families are SmolVLM, Qwen-VL, Gemma 3, and Granite Vision "
            "(loaded in Transformers as `llava_next`). InternVL uses its dedicated loading path."
        )
    loader_kwargs = dict(model_kwargs or {})
    if model_type == "gemma3":
        loader_cls = Gemma3ForConditionalGeneration
    elif model_type in SMOLVLM_MODEL_TYPES or AutoModelForVision2Seq is None:
        loader_cls = AutoModelForImageTextToText
    else:
        loader_cls = AutoModelForVision2Seq
    try:
        return loader_cls.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=compute_dtype,
            trust_remote_code=trust_remote_code,
            **loader_kwargs,
        )
    except TypeError as exc:
        if "attn_implementation" not in str(exc):
            raise
        return loader_cls.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            _attn_implementation=attn_implementation,
            torch_dtype=compute_dtype,
            trust_remote_code=trust_remote_code,
            **loader_kwargs,
        )


def load_model_and_processor(
    *,
    model_id: str,
    cache_dir: str | None,
    device,
    compute_dtype: torch.dtype,
    disable_flash_attn2: bool,
    padding_side: str = "right",
    model_kwargs: dict | None = None,
):
    """Load one supported VLM plus its processor/tokenizer bundle."""
    attn_implementation = "flash_attention_2" if not disable_flash_attn2 else "eager"
    if "internvl" in model_id.lower():
        model_type = "internvl"
        model = AutoModel.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            torch_dtype=compute_dtype,
            low_cpu_mem_usage=True,
            use_flash_attn=not disable_flash_attn2,
            trust_remote_code=True,
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            padding_side=padding_side,
            trust_remote_code=True,
            use_fast=False,
        )
        img_context_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        if hasattr(model, "img_context_token_id"):
            model.img_context_token_id = img_context_token_id
        vision_config = getattr(model.config, "vision_config", None)
        processor = {
            "model_id": model_id,
            "tokenizer": tokenizer,
            "image_size": getattr(model.config, "force_image_size", None)
            or getattr(vision_config, "image_size", 448),
            "normalize_type": (
                "siglip"
                if getattr(vision_config, "model_type", None) == "siglip_vision_model"
                else "imagenet"
            ),
            "max_num_tiles": 6,
            "num_image_token": getattr(model, "num_image_token", 256),
            "img_start_token": "<img>",
            "img_end_token": "</img>",
            "img_context_token": "<IMG_CONTEXT>",
        }
    else:
        processor, tokenizer, model_type = load_processor_and_tokenizer(
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
    "load_model_and_processor",
    "load_processor_and_tokenizer",
    "resolve_model_type",
]
