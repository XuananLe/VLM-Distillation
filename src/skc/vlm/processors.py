from transformers import AutoProcessor, AutoTokenizer, CLIPImageProcessor

from ..config.families import TOKENIZER_ONLY_FAMILIES
from .families import raise_removed_deepseek_vl_support
from .patches import patch_llama_flash_attention2_symbol


def load_remote_tokenizer(model_name: str):
    return AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=False,
    )


def load_remote_processor(model_name: str):
    return AutoProcessor.from_pretrained(model_name, trust_remote_code=True)


def load_deepseek_vl2_processor(model_name: str):
    patch_llama_flash_attention2_symbol()
    try:
        from deepseek_vl2.models import DeepseekVLV2Processor
    except ImportError as exc:
        raise ImportError(
            "DeepSeek-VL2 processor is not available. Install DeepSeek-VL2 package first "
            "(e.g. `pip install -e .` inside DeepSeek-VL2 repo)."
        ) from exc

    return DeepseekVLV2Processor.from_pretrained(model_name)


def load_internvl_processor(model_name: str, model=None):
    tokenizer = load_remote_tokenizer(model_name)
    image_processor = CLIPImageProcessor.from_pretrained(model_name)
    img_context_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
    if model is not None and hasattr(model, "img_context_token_id"):
        model.img_context_token_id = img_context_token_id
    return {
        "tokenizer": tokenizer,
        "image_processor": image_processor,
        "num_image_token": getattr(model, "num_image_token", 256),
        "img_start_token": "<img>",
        "img_end_token": "</img>",
        "img_context_token": "<IMG_CONTEXT>",
    }


def load_phi4_multimodal_processor(model_name: str):
    return AutoProcessor.from_pretrained(model_name, trust_remote_code=True)


def load_vlm_processor(model_name: str, family: str, model=None):
    if family == "deepseek_vl2":
        return load_deepseek_vl2_processor(model_name)
    if family == "removed_deepseek_vl":
        raise_removed_deepseek_vl_support(model_name)
    if family == "internvl_chat":
        return load_internvl_processor(model_name, model)
    if family == "phi4_multimodal":
        return load_phi4_multimodal_processor(model_name)
    if family in TOKENIZER_ONLY_FAMILIES or family == "qwen_vl":
        return load_remote_tokenizer(model_name)
    return load_remote_processor(model_name)
