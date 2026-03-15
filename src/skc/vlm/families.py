from transformers import AutoConfig


def raise_removed_deepseek_vl_support(model_name: str):
    raise RuntimeError(
        f"DeepSeek-VL v1 support was removed from the SKC/audit codepaths for '{model_name}'. "
        "Use src/eval/vlmeval for DeepSeek-VL evaluation instead."
    )


def infer_model_family(model_name: str, model_type: str | None = None) -> str:
    model_name = model_name.lower()
    model_type = (model_type or "").lower()

    if model_type == "deepseek_vl_v2" or "deepseek-vl2" in model_name or "deepseek_vl2" in model_name:
        return "deepseek_vl2"
    if model_type in {"deepseek_vl", "deepseek_vl_chat", "multi_modality"} or "deepseek-vl-" in model_name:
        return "removed_deepseek_vl"
    if model_type == "internvl_chat" or model_type.startswith("internvl"):
        return "internvl_chat"
    if "internvl" in model_name and "-hf" not in model_name:
        return "internvl_chat"
    if model_type.startswith("qwen3_vl") or "qwen3-vl" in model_name:
        return "qwen3_vl"
    if model_type in {"qwen2_vl", "qwen2_5_vl"}:
        return "qwen2_vl"
    if "qwen2-vl" in model_name or "qwen2.5-vl" in model_name:
        return "qwen2_vl"
    if model_type.startswith("qwen2_5_omni") or "qwen2.5-omni" in model_name:
        return "qwen_omni"
    if "qwen-vl-chat" in model_name or model_type == "qwen":
        return "qwen_vl"
    if model_type == "llava_onevision" or "llava-onevision" in model_name:
        return "llava_onevision"
    if model_type == "llava_next" or "llava-v1.6" in model_name or "llava-next" in model_name:
        return "llava_next"
    if model_type == "idefics3" or "idefics3" in model_name:
        return "idefics3"
    if model_type.startswith("gemma3") or "gemma-3" in model_name or "gemma 3" in model_name:
        return "gemma3"
    if model_type in {"phi4_multimodal", "phi4mm"} or "phi-4-multimodal" in model_name:
        return "phi4_multimodal"
    if model_type.startswith("glm4v") or "glm-4v" in model_name:
        return "glm4v"
    if model_type in {"yi_vl", "yi_vl_for_causal_lm"} or "yi-vl" in model_name:
        return "yi_vl"
    if "kimi-vl" in model_name:
        return "kimi_vl"
    return "image_text_to_text"


def resolve_model_family(model_name: str):
    cfg = None
    try:
        cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        family = infer_model_family(model_name, getattr(cfg, "model_type", None))
    except Exception:
        family = infer_model_family(model_name)
    return cfg, family
