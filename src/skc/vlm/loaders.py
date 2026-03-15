from importlib import import_module

import torch
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    Qwen2VLForConditionalGeneration,
)

from .families import raise_removed_deepseek_vl_support
from .patches import (
    patch_dynamic_cache_get_usable_length,
    patch_llama_flash_attention2_symbol,
    patch_phi4_prepare_inputs_for_generation,
    patch_qwen_vl_stream_generator,
)


def move_to_cuda_if_available(model):
    if torch.cuda.is_available():
        return model.cuda()
    return model


def build_load_kwargs(dtype: torch.dtype):
    load_kwargs = {"torch_dtype": dtype}
    auto_load_kwargs = dict(load_kwargs)
    if torch.cuda.is_available():
        auto_load_kwargs["device_map"] = "auto"
    return load_kwargs, auto_load_kwargs


def load_deepseek_vl2(model_name: str, load_kwargs: dict):
    patch_llama_flash_attention2_symbol()
    from deepseek_vl2.models import DeepseekVLV2ForCausalLM  # noqa: F401

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        _attn_implementation="eager",
        **load_kwargs,
    )
    if not hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    return model


def load_qwen3_vl(model_name: str, load_kwargs: dict):
    try:
        if "qwen3-vl-moe" in model_name.lower():
            model_cls = getattr(import_module("transformers"), "Qwen3VLMoeForConditionalGeneration")
        else:
            model_cls = getattr(import_module("transformers"), "Qwen3VLForConditionalGeneration")
    except Exception as exc:
        raise RuntimeError(
            "Loading Qwen3-VL requires a newer Transformers version (>=4.57.0). "
            "Upgrade with `uv pip install --upgrade \"transformers>=4.57.0\"` "
            "or `uv pip install --upgrade git+https://github.com/huggingface/transformers.git`."
        ) from exc

    return model_cls.from_pretrained(model_name, **load_kwargs)


def load_qwen_omni(model_name: str, load_kwargs: dict):
    tfm = import_module("transformers")
    for cls_name in (
        "Qwen2_5OmniThinkerForConditionalGeneration",
        "Qwen2_5OmniForConditionalGeneration",
        "Qwen2_5OmniModel",
    ):
        cls = getattr(tfm, cls_name, None)
        if cls is not None:
            return cls.from_pretrained(model_name, **load_kwargs)
    raise RuntimeError(
        "Qwen2.5-Omni class was not found in installed transformers. "
        "Install a newer 4.x build (e.g. >=4.57.0,<5)."
    )


def load_auto_multimodal(model_name: str, load_kwargs: dict):
    attempts = (
        AutoModelForImageTextToText,
        AutoModelForCausalLM,
        AutoModel,
    )
    last_exc = None
    for cls in attempts:
        try:
            return cls.from_pretrained(model_name, trust_remote_code=True, **load_kwargs)
        except Exception as exc:
            last_exc = exc

    raise RuntimeError(
        f"Failed to load multimodal model '{model_name}' with AutoModelForImageTextToText, "
        "AutoModelForCausalLM, and AutoModel. "
        f"Last error: {type(last_exc).__name__}: {last_exc}"
    ) from last_exc


def load_yi_vl(model_name: str, auto_load_kwargs: dict):
    return AutoModelForImageTextToText.from_pretrained(
        model_name,
        trust_remote_code=True,
        ignore_mismatched_sizes=True,
        **auto_load_kwargs,
    )


def load_phi4_multimodal(model_name: str, auto_load_kwargs: dict):
    patch_dynamic_cache_get_usable_length()
    tfm = import_module("transformers")
    official_cls = getattr(tfm, "Phi4MultimodalForCausalLM", None)
    official_exc = None
    if official_cls is not None:
        try:
            return official_cls.from_pretrained(
                model_name,
                _attn_implementation="eager",
                **auto_load_kwargs,
            )
        except Exception as exc:
            official_exc = exc

    try:
        return AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            _attn_implementation="eager",
            **auto_load_kwargs,
        )
    except Exception as exc:
        error_text = str(exc)
        if "prepare_inputs_for_generation" not in error_text and "Phi4MMModel.prepare_inputs_for_generation" not in error_text:
            raise
        patch_phi4_prepare_inputs_for_generation(model_name)
        return AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            _attn_implementation="eager",
            **auto_load_kwargs,
        )
    except Exception as exc:
        if official_exc is not None:
            raise RuntimeError(
                "Phi-4 multimodal failed with both the official Transformers loader and the "
                f"patched remote-code loader. Official error: {type(official_exc).__name__}: {official_exc}. "
                f"Remote-code error: {type(exc).__name__}: {exc}"
            ) from exc
        raise


def load_internvl_chat(model_name: str, dtype: torch.dtype):
    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=False,
    )
    return move_to_cuda_if_available(model)


def load_qwen2_vl(model_name: str, cfg, auto_load_kwargs: dict):
    if cfg is not None and cfg.model_type == "qwen2_vl":
        return Qwen2VLForConditionalGeneration.from_pretrained(model_name, **auto_load_kwargs)
    return load_auto_multimodal(model_name, auto_load_kwargs)


def load_model_for_family(
    model_name: str,
    family: str,
    cfg,
    dtype: torch.dtype,
    load_kwargs: dict,
    auto_load_kwargs: dict,
):
    if family == "qwen2_vl":
        return load_qwen2_vl(model_name, cfg, auto_load_kwargs)
    if family == "qwen_omni":
        return load_qwen_omni(model_name, auto_load_kwargs)
    if family == "removed_deepseek_vl":
        raise_removed_deepseek_vl_support(model_name)
    if family == "yi_vl":
        return load_yi_vl(model_name, auto_load_kwargs)
    if family == "phi4_multimodal":
        return load_phi4_multimodal(model_name, auto_load_kwargs)
    if family == "qwen_vl":
        patch_qwen_vl_stream_generator()
        return load_auto_multimodal(model_name, auto_load_kwargs)
    if family == "deepseek_vl2":
        return load_deepseek_vl2(model_name, auto_load_kwargs)
    if family == "qwen3_vl":
        return load_qwen3_vl(model_name, auto_load_kwargs)
    if family == "internvl_chat":
        return load_internvl_chat(model_name, dtype)
    return load_auto_multimodal(model_name, auto_load_kwargs)
