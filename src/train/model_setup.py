import importlib

import torch
from transformers import (
    AutoConfig,
    AutoModelForImageTextToText,
    AutoProcessor,
)

try:
    importlib.import_module("pillow_avif")

    _AVIF_SUPPORT = True
except ModuleNotFoundError:
    _AVIF_SUPPORT = False

try:
    from transformers import AutoModelForVision2Seq
except ImportError:
    AutoModelForVision2Seq = None


COMPONENT_ATTRIBUTE_ALIASES = {
    "vision": ("vision_model", "vision_tower"),
    "connector": ("connector", "multi_modal_projector"),
    "text": ("text_model", "language_model"),
}
SUPPORTED_AUTO_MODEL_TYPES = {
    "smolvlm",
    "smolvlm2",
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
    if model_type in {"smolvlm", "smolvlm2"} or AutoModelForVision2Seq is None:
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


def resolve_component_module(model, component: str):
    """Resolve one logical component like vision or connector from the supported VLM wrappers."""
    try:
        aliases = COMPONENT_ATTRIBUTE_ALIASES[component]
    except KeyError as exc:
        raise ValueError(f"Unknown model component: {component!r}") from exc

    roots = [
        model,
        getattr(model, "model", None),
        getattr(model, "base_model", None),
        getattr(getattr(model, "model", None), "base_model", None),
    ]
    seen: set[int] = set()
    for root in roots:
        if root is None or id(root) in seen:
            continue
        seen.add(id(root))
        for alias in aliases:
            module = getattr(root, alias, None)
            if module is not None:
                return module
    raise AttributeError(f"Could not resolve {component} module on the model.")

def build_component_parameter_id_map(
    model,
    *,
    components: tuple[str, ...] = ("vision", "connector"),
) -> dict[int, str]:
    """Map parameter object ids to logical component names for grouped optimizer setup."""
    parameter_ids: dict[int, str] = {}
    for component in components:
        try:
            module = resolve_component_module(model, component)
        except AttributeError:
            continue
        for parameter in module.parameters():
            parameter_ids[id(parameter)] = component
    return parameter_ids


def configure_vision_tower(model, processor, compute_dtype, device):
    """Move the vision tower to the training device/dtype and align processor image settings."""
    if processor is not None and hasattr(processor, "image_processor"):
        processor.image_processor.image_format = "AVIF" if _AVIF_SUPPORT else "JPEG"
        processor.image_processor.do_convert_rgb = True

    vision_tower = resolve_component_module(model, "vision")
    vision_tower.to(dtype=compute_dtype, device=device)


__all__ = [
    "build_component_parameter_id_map",
    "configure_vision_tower",
    "load_model",
    "load_processor_and_tokenizer",
    "resolve_component_module",
    "resolve_model_type",
]
