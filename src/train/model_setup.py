import importlib
from collections import deque

import torch
from transformers import (
    AutoConfig,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
)

try:
    importlib.import_module("pillow_avif")

    _AVIF_SUPPORT = True
except Exception:
    _AVIF_SUPPORT = False

from src.train.log_utils import rank0_print

try:
    from transformers import AutoModelForVision2Seq
except ImportError:
    AutoModelForVision2Seq = None


COMPONENT_ATTRIBUTE_ALIASES = {
    "vision": ("vision_model", "vision_tower"),
    "connector": ("connector", "multi_modal_projector"),
    "text": ("text_model", "language_model"),
}


def get_compute_dtype(training_args) -> torch.dtype:
    if training_args.fp16:
        return torch.float16
    if training_args.bf16:
        return torch.bfloat16
    return torch.float32


def build_model_from_pretrained_args(
    training_args,
    compute_dtype,
    *,
    llm_int8_skip_modules=None,
    include_load_flags: bool = False,
):
    model_kwargs = {"device_map": {"": training_args.device}}
    if training_args.bits not in [4, 8]:
        return model_kwargs

    if include_load_flags:
        model_kwargs.update(
            {
                "load_in_4bit": training_args.bits == 4,
                "load_in_8bit": training_args.bits == 8,
            }
        )

    quantization_kwargs = {
        "load_in_4bit": training_args.bits == 4,
        "load_in_8bit": training_args.bits == 8,
        "llm_int8_threshold": 6.0,
        "llm_int8_has_fp16_weight": False,
        "bnb_4bit_compute_dtype": compute_dtype,
        "bnb_4bit_use_double_quant": training_args.double_quant,
        "bnb_4bit_quant_type": training_args.quant_type,
    }
    if llm_int8_skip_modules:
        quantization_kwargs["llm_int8_skip_modules"] = list(llm_int8_skip_modules)

    model_kwargs["quantization_config"] = BitsAndBytesConfig(**quantization_kwargs)
    return model_kwargs


def get_component_name_spans(*components: str) -> tuple[str, ...]:
    spans: list[str] = []
    for component in components:
        try:
            aliases = COMPONENT_ATTRIBUTE_ALIASES[component]
        except KeyError as exc:
            raise ValueError(f"Unknown model component: {component!r}") from exc
        for alias in aliases:
            if alias not in spans:
                spans.append(alias)
    return tuple(spans)


def build_llm_int8_skip_modules(*components: str) -> list[str]:
    return list(get_component_name_spans(*components))


def extend_lora_namespan_exclude(
    lora_namespan_exclude,
    *,
    exclude_components: tuple[str, ...] = (),
) -> list[str]:
    updated = list(lora_namespan_exclude or [])
    for span in get_component_name_spans(*exclude_components):
        if span not in updated:
            updated.append(span)
    return updated


def resolve_model_type(model_id: str) -> str | None:
    config = AutoConfig.from_pretrained(
        model_id,
        trust_remote_code=True,
    )
    return getattr(config, "model_type", None)


def build_processor_load_kwargs(
    *,
    padding_side: str | None = None,
) -> dict:
    processor_kwargs = {"trust_remote_code": True}
    if padding_side is not None:
        processor_kwargs["padding_side"] = padding_side
    return processor_kwargs


def load_processor_and_tokenizer_backend(
    model_id: str,
    *,
    padding_side: str | None = None,
    cache_dir: str | None = None,
):
    model_type = resolve_model_type(model_id)
    processor_kwargs = build_processor_load_kwargs(
        padding_side=padding_side,
    )
    if cache_dir is not None:
        processor_kwargs["cache_dir"] = cache_dir

    try:
        processor = AutoProcessor.from_pretrained(
            model_id,
            **processor_kwargs,
        )
        tokenizer = getattr(processor, "tokenizer", None)
        return processor, (tokenizer if tokenizer is not None else processor), model_type
    except Exception:
        tokenizer_kwargs = {
            "trust_remote_code": True,
            "use_fast": False,
        }
        if padding_side is not None:
            tokenizer_kwargs["padding_side"] = padding_side
        if cache_dir is not None:
            tokenizer_kwargs["cache_dir"] = cache_dir
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            **tokenizer_kwargs,
        )
        return None, tokenizer, model_type


def resolve_vision_language_model_loader(model_type: str | None):
    if model_type in {"smolvlm", "smolvlm2"}:
        return AutoModelForImageTextToText
    if AutoModelForVision2Seq is not None:
        return AutoModelForVision2Seq
    return AutoModelForImageTextToText


def load_vision_language_model(
    *,
    model_id: str,
    model_type: str | None,
    cache_dir: str | None,
    attn_implementation: str,
    compute_dtype: torch.dtype,
    trust_remote_code: bool = True,
    model_kwargs: dict | None = None,
):
    loader_kwargs = dict(model_kwargs or {})
    loader_cls = resolve_vision_language_model_loader(model_type)
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


def find_target_linear_names(model, num_lora_modules=-1, lora_namespan_exclude=None, verbose=True):
    linear_cls = torch.nn.modules.Linear
    embedding_cls = torch.nn.modules.Embedding
    lora_module_names = []
    lora_namespan_exclude = lora_namespan_exclude or []

    for name, module in model.named_modules():
        if any(ex_keyword in name for ex_keyword in lora_namespan_exclude):
            continue
        if isinstance(module, (linear_cls, embedding_cls)):
            lora_module_names.append(name)

    if num_lora_modules > 0:
        lora_module_names = lora_module_names[-num_lora_modules:]
    if verbose:
        rank0_print(f"Found {len(lora_module_names)} lora modules")
    return lora_module_names


def set_requires_grad(parameters, requires_grad):
    for parameter in parameters:
        parameter.requires_grad = requires_grad


def iter_candidate_model_roots(model):
    queue = deque([model])
    seen: set[int] = set()
    while queue:
        current = queue.popleft()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        for attr in ("model", "base_model"):
            child = getattr(current, attr, None)
            if child is not None:
                queue.append(child)


def resolve_component_module(model, component: str):
    aliases = get_component_name_spans(component)
    for root in iter_candidate_model_roots(model):
        for alias in aliases:
            if hasattr(root, alias):
                return getattr(root, alias)
    raise AttributeError(f"Could not resolve {component} module on the model.")


def resolve_vision_module(model):
    return resolve_component_module(model, "vision")


def resolve_connector_module(model):
    return resolve_component_module(model, "connector")


def resolve_text_backbone(model):
    return resolve_component_module(model, "text")


def set_component_requires_grad(model, component: str, requires_grad: bool):
    module = resolve_component_module(model, component)
    set_requires_grad(module.parameters(), requires_grad)


def build_component_parameter_id_map(
    model,
    *,
    components: tuple[str, ...] = ("vision", "connector"),
) -> dict[int, str]:
    parameter_ids: dict[int, str] = {}
    for component in components:
        try:
            module = resolve_component_module(model, component)
        except AttributeError:
            continue
        for parameter in module.parameters():
            parameter_ids[id(parameter)] = component
    return parameter_ids


def configure_vision_tower(model, processor, training_args, compute_dtype, device):
    if processor is not None and hasattr(processor, "image_processor"):
        processor.image_processor.image_format = "AVIF" if _AVIF_SUPPORT else "JPEG"
        processor.image_processor.do_convert_rgb = True

    vision_tower = resolve_vision_module(model)
    vision_tower.to(dtype=compute_dtype, device=device)

    set_requires_grad(vision_tower.parameters(), not training_args.freeze_vision_tower)
    set_component_requires_grad(model, "connector", not training_args.freeze_connector)


def configure_llm(model, training_args):
    set_requires_grad(model.lm_head.parameters(), not training_args.freeze_llm)
    text_backbone = resolve_text_backbone(model)
    set_requires_grad(text_backbone.parameters(), not training_args.freeze_llm)


def unfreeze_topk_layers(model, k_llm: int = 0, k_vis: int = 0):
    if k_llm:
        text_backbone = None
        try:
            text_backbone = resolve_text_backbone(model)
        except AttributeError:
            text_backbone = None
        layer_stack = getattr(text_backbone, "layers", None) if text_backbone is not None else None
        if layer_stack is not None:
            for layer in layer_stack[-k_llm:]:
                for parameter in layer.parameters():
                    parameter.requires_grad = True

    if k_vis:
        vision_tower = None
        try:
            vision_tower = resolve_vision_module(model)
        except AttributeError:
            vision_tower = None
        vision_blocks = (
            getattr(vision_tower, "blocks", None)
            or getattr(vision_tower, "layers", None)
        )
        if vision_blocks is not None:
            for block in vision_blocks[-k_vis:]:
                for parameter in block.parameters():
                    parameter.requires_grad = True


def log_trainable_parameter_summary(model, header: str):
    rank0_print(header)
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()


__all__ = [
    "build_model_from_pretrained_args",
    "build_component_parameter_id_map",
    "build_llm_int8_skip_modules",
    "build_processor_load_kwargs",
    "configure_llm",
    "configure_vision_tower",
    "extend_lora_namespan_exclude",
    "find_target_linear_names",
    "get_compute_dtype",
    "get_component_name_spans",
    "iter_candidate_model_roots",
    "load_processor_and_tokenizer_backend",
    "load_vision_language_model",
    "log_trainable_parameter_summary",
    "resolve_component_module",
    "resolve_model_type",
    "resolve_vision_language_model_loader",
    "resolve_connector_module",
    "resolve_text_backbone",
    "resolve_vision_module",
    "set_component_requires_grad",
    "set_requires_grad",
    "unfreeze_topk_layers",
]
