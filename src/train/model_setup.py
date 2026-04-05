import torch
from transformers import BitsAndBytesConfig

try:
    import pillow_avif  # noqa: F401

    _AVIF_SUPPORT = True
except Exception:
    _AVIF_SUPPORT = False

from src.train.log_utils import rank0_print


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


def configure_vision_tower(model, processor, training_args, compute_dtype, device):
    if processor is not None and hasattr(processor, "image_processor"):
        processor.image_processor.image_format = "AVIF" if _AVIF_SUPPORT else "JPEG"
        processor.image_processor.do_convert_rgb = True

    vision_tower = model.model.vision_model
    vision_tower.to(dtype=compute_dtype, device=device)

    set_requires_grad(vision_tower.parameters(), not training_args.freeze_vision_tower)
    set_requires_grad(model.model.connector.parameters(), not training_args.freeze_connector)


def configure_llm(model, training_args):
    set_requires_grad(model.lm_head.parameters(), not training_args.freeze_llm)
    set_requires_grad(model.model.text_model.parameters(), not training_args.freeze_llm)


def unfreeze_topk_layers(model, k_llm: int = 0, k_vis: int = 0):
    if k_llm and hasattr(model, "model") and hasattr(model.model, "layers"):
        for layer in model.model.layers[-k_llm:]:
            for parameter in layer.parameters():
                parameter.requires_grad = True

    if k_vis and hasattr(model, "vision_model") and hasattr(model.vision_model, "blocks"):
        for block in model.vision_model.blocks[-k_vis:]:
            for parameter in block.parameters():
                parameter.requires_grad = True


def log_trainable_parameter_summary(model, header: str):
    rank0_print(header)
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()


__all__ = [
    "build_model_from_pretrained_args",
    "configure_llm",
    "configure_vision_tower",
    "find_target_linear_names",
    "get_compute_dtype",
    "log_trainable_parameter_summary",
    "set_requires_grad",
    "unfreeze_topk_layers",
]
