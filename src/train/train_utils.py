import ast
import logging

import torch
import transformers
from transformers import BitsAndBytesConfig

try:
    import pillow_avif  # noqa: F401
    _AVIF_SUPPORT = True
except Exception:
    _AVIF_SUPPORT = False

_LOCAL_RANK = None


def set_local_rank(local_rank) -> None:
    global _LOCAL_RANK
    _LOCAL_RANK = local_rank


def rank0_print(*args):
    if _LOCAL_RANK == 0 or _LOCAL_RANK == "0" or _LOCAL_RANK is None or _LOCAL_RANK == -1:
        print(*args)


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
    model_kwargs = {
        "device_map": {"": training_args.device},
    }
    if training_args.bits not in [4, 8]:
        return model_kwargs

    if include_load_flags:
        model_kwargs.update(
            dict(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
            )
        )

    quantization_kwargs = dict(
        load_in_4bit=training_args.bits == 4,
        load_in_8bit=training_args.bits == 8,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=training_args.double_quant,
        bnb_4bit_quant_type=training_args.quant_type,
    )
    if llm_int8_skip_modules:
        quantization_kwargs["llm_int8_skip_modules"] = list(llm_int8_skip_modules)

    model_kwargs.update(
        dict(
            quantization_config=BitsAndBytesConfig(**quantization_kwargs),
        )
    )
    return model_kwargs


def parse_list_argument(raw_value: str | None, *, arg_name: str, element_type: type = str) -> list:
    """Generic parser for list arguments that supports Python literals or comma-separated strings."""
    if raw_value is None or not raw_value.strip():
        return []

    try:
        parsed = ast.literal_eval(raw_value.strip())
    except (SyntaxError, ValueError):
        parsed = [item.strip() for item in raw_value.split(",") if item.strip()]

    # Normalize to list
    if isinstance(parsed, (str, int)):
        parsed = [parsed]
    elif isinstance(parsed, tuple):
        parsed = list(parsed)
    elif not isinstance(parsed, list):
        raise ValueError(
            f"{arg_name} must be a Python list literal, single value, or comma-separated string."
        )

    # Convert elements to target type
    try:
        if element_type is str:
            result = [str(item).strip() for item in parsed if str(item).strip()]
        elif element_type is int:
            result = [int(item) for item in parsed]
        else:
            result = [element_type(item) for item in parsed]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{arg_name} must contain only {element_type.__name__} values, got: {raw_value!r}") from exc

    return result


def parse_model_id_list(raw_model_ids: str, *, arg_name: str) -> list[str]:
    model_ids = parse_list_argument(raw_model_ids, arg_name=arg_name, element_type=str)
    if not model_ids:
        raise ValueError(f"At least one model ID must be provided via {arg_name}.")
    return model_ids


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


def resolve_attr_path(obj, path: str):
    current = obj
    for part in path.split("."):
        current = getattr(current, part)
    return current


def infer_hidden_size(model) -> int:
    candidate_paths = (
        "config.text_config.hidden_size",
        "config.hidden_size",
        "model.text_model.config.hidden_size",
        "model.config.text_config.hidden_size",
        "model.config.hidden_size",
        "language_model.config.hidden_size",
        "lm_head.in_features",
    )
    for path in candidate_paths:
        try:
            value = resolve_attr_path(model, path)
        except AttributeError:
            continue
        if isinstance(value, int):
            return value
    raise ValueError(f"Could not infer hidden size for model type {type(model).__name__}")


def log_trainable_parameter_summary(model, header: str):
    rank0_print(header)
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param

# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias.items():
            if k in lora_bias_names:
                to_return[k] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        _save_processing_assets(trainer, output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa
        trainer.model.config.save_pretrained(output_dir)
        _save_processing_assets(trainer, output_dir)


def _save_processing_assets(trainer: transformers.Trainer, output_dir: str) -> None:
    if not getattr(trainer.args, "should_save", False):
        return
    if hasattr(trainer, "is_world_process_zero") and not trainer.is_world_process_zero():
        return

    saved = set()
    for asset in (
        getattr(trainer, "processing_class", None),
        getattr(trainer, "processor", None),
        getattr(trainer, "tokenizer", None),
    ):
        if asset is None or id(asset) in saved:
            continue
        if hasattr(asset, "save_pretrained"):
            asset.save_pretrained(output_dir)
            saved.add(id(asset))
            break
