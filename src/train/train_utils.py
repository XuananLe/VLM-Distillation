"""Compatibility facade for training utilities.

This module preserves the existing import surface while delegating to
smaller responsibility-focused modules.
"""

from src.train.arg_utils import parse_list_argument, parse_model_id_list
from src.train.finetune_setup import (
    configure_training_model,
    finalize_quantized_trainable_modules,
    load_training_model_bundle,
    maybe_apply_lora,
    normalize_lora_namespan_exclude,
    prepare_model_for_low_bit_training,
)
from src.train.log_utils import rank0_print, set_local_rank
from src.train.model_setup import (
    build_component_parameter_id_map,
    build_llm_int8_skip_modules,
    build_model_from_pretrained_args,
    build_processor_load_kwargs,
    configure_llm,
    configure_vision_tower,
    extend_lora_namespan_exclude,
    find_target_linear_names,
    get_compute_dtype,
    get_component_name_spans,
    iter_candidate_model_roots,
    load_processor_and_tokenizer_backend,
    load_vision_language_model,
    log_trainable_parameter_summary,
    resolve_component_module,
    resolve_model_type,
    resolve_vision_language_model_loader,
    set_requires_grad,
    set_component_requires_grad,
    unfreeze_topk_layers,
)
from src.train.save_utils import (
    _save_processing_assets,
    get_peft_state_maybe_zero_3,
    get_peft_state_non_lora_maybe_zero_3,
    maybe_zero_3,
    safe_save_model_for_hf_trainer,
)

__all__ = [
    "_save_processing_assets",
    "build_component_parameter_id_map",
    "build_llm_int8_skip_modules",
    "build_model_from_pretrained_args",
    "build_processor_load_kwargs",
    "configure_llm",
    "configure_training_model",
    "configure_vision_tower",
    "extend_lora_namespan_exclude",
    "finalize_quantized_trainable_modules",
    "find_target_linear_names",
    "get_compute_dtype",
    "get_component_name_spans",
    "get_peft_state_maybe_zero_3",
    "get_peft_state_non_lora_maybe_zero_3",
    "iter_candidate_model_roots",
    "load_processor_and_tokenizer_backend",
    "load_vision_language_model",
    "log_trainable_parameter_summary",
    "maybe_zero_3",
    "maybe_apply_lora",
    "normalize_lora_namespan_exclude",
    "parse_list_argument",
    "parse_model_id_list",
    "prepare_model_for_low_bit_training",
    "rank0_print",
    "resolve_component_module",
    "resolve_model_type",
    "resolve_vision_language_model_loader",
    "safe_save_model_for_hf_trainer",
    "set_component_requires_grad",
    "set_local_rank",
    "set_requires_grad",
    "load_training_model_bundle",
    "unfreeze_topk_layers",
]
