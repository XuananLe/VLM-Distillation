"""Compatibility facade for training utilities.

This module preserves the existing import surface while delegating to
smaller responsibility-focused modules.
"""

from src.train.arg_utils import parse_list_argument, parse_model_id_list
from src.train.log_utils import rank0_print, set_local_rank
from src.train.model_setup import (
    build_model_from_pretrained_args,
    configure_llm,
    configure_vision_tower,
    find_target_linear_names,
    get_compute_dtype,
    log_trainable_parameter_summary,
    set_requires_grad,
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
    "build_model_from_pretrained_args",
    "configure_llm",
    "configure_vision_tower",
    "find_target_linear_names",
    "get_compute_dtype",
    "get_peft_state_maybe_zero_3",
    "get_peft_state_non_lora_maybe_zero_3",
    "log_trainable_parameter_summary",
    "maybe_zero_3",
    "parse_list_argument",
    "parse_model_id_list",
    "rank0_print",
    "safe_save_model_for_hf_trainer",
    "set_local_rank",
    "set_requires_grad",
    "unfreeze_topk_layers",
]
