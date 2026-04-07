import importlib
import warnings

from transformers import (
    EarlyStoppingCallback,
)

from src.train.train_utils import rank0_print


def ensure_flash_attention_available() -> None:
    try:
        importlib.import_module("flash_attn")
    except Exception as exc:
        raise RuntimeError(
            "FlashAttention was requested for teacher loading, but the `flash_attn` package "
            "is not available in the current environment."
        ) from exc


def _config_attr(config, name: str, default=None):
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def teacher_supports_flash_attention(teacher_model) -> bool:
    supports_flash_attn = getattr(teacher_model, "_supports_flash_attn", None)
    if supports_flash_attn is not None:
        return bool(supports_flash_attn)

    config = getattr(teacher_model, "config", None)
    model_type = _config_attr(config, "model_type")
    if isinstance(model_type, str) and model_type.startswith("internvl"):
        return True

    vision_config = _config_attr(config, "vision_config")
    if _config_attr(vision_config, "use_flash_attn") is True:
        return True

    llm_config = _config_attr(config, "llm_config")
    if _config_attr(llm_config, "attn_implementation") == "flash_attention_2":
        return True

    return model_type in {"qwen2_vl", "qwen2_5_vl", "gemma3"}


def require_flash_attention_support(teacher_model, teacher_id: str) -> None:
    if teacher_supports_flash_attention(teacher_model):
        return
    warnings.warn(
        f"Teacher model {teacher_id!r} does not advertise FlashAttention support. "
        "Continuing anyway; teacher loading may still fail or fall back internally.",
        stacklevel=2,
    )


def build_trainer_callbacks(*, training_args, data_module):
    trainer_callbacks = []
    if training_args.early_stopping_patience is None:
        return trainer_callbacks

    if data_module["eval_dataset"] is None:
        raise ValueError("Early stopping requires --eval_data_path.")
    if training_args.eval_strategy == "no":
        raise ValueError("Early stopping requires --eval_strategy to run validation.")
    if training_args.metric_for_best_model is None:
        training_args.metric_for_best_model = "eval_loss"
        training_args.greater_is_better = False
    if not training_args.load_best_model_at_end:
        training_args.load_best_model_at_end = True

    trainer_callbacks.append(
        EarlyStoppingCallback(
            early_stopping_patience=training_args.early_stopping_patience,
            early_stopping_threshold=training_args.early_stopping_threshold,
        )
    )
    rank0_print(
        "Early stopping enabled: "
        f"metric={training_args.metric_for_best_model}, "
        f"patience={training_args.early_stopping_patience}, "
        f"threshold={training_args.early_stopping_threshold}"
    )
    return trainer_callbacks
