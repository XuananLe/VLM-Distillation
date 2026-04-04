import torch
from transformers import (
    AutoModel,
    AutoModelForVision2Seq,
    AutoProcessor,
    AutoTokenizer,
    EarlyStoppingCallback,
    Gemma3ForConditionalGeneration,
)

from src.train.train_utils import rank0_print


def load_teacher_models_and_processors(
    *,
    teacher_ids,
    training_args,
    compute_dtype,
    attn_impl: str,
):
    teacher_models = []
    teacher_processors = []

    for teacher_idx, teacher_id in enumerate(teacher_ids, start=1):
        rank0_print(f"\nLoading teacher model {teacher_idx}/{len(teacher_ids)}...")
        if "internvl" in teacher_id.lower():
            teacher_model = AutoModel.from_pretrained(
                teacher_id,
                cache_dir=training_args.cache_dir,
                torch_dtype=compute_dtype,
                low_cpu_mem_usage=True,
                use_flash_attn=not training_args.disable_flash_attn2,
                trust_remote_code=True,
            ).to(training_args.device)
        elif "gemma-3" in teacher_id.lower():
            teacher_model = Gemma3ForConditionalGeneration.from_pretrained(
                teacher_id,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_impl,
                torch_dtype=compute_dtype,
                trust_remote_code=True,
                device_map={"": training_args.device},
            )
        else:
            teacher_model = AutoModelForVision2Seq.from_pretrained(
                teacher_id,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_impl,
                torch_dtype=compute_dtype,
                trust_remote_code=True,
                device_map={"": training_args.device},
            )

        if hasattr(teacher_model.config, "use_cache"):
            teacher_model.config.use_cache = False
        teacher_model._suppress_forward_stdout = "internvl" in teacher_id.lower()
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad_(False)
        teacher_models.append(teacher_model)

        if "internvl" in teacher_id.lower():
            teacher_tokenizer = AutoTokenizer.from_pretrained(
                teacher_id,
                cache_dir=training_args.cache_dir,
                padding_side="right",
                trust_remote_code=True,
                use_fast=False,
            )
            img_context_token_id = teacher_tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
            if hasattr(teacher_model, "img_context_token_id"):
                teacher_model.img_context_token_id = img_context_token_id
            vision_config = getattr(teacher_model.config, "vision_config", None)
            teacher_processors.append(
                {
                    "model_id": teacher_id,
                    "tokenizer": teacher_tokenizer,
                    "image_size": getattr(teacher_model.config, "force_image_size", None)
                    or getattr(vision_config, "image_size", 448),
                    "normalize_type": (
                        "siglip"
                        if getattr(vision_config, "model_type", None) == "siglip_vision_model"
                        else "imagenet"
                    ),
                    "max_num_tiles": 6,
                    "num_image_token": getattr(teacher_model, "num_image_token", 256),
                    "img_start_token": "<img>",
                    "img_end_token": "</img>",
                    "img_context_token": "<IMG_CONTEXT>",
                }
            )
        else:
            teacher_processors.append(
                AutoProcessor.from_pretrained(
                    teacher_id,
                    cache_dir=training_args.cache_dir,
                    padding_side="right",
                    trust_remote_code=True,
                )
            )

        rank0_print(f"Teacher model loaded and frozen: {teacher_id}")

    torch.cuda.empty_cache()
    return teacher_models, teacher_processors


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
