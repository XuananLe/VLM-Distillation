import os
import sys
import importlib
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
from transformers import (
    HfArgumentParser,
)
from src.trainer.distillation_trainer import DistillationTrainer
from src.dataset.sft_data import make_supervised_data_module
from src.params import DataArguments, TrainingArguments
from src.train.distillation_setup import (
    DistillationArguments,
    log_distillation_setup,
    validate_distillation_args,
)
from src.train.train_utils import (
    configure_training_model,
    finalize_quantized_trainable_modules,
    safe_save_model_for_hf_trainer,
    get_compute_dtype,
    get_peft_state_maybe_zero_3,
    get_peft_state_non_lora_maybe_zero_3,
    load_training_model_bundle,
    maybe_apply_lora,
    normalize_lora_namespan_exclude,
    prepare_model_for_low_bit_training,
    set_local_rank,
    rank0_print,
    load_processor_and_tokenizer_backend,
    parse_model_id_list,
)
from src.train.distillation_runtime import (
    build_trainer_callbacks,
)

from PIL import Image, ImageFile

importlib.import_module("pillow_avif")

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


def _uses_wandb(report_to) -> bool:
    if report_to is None:
        return False
    if isinstance(report_to, str):
        parts = [part.strip() for part in report_to.split(",") if part.strip()]
        return "all" in parts or "wandb" in parts
    return "all" in report_to or "wandb" in report_to


def _init_primary_wandb_run(training_args) -> None:
    if not _uses_wandb(training_args.report_to):
        return

    try:
        import wandb
    except Exception:
        return

    if wandb.run is not None:
        return

    init_kwargs = {
        "project": os.getenv("WANDB_PROJECT", "huggingface"),
        "settings": wandb.Settings(
            mode="shared",
            x_primary=True,
            x_label="trainer",
        ),
    }
    wandb_entity = os.getenv("WANDB_ENTITY")
    wandb_run_id = os.getenv("WANDB_RUN_ID")
    wandb_resume = os.getenv("WANDB_RESUME")
    if wandb_entity:
        init_kwargs["entity"] = wandb_entity
    if wandb_run_id:
        init_kwargs["id"] = wandb_run_id
    if wandb_resume:
        init_kwargs["resume"] = wandb_resume
    if training_args.run_name is not None:
        init_kwargs["name"] = training_args.run_name
    wandb.init(**init_kwargs)

def train_distillation():
    """
    Main training function for VLM distillation.

    This script supports CE + KD distillation from one or more frozen teachers.
    """
    global local_rank

    parser = HfArgumentParser(
        (DataArguments, TrainingArguments, DistillationArguments)
    )

    data_args, training_args, distillation_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank
    set_local_rank(local_rank)
    compute_dtype = get_compute_dtype(training_args)
    teacher_ids = parse_model_id_list(
        distillation_args.teacher_model_ids,
        arg_name="--teacher_model_ids",
    )
    validate_distillation_args(distillation_args)
    gradient_checkpointing_kwargs = dict(training_args.gradient_checkpointing_kwargs or {})
    if "use_reentrant" not in gradient_checkpointing_kwargs:
        gradient_checkpointing_kwargs["use_reentrant"] = True

    normalize_lora_namespan_exclude(training_args)

    log_distillation_setup(
        teacher_ids=teacher_ids,
        data_args=data_args,
        training_args=training_args,
        distillation_args=distillation_args,
        gradient_checkpointing_kwargs=gradient_checkpointing_kwargs,
    )

    processor, student_model, _ = load_training_model_bundle(
        model_id=distillation_args.student_model_id,
        training_args=training_args,
        compute_dtype=compute_dtype,
        include_load_flags=True,
    )

    rank0_print("Loading student model...")
    configure_training_model(
        model=student_model,
        processor=processor,
        training_args=training_args,
        compute_dtype=compute_dtype,
    )

    student_model = prepare_model_for_low_bit_training(
        model=student_model,
        training_args=training_args,
        gradient_checkpointing_kwargs=gradient_checkpointing_kwargs,
    )
    student_model = maybe_apply_lora(
        model=student_model,
        training_args=training_args,
    )
    finalize_quantized_trainable_modules(
        model=student_model,
        training_args=training_args,
    )

    if (
        distillation_args.teacher_logits_cache_dir is None
        and distillation_args.teacher_logits_remote_uri is None
    ):
        raise ValueError(
            "Teacher logits require either --teacher_logits_cache_dir (for example /cache "
            "or a writable /tmp path) or --teacher_logits_remote_uri (remote raw cache root)."
        )
    rank0_print("\nUsing cached teacher logits; skipping online teacher model loading.")
    teacher_models, teacher_processors = [], []

    student_loss_tokenizer = None
    teacher_loss_tokenizers = None
    if distillation_args.distillation_loss == "trie_wasserstein_loss":
        rank0_print("Preparing trie-Wasserstein tokenizers...")
        student_loss_tokenizer = getattr(processor, "tokenizer", None) or processor
        teacher_loss_tokenizers = [
            load_processor_and_tokenizer_backend(
                teacher_id,
                cache_dir=training_args.cache_dir,
            )[1]
            for teacher_id in teacher_ids
        ]

    rank0_print("\nPreparing datasets...")
    data_module = make_supervised_data_module(
        processor=processor,
        data_args=data_args,
        teacher_processors=teacher_processors,
        teacher_model_ids=teacher_ids,
        teacher_logits_cache_dir=distillation_args.teacher_logits_cache_dir,
        teacher_logits_remote_uri=distillation_args.teacher_logits_remote_uri,
    )
    _init_primary_wandb_run(training_args)

    rank0_print("\nInitializing distillation trainer...")
    trainer_callbacks = build_trainer_callbacks(
        training_args=training_args,
        data_module=data_module,
    )

    trainer = DistillationTrainer(
        model=student_model,
        teacher_model=teacher_models or None,
        teacher_count=len(teacher_ids),
        student_tokenizer=student_loss_tokenizer,
        teacher_tokenizers=teacher_loss_tokenizers,
        teacher_weighting_strategy=distillation_args.teacher_weighting_strategy,
        loss_function=distillation_args.distillation_loss,
        temperature=distillation_args.temperature,
        student_temperature=distillation_args.student_temperature,
        teacher_temperature=distillation_args.teacher_temperature,
        skip_student_eos=distillation_args.skip_student_eos,
        skip_teacher_eos=distillation_args.skip_teacher_eos,
        alpha=distillation_args.alpha,
        teacher_gate_balance_alpha=distillation_args.teacher_gate_balance_alpha,
        teacher_gate_top_k=distillation_args.teacher_gate_top_k,
        teacher_gate_capacity_factor=distillation_args.teacher_gate_capacity_factor,
        teacher_gate_bias_update_rate=distillation_args.teacher_gate_bias_update_rate,
        teacher_gate_temperature=distillation_args.teacher_gate_temperature,
        teacher_gate_noise_std=distillation_args.teacher_gate_noise_std,
        teacher_gate_entropy_alpha=distillation_args.teacher_gate_entropy_alpha,
        teacher_gate_router_z_loss_alpha=distillation_args.teacher_gate_router_z_loss_alpha,
        teacher_gate_hard_routing_warmup_ratio=distillation_args.teacher_gate_hard_routing_warmup_ratio,
        grace_threshold=distillation_args.grace_threshold,
        grace_warmup_ratio=distillation_args.grace_warmup_ratio,
        grace_epsilon=distillation_args.grace_epsilon,
        grace_softmax_beta=distillation_args.grace_softmax_beta,
        grace_router_blend_lambda=distillation_args.grace_router_blend_lambda,
        grace_ema_decay=distillation_args.grace_ema_decay,
        reinforced_selection_warmup_ratio=distillation_args.reinforced_selection_warmup_ratio,
        reinforced_selection_reward_type=distillation_args.reinforced_selection_reward_type,
        reinforced_selection_reward_ema_decay=distillation_args.reinforced_selection_reward_ema_decay,
        reinforced_selection_policy_alpha=distillation_args.reinforced_selection_policy_alpha,
        trie_wasserstein_rho=distillation_args.trie_wasserstein_rho,
        trie_wasserstein_topk=distillation_args.trie_wasserstein_topk,
        processing_class=processor,
        args=training_args,
        callbacks=trainer_callbacks,
        **data_module,
    )

    rank0_print("\n" + "=" * 80)
    rank0_print("Starting distillation training...")
    rank0_print("=" * 80 + "\n")

    if list(Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    if trainer.state.best_model_checkpoint is not None:
        best_metric_name = training_args.metric_for_best_model or "train_ce_loss"
        rank0_print(f"\nLoading best checkpoint based on {best_metric_name}...")
        rank0_print(f"Best checkpoint: {trainer.state.best_model_checkpoint}")
        rank0_print(f"Best {best_metric_name}: {trainer.state.best_metric:.6f}")
        trainer._load_best_model()

    rank0_print("\nSaving trained model...")
    trainer.save_state()
    student_model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            student_model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            student_model.named_parameters(), require_grad_only=True
        )
        if local_rank == 0 or local_rank == -1:
            student_model.config.save_pretrained(training_args.output_dir)
            student_model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            processor.save_pretrained(training_args.output_dir)
            torch.save(
                non_lora_state_dict,
                os.path.join(training_args.output_dir, "non_lora_state_dict.bin"),
            )
    else:
        safe_save_model_for_hf_trainer(
            trainer=trainer,
            output_dir=training_args.output_dir
        )

    rank0_print("\n" + "=" * 80)
    rank0_print("Training completed successfully!")
    rank0_print(f"Model saved to: {training_args.output_dir}")
    rank0_print("=" * 80)


if __name__ == "__main__":
    train_distillation()
