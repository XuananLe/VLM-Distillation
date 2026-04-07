import os
import sys
import ast
import importlib
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoProcessor,
    HfArgumentParser,
    AutoModelForVision2Seq,
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
    safe_save_model_for_hf_trainer,
    get_compute_dtype,
    get_peft_state_maybe_zero_3,
    get_peft_state_non_lora_maybe_zero_3,
    set_local_rank,
    rank0_print,
    find_target_linear_names,
    configure_vision_tower,
    configure_llm,
    unfreeze_topk_layers,
    log_trainable_parameter_summary,
    build_model_from_pretrained_args,
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
    if training_args.run_name is not None:
        init_kwargs["name"] = training_args.run_name
    wandb.init(**init_kwargs)


def train_distillation():
    """
    Main training function for VLM distillation.

    This script supports CE + ULD distillation from one or more frozen teachers.
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

    if training_args.lora_enable:
        training_args.lora_namespan_exclude = (
            ast.literal_eval(training_args.lora_namespan_exclude)
            if training_args.lora_namespan_exclude is not None
            else []
        )
        if not training_args.vision_lora:
            training_args.lora_namespan_exclude += ["vision_model"]

    log_distillation_setup(
        teacher_ids=teacher_ids,
        data_args=data_args,
        training_args=training_args,
        distillation_args=distillation_args,
        gradient_checkpointing_kwargs=gradient_checkpointing_kwargs,
    )

    attn_impl = "flash_attention_2" if not training_args.disable_flash_attn2 else "eager"

    processor = AutoProcessor.from_pretrained(
        distillation_args.student_model_id,
        padding_side="right",
        trust_remote_code=True,
    )
    model_kwargs = build_model_from_pretrained_args(
        training_args,
        compute_dtype,
        include_load_flags=True,
    )

    rank0_print("Loading student model...")
    student_model = AutoModelForVision2Seq.from_pretrained(
        distillation_args.student_model_id,
        cache_dir=training_args.cache_dir,
        attn_implementation=attn_impl,
        torch_dtype=compute_dtype,
        trust_remote_code=True,
        **model_kwargs,
    )

    configure_vision_tower(
        student_model,
        processor,
        training_args,
        compute_dtype,
        training_args.device,
    )
    configure_llm(student_model, training_args)

    if training_args.unfreeze_topk_llm > 0 or training_args.unfreeze_topk_vision > 0:
        unfreeze_topk_layers(
            student_model,
            k_llm=training_args.unfreeze_topk_llm,
            k_vis=training_args.unfreeze_topk_vision,
        )
    student_model.config.use_cache = False

    if training_args.bits in [4, 8]:
        student_model.config.torch_dtype = (
            torch.float32
            if training_args.fp16
            else (torch.bfloat16 if training_args.bf16 else torch.float32)
        )
        from peft import prepare_model_for_kbit_training

        student_model = prepare_model_for_kbit_training(
            student_model,
            use_gradient_checkpointing=training_args.gradient_checkpointing,
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs,
        )

    if training_args.gradient_checkpointing:
        student_model.enable_input_require_grads()
        training_args.gradient_checkpointing_kwargs = gradient_checkpointing_kwargs

    if training_args.lora_enable:
        lora_namespan_exclude = training_args.lora_namespan_exclude
        lora_target_modules = find_target_linear_names(
            student_model,
            lora_namespan_exclude=lora_namespan_exclude,
            num_lora_modules=training_args.num_lora_modules,
        )
        peft_config = LoraConfig(
            r=training_args.lora_rank,
            lora_alpha=training_args.lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            use_dora=training_args.use_dora,
        )
        if training_args.bits == 16:
            if training_args.bf16:
                student_model.to(torch.bfloat16)
            if training_args.fp16:
                student_model.to(torch.float16)
        student_model = get_peft_model(student_model, peft_config)
        log_trainable_parameter_summary(
            student_model,
            "Trainable parameters after PEFT wrapping:",
        )

        if not training_args.freeze_vision_tower:
            for name, param in student_model.named_parameters():
                if "vision_model" in name:
                    param.requires_grad = True

        if not training_args.freeze_connector:
            for name, param in student_model.named_parameters():
                if "connector" in name:
                    param.requires_grad = True

        if training_args.bits in [4, 8]:
            from peft.tuners.lora import LoraLayer

            for name, module in student_model.named_modules():
                if isinstance(module, LoraLayer) and training_args.bf16:
                    module = module.to(torch.bfloat16)
                if "norm" in name:
                    module = module.to(torch.float32)
                if ("lm_head" in name or "embed_token" in name) and hasattr(module, "weight"):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    if distillation_args.teacher_logits_cache_dir is None:
        raise ValueError(
            "--teacher_logits_cache_dir must be set. Online teacher loading is disabled; "
            "distillation always uses cached teacher logits."
        )
    rank0_print("\nUsing cached teacher logits; skipping online teacher model loading.")
    teacher_models, teacher_processors = [], []

    rank0_print("\nPreparing datasets...")
    data_module = make_supervised_data_module(
        processor=processor,
        data_args=data_args,
        teacher_processors=teacher_processors,
        teacher_model_ids=teacher_ids,
        teacher_logits_cache_dir=distillation_args.teacher_logits_cache_dir,
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
        teacher_weighting_strategy=distillation_args.teacher_weighting_strategy,
        objective_conflict_strategy=distillation_args.objective_conflict_strategy,
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
        gradient_weight_cap=distillation_args.gradient_weight_cap,
        gradient_weight_steps=distillation_args.gradient_weight_steps,
        objective_conflict_cagrad_c=distillation_args.objective_conflict_cagrad_c,
        objective_conflict_cagrad_grid_steps=distillation_args.objective_conflict_cagrad_grid_steps,
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
