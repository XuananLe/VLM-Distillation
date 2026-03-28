import os
import sys
import ast
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
from dataclasses import dataclass, field
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModel,
    AutoProcessor,
    AutoTokenizer,
    EarlyStoppingCallback,
    Gemma3ForConditionalGeneration,
    HfArgumentParser,
    AutoModelForVision2Seq,
)
from src.trainer.distillation_trainer import DistillationTrainer
from src.dataset.sft_data import make_supervised_data_module
from src.params import DataArguments, TrainingArguments
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

import pillow_avif
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


@dataclass
class DistillationArguments:
    """Arguments for knowledge distillation."""

    student_model_id: str = field(
        metadata={"help": "Student model ID or path."}
    )

    teacher_model_ids: str = field(
        metadata={"help": "Teacher model IDs as a Python list literal or comma-separated string."}
    )

    distillation_loss: str = field(
        default="uld_loss",
        metadata={
            "help": "KD loss to use. Supported by src/components/loss.py, e.g. uld_loss, forward_kl, reverse_kl, jensen_shannon_divergence."
        },
    )

    temperature: float = field(
        default=2.0,
        metadata={"help": "Temperature for distillation (higher = softer probabilities)"}
    )

    kd_loss_alpha: float = field(
        default=1.0,
        metadata={"help": "Weight for the distillation term in `ce_loss + alpha * kd_loss`."},
    )

    teacher_gate_balance_alpha: float = field(
        default=1e-2,
        metadata={
            "help": "Weight for the Switch-style load-balancing loss on the learned multi-teacher gate."
        },
    )

    teacher_gate_top_k: int = field(
        default=1,
        metadata={"help": "Number of teachers to keep per sample in the hard top-k teacher gate."},
    )

    teacher_gate_capacity_factor: float = field(
        default=1.25,
        metadata={"help": "Capacity factor for hard top-k teacher routing."},
    )

    teacher_gate_bias_update_rate: float = field(
        default=1e-3,
        metadata={
            "help": "Per-step update rate for the loss-free expert-bias rebalancing on the learned multi-teacher gate."
        },
    )

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
    if distillation_args.kd_loss_alpha < 0.0:
        raise ValueError("--kd_loss_alpha must be >= 0.")
    if distillation_args.teacher_gate_balance_alpha < 0.0:
        raise ValueError("--teacher_gate_balance_alpha must be >= 0.")
    if distillation_args.teacher_gate_top_k < 1:
        raise ValueError("--teacher_gate_top_k must be >= 1.")
    if distillation_args.teacher_gate_capacity_factor <= 0.0:
        raise ValueError("--teacher_gate_capacity_factor must be > 0.")
    if distillation_args.teacher_gate_bias_update_rate < 0.0:
        raise ValueError("--teacher_gate_bias_update_rate must be >= 0.")
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

    rank0_print("=" * 80)
    rank0_print("Logits Distillation Training")
    rank0_print("=" * 80)
    rank0_print(f"Student Model: {distillation_args.student_model_id}")
    rank0_print(f"Teacher Model(s): {teacher_ids}")
    rank0_print(
        "Teacher Weighting: learned gate"
        if len(teacher_ids) > 1
        else "Teacher Weighting: uniform mean"
    )
    rank0_print("Objective: CE + alpha * KD")
    rank0_print(f"KD Function: {distillation_args.distillation_loss}")
    rank0_print(f"KD Loss Alpha: {distillation_args.kd_loss_alpha}")
    if len(teacher_ids) > 1:
        rank0_print(f"Teacher Gate Balance Alpha: {distillation_args.teacher_gate_balance_alpha}")
        rank0_print(f"Teacher Gate Top-K: {distillation_args.teacher_gate_top_k}")
        rank0_print(f"Teacher Gate Capacity Factor: {distillation_args.teacher_gate_capacity_factor}")
        rank0_print(f"Teacher Gate Bias Update Rate: {distillation_args.teacher_gate_bias_update_rate}")
    rank0_print(f"Temperature: {distillation_args.temperature}")
    if training_args.gradient_checkpointing:
        rank0_print(f"Gradient Checkpointing Kwargs: {gradient_checkpointing_kwargs}")
    rank0_print("=" * 80)

    attn_impl = "flash_attention_2" if not training_args.disable_flash_attn2 else "eager"

    processor = AutoProcessor.from_pretrained(
        distillation_args.student_model_id,
        padding_side="right",
        trust_remote_code=True,
    )
    teacher_processors = []

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

    teacher_models = []
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

    rank0_print("\nPreparing datasets...")
    data_module = make_supervised_data_module(
        processor=processor,
        data_args=data_args,
        teacher_processors=teacher_processors,
    )

    rank0_print("\nInitializing distillation trainer...")
    trainer_callbacks = []
    if training_args.early_stopping_patience is not None:
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

    trainer = DistillationTrainer(
        model=student_model,
        teacher_model=teacher_models,
        loss_function=distillation_args.distillation_loss,
        temperature=distillation_args.temperature,
        kd_loss_alpha=distillation_args.kd_loss_alpha,
        teacher_gate_balance_alpha=distillation_args.teacher_gate_balance_alpha,
        teacher_gate_top_k=distillation_args.teacher_gate_top_k,
        teacher_gate_capacity_factor=distillation_args.teacher_gate_capacity_factor,
        teacher_gate_bias_update_rate=distillation_args.teacher_gate_bias_update_rate,
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
        best_metric_name = training_args.metric_for_best_model or "loss"
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
