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
from src.trainer.setup_utils import normalize_teacher_models
from src.dataset.sft_data import make_supervised_data_module
from src.params import DataArguments, TrainingArguments
from src.train.distillation_setup import (
    DistillationArguments,
    log_distillation_setup,
    validate_distillation_args,
)
from src.train.model_setup import (
    configure_vision_tower,
    load_model,
    load_processor_and_tokenizer,
    load_teacher_model_and_processor,
)
from src.train.save_utils import safe_save_model_for_hf_trainer

from PIL import Image, ImageFile

importlib.import_module("pillow_avif")

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

def train_distillation():
    """
    Parse args, load models/data, and run one distillation training job.
    """
    parser = HfArgumentParser(
        (DataArguments, TrainingArguments, DistillationArguments)
    )

    data_args, training_args, distillation_args = parser.parse_args_into_dataclasses()

    compute_dtype = (
        torch.float16 if training_args.fp16
        else torch.bfloat16 if training_args.bf16
        else torch.float32
    )
    teacher_ids = list(distillation_args.teacher_model_ids)
    student_layer_indices = list(distillation_args.student_layer_indices)
    teacher_layer_indices = list(distillation_args.teacher_layer_indices)
    validate_distillation_args(distillation_args)
    if (
        distillation_args.layer_distill_source in {"vision", "model"}
        and distillation_args.layer_distill_weight > 0.0
        and (student_layer_indices or distillation_args.layer_match_json_path)
    ):
        # Layer distillation is only "on" when the user asked for a real hidden-state
        # target, gave it non-zero weight, and provided some student-side layer spec.
        layer_distillation_enabled = True
    else:
        layer_distillation_enabled = False
    gradient_checkpointing_kwargs = dict(training_args.gradient_checkpointing_kwargs or {})
    if "use_reentrant" not in gradient_checkpointing_kwargs:
        gradient_checkpointing_kwargs["use_reentrant"] = True

    log_distillation_setup(
        teacher_ids=teacher_ids,
        student_layer_indices=student_layer_indices,
        teacher_layer_indices=teacher_layer_indices,
        data_args=data_args,
        training_args=training_args,
        distillation_args=distillation_args,
        gradient_checkpointing_kwargs=gradient_checkpointing_kwargs,
    )

    processor, _, student_model_type = load_processor_and_tokenizer(
        distillation_args.student_model_id,
        padding_side="right",
        cache_dir=training_args.cache_dir,
    )
    if processor is None:
        raise ValueError(
            "Training requires an AutoProcessor, but processor loading failed for "
            f"{distillation_args.student_model_id!r}."
        )
    student_model = load_model(
        model_id=distillation_args.student_model_id,
        model_type=student_model_type,
        cache_dir=training_args.cache_dir,
        attn_implementation="flash_attention_2" if not training_args.disable_flash_attn2 else "eager",
        compute_dtype=compute_dtype,
        trust_remote_code=True,
        model_kwargs={"device_map": {"": training_args.device}},
    )

    print("Loading student model...")
    configure_vision_tower(student_model, processor, compute_dtype, training_args.device)
    student_model.config.use_cache = False

    if training_args.gradient_checkpointing:
        student_model.enable_input_require_grads()
        training_args.gradient_checkpointing_kwargs = gradient_checkpointing_kwargs

    if layer_distillation_enabled:
        print(
            "\nLayer distillation enabled; loading live teacher models in addition to cached teacher logits."
        )
        teacher_models = []
        teacher_processors = []
        for teacher_id in teacher_ids:
            print(f"Loading live teacher for layer distillation: {teacher_id}")
            teacher_model, teacher_processor = load_teacher_model_and_processor(
                model_id=teacher_id,
                cache_dir=training_args.cache_dir,
                device=training_args.device,
                compute_dtype=compute_dtype,
                disable_flash_attn2=training_args.disable_flash_attn2,
            )
            teacher_models.append(teacher_model)
            teacher_processors.append(teacher_processor)
        teacher_models, _ = normalize_teacher_models(teacher_models, len(teacher_models))
    else:
        print("\nUsing cached teacher logits; skipping online teacher model loading.")
        teacher_models, teacher_processors = [], []

    student_loss_tokenizer = None
    teacher_loss_tokenizers = None
    if distillation_args.distillation_loss == "trie_wasserstein_loss":
        print("Preparing trie-Wasserstein tokenizers...")
        student_loss_tokenizer = getattr(processor, "tokenizer", None) or processor
        teacher_loss_tokenizers = [
            load_processor_and_tokenizer(
                teacher_id,
                cache_dir=training_args.cache_dir,
            )[1]
            for teacher_id in teacher_ids
        ]

    print("\nPreparing datasets...")
    data_module = make_supervised_data_module(
        processor=processor,
        data_args=data_args,
        teacher_processors=teacher_processors,
        teacher_model_ids=teacher_ids,
        teacher_logits_cache_dir=distillation_args.teacher_logits_cache_dir,
        teacher_logits_remote_uri=distillation_args.teacher_logits_remote_uri,
    )
    print("\nInitializing distillation trainer...")
    trainer = DistillationTrainer(
        model=student_model,
        teacher_model=teacher_models or None,
        teacher_count=len(teacher_ids),
        student_tokenizer=student_loss_tokenizer,
        teacher_tokenizers=teacher_loss_tokenizers,
        teacher_weighting_strategy=distillation_args.teacher_weighting_strategy,
        loss_function=distillation_args.distillation_loss,
        layer_distill_source=distillation_args.layer_distill_source,
        layer_distill_weight=distillation_args.layer_distill_weight,
        layer_match_json_path=distillation_args.layer_match_json_path,
        layer_match_topk=distillation_args.layer_match_topk,
        student_layer_indices=student_layer_indices,
        teacher_layer_indices=teacher_layer_indices,
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
        **data_module,
    )

    print("\n" + "=" * 80)
    print("Starting distillation training...")
    print("=" * 80 + "\n")

    if list(Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    if trainer.state.best_model_checkpoint is not None:
        best_metric_name = training_args.metric_for_best_model or "train_ce_loss"
        print(f"\nLoading best checkpoint based on {best_metric_name}...")
        print(f"Best checkpoint: {trainer.state.best_model_checkpoint}")
        print(f"Best {best_metric_name}: {trainer.state.best_metric:.6f}")
        trainer._load_best_model()

    print("\nSaving trained model...")
    trainer.save_state()
    student_model.config.use_cache = True

    safe_save_model_for_hf_trainer(
        trainer=trainer,
        output_dir=training_args.output_dir
    )

    print("\n" + "=" * 80)
    print("Training completed successfully!")
    print(f"Model saved to: {training_args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    train_distillation()
