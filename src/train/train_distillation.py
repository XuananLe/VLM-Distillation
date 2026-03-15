import torch
from dataclasses import dataclass, field
from typing import Optional
from transformers import (
    AutoProcessor,
    HfArgumentParser,
    AutoModelForVision2Seq,
    BitsAndBytesConfig
)
from src.trainer.distillation_trainer import LogitsDistillationTrainer
from src.dataset.sft_data import make_supervised_data_module
from src.params import ModelArguments, DataArguments, TrainingArguments
from src.train.train_sft import (
    configure_vision_tower,
    configure_llm,
    unfreeze_topk_layers,
    rank0_print,
)
from src.train.train_utils import safe_save_model_for_hf_trainer, get_compute_dtype

import pillow_avif
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

@dataclass
class DistillationArguments:
    """Arguments for knowledge distillation."""

    teacher_model_id: str = field(
        metadata={"help": "The model ID or path for the teacher model"}
    )

    distillation_loss: str = field(
        default="forward_kl",
        metadata={
            "help": "Type of distillation loss to use. Options: forward_kl, reverse_kl, jensen_shannon_divergence, ofa_loss, uld_loss"
        }
    )

    temperature: float = field(
        default=2.0,
        metadata={"help": "Temperature for distillation (higher = softer probabilities)"}
    )

    alpha: float = field(
        default=0.5,
        metadata={
            "help": "Weight for distillation loss vs cross-entropy loss. "
                   "alpha=1.0 means only distillation, alpha=0.0 means only CE"
        }
    )

    ofa_eps: float = field(
        default=1.0,
        metadata={
            "help": "Adaptive target enhancement exponent for OFA-KD. Only used when --distillation_loss ofa_loss."
        }
    )


def _build_model_from_pretrained_args(training_args, compute_dtype):
    model_kwargs = {
        "device_map": {"": training_args.device},
    }
    if training_args.bits not in [4, 8]:
        return model_kwargs

    model_kwargs.update(
        dict(
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type,
            ),
        )
    )
    return model_kwargs


def train_distillation():
    """
    Main training function for VLM distillation.

    This script supports:
    - Knowledge distillation from a larger teacher model to a smaller student model
    - Multiple distillation loss functions (forward KL, reverse KL, Jensen-Shannon)
    - Mixed precision training (fp16/bf16)
    """
    global local_rank

    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, DistillationArguments)
    )

    model_args, data_args, training_args, distillation_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank
    compute_dtype = get_compute_dtype(training_args)

    rank0_print("=" * 80)
    rank0_print("Logits Distillation Training")
    rank0_print("=" * 80)
    rank0_print(f"Student Model: {model_args.model_id}")
    rank0_print(f"Teacher Model: {distillation_args.teacher_model_id}")
    rank0_print(f"Distillation Loss: {distillation_args.distillation_loss}")
    rank0_print(f"Temperature: {distillation_args.temperature}")
    rank0_print(f"Alpha: {distillation_args.alpha}")
    if distillation_args.distillation_loss == "ofa_loss":
        rank0_print(f"OFA eps: {distillation_args.ofa_eps}")
    rank0_print("=" * 80)

    attn_impl = "flash_attention_2" if not training_args.disable_flash_attn2 else "eager"

    processor = AutoProcessor.from_pretrained(
        model_args.model_id,
        padding_side="right",
        trust_remote_code=True,
    )
    teacher_processor = AutoProcessor.from_pretrained(
        distillation_args.teacher_model_id,
        padding_side="right",
        trust_remote_code=True,
    )

    model_kwargs = _build_model_from_pretrained_args(training_args, compute_dtype)

    rank0_print("Loading student model...")
    student_model = AutoModelForVision2Seq.from_pretrained(
        model_args.model_id,
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

    rank0_print("\nLoading teacher model...")
    teacher_model = AutoModelForVision2Seq.from_pretrained(
        distillation_args.teacher_model_id,
        cache_dir=training_args.cache_dir,
        attn_implementation=attn_impl,
        torch_dtype=compute_dtype,
        trust_remote_code=True,
        device_map={"": training_args.device},
    )
    teacher_model.config.use_cache = False
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False
    rank0_print("Teacher model loaded and frozen")
    torch.cuda.empty_cache()

    rank0_print("\nPreparing datasets...")
    data_module = make_supervised_data_module(
        processor=processor,
        data_args=data_args,
        teacher_processor=teacher_processor,
    )

    rank0_print("\nInitializing distillation trainer...")
    trainer = LogitsDistillationTrainer(
        model=student_model,
        teacher_model=teacher_model,
        loss_function=distillation_args.distillation_loss,
        temperature=distillation_args.temperature,
        alpha=distillation_args.alpha,
        ofa_eps=distillation_args.ofa_eps,
        args=training_args,
        **data_module,
    )

    rank0_print("\n" + "=" * 80)
    rank0_print("Starting distillation training...")
    rank0_print("=" * 80 + "\n")

    trainer.train()

    rank0_print("\nSaving trained model...")
    trainer.save_state()

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
