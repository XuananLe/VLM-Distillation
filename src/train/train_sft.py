from pathlib import Path

import torch
from transformers import HfArgumentParser

from src.dataset.sft_data import make_supervised_data_module
from src.params import DataArguments, ModelArguments, TrainingArguments
from src.train.model_setup import load_vlm_bundle
from src.trainer.sft_trainer import SmolVLMSFTTrainer


def set_requires_grad(parameters, requires_grad: bool) -> None:
    for parameter in parameters:
        parameter.requires_grad = requires_grad


def configure_smolvlm_trainable_modules(model, training_args: TrainingArguments) -> None:
    """Keep full SFT as the default while still allowing explicit component freezes."""
    try:
        smolvlm_backbone = model.model
        vision_model = smolvlm_backbone.vision_model
        connector = smolvlm_backbone.connector
        text_model = smolvlm_backbone.text_model
        lm_head = model.lm_head
    except AttributeError as exc:
        raise ValueError(
            "Vanilla SmolVLM SFT expects a SmolVLM/Idefics3-style model with "
            "`model.model.vision_model`, `model.model.connector`, "
            "`model.model.text_model`, and `lm_head`."
        ) from exc

    set_requires_grad(vision_model.parameters(), not training_args.freeze_vision_tower)
    set_requires_grad(connector.parameters(), not training_args.freeze_connector)
    set_requires_grad(text_model.parameters(), not training_args.freeze_llm)
    set_requires_grad(lm_head.parameters(), not training_args.freeze_llm)


def trainable_parameter_summary(model) -> str:
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    percent = 100.0 * trainable / max(total, 1)
    return f"trainable={trainable:,} total={total:,} trainable_percent={percent:.2f}%"


def train():
    """Parse args, build the SFT stack, and run one full supervised fine-tuning job."""
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))

    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if not model_args.model_id:
        raise ValueError("`model_id` must be provided explicitly for SFT training.")

    compute_dtype = torch.float16 if training_args.fp16 else torch.bfloat16 if training_args.bf16 else torch.float32
    model, processor, _, _ = load_vlm_bundle(
        model_id=model_args.model_id,
        cache_dir=training_args.cache_dir,
        device=training_args.device,
        compute_dtype=compute_dtype,
        disable_flash_attn2=training_args.disable_flash_attn2,
    )
    configure_smolvlm_trainable_modules(model, training_args)
    if training_args.local_rank in (-1, 0):
        print(f"Vanilla SmolVLM SFT parameters: {trainable_parameter_summary(model)}")

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}
    model.config.tokenizer_padding_side = processor.tokenizer.padding_side

    data_module = make_supervised_data_module(processor=processor, data_args=data_args)

    trainer = SmolVLMSFTTrainer(model=model, args=training_args, processing_class=processor, **data_module)

    if list(Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()

    model.config.use_cache = True
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    train()
