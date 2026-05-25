from pathlib import Path

import torch
from transformers import HfArgumentParser, Trainer

from src.dataset.sft_data import make_supervised_data_module
from src.params import DataArguments, ModelArguments, TrainingArguments
from src.train.model_setup import load_vlm_components

def train():
    """Parse args, build the SFT stack, and run one full supervised fine-tuning job."""
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if not model_args.model_id:
        raise ValueError("`model_id` must be provided explicitly for SFT training.")

    compute_dtype = (
        torch.float16 if training_args.fp16
        else torch.bfloat16 if training_args.bf16
        else torch.float32
    )
    model, processor, _, _ = load_vlm_components(
        model_id=model_args.model_id,
        cache_dir=training_args.cache_dir,
        device=training_args.device,
        compute_dtype=compute_dtype,
        disable_flash_attn2=training_args.disable_flash_attn2,
    )

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}
    model.config.tokenizer_padding_side = processor.tokenizer.padding_side

    data_module = make_supervised_data_module(processor=processor,
                                              data_args=data_args)

    trainer = Trainer(
        model=model,
        args=training_args,
        processing_class=processor,
        **data_module
    )

    if list(Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()

    model.config.use_cache = True
    trainer.save_model(training_args.output_dir)



if __name__ == "__main__":
    train()
