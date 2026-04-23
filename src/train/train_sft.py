from pathlib import Path

import torch
from transformers import HfArgumentParser, Trainer

from src.dataset.sft_data import make_supervised_data_module
from src.params import DataArguments, ModelArguments, TrainingArguments
from src.train.model_setup import (
    load_model,
    load_processor_and_tokenizer,
)
from src.train.save_utils import safe_save_model_for_hf_trainer

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
    processor, _, model_type = load_processor_and_tokenizer(
        model_args.model_id,
        padding_side="right",
        cache_dir=training_args.cache_dir,
    )
    model = load_model(
        model_id=model_args.model_id,
        model_type=model_type,
        cache_dir=training_args.cache_dir,
        attn_implementation="flash_attention_2" if not training_args.disable_flash_attn2 else "eager",
        compute_dtype=compute_dtype,
        model_kwargs={"device_map": {"": training_args.device}},
    )

    model.config.use_cache = False

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}
    model.config.tokenizer_padding_side = processor.tokenizer.padding_side

    data_module = make_supervised_data_module(processor=processor,
                                              data_args=data_args)

    trainer = Trainer(
        model=model,
        args=training_args,
        **data_module
    )

    if list(Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()

    model.config.use_cache = True
    
    safe_save_model_for_hf_trainer(trainer, output_dir=training_args.output_dir)



if __name__ == "__main__":
    train()
