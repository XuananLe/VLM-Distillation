import torch
from transformers import HfArgumentParser
from src.trainer.sft_trainer import VisionLanguageSFTTrainer
from src.dataset.sft_data import make_supervised_data_module
from src.params import DataArguments, ModelArguments, TrainingArguments
from src.train.model_setup import (
    configure_vision_tower,
    load_model,
    load_processor_and_tokenizer,
)
from src.train.save_utils import safe_save_model_for_hf_trainer
import pathlib

import warnings

# Image handling imports
from PIL import Image, ImageFile

# AVIF support initialization
try:
    from pillow_avif import register_avif_opener
    register_avif_opener()
    AVIF_SUPPORT = True 
except ImportError:
    AVIF_SUPPORT = False
    warnings.warn("AVIF support disabled. Install pillow-avif-plugin for AVIF support.")

# Configure image loading
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

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
    if processor is None:
        raise ValueError(
            "Training requires an AutoProcessor, but processor loading failed for "
            f"{model_args.model_id!r}."
        )
    model = load_model(
        model_id=model_args.model_id,
        model_type=model_type,
        cache_dir=training_args.cache_dir,
        attn_implementation="flash_attention_2" if not training_args.disable_flash_attn2 else "eager",
        compute_dtype=compute_dtype,
        trust_remote_code=True,
        model_kwargs={"device_map": {"": training_args.device}},
    )

    configure_vision_tower(model, processor, compute_dtype, training_args.device)
    model.config.use_cache = False

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}
    # model.config.tokenizer_model_max_length = processor.tokenizer.model_max_length
    model.config.tokenizer_padding_side = processor.tokenizer.padding_side
    model.config.vision_lr = training_args.vision_lr

    data_module = make_supervised_data_module(processor=processor,
                                              data_args=data_args)

    trainer = VisionLanguageSFTTrainer(
        model=model,
        args=training_args,
        **data_module
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()

    model.config.use_cache = True
    
    safe_save_model_for_hf_trainer(trainer, output_dir=training_args.output_dir)



if __name__ == "__main__":
    train()
