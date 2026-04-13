import os
import torch
from transformers import HfArgumentParser
from src.trainer.sft_trainer import VisionLanguageSFTTrainer
from src.dataset.sft_data import make_supervised_data_module
from src.params import DataArguments, ModelArguments, TrainingArguments
from src.train.train_utils import (
    configure_training_model,
    finalize_quantized_trainable_modules,
    get_peft_state_maybe_zero_3,
    get_peft_state_non_lora_maybe_zero_3,
    safe_save_model_for_hf_trainer,
    get_compute_dtype,
    load_training_model_bundle,
    maybe_apply_lora,
    normalize_lora_namespan_exclude,
    prepare_model_for_low_bit_training,
    set_local_rank,
    rank0_print,
)
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

def validate_image_files(data_args):
    """Check image formats and warn about unsupported types"""
    valid_extensions = {'.avif', '.jpg', '.jpeg', '.png', '.webp'}
    invalid_files = []
    
    for img_file in pathlib.Path(data_args.image_folder).rglob('*'):
        if img_file.suffix.lower() not in valid_extensions:
            invalid_files.append(img_file)
    
    if invalid_files:
        warning_msg = f"Found {len(invalid_files)} files with unsupported extensions:\n"
        warning_msg += "\n".join(str(f) for f in invalid_files[:5])
        if len(invalid_files) > 5:
            warning_msg += f"\n...and {len(invalid_files)-5} more"
        
        if data_args.strict_image_validation:
            raise ValueError(f"Invalid image formats detected:\n{warning_msg}")
        else:
            rank0_print(f"WARNING: {warning_msg}")
            rank0_print("Skipping invalid files as strict_image_validation=False")

def train():
    global local_rank

    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if data_args.image_folder:
        validate_image_files(data_args)

    if not model_args.model_id:
        raise ValueError("`model_id` must be provided explicitly for SFT training.")

    if training_args.lora_enable and not training_args.freeze_llm:
        raise ValueError("If `lora_enable` is True, `freeze_llm` must also be True.")

    if not training_args.lora_enable:
        assert not training_args.vision_lora, \
            "Error: training_args.lora_enable is not enabled, but training_args.vision_lora is enabled."

    if training_args.vision_lora and not training_args.freeze_vision_tower:
        raise ValueError("If `vision_lora` is True, `freeze_vision_tower` must also be True.")

    normalize_lora_namespan_exclude(training_args)

    local_rank = training_args.local_rank
    set_local_rank(local_rank)
    compute_dtype = get_compute_dtype(training_args)

    processor, model, _ = load_training_model_bundle(
        model_id=model_args.model_id,
        training_args=training_args,
        compute_dtype=compute_dtype,
        include_load_flags=False,
    )

    configure_training_model(
        model=model,
        processor=processor,
        training_args=training_args,
        compute_dtype=compute_dtype,
    )

    model = prepare_model_for_low_bit_training(
        model=model,
        training_args=training_args,
        gradient_checkpointing_kwargs={"use_reentrant": True},
    )
    model = maybe_apply_lora(
        model=model,
        training_args=training_args,
    )

    # model.config.tokenizer_model_max_length = processor.tokenizer.model_max_length
    model.config.tokenizer_padding_side = processor.tokenizer.padding_side
    model.config.vision_lr = training_args.vision_lr

    finalize_quantized_trainable_modules(
        model=model,
        training_args=training_args,
    )

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
    
    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )

        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters(), require_grad_only=True
        )

        if local_rank == 0 or local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            processor.save_pretrained(training_args.output_dir)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, "non_lora_state_dict.bin"))
    else:
        safe_save_model_for_hf_trainer(trainer, output_dir=training_args.output_dir)



if __name__ == "__main__":
    train()
