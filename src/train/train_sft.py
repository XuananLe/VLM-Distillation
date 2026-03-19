import os
import torch
from peft import LoraConfig, get_peft_model
import ast
from transformers import AutoProcessor, HfArgumentParser, AutoModelForVision2Seq
from src.trainer.sft_trainer import SmolVLMSFTTrainer
from src.dataset.sft_data import make_supervised_data_module
from src.params import DataArguments, ModelArguments, TrainingArguments
from src.train.train_utils import (
    get_peft_state_maybe_zero_3,
    get_peft_state_non_lora_maybe_zero_3,
    safe_save_model_for_hf_trainer,
    get_compute_dtype,
    set_local_rank,
    rank0_print,
    find_target_linear_names,
    configure_vision_tower,
    configure_llm,
    unfreeze_topk_layers,
    build_model_from_pretrained_args,
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

    if training_args.lora_enable and not training_args.freeze_llm:
        raise ValueError("If `lora_enable` is True, `freeze_llm` must also be True.")

    if not training_args.lora_enable:
        assert not training_args.vision_lora, \
            "Error: training_args.lora_enable is not enabled, but training_args.vision_lora is enabled."

    if training_args.vision_lora and not training_args.freeze_vision_tower:
        raise ValueError("If `vision_lora` is True, `freeze_vision_tower` must also be True.")

    if training_args.lora_enable:
        if training_args.lora_namespan_exclude is not None:
            training_args.lora_namespan_exclude = ast.literal_eval(training_args.lora_namespan_exclude)
        else:
            training_args.lora_namespan_exclude = []

        if not training_args.vision_lora:
            training_args.lora_namespan_exclude += ["vision_model"]

    local_rank = training_args.local_rank
    set_local_rank(local_rank)
    compute_dtype = get_compute_dtype(training_args)

    processor = AutoProcessor.from_pretrained(model_args.model_id,
                                            padding_side="right")

    model_from_pretrained_args = build_model_from_pretrained_args(
        training_args,
        compute_dtype,
        llm_int8_skip_modules=["vision_model", "connector"],
    )

    model = AutoModelForVision2Seq.from_pretrained(
        model_args.model_id,
        torch_dtype=compute_dtype,
        attn_implementation="flash_attention_2" if not training_args.disable_flash_attn2 else "eager",
        **model_from_pretrained_args
    )

    configure_llm(model, training_args)
    configure_vision_tower(model, processor, training_args, compute_dtype, training_args.device)

    unfreeze_topk_layers(
        model,
        k_llm=getattr(training_args, "unfreeze_topk_llm", 0),
        k_vis=getattr(training_args, "unfreeze_topk_vision", 0),
    )

    model.config.use_cache = False

    if training_args.bits in [4,8]:
        model.config.torch_dtype = (torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing, gradient_checkpointing_kwargs={"use_reentrant": True})
    
    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}

    if training_args.lora_enable:
        lora_namespan_exclude = training_args.lora_namespan_exclude
        peft_config = LoraConfig(
            r=training_args.lora_rank,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_target_linear_names(model, lora_namespan_exclude=lora_namespan_exclude, num_lora_modules=training_args.num_lora_modules),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA to the model...")
        model = get_peft_model(model, peft_config)

        
        if not training_args.freeze_vision_tower:
            for name, param in model.named_parameters():
                if "vision_model" in name:
                    param.requires_grad = True

        if not training_args.freeze_connector:
            for name, param in model.named_parameters():
                if "connector" in name:
                    param.requires_grad = True

    # model.config.tokenizer_model_max_length = processor.tokenizer.model_max_length
    model.config.tokenizer_padding_side = processor.tokenizer.padding_side
    model.config.vision_lr = training_args.vision_lr

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            
            if 'lm_head' in name or 'embed_token' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    data_module = make_supervised_data_module(processor=processor,
                                              data_args=data_args)

    trainer = SmolVLMSFTTrainer(
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
