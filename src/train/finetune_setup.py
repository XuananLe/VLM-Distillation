import ast

import torch
from peft import LoraConfig, get_peft_model

from src.train.log_utils import rank0_print
from src.train.model_setup import (
    build_llm_int8_skip_modules,
    build_model_from_pretrained_args,
    configure_llm,
    configure_vision_tower,
    extend_lora_namespan_exclude,
    find_target_linear_names,
    load_processor_and_tokenizer_backend,
    load_vision_language_model,
    log_trainable_parameter_summary,
    set_component_requires_grad,
    unfreeze_topk_layers,
)


def normalize_lora_namespan_exclude(training_args) -> None:
    if not training_args.lora_enable:
        return
    training_args.lora_namespan_exclude = (
        ast.literal_eval(training_args.lora_namespan_exclude)
        if training_args.lora_namespan_exclude is not None
        else []
    )
    training_args.lora_namespan_exclude = extend_lora_namespan_exclude(
        training_args.lora_namespan_exclude,
        exclude_components=("vision",) if not training_args.vision_lora else (),
    )


def load_training_model_bundle(
    *,
    model_id: str,
    training_args,
    compute_dtype: torch.dtype,
    padding_side: str = "right",
    include_load_flags: bool = False,
):
    processor, _, model_type = load_processor_and_tokenizer_backend(
        model_id,
        padding_side=padding_side,
        cache_dir=training_args.cache_dir,
    )
    if processor is None:
        raise ValueError(
            "Training requires an AutoProcessor, but processor loading failed for "
            f"{model_id!r}."
        )

    model_kwargs = build_model_from_pretrained_args(
        training_args,
        compute_dtype,
        llm_int8_skip_modules=build_llm_int8_skip_modules("vision", "connector"),
        include_load_flags=include_load_flags,
    )
    model = load_vision_language_model(
        model_id=model_id,
        model_type=model_type,
        cache_dir=training_args.cache_dir,
        attn_implementation="flash_attention_2" if not training_args.disable_flash_attn2 else "eager",
        compute_dtype=compute_dtype,
        trust_remote_code=True,
        model_kwargs=model_kwargs,
    )
    return processor, model, model_type


def configure_training_model(
    *,
    model,
    processor,
    training_args,
    compute_dtype: torch.dtype,
):
    configure_llm(model, training_args)
    configure_vision_tower(model, processor, training_args, compute_dtype, training_args.device)
    unfreeze_topk_layers(
        model,
        k_llm=getattr(training_args, "unfreeze_topk_llm", 0),
        k_vis=getattr(training_args, "unfreeze_topk_vision", 0),
    )
    model.config.use_cache = False


def prepare_model_for_low_bit_training(
    *,
    model,
    training_args,
    gradient_checkpointing_kwargs: dict,
):
    if training_args.bits in [4, 8]:
        model.config.torch_dtype = (
            torch.float32
            if training_args.fp16
            else (torch.bfloat16 if training_args.bf16 else torch.float32)
        )
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=training_args.gradient_checkpointing,
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs,
        )

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()
        training_args.gradient_checkpointing_kwargs = gradient_checkpointing_kwargs

    return model


def maybe_apply_lora(
    *,
    model,
    training_args,
):
    if not training_args.lora_enable:
        return model

    peft_config = LoraConfig(
        r=training_args.lora_rank,
        lora_alpha=training_args.lora_alpha,
        target_modules=find_target_linear_names(
            model,
            lora_namespan_exclude=training_args.lora_namespan_exclude,
            num_lora_modules=training_args.num_lora_modules,
        ),
        lora_dropout=training_args.lora_dropout,
        bias=training_args.lora_bias,
        use_dora=getattr(training_args, "use_dora", False),
    )
    if training_args.bits == 16:
        if training_args.bf16:
            model.to(torch.bfloat16)
        if training_args.fp16:
            model.to(torch.float16)

    rank0_print("Adding LoRA to the model...")
    model = get_peft_model(model, peft_config)
    log_trainable_parameter_summary(
        model,
        "Trainable parameters after PEFT wrapping:",
    )

    if not training_args.freeze_vision_tower:
        set_component_requires_grad(model, "vision", True)

    if not training_args.freeze_connector:
        set_component_requires_grad(model, "connector", True)

    return model


def finalize_quantized_trainable_modules(
    *,
    model,
    training_args,
):
    if training_args.bits not in [4, 8]:
        return

    from peft.tuners.lora import LoraLayer

    for name, module in model.named_modules():
        if isinstance(module, LoraLayer) and training_args.bf16:
            module.to(torch.bfloat16)
        if "norm" in name:
            module.to(torch.float32)
        if ("lm_head" in name or "embed_token" in name) and hasattr(module, "weight"):
            if training_args.bf16 and module.weight.dtype == torch.float32:
                module.to(torch.bfloat16)
