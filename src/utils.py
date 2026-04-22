from peft import PeftModel
import torch
from transformers import AutoConfig, BitsAndBytesConfig
import warnings
import os

from src.train.model_setup import (
    load_processor_and_tokenizer_backend,
    load_vision_language_model,
    resolve_model_type,
)

def create_quantization_config(load_4bit=True, compute_dtype=torch.float16,
                             use_double_quant=True, quant_type='nf4'):
    """
    Create a standard BitsAndBytesConfig for 4-bit quantization.

    Args:
        load_4bit: Whether to load in 4-bit
        compute_dtype: Data type for computation
        use_double_quant: Whether to use double quantization
        quant_type: Quantization type (nf4, fp4)

    Returns:
        BitsAndBytesConfig: Configured quantization config
    """
    return BitsAndBytesConfig(
        load_in_4bit=load_4bit,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_double_quant,
        bnb_4bit_quant_type=quant_type
    )

# This code is borrowed from LLaVA
def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, 
                          device_map="auto", device="cuda", use_flash_attn=False, **kwargs):
    kwargs = dict(kwargs)
    kwargs["device_map"] = device_map
    
    if device != "cuda":
        kwargs['device_map'] = {"":device}
    
    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['quantization_config'] = create_quantization_config()
    else:
        kwargs['torch_dtype'] = torch.float16

    attn_implementation = 'flash_attention_2' if use_flash_attn else 'eager'
    cache_dir = kwargs.pop("cache_dir", None)
    processor_source = model_base or model_path
    processor, _, _ = load_processor_and_tokenizer_backend(
        processor_source,
        cache_dir=cache_dir,
    )
    if processor is None:
        raise ValueError(
            f"Could not load an AutoProcessor for multimodal model {processor_source!r}."
        )

    if 'lora' in model_name.lower() and model_base is None:
        warnings.warn('There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument.')
    if 'lora' in model_name.lower() and model_base is not None:
        lora_cfg_pretrained = AutoConfig.from_pretrained(model_path)
        if hasattr(lora_cfg_pretrained, 'quantization_config'):
            del lora_cfg_pretrained.quantization_config
        print('Loading base vision-language model...')
        model = load_vision_language_model(
            model_id=model_base,
            model_type=resolve_model_type(model_base),
            cache_dir=cache_dir,
            attn_implementation=attn_implementation,
            compute_dtype=kwargs.get("torch_dtype", torch.float16),
            trust_remote_code=True,
            model_kwargs={
                **kwargs,
                "low_cpu_mem_usage": True,
                "config": lora_cfg_pretrained,
            },
        )
        token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
        if model.lm_head.weight.shape[0] != token_num:
            model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
            model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

        print('Loading additional non-LoRA weights...')
        non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_state_dict.bin'), map_location='cpu')
        non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
        if any(k.startswith('model.model.') for k in non_lora_trainables):
            non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
        model.load_state_dict(non_lora_trainables, strict=False)
    
        print('Loading LoRA weights...')
        model = PeftModel.from_pretrained(model, model_path)

        print('Merging LoRA weights...')
        model = model.merge_and_unload()

        print('Model Loaded!!!')

    else:
        model = load_vision_language_model(
            model_id=model_path,
            model_type=resolve_model_type(model_path),
            cache_dir=cache_dir,
            attn_implementation=attn_implementation,
            compute_dtype=kwargs.get("torch_dtype", torch.float16),
            trust_remote_code=True,
            model_kwargs={
                **kwargs,
                "low_cpu_mem_usage": True,
            },
        )

    return processor, model


def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]
