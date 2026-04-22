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


def resolve_module_path(module, path):
    current = module
    if not path:
        return current

    for part in path.split("."):
        if part.isdigit():
            current = current[int(part)]
        else:
            current = getattr(current, part)
    return current


def detect_architecture(model):
    model_name = model.__class__.__name__.lower()

    if "llava" in model_name:
        return "llava"
    if "qwen" in model_name:
        return "qwen-vl"
    if "intern" in model_name:
        return "internvl"
    if "clip" in model_name or "siglip" in model_name:
        return "clip"
    return "generic"


def find_nested_vision_encoder(model):
    best_match = (None, None, 0)
    for name, module in model.named_modules():
        lowered_name = name.lower()
        if not any(token in lowered_name for token in ["vision", "visual", "image", "vit"]):
            continue

        candidate_layers = extract_generic_layers(module)
        if len(candidate_layers) > best_match[2]:
            best_match = (name, module, len(candidate_layers))

    return best_match[0], best_match[1]


def extract_clip_style_layers(vision_encoder):
    layers = []

    if hasattr(vision_encoder, "vision_model"):
        encoder = vision_encoder.vision_model.encoder
    elif hasattr(vision_encoder, "encoder"):
        encoder = vision_encoder.encoder
    else:
        encoder = vision_encoder

    if hasattr(encoder, "layers"):
        for idx, layer in enumerate(encoder.layers):
            layers.append((f"encoder.layers.{idx}", layer))
    elif hasattr(encoder, "layer"):
        for idx, layer in enumerate(encoder.layer):
            layers.append((f"encoder.layer.{idx}", layer))

    return layers


def extract_qwenvl_layers(vision_encoder):
    layers = []

    if hasattr(vision_encoder, "transformer"):
        transformer = vision_encoder.transformer
        if hasattr(transformer, "resblocks"):
            for idx, block in enumerate(transformer.resblocks):
                layers.append((f"transformer.resblocks.{idx}", block))
    elif hasattr(vision_encoder, "blocks"):
        for idx, block in enumerate(vision_encoder.blocks):
            layers.append((f"blocks.{idx}", block))
    elif hasattr(vision_encoder, "layers"):
        for idx, layer in enumerate(vision_encoder.layers):
            layers.append((f"layers.{idx}", layer))

    return layers


def extract_internvl_layers(vision_encoder):
    layers = []

    if hasattr(vision_encoder, "blocks"):
        for idx, block in enumerate(vision_encoder.blocks):
            layers.append((f"blocks.{idx}", block))
    elif hasattr(vision_encoder, "layers"):
        for idx, layer in enumerate(vision_encoder.layers):
            layers.append((f"layers.{idx}", layer))
    elif hasattr(vision_encoder, "encoder"):
        encoder = vision_encoder.encoder
        if hasattr(encoder, "layers"):
            for idx, layer in enumerate(encoder.layers):
                layers.append((f"encoder.layers.{idx}", layer))
        elif hasattr(encoder, "blocks"):
            for idx, block in enumerate(encoder.blocks):
                layers.append((f"encoder.blocks.{idx}", block))

    return layers


def extract_generic_layers(vision_encoder):
    layers = []
    layer_containers = ["layers", "blocks", "encoder", "transformer"]

    for container_name in layer_containers:
        if hasattr(vision_encoder, container_name):
            container = getattr(vision_encoder, container_name)
            if hasattr(container, "__iter__"):
                for idx, layer in enumerate(container):
                    layers.append((f"{container_name}.{idx}", layer))
                break
            if hasattr(container, "layers") or hasattr(container, "blocks"):
                nested = container.layers if hasattr(container, "layers") else container.blocks
                if hasattr(nested, "__iter__"):
                    for idx, layer in enumerate(nested):
                        layers.append((f"{container_name}.{idx}", layer))
                    break

    return layers


def extract_layers_by_architecture(vision_encoder, architecture_type):
    if architecture_type in ["llava", "clip", "siglip"]:
        layers = extract_clip_style_layers(vision_encoder)
    elif architecture_type == "qwen-vl":
        layers = extract_qwenvl_layers(vision_encoder)
    elif architecture_type == "internvl":
        layers = extract_internvl_layers(vision_encoder)
    else:
        layers = extract_generic_layers(vision_encoder)

    if not layers and (hasattr(vision_encoder, "vision_model") or hasattr(vision_encoder, "encoder")):
        layers = extract_clip_style_layers(vision_encoder)
    if not layers and (hasattr(vision_encoder, "transformer") or hasattr(vision_encoder, "blocks")):
        layers = extract_qwenvl_layers(vision_encoder)
    if not layers and hasattr(vision_encoder, "encoder"):
        layers = extract_internvl_layers(vision_encoder)

    return layers


def find_vision_layer_indices(model, architecture_type="auto"):
    vision_layers = {
        "layer_names": [],
        "layer_indices": [],
        "total_layers": 0,
        "encoder_type": None,
        "encoder_path": None,
    }

    if architecture_type == "auto":
        architecture_type = detect_architecture(model)

    vision_encoder_attrs = [
        "vision_tower",
        "model.vision_tower",
        "vision_model.vision_tower",
        "model.vision_model.vision_tower",
        "vision_model.vision_tower_high.vision_tower",
        "model.vision_model.vision_tower_high.vision_tower",
        "vision_model.vision_tower_low.vision_tower",
        "model.vision_model.vision_tower_low.vision_tower",
        "vision_model",
        "model.vision_model",
        "vision",
        "model.vision",
        "visual",
        "model.visual",
        "transformer.visual",
        "model.transformer.visual",
        "vision_encoder",
        "model.vision_encoder",
        "vit",
        "model.vit",
        "image_encoder",
        "model.image_encoder",
    ]

    vision_encoder = None
    for attr_path in vision_encoder_attrs:
        try:
            vision_encoder = resolve_module_path(model, attr_path)
        except (AttributeError, IndexError, KeyError, TypeError):
            continue
        vision_layers["encoder_type"] = attr_path.split(".")[-1]
        vision_layers["encoder_path"] = attr_path
        break

    if vision_encoder is None:
        encoder_path, vision_encoder = find_nested_vision_encoder(model)
        if vision_encoder is not None:
            vision_layers["encoder_type"] = encoder_path.split(".")[-1]
            vision_layers["encoder_path"] = encoder_path

    if vision_encoder is None:
        raise ValueError("No vision encoder found in model")

    layers = extract_layers_by_architecture(vision_encoder, architecture_type)
    vision_layers["layer_names"] = [name for name, _ in layers]
    vision_layers["layer_indices"] = list(range(len(layers)))
    vision_layers["total_layers"] = len(layers)
    return vision_layers


def get_specific_layer(model, layer_index):
    vision_info = find_vision_layer_indices(model)
    architecture_type = detect_architecture(model)

    if layer_index < 0:
        layer_index = vision_info["total_layers"] + layer_index

    if layer_index < 0 or layer_index >= vision_info["total_layers"]:
        raise IndexError(
            f"Layer index {layer_index} out of range (0-{vision_info['total_layers'] - 1})"
        )

    vision_encoder = resolve_module_path(model, vision_info["encoder_path"])
    layers = extract_layers_by_architecture(vision_encoder, architecture_type)
    layer_name, layer = layers[layer_index]
    return layer, layer_name
