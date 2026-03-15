from peft import PeftModel
import torch
from transformers import BitsAndBytesConfig, AutoModelForVision2Seq, AutoProcessor, AutoConfig
import warnings
import os

def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)

# This code is borrowed from LLaVA
def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, 
                          device_map="auto", device="cuda", use_flash_attn=False, **kwargs):
    kwargs = {"device_map": device_map}
    
    if device != "cuda":
        kwargs['device_map'] = {"":device}
    
    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    if use_flash_attn:
        kwargs['_attn_implementation'] = 'flash_attention_2'

    if 'lora' in model_name.lower() and model_base is None:
        warnings.warn('There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument.')
    if 'lora' in model_name.lower() and model_base is not None:
        lora_cfg_pretrained = AutoConfig.from_pretrained(model_path)
        if hasattr(lora_cfg_pretrained, 'quantization_config'):
            del lora_cfg_pretrained.quantization_config
        processor = AutoProcessor.from_pretrained(model_base)
        print('Loading SmolVLM from base model...')
        model = AutoModelForVision2Seq.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
        token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
        if model.lm_head.weight.shape[0] != token_num:
            model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
            model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

        print('Loading additional SmolVLM weights...')
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
        processor = AutoProcessor.from_pretrained(model_base)
        model = AutoModelForVision2Seq.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    return processor, model


def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]


def resolve_module_path(module, path):
    """Resolve a dotted attribute/index path from a root module."""
    current = module
    if not path:
        return current

    for part in path.split('.'):
        if part.isdigit():
            current = current[int(part)]
        else:
            current = getattr(current, part)
    return current
    

def find_vision_layer_indices(model, architecture_type="auto"):
    """
    Find all vision encoder layers in a Vision-Language Model
    
    Args:
        model: The VLM model (PyTorch/Transformers)
        architecture_type: "llava", "qwen-vl", "internvl", "auto"
    
    Returns:
        dict with layer names, indices, and metadata
    """
    vision_layers = {
        'layer_names': [],
        'layer_indices': [],
        'total_layers': 0,
        'encoder_type': None,
        'encoder_path': None,
    }
    
    # Auto-detect architecture
    if architecture_type == "auto":
        architecture_type = detect_architecture(model)
    
    # Define common vision encoder attribute names
    vision_encoder_attrs = [
        'vision_tower',            # LLaVA, LLaVA-NeXT
        'model.vision_tower',
        'vision_model.vision_tower',
        'model.vision_model.vision_tower',
        'vision_model.vision_tower_high.vision_tower',
        'model.vision_model.vision_tower_high.vision_tower',
        'vision_model.vision_tower_low.vision_tower',
        'model.vision_model.vision_tower_low.vision_tower',
        'vision_model',            # CLIP-based models
        'model.vision_model',
        'vision',                  # DeepSeek-VL2
        'model.vision',
        'visual',                  # Qwen-VL, InternVL
        'model.visual',
        'transformer.visual',      # Qwen-VL-Chat
        'model.transformer.visual',
        'vision_encoder',          # Generic
        'model.vision_encoder',
        'vit',                     # Some custom implementations
        'model.vit',
        'image_encoder',           # Alternative naming
        'model.image_encoder',
    ]
    
    # Find vision encoder module
    vision_encoder = None
    for attr_path in vision_encoder_attrs:
        try:
            vision_encoder = resolve_module_path(model, attr_path)
        except (AttributeError, IndexError, KeyError, TypeError):
            continue
        vision_layers['encoder_type'] = attr_path.split('.')[-1]
        vision_layers['encoder_path'] = attr_path
        break
    
    if vision_encoder is None:
        # Try nested search
        encoder_path, vision_encoder = find_nested_vision_encoder(model)
        if vision_encoder is not None:
            vision_layers['encoder_type'] = encoder_path.split('.')[-1]
            vision_layers['encoder_path'] = encoder_path
    
    if vision_encoder is None:
        raise ValueError("No vision encoder found in model")
    
    # Extract layers based on architecture
    if architecture_type in ["llava", "clip", "siglip"]:
        layers = extract_clip_style_layers(vision_encoder)
    elif architecture_type == "qwen-vl":
        layers = extract_qwenvl_layers(vision_encoder)
    elif architecture_type == "internvl":
        layers = extract_internvl_layers(vision_encoder)
    else:
        layers = extract_generic_layers(vision_encoder)

    if not layers and (hasattr(vision_encoder, 'vision_model') or hasattr(vision_encoder, 'encoder')):
        layers = extract_clip_style_layers(vision_encoder)
    if not layers and (hasattr(vision_encoder, 'transformer') or hasattr(vision_encoder, 'blocks')):
        layers = extract_qwenvl_layers(vision_encoder)
    if not layers and hasattr(vision_encoder, 'encoder'):
        layers = extract_internvl_layers(vision_encoder)
    
    vision_layers['layer_names'] = [name for name, _ in layers]
    vision_layers['layer_indices'] = list(range(len(layers)))
    vision_layers['total_layers'] = len(layers)
    
    return vision_layers


def detect_architecture(model):
    """Auto-detect VLM architecture type"""
    model_name = model.__class__.__name__.lower()
    
    if 'llava' in model_name:
        return 'llava'
    elif 'qwen' in model_name:
        return 'qwen-vl'
    elif 'intern' in model_name:
        return 'internvl'
    elif 'clip' in model_name or 'siglip' in model_name:
        return 'clip'
    else:
        return 'generic'


def find_nested_vision_encoder(model):
    """Recursively search for vision encoder in nested modules"""
    best_match = (None, None, 0)
    for name, module in model.named_modules():
        lowered_name = name.lower()
        if not any(x in lowered_name for x in ['vision', 'visual', 'image', 'vit']):
            continue

        candidate_layers = extract_generic_layers(module)
        if len(candidate_layers) > best_match[2]:
            best_match = (name, module, len(candidate_layers))

    return best_match[0], best_match[1]


def extract_clip_style_layers(vision_encoder):
    """Extract layers from CLIP/SigLIP style encoders"""
    layers = []
    
    # CLIP usually has: vision_model.encoder.layers
    if hasattr(vision_encoder, 'vision_model'):
        encoder = vision_encoder.vision_model.encoder
    elif hasattr(vision_encoder, 'encoder'):
        encoder = vision_encoder.encoder
    else:
        encoder = vision_encoder
    
    if hasattr(encoder, 'layers'):
        for idx, layer in enumerate(encoder.layers):
            layer_name = f"encoder.layers.{idx}"
            layers.append((layer_name, layer))
    elif hasattr(encoder, 'layer'):
        for idx, layer in enumerate(encoder.layer):
            layer_name = f"encoder.layer.{idx}"
            layers.append((layer_name, layer))
    
    return layers


def extract_qwenvl_layers(vision_encoder):
    """Extract layers from Qwen-VL style encoders"""
    layers = []
    
    # Qwen-VL may have transformer blocks
    if hasattr(vision_encoder, 'transformer'):
        transformer = vision_encoder.transformer
        if hasattr(transformer, 'resblocks'):
            for idx, block in enumerate(transformer.resblocks):
                layer_name = f"transformer.resblocks.{idx}"
                layers.append((layer_name, block))
    elif hasattr(vision_encoder, 'blocks'):
        for idx, block in enumerate(vision_encoder.blocks):
            layer_name = f"blocks.{idx}"
            layers.append((layer_name, block))
    elif hasattr(vision_encoder, 'layers'):
        for idx, layer in enumerate(vision_encoder.layers):
            layer_name = f"layers.{idx}"
            layers.append((layer_name, layer))
    
    return layers


def extract_internvl_layers(vision_encoder):
    """Extract layers from InternVL style encoders"""
    layers = []
    
    # InternViT structure
    if hasattr(vision_encoder, 'blocks'):
        for idx, block in enumerate(vision_encoder.blocks):
            layer_name = f"blocks.{idx}"
            layers.append((layer_name, block))
    elif hasattr(vision_encoder, 'layers'):
        for idx, layer in enumerate(vision_encoder.layers):
            layer_name = f"layers.{idx}"
            layers.append((layer_name, layer))
    elif hasattr(vision_encoder, 'encoder'):
        encoder = vision_encoder.encoder
        if hasattr(encoder, 'layers'):
            for idx, layer in enumerate(encoder.layers):
                layer_name = f"encoder.layers.{idx}"
                layers.append((layer_name, layer))
        elif hasattr(encoder, 'blocks'):
            for idx, block in enumerate(encoder.blocks):
                layer_name = f"encoder.blocks.{idx}"
                layers.append((layer_name, block))
    
    return layers


def extract_generic_layers(vision_encoder):
    """Generic layer extraction for unknown architectures"""
    layers = []
    
    # Common patterns
    layer_containers = ['layers', 'blocks', 'encoder', 'transformer']
    
    for container_name in layer_containers:
        if hasattr(vision_encoder, container_name):
            container = getattr(vision_encoder, container_name)
            
            # If it's a ModuleList or Sequential
            if hasattr(container, '__iter__'):
                for idx, layer in enumerate(container):
                    layer_name = f"{container_name}.{idx}"
                    layers.append((layer_name, layer))
                break
            # If it has nested layers
            elif hasattr(container, 'layers') or hasattr(container, 'blocks'):
                nested = container.layers if hasattr(container, 'layers') else container.blocks
                if hasattr(nested, '__iter__'):
                    for idx, layer in enumerate(nested):
                        layer_name = f"{container_name}.{idx}"
                        layers.append((layer_name, layer))
                    break
    
    return layers


def get_specific_layer(model, layer_index):
    """Get a specific vision layer by index"""
    vision_info = find_vision_layer_indices(model)
    architecture_type = detect_architecture(model)

    if layer_index < 0:
        layer_index = vision_info['total_layers'] + layer_index

    if layer_index < 0 or layer_index >= vision_info['total_layers']:
        raise IndexError(f"Layer index {layer_index} out of range (0-{vision_info['total_layers']-1})")

    vision_encoder = resolve_module_path(model, vision_info['encoder_path'])
    if architecture_type in ["llava", "clip", "siglip"]:
        layers = extract_clip_style_layers(vision_encoder)
    elif architecture_type == "qwen-vl":
        layers = extract_qwenvl_layers(vision_encoder)
    elif architecture_type == "internvl":
        layers = extract_internvl_layers(vision_encoder)
    else:
        layers = extract_generic_layers(vision_encoder)

    if not layers and (hasattr(vision_encoder, 'vision_model') or hasattr(vision_encoder, 'encoder')):
        layers = extract_clip_style_layers(vision_encoder)
    if not layers and (hasattr(vision_encoder, 'transformer') or hasattr(vision_encoder, 'blocks')):
        layers = extract_qwenvl_layers(vision_encoder)
    if not layers and hasattr(vision_encoder, 'encoder'):
        layers = extract_internvl_layers(vision_encoder)

    layer_name, layer = layers[layer_index]
    return layer, layer_name


def extract_vision_features_from_layer(model, model_inputs, layer_index=-1):
    """Extract features from a specific vision layer using a forward hook."""
    features = {}
    layer, layer_name = get_specific_layer(model, layer_index)

    def hook_fn(module, inputs, output):
        value = output[0] if isinstance(output, (tuple, list)) else output
        features['output'] = value.detach()

    handle = layer.register_forward_hook(hook_fn)

    with torch.no_grad():
        if isinstance(model_inputs, dict):
            _ = model(**model_inputs)
        else:
            _ = model(model_inputs)

    handle.remove()
    return features['output'], layer_name
