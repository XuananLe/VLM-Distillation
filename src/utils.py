def resolve_module_path(module, path):
    """Resolve a dotted attribute/index path against a nested module tree."""
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
    """Infer the broad VLM architecture family from the model class name."""
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
    """Search named modules for the deepest vision-like encoder that exposes transformer layers."""
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
    """Extract ordered transformer layers from CLIP-style vision encoders."""
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
    """Extract ordered transformer layers from Qwen-VL style vision encoders."""
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
    """Extract ordered transformer layers from InternVL-style vision encoders."""
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
    """Extract ordered layers from generic iterable vision backbones."""
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
    """Dispatch to the architecture-specific vision-layer extractor with fallbacks."""
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
    """Locate the vision encoder on a model and return its ordered layer metadata."""
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
    """Return one resolved vision layer module and its name by index."""
    vision_info = find_vision_layer_indices(model)
    architecture_type = detect_architecture(model)

    if layer_index < 0:
        layer_index = vision_info["total_layers"] + layer_index

    if layer_index < 0 or layer_index >= vision_info["total_layers"]:
        raise IndexError(f"Layer index {layer_index} out of range (0-{vision_info['total_layers'] - 1})")

    vision_encoder = resolve_module_path(model, vision_info["encoder_path"])
    layers = extract_layers_by_architecture(vision_encoder, architecture_type)
    layer_name, layer = layers[layer_index]
    return layer, layer_name
