from dataclasses import dataclass


@dataclass(frozen=True)
class VisionLayerSpec:
    family: str
    docs: tuple[str, ...]
    encoder_path_hints: tuple[str, ...]
    layer_stack_hints: tuple[str, ...]
    depth_config_hints: tuple[str, ...]
    notes: str = ""
    requires_manual_review: bool = False


FAMILY_VISION_SPECS = {
    "qwen2_vl": VisionLayerSpec(
        family="qwen2_vl",
        docs=("https://huggingface.co/docs/transformers/en/model_doc/qwen2_vl",),
        encoder_path_hints=("visual", "model.visual"),
        layer_stack_hints=("blocks",),
        depth_config_hints=("vision_config.depth", "vision_config.num_hidden_layers"),
        notes="Qwen2-VL and Qwen2.5-VL use a ViT-style visual stack exposed through the visual module.",
    ),
    "qwen3_vl": VisionLayerSpec(
        family="qwen3_vl",
        docs=("https://huggingface.co/docs/transformers/en/model_doc/qwen3_vl",),
        encoder_path_hints=("visual", "model.visual"),
        layer_stack_hints=("blocks",),
        depth_config_hints=("vision_config.depth", "vision_config.num_hidden_layers"),
        notes="Qwen3-VL keeps the visual transformer stack under the visual tower.",
    ),
    "qwen_vl": VisionLayerSpec(
        family="qwen_vl",
        docs=(
            "https://huggingface.co/Qwen/Qwen-VL-Chat",
            "https://github.com/QwenLM/Qwen-VL",
        ),
        encoder_path_hints=("transformer.visual", "model.transformer.visual", "visual"),
        layer_stack_hints=("resblocks", "blocks"),
        depth_config_hints=("visual.num_layers", "vision_config.num_hidden_layers"),
        notes="Qwen-VL-Chat uses the visual transformer inside transformer.visual.",
    ),
    "qwen_omni": VisionLayerSpec(
        family="qwen_omni",
        docs=("https://huggingface.co/docs/transformers/en/model_doc/qwen2_5_omni",),
        encoder_path_hints=("visual", "model.visual"),
        layer_stack_hints=("blocks",),
        depth_config_hints=("thinker_config.vision_config.depth", "thinker_config.vision_config.num_hidden_layers"),
        notes="Qwen2.5-Omni routes image understanding through the thinker vision encoder, exposed as visual.blocks.",
    ),
    "llava_next": VisionLayerSpec(
        family="llava_next",
        docs=("https://huggingface.co/docs/transformers/en/model_doc/llava_next",),
        encoder_path_hints=(
            "vision_tower",
            "model.vision_tower",
            "vision_model",
            "model.vision_model",
        ),
        layer_stack_hints=("vision_model.encoder.layers", "encoder.layers", "layers"),
        depth_config_hints=("vision_config.num_hidden_layers",),
        notes="LLaVA-NeXT uses a CLIP-style vision tower with encoder.layers.",
    ),
    "llava_onevision": VisionLayerSpec(
        family="llava_onevision",
        docs=("https://huggingface.co/docs/transformers/en/model_doc/llava_onevision",),
        encoder_path_hints=(
            "vision_tower",
            "model.vision_tower",
            "vision_model",
            "model.vision_model",
        ),
        layer_stack_hints=("vision_model.encoder.layers", "encoder.layers", "layers"),
        depth_config_hints=("vision_config.num_hidden_layers",),
        notes="LLaVA-OneVision uses a SigLIP/CLIP-style encoder stack.",
    ),
    "idefics3": VisionLayerSpec(
        family="idefics3",
        docs=("https://huggingface.co/docs/transformers/en/model_doc/idefics3",),
        encoder_path_hints=("vision_model", "model.vision_model", "vision_tower", "model.vision_tower"),
        layer_stack_hints=("encoder.layers", "vision_model.encoder.layers", "layers"),
        depth_config_hints=("vision_config.num_hidden_layers",),
        notes="Idefics3 uses a SigLIP vision backbone exposed as a transformer encoder.",
    ),
    "gemma3": VisionLayerSpec(
        family="gemma3",
        docs=("https://huggingface.co/docs/transformers/en/model_doc/gemma3",),
        encoder_path_hints=("vision_tower", "model.vision_tower", "vision_model", "model.vision_model"),
        layer_stack_hints=("vision_model.encoder.layers", "encoder.layers", "layers"),
        depth_config_hints=("vision_config.num_hidden_layers",),
        notes="Gemma 3 multimodal checkpoints use a SigLIP-style vision encoder.",
    ),
    "phi4_multimodal": VisionLayerSpec(
        family="phi4_multimodal",
        docs=("https://huggingface.co/docs/transformers/en/model_doc/phi4_multimodal",),
        encoder_path_hints=("vision_model", "model.vision_model", "vision_encoder", "model.vision_encoder"),
        layer_stack_hints=("encoder.layers", "vision_model.encoder.layers", "layers", "blocks"),
        depth_config_hints=("vision_config.num_hidden_layers", "vision_config.depth"),
        notes="Phi-4 multimodal uses a dedicated vision encoder tower.",
    ),
    "internvl_chat": VisionLayerSpec(
        family="internvl_chat",
        docs=(
            "https://huggingface.co/docs/transformers/en/model_doc/internvl",
            "https://github.com/OpenGVLab/InternVL",
        ),
        encoder_path_hints=("visual", "model.visual", "vision_model", "model.vision_model"),
        layer_stack_hints=("blocks", "encoder.layers", "layers"),
        depth_config_hints=("vision_config.num_hidden_layers", "vision_config.depth", "vision_config.num_layers"),
        notes="InternVL models expose an InternViT stack, usually as visual.blocks.",
    ),
    "yi_vl": VisionLayerSpec(
        family="yi_vl",
        docs=("https://huggingface.co/01-ai/Yi-VL-6B",),
        encoder_path_hints=("vision_tower", "model.vision_tower", "vision_model", "model.vision_model", "visual"),
        layer_stack_hints=("vision_model.encoder.layers", "encoder.layers", "layers", "blocks"),
        depth_config_hints=("vision_config.num_hidden_layers", "vision_tower.config.num_hidden_layers"),
        notes="Yi-VL checkpoints typically wrap a CLIP-like visual encoder.",
    ),
    "deepseek_vl2": VisionLayerSpec(
        family="deepseek_vl2",
        docs=(
            "https://github.com/deepseek-ai/DeepSeek-VL2",
            "https://huggingface.co/deepseek-ai/deepseek-vl2-tiny",
        ),
        encoder_path_hints=("vision", "model.vision"),
        layer_stack_hints=("blocks",),
        depth_config_hints=("vision_config.layers",),
        notes="DeepSeek-VL2 uses a single VisionTransformer as self.vision, but the current Modal runtime cannot complete a real forward due an upstream attention-signature mismatch.",
        requires_manual_review=True,
    ),
    "kimi_vl": VisionLayerSpec(
        family="kimi_vl",
        docs=("https://huggingface.co/moonshotai/Kimi-VL-A3B-Instruct",),
        encoder_path_hints=("vision_tower", "model.vision_tower", "vision_model", "model.vision_model", "visual"),
        layer_stack_hints=("vision_model.encoder.layers", "encoder.layers", "layers", "blocks"),
        depth_config_hints=("vision_config.num_hidden_layers", "vision_config.depth"),
        notes="Kimi-VL uses a standalone vision tower; verify the final block against the MoonViT stack if remote code changes.",
    ),
}


MODEL_PRESETS = {
    "vision_benchmark_under_13b": (
        "Qwen/Qwen3-VL-8B-Instruct",
        "Qwen/Qwen3-VL-4B-Instruct",
        "01-ai/Yi-VL-6B",
        "Qwen/Qwen-VL-Chat",
        "llava-hf/llava-v1.6-mistral-7b-hf",
        "Qwen/Qwen2-VL-7B-Instruct",
        "llava-hf/llava-onevision-qwen2-7b-ov-hf",
        "Qwen/Qwen2.5-VL-7B-Instruct",
        "OpenGVLab/InternVL2_5-8B",
        "OpenGVLab/InternVL2_5-4B",
        "OpenGVLab/InternVL3-8B",
        "OpenGVLab/InternVL3-1B",
        "HuggingFaceM4/Idefics3-8B-Llama3",
        "microsoft/Phi-4-multimodal-instruct",
        "google/gemma-3-4b-it",
        "google/gemma-3-12b-it",
        "Qwen/Qwen2.5-Omni-7B",
        "deepseek-ai/deepseek-vl2-tiny",
    ),
    "vision_benchmark_current_working_set": (
        "Qwen/Qwen2-VL-7B-Instruct",
        "Qwen/Qwen2.5-VL-7B-Instruct",
        "Qwen/Qwen3-VL-4B-Instruct",
        "Qwen/Qwen3-VL-8B-Instruct",
        "llava-hf/llava-v1.6-mistral-7b-hf",
        "OpenGVLab/InternVL2_5-4B",
        "OpenGVLab/InternVL3-1B",
        "HuggingFaceM4/Idefics3-8B-Llama3",
    ),
    "vision_benchmark_audit_pass_set": (
        "Qwen/Qwen2-VL-7B-Instruct",
        "Qwen/Qwen2.5-VL-7B-Instruct",
        "Qwen/Qwen3-VL-4B-Instruct",
        "Qwen/Qwen3-VL-8B-Instruct",
        "llava-hf/llava-v1.6-mistral-7b-hf",
        "OpenGVLab/InternVL2_5-4B",
        "OpenGVLab/InternVL3-1B",
        "HuggingFaceM4/Idefics3-8B-Llama3",
        "Qwen/Qwen2.5-Omni-7B",
        "deepseek-ai/deepseek-vl2-tiny",
    ),
}

__all__ = ["FAMILY_VISION_SPECS", "MODEL_PRESETS", "VisionLayerSpec"]
