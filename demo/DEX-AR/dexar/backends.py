import math
from collections import deque
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import torch


def iter_model_roots(model):
    queue = deque([model])
    seen: set[int] = set()
    while queue:
        current = queue.popleft()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        for attr in ("model", "base_model"):
            child = getattr(current, attr, None)
            if child is not None:
                queue.append(child)


def resolve_attr(model, aliases: tuple[str, ...]) -> Any:
    for root in iter_model_roots(model):
        for alias in aliases:
            if hasattr(root, alias):
                return getattr(root, alias)
    raise AttributeError(f"Could not resolve any of {aliases!r} on the model.")


def resolve_text_backbone(model):
    return resolve_attr(model, ("language_model", "text_model"))


def resolve_lm_head(model, text_backbone):
    for candidate in (model, text_backbone, getattr(text_backbone, "model", None)):
        if candidate is not None and hasattr(candidate, "lm_head"):
            return getattr(candidate, "lm_head")
    raise AttributeError("Could not resolve lm_head on the model.")


def resolve_norm(text_backbone):
    for candidate in (text_backbone, getattr(text_backbone, "model", None)):
        if candidate is not None:
            for attr_name in ("norm", "embedding_norm"):
                if hasattr(candidate, attr_name):
                    return getattr(candidate, attr_name)
    raise AttributeError("Could not resolve the final normalization layer.")


def resolve_layers(text_backbone):
    for candidate in (getattr(text_backbone, "model", None), text_backbone):
        if candidate is None:
            continue
        layers = getattr(candidate, "layers", None)
        if layers is not None:
            return layers
    raise AttributeError("Could not resolve decoder layers on the text backbone.")


def identity_norm():
    from torch import nn

    return nn.Identity()


def resolve_device_map(device: str):
    if device in {"auto", "cpu", "cuda"}:
        return device
    return {"": device}


def resolve_torch_dtype(device: str) -> torch.dtype:
    if isinstance(device, str) and device.startswith("cuda"):
        return torch.float16
    return torch.float32


def load_model_with_eager_attention(model_cls, model_name: str, *, device: str):
    common_kwargs = {
        "torch_dtype": resolve_torch_dtype(device),
        "device_map": resolve_device_map(device),
    }
    try:
        return model_cls.from_pretrained(
            model_name,
            attn_implementation="eager",
            **common_kwargs,
        )
    except TypeError as exc:
        if "attn_implementation" not in str(exc):
            raise
        return model_cls.from_pretrained(
            model_name,
            _attn_implementation="eager",
            **common_kwargs,
        )


@dataclass
class EncodedPrompt:
    model_inputs: dict[str, torch.Tensor]
    prompt_input_ids: torch.Tensor
    prompt_image_mask: torch.Tensor
    image_grid: tuple[int, int]


@dataclass
class DexarBackend:
    family: str
    model: Any
    processor: Any
    text_backbone: Any
    lm_head: Any
    norm: Any
    layers: Any
    image_token_id: int
    default_prompt: str
    recommended_image_size: int
    base_image_seq_len: int | None = None
    spatial_merge_size: int | None = None
    attention_layer_indices: tuple[int, ...] | None = None
    custom_generate_answer: Any | None = None

    @property
    def num_layers(self) -> int:
        return len(self.layers)

    @classmethod
    def from_pretrained(cls, model_name: str, device: str):
        from transformers import AutoConfig

        if "florence-2" in model_name.lower() or "florence2" in model_name.lower():
            return load_florence2_backend(model_name, device)

        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        model_type = getattr(config, "model_type", None)

        if model_type == "llava":
            return load_llava_backend(model_name, device)
        if model_type == "paligemma":
            return load_paligemma_backend(model_name, device)
        if model_type in {"smolvlm", "idefics3"}:
            return load_smolvlm_backend(model_name, device)
        if model_type in {"qwen2_vl", "qwen2_5_vl"}:
            return load_qwen2vl_backend(model_name, device, model_type=model_type)
        if model_type == "lfm2_vl":
            return load_lfm2vl_backend(model_name, device)
        if model_type == "gemma3":
            return load_gemma3_backend(model_name, device)
        if model_type == "internvl_chat" or "internvl" in model_name.lower():
            return load_internvl_backend(model_name, device)

        raise ValueError(
            f"Unsupported model type {model_type!r} for DEX-AR. "
            "Supported families: LLaVA, SmolVLM/Idefics3, Qwen2-VL, "
            "LFM2-VL, Gemma 3, and InternVL."
        )

    def enable_dexar_gradients(self) -> None:
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        for parameter in self.text_backbone.parameters():
            parameter.requires_grad = True
        for parameter in self.lm_head.parameters():
            parameter.requires_grad = True
        self.model.eval()
        configs = (
            getattr(self.model, "config", None),
            getattr(getattr(self.model, "config", None), "text_config", None),
            getattr(getattr(self.model, "language_model", None), "config", None),
            getattr(self.text_backbone, "config", None),
        )
        seen_config_ids: set[int] = set()
        for config in configs:
            if config is not None:
                if id(config) in seen_config_ids:
                    continue
                seen_config_ids.add(id(config))
                if getattr(config, "_attn_implementation", None) != "eager":
                    config._attn_implementation = "eager"
                config.output_attentions = True
                config.output_hidden_states = True

    def encode_prompt(self, prompt: str, image, device: torch.device) -> EncodedPrompt:
        if self.family not in {"paligemma", "florence2"} and prompt.count("<image>") != 1:
            raise ValueError(
                "DEX-AR currently supports prompts with exactly one <image> placeholder."
            )

        if self.family == "internvl":
            return self._encode_internvl_prompt(prompt=prompt, image=image, device=device)

        if self.family == "paligemma":
            clean_prompt = prompt.replace("<image>", "").strip()
            inputs = self.processor(
                text=clean_prompt,
                images=image,
                return_tensors="pt",
            )
        elif self.family == "florence2":
            clean_prompt = prompt.replace("<image>", "").strip()
            inputs = self.processor(
                text=clean_prompt,
                images=image,
                return_tensors="pt",
            )
        elif self.family in {"qwen2vl", "lfm2vl", "gemma3"}:
            prefix, suffix = prompt.split("<image>")
            content = []
            prefix = prefix.strip()
            suffix = suffix.strip()
            if prefix:
                content.append({"type": "text", "text": prefix})
            if self.family == "gemma3":
                content.append({"type": "image", "image": image})
            else:
                content.append({"type": "image"})
            if suffix:
                content.append({"type": "text", "text": suffix})
            messages = [{"role": "user", "content": content}]
            if self.family == "gemma3":
                inputs = self.processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                    add_generation_prompt=True,
                    do_pan_and_scan=False,
                )
            else:
                prompt_text = self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                inputs = self.processor(
                    text=[prompt_text],
                    images=[image],
                    return_tensors="pt",
                )
        else:
            inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        model_inputs = {
            key: value.to(device) for key, value in inputs.items() if torch.is_tensor(value)
        }

        if "input_ids" not in model_inputs:
            raise ValueError("Processor output did not include input_ids.")

        prompt_input_ids = model_inputs["input_ids"]
        prompt_image_mask = prompt_input_ids[0] == self.image_token_id
        num_image_tokens = int(prompt_image_mask.sum().item())
        if self.family == "florence2" and num_image_tokens == 0:
            image_seq_length = getattr(self.processor, "image_seq_length", None)
            if image_seq_length is None and hasattr(self.processor, "image_processor"):
                image_seq_length = getattr(self.processor.image_processor, "image_seq_length", None)
            if image_seq_length is None:
                raise ValueError(
                    "Florence-2 processor output did not include image tokens and "
                    "no image_seq_length was available to synthesize the image mask."
                )
            synthetic_image_ids = torch.full(
                (1, int(image_seq_length)),
                int(self.image_token_id),
                device=prompt_input_ids.device,
                dtype=prompt_input_ids.dtype,
            )
            prompt_input_ids = torch.cat([synthetic_image_ids, prompt_input_ids], dim=1)
            prompt_image_mask = prompt_input_ids[0] == self.image_token_id
            num_image_tokens = int(prompt_image_mask.sum().item())
        if num_image_tokens == 0:
            raise ValueError(
                "No image tokens were found in the encoded prompt. "
                "Check that the processor and prompt format match the selected model."
            )

        if (
            self.family == "smolvlm"
            and self.base_image_seq_len is not None
            and num_image_tokens != self.base_image_seq_len
        ):
            raise ValueError(
                "SmolVLM support currently handles only a single unsplit image region. "
                f"Expected {self.base_image_seq_len} image tokens, got {num_image_tokens}. "
                "Resize the image to the model's base resolution and keep a single <image>."
            )

        if self.family == "qwen2vl":
            image_grid_thw = model_inputs.get("image_grid_thw")
            if image_grid_thw is None:
                raise ValueError("Qwen2-VL processor output did not include image_grid_thw.")
            if image_grid_thw.shape[0] != 1:
                raise ValueError("DEX-AR currently supports a single Qwen image per prompt.")

            temporal_grid, height_grid, width_grid = (
                int(x) for x in image_grid_thw[0].tolist()
            )
            if temporal_grid != 1:
                raise ValueError(
                    "DEX-AR currently supports only static images for Qwen2-VL."
                )

            merge_size = self.spatial_merge_size or 1
            if height_grid % merge_size != 0 or width_grid % merge_size != 0:
                raise ValueError(
                    "Qwen2-VL image grid is not divisible by the spatial merge size."
                )

            image_grid = (height_grid // merge_size, width_grid // merge_size)
            expected_num_tokens = image_grid[0] * image_grid[1]
            if num_image_tokens != expected_num_tokens:
                raise ValueError(
                    "Qwen2-VL image token count does not match the merged spatial grid. "
                    f"Expected {expected_num_tokens}, got {num_image_tokens}."
                )
        elif self.family == "lfm2vl":
            spatial_shapes = model_inputs.get("spatial_shapes")
            if spatial_shapes is None:
                raise ValueError("LFM2-VL processor output did not include spatial_shapes.")
            if spatial_shapes.shape[0] != 1:
                raise ValueError("DEX-AR currently supports a single LFM2-VL image per prompt.")

            height_grid, width_grid = (int(x) for x in spatial_shapes[0].tolist())
            downsample_factor = getattr(self.model.config, "downsample_factor", None)
            if downsample_factor is None and hasattr(self.processor, "image_processor"):
                downsample_factor = getattr(self.processor.image_processor, "downsample_factor", None)
            downsample_factor = int(downsample_factor or 1)
            image_grid = (
                math.ceil(height_grid / downsample_factor),
                math.ceil(width_grid / downsample_factor),
            )
            expected_num_tokens = image_grid[0] * image_grid[1]
            if num_image_tokens != expected_num_tokens:
                raise ValueError(
                    "LFM2-VL image token count does not match the projected spatial grid. "
                    f"Expected {expected_num_tokens}, got {num_image_tokens}."
                )
        elif self.family == "florence2":
            spatial_tokens = num_image_tokens - 1
            side = math.isqrt(spatial_tokens)
            if side * side != spatial_tokens:
                side = math.isqrt(num_image_tokens)
                if side * side != num_image_tokens:
                    raise ValueError(
                        "Florence-2 image token count is not square after removing "
                        f"the global token: {num_image_tokens} tokens."
                    )
            else:
                image_positions = prompt_image_mask.nonzero(as_tuple=False).flatten()
                prompt_image_mask = prompt_image_mask.clone()
                prompt_image_mask[image_positions[0]] = False
            image_grid = (side, side)
        else:
            side = math.isqrt(num_image_tokens)
            if side * side != num_image_tokens:
                raise ValueError(
                    f"DEX-AR currently expects a square image token grid, got {num_image_tokens} tokens."
                )
            image_grid = (side, side)

        return EncodedPrompt(
            model_inputs=model_inputs,
            prompt_input_ids=prompt_input_ids,
            prompt_image_mask=prompt_image_mask,
            image_grid=image_grid,
        )

    def _encode_internvl_prompt(self, prompt: str, image, device: torch.device) -> EncodedPrompt:
        from src.dataset.internvl_utils import (
            INTERNVL_IMG_CONTEXT_TOKEN,
            INTERNVL_IMG_END_TOKEN,
            INTERNVL_IMG_START_TOKEN,
            build_internvl_pixel_values,
        )

        tokenizer = self.processor.tokenizer
        clean_prompt = prompt.strip()
        if "<image>" not in clean_prompt:
            clean_prompt = "<image>\n" + clean_prompt

        template = getattr(self.model, "conv_template", None)
        if template is None:
            query = f"User: {clean_prompt}\nAssistant:"
        else:
            import copy

            template = copy.deepcopy(template)
            template.system_message = getattr(self.model, "system_message", template.system_message)
            template.append_message(template.roles[0], clean_prompt)
            template.append_message(template.roles[1], None)
            query = template.get_prompt()

        image_processor_cfg = getattr(self.processor, "image_processor_cfg", {})
        pixel_values = build_internvl_pixel_values(image, image_processor_cfg)
        num_patches = int(pixel_values.shape[0])
        num_image_token = int(getattr(self.model, "num_image_token", self.base_image_seq_len or 256))
        image_tokens = (
            INTERNVL_IMG_START_TOKEN
            + INTERNVL_IMG_CONTEXT_TOKEN * num_image_token * num_patches
            + INTERNVL_IMG_END_TOKEN
        )
        query = query.replace("<image>", image_tokens, 1)

        image_token_id = tokenizer.convert_tokens_to_ids(INTERNVL_IMG_CONTEXT_TOKEN)
        self.model.img_context_token_id = image_token_id

        tokenized = tokenizer(query, return_tensors="pt")
        model_inputs = {
            "input_ids": tokenized["input_ids"].to(device),
            "attention_mask": tokenized["attention_mask"].to(device),
            "pixel_values": pixel_values.to(device=device, dtype=next(self.model.parameters()).dtype),
            "image_flags": torch.ones((num_patches, 1), device=device, dtype=torch.long),
        }

        prompt_input_ids = model_inputs["input_ids"]
        prompt_image_mask = prompt_input_ids[0] == image_token_id
        num_image_tokens = int(prompt_image_mask.sum().item())
        if num_image_tokens == 0:
            raise ValueError("No InternVL image context tokens were found in the encoded prompt.")

        tokens_per_patch = num_image_tokens // num_patches
        side = math.isqrt(tokens_per_patch)
        if side * side != tokens_per_patch:
            raise ValueError(
                "InternVL image token count per patch is not square: "
                f"{tokens_per_patch} tokens."
            )
        if num_patches != 1:
            raise ValueError(
                "DEX-AR currently runs InternVL with one image tile for interpretable maps; "
                f"got {num_patches} tiles."
            )

        return EncodedPrompt(
            model_inputs=model_inputs,
            prompt_input_ids=prompt_input_ids,
            prompt_image_mask=prompt_image_mask,
            image_grid=(side, side),
        )


def build_backend(
    *,
    family: str,
    model,
    processor,
    image_token_id: int,
    default_prompt: str,
    recommended_image_size: int,
    base_image_seq_len: int | None = None,
    spatial_merge_size: int | None = None,
    text_backbone=None,
    norm=None,
    lm_head=None,
    custom_generate_answer=None,
) -> DexarBackend:
    if text_backbone is None:
        text_backbone = resolve_text_backbone(model)
    if norm is None:
        norm = resolve_norm(text_backbone)
    if lm_head is None:
        lm_head = resolve_lm_head(model, text_backbone)
    return DexarBackend(
        family=family,
        model=model,
        processor=processor,
        text_backbone=text_backbone,
        lm_head=lm_head,
        norm=norm,
        layers=resolve_layers(text_backbone),
        image_token_id=image_token_id,
        default_prompt=default_prompt,
        recommended_image_size=recommended_image_size,
        base_image_seq_len=base_image_seq_len,
        spatial_merge_size=spatial_merge_size,
        custom_generate_answer=custom_generate_answer,
        attention_layer_indices=tuple(
            index
            for index, layer_type in enumerate(getattr(text_backbone.config, "layer_types", ()))
            if layer_type == "full_attention"
        )
        or None,
    )


def load_llava_backend(model_name: str, device: str) -> DexarBackend:
    from transformers import AutoProcessor, LlavaForConditionalGeneration

    model = load_model_with_eager_attention(
        LlavaForConditionalGeneration,
        model_name,
        device=device,
    )

    if "bak" in model_name.lower():
        processor = AutoProcessor.from_pretrained(
            model_name,
            revision="a92a28c845fbe89d009f211ce3d0d7aa6d42e948",
        )
    else:
        processor = AutoProcessor.from_pretrained(model_name, revision="a272c74")

    processor.patch_size = model.config.vision_config.patch_size
    processor.vision_feature_select_strategy = model.config.vision_feature_select_strategy
    processor.num_additional_image_tokens = 1
    image_token = getattr(processor, "image_token", "<image>")
    image_token_id = processor.tokenizer.convert_tokens_to_ids(image_token)

    return build_backend(
        family="llava",
        model=model,
        processor=processor,
        image_token_id=image_token_id,
        default_prompt="USER: <image>\nDescribe the image. ASSISTANT:",
        recommended_image_size=336,
    )


def load_paligemma_backend(model_name: str, device: str) -> DexarBackend:
    from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

    model = load_model_with_eager_attention(
        PaliGemmaForConditionalGeneration,
        model_name,
        device=device,
    )
    processor = AutoProcessor.from_pretrained(model_name)

    image_token_id = getattr(model.config, "image_token_index", None)
    if image_token_id is None:
        image_token_id = getattr(model.config, "image_token_id", None)
    if image_token_id is None:
        image_token_id = processor.tokenizer.convert_tokens_to_ids("<image>")

    text_backbone = getattr(model, "language_model", None)
    if text_backbone is None:
        text_backbone = resolve_text_backbone(model)

    vision_config = getattr(model.config, "vision_config", None)
    recommended_image_size = int(getattr(vision_config, "image_size", 224) or 224)

    return build_backend(
        family="paligemma",
        model=model,
        processor=processor,
        text_backbone=text_backbone,
        image_token_id=image_token_id,
        default_prompt="cap en\n",
        recommended_image_size=recommended_image_size,
    )


def load_florence2_backend(model_name: str, device: str) -> DexarBackend:
    from transformers import AutoProcessor, Florence2ForConditionalGeneration

    if model_name.lower().startswith("microsoft/florence-2"):
        try:
            return load_microsoft_florence2_remote_backend(model_name, device)
        except Exception as exc:
            print(
                "[dexar] Microsoft Florence-2 remote-code load failed; "
                f"falling back to native Transformers Florence-2 path: {type(exc).__name__}: {exc}",
                flush=True,
            )

    load_kwargs = {
        "device_map": resolve_device_map(device),
        "torch_dtype": resolve_torch_dtype(device),
        "trust_remote_code": False,
    }
    try:
        model = Florence2ForConditionalGeneration.from_pretrained(
            model_name,
            attn_implementation="eager",
            **load_kwargs,
        )
    except TypeError:
        model = Florence2ForConditionalGeneration.from_pretrained(
            model_name,
            **load_kwargs,
        )

    try:
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=False)
    except (AttributeError, OSError, ValueError):
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    image_token_id = getattr(model.config, "image_token_id", None)
    if image_token_id is None:
        image_token_id = processor.tokenizer.convert_tokens_to_ids("<image>")

    language_model = model.model.language_model
    text_backbone = language_model.decoder

    return build_backend(
        family="florence2",
        model=model,
        processor=processor,
        text_backbone=text_backbone,
        norm=identity_norm(),
        image_token_id=image_token_id,
        default_prompt="<DETAILED_CAPTION>",
        recommended_image_size=768,
        custom_generate_answer=generate_florence2_answer,
    )


def load_microsoft_florence2_remote_backend(model_name: str, device: str) -> DexarBackend:
    from transformers import AutoModelForCausalLM, AutoProcessor

    common_kwargs = {
        "device_map": resolve_device_map(device),
        "torch_dtype": resolve_torch_dtype(device),
        "trust_remote_code": True,
    }
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            attn_implementation="eager",
            **common_kwargs,
        )
    except TypeError as exc:
        if "attn_implementation" not in str(exc):
            raise
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            _attn_implementation="eager",
            **common_kwargs,
        )
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    language_model = model.language_model
    text_model = language_model.model
    image_size = 768
    processor_size = getattr(getattr(processor, "image_processor", None), "size", None)
    if isinstance(processor_size, dict):
        image_size = int(
            processor_size.get("height")
            or processor_size.get("shortest_edge")
            or processor_size.get("longest_edge")
            or image_size
        )

    return build_backend(
        family="florence2",
        model=model,
        processor=processor,
        text_backbone=text_model.decoder,
        lm_head=language_model.lm_head,
        norm=identity_norm(),
        image_token_id=-1,
        default_prompt="<DETAILED_CAPTION>",
        recommended_image_size=image_size,
        custom_generate_answer=generate_florence2_answer,
    )


def load_smolvlm_backend(model_name: str, device: str) -> DexarBackend:
    from transformers import AutoProcessor, Idefics3ForConditionalGeneration

    model = load_model_with_eager_attention(
        Idefics3ForConditionalGeneration,
        model_name,
        device=device,
    )
    processor = AutoProcessor.from_pretrained(model_name)
    if hasattr(processor, "image_processor"):
        image_processor = processor.image_processor
        if hasattr(image_processor, "max_image_size"):
            image_processor.size = dict(image_processor.max_image_size)
        if hasattr(image_processor, "do_image_splitting"):
            image_processor.do_image_splitting = False

    image_token_id = getattr(processor, "image_token_id", None)
    if image_token_id is None:
        image_token = getattr(processor, "image_token", "<image>")
        image_token_id = processor.tokenizer.convert_tokens_to_ids(image_token)

    base_image_seq_len = getattr(processor, "image_seq_len", None)
    if base_image_seq_len is None:
        base_image_seq_len = getattr(getattr(model, "model", None), "image_seq_len", None)
    recommended_image_size = 384
    if hasattr(processor, "image_processor") and hasattr(processor.image_processor, "max_image_size"):
        recommended_image_size = processor.image_processor.max_image_size.get(
            "longest_edge",
            recommended_image_size,
        )

    return build_backend(
        family="smolvlm",
        model=model,
        processor=processor,
        image_token_id=image_token_id,
        default_prompt="<|im_start|>User:<image>Describe the image.<end_of_utterance>\nAssistant:",
        recommended_image_size=recommended_image_size,
        base_image_seq_len=base_image_seq_len,
    )


def load_qwen2vl_backend(
    model_name: str,
    device: str,
    *,
    model_type: str,
) -> DexarBackend:
    from transformers import AutoProcessor

    if model_type == "qwen2_5_vl":
        from transformers import Qwen2_5_VLForConditionalGeneration as QwenModelClass
    else:
        from transformers import Qwen2VLForConditionalGeneration as QwenModelClass

    model = load_model_with_eager_attention(
        QwenModelClass,
        model_name,
        device=device,
    )
    processor = AutoProcessor.from_pretrained(model_name)

    image_token_id = getattr(processor, "image_token_id", None)
    if image_token_id is None:
        image_token_id = getattr(model.config, "image_token_id", None)
    if image_token_id is None:
        image_token = getattr(processor, "image_token", "<|image_pad|>")
        image_token_id = processor.tokenizer.convert_tokens_to_ids(image_token)

    spatial_merge_size = getattr(
        getattr(model.config, "vision_config", None),
        "spatial_merge_size",
        None,
    )
    if spatial_merge_size is None and hasattr(processor, "image_processor"):
        spatial_merge_size = getattr(processor.image_processor, "merge_size", None)

    text_backbone = getattr(getattr(model, "model", None), "language_model", None)
    if text_backbone is None:
        text_backbone = resolve_text_backbone(model)

    return build_backend(
        family="qwen2vl",
        model=model,
        processor=processor,
        text_backbone=text_backbone,
        image_token_id=image_token_id,
        default_prompt="<image>Describe the image.",
        recommended_image_size=448,
        spatial_merge_size=spatial_merge_size,
    )


def load_lfm2vl_backend(model_name: str, device: str) -> DexarBackend:
    from transformers import AutoModelForImageTextToText, AutoProcessor

    load_kwargs = {
        "device_map": resolve_device_map(device),
        "dtype": torch.bfloat16 if isinstance(device, str) and device.startswith("cuda") else torch.float32,
    }
    try:
        model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            attn_implementation="eager",
            **load_kwargs,
        )
    except TypeError as exc:
        if "attn_implementation" not in str(exc):
            raise
        model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            _attn_implementation="eager",
            **load_kwargs,
        )
    processor = AutoProcessor.from_pretrained(model_name)
    if hasattr(processor, "image_processor"):
        image_processor = processor.image_processor
        if hasattr(image_processor, "do_image_splitting"):
            image_processor.do_image_splitting = False
        if hasattr(image_processor, "use_thumbnail"):
            image_processor.use_thumbnail = False

    image_token_id = getattr(processor, "image_token_id", None)
    if image_token_id is None:
        image_token_id = getattr(model.config, "image_token_id", None)
    if image_token_id is None:
        image_token = getattr(processor, "image_token", "<image>")
        image_token_id = processor.tokenizer.convert_tokens_to_ids(image_token)

    text_backbone = getattr(getattr(model, "model", None), "language_model", None)
    if text_backbone is None:
        text_backbone = resolve_text_backbone(model)

    return build_backend(
        family="lfm2vl",
        model=model,
        processor=processor,
        text_backbone=text_backbone,
        image_token_id=image_token_id,
        default_prompt="<image>Describe the image.",
        recommended_image_size=512,
    )


def load_gemma3_backend(model_name: str, device: str) -> DexarBackend:
    from transformers import AutoProcessor, Gemma3ForConditionalGeneration

    load_kwargs = {
        "device_map": resolve_device_map(device),
        "dtype": torch.bfloat16 if isinstance(device, str) and device.startswith("cuda") else torch.float32,
        "trust_remote_code": True,
    }
    try:
        model = Gemma3ForConditionalGeneration.from_pretrained(
            model_name,
            attn_implementation="eager",
            **load_kwargs,
        )
    except TypeError as exc:
        if "attn_implementation" not in str(exc):
            raise
        model = Gemma3ForConditionalGeneration.from_pretrained(
            model_name,
            _attn_implementation="eager",
            **load_kwargs,
        )

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, use_fast=False)
    image_token_id = getattr(model.config, "image_token_index", None)
    if image_token_id is None:
        image_token_id = getattr(model.config, "image_token_id", None)
    if image_token_id is None:
        image_token_id = processor.tokenizer.convert_tokens_to_ids("<image_soft_token>")

    text_backbone = getattr(getattr(model, "model", None), "language_model", None)
    if text_backbone is None:
        text_backbone = resolve_text_backbone(model)

    return build_backend(
        family="gemma3",
        model=model,
        processor=processor,
        text_backbone=text_backbone,
        image_token_id=image_token_id,
        default_prompt="<image>Describe the image.",
        recommended_image_size=896,
    )


def load_internvl_backend(model_name: str, device: str) -> DexarBackend:
    from transformers import AutoTokenizer

    from src.dataset.internvl_utils import (
        INTERNVL_IMAGE_SIZE,
        INTERNVL_IMG_CONTEXT_TOKEN,
        INTERNVL_MAX_NUM_TILES,
        INTERNVL_NUM_IMAGE_TOKEN,
    )
    from src.train.internvl_compat import load_internvl_model

    resolved_device = "cuda" if device == "auto" and torch.cuda.is_available() else device
    dtype = torch.bfloat16 if isinstance(resolved_device, str) and resolved_device.startswith("cuda") else torch.float32
    model = load_internvl_model(
        model_id=model_name,
        cache_dir=None,
        device=resolved_device,
        compute_dtype=dtype,
        use_flash_attn=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=True,
        padding_side="right",
    )
    image_token_id = tokenizer.convert_tokens_to_ids(INTERNVL_IMG_CONTEXT_TOKEN)
    model.img_context_token_id = image_token_id

    vision_config = getattr(model.config, "vision_config", None)
    image_size = (
        getattr(model.config, "force_image_size", None)
        or getattr(vision_config, "image_size", INTERNVL_IMAGE_SIZE)
    )
    processor = SimpleNamespace(
        tokenizer=tokenizer,
        image_processor_cfg={
            "model_id": model_name,
            "tokenizer": tokenizer,
            "image_size": image_size,
            "normalize_type": "imagenet",
            "max_num_tiles": 1,
            "num_image_token": getattr(model, "num_image_token", INTERNVL_NUM_IMAGE_TOKEN),
        },
    )

    return build_backend(
        family="internvl",
        model=model,
        processor=processor,
        text_backbone=model.language_model,
        image_token_id=image_token_id,
        default_prompt="<image>\nDescribe the image.",
        recommended_image_size=image_size,
        base_image_seq_len=getattr(model, "num_image_token", INTERNVL_NUM_IMAGE_TOKEN),
        custom_generate_answer=generate_internvl_answer,
    )


def generate_internvl_answer(
    backend: DexarBackend,
    image,
    prompt: str,
    max_new_tokens: int,
    device: torch.device,
) -> str:
    encoded_prompt = backend.encode_prompt(prompt=prompt, image=image, device=device)
    generation_inputs = {
        key: value
        for key, value in encoded_prompt.model_inputs.items()
        if key in {"input_ids", "attention_mask", "pixel_values"}
    }
    tokenizer = backend.processor.tokenizer
    eos_token_id = tokenizer.eos_token_id
    template = getattr(backend.model, "conv_template", None)
    if template is not None:
        candidate_eos = tokenizer.convert_tokens_to_ids(str(template.sep).strip())
        if candidate_eos is not None and candidate_eos >= 0:
            eos_token_id = candidate_eos

    with torch.inference_mode():
        generated_ids = backend.model.generate(
            **generation_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=eos_token_id,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    generated_text = tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    if template is not None:
        generated_text = generated_text.split(str(template.sep).strip())[0].strip()
    return generated_text


def generate_florence2_answer(
    backend: DexarBackend,
    image,
    prompt: str,
    max_new_tokens: int,
    device: torch.device,
) -> str:
    encoded_prompt = backend.encode_prompt(prompt=prompt, image=image, device=device)
    encoder_input_ids = encoded_prompt.model_inputs["input_ids"]
    encoder_attention_mask = encoded_prompt.model_inputs.get("attention_mask")
    if encoder_attention_mask is None:
        encoder_attention_mask = torch.ones_like(encoder_input_ids, device=device)
    model_static_inputs = {
        key: value
        for key, value in encoded_prompt.model_inputs.items()
        if key not in {"input_ids", "attention_mask"}
    }
    if "pixel_values" in model_static_inputs:
        model_static_inputs["pixel_values"] = model_static_inputs["pixel_values"].to(
            dtype=next(backend.model.parameters()).dtype
        )
    tokenizer = backend.processor.tokenizer

    text_config = getattr(backend.model.config, "text_config", backend.model.config)
    decoder_start_token_id = getattr(text_config, "decoder_start_token_id", None)
    if decoder_start_token_id is None:
        decoder_start_token_id = getattr(backend.model.config, "decoder_start_token_id", None)
    if decoder_start_token_id is None:
        decoder_start_token_id = tokenizer.bos_token_id
    if decoder_start_token_id is None:
        raise ValueError("Could not resolve decoder_start_token_id for Florence-2 generation.")

    decoder_input_ids = torch.full(
        (1, 1),
        int(decoder_start_token_id),
        device=device,
        dtype=encoder_input_ids.dtype,
    )
    decoder_attention_mask = torch.ones_like(decoder_input_ids, device=device)
    generated_token_ids = []

    with torch.inference_mode():
        for _ in range(max_new_tokens):
            outputs = backend.model(
                input_ids=encoder_input_ids,
                attention_mask=encoder_attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
                use_cache=False,
                **model_static_inputs,
            )
            next_token_id = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated_token_ids.append(next_token_id)
            decoder_input_ids = torch.cat([decoder_input_ids, next_token_id], dim=-1)
            decoder_attention_mask = torch.cat(
                [
                    decoder_attention_mask,
                    torch.ones((1, 1), device=device, dtype=decoder_attention_mask.dtype),
                ],
                dim=1,
            )
            if tokenizer.eos_token_id is not None and int(next_token_id.item()) == int(tokenizer.eos_token_id):
                break

    generated_ids = (
        torch.cat(generated_token_ids, dim=-1)
        if generated_token_ids
        else torch.empty((1, 0), device=device, dtype=encoder_input_ids.dtype)
    )

    generated_text = backend.processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    task_prompt = prompt.strip().split()[0] if prompt.strip().startswith("<") else None
    post_process = getattr(backend.processor, "post_process_generation", None)
    if task_prompt and post_process is not None:
        try:
            parsed = post_process(
                generated_text,
                task=task_prompt,
                image_size=(image.width, image.height),
            )
            if isinstance(parsed, dict) and task_prompt in parsed:
                value = parsed[task_prompt]
                if isinstance(value, str):
                    generated_text = value.strip()
        except (KeyError, TypeError, ValueError):
            pass

    return generated_text
