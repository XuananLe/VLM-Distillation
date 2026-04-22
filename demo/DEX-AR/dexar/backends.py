import math
from collections import deque
from dataclasses import dataclass
from typing import Any

import torch


def _iter_model_roots(model):
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


def _resolve_attr(model, aliases: tuple[str, ...]) -> Any:
    for root in _iter_model_roots(model):
        for alias in aliases:
            if hasattr(root, alias):
                return getattr(root, alias)
    raise AttributeError(f"Could not resolve any of {aliases!r} on the model.")


def _resolve_text_backbone(model):
    return _resolve_attr(model, ("language_model", "text_model"))


def _resolve_lm_head(model, text_backbone):
    for candidate in (model, text_backbone, getattr(text_backbone, "model", None)):
        if candidate is not None and hasattr(candidate, "lm_head"):
            return getattr(candidate, "lm_head")
    raise AttributeError("Could not resolve lm_head on the model.")


def _resolve_norm(text_backbone):
    for candidate in (text_backbone, getattr(text_backbone, "model", None)):
        if candidate is not None and hasattr(candidate, "norm"):
            return getattr(candidate, "norm")
    raise AttributeError("Could not resolve the final normalization layer.")


def _resolve_layers(text_backbone):
    for candidate in (getattr(text_backbone, "model", None), text_backbone):
        if candidate is None:
            continue
        layers = getattr(candidate, "layers", None)
        if layers is not None:
            return layers
    raise AttributeError("Could not resolve decoder layers on the text backbone.")


def _resolve_device_map(device: str):
    if device in {"auto", "cpu", "cuda"}:
        return device
    return {"": device}


def _resolve_torch_dtype(device: str) -> torch.dtype:
    if isinstance(device, str) and device.startswith("cuda"):
        return torch.float16
    return torch.float32


def _load_model_with_eager_attention(model_cls, model_name: str, *, device: str):
    common_kwargs = {
        "torch_dtype": _resolve_torch_dtype(device),
        "device_map": _resolve_device_map(device),
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

    @property
    def num_layers(self) -> int:
        return len(self.layers)

    @classmethod
    def from_pretrained(cls, model_name: str, device: str):
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        model_type = getattr(config, "model_type", None)

        if model_type == "llava":
            return _load_llava_backend(model_name, device)
        if model_type in {"smolvlm", "idefics3"}:
            return _load_smolvlm_backend(model_name, device)
        if model_type in {"qwen2_vl", "qwen2_5_vl"}:
            return _load_qwen2vl_backend(model_name, device, model_type=model_type)

        raise ValueError(
            f"Unsupported model type {model_type!r} for DEX-AR. "
            "Supported families: LLaVA, SmolVLM/Idefics3, and Qwen2-VL."
        )

    def enable_dexar_gradients(self) -> None:
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        for parameter in self.text_backbone.parameters():
            parameter.requires_grad = True
        for parameter in self.lm_head.parameters():
            parameter.requires_grad = True
        self.model.eval()
        self.model.config.output_attentions = True
        self.model.config.output_hidden_states = True

    def encode_prompt(self, prompt: str, image, device: torch.device) -> EncodedPrompt:
        if prompt.count("<image>") != 1:
            raise ValueError(
                "DEX-AR currently supports prompts with exactly one <image> placeholder."
            )

        if self.family == "qwen2vl":
            prefix, suffix = prompt.split("<image>")
            content = []
            prefix = prefix.strip()
            suffix = suffix.strip()
            if prefix:
                content.append({"type": "text", "text": prefix})
            content.append({"type": "image"})
            if suffix:
                content.append({"type": "text", "text": suffix})
            prompt_text = self.processor.apply_chat_template(
                [{"role": "user", "content": content}],
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


def _build_backend(
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
) -> DexarBackend:
    if text_backbone is None:
        text_backbone = _resolve_text_backbone(model)
    return DexarBackend(
        family=family,
        model=model,
        processor=processor,
        text_backbone=text_backbone,
        lm_head=_resolve_lm_head(model, text_backbone),
        norm=_resolve_norm(text_backbone),
        layers=_resolve_layers(text_backbone),
        image_token_id=image_token_id,
        default_prompt=default_prompt,
        recommended_image_size=recommended_image_size,
        base_image_seq_len=base_image_seq_len,
        spatial_merge_size=spatial_merge_size,
    )


def _load_llava_backend(model_name: str, device: str) -> DexarBackend:
    from transformers import AutoProcessor, LlavaForConditionalGeneration

    model = _load_model_with_eager_attention(
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

    return _build_backend(
        family="llava",
        model=model,
        processor=processor,
        image_token_id=image_token_id,
        default_prompt="USER: <image>\nDescribe the image. ASSISTANT:",
        recommended_image_size=336,
    )


def _load_smolvlm_backend(model_name: str, device: str) -> DexarBackend:
    from transformers import AutoProcessor, Idefics3ForConditionalGeneration

    model = _load_model_with_eager_attention(
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

    return _build_backend(
        family="smolvlm",
        model=model,
        processor=processor,
        image_token_id=image_token_id,
        default_prompt="<|im_start|>User:<image>Describe the image.<end_of_utterance>\nAssistant:",
        recommended_image_size=recommended_image_size,
        base_image_seq_len=base_image_seq_len,
    )


def _load_qwen2vl_backend(
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

    model = _load_model_with_eager_attention(
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
        text_backbone = _resolve_text_backbone(model)

    return _build_backend(
        family="qwen2vl",
        model=model,
        processor=processor,
        text_backbone=text_backbone,
        image_token_id=image_token_id,
        default_prompt="<image>Describe the image.",
        recommended_image_size=448,
        spatial_merge_size=spatial_merge_size,
    )
