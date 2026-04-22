import re
from typing import Dict

import torch


def forward_with_kwarg_retry(model, call_inputs):
    inputs = dict(call_inputs)
    while True:
        try:
            return model(**inputs)
        except TypeError as exc:
            match = re.search(r"unexpected keyword argument '([^']+)'", str(exc))
            if not match:
                raise
            bad_key = match.group(1)
            if bad_key not in inputs:
                raise
            inputs.pop(bad_key, None)


def unwrap_tensor(output):
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)):
        for item in output:
            tensor = unwrap_tensor(item)
            if tensor is not None:
                return tensor
        return None
    if isinstance(output, dict):
        for value in output.values():
            tensor = unwrap_tensor(value)
            if tensor is not None:
                return tensor
        return None
    for attr_name in ("last_hidden_state", "hidden_states"):
        value = getattr(output, attr_name, None)
        if isinstance(value, (tuple, list)) and value:
            return unwrap_tensor(value[-1])
        tensor = unwrap_tensor(value)
        if tensor is not None:
            return tensor
    return None


def get_hidden_states_from_outputs(outputs):
    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states is not None:
        return tuple(hidden_states)

    if isinstance(outputs, dict):
        hidden_states = outputs.get("hidden_states")
        if hidden_states is not None:
            return tuple(hidden_states)

    for attr_name in (
        "language_model_outputs",
        "language_model_output",
        "text_model_output",
        "text_outputs",
        "model_outputs",
    ):
        nested = getattr(outputs, attr_name, None)
        if nested is None and isinstance(outputs, dict):
            nested = outputs.get(attr_name)
        nested_hidden_states = getattr(nested, "hidden_states", None) if nested is not None else None
        if nested_hidden_states is not None:
            return tuple(nested_hidden_states)

    if isinstance(outputs, (tuple, list)):
        for item in outputs:
            if item is None:
                continue
            nested_hidden_states = getattr(item, "hidden_states", None)
            if nested_hidden_states is not None:
                return tuple(nested_hidden_states)

    raise RuntimeError("Model forward output did not expose `hidden_states`.")


def get_decoder_hidden_states(outputs):
    hidden_states = get_hidden_states_from_outputs(outputs)
    if not hidden_states:
        raise RuntimeError("Model returned an empty hidden_states tuple.")
    return list(hidden_states if len(hidden_states) == 1 else hidden_states[1:])


def pool_model_hidden_states(hidden_states: torch.Tensor, attention_mask) -> torch.Tensor:
    if hidden_states.ndim == 1:
        return hidden_states.unsqueeze(0)
    if hidden_states.ndim == 2:
        return hidden_states.mean(dim=0, keepdim=True)
    if hidden_states.ndim != 3:
        raise ValueError(
            f"Unsupported hidden-state shape for model-layer pooling: {tuple(hidden_states.shape)}"
        )

    if attention_mask is not None and attention_mask.ndim == 2 and attention_mask.shape == hidden_states.shape[:2]:
        mask = attention_mask.to(device=hidden_states.device, dtype=hidden_states.dtype).unsqueeze(-1)
        return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    return hidden_states.mean(dim=1)


def infer_batch_size(inputs: Dict[str, torch.Tensor]) -> int:
    for value in inputs.values():
        if isinstance(value, torch.Tensor) and value.ndim > 0:
            return int(value.shape[0])
    raise ValueError("Could not infer batch size from model inputs.")
