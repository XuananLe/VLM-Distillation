import inspect
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


def infer_batch_size(inputs: Dict[str, torch.Tensor]) -> int:
    for value in inputs.values():
        if isinstance(value, torch.Tensor) and value.ndim > 0:
            return int(value.shape[0])
    raise ValueError("Could not infer batch size from model inputs.")


def pool_vision_features(features: torch.Tensor, batch_size: int) -> torch.Tensor:
    if features.ndim == 1:
        return features.unsqueeze(0)
    if features.ndim == 2:
        if features.shape[0] == batch_size:
            return features
        if batch_size == 1:
            return features.reshape(-1, features.shape[-1]).mean(dim=0, keepdim=True)
    if features.ndim == 3 and features.shape[0] == batch_size:
        return features.mean(dim=1)
    if features.ndim == 4 and features.shape[0] == batch_size:
        return features.flatten(start_dim=2).mean(dim=2)
    if features.ndim >= 2 and batch_size == 1:
        return features.reshape(-1, features.shape[-1]).mean(dim=0, keepdim=True)
    raise ValueError(
        f"Unsupported vision feature shape {tuple(features.shape)} for batch size {batch_size}."
    )


def prepare_forward_inputs(model, batch):
    parameter = next(model.parameters())
    model_device = parameter.device
    model_dtype = parameter.dtype

    inputs = {}
    for key, value in batch.items():
        if key == "labels" or value is None:
            continue
        if not isinstance(value, torch.Tensor):
            inputs[key] = value
            continue
        tensor = value.to(model_device)
        if torch.is_floating_point(tensor):
            tensor = tensor.to(model_dtype)
        inputs[key] = tensor

    signature = inspect.signature(model.forward)
    accepts_var_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if accepts_var_kwargs:
        return inputs

    allowed = set(signature.parameters.keys())
    return {key: value for key, value in inputs.items() if key in allowed}
