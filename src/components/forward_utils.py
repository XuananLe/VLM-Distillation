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


def infer_batch_size(inputs: Dict[str, torch.Tensor]) -> int:
    for value in inputs.values():
        if isinstance(value, torch.Tensor) and value.ndim > 0:
            return int(value.shape[0])
    raise ValueError("Could not infer batch size from model inputs.")
