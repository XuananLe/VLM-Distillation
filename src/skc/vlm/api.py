import torch

from .families import resolve_model_family
from .loaders import build_load_kwargs, load_model_for_family


def load_vlm(model_name: str, dtype: torch.dtype):
    cfg, family = resolve_model_family(model_name)
    load_kwargs, auto_load_kwargs = build_load_kwargs(dtype)
    model = load_model_for_family(
        model_name=model_name,
        family=family,
        cfg=cfg,
        dtype=dtype,
        load_kwargs=load_kwargs,
        auto_load_kwargs=auto_load_kwargs,
    )

    if not torch.cuda.is_available():
        model = model.to("cpu")

    model.eval()
    return model, family
