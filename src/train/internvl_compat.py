from contextlib import contextmanager

import torch
import transformers.modeling_utils as modeling_utils
from transformers import AutoModel
from transformers.modeling_utils import PreTrainedModel

# https://huggingface.co/OpenGVLab/InternVL3-1B/tree/main


@contextmanager
def internvl_transformers5_load_context():
    # - OpenGVLab's InternVL3 model card loads with AutoModel + trust_remote_code.
    # - Transformers 5 initializes PreTrainedModel subclasses on the meta device.
    # - InternVL3 remote code calls Tensor.item() in __init__, which fails on meta
    #   tensors, then later lacks Transformers 5's all_tied_weights_keys field.
    # This scoped patch disables meta init only while loading InternVL.
    original_get_init_context = PreTrainedModel.get_init_context
    original_mark_tied_weights_as_initialized = getattr(PreTrainedModel, "mark_tied_weights_as_initialized", None)

    @classmethod
    def cpu_init_context(cls, *args, **kwargs):
        dtype = kwargs.get("dtype", args[0] if args and not isinstance(args[0], bool) else None)
        if dtype is not None and hasattr(modeling_utils, "local_torch_dtype") and hasattr(modeling_utils, "init"):
            return [
                modeling_utils.local_torch_dtype(dtype, cls.__name__),
                modeling_utils.init.no_tie_weights(),
            ]
        if hasattr(modeling_utils, "no_init_weights"):
            return [modeling_utils.no_init_weights()]
        return []

    def mark_tied_weights_as_initialized(self):
        if not hasattr(self, "all_tied_weights_keys"):
            self.all_tied_weights_keys = {}
        return original_mark_tied_weights_as_initialized(self)

    PreTrainedModel.get_init_context = cpu_init_context
    if original_mark_tied_weights_as_initialized is not None:
        PreTrainedModel.mark_tied_weights_as_initialized = mark_tied_weights_as_initialized
    try:
        yield
    finally:
        PreTrainedModel.get_init_context = original_get_init_context
        if original_mark_tied_weights_as_initialized is not None:
            PreTrainedModel.mark_tied_weights_as_initialized = original_mark_tied_weights_as_initialized


def load_internvl_model(
    *,
    model_id: str,
    cache_dir: str | None,
    device,
    compute_dtype: torch.dtype,
    use_flash_attn: bool,
):
    """Load InternVL remote-code models under Transformers 5 and move them to one device."""
    with internvl_transformers5_load_context():
        return AutoModel.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            dtype=compute_dtype,
            low_cpu_mem_usage=False,
            use_flash_attn=use_flash_attn,
            trust_remote_code=True,
        ).to(device)
