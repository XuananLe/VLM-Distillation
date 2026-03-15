import inspect
import sys
import types


def patch_llama_flash_attention2_symbol() -> None:
    try:
        from transformers.models.llama import modeling_llama
    except Exception:
        return

    if hasattr(modeling_llama, "LlamaFlashAttention2"):
        return
    if hasattr(modeling_llama, "LlamaAttention"):
        modeling_llama.LlamaFlashAttention2 = modeling_llama.LlamaAttention


def patch_qwen_vl_stream_generator() -> None:
    if "transformers_stream_generator" in sys.modules:
        return

    shim = types.ModuleType("transformers_stream_generator")

    def init_stream_support(*_, **__):
        return None

    shim.init_stream_support = init_stream_support
    sys.modules["transformers_stream_generator"] = shim


def patch_dynamic_cache_get_usable_length() -> bool:
    try:
        from transformers.cache_utils import DynamicCache
    except Exception:
        return False

    if hasattr(DynamicCache, "get_usable_length"):
        return False

    def get_usable_length(self, new_seq_length=0, layer_idx=0):
        del new_seq_length
        return self.get_seq_length(layer_idx)

    DynamicCache.get_usable_length = get_usable_length
    return True


def build_phi4_prepare_inputs_for_generation():
    def prepare_inputs_for_generation(self, input_ids=None, **kwargs):
        data = dict(kwargs)
        if input_ids is not None:
            data["input_ids"] = input_ids
        return data

    return prepare_inputs_for_generation


def patch_phi4_prepare_inputs_for_generation(model_name: str | None = None) -> bool:
    patched = False
    candidate_classes = []

    if model_name is not None:
        try:
            from transformers.dynamic_module_utils import get_class_from_dynamic_module

            candidate_classes.append(
                get_class_from_dynamic_module("modeling_phi4mm.Phi4MMModel", model_name)
            )
        except Exception:
            pass

    for module in list(sys.modules.values()):
        cls = getattr(module, "Phi4MMModel", None)
        if cls is None:
            continue
        candidate_classes.append(cls)

    prepare_inputs_for_generation = build_phi4_prepare_inputs_for_generation()
    seen = set()
    for cls in candidate_classes:
        if cls is None or id(cls) in seen:
            continue
        if not inspect.isclass(cls):
            continue
        seen.add(id(cls))
        if any("prepare_inputs_for_generation" in base.__dict__ for base in cls.mro()):
            continue
        cls.prepare_inputs_for_generation = prepare_inputs_for_generation
        patched = True

    return patched
