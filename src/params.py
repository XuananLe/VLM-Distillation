from dataclasses import dataclass, field
from typing import Optional

try:
    from accelerate.utils import ParallelismConfig as _PC
except Exception:
    class _PC:
        pass

import transformers.training_args as _ta
if not hasattr(_ta, "ParallelismConfig"):
    _ta.ParallelismConfig = _PC

from transformers import TrainingArguments as HFTrainingArguments



@dataclass
class ModelArguments:
    model_id: Optional[str] = field(default="HuggingFaceTB/SmolVLM-Instruct")


@dataclass
class TrainingArguments(HFTrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    seed: int = field(default=42)
    data_seed: int = field(default=42)
    adam_beta1: float = field(default=0.9)
    adam_beta2: float = field(default=0.98)
    adam_epsilon: float = field(default=1e-7)

    freeze_vision_tower: bool = field(default=False)
    freeze_llm: bool = field(default=False)
    freeze_connector: bool = field(default=False)
    disable_flash_attn2: bool = field(default=False)
    unfreeze_topk_llm: int = 0
    unfreeze_topk_vision: int = 0

    max_seq_length: int = field(
        default=16384, # This is the default value of the SmolVLM model
        metadata={
            "help":
                "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )

    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    vision_lora: bool = False
    use_dora: bool = False
    lora_rank: int = 64
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    vision_lr: Optional[float] = None
    connector_lr: Optional[float] = None
    early_stopping_patience: Optional[int] = field(
        default=None,
        metadata={"help": "Stop training when the validation metric fails to improve for this many evaluation calls."},
    )
    early_stopping_threshold: float = field(
        default=0.0,
        metadata={"help": "Minimum absolute improvement required to reset early stopping patience."},
    )
    lora_namespan_exclude: str = field(default=None, metadata={"help": "List of namespan to exclude for LoRA"})
    num_lora_modules: int = -1

    def __post_init__(self):
        # TRL 0.17 mutates transformers.training_args._VALID_DICT_FIELDS to include
        # fields like `model_init_kwargs` that do not exist on plain
        # transformers.TrainingArguments in transformers 4.47.x.
        original_valid_dict_fields = None
        if hasattr(_ta, "_VALID_DICT_FIELDS"):
            original_valid_dict_fields = list(_ta._VALID_DICT_FIELDS)
            _ta._VALID_DICT_FIELDS = [
                field_name for field_name in original_valid_dict_fields if hasattr(self, field_name)
            ]

        try:
            super().__post_init__()
        finally:
            if original_valid_dict_fields is not None:
                _ta._VALID_DICT_FIELDS = original_valid_dict_fields


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: Optional[str] = field(
        default=None, metadata={"help": "Optional path to the validation data."}
    )
    lazy_preprocess: bool = False
    image_folder: Optional[str] = field(default=None)
    max_num_frames: int = 10
    strict_image_validation: bool = False
