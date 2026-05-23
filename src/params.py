from dataclasses import dataclass, field
from typing import Optional

import transformers.training_args as _ta
from transformers import TrainingArguments as HFTrainingArguments



@dataclass
class ModelArguments:
    """Minimal model-selection arguments for the SFT entrypoint."""
    model_id: Optional[str] = field(default=None)


@dataclass
class TrainingArguments(HFTrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    seed: int = field(default=42)
    data_seed: int = field(default=42)
    adam_beta1: float = field(default=0.9)
    adam_beta2: float = field(default=0.98)
    adam_epsilon: float = field(default=1e-7)

    disable_flash_attn2: bool = field(default=False)
    freeze_vision_tower: bool = field(default=False)
    freeze_llm: bool = field(default=False)
    freeze_connector: bool = field(default=False)
    vision_lr: Optional[float] = field(default=None)
    connector_lr: Optional[float] = field(default=None)

    max_seq_length: int = field(
        default=16384,
        metadata={
            "help":
                "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    def __post_init__(self):
        """Restore compatibility with TRL-mutated TrainingArguments field validation."""
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
    """Dataset paths and image-loading options shared by SFT and distillation."""
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: Optional[str] = field(
        default=None, metadata={"help": "Optional path to the validation data."}
    )
    lazy_preprocess: bool = False
    image_folder: Optional[str] = field(default=None)
    strict_image_validation: bool = False
