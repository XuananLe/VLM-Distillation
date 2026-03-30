import os
from typing import override

import torch
import torch.nn as nn

from transformers import Trainer
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    PREFIX_CHECKPOINT_DIR,
    logger,
)
from transformers.pytorch_utils import (
    ALL_LAYERNORM_LAYERS
)
from src.train.train_utils import get_peft_state_non_lora_maybe_zero_3, _save_processing_assets

class SmolVLMSFTTrainer(Trainer):

    def _save_non_lora_weights(self, output_dir: str, *, require_grad_only: bool) -> None:
        if not self.args.lora_enable:
            return
        torch.save(
            get_peft_state_non_lora_maybe_zero_3(
                self.model.named_parameters(),
                require_grad_only=require_grad_only,
            ),
            os.path.join(output_dir, "non_lora_state_dict.bin"),
        )

    @override
    def create_optimizer(self):
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        if self.optimizer is None:
            opt_model = self.model
            named_params = [(name, param) for name, param in opt_model.named_parameters() if param.requires_grad]
            decay_names = {
                name for name in get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS) if "bias" not in name
            }
            lr_map = {
                name: lr
                for name, lr in (
                    ("vision_model", self.args.vision_lr),
                    ("connector", self.args.connector_lr),
                )
                if lr is not None
            }
            param_groups = {(module_name, decay): [] for module_name in (None, *lr_map) for decay in (True, False)}
            for name, param in named_params:
                module_name = next((key for key in lr_map if key in name), None)
                param_groups[(module_name, name in decay_names)].append(param)

            optimizer_grouped_parameters = []
            for (module_name, decay), params in param_groups.items():
                if not params:
                    continue
                group = {
                    "params": params,
                    "weight_decay": self.args.weight_decay if decay else 0.0,
                }
                if module_name is not None:
                    group["lr"] = lr_map[module_name]
                optimizer_grouped_parameters.append(group)

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                        logger.info(f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(module, "weight", {"optim_bits": 32})
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer

    @override
    def _save_checkpoint(self, model, trial):
        super()._save_checkpoint(model, trial)
        output_dir = os.path.join(
            self._get_output_dir(trial=trial),
            f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}",
        )
        _save_processing_assets(self, output_dir)
        self._save_non_lora_weights(
            output_dir,
            require_grad_only=getattr(self, "non_lora_require_grad_only", False),
        )
