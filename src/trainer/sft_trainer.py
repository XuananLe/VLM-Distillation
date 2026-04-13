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
from src.train.train_utils import (
    _save_processing_assets,
    build_component_parameter_id_map,
    get_peft_state_non_lora_maybe_zero_3,
)

class VisionLanguageSFTTrainer(Trainer):
    def has_broken_deepspeed_wrapper(self) -> bool:
        accelerator = getattr(self, "accelerator", None)
        if accelerator is None:
            return False
        if getattr(accelerator, "deepspeed_engine_wrapped", None) is not None:
            return False
        distributed_type = getattr(accelerator, "distributed_type", None)
        return str(distributed_type).endswith("DEEPSPEED")

    def resolve_deepspeed_engine(self, model):
        accelerator = getattr(self, "accelerator", None)
        for candidate in (
            model,
            getattr(self, "model_wrapped", None),
            getattr(accelerator, "deepspeed_engine", None) if accelerator is not None else None,
            getattr(self, "model", None),
        ):
            if candidate is not None and hasattr(candidate, "backward") and hasattr(candidate, "step"):
                return candidate
        raise RuntimeError(
            "DeepSpeed training is enabled, but no engine exposing backward()/step() "
            "could be resolved from the trainer state."
        )

    def manual_deepspeed_training_step(self, model, inputs, num_items_in_batch=None) -> torch.Tensor:
        model.train()
        if hasattr(self.optimizer, "train"):
            self.optimizer.train()

        inputs = self._prepare_inputs(inputs)
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)
        del inputs

        if self.args.n_gpu > 1:
            loss = loss.mean()

        deepspeed_engine = self.resolve_deepspeed_engine(model)
        deepspeed_engine.backward(
            loss,
            sync_gradients=getattr(self, "sync_gradients", True),
            scale_wrt_gas=False,
        )
        deepspeed_engine.step()

        gradient_accumulation_steps = getattr(
            self,
            "current_gradient_accumulation_steps",
            self.args.gradient_accumulation_steps,
        )
        return loss.detach() / gradient_accumulation_steps

    @override
    def training_step(self, model, inputs, num_items_in_batch=None) -> torch.Tensor:
        try:
            return super().training_step(
                model,
                inputs,
                num_items_in_batch=num_items_in_batch,
            )
        except AttributeError as exc:
            if (
                not self.has_broken_deepspeed_wrapper()
                or "'NoneType' object has no attribute 'backward'" not in str(exc)
            ):
                raise
            logger.warning(
                "Accelerate reported a missing DeepSpeed engine wrapper during backward; "
                "falling back to manual DeepSpeed engine.backward()/step()."
            )
            return self.manual_deepspeed_training_step(
                model,
                inputs,
                num_items_in_batch=num_items_in_batch,
            )

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
                component: lr
                for component, lr in (
                    ("vision", self.args.vision_lr),
                    ("connector", self.args.connector_lr),
                )
                if lr is not None
            }
            component_parameter_ids = build_component_parameter_id_map(
                opt_model,
                components=tuple(lr_map),
            )
            param_groups = {(component, decay): [] for component in (None, *lr_map) for decay in (True, False)}
            for name, param in named_params:
                component = component_parameter_ids.get(id(param))
                param_groups[(component, name in decay_names)].append(param)

            optimizer_grouped_parameters = []
            for (component, decay), params in param_groups.items():
                if not params:
                    continue
                group = {
                    "params": params,
                    "weight_decay": self.args.weight_decay if decay else 0.0,
                }
                if component is not None:
                    group["lr"] = lr_map[component]
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
