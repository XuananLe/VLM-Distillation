import os
from collections import deque
from typing import override

import torch
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
)

class VisionLanguageSFTTrainer(Trainer):
    """Trainer with VLM-specific optimizer grouping and DeepSpeed recovery hooks."""
    _DEEPSPEED_CANDIDATE_ATTRS = (
        "deepspeed",
        "deepspeed_engine",
        "deepspeed_engine_wrapped",
        "engine",
        "module",
        "model",
        "model_wrapped",
        "wrapped_model",
    )

    def has_broken_deepspeed_wrapper(self) -> bool:
        """Return whether Accelerate lost its DeepSpeed wrapper even though DeepSpeed is active."""
        accelerator = getattr(self, "accelerator", None)
        if accelerator is None:
            return False
        if getattr(accelerator, "deepspeed_engine_wrapped", None) is not None:
            return False
        distributed_type = getattr(accelerator, "distributed_type", None)
        return str(distributed_type).endswith("DEEPSPEED")

    def _iter_deepspeed_candidates(self, model):
        """Yield objects that may hold the active DeepSpeed engine inside trainer state."""
        accelerator = getattr(self, "accelerator", None)
        queue = deque(
            (
                model,
                getattr(self, "model_wrapped", None),
                getattr(self, "deepspeed", None),
                getattr(accelerator, "deepspeed_engine", None) if accelerator is not None else None,
                getattr(accelerator, "deepspeed_engine_wrapped", None) if accelerator is not None else None,
                getattr(accelerator, "_models", None) if accelerator is not None else None,
                getattr(self, "model", None),
            )
        )
        seen: set[int] = set()

        while queue:
            candidate = queue.popleft()
            if candidate is None:
                continue

            if isinstance(candidate, dict):
                queue.extend(candidate.values())
                continue
            if isinstance(candidate, (list, tuple, set)):
                queue.extend(candidate)
                continue

            candidate_id = id(candidate)
            if candidate_id in seen:
                continue
            seen.add(candidate_id)
            yield candidate

            for attr_name in self._DEEPSPEED_CANDIDATE_ATTRS:
                nested_candidate = getattr(candidate, attr_name, None)
                if nested_candidate is not None:
                    queue.append(nested_candidate)

    def _restore_deepspeed_wrapper(self, engine) -> None:
        """Recreate Accelerate's DeepSpeed wrapper around a resolved engine when it went missing."""
        accelerator = getattr(self, "accelerator", None)
        if accelerator is None or getattr(accelerator, "deepspeed_engine_wrapped", None) is not None:
            return

        try:
            from accelerate.utils.deepspeed import DeepSpeedEngineWrapper
        except ImportError as exc:
            raise RuntimeError(
                "DeepSpeed engine wrapper restoration requires "
                "`accelerate.utils.deepspeed.DeepSpeedEngineWrapper`."
            ) from exc

        accelerator.deepspeed_engine_wrapped = DeepSpeedEngineWrapper(engine)

    def resolve_deepspeed_engine(self, model):
        """Find the live DeepSpeed engine object that exposes backward and step."""
        checked_candidates: list[str] = []
        for candidate in self._iter_deepspeed_candidates(model):
            checked_candidates.append(type(candidate).__name__)
            if hasattr(candidate, "backward") and hasattr(candidate, "step"):
                self._restore_deepspeed_wrapper(candidate)
                return candidate
        raise RuntimeError(
            "DeepSpeed training is enabled, but no engine exposing backward()/step() "
            "could be resolved from the trainer state. "
            f"Checked candidates: {checked_candidates}"
        )

    def manual_deepspeed_training_step(self, model, inputs, num_items_in_batch=None) -> torch.Tensor:
        """Run a manual DeepSpeed backward/step path when the normal wrapper is broken."""
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
        accelerator = getattr(self, "accelerator", None)
        sync_gradients = getattr(accelerator, "sync_gradients", getattr(self, "sync_gradients", True))

        if hasattr(deepspeed_engine, "set_gradient_accumulation_boundary"):
            deepspeed_engine.set_gradient_accumulation_boundary(is_boundary=sync_gradients)
            deepspeed_engine.backward(loss, scale_wrt_gas=False)
            if sync_gradients:
                deepspeed_engine.step()
        else:
            deepspeed_engine.backward(
                loss,
                sync_gradients=sync_gradients,
                scale_wrt_gas=False,
            )
            if sync_gradients and hasattr(deepspeed_engine, "step"):
                deepspeed_engine.step()

        gradient_accumulation_steps = getattr(
            self,
            "current_gradient_accumulation_steps",
            self.args.gradient_accumulation_steps,
        )
        return loss.detach() / gradient_accumulation_steps

    @override
    def training_step(self, model, inputs, num_items_in_batch=None) -> torch.Tensor:
        """Delegate to Trainer.training_step and fall back to manual DeepSpeed recovery on wrapper bugs."""
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

    @override
    def create_optimizer(self):
        """Build the optimizer with optional vision and connector learning-rate overrides."""
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
        return self.optimizer

    @override
    def _save_checkpoint(self, model, trial):
        """Save the checkpoint and persist the active processor/tokenizer assets beside it."""
        super()._save_checkpoint(model, trial)
        output_dir = os.path.join(
            self._get_output_dir(trial=trial),
            f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}",
        )
        _save_processing_assets(self, output_dir)
