import contextlib
import io
import os
import torch.nn.functional as F
from transformers import PreTrainedModel
import inspect
import torch
from transformers.trainer import PREFIX_CHECKPOINT_DIR, TRAINER_STATE_NAME, ExportableState, SaveStrategy
from src.components.vision_forward import forward_with_kwarg_retry
from src.trainer.sft_trainer import SmolVLMSFTTrainer
from src.train.train_utils import get_peft_state_non_lora_maybe_zero_3


class DistillationTrainer(SmolVLMSFTTrainer):
    def __init__(
        self,
        teacher_model: PreTrainedModel = None,
        loss_function: str = "forward_kl",
        temperature: float = 2.0,
        alpha: float = 0.5,
        representation_loss_weight: float = 0.0,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        from src.components import loss as distillation_loss_module
        self.distillation_loss_fn = getattr(distillation_loss_module, loss_function)
        self.loss_function = loss_function
        self.distillation_loss_params = inspect.signature(self.distillation_loss_fn).parameters

        if teacher_model is None:
            raise ValueError("teacher_model must be provided for distillation.")
        if isinstance(teacher_model, (list, tuple)):
            self.teacher_models = list(teacher_model)
        else:
            self.teacher_models = [teacher_model]
        for model in self.teacher_models:
            model.eval()
        self.temperature = temperature
        self.alpha = alpha
        self.representation_loss_weight = representation_loss_weight
        self.latest_ce_loss = None

        print(f"Distillation Trainer initialized:")
        print(f"  - Teachers: {len(self.teacher_models)}")
        print(f"  - Loss function: {loss_function}")
        print(f"  - Temperature: {temperature}")
        print(f"  - Alpha: {alpha}")
        if representation_loss_weight > 0:
            print(f"  - Representation loss weight: {representation_loss_weight}")

    @staticmethod
    def _teacher_prefixes(inputs) -> list[str]:
        prefixes = []
        if "teacher_input_ids" in inputs:
            prefixes.append("teacher")
        index = 0
        while f"teacher_{index}_input_ids" in inputs:
            prefixes.append(f"teacher_{index}")
            index += 1
        return prefixes

    @staticmethod
    def _teacher_batch_from_inputs(inputs, prefix: str):
        teacher_inputs = {
            "input_ids": inputs[f"{prefix}_input_ids"],
            "attention_mask": inputs[f"{prefix}_attention_mask"],
            "pixel_values": inputs[f"{prefix}_pixel_values"],
        }
        pixel_attention_key = f"{prefix}_pixel_attention_mask"
        if pixel_attention_key in inputs:
            teacher_inputs["pixel_attention_mask"] = inputs[pixel_attention_key]
        image_grid_key = f"{prefix}_image_grid_thw"
        if image_grid_key in inputs:
            teacher_inputs["image_grid_thw"] = inputs[image_grid_key]
        image_flags_key = f"{prefix}_image_flags"
        if image_flags_key in inputs:
            teacher_inputs["image_flags"] = inputs[image_flags_key]
        return teacher_inputs, inputs[f"{prefix}_labels"]

    def _compute_single_teacher_loss(
        self,
        student_logits: torch.Tensor,
        student_labels: torch.Tensor,
        teacher_logits: torch.Tensor,
        teacher_labels: torch.Tensor,
    ) -> torch.Tensor:
        student_mask = student_labels != -100
        teacher_mask = teacher_labels != -100

        if student_mask.sum() == 0 or teacher_mask.sum() == 0:
            return torch.tensor(0.0, device=student_logits.device)

        if self.loss_function == "uld_loss":
            sample_losses = []
            for i in range(student_logits.size(0)):
                student_logits_masked = student_logits[i][student_mask[i]]
                teacher_logits_masked = teacher_logits[i][teacher_mask[i]]
                if student_logits_masked.size(0) == 0 or teacher_logits_masked.size(0) == 0:
                    continue
                min_len = min(student_logits_masked.size(0), teacher_logits_masked.size(0))
                sample_losses.append(
                    self.distillation_loss_fn(
                        student_logits=student_logits_masked[:min_len],
                        teacher_logits=teacher_logits_masked[:min_len],
                        temperature=self.temperature,
                    )
                )
            if sample_losses:
                return torch.stack(sample_losses).mean()
            return torch.tensor(0.0, device=student_logits.device)

        student_logits_masked = student_logits.view(-1, student_logits.size(-1))[student_mask.view(-1)]
        teacher_logits_masked = teacher_logits.view(-1, teacher_logits.size(-1))[teacher_mask.view(-1)]
        student_labels_masked = student_labels.view(-1)[student_mask.view(-1)]
        min_len = min(student_logits_masked.size(0), teacher_logits_masked.size(0))
        loss_kwargs = dict(
            student_logits=student_logits_masked[:min_len],
            teacher_logits=teacher_logits_masked[:min_len],
            temperature=self.temperature,
        )
        if "labels" in self.distillation_loss_params:
            loss_kwargs["labels"] = student_labels_masked[:min_len]
        return self.distillation_loss_fn(**loss_kwargs)

    @staticmethod
    def _extract_last_hidden_state(model_outputs):
        hidden_states = getattr(model_outputs, "hidden_states", None)
        if isinstance(hidden_states, (tuple, list)) and hidden_states:
            return hidden_states[-1]

        last_hidden_state = getattr(model_outputs, "last_hidden_state", None)
        if isinstance(last_hidden_state, torch.Tensor):
            return last_hidden_state

        if isinstance(model_outputs, (tuple, list)):
            for item in model_outputs:
                if isinstance(item, torch.Tensor) and item.ndim >= 3:
                    return item

        raise ValueError("Could not extract a final hidden-state tensor from model outputs.")

    @staticmethod
    def _teacher_adapter_modules(model):
        if hasattr(model, "teacher_output_adapters"):
            return model.teacher_output_adapters
        if hasattr(model, "module") and hasattr(model.module, "teacher_output_adapters"):
            return model.module.teacher_output_adapters
        return None

    @staticmethod
    def _forward_model(model, model_inputs, *, output_hidden_states: bool):
        call_inputs = dict(model_inputs)
        call_inputs["return_dict"] = True
        if output_hidden_states:
            call_inputs["output_hidden_states"] = True
        return forward_with_kwarg_retry(model, call_inputs)

    @staticmethod
    def _representation_mask(labels: torch.Tensor) -> torch.Tensor:
        return labels != -100

    def _compute_representation_loss(
        self,
        student_hidden_states: torch.Tensor,
        student_labels: torch.Tensor,
        teacher_hidden_states: torch.Tensor,
        teacher_labels: torch.Tensor,
    ) -> torch.Tensor:
        student_mask = self._representation_mask(student_labels)
        teacher_mask = self._representation_mask(teacher_labels)

        if student_mask.sum() == 0 or teacher_mask.sum() == 0:
            return torch.tensor(0.0, device=student_hidden_states.device)

        sample_losses = []
        for i in range(student_hidden_states.size(0)):
            student_hidden_masked = student_hidden_states[i][student_mask[i]]
            teacher_hidden_masked = teacher_hidden_states[i][teacher_mask[i]]
            if student_hidden_masked.size(0) == 0 or teacher_hidden_masked.size(0) == 0:
                continue
            min_len = min(student_hidden_masked.size(0), teacher_hidden_masked.size(0))
            sample_losses.append(
                F.mse_loss(
                    student_hidden_masked[:min_len],
                    teacher_hidden_masked[:min_len].to(student_hidden_masked.dtype),
                )
            )

        if sample_losses:
            return torch.stack(sample_losses).mean()
        return torch.tensor(0.0, device=student_hidden_states.device)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Use the device of the input tensors as the reference, not model.device.
        # With DeepSpeed ZeRO-2/3 the student model's .device property can return
        # an unexpected value because parameters are sharded across ranks.
        target_device = inputs["input_ids"].device
        for index, teacher_model in enumerate(self.teacher_models):
            if next(teacher_model.parameters()).device != target_device:
                self.teacher_models[index] = teacher_model.to(target_device)

        teacher_prefixes = self._teacher_prefixes(inputs)
        student_inputs = {k: v for k, v in inputs.items() if not k.startswith("teacher")}

        teacher_batches = []
        if teacher_prefixes:
            for prefix in teacher_prefixes:
                teacher_batches.append(self._teacher_batch_from_inputs(inputs, prefix))
        else:
            if len(self.teacher_models) != 1:
                raise ValueError(
                    "Multiple teachers require explicit teacher-specific batch inputs."
                )
            teacher_batches.append(
                (
                    {k: v for k, v in student_inputs.items() if k != "labels"},
                    student_inputs.get("labels"),
                )
            )

        if len(teacher_batches) != len(self.teacher_models):
            raise ValueError(
                f"Teacher inputs/model count mismatch: got {len(teacher_batches)} teacher batches "
                f"for {len(self.teacher_models)} teacher models."
            )

        need_hidden_states = self.representation_loss_weight > 0
        student_outputs = self._forward_model(
            model,
            student_inputs,
            output_hidden_states=need_hidden_states,
        )
        student_logits = student_outputs.logits
        student_hidden_states = (
            self._extract_last_hidden_state(student_outputs) if need_hidden_states else None
        )

        student_labels = student_inputs.get("labels")
        assert student_labels is not None, "Labels must be provided for distillation loss masking"

        teacher_losses = []
        teacher_hidden_targets = []
        with torch.no_grad():
            for teacher_model, (teacher_inputs, teacher_labels) in zip(self.teacher_models, teacher_batches):
                if getattr(teacher_model, "_suppress_forward_stdout", False):
                    with contextlib.redirect_stdout(io.StringIO()):
                        teacher_outputs = self._forward_model(
                            teacher_model,
                            teacher_inputs,
                            output_hidden_states=need_hidden_states,
                        )
                else:
                    teacher_outputs = self._forward_model(
                        teacher_model,
                        teacher_inputs,
                        output_hidden_states=need_hidden_states,
                    )
                teacher_logits = teacher_outputs.logits.detach()
                teacher_losses.append(
                    self._compute_single_teacher_loss(
                        student_logits=student_logits,
                        student_labels=student_labels,
                        teacher_logits=teacher_logits,
                        teacher_labels=teacher_labels,
                    )
                )
                if need_hidden_states:
                    teacher_hidden_targets.append(
                        (
                            self._extract_last_hidden_state(teacher_outputs).detach(),
                            teacher_labels,
                        )
                    )

        if teacher_losses:
            distillation_loss = torch.stack(teacher_losses).mean()
        else:
            distillation_loss = torch.tensor(0.0, device=student_logits.device)

        representation_loss = torch.tensor(0.0, device=student_logits.device)
        if need_hidden_states:
            teacher_adapters = self._teacher_adapter_modules(model)
            if teacher_adapters is None:
                raise ValueError("Representation distillation requested but no teacher_output_adapters found.")
            if len(teacher_adapters) != len(teacher_hidden_targets):
                raise ValueError(
                    "Teacher adapter count does not match teacher hidden-state targets: "
                    f"{len(teacher_adapters)} vs {len(teacher_hidden_targets)}"
                )

            representation_losses = []
            for teacher_adapter, (teacher_hidden_states, teacher_labels) in zip(
                teacher_adapters, teacher_hidden_targets
            ):
                adapter_dtype = next(teacher_adapter.parameters()).dtype
                adapted_teacher_hidden_states = teacher_adapter(
                    teacher_hidden_states.to(dtype=adapter_dtype)
                )
                representation_losses.append(
                    self._compute_representation_loss(
                        student_hidden_states=student_hidden_states,
                        student_labels=student_labels,
                        teacher_hidden_states=adapted_teacher_hidden_states,
                        teacher_labels=teacher_labels,
                    )
                )

            if representation_losses:
                representation_loss = torch.stack(representation_losses).mean()
                distillation_loss = (
                    distillation_loss
                    + self.representation_loss_weight * representation_loss
                )

        ce_loss = student_outputs.loss
        self.latest_ce_loss = ce_loss.detach().float().item()
        loss = self.alpha * distillation_loss + (1 - self.alpha) * ce_loss

        if self.state.global_step % self.args.logging_steps == 0:
            metrics = {
                "distillation_loss": distillation_loss.item(),
                "ce_loss": ce_loss.item(),
            }
            if need_hidden_states:
                metrics["representation_loss"] = representation_loss.item()
            self.log(metrics)

        return (loss, student_outputs) if return_outputs else loss

    def _save_checkpoint(self, model, trial):
        current_ce_loss = self.latest_ce_loss
        improved = (
            current_ce_loss is not None
            and (self.state.best_metric is None or current_ce_loss < self.state.best_metric)
        )
        if improved:
            self.state.best_metric = current_ce_loss
            self.state.best_global_step = self.state.global_step

        if self.args.lora_enable:
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            if self.hp_search_backend is None and trial is None:
                self.store_flos()

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)
            self.save_model(output_dir, _internal_call=True)
            self._save_processor_assets(output_dir)
            non_lora_weights = get_peft_state_non_lora_maybe_zero_3(
                self.model.named_parameters(),
                require_grad_only=True,
            )
            torch.save(non_lora_weights, os.path.join(output_dir, "non_lora_state_dict.bin"))

            if not self.args.save_only_model:
                self._save_optimizer_and_scheduler(output_dir)
                self._save_scaler(output_dir)
                self._save_rng_state(output_dir)

            if self.args.should_save:
                for cb in [
                    cb for cb in self.callback_handler.callbacks + [self.control] if isinstance(cb, ExportableState)
                ]:
                    cb_name = cb.__class__.__name__
                    cb_state = cb.state()
                    if isinstance(self.state.stateful_callbacks[cb_name], list):
                        self.state.stateful_callbacks[cb_name].append(cb_state)
                    else:
                        self.state.stateful_callbacks[cb_name] = cb_state
                self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))

            if self.args.push_to_hub:
                self._push_from_checkpoint(output_dir)
        else:
            super()._save_checkpoint(model, trial)

        if improved and self.args.should_save:
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
            run_dir = self._get_output_dir(trial=trial)
            self.state.best_model_checkpoint = os.path.join(run_dir, checkpoint_folder)
            print(
                f"New best checkpoint by train ce_loss: {self.state.best_model_checkpoint} "
                f"(ce_loss={self.state.best_metric:.6f})"
            )

    def _load_best_model(self):
        super()._load_best_model()

        if not getattr(self.args, "lora_enable", False):
            return

        checkpoint_dir = self.state.best_model_checkpoint
        if not checkpoint_dir:
            return

        non_lora_path = os.path.join(checkpoint_dir, "non_lora_state_dict.bin")
        if not os.path.exists(non_lora_path):
            return

        non_lora_state_dict = torch.load(non_lora_path, map_location="cpu")
        if not non_lora_state_dict:
            return

        _, unexpected_keys = self.model.load_state_dict(non_lora_state_dict, strict=False)
        if unexpected_keys:
            print(f"Loaded best non-LoRA weights with unexpected keys: {unexpected_keys}")
