import contextlib
import io
import os
from transformers import PreTrainedModel
import inspect
import torch
from transformers.trainer import PREFIX_CHECKPOINT_DIR, TRAINER_STATE_NAME, ExportableState, SaveStrategy
from src.components.forward_utils import (
    forward_with_kwarg_retry,
    infer_batch_size,
    unwrap_tensor,
)
from src.components.skc import linear_cka_loss
from src.components.vision_forward import infer_vision_group_counts, pool_vision_features
from src.trainer.sft_trainer import SmolVLMSFTTrainer
from src.train.train_utils import get_peft_state_non_lora_maybe_zero_3
from src.utils import find_vision_layer_indices, get_specific_layer


class DistillationTrainer(SmolVLMSFTTrainer):
    def __init__(
        self,
        teacher_model: PreTrainedModel = None,
        loss_function: str = "forward_kl",
        temperature: float = 2.0,
        alpha: float = 0.5,
        layer_distill_source: str = "none",
        layer_distill_weight: float = 0.0,
        student_layer_indices: list[int] | None = None,
        teacher_layer_indices: list[int] | None = None,
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
        self.latest_ce_loss = None
        self.layer_distill_source = layer_distill_source
        self.layer_distill_weight = layer_distill_weight
        self.student_layer_indices = list(student_layer_indices or [])
        self.teacher_layer_indices = list(teacher_layer_indices) if teacher_layer_indices is not None else None
        self.layer_distillation_enabled = (
            self.layer_distill_source == "vision"
            and self.layer_distill_weight > 0.0
            and bool(self.student_layer_indices)
        )
        self.teacher_layer_pairs = []

        print(f"Distillation Trainer initialized:")
        print(f"  - Teachers: {len(self.teacher_models)}")
        print(f"  - Loss function: {loss_function}")
        print(f"  - Temperature: {temperature}")
        print(f"  - Alpha: {alpha}")
        if self.layer_distillation_enabled:
            self._initialize_vision_layer_distillation()
            print("  - Layer distillation: enabled")
            print(f"  - Layer distill source: {self.layer_distill_source}")
            print(f"  - Layer distill weight: {self.layer_distill_weight}")
            print(f"  - Student vision layers: {self.student_layer_indices}")
            for teacher_index, layer_pairs in enumerate(self.teacher_layer_pairs):
                print(f"  - Teacher {teacher_index} layer pairs: {layer_pairs}")
        else:
            print("  - Layer distillation: disabled")

    @staticmethod
    def _unwrap_layer_model(model):
        if hasattr(model, "get_base_model"):
            return model.get_base_model()
        return model

    @staticmethod
    def _resolve_layer_indices(total_layers: int, layer_indices: list[int], label: str) -> list[int]:
        resolved = []
        for layer_index in layer_indices:
            normalized = total_layers + layer_index if layer_index < 0 else layer_index
            if normalized < 0 or normalized >= total_layers:
                raise IndexError(
                    f"{label} layer index {layer_index} resolved to {normalized}, "
                    f"but valid range is 0-{total_layers - 1}."
                )
            resolved.append(int(normalized))
        return resolved

    @staticmethod
    def _auto_map_teacher_layers(
        student_layer_indices: list[int],
        student_total_layers: int,
        teacher_total_layers: int,
    ) -> list[int]:
        if teacher_total_layers < 1:
            raise ValueError("Teacher model does not expose any vision layers.")
        if teacher_total_layers == 1:
            return [0 for _ in student_layer_indices]
        denominator = max(student_total_layers - 1, 1)
        return [
            int(round((student_layer_index / denominator) * (teacher_total_layers - 1)))
            for student_layer_index in student_layer_indices
        ]

    def _initialize_vision_layer_distillation(self) -> None:
        if self.layer_distill_source != "vision":
            raise ValueError(
                f"Unsupported layer_distill_source={self.layer_distill_source!r}. "
                "Only 'vision' is implemented."
            )

        student_vision_info = find_vision_layer_indices(self._unwrap_layer_model(self.model))
        self.student_layer_indices = self._resolve_layer_indices(
            student_vision_info["total_layers"],
            self.student_layer_indices,
            "Student vision",
        )
        if self.teacher_layer_indices is not None and len(self.teacher_layer_indices) != len(self.student_layer_indices):
            raise ValueError(
                "--teacher_layer_indices must have the same number of entries as --student_layer_indices."
            )

        self.teacher_layer_pairs = []
        for teacher_index, teacher_model in enumerate(self.teacher_models):
            teacher_vision_info = find_vision_layer_indices(self._unwrap_layer_model(teacher_model))
            if self.teacher_layer_indices is None:
                teacher_indices = self._auto_map_teacher_layers(
                    self.student_layer_indices,
                    student_vision_info["total_layers"],
                    teacher_vision_info["total_layers"],
                )
            else:
                teacher_indices = self._resolve_layer_indices(
                    teacher_vision_info["total_layers"],
                    self.teacher_layer_indices,
                    f"Teacher {teacher_index} vision",
                )
            self.teacher_layer_pairs.append(list(zip(self.student_layer_indices, teacher_indices)))

    @staticmethod
    def _register_layer_hooks(model, layer_indices: list[int]):
        raw_outputs = {}
        handles = []
        layer_model = DistillationTrainer._unwrap_layer_model(model)

        for layer_index in layer_indices:
            layer, _ = get_specific_layer(layer_model, layer_index)

            def make_hook(index: int):
                def hook(module, hook_inputs, output):
                    del module, hook_inputs
                    tensor = unwrap_tensor(output)
                    if tensor is not None:
                        raw_outputs[index] = tensor
                return hook

            handles.append(layer.register_forward_hook(make_hook(layer_index)))

        return raw_outputs, handles

    @staticmethod
    def _remove_hook_handles(handles) -> None:
        for handle in handles:
            handle.remove()

    @staticmethod
    def _pool_vision_representations(
        raw_outputs: dict[int, torch.Tensor],
        layer_indices: list[int],
        batch_size: int,
        model_inputs,
    ) -> dict[int, torch.Tensor]:
        group_counts = infer_vision_group_counts(model_inputs, batch_size)
        pooled_outputs = {}
        for layer_index in layer_indices:
            if layer_index not in raw_outputs:
                raise RuntimeError(
                    f"Vision layer {layer_index} did not produce hook features during the forward pass."
                )
            pooled_outputs[layer_index] = pool_vision_features(
                raw_outputs[layer_index],
                batch_size,
                group_counts=group_counts,
            )
        return pooled_outputs

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
    def _forward_model(model, model_inputs):
        call_inputs = dict(model_inputs)
        call_inputs["return_dict"] = True
        return forward_with_kwarg_retry(model, call_inputs)

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

        student_layer_outputs = None
        student_hook_handles = []
        if self.layer_distillation_enabled:
            student_layer_outputs, student_hook_handles = self._register_layer_hooks(
                model,
                self.student_layer_indices,
            )
        try:
            student_outputs = self._forward_model(model, student_inputs)
        finally:
            self._remove_hook_handles(student_hook_handles)
        student_logits = student_outputs.logits

        student_labels = student_inputs.get("labels")
        assert student_labels is not None, "Labels must be provided for distillation loss masking"
        layer_distillation_losses = []
        if self.layer_distillation_enabled:
            student_batch_size = infer_batch_size(student_inputs)
            student_layer_representations = self._pool_vision_representations(
                student_layer_outputs,
                self.student_layer_indices,
                student_batch_size,
                student_inputs,
            )

        teacher_losses = []
        for teacher_index, (teacher_model, (teacher_inputs, teacher_labels)) in enumerate(
            zip(self.teacher_models, teacher_batches)
        ):
            teacher_layer_outputs = None
            teacher_hook_handles = []
            teacher_layer_indices = []
            if self.layer_distillation_enabled:
                teacher_layer_indices = [teacher_layer for _, teacher_layer in self.teacher_layer_pairs[teacher_index]]
                teacher_layer_outputs, teacher_hook_handles = self._register_layer_hooks(
                    teacher_model,
                    teacher_layer_indices,
                )
            try:
                with torch.no_grad():
                    if getattr(teacher_model, "_suppress_forward_stdout", False):
                        with contextlib.redirect_stdout(io.StringIO()):
                            teacher_outputs = self._forward_model(teacher_model, teacher_inputs)
                    else:
                        teacher_outputs = self._forward_model(teacher_model, teacher_inputs)
            finally:
                self._remove_hook_handles(teacher_hook_handles)
            teacher_logits = teacher_outputs.logits.detach()
            teacher_losses.append(
                self._compute_single_teacher_loss(
                    student_logits=student_logits,
                    student_labels=student_labels,
                    teacher_logits=teacher_logits,
                    teacher_labels=teacher_labels,
                )
            )
            if self.layer_distillation_enabled:
                teacher_batch_size = infer_batch_size(teacher_inputs)
                teacher_layer_representations = self._pool_vision_representations(
                    teacher_layer_outputs,
                    teacher_layer_indices,
                    teacher_batch_size,
                    teacher_inputs,
                )
                pair_losses = []
                for student_layer_index, teacher_layer_index in self.teacher_layer_pairs[teacher_index]:
                    pair_losses.append(
                        linear_cka_loss(
                            student_layer_representations[student_layer_index],
                            teacher_layer_representations[teacher_layer_index],
                        )
                    )
                if pair_losses:
                    layer_distillation_losses.append(torch.stack(pair_losses).mean())

        if teacher_losses:
            distillation_loss = torch.stack(teacher_losses).mean()
        else:
            distillation_loss = torch.tensor(0.0, device=student_logits.device)
        if layer_distillation_losses:
            layer_distillation_loss = torch.stack(layer_distillation_losses).mean()
        else:
            layer_distillation_loss = torch.tensor(0.0, device=student_logits.device)

        ce_loss = student_outputs.loss
        self.latest_ce_loss = ce_loss.detach().float().item()
        loss = (
            self.alpha * distillation_loss
            + (1 - self.alpha) * ce_loss
            + self.layer_distill_weight * layer_distillation_loss
        )

        if self.state.global_step % self.args.logging_steps == 0:
            metrics = {
                "distillation_loss": distillation_loss.item(),
                "ce_loss": ce_loss.item(),
            }
            if self.layer_distillation_enabled:
                metrics["vision_layer_distill_loss"] = layer_distillation_loss.item()
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
