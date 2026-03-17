import contextlib
import io
import os
from transformers import Trainer, PreTrainedModel
import inspect
import torch
from transformers.trainer import PREFIX_CHECKPOINT_DIR


class LogitsDistillationTrainer(Trainer):
    def __init__(
        self,
        teacher_model: PreTrainedModel = None,
        loss_function: str = "forward_kl",
        temperature: float = 2.0,
        alpha: float = 0.5,
        ofa_eps: float = 1.0,
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
        self.ofa_eps = ofa_eps
        self.latest_ce_loss = None

        print(f"Distillation Trainer initialized:")
        print(f"  - Teachers: {len(self.teacher_models)}")
        print(f"  - Loss function: {loss_function}")
        print(f"  - Temperature: {temperature}")
        print(f"  - Alpha: {alpha}")
        if "eps" in self.distillation_loss_params:
            print(f"  - OFA eps: {ofa_eps}")

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
        if "eps" in self.distillation_loss_params:
            loss_kwargs["eps"] = self.ofa_eps
        return self.distillation_loss_fn(**loss_kwargs)

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

        student_outputs = model(**student_inputs)
        student_logits = student_outputs.logits

        student_labels = student_inputs.get("labels")
        assert student_labels is not None, "Labels must be provided for distillation loss masking"

        teacher_losses = []
        with torch.no_grad():
            for teacher_model, (teacher_inputs, teacher_labels) in zip(self.teacher_models, teacher_batches):
                if getattr(teacher_model, "_suppress_forward_stdout", False):
                    with contextlib.redirect_stdout(io.StringIO()):
                        teacher_outputs = teacher_model(**teacher_inputs)
                else:
                    teacher_outputs = teacher_model(**teacher_inputs)
                teacher_logits = teacher_outputs.logits.detach()
                teacher_losses.append(
                    self._compute_single_teacher_loss(
                        student_logits=student_logits,
                        student_labels=student_labels,
                        teacher_logits=teacher_logits,
                        teacher_labels=teacher_labels,
                    )
                )

        if teacher_losses:
            distillation_loss = torch.stack(teacher_losses).mean()
        else:
            distillation_loss = torch.tensor(0.0, device=student_logits.device)

        ce_loss = student_outputs.loss
        self.latest_ce_loss = ce_loss.detach().float().item()
        loss = self.alpha * distillation_loss + (1 - self.alpha) * ce_loss

        if self.state.global_step % self.args.logging_steps == 0:
            self.log({
                "distillation_loss": distillation_loss.item(),
                "ce_loss": ce_loss.item(),
            })

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

        super()._save_checkpoint(model, trial)

        if improved and self.args.should_save:
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
            run_dir = self._get_output_dir(trial=trial)
            self.state.best_model_checkpoint = os.path.join(run_dir, checkpoint_folder)
            print(
                f"New best checkpoint by train ce_loss: {self.state.best_model_checkpoint} "
                f"(ce_loss={self.state.best_metric:.6f})"
            )
