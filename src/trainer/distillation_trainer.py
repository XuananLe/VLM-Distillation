import os

import torch
from transformers import PreTrainedModel

from src.components.forward_utils import (
    forward_with_kwarg_retry,
    infer_batch_size,
)
from src.components.teacher_gate import Gate
from src.trainer.sft_trainer import SmolVLMSFTTrainer
from src.trainer.distillation_utils import (
    build_teacher_batches,
    compute_teacher_forward,
    release_eval_memory,
)


class DistillationTrainer(SmolVLMSFTTrainer):
    def __init__(
        self,
        teacher_model: PreTrainedModel = None,
        loss_function: str = "uld_loss",
        temperature: float = 2.0,
        kd_loss_alpha: float = 1.0,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        from src.components import loss as distillation_loss_module
        if kd_loss_alpha < 0.0:
            raise ValueError("DistillationTrainer requires `kd_loss_alpha >= 0`.")
        if not hasattr(distillation_loss_module, loss_function):
            raise ValueError(f"Unknown distillation loss: {loss_function!r}")
        self.loss_function = loss_function
        self.distillation_loss_fn = getattr(distillation_loss_module, loss_function)

        if teacher_model is None:
            raise ValueError("teacher_model must be provided for distillation.")
        self.teacher_models = (
            list(teacher_model)
            if isinstance(teacher_model, (list, tuple))
            else [teacher_model]
        )
        for model in self.teacher_models:
            model.eval()
            for param in model.parameters():
                param.requires_grad = False

        self.teacher_gate = None
        if len(self.teacher_models) > 1:
            self.teacher_gate = Gate(self.model, len(self.teacher_models))
            self.model.teacher_gate = self.teacher_gate

        self.temperature = temperature
        self.kd_loss_alpha = kd_loss_alpha
        self.non_lora_require_grad_only = True
        self.eval_ce_loss_sum = 0.0
        self.eval_ce_loss_count = 0

        print(f"Distillation Trainer initialized:")
        print(f"  - Teachers: {len(self.teacher_models)}")
        print(
            "  - Teacher weighting: learned gate"
            if len(self.teacher_models) > 1
            else "  - Teacher weighting: uniform mean"
        )
        print(f"  - Loss function: {loss_function}")
        print(f"  - Temperature: {temperature}")
        print(f"  - KD alpha: {kd_loss_alpha}")
        print("  - Loss weighting: CE + alpha * KD")

    def _prepare_inputs(self, inputs):
        if not isinstance(inputs, dict):
            return super()._prepare_inputs(inputs)

        teacher_inputs = {
            key: value for key, value in inputs.items() if key.startswith("teacher")
        }
        student_inputs = {
            key: value for key, value in inputs.items() if not key.startswith("teacher")
        }

        prepared_inputs = super()._prepare_inputs(student_inputs)
        prepared_inputs.update(teacher_inputs)
        return prepared_inputs

    def _compute_single_teacher_loss(
        self,
        student_logits: torch.Tensor,
        student_labels: torch.Tensor,
        teacher_logits: torch.Tensor,
        teacher_labels: torch.Tensor,
    ) -> torch.Tensor:
        student_mask = student_labels != -100
        teacher_mask = teacher_labels != -100

        sample_losses = []
        for i in range(student_logits.size(0)):
            student_logits_masked = student_logits[i][student_mask[i]]
            teacher_logits_masked = teacher_logits[i][teacher_mask[i]]
            min_len = min(student_logits_masked.size(0), teacher_logits_masked.size(0))
            if min_len > 0:
                sample_losses.append(
                    self.distillation_loss_fn(
                        student_logits=student_logits_masked[:min_len],
                        teacher_logits=teacher_logits_masked[:min_len],
                        temperature=self.temperature,
                    )
                )
            else:
                sample_losses.append(student_logits.new_zeros(()))
        if not sample_losses:
            return student_logits.new_zeros((student_logits.size(0),))
        return torch.stack(sample_losses)

    def _compute_teacher_loss_matrix(
        self,
        *,
        student_logits: torch.Tensor,
        student_labels: torch.Tensor,
        teacher_batches,
    ) -> torch.Tensor:
        return torch.stack(
            [
                self._compute_single_teacher_loss(
                    student_logits=student_logits,
                    student_labels=student_labels,
                    teacher_logits=compute_teacher_forward(
                        teacher_model,
                        self._prepare_input(teacher_inputs),
                        output_hidden_states=False,
                        suppress_stdout=getattr(teacher_model, "_suppress_forward_stdout", False),
                    ).logits.detach(),
                    teacher_labels=self._prepare_input(teacher_labels),
                )
                for teacher_model, (teacher_inputs, teacher_labels) in zip(self.teacher_models, teacher_batches)
            ],
            dim=-1,
        )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        target_device = inputs["input_ids"].device
        for index, teacher_model in enumerate(self.teacher_models):
            teacher_device = next(teacher_model.parameters()).device
            if teacher_device != target_device:
                self.teacher_models[index] = teacher_model.to(target_device)

        student_inputs = {k: v for k, v in inputs.items() if not k.startswith("teacher")}
        teacher_batches = build_teacher_batches(
            inputs,
            student_inputs,
            len(self.teacher_models),
        )

        if self.teacher_gate is not None:
            self.teacher_gate.reset()
        student_outputs = forward_with_kwarg_retry(
            model,
            {**student_inputs, "return_dict": True, "output_hidden_states": False},
        )
        student_logits = student_outputs.logits
        teacher_gate_weights = (
            self.teacher_gate(
                labels=student_inputs["labels"],
                attention_mask=student_inputs.get("attention_mask"),
            )
            if self.teacher_gate is not None
            else None
        )
        teacher_loss_matrix = self._compute_teacher_loss_matrix(
            student_logits=student_logits,
            student_labels=student_inputs["labels"],
            teacher_batches=teacher_batches,
        )
        distillation_loss = teacher_loss_matrix.mean()
        if teacher_gate_weights is not None:
            distillation_loss = (teacher_loss_matrix * teacher_gate_weights).sum(dim=-1).mean()

        ce_loss = student_outputs.loss
        if not model.training:
            batch_size = infer_batch_size(student_inputs)
            self.eval_ce_loss_sum += ce_loss.detach().float().item() * batch_size
            self.eval_ce_loss_count += batch_size

        loss = ce_loss + distillation_loss * self.kd_loss_alpha

        if self.state.global_step % self.args.logging_steps == 0:
            metrics = {
                "loss": loss.item(),
                "distillation_loss": distillation_loss.item(),
                "ce_loss": ce_loss.item(),
            }
            if teacher_gate_weights is not None:
                metrics.update(
                    {
                        f"teacher_gate_w_{teacher_index}": weight.item()
                        for teacher_index, weight in enumerate(teacher_gate_weights.detach().mean(dim=0))
                    }
                )
            self.log(metrics)

        return (loss, student_outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        if prediction_loss_only:
            has_labels = False if len(self.label_names) == 0 else all(
                inputs.get(k) is not None for k in self.label_names
            )
            return_loss = inputs.get("return_loss")
            if return_loss is None:
                return_loss = self.can_return_loss
            loss_without_labels = len(self.label_names) == 0 and return_loss

            if has_labels or loss_without_labels:
                prepared_inputs = self._prepare_inputs(inputs)
                with torch.no_grad():
                    with self.compute_loss_context_manager():
                        num_items_in_batch = self._get_num_items_in_batch([prepared_inputs], self.args.device)
                        loss = self.compute_loss(
                            model,
                            prepared_inputs,
                            return_outputs=False,
                            num_items_in_batch=num_items_in_batch,
                        )
                    loss = loss.detach().mean()
                return (loss, None, None)

        return super().prediction_step(
            model,
            inputs,
            prediction_loss_only=prediction_loss_only,
            ignore_keys=ignore_keys,
        )

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        self.eval_ce_loss_sum = 0.0
        self.eval_ce_loss_count = 0
        try:
            return super().evaluate(
                eval_dataset=eval_dataset,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )
        finally:
            release_eval_memory()

    def log(self, logs: dict[str, float], start_time=None) -> None:
        if self.eval_ce_loss_count > 0:
            eval_prefixes = [
                key[:-5]
                for key in logs
                if key.startswith("eval") and key.endswith("_loss") and not key.endswith("_ce_loss")
            ]
            if eval_prefixes:
                stats = torch.tensor(
                    [self.eval_ce_loss_sum, float(self.eval_ce_loss_count)],
                    device=self.args.device,
                    dtype=torch.float64,
                )
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)
                logs = dict(logs)
                eval_ce_loss = (stats[0] / stats[1]).item()
                for prefix in eval_prefixes:
                    logs[f"{prefix}_ce_loss"] = eval_ce_loss
                self.eval_ce_loss_sum = 0.0
                self.eval_ce_loss_count = 0

        super().log(logs, start_time=start_time)

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
        _, unexpected_keys = self.model.load_state_dict(non_lora_state_dict, strict=False)
        if unexpected_keys:
            print(f"Loaded best non-LoRA weights with unexpected keys: {unexpected_keys}")
