import math
import os

import torch
import torch.nn.functional as F
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
        alpha: float = 1.0,
        gradient_alignment_threshold: float = 0.0,
        gradient_alignment_warmup_ratio: float = 0.0,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        from src.components import loss as distillation_loss_module
        if alpha < 0.0:
            raise ValueError("DistillationTrainer requires `alpha >= 0`.")
        if gradient_alignment_warmup_ratio < 0.0:
            raise ValueError("DistillationTrainer requires `gradient_alignment_warmup_ratio >= 0`.")
        if not hasattr(distillation_loss_module, loss_function):
            raise ValueError(f"Unknown distillation loss: {loss_function!r}")
        self.loss_function = loss_function
        self.distillation_loss_fn = getattr(distillation_loss_module, loss_function)
        self.distillation_logit_grad_fn = distillation_loss_module.distillation_logit_grad

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
        self.alpha = alpha
        self.gradient_alignment_threshold = gradient_alignment_threshold
        self.gradient_alignment_warmup_ratio = gradient_alignment_warmup_ratio
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
        print(f"  - Alpha: {alpha}")
        if self.teacher_gate is not None:
            print("  - Routing: learned gate + gradient alignment filter")
            print(f"  - Gradient alignment threshold: {gradient_alignment_threshold}")
            print(f"  - Gradient alignment warmup ratio: {gradient_alignment_warmup_ratio}")
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

    def should_apply_gradient_alignment_routing(self) -> bool:
        if (
            self.teacher_gate is None
            or not self.model.training
        ):
            return False

        if self.gradient_alignment_warmup_ratio <= 0.0:
            return True

        total_steps = max(self.state.max_steps, getattr(self.args, "max_steps", 0))
        if total_steps <= 0:
            return True

        warmup_steps = math.ceil(total_steps * self.gradient_alignment_warmup_ratio)
        return self.state.global_step >= warmup_steps

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
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        collect_alignment_tensors = self.should_apply_gradient_alignment_routing()
        teacher_losses = []
        teacher_logits_list = []
        teacher_labels_list = []
        for teacher_model, (teacher_inputs, teacher_labels) in zip(self.teacher_models, teacher_batches):
            prepared_teacher_labels = self._prepare_input(teacher_labels)
            teacher_logits = compute_teacher_forward(
                teacher_model,
                self._prepare_input(teacher_inputs),
                output_hidden_states=False,
                suppress_stdout=getattr(teacher_model, "_suppress_forward_stdout", False),
            ).logits.detach()
            teacher_losses.append(
                self._compute_single_teacher_loss(
                    student_logits=student_logits,
                    student_labels=student_labels,
                    teacher_logits=teacher_logits,
                    teacher_labels=prepared_teacher_labels,
                )
            )
            if collect_alignment_tensors:
                teacher_logits_list.append(teacher_logits.to(device="cpu", copy=True))
                teacher_labels_list.append(prepared_teacher_labels.to(device="cpu", copy=True))
            del teacher_logits
        return torch.stack(teacher_losses, dim=-1), teacher_logits_list, teacher_labels_list

    def _compute_ce_alignment_grad(
        self,
        student_logits: torch.Tensor,
        student_labels: torch.Tensor,
    ) -> torch.Tensor:
        probs = torch.softmax(student_logits.float(), dim=-1)
        flat_probs = probs.view(-1, probs.size(-1))
        flat_labels = student_labels.view(-1)
        valid_mask = flat_labels.ne(-100)
        if valid_mask.any():
            flat_probs[valid_mask, flat_labels[valid_mask]] -= 1.0
        grad = flat_probs.view_as(probs)
        return grad * student_labels.ne(-100).unsqueeze(-1)

    def _compute_kd_alignment_grad(
        self,
        student_logits: torch.Tensor,
        student_labels: torch.Tensor,
        teacher_logits: torch.Tensor,
        teacher_labels: torch.Tensor,
    ) -> torch.Tensor:
        kd_grad = torch.zeros_like(student_logits, dtype=torch.float32)
        for sample_index in range(student_logits.size(0)):
            student_positions = student_labels[sample_index].ne(-100).nonzero(as_tuple=False).squeeze(-1)
            teacher_positions = teacher_labels[sample_index].ne(-100).nonzero(as_tuple=False).squeeze(-1)
            matched_tokens = min(student_positions.numel(), teacher_positions.numel())
            if matched_tokens == 0:
                continue

            student_positions = student_positions[:matched_tokens]
            teacher_positions = teacher_positions[:matched_tokens]
            kd_grad[sample_index, student_positions] = self.distillation_logit_grad_fn(
                self.loss_function,
                student_logits[sample_index, student_positions],
                teacher_logits[sample_index, teacher_positions].to(
                    device=student_logits.device,
                    dtype=student_logits.dtype,
                ),
                temperature=self.temperature,
            )
        return kd_grad

    def apply_gradient_alignment_routing(
        self,
        *,
        teacher_gate_weights: torch.Tensor | None,
        student_logits: torch.Tensor,
        teacher_logits_list: list[torch.Tensor],
        teacher_labels_list: list[torch.Tensor],
        student_inputs,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        if (
            teacher_gate_weights is None
            or not self.should_apply_gradient_alignment_routing()
            or not teacher_logits_list
        ):
            return teacher_gate_weights, None, None

        pooled_ce_grad = self.teacher_gate.pool_tensor(
            self._compute_ce_alignment_grad(student_logits, student_inputs["labels"]),
            labels=student_inputs["labels"],
            attention_mask=student_inputs.get("attention_mask"),
        )

        alignment_scores = []
        alignment_active = []
        for teacher_logits, teacher_labels in zip(teacher_logits_list, teacher_labels_list):
            pooled_kd_grad = self.teacher_gate.pool_tensor(
                self._compute_kd_alignment_grad(
                    student_logits=student_logits,
                    student_labels=student_inputs["labels"],
                    teacher_logits=teacher_logits.to(device=student_logits.device),
                    teacher_labels=teacher_labels.to(device=student_logits.device),
                ),
                labels=student_inputs["labels"],
                attention_mask=student_inputs.get("attention_mask"),
            )
            agreement = F.cosine_similarity(
                pooled_ce_grad,
                pooled_kd_grad,
                dim=-1,
                eps=1e-8,
            )
            alignment_scores.append(agreement)
            alignment_active.append(agreement > self.gradient_alignment_threshold)

        alignment_scores = torch.stack(alignment_scores, dim=-1)
        alignment_active = torch.stack(alignment_active, dim=-1)

        missing_samples = ~alignment_active.any(dim=-1)
        if missing_samples.any():
            fallback_indices = alignment_scores.argmax(dim=-1, keepdim=True)
            fallback_mask = torch.zeros_like(alignment_active)
            fallback_mask.scatter_(1, fallback_indices, True)
            alignment_active = alignment_active | (fallback_mask & missing_samples.unsqueeze(-1))

        effective_weights = teacher_gate_weights * alignment_active.to(dtype=teacher_gate_weights.dtype)
        effective_weights = effective_weights / effective_weights.sum(dim=-1, keepdim=True).clamp(
            min=torch.finfo(effective_weights.dtype).eps
        )
        return effective_weights, alignment_scores, alignment_active

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
        teacher_loss_matrix, teacher_logits_list, teacher_labels_list = self._compute_teacher_loss_matrix(
            student_logits=student_logits,
            student_labels=student_inputs["labels"],
            teacher_batches=teacher_batches,
        )
        ce_loss = student_outputs.loss
        effective_teacher_gate_weights, teacher_alignment_scores, teacher_alignment_active = (
            self.apply_gradient_alignment_routing(
                teacher_gate_weights=teacher_gate_weights,
                student_logits=student_logits,
                teacher_logits_list=teacher_logits_list,
                teacher_labels_list=teacher_labels_list,
                student_inputs=student_inputs,
            )
        )
        distillation_loss = teacher_loss_matrix.mean()
        if effective_teacher_gate_weights is not None:
            distillation_loss = (teacher_loss_matrix * effective_teacher_gate_weights).sum(dim=-1).mean()
        elif teacher_gate_weights is not None:
            distillation_loss = (teacher_loss_matrix * teacher_gate_weights).sum(dim=-1).mean()

        if not model.training:
            batch_size = infer_batch_size(student_inputs)
            self.eval_ce_loss_sum += ce_loss.detach().float().item() * batch_size
            self.eval_ce_loss_count += batch_size

        loss = ce_loss + distillation_loss * self.alpha

        if self.state.global_step % self.args.logging_steps == 0:
            metrics = {
                "loss": loss.item(),
                "distillation_loss": distillation_loss.item(),
                "ce_loss": ce_loss.item(),
            }
            if teacher_gate_weights is not None:
                logged_weights = (
                    effective_teacher_gate_weights
                    if effective_teacher_gate_weights is not None
                    else teacher_gate_weights
                )
                metrics.update(
                    {
                        f"teacher_gate_w_{teacher_index}": weight.item()
                        for teacher_index, weight in enumerate(logged_weights.detach().mean(dim=0))
                    }
                )
                if teacher_alignment_scores is not None and teacher_alignment_active is not None:
                    metrics.update(
                        {
                            f"teacher_alignment_score_{teacher_index}": score.item()
                            for teacher_index, score in enumerate(teacher_alignment_scores.detach().mean(dim=0))
                        }
                    )
                    metrics.update(
                        {
                            f"teacher_alignment_active_{teacher_index}": active.item()
                            for teacher_index, active in enumerate(
                                teacher_alignment_active.detach().to(dtype=logged_weights.dtype).mean(dim=0)
                            )
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
