import os
from typing import override

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel

from src.components.forward_utils import (
    forward_with_kwarg_retry,
    infer_batch_size,
)
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
        student_temperature: float | None = None,
        teacher_temperature: float | None = None,
        skip_student_eos: bool = False,
        skip_teacher_eos: bool = False,
        orientation_vote_margin: float = 0.02,
        alpha: float = 1.0,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        from src.components import loss as distillation_loss_module
        if alpha < 0.0:
            raise ValueError("DistillationTrainer requires `alpha >= 0`.")
        if orientation_vote_margin < 0.0:
            raise ValueError("DistillationTrainer requires `orientation_vote_margin >= 0`.")
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

        self.temperature = temperature
        self.student_temperature = (
            float(temperature) if student_temperature is None else float(student_temperature)
        )
        self.teacher_temperature = (
            float(temperature) if teacher_temperature is None else float(teacher_temperature)
        )
        self.skip_student_eos = skip_student_eos
        self.skip_teacher_eos = skip_teacher_eos
        self.orientation_vote_margin = float(orientation_vote_margin)
        self.alpha = alpha
        self.non_lora_require_grad_only = True
        self.eval_ce_loss_sum = 0.0
        self.eval_ce_loss_count = 0

        print(f"Distillation Trainer initialized:")
        print(f"  - Teachers: {len(self.teacher_models)}")
        print(
            "  - Teacher weighting: orientation vote"
            if len(self.teacher_models) > 1
            else "  - Teacher weighting: uniform mean"
        )
        print(f"  - Loss function: {loss_function}")
        print(f"  - Student temperature: {self.student_temperature}")
        print(f"  - Teacher temperature: {self.teacher_temperature}")
        print(f"  - Skip student EOS: {self.skip_student_eos}")
        print(f"  - Skip teacher EOS: {self.skip_teacher_eos}")
        print(f"  - Orientation vote margin: {self.orientation_vote_margin}")
        print(f"  - Alpha: {alpha}")
        if len(self.teacher_models) > 1:
            print("  - Routing: Orientation vote")
        print("  - Loss weighting: CE + alpha * KD")

    @override
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

    def prepare_distillation_sequences(
        self,
        *,
        student_logits: torch.Tensor,
        student_labels: torch.Tensor,
        teacher_logits: torch.Tensor,
        teacher_labels: torch.Tensor,
        ce_grad: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        student_logits_masked = student_logits[student_labels != -100]
        teacher_logits_masked = teacher_logits[teacher_labels != -100]
        ce_grad_masked = ce_grad[student_labels != -100] if ce_grad is not None else None

        if self.skip_student_eos and student_logits_masked.size(0) > 0:
            student_logits_masked = student_logits_masked[:-1]
            if ce_grad_masked is not None:
                ce_grad_masked = ce_grad_masked[:-1]
        if self.skip_teacher_eos and teacher_logits_masked.size(0) > 0:
            teacher_logits_masked = teacher_logits_masked[:-1]

        min_len = min(student_logits_masked.size(0), teacher_logits_masked.size(0))
        if ce_grad_masked is not None:
            min_len = min(min_len, ce_grad_masked.size(0))

        if min_len == 0:
            return (
                student_logits.new_zeros((0, student_logits.size(-1))),
                teacher_logits.new_zeros((0, teacher_logits.size(-1))),
                None if ce_grad_masked is None else ce_grad.new_zeros((0, ce_grad.size(-1))),
            )

        return (
            student_logits_masked[:min_len],
            teacher_logits_masked[:min_len],
            None if ce_grad_masked is None else ce_grad_masked[:min_len],
        )

    def compute_single_teacher_loss(
        self,
        student_logits: torch.Tensor,
        student_labels: torch.Tensor,
        teacher_logits: torch.Tensor,
        teacher_labels: torch.Tensor,
    ) -> torch.Tensor:
        sample_losses = []
        for i in range(student_logits.size(0)):
            student_logits_masked, teacher_logits_masked, _ = self.prepare_distillation_sequences(
                student_logits=student_logits[i],
                student_labels=student_labels[i],
                teacher_logits=teacher_logits[i],
                teacher_labels=teacher_labels[i],
            )
            if student_logits_masked.size(0) > 0:
                sample_losses.append(
                    self.distillation_loss_fn(
                        student_logits=student_logits_masked,
                        teacher_logits=teacher_logits_masked,
                        temperature=self.temperature,
                        student_temperature=self.student_temperature,
                        teacher_temperature=self.teacher_temperature,
                    )
                )
            else:
                sample_losses.append(student_logits.new_zeros(()))
        if not sample_losses:
            return student_logits.new_zeros((student_logits.size(0),))
        return torch.stack(sample_losses)

    @staticmethod
    def compute_ce_logit_grad(
        student_logits: torch.Tensor,
        student_labels: torch.Tensor,
    ) -> torch.Tensor:
        ce_grad = F.softmax(student_logits.float(), dim=-1)
        label_mask = student_labels != -100
        if not label_mask.any():
            return torch.zeros_like(student_logits, dtype=torch.float32)

        batch_idx, token_idx = label_mask.nonzero(as_tuple=True)
        label_idx = student_labels[batch_idx, token_idx]
        ce_grad[batch_idx, token_idx, label_idx] -= 1.0
        ce_grad[~label_mask] = 0.0
        return ce_grad

    def compute_orientation_vote_matrix(
        self,
        *,
        student_logits: torch.Tensor,
        student_labels: torch.Tensor,
        teacher_logits: torch.Tensor,
        teacher_labels: torch.Tensor,
        ce_grad: torch.Tensor,
    ) -> torch.Tensor:
        from src.components.loss import distillation_logit_grad

        sample_votes = []

        for i in range(student_logits.size(0)):
            student_logits_masked, teacher_logits_masked, ce_grad_masked = self.prepare_distillation_sequences(
                student_logits=student_logits[i],
                student_labels=student_labels[i],
                teacher_logits=teacher_logits[i],
                teacher_labels=teacher_labels[i],
                ce_grad=ce_grad[i],
            )
            if student_logits_masked.size(0) == 0 or ce_grad_masked is None:
                sample_votes.append(student_logits.new_zeros((), dtype=torch.float32))
                continue

            kd_grad = distillation_logit_grad(
                loss_function=self.loss_function,
                student_logits=student_logits_masked,
                teacher_logits=teacher_logits_masked,
                temperature=self.temperature,
                student_temperature=self.student_temperature,
                teacher_temperature=self.teacher_temperature,
            )
            alignment = F.cosine_similarity(
                ce_grad_masked.reshape(1, -1),
                kd_grad.reshape(1, -1),
                dim=-1,
                eps=1e-8,
            ).squeeze(0)
            vote = torch.where(
                alignment > self.orientation_vote_margin,
                torch.ones_like(alignment),
                torch.where(
                    alignment < -self.orientation_vote_margin,
                    -torch.ones_like(alignment),
                    torch.zeros_like(alignment),
                ),
            )
            sample_votes.append(vote)

        if not sample_votes:
            return student_logits.new_zeros((student_logits.size(0),), dtype=torch.float32)
        return torch.stack(sample_votes)

    @staticmethod
    def apply_orientation_vote_routing(
        teacher_loss_matrix: torch.Tensor,
        teacher_vote_matrix: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        keep_mask = teacher_vote_matrix > 0
        route_weights = keep_mask.float()
        route_weights = route_weights / route_weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
        routed_teacher_loss = (teacher_loss_matrix * route_weights).sum(dim=-1)
        kd_sample_mask = keep_mask.any(dim=-1)
        if kd_sample_mask.any():
            distillation_loss = routed_teacher_loss[kd_sample_mask].mean()
        else:
            distillation_loss = teacher_loss_matrix.new_zeros(())
        return distillation_loss, route_weights, kd_sample_mask

    def compute_teacher_loss_matrix(
        self,
        *,
        student_logits: torch.Tensor,
        student_labels: torch.Tensor,
        teacher_batches,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        teacher_losses = []
        teacher_votes = []
        ce_grad = None
        if len(self.teacher_models) > 1:
            ce_grad = self.compute_ce_logit_grad(student_logits, student_labels)

        for teacher_model, (teacher_inputs, teacher_labels) in zip(self.teacher_models, teacher_batches):
            prepared_teacher_labels = self._prepare_input(teacher_labels)
            teacher_logits = compute_teacher_forward(
                teacher_model,
                self._prepare_input(teacher_inputs),
                output_hidden_states=False,
                suppress_stdout=getattr(teacher_model, "_suppress_forward_stdout", False),
            ).logits.detach()
            teacher_losses.append(
                self.compute_single_teacher_loss(
                    student_logits=student_logits,
                    student_labels=student_labels,
                    teacher_logits=teacher_logits,
                    teacher_labels=prepared_teacher_labels,
                    )
                )
            if ce_grad is not None:
                teacher_votes.append(
                    self.compute_orientation_vote_matrix(
                        student_logits=student_logits,
                        student_labels=student_labels,
                        teacher_logits=teacher_logits,
                        teacher_labels=prepared_teacher_labels,
                        ce_grad=ce_grad,
                    )
                )
            del teacher_logits

        teacher_loss_matrix = torch.stack(teacher_losses, dim=-1)
        teacher_vote_matrix = (
            torch.stack(teacher_votes, dim=-1)
            if teacher_votes
            else None
        )
        return teacher_loss_matrix, teacher_vote_matrix, None

    @override
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

        student_outputs = forward_with_kwarg_retry(
            model,
            {**student_inputs, "return_dict": True, "output_hidden_states": False},
        )
        student_logits = student_outputs.logits
        teacher_loss_matrix, teacher_vote_matrix, _ = self.compute_teacher_loss_matrix(
            student_logits=student_logits,
            student_labels=student_inputs["labels"],
            teacher_batches=teacher_batches,
        )
        ce_loss = student_outputs.loss
        if teacher_vote_matrix is not None:
            distillation_loss, route_weights, kd_sample_mask = self.apply_orientation_vote_routing(
                teacher_loss_matrix=teacher_loss_matrix,
                teacher_vote_matrix=teacher_vote_matrix,
            )
        else:
            distillation_loss = teacher_loss_matrix.mean()
            route_weights = None
            kd_sample_mask = None

        if not model.training:
            batch_size = infer_batch_size(student_inputs)
            self.eval_ce_loss_sum += ce_loss.detach().float().item() * batch_size
            self.eval_ce_loss_count += batch_size

        loss = ce_loss + self.alpha * distillation_loss

        if self.state.global_step % self.args.logging_steps == 0:
            metrics = {
                "loss": loss.item(),
                "distillation_loss": distillation_loss.item(),
                "ce_loss": ce_loss.item(),
            }
            if route_weights is not None:
                for teacher_idx in range(route_weights.size(-1)):
                    metrics[f"teacher_vote_w_{teacher_idx}"] = route_weights[:, teacher_idx].mean().item()
                    metrics[f"teacher_vote_{teacher_idx}"] = teacher_vote_matrix[:, teacher_idx].float().mean().item()
                    metrics[f"teacher_vote_pos_rate_{teacher_idx}"] = (
                        (teacher_vote_matrix[:, teacher_idx] > 0).float().mean().item()
                    )
                    metrics[f"teacher_vote_zero_rate_{teacher_idx}"] = (
                        (teacher_vote_matrix[:, teacher_idx] == 0).float().mean().item()
                    )
                    metrics[f"teacher_vote_neg_rate_{teacher_idx}"] = (
                        (teacher_vote_matrix[:, teacher_idx] < 0).float().mean().item()
                    )
                metrics["teacher_vote_kd_sample_rate"] = kd_sample_mask.float().mean().item()
            self.log(metrics)

        return (loss, student_outputs) if return_outputs else loss

    @override
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

    @override
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

    @override
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

    @override
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
