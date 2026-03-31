import math
import os
from typing import override

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
    select_labels_at_positions,
    select_supervised_logit_positions,
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
        alpha: float = 1.0,
        teacher_gate_balance_alpha: float = 1e-2,
        teacher_gate_top_k: int = 1,
        teacher_gate_capacity_factor: float = 1.25,
        teacher_gate_bias_update_rate: float = 1e-3,
        gradient_alignment_threshold: float = 0.0,
        gradient_alignment_warmup_ratio: float = 0.0,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        from src.components import loss as distillation_loss_module

        if alpha < 0.0:
            raise ValueError("DistillationTrainer requires `alpha >= 0`.")
        if teacher_gate_balance_alpha < 0.0:
            raise ValueError("DistillationTrainer requires `teacher_gate_balance_alpha >= 0`.")
        if teacher_gate_top_k < 1:
            raise ValueError("DistillationTrainer requires `teacher_gate_top_k >= 1`.")
        if teacher_gate_capacity_factor <= 0.0:
            raise ValueError("DistillationTrainer requires `teacher_gate_capacity_factor > 0`.")
        if teacher_gate_bias_update_rate < 0.0:
            raise ValueError("DistillationTrainer requires `teacher_gate_bias_update_rate >= 0`.")
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
            self.teacher_gate = Gate(
                self.model,
                len(self.teacher_models),
                bias_update_rate=teacher_gate_bias_update_rate,
            )
            self.model.teacher_gate = self.teacher_gate

        self.temperature = temperature
        self.student_temperature = (
            float(temperature) if student_temperature is None else float(student_temperature)
        )
        self.teacher_temperature = (
            float(temperature) if teacher_temperature is None else float(teacher_temperature)
        )
        self.skip_student_eos = skip_student_eos
        self.skip_teacher_eos = skip_teacher_eos
        self.alpha = alpha
        self.teacher_gate_balance_alpha = teacher_gate_balance_alpha
        self.teacher_gate_top_k = teacher_gate_top_k
        self.teacher_gate_capacity_factor = teacher_gate_capacity_factor
        self.teacher_gate_bias_update_rate = teacher_gate_bias_update_rate
        self.gradient_alignment_threshold = gradient_alignment_threshold
        self.gradient_alignment_warmup_ratio = gradient_alignment_warmup_ratio
        self.non_lora_require_grad_only = True
        self.eval_ce_loss_sum = 0.0
        self.eval_ce_loss_count = 0

        print("Distillation Trainer initialized:")
        print(f"  - Teachers: {len(self.teacher_models)}")
        print(
            "  - Teacher weighting: learned deep gate + balancing + gradient alignment"
            if len(self.teacher_models) > 1
            else "  - Teacher weighting: uniform mean"
        )
        print(f"  - Loss function: {loss_function}")
        print(f"  - Student temperature: {self.student_temperature}")
        print(f"  - Teacher temperature: {self.teacher_temperature}")
        print(f"  - Skip student EOS: {self.skip_student_eos}")
        print(f"  - Skip teacher EOS: {self.skip_teacher_eos}")
        print(f"  - Alpha: {alpha}")
        if self.teacher_gate is not None:
            print(f"  - Teacher gate balance alpha: {teacher_gate_balance_alpha}")
            print(f"  - Teacher gate top-k: {teacher_gate_top_k}")
            print(f"  - Teacher gate capacity factor: {teacher_gate_capacity_factor}")
            print(f"  - Teacher gate bias update rate: {teacher_gate_bias_update_rate}")
            print(f"  - Gradient alignment threshold: {gradient_alignment_threshold}")
            print(f"  - Gradient alignment warmup ratio: {gradient_alignment_warmup_ratio}")
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

    def should_apply_gradient_alignment_routing(self) -> bool:
        if self.teacher_gate is None or not self.model.training:
            return False

        if self.gradient_alignment_warmup_ratio <= 0.0:
            return True

        total_steps = max(self.state.max_steps, getattr(self.args, "max_steps", 0))
        if total_steps <= 0:
            return True

        warmup_steps = math.ceil(total_steps * self.gradient_alignment_warmup_ratio)
        return self.state.global_step >= warmup_steps

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
    def get_supervised_positions(
        labels: torch.Tensor,
        *,
        skip_last: bool = False,
    ) -> torch.Tensor:
        positions = labels.ne(-100).nonzero(as_tuple=False).squeeze(-1)
        if skip_last and positions.numel() > 0:
            positions = positions[:-1]
        return positions

    def compute_pooled_ce_alignment_grad(
        self,
        student_logits: torch.Tensor,
        student_labels: torch.Tensor,
    ) -> torch.Tensor:
        pooled_grads = []
        vocab_size = student_logits.size(-1)
        zero_grad = student_logits.new_zeros((vocab_size,), dtype=torch.float32)

        for sample_index in range(student_logits.size(0)):
            positions = self.get_supervised_positions(student_labels[sample_index])
            if positions.numel() == 0:
                pooled_grads.append(zero_grad)
                continue

            sample_labels = student_labels[sample_index, positions]
            sample_grad = F.softmax(student_logits[sample_index, positions].float(), dim=-1)
            sample_grad[torch.arange(sample_labels.numel(), device=sample_labels.device), sample_labels] -= 1.0
            pooled_grads.append(sample_grad.sum(dim=0) / positions.numel())

        if not pooled_grads:
            return student_logits.new_zeros((0, vocab_size), dtype=torch.float32)
        return torch.stack(pooled_grads, dim=0)

    def compute_pooled_kd_alignment_grad(
        self,
        *,
        student_logits: torch.Tensor,
        student_labels: torch.Tensor,
        teacher_logits: torch.Tensor,
        teacher_labels: torch.Tensor,
    ) -> torch.Tensor:
        pooled_grads = []
        vocab_size = student_logits.size(-1)
        zero_grad = student_logits.new_zeros((vocab_size,), dtype=torch.float32)

        for sample_index in range(student_logits.size(0)):
            supervised_student_count = int(student_labels[sample_index].ne(-100).sum().item())
            if supervised_student_count == 0:
                pooled_grads.append(zero_grad)
                continue

            student_positions = self.get_supervised_positions(
                student_labels[sample_index],
                skip_last=self.skip_student_eos,
            )
            teacher_positions = self.get_supervised_positions(
                teacher_labels[sample_index],
                skip_last=self.skip_teacher_eos,
            )

            matched_tokens = min(student_positions.numel(), teacher_positions.numel())
            if matched_tokens == 0:
                pooled_grads.append(zero_grad)
                continue

            student_positions = student_positions[:matched_tokens]
            teacher_positions = teacher_positions[:matched_tokens]
            sample_kd_grad = self.distillation_logit_grad_fn(
                self.loss_function,
                student_logits[sample_index, student_positions],
                teacher_logits[sample_index, teacher_positions].to(
                    device=student_logits.device,
                    dtype=student_logits.dtype,
                ),
                temperature=self.temperature,
                student_temperature=self.student_temperature,
                teacher_temperature=self.teacher_temperature,
            )
            pooled_grads.append(sample_kd_grad.sum(dim=0) / supervised_student_count)

        if not pooled_grads:
            return student_logits.new_zeros((0, vocab_size), dtype=torch.float32)
        return torch.stack(pooled_grads, dim=0)

    def compute_teacher_gate_balance_loss(
        self,
        teacher_gate_weights: torch.Tensor,
        routing_scores: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, num_teachers = teacher_gate_weights.shape
        top_k = min(self.teacher_gate_top_k, num_teachers)
        soft_load = teacher_gate_weights.mean(dim=0)
        topk_teachers = (teacher_gate_weights if routing_scores is None else routing_scores).topk(
            top_k,
            dim=-1,
        ).indices
        hard_load = torch.nn.functional.one_hot(
            topk_teachers,
            num_classes=num_teachers,
        ).to(dtype=teacher_gate_weights.dtype).sum(dim=1).sum(dim=0) / (batch_size * top_k)
        balance_loss = num_teachers * (soft_load * hard_load).sum()
        return balance_loss, soft_load, hard_load

    def apply_teacher_gate_constraints(
        self,
        routing_scores: torch.Tensor,
        teacher_gate_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, int, torch.Tensor, torch.Tensor]:
        batch_size, num_teachers = teacher_gate_weights.shape
        top_k = min(self.teacher_gate_top_k, num_teachers)
        capacity = max(
            1,
            math.ceil(self.teacher_gate_capacity_factor * batch_size * top_k / num_teachers),
        )

        topk_scores, topk_indices = routing_scores.topk(top_k, dim=-1)
        routed_weights = torch.zeros_like(teacher_gate_weights)
        assignment_mask = torch.zeros_like(teacher_gate_weights, dtype=torch.bool)

        for teacher_index in range(num_teachers):
            candidate_mask = topk_indices == teacher_index
            if not candidate_mask.any():
                continue
            sample_indices, topk_slots = candidate_mask.nonzero(as_tuple=True)
            candidate_scores = topk_scores[sample_indices, topk_slots]
            if candidate_scores.numel() > capacity:
                keep_indices = candidate_scores.topk(capacity, largest=True, sorted=False).indices
                sample_indices = sample_indices[keep_indices]
            routed_weights[sample_indices, teacher_index] = teacher_gate_weights[sample_indices, teacher_index]
            assignment_mask[sample_indices, teacher_index] = True

        missing_indices = (routed_weights.sum(dim=-1) == 0).nonzero(as_tuple=True)[0]
        if missing_indices.numel() > 0:
            fallback_indices = routing_scores.argmax(dim=-1)
            routed_weights[missing_indices, fallback_indices[missing_indices]] = teacher_gate_weights[
                missing_indices,
                fallback_indices[missing_indices],
            ]
            assignment_mask[missing_indices, fallback_indices[missing_indices]] = True

        routed_weights = routed_weights / routed_weights.sum(dim=-1, keepdim=True).clamp(
            min=torch.finfo(routed_weights.dtype).eps
        )
        expert_load = assignment_mask.float().sum(dim=0)
        assignment_rate = expert_load / max(batch_size, 1)
        return routed_weights, capacity, assignment_rate, expert_load

    def compute_teacher_loss_matrix(
        self,
        *,
        student_logits: torch.Tensor,
        student_labels: torch.Tensor,
        student_attention_mask: torch.Tensor | None,
        teacher_batches,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        collect_alignment_tensors = self.should_apply_gradient_alignment_routing()
        teacher_losses = []
        alignment_scores = []
        alignment_active = []

        pooled_ce_grad = None
        if collect_alignment_tensors:
            with torch.no_grad():
                pooled_ce_grad = self.compute_pooled_ce_alignment_grad(
                    student_logits.detach(),
                    student_labels,
                )

        for teacher_model, (teacher_inputs, teacher_labels) in zip(self.teacher_models, teacher_batches):
            prepared_teacher_labels = self._prepare_input(teacher_labels)
            teacher_logit_positions = select_supervised_logit_positions(prepared_teacher_labels)
            if teacher_logit_positions is None and not prepared_teacher_labels.ne(-100).any():
                zero_losses = student_logits.new_zeros((student_logits.size(0),))
                teacher_losses.append(zero_losses)
                if pooled_ce_grad is not None:
                    zero_alignment = pooled_ce_grad.new_zeros((student_logits.size(0),))
                    alignment_scores.append(zero_alignment)
                    alignment_active.append(zero_alignment > self.gradient_alignment_threshold)
                continue

            teacher_outputs = compute_teacher_forward(
                teacher_model,
                self._prepare_input(teacher_inputs),
                output_hidden_states=False,
                suppress_stdout=getattr(teacher_model, "_suppress_forward_stdout", False),
                logits_to_keep=teacher_logit_positions,
            )
            teacher_logits = teacher_outputs.logits.detach()
            del teacher_outputs

            selected_teacher_labels = select_labels_at_positions(
                prepared_teacher_labels,
                teacher_logit_positions,
            )
            effective_teacher_labels = (
                selected_teacher_labels
                if selected_teacher_labels.size(1) == teacher_logits.size(1)
                else prepared_teacher_labels
            )
            teacher_losses.append(
                self.compute_single_teacher_loss(
                    student_logits=student_logits,
                    student_labels=student_labels,
                    teacher_logits=teacher_logits,
                    teacher_labels=effective_teacher_labels,
                )
            )
            if pooled_ce_grad is not None:
                with torch.no_grad():
                    pooled_kd_grad = self.compute_pooled_kd_alignment_grad(
                        student_logits=student_logits.detach(),
                        student_labels=student_labels,
                        teacher_logits=teacher_logits,
                        teacher_labels=effective_teacher_labels,
                    )
                    agreement = F.cosine_similarity(pooled_ce_grad, pooled_kd_grad, dim=-1, eps=1e-8)
                alignment_scores.append(agreement)
                alignment_active.append(agreement > self.gradient_alignment_threshold)
                del pooled_kd_grad
            del teacher_logits

        teacher_loss_matrix = torch.stack(teacher_losses, dim=-1)
        if not alignment_scores:
            return teacher_loss_matrix, None, None
        return (
            teacher_loss_matrix,
            torch.stack(alignment_scores, dim=-1),
            torch.stack(alignment_active, dim=-1),
        )

    def apply_gradient_alignment_routing(
        self,
        *,
        teacher_gate_weights: torch.Tensor | None,
        teacher_alignment_scores: torch.Tensor | None,
        teacher_alignment_active: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        if (
            teacher_gate_weights is None
            or teacher_alignment_scores is None
            or teacher_alignment_active is None
        ):
            return teacher_gate_weights, teacher_alignment_scores, teacher_alignment_active

        available_mask = teacher_gate_weights > 0
        teacher_alignment_active = teacher_alignment_active & available_mask
        missing_samples = ~teacher_alignment_active.any(dim=-1)
        if missing_samples.any():
            masked_scores = teacher_alignment_scores.masked_fill(~available_mask, float("-inf"))
            fallback_indices = masked_scores.argmax(dim=-1, keepdim=True)
            fallback_mask = torch.zeros_like(teacher_alignment_active)
            fallback_mask.scatter_(1, fallback_indices, True)
            teacher_alignment_active = teacher_alignment_active | (fallback_mask & missing_samples.unsqueeze(-1))

        effective_weights = teacher_gate_weights * teacher_alignment_active.to(dtype=teacher_gate_weights.dtype)
        effective_weights = effective_weights / effective_weights.sum(dim=-1, keepdim=True).clamp(
            min=torch.finfo(effective_weights.dtype).eps
        )
        return effective_weights, teacher_alignment_scores, teacher_alignment_active

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

        if self.teacher_gate is not None:
            self.teacher_gate.reset()
        student_outputs = forward_with_kwarg_retry(
            model,
            {**student_inputs, "return_dict": True, "output_hidden_states": False},
        )
        student_logits = student_outputs.logits
        teacher_gate_logits = (
            self.teacher_gate.compute_router_logits(
                labels=student_inputs["labels"],
                attention_mask=student_inputs.get("attention_mask"),
            )
            if self.teacher_gate is not None
            else None
        )
        teacher_gate_weights = (
            torch.softmax(teacher_gate_logits, dim=-1)
            if teacher_gate_logits is not None
            else None
        )
        teacher_gate_routing_scores = (
            self.teacher_gate.apply_expert_bias(teacher_gate_logits)
            if teacher_gate_logits is not None
            else None
        )
        teacher_gate_balance_loss = None
        teacher_gate_soft_load = None
        teacher_gate_hard_load = None
        teacher_gate_capacity = None
        teacher_gate_assignment_rate = None
        teacher_gate_expert_load = None
        routed_teacher_gate_weights = teacher_gate_weights
        if teacher_gate_weights is not None:
            (
                teacher_gate_balance_loss,
                teacher_gate_soft_load,
                teacher_gate_hard_load,
            ) = self.compute_teacher_gate_balance_loss(
                teacher_gate_weights,
                routing_scores=teacher_gate_routing_scores,
            )
            (
                routed_teacher_gate_weights,
                teacher_gate_capacity,
                teacher_gate_assignment_rate,
                teacher_gate_expert_load,
            ) = self.apply_teacher_gate_constraints(
                teacher_gate_routing_scores,
                teacher_gate_weights,
            )
            if model.training:
                self.teacher_gate.update_expert_bias(teacher_gate_expert_load.detach())
        teacher_loss_matrix, teacher_alignment_scores, teacher_alignment_active = self.compute_teacher_loss_matrix(
            student_logits=student_logits,
            student_labels=student_inputs["labels"],
            student_attention_mask=student_inputs.get("attention_mask"),
            teacher_batches=teacher_batches,
        )
        ce_loss = student_outputs.loss
        effective_teacher_gate_weights, teacher_alignment_scores, teacher_alignment_active = (
            self.apply_gradient_alignment_routing(
                teacher_gate_weights=routed_teacher_gate_weights,
                teacher_alignment_scores=teacher_alignment_scores,
                teacher_alignment_active=teacher_alignment_active,
            )
        )
        distillation_loss = teacher_loss_matrix.mean()
        if effective_teacher_gate_weights is not None:
            distillation_loss = (teacher_loss_matrix * effective_teacher_gate_weights).sum(dim=-1).mean()
        elif routed_teacher_gate_weights is not None:
            distillation_loss = (teacher_loss_matrix * routed_teacher_gate_weights).sum(dim=-1).mean()

        if not model.training:
            batch_size = infer_batch_size(student_inputs)
            self.eval_ce_loss_sum += ce_loss.detach().float().item() * batch_size
            self.eval_ce_loss_count += batch_size

        loss = ce_loss + self.alpha * distillation_loss
        if teacher_gate_balance_loss is not None:
            loss = loss + teacher_gate_balance_loss * self.teacher_gate_balance_alpha

        if self.state.global_step % self.args.logging_steps == 0:
            metrics = {
                "loss": loss.item(),
                "distillation_loss": distillation_loss.item(),
                "ce_loss": ce_loss.item(),
            }
            if routed_teacher_gate_weights is not None:
                logged_weights = (
                    effective_teacher_gate_weights
                    if effective_teacher_gate_weights is not None
                    else routed_teacher_gate_weights
                )
                metrics.update(
                    {
                        f"teacher_gate_w_{teacher_index}": weight.item()
                        for teacher_index, weight in enumerate(logged_weights.detach().mean(dim=0))
                    }
                )
                metrics["teacher_gate_balance_loss"] = teacher_gate_balance_loss.item()
                metrics["teacher_gate_capacity"] = float(teacher_gate_capacity)
                metrics.update(
                    {
                        f"teacher_gate_soft_load_{teacher_index}": load.item()
                        for teacher_index, load in enumerate(teacher_gate_soft_load.detach())
                    }
                )
                metrics.update(
                    {
                        f"teacher_gate_hard_load_{teacher_index}": load.item()
                        for teacher_index, load in enumerate(teacher_gate_hard_load.detach())
                    }
                )
                metrics.update(
                    {
                        f"teacher_gate_assignment_rate_{teacher_index}": rate.item()
                        for teacher_index, rate in enumerate(teacher_gate_assignment_rate.detach())
                    }
                )
                metrics.update(
                    {
                        f"teacher_gate_bias_{teacher_index}": bias.item()
                        for teacher_index, bias in enumerate(self.teacher_gate.expert_bias.detach())
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
