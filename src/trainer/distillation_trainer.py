import time
from typing import override

from transformers import PreTrainedModel

from src.trainer.checkpoint_utils import (
    load_best_non_lora_weights,
    update_best_checkpoint_by_train_ce,
    update_eval_ce_logs,
)
from src.trainer.metrics_utils import build_distillation_train_metrics
from src.trainer.setup_utils import (
    log_distillation_trainer_setup,
    maybe_create_teacher_gate,
    normalize_teacher_models,
    validate_distillation_trainer_args,
)
from src.trainer.sft_trainer import SmolVLMSFTTrainer
from src.trainer.distillation_utils import release_eval_memory
from src.trainer.step_utils import (
    apply_alignment_and_compute_distillation_loss,
    apply_teacher_gate_routing,
    compute_teacher_losses_and_alignment,
    compute_total_loss,
    move_teacher_models_to_device,
    prepare_teacher_batches,
    run_student_forward_and_teacher_gate,
    update_eval_ce_stats,
)


class DistillationTrainer(SmolVLMSFTTrainer):
    def __init__(
        self,
        teacher_model: PreTrainedModel = None,
        teacher_count: int | None = None,
        teacher_weighting_strategy: str = "routing",
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
        teacher_gate_router_z_loss_alpha: float = 1e-3,
        gradient_alignment_threshold: float = 0.0,
        gradient_alignment_warmup_ratio: float = 0.0,
        gradient_alignment_epsilon: float = 0.01,
        gradient_alignment_softmax_beta: float = 20.0,
        gradient_alignment_router_blend_lambda: float = 0.5,
        gradient_alignment_ema_decay: float = 0.9,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        from src.components import loss as distillation_loss_module

        validate_distillation_trainer_args(
            alpha=alpha,
            teacher_gate_balance_alpha=teacher_gate_balance_alpha,
            teacher_gate_top_k=teacher_gate_top_k,
            teacher_gate_capacity_factor=teacher_gate_capacity_factor,
            teacher_gate_bias_update_rate=teacher_gate_bias_update_rate,
            teacher_gate_router_z_loss_alpha=teacher_gate_router_z_loss_alpha,
            gradient_alignment_warmup_ratio=gradient_alignment_warmup_ratio,
            gradient_alignment_epsilon=gradient_alignment_epsilon,
            gradient_alignment_softmax_beta=gradient_alignment_softmax_beta,
            gradient_alignment_router_blend_lambda=gradient_alignment_router_blend_lambda,
            gradient_alignment_ema_decay=gradient_alignment_ema_decay,
            teacher_weighting_strategy=teacher_weighting_strategy,
            loss_function=loss_function,
            distillation_loss_module=distillation_loss_module,
        )

        self.loss_function = loss_function
        self.distillation_loss_fn = getattr(distillation_loss_module, loss_function)
        self.distillation_logit_grad_fn = distillation_loss_module.distillation_logit_grad
        self.teacher_weighting_strategy = teacher_weighting_strategy

        self.teacher_models, self.num_teachers = normalize_teacher_models(
            teacher_model,
            teacher_count,
        )
        self.teacher_gate = maybe_create_teacher_gate(
            model=self.model,
            num_teachers=self.num_teachers,
            teacher_weighting_strategy=self.teacher_weighting_strategy,
            teacher_gate_bias_update_rate=teacher_gate_bias_update_rate,
        )

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
        self.teacher_gate_router_z_loss_alpha = teacher_gate_router_z_loss_alpha
        self.gradient_alignment_threshold = gradient_alignment_threshold
        self.gradient_alignment_warmup_ratio = gradient_alignment_warmup_ratio
        self.gradient_alignment_epsilon = gradient_alignment_epsilon
        self.gradient_alignment_softmax_beta = gradient_alignment_softmax_beta
        self.gradient_alignment_router_blend_lambda = gradient_alignment_router_blend_lambda
        self.gradient_alignment_ema_decay = gradient_alignment_ema_decay
        self.non_lora_require_grad_only = True
        self.eval_ce_loss_sum = 0.0
        self.eval_ce_loss_count = 0
        self.last_compute_loss_end_time = None
        self.latest_train_ce_loss = None
        self.teacher_alignment_score_ema = None

        log_distillation_trainer_setup(
            num_teachers=self.num_teachers,
            teacher_weighting_strategy=self.teacher_weighting_strategy,
            loss_function=loss_function,
            student_temperature=self.student_temperature,
            teacher_temperature=self.teacher_temperature,
            skip_student_eos=self.skip_student_eos,
            skip_teacher_eos=self.skip_teacher_eos,
            alpha=alpha,
            teacher_gate=self.teacher_gate,
            teacher_gate_balance_alpha=teacher_gate_balance_alpha,
            teacher_gate_top_k=teacher_gate_top_k,
            teacher_gate_capacity_factor=teacher_gate_capacity_factor,
            teacher_gate_bias_update_rate=teacher_gate_bias_update_rate,
            teacher_gate_router_z_loss_alpha=teacher_gate_router_z_loss_alpha,
            gradient_alignment_threshold=gradient_alignment_threshold,
            gradient_alignment_warmup_ratio=gradient_alignment_warmup_ratio,
            gradient_alignment_epsilon=gradient_alignment_epsilon,
            gradient_alignment_softmax_beta=gradient_alignment_softmax_beta,
            gradient_alignment_router_blend_lambda=gradient_alignment_router_blend_lambda,
            gradient_alignment_ema_decay=gradient_alignment_ema_decay,
        )

    def tracks_best_checkpoint_by_train_ce(self) -> bool:
        metric_name = getattr(self.args, "metric_for_best_model", None)
        return (
            getattr(self.args, "eval_strategy", "no") == "no"
            and metric_name in (None, "train_ce_loss", "ce_loss", "train/ce_loss")
        )

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

    def should_apply_gradient_alignment_routing(self) -> bool:
        if self.teacher_gate is None or not self.model.training:
            return False

        if self.gradient_alignment_warmup_ratio <= 0.0:
            return True

        total_steps = max(self.state.max_steps, getattr(self.args, "max_steps", 0))
        if total_steps <= 0:
            return True
        import math
        warmup_steps = math.ceil(total_steps * self.gradient_alignment_warmup_ratio)
        return self.state.global_step >= warmup_steps

    @override
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        compute_loss_start_time = time.perf_counter()
        outside_compute_loss_time = (
            None
            if self.last_compute_loss_end_time is None
            else compute_loss_start_time - self.last_compute_loss_end_time
        )
        move_teacher_models_to_device(self.teacher_models, inputs["input_ids"].device)

        student_inputs = {k: v for k, v in inputs.items() if not k.startswith("teacher")}
        teacher_batches, cached_teacher_batches = prepare_teacher_batches(
            inputs=inputs,
            student_inputs=student_inputs,
            num_teachers=self.num_teachers,
            teacher_models=self.teacher_models,
        )

        student_and_gate = run_student_forward_and_teacher_gate(
            trainer=self,
            model=model,
            student_inputs=student_inputs,
        )
        gate_routing = apply_teacher_gate_routing(
            trainer=self,
            model=model,
            teacher_gate_logits=student_and_gate["teacher_gate_logits"],
            teacher_gate_weights=student_and_gate["teacher_gate_weights"],
            teacher_gate_routing_scores=student_and_gate["teacher_gate_routing_scores"],
        )
        teacher_losses = compute_teacher_losses_and_alignment(
            trainer=self,
            student_logits=student_and_gate["student_logits"],
            student_labels=student_inputs["labels"],
            teacher_batches=teacher_batches,
            cached_teacher_batches=cached_teacher_batches,
        )
        ce_loss = student_and_gate["student_outputs"].loss
        if model.training:
            self.latest_train_ce_loss = ce_loss.detach().float().item()
        alignment_and_loss = apply_alignment_and_compute_distillation_loss(
            trainer=self,
            routed_teacher_gate_weights=gate_routing["routed_teacher_gate_weights"],
            teacher_alignment_scores=teacher_losses["teacher_alignment_scores"],
            teacher_alignment_active=teacher_losses["teacher_alignment_active"],
            teacher_loss_matrix=teacher_losses["teacher_loss_matrix"],
        )

        update_eval_ce_stats(
            trainer=self,
            student_inputs=student_inputs,
            ce_loss=ce_loss,
        )
        loss = compute_total_loss(
            trainer=self,
            ce_loss=ce_loss,
            distillation_loss=alignment_and_loss["distillation_loss"],
            teacher_gate_balance_loss=gate_routing["teacher_gate_balance_loss"],
            teacher_gate_z_loss=gate_routing["teacher_gate_z_loss"],
        )

        compute_loss_time = time.perf_counter() - compute_loss_start_time
        self.last_compute_loss_end_time = time.perf_counter()

        if self.state.global_step % self.args.logging_steps == 0:
            metrics = build_distillation_train_metrics(
                loss=loss,
                distillation_loss=alignment_and_loss["distillation_loss"],
                ce_loss=ce_loss,
                compute_loss_time=compute_loss_time,
                student_forward_time=student_and_gate["student_forward_time"],
                teacher_gate_time=student_and_gate["teacher_gate_time"],
                routing_constraint_time=gate_routing["routing_constraint_time"],
                teacher_loss_matrix_time=teacher_losses["teacher_loss_matrix_time"],
                alignment_routing_time=alignment_and_loss["alignment_routing_time"],
                outside_compute_loss_time=outside_compute_loss_time,
                teacher_loss_matrix=teacher_losses["teacher_loss_matrix"],
                routed_teacher_gate_weights=gate_routing["routed_teacher_gate_weights"],
                effective_teacher_gate_weights=alignment_and_loss["effective_teacher_gate_weights"],
                teacher_gate_weights=student_and_gate["teacher_gate_weights"],
                teacher_gate_logits=student_and_gate["teacher_gate_logits"],
                teacher_gate_routing_scores=student_and_gate["teacher_gate_routing_scores"],
                teacher_gate_balance_loss=gate_routing["teacher_gate_balance_loss"],
                teacher_gate_z_loss=gate_routing["teacher_gate_z_loss"],
                teacher_gate_capacity=gate_routing["teacher_gate_capacity"],
                teacher_gate_routing_fallback_rate=gate_routing["teacher_gate_routing_fallback_rate"],
                teacher_gate_soft_load=gate_routing["teacher_gate_soft_load"],
                teacher_gate_hard_load=gate_routing["teacher_gate_hard_load"],
                teacher_gate_assignment_rate=gate_routing["teacher_gate_assignment_rate"],
                teacher_gate_bias=None if self.teacher_gate is None else self.teacher_gate.expert_bias,
                teacher_alignment_scores=alignment_and_loss["teacher_alignment_scores"],
                teacher_alignment_active=alignment_and_loss["teacher_alignment_active"],
                teacher_alignment_score_ema=self.teacher_alignment_score_ema,
                teacher_alignment_weights=alignment_and_loss["teacher_alignment_weights"],
                teacher_alignment_fallback_rate=alignment_and_loss["teacher_alignment_fallback_rate"],
                alignment_warmup_active=self.should_apply_gradient_alignment_routing(),
            )
            self.log(metrics)

        return (loss, student_and_gate["student_outputs"]) if return_outputs else loss

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
                import torch
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
        super().log(update_eval_ce_logs(self, logs), start_time=start_time)

    @override
    def _save_checkpoint(self, model, trial):
        super()._save_checkpoint(model, trial)
        update_best_checkpoint_by_train_ce(self, trial)

    @override
    def _load_best_model(self):
        super()._load_best_model()
        load_best_non_lora_weights(self)
