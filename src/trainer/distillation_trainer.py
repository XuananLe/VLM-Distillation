import time
from typing import override

from transformers import PreTrainedModel, Trainer

from src.trainer.metrics_utils import build_distillation_train_metrics
from src.trainer.setup_utils import (
    log_distillation_trainer_setup,
    maybe_create_teacher_gate,
    maybe_create_reinforced_teacher_selector,
    normalize_teacher_models,
)
from src.trainer.distillation_utils import setup_layer_matching
from src.trainer.step_utils import (
    apply_grace_and_compute_distillation_loss,
    apply_teacher_gate_routing,
    compute_layer_distillation_loss,
    compute_teacher_losses_and_grace,
    compute_total_loss,
    prepare_teacher_batches,
    run_student_forward_and_teacher_gate,
)


class DistillationTrainer(Trainer):
    """Trainer that combines CE, KD, routing, GRACE, and optional layer distillation."""
    def __init__(
        self,
        teacher_model: PreTrainedModel = None,
        teacher_count: int | None = None,
        student_tokenizer=None,
        teacher_tokenizers=None,
        teacher_weighting_strategy: str = "routing",
        loss_function: str = "uld_loss",
        layer_distill_source: str = "none",
        layer_distill_weight: float = 0.0,
        layer_match_json_path: str | None = None,
        layer_match_topk: int = 1,
        student_layer_indices: list[int] | None = None,
        teacher_layer_indices: list[int] | None = None,
        student_temperature: float = 2.0,
        teacher_temperature: float = 2.0,
        skip_student_eos: bool = False,
        skip_teacher_eos: bool = False,
        alpha: float = 1.0,
        teacher_gate_balance_alpha: float = 1e-2,
        teacher_gate_top_k: int = 1,
        teacher_gate_capacity_factor: float = 1.25,
        teacher_gate_bias_update_rate: float = 1e-3,
        teacher_gate_temperature: float = 1.5,
        teacher_gate_noise_std: float = 0.01,
        teacher_gate_entropy_alpha: float = 1e-3,
        teacher_gate_router_z_loss_alpha: float = 1e-3,
        teacher_gate_hard_routing_warmup_ratio: float = 0.2,
        grace_threshold: float = 0.0,
        grace_warmup_ratio: float = 0.0,
        grace_epsilon: float = 0.01,
        grace_softmax_beta: float = 20.0,
        grace_router_blend_lambda: float = 0.5,
        grace_ema_decay: float = 0.9,
        reinforced_selection_warmup_ratio: float = 0.1,
        reinforced_selection_reward_type: str = "reward2",
        reinforced_selection_reward_ema_decay: float = 0.9,
        reinforced_selection_policy_alpha: float = 1.0,
        trie_wasserstein_rho: float = 0.7,
        trie_wasserstein_topk: int = 64,
        *args,
        **kwargs
    ):
        """Initialize distillation-specific models, losses, routing modules, and trainer state."""
        super().__init__(*args, **kwargs)

        from src.components import loss as distillation_loss_module

        self.loss_function = loss_function
        (
            self.distillation_prepare_batch_fn,
            self.distillation_loss_fn,
        ) = distillation_loss_module.build_distillation_loss(
            loss_function=loss_function,
            student_tokenizer=student_tokenizer,
            teacher_tokenizers=teacher_tokenizers,
            trie_wasserstein_rho=trie_wasserstein_rho,
            trie_wasserstein_topk=trie_wasserstein_topk,
        )
        self.teacher_weighting_strategy = teacher_weighting_strategy

        self.teacher_models, self.num_teachers = normalize_teacher_models(
            teacher_model,
            teacher_count,
        )
        self.layer_distill_source = layer_distill_source
        self.layer_distill_weight = layer_distill_weight
        self.layer_match_json_path = layer_match_json_path
        self.layer_match_topk = layer_match_topk
        self.student_layer_indices = []
        self.teacher_layer_indices = list(teacher_layer_indices or [])
        self.teacher_layer_soft_matches = []
        self.layer_distillation_enabled = (
            layer_distill_source in {"vision", "model"}
            and layer_distill_weight > 0.0
            and (bool(student_layer_indices) or bool(layer_match_json_path))
        )
        if self.layer_distillation_enabled:
            if not self.teacher_models:
                raise ValueError(
                    "Layer distillation requires live teacher models to be loaded."
                )
            if teacher_layer_indices is None and not layer_match_json_path:
                raise ValueError(
                    "`teacher_layer_indices` must be provided when layer distillation is enabled."
                )
            self.student_layer_indices, self.teacher_layer_soft_matches = setup_layer_matching(
                self.model,
                self.teacher_models,
                layer_match_json_path,
                layer_match_topk,
                layer_distill_source,
                list(student_layer_indices or []),
                self.teacher_layer_indices,
            )
        self.teacher_gate = maybe_create_teacher_gate(
            model=self.model,
            num_teachers=self.num_teachers,
            teacher_weighting_strategy=self.teacher_weighting_strategy,
            teacher_gate_bias_update_rate=teacher_gate_bias_update_rate,
            teacher_gate_temperature=teacher_gate_temperature,
            teacher_gate_noise_std=teacher_gate_noise_std,
        )
        self.reinforced_teacher_selector = maybe_create_reinforced_teacher_selector(
            model=self.model,
            num_teachers=self.num_teachers,
            teacher_weighting_strategy=self.teacher_weighting_strategy,
        )

        self.student_temperature = float(student_temperature)
        self.teacher_temperature = float(teacher_temperature)
        self.skip_student_eos = skip_student_eos
        self.skip_teacher_eos = skip_teacher_eos
        self.alpha = alpha
        self.teacher_gate_balance_alpha = teacher_gate_balance_alpha
        self.teacher_gate_top_k = teacher_gate_top_k
        self.teacher_gate_capacity_factor = teacher_gate_capacity_factor
        self.teacher_gate_bias_update_rate = teacher_gate_bias_update_rate
        self.teacher_gate_temperature = teacher_gate_temperature
        self.teacher_gate_noise_std = teacher_gate_noise_std
        self.teacher_gate_entropy_alpha = teacher_gate_entropy_alpha
        self.teacher_gate_router_z_loss_alpha = teacher_gate_router_z_loss_alpha
        self.teacher_gate_hard_routing_warmup_ratio = teacher_gate_hard_routing_warmup_ratio
        self.grace_threshold = grace_threshold
        self.grace_warmup_ratio = grace_warmup_ratio
        self.grace_epsilon = grace_epsilon
        self.grace_softmax_beta = grace_softmax_beta
        self.grace_router_blend_lambda = grace_router_blend_lambda
        self.grace_ema_decay = grace_ema_decay
        self.reinforced_selection_warmup_ratio = reinforced_selection_warmup_ratio
        self.reinforced_selection_reward_type = reinforced_selection_reward_type
        self.reinforced_selection_reward_ema_decay = reinforced_selection_reward_ema_decay
        self.reinforced_selection_policy_alpha = reinforced_selection_policy_alpha
        self.trie_wasserstein_rho = trie_wasserstein_rho
        self.trie_wasserstein_topk = trie_wasserstein_topk
        self.last_compute_loss_end_time = None
        self.teacher_grace_score_ema = None
        self.reinforced_selection_reward_baseline = None

        log_distillation_trainer_setup(
            num_teachers=self.num_teachers,
            teacher_weighting_strategy=self.teacher_weighting_strategy,
            loss_function=loss_function,
            layer_distillation_enabled=self.layer_distillation_enabled,
            layer_distill_source=self.layer_distill_source,
            layer_distill_weight=self.layer_distill_weight,
            layer_match_json_path=self.layer_match_json_path,
            layer_match_topk=self.layer_match_topk,
            student_layer_indices=self.student_layer_indices,
            teacher_layer_soft_matches=self.teacher_layer_soft_matches,
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
            teacher_gate_temperature=teacher_gate_temperature,
            teacher_gate_noise_std=teacher_gate_noise_std,
            teacher_gate_entropy_alpha=teacher_gate_entropy_alpha,
            teacher_gate_router_z_loss_alpha=teacher_gate_router_z_loss_alpha,
            teacher_gate_hard_routing_warmup_ratio=teacher_gate_hard_routing_warmup_ratio,
            grace_threshold=grace_threshold,
            grace_warmup_ratio=grace_warmup_ratio,
            grace_epsilon=grace_epsilon,
            grace_softmax_beta=grace_softmax_beta,
            grace_router_blend_lambda=grace_router_blend_lambda,
            grace_ema_decay=grace_ema_decay,
            reinforced_selection_warmup_ratio=reinforced_selection_warmup_ratio,
            reinforced_selection_reward_type=reinforced_selection_reward_type,
            reinforced_selection_reward_ema_decay=reinforced_selection_reward_ema_decay,
            reinforced_selection_policy_alpha=reinforced_selection_policy_alpha,
            trie_wasserstein_rho=trie_wasserstein_rho,
            trie_wasserstein_topk=trie_wasserstein_topk,
        )

    @override
    def _prepare_inputs(self, inputs):
        """Prepare student inputs on-device while preserving teacher-prefixed tensors unchanged."""
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

    def should_apply_grace_routing(self) -> bool:
        """Return whether GRACE refinement is active at the current global step."""
        if self.teacher_gate is None or not self.model.training:
            return False

        if self.grace_warmup_ratio <= 0.0:
            return True

        total_steps = max(self.state.max_steps, getattr(self.args, "max_steps", 0))
        if total_steps <= 0:
            return True
        import math
        warmup_steps = math.ceil(total_steps * self.grace_warmup_ratio)
        return self.state.global_step >= warmup_steps

    def current_teacher_gate_top_k(self) -> int:
        """Return the effective teacher-gate top-k after warmup scheduling."""
        if self.teacher_gate is None:
            return self.teacher_gate_top_k
        if not self.model.training or self.teacher_gate_hard_routing_warmup_ratio <= 0.0:
            return self.teacher_gate_top_k
        total_steps = max(self.state.max_steps, getattr(self.args, "max_steps", 0))
        if total_steps <= 0:
            return self.teacher_gate_top_k
        import math
        warmup_steps = math.ceil(total_steps * self.teacher_gate_hard_routing_warmup_ratio)
        if self.state.global_step < warmup_steps:
            return self.num_teachers
        return self.teacher_gate_top_k

    def reinforced_selection_warmup_active(self) -> bool:
        """Return whether reinforced teacher selection is still in its all-teachers warmup phase."""
        if self.reinforced_teacher_selector is None or not self.model.training:
            return False
        if self.reinforced_selection_warmup_ratio <= 0.0:
            return False
        total_steps = max(self.state.max_steps, getattr(self.args, "max_steps", 0))
        if total_steps <= 0:
            return False
        import math
        warmup_steps = math.ceil(total_steps * self.reinforced_selection_warmup_ratio)
        return self.state.global_step < warmup_steps

    @override
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Run one full distillation step and return the combined training loss."""
        compute_loss_start_time = time.perf_counter()
        outside_compute_loss_time = (
            None
            if self.last_compute_loss_end_time is None
            else compute_loss_start_time - self.last_compute_loss_end_time
        )

        student_inputs = {k: v for k, v in inputs.items() if not k.startswith("teacher")}
        teacher_batches, cached_teacher_batches = prepare_teacher_batches(
            inputs=inputs,
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
        ce_loss = student_and_gate["student_outputs"].loss
        teacher_losses = compute_teacher_losses_and_grace(
            trainer=self,
            model=model,
            student_logits=student_and_gate["student_logits"],
            student_labels=student_inputs["labels"],
            ce_loss=ce_loss,
            teacher_batches=teacher_batches,
            cached_teacher_batches=cached_teacher_batches,
        )
        grace_and_loss = apply_grace_and_compute_distillation_loss(
            trainer=self,
            routed_teacher_gate_weights=gate_routing["routed_teacher_gate_weights"],
            teacher_grace_scores=teacher_losses["teacher_grace_scores"],
            teacher_grace_active=teacher_losses["teacher_grace_active"],
            teacher_loss_matrix=teacher_losses["teacher_loss_matrix"],
            teacher_logits_batches=teacher_losses["teacher_logit_batches"],
            teacher_label_batches=teacher_losses["teacher_label_batches"],
            labels=student_inputs["labels"],
            attention_mask=student_inputs.get("attention_mask"),
            ce_loss=ce_loss,
        )
        layer_and_loss = compute_layer_distillation_loss(
            trainer=self,
            teacher_batches=teacher_batches,
            student_layer_representations=student_and_gate["student_layer_representations"],
        )
        loss = compute_total_loss(
            trainer=self,
            ce_loss=ce_loss,
            distillation_loss=grace_and_loss["distillation_loss"],
            layer_distillation_loss=layer_and_loss["layer_distillation_loss"],
            teacher_gate_balance_loss=gate_routing["teacher_gate_balance_loss"],
            teacher_gate_entropy_loss=gate_routing["teacher_gate_entropy_loss"],
            teacher_gate_z_loss=gate_routing["teacher_gate_z_loss"],
            teacher_selection_policy_loss=grace_and_loss["teacher_selection_policy_loss"],
        )

        compute_loss_time = time.perf_counter() - compute_loss_start_time
        self.last_compute_loss_end_time = time.perf_counter()

        if self.state.global_step % self.args.logging_steps == 0:
            metrics = build_distillation_train_metrics(
                loss=loss,
                distillation_loss=grace_and_loss["distillation_loss"],
                ce_loss=ce_loss,
                layer_distillation_loss=layer_and_loss["layer_distillation_loss"],
                layer_distillation_time=layer_and_loss["layer_distillation_time"],
                layer_distill_source=self.layer_distill_source if self.layer_distillation_enabled else None,
                compute_loss_time=compute_loss_time,
                student_forward_time=student_and_gate["student_forward_time"],
                teacher_gate_time=student_and_gate["teacher_gate_time"],
                routing_constraint_time=gate_routing["routing_constraint_time"],
                teacher_loss_matrix_time=teacher_losses["teacher_loss_matrix_time"],
                grace_routing_time=grace_and_loss["grace_routing_time"],
                outside_compute_loss_time=outside_compute_loss_time,
                teacher_loss_matrix=teacher_losses["teacher_loss_matrix"],
                routed_teacher_gate_weights=gate_routing["routed_teacher_gate_weights"],
                effective_teacher_gate_weights=grace_and_loss["effective_teacher_gate_weights"],
                teacher_gate_weights=student_and_gate["teacher_gate_weights"],
                teacher_gate_logits=student_and_gate["teacher_gate_logits"],
                teacher_gate_routing_scores=student_and_gate["teacher_gate_routing_scores"],
                teacher_gate_balance_loss=gate_routing["teacher_gate_balance_loss"],
                teacher_gate_entropy_loss=gate_routing["teacher_gate_entropy_loss"],
                teacher_gate_z_loss=gate_routing["teacher_gate_z_loss"],
                teacher_gate_capacity=gate_routing["teacher_gate_capacity"],
                teacher_gate_routing_fallback_rate=gate_routing["teacher_gate_routing_fallback_rate"],
                teacher_gate_soft_load=gate_routing["teacher_gate_soft_load"],
                teacher_gate_hard_load=gate_routing["teacher_gate_hard_load"],
                teacher_gate_assignment_rate=gate_routing["teacher_gate_assignment_rate"],
                teacher_gate_bias=None if self.teacher_gate is None else self.teacher_gate.expert_bias,
                teacher_grace_scores=grace_and_loss["teacher_grace_scores"],
                teacher_grace_active=grace_and_loss["teacher_grace_active"],
                teacher_grace_score_ema=self.teacher_grace_score_ema,
                teacher_grace_weights=grace_and_loss["teacher_grace_weights"],
                teacher_grace_fallback_rate=grace_and_loss["teacher_grace_fallback_rate"],
                reinforced_selection_metrics=grace_and_loss["reinforced_selection_metrics"],
                grace_warmup_active=self.should_apply_grace_routing(),
            )
            self.log(metrics)

        return (loss, student_and_gate["student_outputs"]) if return_outputs else loss
