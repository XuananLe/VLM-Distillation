from typing import override

from transformers import Trainer

from src.components.teacher_gate import Gate
from src.trainer.metrics_utils import build_distillation_train_metrics
from src.trainer.setup_utils import (
    log_distillation_trainer_setup,
    resolve_reinforced_teacher_selector,
)
from src.trainer.step_utils import (
    build_student_forward_state,
    build_teacher_loss_state,
    compute_total_loss,
    prepare_teacher_batch_sources,
    resolve_teacher_gate_state,
    resolve_teacher_weighting_state,
)
from src.trainer.teacher_loss_utils import resolve_teacher_target_batches


class DistillationTrainer(Trainer):
    def __init__(
        self,
        teacher_count: int | None = None,
        student_tokenizer=None,
        teacher_tokenizers=None,
        teacher_weighting_strategy: str = "routing",
        loss_function: str = "uld_loss",
        student_temperature: float = 2.0,
        teacher_temperature: float = 2.0,
        alpha: float = 1.0,
        teacher_gate_top_k: int = 1,
        teacher_gate_entropy_alpha: float = 1e-3,
        teacher_gate_router_z_loss_alpha: float = 1e-3,
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
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        from src.components import loss as distillation_loss_module

        self.loss_function = loss_function
        self.alpha = alpha
        if self.alpha == 0.0:
            def prepare_teacher_batch(**kwargs) -> None:
                del kwargs

            def compute_distillation_loss(**kwargs):
                return kwargs["student_logits"].new_zeros(())

            self.distillation_prepare_batch_fn = prepare_teacher_batch
            self.distillation_loss_fn = compute_distillation_loss
        else:
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
        self.teacher_weighting_strategy = "uniform_mean" if self.alpha == 0.0 else teacher_weighting_strategy

        self.num_teachers = int(teacher_count or 0)
        if self.num_teachers < 1:
            raise ValueError("DistillationTrainer requires at least one teacher.")
        self.teacher_gate = None
        if self.teacher_weighting_strategy == "routing":
            if self.num_teachers <= 1:
                raise ValueError(
                    "Teacher routing requires at least two teachers; use single-teacher distillation instead."
                )
            self.teacher_gate = Gate(
                self.model,
                self.num_teachers,
            )
            self.model.teacher_gate = self.teacher_gate
        self.reinforced_teacher_selector = resolve_reinforced_teacher_selector(
            model=self.model,
            num_teachers=self.num_teachers,
            teacher_weighting_strategy=self.teacher_weighting_strategy,
        )

        self.student_temperature = float(student_temperature)
        self.teacher_temperature = float(teacher_temperature)
        self.teacher_gate_top_k = teacher_gate_top_k
        self.teacher_gate_entropy_alpha = teacher_gate_entropy_alpha
        self.teacher_gate_router_z_loss_alpha = teacher_gate_router_z_loss_alpha
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
        self.teacher_grace_score_ema = None
        self.reinforced_selection_reward_baseline = None

        log_distillation_trainer_setup(
            num_teachers=self.num_teachers,
            teacher_weighting_strategy=self.teacher_weighting_strategy,
            loss_function=loss_function,
            student_temperature=self.student_temperature,
            teacher_temperature=self.teacher_temperature,
            alpha=alpha,
            teacher_gate=self.teacher_gate,
            teacher_gate_top_k=teacher_gate_top_k,
            teacher_gate_entropy_alpha=teacher_gate_entropy_alpha,
            teacher_gate_router_z_loss_alpha=teacher_gate_router_z_loss_alpha,
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
        # Hugging Face Trainer calls this private hook by name, so this override
        # must keep the framework method name even though project helpers avoid it.
        if not isinstance(inputs, dict):
            return super()._prepare_inputs(inputs)

        teacher_inputs = {key: value for key, value in inputs.items() if key.startswith("teacher")}
        student_inputs = {key: value for key, value in inputs.items() if not key.startswith("teacher")}

        prepared_inputs = super()._prepare_inputs(student_inputs)
        prepared_inputs.update(teacher_inputs)
        return prepared_inputs

    def should_apply_grace_routing(self) -> bool:
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

    def reinforced_selection_warmup_active(self) -> bool:
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
        student_inputs = {k: v for k, v in inputs.items() if not k.startswith("teacher")}

        student_forward_state = build_student_forward_state(
            trainer=self,
            model=model,
            student_inputs=student_inputs,
        )
        teacher_gate_state = resolve_teacher_gate_state(
            trainer=self,
            teacher_router_logits=student_forward_state["teacher_router_logits"],
            teacher_router_weights=student_forward_state["teacher_router_weights"],
        )
        ce_loss = student_forward_state["student_outputs"].loss
        if self.alpha == 0.0:
            teacher_loss_state = {
                "teacher_loss_matrix": ce_loss.new_zeros(
                    (
                        student_forward_state["student_logits"].size(0),
                        self.num_teachers,
                    )
                ),
                "teacher_grace_scores": None,
                "teacher_grace_active_mask": None,
                "selection_teacher_logits": None,
                "selection_teacher_labels": None,
            }
            teacher_weighting_state = {
                "teacher_mix_weights": None,
                "teacher_grace_scores": None,
                "teacher_grace_active_mask": None,
                "teacher_grace_weights": None,
                "teacher_grace_fallback_rate": None,
                "distillation_loss": ce_loss.new_zeros(()),
                "teacher_selection_policy_loss": None,
                "reinforced_selection_metrics": None,
            }
        else:
            teacher_batch_sources = prepare_teacher_batch_sources(
                inputs=inputs,
                num_teachers=self.num_teachers,
            )
            teacher_target_batches = resolve_teacher_target_batches(
                student_logits=student_forward_state["student_logits"],
                cached_teacher_target_batches=teacher_batch_sources.cached_teacher_target_batches,
                prepare_input_fn=self._prepare_input,
            )
            teacher_loss_state = build_teacher_loss_state(
                trainer=self,
                model=model,
                student_logits=student_forward_state["student_logits"],
                student_labels=student_inputs["labels"],
                teacher_target_batches=teacher_target_batches,
            )
            teacher_weighting_state = resolve_teacher_weighting_state(
                trainer=self,
                routed_teacher_weights=teacher_gate_state["routed_teacher_weights"],
                teacher_grace_scores=teacher_loss_state["teacher_grace_scores"],
                teacher_grace_active_mask=teacher_loss_state["teacher_grace_active_mask"],
                teacher_loss_matrix=teacher_loss_state["teacher_loss_matrix"],
                selection_teacher_logits=teacher_loss_state["selection_teacher_logits"],
                selection_teacher_labels=teacher_loss_state["selection_teacher_labels"],
                student_labels=student_inputs["labels"],
                student_ce_loss=ce_loss,
            )
        loss = compute_total_loss(
            trainer=self,
            ce_loss=ce_loss,
            distillation_loss=teacher_weighting_state["distillation_loss"],
            teacher_gate_entropy_loss=teacher_gate_state["teacher_gate_entropy_loss"],
            teacher_gate_z_loss=teacher_gate_state["teacher_gate_z_loss"],
            teacher_selection_policy_loss=teacher_weighting_state["teacher_selection_policy_loss"],
        )

        if self.state.global_step % self.args.logging_steps == 0:
            metrics = build_distillation_train_metrics(
                loss=loss,
                distillation_loss=teacher_weighting_state["distillation_loss"],
                ce_loss=ce_loss,
            )
            self.log(metrics)

        return (loss, student_forward_state["student_outputs"]) if return_outputs else loss
