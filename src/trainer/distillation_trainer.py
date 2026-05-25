import math
from typing import override

import torch
from einops import einsum
from transformers import Trainer

from src.components.reinforced_teacher_selection import (
    ReinforcedTeacherSelectionPolicy,
    compute_reinforced_selection_state,
)
from src.trainer.teacher_loss_utils import compute_teacher_loss_matrix


class DistillationTrainer(Trainer):
    def __init__(
        self,
        teacher_count: int | None = None,
        student_tokenizer=None,
        teacher_tokenizers=None,
        teacher_weighting_strategy: str = "reinforced_selection",
        loss_function: str = "uld_loss",
        student_temperature: float = 2.0,
        teacher_temperature: float = 2.0,
        alpha: float = 1.0,
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

        self.alpha = alpha
        if self.alpha == 0.0:
            def prepare_teacher_batch(**_kwargs) -> None:
                return None

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

        self.reinforced_teacher_selector = None
        if self.teacher_weighting_strategy == "reinforced_selection":
            if self.num_teachers <= 1:
                raise ValueError(
                    "Reinforced teacher selection requires at least two teachers; use single-teacher distillation instead."
                )
            self.reinforced_teacher_selector = ReinforcedTeacherSelectionPolicy(
                self.model,
                self.num_teachers,
            )
            self.model.reinforced_teacher_selector = self.reinforced_teacher_selector

        self.student_temperature = float(student_temperature)
        self.teacher_temperature = float(teacher_temperature)
        self.reinforced_selection_warmup_ratio = reinforced_selection_warmup_ratio
        self.reinforced_selection_reward_type = reinforced_selection_reward_type
        self.reinforced_selection_reward_ema_decay = reinforced_selection_reward_ema_decay
        self.reinforced_selection_policy_alpha = reinforced_selection_policy_alpha
        self.reinforced_selection_reward_baseline = None

        print("Distillation Trainer initialized:")
        print(f"  - Teachers: {self.num_teachers}")
        if alpha == 0.0:
            print("  - Teacher weighting: disabled because alpha is 0")
        elif self.num_teachers > 1 and self.teacher_weighting_strategy == "reinforced_selection":
            print("  - Teacher weighting: reinforced teacher selection")
        elif self.num_teachers == 1:
            print("  - Teacher weighting: single teacher")
        else:
            print("  - Teacher weighting: uniform mean")
        print(f"  - Loss function: {loss_function}")
        if loss_function == "trie_wasserstein_loss":
            print(f"  - Trie Wasserstein rho: {trie_wasserstein_rho}")
            print(f"  - Trie Wasserstein top-k: {trie_wasserstein_topk}")
        print(f"  - Student temperature: {self.student_temperature}")
        print(f"  - Teacher temperature: {self.teacher_temperature}")
        print("  - Drop final supervised token for KD: True")
        print(f"  - Alpha: {alpha}")
        print(f"  - KD weight: {alpha}")
        print("  - CE weight: 1.0")
        if self.reinforced_teacher_selector is not None:
            print(f"  - Reinforced selection warmup ratio: {reinforced_selection_warmup_ratio}")
            print(f"  - Reinforced selection reward type: {reinforced_selection_reward_type}")
            print(f"  - Reinforced selection reward EMA decay: {reinforced_selection_reward_ema_decay}")
            print(f"  - Reinforced selection policy alpha: {reinforced_selection_policy_alpha}")
        print("  - Loss weighting: CE + alpha * KD")

    def _prepare_inputs(self, inputs):
        if not isinstance(inputs, dict):
            return super()._prepare_inputs(inputs)

        teacher_inputs = {key: value for key, value in inputs.items() if key.startswith("teacher")}
        student_inputs = {key: value for key, value in inputs.items() if not key.startswith("teacher")}

        prepared_inputs = super()._prepare_inputs(student_inputs)
        prepared_inputs.update(teacher_inputs)
        return prepared_inputs

    def reinforced_selection_warmup_active(self) -> bool:
        if self.reinforced_teacher_selector is None or not self.model.training:
            return False
        if self.reinforced_selection_warmup_ratio <= 0.0:
            return False
        total_steps = max(self.state.max_steps, getattr(self.args, "max_steps", 0))
        if total_steps <= 0:
            return False
        warmup_steps = math.ceil(total_steps * self.reinforced_selection_warmup_ratio)
        return self.state.global_step < warmup_steps

    @override
    def compute_loss(self, model, inputs, **_kwargs):
        student_inputs = {k: v for k, v in inputs.items() if not k.startswith("teacher")}

        student_outputs = model(
            **student_inputs,
            return_dict=True,
        )
        student_logits = student_outputs.logits
        ce_loss = student_outputs.loss

        teacher_selection_policy_loss = None
        if self.alpha == 0.0:
            distillation_loss = ce_loss.new_zeros(())
        else:
            teacher_prefixes = [f"teacher_{teacher_index}" for teacher_index in range(self.num_teachers)]
            teacher_target_batches = [
                (
                    self._prepare_input(inputs[f"{prefix}_cached_logits"]).to(dtype=student_logits.dtype),
                    self._prepare_input(inputs[f"{prefix}_cached_labels"]),
                )
                for prefix in teacher_prefixes
            ]
            teacher_loss_matrix, selection_teacher_logits, selection_teacher_labels = compute_teacher_loss_matrix(
                student_logits=student_logits,
                student_labels=student_inputs["labels"],
                teacher_target_batches=teacher_target_batches,
                collect_teacher_targets_for_selection=self.teacher_weighting_strategy == "reinforced_selection",
                distillation_prepare_batch_fn=self.distillation_prepare_batch_fn,
                distillation_loss_fn=self.distillation_loss_fn,
                student_temperature=self.student_temperature,
                teacher_temperature=self.teacher_temperature,
            )

            if self.teacher_weighting_strategy == "reinforced_selection":
                reinforced_state = compute_reinforced_selection_state(
                    selector=self.reinforced_teacher_selector,
                    teacher_loss_matrix=teacher_loss_matrix,
                    selection_teacher_logits=selection_teacher_logits or [],
                    selection_teacher_labels=selection_teacher_labels or [],
                    student_labels=student_inputs["labels"],
                    student_ce_loss=ce_loss,
                    teacher_temperature=self.teacher_temperature,
                    warmup_active=self.reinforced_selection_warmup_active(),
                    reward_type=self.reinforced_selection_reward_type,
                    prev_reward_baseline=self.reinforced_selection_reward_baseline,
                    reward_ema_decay=self.reinforced_selection_reward_ema_decay,
                )
                self.reinforced_selection_reward_baseline = reinforced_state["next_reward_baseline"]
                distillation_loss = reinforced_state["distillation_loss"]
                teacher_selection_policy_loss = reinforced_state["policy_loss"]
            else:
                distillation_loss = teacher_loss_matrix.mean()

        loss = ce_loss + self.alpha * distillation_loss
        if teacher_selection_policy_loss is not None:
            loss = loss + teacher_selection_policy_loss * self.reinforced_selection_policy_alpha

        if self.state.global_step % self.args.logging_steps == 0:
            self.log(
                {
                    "loss": loss.item(),
                    "ce_loss": ce_loss.item(),
                    "kd_loss": distillation_loss.item(),
                }
            )

        return loss
