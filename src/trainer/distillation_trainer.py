import math
from typing import override

import torch
from einops import einsum
from transformers import Trainer

from src.components.grace import apply_grace_routing
from src.components.teacher_gate import Gate
from src.trainer.teacher_loss_utils import compute_teacher_loss_matrix


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
        self.teacher_grace_score_ema = None

        print("Distillation Trainer initialized:")
        print(f"  - Teachers: {self.num_teachers}")
        if alpha == 0.0:
            print("  - Teacher weighting: disabled because alpha is 0")
        elif self.num_teachers > 1 and self.teacher_weighting_strategy == "routing":
            print("  - Teacher weighting: learned deep gate + GRACE routing")
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
        if self.teacher_gate is not None:
            print(f"  - Teacher gate top-k: {teacher_gate_top_k}")
            print(f"  - Teacher gate entropy alpha: {teacher_gate_entropy_alpha}")
            print(f"  - Teacher gate router z-loss alpha: {teacher_gate_router_z_loss_alpha}")
            print(f"  - GRACE threshold: {grace_threshold}")
            print(f"  - GRACE warmup ratio: {grace_warmup_ratio}")
            print(f"  - GRACE epsilon: {grace_epsilon}")
            print(f"  - GRACE softmax beta: {grace_softmax_beta}")
            print(f"  - GRACE router blend lambda: {grace_router_blend_lambda}")
            print(f"  - GRACE EMA decay: {grace_ema_decay}")
        print("  - Loss weighting: CE + alpha * KD")

    def _prepare_inputs(self, inputs):
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
        warmup_steps = math.ceil(total_steps * self.grace_warmup_ratio)
        return self.state.global_step >= warmup_steps

    @override
    def compute_loss(self, model, inputs, **_kwargs):
        student_inputs = {k: v for k, v in inputs.items() if not k.startswith("teacher")}

        student_outputs = model(
            **student_inputs,
            return_dict=True,
        )
        student_logits = student_outputs.logits
        teacher_router_logits = (
            self.teacher_gate.compute_router_logits(
                student_labels=student_inputs["labels"],
            )
            if self.teacher_gate is not None
            else None
        )
        teacher_router_weights = torch.softmax(teacher_router_logits, dim=-1) if teacher_router_logits is not None else None
        teacher_gate_entropy_loss = None
        teacher_gate_z_loss = None
        routed_teacher_weights = teacher_router_weights
        teacher_mix_weights = None
        teacher_grace_active_mask = None
        if teacher_router_weights is not None:
            teacher_gate_z_loss = torch.logsumexp(teacher_router_logits.float(), dim=-1).square().mean().to(
                dtype=teacher_router_logits.dtype
            )
            teacher_gate_entropy_loss = (
                (teacher_router_weights * torch.log_softmax(teacher_router_logits, dim=-1))
                .sum(dim=-1)
                .mean()
                .to(dtype=teacher_router_weights.dtype)
            )

            num_teachers = teacher_router_weights.shape[1]
            top_k = max(1, min(self.teacher_gate_top_k, num_teachers))
            topk_indices = teacher_router_logits.topk(top_k, dim=-1).indices
            topk_mask = (
                torch.nn.functional.one_hot(
                    topk_indices,
                    num_classes=num_teachers,
                )
                .sum(dim=1)
                .to(dtype=torch.bool)
            )
            routed_teacher_weights = torch.where(
                topk_mask,
                teacher_router_weights,
                torch.zeros_like(teacher_router_weights),
            )
            routed_teacher_weights = routed_teacher_weights / routed_teacher_weights.sum(dim=-1, keepdim=True).clamp(
                min=torch.finfo(routed_teacher_weights.dtype).eps
            )

        ce_loss = student_outputs.loss
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
            (
                teacher_loss_matrix,
                teacher_grace_scores,
                teacher_grace_active_mask,
            ) = compute_teacher_loss_matrix(
                student_logits=student_logits,
                student_labels=student_inputs["labels"],
                model=model,
                teacher_target_batches=teacher_target_batches,
                collect_grace_tensors=self.should_apply_grace_routing(),
                grace_threshold=self.grace_threshold,
                distillation_prepare_batch_fn=self.distillation_prepare_batch_fn,
                distillation_loss_fn=self.distillation_loss_fn,
                student_temperature=self.student_temperature,
                teacher_temperature=self.teacher_temperature,
            )

            if not self.should_apply_grace_routing():
                if routed_teacher_weights is not None:
                    teacher_mix_weights = routed_teacher_weights.to(dtype=teacher_loss_matrix.dtype)
                    teacher_mix_weights = teacher_mix_weights / teacher_mix_weights.sum(dim=-1, keepdim=True).clamp(
                        min=torch.finfo(teacher_mix_weights.dtype).eps
                    )
            else:
                (
                    teacher_mix_weights,
                    self.teacher_grace_score_ema,
                ) = apply_grace_routing(
                    routed_teacher_weights=routed_teacher_weights,
                    teacher_grace_scores=teacher_grace_scores,
                    teacher_grace_active_mask=teacher_grace_active_mask,
                    prev_grace_score_ema=self.teacher_grace_score_ema,
                    grace_ema_decay=self.grace_ema_decay,
                    grace_softmax_beta=self.grace_softmax_beta,
                    grace_router_blend_lambda=self.grace_router_blend_lambda,
                    grace_epsilon=self.grace_epsilon,
                )

            distillation_loss = teacher_loss_matrix.mean()
            if teacher_mix_weights is not None:
                distillation_loss = einsum(
                    teacher_loss_matrix,
                    teacher_mix_weights,
                    "batch teacher, batch teacher -> batch",
                ).mean()
            elif routed_teacher_weights is not None:
                distillation_loss = einsum(
                    teacher_loss_matrix,
                    routed_teacher_weights,
                    "batch teacher, batch teacher -> batch",
                ).mean()

        loss = ce_loss + self.alpha * distillation_loss
        if teacher_gate_entropy_loss is not None:
            loss = loss + teacher_gate_entropy_loss * self.teacher_gate_entropy_alpha
        if teacher_gate_z_loss is not None:
            loss = loss + teacher_gate_z_loss * self.teacher_gate_router_z_loss_alpha

        if self.state.global_step % self.args.logging_steps == 0:
            metrics = {
                "loss": loss.item(),
                "ce_loss": ce_loss.item(),
                "kd_loss": distillation_loss.item(),
            }
            logged_teacher_mix_weights = teacher_mix_weights if teacher_mix_weights is not None else routed_teacher_weights
            if logged_teacher_mix_weights is not None:
                safe_mix_weights = logged_teacher_mix_weights.clamp(
                    min=torch.finfo(logged_teacher_mix_weights.dtype).eps
                )
                metrics["teacher_mix_entropy"] = (
                    -(safe_mix_weights * safe_mix_weights.log()).sum(dim=-1).mean().item()
                )
            if routed_teacher_weights is not None:
                metrics["teacher_gate_entropy_loss"] = (
                    teacher_gate_entropy_loss.item() if teacher_gate_entropy_loss is not None else 0.0
                )
                metrics["teacher_gate_router_z_loss"] = teacher_gate_z_loss.item() if teacher_gate_z_loss is not None else 0.0
                safe_router_weights = teacher_router_weights.clamp(min=torch.finfo(teacher_router_weights.dtype).eps)
                safe_routed_weights = routed_teacher_weights.clamp(min=torch.finfo(routed_teacher_weights.dtype).eps)
                metrics["teacher_gate_router_entropy"] = (
                    -(safe_router_weights * safe_router_weights.log()).sum(dim=-1).mean().item()
                )
                metrics["teacher_gate_routed_entropy"] = (
                    -(safe_routed_weights * safe_routed_weights.log()).sum(dim=-1).mean().item()
                )
                metrics["teacher_gate_active_teachers"] = (
                    (routed_teacher_weights > 0).to(dtype=routed_teacher_weights.dtype).sum(dim=-1).mean().item()
                )
                metrics["teacher_grace_routing_active"] = float(self.should_apply_grace_routing())
                if teacher_grace_active_mask is not None:
                    teacher_grace_active = teacher_grace_active_mask.detach().to(dtype=routed_teacher_weights.dtype)
                    metrics.update(
                        {
                            f"teacher_grace_active_{teacher_index}": active_rate.item()
                            for teacher_index, active_rate in enumerate(teacher_grace_active.mean(dim=0))
                        }
                    )
                    metrics["teacher_grace_active_teachers"] = teacher_grace_active.sum(dim=-1).mean().item()
                    if self.teacher_grace_score_ema is not None:
                        metrics.update(
                            {
                                f"teacher_grace_score_ema_{teacher_index}": score.item()
                                for teacher_index, score in enumerate(self.teacher_grace_score_ema.detach())
                            }
                        )
            self.log(metrics)

        return loss
