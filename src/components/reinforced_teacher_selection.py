import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange
from src.components.pooling import masked_mean_pool_sequence
from src.components.teacher_gate import Gate
# https://arxiv.org/pdf/2012.06048

def compute_reinforced_teacher_descriptor_stats(
    *,
    teacher_logits: torch.Tensor,
    teacher_labels: torch.Tensor,
    teacher_temperature: float,
    skip_teacher_eos: bool,
) -> torch.Tensor:
    # Build simple per-sample teacher descriptors from supervised answer positions only.
    stats = []
    for sample_index, (sample_logits, sample_labels) in enumerate(zip(teacher_logits, teacher_labels)):
        positions = sample_labels.ne(-100).nonzero(as_tuple=False).squeeze(-1)
        if skip_teacher_eos and positions.numel() > 0:
            positions = positions[:-1]
        if positions.numel() == 0:
            raise ValueError(
                "Teacher labels contain no supervised answer tokens for reinforced selection "
                f"at sample {sample_index}."
            )

        supervised_logits = sample_logits.index_select(dim=0, index=positions)
        teacher_probs = F.softmax(supervised_logits.float() / teacher_temperature, dim=-1)
        mean_confidence = teacher_probs.max(dim=-1).values.mean()
        token_entropy = -(teacher_probs * teacher_probs.clamp_min(torch.finfo(teacher_probs.dtype).eps).log()).sum(dim=-1)
        normalized_entropy = token_entropy / max(math.log(teacher_probs.size(-1)), 1.0)
        stats.append(
            torch.stack(
                [
                    mean_confidence,
                    normalized_entropy.mean(),
                ]
            )
        )
    return torch.stack(stats, dim=0)


def build_reinforced_selection_teacher_features(
    *,
    selection_teacher_logits: list[torch.Tensor],
    selection_teacher_labels: list[torch.Tensor],
    teacher_loss_matrix: torch.Tensor,
    teacher_temperature: float,
    skip_teacher_eos: bool,
) -> torch.Tensor:
    if not selection_teacher_logits or not selection_teacher_labels:
        raise ValueError("Reinforced teacher selection requires teacher logits and labels.")
    if len(selection_teacher_logits) != len(selection_teacher_labels):
        raise ValueError(
            "Teacher logits/labels batch counts must match for reinforced selection. "
            f"logits={len(selection_teacher_logits)}, labels={len(selection_teacher_labels)}"
        )

    teacher_features = []
    for teacher_index, (teacher_logits, teacher_labels) in enumerate(
        zip(selection_teacher_logits, selection_teacher_labels)
    ):
        descriptor_stats = compute_reinforced_teacher_descriptor_stats(
            teacher_logits=teacher_logits,
            teacher_labels=teacher_labels,
            teacher_temperature=teacher_temperature,
            skip_teacher_eos=skip_teacher_eos,
        ).to(device=teacher_loss_matrix.device, dtype=teacher_loss_matrix.dtype)
        teacher_features.append(
            torch.cat(
                [
                    # Each teacher contributes [KD loss, mean confidence, mean normalized entropy].
                    teacher_loss_matrix[:, teacher_index : teacher_index + 1],
                    descriptor_stats,
                ],
                dim=-1,
            )
        )
    return torch.stack(teacher_features, dim=1)


class ReinforcedTeacherSelectionPolicy(nn.Module):
    def __init__(self, model: nn.Module, num_teachers: int, teacher_feature_dim: int = 3):
        super().__init__()
        hidden_size, hook_module = Gate.resolve_gate_source(model)
        self.num_teachers = num_teachers
        self.teacher_feature_dim = teacher_feature_dim
        self.hidden_state = None
        self.state_normalizer = nn.LayerNorm(hidden_size + num_teachers * teacher_feature_dim)
        self.policy = nn.Linear(hidden_size + num_teachers * teacher_feature_dim, num_teachers)
        nn.init.xavier_uniform_(self.policy.weight)
        nn.init.zeros_(self.policy.bias)
        # Reuse the same pre-lm-head hidden state that the router sees.
        self.hook_handle = hook_module.register_forward_pre_hook(self.capture_hidden_state)

    def capture_hidden_state(self, module, args):
        del module
        if not args or not torch.is_tensor(args[0]):
            raise RuntimeError("Reinforced teacher selector hook did not receive the pre-lm-head hidden state.")
        if self.hidden_state is not None:
            raise RuntimeError("Reinforced teacher selector hidden state from the previous forward was not consumed.")
        self.hidden_state = args[0]

    def pool_hidden_state(
        self,
        *,
        student_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Pool the cached student hidden state per sample for the selector policy."""
        if self.hidden_state is None:
            raise RuntimeError("Reinforced teacher selector hidden state was not captured.")
        hidden_state = self.hidden_state
        self.hidden_state = None
        return masked_mean_pool_sequence(hidden_state, student_labels)

    def compute_policy_logits(
        self,
        *,
        teacher_features: torch.Tensor,
        student_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Bernoulli logits over teachers; input is teacher descriptors plus pooled student context, output is [batch, teacher] logits, and this exists to parameterize the policy."""
        pooled_hidden = self.pool_hidden_state(
            student_labels=student_labels,
        ).detach()
        # The policy state is student context plus all teacher descriptors flattened together.
        flattened_teacher_features = rearrange(teacher_features, "b teacher feat -> b (teacher feat)")
        policy_state = torch.cat([pooled_hidden, flattened_teacher_features], dim=-1)
        return self.policy(self.state_normalizer(policy_state))


def compute_reinforced_selection_state(
    *,
    selector: ReinforcedTeacherSelectionPolicy,
    teacher_loss_matrix: torch.Tensor,
    selection_teacher_logits: list[torch.Tensor],
    selection_teacher_labels: list[torch.Tensor],
    student_labels: torch.Tensor,
    student_ce_loss: torch.Tensor,
    teacher_temperature: float,
    skip_teacher_eos: bool,
    warmup_active: bool,
    reward_type: str,
    prev_reward_baseline: torch.Tensor | None,
    reward_ema_decay: float,
) -> dict[str, torch.Tensor | dict[str, float] | None]:
    """Compute one reinforced-selection step state; input is selector state plus per-teacher losses/logits, output is a dict of losses, weights, and metrics, and this exists to isolate REINFORCE bookkeeping from the trainer."""
    teacher_features = build_reinforced_selection_teacher_features(
        selection_teacher_logits=selection_teacher_logits,
        selection_teacher_labels=selection_teacher_labels,
        teacher_loss_matrix=teacher_loss_matrix,
        teacher_temperature=teacher_temperature,
        skip_teacher_eos=skip_teacher_eos,
    )
    policy_logits = selector.compute_policy_logits(
        teacher_features=teacher_features,
        student_labels=student_labels,
    )
    policy_probs = torch.sigmoid(policy_logits).clamp(
        min=torch.finfo(policy_logits.dtype).eps,
        max=1.0 - torch.finfo(policy_logits.dtype).eps,
    )
    # Independent Bernoulli policy per teacher: p_t = sigmoid(logit_t).

    if warmup_active:
        # Warmup keeps all teachers active and trains the selector to predict "on"
        # before sampling-based REINFORCE starts.
        selection_mask = torch.ones_like(policy_probs, dtype=torch.bool)
        policy_loss = F.binary_cross_entropy_with_logits(
            policy_logits,
            torch.ones_like(policy_logits),
        )
        reward = None
        next_reward_baseline = prev_reward_baseline
        fallback_rate = policy_probs.new_zeros(())
    else:
        if selector.training:
            selection_mask = torch.bernoulli(policy_probs).to(dtype=torch.bool)
        else:
            selection_mask = policy_probs >= 0.5
        empty_selection = ~selection_mask.any(dim=-1)
        if empty_selection.any():
            # Never allow an all-zero teacher set; fall back to the highest-probability teacher.
            fallback_indices = policy_probs[empty_selection].argmax(dim=-1, keepdim=True)
            selection_mask = selection_mask.clone()
            selection_mask[empty_selection] = False
            selection_mask[empty_selection].scatter_(dim=-1, index=fallback_indices, value=True)
        fallback_rate = empty_selection.float().mean()

        selection_mask_float = selection_mask.to(dtype=teacher_loss_matrix.dtype)
        if selector.training:
            # Bernoulli log-prob of the sampled multi-teacher action for REINFORCE.
            # log pi(a|s) = sum_t [a_t log p_t + (1-a_t) log(1-p_t)].
            log_prob = (
                selection_mask_float * policy_probs.log()
                + (1.0 - selection_mask_float) * (1.0 - policy_probs).log()
            ).sum(dim=-1).mean()
            # reward1: R = -CE
            # reward2: R = -CE - mean_selected_teacher_KD
            reward = -student_ce_loss.detach()
            if reward_type == "reward2":
                reward = reward - (
                    einsum(
                        teacher_loss_matrix,
                        selection_mask_float,
                        "batch teacher, batch teacher -> batch",
                    )
                    / selection_mask_float.sum(dim=-1).clamp(min=1.0)
                ).mean().detach()
            baseline_source = reward.detach().to(dtype=policy_logits.dtype, device=policy_logits.device)
            reward_baseline = (
                None
                if prev_reward_baseline is None
                else prev_reward_baseline.to(
                    dtype=baseline_source.dtype,
                    device=baseline_source.device,
                )
            )
            next_reward_baseline = (
                baseline_source
                if reward_baseline is None
                else reward_ema_decay * reward_baseline + (1.0 - reward_ema_decay) * baseline_source
            )
            # Advantage estimate: A = R - b.
            advantage = baseline_source if reward_baseline is None else baseline_source - reward_baseline
            # REINFORCE objective: L_policy = -A * log pi(a|s).
            policy_loss = -(advantage.detach() * log_prob)
        else:
            reward = None
            policy_loss = None
            next_reward_baseline = prev_reward_baseline

    # Normalize the selected teacher mask into mixture weights for the KD loss.
    # w_t = a_t / sum_j a_j, then L_KD = sum_t w_t * ell_t.
    selection_weights = selection_mask.to(dtype=teacher_loss_matrix.dtype)
    selection_weights = selection_weights / selection_weights.sum(dim=-1, keepdim=True).clamp(min=1.0)
    distillation_loss = einsum(
        teacher_loss_matrix,
        selection_weights,
        "batch teacher, batch teacher -> batch",
    ).mean()
    policy_entropy = (
        -(policy_probs * policy_probs.log() + (1.0 - policy_probs) * (1.0 - policy_probs).log())
        .sum(dim=-1)
        .mean()
    )

    metrics = {
        "reinforced_selection_entropy": policy_entropy.detach().float().item(),
        "reinforced_selection_fallback_rate": fallback_rate.detach().float().item(),
        "reinforced_selection_warmup_active": float(warmup_active),
    }
    metrics.update(
        {
            f"reinforced_selection_prob_{teacher_index}": prob.item()
            for teacher_index, prob in enumerate(policy_probs.detach().mean(dim=0))
        }
    )
    metrics.update(
        {
            f"reinforced_selection_select_rate_{teacher_index}": rate.item()
            for teacher_index, rate in enumerate(selection_weights.detach().mean(dim=0))
        }
    )
    if policy_loss is not None:
        metrics["reinforced_selection_policy_loss"] = policy_loss.detach().float().item()
    if reward is not None:
        metrics["reinforced_selection_reward"] = reward.detach().float().item()

    return {
        "distillation_loss": distillation_loss,
        "selection_weights": selection_weights,
        "policy_probs": policy_probs,
        "policy_loss": policy_loss,
        "reward": reward,
        "policy_entropy": policy_entropy,
        "fallback_rate": fallback_rate,
        "next_reward_baseline": next_reward_baseline,
        "metrics": metrics,
    }


__all__ = [
    "ReinforcedTeacherSelectionPolicy",
    "build_reinforced_selection_teacher_features",
    "compute_reinforced_selection_state",
]
