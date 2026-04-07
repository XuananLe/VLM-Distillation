import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.components.teacher_gate import Gate


def _compute_reinforced_teacher_descriptor_stats(
    *,
    teacher_logits: torch.Tensor,
    teacher_labels: torch.Tensor,
    teacher_temperature: float,
    skip_teacher_eos: bool,
) -> torch.Tensor:
    stats = []
    for sample_logits, sample_labels in zip(teacher_logits, teacher_labels):
        positions = sample_labels.ne(-100).nonzero(as_tuple=False).squeeze(-1)
        if skip_teacher_eos and positions.numel() > 0:
            positions = positions[:-1]
        if positions.numel() == 0:
            stats.append(sample_logits.new_zeros((2,), dtype=torch.float32))
            continue

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
    teacher_logits_batches: list[torch.Tensor],
    teacher_label_batches: list[torch.Tensor],
    teacher_loss_matrix: torch.Tensor,
    teacher_temperature: float,
    skip_teacher_eos: bool,
) -> torch.Tensor:
    if not teacher_logits_batches or not teacher_label_batches:
        raise ValueError("Reinforced teacher selection requires teacher logits and labels.")
    if len(teacher_logits_batches) != len(teacher_label_batches):
        raise ValueError(
            "Teacher logits/labels batch counts must match for reinforced selection. "
            f"logits={len(teacher_logits_batches)}, labels={len(teacher_label_batches)}"
        )

    teacher_features = []
    for teacher_index, (teacher_logits, teacher_labels) in enumerate(
        zip(teacher_logits_batches, teacher_label_batches)
    ):
        descriptor_stats = _compute_reinforced_teacher_descriptor_stats(
            teacher_logits=teacher_logits,
            teacher_labels=teacher_labels,
            teacher_temperature=teacher_temperature,
            skip_teacher_eos=skip_teacher_eos,
        ).to(device=teacher_loss_matrix.device, dtype=teacher_loss_matrix.dtype)
        teacher_features.append(
            torch.cat(
                [
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
        self.hook_handle = hook_module.register_forward_pre_hook(self.capture_hidden_state)

    def capture_hidden_state(self, _module, args):
        if args:
            self.hidden_state = args[0]

    def reset(self) -> None:
        self.hidden_state = None

    def pool_hidden_state(
        self,
        *,
        labels: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.hidden_state is None:
            raise RuntimeError("Reinforced teacher selector hidden state was not captured.")
        hidden_state = self.hidden_state
        self.hidden_state = None

        label_mask = labels.ne(-100)
        fallback_mask = (
            attention_mask.bool()
            if attention_mask is not None
            else torch.ones_like(label_mask, dtype=torch.bool)
        )
        pool_mask = label_mask
        missing_supervised = ~pool_mask.any(dim=1)
        if missing_supervised.any():
            pool_mask = pool_mask.clone()
            pool_mask[missing_supervised] = fallback_mask[missing_supervised]

        pool_mask = pool_mask.to(dtype=hidden_state.dtype)
        pooled_hidden = (hidden_state * pool_mask.unsqueeze(-1)).sum(dim=1)
        return pooled_hidden / pool_mask.sum(dim=1, keepdim=True).clamp(min=1.0)

    def compute_policy_logits(
        self,
        *,
        teacher_features: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        pooled_hidden = self.pool_hidden_state(
            labels=labels,
            attention_mask=attention_mask,
        ).detach()
        state = torch.cat([pooled_hidden, teacher_features.flatten(start_dim=1)], dim=-1)
        return self.policy(self.state_normalizer(state))


def compute_reinforced_selection_state(
    *,
    selector: ReinforcedTeacherSelectionPolicy,
    teacher_loss_matrix: torch.Tensor,
    teacher_logits_batches: list[torch.Tensor],
    teacher_label_batches: list[torch.Tensor],
    labels: torch.Tensor,
    attention_mask: torch.Tensor | None,
    ce_loss: torch.Tensor,
    teacher_temperature: float,
    skip_teacher_eos: bool,
    warmup_active: bool,
    reward_type: str,
    prev_reward_baseline: torch.Tensor | None,
    reward_ema_decay: float,
) -> dict[str, torch.Tensor | dict[str, float] | None]:
    teacher_features = build_reinforced_selection_teacher_features(
        teacher_logits_batches=teacher_logits_batches,
        teacher_label_batches=teacher_label_batches,
        teacher_loss_matrix=teacher_loss_matrix,
        teacher_temperature=teacher_temperature,
        skip_teacher_eos=skip_teacher_eos,
    )
    policy_logits = selector.compute_policy_logits(
        teacher_features=teacher_features,
        labels=labels,
        attention_mask=attention_mask,
    )
    policy_probs = torch.sigmoid(policy_logits).clamp(
        min=torch.finfo(policy_logits.dtype).eps,
        max=1.0 - torch.finfo(policy_logits.dtype).eps,
    )

    if warmup_active:
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
            fallback_indices = policy_probs[empty_selection].argmax(dim=-1, keepdim=True)
            selection_mask = selection_mask.clone()
            selection_mask[empty_selection] = False
            selection_mask[empty_selection].scatter_(dim=-1, index=fallback_indices, value=True)
        fallback_rate = empty_selection.float().mean()

        selection_mask_float = selection_mask.to(dtype=teacher_loss_matrix.dtype)
        if selector.training:
            log_prob = (
                selection_mask_float * policy_probs.log()
                + (1.0 - selection_mask_float) * (1.0 - policy_probs).log()
            ).sum(dim=-1).mean()
            reward = -ce_loss.detach()
            if reward_type == "reward2":
                reward = reward - (
                    (teacher_loss_matrix * selection_mask_float).sum(dim=-1)
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
            advantage = baseline_source if reward_baseline is None else baseline_source - reward_baseline
            policy_loss = -(advantage.detach() * log_prob)
        else:
            reward = None
            policy_loss = None
            next_reward_baseline = prev_reward_baseline

    selection_weights = selection_mask.to(dtype=teacher_loss_matrix.dtype)
    selection_weights = selection_weights / selection_weights.sum(dim=-1, keepdim=True).clamp(min=1.0)
    distillation_loss = (teacher_loss_matrix * selection_weights).sum(dim=-1).mean()
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
