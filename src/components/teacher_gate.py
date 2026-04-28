import torch
import torch.nn as nn
import torch.nn.functional as F

from src.components.pooling import masked_mean_pool_sequence


class DeepRouter(nn.Module):
    """Small MLP router over pooled student features; it takes [batch, hidden] inputs, returns [batch, teacher] logits, and exists to learn non-uniform teacher routing."""
    def __init__(self, input_size: int, num_experts: int):
        """Initialize the router block; input is pooled feature size and teacher count, output is an initialized module, and this exists to keep router architecture local to one class."""
        super().__init__()
        self.hidden_size = self.resolve_hidden_size(num_experts)
        self.normalizer = nn.LayerNorm(input_size)
        self.up_proj = nn.Linear(input_size, self.hidden_size * 2)
        self.down_proj = nn.Linear(self.hidden_size, num_experts)

        nn.init.xavier_uniform_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)
        nn.init.xavier_uniform_(self.down_proj.weight)
        nn.init.zeros_(self.down_proj.bias)

    @staticmethod
    def resolve_hidden_size(num_experts: int) -> int:
        """Choose router width from expert count; input is number of experts, output is an int hidden size, and this exists as a lightweight capacity heuristic."""
        # This size ladder is heuristic: small teacher sets do not need a wide router,
        # but larger mixtures get a wider hidden layer to avoid a severe bottleneck.
        if num_experts <= 4:
            return 64
        if num_experts <= 16:
            return 128
        if num_experts <= 32:
            return 256
        return 512

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Map pooled features to teacher logits; input is [batch, hidden], output is [batch, teacher], and this exists as the learned routing-score block."""
        hidden = self.normalizer(inputs)
        # Two-projection SwiGLU-style router block before the final expert logits.
        gate, value = self.up_proj(hidden).chunk(2, dim=-1)
        hidden = F.silu(gate) * value
        return self.down_proj(hidden)


class Gate(nn.Module):
    """Teacher-routing module attached to the student; it reuses the hidden state before lm_head and exists to score teachers without another student forward."""
    def __init__(
        self,
        model: nn.Module,
        num_teachers: int,
        router_temperature: float = 1.0,
        router_noise_std: float = 0.0,
    ):
        """Initialize routing state and register the pre-lm-head hook; input is the student model plus routing hyperparameters, output is an initialized gate, and this exists to keep routing setup out of trainer code."""
        super().__init__()
        hidden_size, hook_module = self.resolve_gate_source(model)
        self.router = DeepRouter(hidden_size, num_teachers)

        self.router_temperature = router_temperature
        self.router_noise_std = router_noise_std
        self.hidden_state = None
        # Capture the hidden state immediately before lm_head so routing uses the
        # same student context as the token prediction head.
        self.hook_handle = hook_module.register_forward_pre_hook(self.capture_hidden_state)

    @staticmethod
    def resolve_gate_source(model: nn.Module) -> tuple[int, nn.Module]:
        """Find the hidden size and hook source for routing; input is the student model, output is (hidden_size, lm_head-like module), and this exists so routing stays model-family agnostic."""
        lm_head = getattr(model, "lm_head", None)
        if lm_head is None:
            raise ValueError("Teacher gate requires the model to expose `lm_head`.")

        config = getattr(model, "config", None)
        text_config = getattr(config, "text_config", None) if config is not None else None
        hidden_size = (
            getattr(text_config, "hidden_size", None)
            or getattr(config, "hidden_size", None)
            or getattr(lm_head, "in_features", None)
        )
        if hidden_size is None:
            raise ValueError("Teacher gate could not infer the student hidden size from config or lm_head.")
        return int(hidden_size), lm_head

    def capture_hidden_state(self, _module, args):
        """Cache the hidden state seen by lm_head; input is hook args, output is None, and this exists so routing can reuse student context after the main forward."""
        if args:
            self.hidden_state = args[0]

    def reset(self) -> None:
        """Clear the cached hidden state; input/output are None, and this exists to avoid reusing stale routing context across steps."""
        self.hidden_state = None

    def pool_tensor(
        self,
        tensor: torch.Tensor,
        *,
        labels: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Pool a token sequence into one vector per sample; input is [batch, tokens, hidden] plus masks, output is [batch, hidden], and this exists to share pooling behavior with the selector."""
        return masked_mean_pool_sequence(
            tensor,
            labels=labels,
            attention_mask=attention_mask,
        )

    def pool_features(
        self,
        *,
        labels: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Pool the cached student hidden state for routing; input is labels plus optional attention mask, output is [batch, hidden], and this exists to turn token context into router input."""
        if self.hidden_state is None:
            raise RuntimeError("Teacher gate hidden state was not captured during the student forward pass.")
        pooled_features = self.pool_tensor(
            self.hidden_state,
            labels=labels,
            attention_mask=attention_mask,
        )
        self.hidden_state = None
        return pooled_features

    def prepare_router_module(self, reference: torch.Tensor) -> None:
        """Move the router to the reference device/dtype; input is a reference tensor, output is None, and this exists because the gate is auxiliary to the base model."""
        target_dtype = reference.dtype if reference.is_floating_point() else None
        router_param = next(self.router.parameters(), None)
        if router_param is None:
            return

        if router_param.device != reference.device or (
            target_dtype is not None and router_param.dtype != target_dtype
        ):
            self.router.to(device=reference.device, dtype=target_dtype)

    def compute_router_logits(
        self,
        *,
        labels: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute raw router logits over teachers; input is batch masks, output is [batch, teacher] logits, and this exists as the main routing-score path."""
        pooled_features = self.pool_features(labels=labels, attention_mask=attention_mask)
        self.prepare_router_module(pooled_features)
        return self.router(pooled_features)

    def prepare_routing_scores(self, router_logits: torch.Tensor) -> torch.Tensor:
        """Turn raw logits into routing scores; input is router logits, output is temperature/noise-adjusted scores, and this exists to separate scoring policy from final softmax."""
        routing_scores = router_logits
        if self.training and self.router_noise_std > 0.0:
            routing_scores = routing_scores + torch.randn_like(routing_scores) * self.router_noise_std
        return routing_scores / self.router_temperature

    def forward(
        self,
        *,
        labels: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return soft routing weights over teachers; input is batch masks, output is [batch, teacher] probabilities, and this exists for trainer code that wants ready-to-use gate weights."""
        router_logits = self.compute_router_logits(labels=labels, attention_mask=attention_mask)
        return torch.softmax(self.prepare_routing_scores(router_logits), dim=-1)


def maybe_create_teacher_gate(
    *,
    model,
    num_teachers: int,
    teacher_weighting_strategy: str,
    teacher_gate_temperature: float,
    teacher_gate_noise_std: float,
):
    """Build and attach the routing gate only for multi-teacher routing runs."""
    if num_teachers <= 1 or teacher_weighting_strategy != "routing":
        return None

    # The gate is attached to the student model so its forward hook can reuse the
    # same hidden state captured immediately before the student's lm_head.
    teacher_gate = Gate(
        model,
        num_teachers,
        router_temperature=teacher_gate_temperature,
        router_noise_std=teacher_gate_noise_std,
    )
    model.teacher_gate = teacher_gate
    return teacher_gate


__all__ = [
    "DeepRouter",
    "Gate",
    "maybe_create_teacher_gate",
]
