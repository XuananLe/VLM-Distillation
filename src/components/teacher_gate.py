import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce


class DeepRouter(nn.Module):
    def __init__(self, input_size: int, num_experts: int):
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
        if num_experts <= 4:
            return 64
        if num_experts <= 16:
            return 128
        if num_experts <= 32:
            return 256
        return 512

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.normalizer(inputs)
        gate, value = self.up_proj(hidden).chunk(2, dim=-1)
        hidden = F.silu(gate) * value
        return self.down_proj(hidden)


class Gate(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        num_teachers: int,
        bias_update_rate: float = 0.0,
        router_temperature: float = 1.0,
        router_noise_std: float = 0.0,
    ):
        super().__init__()
        hidden_size, hook_module = self.resolve_gate_source(model)
        self.router = DeepRouter(hidden_size, num_teachers)

        self.bias_update_rate = bias_update_rate
        self.router_temperature = router_temperature
        self.router_noise_std = router_noise_std
        self.hidden_state = None
        self.register_buffer("expert_bias", torch.zeros(num_teachers))
        self.hook_handle = hook_module.register_forward_pre_hook(self.capture_hidden_state)


    @staticmethod
    def resolve_gate_source(model: nn.Module) -> tuple[int, nn.Module]:
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
        if args:
            self.hidden_state = args[0]

    def reset(self) -> None:
        self.hidden_state = None

    def pool_tensor(
        self,
        tensor: torch.Tensor,
        *,
        labels: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        label_mask = labels.ne(other=-100)
        if attention_mask is not None:
            fallback_mask = attention_mask.bool()
        else:
            fallback_mask = torch.ones_like(label_mask, dtype=torch.bool)

        gate_mask = label_mask
        missing_supervised = ~gate_mask.any(dim=1)
        if missing_supervised.any():
            gate_mask = gate_mask.clone()
            gate_mask[missing_supervised] = fallback_mask[missing_supervised]

        gate_mask = gate_mask.to(dtype=tensor.dtype)
        masked_tensor = tensor * rearrange(gate_mask, "b t -> b t 1")
        pooled_tensor = reduce(masked_tensor, "b t d -> b d", "sum")
        pooled_denominator = reduce(gate_mask, "b t -> b 1", "sum").clamp(min=1.0)
        return pooled_tensor / pooled_denominator

    def pool_features(
        self,
        *,
        labels: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
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
        pooled_features = self.pool_features(labels=labels, attention_mask=attention_mask)
        self.prepare_router_module(pooled_features)
        return self.router(pooled_features)

    def apply_expert_bias(self, router_logits: torch.Tensor) -> torch.Tensor:
        return router_logits + self.expert_bias.to(device=router_logits.device, dtype=router_logits.dtype)

    def prepare_routing_scores(self, router_logits: torch.Tensor) -> torch.Tensor:
        routing_scores = self.apply_expert_bias(router_logits)
        if self.training and self.router_noise_std > 0.0:
            routing_scores = routing_scores + torch.randn_like(routing_scores) * self.router_noise_std
        return routing_scores / self.router_temperature

    @torch.no_grad()
    def update_expert_bias(self, expert_load: torch.Tensor) -> None:
        if self.bias_update_rate <= 0.0:
            return
        violation = expert_load - expert_load.mean()
        self.expert_bias.sub_(
            self.bias_update_rate * violation.to(device=self.expert_bias.device, dtype=self.expert_bias.dtype)
        )

    def forward(
        self,
        *,
        labels: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        router_logits = self.compute_router_logits(labels=labels, attention_mask=attention_mask)
        return torch.softmax(self.prepare_routing_scores(router_logits), dim=-1)
