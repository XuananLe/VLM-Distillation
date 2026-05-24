import torch
import torch.nn as nn
import torch.nn.functional as F

from src.components.pooling import masked_mean_pool_sequence

ROUTER_HIDDEN_SIZE = 64


class DeepRouter(nn.Module):
    def __init__(self, input_size: int, num_experts: int):
        super().__init__()
        self.normalizer = nn.LayerNorm(input_size)
        self.up_proj = nn.Linear(input_size, ROUTER_HIDDEN_SIZE * 2)
        self.down_proj = nn.Linear(ROUTER_HIDDEN_SIZE, num_experts)

        nn.init.xavier_uniform_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)
        nn.init.xavier_uniform_(self.down_proj.weight)
        nn.init.zeros_(self.down_proj.bias)

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
    ):
        super().__init__()
        hidden_size = int(model.config.text_config.hidden_size)
        lm_head = model.lm_head
        self.router = DeepRouter(hidden_size, num_teachers).to(device=lm_head.weight.device, dtype=lm_head.weight.dtype)
        self.hidden_state: torch.Tensor | None = None
        self.hook_handle = lm_head.register_forward_pre_hook(self.capture_hidden_state)

    def capture_hidden_state(self, _module, args):
        if not args or not torch.is_tensor(args[0]):
            raise RuntimeError("Teacher gate hook did not receive the pre-lm-head hidden state.")
        if self.hidden_state is not None:
            raise RuntimeError("Teacher gate hidden state from the previous forward was not consumed.")
        self.hidden_state = args[0]

    def compute_router_logits(
        self,
        *,
        student_labels: torch.Tensor,
    ) -> torch.Tensor:
        if self.hidden_state is None:
            raise RuntimeError("Teacher gate hidden state was not captured during the student forward pass.")
        hidden_state = self.hidden_state
        self.hidden_state = None
        pooled_features = masked_mean_pool_sequence(hidden_state, student_labels)
        return self.router(pooled_features)


__all__ = [
    "DeepRouter",
    "Gate",
]
