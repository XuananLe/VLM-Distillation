import torch
import torch.nn as nn
import torch.nn.functional as F

from src.components.pooling import masked_mean_pool_sequence

SMOLVLM_MODEL_TYPES = {"smolvlm", "smolvlm2", "idefics3"}

# student hidden size D
# teacher count N
# router hidden size H
class DeepRouter(nn.Module):
    def __init__(self, input_size: int, num_experts: int):
        super().__init__()
        self.hidden_size = self.resolve_hidden_size(num_experts)
        self.normalizer = nn.LayerNorm(input_size) # 2 * D
        self.up_proj = nn.Linear(input_size, self.hidden_size * 2)
        self.down_proj = nn.Linear(self.hidden_size, num_experts)
    
        nn.init.xavier_uniform_(self.up_proj.weight)  # inplace
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
        # inputs [batch_size, hidden_features]
        hidden = self.normalizer(inputs) # [batch_size, 2 * hidden_features]
        # Two-projection SwiGLU-style router block before the final expert logits.
        gate, value = self.up_proj(hidden).chunk(2, dim=-1)
        hidden = F.silu(gate) * value 
        return self.down_proj(hidden)


class Gate(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        num_teachers: int,
    ):
        #  [batch, seq_len, hidden_dim] -> [batch, num_teachers]
        super().__init__()
        hidden_size, hook_module = self.resolve_gate_source(model)
        self.router = DeepRouter(hidden_size, num_teachers)

        self.hidden_state = None
        # Capture the hidden state immediately before lm_head so routing uses the
        # same student context as the token prediction head.
        self.hook_handle = hook_module.register_forward_pre_hook(self.capture_hidden_state)

    @staticmethod
    def resolve_gate_source(model: nn.Module) -> tuple[int, nn.Module]:
        model_type = model.config.model_type
        hidden_size = int(model.config.text_config.hidden_size)
        lm_head = model.lm_head

        if model_type not in SMOLVLM_MODEL_TYPES:
            raise ValueError(f"Teacher gate only supports SmolVLM-style students, got model_type={model_type!r}.")
        if not isinstance(lm_head, nn.Linear):
            raise ValueError("Teacher gate expects SmolVLM `lm_head` to be an nn.Linear module.")
        if lm_head.in_features != hidden_size:
            raise ValueError(
                f"SmolVLM lm_head input size does not match text hidden size: {lm_head.in_features} != {hidden_size}."
            )
        return int(hidden_size), lm_head

    def capture_hidden_state(self, _module, args):
        if not args or not torch.is_tensor(args[0]):
            raise RuntimeError("Teacher gate hook did not receive the pre-lm-head hidden state.")
        if self.hidden_state is not None:
            raise RuntimeError("Teacher gate hidden state from the previous forward was not consumed.")
        self.hidden_state = args[0]

    def pool_features(
        self,
        *,
        student_labels: torch.Tensor,
    ) -> torch.Tensor:
        if self.hidden_state is None:
            raise RuntimeError("Teacher gate hidden state was not captured during the student forward pass.")
        hidden_state = self.hidden_state
        self.hidden_state = None
        return masked_mean_pool_sequence(hidden_state, student_labels)

    def prepare_router_module(self, reference: torch.Tensor) -> None:
        target_dtype = reference.dtype if reference.is_floating_point() else None
        router_param = next(self.router.parameters(), None)
        if router_param is None:
            return

        if router_param.device != reference.device or (target_dtype is not None and router_param.dtype != target_dtype):
            self.router.to(device=reference.device, dtype=target_dtype)

    def compute_router_logits(
        self,
        *,
        student_labels: torch.Tensor,
    ) -> torch.Tensor:
        pooled_features = self.pool_features(
            student_labels=student_labels,
        )
        self.prepare_router_module(pooled_features)
        return self.router(pooled_features)

    def forward(
        self,
        *,
        student_labels: torch.Tensor,
    ) -> torch.Tensor:
        router_logits = self.compute_router_logits(
            student_labels=student_labels,
        )
        return torch.softmax(router_logits, dim=-1)


__all__ = [
    "DeepRouter",
    "Gate",
]
