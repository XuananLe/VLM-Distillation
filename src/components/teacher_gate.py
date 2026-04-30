import torch
import torch.nn as nn
import torch.nn.functional as F

from src.components.pooling import masked_mean_pool_sequence

SMOLVLM_MODEL_TYPES = {"smolvlm", "smolvlm2", "idefics3"}


class DeepRouter(nn.Module):
    def __init__(self, input_size: int, num_experts: int):
        super().__init__()
        self.hidden_size = self.resolve_hidden_size(num_experts)
        self.normalizer = nn.LayerNorm(input_size)
        self.up_proj = nn.Linear(input_size, self.hidden_size * 2)
        self.down_proj = nn.Linear(self.hidden_size, num_experts)

        nn.init.xavier_uniform_(self.up_proj.weight) # inplace
        nn.init.zeros_(self.up_proj.bias)
        nn.init.xavier_uniform_(self.down_proj.weight)
        nn.init.zeros_(self.down_proj.bias)

    @staticmethod
    def resolve_hidden_size(num_experts: int) -> int:
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
        hidden = self.normalizer(inputs)
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
        """Initialize routing state and register the pre-lm-head hook; input is the student model plus routing hyperparameters, output is an initialized gate, and this exists to keep routing setup out of trainer code."""
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
            raise ValueError(
                "Teacher gate only supports SmolVLM-style students, got "
                f"model_type={model_type!r}."
            )
        if not isinstance(lm_head, nn.Linear):
            raise ValueError(
                "Teacher gate expects SmolVLM `lm_head` to be an nn.Linear module."
            )
        if lm_head.in_features != hidden_size:
            raise ValueError(
                "SmolVLM lm_head input size does not match text hidden size: "
                f"{lm_head.in_features} != {hidden_size}."
            )
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
        return masked_mean_pool_sequence(
            tensor,
            labels=labels,
            attention_mask=attention_mask,
        )

    def pool_features(
        self,
        *,
        student_labels: torch.Tensor,
        student_attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Pool the cached student hidden state over supervised answer tokens for routing."""
        if self.hidden_state is None:
            raise RuntimeError("Teacher gate hidden state was not captured during the student forward pass.")
        pooled_features = self.pool_tensor(
            self.hidden_state,
            labels=student_labels,
            attention_mask=student_attention_mask,
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
        student_labels: torch.Tensor,
        student_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pooled_features = self.pool_features(
            student_labels=student_labels,
            student_attention_mask=student_attention_mask,
        )
        self.prepare_router_module(pooled_features)
        return self.router(pooled_features)

    def forward(
        self,
        *,
        student_labels: torch.Tensor,
        student_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return soft routing weights over teachers; input is batch masks, output is [batch, teacher] probabilities, and this exists for trainer code that wants ready-to-use gate weights."""
        router_logits = self.compute_router_logits(
            student_labels=student_labels,
            student_attention_mask=student_attention_mask,
        )
        return torch.softmax(router_logits, dim=-1)

__all__ = [
    "DeepRouter",
    "Gate",
]
