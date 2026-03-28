import torch
import torch.nn as nn


SUPPORTED_SMOLVLM_HIDDEN_SIZES = {
    "huggingfacetb/smolvlm-500m-instruct": 960,
    "huggingfacetb/smolvlm-256m-instruct": 576,
}


class Gate(nn.Module):
    def __init__(self, model: nn.Module, num_teachers: int, bias_update_rate: float = 0.0):
        super().__init__()
        hidden_size, hook_module = self.resolve_gate_source(model)
        self.normalizer = nn.LayerNorm(hidden_size)
        self.router = nn.Linear(hidden_size, out_features=num_teachers)
        nn.init.xavier_uniform_(self.router.weight)
        nn.init.xavier_uniform_(self.router.bias)

        self.bias_update_rate = bias_update_rate
        self.hidden_state = None
        self.register_buffer("expert_bias", torch.zeros(num_teachers))
        self.hook_handle = hook_module.register_forward_pre_hook(self.capture_hidden_state)


    @staticmethod
    def resolve_gate_source(model: nn.Module) -> tuple[int, nn.Module]:
        smolvlm_source = Gate.resolve_smolvlm_gate_source(model)
        if smolvlm_source is not None:
            return smolvlm_source

        lm_head = model.get_output_embeddings()
        if lm_head is None:
            raise ValueError(
                "Gate currently supports SmolVLM-500M, SmolVLM-256M, "
                "or models exposing an output head with an explicit input dimension."
            )
        if hasattr(lm_head, "in_features"):
            return lm_head.in_features, lm_head
        if hasattr(lm_head, "weight") and lm_head.weight.ndim == 2:
            return lm_head.weight.shape[1], lm_head
        raise ValueError("Could not resolve gate source from the model output head.")

    @staticmethod
    def resolve_smolvlm_gate_source(model: nn.Module) -> tuple[int, nn.Module] | None:
        config = getattr(model, "config", None)
        model_name = str(getattr(config, "_name_or_path", "")).lower().strip()
        if model_name not in SUPPORTED_SMOLVLM_HIDDEN_SIZES:
            return None

        model_type = getattr(config, "model_type", None)
        architectures = getattr(config, "architectures", None) or []
        if model_type != "idefics3" or "Idefics3ForConditionalGeneration" not in architectures:
            raise ValueError(
                f"Expected {model_name} to be an Idefics3ForConditionalGeneration checkpoint, "
                f"got model_type={model_type!r}, architectures={architectures!r}."
            )

        text_config = getattr(config, "text_config", None)
        hidden_size = getattr(text_config, "hidden_size", None)
        expected_hidden_size = SUPPORTED_SMOLVLM_HIDDEN_SIZES[model_name]
        if hidden_size != expected_hidden_size:
            raise ValueError(
                f"Expected {model_name} text hidden size {expected_hidden_size}, got {hidden_size}."
            )

        lm_head = getattr(model, "lm_head", None)
        if lm_head is None:
            raise ValueError(f"Expected {model_name} to expose `lm_head` for learned teacher gating.")
        return hidden_size, lm_head

    def capture_hidden_state(self, _module, args):
        if args:
            self.hidden_state = args[0]

    def reset(self) -> None:
        self.hidden_state = None

    def pool_features(
        self,
        *,
        labels: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.hidden_state is None:
            raise RuntimeError("Teacher gate hidden state was not captured during the student forward pass.")

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

        gate_mask = gate_mask.to(dtype=self.hidden_state.dtype)
        pooled_features = (self.hidden_state * gate_mask.unsqueeze(-1)).sum(dim=1)
        pooled_features = pooled_features / gate_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        self.hidden_state = None
        return pooled_features

    def compute_router_logits(
        self,
        *,
        labels: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pooled_features = self.pool_features(labels=labels, attention_mask=attention_mask)
        return self.router(self.normalizer(pooled_features))

    def apply_expert_bias(self, router_logits: torch.Tensor) -> torch.Tensor:
        return router_logits + self.expert_bias.to(device=router_logits.device, dtype=router_logits.dtype)

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
        return torch.softmax(
            self.compute_router_logits(labels=labels, attention_mask=attention_mask),
            dim=-1,
        )
