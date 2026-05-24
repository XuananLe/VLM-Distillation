import torch

def trainable_parameters(model) -> tuple[torch.nn.Parameter, ...]:
    parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    if not parameters:
        raise ValueError("GRACE parameter gradients require at least one trainable student parameter.")
    return parameters



# d loss / d w_i
# https://gist.github.com/Lyken17/91b81526a8245a028d4f85ccc9191884
def compute_parameter_grads(
    *,
    loss: torch.Tensor,
    parameters: tuple[torch.nn.Parameter, ...],
) -> tuple[torch.Tensor | None, ...]:
    return tuple(
        None if grad is None else grad.detach()
        for grad in torch.autograd.grad(
            loss,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
    )

# cosine = dot(CE_grad, KD_grad) / (||CE_grad|| * ||KD_grad||)
# https://www.vegardstikbakke.com/python-keyword-only/
def parameter_gradient_cosine(
    ce_grads: tuple[torch.Tensor | None, ...],
    kd_grads: tuple[torch.Tensor | None, ...],
    *,
    loss: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    dot = loss.new_zeros((), dtype=torch.float32)
    ce_norm = loss.new_zeros((), dtype=torch.float32)
    kd_norm = loss.new_zeros((), dtype=torch.float32)

    for ce_grad, kd_grad in zip(ce_grads, kd_grads):
        if ce_grad is not None:
            ce_grad = ce_grad.float()
            ce_norm = ce_norm + ce_grad.square().sum().to(ce_norm.device)
        if kd_grad is not None:
            kd_grad = kd_grad.float()
            kd_norm = kd_norm + kd_grad.square().sum().to(kd_norm.device)
        if ce_grad is not None and kd_grad is not None:
            dot = dot + (ce_grad * kd_grad).sum().to(dot.device)

    denominator = ce_norm.sqrt() * kd_norm.sqrt()
    return dot / denominator.clamp_min(eps)


__all__ = [
    "compute_parameter_grads",
    "parameter_gradient_cosine",
    "trainable_parameters",
]
