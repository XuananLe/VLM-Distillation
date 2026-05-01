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
    reference_grads: tuple[torch.Tensor | None, ...],
    candidate_grads: tuple[torch.Tensor | None, ...],
    *,
    loss: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    dot = loss.new_zeros((), dtype=torch.float32)
    reference_norm = loss.new_zeros((), dtype=torch.float32)
    candidate_norm = loss.new_zeros((), dtype=torch.float32)

    for reference_grad, candidate_grad in zip(reference_grads, candidate_grads):
        if reference_grad is not None:
            reference = reference_grad.float()
            reference_norm = reference_norm + reference.square().sum().to(reference_norm.device)
        if candidate_grad is not None:
            candidate = candidate_grad.float()
            candidate_norm = candidate_norm + candidate.square().sum().to(candidate_norm.device)
        if reference_grad is not None and candidate_grad is not None:
            dot = dot + (reference * candidate).sum().to(dot.device)

    denominator = reference_norm.sqrt() * candidate_norm.sqrt()
    return dot / denominator.clamp_min(eps)


__all__ = [
    "compute_parameter_grads",
    "parameter_gradient_cosine",
    "trainable_parameters",
]
