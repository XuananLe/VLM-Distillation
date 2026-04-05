import torch


def _dot(vec_a: torch.Tensor, vec_b: torch.Tensor) -> torch.Tensor:
    return torch.dot(vec_a.float(), vec_b.float())


def _min_norm_element_from_two(
    v1v1: torch.Tensor,
    v1v2: torch.Tensor,
    v2v2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if v1v2 >= v1v1:
        gamma = v1v1.new_tensor(0.999)
        return gamma, v1v1
    if v1v2 >= v2v2:
        gamma = v1v1.new_tensor(0.001)
        return gamma, v2v2
    gamma = -((v1v2 - v2v2) / (v1v1 + v2v2 - 2.0 * v1v2).clamp_min(1e-12))
    cost = v2v2 + gamma * (v1v2 - v2v2)
    return gamma, cost


def pcgrad_weights(
    ce_grad: torch.Tensor,
    kd_grad: torch.Tensor,
) -> torch.Tensor:
    ce_sq = _dot(ce_grad, ce_grad).clamp_min(1e-12)
    kd_sq = _dot(kd_grad, kd_grad).clamp_min(1e-12)
    cross = _dot(ce_grad, kd_grad)

    ce_weight = ce_grad.new_tensor(1.0)
    kd_weight = kd_grad.new_tensor(1.0)
    if cross < 0:
        ce_weight = ce_weight - cross / ce_sq
        kd_weight = kd_weight - cross / kd_sq
    return torch.stack([ce_weight, kd_weight]).to(dtype=ce_grad.dtype, device=ce_grad.device)


def mgda_weights(
    ce_grad: torch.Tensor,
    kd_grad: torch.Tensor,
) -> torch.Tensor:
    gamma, _ = _min_norm_element_from_two(
        _dot(ce_grad, ce_grad),
        _dot(ce_grad, kd_grad),
        _dot(kd_grad, kd_grad),
    )
    return torch.stack([gamma, 1.0 - gamma]).to(dtype=ce_grad.dtype, device=ce_grad.device)


def cagrad_weights(
    ce_grad: torch.Tensor,
    kd_grad: torch.Tensor,
    *,
    cagrad_c: float = 0.5,
    grid_steps: int = 257,
) -> torch.Tensor:
    device = ce_grad.device
    dtype = ce_grad.dtype
    grads = torch.stack([ce_grad.float(), kd_grad.float()], dim=0)
    g0 = grads.mean(dim=0)
    if g0.norm() <= torch.finfo(g0.dtype).eps:
        return grads.new_tensor([0.5, 0.5]).to(dtype=dtype, device=device)

    gram = grads @ grads.transpose(0, 1)
    midpoint = grads.new_tensor([0.5, 0.5])
    c_value = float(cagrad_c) * g0.norm().item()
    xs = torch.linspace(0.0, 1.0, steps=max(int(grid_steps), 2), device=grads.device)
    candidates = torch.stack([xs, 1.0 - xs], dim=-1)

    left_term = candidates @ gram @ midpoint
    quad_term = torch.sum((candidates @ gram) * candidates, dim=-1).clamp_min(1e-8).sqrt()
    objective = left_term + c_value * quad_term
    best_index = torch.argmin(objective)
    best_weights = candidates[best_index]

    gw = (best_weights.view(-1, 1) * grads).sum(dim=0)
    gw_norm = gw.norm().clamp_min(1e-8)
    lambda_value = c_value / gw_norm
    combined_weights = (midpoint + lambda_value * best_weights) / (1.0 + lambda_value)
    return combined_weights.to(dtype=dtype, device=device)


def resolve_objective_conflict_weights(
    *,
    strategy: str,
    ce_grad: torch.Tensor,
    kd_grad: torch.Tensor,
    cagrad_c: float = 0.5,
    cagrad_grid_steps: int = 257,
) -> tuple[torch.Tensor, torch.Tensor]:
    if kd_grad.norm() <= torch.finfo(kd_grad.dtype).eps:
        weights = torch.tensor([1.0, 0.0], device=ce_grad.device, dtype=ce_grad.dtype)
    elif ce_grad.norm() <= torch.finfo(ce_grad.dtype).eps:
        weights = torch.tensor([0.0, 1.0], device=ce_grad.device, dtype=ce_grad.dtype)
    elif strategy == "pcgrad":
        weights = pcgrad_weights(ce_grad, kd_grad)
    elif strategy == "cagrad":
        weights = cagrad_weights(
            ce_grad,
            kd_grad,
            cagrad_c=cagrad_c,
            grid_steps=cagrad_grid_steps,
        )
    elif strategy == "mgda":
        weights = mgda_weights(ce_grad, kd_grad)
    elif strategy == "fixed":
        weights = torch.tensor([1.0, 1.0], device=ce_grad.device, dtype=ce_grad.dtype)
    else:
        raise ValueError(f"Unknown objective conflict strategy: {strategy!r}")

    cosine = torch.nn.functional.cosine_similarity(
        ce_grad.unsqueeze(0).float(),
        kd_grad.unsqueeze(0).float(),
        dim=-1,
        eps=1e-8,
    ).squeeze(0).to(dtype=ce_grad.dtype, device=ce_grad.device)
    return weights, cosine


__all__ = [
    "resolve_objective_conflict_weights",
]
