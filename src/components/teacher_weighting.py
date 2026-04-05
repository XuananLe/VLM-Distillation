import torch


def project_to_capped_simplex(
    values: torch.Tensor,
    upper_bound: float,
    *,
    max_bisection_steps: int = 64,
) -> torch.Tensor:
    num_weights = values.numel()
    upper_bound = max(float(upper_bound), 1.0 / max(num_weights, 1))
    if num_weights == 1:
        return values.new_ones((1,))

    lower = (values - upper_bound).min().item()
    upper = values.max().item()
    projected = values
    for _ in range(max_bisection_steps):
        midpoint = 0.5 * (lower + upper)
        projected = torch.clamp(values - midpoint, min=0.0, max=upper_bound)
        total = projected.sum().item()
        if total > 1.0:
            lower = midpoint
        else:
            upper = midpoint

    projected = torch.clamp(values - upper, min=0.0, max=upper_bound)
    total = projected.sum()
    if total <= torch.finfo(projected.dtype).eps:
        return projected.new_full((num_weights,), 1.0 / num_weights)
    return projected / total


def solve_gradient_weight_vector(
    gradient_vectors: torch.Tensor,
    *,
    weight_cap: float = 1.0,
    max_steps: int = 50,
    tolerance: float = 1e-6,
) -> torch.Tensor:
    if gradient_vectors.ndim != 2:
        raise ValueError(
            "gradient_vectors must have shape [num_teachers, feature_dim]. "
            f"Got {tuple(gradient_vectors.shape)}."
        )

    num_teachers = gradient_vectors.size(0)
    if num_teachers == 1:
        return gradient_vectors.new_ones((1,))

    working_vectors = gradient_vectors.float()
    gram_matrix = working_vectors @ working_vectors.transpose(0, 1)
    if gram_matrix.abs().max() <= torch.finfo(gram_matrix.dtype).eps:
        return gradient_vectors.new_full((num_teachers,), 1.0 / num_teachers)

    max_eigenvalue = torch.linalg.eigvalsh(gram_matrix).amax().clamp_min(
        torch.finfo(gram_matrix.dtype).eps
    )
    step_size = 1.0 / (2.0 * max_eigenvalue)
    weights = gram_matrix.new_full((num_teachers,), 1.0 / num_teachers)
    effective_cap = max(float(weight_cap), 1.0 / num_teachers)

    for _ in range(max_steps):
        updated_weights = project_to_capped_simplex(
            weights - step_size * (2.0 * (gram_matrix @ weights)),
            effective_cap,
        )
        if torch.max(torch.abs(updated_weights - weights)).item() <= tolerance:
            weights = updated_weights
            break
        weights = updated_weights

    return weights.to(dtype=gradient_vectors.dtype, device=gradient_vectors.device)


__all__ = ["solve_gradient_weight_vector", "project_to_capped_simplex"]
