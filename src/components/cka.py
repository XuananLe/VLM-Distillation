from typing import Tuple

import torch


def center_gram(gram_matrix: torch.Tensor) -> torch.Tensor:
    num_samples = gram_matrix.size(0)
    centering = (
        torch.eye(num_samples, dtype=gram_matrix.dtype, device=gram_matrix.device)
        - 1.0 / num_samples
    )
    return centering @ gram_matrix @ centering


def linear_cka_tensor(features_a: torch.Tensor, features_b: torch.Tensor) -> torch.Tensor:
    """
    Linear CKA as a tensor scalar.

    The two feature matrices must share the sample axis but may have different
    feature dimensions.
    """
    gram_a = center_gram(features_a @ features_a.T)
    gram_b = center_gram(features_b @ features_b.T)
    numerator = (gram_a * gram_b).sum()
    denominator = torch.norm(gram_a, p="fro") * torch.norm(gram_b, p="fro")
    return (numerator / denominator.clamp_min(1e-10)).clamp(0.0, 1.0)


def linear_cka(features_a: torch.Tensor, features_b: torch.Tensor) -> float:
    return linear_cka_tensor(features_a, features_b).item()


def linear_cka_loss(features_a: torch.Tensor, features_b: torch.Tensor) -> torch.Tensor:
    """
    Differentiable CKA loss used by older layer-distillation code paths.
    """
    if features_a.shape[0] != features_b.shape[0]:
        raise ValueError(
            f"Sample count mismatch: {features_a.shape[0]} vs {features_b.shape[0]}."
        )
    if features_a.shape[0] < 2:
        return features_a.new_tensor(0.0)

    centered_a = features_a.float() - features_a.float().mean(dim=0, keepdim=True)
    centered_b = features_b.float() - features_b.float().mean(dim=0, keepdim=True)
    return 1.0 - torch.sqrt(linear_cka_tensor(centered_a, centered_b))


def compute_skc_from_matrices(
    features_a: torch.Tensor,
    features_b: torch.Tensor,
) -> Tuple[float, float, float, float, float, float]:
    """
    Compatibility helper used by the plotting script.

    Returns:
        (skc, cka, 0.0, 0.0, 0.0, 0.0)
    """
    if features_a.shape[0] != features_b.shape[0]:
        raise AssertionError(
            f"Sample count mismatch: {features_a.shape[0]} vs {features_b.shape[0]}."
        )

    centered_a = features_a - features_a.mean(dim=0)
    centered_b = features_b - features_b.mean(dim=0)
    cka = linear_cka(centered_a, centered_b)
    return cka, cka, 0.0, 0.0, 0.0, 0.0
