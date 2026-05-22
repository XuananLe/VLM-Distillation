import torch
from einops import einsum


def center_gram(gram_matrix: torch.Tensor) -> torch.Tensor:
    """Center a Gram matrix over its sample axis before CKA."""
    row_mean = gram_matrix.mean(dim=1, keepdim=True)
    column_mean = gram_matrix.mean(dim=0, keepdim=True)
    grand_mean = gram_matrix.mean()
    return gram_matrix - row_mean - column_mean + grand_mean


def linear_cka_tensor(features_a: torch.Tensor, features_b: torch.Tensor) -> torch.Tensor:
    """
    Linear CKA as a tensor scalar.

    The two feature matrices must share the sample axis but may have different
    feature dimensions.
    """
    # Linear CKA = <K_c, L_c> / (||K_c||_F ||L_c||_F), with K = X X^T and    L = Y Y^T.
    gram_a = center_gram(features_a @ features_a.T)
    gram_b = center_gram(features_b @ features_b.T)
    numerator = einsum(gram_a, gram_b, "sample_a sample_b, sample_a sample_b ->")
    denominator = torch.norm(gram_a, p="fro") * torch.norm(gram_b, p="fro")
    return (numerator / denominator.clamp_min(1e-10)).clamp(0.0, 1.0)


def linear_cka(features_a: torch.Tensor, features_b: torch.Tensor) -> float:
    """Return linear CKA as a Python float for reporting code."""
    return linear_cka_tensor(features_a, features_b).item()


def linear_cka_loss(features_a: torch.Tensor, features_b: torch.Tensor) -> torch.Tensor:
    """
    Differentiable CKA loss for aligned feature or probability matrices.
    """
    if features_a.shape[0] != features_b.shape[0]:
        raise ValueError(f"Sample count mismatch: {features_a.shape[0]} vs {features_b.shape[0]}.")
    if features_a.shape[0] < 2:
        return features_a.new_tensor(0.0)

    centered_a = features_a.float() - features_a.float().mean(dim=0, keepdim=True)
    centered_b = features_b.float() - features_b.float().mean(dim=0, keepdim=True)
    # The training loss uses 1 - sqrt(CKA) to keep identical representations at 0.
    return 1.0 - torch.sqrt(linear_cka_tensor(centered_a, centered_b))
