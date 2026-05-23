import torch
from einops import einsum


def center_gram(gram_matrix: torch.Tensor) -> torch.Tensor:
    """Center a Gram matrix over its sample axis before CKA."""
    row_mean = gram_matrix.mean(dim=1, keepdim=True)
    column_mean = gram_matrix.mean(dim=0, keepdim=True)
    grand_mean = gram_matrix.mean()
    return gram_matrix - row_mean - column_mean + grand_mean


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

    gram_a = center_gram(centered_a @ centered_a.T)
    gram_b = center_gram(centered_b @ centered_b.T)
    numerator = einsum(gram_a, gram_b, "sample_a sample_b, sample_a sample_b ->")
    denominator = torch.norm(gram_a, p="fro") * torch.norm(gram_b, p="fro")
    cka = (numerator / denominator.clamp_min(1e-10)).clamp(0.0, 1.0)

    # The training loss uses 1 - sqrt(CKA) to keep identical representations at 0.
    return 1.0 - torch.sqrt(cka)
