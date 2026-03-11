import torch
import torch.nn.functional as F
from typing import Dict, List, Optional


def pooled_embedding(
    model,
    inputs: Dict[str, torch.Tensor],
    layer_index: int = -1,
) -> torch.Tensor:
    """Mean-pool hidden states from one layer over the sequence dimension.

    Args:
        model:       Transformer model (call inside torch.no_grad()).
        inputs:      Input dict; labels are stripped before the forward pass.
        layer_index: Layer to extract from (-1 = last transformer layer).

    Returns:
        Tensor of shape (B, H) — one vector per sample.
    """
    fwd_inputs = {k: v for k, v in inputs.items() if k != "labels"}
    outputs = model(**fwd_inputs, output_hidden_states=True)
    hs = outputs.hidden_states[layer_index]  # (B, T, H)
    return hs.mean(dim=1)                    # (B, H)


def skc_score(
    model_a,
    model_b,
    dataloader,
    k: int = 32,
    layer_index: int = -1,
) -> float:
    """Spectral Knowledge Complementarity (SKC) score.

    Measures how complementary two teacher models are on a probe dataset.
    High score → complementary (low redundancy); low score → redundant.

    Algorithm:
        1. Extract mean-pooled representations from both models over the probe set.
        2. Center both representation matrices column-wise.
        3. Compute economy SVD of each matrix.
        4. Compute JSD between the spectral energy distributions (captures
           differences in which principal directions carry the most variance).
        5. Compute the normalised squared Frobenius norm of the top-k
           left-singular-vector overlap (captures whether the models span
           similar sample-space subspaces).
        6. skc = SpectralDiff * (1 - SubspaceOverlap)

    Args:
        model_a:     Teacher A (should be in eval mode, no_grad is applied here).
        model_b:     Teacher B (same).
        dataloader:  Yields input dicts compatible with both models.
                     Batches must not contain "labels".
        k:           Number of top sample-space singular vectors for the
                     subspace overlap term.
        layer_index: Layer from which to extract representations (-1 = last).

    Returns:
        SKC score in [0, 1].
    """
    H_A_list: List[torch.Tensor] = []
    H_B_list: List[torch.Tensor] = []

    with torch.no_grad():
        for batch in dataloader:
            H_A_list.append(pooled_embedding(model_a, batch, layer_index).float().cpu())
            H_B_list.append(pooled_embedding(model_b, batch, layer_index).float().cpu())

    H_A = torch.cat(H_A_list, dim=0)  # (N, H_A)
    H_B = torch.cat(H_B_list, dim=0)  # (N, H_B)

    # Center column-wise
    H_A = H_A - H_A.mean(dim=0)
    H_B = H_B - H_B.mean(dim=0)

    # Economy SVD: U (N, r), S (r,), Vh (r, H)
    U_A, S_A, _ = torch.linalg.svd(H_A, full_matrices=False)
    U_B, S_B, _ = torch.linalg.svd(H_B, full_matrices=False)

    # Spectral energy distributions
    lambda_A = S_A ** 2
    lambda_B = S_B ** 2

    L = max(lambda_A.size(0), lambda_B.size(0))
    p_A = F.pad(lambda_A, (0, L - lambda_A.size(0)))
    p_B = F.pad(lambda_B, (0, L - lambda_B.size(0)))
    p_A = p_A / p_A.sum()
    p_B = p_B / p_B.sum()

    # Jensen-Shannon divergence (base-2, so JSD ∈ [0, 1])
    m = 0.5 * (p_A + p_B)
    jsd = 0.5 * (
        torch.where(p_A > 0, p_A * (p_A / m).log2(), p_A.new_zeros(())).sum()
        + torch.where(p_B > 0, p_B * (p_B / m).log2(), p_B.new_zeros(())).sum()
    )
    spectral_diff = jsd.clamp(0.0, 1.0).item()

    # Subspace overlap via top-k left singular vectors
    k_eff = min(k, U_A.size(1), U_B.size(1))
    M = U_A[:, :k_eff].T @ U_B[:, :k_eff]          # (k_eff, k_eff)
    subspace_overlap = (M ** 2).sum().item() / k_eff  # normalised ∈ [0, 1]

    return spectral_diff * (1.0 - subspace_overlap)
