from typing import Dict, Tuple
import torch
from tqdm import tqdm
from src.utils import get_specific_layer
from src.components.forward_utils import (
    forward_with_kwarg_retry,
    infer_batch_size,
    prepare_forward_inputs,
    unwrap_tensor,
)
from src.components.vision_forward import pool_vision_features


def extract_sample_representations(
    model,
    inputs: Dict[str, torch.Tensor],
    layer_index: int = -1,
) -> torch.Tensor:
    """
    Pooled representation from a specific vision encoder layer per sample.
    Returns: (B, H)
    """
    fwd_inputs = prepare_forward_inputs(model, inputs)
    batch_size = infer_batch_size(fwd_inputs)
    layer, layer_name = get_specific_layer(model, layer_index)
    features = {}

    def hook_fn(module, hook_inputs, output):
        del module, hook_inputs
        tensor = unwrap_tensor(output)
        if tensor is None:
            raise RuntimeError(f"Hook output for vision layer '{layer_name}' did not contain a tensor.")
        features["output"] = tensor.detach()

    handle = layer.register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            forward_with_kwarg_retry(model, fwd_inputs)
    finally:
        handle.remove()

    if "output" not in features:
        raise RuntimeError(
            f"Vision layer '{layer_name}' did not produce hook features during the forward pass."
        )

    return pool_vision_features(features["output"], batch_size)


# ---------------------------------------------------------------------------
# CKA — sample-relationship similarity
# ---------------------------------------------------------------------------

def _center_gram(K: torch.Tensor) -> torch.Tensor:
    n = K.size(0)
    H = torch.eye(n, dtype=K.dtype, device=K.device) - 1.0 / n
    return H @ K @ H


def linear_cka_tensor(H_A: torch.Tensor, H_B: torch.Tensor) -> torch.Tensor:
    """
    Linear CKA as a tensor scalar. Keeps gradients intact for training use.
    """
    K_A = _center_gram(H_A @ H_A.T)
    K_B = _center_gram(H_B @ H_B.T)
    num = (K_A * K_B).sum()
    denom = torch.norm(K_A, p="fro") * torch.norm(K_B, p="fro")
    return (num / denom.clamp_min(1e-10)).clamp(0.0, 1.0)


def linear_cka(H_A: torch.Tensor, H_B: torch.Tensor) -> float:
    """
    Linear CKA. Invariant to rotation and isotropic scaling.
    Returns float in [0, 1]: 1 = identical, 0 = orthogonal.
    """
    return linear_cka_tensor(H_A, H_B).item()


def linear_cka_loss(H_A: torch.Tensor, H_B: torch.Tensor) -> torch.Tensor:
    """
    Differentiable loss form of linear CKA for training.
    """
    if H_A.shape[0] != H_B.shape[0]:
        raise ValueError(f"Sample count mismatch: {H_A.shape[0]} vs {H_B.shape[0]}.")
    if H_A.shape[0] < 2:
        return H_A.new_tensor(0.0)

    H_A = H_A.float() - H_A.float().mean(dim=0, keepdim=True)
    H_B = H_B.float() - H_B.float().mean(dim=0, keepdim=True)
    return 1.0 - torch.sqrt(linear_cka_tensor(H_A, H_B))


def compute_skc_from_matrices(
    H_A: torch.Tensor,
    H_B: torch.Tensor,
) -> Tuple[float, float, float, float, float, float]:
    """
    Compute score using linear CKA only.

    Returns:
        (skc, cka, 0.0, 0.0, 0.0, 0.0)
    """
    assert H_A.shape[0] == H_B.shape[0], (
        f"Sample count mismatch: {H_A.shape[0]} vs {H_B.shape[0]}."
    )

    H_A = H_A - H_A.mean(dim=0)
    H_B = H_B - H_B.mean(dim=0)

    cka = linear_cka(H_A, H_B)
    return cka, cka, 0.0, 0.0, 0.0, 0.0


def skc_score(
    model_a,
    model_b,
    dataloader_a,
    dataloader_b,
    layer_index: int = -1,
    verbose: bool = True,
    **kwargs,  # Accept unused args for API compat
) -> float:
    """
    Compute the score between two models using linear CKA only.

    Args:
        model_a, model_b      : models to compare
        dataloader_a/b        : dataloaders yielding the same samples
        layer_index           : which hidden layer to extract (-1 = last)

    Returns:
        CKA in [0, 1]:
            1.0 = identical sample geometry
            0.0 = orthogonal sample geometry
    """
    del kwargs
    reps_a, reps_b = [], []

    for batch_a, batch_b in tqdm(
        zip(dataloader_a, dataloader_b),
        total=len(dataloader_a),
        desc="Extracting representations",
    ):
        reps_a.append(
            extract_sample_representations(model_a, batch_a, layer_index).float().cpu()
        )
        reps_b.append(
            extract_sample_representations(model_b, batch_b, layer_index).float().cpu()
        )

    H_A = torch.cat(reps_a, dim=0)
    H_B = torch.cat(reps_b, dim=0)

    skc, cka, _, _, _, _ = compute_skc_from_matrices(H_A, H_B)

    if verbose:
        print()
        print(f"  CKA similarity : {cka:.4f}  (1=identical, 0=orthogonal)")
        print(f"  SKC = CKA      : {skc:.4f}")
        print()

        if skc > 0.90:
            print("  → REDUNDANT: models encode visual features nearly identically")
        elif skc > 0.75:
            print("  → MODERATE: some shared visual encoding, limited complementarity")
        else:
            print("  → COMPLEMENTARY: models extract diverse visual features")

    return skc
