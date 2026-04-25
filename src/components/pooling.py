import torch
from einops import rearrange, reduce


def masked_mean_pool_sequence(
    tensor: torch.Tensor,
    *,
    labels: torch.Tensor,
    attention_mask: torch.Tensor | None,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Mean-pool a sequence over supervised answer tokens."""
    label_mask = labels.ne(ignore_index)
    del attention_mask

    pool_mask = label_mask
    missing_supervised = ~pool_mask.any(dim=1)
    if missing_supervised.any():
        bad_indices = missing_supervised.nonzero(as_tuple=True)[0].tolist()
        raise ValueError(
            "Cannot pool sequence: samples contain no supervised answer tokens. "
            f"indices={bad_indices}"
        )

    pool_mask = pool_mask.to(dtype=tensor.dtype)
    # Mean-pool over selected sequence positions.
    masked_tensor = tensor * rearrange(pool_mask, "b t -> b t 1")
    pooled_tensor = reduce(masked_tensor, "b t d -> b d", "sum")
    pooled_denominator = reduce(pool_mask, "b t -> b 1", "sum").clamp(min=1.0)
    return pooled_tensor / pooled_denominator
