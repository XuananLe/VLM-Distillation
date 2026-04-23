import torch
from einops import rearrange, reduce


def masked_mean_pool_sequence(
    tensor: torch.Tensor,
    *,
    labels: torch.Tensor,
    attention_mask: torch.Tensor | None,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Mean-pool a sequence over supervised tokens with an attention-mask fallback."""
    label_mask = labels.ne(ignore_index)
    fallback_mask = (
        attention_mask.bool()
        if attention_mask is not None
        else torch.ones_like(label_mask, dtype=torch.bool)
    )

    pool_mask = label_mask
    missing_supervised = ~pool_mask.any(dim=1)
    if missing_supervised.any():
        # Some prompts may contain no supervised answer tokens after masking; in that
        # case fall back to the general attention mask so the sample still has context.
        pool_mask = pool_mask.clone()
        pool_mask[missing_supervised] = fallback_mask[missing_supervised]

    pool_mask = pool_mask.to(dtype=tensor.dtype)
    # Mean-pool over selected sequence positions.
    masked_tensor = tensor * rearrange(pool_mask, "b t -> b t 1")
    pooled_tensor = reduce(masked_tensor, "b t d -> b d", "sum")
    pooled_denominator = reduce(pool_mask, "b t -> b 1", "sum").clamp(min=1.0)
    return pooled_tensor / pooled_denominator
