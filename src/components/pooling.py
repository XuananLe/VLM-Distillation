import torch
from einops import rearrange, reduce

# [batch, sequence_length, hidden_dim] -> [batch, hidden_dim]
def masked_mean_pool_sequence(
    tensor: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    label_mask = labels.ne(ignore_index)
    pool_mask = label_mask
    pool_mask = pool_mask.to(dtype=tensor.dtype) # chuyen thanh tensor chua 1 va 0 
    masked_tensor = tensor * rearrange(pool_mask, "b t -> b t 1")
    pooled_tensor = reduce(masked_tensor, "b t d -> b d", "sum")
    pooled_denominator = reduce(pool_mask, "b t -> b 1", "sum").clamp(min=1.0)
    return pooled_tensor / pooled_denominator # masked mean = masked sum / masked count
