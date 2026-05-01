import torch

# [batch, sequence_length, hidden_dim] -> [batch, hidden_dim]
def masked_mean_pool_sequence(
    tensor: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    label_mask = labels.ne(ignore_index)
    pooled_tensor = torch.masked.mean(
        tensor,
        dim=1,
        mask=label_mask.unsqueeze(-1),
    )
    return torch.nan_to_num(pooled_tensor)
