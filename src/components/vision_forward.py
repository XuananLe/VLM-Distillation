import torch


def infer_vision_group_counts(model_inputs, batch_size: int) -> list[int] | None:
    image_grid_thw = model_inputs.get("image_grid_thw")
    if isinstance(image_grid_thw, torch.Tensor) and image_grid_thw.ndim == 2 and image_grid_thw.shape[-1] == 3:
        if image_grid_thw.shape[0] == batch_size:
            return image_grid_thw.to(dtype=torch.long).prod(dim=-1).tolist()

    pixel_attention_mask = model_inputs.get("pixel_attention_mask")
    if isinstance(pixel_attention_mask, torch.Tensor) and pixel_attention_mask.ndim >= 3:
        flat_mask = pixel_attention_mask.reshape(
            pixel_attention_mask.shape[0],
            pixel_attention_mask.shape[1],
            -1,
        )
        return flat_mask.any(dim=-1).sum(dim=-1).to(dtype=torch.long).tolist()

    pixel_values = model_inputs.get("pixel_values")
    if isinstance(pixel_values, torch.Tensor) and pixel_values.ndim >= 5:
        flat_pixels = pixel_values.reshape(
            pixel_values.shape[0],
            pixel_values.shape[1],
            -1,
        )
        return flat_pixels.abs().sum(dim=-1).ne(0).sum(dim=-1).to(dtype=torch.long).tolist()

    return None


def pool_vision_features(
    features: torch.Tensor,
    batch_size: int,
    group_counts: list[int] | None = None,
) -> torch.Tensor:
    if group_counts is not None and features.ndim in (2, 3):
        total_groups = sum(int(count) for count in group_counts)
        if len(group_counts) == batch_size and features.shape[0] == total_groups:
            hidden_size = features.shape[-1]
            pooled = []
            start = 0
            for count in group_counts:
                count = int(count)
                if count <= 0:
                    pooled.append(features.new_zeros(hidden_size))
                    continue
                stop = start + count
                pooled.append(features[start:stop].reshape(-1, hidden_size).mean(dim=0))
                start = stop
            if start != features.shape[0]:
                raise ValueError(
                    f"Vision grouping consumed {start} entries, expected {features.shape[0]}."
                )
            return torch.stack(pooled, dim=0)

    if features.ndim == 1:
        return features.unsqueeze(0)
    if features.ndim == 2:
        if features.shape[0] == batch_size:
            return features
        if batch_size == 1:
            return features.reshape(-1, features.shape[-1]).mean(dim=0, keepdim=True)
    if features.ndim == 3 and features.shape[0] == batch_size:
        return features.mean(dim=1)
    if features.ndim == 4 and features.shape[0] == batch_size:
        return features.flatten(start_dim=2).mean(dim=2)
    if features.ndim >= 2 and batch_size == 1:
        return features.reshape(-1, features.shape[-1]).mean(dim=0, keepdim=True)
    raise ValueError(
        f"Unsupported vision feature shape {tuple(features.shape)} for batch size {batch_size}."
    )
