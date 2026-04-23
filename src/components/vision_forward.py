import torch
from einops import rearrange, reduce


def infer_vision_group_counts(model_inputs, batch_size: int) -> list[int] | None:
    """Infer how many packed vision groups belong to each sample in a batch."""
    image_grid_thw = model_inputs.get("image_grid_thw")
    if isinstance(image_grid_thw, torch.Tensor) and image_grid_thw.ndim == 2 and image_grid_thw.shape[-1] == 3:
        if image_grid_thw.shape[0] == batch_size:
            # Qwen-style image_grid_thw encodes how many vision groups belong to each sample.
            return image_grid_thw.to(dtype=torch.long).prod(dim=-1).tolist()

    pixel_attention_mask = model_inputs.get("pixel_attention_mask")
    if isinstance(pixel_attention_mask, torch.Tensor) and pixel_attention_mask.ndim >= 3:
        flat_mask = rearrange(pixel_attention_mask, "b g ... -> b g (...)")
        return flat_mask.any(dim=-1).sum(dim=-1).to(dtype=torch.long).tolist()

    pixel_values = model_inputs.get("pixel_values")
    if isinstance(pixel_values, torch.Tensor) and pixel_values.ndim >= 5:
        flat_pixels = rearrange(pixel_values, "b g ... -> b g (...)")
        return flat_pixels.abs().sum(dim=-1).ne(0).sum(dim=-1).to(dtype=torch.long).tolist()

    return None


def pool_vision_features(
    features: torch.Tensor,
    batch_size: int,
    group_counts: list[int] | None = None,
) -> torch.Tensor:
    """Collapse vision features into one vector per sample for downstream matching."""
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
                # Collapse all vision groups that belong to one sample into a single vector.
                pooled_chunk = rearrange(features[start:stop], "... d -> (...) d")
                pooled.append(reduce(pooled_chunk, "n d -> d", "mean"))
                start = stop
            if start != features.shape[0]:
                raise ValueError(
                    f"Vision grouping consumed {start} entries, expected {features.shape[0]}."
                )
            return torch.stack(pooled, dim=0)

    if features.ndim == 1:
        return rearrange(features, "d -> 1 d")
    if features.ndim == 2:
        if features.shape[0] == batch_size:
            return features
        if batch_size == 1:
            return reduce(features, "n d -> 1 d", "mean")
    if features.ndim == 3 and features.shape[0] == batch_size:
        return reduce(features, "b t d -> b d", "mean")
    if features.ndim == 4 and features.shape[0] == batch_size:
        return reduce(features, "b c h w -> b c", "mean")
    if features.ndim >= 2 and batch_size == 1:
        return reduce(rearrange(features, "... d -> (...) d"), "n d -> 1 d", "mean")
    raise ValueError(
        f"Unsupported vision feature shape {tuple(features.shape)} for batch size {batch_size}."
    )
