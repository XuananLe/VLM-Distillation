import os
import math
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from typing import Optional, List


def _should_show_plots() -> bool:
    return os.environ.get("DEXAR_SHOW_PLOTS", "1").lower() not in {"0", "false", "no"}


def min_max(x: torch.Tensor) -> torch.Tensor:
    """Min-max normalization."""
    min_value = torch.min(x)
    max_value = torch.max(x)
    denom = max_value - min_value
    if not torch.isfinite(denom) or denom <= 0:
        return torch.zeros_like(x)
    return (x - min_value) / denom


def topk_norm(x: torch.Tensor, k: int, dim: int = -1) -> torch.Tensor:
    """Top-k norm scoring: average of top-k absolute values along dim."""
    return torch.topk(torch.abs(x), k=k, dim=dim)[0].sum(dim=dim) / k


def _resize_heatmap_to_image(heatmap: torch.Tensor, image_size: tuple[int, int]) -> np.ndarray:
    """Resize a 2D heatmap to the image size while keeping the source image sharp."""
    image_width, image_height = image_size
    heatmap_np = heatmap.detach().float().cpu().numpy()
    heatmap_np = np.nan_to_num(heatmap_np, nan=0.0, posinf=0.0, neginf=0.0)
    heatmap_np = np.clip(heatmap_np, 0.0, 1.0)

    if heatmap_np.shape == (image_height, image_width):
        return heatmap_np

    interpolation = cv2.INTER_CUBIC
    if heatmap_np.shape[0] > image_height or heatmap_np.shape[1] > image_width:
        interpolation = cv2.INTER_AREA

    return cv2.resize(heatmap_np, (image_width, image_height), interpolation=interpolation)


def _overlay_heatmap(
    image: Image.Image,
    heatmap: torch.Tensor,
    alpha: float,
) -> np.ndarray:
    """Render a transparent heatmap overlay on top of the original-resolution image."""
    base_image = image.convert("RGB")
    image_width, image_height = base_image.size
    heatmap_resized = _resize_heatmap_to_image(heatmap, (image_width, image_height))

    heatmap_uint8 = (heatmap_resized * 255).astype("uint8")
    image_bgr = cv2.cvtColor(np.array(base_image), cv2.COLOR_RGB2BGR)
    heatmap_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_TURBO)

    # Keep low-activation regions transparent so the underlying image stays legible.
    alpha_mask = np.clip((heatmap_resized - 0.05) / 0.95, 0.0, 1.0)
    alpha_mask = np.power(alpha_mask, 0.85) * alpha
    alpha_mask = alpha_mask[..., None].astype("float32")

    overlay_bgr = image_bgr.astype("float32") * (1.0 - alpha_mask)
    overlay_bgr += heatmap_bgr.astype("float32") * alpha_mask
    overlay_bgr = np.clip(overlay_bgr, 0.0, 255.0).astype("uint8")
    return cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)


def _figure_size_for_image(image_size: tuple[int, int], base_width: float = 8.0) -> tuple[float, float]:
    image_width, image_height = image_size
    aspect_ratio = image_height / max(image_width, 1)
    return base_width, max(3.0, base_width * aspect_ratio)


def visualize(
    image: Image.Image,
    heatmap: torch.Tensor,
    alpha: float = 0.6,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> None:
    """Overlay a heatmap on a PIL image.

    Args:
        image: PIL Image to overlay on.
        heatmap: 2D tensor [H, W] with values in [0, 1].
        alpha: Blending factor for the heatmap overlay.
        title: Optional title for the plot.
        save_path: If provided, saves the figure to this path.
    """
    viz = _overlay_heatmap(image=image, heatmap=heatmap, alpha=alpha)

    fig, ax = plt.subplots(figsize=_figure_size_for_image(image.size))
    ax.imshow(viz, interpolation="nearest")
    ax.axis("off")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight", pad_inches=0)
    if _should_show_plots():
        plt.show()
    plt.close(fig)


def visualize_multi(
    image: Image.Image,
    heatmaps: torch.Tensor,
    tokens: Optional[List[str]] = None,
    alpha: float = 0.6,
    save_path: Optional[str] = None,
) -> None:
    """Overlay multiple per-token heatmaps on a PIL image.

    Args:
        image: PIL Image to overlay on.
        heatmaps: 3D tensor [num_tokens, H, W] with values in [0, 1].
        tokens: Optional list of token strings for titles.
        alpha: Blending factor for the heatmap overlay.
        save_path: If provided, saves the figure to this path prefix.
    """
    num_tokens = heatmaps.shape[0]
    base_image = image.convert("RGB")

    if tokens is None:
        tokens = [str(i) for i in range(num_tokens)]

    num_panels = num_tokens + 1
    cols = min(4, num_panels)
    rows = math.ceil(num_panels / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.atleast_1d(axes).reshape(rows, cols)
    flat_axes = axes.ravel()

    overlays = []
    flat_axes[0].imshow(base_image)
    flat_axes[0].set_title("Original")
    flat_axes[0].axis("off")

    for i in range(num_tokens):
        viz = _overlay_heatmap(image=base_image, heatmap=heatmaps[i], alpha=alpha)
        overlays.append(viz)
        flat_axes[i + 1].imshow(viz, interpolation="nearest")
        flat_axes[i + 1].set_title(tokens[i])
        flat_axes[i + 1].axis("off")

    for ax in flat_axes[num_panels:]:
        ax.axis("off")

    fig.tight_layout()
    if save_path is not None:
        stem, ext = os.path.splitext(save_path)
        ext = ext or ".png"
        fig.savefig(save_path, dpi=200, bbox_inches="tight", pad_inches=0)
        for i in range(num_tokens):
            token_safe = tokens[i].strip().replace(" ", "_").replace("/", "_")
            individual_path = f"{stem}_{token_safe}{ext}"
            fig_i, ax_i = plt.subplots(figsize=_figure_size_for_image(base_image.size))
            ax_i.imshow(overlays[i], interpolation="nearest")
            ax_i.set_title(tokens[i])
            ax_i.axis("off")
            fig_i.tight_layout()
            fig_i.savefig(individual_path, dpi=200, bbox_inches="tight", pad_inches=0)
            plt.close(fig_i)
    if _should_show_plots():
        plt.show()
    plt.close(fig)
