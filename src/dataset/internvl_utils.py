from PIL import Image
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

INTERNVL_MEAN = (0.485, 0.456, 0.406)
INTERNVL_STD = (0.229, 0.224, 0.225)
INTERNVL_SIGLIP_MEAN = (0.5, 0.5, 0.5)
INTERNVL_SIGLIP_STD = (0.5, 0.5, 0.5)


def _build_internvl_transform(input_size: int, normalize_type: str) -> T.Compose:
    """Build the torchvision preprocessing pipeline used by InternVL image tiles."""
    if normalize_type == "siglip":
        mean, std = INTERNVL_SIGLIP_MEAN, INTERNVL_SIGLIP_STD
    else:
        mean, std = INTERNVL_MEAN, INTERNVL_STD
    return T.Compose(
        [
            T.Lambda(lambda image: image.convert("RGB") if image.mode != "RGB" else image),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )


def _find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """Pick the tiling aspect ratio that best matches the source image geometry."""
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            # InternVL breaks exact aspect-ratio ties by preferring a denser tile
            # grid only when the source image area is large enough to justify it.
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def _dynamic_preprocess_internvl(
    image: Image.Image,
    *,
    image_size: int,
    max_num_tiles: int,
    min_num_tiles: int = 1,
    use_thumbnail: bool = True,
) -> list[Image.Image]:
    """Split one image into InternVL-style tiles plus an optional thumbnail tile."""
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j)
        for n in range(min_num_tiles, max_num_tiles + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if min_num_tiles <= i * j <= max_num_tiles
    )
    target_ratios = sorted(target_ratios, key=lambda ratio: ratio[0] * ratio[1])
    target_aspect_ratio = _find_closest_aspect_ratio(
        aspect_ratio,
        target_ratios,
        orig_width,
        orig_height,
        image_size,
    )
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_image = image.resize((target_width, target_height), Image.BICUBIC)
    processed_images = []
    for block_index in range(blocks):
        box = (
            (block_index % (target_width // image_size)) * image_size,
            (block_index // (target_width // image_size)) * image_size,
            ((block_index % (target_width // image_size)) + 1) * image_size,
            ((block_index // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_image.crop(box))
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size), Image.BICUBIC))
    return processed_images


def _build_internvl_pixel_values(image: Image.Image, teacher_processor: dict) -> torch.Tensor:
    """Convert one image into the stacked tensor tiles expected by InternVL teachers."""
    # 448 and 6 are the repo's fallback InternVL preprocessing defaults when the
    # live teacher config does not expose force_image_size / tiling metadata.
    image_size = teacher_processor.get("image_size", 448)
    max_num_tiles = teacher_processor.get("max_num_tiles", 6)
    normalize_type = teacher_processor.get("normalize_type", "imagenet")
    transform = _build_internvl_transform(image_size, normalize_type)
    processed_images = _dynamic_preprocess_internvl(
        image,
        image_size=image_size,
        max_num_tiles=max_num_tiles,
        use_thumbnail=True,
    )
    return torch.stack([transform(processed_image) for processed_image in processed_images])
