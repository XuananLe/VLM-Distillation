"""DEX-AR usage example.

Produces per-token heatmaps and a sentence-level explainability map
for a given image and target sentence.

Usage:
    python playground.py
"""

import os

from PIL import Image
import torch
from dexar import DexarWrapper, visualize, visualize_multi


def main():
    output_root = os.environ.get("DEXAR_OUTPUT_DIR", ".")
    filtered_dir = os.path.join(output_root, "filtered")
    unfiltered_dir = os.path.join(output_root, "unfiltered")
    os.makedirs(filtered_dir, exist_ok=True)
    os.makedirs(unfiltered_dir, exist_ok=True)

    model_name = os.environ.get("DEXAR_MODEL_NAME", "llava-hf/llava-1.5-7b-hf")
    device = os.environ.get("DEXAR_DEVICE", "cuda")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    layer_index = int(os.environ.get("DEXAR_LAYER_INDEX", "0"))
    target_sentence = os.environ.get(
        "DEXAR_TARGET_SENTENCE",
        "The image features a dog and a cat sitting together in a grassy field",
    )
    disable_resize = os.environ.get("DEXAR_DISABLE_RESIZE", "0").lower() in {
        "1",
        "true",
        "yes",
    }

    print(
        {
            "model_name": model_name,
            "device": device,
            "layer_index": layer_index,
            "output_root": output_root,
            "disable_resize": disable_resize,
        }
    )
    # --- Load model ---
    model = DexarWrapper.from_pretrained(
        model_name, device=device, layer_index=layer_index
    )

    # --- Load a sample image ---
    path_to_image = os.environ.get("DEXAR_IMAGE_PATH", "./assets/cat_and_dog.jpg")
    image_size = model.recommended_image_size
    image = Image.open(path_to_image).convert("RGB")
    if not disable_resize:
        image = image.resize((image_size, image_size))

    # --- Compute DEX-AR ---
    prompt = os.environ.get("DEXAR_PROMPT", model.default_prompt)
    result = model.compute_dexar(
        image=image,
        target_sentence=target_sentence,
        prompt=prompt,
    )

    # --- Print token info ---
    print("Tokens:", result.tokens)
    print("Token weights (delta^t):", result.token_weights)
    print("Per-token heatmaps shape:", result.per_token_heatmaps.shape)
    print("Sentence heatmap shape:", result.sentence_heatmap.shape)

    # --- Filtered: sentence-level heatmap (with head filtering, Eq. 5) ---
    visualize(
        image=image, heatmap=result.sentence_heatmap,
        title="Sentence heatmap (filtered)",
        save_path=os.path.join(filtered_dir, "sentence_heatmap.png"),
    )

    # --- Filtered: per-token heatmaps ---
    visualize_multi(
        image=image, heatmaps=result.per_token_heatmaps, tokens=result.tokens,
        save_path=os.path.join(filtered_dir, "per_token_heatmaps.png"),
    )

    # --- Unfiltered: sentence-level heatmap (no head filtering) ---
    visualize(
        image=image, heatmap=result.sentence_heatmap_unfiltered,
        title="Sentence heatmap (unfiltered)",
        save_path=os.path.join(unfiltered_dir, "sentence_heatmap.png"),
    )

    # --- Unfiltered: per-token heatmaps ---
    visualize_multi(
        image=image, heatmaps=result.per_token_heatmaps_unfiltered, tokens=result.tokens,
        save_path=os.path.join(unfiltered_dir, "per_token_heatmaps.png"),
    )


if __name__ == "__main__":
    main()
