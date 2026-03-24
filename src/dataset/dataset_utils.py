"""Helper utilities for data processing and formatting."""
import random
from typing import Optional
from torch.utils.data import Subset

from src.constants import LLAVA_IMAGE_TOKEN, LLAVA_VIDEO_TOKEN


def replace_image_tokens(input_string, start_count=1):
    """Replace LLAVA image tokens with numbered image tokens."""
    count = start_count

    if LLAVA_IMAGE_TOKEN not in input_string:
        return input_string, count

    while LLAVA_IMAGE_TOKEN+'\n' in input_string:
        input_string = input_string.replace(LLAVA_IMAGE_TOKEN+'\n', "<image>", 1)
        count += 1

    return input_string, count


def video_to_image_tokens(input_string, num_frames):
    """Convert video tokens to multiple image tokens based on number of frames."""
    frame_tokens = "\n".join([LLAVA_IMAGE_TOKEN] * num_frames)
    input_string = input_string.replace(LLAVA_VIDEO_TOKEN, frame_tokens)
    return input_string


def llava_to_openai(conversations, is_video=False, num_frames=None):
    """Convert LLaVA format conversations to OpenAI format."""
    role_mapping = {"human": "user", "gpt": "assistant"}

    transformed_data = []
    image_count = 1
    for conversation in conversations:
        if is_video:
            conversation['value'] = video_to_image_tokens(conversation["value"], num_frames)

        transformed_content, image_count = replace_image_tokens(conversation["value"], image_count)
        transformed_entry = {
            "role": role_mapping.get(conversation["from"], conversation["from"]),
            "content": transformed_content
        }
        transformed_data.append(transformed_entry)

    return transformed_data


def subset_dataset(dataset, subset_size: Optional[int], data_path: Optional[str]):
    """Create a subset of the dataset with optional stratification."""
    if subset_size is None:
        return dataset
    if subset_size <= 0:
        raise ValueError("--train_subset_size and --eval_subset_size must be > 0 when set.")

    subset_size = min(subset_size, len(dataset))
    dataset_name = str(data_path).lower()
    rng = random.Random(42)

    # Special handling for ChartQA dataset with stratification
    if "chartqa" in dataset_name and hasattr(dataset, "list_data_dict"):
        split_key = None
        for candidate in ("chartqa_split", "split"):
            if dataset.list_data_dict and all(candidate in row for row in dataset.list_data_dict):
                split_key = candidate
                break

        if split_key is not None:
            # Stratified sampling by split
            grouped_indices = {}
            for idx, row in enumerate(dataset.list_data_dict):
                grouped_indices.setdefault(str(row[split_key]), []).append(idx)

            group_names = sorted(grouped_indices)
            base = subset_size // len(group_names)
            remainder = subset_size % len(group_names)

            selected = []
            leftovers = []
            for idx, group_name in enumerate(group_names):
                group = grouped_indices[group_name]
                rng.shuffle(group)
                take = min(base + int(idx < remainder), len(group))
                selected.extend(group[:take])
                leftovers.extend(group[take:])

            if len(selected) < subset_size:
                rng.shuffle(leftovers)
                selected.extend(leftovers[: subset_size - len(selected)])

            indices = sorted(selected)
        else:
            indices = sorted(rng.sample(range(len(dataset)), subset_size))

    # Random sampling for other datasets
    elif any(name in dataset_name for name in ("docvqa", "textvqa")):
        indices = sorted(rng.sample(range(len(dataset)), subset_size))
    else:
        # Sequential indices for unknown datasets
        indices = range(subset_size)

    return Subset(dataset, indices)
