"""Data collator for supervised fine-tuning datasets."""
import re
import torch
from typing import Dict, Optional

from src.constants import IGNORE_INDEX
from .data_utils import pad_sequence

_DUMMY_PIXEL_VALUES = (1, 13, 3, 384, 384)
_DUMMY_PIXEL_MASK = (1, 13, 384, 384)


def _pad_frames(tensors, pad_value=0):
    """Pad a list of (1, T, ...) tensors to (B, T_max, ...) along the frame dim."""
    T_max = max(t.size(1) for t in tensors)
    out = torch.full(
        (len(tensors), T_max) + tensors[0].shape[2:],
        fill_value=pad_value,
        dtype=tensors[0].dtype,
        device=tensors[0].device,
    )
    for i, t in enumerate(tensors):
        out[i, :t.size(1)] = t[0]
    return out


def pad_pixel_values(pixel_values_list, pad_value=0.0):
    """Pad pixel values along the frame dimension."""
    return _pad_frames(pixel_values_list, pad_value)


def pad_pixel_attention_masks(mask_list, pad_value=0):
    """Pad pixel attention masks along the frame dimension."""
    return _pad_frames(mask_list, pad_value)


class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    def __init__(
        self,
        pad_token_id: int,
        teacher_pad_token_id: Optional[int] = None,
        teacher_pad_token_ids: Optional[list[Optional[int]]] = None,
    ):
        self.pad_token_id         = pad_token_id
        self.teacher_pad_token_id = teacher_pad_token_id
        self.teacher_pad_token_ids = teacher_pad_token_ids or []

    def _teacher_input_prefixes(self, example: Dict[str, torch.Tensor]) -> list[str]:
        """Extract teacher input prefixes from an example."""
        if "teacher_input_ids" in example:
            return ["teacher"]

        prefixes = []
        for key in example:
            match = re.fullmatch(r"(teacher_\d+)_input_ids", key)
            if match:
                prefixes.append(match.group(1))
        return sorted(prefixes, key=lambda prefix: int(prefix.split("_")[1]))

    def _teacher_pad_for_prefix(self, prefix: str) -> int:
        """Get the padding token ID for a specific teacher prefix."""
        if prefix == "teacher":
            return self.teacher_pad_token_id or self.pad_token_id

        teacher_index = int(prefix.split("_")[1])
        if teacher_index < len(self.teacher_pad_token_ids):
            teacher_pad = self.teacher_pad_token_ids[teacher_index]
            if teacher_pad is not None:
                return teacher_pad
        return self.pad_token_id

    def _collate_teacher_batch(self, examples, batch_dict, prefix: str) -> None:
        """Collate teacher inputs from examples into the batch dictionary."""
        teacher_pad = self._teacher_pad_for_prefix(prefix)
        teacher_input_ids = pad_sequence(
            [e[f"{prefix}_input_ids"] for e in examples],
            padding_side="right",
            padding_value=teacher_pad,
        )
        teacher_labels = pad_sequence(
            [e[f"{prefix}_labels"] for e in examples],
            padding_side="right",
            padding_value=IGNORE_INDEX,
        )
        batch_dict.update(
            {
                f"{prefix}_input_ids": teacher_input_ids,
                f"{prefix}_labels": teacher_labels,
                f"{prefix}_attention_mask": teacher_input_ids != teacher_pad,
            }
        )

        pixel_key = f"{prefix}_pixel_values"
        if pixel_key not in examples[0]:
            return

        teacher_pixel_values = [e[pixel_key] for e in examples]
        if teacher_pixel_values[0].dim() == 5:
            batch_dict[pixel_key] = pad_pixel_values(teacher_pixel_values, pad_value=0.0)
        else:
            batch_dict[pixel_key] = torch.cat(teacher_pixel_values, dim=0)

        pixel_attention_key = f"{prefix}_pixel_attention_mask"
        teacher_pixel_attention_masks = [e.get(pixel_attention_key) for e in examples]
        if teacher_pixel_attention_masks[0] is not None:
            batch_dict[pixel_attention_key] = pad_pixel_attention_masks(
                teacher_pixel_attention_masks,
                pad_value=0,
            )

        image_grid_key = f"{prefix}_image_grid_thw"
        if image_grid_key in examples[0]:
            batch_dict[image_grid_key] = torch.cat(
                [e[image_grid_key] for e in examples],
                dim=0,
            )

        image_flags_key = f"{prefix}_image_flags"
        if image_flags_key in examples[0]:
            batch_dict[image_flags_key] = torch.cat(
                [e[image_flags_key] for e in examples],
                dim=0,
            )

    def __call__(self, examples):
        """Collate a batch of examples."""
        batch_input_ids            = [e["input_ids"]                    for e in examples]
        batch_label_ids            = [e["labels"]                       for e in examples]
        batch_pixel_values         = [e.get("pixel_values")             for e in examples]
        batch_pixel_attention_mask = [e.get("pixel_attention_mask")     for e in examples]

        input_ids = pad_sequence(
            batch_input_ids, padding_side='right', padding_value=self.pad_token_id
        )
        attention_mask = input_ids != self.pad_token_id
        labels         = pad_sequence(batch_label_ids, padding_side='right', padding_value=IGNORE_INDEX)
        pixel_values   = pad_pixel_values(batch_pixel_values, pad_value=0.0)
        pixel_attention_mask = pad_pixel_attention_masks(batch_pixel_attention_mask, pad_value=0)

        batch_dict = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )
        if pixel_values is not None:
            batch_dict.update(pixel_values=pixel_values, pixel_attention_mask=pixel_attention_mask)

        for prefix in self._teacher_input_prefixes(examples[0]):
            self._collate_teacher_batch(examples, batch_dict, prefix)

        return batch_dict
