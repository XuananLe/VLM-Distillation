import torch
from typing import Dict, Optional

from src.constants import IGNORE_INDEX
from .data_utils import pad_frames, pad_sequence


class DataCollatorForSupervisedDataset:

    def __init__(
        self,
        pad_token_id: int,
        teacher_pad_token_ids: Optional[list[Optional[int]]] = None,
    ):
        self.pad_token_id = pad_token_id
        self.teacher_pad_token_ids = teacher_pad_token_ids or []

    def teacher_prefixes(self, example: Dict[str, torch.Tensor], suffix: str) -> list[str]:
        marker = f"_{suffix}"
        prefixes = [
            key.removesuffix(marker)
            for key in example
            if key.startswith("teacher_") and key.endswith(marker)
        ]
        return sorted(prefixes, key=lambda prefix: int(prefix.split("_")[1]))

    def teacher_pad_for_prefix(self, prefix: str) -> int:
        teacher_index = int(prefix.split("_")[1])
        if teacher_index < len(self.teacher_pad_token_ids):
            teacher_pad = self.teacher_pad_token_ids[teacher_index]
            if teacher_pad is not None:
                return teacher_pad
        return self.pad_token_id

    def collate_teacher_batch(self, examples, batch_dict, prefix: str) -> None:
        teacher_pad = self.teacher_pad_for_prefix(prefix)
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
            batch_dict[pixel_key] = pad_frames(teacher_pixel_values, pad_value=0.0)
        else:
            batch_dict[pixel_key] = torch.cat(teacher_pixel_values, dim=0)

        pixel_attention_key = f"{prefix}_pixel_attention_mask"
        teacher_pixel_attention_masks = [e.get(pixel_attention_key) for e in examples]
        if teacher_pixel_attention_masks[0] is not None:
            batch_dict[pixel_attention_key] = pad_frames(
                teacher_pixel_attention_masks,
                pad_value=0,
            )

        image_grid_key = f"{prefix}_image_grid_thw"
        if image_grid_key in examples[0]:
            batch_dict[image_grid_key] = torch.cat(
                [e[image_grid_key] for e in examples],
                dim=0,
            )

        image_sizes_key = f"{prefix}_image_sizes"
        if image_sizes_key in examples[0]:
            batch_dict[image_sizes_key] = torch.cat(
                [e[image_sizes_key] for e in examples],
                dim=0,
            )

        image_flags_key = f"{prefix}_image_flags"
        if image_flags_key in examples[0]:
            batch_dict[image_flags_key] = torch.cat(
                [e[image_flags_key] for e in examples],
                dim=0,
            )

    def collate_cached_teacher_batch(self, examples, batch_dict, prefix: str) -> None:
        batch_dict[f"{prefix}_cached_logits"] = pad_sequence(
            [e[f"{prefix}_cached_logits"] for e in examples],
            padding_side="right",
            padding_value=0.0,
        )
        batch_dict[f"{prefix}_cached_labels"] = pad_sequence(
            [e[f"{prefix}_cached_labels"] for e in examples],
            padding_side="right",
            padding_value=IGNORE_INDEX,
        )

    def __call__(self, examples):
        batch_input_ids            = [e["input_ids"] for e in examples]
        batch_label_ids            = [e["labels"] for e in examples]
        batch_pixel_values         = [e.get("pixel_values") for e in examples]
        batch_pixel_attention_mask = [e.get("pixel_attention_mask") for e in examples]

        input_ids = pad_sequence(
            batch_input_ids, padding_side='right', padding_value=self.pad_token_id
        )
        attention_mask = input_ids != self.pad_token_id
        labels         = pad_sequence(batch_label_ids, padding_side='right', padding_value=IGNORE_INDEX)
        pixel_values = pad_frames(batch_pixel_values, pad_value=0.0)
        pixel_attention_mask = pad_frames(batch_pixel_attention_mask, pad_value=0)

        batch_dict = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )
        if pixel_values is not None:
            batch_dict.update(pixel_values=pixel_values, pixel_attention_mask=pixel_attention_mask)

        for prefix in self.teacher_prefixes(examples[0], "input_ids"):
            self.collate_teacher_batch(examples, batch_dict, prefix)
        for prefix in self.teacher_prefixes(examples[0], "cached_logits"):
            self.collate_cached_teacher_batch(examples, batch_dict, prefix)

        return batch_dict
