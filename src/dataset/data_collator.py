from src.constants import IGNORE_INDEX

from .data_utils import pad_frames, pad_sequence


class DataCollatorForSupervisedDataset:
    def __init__(
        self,
        pad_token_id: int,
    ):
        self.pad_token_id = pad_token_id

    def __call__(self, examples):
        batch_input_ids = [e["input_ids"] for e in examples]
        batch_attention_masks = [e["attention_mask"] for e in examples]
        batch_label_ids = [e["labels"] for e in examples]
        batch_pixel_values = [e.get("pixel_values") for e in examples]
        batch_pixel_attention_mask = [e.get("pixel_attention_mask") for e in examples]

        input_ids = pad_sequence(batch_input_ids, padding_side="right", padding_value=self.pad_token_id)
        attention_mask = pad_sequence(batch_attention_masks, padding_side="right", padding_value=0)
        labels = pad_sequence(batch_label_ids, padding_side="right", padding_value=IGNORE_INDEX)
        pixel_values = pad_frames(batch_pixel_values, pad_value=0.0)

        batch_dict = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
        )
        if pixel_values is not None:
            batch_dict["pixel_values"] = pixel_values
            if batch_pixel_attention_mask[0] is not None:
                batch_dict["pixel_attention_mask"] = pad_frames(batch_pixel_attention_mask, pad_value=0)

        teacher_prefixes = [
            key.removesuffix("_cached_logits")
            for key in examples[0]
            if key.startswith("teacher_") and key.endswith("_cached_logits")
        ]
        teacher_prefixes.sort(key=lambda prefix: int(prefix.split("_")[1]))

        for prefix in teacher_prefixes:
            batch_dict[f"{prefix}_cached_logits"] = pad_sequence(
                [example[f"{prefix}_cached_logits"] for example in examples],
                padding_side="right",
                padding_value=0.0,
            )
            batch_dict[f"{prefix}_cached_labels"] = pad_sequence(
                [example[f"{prefix}_cached_labels"] for example in examples],
                padding_side="right",
                padding_value=IGNORE_INDEX,
            )

        return batch_dict
