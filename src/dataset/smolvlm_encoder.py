from typing import Dict

import torch
import transformers
from PIL import Image

from src.constants import IGNORE_INDEX, LLAVA_IMAGE_TOKEN


def smolvlm_encode_conversation(
    sources,
    image: Image.Image,
    processor: transformers.ProcessorMixin,
) -> Dict[str, torch.Tensor]:
    user_input, gpt_response = sources
    user_text = user_input["value"].replace(LLAVA_IMAGE_TOKEN, "").strip()

    user_message = {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": user_text},
        ],
    }
    assistant_message = {
        "role": "assistant",
        "content": [{"type": "text", "text": gpt_response["value"]}],
    }
    prompt_messages = [user_message]
    full_messages = [user_message, assistant_message]

    prompt_enc = processor.apply_chat_template(
        prompt_messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
    )
    full_enc = processor.apply_chat_template(
        full_messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=False,
    )

    prompt_ids = prompt_enc["input_ids"]
    full_ids = full_enc["input_ids"]
    # need this to build labels
    response_ids = full_ids[:, prompt_ids.size(1) :]
    full_attention_mask = full_enc.get("attention_mask", torch.ones_like(full_ids, dtype=torch.long))
    input_ids = full_ids.squeeze(0).to(torch.long)
    attention_mask = full_attention_mask.squeeze(0).to(torch.long)
    labels = torch.cat(
        [
            torch.full((prompt_ids.size(1),), IGNORE_INDEX, dtype=torch.long),
            response_ids.squeeze(0).to(torch.long),
        ],
        dim=0,
    )

    return dict(
        input_ids=input_ids,
        labels=labels, # [-100, -100, ...]
        attention_mask=attention_mask,
        pixel_values=full_enc.get("pixel_values", prompt_enc.get("pixel_values")),
        pixel_attention_mask=full_enc.get("pixel_attention_mask", prompt_enc.get("pixel_attention_mask")),
    )


__all__ = [
    "smolvlm_encode_conversation",
]
