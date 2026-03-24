"""Encoding utilities for processing conversations and images."""
import torch
import transformers
from typing import Dict

from src.constants import LLAVA_IMAGE_TOKEN, IGNORE_INDEX

EOS_TOKEN = "<end_of_utterance>"


def encode_smolvlm_conversation(
    sources,
    images,
    processor: transformers.ProcessorMixin,
) -> Dict[str, torch.Tensor]:
    """Encode a conversation with the given processor.

    Returns input_ids, labels, attention_mask, pixel_values,
    pixel_attention_mask.  pixel_values / pixel_attention_mask are None
    when there are no images in the sample (caller sets dummy tensors).
    """
    all_input_ids = [torch.tensor([1])]   # bos token id
    all_labels    = [torch.tensor([-100])] # mask bos token

    pixel_values          = None
    pixel_attention_mask  = None

    for idx, j in enumerate(range(0, len(sources), 2)):
        user_input   = sources[j]
        gpt_response = sources[j + 1]
        is_last_turn = (idx == (len(sources) // 2 - 1))

        if user_input['content'].startswith(LLAVA_IMAGE_TOKEN):
            user_prompt = f"User:{user_input['content']}{EOS_TOKEN}\nAssistant: "
        else:
            user_prompt = f"User: {user_input['content']}{EOS_TOKEN}\nAssistant: "

        gpt_prompt = (
            f"{gpt_response['content']}{EOS_TOKEN}"
            if is_last_turn
            else f"{gpt_response['content']}{EOS_TOKEN}\n"
        )

        if LLAVA_IMAGE_TOKEN in user_prompt:
            enc = processor(text=user_prompt, images=images, return_tensors='pt')
            prompt_input_ids     = enc['input_ids']
            pixel_values         = enc.get('pixel_values', None)
            pixel_attention_mask = enc.get('pixel_attention_mask', None)
        else:
            prompt_input_ids = processor.tokenizer(
                user_prompt, add_special_tokens=False, return_tensors='pt'
            )['input_ids']

        response_input_ids = processor.tokenizer(
            gpt_prompt, add_special_tokens=False, return_tensors='pt'
        )['input_ids']

        input_ids = torch.cat([prompt_input_ids, response_input_ids], dim=1).squeeze(0)
        labels = torch.cat(
            [
                torch.tensor([IGNORE_INDEX] * len(prompt_input_ids[0])),
                response_input_ids.squeeze(0),
            ],
            dim=0,
        )

        all_input_ids.append(input_ids)
        all_labels.append(labels)

    input_ids      = torch.cat(all_input_ids, dim=0).to(torch.long)
    labels         = torch.cat(all_labels,    dim=0).to(torch.long)
    attention_mask = torch.ones_like(input_ids)

    return dict(
        input_ids=input_ids,
        labels=labels,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        pixel_attention_mask=pixel_attention_mask,
    )


def encode_qwen_conversation(
    sources,
    images,
    processor: transformers.ProcessorMixin,
) -> Dict[str, torch.Tensor]:
    """Encode a conversation with a Qwen2.5-VL / Qwen3-VL processor.

    Uses ``apply_chat_template`` for correct image-token injection and
    returns ``image_grid_thw`` instead of ``pixel_attention_mask``.
    """
    all_input_ids = []
    all_labels    = []

    pixel_values   = None
    image_grid_thw = None
    image_idx      = 0

    for idx, j in enumerate(range(0, len(sources), 2)):
        user_input   = sources[j]
        gpt_response = sources[j + 1]
        is_last_turn = (idx == (len(sources) // 2 - 1))

        user_text = user_input['content']
        has_image = LLAVA_IMAGE_TOKEN in user_text and images is not None
        n_images  = user_text.count(LLAVA_IMAGE_TOKEN)
        clean_text = user_text.replace(LLAVA_IMAGE_TOKEN, "").strip()

        # Build Qwen content list for this user turn
        if has_image:
            turn_images  = images[image_idx: image_idx + n_images]
            image_idx   += n_images
            user_content = [{"type": "image"}] * n_images + [{"type": "text", "text": clean_text}]
        else:
            turn_images  = None
            user_content = clean_text

        # Encode prompt (everything up to and including <|im_start|>assistant\n)
        prompt_text = processor.apply_chat_template(
            [{"role": "user", "content": user_content}],
            tokenize=False,
            add_generation_prompt=True,
        )
        if has_image:
            enc             = processor(text=[prompt_text], images=turn_images, return_tensors="pt")
            prompt_ids      = enc["input_ids"]
            pixel_values    = enc.get("pixel_values", None)
            image_grid_thw  = enc.get("image_grid_thw", None)
        else:
            prompt_ids = processor.tokenizer(
                prompt_text, add_special_tokens=False, return_tensors="pt"
            )["input_ids"]

        # Encode response (just the assistant's reply + optional endoftext token)
        gpt_content = gpt_response["content"]
        response_text = processor.apply_chat_template(
            [{"role": "user", "content": "x"}, {"role": "assistant", "content": gpt_content}],
            tokenize=False,
            add_generation_prompt=False,
        )
        response_start_idx = response_text.find(gpt_content)
        if response_start_idx == -1:
            raise ValueError(f"Could not find assistant response '{gpt_content}' in template output.")

        response_only = response_text[response_start_idx:]
        response_ids = processor.tokenizer(
            response_only, add_special_tokens=False, return_tensors="pt"
        )["input_ids"]

        prompt_ids   = prompt_ids.squeeze(0)
        response_ids = response_ids.squeeze(0)
        input_ids    = torch.cat([prompt_ids, response_ids], dim=0)
        labels       = torch.cat(
            [torch.full_like(prompt_ids, IGNORE_INDEX), response_ids],
            dim=0,
        )

        all_input_ids.append(input_ids)
        all_labels.append(labels)

    input_ids      = torch.cat(all_input_ids, dim=0).to(torch.long)
    labels         = torch.cat(all_labels,    dim=0).to(torch.long)
    attention_mask = torch.ones_like(input_ids)

    return dict(
        input_ids=input_ids,
        labels=labels,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
    )
