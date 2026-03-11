import copy
import os
from typing import Dict, Optional
import torch
import transformers
import ujson as json
from torch.utils.data import Dataset
from PIL import Image

from src.params import DataArguments
from src.constants import *
from .data_utils import pad_sequence, encode_video

EOS_TOKEN = "<end_of_utterance>"
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
    return _pad_frames(pixel_values_list, pad_value)

def pad_pixel_attention_masks(mask_list, pad_value=0):
    return _pad_frames(mask_list, pad_value)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path: str | list,
        processor: transformers.ProcessorMixin,
        data_args: DataArguments,
        padding=True,
        teacher_processor: Optional[transformers.ProcessorMixin] = None,
    ):
        super(SupervisedDataset, self).__init__()
        if isinstance(data_path, str):
            list_data_dict = json.load(open(data_path, "r"))
        else:
            list_data_dict = data_path

        self.processor = processor
        self.teacher_processor = teacher_processor
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.padding = padding
        self.max_num_frames = data_args.max_num_frames

    def __len__(self):
        return len(self.list_data_dict)

    @staticmethod
    def _dummy_pixel_tensors():
        return (
            torch.zeros(_DUMMY_PIXEL_VALUES),
            torch.zeros(_DUMMY_PIXEL_MASK),
        )

    def _encode_conversation(
        self,
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

    def qwen_encode_conversation(
        self,
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

            # Encode response — append <|im_end|> so the model learns to stop
            suffix = "" if is_last_turn else "\n"
            response_text = gpt_response['content'] + "<|im_end|>" + suffix
            response_ids  = processor.tokenizer(
                response_text, add_special_tokens=False, return_tensors="pt"
            )["input_ids"]

            input_ids = torch.cat([prompt_ids, response_ids], dim=1).squeeze(0)
            labels    = torch.cat(
                [
                    torch.tensor([IGNORE_INDEX] * len(prompt_ids[0])),
                    response_ids.squeeze(0),
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
            pixel_attention_mask=None,   # Qwen uses image_grid_thw instead
            image_grid_thw=image_grid_thw,
        )

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        is_video = False
        num_frames = None
        images = None

        if "image" in sources:
            image_files = sources["image"]
            image_folder = self.data_args.image_folder
            if isinstance(image_files, str):
                image_files = [image_files]

            images = []
            for image_file in image_files:
                resolved_path = image_file
                if not os.path.exists(resolved_path):
                    resolved_path = os.path.join(image_folder, image_file)
                images.append(Image.open(resolved_path).convert("RGB"))
        elif "video" in sources:
            video_file = sources["video"]
            video_folder = self.data_args.image_folder
            if not os.path.exists(video_file):
                video_file = os.path.join(video_folder, video_file)
            images = encode_video(video_file, self.max_num_frames)
            is_video = True
            num_frames = len(images)

        sources = copy.deepcopy(
            llava_to_openai(sources['conversations'], is_video=is_video, num_frames=num_frames)
        )

        data_dict = self._encode_conversation(sources, images, self.processor)
        if data_dict["pixel_values"] is None:
            pixel_values, pixel_attention_mask = self._dummy_pixel_tensors()
            data_dict["pixel_values"] = pixel_values
            data_dict["pixel_attention_mask"] = pixel_attention_mask

        if self.teacher_processor is None:
            return data_dict

        teacher_processor_name = type(self.teacher_processor).__name__
        is_qwen_teacher = "Qwen" in teacher_processor_name

        if is_qwen_teacher:
            teacher_data = self.qwen_encode_conversation(sources, images, self.teacher_processor)
        else:
            teacher_data = self._encode_conversation(sources, images, self.teacher_processor)

        data_dict["teacher_input_ids"] = teacher_data["input_ids"]
        data_dict["teacher_labels"] = teacher_data["labels"]
        data_dict["teacher_attention_mask"] = teacher_data["attention_mask"]

        if teacher_data["pixel_values"] is None and not is_qwen_teacher:
            pixel_values, pixel_attention_mask = self._dummy_pixel_tensors()
            teacher_data["pixel_values"] = pixel_values
            teacher_data["pixel_attention_mask"] = pixel_attention_mask

        data_dict["teacher_pixel_values"] = teacher_data["pixel_values"]
        data_dict["teacher_pixel_attention_mask"] = teacher_data["pixel_attention_mask"]
        if teacher_data.get("image_grid_thw") is not None:
            data_dict["teacher_image_grid_thw"] = teacher_data["image_grid_thw"]

        return data_dict

class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    def __init__(self, pad_token_id: int, teacher_pad_token_id: Optional[int] = None):
        self.pad_token_id         = pad_token_id
        self.teacher_pad_token_id = teacher_pad_token_id

    def __call__(self, examples):
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

        # Collate teacher inputs when present
        if "teacher_input_ids" in examples[0]:
            teacher_pad = self.teacher_pad_token_id or self.pad_token_id
            teacher_input_ids = pad_sequence(
                [e["teacher_input_ids"] for e in examples],
                padding_side="right",
                padding_value=teacher_pad,
            )
            teacher_labels = pad_sequence(
                [e["teacher_labels"] for e in examples],
                padding_side="right",
                padding_value=IGNORE_INDEX,
            )
            batch_dict.update(
                teacher_input_ids=teacher_input_ids,
                teacher_labels=teacher_labels,
                teacher_attention_mask=teacher_input_ids != teacher_pad,
            )

            if "teacher_pixel_values" in examples[0]:
                teacher_pixel_values = [e["teacher_pixel_values"] for e in examples]
                if teacher_pixel_values[0].dim() == 5:
                    batch_dict["teacher_pixel_values"] = pad_pixel_values(teacher_pixel_values, pad_value=0.0)
                else:
                    batch_dict["teacher_pixel_values"] = torch.cat(teacher_pixel_values, dim=0)

                teacher_pixel_attention_masks = [e.get("teacher_pixel_attention_mask") for e in examples]
                if teacher_pixel_attention_masks[0] is not None:
                    batch_dict["teacher_pixel_attention_mask"] = pad_pixel_attention_masks(
                        teacher_pixel_attention_masks,
                        pad_value=0,
                    )

                if "teacher_image_grid_thw" in examples[0]:
                    batch_dict["teacher_image_grid_thw"] = torch.cat(
                        [e["teacher_image_grid_thw"] for e in examples],
                        dim=0,
                    )

        return batch_dict

def replace_image_tokens(input_string, start_count=1):
    count = start_count

    if LLAVA_IMAGE_TOKEN not in input_string:
        return input_string, count

    while LLAVA_IMAGE_TOKEN+'\n' in input_string:
        input_string = input_string.replace(LLAVA_IMAGE_TOKEN+'\n', "<image>", 1)
        count += 1

    return input_string, count

def video_to_image_tokens(input_string, num_frames):

    frame_tokens = "\n".join([LLAVA_IMAGE_TOKEN] * num_frames)
    input_string = input_string.replace(LLAVA_VIDEO_TOKEN, frame_tokens)

    return input_string

def llava_to_openai(conversations, is_video=False, num_frames=None):

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

def make_supervised_data_module(
    processor,
    data_args,
    teacher_processor: Optional[transformers.ProcessorMixin] = None,
):
    """Make dataset and collator for supervised fine-tuning."""
    sft_dataset = SupervisedDataset(
        data_path=data_args.data_path,
        processor=processor,
        data_args=data_args,
        teacher_processor=teacher_processor,
    )
    teacher_pad = (
        teacher_processor.tokenizer.pad_token_id if teacher_processor is not None else None
    )
    data_collator = DataCollatorForSupervisedDataset(
        pad_token_id=processor.tokenizer.pad_token_id,
        teacher_pad_token_id=teacher_pad,
    )

    return dict(train_dataset=sft_dataset, eval_dataset=None, data_collator=data_collator)
