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

def pad_pixel_values(pixel_values_list, pad_value=0.0):
    batch_size = len(pixel_values_list)
    frame_lengths = [pv.shape[1] for pv in pixel_values_list]
    T_max = max(frame_lengths)
    _, _, C, H, W = pixel_values_list[0].shape
    dtype = pixel_values_list[0].dtype
    device = pixel_values_list[0].device

    output = torch.full((batch_size, T_max, C, H, W),
                        fill_value=pad_value,
                        dtype=dtype,
                        device=device)

    for i, pv in enumerate(pixel_values_list):
        t_i = pv.shape[1]
        output[i, :t_i] = pv[0]
    return output

def pad_pixel_attention_masks(mask_list, pad_value=0):
    batch_size = len(mask_list)
    frame_lengths = [mask.shape[1] for mask in mask_list]
    T_max = max(frame_lengths)

    _, _, H, W = mask_list[0].shape
    dtype = mask_list[0].dtype
    device = mask_list[0].device

    output = torch.full(
        (batch_size, T_max, H, W),
        fill_value=pad_value,
        dtype=dtype,
        device=device
    )

    for i, m in enumerate(mask_list):
        t_i = m.shape[1]
        output[i, :t_i] = m[0]

    return output

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
        attention_mask = (input_ids > -1000000).to(torch.long)

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            pixel_attention_mask=pixel_attention_mask,
        )

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources   = self.list_data_dict[i]
        is_video  = False
        num_frames = None
        images    = None

        if "image" in sources:
            image_files  = sources["image"]
            image_folder = self.data_args.image_folder
            if isinstance(image_files, str):
                image_files = [image_files]
            images = []
            for image_file in image_files:
                if not os.path.exists(image_file):
                    image_file = os.path.join(image_folder, image_file)
                images.append(Image.open(image_file).convert("RGB"))

        elif "video" in sources:
            video_file   = sources["video"]
            video_folder = self.data_args.image_folder
            if not os.path.exists(video_file):
                video_file = os.path.join(video_folder, video_file)
            images     = encode_video(video_file, self.max_num_frames)
            is_video   = True
            num_frames = len(images)

        sources = copy.deepcopy(
            llava_to_openai(sources['conversations'], is_video=is_video, num_frames=num_frames)
        )

        data_dict = self._encode_conversation(sources, images, self.processor)

        # Fill in dummy pixel tensors for the no-image case (required by DeepSpeed ZeRO-3
        # so that every sample has the same activation-graph shape within a batch).
        if data_dict['pixel_values'] is None:
            data_dict['pixel_values']         = torch.zeros(1, 13, 3, 384, 384)
            data_dict['pixel_attention_mask'] = torch.zeros(1, 13, 384, 384)

        # Encode the same sample with the teacher's processor when doing
        # cross-model distillation.  The teacher may have a different patch_size
        # or image-token configuration, so it must process images independently.
        if self.teacher_processor is not None:
            teacher_data = self._encode_conversation(sources, images, self.teacher_processor)

            if teacher_data['pixel_values'] is None:
                teacher_data['pixel_values']         = torch.zeros(1, 13, 3, 384, 384)
                teacher_data['pixel_attention_mask'] = torch.zeros(1, 13, 384, 384)

            data_dict["teacher_input_ids"]            = teacher_data["input_ids"]
            data_dict["teacher_labels"]               = teacher_data["labels"]
            data_dict["teacher_attention_mask"]       = teacher_data["attention_mask"]
            data_dict["teacher_pixel_values"]         = teacher_data["pixel_values"]
            data_dict["teacher_pixel_attention_mask"] = teacher_data["pixel_attention_mask"]

        return data_dict

class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    def __init__(self, pad_token_id: int, teacher_pad_token_id: Optional[int] = None):
        self.pad_token_id         = pad_token_id
        self.teacher_pad_token_id = teacher_pad_token_id

    def __call__(self, examples):
        batch_input_ids           = []
        batch_label_ids           = []
        batch_pixel_values        = []
        batch_pixel_attention_mask = []

        for example in examples:
            batch_input_ids.append(example["input_ids"])
            batch_label_ids.append(example["labels"])
            batch_pixel_values.append(example.get("pixel_values"))
            batch_pixel_attention_mask.append(example.get("pixel_attention_mask"))

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
                padding_side='right',
                padding_value=teacher_pad,
            )
            teacher_labels = pad_sequence(
                [e["teacher_labels"] for e in examples],
                padding_side='right',
                padding_value=IGNORE_INDEX,
            )
            teacher_pixel_values = pad_pixel_values(
                [e["teacher_pixel_values"] for e in examples], pad_value=0.0
            )
            teacher_pixel_attention_mask = pad_pixel_attention_masks(
                [e["teacher_pixel_attention_mask"] for e in examples], pad_value=0
            )
            batch_dict.update(
                teacher_input_ids=teacher_input_ids,
                teacher_labels=teacher_labels,
                teacher_attention_mask=(teacher_input_ids != teacher_pad),
                teacher_pixel_values=teacher_pixel_values,
                teacher_pixel_attention_mask=teacher_pixel_attention_mask,
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
