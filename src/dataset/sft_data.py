import copy
import os
import random
import re
from dataclasses import replace
from typing import Dict, Optional
import torch
import transformers
import ujson as json
from torch.utils.data import Dataset, Subset
from PIL import Image

from src.params import DataArguments
from src.constants import *
from .data_utils import pad_sequence, encode_video
from .data_collator import DataCollatorForSupervisedDataset, pad_pixel_values, pad_pixel_attention_masks

EOS_TOKEN = "<end_of_utterance>"
_DUMMY_PIXEL_VALUES = (1, 13, 3, 384, 384)
_DUMMY_PIXEL_MASK = (1, 13, 384, 384)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path: str | list,
        processor: transformers.ProcessorMixin,
        data_args: DataArguments,
        padding=True,
        teacher_processors: Optional[list[transformers.ProcessorMixin]] = None,
        teacher_processor: Optional[transformers.ProcessorMixin] = None,
    ):
        super(SupervisedDataset, self).__init__()
        if isinstance(data_path, str):
            list_data_dict = json.load(open(data_path, "r"))
        else:
            list_data_dict = data_path

        self.processor = processor
        self.teacher_processors = list(teacher_processors or [])
        if teacher_processor is not None:
            self.teacher_processors.append(teacher_processor)
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

    def gemma3_encode_conversation(
        self,
        sources,
        images,
        processor: transformers.ProcessorMixin,
    ) -> Dict[str, torch.Tensor]:
        """Encode a conversation with a Gemma 3 processor.

        Gemma 3 requires multimodal prompts to be built through
        ``apply_chat_template`` so image placeholders are injected before the
        processor receives image tensors.
        """
        all_input_ids = []
        all_labels = []

        pixel_values = None
        image_idx = 0

        for j in range(0, len(sources), 2):
            user_input = sources[j]
            gpt_response = sources[j + 1]

            user_text = user_input["content"]
            has_image = LLAVA_IMAGE_TOKEN in user_text and images is not None
            n_images = user_text.count(LLAVA_IMAGE_TOKEN)
            clean_text = user_text.replace(LLAVA_IMAGE_TOKEN, "").strip()

            user_content = []
            if has_image:
                turn_images = images[image_idx: image_idx + n_images]
                image_idx += n_images
                user_content.extend({"type": "image", "image": image} for image in turn_images)
            else:
                turn_images = None

            if clean_text:
                user_content.append({"type": "text", "text": clean_text})
            if not user_content:
                user_content = [{"type": "text", "text": ""}]

            prompt_messages = [{"role": "user", "content": user_content}]
            full_messages = [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": [{"type": "text", "text": gpt_response["content"]}]},
            ]

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
            if prompt_ids.size(1) > full_ids.size(1):
                raise ValueError(
                    "Gemma 3 prompt encoding is longer than full conversation encoding."
                )

            response_ids = full_ids[:, prompt_ids.size(1):]
            input_ids = full_ids.squeeze(0)
            labels = torch.cat(
                [
                    torch.full((prompt_ids.size(1),), IGNORE_INDEX, dtype=torch.long),
                    response_ids.squeeze(0).to(torch.long),
                ],
                dim=0,
            )

            all_input_ids.append(input_ids.to(torch.long))
            all_labels.append(labels)

            if has_image:
                pixel_values = full_enc.get("pixel_values", prompt_enc.get("pixel_values"))

        input_ids = torch.cat(all_input_ids, dim=0).to(torch.long)
        labels = torch.cat(all_labels, dim=0).to(torch.long)
        attention_mask = torch.ones_like(input_ids)

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            pixel_attention_mask=None,
        )

    def _encode_teacher_data(
        self,
        sources,
        images,
        teacher_processor,
    ) -> Dict[str, torch.Tensor]:
        if isinstance(teacher_processor, dict):
            tokenizer = teacher_processor["tokenizer"]
            image_processor = teacher_processor["image_processor"]
            num_image_token = teacher_processor.get("num_image_token", 256)
            img_start_token = teacher_processor.get("img_start_token", "<img>")
            img_end_token = teacher_processor.get("img_end_token", "</img>")
            img_context_token = teacher_processor.get("img_context_token", "<IMG_CONTEXT>")

            all_input_ids = [torch.tensor([tokenizer.bos_token_id or 1])]
            all_labels = [torch.tensor([IGNORE_INDEX])]
            pixel_values = None
            image_flags = None

            for idx, j in enumerate(range(0, len(sources), 2)):
                user_input = sources[j]
                gpt_response = sources[j + 1]
                is_last_turn = (idx == (len(sources) // 2 - 1))

                if user_input["content"].startswith(LLAVA_IMAGE_TOKEN):
                    user_prompt = f"User:{user_input['content']}{EOS_TOKEN}\nAssistant: "
                else:
                    user_prompt = f"User: {user_input['content']}{EOS_TOKEN}\nAssistant: "

                gpt_prompt = (
                    f"{gpt_response['content']}{EOS_TOKEN}"
                    if is_last_turn
                    else f"{gpt_response['content']}{EOS_TOKEN}\n"
                )

                if LLAVA_IMAGE_TOKEN in user_prompt and images is not None:
                    pixel_values = image_processor(images=images, return_tensors="pt").pixel_values
                    if pixel_values.dim() == 3:
                        pixel_values = pixel_values.unsqueeze(0)
                    num_patches = pixel_values.shape[0]
                    image_tokens = (
                        img_start_token
                        + img_context_token * (num_image_token * num_patches)
                        + img_end_token
                    )
                    user_prompt = user_prompt.replace(LLAVA_IMAGE_TOKEN, image_tokens)
                    image_flags = torch.ones((num_patches, 1), dtype=torch.long)

                prompt_input_ids = tokenizer(
                    user_prompt, add_special_tokens=False, return_tensors="pt"
                )["input_ids"]
                response_input_ids = tokenizer(
                    gpt_prompt, add_special_tokens=False, return_tensors="pt"
                )["input_ids"]

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

            teacher_data = dict(
                input_ids=torch.cat(all_input_ids, dim=0).to(torch.long),
                labels=torch.cat(all_labels, dim=0).to(torch.long),
                attention_mask=None,
                pixel_values=pixel_values,
                pixel_attention_mask=None,
                image_flags=image_flags,
            )
        elif "Gemma3" in type(teacher_processor).__name__:
            teacher_data = self.gemma3_encode_conversation(sources, images, teacher_processor)
        elif "Qwen" in type(teacher_processor).__name__:
            teacher_data = self.qwen_encode_conversation(sources, images, teacher_processor)
        else:
            teacher_data = self._encode_conversation(sources, images, teacher_processor)

        if teacher_data["attention_mask"] is None:
            teacher_data["attention_mask"] = torch.ones_like(teacher_data["input_ids"])

        if teacher_data["pixel_values"] is None and isinstance(teacher_processor, dict):
            teacher_data["pixel_values"] = torch.zeros((1, 3, 448, 448))
            teacher_data["image_flags"] = torch.zeros((1, 1), dtype=torch.long)
        elif teacher_data["pixel_values"] is None and "Qwen" not in type(teacher_processor).__name__:
            pixel_values, pixel_attention_mask = self._dummy_pixel_tensors()
            teacher_data["pixel_values"] = pixel_values
            teacher_data["pixel_attention_mask"] = pixel_attention_mask

        return teacher_data

    @staticmethod
    def _teacher_prefix(index: int, count: int) -> str:
        if count == 1:
            return "teacher"
        return f"teacher_{index}"

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

        if not self.teacher_processors:
            return data_dict

        teacher_count = len(self.teacher_processors)
        for teacher_index, teacher_processor in enumerate(self.teacher_processors):
            teacher_data = self._encode_teacher_data(sources, images, teacher_processor)
            prefix = self._teacher_prefix(teacher_index, teacher_count)

            data_dict[f"{prefix}_input_ids"] = teacher_data["input_ids"]
            data_dict[f"{prefix}_labels"] = teacher_data["labels"]
            data_dict[f"{prefix}_attention_mask"] = teacher_data["attention_mask"]
            data_dict[f"{prefix}_pixel_values"] = teacher_data["pixel_values"]
            data_dict[f"{prefix}_pixel_attention_mask"] = teacher_data["pixel_attention_mask"]
            if teacher_data.get("image_grid_thw") is not None:
                data_dict[f"{prefix}_image_grid_thw"] = teacher_data["image_grid_thw"]
            if teacher_data.get("image_flags") is not None:
                data_dict[f"{prefix}_image_flags"] = teacher_data["image_flags"]

        return data_dict

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


def subset_dataset(dataset, subset_size: Optional[int], data_path: Optional[str]):
    if subset_size is None:
        return dataset
    if subset_size <= 0:
        raise ValueError("--train_subset_size and --eval_subset_size must be > 0 when set.")

    subset_size = min(subset_size, len(dataset))
    dataset_name = str(data_path).lower()
    rng = random.Random(42)
    if "chartqa" in dataset_name and hasattr(dataset, "list_data_dict"):
        split_key = None
        for candidate in ("chartqa_split", "split"):
            if dataset.list_data_dict and all(candidate in row for row in dataset.list_data_dict):
                split_key = candidate
                break
        if split_key is not None:
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
    elif any(name in dataset_name for name in ("docvqa", "textvqa")):
        indices = sorted(rng.sample(range(len(dataset)), subset_size))
    else:
        indices = range(subset_size)
    return Subset(dataset, indices)

def make_supervised_data_module(
    processor,
    data_args,
    teacher_processors: Optional[list[transformers.ProcessorMixin]] = None,
    teacher_processor: Optional[transformers.ProcessorMixin] = None,
):
    """Make dataset and collator for supervised fine-tuning."""
    normalized_teacher_processors = list(teacher_processors or [])
    if teacher_processor is not None:
        normalized_teacher_processors.append(teacher_processor)
    sft_dataset = SupervisedDataset(
        data_path=data_args.data_path,
        processor=processor,
        data_args=data_args,
        teacher_processors=normalized_teacher_processors,
    )
    sft_dataset = subset_dataset(
        sft_dataset,
        data_args.train_subset_size,
        data_args.data_path,
    )
    eval_dataset = None
    if data_args.eval_data_path:
        eval_dataset = SupervisedDataset(
            data_path=data_args.eval_data_path,
            processor=processor,
            data_args=replace(data_args, data_path=data_args.eval_data_path),
            teacher_processors=normalized_teacher_processors,
        )
        eval_dataset = subset_dataset(
            eval_dataset,
            data_args.eval_subset_size,
            data_args.eval_data_path,
        )
    teacher_pad = None
    if len(normalized_teacher_processors) == 1:
        if isinstance(normalized_teacher_processors[0], dict):
            teacher_pad = normalized_teacher_processors[0]["tokenizer"].pad_token_id
        elif hasattr(normalized_teacher_processors[0], "tokenizer"):
            teacher_pad = normalized_teacher_processors[0].tokenizer.pad_token_id
        else:
            teacher_pad = normalized_teacher_processors[0].pad_token_id
    teacher_pad_ids = []
    for teacher_processor in normalized_teacher_processors:
        if isinstance(teacher_processor, dict):
            teacher_pad_ids.append(teacher_processor["tokenizer"].pad_token_id)
        elif hasattr(teacher_processor, "tokenizer"):
            teacher_pad_ids.append(teacher_processor.tokenizer.pad_token_id)
        else:
            teacher_pad_ids.append(teacher_processor.pad_token_id)
    data_collator = DataCollatorForSupervisedDataset(
        pad_token_id=processor.tokenizer.pad_token_id,
        teacher_pad_token_id=teacher_pad,
        teacher_pad_token_ids=teacher_pad_ids,
    )

    return dict(train_dataset=sft_dataset, eval_dataset=eval_dataset, data_collator=data_collator)
