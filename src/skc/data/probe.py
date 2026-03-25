import os
import tempfile

import torch
from torch.utils.data import DataLoader, Dataset

from ..config.families import LLAVA_FAMILIES, TOKENIZER_ONLY_FAMILIES

EXTRA_BATCH_KEYS = (
    "pixel_values",
    "images",
    "images_seq_mask",
    "images_emb_mask",
    "images_spatial_crop",
    "image_flags",
    "image_sizes",
    "image_grid_thw",
    "video_grid_thw",
)


class ProbeDataset(Dataset):
    def __init__(self, probe_samples, processor, family: str, model=None):
        self.probe_samples = probe_samples
        self.processor = processor
        self.family = family
        self.model = model
        self._qwen_vl_image_paths = {}
        self.init_family_state()

    def __len__(self):
        return len(self.probe_samples)

    @staticmethod
    def batch_get(batch, key):
        if isinstance(batch, dict):
            return batch.get(key)
        return getattr(batch, key, None)

    def init_family_state(self) -> None:
        if self.family != "qwen_vl":
            return

        image_dir = tempfile.mkdtemp(prefix="skc_qwen_vl_")
        for index, sample in enumerate(self.probe_samples):
            img_path = os.path.join(image_dir, f"sample_{index}.png")
            sample["image"].save(img_path, format="PNG")
            self._qwen_vl_image_paths[index] = img_path

    def encode_with_chat_template(self, image, question):
        if not hasattr(self.processor, "apply_chat_template"):
            return None

        conversation_inline = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            }
        ]
        try:
            encoded = self.processor.apply_chat_template(
                conversation_inline,
                tokenize=True,
                add_generation_prompt=False,
                return_dict=True,
                return_tensors="pt",
            )
            if self.batch_get(encoded, "input_ids") is not None:
                return encoded
        except Exception:
            pass

        conversation_prompt = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            }
        ]
        try:
            prompt = self.processor.apply_chat_template(
                conversation_prompt,
                tokenize=False,
                add_generation_prompt=False,
            )
            encoded = self.processor(images=image, text=prompt, return_tensors="pt")
            if self.batch_get(encoded, "input_ids") is not None:
                return encoded
        except Exception:
            pass

        return None

    def attach_answer_token_mask(self, batch, prompt_length: int):
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask")
        full_length = int(input_ids.shape[1])
        start = max(0, min(int(prompt_length), full_length))

        answer_token_mask = torch.zeros_like(input_ids, dtype=torch.long)
        answer_token_mask[:, start:] = 1
        if attention_mask is not None:
            answer_token_mask = answer_token_mask * attention_mask.to(torch.long)

        if int(answer_token_mask.sum().item()) == 0:
            if attention_mask is not None:
                last_positions = attention_mask.to(torch.long).sum(dim=1).clamp_min(1) - 1
            else:
                last_positions = torch.full(
                    (input_ids.shape[0],),
                    full_length - 1,
                    dtype=torch.long,
                    device=input_ids.device,
                )
            answer_token_mask.scatter_(1, last_positions.unsqueeze(1), 1)

        labels = torch.full_like(input_ids, -100)
        labels[answer_token_mask.bool()] = input_ids[answer_token_mask.bool()]
        batch["labels"] = labels
        batch["answer_token_mask"] = answer_token_mask
        return batch

    def encode_answer_supervision(self, image, question, answer):
        if not answer or not hasattr(self.processor, "apply_chat_template"):
            return None

        prompt_inline = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            }
        ]
        full_inline = prompt_inline + [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": answer}],
            }
        ]

        try:
            prompt_encoded = self.processor.apply_chat_template(
                prompt_inline,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            full_encoded = self.processor.apply_chat_template(
                full_inline,
                tokenize=True,
                add_generation_prompt=False,
                return_dict=True,
                return_tensors="pt",
            )
            prompt_input_ids = self.batch_get(prompt_encoded, "input_ids")
            full_input_ids = self.batch_get(full_encoded, "input_ids")
            if prompt_input_ids is not None and full_input_ids is not None:
                batch = self.build_model_batch(full_encoded)
                return self.attach_answer_token_mask(batch, int(prompt_input_ids.shape[1]))
        except Exception:
            pass

        prompt_conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            }
        ]
        full_conversation = prompt_conversation + [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": answer}],
            }
        ]

        try:
            prompt_text = self.processor.apply_chat_template(
                prompt_conversation,
                tokenize=False,
                add_generation_prompt=True,
            )
            full_text = self.processor.apply_chat_template(
                full_conversation,
                tokenize=False,
                add_generation_prompt=False,
            )
            prompt_encoded = self.processor(images=image, text=prompt_text, return_tensors="pt")
            full_encoded = self.processor(images=image, text=full_text, return_tensors="pt")
            prompt_input_ids = self.batch_get(prompt_encoded, "input_ids")
            full_input_ids = self.batch_get(full_encoded, "input_ids")
            if prompt_input_ids is not None and full_input_ids is not None:
                batch = self.build_model_batch(full_encoded)
                return self.attach_answer_token_mask(batch, int(prompt_input_ids.shape[1]))
        except Exception:
            pass

        return None

    @staticmethod
    def build_text_batch(model_inputs):
        return {
            "input_ids": model_inputs["input_ids"],
            "attention_mask": model_inputs.get("attention_mask"),
        }

    def encode_text_only(self, tokenizer, text):
        return self.build_text_batch(tokenizer(text, return_tensors="pt"))

    def encode_text_and_image(self, prompt, image):
        return self.processor(
            text=prompt,
            images=[image],
            return_tensors="pt",
        )

    def encode_phi4_multimodal(self, image, question):
        prompt = f"<|user|><|image_1|>{question}<|end|><|assistant|>"
        return self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
        )

    def encode_conversation(self, conversation, image, **kwargs):
        return self.processor(
            conversations=conversation,
            images=[image],
            force_batchify=True,
            **kwargs,
        )

    def encode_deepseek_vl2(self, image, question):
        conversation = [
            {
                "role": "<|User|>",
                "content": f"<image>\n{question}",
                "images": ["image_0"],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]
        return self.encode_conversation(conversation, image, system_prompt="")

    def encode_qwen_vl(self, index, question):
        tokenizer = self.processor
        image_path = self._qwen_vl_image_paths.get(index)
        if hasattr(tokenizer, "from_list_format") and image_path is not None:
            query = tokenizer.from_list_format(
                [{"image": image_path}, {"text": question}]
            )
        else:
            query = question

        return self.encode_text_only(tokenizer, query)

    def encode_tokenizer_only(self, question):
        return self.encode_text_only(self.processor, question)

    def encode_internvl_chat(self, image, question):
        tokenizer = self.processor["tokenizer"]
        image_processor = self.processor["image_processor"]
        num_image_token = self.processor["num_image_token"]
        img_start_token = self.processor["img_start_token"]
        img_end_token = self.processor["img_end_token"]
        img_context_token = self.processor["img_context_token"]

        pixel_values = image_processor(images=image, return_tensors="pt").pixel_values
        if pixel_values.dim() == 3:
            pixel_values = pixel_values.unsqueeze(0)
        num_patches = pixel_values.shape[0]

        query = f"<image>\n{question}"
        image_tokens = img_start_token + img_context_token * (num_image_token * num_patches) + img_end_token
        query = query.replace("<image>", image_tokens, 1)

        return {
            **self.build_text_batch(tokenizer(query, return_tensors="pt")),
            "pixel_values": pixel_values,
            "image_flags": torch.ones((num_patches, 1), dtype=torch.long),
        }

    def build_llava_prompt(self, question):
        if hasattr(self.processor, "apply_chat_template"):
            return self.processor.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": question},
                        ],
                    }
                ],
                tokenize=False,
                add_generation_prompt=False,
            )
        return f"<image>\n{question}"

    def encode_llava_family(self, image, question):
        prompt = self.build_llava_prompt(question)
        return self.encode_text_and_image(prompt, image)

    def encode_default_family(self, image, question):
        encoded = self.encode_with_chat_template(image, question)
        if encoded is not None:
            return encoded
        return self.encode_text_and_image(f"<image>\n{question}", image)

    def build_model_batch(self, encoded):
        input_ids = self.batch_get(encoded, "input_ids")
        if input_ids is None:
            raise ValueError(f"Processor for family '{self.family}' did not return input_ids.")

        input_ids = input_ids.to(torch.long)
        attention_mask = self.batch_get(encoded, "attention_mask")
        batch = {
            "input_ids": input_ids,
            "labels": torch.full_like(input_ids, -100),
            "attention_mask": (
                attention_mask.to(torch.long)
                if attention_mask is not None
                else torch.ones_like(input_ids, dtype=torch.long)
            ),
        }

        for key in EXTRA_BATCH_KEYS:
            value = self.batch_get(encoded, key)
            if value is not None:
                batch[key] = value

        encoded_items = encoded.items() if hasattr(encoded, "items") else []
        for key, value in encoded_items:
            if key in {"input_ids", "attention_mask"}:
                continue
            if value is not None:
                batch[key] = value

        return batch

    def encode_sample(self, index, image, question):
        answer = self.probe_samples[index].get("answer")
        if answer is not None:
            answer_batch = self.encode_answer_supervision(image, question, answer)
            if answer_batch is not None:
                return answer_batch

        if self.family == "deepseek_vl2":
            return self.encode_deepseek_vl2(image, question)
        if self.family == "qwen_vl":
            return self.encode_qwen_vl(index, question)
        if self.family == "phi4_multimodal":
            return self.encode_phi4_multimodal(image, question)
        if self.family in TOKENIZER_ONLY_FAMILIES:
            return self.encode_tokenizer_only(question)
        if self.family == "internvl_chat":
            return self.encode_internvl_chat(image, question)
        if self.family in LLAVA_FAMILIES:
            return self.encode_llava_family(image, question)
        return self.encode_default_family(image, question)

    def __getitem__(self, index):
        sample = self.probe_samples[index]
        question = sample["question"]
        image = sample["image"]
        encoded = self.encode_sample(index, image, question)
        return self.build_model_batch(encoded)


def build_loader(processor, probe_samples, family, model=None) -> DataLoader:
    dataset = ProbeDataset(probe_samples, processor, family, model=model)
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda batch: batch[0],
        num_workers=0,
    )
