from .base import BaseModel
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText


class LFM2VL(BaseModel):
    INTERLEAVE = True

    def __init__(self, model_path, **kwargs):
        self.default_instruction_prompt = (
            "\nPlease answer directly with only the final answer, "
            "do not give any explanation."
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        attn_implementation = "flash_attention_2" if self.device == "cuda" else "eager"
        torch_dtype = torch.bfloat16 if self.device == "cuda" else torch.float32

        self.processor = AutoProcessor.from_pretrained(
            model_path,
            max_image_tokens=256,
            trust_remote_code=True,
        )
        self.model = (
            AutoModelForImageTextToText.from_pretrained(
                model_path,
                attn_implementation=attn_implementation,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
            )
            .to(self.device)
            .eval()
        )

        kwargs_default = {"max_new_tokens": 1024, "use_cache": True}
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default

    def custom_instruction_prompt_by_dataset(self, dataset):
        if dataset == "MathVista_MINI" or dataset == "MM-IFEval" or dataset == "MMVet":
            return ""
        else:
            return self.default_instruction_prompt

    def message_to_chat_messages(self, message, instruction_prompt, dataset):
        single_turn_messages = []

        for item in message:
            if item["type"] == "image":
                single_turn_messages.append(
                    {"type": "image", "image": Image.open(item["value"]).convert("RGB")}
                )
            elif item["type"] == "text":
                single_turn_messages.append({"type": "text", "text": item["value"]})
        if instruction_prompt:
            single_turn_messages.append({"type": "text", "text": instruction_prompt})

        if dataset == "MM-IFEval":
            # move images to the beginning of the conversation list, in the same order as they appear in the message
            index = 0
            image_index = 0
            while index < len(single_turn_messages):
                if single_turn_messages[index]["type"] == "image":
                    single_turn_messages.insert(
                        image_index, single_turn_messages.pop(index)
                    )
                    image_index += 1
                index += 1

        return [{"role": "user", "content": single_turn_messages}]

    def _move_inputs(self, inputs):
        moved = {}
        for key, value in inputs.items():
            if torch.is_tensor(value):
                tensor_kwargs = {"device": self.device}
                if value.is_floating_point():
                    tensor_kwargs["dtype"] = self.model.dtype
                moved[key] = value.to(**tensor_kwargs)
            else:
                moved[key] = value
        return moved

    def generate_inner(self, message, dataset=None):
        instruction_prompt = self.custom_instruction_prompt_by_dataset(dataset)

        chat_messages = self.message_to_chat_messages(message, instruction_prompt, dataset)
        generation_inputs = self.processor.apply_chat_template(
            chat_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        generation_inputs = self._move_inputs(generation_inputs)
        input_len = generation_inputs["input_ids"].shape[-1]

        history = self.model.generate(**generation_inputs, **self.kwargs)
        generated_tokens = history[:, input_len:]
        return self.processor.batch_decode(generated_tokens, skip_special_tokens=True)[0].strip()

    def chat_inner(self, message, dataset=None):
        return self.generate_inner(message, dataset)
