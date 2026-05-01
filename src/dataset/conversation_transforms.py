from src.constants import LLAVA_IMAGE_TOKEN


def replace_image_tokens(input_string):
    if LLAVA_IMAGE_TOKEN not in input_string:
        return input_string

    while LLAVA_IMAGE_TOKEN + "\n" in input_string:
        input_string = input_string.replace(LLAVA_IMAGE_TOKEN + "\n", "<image>", 1)

    return input_string


# https://huggingface.co/docs/transformers/en/chat_templating
def llava_to_openai(conversations):
    role_mapping = {"human": "user", "gpt": "assistant"}

    transformed_data = []
    for conversation in conversations:
        transformed_content = replace_image_tokens(conversation["value"])
        transformed_data.append(
            {
                "role": role_mapping.get(conversation["from"], conversation["from"]),
                "content": transformed_content,
            }
        )

    return transformed_data


__all__ = [
    "llava_to_openai",
    "replace_image_tokens",
]
