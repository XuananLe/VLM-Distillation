from src.constants import LLAVA_IMAGE_TOKEN


def replace_image_tokens(input_string, start_count=1):
    """Replace LLaVA image markers with OpenAI-style image placeholders."""
    count = start_count

    if LLAVA_IMAGE_TOKEN not in input_string:
        return input_string, count

    while LLAVA_IMAGE_TOKEN + "\n" in input_string:
        input_string = input_string.replace(LLAVA_IMAGE_TOKEN + "\n", "<image>", 1)
        count += 1

    return input_string, count

def llava_to_openai(conversations):
    """Convert LLaVA-style conversation dicts into the repo's OpenAI-style message format."""
    role_mapping = {"human": "user", "gpt": "assistant"}

    transformed_data = []
    image_count = 1
    for conversation in conversations:
        transformed_content, image_count = replace_image_tokens(
            conversation["value"],
            image_count,
        )
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
