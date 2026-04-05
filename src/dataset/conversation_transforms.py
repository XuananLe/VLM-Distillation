from src.constants import LLAVA_IMAGE_TOKEN, LLAVA_VIDEO_TOKEN


def replace_image_tokens(input_string, start_count=1):
    count = start_count

    if LLAVA_IMAGE_TOKEN not in input_string:
        return input_string, count

    while LLAVA_IMAGE_TOKEN + "\n" in input_string:
        input_string = input_string.replace(LLAVA_IMAGE_TOKEN + "\n", "<image>", 1)
        count += 1

    return input_string, count


def video_to_image_tokens(input_string, num_frames):
    frame_tokens = "\n".join([LLAVA_IMAGE_TOKEN] * num_frames)
    return input_string.replace(LLAVA_VIDEO_TOKEN, frame_tokens)


def llava_to_openai(conversations, is_video=False, num_frames=None):
    role_mapping = {"human": "user", "gpt": "assistant"}

    transformed_data = []
    image_count = 1
    for conversation in conversations:
        if is_video:
            conversation["value"] = video_to_image_tokens(conversation["value"], num_frames)

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
    "video_to_image_tokens",
]
