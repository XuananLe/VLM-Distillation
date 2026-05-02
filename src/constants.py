IGNORE_INDEX = -100

LLAVA_IMAGE_TOKEN = "<image>"

# Byte values live in [0, 255], so 256 is a clean end-of-token marker.
EOS_SENTINEL = 256

TOKENIZER_VOCAB_SIZES = {
    "HuggingFaceTB/SmolVLM-256M-Instruct": 49280,
    "HuggingFaceTB/SmolVLM-500M-Instruct": 49280,
    "OpenGVLab/InternVL2-1B": 151655,
    "Qwen/Qwen2.5-VL-3B-Instruct": 151665,
    "Qwen/Qwen2-VL-2B-Instruct": 151657,
    "ibm-granite/granite-vision-3.1-2b-preview": 49156,
    "google/gemma-3-4b-it": 262145,
}
