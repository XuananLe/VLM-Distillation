import json
import os
import subprocess

import modal


MINUTES = 60

QWEN_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
QWEN_SERVED_MODEL_NAME = "qwen2.5-vl-3b-instruct"
OUTPUT_VOLUME_MOUNT_PATH = "/mnt/output-vol"
SMOLVLM_SOURCE_MODEL_ID = f"{OUTPUT_VOLUME_MOUNT_PATH}/smolvlm-500m-checkpoint-450"
SMOLVLM_MODEL_ID = SMOLVLM_SOURCE_MODEL_ID
SMOLVLM_SERVED_MODEL_NAME = "smolvlm-500m-checkpoint-450"
SMOLVLM_ORIGINAL_MODEL_ID = "HuggingFaceTB/SmolVLM-500M-Instruct"
SMOLVLM_ORIGINAL_SERVED_MODEL_NAME = "smolvlm-500m-instruct"
QWEN3_MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
QWEN3_SERVED_MODEL_NAME = "qwen3-vl-4b-instruct"

VLLM_PORT = 8000
OPENWEBUI_PORT = 8080
DEFAULT_GENERATION_CONFIG = {"temperature": 0.0}

QWEN_VLLM_PUBLIC_BASE_URL = "https://ise-lab--qwen-vl-vllm.modal.run/v1"
SMOLVLM_VLLM_PUBLIC_BASE_URL = "https://ise-lab--smolvlm-vllm.modal.run/v1"
SMOLVLM_ORIGINAL_VLLM_PUBLIC_BASE_URL = (
    "https://ise-lab--smolvlm-original-vllm.modal.run/v1"
)
QWEN3_VLLM_PUBLIC_BASE_URL = "https://ise-lab--qwen3-vl-vllm.modal.run/v1"
OPENAI_BASE_URL_LIST = [
    QWEN_VLLM_PUBLIC_BASE_URL,
    SMOLVLM_VLLM_PUBLIC_BASE_URL,
    SMOLVLM_ORIGINAL_VLLM_PUBLIC_BASE_URL,
    QWEN3_VLLM_PUBLIC_BASE_URL,
]
OPENAI_BASE_URLS = ";".join(OPENAI_BASE_URL_LIST)

api_secret = modal.Secret.from_name("qwen-vl-vllm-api-key")
huggingface_secret = modal.Secret.from_name("huggingface-secret")

hf_cache = modal.Volume.from_name("qwen-vl-huggingface-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("qwen-vl-vllm-cache", create_if_missing=True)
output_volume = modal.Volume.from_name("output-vol", create_if_missing=True)
openwebui_data = modal.Volume.from_name(
    "openwebui-vlm-qwen-smolvlm-gemma-qwen3-internvl-data", create_if_missing=True
)

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install(
        "vllm==0.19.0",
        "qwen-vl-utils==0.0.14",
    )
    .env(
        {
            "HF_XET_HIGH_PERFORMANCE": "1",
            "VLLM_IMAGE_FETCH_TIMEOUT": "30",
            "VLLM_VIDEO_FETCH_TIMEOUT": "60",
        }
    )
)

checkpoint_vllm_image = vllm_image.run_commands(
    "python -m pip install --upgrade transformers==5.1.0 huggingface-hub==1.12.0"
)

openwebui_image = (
    modal.Image.from_registry("ghcr.io/open-webui/open-webui:main")
    .env(
        {
            "HOST": "0.0.0.0",
            "PORT": str(OPENWEBUI_PORT),
            "DATA_DIR": "/data",
            "WEBUI_NAME": "VLM vLLM Demo",
            "ENABLE_OLLAMA_API": "false",
            "ENABLE_OPENAI_API": "true",
            "OPENAI_API_BASE_URL": QWEN_VLLM_PUBLIC_BASE_URL,
            "OPENAI_API_BASE_URLS": OPENAI_BASE_URLS,
            "DEFAULT_MODELS": QWEN_SERVED_MODEL_NAME,
            "DEFAULT_MODEL_PARAMS": json.dumps(DEFAULT_GENERATION_CONFIG),
        }
    )
)

app = modal.App("qwen-vl-openwebui-demo")


@app.function(
    image=vllm_image,
    gpu="L4",
    cpu=4,
    memory=32768,
    secrets=[api_secret, huggingface_secret],
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/root/.cache/vllm": vllm_cache,
    },
    scaledown_window=15 * MINUTES,
    timeout=20 * MINUTES,
)
@modal.concurrent(max_inputs=20)
@modal.web_server(VLLM_PORT, startup_timeout=20 * MINUTES, label="qwen-vl-vllm")
def serve_qwen():
    api_key = os.environ["VLLM_API_KEY"]
    mm_limits = {"image": 2, "video": 0}
    mm_processor_kwargs = {
        "min_pixels": 256 * 28 * 28,
        "max_pixels": 1280 * 28 * 28,
    }

    cmd = [
        "vllm",
        "serve",
        QWEN_MODEL_ID,
        "--served-model-name",
        QWEN_SERVED_MODEL_NAME,
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--dtype",
        "half",
        "--max-model-len",
        "8192",
        "--max-num-seqs",
        "4",
        "--gpu-memory-utilization",
        "0.85",
        "--limit-mm-per-prompt",
        json.dumps(mm_limits),
        "--mm-processor-kwargs",
        json.dumps(mm_processor_kwargs),
        "--uvicorn-log-level",
        "info",
        "--enforce-eager",
        "--override-generation-config",
        json.dumps(DEFAULT_GENERATION_CONFIG),
        "--api-key",
        api_key,
    ]

    print("Starting vLLM for", QWEN_MODEL_ID, "as", QWEN_SERVED_MODEL_NAME)
    subprocess.Popen(cmd)


@app.function(
    image=checkpoint_vllm_image,
    gpu="L4",
    cpu=2,
    memory=16384,
    secrets=[api_secret, huggingface_secret],
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/root/.cache/vllm": vllm_cache,
        OUTPUT_VOLUME_MOUNT_PATH: output_volume,
    },
    scaledown_window=15 * MINUTES,
    timeout=20 * MINUTES,
)
@modal.concurrent(max_inputs=20)
@modal.web_server(VLLM_PORT, startup_timeout=20 * MINUTES, label="smolvlm-vllm")
def serve_smolvlm():
    api_key = os.environ["VLLM_API_KEY"]
    mm_limits = {"image": 2, "video": 0}

    cmd = [
        "vllm",
        "serve",
        SMOLVLM_MODEL_ID,
        "--trust-remote-code",
        "--served-model-name",
        SMOLVLM_SERVED_MODEL_NAME,
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--dtype",
        "half",
        "--max-model-len",
        "4096",
        "--max-num-seqs",
        "8",
        "--gpu-memory-utilization",
        "0.70",
        "--limit-mm-per-prompt",
        json.dumps(mm_limits),
        "--uvicorn-log-level",
        "info",
        "--override-generation-config",
        json.dumps(DEFAULT_GENERATION_CONFIG),
        "--api-key",
        api_key,
    ]

    print("Starting vLLM for", SMOLVLM_MODEL_ID, "as", SMOLVLM_SERVED_MODEL_NAME)
    subprocess.Popen(cmd)


@app.function(
    image=vllm_image,
    gpu="L4",
    cpu=2,
    memory=16384,
    secrets=[api_secret, huggingface_secret],
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/root/.cache/vllm": vllm_cache,
    },
    scaledown_window=15 * MINUTES,
    timeout=20 * MINUTES,
)
@modal.concurrent(max_inputs=20)
@modal.web_server(VLLM_PORT, startup_timeout=20 * MINUTES, label="smolvlm-original-vllm")
def serve_smolvlm_original():
    api_key = os.environ["VLLM_API_KEY"]
    mm_limits = {"image": 2, "video": 0}

    cmd = [
        "vllm",
        "serve",
        SMOLVLM_ORIGINAL_MODEL_ID,
        "--served-model-name",
        SMOLVLM_ORIGINAL_SERVED_MODEL_NAME,
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--dtype",
        "half",
        "--max-model-len",
        "4096",
        "--max-num-seqs",
        "8",
        "--gpu-memory-utilization",
        "0.70",
        "--limit-mm-per-prompt",
        json.dumps(mm_limits),
        "--uvicorn-log-level",
        "info",
        "--override-generation-config",
        json.dumps(DEFAULT_GENERATION_CONFIG),
        "--api-key",
        api_key,
    ]

    print(
        "Starting vLLM for",
        SMOLVLM_ORIGINAL_MODEL_ID,
        "as",
        SMOLVLM_ORIGINAL_SERVED_MODEL_NAME,
    )
    subprocess.Popen(cmd)


@app.function(
    image=vllm_image,
    gpu="L4",
    cpu=4,
    memory=32768,
    secrets=[api_secret, huggingface_secret],
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/root/.cache/vllm": vllm_cache,
    },
    scaledown_window=15 * MINUTES,
    timeout=20 * MINUTES,
)
@modal.concurrent(max_inputs=20)
@modal.web_server(VLLM_PORT, startup_timeout=20 * MINUTES, label="qwen3-vl-vllm")
def serve_qwen3():
    api_key = os.environ["VLLM_API_KEY"]
    mm_limits = {"image": 2, "video": 0}
    mm_processor_kwargs = {
        "min_pixels": 256 * 28 * 28,
        "max_pixels": 1280 * 28 * 28,
    }

    cmd = [
        "vllm",
        "serve",
        QWEN3_MODEL_ID,
        "--served-model-name",
        QWEN3_SERVED_MODEL_NAME,
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--dtype",
        "half",
        "--max-model-len",
        "8192",
        "--max-num-seqs",
        "4",
        "--gpu-memory-utilization",
        "0.85",
        "--limit-mm-per-prompt",
        json.dumps(mm_limits),
        "--mm-processor-kwargs",
        json.dumps(mm_processor_kwargs),
        "--uvicorn-log-level",
        "info",
        "--enforce-eager",
        "--override-generation-config",
        json.dumps(DEFAULT_GENERATION_CONFIG),
        "--api-key",
        api_key,
    ]

    print("Starting vLLM for", QWEN3_MODEL_ID, "as", QWEN3_SERVED_MODEL_NAME)
    subprocess.Popen(cmd)


@app.function(
    image=openwebui_image,
    volumes={"/data": openwebui_data},
    secrets=[api_secret],
    max_containers=1,
    scaledown_window=10 * MINUTES,
    timeout=60 * MINUTES,
)
@modal.concurrent(max_inputs=100)
@modal.web_server(OPENWEBUI_PORT, startup_timeout=180, label="webui")
def webui():
    api_key = os.environ["OPENAI_API_KEY"]
    os.environ["OPENAI_API_KEY"] = api_key
    os.environ["OPENAI_API_KEYS"] = ";".join([api_key] * len(OPENAI_BASE_URL_LIST))
    os.environ["OPENAI_API_BASE_URL"] = QWEN_VLLM_PUBLIC_BASE_URL
    os.environ["OPENAI_API_BASE_URLS"] = OPENAI_BASE_URLS
    subprocess.Popen(["bash", "start.sh"], cwd="/app/backend")
