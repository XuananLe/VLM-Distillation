import os
import subprocess
from pathlib import Path

import modal

ROOT_DIR = Path("/root/VLM-Distillation")
WANDB_PROJECT = "VLM-Distillation"
MODEL_DIR = Path("/models")
DATASET_DIR = Path("/data")
OUTPUT_DIR = Path("/output")
model_volume = modal.Volume.from_name("model-weights-vol", create_if_missing=True)
dataset_volume = modal.Volume.from_name("vlm-distillation-data", create_if_missing=True)
output_volume = modal.Volume.from_name("output-vol", create_if_missing=True)

base_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04",
        add_python="3.12"
    )
    .apt_install(
        "git",
        "curl",
        "ffmpeg",
        "libc-bin",
        "build-essential",
        "clang",
        "ninja-build",
        "pkg-config",
        "libgl1",
        "libglib2.0-0",
    )
    .add_local_dir(
        ".",
        remote_path=ROOT_DIR,
        copy=True,
        ignore=modal.FilePatternMatcher.from_file(".gitignore"),
    )
    .uv_pip_install(
        "accelerate",
        "anls>=0.0.2",
        "apted>=1.0.3",
        "bert_score",
        "cairosvg",
        "colormath>=3.0.0",
        "datasets",
        "distance>=0.1.3",
        "Pillow",
        "tqdm",
        "pillow-avif-plugin",
        "attrdict",
        "timm",
        "ujson",
        "decord",
        "hf-transfer",
        "google-genai",
        "gradio",
        "huggingface_hub",
        "wandb",
        "jieba>=0.42.1",
        "json_repair",
        "levenshtein>=0.27.1",
        "lpips",
        "math-verify",
        "nltk",
        "numpy",
        "sentencepiece",
        "scipy",
        "matplotlib",
        "backoff",
        "openai",
        "openai-clip",
        "pdf2image>=1.17.0",
        "polygon3>=3.0.9.1",
        "protobuf",
        "qwen_vl_utils",
        "requests",
        "rich",
        "scikit-image",
        "scikit-learn",
        "sentence_transformers",
        "tiktoken",
        "einops",
        "transformers>=4.57.0,<5",
        "trl==0.17.0",
        "peft==0.15.2",
        "sty",
        "portalocker",
        "validators",
        "python-dotenv",
        "tabulate",
        "timeout-decorator",
        "imageio",
        "openpyxl",
        "xlsxwriter",
        "jsonlines",
        "termcolor",
        "unidecode",
        "opencv-python-headless",
        "natsort",
        "sacrebleu",
        "ipdb",
        "rapidfuzz",
        "editdistance",
        "word2number",
        "omegaconf>=2.4.0.dev4",
        "antlr4-python3-runtime==4.11.1",
        "pylatexenc",
        "torchmetrics",
        "typing_extensions",
        "zss>=1.2.0",
    )
    .uv_pip_install("num2words")
    .uv_pip_install(
        "torch==2.8.0",
        "torchvision==0.23.0",
        "torchaudio==2.8.0",
        extra_options="--torch-backend=cu126",
    )
    .uv_pip_install("xformers==0.0.32.post2")
    .uv_pip_install("wheel", "packaging", "psutil", "ninja", "setuptools", "deepspeed")
    .uv_pip_install(
        "git+https://github.com/deepseek-ai/DeepSeek-VL2.git",
        extra_options="--no-deps",
    )
    .uv_pip_install(
        "git+https://github.com/deepseek-ai/DeepSeek-VL.git",
        extra_options="--no-deps",
    )
    .uv_pip_install(
        "flash-attn==2.8.3",
        extra_options="--no-build-isolation",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "WANDB_PROJECT": WANDB_PROJECT,
            "ACCELERATE_LOG_LEVEL": "error",
            "XFORMERS_IGNORE_FLASH_VERSION_CHECK": "1",
            "MAX_JOBS": "1",
        }
    )
)

app = modal.App(
    image=base_image,
    secrets=[modal.Secret.from_name("wandb-secret"), modal.Secret.from_name("huggingface-secret")],
    volumes={
        MODEL_DIR.as_posix(): model_volume,
        DATASET_DIR.as_posix(): dataset_volume,
        OUTPUT_DIR.as_posix(): output_volume,
    },
)


@app.function(gpu="L4", timeout=60 * 60 * 12)
def exec_cmd(cmd: str):
    cmd = cmd.strip()
    if not cmd:
        raise ValueError("cmd must be non-empty")

    env = os.environ.copy()
    print(f"WANDB_API_KEY: {env.get('WANDB_API_KEY', 'not set')}")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("WANDB_MODE", "online")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("PYTHONPATH", f"{ROOT_DIR}:{ROOT_DIR / 'src'}")
    env.setdefault("HF_DATASETS_CACHE", str(DATASET_DIR / ".hf_cache" / "datasets"))
    env.setdefault("HF_HUB_CACHE", str(MODEL_DIR / ".hf_cache" / "hub"))
    env.setdefault("HF_HOME", str(MODEL_DIR / ".hf_cache"))

    os.makedirs(env["HF_DATASETS_CACHE"], exist_ok=True)
    os.makedirs(env["HF_HUB_CACHE"], exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(DATASET_DIR / ".hf_cache" / "datasets", exist_ok=True)
    os.makedirs(MODEL_DIR / ".hf_cache" / "hub", exist_ok=True)

    print("[exec] Command:")
    print(cmd)

    bash_cmd = f"set -euxo pipefail; {cmd}"
    proc = subprocess.Popen(
        bash_cmd,
        shell=True,
        executable="/bin/bash",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line.rstrip())
    finally:
        returncode = proc.wait()

    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, cmd)


@app.local_entrypoint()
def run():
    cmd = {
        "train": f"cd {ROOT_DIR} && bash scripts/train/distill_single_teacher.sh",
    }
    exec_cmd.remote(cmd['train'])
