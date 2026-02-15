import os
import subprocess
from pathlib import Path

import modal

ROOT_DIR = Path("/root/VLM-Distillation")
WANDB_PROJECT = "VLM-Distillation"
MODEL_DIR = Path("/models")
DATASET_DIR = Path("/dataset")
OUTPUT_DIR = Path("/outputs")
flash_attn_release = (
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/"
    "flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
)
model_volume = modal.Volume.from_name("model-weights-vol", create_if_missing=True)
dataset_volume = modal.Volume.from_name("dataset-vol", create_if_missing=True)
output_volume = modal.Volume.from_name("output-vol", create_if_missing=True)

base_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install(
        "git",
        "curl",
        "ffmpeg",
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
    .pip_install_from_requirements("./requirements.txt")
    .uv_pip_install(
        flash_attn_release,
        "pillow-avif-plugin"
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "WANDB_PROJECT": WANDB_PROJECT,
            "ACCELERATE_LOG_LEVEL": "error",
        }
    )
)

app = modal.App(
    image=base_image,
    secrets=[modal.Secret.from_name("wandb-secret")],
    volumes={
        MODEL_DIR.as_posix(): model_volume,
        DATASET_DIR.as_posix(): dataset_volume,
        OUTPUT_DIR.as_posix(): output_volume,
    },
)


@app.function(gpu="T4", timeout=60 * 60 * 12)
def exec_cmd(cmd: str):
    cmd = cmd.strip()
    if not cmd:
        raise ValueError("cmd must be non-empty")

    env = os.environ.copy()
    # print the wandb api key for debugging
    print(f"WANDB_API_KEY: {env.get('WANDB_API_KEY', 'not set')}")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("WANDB_MODE", "online")
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
        "train": f"cd {ROOT_DIR} && python -m src.train",
        "evaluate": f"cd {ROOT_DIR} && python -m src.evaluate",
    }
    exec_cmd.remote(cmd['train'])
