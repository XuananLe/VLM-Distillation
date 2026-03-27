import os
import subprocess
from pathlib import Path

import modal

ROOT_DIR = Path("/root/VLM-Distillation")
WANDB_PROJECT = "VLM-Distillation"
MODEL_DIR = Path("/models")
DATASET_DIR = Path("/data")
OUTPUT_DIR = Path("/output")
CACHE_DIR = Path("/cache")
model_volume = modal.Volume.from_name("model-weights-vol", create_if_missing=True)
dataset_volume = modal.Volume.from_name("vlm-distillation-data", create_if_missing=True)
output_volume = modal.Volume.from_name("output-vol", create_if_missing=True)
cache_volume = modal.Volume.from_name("cache-vol", create_if_missing=True)
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
    .pip_install(
        "wheel==0.46.3",
        "setuptools==69.0.3",
        "packaging==26.0",
        "ninja==1.13.0",
        "psutil==7.2.2",
    )
    .pip_install(
        "torch==2.8.0+cu126",
        "torchvision==0.23.0+cu126",
        "torchaudio==2.8.0+cu126",
        extra_index_url="https://download.pytorch.org/whl/cu126",
    )
    .pip_install_from_requirements(
        "requirements.txt",
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
        CACHE_DIR.as_posix(): cache_volume,
    },
)


@app.function(gpu = "A100-80GB", timeout=60 * 60 * 12)
def exec_cmd(cmd: str) -> None:
    cmd = cmd.strip()
    if not cmd:
        raise ValueError("cmd must be non-empty")

    env = os.environ.copy()
    print(f"WANDB_API_KEY set: {'WANDB_API_KEY' in env}")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("WANDB_MODE", "online")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env.setdefault("HF_DATASETS_CACHE", str(DATASET_DIR / ".hf_cache" / "datasets"))
    env.setdefault("HF_HUB_CACHE", str(MODEL_DIR / ".hf_cache" / "hub"))
    env.setdefault("HF_HOME", str(MODEL_DIR / ".hf_cache"))

    pythonpath_entries = [str(ROOT_DIR), str(ROOT_DIR / "src")]
    existing_pythonpath = env.get("PYTHONPATH")
    if existing_pythonpath:
        pythonpath_entries.extend(path for path in existing_pythonpath.split(os.pathsep) if path)
    env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(pythonpath_entries))

    os.makedirs(env["HF_DATASETS_CACHE"], exist_ok=True)
    os.makedirs(env["HF_HUB_CACHE"], exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
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

    for volume in (model_volume, dataset_volume, output_volume, cache_volume):
        volume.commit()

    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, cmd)


@app.local_entrypoint()
def run():
    cmd = { 
        "train": f"cd {ROOT_DIR} && bash scripts/train/distill_single_teacher.sh",
        "eval": f"cd /root/VLM-Distillation/src/eval && python run.py --data DocVQA_VAL --model SmolVLM-500M --work-dir /output/vlmeval/SmolVLM-500M-Sample-Normal --subset-size 1037",
    }
    exec_cmd.remote(cmd['eval'])
