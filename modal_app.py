import os
import subprocess
from pathlib import Path

import modal

LOCAL_ROOT_DIR = Path(__file__).resolve().parent
ROOT_DIR = Path("/root/VLM-Distillation")
WANDB_PROJECT = "VLM-Distillation"
MODEL_DIR = Path("/models")
DATASET_DIR = Path("/data")
OUTPUT_DIR = Path("/output")
CACHE_DIR = Path("/cache")
MODAL_TRANSFORMERS_VERSION = os.environ.get("MODAL_TRANSFORMERS_VERSION", "5.1.0")
MODAL_FLASH_ATTN_VERSION = os.environ.get("MODAL_FLASH_ATTN_VERSION", "2.8.3")
MODAL_IMAGE_BUILD_GPU = os.environ.get("MODAL_IMAGE_BUILD_GPU", "L4")
R2_SECRET_NAME = os.environ.get("MODAL_R2_SECRET_NAME", "cloudflare-r2-secret")
R2_ENDPOINT_URL = os.environ.get(
    "MODAL_R2_ENDPOINT_URL",
    "https://7f19258d1ceabe6abc46808eca312b1e.r2.cloudflarestorage.com",
)
R2_CACHE_BUCKET = os.environ.get("MODAL_R2_CACHE_BUCKET", "google-drive-backup")
R2_CACHE_PREFIX = os.environ.get(
    "MODAL_R2_CACHE_PREFIX",
    "1W9sUXnqNdR2qVHr8PrgVAtM6J2SXsJ-7",
)
MODAL_REQUIREMENTS_PATH = Path("/tmp/vlm-distillation-modal-requirements.txt")
MODAL_REQUIREMENTS_BLOCKLIST = (
    "torch==",
    "torchvision==",
    "torchaudio==",
    "nvidia-",
    "transformers==",
    "flash_attn==",
    "flash-attn==",
)
model_volume = modal.Volume.from_name("model-weights-vol", create_if_missing=True)
dataset_volume = modal.Volume.from_name("vlm-distillation-data", create_if_missing=True)
output_volume = modal.Volume.from_name("output-vol", create_if_missing=True)


def build_modal_requirements_file() -> Path:
    source_path = LOCAL_ROOT_DIR / "requirements.txt"
    if not source_path.exists():
        source_path = ROOT_DIR / "requirements.txt"
    filtered_lines = []
    for raw_line in source_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith(MODAL_REQUIREMENTS_BLOCKLIST):
            continue
        filtered_lines.append(line)
    MODAL_REQUIREMENTS_PATH.write_text(
        "\n".join(filtered_lines) + "\n",
        encoding="utf-8",
    )
    return MODAL_REQUIREMENTS_PATH


build_modal_requirements_file()

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
    .uv_pip_install(
        "wheel==0.46.3",
        "setuptools==69.0.3",
        "packaging==26.0",
        "ninja==1.13.0",
        "psutil==7.2.2",
        "numpy<2.2",
        "mosaicml-streaming==0.13.0",
    )
    .uv_pip_install(
        "torch==2.8.0+cu126",
        "torchvision==0.23.0+cu126",
        "torchaudio==2.8.0+cu126",
        extra_index_url="https://download.pytorch.org/whl/cu126",
    )
    .uv_pip_install(
        f"flash-attn=={MODAL_FLASH_ATTN_VERSION}",
        extra_options="--no-build-isolation",
        gpu=MODAL_IMAGE_BUILD_GPU,
    )
    .pip_install_from_requirements(
        MODAL_REQUIREMENTS_PATH,
        extra_options="--no-build-isolation",
    )
    .uv_pip_install(
        f"transformers=={MODAL_TRANSFORMERS_VERSION}",
    )
    .run_commands(
        "python -c \"import flash_attn, transformers; "
        "symbols = ('AutoModelForImageTextToText', 'AutoProcessor'); "
        "missing = [symbol for symbol in symbols if not hasattr(transformers, symbol)]; "
        f"assert transformers.__version__ == '{MODAL_TRANSFORMERS_VERSION}'; "
        "assert not missing, f'Missing transformers symbols: {missing}'; "
        "print('flash_attn import OK'); "
        "print('transformers', transformers.__version__)\""
    )
    .add_local_dir(
        ".",
        remote_path=ROOT_DIR,
        copy=True,
        ignore=modal.FilePatternMatcher.from_file(".gitignore"),
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "WANDB_PROJECT": WANDB_PROJECT,
            "ACCELERATE_LOG_LEVEL": "error",
            "XFORMERS_IGNORE_FLASH_VERSION_CHECK": "1",
            "MAX_JOBS": "1",
            "S3_ENDPOINT_URL": R2_ENDPOINT_URL,
            "AWS_DEFAULT_REGION": os.environ.get("AWS_DEFAULT_REGION", "auto"),
        }
    )
)

def build_r2_mount(bucket_name: str, key_prefix: str | None) -> modal.CloudBucketMount:
    if not R2_ENDPOINT_URL:
        raise ValueError(
            "MODAL_R2_ENDPOINT_URL must be set when enabling Cloudflare R2 mounts."
        )
    normalized_prefix = None
    if key_prefix:
        normalized_prefix = key_prefix if key_prefix.endswith("/") else f"{key_prefix}/"
    return modal.CloudBucketMount(
        bucket_name=bucket_name,
        bucket_endpoint_url=R2_ENDPOINT_URL,
        key_prefix=normalized_prefix,
        secret=modal.Secret.from_name(R2_SECRET_NAME),
        read_only=True,
    )


def build_modal_mounts() -> tuple[dict[str, object], list[modal.Volume]]:
    mounts: dict[str, object] = {
        MODEL_DIR.as_posix(): model_volume,
        DATASET_DIR.as_posix(): dataset_volume,
        OUTPUT_DIR.as_posix(): output_volume,
        CACHE_DIR.as_posix(): build_r2_mount(
            bucket_name=R2_CACHE_BUCKET,
            key_prefix=R2_CACHE_PREFIX,
        ),
    }
    return mounts, [model_volume, dataset_volume, output_volume]


def _build_modal_secrets() -> list[modal.Secret]:
    return [
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name(R2_SECRET_NAME),
    ]


app_mounts, committable_volumes = build_modal_mounts()
app = modal.App(
    image=base_image,
    secrets=_build_modal_secrets(),
    volumes=app_mounts,
)


def _prepare_modal_runtime_env() -> dict[str, str]:
    env = os.environ.copy()
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
    os.makedirs(DATASET_DIR / ".hf_cache" / "datasets", exist_ok=True)
    os.makedirs(MODEL_DIR / ".hf_cache" / "hub", exist_ok=True)
    return env


@app.function(gpu = "L4", timeout=60 * 60 * 24)
def exec_cmd(cmd: str) -> None:
    cmd = cmd.strip()
    if not cmd:
        raise ValueError("cmd must be non-empty")

    env = _prepare_modal_runtime_env()
    print(f"WANDB_API_KEY set: {'WANDB_API_KEY' in env}")

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

    for volume in committable_volumes:
        volume.commit()

    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, cmd)


@app.local_entrypoint()
def run(
cmd = r"""
cd /root/VLM-Distillation
python -c "print('Pass a command via --cmd to run a workload.')"
wait
"""

):
    exec_cmd.remote(cmd)
