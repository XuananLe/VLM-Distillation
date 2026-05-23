import json
import os
import shlex
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
SMOKE_DATA_DIR = Path("/tmp/vlm-distillation-smoke")
MODAL_TRANSFORMERS_VERSION = os.environ.get("MODAL_TRANSFORMERS_VERSION", "5.1.0")
MODAL_FLASH_ATTN_VERSION = os.environ.get("MODAL_FLASH_ATTN_VERSION", "2.8.3")
MODAL_IMAGE_BUILD_GPU = os.environ.get("MODAL_IMAGE_BUILD_GPU", "L4")
MODAL_GPU = os.environ.get("MODAL_GPU", "A100-80GB")
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
    "numpy==",
    "deepspeed==",
    "transformers==",
    "flash_attn==",
    "flash-attn==",
    "xformers==",
    "manimgl==",
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
    modal.Image.from_registry("nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04", add_python="3.12")
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
        "deepspeed==0.18.8",
    )
    .run_commands(
        'python -c "import flash_attn, transformers; '
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
        raise ValueError("MODAL_R2_ENDPOINT_URL must be set when enabling Cloudflare R2 mounts.")
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


def build_modal_secrets() -> list[modal.Secret]:
    return [
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name(R2_SECRET_NAME),
    ]


app = modal.App(
    "vlm-distillation",
    image=base_image,
    secrets=build_modal_secrets(),
)
app_mounts, committable_volumes = build_modal_mounts()


def shell_join(args: list[str]) -> str:
    return " ".join(shlex.quote(str(arg)) for arg in args)


def replace_path_with_symlink(path: Path, target: Path) -> None:
    if path.is_symlink() and path.resolve() == target:
        return
    if path.exists() or path.is_symlink():
        if path.is_dir() and not path.is_symlink():
            return
        path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.symlink_to(target)


def prepare_modal_filesystem() -> None:
    replace_path_with_symlink(ROOT_DIR / "data", DATASET_DIR)
    replace_path_with_symlink(ROOT_DIR / "output", OUTPUT_DIR)
    workspace_root = Path("/workspace")
    workspace_root.mkdir(parents=True, exist_ok=True)
    replace_path_with_symlink(workspace_root / "VLM-Distillation", ROOT_DIR)
    replace_path_with_symlink(workspace_root / "data", DATASET_DIR)
    replace_path_with_symlink(workspace_root / "cache", CACHE_DIR)


def prepare_modal_runtime_env() -> dict[str, str]:
    prepare_modal_filesystem()
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


def exec_cmd_impl(cmd: str) -> None:
    cmd = cmd.strip()
    if not cmd:
        raise ValueError("cmd must be non-empty")

    env = prepare_modal_runtime_env()
    print(f"WANDB_API_KEY set: {'WANDB_API_KEY' in env}")

    print("[exec] Command:")
    print(cmd)

    bash_cmd = f"set -euxo pipefail; {cmd}"
    proc = subprocess.Popen(
        bash_cmd,
        shell=True,
        executable="/bin/bash",
        cwd=ROOT_DIR,
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


def write_smoke_dataset() -> Path:
    from PIL import Image, ImageDraw

    image_dir = SMOKE_DATA_DIR / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    image_path = image_dir / "smoke.png"

    image = Image.new("RGB", (224, 224), color=(245, 245, 245))
    draw = ImageDraw.Draw(image)
    draw.rectangle((48, 48, 176, 176), fill=(220, 40, 40))
    image.save(image_path)

    samples = [
        {
            "id": "modal-smoke-0",
            "image": image_path.name,
            "conversations": [
                {
                    "from": "human",
                    "value": "<image>\nWhat color is the square?",
                },
                {
                    "from": "gpt",
                    "value": "The square is red.",
                },
            ],
        }
    ]
    data_path = SMOKE_DATA_DIR / "train_llava.json"
    data_path.write_text(json.dumps(samples), encoding="utf-8")
    return data_path


def build_smoke_training_cmd(data_path: Path, max_steps: int) -> str:
    output_dir = OUTPUT_DIR / "modal_smoke_smolvlm_256m"
    train_args = [
        "python",
        "src/train/train_distillation.py",
        "--student_model_id",
        "HuggingFaceTB/SmolVLM-256M-Instruct",
        "--teacher_model_ids",
        "google/gemma-3-4b-it",
        "OpenGVLab/InternVL2-1B",
        "Qwen/Qwen2.5-VL-3B-Instruct",
        "Qwen/Qwen2-VL-2B-Instruct",
        "--data_path",
        data_path.as_posix(),
        "--image_folder",
        (SMOKE_DATA_DIR / "images").as_posix(),
        "--distillation_loss",
        "trie_wasserstein_loss",
        "--bf16",
        "True",
        "--output_dir",
        output_dir.as_posix(),
        "--student_temperature",
        "1.0",
        "--teacher_temperature",
        "1.0",
        "--alpha",
        "0.0",
        "--num_train_epochs",
        "1",
        "--max_steps",
        str(max_steps),
        "--per_device_train_batch_size",
        "1",
        "--learning_rate",
        "1e-5",
        "--warmup_ratio",
        "0.0",
        "--lr_scheduler_type",
        "constant",
        "--tf32",
        "True",
        "--gradient_checkpointing",
        "False",
        "--logging_steps",
        "1",
        "--save_strategy",
        "no",
        "--dataloader_num_workers",
        "0",
        "--remove_unused_columns",
        "False",
        "--report_to",
        "none",
        "--disable_tqdm",
        "True",
    ]
    return " && ".join(
        [
            "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader",
            shell_join(train_args),
        ]
    )


@app.function(
    gpu=MODAL_GPU,
    timeout=60 * 60 * 24,
    volumes=app_mounts,
)
def exec_cmd(cmd: str) -> None:
    exec_cmd_impl(cmd)


@app.function(
    gpu=MODAL_GPU,
    timeout=60 * 60 * 24,
    volumes=app_mounts,
)
def smoke_train(max_steps: int = 1) -> None:
    if max_steps < 1:
        raise ValueError("max_steps must be >= 1")
    data_path = write_smoke_dataset()
    exec_cmd_impl(build_smoke_training_cmd(data_path, max_steps=max_steps))


@app.local_entrypoint()
def run(cmd: str = r"""""", smoke: bool = True, max_steps: int = 1, wait: bool = True) -> None:
    if cmd:
        call = exec_cmd.spawn(cmd)
    elif smoke:
        call = smoke_train.spawn(max_steps=max_steps)
    else:
        raise ValueError("Pass --cmd or leave --smoke enabled to run the one-step training smoke job.")
    print(f"Triggered Modal function call: {getattr(call, 'object_id', call)}")
    if wait:
        call.get()
