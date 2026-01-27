import modal
import subprocess
import os
from pathlib import Path
import wandb



# DONT CHANGE THESE PATHS
MODEL_DIR = Path("/models")
DATASET_DIR = Path("/dataset")
OUTPUT_DIR = Path("/outputs")
ROOT_DIR = Path("/root/VLM-Distillation")
EVAL_DIR = ROOT_DIR / "VLMEvalKit"
volume = modal.Volume.from_name("model-weights-vol", create_if_missing=True)
dataset_volume = modal.Volume.from_name("dataset-vol", create_if_missing=True)
output_volume = modal.Volume.from_name("output-vol", create_if_missing=True)
#######################


# Change this to a unique name for your Modal app
MODAL_RUN_NAME = "Test Modal Run An"
WANDB_PROJECT = "VLM-Distillation"
# Log config to wandb
config = {
    "hidden_layer_sizes": [32, 64],
    "kernel_sizes": [3],
    "activation": "ReLU",
    "pool_sizes": [2],
    "dropout": 0.5,
    "num_classes": 10,
}

# Run name
RUN = wandb.init(project=WANDB_PROJECT, config = config)

base_image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .apt_install(
        "git",
        "libgl1",          # provides libGL.so.1
        "libglib2.0-0",    # often needed by opencv
    )
    .pip_install("opencv-python-headless")
    .pip_install_from_requirements("./VLMEvalKit/requirements.txt")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_dir(
        ".",
        remote_path=ROOT_DIR,
        ignore=modal.FilePatternMatcher.from_file(".gitignore")
    )
)

app = modal.App(
    name=MODAL_RUN_NAME,
    image=base_image,
    secrets=[
        modal.Secret.from_name("wandb-secret")
    ],
    volumes={
        MODEL_DIR.as_posix(): volume,
        DATASET_DIR.as_posix(): dataset_volume,
        OUTPUT_DIR.as_posix(): output_volume,
    },
)

@app.function(
            gpu = "L4",
            timeout=60 * 60 * 12)
def exec_cmd(cmd):
    cmd = cmd.strip()
    env = os.environ.copy()
    env.setdefault("WANDB_PROJECT", WANDB_PROJECT)
    env.setdefault("PYTHONUNBUFFERED", "1")  # ensure unbuffered output for python cmds
    env.setdefault("HF_DATASETS_CACHE", str(DATASET_DIR / ".hf_cache" / "datasets"))
    env.setdefault("HF_HUB_CACHE", str(MODEL_DIR / ".hf_cache" / "hub"))
    os.environ.setdefault("WANDB_MODE", "online")
    os.makedirs(env.get("HF_DATASETS_CACHE", "/tmp"), exist_ok=True)
    os.makedirs(env.get("HF_HUB_CACHE", "/tmp"), exist_ok=True)

    print("[exec] Starting")
    if not cmd:
        print("[exec] No command provided; nothing to run.")
        return
    print("[exec] Command:\n" + cmd)

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
            print(f"{line.rstrip()}")
    finally:
        returncode = proc.wait()

    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, cmd)

    print("[exec] Finished successfully")


@app.local_entrypoint()
def run(): 
    exec_cmd.remote(cmd=f"cd {ROOT_DIR} && echo 'Hello world'")