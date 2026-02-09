#!/usr/bin/env python3
"""
modal_app.py - Modal entrypoint for VLM fine-tuning and evaluation.

Usage:
  modal run modal_app.py --cmd "train --model HuggingFaceTB/SmolVLM-500M-Instruct --dataset lmms-lab/textvqa"
  modal run modal_app.py --cmd "eval --dataset lmms-lab/textvqa --checkpoint /outputs/run_name/checkpoints/final"
  modal run modal_app.py --cmd doctor
  modal run modal_app.py --cmd smoke

Commands:
  doctor   - Check environment
  smoke    - Run smoke test
  train    - Run training (uses A100)
  eval     - Run evaluation (uses L4)
"""

import os
import subprocess
from pathlib import Path

import modal

# =============================================================================
# MODAL CONFIGURATION
# =============================================================================

# Fixed paths (DO NOT CHANGE)
MODEL_DIR = Path("/models")
DATASET_DIR = Path("/dataset")
OUTPUT_DIR = Path("/outputs")
ROOT_DIR = Path("/root/VLM-Distillation")

# Volumes
volume = modal.Volume.from_name("model-weights-vol", create_if_missing=True)
dataset_volume = modal.Volume.from_name("dataset-vol", create_if_missing=True)
output_volume = modal.Volume.from_name("output-vol", create_if_missing=True)

# App settings
MODAL_APP_NAME = "VLM-Distillation"
WANDB_PROJECT = "VLM-Distillation"

# Base image with all dependencies
base_image = (
    modal.Image.debian_slim()
    .apt_install(
        "git",
        "libgl1",
        "libglib2.0-0",
    )
    .pip_install("opencv-python-headless")
    .pip_install_from_requirements("./eval/VLMEvalKit/requirements.txt")
    .pip_install(
        "peft",
        "trl",
        "bitsandbytes",
        "accelerate",
        "wandb",
        "groq",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_dir(
        ".",
        remote_path=ROOT_DIR,
        ignore=modal.FilePatternMatcher.from_file(".gitignore")
    )
)

app = modal.App(
    name=MODAL_APP_NAME,
    image=base_image,
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("groq-secret", required_keys=["GROQ_API_KEY"]),
    ],
    volumes={
        MODEL_DIR.as_posix(): volume,
        DATASET_DIR.as_posix(): dataset_volume,
        OUTPUT_DIR.as_posix(): output_volume,
    },
)


def _setup_env() -> dict:
    """Set up environment variables for execution."""
    env = os.environ.copy()
    env["WANDB_PROJECT"] = WANDB_PROJECT
    env["WANDB_MODE"] = "online"
    env["PYTHONUNBUFFERED"] = "1"
    env["HF_DATASETS_CACHE"] = str(DATASET_DIR / ".hf_cache" / "datasets")
    env["HF_HUB_CACHE"] = str(MODEL_DIR / ".hf_cache" / "hub")
    env["TRANSFORMERS_CACHE"] = str(MODEL_DIR / ".hf_cache" / "hub")
    env["OUTPUT_DIR"] = str(OUTPUT_DIR)

    # Ensure cache directories exist
    os.makedirs(env["HF_DATASETS_CACHE"], exist_ok=True)
    os.makedirs(env["HF_HUB_CACHE"], exist_ok=True)

    return env


def _run_command(cmd: str, env: dict):
    """Execute a shell command with streaming output."""
    print(f"[modal] Executing: {cmd}")

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

    print("[modal] Command completed successfully")


# =============================================================================
# TRAINING (A100)
# =============================================================================

@app.function(
    gpu="A100",
    timeout=60 * 60 * 12,  # 12 hours
)
def run_train(cmd: str):
    """Run training command on A100 GPU."""
    env = _setup_env()
    full_cmd = f"cd {ROOT_DIR} && python finetune.py train {cmd}"
    _run_command(full_cmd, env)


# =============================================================================
# EVALUATION (L4)
# =============================================================================

@app.function(
    gpu="L4",
    timeout=60 * 60 * 4,  # 4 hours
)
def run_eval(cmd: str):
    """Run evaluation command on L4 GPU."""
    env = _setup_env()
    full_cmd = f"cd {ROOT_DIR} && python eval.py {cmd}"
    _run_command(full_cmd, env)


# =============================================================================
# UTILITY COMMANDS (CPU or small GPU)
# =============================================================================

@app.function(
    timeout=60 * 30,  # 30 minutes
)
def run_utility(script: str, cmd: str):
    """Run utility commands (doctor, preflight, smoke)."""
    env = _setup_env()
    full_cmd = f"cd {ROOT_DIR} && python {script} {cmd}"
    _run_command(full_cmd, env)


@app.function(
    timeout=60 * 5,  # 5 minutes
)
def list_outputs():
    """List all outputs on the Modal volume."""
    import os
    from pathlib import Path
    
    output_dir = Path("/outputs")
    
    print("=" * 60)
    print("Modal Output Volume Contents")
    print("=" * 60)
    
    if not output_dir.exists():
        print("  (empty - no outputs yet)")
        return
    
    for run_dir in sorted(output_dir.iterdir()):
        if run_dir.is_dir():
            print(f"\n📁 {run_dir.name}/")
            
            # Check for checkpoints
            ckpt_dir = run_dir / "checkpoints"
            if ckpt_dir.exists():
                for ckpt in sorted(ckpt_dir.iterdir()):
                    print(f"    └── checkpoints/{ckpt.name}/")
            
            # Check for eval results
            eval_dir = run_dir / "eval"
            if eval_dir.exists():
                for ds_dir in sorted(eval_dir.iterdir()):
                    if ds_dir.is_dir():
                        for ckpt_dir in sorted(ds_dir.iterdir()):
                            print(f"    └── eval/{ds_dir.name}/{ckpt_dir.name}/")
    
    print("\n" + "=" * 60)
    print("To run eval with a checkpoint:")
    print('  modal run modal_app.py --cmd "eval --dataset lmms-lab/textvqa --checkpoint /outputs/<run_name>/checkpoints/final"')
    print("=" * 60)


# =============================================================================
# LOCAL ENTRYPOINT
# =============================================================================

@app.local_entrypoint()
def main(cmd: str = "doctor"):
    """
    Modal entrypoint - dispatch to appropriate function based on command.
    
    Examples:
      modal run modal_app.py --cmd doctor
      modal run modal_app.py --cmd smoke
      modal run modal_app.py --cmd "train --model HuggingFaceTB/SmolVLM-500M-Instruct --dataset lmms-lab/textvqa"
      modal run modal_app.py --cmd "eval --dataset lmms-lab/textvqa"
    """
    cmd = cmd.strip()
    parts = cmd.split(maxsplit=1)
    action = parts[0] if parts else "doctor"
    args = parts[1] if len(parts) > 1 else ""

    print(f"[modal] Action: {action}")
    print(f"[modal] Args: {args}")

    if action == "train":
        # Training runs on A100
        run_train.remote(args)
    elif action == "eval":
        # Evaluation runs on L4
        run_eval.remote(args)
    elif action == "list" or action == "ls":
        # List outputs on Modal volume
        list_outputs.remote()
    elif action == "doctor":
        # Doctor is a finetune.py subcommand
        run_utility.remote("finetune.py", "doctor")
    elif action == "preflight":
        # Preflight is a finetune.py subcommand
        run_utility.remote("finetune.py", f"preflight {args}")
    elif action == "smoke":
        # Smoke test is a finetune.py subcommand
        run_utility.remote("finetune.py", f"smoke {args}")
    else:
        print(f"[modal] Unknown command: {action}")
        print("[modal] Available commands: doctor, preflight, smoke, train, eval")
        raise ValueError(f"Unknown command: {action}")
