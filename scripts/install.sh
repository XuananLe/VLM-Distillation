#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
PYTHON_BIN="${PYTHON_BIN:-}"
UV_BIN="${UV_BIN:-}"
UV_INSTALL_DIR="${UV_INSTALL_DIR:-$ROOT_DIR/.uv-bin}"

TORCH_VERSION="${TORCH_VERSION:-2.8.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.23.0}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.8.0}"
TORCH_CUDA_TAG="${TORCH_CUDA_TAG:-cu126}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/${TORCH_CUDA_TAG}}"

FLASH_ATTN_VERSION="${FLASH_ATTN_VERSION:-2.8.3}"
ACCELERATE_VERSION="${ACCELERATE_VERSION:-1.13.0}"
DEEPSPEED_VERSION="${DEEPSPEED_VERSION:-0.18.8}"
TRANSFORMERS_VERSION="${TRANSFORMERS_VERSION:-5.1.0}"
MODAL_VERSION="${MODAL_VERSION:-1.3.2}"
STREAMING_VERSION="${STREAMING_VERSION:-0.13.0}"

INSTALL_SYSTEM_DEPS="${INSTALL_SYSTEM_DEPS:-1}"
INSTALL_WRITING_DEPS="${INSTALL_WRITING_DEPS:-0}"
FILTERED_REQUIREMENTS_PATH=""

log() {
  printf '[install] %s\n' "$*"
}

die() {
  printf '[install] error: %s\n' "$*" >&2
  exit 1
}

command_exists() {
  command -v "$1" >/dev/null 2>&1
}

cleanup() {
  if [[ -n "${FILTERED_REQUIREMENTS_PATH:-}" ]]; then
    rm -f "${FILTERED_REQUIREMENTS_PATH}"
  fi
}

trap cleanup EXIT

pick_uv() {
  if [[ -n "${UV_BIN}" ]]; then
    command_exists "${UV_BIN}" || die "UV_BIN=${UV_BIN} was not found."
    printf '%s\n' "${UV_BIN}"
    return
  fi

  if command_exists uv; then
    printf '%s\n' "uv"
    return
  fi

  mkdir -p "${UV_INSTALL_DIR}"
  export PATH="${UV_INSTALL_DIR}:${PATH}"

  if command_exists uv; then
    printf '%s\n' "uv"
    return
  fi

  log "Installing uv via the official standalone installer."
  if command_exists curl; then
    curl -LsSf https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL="${UV_INSTALL_DIR}" sh
  elif command_exists wget; then
    wget -qO- https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL="${UV_INSTALL_DIR}" sh
  else
    die "Neither curl nor wget is available to install uv."
  fi

  command_exists uv || die "uv installation succeeded but uv is still not on PATH."
  printf '%s\n' "uv"
}

pick_python() {
  if [[ -n "${PYTHON_BIN}" ]]; then
    command_exists "${PYTHON_BIN}" || die "PYTHON_BIN=${PYTHON_BIN} was not found."
    printf '%s\n' "${PYTHON_BIN}"
    return
  fi

  local candidates=(python3.12 python3.11 python3)
  local candidate
  for candidate in "${candidates[@]}"; do
    if command_exists "${candidate}"; then
      local version
      version="$("${candidate}" - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"
      case "${version}" in
        3.11|3.12)
          printf '%s\n' "${candidate}"
          return
          ;;
      esac
    fi
  done

  die "Python 3.11 or 3.12 is required."
}

maybe_install_apt_packages() {
  [[ "${INSTALL_SYSTEM_DEPS}" == "1" ]] || return 0
  command_exists apt-get || return 0

  local sudo_cmd=()
  if [[ "${EUID}" -ne 0 ]]; then
    command_exists sudo || die "apt-get is available but sudo is not installed."
    sudo_cmd=(sudo)
  fi

  local packages=(
    build-essential
    ca-certificates
    curl
    ffmpeg
    git
    git-lfs
    libaio-dev
    libcairo2-dev
    libffi-dev
    libgl1
    libglib2.0-0
    libjpeg-dev
    libpng-dev
    libssl-dev
    ninja-build
    pkg-config
    poppler-utils
    python3-dev
    python3-venv
    rsync
    wget
    zlib1g-dev
  )

  if [[ "${INSTALL_WRITING_DEPS}" == "1" ]]; then
    packages+=(
      latexmk
      texlive-fonts-recommended
      texlive-lang-other
      texlive-latex-extra
      texlive-science
    )
  fi

  log "Installing Ubuntu/Debian system packages."
  "${sudo_cmd[@]}" apt-get update
  DEBIAN_FRONTEND=noninteractive "${sudo_cmd[@]}" apt-get install -y "${packages[@]}"
  if command_exists git; then
    git lfs install >/dev/null 2>&1 || true
  fi
}

require_cuda_toolkit() {
  if ! command_exists nvidia-smi; then
    die "nvidia-smi was not found. Install an NVIDIA driver before running this script."
  fi
  if ! command_exists nvcc; then
    die "nvcc was not found. FlashAttention requires a CUDA toolkit install; use CUDA 12.6 to match the pinned PyTorch wheels."
  fi
}

build_filtered_requirements() {
  local src_file="$1"
  local dst_file="$2"

  python - <<'PY' "${src_file}" "${dst_file}"
from pathlib import Path
import sys

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
blocked_prefixes = (
    "torch==",
    "torchvision==",
    "torchaudio==",
    "nvidia-",
    "accelerate==",
    "deepspeed==",
    "transformers==",
    "flash_attn==",
    "flash-attn==",
)

lines = []
for raw_line in src.read_text().splitlines():
    line = raw_line.strip()
    if not line or line.startswith("#"):
        continue
    lowered = line.lower()
    if lowered.startswith(blocked_prefixes):
        continue
    lines.append(line)

dst.write_text("\n".join(lines) + "\n")
PY
}

main() {
  maybe_install_apt_packages
  require_cuda_toolkit

  local python_exec
  python_exec="$(pick_python)"
  log "Using ${python_exec}."

  local uv_exec
  uv_exec="$(pick_uv)"
  log "Using $(${uv_exec} --version)."

  if [[ ! -d "${VENV_DIR}" ]]; then
    log "Creating virtual environment at ${VENV_DIR}."
    "${uv_exec}" venv "${VENV_DIR}" --python "${python_exec}" --seed
  fi

  # shellcheck disable=SC1090
  source "${VENV_DIR}/bin/activate"

  export PYTHONNOUSERSITE=1
  export UV_PROJECT_ENVIRONMENT="${VENV_DIR}"
  log "Using virtualenv interpreter ${VENV_DIR}/bin/python."

  log "Upgrading seed packaging tooling."
  "${uv_exec}" pip install --python "${VENV_DIR}/bin/python" --upgrade pip setuptools wheel packaging

  log "Installing PyTorch ${TORCH_VERSION} (${TORCH_CUDA_TAG})."
  "${uv_exec}" pip install --python "${VENV_DIR}/bin/python" \
    --index-url "${TORCH_INDEX_URL}" \
    "torch==${TORCH_VERSION}+${TORCH_CUDA_TAG}" \
    "torchvision==${TORCHVISION_VERSION}+${TORCH_CUDA_TAG}" \
    "torchaudio==${TORCHAUDIO_VERSION}+${TORCH_CUDA_TAG}"

  FILTERED_REQUIREMENTS_PATH="$(mktemp)"
  build_filtered_requirements "${ROOT_DIR}/requirements.txt" "${FILTERED_REQUIREMENTS_PATH}"

  log "Installing Python dependencies from requirements.txt."
  "${uv_exec}" pip install --python "${VENV_DIR}/bin/python" -r "${FILTERED_REQUIREMENTS_PATH}"

  log "Installing Modal and streaming cache support."
  "${uv_exec}" pip install --python "${VENV_DIR}/bin/python" \
    "modal==${MODAL_VERSION}" \
    "mosaicml-streaming==${STREAMING_VERSION}"

  log "Installing training runtime pins."
  "${uv_exec}" pip install --python "${VENV_DIR}/bin/python" --upgrade \
    "accelerate==${ACCELERATE_VERSION}" \
    "deepspeed==${DEEPSPEED_VERSION}" \
    "transformers==${TRANSFORMERS_VERSION}"

  log "Installing FlashAttention ${FLASH_ATTN_VERSION}."
  "${uv_exec}" pip install --python "${VENV_DIR}/bin/python" --no-build-isolation "flash-attn==${FLASH_ATTN_VERSION}"

  log "Verifying core imports."
  python - <<'PY'
import importlib
import sys

import accelerate
import deepspeed
import torch
import transformers

required_modules = [
    "accelerate",
    "flash_attn",
    "modal",
    "streaming",
]
for name in required_modules:
    importlib.import_module(name)

required_symbols = [
    "AutoModelForImageTextToText",
    "AutoProcessor",
]
missing = [name for name in required_symbols if not hasattr(transformers, name)]
if missing:
    raise RuntimeError(f"transformers is missing expected symbols: {missing}")

print("python:", sys.version)
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("accelerate:", accelerate.__version__)
print("deepspeed:", deepspeed.__version__)
print("transformers:", transformers.__version__)
PY

  cat <<EOF

[install] environment is ready.
[install] activate it with:
  source "${VENV_DIR}/bin/activate"

[install] common next steps:
  "${VENV_DIR}/bin/python" src/load_data.py --dataset docvqa --output-root data
  bash "${ROOT_DIR}/scripts/train/distill_multi_teachers.sh"

EOF
}

main "$@"
