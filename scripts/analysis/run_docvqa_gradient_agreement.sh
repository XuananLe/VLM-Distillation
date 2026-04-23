#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if [[ -x "./venv/bin/python" ]]; then
  DEFAULT_PYTHON_BIN="./venv/bin/python"
else
  DEFAULT_PYTHON_BIN="python"
fi

STUDENT_MODEL_ID="${STUDENT_MODEL_ID:-HuggingFaceTB/SmolVLM-500M-Instruct}"
TEACHER_MODEL_ID="${TEACHER_MODEL_ID:-Qwen/Qwen2-VL-2B-Instruct}"
DATASET="${DATASET:-docvqa}"
SPLIT="${SPLIT:-train}"
SUBSET_SIZE="${SUBSET_SIZE:-1000}"
OFFSET="${OFFSET:-0}"
BATCH_SIZE="${BATCH_SIZE:-1}"
STUDENT_TEMPERATURE="${STUDENT_TEMPERATURE:-2.0}"
TEACHER_TEMPERATURE="${TEACHER_TEMPERATURE:-2.0}"
LOSS_FUNCTION="${LOSS_FUNCTION:-uld_loss}"
DTYPE="${DTYPE:-bfloat16}"
DEVICE="${DEVICE:-cuda}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
CACHE_DIR="${CACHE_DIR:-${HF_HOME:-}}"
OUTPUT_JSON="${OUTPUT_JSON:-/output/gradient_agreement/docvqa/qwen2_vl_2b_vs_smolvlm500m_1000samples/report.json}"
PROGRESS_EVERY="${PROGRESS_EVERY:-25}"
PYTHON_BIN="${PYTHON_BIN:-$DEFAULT_PYTHON_BIN}"

cmd=(
  "$PYTHON_BIN"
  scripts/analysis/gradient_agreement.py
  --student-model-id "$STUDENT_MODEL_ID" \
  --teacher-model-id "$TEACHER_MODEL_ID" \
  --dataset "$DATASET" \
  --split "$SPLIT" \
  --subset-size "$SUBSET_SIZE" \
  --offset "$OFFSET" \
  --batch-size "$BATCH_SIZE" \
  --student-temperature "$STUDENT_TEMPERATURE" \
  --teacher-temperature "$TEACHER_TEMPERATURE" \
  --loss-function "$LOSS_FUNCTION" \
  --dtype "$DTYPE" \
  --device "$DEVICE" \
  --attn-implementation "$ATTN_IMPLEMENTATION" \
  --output-json "$OUTPUT_JSON" \
  --progress-every "$PROGRESS_EVERY"
)

if [[ -n "$CACHE_DIR" ]]; then
  cmd+=(--cache-dir "$CACHE_DIR")
fi

"${cmd[@]}"
