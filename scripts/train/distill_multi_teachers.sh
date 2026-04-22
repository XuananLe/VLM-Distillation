#!/bin/bash

export PYTHONPATH=src:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

WANDB_RESUME_RUN_PATH=""
WANDB_RESUME_MODE="${WANDB_RESUME_MODE:-allow}"
if [[ -n "${WANDB_RESUME_RUN_PATH}" ]]; then
    IFS='/' read -r WANDB_RESUME_ENTITY WANDB_RESUME_PROJECT WANDB_RESUME_ID <<< "${WANDB_RESUME_RUN_PATH}"
    if [[ -z "${WANDB_RESUME_ENTITY}" || -z "${WANDB_RESUME_PROJECT}" || -z "${WANDB_RESUME_ID}" ]]; then
        exit 1
        echo "Invalid WANDB_RESUME_RUN_PATH: ${WANDB_RESUME_RUN_PATH}" >&2
        echo "Expected format: <entity>/<project>/<run_id>" >&2
    fi
    export WANDB_ENTITY="${WANDB_ENTITY:-$WANDB_RESUME_ENTITY}"
    export WANDB_PROJECT="${WANDB_PROJECT:-$WANDB_RESUME_PROJECT}"
    export WANDB_RUN_ID="${WANDB_RUN_ID:-$WANDB_RESUME_ID}"
    export WANDB_RESUME="${WANDB_RESUME:-$WANDB_RESUME_MODE}"
fi

# Models
TEACHER_MODEL_1="Qwen/Qwen2.5-VL-3B-Instruct"
TEACHER_MODEL_2="Qwen/Qwen2-VL-2B-Instruct"
TEACHER_MODEL_3="ibm-granite/granite-vision-3.1-2b-preview"
TEACHER_MODEL_4="google/gemma-3-4b-it"
TEACHER_MODEL_IDS="[\"${TEACHER_MODEL_1}\", \"${TEACHER_MODEL_2}\", \"${TEACHER_MODEL_3}\", \"${TEACHER_MODEL_4}\"]"
STUDENT_MODEL="HuggingFaceTB/SmolVLM-500M-Instruct"

# Distillation strategy
# Available strategies: uniform_mean, routing, reinforced_selection
TEACHER_WEIGHTING_STRATEGY="${TEACHER_WEIGHTING_STRATEGY:-routing}"

DISTILLATION_LOSS="trie_wasserstein_loss"
TEMPERATURE=1.0
STUDENT_TEMPERATURE="${STUDENT_TEMPERATURE:-$TEMPERATURE}"
TEACHER_TEMPERATURE="${TEACHER_TEMPERATURE:-$TEMPERATURE}"
# Trie-loss sensitivity knobs
TRIE_WASSERSTEIN_RHO="${TRIE_WASSERSTEIN_RHO:-0.7}"
TRIE_WASSERSTEIN_TOPK="${TRIE_WASSERSTEIN_TOPK:-64}"

# For fixed weighting strategy and as a base alpha for other strategies
ALPHA=0.5

# Router-based weighting knobs
TEACHER_GATE_TOP_K="${TEACHER_GATE_TOP_K:-2}"
TEACHER_GATE_BALANCE_ALPHA="${TEACHER_GATE_BALANCE_ALPHA:-0.01}"
TEACHER_GATE_CAPACITY_FACTOR="${TEACHER_GATE_CAPACITY_FACTOR:-1.25}"
TEACHER_GATE_BIAS_UPDATE_RATE="${TEACHER_GATE_BIAS_UPDATE_RATE:-5e-4}"
TEACHER_GATE_TEMPERATURE="${TEACHER_GATE_TEMPERATURE:-1.5}"
TEACHER_GATE_NOISE_STD="${TEACHER_GATE_NOISE_STD:-0.01}"
TEACHER_GATE_ENTROPY_ALPHA="${TEACHER_GATE_ENTROPY_ALPHA:-1e-3}"
TEACHER_GATE_ROUTER_Z_LOSS_ALPHA="${TEACHER_GATE_ROUTER_Z_LOSS_ALPHA:-1e-3}"
TEACHER_GATE_HARD_ROUTING_WARMUP_RATIO="${TEACHER_GATE_HARD_ROUTING_WARMUP_RATIO:-0.2}"

# GRACE weighting knobs
GRACE_THRESHOLD="${GRACE_THRESHOLD:--0.005}"
GRACE_WARMUP_RATIO="${GRACE_WARMUP_RATIO:-0.01}"
GRACE_EPSILON="${GRACE_EPSILON:-5e-4}"
GRACE_SOFTMAX_BETA="${GRACE_SOFTMAX_BETA:-20.0}"
GRACE_ROUTER_BLEND_LAMBDA="${GRACE_ROUTER_BLEND_LAMBDA:-0.4}"
GRACE_EMA_DECAY="${GRACE_EMA_DECAY:-0.0}"

# Reinforced teacher-selection knobs
REINFORCED_SELECTION_WARMUP_RATIO="${REINFORCED_SELECTION_WARMUP_RATIO:-0.1}"
REINFORCED_SELECTION_REWARD_TYPE="${REINFORCED_SELECTION_REWARD_TYPE:-reward2}"
REINFORCED_SELECTION_REWARD_EMA_DECAY="${REINFORCED_SELECTION_REWARD_EMA_DECAY:-0.9}"
REINFORCED_SELECTION_POLICY_ALPHA="${REINFORCED_SELECTION_POLICY_ALPHA:-1.0}"

# Teacher logits can be read either from a local cache root (for example /cache)
# or directly from the remote raw .pt cache layout.
# To force local mode, set TEACHER_LOGITS_REMOTE_URI="" and optionally
# TEACHER_LOGITS_CACHE_DIR=/cache.
TEACHER_LOGITS_REMOTE_URI="${TEACHER_LOGITS_REMOTE_URI-}"
if [[ -n "${TEACHER_LOGITS_REMOTE_URI}" ]]; then
    TEACHER_LOGITS_CACHE_DIR="${TEACHER_LOGITS_CACHE_DIR:-/tmp/teacher-logits-streaming}"
else
    TEACHER_LOGITS_CACHE_DIR="${TEACHER_LOGITS_CACHE_DIR:-/workspace/cache}"
fi

# Dataset and runtime
NUM_TEACHERS=4
DATASET_NAME="docvqa"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1.0}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-80}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
LOGGING_STEPS="${LOGGING_STEPS:-5}"
AUTO_STOP_VAST_INSTANCE="${AUTO_STOP_VAST_INSTANCE:-1}"

# Naming
STUDENT_NAME="${STUDENT_MODEL##*/}"
TEACHER_NAME_1="${TEACHER_MODEL_1##*/}"
TEACHER_NAME_2="${TEACHER_MODEL_2##*/}"
TEACHER_NAME_3="${TEACHER_MODEL_3##*/}"
TEACHER_NAME_4="${TEACHER_MODEL_4##*/}"
TRIE_WASSERSTEIN_RHO_TAG="${TRIE_WASSERSTEIN_RHO//./p}"
RUN_TAG="${RUN_TAG:-trie_sensitivity_rho${TRIE_WASSERSTEIN_RHO_TAG}_topk${TRIE_WASSERSTEIN_TOPK}_$(date +%Y%m%d_%H%M)}"
OUTPUT_DIR="output/${DISTILLATION_LOSS}_${NUM_TEACHERS}_teachers_${TEACHER_NAME_1}_${TEACHER_NAME_2}_${TEACHER_NAME_3}_${TEACHER_NAME_4}_${STUDENT_NAME}_${DATASET_NAME}_${RUN_TAG}"

EXTRA_ARGS=(
    --teacher_logits_cache_dir "$TEACHER_LOGITS_CACHE_DIR"
)
if [[ -n "${TEACHER_LOGITS_REMOTE_URI}" ]]; then
    EXTRA_ARGS+=(--teacher_logits_remote_uri "$TEACHER_LOGITS_REMOTE_URI")
fi

stop_vast_instance() {
    if [[ "${AUTO_STOP_VAST_INSTANCE}" == "0" ]]; then
        return 0
    fi

    local vast_instance_id="${VAST_INSTANCE_ID:-${CONTAINER_ID:-${VAST_CONTAINERLABEL:-}}}"
    vast_instance_id="${vast_instance_id#C.}"

    if [[ -z "${vast_instance_id}" ]]; then
        echo "Skipping Vast.ai stop: instance id not found in VAST_CONTAINERLABEL/CONTAINER_ID/VAST_INSTANCE_ID." >&2
        return 0
    fi

    if ! command -v vastai >/dev/null 2>&1; then
        echo "Installing Vast.ai CLI before stopping instance ${vast_instance_id}..."
        if ! python -m pip install --disable-pip-version-check -q vastai; then
            echo "Failed to install Vast.ai CLI; instance ${vast_instance_id} was not stopped." >&2
            return 1
        fi
    fi

    local stop_args=()
    if [[ -n "${CONTAINER_API_KEY:-}" ]]; then
        stop_args+=(--api-key "${CONTAINER_API_KEY}")
    fi

    echo "Stopping Vast.ai instance ${vast_instance_id}..."
    if ! vastai stop instance "${stop_args[@]}" "${vast_instance_id}"; then
        echo "Failed to stop Vast.ai instance ${vast_instance_id}." >&2
        return 1
    fi
}

deepspeed src/train/train_distillation.py \
    --deepspeed scripts/deepspeed/zero2.json \
    --student_model_id "$STUDENT_MODEL" \
    --teacher_model_ids "$TEACHER_MODEL_IDS" \
    --teacher_weighting_strategy "$TEACHER_WEIGHTING_STRATEGY" \
    --data_path data/${DATASET_NAME}/train_llava.json \
    --image_folder data/${DATASET_NAME}/images \
    --distillation_loss "$DISTILLATION_LOSS" \
    --trie_wasserstein_rho "$TRIE_WASSERSTEIN_RHO" \
    --trie_wasserstein_topk "$TRIE_WASSERSTEIN_TOPK" \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir "$OUTPUT_DIR" \
    --temperature "$TEMPERATURE" \
    --student_temperature "$STUDENT_TEMPERATURE" \
    --teacher_temperature "$TEACHER_TEMPERATURE" \
    --alpha "$ALPHA" \
    --teacher_gate_balance_alpha "$TEACHER_GATE_BALANCE_ALPHA" \
    --teacher_gate_top_k "$TEACHER_GATE_TOP_K" \
    --teacher_gate_capacity_factor "$TEACHER_GATE_CAPACITY_FACTOR" \
    --teacher_gate_bias_update_rate "$TEACHER_GATE_BIAS_UPDATE_RATE" \
    --teacher_gate_temperature "$TEACHER_GATE_TEMPERATURE" \
    --teacher_gate_noise_std "$TEACHER_GATE_NOISE_STD" \
    --teacher_gate_entropy_alpha "$TEACHER_GATE_ENTROPY_ALPHA" \
    --teacher_gate_router_z_loss_alpha "$TEACHER_GATE_ROUTER_Z_LOSS_ALPHA" \
    --teacher_gate_hard_routing_warmup_ratio "$TEACHER_GATE_HARD_ROUTING_WARMUP_RATIO" \
    --grace_threshold "$GRACE_THRESHOLD" \
    --grace_warmup_ratio "$GRACE_WARMUP_RATIO" \
    --grace_epsilon "$GRACE_EPSILON" \
    --grace_softmax_beta "$GRACE_SOFTMAX_BETA" \
    --grace_router_blend_lambda "$GRACE_ROUTER_BLEND_LAMBDA" \
    --grace_ema_decay "$GRACE_EMA_DECAY" \
    --reinforced_selection_warmup_ratio "$REINFORCED_SELECTION_WARMUP_RATIO" \
    --reinforced_selection_reward_type "$REINFORCED_SELECTION_REWARD_TYPE" \
    --reinforced_selection_reward_ema_decay "$REINFORCED_SELECTION_REWARD_EMA_DECAY" \
    --reinforced_selection_policy_alpha "$REINFORCED_SELECTION_POLICY_ALPHA" \
    --num_train_epochs "$NUM_TRAIN_EPOCHS" \
    --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --learning_rate 1e-5 \
    --vision_lr 2e-6 \
    --connector_lr 1e-5 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --freeze_vision_tower False \
    --freeze_llm False \
    --freeze_connector False \
    --tf32 True \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --logging_steps "$LOGGING_STEPS" \
    --save_strategy steps \
    --save_steps 150 \
    --save_total_limit 100 \
    --save_only_model False \
    --eval_strategy no \
    --dataloader_num_workers 4 \
    --remove_unused_columns False \
    --report_to wandb \
    "${EXTRA_ARGS[@]}"

TRAIN_EXIT_CODE=$?
if [[ "${TRAIN_EXIT_CODE}" -eq 1 ]]; then
    echo "Training exited with code 1; leaving the instance running for inspection." >&2
    while true; do
        sleep 60000
    done
fi
stop_vast_instance
exit "${TRAIN_EXIT_CODE}"
