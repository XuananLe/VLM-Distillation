#!/bin/bash

export PYTHONPATH=src:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Models
TEACHER_MODEL_1="Qwen/Qwen2-VL-2B-Instruct"
TEACHER_MODEL_2="Qwen/Qwen2.5-VL-3B-Instruct"
TEACHER_MODEL_IDS="[\"${TEACHER_MODEL_1}\", \"${TEACHER_MODEL_2}\"]"
STUDENT_MODEL="HuggingFaceTB/SmolVLM-500M-Instruct"

# Distillation strategy
# Available strategies: uniform_mean, routing, gradient_optimal, reinforced_selection
TEACHER_WEIGHTING_STRATEGY="${TEACHER_WEIGHTING_STRATEGY:-routing}"
# Objective conflict strategies: fixed, pcgrad, cagrad, mgda
OBJECTIVE_CONFLICT_STRATEGY="${OBJECTIVE_CONFLICT_STRATEGY:-fixed}"
OBJECTIVE_CONFLICT_CAGRAD_C="${OBJECTIVE_CONFLICT_CAGRAD_C:-0.5}"
OBJECTIVE_CONFLICT_CAGRAD_GRID_STEPS="${OBJECTIVE_CONFLICT_CAGRAD_GRID_STEPS:-257}"
DISTILLATION_LOSS="uld_loss"
TEMPERATURE=1.0
STUDENT_TEMPERATURE="${STUDENT_TEMPERATURE:-$TEMPERATURE}"
TEACHER_TEMPERATURE="${TEACHER_TEMPERATURE:-$TEMPERATURE}"

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
GRACE_THRESHOLD="${GRACE_THRESHOLD:--0.02}"
GRACE_WARMUP_RATIO="${GRACE_WARMUP_RATIO:-0.2}"
GRACE_EPSILON="${GRACE_EPSILON:-0.01}"
GRACE_SOFTMAX_BETA="${GRACE_SOFTMAX_BETA:-20.0}"
GRACE_ROUTER_BLEND_LAMBDA="${GRACE_ROUTER_BLEND_LAMBDA:-0.5}"
GRACE_EMA_DECAY="${GRACE_EMA_DECAY:-0.9}"

# Gradient-optimized weighting knobs
GRADIENT_WEIGHT_CAP="${GRADIENT_WEIGHT_CAP:-1.0}"
GRADIENT_WEIGHT_STEPS="${GRADIENT_WEIGHT_STEPS:-50}"

# Reinforced teacher-selection knobs
REINFORCED_SELECTION_WARMUP_RATIO="${REINFORCED_SELECTION_WARMUP_RATIO:-0.1}"
REINFORCED_SELECTION_REWARD_TYPE="${REINFORCED_SELECTION_REWARD_TYPE:-reward2}"
REINFORCED_SELECTION_REWARD_EMA_DECAY="${REINFORCED_SELECTION_REWARD_EMA_DECAY:-0.9}"
REINFORCED_SELECTION_POLICY_ALPHA="${REINFORCED_SELECTION_POLICY_ALPHA:-1.0}"

# Optional cached teacher logits
TEACHER_LOGITS_CACHE_DIR="${TEACHER_LOGITS_CACHE_DIR:-}"

# Dataset and runtime
NUM_TEACHERS=2
DATASET_NAME="docvqa"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-18}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"

# Naming
STUDENT_NAME="${STUDENT_MODEL##*/}"
TEACHER_NAME_1="${TEACHER_MODEL_1##*/}"
TEACHER_NAME_2="${TEACHER_MODEL_2##*/}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M)}"
OUTPUT_DIR="/output/${DISTILLATION_LOSS}_${NUM_TEACHERS}_teachers_${TEACHER_NAME_1}_${TEACHER_NAME_2}_${STUDENT_NAME}_${DATASET_NAME}_${RUN_TAG}"

EXTRA_ARGS=()
if [[ -n "$TEACHER_LOGITS_CACHE_DIR" ]]; then
    EXTRA_ARGS+=(--teacher_logits_cache_dir "$TEACHER_LOGITS_CACHE_DIR")
fi

deepspeed src/train/train_distillation.py \
    --deepspeed scripts/deepspeed/zero2.json \
    --student_model_id "$STUDENT_MODEL" \
    --teacher_model_ids "$TEACHER_MODEL_IDS" \
    --teacher_weighting_strategy "$TEACHER_WEIGHTING_STRATEGY" \
    --objective_conflict_strategy "$OBJECTIVE_CONFLICT_STRATEGY" \
    --objective_conflict_cagrad_c "$OBJECTIVE_CONFLICT_CAGRAD_C" \
    --objective_conflict_cagrad_grid_steps "$OBJECTIVE_CONFLICT_CAGRAD_GRID_STEPS" \
    --data_path /data/${DATASET_NAME}/train_llava.json \
    --image_folder /data/${DATASET_NAME}/images \
    --distillation_loss "$DISTILLATION_LOSS" \
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
    --gradient_weight_cap "$GRADIENT_WEIGHT_CAP" \
    --gradient_weight_steps "$GRADIENT_WEIGHT_STEPS" \
    --num_train_epochs 1 \
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
