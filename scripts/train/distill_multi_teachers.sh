#!/bin/bash

export PYTHONPATH=src:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TEACHER_MODEL_1="Qwen/Qwen2-VL-2B-Instruct"
TEACHER_MODEL_2="Qwen/Qwen2.5-VL-3B-Instruct"
TEACHER_MODEL_IDS="[\"${TEACHER_MODEL_1}\", \"${TEACHER_MODEL_2}\"]"
STUDENT_MODEL="HuggingFaceTB/SmolVLM-500M-Instruct"
TEACHER_WEIGHTING_STRATEGY="${TEACHER_WEIGHTING_STRATEGY:-routing}"
DISTILLATION_LOSS="uld_loss"
TEMPERATURE=1.0
STUDENT_TEMPERATURE="${STUDENT_TEMPERATURE:-$TEMPERATURE}"
TEACHER_TEMPERATURE="${TEACHER_TEMPERATURE:-$TEMPERATURE}"
ALPHA=0.5
TEACHER_GATE_TOP_K="${TEACHER_GATE_TOP_K:-2}"
TEACHER_GATE_BALANCE_ALPHA="${TEACHER_GATE_BALANCE_ALPHA:-0.01}"
TEACHER_GATE_CAPACITY_FACTOR="${TEACHER_GATE_CAPACITY_FACTOR:-1.25}"
TEACHER_GATE_BIAS_UPDATE_RATE="${TEACHER_GATE_BIAS_UPDATE_RATE:-5e-4}"
TEACHER_GATE_ROUTER_Z_LOSS_ALPHA="${TEACHER_GATE_ROUTER_Z_LOSS_ALPHA:-1e-3}"
GRADIENT_ALIGNMENT_THRESHOLD="${GRADIENT_ALIGNMENT_THRESHOLD:--0.02}"
GRADIENT_ALIGNMENT_WARMUP_RATIO="${GRADIENT_ALIGNMENT_WARMUP_RATIO:-0.2}"
GRADIENT_ALIGNMENT_EPSILON="${GRADIENT_ALIGNMENT_EPSILON:-0.01}"
GRADIENT_ALIGNMENT_SOFTMAX_BETA="${GRADIENT_ALIGNMENT_SOFTMAX_BETA:-20.0}"
GRADIENT_ALIGNMENT_ROUTER_BLEND_LAMBDA="${GRADIENT_ALIGNMENT_ROUTER_BLEND_LAMBDA:-0.5}"
GRADIENT_ALIGNMENT_EMA_DECAY="${GRADIENT_ALIGNMENT_EMA_DECAY:-0.9}"
NUM_TEACHERS=2
DATASET_NAME="docvqa"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-18}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"
DATALOADER_PERSISTENT_WORKERS="${DATALOADER_PERSISTENT_WORKERS:-True}"
DATALOADER_PREFETCH_FACTOR="${DATALOADER_PREFETCH_FACTOR:-4}"
STUDENT_NAME="${STUDENT_MODEL##*/}"
TEACHER_NAME_1="${TEACHER_MODEL_1##*/}"
TEACHER_NAME_2="${TEACHER_MODEL_2##*/}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M)}"
OUTPUT_DIR="/output/${DISTILLATION_LOSS}_${NUM_TEACHERS}_teachers_${TEACHER_NAME_1}_${TEACHER_NAME_2}_${STUDENT_NAME}_${DATASET_NAME}_${RUN_TAG}"

deepspeed src/train/train_distillation.py \
    --deepspeed scripts/deepspeed/zero2.json \
    --student_model_id "$STUDENT_MODEL" \
    --teacher_model_ids "$TEACHER_MODEL_IDS" \
    --teacher_weighting_strategy "$TEACHER_WEIGHTING_STRATEGY" \
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
    --teacher_gate_router_z_loss_alpha "$TEACHER_GATE_ROUTER_Z_LOSS_ALPHA" \
    --gradient_alignment_threshold "$GRADIENT_ALIGNMENT_THRESHOLD" \
    --gradient_alignment_warmup_ratio "$GRADIENT_ALIGNMENT_WARMUP_RATIO" \
    --gradient_alignment_epsilon "$GRADIENT_ALIGNMENT_EPSILON" \
    --gradient_alignment_softmax_beta "$GRADIENT_ALIGNMENT_SOFTMAX_BETA" \
    --gradient_alignment_router_blend_lambda "$GRADIENT_ALIGNMENT_ROUTER_BLEND_LAMBDA" \
    --gradient_alignment_ema_decay "$GRADIENT_ALIGNMENT_EMA_DECAY" \
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
    --dataloader_num_workers "$DATALOADER_NUM_WORKERS" \
    --dataloader_persistent_workers "$DATALOADER_PERSISTENT_WORKERS" \
    --dataloader_prefetch_factor "$DATALOADER_PREFETCH_FACTOR" \
    --remove_unused_columns False \
    --report_to wandb
