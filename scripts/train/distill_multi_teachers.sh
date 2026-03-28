#!/bin/bash

export PYTHONPATH=src:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TEACHER_MODEL_1="Qwen/Qwen2-VL-2B-Instruct"
TEACHER_MODEL_2="Qwen/Qwen2.5-VL-3B-Instruct"
TEACHER_MODEL_3="OpenGVLab/InternVL3-1B"
TEACHER_MODEL_4="google/gemma-3-4b-it"
TEACHER_MODEL_IDS="[\"${TEACHER_MODEL_1}\", \"${TEACHER_MODEL_2}\", \"${TEACHER_MODEL_3}\", \"${TEACHER_MODEL_4}\"]"
STUDENT_MODEL="HuggingFaceTB/SmolVLM-500M-Instruct"
DISTILLATION_LOSS="uld_loss"
TEMPERATURE=1.0
KD_LOSS_ALPHA=1.5
TEACHER_GATE_BALANCE_ALPHA=5e-2
TEACHER_GATE_TOP_K=2
TEACHER_GATE_CAPACITY_FACTOR=1.25
TEACHER_GATE_BIAS_UPDATE_RATE=1e-3
NUM_TEACHERS=4
DATASET_NAME="chartqa"
EVAL_SPLIT="val"
PER_DEVICE_TRAIN_BATCH_SIZE=32
GRADIENT_ACCUMULATION_STEPS=1
STUDENT_NAME="${STUDENT_MODEL##*/}"
TEACHER_NAME_1="${TEACHER_MODEL_1##*/}"
TEACHER_NAME_2="${TEACHER_MODEL_2##*/}"
TEACHER_NAME_3="${TEACHER_MODEL_3##*/}"
TEACHER_NAME_4="${TEACHER_MODEL_4##*/}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="/output/${DISTILLATION_LOSS}_${NUM_TEACHERS}_teachers_${TEACHER_NAME_1}_${TEACHER_NAME_2}_${TEACHER_NAME_3}_${TEACHER_NAME_4}_${STUDENT_NAME}_${DATASET_NAME}_${RUN_TAG}"

deepspeed src/train/train_distillation.py \
    --deepspeed scripts/deepspeed/zero2.json \
    --student_model_id "$STUDENT_MODEL" \
    --teacher_model_ids "$TEACHER_MODEL_IDS" \
    --data_path /data/chartqa/train_llava.json \
    --eval_data_path /data/${DATASET_NAME}/${EVAL_SPLIT}_llava.json \
    --image_folder /data/chartqa/images \
    --distillation_loss "$DISTILLATION_LOSS" \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir "$OUTPUT_DIR" \
    --temperature "$TEMPERATURE" \
    --kd_loss_alpha "$KD_LOSS_ALPHA" \
    --teacher_gate_balance_alpha "$TEACHER_GATE_BALANCE_ALPHA" \
    --teacher_gate_top_k "$TEACHER_GATE_TOP_K" \
    --teacher_gate_capacity_factor "$TEACHER_GATE_CAPACITY_FACTOR" \
    --teacher_gate_bias_update_rate "$TEACHER_GATE_BIAS_UPDATE_RATE" \
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
    --tf32 False \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --logging_steps 10 \
    --save_strategy steps \
    --save_steps 100 \
    --save_total_limit 10 \
    --save_only_model False \
    --eval_strategy steps \
    --eval_steps 100 \
    --per_device_eval_batch_size 20 \
    --load_best_model_at_end True \
    --metric_for_best_model eval_loss \
    --greater_is_better False \
    --dataloader_num_workers 4 \
    --remove_unused_columns False \
    --report_to wandb
