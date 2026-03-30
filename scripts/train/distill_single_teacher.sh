#!/bin/bash

export PYTHONPATH=src:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TEACHER_MODEL="Qwen/Qwen2-VL-2B-Instruct"
TEACHER_MODEL_IDS="[\"${TEACHER_MODEL}\"]"
STUDENT_MODEL="HuggingFaceTB/SmolVLM-500M-Instruct"
DISTILLATION_LOSS="uld_loss"
TEMPERATURE=1.0
STUDENT_TEMPERATURE="${STUDENT_TEMPERATURE:-$TEMPERATURE}"
TEACHER_TEMPERATURE="${TEACHER_TEMPERATURE:-$TEMPERATURE}"
ALPHA=0.5
DATASET_NAME="textvqa"
EVAL_SPLIT="validation"
POST_SAVE_EVAL_ROOT="${POST_SAVE_EVAL_ROOT:-/output/vlmeval}"
STUDENT_NAME="${STUDENT_MODEL##*/}"
TEACHER_NAME="${TEACHER_MODEL##*/}"
OUTPUT_DIR="/output/${DISTILLATION_LOSS}_single_teacher_${TEACHER_NAME}_${STUDENT_NAME}_${DATASET_NAME}"

deepspeed src/train/train_distillation.py \
    --deepspeed scripts/deepspeed/zero2.json \
    --student_model_id "$STUDENT_MODEL" \
    --teacher_model_ids "$TEACHER_MODEL_IDS" \
    --data_path /data/${DATASET_NAME}/train_llava.json \
    --eval_data_path /data/${DATASET_NAME}/${EVAL_SPLIT}_llava.json \
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
    --post_save_eval_root "$POST_SAVE_EVAL_ROOT" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 1 \
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
    --logging_steps 10 \
    --save_strategy steps \
    --save_steps 100 \
    --save_total_limit 3 \
    --save_only_model False \
    --eval_strategy no \
    --dataloader_num_workers 4 \
    --remove_unused_columns False \
    --report_to wandb
