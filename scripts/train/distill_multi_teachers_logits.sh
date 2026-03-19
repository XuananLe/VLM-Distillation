#!/bin/bash

export PYTHONPATH=src:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TEACHER_MODEL_1="Qwen/Qwen2-VL-2B-Instruct"
TEACHER_MODEL_2="Qwen/Qwen2.5-VL-3B-Instruct"
TEACHER_MODEL_IDS="[\"${TEACHER_MODEL_1}\", \"${TEACHER_MODEL_2}\"]"
STUDENT_MODEL="HuggingFaceTB/SmolVLM-256M-Instruct"
DISTILLATION_LOSS="uld_loss"
TEMPERATURE=1.0
ALPHA=0.5
REPRESENTATION_LOSS_WEIGHT=0.2
NUM_TEACHERS=2
DATASET_NAME="textvqa"
PER_DEVICE_TRAIN_BATCH_SIZE=26
GRADIENT_ACCUMULATION_STEPS=1
TEACHER_NAME_1="${TEACHER_MODEL_1##*/}"
TEACHER_NAME_2="${TEACHER_MODEL_2##*/}"
OUTPUT_DIR="/output/${DISTILLATION_LOSS}_${NUM_TEACHERS}_teachers_${TEACHER_NAME_1}_${TEACHER_NAME_2}_${DATASET_NAME}"

deepspeed src/train/train_distillation.py \
    --deepspeed scripts/deepspeed/zero2.json \
    --student_model_id "$STUDENT_MODEL" \
    --teacher_model_ids "$TEACHER_MODEL_IDS" \
    --data_path /data/textvqa/train_llava.json \
    --image_folder /data/textvqa/images \
    --distillation_loss "$DISTILLATION_LOSS" \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir "$OUTPUT_DIR" \
    --temperature "$TEMPERATURE" \
    --alpha "$ALPHA" \
    --representation_loss_weight "$REPRESENTATION_LOSS_WEIGHT" \
    --num_train_epochs 1 \
    --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --learning_rate 1e-5 \
    --vision_lr 2e-6 \
    --connector_lr 1e-5 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --lora_enable True \
    --freeze_vision_tower True \
    --freeze_llm True \
    --freeze_connector True \
    --tf32 True \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --logging_steps 1 \
    --save_strategy steps \
    --save_steps 150 \
    --save_total_limit 10 \
    --eval_strategy no \
    --dataloader_num_workers 4 \
    --remove_unused_columns False \
    --report_to wandb
