#!/bin/bash

export PYTHONPATH=src:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TEACHER_MODEL="Qwen/Qwen2-VL-2B-Instruct"
TEACHER_MODEL_IDS="[\"${TEACHER_MODEL}\"]"
STUDENT_MODEL="HuggingFaceTB/SmolVLM-500M-Instruct"
DISTILLATION_LOSS="uld_loss"
TEMPERATURE=1.0
ALPHA=0.5
LOSS_WEIGHTING="gradnorm"
GRADNORM_ALPHA=1.5
GRADNORM_LR=0.025
LAYER_DISTILL_SOURCE="vision"
LAYER_DISTILL_WEIGHT=0.2
STUDENT_LAYER_INDICES="0,1,3,4,5,6,7,8"
TEACHER_LAYER_INDICES="1,4,20,20,20,21,21,21"
DATASET_NAME="docvqa"
STUDENT_NAME="${STUDENT_MODEL##*/}"
TEACHER_NAME="${TEACHER_MODEL##*/}"
OUTPUT_DIR="/output/${DISTILLATION_LOSS}_single_teacher_${TEACHER_NAME}_${STUDENT_NAME}_${DATASET_NAME}"


deepspeed src/train/train_distillation.py \
    --deepspeed scripts/deepspeed/zero2.json \
    --student_model_id "$STUDENT_MODEL" \
    --teacher_model_ids "$TEACHER_MODEL_IDS" \
    --data_path /data/${DATASET_NAME}/train_llava.json \
    --image_folder /data/${DATASET_NAME}/images \
    --distillation_loss "$DISTILLATION_LOSS" \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir "$OUTPUT_DIR" \
    --temperature "$TEMPERATURE" \
    --alpha "$ALPHA" \
    --loss_weighting "$LOSS_WEIGHTING" \
    --gradnorm_alpha "$GRADNORM_ALPHA" \
    --gradnorm_lr "$GRADNORM_LR" \
    --layer_distill_source "$LAYER_DISTILL_SOURCE" \
    --layer_distill_weight "$LAYER_DISTILL_WEIGHT" \
    --student_layer_indices "$STUDENT_LAYER_INDICES" \
    --teacher_layer_indices "$TEACHER_LAYER_INDICES" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 20 \
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
    --logging_steps 1 \
    --save_strategy steps \
    --save_steps 100 \
    --save_total_limit 3 \
    --eval_strategy no \
    --dataloader_num_workers 4 \
    --remove_unused_columns False \
    --report_to wandb
