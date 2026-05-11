#!/bin/bash

export PYTHONPATH=src:$PYTHONPATH

MODEL_NAME="${MODEL_NAME:-HuggingFaceTB/SmolVLM-500M-Instruct}"
DATASET_NAME="${DATASET_NAME:-docvqa}"
DATA_PATH="${DATA_PATH:-data/${DATASET_NAME}/train_llava.json}"
IMAGE_FOLDER="${IMAGE_FOLDER:-data/${DATASET_NAME}/images}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1.0}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-40}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
LOGGING_STEPS="${LOGGING_STEPS:-5}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-scripts/deepspeed/zero3.json}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
VISION_LR="${VISION_LR:-2e-6}"
CONNECTOR_LR="${CONNECTOR_LR:-1e-5}"
REPORT_TO="${REPORT_TO:-wandb}"

MODEL_TAG="${MODEL_NAME##*/}"
RUN_TAG="${RUN_TAG:-vanilla_sft_${MODEL_TAG}_${DATASET_NAME}_$(date +%Y%m%d_%H%M)}"
OUTPUT_DIR="${OUTPUT_DIR:-output/${RUN_TAG}}"

deepspeed src/train/train_sft.py \
    --deepspeed "$DEEPSPEED_CONFIG" \
    --model_id "$MODEL_NAME" \
    --data_path "$DATA_PATH" \
    --image_folder "$IMAGE_FOLDER" \
    --freeze_vision_tower False \
    --freeze_llm False \
    --freeze_connector False \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs "$NUM_TRAIN_EPOCHS" \
    --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --learning_rate "$LEARNING_RATE" \
    --vision_lr "$VISION_LR" \
    --connector_lr "$CONNECTOR_LR" \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps "$LOGGING_STEPS" \
    --tf32 True \
    --gradient_checkpointing True \
    --report_to "$REPORT_TO" \
    --lazy_preprocess True \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 10 \
    --dataloader_num_workers 4
