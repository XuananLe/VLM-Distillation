#!/bin/bash

MODEL_NAME="HuggingFaceTB/SmolVLM-Instruct"

export PYTHONPATH=src:$PYTHONPATH

deepspeed src/train/train_sft.py \
    --deepspeed scripts/zero3.json \
    --model_id $MODEL_NAME \
    --data_path /path/to/your/training/data.json \
    --image_folder /path/to/your/image/folder \
    --freeze_vision_tower False \
    --freeze_llm False \
    --freeze_connector False \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir output/testing \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-5 \
    --connector_lr 1e-5 \
    --vision_lr 2e-6 \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --report_to tensorboard \
    --lazy_preprocess True \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 10 \
    --dataloader_num_workers 4