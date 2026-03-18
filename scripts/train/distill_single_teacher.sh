#!/bin/bash

export PYTHONPATH=src:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TEACHER_MODEL="Qwen/Qwen2-VL-2B-Instruct"
STUDENT_MODEL="HuggingFaceTB/SmolVLM-256M-Instruct"
DISTILLATION_LOSS="uld_loss"
TEMPERATURE=1.0
ALPHA=0.5


deepspeed src/train/train_distillation.py \
    --deepspeed scripts/deepspeed/zero2.json \
    --model_id "$STUDENT_MODEL" \
    --teacher_model_id "$TEACHER_MODEL" \
    --data_path /data/textvqa/train_llava.json \
    --image_folder /data/textvqa/images \
    --distillation_loss "$DISTILLATION_LOSS" \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir /output/uld_qwen_smolvlm_500m_textvqa \
    --temperature "$TEMPERATURE" \
    --alpha "$ALPHA" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 28 \
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
    --save_steps 400 \
    --save_total_limit 3 \
    --eval_strategy no \
    --dataloader_num_workers 4 \
    --remove_unused_columns False \
    --report_to wandb
