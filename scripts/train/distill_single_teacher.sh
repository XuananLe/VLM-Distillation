#!/bin/bash

export PYTHONPATH=src:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TEACHER_MODEL="Qwen/Qwen2-VL-2B-Instruct"
TEACHER_MODEL_IDS="[\"${TEACHER_MODEL}\"]"
STUDENT_MODEL="HuggingFaceTB/SmolVLM-500M-Instruct"
DISTILLATION_LOSS="uld_loss"
TEMPERATURE=1.0
LOSS_WEIGHTING="gradnorm"
GRADNORM_ALPHA=1.5
GRADNORM_LR=0.025
LAYER_DISTILL_SOURCE="model"
LAYER_MATCH_JSON_PATH="artifacts/cka_plots_last_layer/chartqa/qwen/qwen2_vl_2b_vs_smolvlm500m/matrix.json"
LAYER_MATCH_TOPK=3
DATASET_NAME="chartqa"
EVAL_SPLIT="val"
TRAIN_SUBSET_SIZE=""
EVAL_SUBSET_SIZE=""
EARLY_STOPPING_PATIENCE="3"
EARLY_STOPPING_THRESHOLD="0.002"
STUDENT_NAME="${STUDENT_MODEL##*/}"
TEACHER_NAME="${TEACHER_MODEL##*/}"
OUTPUT_DIR="/output/${DISTILLATION_LOSS}_single_teacher_${TEACHER_NAME}_${STUDENT_NAME}_${DATASET_NAME}_gradnorm_last10layers_sample"

LAYER_MATCH_ARGS=()
if [[ -n "$LAYER_MATCH_JSON_PATH" ]]; then
    LAYER_MATCH_ARGS=(
        --layer_match_json_path "$LAYER_MATCH_JSON_PATH"
        --layer_match_topk "$LAYER_MATCH_TOPK"
    )
else
    echo "LAYER_MATCH_JSON_PATH must be set for soft layer matching." >&2
    exit 1
fi

SUBSET_ARGS=()
if [[ -n "$TRAIN_SUBSET_SIZE" ]]; then
    SUBSET_ARGS+=(--train_subset_size "$TRAIN_SUBSET_SIZE")
fi
if [[ -n "$EVAL_SUBSET_SIZE" ]]; then
    SUBSET_ARGS+=(--eval_subset_size "$EVAL_SUBSET_SIZE")
fi

EARLY_STOPPING_ARGS=()
if [[ -n "$EARLY_STOPPING_PATIENCE" ]]; then
    EARLY_STOPPING_ARGS+=(
        --early_stopping_patience "$EARLY_STOPPING_PATIENCE"
        --early_stopping_threshold "$EARLY_STOPPING_THRESHOLD"
    )
fi

deepspeed src/train/train_distillation.py \
    --deepspeed scripts/deepspeed/zero2.json \
    --student_model_id "$STUDENT_MODEL" \
    --teacher_model_ids "$TEACHER_MODEL_IDS" \
    --data_path data/${DATASET_NAME}/train_llava.json \
    --eval_data_path data/${DATASET_NAME}/${EVAL_SPLIT}_llava.json \
    --image_folder data/${DATASET_NAME}/images \
    --distillation_loss "$DISTILLATION_LOSS" \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir "$OUTPUT_DIR" \
    --temperature "$TEMPERATURE" \
    --loss_weighting "$LOSS_WEIGHTING" \
    --gradnorm_alpha "$GRADNORM_ALPHA" \
    --gradnorm_lr "$GRADNORM_LR" \
    --layer_distill_source "$LAYER_DISTILL_SOURCE" \
    "${LAYER_MATCH_ARGS[@]}" \
    "${SUBSET_ARGS[@]}" \
    "${EARLY_STOPPING_ARGS[@]}" \
    --num_train_epochs 1.5 \
    --per_device_train_batch_size 18 \
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
    --eval_strategy steps \
    --eval_steps 100 \
    --per_device_eval_batch_size 20 \
    --load_best_model_at_end True \
    --metric_for_best_model eval_loss \
    --greater_is_better False \
    --dataloader_num_workers 4 \
    --remove_unused_columns False \
    --report_to wandb
