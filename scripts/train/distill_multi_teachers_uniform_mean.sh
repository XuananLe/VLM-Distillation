#!/bin/bash

export PYTHONPATH=src:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TEACHER_MODEL_1="google/gemma-3-4b-it"
TEACHER_MODEL_2="OpenGVLab/InternVL2-1B"
TEACHER_MODEL_3="Qwen/Qwen2.5-VL-3B-Instruct"
TEACHER_MODEL_4="Qwen/Qwen2-VL-2B-Instruct"
TEACHER_MODEL_IDS=("${TEACHER_MODEL_1}" "${TEACHER_MODEL_2}" "${TEACHER_MODEL_3}" "${TEACHER_MODEL_4}")
STUDENT_MODEL="HuggingFaceTB/SmolVLM-256M-Instruct"

# uld_loss, trie_wasserstein_loss, cka_loss, forward_kl, reverse_kl, jensen_shannon_divergence
DISTILLATION_LOSS="${DISTILLATION_LOSS:-trie_wasserstein_loss}"
STUDENT_TEMPERATURE="${STUDENT_TEMPERATURE:-1.0}"
TEACHER_TEMPERATURE="${TEACHER_TEMPERATURE:-1.0}"

TRIE_WASSERSTEIN_RHO="${TRIE_WASSERSTEIN_RHO:-0.5}"
TRIE_WASSERSTEIN_TOPK="${TRIE_WASSERSTEIN_TOPK:-32}"

# Loss = ce_loss + alpha * mean(kd_loss over teachers)
ALPHA="${ALPHA:-0.5}"
TEACHER_WEIGHTING_STRATEGY="uniform_mean"

TEACHER_LOGITS_CACHE_DIR="${TEACHER_LOGITS_CACHE_DIR:-/workspace/cache}"

NUM_TEACHERS=4
DATASET_NAME="${DATASET_NAME:-textvqa}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1.0}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-40}"
LOGGING_STEPS="${LOGGING_STEPS:-5}"
AUTO_STOP_VAST_INSTANCE="${AUTO_STOP_VAST_INSTANCE:-0}"

STUDENT_NAME="${STUDENT_MODEL##*/}"
TEACHER_NAME_1="${TEACHER_MODEL_1##*/}"
TEACHER_NAME_2="${TEACHER_MODEL_2##*/}"
TEACHER_NAME_3="${TEACHER_MODEL_3##*/}"
TEACHER_NAME_4="${TEACHER_MODEL_4##*/}"
ALPHA_TAG="${ALPHA//./p}"
TRIE_WASSERSTEIN_RHO_TAG="${TRIE_WASSERSTEIN_RHO//./p}"
LOSS_TAG="${DISTILLATION_LOSS}_alpha${ALPHA_TAG}"
if [[ "${DISTILLATION_LOSS}" == "trie_wasserstein_loss" ]]; then
    LOSS_TAG="${DISTILLATION_LOSS}_rho${TRIE_WASSERSTEIN_RHO_TAG}_topk${TRIE_WASSERSTEIN_TOPK}_alpha${ALPHA_TAG}"
fi
RUN_TAG="${RUN_TAG:-uniform_mean_${DATASET_NAME}_${LOSS_TAG}_$(date +%Y%m%d_%H%M)}"
OUTPUT_DIR="output/${DISTILLATION_LOSS}_${TEACHER_WEIGHTING_STRATEGY}_${NUM_TEACHERS}_teachers_${TEACHER_NAME_1}_${TEACHER_NAME_2}_${TEACHER_NAME_3}_${TEACHER_NAME_4}_${STUDENT_NAME}_${DATASET_NAME}_${RUN_TAG}"

stop_vast_instance() {
    if [[ "${AUTO_STOP_VAST_INSTANCE}" == "0" ]]; then
        return 0
    fi
    local vast_instance_id="${VAST_INSTANCE_ID:-${CONTAINER_ID:-${VAST_CONTAINERLABEL:-}}}"
    vast_instance_id="${vast_instance_id#C.}"

    local stop_args=()
    if [[ -n "${CONTAINER_API_KEY:-}" ]]; then
        stop_args+=(--api-key "${CONTAINER_API_KEY}")
    fi

    vastai stop instance "${stop_args[@]}" "${vast_instance_id}"
}

python src/train/train_distillation.py \
    --student_model_id "$STUDENT_MODEL" \
    --teacher_model_ids "${TEACHER_MODEL_IDS[@]}" \
    --teacher_weighting_strategy "$TEACHER_WEIGHTING_STRATEGY" \
    --data_path data/${DATASET_NAME}/train_llava.json \
    --image_folder data/${DATASET_NAME}/images \
    --distillation_loss "$DISTILLATION_LOSS" \
    --trie_wasserstein_rho "$TRIE_WASSERSTEIN_RHO" \
    --trie_wasserstein_topk "$TRIE_WASSERSTEIN_TOPK" \
    --bf16 True \
    --output_dir "$OUTPUT_DIR" \
    --student_temperature "$STUDENT_TEMPERATURE" \
    --teacher_temperature "$TEACHER_TEMPERATURE" \
    --alpha "$ALPHA" \
    --num_train_epochs "$NUM_TRAIN_EPOCHS" \
    --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
    --learning_rate 1e-5 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --tf32 True \
    --gradient_checkpointing True \
    --logging_steps "$LOGGING_STEPS" \
    --save_strategy steps \
    --save_steps 150 \
    --save_total_limit 100 \
    --dataloader_num_workers 4 \
    --remove_unused_columns False \
    --report_to wandb \
    --teacher_logits_cache_dir "$TEACHER_LOGITS_CACHE_DIR"

TRAIN_EXIT_CODE=$?
if [[ "${TRAIN_EXIT_CODE}" -eq 1 ]]; then
    while true; do
        sleep 60000
    done
fi
stop_vast_instance
exit "${TRAIN_EXIT_CODE}"
