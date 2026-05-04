#!/bin/bash

export PYTHONPATH=src:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

TEACHER_MODEL_1="google/gemma-3-4b-it"
TEACHER_MODEL_2="Qwen/Qwen2.5-VL-3B-Instruct"
TEACHER_MODEL_3="Qwen/Qwen2-VL-2B-Instruct"
TEACHER_MODEL_IDS=("${TEACHER_MODEL_1}" "${TEACHER_MODEL_2}" "${TEACHER_MODEL_3}")
STUDENT_MODEL="HuggingFaceTB/SmolVLM-256M-Instruct"

# Logits KD is disabled by ALPHA=0. This placeholder is kept for parser compatibility.
DISTILLATION_LOSS="${DISTILLATION_LOSS:-uld_loss}"
STUDENT_TEMPERATURE="${STUDENT_TEMPERATURE:-1.0}"
TEACHER_TEMPERATURE="${TEACHER_TEMPERATURE:-1.0}"

# Loss = ce_loss + layer_weight * CKA layer loss. Alpha stays zero so logits KD is skipped.
ALPHA="${ALPHA:-0.0}"
TEACHER_WEIGHTING_STRATEGY="uniform_mean"
LAYER_DISTILL_SOURCE="vision"
LAYER_DISTILL_WEIGHT="${LAYER_DISTILL_WEIGHT:-0.1}"
LAYER_MATCH_TOPK="${LAYER_MATCH_TOPK:-1}"
CKA_LAYER_MATCH_ROOT="${CKA_LAYER_MATCH_ROOT:-${REPO_ROOT}/artifacts/cka_plots_vison_layer}"

# Resolved from artifacts/cka_plots_vison_layer/textvqa with LAYER_MATCH_TOPK=1.
# These are documented here so the startup config is easy to read. The trainer
# still consumes LAYER_MATCH_JSON_PATH because each teacher has different matched
# teacher layers, while --teacher_layer_indices only supports one shared list.
STUDENT_LAYER_INDICES=(0 1 2 3 4 5 6 7 8 9 10 11)
TEACHER_1_LAYER_INDICES=(24 25 25 25 25 25 25 25 25 25 25 25)
TEACHER_2_LAYER_INDICES=(1 3 5 9 7 10 10 12 12 12 12 16)
TEACHER_3_LAYER_INDICES=(0 1 1 6 6 9 9 10 10 10 10 30)

NUM_TEACHERS=3
DATASET_NAME="${DATASET_NAME:-textvqa}"
LAYER_MATCH_JSON_PATH="${LAYER_MATCH_JSON_PATH:-${CKA_LAYER_MATCH_ROOT}/${DATASET_NAME}}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-0.5}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-40}"
LOGGING_STEPS="${LOGGING_STEPS:-5}"
AUTO_STOP_VAST_INSTANCE="${AUTO_STOP_VAST_INSTANCE:-0}"

STUDENT_NAME="${STUDENT_MODEL##*/}"
TEACHER_NAME_1="${TEACHER_MODEL_1##*/}"
TEACHER_NAME_2="${TEACHER_MODEL_2##*/}"
TEACHER_NAME_3="${TEACHER_MODEL_3##*/}"
ALPHA_TAG="${ALPHA//./p}"
LAYER_WEIGHT_TAG="${LAYER_DISTILL_WEIGHT//./p}"
LOSS_TAG="layer_only_alpha${ALPHA_TAG}"
RUN_TAG="${RUN_TAG:-uniform_mean_${DATASET_NAME}_${LOSS_TAG}_vision_layer_cka_topk${LAYER_MATCH_TOPK}_w${LAYER_WEIGHT_TAG}_$(date +%Y%m%d_%H%M)}"
OUTPUT_DIR="output/${DISTILLATION_LOSS}_${TEACHER_WEIGHTING_STRATEGY}_${NUM_TEACHERS}_teachers_${TEACHER_NAME_1}_${TEACHER_NAME_2}_${TEACHER_NAME_3}_${STUDENT_NAME}_${DATASET_NAME}_${RUN_TAG}"

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

echo "Resolved CKA student layer indices: ${STUDENT_LAYER_INDICES[*]}"
echo "Resolved CKA teacher 1 (${TEACHER_MODEL_1}) layer indices: ${TEACHER_1_LAYER_INDICES[*]}"
echo "Resolved CKA teacher 2 (${TEACHER_MODEL_2}) layer indices: ${TEACHER_2_LAYER_INDICES[*]}"
echo "Resolved CKA teacher 3 (${TEACHER_MODEL_3}) layer indices: ${TEACHER_3_LAYER_INDICES[*]}"

python src/train/train_distillation.py \
    --student_model_id "$STUDENT_MODEL" \
    --teacher_model_ids "${TEACHER_MODEL_IDS[@]}" \
    --teacher_weighting_strategy "$TEACHER_WEIGHTING_STRATEGY" \
    --data_path data/${DATASET_NAME}/train_llava.json \
    --image_folder data/${DATASET_NAME}/images \
    --distillation_loss "$DISTILLATION_LOSS" \
    --layer_distill_source "$LAYER_DISTILL_SOURCE" \
    --layer_distill_weight "$LAYER_DISTILL_WEIGHT" \
    --layer_match_json_path "$LAYER_MATCH_JSON_PATH" \
    --layer_match_topk "$LAYER_MATCH_TOPK" \
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
    --report_to wandb

TRAIN_EXIT_CODE=$?
if [[ "${TRAIN_EXIT_CODE}" -eq 1 ]]; then
    while true; do
        sleep 60000
    done
fi
stop_vast_instance
exit "${TRAIN_EXIT_CODE}"
