#!/usr/bin/env bash

set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    cat <<'EOF'
Usage: monitor_tmux_training.sh <tmux-pane> <output-dir> [duration-seconds] [interval-seconds] [checkpoint-dir]

Example:
  monitor_tmux_training.sh ssh_tmux:0.0 monitoring/run1 7200 60 output/my_run
EOF
    exit 0
fi

PANE_TARGET="${1:?tmux pane target is required, e.g. ssh_tmux:0.0}"
OUTPUT_DIR="${2:?output directory is required}"
DURATION_SECONDS="${3:-7200}"
INTERVAL_SECONDS="${4:-60}"
CHECKPOINT_DIR="${5:-}"

mkdir -p "${OUTPUT_DIR}"

PANE_LOG="${OUTPUT_DIR}/pane.log"
SNAPSHOT_LOG="${OUTPUT_DIR}/snapshots.log"
SUMMARY_TSV="${OUTPUT_DIR}/summary.tsv"
STATUS_FILE="${OUTPUT_DIR}/status.env"

START_EPOCH="$(date +%s)"
END_EPOCH="$((START_EPOCH + DURATION_SECONDS))"
PANE_LOG_QUOTED="$(printf '%q' "${PANE_LOG}")"

tmux_has_active_pipe() {
    tmux display-message -p -t "${PANE_TARGET}" '#{?pane_pipe,1,0}'
}

disable_pipe() {
    if [[ "$(tmux_has_active_pipe)" == "1" ]]; then
        tmux pipe-pane -t "${PANE_TARGET}"
    fi
}

cleanup() {
    disable_pipe
    {
        echo "finished_at=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
        echo "finished_epoch=$(date +%s)"
        echo "state=finished"
    } >> "${STATUS_FILE}"
}

trap cleanup EXIT INT TERM

if [[ "$(tmux_has_active_pipe)" == "1" ]]; then
    echo "tmux pane ${PANE_TARGET} already has an active pipe-pane" >&2
    exit 1
fi

cat > "${STATUS_FILE}" <<EOF
pane_target=${PANE_TARGET}
output_dir=${OUTPUT_DIR}
started_at=$(date -u '+%Y-%m-%dT%H:%M:%SZ')
started_epoch=${START_EPOCH}
duration_seconds=${DURATION_SECONDS}
interval_seconds=${INTERVAL_SECONDS}
checkpoint_dir=${CHECKPOINT_DIR}
state=running
EOF

printf '%s\n' \
    $'timestamp\telapsed_sec\tpane_pid\tpane_command\tgpu_util_pct\tgpu_mem_pct\tgpu_mem_used_mb\tgpu_mem_total_mb\tgpu_power_w\tsm_clock_mhz\tmem_clock_mhz\tpcie_gen\tpcie_width\tlatest_checkpoint\tlatest_checkpoint_mtime\tlast_pane_line' \
    > "${SUMMARY_TSV}"

tmux pipe-pane -t "${PANE_TARGET}" "stdbuf -oL -eL cat >> ${PANE_LOG_QUOTED}"

while (( "$(date +%s)" < END_EPOCH )); do
    TIMESTAMP="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    NOW_EPOCH="$(date +%s)"
    ELAPSED_SECONDS="$((NOW_EPOCH - START_EPOCH))"

    PANE_PID="$(tmux display-message -p -t "${PANE_TARGET}" '#{pane_pid}')"
    PANE_COMMAND="$(tmux display-message -p -t "${PANE_TARGET}" '#{pane_current_command}')"
    LAST_PANE_LINE="$(
        tmux capture-pane -pt "${PANE_TARGET}" -S -20 \
        | tail -n 1 \
        | tr '\r' ' ' \
        | tr '\t' ' ' \
        | sed 's/[[:space:]]\+/ /g; s/^ //; s/ $//'
    )"

    GPU_LINE="$(
        nvidia-smi \
            --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,clocks.sm,clocks.mem,pcie.link.gen.current,pcie.link.width.current \
            --format=csv,noheader,nounits \
        | head -n 1
    )"
    IFS=',' read -r GPU_UTIL GPU_MEM_UTIL GPU_MEM_USED GPU_MEM_TOTAL GPU_POWER SM_CLOCK MEM_CLOCK PCIE_GEN PCIE_WIDTH <<< "${GPU_LINE}"
    GPU_UTIL="$(echo "${GPU_UTIL}" | xargs)"
    GPU_MEM_UTIL="$(echo "${GPU_MEM_UTIL}" | xargs)"
    GPU_MEM_USED="$(echo "${GPU_MEM_USED}" | xargs)"
    GPU_MEM_TOTAL="$(echo "${GPU_MEM_TOTAL}" | xargs)"
    GPU_POWER="$(echo "${GPU_POWER}" | xargs)"
    SM_CLOCK="$(echo "${SM_CLOCK}" | xargs)"
    MEM_CLOCK="$(echo "${MEM_CLOCK}" | xargs)"
    PCIE_GEN="$(echo "${PCIE_GEN}" | xargs)"
    PCIE_WIDTH="$(echo "${PCIE_WIDTH}" | xargs)"

    TRAIN_PROCESSES="$(ps -eo pid,ppid,%cpu,%mem,cmd --sort=-%cpu | rg 'deepspeed|train_distillation.py' || true)"
    GPU_PROCESSES="$(nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader,nounits || true)"

    LATEST_CHECKPOINT=""
    LATEST_CHECKPOINT_MTIME=""
    if [[ -n "${CHECKPOINT_DIR}" && -d "${CHECKPOINT_DIR}" ]]; then
        LATEST_CHECKPOINT="$(
            find "${CHECKPOINT_DIR}" -maxdepth 1 -mindepth 1 -type d -name 'checkpoint-*' \
            | sort -V \
            | tail -n 1
        )"
        if [[ -n "${LATEST_CHECKPOINT}" ]]; then
            LATEST_CHECKPOINT="$(basename "${LATEST_CHECKPOINT}")"
            LATEST_CHECKPOINT_MTIME="$(date -u -r "${CHECKPOINT_DIR}/${LATEST_CHECKPOINT}" '+%Y-%m-%dT%H:%M:%SZ')"
        fi
    fi

    {
        echo "=== ${TIMESTAMP} ==="
        echo "pane_pid=${PANE_PID}"
        echo "pane_command=${PANE_COMMAND}"
        echo "gpu_util_pct=${GPU_UTIL}"
        echo "gpu_mem_pct=${GPU_MEM_UTIL}"
        echo "gpu_mem_used_mb=${GPU_MEM_USED}"
        echo "gpu_mem_total_mb=${GPU_MEM_TOTAL}"
        echo "gpu_power_w=${GPU_POWER}"
        echo "sm_clock_mhz=${SM_CLOCK}"
        echo "mem_clock_mhz=${MEM_CLOCK}"
        echo "pcie_gen=${PCIE_GEN}"
        echo "pcie_width=${PCIE_WIDTH}"
        echo "latest_checkpoint=${LATEST_CHECKPOINT}"
        echo "latest_checkpoint_mtime=${LATEST_CHECKPOINT_MTIME}"
        echo "last_pane_line=${LAST_PANE_LINE}"
        echo "-- train_processes --"
        echo "${TRAIN_PROCESSES}"
        echo "-- gpu_processes --"
        echo "${GPU_PROCESSES}"
        echo
    } >> "${SNAPSHOT_LOG}"

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "${TIMESTAMP}" \
        "${ELAPSED_SECONDS}" \
        "${PANE_PID}" \
        "${PANE_COMMAND}" \
        "${GPU_UTIL}" \
        "${GPU_MEM_UTIL}" \
        "${GPU_MEM_USED}" \
        "${GPU_MEM_TOTAL}" \
        "${GPU_POWER}" \
        "${SM_CLOCK}" \
        "${MEM_CLOCK}" \
        "${PCIE_GEN}" \
        "${PCIE_WIDTH}" \
        "${LATEST_CHECKPOINT}" \
        "${LATEST_CHECKPOINT_MTIME}" \
        "${LAST_PANE_LINE}" \
        >> "${SUMMARY_TSV}"

    sleep "${INTERVAL_SECONDS}"
done
