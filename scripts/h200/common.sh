#!/usr/bin/env bash

PROJECT_DIR="${PROJECT_DIR:-/workspace}"
RESULTS_ROOT="${RESULTS_ROOT:-/resultados/ollaya-ids}"
CONFIG_FILE="${CONFIG_FILE:-${PROJECT_DIR}/configs/h200.yaml}"
LOG_DIR="${RESULTS_ROOT}/logs"
RUN_DIR="${RESULTS_ROOT}/runs"
KEV_PYTHON="${KEV_PYTHON:-/opt/kev/.venv/bin/python}"

mkdir -p "${LOG_DIR}" "${RUN_DIR}"

wait_for_http() {
    local url="$1"
    local attempts="${2:-120}"
    local count=0
    until curl --silent --fail "${url}" >/dev/null; do
        count=$((count + 1))
        if (( count >= attempts )); then
            echo "O serviço não respondeu em ${url}."
            return 1
        fi
        sleep 2
    done
}
