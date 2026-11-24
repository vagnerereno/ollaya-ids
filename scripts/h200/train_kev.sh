#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3.5-4B-Base}"
INIT_FROM="${INIT_FROM:-jaredpalmer/kev-4b@v1.0}"
EPOCHS="${EPOCHS:-2}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
BATCH_SIZE="${BATCH_SIZE:-4}"
ACCUMULATION_STEPS="${ACCUMULATION_STEPS:-2}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${RUN_DIR}/ereno-kev-4b}"
DATA_DIR="${RESULTS_ROOT}/artifacts/kev"

nvidia-smi
ollaya-ids --config "${CONFIG_FILE}" export-kev

"${KEV_PYTHON}" -m kev.train \
    --data "${DATA_DIR}/train.jsonl" \
    --base "${BASE_MODEL}" \
    --init_from "${INIT_FROM}" \
    --epochs "${EPOCHS}" \
    --lr "${LEARNING_RATE}" \
    --batch "${BATCH_SIZE}" \
    --accum "${ACCUMULATION_STEPS}" \
    --dtype bf16 \
    --weights_dtype bf16 \
    --checkpointing 1 \
    --device cuda \
    --out "${CHECKPOINT_DIR}" \
    2>&1 | tee "${LOG_DIR}/kev-training.log"

"${KEV_PYTHON}" -m kev.benchmark \
    --run "${CHECKPOINT_DIR}" \
    --data "${DATA_DIR}/validation.jsonl" \
    --out "${CHECKPOINT_DIR}-validation" \
    2>&1 | tee "${LOG_DIR}/kev-validation.log"
