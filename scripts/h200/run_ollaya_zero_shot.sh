#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

MODEL="${OLLAYA_MODEL:-kev:4b}"
MAX_TEST_SAMPLES="${MAX_TEST_SAMPLES:-2000}"
SERVER_LOG="${LOG_DIR}/ollaya-server.log"

ollaya serve >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!
trap 'kill "${SERVER_PID}" 2>/dev/null || true' EXIT

wait_for_http http://127.0.0.1:11435/
ollaya pull "${MODEL}"

ollaya-ids --config "${CONFIG_FILE}" evaluate-api \
    --base-url http://127.0.0.1:11435 \
    --endpoint /v1/systemone \
    --model "${MODEL}" \
    --max-samples "${MAX_TEST_SAMPLES}" \
    --run-name kev-zero-shot
