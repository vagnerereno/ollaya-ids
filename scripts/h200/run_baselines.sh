#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

nvidia-smi
ollaya-ids --config "${CONFIG_FILE}" prepare
ollaya-ids --config "${CONFIG_FILE}" baseline --model random-forest
ollaya-ids --config "${CONFIG_FILE}" baseline --model xgboost --device cuda
