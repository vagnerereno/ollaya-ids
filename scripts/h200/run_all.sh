#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "${SCRIPT_DIR}/verify_environment.sh"
bash "${SCRIPT_DIR}/run_baselines.sh"
bash "${SCRIPT_DIR}/run_ollaya_zero_shot.sh"
bash "${SCRIPT_DIR}/train_kev.sh"
bash "${SCRIPT_DIR}/evaluate_finetuned_kev.sh"
ollaya-ids --config /workspace/configs/h200.yaml compare
