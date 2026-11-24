#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-ollaya-ids:h200}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

docker build --file "${PROJECT_DIR}/Dockerfile.h200" --tag "${IMAGE_NAME}" "${PROJECT_DIR}"
