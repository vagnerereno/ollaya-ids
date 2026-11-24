#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Uso: $0 GPU_INDEX [RESULTS_DIR]"
    echo "Consulte uso_das_gpus antes de escolher a GPU."
    exit 2
fi

GPU_INDEX="$1"
RESULTS_DIR="${2:-${HOME}/resultados}"
IMAGE_NAME="${IMAGE_NAME:-ollaya-ids:h200}"
CONTAINER_NAME="${CONTAINER_NAME:-ollaya-ids-h200-${GPU_INDEX}}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CACHE_DIR="${HOME}/.cache/ollaya-ids"
OLLAYA_DIR="${HOME}/.ollaya"
GPU_REQUEST="\"device=${GPU_INDEX}\""

mkdir -p "${RESULTS_DIR}" "${CACHE_DIR}" "${OLLAYA_DIR}"

if docker ps -a --format '{{.Names}}' | grep -Fxq "${CONTAINER_NAME}"; then
    echo "Já existe um container chamado ${CONTAINER_NAME}."
    exit 1
fi

docker run -itd \
    --name "${CONTAINER_NAME}" \
    --gpus "${GPU_REQUEST}" \
    --rm \
    --shm-size=16g \
    --volume "${PROJECT_DIR}:/workspace" \
    --volume "${RESULTS_DIR}:/resultados" \
    --volume "${CACHE_DIR}:/root/.cache" \
    --volume "${OLLAYA_DIR}:/root/.ollaya" \
    --workdir /workspace \
    "${IMAGE_NAME}" bash

echo "Container: ${CONTAINER_NAME}"
echo "Entrar: docker attach ${CONTAINER_NAME}"
echo "Validar ambiente: docker exec -it ${CONTAINER_NAME} bash scripts/h200/verify_environment.sh"
echo "Executar tudo: docker exec -it ${CONTAINER_NAME} bash scripts/h200/run_all.sh"
