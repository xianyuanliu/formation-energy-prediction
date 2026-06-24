#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_NAME="${IMAGE_NAME:-formation-energy-prediction:dgl}"
BASE_IMAGE="${BASE_IMAGE:-nvcr.io/nvidia/dgl:25.01-py3}"

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is not installed. Install Docker Engine and NVIDIA Container Toolkit first." >&2
  exit 127
fi

docker build \
  --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
  -f "${REPO_DIR}/docker/Dockerfile.dgl" \
  -t "${IMAGE_NAME}" \
  "${REPO_DIR}"

echo "Built ${IMAGE_NAME} from ${BASE_IMAGE}"
