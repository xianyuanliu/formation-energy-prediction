#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_NAME="${IMAGE_NAME:-formation-energy-prediction:dgl}"

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is not installed. Install Docker Engine and NVIDIA Container Toolkit first." >&2
  exit 127
fi

TTY_ARGS=()
if [[ -t 0 && -t 1 ]]; then
  TTY_ARGS=(-it)
fi

CONTAINER_USER="${CONTAINER_USER:-$(id -u):$(id -g)}"

docker run --rm \
  "${TTY_ARGS[@]}" \
  --user "${CONTAINER_USER}" \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e HOME=/tmp \
  -e MATGL_BACKEND=dgl \
  -e PYTHONDONTWRITEBYTECODE=1 \
  -v "${REPO_DIR}:/workspace/formation-energy-prediction" \
  -w /workspace/formation-energy-prediction \
  "${IMAGE_NAME}" \
  "$@"
