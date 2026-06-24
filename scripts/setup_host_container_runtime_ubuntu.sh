#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" != "--yes" ]]; then
  cat >&2 <<'MSG'
This installs Docker Engine and NVIDIA Container Toolkit on Ubuntu using sudo.
It modifies apt repositories, /etc/docker/daemon.json, and restarts Docker.

Re-run with:
  ./scripts/setup_host_container_runtime_ubuntu.sh --yes
MSG
  exit 2
fi

. /etc/os-release
CODENAME="${UBUNTU_CODENAME:-${VERSION_CODENAME:-}}"
if [[ -z "${CODENAME}" ]]; then
  echo "Could not determine Ubuntu codename from /etc/os-release." >&2
  exit 1
fi

TARGET_USER="${TARGET_USER:-${SUDO_USER:-${USER}}}"
if ! id "${TARGET_USER}" >/dev/null 2>&1; then
  echo "Target user does not exist: ${TARGET_USER}" >&2
  exit 1
fi

sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg

sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu ${CODENAME} stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list >/dev/null

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null

sudo apt-get update
sudo apt-get install -y \
  docker-ce \
  docker-ce-cli \
  containerd.io \
  docker-buildx-plugin \
  docker-compose-plugin \
  nvidia-container-toolkit

sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

sudo usermod -aG docker "${TARGET_USER}"

cat <<MSG
Docker and NVIDIA Container Toolkit are installed.

Added ${TARGET_USER} to the docker group.

Log out and back in as ${TARGET_USER}, or run \`newgrp docker\`, so the shell can
use Docker without sudo. Then validate GPU access:

  docker run --rm --gpus all nvcr.io/nvidia/cuda:13.0.0-base-ubuntu24.04 nvidia-smi
MSG
