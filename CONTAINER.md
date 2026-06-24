# Container Setup

This project needs DGL for the MatGL and ALIGNN paths. On Blackwell GPUs, a
plain virtual environment is awkward because public DGL pip wheels currently
stop at older CUDA builds. The container setup uses NVIDIA's DGL image as the
base so CUDA, PyTorch, and DGL stay matched.

## System Prerequisites

Install Docker Engine and NVIDIA Container Toolkit on the host. After that,
this should work:

```bash
docker run --rm --gpus all nvcr.io/nvidia/cuda:13.0.0-base-ubuntu24.04 nvidia-smi
```

On Ubuntu, a helper script is included. It uses `sudo`, follows the official
Docker and NVIDIA Container Toolkit apt repository flow, and adds the current
user to the `docker` group:

```bash
./scripts/setup_host_container_runtime_ubuntu.sh --yes
```

After the install, log out and back in, or run `newgrp docker`.

## Build

```bash
cd ~/formation-energy-prediction
./scripts/build_dgl_container.sh
```

The default base image is `nvcr.io/nvidia/dgl:25.01-py3`, the first NVIDIA
DGL container line documented as Blackwell-optimized. Override it if needed:

```bash
BASE_IMAGE=nvcr.io/nvidia/dgl:25.08-py3 ./scripts/build_dgl_container.sh
```

## Smoke Tests

```bash
./scripts/run_dgl_container.sh python scripts/check_dgl_container_env.py
./scripts/run_dgl_container.sh python main.py --help
```

## Run Training

Run `main.py` through the wrapper script. Repository YAML configs are the safest
entrypoint because the training code merges config defaults during startup:

```bash
./scripts/run_dgl_container.sh python main.py \
  --config examples/configs/133/cgcnn_crystalsys.yaml
```

For an interactive shell inside the container:

```bash
./scripts/run_dgl_container.sh bash
```

The wrapper runs as the host UID/GID by default, so outputs written under the
repository stay owned by the host user. Override with `CONTAINER_USER=0:0` only
for maintenance tasks that really need root inside the container.
