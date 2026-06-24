# 🧪 Formation Energy Prediction

## Overview
This repository predicts formation energy using a multimodal model built on a crystal graph (CIF structure) plus optional XRD features and space-group text embeddings. It was developed for the [KRICT Hackathon 2025](https://gitlab.chemdx.org/global-network/2025-krict-chemdx-hackathon/-/wikis/home).

## Environment Setup

Choose the setup path based on the GPU architecture.

### Pre-Blackwell GPUs: virtual environment setup

For GPUs with readily available PyTorch/DGL CUDA wheels, use the original
virtual-environment workflow. To ensure optimal configuration for each model
family, install dependencies in separate environments.

1. **ALIGNN Models**:
   ```bash
   pip install -r requirements_alignn.txt
   ```

2. **MatGL Models** (CHGNet, M3GNet, TensorNet, QET):
   ```bash
   pip install -r requirements_matgl.txt
   ```

3. **Others (CGCNN, MPNN)**:
   The base models `cgcnn` and `mpnn` are functional in both environments.

### Blackwell GPUs / CUDA 13.0 hosts: container setup

On Blackwell-generation GPUs such as RTX PRO 6000 Blackwell, the host driver may
report CUDA 13.0 while public DGL pip wheels lag behind the required CUDA/PyTorch
stack. The SJK branch therefore uses a container workflow based on NVIDIA's DGL
image so CUDA, PyTorch, and DGL remain matched.

Install Docker Engine and NVIDIA Container Toolkit on the host. On Ubuntu:

```bash
./scripts/setup_host_container_runtime_ubuntu.sh --yes
newgrp docker
```

Validate GPU access from Docker:

```bash
docker run --rm --gpus all nvcr.io/nvidia/cuda:13.0.0-base-ubuntu24.04 nvidia-smi
```

Build the project image:

```bash
./scripts/build_dgl_container.sh
```

Run smoke tests:

```bash
./scripts/run_dgl_container.sh python scripts/check_dgl_container_env.py
./scripts/run_dgl_container.sh python main.py --help
```

Run training through the container wrapper:

```bash
./scripts/run_dgl_container.sh python main.py \
  --config examples/configs/133/cgcnn_crystalsys.yaml
```

The wrapper mounts this repository into the container and runs as the host
UID/GID by default, so outputs written under the repository remain editable from
the host. See [`CONTAINER.md`](CONTAINER.md) for the full container manual.

## Repository Layout
- `main.py`: training + validation + test evaluation entrypoint.
- `data.py`: CIF dataset loader and batching utilities.
- `models/`: CGCNN backbone and XRD/text feature extractors.
- `scripts/train.sh`: example training command.
- `scripts/build_dgl_container.sh`, `scripts/run_dgl_container.sh`: Blackwell
  container build/run helpers.
- `CONTAINER.md`: detailed container setup notes for Blackwell/CUDA 13.0 hosts.
- `pretrained_models/`: pre-trained CGCNN weights.
- `data_preprocessing/`, `data_preprocessing_update/`: dataset prep scripts and artifacts.
- `data/`: expected dataset location.


## Data Format
By default `main.py` expects data under `data/` (configurable via `--data_path` and `--base_data_dir`). 

### CSV File Requirements
CSV files used for training and testing must contain a header row with the following **exact** column names:
- `file_name`: CIF ID matching the filename (e.g., if it's `123.cif`, use `123`)
- `value_per_atom`: Target formation energy value per atom.
- `space_group`: Space group identifier (used as a key for text embeddings).

### Common Assets
Common resource files are stored in a directory specified by `--base_data_dir` (default is `data/`):
- `data/atom_init.json`: (Required) JSON file for element features.
- `data/XRD_data.csv`: (Optional) XRD features. Required if `--xrd True`.
- `data/space_group_embeddings.csv`: (Optional) Text embeddings. Required if `--text True`.

---

## Usage Examples (Dataset Splitting)

The script `main.py` supports flexible dataset management.

### Case 1: Specifying a single file for both Training and Testing
Manually assign one CSV file for training and one for testing.
```bash
python main.py --data_path data/split_both_hhi --train_file train.csv --test_file test.csv
```

### Case 2: Specifying multiple files for Training
You can list multiple CSV files for the training set by separating them with spaces. The data will be merged automatically.
```bash
python main.py --data_path data/split_rand02 --train_file chunk1.csv chunk2.csv chunk3.csv --test_file chunk4.csv
```

### Case 3: Auto-Discovery Mode (Specifying ONLY the Test file)
If you specify only the `--test_file`, the script searches the `--data_path` and uses all other `.csv` files (excluding auxiliary files like `XRD_data.csv`) as the training set.
```bash
python main.py --data_path data/split_rand02 --test_file chunk1.csv
```

### Case 4: Specifying multiple files for both Training and Testing
Both arguments support multiple files. Useful for combining specific chunks for evaluation.
```bash
python main.py --data_path data/split_folder --train_file chunk1.csv chunk2.csv --test_file chunk4.csv chunk5.csv
```

### Case 5: Single File Mode with Ratio Split
If you have one large CSV file and want the script to handle the train/val/test split automatically by percentage.
```bash
python main.py --data_path data/folder --train_file my_data.csv --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1
```
*Note: If `--train_file` is omitted, it defaults to `id_prop.csv`.*

---

## Training Options

### Graph Architectures
You can choose from several Graph Neural Network architectures via the `--graph_type` argument:
- `cgcnn`: Crystal Graph Convolutional Neural Network (Default)
- `mpnn`: Message Passing Neural Network
- `chgnet`: Crystalline Hamiltonian Graph Network (via MatGL)
- `m3gnet`: Materials 3-body Graph Network (via MatGL)
- `tensornet`: TensorNet (via MatGL)
- `qet`: Quantum Electrostatic Transformer (via MatGL)
- `alignn`: Atomistic Line Graph Neural Network (via ALIGNN)

Example:
```bash
python main.py --data_path data/split_folder --graph_type chgnet --epochs 50
```

When using the Blackwell container workflow, prefix the same command with
`./scripts/run_dgl_container.sh`:

```bash
./scripts/run_dgl_container.sh python main.py \
  --data_path data/split_folder \
  --graph_type chgnet \
  --epochs 50
```

### Multimodal Features
Toggle XRD and Text features as needed:
```bash
# Full Multimodal
python main.py --xrd True --text True

# Structure-Only (CGCNN-like)
python main.py --xrd False --text False
```

### Optimization Parameters
```bash
python main.py --lr 0.001 --batch-size 128 --optim Adam --epochs 100
```

## Outputs and Checkpoints

The script provides flexible management for training results and checkpoints.

### Default Behavior
By default, the following files are generated in the current working directory:
- `checkpoint.pth.tar`: Latest training state.
- `model_best.pth.tar`: Best model weights based on validation MAE.
- `test_results.csv`: Final predictions on the test set.
- `train_emb.npy`, `test_emb.npy`: Latent embeddings for UMAP analysis.
- `ood_umap_kde.png`: Visualization of data distribution.

### Organizing Results into a Folder
You can automatically move all relevant outputs to a specific directory using `--result_dir`.

```bash
# Organize results into 'experiment_1' folder
python main.py --result_dir experiment_1
```

### Customizing Output Files (Wildcard Support)
You can specify exactly which files to move using `--result_files`. This supports **wildcards (`*`)**, making it easy to include environment-specific log files (e.g., Slurm logs).

```bash
# Move only the best model and all Slurm output files
python main.py \
  --result_dir experiment_results \
  --result_files model_best.pth.tar test_results.csv slurm-*.out
```

## Inference / Evaluation
`main.py` automatically evaluates the best checkpoint on the test split at the end of training. Results are stored in `test_results.csv`.

## License
See `LICENSE`.
