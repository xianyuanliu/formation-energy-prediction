# 🧪 Formation Energy Prediction

## Overview
This repository predicts formation energy using a multimodal model built on a crystal graph (CIF structure) plus optional XRD features and space-group text embeddings. It was developed for the [KRICT Hackathon 2025](https://gitlab.chemdx.org/global-network/2025-krict-chemdx-hackathon/-/wikis/home).

## Environment Setup
1. Install PyTorch with CUDA support:

```bash
pip install torch==2.4.0+cu121 torchvision==0.19.0+cu121 torchaudio==2.4.0+cu121 --index-url https://download.pytorch.org/whl/cu121
```

2. Install torchdata (required by matgl):

```bash
pip install torchdata==0.8.0
```

3. Install DGL with CUDA support:

```bash
pip install dgl -f https://data.dgl.ai/wheels/cu121/repo.html
```

4. Install remaining dependencies:

```bash
pip install -r requirements.txt
```

## Repository Layout
- `main.py`: training + validation + test evaluation entrypoint.
- `data.py`: CIF dataset loader and batching utilities.
- `models/`: CGCNN backbone and XRD/text feature extractors.
- `scripts/train.sh`: example training command.
- `pretrained_models/`: pre-trained CGCNN weights.
- `data_preprocessing/`, `data_preprocessing_update/`: dataset prep scripts and artifacts.
- `data/`: expected dataset location.


## Data Format
By default `main.py` expects data under `data/cifs` (configurable via `--data_path`). The current loader expects this structure:

```
data/
  cifs/
    1_MatDX_EF_modified.csv
    atom_init.json
    <cif_id>.cif
  XRD_data.csv
  SG_text_data.csv
```

Expected columns:
- `data/cifs/1_MatDX_EF_modified.csv` (header row is skipped)
  - column 0: `cif_id` (must match CIF filenames and XRD `Composition` keys)
  - column 2: `space_group`
  - column 3: target formation energy per atom
- `data/XRD_data.csv`
  - `Composition` column
  - XRD feature columns (`xrd_0`, `xrd_1`, ... )
- `data/SG_text_data.csv`
  - `space_group` column
  - Pre-computed text embeddings (`emb_000`, `emb_001`, ... )

If you use a different naming scheme, adjust paths in `data.py`.

## Training
Run training with defaults:

```bash
python main.py --data_path data/cifs
```

Example with explicit split ratios:

```bash
python main.py \
  --data_path data/cifs \
  --train-ratio 0.6 \
  --val-ratio 0.2 \
  --test-ratio 0.2 \
  --graph_type mpnn
```

You can also use the convenience script:

```bash
bash scripts/train.sh
```

## Outputs and Checkpoints
Training saves `checkpoint.pth.tar` in the current working directory. The "best" checkpoint is copied to `../model_best.pth.tar` (relative to where you run the command). `main.py` expects `model_best.pth.tar` in the working directory for its final test evaluation, so copy it back or adjust the path if needed.

## Inference / Evaluation
`main.py` automatically evaluates the best checkpoint on the test split at the end of training.

There is also a legacy `test.py` script for the CGCNN baseline and the pre-trained weights in `pretrained_models/`. It may require updates to match the current multimodal dataset loader.

## License
See `LICENSE`.
