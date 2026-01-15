#  🧪 Formation Energy Prediction

## Overview
This repository contains code for predicting the formation energy of materials using multimodal  machine learning techniques. 
The primary focus is on implementing a Graph Neural Network (GNN) model to learn from material structures, text descriptions, and XRD structures and predict their formation energies.
This is part of a project for the [KRICT Hackathon 2025](https://gitlab.chemdx.org/global-network/2025-krict-chemdx-hackathon/-/wikis/home), an amazing event organised by the Korea Research Institute of Chemical Technology.

## Environment Setup
1. Install PyTorch with CUDA support:
`pip install torch==2.4.0+cu121 torchvision==0.19.0+cu121 torchaudio==2.4.0+cu121 --index-url https://download.pytorch.org/whl/cu121`
2. Install torchdata (required by matgl)
`pip install torchdata==0.8.0`
3. Install DGL with CUDA support:
`pip install dgl -f https://data.dgl.ai/wheels/cu121/repo.html`
4. Install remaining dependencies:
`pip install -r requirements.txt`

## Data
The dataset used in this project comprises a collection of materials, along with their corresponding formation energies, structures, space groups, and XRD patterns. The data is stored in a structured format, with each material represented by its features and labels. The KRICT Hackathon organiser provides the data.

## Usage
To run the model, you can run the following command:

```bash
python main.py
```
pip install torch==2.4.0+cu121 torchvision==0.19.0+cu121 torchaudio==2.4.0+cu121 --index-url https://download.pytorch.org/whl/cu121