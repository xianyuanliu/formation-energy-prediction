Formation Energy Prediction

## Overview
This repository contains code for predicting the formation energy of materials using multimodal  machine learning techniques. 
The primary focus is on implementing a Graph Neural Network (GNN) model to learn from material structures, text description, and XRD structure and predict their formation energies.
This is part of a project for the KRICT Hackathon, an amazing event organized by the Korea Research Institute of Chemical Technology.

## Installation
To set up the environment, you can run `pip install -requirements.txt` to install the necessary dependencies.

## Data
The dataset used in this project is a collection of materials with their corresponding formation energies, structures, space groups, and XRD patterns. The data is stored in a structured format, with each material represented by its features and labels. The data is provided by KRICT Hackathon organisor.

## Usage
To run the model, you can run the following command:

```bash
python main.py
```