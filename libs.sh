#!/bin/bash

# Initialize Conda for shell activation
conda init
source ~/.zshrc # Or ~/.bash_profile

# Create a new Conda environment or activate an existing one
# conda create --prefix /Users/abraar/Downloads/Thesis/.conda python=3.12 -y
conda activate /Users/abraar/Downloads/Thesis/.conda

# Set the solver to classic to avoid libmamba issues
conda config --set solver classic

# Install essential libraries
conda install numpy pandas tqdm matplotlib seaborn psutil -y

# Install PyTorch with MPS support (optimized for Apple Silicon)
conda install pytorch torchvision torchaudio -c pytorch -y

# Install NLP-related tools for tokenization and preprocessing
conda install -c conda-forge sentencepiece sacremoses nltk -y

# Additional tools for optimization and experiment tracking
conda install -c conda-forge optuna wandb tensorboard -y

# Use pip to install additional packages
pip install torch numpy transformers datasets tiktoken wandb tqdm torchmetrics micrograd




# to export : conda env export --prefix /Users/abraar/Downloads/Thesis/.conda > environment.yml