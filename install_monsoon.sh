#!/bin/bash
# =============================================================================
# Monsoon HPC First-Time Installation Script for BioBERT Pipeline
# =============================================================================
# Run this ONCE to set up the environment from scratch.
# Usage: bash install_monsoon.sh
# =============================================================================

set -e  # Exit on any error

echo "=============================================="
echo "BioBERT Pipeline - Monsoon Installation"
echo "=============================================="

# --- Redirect all caches to scratch ---
export PIP_CACHE_DIR=/scratch/bmb646/.pip_cache
export MPLCONFIGDIR=/scratch/bmb646/.matplotlib
export HF_HOME=/scratch/bmb646/.huggingface
export TRANSFORMERS_CACHE=/scratch/bmb646/.huggingface
export XDG_CACHE_HOME=/scratch/bmb646/.cache

# Create directories
mkdir -p $PIP_CACHE_DIR
mkdir -p $MPLCONFIGDIR
mkdir -p $HF_HOME
mkdir -p $XDG_CACHE_HOME
mkdir -p /scratch/bmb646/envs
mkdir -p /scratch/bmb646/biobert_dualsolo

# --- Load anaconda ---
echo "Loading anaconda module..."
module load anaconda3

# --- Configure conda to use scratch ---
echo "Configuring conda to use scratch directories..."
conda config --add pkgs_dirs /scratch/bmb646/.conda/pkgs
conda config --add envs_dirs /scratch/bmb646/envs

# --- Create environment in scratch ---
echo "Creating conda environment..."
conda create --prefix /scratch/bmb646/envs/nlp_bio python=3.9 -y

# --- Activate environment ---
echo "Activating environment..."
conda activate /scratch/bmb646/envs/nlp_bio

# --- Install packages via conda (more compatible with HPC) ---
echo "Installing base packages via conda..."
conda install -y \
    numpy=1.24 \
    pandas \
    matplotlib \
    seaborn \
    scikit-learn \
    scipy \
    pip

# --- Set library path for C++ compatibility ---
export LD_LIBRARY_PATH=/scratch/bmb646/envs/nlp_bio/lib:$LD_LIBRARY_PATH

# --- Install PyTorch with CUDA 11.7 support ---
echo "Installing PyTorch with CUDA support..."
pip install torch --index-url https://download.pytorch.org/whl/cu117

# --- Install transformers and related packages ---
echo "Installing transformers..."
pip install "transformers==4.40.0" accelerate datasets

# --- Verify installation ---
echo ""
echo "=============================================="
echo "Verifying installation..."
echo "=============================================="

python -c "
import torch
import transformers
import sklearn
import matplotlib
import pandas
import numpy

print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'Transformers: {transformers.__version__}')
print(f'NumPy: {numpy.__version__}')
print(f'Scikit-learn: {sklearn.__version__}')
print('All packages installed successfully!')
"

echo ""
echo "=============================================="
echo "Installation Complete!"
echo "=============================================="
echo ""
echo "Next steps:"
echo "1. Upload your data files to /scratch/bmb646/biobert_dualsolo/"
echo "   - generate_report_v2.py"
echo "   - train.csv"
echo "   - test.csv"
echo ""
echo "2. For future sessions, run:"
echo "   source /scratch/bmb646/biobert_dualsolo/setup_monsoon.sh"
echo ""
echo "3. Then run the pipeline:"
echo "   python generate_report_v2.py"
echo "=============================================="
