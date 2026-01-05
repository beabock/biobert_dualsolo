#!/bin/bash
# =============================================================================
# Monsoon HPC Setup Script for BioBERT Pipeline
# =============================================================================
# Usage: source /scratch/bmb646/setup_monsoon.sh
# =============================================================================

echo "Setting up BioBERT environment on Monsoon..."

# --- Redirect all caches to scratch (avoid home quota issues) ---
export PIP_CACHE_DIR=/scratch/bmb646/.pip_cache
export MPLCONFIGDIR=/scratch/bmb646/.matplotlib
export HF_HOME=/scratch/bmb646/.huggingface
export TRANSFORMERS_CACHE=/scratch/bmb646/.huggingface
export XDG_CACHE_HOME=/scratch/bmb646/.cache

# --- Suppress tokenizer parallelism warning ---
export TOKENIZERS_PARALLELISM=false

# Create cache directories
mkdir -p $PIP_CACHE_DIR
mkdir -p $MPLCONFIGDIR
mkdir -p $HF_HOME
mkdir -p $XDG_CACHE_HOME

# --- Use conda's C++ libraries instead of system (fixes GLIBCXX errors) ---
export LD_LIBRARY_PATH=/scratch/bmb646/envs/nlp_bio/lib:$LD_LIBRARY_PATH

# --- Load anaconda module ---
module load anaconda3

# --- Activate the environment ---
conda activate /scratch/bmb646/envs/nlp_bio

# --- Navigate to project directory ---
cd /scratch/bmb646/biobert_dualsolo

# --- Verify setup ---
echo ""
echo "=============================================="
echo "Environment Setup Complete"
echo "=============================================="
echo "Python: $(which python)"
echo "Working dir: $(pwd)"
echo ""

# Check GPU
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')" 2>/dev/null || echo "PyTorch not yet installed or CUDA check failed"

echo ""
echo "To run the pipeline: python generate_report_v2.py"
echo "=============================================="
