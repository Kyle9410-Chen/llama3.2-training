#!/bin/bash
#SBATCH --job-name=hf-upload
#SBATCH --partition=dev
#SBATCH --account=ACD115198
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# usage: sbatch src/llama3_2_training/sbatch/upload.sh [model_dir] [repo_id] [--message MSG]
#   model_dir / repo_id default to UPLOAD_MODEL_PATH / HF_REPO_ID in .env
#   e.g. sbatch src/llama3_2_training/sbatch/upload.sh out/muon-1h-535585/final your-name/llama1b-muon

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

module purge

export HF_HOME=/work/${USER}/.cache/huggingface

echo "host: $(hostname) | start: $(date)"
echo "args: $*"

uv run --frozen --env-file .env python -m llama3_2_training.upload "$@"

echo "done: $(date)"
