#!/bin/bash
#SBATCH --job-name=dolma3-tokenize
#SBATCH --partition=dev
#SBATCH --account=ACD115198
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=96
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

module purge

export HF_HOME=/work/${USER}/.cache/huggingface
export HF_HOME=/work/${USER}/.cache/huggingface
export TOKENIZERS_PARALLELISM=false
export NUM_PROC=$SLURM_CPUS_PER_TASK

echo "host: $(hostname) | cpus: $NUM_PROC | start: $(date)"

uv run --frozen --env-file .env python -m llama3_2_training.tokenize

echo "done: $(date)"
ls -lh data/
cat data/meta.json
