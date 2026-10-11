#!/bin/bash
#SBATCH --job-name=dolma3-tokenize
#SBATCH --partition=dev
#SBATCH --account=ACD115198
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

source ~/miniconda3/etc/profile.d/conda.sh
conda activate lab5

export HF_HOME=/path/to/big_disk/hf_cache
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false
export NUM_PROC=$SLURM_CPUS_PER_TASK

echo "host: $(hostname) | cpus: $NUM_PROC | start: $(date)"
df -h "$HF_HOME" .

python prepare_data.py

echo "done: $(date)"
ls -lh data/
cat data/meta.json
