#!/bin/bash
#SBATCH --job-name=dolma3-tokenize
#SBATCH --partition=8gpus
#SBATCH --account=ACD115198
#SBATCH --exclude=25a-hgpn001,25a-hgpn003,25a-hgpn004,25a-hgpn007,25a-hgpn071
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=48
#SBATCH --time=05:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

module purge

export HF_HOME=/work/${USER}/.cache/huggingface
export TOKENIZERS_PARALLELISM=false
export NUM_PROC=$SLURM_CPUS_PER_TASK

echo "host: $(hostname) | cpus: $NUM_PROC | start: $(date)"

# 1. tokenize 8B tokens 到新檔（同 seed、同 cache，前 6.2B 應與現有 train.bin 完全相同）
export TRAIN_TOKENS=8e9 TRAIN_OUT=train_8b.bin
uv run --frozen --env-file .env python -m llama3_2_training.tokenize
cat data/meta.json

# 2. 確認前 1B tokens（4GB，先前實驗用到的部分）與舊 train.bin 相同，才取代
if [ -f data/train.bin ] && ! cmp -n 4000000000 data/train.bin data/train_8b.bin; then
    echo "ERROR: first 1B tokens differ from the existing train.bin; keeping both files, skipping repack" >&2
    exit 1
fi
mv data/train_8b.bin data/train.bin
echo "train.bin replaced: $(date)"

# 3. 全量 repack（truncate：每篇只留前 8192 tokens，與 eval 計算方式一致）
uv run --frozen python -m llama3_2_training.repack --long_docs truncate

echo "done: $(date)"
ls -lh data/
