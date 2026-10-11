#!/bin/bash
#SBATCH --job-name=llama1b-adamw-1h
#SBATCH --partition=8gpu
#SBATCH --account=ACD115198
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --time=01:10:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err


set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs out

module purge
module load cuda/12.6

export HF_HOME=/path/to/big_disk/hf_cache
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export NCCL_DEBUG=WARN

RUN_NAME=adamw-1h-${SLURM_JOB_ID}
MAX_TOKENS=5e8
MICRO_SEQS=2
GRAD_ACCUM=4
LR=4e-4
WARMUP=100

uv sync --frozen

echo "host: $(hostname) | start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv

torchrun --standalone --nproc_per_node=8 train.py \
    --data_dir data \
    --out_dir out \
    --run_name "$RUN_NAME" \
    --max_tokens "$MAX_TOKENS" \
    --micro_seqs "$MICRO_SEQS" \
    --grad_accum "$GRAD_ACCUM" \
    --lr "$LR" \
    --warmup_steps "$WARMUP" \
    --logging_steps 10 \
    --eval_interval 0.1 \
    --eval_docs 2000 \
    --save_every 500

echo "done: $(date)"