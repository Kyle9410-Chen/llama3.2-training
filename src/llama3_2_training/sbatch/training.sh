#!/bin/bash
#SBATCH --job-name=llama1b-muon-3h
#SBATCH --partition=8gpus
#SBATCH --account=ACD115198
#SBATCH --exclude=25a-hgpn001,25a-hgpn003
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err


set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs out

module purge
module load cuda/13.0

export HF_HOME=/work/${USER}/.cache/huggingface
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export NCCL_DEBUG=WARN

OPTIMIZER=muon
RUN_NAME=${OPTIMIZER}-3h-${SLURM_JOB_ID}
MAX_TOKENS=1e9
MICRO_SEQS=2
LR=2e-4
WARMUP=400
DECAY_RATIO=0.2
DECAY_SHAPE=linear
SAVE_RESERVE_MIN=5

JOB_END=$(squeue -h -j "$SLURM_JOB_ID" -o %e)
DEADLINE=$(date -d "$JOB_END" +%s)
echo "job end: $JOB_END (deadline $DEADLINE)"

uv sync --frozen

echo "host: $(hostname) | start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv

uv run --frozen --env-file .env torchrun --standalone --nproc_per_node=8 \
    -m llama3_2_training.main \
    --data_dir data \
    --out_dir out \
    --run_name "$RUN_NAME" \
    --max_tokens "$MAX_TOKENS" \
    --micro_seqs "$MICRO_SEQS" \
    --optimizer "$OPTIMIZER" \
    --lr "$LR" \
    --warmup_steps "$WARMUP" \
    --decay_ratio "$DECAY_RATIO" \
    --decay_shape "$DECAY_SHAPE" \
    --deadline "$DEADLINE" \
    --save_reserve_min "$SAVE_RESERVE_MIN" \
    --logging_steps 10 \
    --eval_interval 0.1 \
    --eval_docs 2000 \
    --save_every 500

echo "done: $(date)"