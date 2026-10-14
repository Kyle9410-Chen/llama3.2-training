#!/bin/bash
#SBATCH --job-name=llama1b-lrs-sweep
#SBATCH --partition=8gpus
#SBATCH --account=ACD115198
#SBATCH --exclude=25a-hgpn001,25a-hgpn003,25a-hgpn004,25a-hgpn007,25a-hgpn071
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

# usage (one step):  uv run --frozen --env-file .env python -m llama3_2_training.launch_sweep [--parallel N]
# usage (manual):
#   1. uv run --frozen --env-file .env wandb sweep src/llama3_2_training/sweeps/lrs.yaml
#   2. sbatch --array=0-5 src/llama3_2_training/sbatch/sweep.sh <entity/project/sweep_id>
# 每個 array task 佔一個 8-GPU 節點，用 wandb agent 從 sweep 領一組設定跑一次

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs out

if [ $# -lt 1 ]; then
    echo "usage: sbatch --array=0-N $0 <entity/project/sweep_id>" >&2
    exit 1
fi
SWEEP_ID=$1

module purge
module load cuda/13.0

export HF_HOME=/work/${USER}/.cache/huggingface
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export NCCL_DEBUG=WARN
# 單節點：GPU 間走 NVLink，socket 只用於 bootstrap；有些節點 NCCL 會選到不通的網卡
# （hgpn160: connect ... No route to host），固定用 loopback
export NCCL_SOCKET_IFNAME=lo
# 從已經跑過 wandb 的環境送出時，WANDB_SERVICE 會指向 login node 的 socket，必須清掉
unset WANDB_SERVICE
# agent 的暫存（sweep config 檔等）放節點 tmp，不留在 repo
export WANDB_DIR=$(mktemp -d)
trap 'rm -rf "$WANDB_DIR"' EXIT

# main.py 會從 TRAIN_DEADLINE 讀 deadline（安全網）
# array 的最後一個 task 的 SLURM_JOB_ID 等於整個 array 的 id，直接 squeue 會列出所有 task，
# 所以用 <array_id>_<task_id> 只查自己
JOB_REF=${SLURM_ARRAY_JOB_ID:+${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}}
JOB_END=$(squeue -h -j "${JOB_REF:-$SLURM_JOB_ID}" -o %e)
TRAIN_DEADLINE=$(date -d "$JOB_END" +%s) || { echo "cannot parse job end time: '$JOB_END'" >&2; exit 1; }
export TRAIN_DEADLINE
echo "job end: $JOB_END (deadline $TRAIN_DEADLINE)"

uv sync --frozen

echo "host: $(hostname) | sweep: $SWEEP_ID | array task: ${SLURM_ARRAY_TASK_ID:-none} | start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv
# 有些節點的 GPU 處於 "GPU requires reset"，nvidia-smi 少列一張（hgpn071 只有 7 張）；
# 在開始訓練（sweep：在 wandb agent 領走 run）之前就結束，免得浪費 sweep 的組合
N_GPU=$(nvidia-smi -L | grep -c '^GPU')
if [ "$N_GPU" -ne 8 ]; then
    echo "ERROR: only $N_GPU/8 GPUs visible on $(hostname); add it to --exclude and resubmit" >&2
    exit 1
fi

uv run --frozen --env-file .env wandb agent --count 1 "$SWEEP_ID"

echo "done: $(date)"
