#!/bin/bash
#SBATCH --job-name=llama1b-muon-3h
#SBATCH --partition=8gpus
#SBATCH --account=ACD115198
#SBATCH --exclude=25a-hgpn001,25a-hgpn003,25a-hgpn004,25a-hgpn007,25a-hgpn071
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
# 單節點：GPU 間走 NVLink，socket 只用於 bootstrap；有些節點 NCCL 會選到不通的網卡
# （hgpn160: connect ... No route to host），固定用 loopback
export NCCL_SOCKET_IFNAME=lo

# 預設為目前最佳設定（EXPERIMENTS.md 實驗 7、8：global batch 262K + lr 1.2e-3，1B tokens eval ppl 19.11）
OPTIMIZER=muon
RUN_NAME=${OPTIMIZER}-3h-${SLURM_JOB_ID}
MAX_TOKENS=1e9
MICRO_SEQS=4
TRAIN_FILE=train_packed_truncate.bin
LR=1.2e-3
WARMUP=200
DECAY_RATIO=0.2
DECAY_SHAPE=linear
SAVE_RESERVE_MIN=5

# 若以 array 送出，只查自己這個 task（array 最後一個 task 的 SLURM_JOB_ID 會列出整個 array）
JOB_REF=${SLURM_ARRAY_JOB_ID:+${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}}
JOB_END=$(squeue -h -j "${JOB_REF:-$SLURM_JOB_ID}" -o %e)
DEADLINE=$(date -d "$JOB_END" +%s) || { echo "cannot parse job end time: '$JOB_END'" >&2; exit 1; }
echo "job end: $JOB_END (deadline $DEADLINE)"

uv sync --frozen

echo "host: $(hostname) | start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv
# 有些節點的 GPU 處於 "GPU requires reset"，nvidia-smi 少列一張（hgpn071 只有 7 張）；
# 在開始訓練（sweep：在 wandb agent 領走 run）之前就結束，免得浪費 sweep 的組合
N_GPU=$(nvidia-smi -L | grep -c '^GPU')
if [ "$N_GPU" -ne 8 ]; then
    echo "ERROR: only $N_GPU/8 GPUs visible on $(hostname); add it to --exclude and resubmit" >&2
    exit 1
fi

uv run --frozen --env-file .env torchrun --standalone --nproc_per_node=8 \
    -m llama3_2_training.main \
    --data_dir data \
    --train_file "$TRAIN_FILE" \
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