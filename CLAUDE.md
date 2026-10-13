# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Pretraining a Llama-3.2-1B-architecture model from scratch (random init) on `allenai/dolma3_mix-150B-1025`, on a Slurm cluster with 8× H200 nodes. There is no test suite or linter; correctness is checked with short smoke runs on the `dev` partition (see below).

## Environment & commands

- Python 3.14, dependencies managed by `uv` (`uv.lock` is committed; scripts use `--frozen`). Add deps with `uv add`, never plain `pip install` — `uv sync --frozen` in `training.sh` removes anything not in the lock.
- Always run modules with `-m` from the repo root (e.g. `python -m llama3_2_training.main`). Running `src/llama3_2_training/main.py` as a file puts that directory on `sys.path`, and the project's `tokenize.py` shadows the stdlib `tokenize`, crashing at import.
- Secrets/config live in `.env` (`HF_TOKEN`, `WANDB_API_KEY`, `WANDB_ENTITY`, `WANDB_PROJECT`, `HF_REPO_ID`, `UPLOAD_MODEL_PATH`) and are loaded with `uv run --env-file .env`.
- `HF_HOME=/work/${USER}/.cache/huggingface` (Llama-3.2-1B tokenizer and hub kernels are cached there).

Slurm jobs (submit from the repo root; logs go to `logs/%x-%j.{out,err}`):

```bash
sbatch src/llama3_2_training/sbatch/download.sh   # tokenize dataset -> data/
sbatch src/llama3_2_training/sbatch/training.sh   # 8-GPU training run
sbatch src/llama3_2_training/sbatch/upload.sh [model_dir] [repo_id]   # push out/<run>/final to HF Hub
```

Smoke test (the de-facto "test"): short run on `dev` with wandb disabled, then delete the output:

```bash
srun --partition=dev --account=ACD115198 --exclude=25a-hgpn001,25a-hgpn003 --nodes=1 --ntasks=1 \
  --gpus-per-node=2 --cpus-per-task=8 --time=00:10:00 bash -c \
  "export HF_HOME=/work/\$USER/.cache/huggingface WANDB_MODE=disabled; \
   uv run --frozen --env-file .env torchrun --standalone --nproc_per_node=2 -m llama3_2_training.main \
   --out_dir out/_smoke --run_name smoke --micro_seqs 1 --max_tokens 3e6 --warmup_steps 20 \
   --eval_interval 0.25 --eval_docs 50 --save_every 100000"
rm -rf out/_smoke
```

Cluster notes: partitions are `dev` (≤4h) and `8gpus`/`16gpus`/…; the `dev` partition rejects jobs without `--gpus-per-node`. Nodes `25a-hgpn001` and `25a-hgpn003` have GPUs in "GPU requires reset" state (PyTorch sees no GPU / `invalid device ordinal`) — keep them excluded.

## Data pipeline

`tokenize.py` (run by `download.sh`) loads the full dataset, shuffles with a fixed seed, holds out the last 50k docs, and writes:

- `data/train.bin` — flat `uint32` token stream, exactly 6.2B tokens, each doc as `BOS ... EOS` concatenated with no padding.
- `data/eval.bin` + `data/eval_offsets.npy` — 10k held-out docs with per-doc offsets.
- `data/meta.json` — provenance.

`download.py` is a separate shard-subset downloader (`snapshot_download` into `dolma3_6b_raw/`); `tokenize.py` does not depend on it.

## Training (`main.py`) — key design points

- **Packing without padding**: `TrainData.get(step)` reads a fixed `micro_seqs × 8192` window from `train.bin`, indexed deterministically by `step` and rank. `make_position_ids` resets positions at every BOS and every 8192 boundary; labels at position 0 are masked. Document isolation inside a packed row relies on the flash-attention kernel's position_ids-based varlen handling, so the attention implementation must be a flash-attn kernel (`kernels-community/flash-attn2` via the `kernels` package; `flash-attn3` also works on Hopper). Plain `flash_attention_2` is not installed.
- **No gradient accumulation**: one forward/backward per step; global batch = `micro_seqs × 8192 × world_size` tokens. `micro_seqs` 8 OOMs on H200 without activation checkpointing.
- **Token-driven schedule**: `max_steps = min(max_tokens, len(train.bin)) // tokens_per_step` determines stopping, the WSD LR schedule (`scheduler.wsd_lr`: step-based warmup → stable → decay over the final `decay_ratio` of steps), and eval cadence (every `max_steps × eval_interval` steps).
- **Deadline safety net**: `training.sh` reads the job end time from `squeue` and passes `--deadline`. Each step, rank 0's clock and EMA step time are broadcast (`sync_from_rank0`) so all ranks agree; if the next step wouldn't finish before `deadline - save_reserve_min`, all ranks break together, run a final eval, and still save `final/` (with a warning that LR decay is incomplete). Any new early-exit condition must be decided collectively the same way, or DDP will hang.
- **Optimizer**: `--optimizer muon` (default) uses `torch.optim.Muon` (`adjust_lr_fn="match_rms_adamw"`, so it shares the AdamW-scale `--lr`) for 2D hidden weights, and AdamW for embeddings (tied with `lm_head`) and norms. `MultiOptimizer` wraps both so the loop and checkpoints treat them as one; checkpoint `"opt"` is a list of state dicts.
- **Outputs**: `out/<run_name>/ckpt.pt` (model + optimizer, every `--save_every` steps, auto-resumed if present) and `out/<run_name>/final/` (bf16 safetensors + tokenizer, HF `from_pretrained`-loadable). `RUN_NAME` includes `$SLURM_JOB_ID`, so each submission is a fresh run.
- **wandb**: rank 0 only; run files are written to a `tempfile.mkdtemp()` dir and deleted after `wandb.finish()`, so nothing is kept locally (no offline resync possible). Use `WANDB_MODE=disabled` for smoke tests.
- Log/error/help strings are in English; code comments are in Traditional Chinese.
