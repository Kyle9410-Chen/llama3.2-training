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

# wandb sweep: create the sweep from a yaml and submit one array task (8-GPU node + `wandb agent --count 1`) per run
uv run --frozen --env-file .env python -m llama3_2_training.launch_sweep src/llama3_2_training/sweeps/lrs.yaml [--parallel N] [--dry_run]
```

Smoke test (the de-facto "test"): short run on `dev` with wandb disabled, then delete the output:

```bash
srun --partition=dev --account=ACD115198 --exclude=25a-hgpn001,25a-hgpn003,25a-hgpn004,25a-hgpn007,25a-hgpn071 --nodes=1 --ntasks=1 \
  --gpus-per-node=2 --cpus-per-task=8 --time=00:10:00 bash -c \
  "export HF_HOME=/work/\$USER/.cache/huggingface WANDB_MODE=disabled; \
   uv run --frozen --env-file .env torchrun --standalone --nproc_per_node=2 -m llama3_2_training.main \
   --out_dir out/_smoke --run_name smoke --micro_seqs 1 --max_tokens 3e6 --warmup_steps 20 \
   --eval_interval 0.25 --eval_docs 50 --save_every 100000 --no-compile"
rm -rf out/_smoke
```

Cluster notes: partitions are `dev` (≤4h) and `8gpus`/`16gpus`/…; the `dev` partition rejects jobs without `--gpus-per-node`. Nodes `25a-hgpn001`, `25a-hgpn003`, `25a-hgpn004`, `25a-hgpn007` and `25a-hgpn071` (only 7 GPUs visible) have GPUs in "GPU requires reset" state (PyTorch sees no GPU / `invalid device ordinal`) — keep them excluded. `training.sh` and `sweep.sh` also check that `nvidia-smi` sees all 8 GPUs and exit before training (before `wandb agent` takes a sweep run) if not.

## Data pipeline

`tokenize.py` (run by `download.sh`) loads the full dataset, shuffles with a fixed seed, holds out the last 50k docs, and writes:

- `data/train.bin` — flat `uint32` token stream, `TRAIN_TOKENS` tokens (default 8B; was 6.2B — same seed, so the 6.2B prefix is identical), each doc as `BOS ... EOS` concatenated with no padding. Files are written to `.tmp` and renamed, so readers of the old file are unaffected.
- `data/eval.bin` + `data/eval_offsets.npy` — 10k held-out docs with per-doc offsets.
- `data/meta.json` — provenance.

`repack.py` rebuilds a training file from `train.bin` without re-tokenizing: it finds docs by BOS and best-fit-packs whole docs into 8192-token sequences (padding with `PAD_ID=128004`, ~0%; `make_inputs` masks pad labels), so short docs are never split across sequences. `--long_docs truncate` keeps only each doc's first 8192 tokens (matching how `build_eval_rows` scores eval docs); `--long_docs split` keeps tails as BOS-less chunks that are always placed at a sequence start (positions only reset at BOS or 8192 boundaries, so a BOS-less chunk mid-sequence would merge with the previous doc). Select the file with `main.py --train_file`; the default (in `main.py` and `training.sh`) is `train_packed_truncate.bin`, which beat plain `train.bin` (eval ppl 19.64 vs 20.55; `split` 20.09) at 1B tokens, lr 6e-4. `download.sh` runs tokenize → prefix check against the old `train.bin` → full `repack --long_docs truncate` in one job (`8gpus`, 4 GPUs, 48 CPUs, 5h; `dev` caps at 4h). In plain `train.bin`, ~34% of the first 1B tokens are trained without their doc's start in context (10.6% from short docs split at boundaries, 23.9% from >8192-token docs).

`download.py` is a separate shard-subset downloader (`snapshot_download` into `dolma3_6b_raw/`); `tokenize.py` does not depend on it.

## Training (`main.py`) — key design points

- **Packing without padding**: `TrainData.get(step)` reads a fixed `micro_seqs × 8192` window from `train.bin`, indexed deterministically by `step` and rank. `make_position_ids` resets positions at every BOS and every 8192 boundary; labels at position 0 are masked. Document isolation inside a packed row relies on the flash-attention kernel's position_ids-based varlen handling, so the attention implementation must be a flash-attn varlen kernel: `--attn_impl` defaults to `kernels-community/flash-attn3` (via the `kernels` package; ~10% faster than `kernels-community/flash-attn2` on H200, both verified to isolate packed docs). Plain `flash_attention_2` is not installed.
- **No gradient accumulation**: one forward/backward per step; global batch = `micro_seqs × 8192 × world_size` tokens. `micro_seqs` 8 OOMs on H200 without activation checkpointing; 4 fits (~133 GB reserved per GPU). Measured on 8×H200: `micro_seqs` 1/2/4 → ~318K/372K/429K tok/s. For batch sweeps, `--lr_ref_batch` treats `--lr` as the peak LR at that global batch and scales it by sqrt(batch ratio), and `--warmup_tokens` overrides `--warmup_steps`; the effective values overwrite `args`, but under a sweep wandb keeps the swept `lr` in the run config, so the effective LR is logged as `peak_lr` (and in `train/learning_rate`).
- **`training.sh` defaults** are the best 1B-token setting so far (see `EXPERIMENTS.md`): `MICRO_SEQS=4` (262K tokens/step), `LR=1.2e-3`, `WARMUP=200`, WSD linear decay over the last 20% to 0, `train_packed_truncate.bin`. `main.py`'s argparse defaults match these (plus `--eval_interval 0.1`, `--eval_docs 2000`, `--save_every 500`), so a bare `main.py` run uses the same setting.
- **Token-driven schedule**: `max_steps = min(max_tokens, len(train.bin)) // tokens_per_step` determines stopping, the WSD LR schedule (`scheduler.wsd_lr`: step-based warmup → stable → decay over the final `decay_ratio` of steps), and eval cadence (every `max_steps × eval_interval` steps).
- **Deadline safety net**: `training.sh` reads the job end time from `squeue` and passes `--deadline`. Each step, rank 0's clock and EMA step time are broadcast (`sync_from_rank0`) so all ranks agree; if the next step wouldn't finish before `deadline - save_reserve_min`, all ranks break together, run a final eval, and still save `final/` (with a warning that LR decay is incomplete). Any new early-exit condition must be decided collectively the same way, or DDP will hang.
- **Optimizer**: `--optimizer muon` (default) uses `torch.optim.Muon` (`adjust_lr_fn="match_rms_adamw"`, so it shares the AdamW-scale `--lr`) for 2D hidden weights, and AdamW for embeddings (tied with `lm_head`) and norms. `MultiOptimizer` wraps both so the loop and checkpoints treat them as one; checkpoint `"opt"` is a list of state dicts.
- **Throughput**: `--compile` (default on) wraps the DDP model in `torch.compile`; `flash_attention_forward` hits dynamo's recompile limit once per rank and falls back to eager (expected, attention is an opaque kernel). `--distributed_muon` (default on, `distributed_muon.py`) assigns each Muon matrix to one owner rank for Newton–Schulz and broadcasts the updated weights, instead of every rank repeating NS for all matrices. Together these took 8×H200 from ~250K to ~350K tok/s with identical loss (~387K with FA3). With distributed Muon the momentum lives only on its owner, so checkpoints are `ckpt.pt` (model + non-sharded opt state, rank 0) plus `opt-rank{r}.pt` per rank; resume requires the same world size.
- **HF Hub rate limits**: every rank queries the Hub for the flash-attn kernel at startup; many concurrent jobs (sweeps) get `429`. `build_model` is wrapped in `hub_retry.retry_on_rate_limit` (exponential backoff). `HF_HUB_OFFLINE=1` does not work here: `kernels` caches only the matching build variant, which `huggingface_hub` treats as an incomplete snapshot.
- **Outputs**: `out/<run_name>/ckpt.pt` (model + optimizer, every `--save_every` steps, auto-resumed if present) and `out/<run_name>/final/` (bf16 safetensors + tokenizer, HF `from_pretrained`-loadable). `RUN_NAME` includes `$SLURM_JOB_ID`, so each submission is a fresh run. Under a wandb sweep agent, `main.py` appends `$WANDB_RUN_ID` to `run_name` so parallel sweep runs don't share (and resume from) the same `out/` dir; the deadline is passed via `TRAIN_DEADLINE` because the sweep `command` can't carry per-job values.
- **wandb**: rank 0 only; run files are written to a `tempfile.mkdtemp()` dir and deleted after `wandb.finish()`, so nothing is kept locally (no offline resync possible). Use `WANDB_MODE=disabled` for smoke tests.
- Log/error/help strings are in English; code comments are in Traditional Chinese.
