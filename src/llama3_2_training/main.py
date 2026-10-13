import argparse
import math
import os
import shutil
import tempfile
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig

from llama3_2_training.scheduler import wsd_lr

SEQ_LEN = 8192
BOS_ID = 128000
EOS_ID = 128001
TOKENIZER = "meta-llama/Llama-3.2-1B"

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data")
    p.add_argument("--out_dir", default="out")
    p.add_argument("--run_name", default="muon-baseline")
    p.add_argument("--wandb_project", default="lab5-training-llama")
    # deadline
    p.add_argument("--deadline", type=float,
                   default=float(os.environ["TRAIN_DEADLINE"]) if "TRAIN_DEADLINE" in os.environ else None,
                   help="safety net: unix timestamp to stop early and save if max_tokens is not reached (usually the Slurm job end time)")
    p.add_argument("--time_budget_min", type=float, default=None,
                   help="minutes available from launch, used when --deadline is not given")
    p.add_argument("--save_reserve_min", type=float, default=5.0,
                   help="minutes reserved before the deadline for the final eval and model save")
    # token
    p.add_argument("--max_tokens", type=float, default=6e9,
                   help="training token budget; sets total steps, WSD schedule and eval interval")
    p.add_argument("--micro_seqs", type=int, default=2,
                   help="number of 8192-token sequences per GPU per micro-step")
    # optimizer
    p.add_argument("--optimizer", choices=["muon", "adamw"], default="muon",
                   help="muon: Muon for hidden 2D weights + AdamW for embeddings/norms; "
                        "adamw: AdamW for everything")
    p.add_argument("--lr", type=float, default=4e-4,
                   help="peak LR, shared by Muon (RMS-matched to AdamW) and AdamW")
    p.add_argument("--min_lr_ratio", type=float, default=0.1)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--decay_ratio", type=float, default=0.1,
                   help="fraction of total steps (at the end) used for LR decay in WSD")
    p.add_argument("--decay_shape", choices=["linear", "sqrt", "cosine"], default="linear",
                   help="shape of the WSD decay phase (sqrt = 1 - sqrt(progress))")
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--muon_momentum", type=float, default=0.95)
    p.add_argument("--muon_ns_steps", type=int, default=5)
    p.add_argument("--grad_clip", type=float, default=1.0)
    # logging / eval / ckpt
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--eval_interval", type=float, default=0.05,
                   help="run eval every this fraction of total steps (spec requires <= 0.1)")
    p.add_argument("--eval_docs", type=int, default=10000)
    p.add_argument("--save_every", type=int, default=1000)
    p.add_argument("--push_to_hub", default=None, help="e.g. your-name/llama1b-adamw")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()

def build_model():
    config = LlamaConfig(
        vocab_size=128256,
        hidden_size=2048,
        intermediate_size=8192,
        num_hidden_layers=16,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=64,
        hidden_act="silu",
        max_position_embeddings=SEQ_LEN,
        rms_norm_eps=1e-5,
        rope_theta=500000.0,
        rope_scaling=None,
        tie_word_embeddings=True,
        attention_bias=False,
        mlp_bias=False,
        bos_token_id=BOS_ID,
        eos_token_id=EOS_ID,
        use_cache=False,
    )

    return AutoModelForCausalLM.from_config(config, attn_implementation="kernels-community/flash-attn2")


def make_position_ids(x, seq_len=None):
    idx = torch.arange(x.numel(), device=x.device)
    is_start = x == BOS_ID
    if seq_len is not None:
        is_start = is_start | (idx % seq_len == 0)
    is_start[0] = True
    start_idx = torch.where(is_start, idx, torch.zeros_like(idx))
    start_idx = torch.cummax(start_idx, dim=0).values
    return idx - start_idx


def make_inputs(x, seq_len=None):
    pos = make_position_ids(x, seq_len)
    labels = x.clone()
    labels[pos == 0] = -100
    return x[None], pos[None], labels[None]


class TrainData:
    def __init__(self, path, micro_tokens, rank, world):
        self.data = np.memmap(path, dtype=np.uint32, mode="r")
        self.micro_tokens = micro_tokens
        self.rank, self.world = rank, world

    def __len__(self):
        return len(self.data)

    def get(self, micro_idx):
        i = micro_idx * self.world + self.rank
        s = i * self.micro_tokens
        e = s + self.micro_tokens
        assert e <= len(self.data)
        return torch.from_numpy(self.data[s:e].astype(np.int64))


def build_eval_rows(data_dir, n_docs, row_tokens):
    data = np.fromfile(f"{data_dir}/eval.bin", dtype=np.uint32)
    off = np.load(f"{data_dir}/eval_offsets.npy")
    n_docs = min(n_docs, len(off) - 1)
    rows, cur, cur_len = [], [], 0
    for i in range(n_docs):
        doc = data[off[i]:off[i + 1]][:SEQ_LEN]
        if cur and cur_len + len(doc) > row_tokens:
            rows.append(np.concatenate(cur))
            cur, cur_len = [], 0
        cur.append(doc)
        cur_len += len(doc)
    if cur:
        rows.append(np.concatenate(cur))
    return rows


@torch.no_grad()
def evaluate(model, rows, device):
    model.eval()
    nll = torch.zeros((), device=device, dtype=torch.float64)
    cnt = torch.zeros((), device=device, dtype=torch.float64)
    for row in rows:
        x = torch.from_numpy(row.astype(np.int64)).to(device, non_blocking=True)
        ids, pos, labels = make_inputs(x)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(input_ids=ids, position_ids=pos, labels=labels).loss
        n = (labels[:, 1:] != -100).sum()
        nll += loss.double() * n
        cnt += n
    dist.all_reduce(nll)
    dist.all_reduce(cnt)
    model.train()
    return math.exp((nll / cnt).item())


class MultiOptimizer:
    """把多個 optimizer 包成一個，讓 training loop 與 checkpoint 不用分開處理。"""

    def __init__(self, *opts):
        self.opts = opts
        self.param_groups = [g for o in opts for g in o.param_groups]

    def step(self):
        for o in self.opts:
            o.step()

    def zero_grad(self, set_to_none=True):
        for o in self.opts:
            o.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return [o.state_dict() for o in self.opts]

    def load_state_dict(self, states):
        assert len(states) == len(self.opts), "checkpoint optimizer layout does not match"
        for o, sd in zip(self.opts, states):
            o.load_state_dict(sd)


def build_optimizer(model, args):
    # Muon 只用在 transformer block 內的 2D 權重；embedding（與 tied lm_head）與 norm 用 AdamW
    muon, decay, no_decay = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2:
            no_decay.append(p)
        elif args.optimizer == "muon" and "embed_tokens" not in name and "lm_head" not in name:
            muon.append(p)
        else:
            decay.append(p)
    adamw = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(args.beta1, args.beta2), eps=args.eps, fused=True)
    if not muon:
        return MultiOptimizer(adamw)
    muon_opt = torch.optim.Muon(muon, lr=args.lr, weight_decay=args.weight_decay,
                                momentum=args.muon_momentum, nesterov=True,
                                ns_steps=args.muon_ns_steps, adjust_lr_fn="match_rms_adamw")
    return MultiOptimizer(muon_opt, adamw)


def sync_from_rank0(values, device):
    t = torch.tensor(values, dtype=torch.float64, device=device)
    dist.broadcast(t, 0)
    return t.tolist()


def save_final(raw_model, args, tokens_seen, step):
    final_dir = os.path.join(args.out_dir, args.run_name, "final")
    raw_model.to(torch.bfloat16).save_pretrained(final_dir)
    print(f"saved model to {final_dir} | step {step} | total tokens {tokens_seen:,}", flush=True)
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    tok.save_pretrained(final_dir)
    if args.push_to_hub:
        raw_model.push_to_hub(args.push_to_hub)
        tok.push_to_hub(args.push_to_hub)


def main():
    t_launch = time.time()
    args = get_args()
    # deadline 只是安全網：token 用完前時間就到的話，提早停下並存模型
    if args.deadline is not None:
        deadline = args.deadline
    elif args.time_budget_min is not None:
        deadline = t_launch + args.time_budget_min * 60
    else:
        deadline = math.inf

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    is_main = rank == 0

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    micro_tokens = args.micro_seqs * SEQ_LEN
    tokens_per_step = micro_tokens * world

    train_data = TrainData(f"{args.data_dir}/train.bin", micro_tokens, rank, world)
    max_steps = int(min(args.max_tokens, len(train_data)) // tokens_per_step)
    eval_rows = build_eval_rows(args.data_dir, args.eval_docs, micro_tokens)[rank::world]

    model = build_model().to(device)
    raw_model = model
    opt = build_optimizer(raw_model, args)

    start_step, tokens_seen = 0, 0
    ckpt_path = os.path.join(args.out_dir, args.run_name, "ckpt.pt")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        raw_model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        start_step, tokens_seen = ckpt["step"], ckpt["tokens_seen"]
        if is_main:
            print(f"resumed from step {start_step}")
        del ckpt

    model = DDP(model, device_ids=[local_rank])

    if is_main:
        import wandb
        n_params = sum(p.numel() for p in raw_model.parameters())
        # wandb 一定要先寫本機暫存才能上傳；放到節點的 tmp，上傳完就刪，不留在 repo
        wandb_dir = tempfile.mkdtemp(prefix="wandb-")
        wandb.init(project=args.wandb_project, name=args.run_name, dir=wandb_dir,
                   config={**vars(args), "world_size": world,
                           "tokens_per_step": tokens_per_step,
                           "max_steps": max_steps, "n_params": n_params},
                   resume="allow")

    t_start, t_end = sync_from_rank0([time.time(), deadline - args.save_reserve_min * 60], device)
    budget = t_end - t_start
    if budget <= 0:
        raise SystemExit(f"not enough time: only {(deadline - t_start) / 60:.1f} min left before deadline, "
                         f"but {args.save_reserve_min} min is reserved for saving")
    eval_every = max(1, int(max_steps * args.eval_interval))
    if is_main:
        limit = "no time limit" if math.isinf(budget) else f"time limit {budget/60:.1f} min"
        print(f"params {n_params/1e9:.3f}B | tokens/step {tokens_per_step:,} | "
              f"max steps {max_steps:,} ({max_steps * tokens_per_step:,} tokens) | {limit} | "
              f"eval every {eval_every} steps", flush=True)

    model.train()
    window_loss, window_n = 0.0, 0
    step, step_time = start_step, 0.0
    last_eval_step = None
    stop_reason = "max_tokens"
    while step < max_steps:
        now, step_time = sync_from_rank0([time.time(), step_time], device)
        if now + step_time > t_end:
            stop_reason = "time"
            if is_main:
                print(f"WARNING: time limit reached at step {step}/{max_steps} before the token "
                      f"budget was used up; LR decay is incomplete", flush=True)
            break
        progress = step / max_steps
        t0 = time.perf_counter()

        lr = wsd_lr(step, max_steps, peak_lr=args.lr, warmup_steps=args.warmup_steps,
                    decay_ratio=args.decay_ratio, min_lr_ratio=args.min_lr_ratio,
                    decay_shape=args.decay_shape)
        for g in opt.param_groups:
            g["lr"] = lr

        x = train_data.get(step).to(device, non_blocking=True)
        ids, pos, labels = make_inputs(x, seq_len=SEQ_LEN)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(input_ids=ids, position_ids=pos, labels=labels).loss
        loss.backward()
        step_loss = loss.detach()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        opt.zero_grad(set_to_none=True)

        dist.all_reduce(step_loss, op=dist.ReduceOp.AVG)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        step_time = dt if step_time == 0.0 else 0.9 * step_time + 0.1 * dt
        tokens_seen += tokens_per_step
        step += 1

        window_loss += step_loss.item()
        window_n += 1

        if step % args.logging_steps == 0 and is_main:
            log = {
                "train/loss": window_loss / window_n,
                "train/grad_norm": grad_norm.item(),
                "train/learning_rate": lr,
                "train/tokens_per_second": tokens_per_step / dt,
                "train/total_tokens_seen": tokens_seen,
                "train/token_progress": progress,
            }
            wandb.log(log, step=step)
            print(f"step {step}/{max_steps} ({progress:.1%}) | loss {log['train/loss']:.4f} | "
                  f"gnorm {log['train/grad_norm']:.3f} | lr {lr:.2e} | "
                  f"{log['train/tokens_per_second']/1e3:.1f}K tok/s", flush=True)
        if step % args.logging_steps == 0:
            window_loss, window_n = 0.0, 0

        if step % eval_every == 0:
            ppl = evaluate(raw_model, eval_rows, device)
            last_eval_step = step
            if is_main:
                wandb.log({"eval/perplexity": ppl, "train/total_tokens_seen": tokens_seen},
                          step=step)
                print(f"[eval] step {step} | ppl {ppl:.3f}", flush=True)

        if step % args.save_every == 0 and is_main:
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            torch.save({"model": raw_model.state_dict(), "opt": opt.state_dict(),
                        "step": step, "tokens_seen": tokens_seen}, ckpt_path + ".tmp")
            os.replace(ckpt_path + ".tmp", ckpt_path)

    if is_main:
        print(f"stopped at step {step} ({stop_reason}) | {tokens_seen:,} tokens | "
              f"{(time.time() - t_start) / 60:.1f} min", flush=True)
    if last_eval_step != step:
        ppl = evaluate(raw_model, eval_rows, device)
        if is_main:
            wandb.log({"eval/perplexity": ppl, "train/total_tokens_seen": tokens_seen}, step=step)
            print(f"[eval] final step {step} | ppl {ppl:.3f}", flush=True)

    if is_main:
        save_final(raw_model, args, tokens_seen, step)
        wandb.finish()
        shutil.rmtree(wandb_dir, ignore_errors=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()