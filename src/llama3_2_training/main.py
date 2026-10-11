import argparse
import math
import os
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig

SEQ_LEN = 8192
BOS_ID = 128000
EOS_ID = 128001
TOKENIZER = "meta-llama/Llama-3.2-1B"

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data")
    p.add_argument("--out_dir", default="out")
    p.add_argument("--run_name", default="adamw-baseline")
    p.add_argument("--wandb_project", default="lab5-training-llama")
    # token budget 與 batch
    p.add_argument("--max_tokens", type=float, default=6e9)
    p.add_argument("--micro_seqs", type=int, default=2,
                   help="每張 GPU 每個 micro-step 處理幾條 8192 序列")
    p.add_argument("--grad_accum", type=int, default=4)
    # optimizer
    p.add_argument("--lr", type=float, default=4e-4)
    p.add_argument("--min_lr_ratio", type=float, default=0.1)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--grad_clip", type=float, default=1.0)
    # logging / eval / ckpt
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--eval_interval", type=float, default=0.05,
                   help="每隔總步數的多少比例 eval 一次（spec 要求 <= 0.1）")
    p.add_argument("--eval_docs", type=int, default=10000)
    p.add_argument("--save_every", type=int, default=1000)
    p.add_argument("--push_to_hub", default=None, help="例如 your-name/llama1b-adamw")
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

    return AutoModelForCausalLM.from_config(config, attn_implementation="flash_attention_2")


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


def build_optimizer(model, args):
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim < 2 else decay).append(p)
    groups = [
        {"params": decay, "weight_decay": args.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=args.lr, betas=(args.beta1, args.beta2),
                             eps=args.eps, fused=True)


def lr_at(step, total_steps, args):
    if step < args.warmup_steps:
        return args.lr * (step + 1) / args.warmup_steps
    progress = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
    cosine = 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
    return args.lr * (args.min_lr_ratio + (1 - args.min_lr_ratio) * cosine)


def main():
    args = get_args()

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
    tokens_per_step = micro_tokens * args.grad_accum * world
    total_steps = int(args.max_tokens // tokens_per_step)
    eval_every = max(1, int(total_steps * args.eval_interval))

    train_data = TrainData(f"{args.data_dir}/train.bin", micro_tokens, rank, world)
    assert total_steps * tokens_per_step <= len(train_data), "train.bin 的 token 數不足"
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
        wandb.init(project=args.wandb_project, name=args.run_name,
                   config={**vars(args), "world_size": world,
                           "tokens_per_step": tokens_per_step,
                           "total_steps": total_steps, "n_params": n_params},
                   resume="allow")
        print(f"params {n_params/1e9:.3f}B | tokens/step {tokens_per_step:,} | "
              f"steps {total_steps:,} | eval every {eval_every} steps")

    model.train()
    window_loss, window_n = 0.0, 0
    for step in range(start_step, total_steps):
        t0 = time.perf_counter()

        lr = lr_at(step, total_steps, args)
        for g in opt.param_groups:
            g["lr"] = lr

        step_loss = torch.zeros((), device=device)
        for m in range(args.grad_accum):
            x = train_data.get(step * args.grad_accum + m).to(device, non_blocking=True)
            ids, pos, labels = make_inputs(x, seq_len=SEQ_LEN)
            sync_ctx = model.no_sync() if m < args.grad_accum - 1 else nullcontext()
            with sync_ctx:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = model(input_ids=ids, position_ids=pos, labels=labels).loss
                (loss / args.grad_accum).backward()
            step_loss += loss.detach() / args.grad_accum

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        opt.zero_grad(set_to_none=True)

        dist.all_reduce(step_loss, op=dist.ReduceOp.AVG)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        tokens_seen += tokens_per_step

        window_loss += step_loss.item()
        window_n += 1

        if (step + 1) % args.logging_steps == 0 and is_main:
            log = {
                "train/loss": window_loss / window_n,
                "train/grad_norm": grad_norm.item(),
                "train/learning_rate": lr,
                "train/tokens_per_second": tokens_per_step / dt,
                "train/total_tokens_seen": tokens_seen,
            }
            wandb.log(log, step=step + 1)
            print(f"step {step+1}/{total_steps} | loss {log['train/loss']:.4f} | "
                  f"gnorm {log['train/grad_norm']:.3f} | lr {lr:.2e} | "
                  f"{log['train/tokens_per_second']/1e3:.1f}K tok/s")
        if (step + 1) % args.logging_steps == 0:
            window_loss, window_n = 0.0, 0

        if (step + 1) % eval_every == 0 or step + 1 == total_steps:
            ppl = evaluate(raw_model, eval_rows, device)
            if is_main:
                wandb.log({"eval/perplexity": ppl, "train/total_tokens_seen": tokens_seen},
                          step=step + 1)
                print(f"[eval] step {step+1} | ppl {ppl:.3f}")

        if (step + 1) % args.save_every == 0 and is_main:
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            torch.save({"model": raw_model.state_dict(), "opt": opt.state_dict(),
                        "step": step + 1, "tokens_seen": tokens_seen}, ckpt_path + ".tmp")
            os.replace(ckpt_path + ".tmp", ckpt_path)

    if is_main:
        final_dir = os.path.join(args.out_dir, args.run_name, "final")
        raw_model.to(torch.bfloat16).save_pretrained(final_dir)
        tok = AutoTokenizer.from_pretrained(TOKENIZER)
        tok.save_pretrained(final_dir)
        if args.push_to_hub:
            raw_model.push_to_hub(args.push_to_hub)
            tok.push_to_hub(args.push_to_hub)
        print(f"saved to {final_dir} | total tokens {tokens_seen:,}")
        wandb.finish()

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()