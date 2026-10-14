import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import json
import numpy as np
from datasets import load_dataset, Features, Sequence, Value
from transformers import AutoTokenizer

DATASET = "allenai/dolma3_mix-150B-1025"
TOKENIZER = "meta-llama/Llama-3.2-1B"
OUT_DIR = "data"
SEED = 42
N_EVAL = 50_000
EVAL_KEEP = 10_000
TRAIN_TOKENS = int(float(os.environ.get("TRAIN_TOKENS", 8e9)))
TRAIN_OUT = os.environ.get("TRAIN_OUT", "train.bin")
CHUNK_DOCS = 500_000
NUM_PROC = int(os.environ.get("NUM_PROC", 64))

tok = AutoTokenizer.from_pretrained(TOKENIZER)
EOS_ID = tok.eos_token_id


def tokenize(batch):
    ids = tok(batch["text"], add_special_tokens=True)["input_ids"]
    return {"input_ids": [x + [EOS_ID] for x in ids]}


def tokenize_part(part):
    return part.map(
        tokenize,
        batched=True,
        batch_size=1000,
        num_proc=NUM_PROC,
        remove_columns=part.column_names,
        features=Features({"input_ids": Sequence(Value("int32"))}),
    )


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    ds = load_dataset(DATASET, split="train", num_proc=NUM_PROC)
    ds = ds.shuffle(seed=SEED)
    n = len(ds)
    train = ds.select(range(n - N_EVAL))
    evals = ds.select(range(n - N_EVAL, n))
    print(f"total {n:,} docs | train {len(train):,} | eval {len(evals):,}")

    ev = tokenize_part(evals.select(range(EVAL_KEEP)))
    arrs, lens = [], []
    for b in ev.iter(batch_size=1000):
        for x in b["input_ids"]:
            arrs.append(np.asarray(x, dtype=np.uint32))
            lens.append(len(x))
    offsets = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    # 先寫暫存檔再 rename，正在讀舊檔的 job 不受影響，中途失敗也不會留下半個檔
    np.concatenate(arrs).tofile(f"{OUT_DIR}/eval.bin.tmp")
    os.replace(f"{OUT_DIR}/eval.bin.tmp", f"{OUT_DIR}/eval.bin")
    np.save(f"{OUT_DIR}/eval_offsets.tmp.npy", offsets)
    os.replace(f"{OUT_DIR}/eval_offsets.tmp.npy", f"{OUT_DIR}/eval_offsets.npy")
    print(f"eval: {EVAL_KEEP:,} docs, {offsets[-1]:,} tokens")

    out_path = f"{OUT_DIR}/{TRAIN_OUT}"
    out = np.memmap(out_path + ".tmp", dtype=np.uint32, mode="w+",
                    shape=(TRAIN_TOKENS,))
    pos, start, n_docs = 0, 0, 0
    while pos < TRAIN_TOKENS and start < len(train):
        end = min(start + CHUNK_DOCS, len(train))
        part = tokenize_part(train.select(range(start, end)))
        for b in part.iter(batch_size=10_000):
            arr = np.concatenate([np.asarray(x, dtype=np.uint32) for x in b["input_ids"]])
            take = min(len(arr), TRAIN_TOKENS - pos)
            out[pos:pos + take] = arr[:take]
            pos += take
            n_docs += len(b["input_ids"])
            if pos >= TRAIN_TOKENS:
                break
        print(f"train: docs [{start:,}, {end:,}) -> {pos:,} / {TRAIN_TOKENS:,} tokens")
        start = end
    out.flush()
    del out
    os.replace(out_path + ".tmp", out_path)

    with open(f"{OUT_DIR}/meta.json", "w") as f:
        json.dump({
            "dataset": DATASET, "tokenizer": TOKENIZER, "seed": SEED,
            "n_total_docs": n, "n_eval_reserved": N_EVAL, "eval_docs_kept": EVAL_KEEP,
            "train_tokens": int(pos), "train_docs_used_approx": n_docs,
        }, f, indent=2)


if __name__ == "__main__":
    main()