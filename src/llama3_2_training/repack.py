import argparse
import json
import os
import time

import numpy as np

SEQ_LEN = 8192
BOS_ID = 128000
PAD_ID = 128004  # <|finetune_right_pad_id|>，train/eval 資料中都沒出現過


def get_args():
    p = argparse.ArgumentParser(
        description="Repack train.bin with best-fit packing so documents are not split across 8192-token sequences")
    p.add_argument("--src", default="data/train.bin")
    p.add_argument("--out", default=None, help="default: data/train_packed_<long_docs>.bin")
    p.add_argument("--long_docs", choices=["truncate", "split"], default="truncate",
                   help="docs longer than 8192: keep only the first 8192 tokens (truncate, like eval) "
                        "or cut into 8192-token chunks that are packed as separate items (split)")
    p.add_argument("--max_out_tokens", type=float, default=None,
                   help="stop once the packed output reaches this many positions (incl. padding); default: use all docs")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def doc_starts(data, chunk=200_000_000):
    return np.concatenate([np.flatnonzero(np.asarray(data[o:o + chunk]) == BOS_ID) + o
                           for o in range(0, len(data), chunk)])


def make_items(starts, n_tokens, long_docs, budget):
    """把文件轉成要裝箱的 (起點, 長度)；依原始順序取，直到真實 token 數達到 budget。"""
    ends = np.append(starts[1:], n_tokens)
    lens = ends - starts
    if long_docs == "truncate":
        item_s, item_l = starts, np.minimum(lens, SEQ_LEN)
        n_parts = np.ones(len(starts), dtype=np.int64)
        part = np.zeros(len(starts), dtype=np.int64)
    else:
        n_parts = (lens + SEQ_LEN - 1) // SEQ_LEN
        doc_idx = np.repeat(np.arange(len(starts)), n_parts)
        part = np.arange(len(doc_idx)) - np.repeat(np.cumsum(n_parts) - n_parts, n_parts)
        item_s = starts[doc_idx] + part * SEQ_LEN
        item_l = np.minimum(ends[doc_idx] - item_s, SEQ_LEN)
    if budget is not None:
        k = int(np.searchsorted(np.cumsum(item_l), budget)) + 1
        item_s, item_l, part = item_s[:k], item_l[:k], part[:k]
    n_docs = int(np.searchsorted(np.cumsum(n_parts), len(item_s), side="left")) + 1
    # 長文件後續的 chunk 沒有 BOS，main.py 只在 BOS 與 8192 邊界重設 position，
    # 所以它們必須放在 sequence 開頭（must_open），不能接在別的文件後面
    return item_s, item_l, part > 0, min(n_docs, len(starts)), int(ends[min(n_docs, len(starts)) - 1])


def best_fit_decreasing(lengths, rng, must_open=None):
    """Best-fit decreasing：每個 item 放進「剩餘空間最小但放得下」的 bin。

    剩餘容量只有 0..SEQ_LEN 共 SEQ_LEN+1 種，用線段樹找「>= x 的最小非空容量」，每個 item O(log SEQ_LEN)。
    """
    order = rng.permutation(len(lengths))                    # 同長度的 item 隨機排列
    order = order[np.argsort(-lengths[order], kind="stable")]
    size = 1 << (SEQ_LEN).bit_length()                       # 葉節點數 >= SEQ_LEN + 1
    tree = [0] * (2 * size)                                  # 每個節點：子樹內非空容量的 bin 數
    by_cap = [[] for _ in range(SEQ_LEN + 1)]                # 容量 -> bin id 清單
    bins = []

    def update(cap, delta):
        i = cap + size
        while i:
            tree[i] += delta
            i >>= 1

    def query(x, i=1, lo=0, hi=size - 1):
        # 最小的 cap >= x 且有 bin；優先走左子樹（容量小），沒有則回傳 None
        if hi < x or tree[i] == 0:
            return None
        if lo == hi:
            return lo
        mid = (lo + hi) // 2
        r = query(x, 2 * i, lo, mid)
        return r if r is not None else query(x, 2 * i + 1, mid + 1, hi)

    for idx in order:
        x = int(lengths[idx])
        cap = None if must_open is not None and must_open[idx] else query(x)
        if cap is None:
            bins.append([idx])
            rem = SEQ_LEN - x
        else:
            b = by_cap[cap].pop()
            update(cap, -1)
            bins[b].append(idx)
            rem = cap - x
            if rem:
                by_cap[rem].append(b)
                update(rem, 1)
            continue
        if rem:
            by_cap[rem].append(len(bins) - 1)
            update(rem, 1)
    return bins


def main():
    args = get_args()
    out = args.out or f"data/train_packed_{args.long_docs}.bin"
    rng = np.random.default_rng(args.seed)
    t0 = time.time()

    src = np.memmap(args.src, dtype=np.uint32, mode="r")
    starts = doc_starts(src)
    # 預估：輸出 = 真實 token + padding（BFD 的 padding 很少），先以 max_out_tokens 當真實 token 上限取文件
    item_s, item_l, must_open, n_docs, src_used = make_items(starts, len(src), args.long_docs, args.max_out_tokens)
    print(f"[{time.time() - t0:.0f}s] {len(starts):,} docs in {args.src} | using {n_docs:,} docs "
          f"({src_used:,} source tokens) -> {len(item_l):,} items, {int(item_l.sum()):,} real tokens", flush=True)

    bins = best_fit_decreasing(item_l, rng, must_open)
    bins = [bins[i] for i in rng.permutation(len(bins))]     # 打亂 bin 順序
    n_pos = len(bins) * SEQ_LEN
    print(f"[{time.time() - t0:.0f}s] packed into {len(bins):,} sequences of {SEQ_LEN} | "
          f"padding {1 - item_l.sum() / n_pos:.3%}", flush=True)

    dst = np.memmap(out + ".tmp", dtype=np.uint32, mode="w+", shape=(n_pos,))
    for b, items in enumerate(bins):
        pos = b * SEQ_LEN
        for i in items:
            s, l = int(item_s[i]), int(item_l[i])
            dst[pos:pos + l] = src[s:s + l]
            pos += l
        dst[pos:(b + 1) * SEQ_LEN] = PAD_ID
        if b % 20000 == 0 and b:
            print(f"[{time.time() - t0:.0f}s] wrote {b:,}/{len(bins):,} sequences", flush=True)
    dst.flush()
    del dst
    os.replace(out + ".tmp", out)

    meta = {
        "src": args.src, "long_docs": args.long_docs, "seed": args.seed, "seq_len": SEQ_LEN, "pad_id": PAD_ID,
        "src_docs_used": n_docs, "src_tokens_used": src_used, "items": int(len(item_l)),
        "real_tokens": int(item_l.sum()), "positions": n_pos, "sequences": len(bins),
        "padding_fraction": float(1 - item_l.sum() / n_pos),
        "dropped_tokens": int(src_used - item_l.sum()),
    }
    with open(out.replace(".bin", ".json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[{time.time() - t0:.0f}s] done: {out}\n{json.dumps(meta, indent=2)}", flush=True)


if __name__ == "__main__":
    main()
