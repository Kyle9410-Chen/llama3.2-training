# 實驗記錄

Llama-3.2-1B 架構（1,235,814,400 參數，random init）在 `allenai/dolma3_mix-150B-1025` 上從頭預訓練。
所有數字取自 wandb project `cerulean-labs/lab5-training-llama`（只列本 repo 的 run）與 Slurm log。
最後更新：2026-10-14。

## 共同設定

除非另外註明，所有 run 都是：

| 項目 | 值 |
|---|---|
| 硬體 | 1 節點 × 8 H200（`8gpus` partition），DDP |
| 序列長度 | 8192 |
| Global batch | `micro_seqs=2` × 8192 × 8 GPU = **131,072 tokens/step**（無 gradient accumulation） |
| 訓練量 | 1e9 tokens = **7,629 steps** |
| Optimizer | Muon（2D hidden 權重，`match_rms_adamw`，momentum 0.95，NS 5 步）+ AdamW（embedding/norm），共用 `--lr`；wd 0.1，β=(0.9, 0.95)，grad clip 1.0 |
| Warmup | 400 steps |
| Eval | held-out 文件前 2,000 篇，每篇取前 8192 tokens（從 BOS 起算），`eval/perplexity`；每 762 steps（10%）一次 |
| Seed | 0（每個設定只跑一個 seed） |

---

## 實驗總覽

| # | 實驗 | wandb | 結論 |
|---|---|---|---|
| 0 | 初期 1 小時 run（AdamW / Muon） | `os40bgpm`, `xco4bre0` | AdamW lr 3e-3 發散；之後一律用 Muon |
| 1 | LR scheduler（WSD vs 全程 decay） | sweep `9uu69baq` | **WSD linear、最後 20% decay** 最好（23.05） |
| 2 | 吞吐量優化 | dev 節點 benchmark | 249K → **387K tok/s**（+55%），loss 不變 |
| 3 | Peak LR | sweep `gzrjyatz` | **lr 6e-4** 最好（20.55） |
| 4 | 資料分析：文件被切斷的比例 | — | 34.4% 的訓練 token 沒有文件開頭前文 |
| 5 | Best-fit packing | sweep `xpiela5w` | **truncate** 最好（19.64，−4.4%） |
| 6 | 資料擴充到 8B + 全量 repack | Slurm job 539973 | `train_packed_truncate.bin` 6.6B 真實 tokens |
| 7 | Global batch size（65K / 131K / 262K） | sweep `gotyxib5` | **262K + lr 1.2e-3** 最好（19.11），吞吐量也最高 |
| 8 | Peak LR @ 262K batch | sweep `56sz8t18` | 最佳 lr 在 **1.2e-3** 附近；2e-3 開始變差，4e-3 發散 |

**目前最佳設定（ppl 19.11，8B packed 檔）**：Muon，global batch 262,144（`micro_seqs=4`），lr 1.2e-3，warmup 200 步，WSD linear decay 最後 20% 降到 0，`train_packed_truncate.bin`，1B tokens。

---

## 0. 初期 1 小時 run

在 token 驅動排程之前，LR 排程是按時間的 cosine（預設 6B tokens 的長度），1 小時內只跑到一小段。

| run | optimizer | lr | batch | 結果 |
|---|---|---|---|---|
| `adamw-1h-535317`（`os40bgpm`） | AdamW | 3e-3 | 524,288 tokens/step（grad_accum 4） | **發散**：step 261 ppl 366 → step 510 ppl 479，之後 crash；355K tok/s（FA3） |
| `muon-1h-535585`（`xco4bre0`） | Muon | 2e-4 | 131,072 tokens/step | 1 小時 6,926 steps / 0.91B tokens，ppl **26.56**；~250K tok/s（FA2，未 compile） |

`muon-1h` 的 eval：step 705 → 174.95，2081 → 44.59，4156 → 31.23，6926 → 26.56。LR 在這時還沒 decay（cosine 是照 6B 排的），所以這個數字偏高。

之後改成：訓練由 `MAX_TOKENS` 決定步數，排程依步數計算，時間只當安全網（deadline）。

---

## 1. LR scheduler sweep（`9uu69baq`）

固定 lr 2e-4，`min_lr_ratio` 0.1（decay 到 peak 的 10%）。變因：`decay_shape` ∈ {linear, sqrt, cosine} × `decay_ratio` ∈ {0.2, 1.0}。
`decay_ratio 1.0` = warmup 後整段都在 decay（傳統 schedule）；`0.2` = WSD（stable 到 80%，最後 20% decay）。

| decay_shape | decay_ratio | 最終 ppl | run |
|---|---|---|---|
| **linear** | **0.2** | **23.05** | `bnqdy58c` |
| sqrt | 0.2 | 23.11 | `y1zadzku` |
| cosine | 1.0 | 25.53 | `k65f37st` |
| linear | 1.0 | 25.53 | `4a0dgbch` |
| sqrt | 1.0 | 28.28 | `7b3kdjs9` |
| cosine | 0.2 | — | `2el0wizu`（啟動就 crash，沒有重跑） |

各 eval 點 ppl：

| step | linear 0.2 | sqrt 0.2 | linear 1.0 | cosine 1.0 | sqrt 1.0 |
|---|---|---|---|---|---|
| 762 | 156.98 | 156.98 | 157.70 | 157.00 | 163.29 |
| 2286 | 42.00 | 41.98 | 42.71 | 41.98 | 46.11 |
| 4572 | 30.14 | 30.14 | 30.45 | 29.62 | 33.13 |
| 6096 | 27.33 | 27.34 | 27.38 | 26.76 | 30.05 |
| 6858 | 24.81 | 24.31 | 26.29 | 25.98 | 29.02 |
| 7629 | **23.05** | 23.11 | 25.53 | 25.53 | 28.28 |

**解讀**
- WSD 在 stable 階段 ppl 比全程 decay 的 cosine 高（6096 步時 27.33 vs 26.76），但最後 20% 的 decay 會一口氣降下來，最終低 2.5 ppl。
- WSD 中 linear 跟 sqrt 幾乎一樣（差 0.06）。
- `k65f37st` 跑在 `25a-hgpn083`，只有 141K tok/s（其他約 247K），疑似節點 CPU / Weka 搶資源；不影響 ppl。

---

## 2. 吞吐量優化

都在同一台 dev 節點（8×H200）上各跑 250 steps 量測。

| 設定 | tok/s | 相對原始 |
|---|---|---|
| 原始（FA2，eager，每個 rank 都算全部 Muon NS） | 249K | — |
| + `torch.compile` | ~294K | +18% |
| + 分散式 Muon（只開這個） | ~281K | +13% |
| compile + 分散式 Muon（FA2） | 351–353K | +41% |
| **compile + 分散式 Muon + FA3** | **387K** | **+55%** |

單 GPU 每 step 時間拆解（`micro_seqs=2`）：

| 項目 | 原始 | 優化後 |
|---|---|---|
| forward + backward | 380 ms | 285 ms（compile） |
| Muon step | 94 ms | 11 ms（分散式 Muon） |

**正確性檢查**
- compile + 分散式 Muon 的 loss 在第 250 步 6.2157 vs 原始 6.2154。
- 分散式 Muon 跟原本 Muon 的權重差（5.37e-3）跟原本 Muon 自己跑兩次的差（5.35e-3）相同，差異來自 flash-attn backward 的不確定性。
- FA2、FA3 都驗證過：多篇文件 pack 在同一條序列時，logits 跟各自單獨算完全相同（差 0.0）。
- 實際訓練對照：`r6e2ncwa`（lr 2e-4，有全部加速）跟 `bnqdy58c`（同 lr，沒有加速，只有 `min_lr_ratio` 不同）的 eval 曲線在 decay 前幾乎重合（step 762：156.96 vs 156.98；step 6096：27.31 vs 27.33）。

每個 1B-token run 從約 71 分鐘縮到約 45 分鐘。

---

## 3. Peak LR sweep（`gzrjyatz`）

WSD linear、decay 最後 20%、`min_lr_ratio` 0（decay 到 0）。變因：lr ∈ {2e-4, 6e-4, 2e-3, 6e-3}。

| lr | 最終 ppl | run | 備註 |
|---|---|---|---|
| 2e-4 | 23.08 | `r6e2ncwa` | |
| **6e-4** | **20.55** | `e7e1zv32` | 原本的 task（`v51nxx15`）NCCL 錯誤 crash，手動重跑並掛回 sweep |
| 2e-3 | 35.43 | `9kg07v5k` | stable 階段 ppl 停在 ~46 後回升到 50，decay 後才降 |
| 6e-3 | 1309.84 | `kd30t48t` | 發散 |

各 eval 點 ppl：

| step | 2e-4 | 6e-4 | 2e-3 | 6e-3 |
|---|---|---|---|---|
| 762 | 156.96 | 70.42 | 67.16 | 245.6 |
| 1524 | 56.44 | 42.97 | 51.09 | 535.2 |
| 3048 | 36.15 | 32.06 | 45.86 | 473.8 |
| 4572 | 30.14 | 27.75 | 46.24 | 461.5 |
| 6096 | 27.31 | 25.78 | 50.12 | 1961.7 |
| 7629 | 23.08 | **20.55** | 35.43 | 1309.8 |

**解讀**
- 6e-4 在每個 eval 點都最好（除了第一個點 2e-3 略低）。最佳值落在 6e-4 附近，2e-3 已經太大；6e-4 和 2e-3 之間（例如 1e-3）還沒測。
- `min_lr_ratio` 0.1 → 0：lr 2e-4 時 23.05 vs 23.08，幾乎沒差。之後預設改為 0。
- AdamW（embedding / norm）跟 Muon 共用 lr，2e-3 以上的不穩定可能部分來自 AdamW 那側，尚未分開驗證。

---

## 4. 資料分析：沒有文件開頭前文的 token

原本的 `train.bin` 是把 `BOS ... EOS` 文件直接串接，再按 8192 切成固定 window。用實際訓練的前 1B tokens 統計：

| 項目 | 值 |
|---|---|
| 文件數 / 平均長度 / 中位數 | 66 萬篇 / 1,512 / 697 tokens |
| > 8192 tokens 的文件 | 2.4% 的文件，但佔 **30.3%** 的 tokens |
| 被 window 邊界切斷的文件 | 15.2% |
| 訓練時沒有文件開頭前文的 token | **34.4%** |
| ├ 短文件（≤ 8192）跨 window 邊界 | 10.6% |
| └ 長文件第 8192 個 token 之後 | 23.9% |
| Attention 計算量（相對整條 8192 單一文件） | 43.6% |

Eval 一律從文件開頭算前 8192 tokens，所以這 34.4% 的 token 跟 eval 的情境不一致。這是實驗 5 的動機。

---

## 5. Packing sweep（`xpiela5w`）

lr 6e-4，其他設定跟實驗 3 的 `e7e1zv32` 完全相同，只換訓練檔。`repack.py` 用 best-fit decreasing 把完整文件裝進 8192 的序列，padding 不算 loss。

- `truncate`：長文件只留前 8192 tokens（跟 eval 一致）。
- `split`：長文件切成 8192 的段落，沒有 BOS 的段落一律放在序列開頭。

| 訓練檔 | 最終 ppl | 相對 baseline | 沒有開頭前文的 token | run |
|---|---|---|---|---|
| `train.bin`（baseline） | 20.55 | — | 34.4% | `e7e1zv32` |
| `train_packed_split.bin` | 20.09 | −2.2% | ~16% | `j28c907z` |
| **`train_packed_truncate.bin`** | **19.64** | **−4.4%** | 0% | `xg8uoqm9` |

| step | baseline | split | truncate |
|---|---|---|---|
| 762 | 70.42 | 65.79 | 61.66 |
| 1524 | 42.97 | 41.15 | 39.53 |
| 3048 | 32.06 | 30.98 | 30.10 |
| 4572 | 27.75 | 27.11 | 26.36 |
| 6096 | 25.78 | 24.85 | 24.14 |
| 7629 | 20.55 | 20.09 | **19.64** |

**解讀**
- 每個 eval 點的排序都是 truncate < split < baseline，沒有交叉，不太像單一 seed 的雜訊（但每組只跑了一個 seed）。
- 改善幅度大致跟「修掉多少沒有前文的 token」成正比。
- Padding ~0%，吞吐量沒有變差（三個 run 都在 373–389K tok/s）。
- train loss 不能拿來比，三者的訓練分布不同。
- truncate 有一部分是針對 eval 計算方式最佳化，代價是看不到長文件 8192 之後的內容。

之後 `main.py` / `training.sh` 預設改用 `train_packed_truncate.bin`。

---

## 6. 資料擴充到 8B tokens + 全量 repack（Slurm job 539973）

`download.sh` 一個 job 做完：tokenize → 檢查前綴 → repack。`8gpus` partition，4 GPU / 48 CPU，花了 2 小時 10 分。

| 項目 | 值 |
|---|---|
| `train.bin` | 8,000,000,000 tokens（32 GB），5,294,559 篇文件（原本 6.2B） |
| 前綴檢查 | 新檔前 1B tokens 跟舊 `train.bin` 完全相同（同 seed，可重現） |
| `train_packed_truncate.bin` 真實 tokens | **6,615,431,252** |
| 截掉的 tokens | 1,384,568,748（17.3%） |
| Padding | 0.00015% |
| 序列數 | 807,549 × 8192（25 GB），可跑約 50.5K steps |

**注意**：新的 packed 檔是把 8B 範圍內的序列整體打亂，前 1B tokens 的文件組成跟實驗 5 用的檔案不同，所以新的 1B 實驗不能直接跟 19.64 比。要比較的話，先用新檔重跑一次 lr 6e-4 的 baseline。`train_packed_split.bin` 仍是舊版（從約 1.1B 來源產生）。

---

## 7. Global batch size sweep（`gotyxib5`）

使用新的 8B `train_packed_truncate.bin`。WSD linear、decay 最後 20% 降到 0、1B tokens。
變因：`micro_seqs` ∈ {1, 2, 4}（global batch 65K / 131K / 262K）× 2 個 lr。
lr 依 √batch 從 131K 換算（`--lr_ref_batch 131072`），每個 batch 測「√ 縮放值」和「再 ×√2」兩點；warmup 固定 52.4M tokens（`--warmup_tokens`）。

先在 dev 節點量吞吐量與記憶體（200 步，job 541166，`hgpn085`）：

| micro_seqs | Global batch | tok/s | 每 GPU 記憶體（nvidia-smi） |
|---|---|---|---|
| 1 | 65,536 | ~318K | 49 GB |
| 2 | 131,072 | ~372K | 87 GB |
| 4 | 262,144 | ~429K | 133 GB / 141 GB |

結果（Slurm array 541342）：

| Global batch | 步數 | warmup | 實際 lr | 最終 ppl | tok/s | 時間 | run |
|---|---|---|---|---|---|---|---|
| 65K | 15,258 | 800 | 4.24e-4 | 20.62 | 316K | 58 分 | `yj3m11da` |
| 65K | 15,258 | 800 | 6.0e-4 | 20.28 | 319K | 58 分 | `2kun7p5n` |
| 131K | 7,629 | 400 | 6.0e-4 | 19.71 | 381K | 47 分 | `29s1oew5`（新資料 baseline） |
| 131K | 7,629 | 400 | 8.49e-4 | 19.38 | 379K | 47 分 | `kjhf3sq0` |
| 262K | 3,814 | 200 | 8.49e-4 | 19.41 | 428K | 42 分 | `kdx4a2iz` |
| **262K** | 3,814 | 200 | **1.2e-3** | **19.11** | 427K | 42 分 | `jz59iedj` |

同 token 數的 eval ppl（每個 batch 取較好的 lr）：

| tokens | 65K / 6e-4 | 131K / 8.49e-4 | 262K / 1.2e-3 |
|---|---|---|---|
| 100M | 57.73 | 57.79 | 65.08 |
| 200M | 40.57 | 39.15 | 39.87 |
| 400M | 31.75 | 29.93 | 29.42 |
| 600M | 28.16 | 26.28 | 25.43 |
| 800M | 26.09 | 24.12 | 23.20 |
| 900M | 22.74 | 21.38 | 20.84 |
| 1B | 20.28 | 19.38 | **19.11** |

**解讀**
- **batch 越大越好**：最佳 ppl 65K 20.28 → 131K 19.38 → 262K 19.11。262K 只在最前面 100M tokens 落後（步數少、warmup 後 lr 才剛到高點），200M 之後就一直領先。
- **每個 batch 都是較高的 lr 比較好**：最佳 lr 可能還在更高的地方，尤其 262K 的 1.2e-3 是這次測到的上限，還沒看到 lr 太高的跡象。
- **新資料 baseline**：131K / 6e-4 是 19.71，跟舊 packed 檔的 19.64 很接近，表示換成 8B 範圍的資料沒有明顯影響。
- **吞吐量**：262K 比 131K 快 12%，在固定時間的訓練裡又多一個優勢。
- `micro_seqs=4` 每 GPU 只剩約 8 GB 記憶體餘裕，再大的 batch 需要 gradient accumulation 或 activation checkpointing。
- 注意：wandb config 裡的 `lr` 是 sweep 給的 131K 參考值，不是實際 lr；實際 lr 看 `train/learning_rate`（之後的 run 另外記成 config `peak_lr`）。

---

## 8. Peak LR sweep @ 262K batch（`56sz8t18`）

設定同實驗 7 的 262K 組（`micro_seqs=4`，warmup 200 步，WSD linear 0.2 → 0，8B packed 檔，1B tokens）。lr ∈ {6e-4, 1e-3, 2e-3, 4e-3}，再加上實驗 7 已有的 8.49e-4、1.2e-3 兩點。
第一次送出（sweep `xnn4s1gf`，job 542635）4 個 task 都排到 `25a-hgpn071`（只看得到 7 張 GPU，`invalid device ordinal`），各約 30 秒就失敗；排除該節點後重送為 `56sz8t18`（job 542774）。

| lr | 最終 ppl | run |
|---|---|---|
| 6e-4 | 19.91 | `6zyb82yv` |
| 8.49e-4 | 19.41 | `kdx4a2iz`（實驗 7） |
| 1e-3 | 19.26 | `yeg1mmys` |
| **1.2e-3** | **19.11** | `jz59iedj`（實驗 7） |
| 2e-3 | 19.42 | `agfx96bk` |
| 4e-3 | 95.87 | `5u595mlk`（發散） |

| tokens | 6e-4 | 1e-3 | 1.2e-3 | 2e-3 | 4e-3 |
|---|---|---|---|---|---|
| 100M | 101.2 | 69.09 | 65.08 | 61.97 | 68.40 |
| 200M | 42.57 | 39.95 | 39.87 | 41.58 | 58.43 |
| 400M | 30.21 | 29.58 | 29.42 | 31.57 | 118.43 |
| 600M | 26.06 | 25.58 | 25.43 | 26.81 | 185.44 |
| 800M | 23.75 | 23.31 | 23.20 | 24.10 | 160.43 |
| 1B | 19.91 | 19.26 | **19.11** | 19.42 | 95.87 |

**解讀**
- lr 曲線很平滑：6e-4 → 1.2e-3 一路變好，2e-3 開始變差，最佳值落在 1.2e-3 到 2e-3 之間偏 1.2e-3 的一側。再細掃（例如 1.5e-3）預期最多改善 ~0.1 ppl，不太值得。
- 2e-3 前 100M tokens 最低（61.97），但 200M 之後就落後，跟 131K 時 2e-3 的情況類似：高 lr 一開始降得快，stable 階段停滯。
- 4e-3 在 200M 之後 ppl 回升到 185，發散；decay 階段降回 96，但已經沒救。
- 262K 下最佳 lr（~1.2e-3）是 131K（~6e-4–8.5e-4）的約 1.4–2 倍，大致符合 √batch 縮放。
- 吞吐量 428–431K tok/s，每個 run 42 分鐘。

---

## 尚未做 / 待辦

- 524K batch（要加回 gradient accumulation 或用 activation checkpointing）。
- AdamW baseline 的 LR sweep（建議 {3e-4, 1e-3, 3e-3, 6e-3}），量化 Muon 帶來多少改善。
- 分開 Muon 跟 AdamW 的 lr（`--adamw_lr`）。
- cosine + decay 0.2 那組 crash 後沒有重跑。
- 所有結論都是單一 seed。

## 叢集問題記錄

| 節點 | 問題 | 處理 |
|---|---|---|
| `25a-hgpn001`, `003`, `004`, `007`, `071` | GPU 處於 "GPU requires reset"，PyTorch 看不到 GPU（`071` 只看得到 7 張） | 所有 sbatch 都 exclude；`training.sh` / `sweep.sh` 啟動時檢查是否有 8 張 GPU |
| `25a-hgpn083` | 吞吐量只有正常的 ~57%（141K tok/s），GPU 使用率在 30–100% 間跳動 | 未處理，疑似 CPU / Weka 搶資源 |
| `25a-hgpn160` | NCCL `connect ... No route to host` | `NCCL_SOCKET_IFNAME=lo`（單節點只需要 loopback） |
| — | 多個 job 同時啟動時，HF Hub 查 kernel 回 429 | `hub_retry.py` 指數退避重試 |
