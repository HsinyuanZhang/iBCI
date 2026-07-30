# 03 — 检测点、验收顺序与容差

验证分层（任何层失败先在相邻层定位，不从最终 R² 反推）：

```text
L0 FP32 reference → L1 FP32 compiled → L2 fake-quant → L3 integer SW engine
→ L4 cycle-accurate sim → L5 RTL sim → L6 gate-level → L7 FPGA → L8 silicon
```

RTL 侧主要对 L3（integer software engine）做**逐元素 exact** 对齐。

---

## 1. B3 分层检查点 S0–S7

由 `golden/export_golden.py` 导出，逐 stage 对照。

| ID | 名称 | shape | dtype | 用途 |
|---|---|---|---|---|
| S0 | input (`calib_i8`) | `[M,T,N]` | INT8 | 输入 |
| S1 | pre accumulator (`pre_linear`) | `[M,N,D]` | INT32 | MAC+bias |
| S2 | pre ReLU/requant (`feat_i8`) | `[M,N,D]` | INT8 | ReLU 后、累加前 |
| S3 | `sum_feat_i32` | `[N,D]` | INT32 | DeepSets 累加 |
| S4 | `mean_i32` | `[N,D]` | INT32 | reciprocal mean |
| S5 | post0 relu | `[N,D]` | INT8 | |
| S6 | post1 relu | `[N,D]` | INT8 | |
| S7 | `E_i8` | `[N,W]` | INT8 | 最终 identity |

每个 checkpoint 记录：logical shape、linear layout、dtype、scale、hash、valid range、expected saturation。

---

## 2. 验收顺序（由易到难）

1. **tiny / M=1**：`--profile tiny`，单 neuron、小 `T/D/W`，所有整数 stage 逐元素相等。
2. **N>1**：验证权重共享（同套权重跑多 neuron）。
3. **M>1**：SUM 累加 + reciprocal mean + post MLP。
4. **full-shape**：`[33,100,96] → [96,50]` bit-exact。
5. **多 calibration draw**：检查状态机与 reciprocal，而非重新拟合 scale。
6. **model release**：切到冻结 release 包后重复 full-shape exact，再做 RTL/网表 sign-off。

---

## 3. 容差策略

| 对照 | 容差 |
|---|---|
| integer SW engine ↔ RTL（所有合同 stage） | **exact（逐元素相等）** |
| requant rounding / reciprocal mean / illegal-overflow / model hash | **exact** |
| FP32 reference ↔ algebraic compiled graph | 定义容差（默认 `atol=1e-5, rtol=1e-5`） |
| approximate LN/softmax ↔ accurate integer（DEFERRED） | 报 max_abs/mean_abs/RMSE/percentile/task-R²/worst-session |

FP32 逐层对照用 `b3_hw_golden.py --compare`；整数对照用 `export_golden.py` 的 `int8_stages/*` 逐元素比。

---

## 4. Metamorphic 测试（必须包含）

1. neuron 同步置换（online + calib 一起置换 → 输出同步置换）
2. trial 顺序置换（`E` 不变，浮点累加误差除外）
3. batch duplication
4. padding / mask 等价
5. split/cached vs unsplit（decoder，DEFERRED）
6. query cache invalidation on model reload
7. `E` invalidation on session reset
8. `N` tail tile（非 8 倍数）
9. `M=1` 与 `M=max`
10. zero / max / negative INT8 输入
11. abort & restart
12. DMA backpressure

---

## 5. RTL 单元验收

- **PE**：signed 乘法、acc clear/hold、overflow assertion、Mode A/B mapping、partial output/token mask。
- **SRAM**：bank mapping、同时读写、conflict/stall、地址越界、model CRC、retention/reset。
- **Postprocess**：INT64 product、rounding、shift、saturation、ReLU、residual add。
- **（DEFERRED）NLU**：reduction/max/sum/sumsq/reciprocal/rsqrt/exp LUT/softmax。

---

## 6. Cycle model 需建模项

PE compute、tile tails、weight/activation SRAM 带宽、bank conflict、postprocess overlap、reduction、（DEFERRED）LN/softmax、DMA、controller bubble、debug on/off。  
输出：per-layer cycles、per-layer 利用率、stall 拆分、per-block read/write、peak live memory、端到端 cycles、deadline margin。

---

## 7. 回归矩阵（关键行）

| Profile | M | N | 用途 |
|---|---:|---:|---|
| tiny encoder | 1 | 1/9 | 逐元素 bring-up |
| B3 M2 | 33 | 96 | encoder full |
| B3T M2 | 33 | 96 | 候选（待 W8A8 release） |
| tail stress | 7/33 | 159 | mask/tail |

decoder 相关行（decoder tiny / M2 small–baseline / SUA variable / M1-shape）见 `hardware_pe_sram/07 §11`，第一版 `DEFERRED`。

---

## 8. Tapeout 前 sign-off（节选）

contract version 冻结 · model release 冻结 · 所有 hash 记录 · integer SW == RTL exact · gate-level 回归 · SRAM macro 集成 · worst-shape timing/energy · session reset/error handling · permutation tests · formal/development evidence 标注 · 与专用 baseline 的 PPA 对照 · model reload+CRC · silicon debug 可观测性。
