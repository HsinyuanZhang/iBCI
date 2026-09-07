# Movement-aligned general carrier：修订后的开发协议

**状态：** development-only protocol freeze，2026-07-31。  
**来源纪律：** 本协议是在审阅
`HANDOFF_GENERAL_CARRIER_PROGRAM.md` 并查看 M2 held-in calibration 数据后形成，
因此不是事前注册。所有阈值在正式 Gate-A artifact 与任何新 GPU cell 生成前固定；
本地 FALCON held-out-calib 和隐藏 EvalAI query/test 在 Gate A/B 中保持关闭。

> **Resolved 2026-08-01:** Gate A establishes support-only encoding value,
> but strict M24 held-out `K4-T4=+0.018987` misses the `+0.03` primary margin.
> The full CPU factorial finds reliable population/weighted W and reliable
> `||W||/b`, but unreliable typical per-channel direction (`.1837` cosine at
> M24); balancing and ridge do not rescue it. The conditional K4/KS4 GPU
> precision replication was not launched. Authoritative results:
> [`M2_CPU_CORRECTION_AND_CARRIER_RESULTS_20260801.md`](M2_CPU_CORRECTION_AND_CARRIER_RESULTS_20260801.md).

## 1. 被拒绝的原始定义

不实施下面的 whole-trial estimator：

```text
rate_i(trial) = b_i + w_i * mean_behavior(trial)
```

RT 每条 trial 含四次 reach，整 trial 均值会抵消；CO/M2 的 trial 也包含静止、启动、
到达等区间，整 trial 平均速度不等价于目标方向。该定义的 negative 或 positive result
均难以解释。

## 2. 首个候选：M2 movement-aligned K4

首轮只测试原生 FALCON M2，因为它已有严格的 native-MUA F0/T4/TS4 internal-LOSO
基线，输出与 carrier 都是二维，而且四维 side-feature width 可以完全保持不变。

每个 calibration trial 内构造不跨 trial 边界的非重叠 100 ms block：

```text
R_i[k] = sum neural_i[t : t+5]                 # 5 x 20 ms bins
Y[k]   = mean velocity[t+2 : t+7]              # neural leads behavior by 40 ms
R_i[k] = b_i + W_ix Y_x[k] + W_iy Y_y[k] + e_i[k]
K4_i   = [W_ix, W_iy, ||W_i||_2, b_i]
```

运动 block 沿用 FALCON 已有逐 bin still 判定：neural block 和相应的 shifted
behavior block 中，每个样本都必须满足 `~all(abs(velocity)<0.001)`；首轮 estimator
只用这些全程 active 的 blocks。拟合为带截距 OLS（两个 predictor，不搜索 ridge
alpha）。任何 response/label pair 必须具有相同 session、trial ID 和 split ID。

训练 session 的 K4 rows 提供四维 mean/std；validation/session-calibration K4 只能用
相应 calibration support，并用训练侧统计归一化。不得读取 minival/query behavior
来构造 side feature。

标签声明：K4 使用逐时间 bin 的连续速度标签，信息量显著大于 T4 的每 trial 一个
target-direction label。相同 trial 数不等于相同 label information；首轮只能声称
“相同 calibration-trial prefix 下的 dense-label carrier”。

## 3. Gate A：0-GPU carrier validity

使用七个 M2 held-in-calib sessions；每个 session 的 first-33 prefix 固定拆为：

```text
A = trials[0:17]       # fit K4
B = trials[17:33]      # untouched carrier evaluation
```

主指标是 B 上的 neural-response MSE，不是 decoding R2。比较：

1. `K4-aligned`：A 上拟合的 `W,b`；
2. `rate-only`：A 上每 channel 的均值，无行为项；
3. `W-shuffle`：`b_i` 保持 channel 对齐，只对完整二维 `W_i` rows 做 permutation；
4. 固定 seeds `0..99` 的 100 个 W-only permutations，报告 median null 和
   aligned 胜过 null 的次数。

Gate A 只有同时满足以下条件才通过：

- `mean(MSE_K4 / MSE_rate-only) <= 0.95`；
- `mean(MSE_K4 / median_MSE_W-shuffle) <= 0.95`；
- 两个 paired improvement 都是 7/7 session 为正；
- 两个 7-session exact two-sided Wilcoxon `p <= 0.05`；
- A/B 的 flattened `W` Pearson correlation 在每个 session 都大于 `0.5`；
- 完整 artifact 明确记录文件、trial/block counts、lag、block width、active rule、
  split、permutation seeds 和 `no_heldout_files_opened=true`。

`carrier inverse` 和 calibration-only ridge/Wiener 只作为诊断/竞争基线，不参与
Gate A。原因是 encoding model 的简单伪逆对异方差 spike noise 很敏感；它失败不能
反证 channel-attached encoding signature 不存在。

在 protocol freeze 前完成的探索性审计（不是正式 Gate-A artifact）得到：早期的
block-mean active 版本相对 rate-only 平均 MSE 降低约 `5.86%`。在把 active 条件收紧
为上述逐样本规则后，降低约 `6.56%`，相对 100-permutation median W-shuffle 降低约
`15.18%`；两项均为 7/7 sessions、exact `p=0.015625`，每个 session 的 aligned K4
都胜过 100/100 null。该结果只说明正式 Gate A 值得实现，不能作为 decoding 改进
结论。

## 4. Gate B：唯一首轮 GPU screen

仅在正式 Gate A 通过后启动，由 Terra subagent 具体负责。首轮固定为 M2
`fold1/seed42` 的两条新 arm，使用两个 GPU 并行：

| arm | 网络 | side feature | support |
|---|---|---|---|
| `k4` | 与现有 T4 完全相同的 B3S、`side_dim=4` | aligned K4 | chronological first 33 |
| `ks4` | 同一 B3S | 完整 K4 row 的 deterministic nonidentity channel shuffle | 同上 |

两臂必须匹配既有 `native_mua_t4_v1` 的 fold/seed、teacher、decoder freeze、loss、
12 epochs、checkpoint selection 和 held-out exclusion。F0/T4/TS4 使用同一 cell 的
既有严格 artifact，不重跑。不能 warm-start T4，不得用 K4 validation score选择 lag、
block width、active threshold 或 normalization。

首轮 decoding gate：

```text
K4 - T4  >= +0.03 R2
K4 - KS4 >= +0.03 R2
K4 - F0  >= +0.03 R2
```

`fold1/seed42` 只是机制 screen。若三项全过，补齐既有矩阵的
`fold1/seed43`、`fold2/seed42`；完整三-cell block 才能支持 internal-LOSO
development claim。若 K4 很早即低于 T4 超过 `0.05 R2`，或 K4 与 KS4 几乎相同，
停止该 cell 后续 epoch/replication，不以换 seed 挽救。

同标签 direct ridge/Wiener 必须单列报告。只有 K4-SPINT 在完整三-cell block 中超过
它，才可把收益解释为“跨 session 预训练模型利用了 encoding signature”；否则只能
把 K4 视为 supervised calibration descriptor，不能主张优于经典 few-shot decoder。

## 5. 只允许一次的优化分支

若 Gate A 通过但 Gate B 为小幅 negative/indeterminate（`K4-T4` 位于
`[-0.03,+0.03)` 且 `K4-KS4` 为正），允许唯一一次优化：

- 保持相同 100 ms/40 ms/active/split，不搜索时序参数；
- 将 K4 direction coefficient 做 A 内解析 reliability shrinkage，目标只能是零或
  training-session prior；
- shrinkage strength 仅由 train sessions nested-LOSO 选择；
- 重跑同一个 fold1/seed42 K4/KS4 pair。

若优化后仍未达到三项 `+0.03`，终止 general-carrier GPU 主线。不得继续堆 FiLM、
cross-attention 或更大 side dimension。

## 6. 后续顺序（仅在三-cell positive 后）

1. `M_K4=10/15/20/33`，共同 evaluation window，明确区分 trial budget 与 dense-label
   information；探索性 common-window proxy 显示 M10 才开始稳定超过 rate-only，M5
   不足，因此 M10 是首个候选而不是 M5。
2. RT：按四个 `go_cue_time_array` reach 分段，复用相同 trial-boundary/segment-boundary
   防泄漏约束；绝不退回 whole-trial mean。
3. M1：16-D EMG 形成 18-D `[W,||W||,b]`，必须单独做 parameter-matched control，
   不能冒充四维同架构实验。
4. 取得稳定候选后，才申请 FALCON hidden EvalAI test；随后再做 encoder-only INT8，
   decoder 量化继续复用其他平台已有结论。
