# 测量协议 V4：估计量、不确定度与门槛

**状态：冻结协议，2026-07-25 建立。**
**适用范围：此后所有 DANDI 000688 SUA 与 FALCON MUA 的变体比较。**
**取代：`attention_arch_screen_v3` 及更早各版本使用的 best-checkpoint 口径。**

## 1. 为什么需要 V4

`attention_arch_screen_v3` 的结果不可用。诊断见
[`CURRENT_RESULTS.md`](CURRENT_RESULTS.md) §H，三条独立缺陷：

| # | 缺陷 | 后果 |
|---|---|---|
| H.1 | validation R² 的 epoch 间 `σ_SUA = 0.0388`、`σ_MUA = 0.0245` | 预注册门槛 `+0.005` 比噪声底低约 8 倍，不可能被有意义地通过 |
| H.2 | checkpoint 按 val 取 argmax，但各变体 epoch 数不等（SUA 6–10，MUA 8–40） | `E[max]` 随抽样次数增长，跑得久的变体被系统性高估 |
| H.4 | 两个 seed 共用 Hydra run 目录 | seed 归属不可考 |

并且换用同样合理的估计量后 `B15−B15P` 从 `+0.0064` 变成 `+0.0462`、
`B15P−B3` 符号翻转（§H.3）。**结论由估计量决定而非由架构决定**，这是
V3 必须作废的根本原因。

## 2. V4 的三处硬性改变

### 2.1 唯一 run 目录（对应 M1）

每个 `(variant, fold/task, seed)` 必须拥有独占的 run 目录。训练启动前断言
目标目录不含既有 checkpoint 或 tfevents，否则**直接 abort**。禁止依赖
时间戳生成目录名。

### 2.2 固定 epoch 预算（对应 M2）

关闭 early stopping。所有变体训练**完全相同**的 epoch 数。这消除 H.2 的
不均等 max-of-N 偏置——该偏置无法事后校正，只能靠等预算消除。

### 2.3 确定性 checkpoint 规则（对应 M3）

**不再对噪声 validation 取 argmax。** 预先固定：

- 训练 `E = 12` 个 epoch；
- 变体分数 = 协议指标在 epoch `5,6,7,8,9,10,11,12` 上的**算术平均**
  （4 epoch burn-in + 8 epoch 尾部平均）。

协议指标沿用既有的固定前向 calibration 评价：`first / n = 30 / pool = 50`，
在 6 个 validation session 上计算，encoder 与 decoder 全程冻结。

这是 V4 的核心。噪声主要通过"选择"这一步被注入并放大，而平均可以把它
按约 `1/√K_eff` 压下去（`K_eff < 8`，因为相邻 epoch 存在自相关）。

## 3. 必须报告不确定度

V3 只报点估计，这是把"未通过"误读成"已否定"的直接原因。V4 要求每个
比较同时给出：

1. **点估计**：paired delta 的 per-session seed 均值再取均值；
2. **窗口内离散度**：epoch 5–12 之间的 std，作为单次 run 的噪声度量；
3. **跨 seed 离散度**：各 seed 变体分数的 std；
4. **`σ_delta` 估计**：由上面两项合成，用于判定分辨力。

`σ_delta` 必须**从本轮数据实测**，不得沿用本文的先验估计。

## 4. 门槛（按可分辨性与部署相关性同时设定）

### 4.1 分辨力估算（已用 v3 曲线实测标定）

平均能压掉多少噪声取决于 epoch 序列的自相关。从 v3 的 8 条 SUA 曲线实测：

- 原始 lag-1 自相关 `= −0.233`；
- 但**均值中心化的短序列即使是白噪声，lag-1 期望也约为 `−1/(n−1)`**；
  这些序列长 6–10，该伪影期望为 `−0.165`；
- 偏置校正后 `ρ₁ ≈ −0.068`，与 0 无实质差别。

**结论：epoch 序列近似白噪声，8 epoch 平均可获得接近完整的 `√8` 降噪。**
（初版本文曾保守假设 `n_eff = 4`，该假设偏悲观，现按实测改为 `n_eff = 8`。）

```
σ_epoch   ≈ 0.0388                   (v3 实测，8 runs)
σ_run     ≈ 0.0388 / √8  ≈ 0.0137
σ_variant ≈ 0.0137 / √3  ≈ 0.0079    (3 seeds)
σ_delta   ≈ 0.0079 × √2  ≈ 0.0112
```

> ## 2026-07-26 修订：上式是错的，漏了 σ_seed
>
> `side_feature_ablation_v2`（首个 V4 screen，15 runs）实测：
>
> | 分量 | 实测值 |
> |---|---:|
> | 窗口内 std → `σ_run` | **0.0122** |
> | **跨 seed std → `σ_seed`** | **0.0385** |
> | `σ_delta`（3 seeds） | **0.024–0.039** |
> | `2σ_delta` | **0.048–0.077** |
>
> 上式隐含假设"不同 seed 只差 epoch 噪声"，即 `σ_variant = σ_run/√3`。
> **这是错的。** 不同 seed 有不同初始化、收敛到不同的解，存在一个不可约的
> `σ_seed`，而且它是**主导项**（`0.0385` vs `0.0122`）。正确的分解是：
>
> ```
> σ_variant = sqrt(σ_seed² + σ_epoch²/8) / √n_seeds
> σ_delta   = sqrt(σ_variant,A² + σ_variant,B²)
> ```
>
> M3 只压掉了 epoch 分量（v3 的 `0.0388` → `0.0122`，其中 burn-in 排除
> 贡献了约一半，超出预期），对 `σ_seed` 完全无效。
>
> **后果：`+0.03` 门槛落在噪声底之下（`2σ_delta = 0.048–0.077`）。**
> 要在 2σ 水平分辨 `+0.03` 需要约 **13 个 seed**。这与 v3 把门槛设在
> `+0.005` 是同一类错误，只是没那么极端。
>
> **今后写门槛前必须先估 `σ_seed`，不能只估 epoch 噪声。** 这是本协议
> 第二次因为低估噪声底而设出不可达门槛。

因此 v3 曲线给出的 **2σ ≈ `+0.022`** 是**乐观下界**，真实值见上表。要分辨
`+0.005` 需要约 240 次 run，本项目不具备该预算。

`σ_delta` 必须在每轮数据上重新实测（见 §3），不得沿用本节任何先验值。

### 4.2 部署相关性

按 [`ASIC_DEPLOYMENT_CHARTER.md`](ASIC_DEPLOYMENT_CHARTER.md) 的定位，
本项目的贡献点是可重构能力而非精度。`+0.005` 的 R² 差异**本来就不足以
为 `O(N²)` attention 和全局 neuron 缓冲付硅片代价**。

两条路径给出一致的量级，因此：

| Gate | 条件 |
|---|---|
| 变体可用 | 变体分数 > 0 |
| **机制有效** | paired delta 的 mean ≥ **`+0.03`**，且 6 个 session 中至少 5 个为正，且 3 个 seed 的逐 seed 均值全部为正 |
| **判定为无效** | mean delta ≤ `+0.03` **且** `|mean delta| > 2σ_delta` |
| **判定为不确定** | 以上两条都不满足 |

### 4.2b 2026-07-27 修正：`ineffective` 的判据原本是错的

> **这是修 bug，不是改阈值。** `+0.03` 的门槛值、seed 数、估计量窗口一律不变。

原判据 `|mean_delta| > 2σ_delta` 的含义是"效应**确定非零**"——对**正向**效应
而言恰好是反的。实证：`B3T − B3 = +0.0324`、配对 `2σ = 0.0203`、三个 seed 全正
（3.2 SE），却被判成 `ineffective`。**把一个确定为正的效应标成"无效"是明显
错误的**，与 V3 把"不确定"读成"已否定"属同一类混淆，只是方向相反。

**正确判据**：`ineffective` 应表示「能自信排除**至少 threshold 大小**的效应」：

```
ineffective  ⟺  mean_delta + 2·σ_delta_paired < effective_mean_delta
```

同时补一个此前缺失的状态。完整四态：

| 状态 | 条件 |
|---|---|
| `effective` | mean ≥ threshold **且** ≥5/6 session 为正 **且** 各 seed 均值全为正 |
| `effective_heterogeneous` | `mean − 2σ > 0`（确定为正）**且** 各 seed 均值全为正，但 session 一致性不满足 |
| `ineffective` | `mean + 2σ < threshold`（自信排除该量级效应） |
| `indeterminate` | 以上皆不满足 |

组级（多配对）逻辑：`effective` 要求**所有**配对满足（AND）；`ineffective`
只要**任一**配对自信排除即可（OR）——若某变体相对自己的置换对照已被自信排除，
它就不可能是 effective。原实现对 `ineffective` 也用 AND，过度保守。

### 4.2c 配对估计量（2026-07-27）

`σ_delta` 原用 `sqrt(σ_A²+σ_B²)/√n` 由两臂**独立**的跨 seed std 合成，
假设两臂 seed 效应互不相关。**但两臂用的是同一批 seed**，seed 难度共享、
在差值中大部分抵消。实测 `B3T/B3` 的两臂 seed 相关 `ρ ≈ 0.90`，quadrature
估计偏大 **2.4 倍**，会系统性把真实效应误判为 `indeterminate`。

```
σ_delta_paired = stdev(逐 seed 的 mean delta, ddof=1) / √n_seeds
```

判定一律使用配对值；输出必须同时记录 `σ_delta_paired`、
`σ_delta_unpaired_quadrature` 与隐含 `ρ`，便于读者判断差异来源。
`n_seeds < 2` 时配对 std 未定义，必须 raise，不得静默回退。

### 4.3 「无效」与「不确定」必须分开

这是 V4 相对 V3 最重要的改进。V3 把所有未过门槛的情形都记成
`gate: false`，而我把它读成了"已否定"。V4 要求聚合器显式输出三态：
`effective` / `ineffective` / `indeterminate`，并且只有 `ineffective`
才允许写成阴性结论。

## 5. 运行矩阵

### 5.1 Attention 重跑（`attention_arch_screen_v4`）

| | 值 |
|---|---|
| 变体 | B3, B15P, B15D, B15 |
| SUA | DANDI 000688 sub-C CO，`27/6/6`，seeds 42/43/44 → 12 runs |
| MUA | FALCON M2 internal LOSO，fold1/fold2 × seeds 42/43 → 后续阶段 |

先跑 SUA 12 个 run。MUA 在 SUA 结果出来、协议被确认有效后再排期——
MUA 的 `σ` 更小（0.0245），但 fold2 明显比 fold1 噪声大（0.024–0.077 vs
0.008–0.017），需要单独评估。

### 5.2 侧信息消融

见 [`UNIT_SIDE_FEATURE_ABLATION.md`](UNIT_SIDE_FEATURE_ABLATION.md)，
其门槛与估计量一并改用本协议。

## 6. 数据隔离（不变）

- 只使用 train + validation sessions；
- **6 个 test session 的 spike / behavior / trial 一律不加载**，只允许读
  test NWB 的 unit-table 行数以固定 `N < 100` regime；
- 不创建、不修改、不删除 formal-test receipt；
- 本协议下的一切结果都是 validation development evidence。

## 7. 本协议不能声称的内容

- 更低的估计量方差不等于更高的外部效度——它只是让同一批 validation
  session 上的比较可重复；
- `σ_delta` 由 6 个 session × 3 个 seed 估计，本身也有不确定度；
- V4 仍然无法分辨 `+0.005` 量级的效应，这是预算决定的，不是方法缺陷；
- 通过机制门槛只说明"值得投入 replication"，不构成 formal held-out 结论。
