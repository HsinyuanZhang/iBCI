# Decision — B1 Shared-SPINT Alignment Hold

日期：2026-09-03  
数据集：FALCON B1 vocal / DANDI `001046`  
状态：**SUPERSEDED（2026-09-03）。B1 已由
`CLOSURE_B1_REMOVED_FROM_MAINLINE_20260903.md` 关闭并移出主线；本文 §3 的 ARTP-P 正证据含
minival/calib 重复 trial 造成的模板自检索，去重后日期等权增益为 +1.28%（见 closure §2）；
§8–§9 的 network-alignment cell 作废。以下正文仅作历史记录。**

原状态：HOLD；只整理结论，不授权实现、训练、Docker、EvalAI 或论文主张

## 1. 决策

结论分成两层：

1. **网络结构可以复用。** B1 可以直接使用 M1/M2 当前的
   `StreamingSpintModel + SideFeatureEarlyPoolEncoder (B3S)`。所需变化是数据窗口、
   calibration trial 长度、输出维度、carrier 维度、B1 输出包装和 metric，不需要重写
   unit-token Transformer 或 activity-identity 主干。
2. **当前 B1 正结果不能直接算作共享网络的跨数据集验证。** ARTP-P 的关键能力是用
   query neural activity 给三条有标签的 M3 声谱模板分配权重。显式保存和读取
   `3 × 158 × 880` 标签模板不是 SFC9/T4 carrier 的小改动，而是一条 task-specific
   output-value memory。

因此当前执行决定是：

- B1 暂不写入本次 T4/activity-memory 工作的主结果；
- 现有 ARTP-P 结果、payload、SFCJ/TARM 负结果全部原样保留；
- 不继续实现、不启动 GPU、不提交 EvalAI；
- 将来只有一个允许重开的 network-alignment cell，见 §8；
- 该 cell 失败后，B1 从本次工作中完全移除，不再继续扫 rank、pooling、gate、memory
  capacity 或训练配方。

## 2. 名称边界

必须保留下列区别：

| 名称 | 含义 | 是否为正式 Tuning Profile |
|---|---|---:|
| `SFC9` | 每通道的声学功能描述：intercept + 8 个 source-frozen spectral-PC tuning 系数 | 是 |
| `CAP` | M3-frozen per-channel activity mean/std，用于归一化神经活动 | 否 |
| `TRP` | 三条 M3 声谱模板的可靠性权重 | 否 |
| `ARTP-P` | raw/profiled activity similarity + TRP，对三条声谱模板加权 | 否 |

论文或代码不得把 ARTP-P 的 `profile` 写成 SFC9/Tuning Profile，也不得把 ARTP-P 的正数字
归因给 T4-like carrier。

## 3. 当前 B1 正证据

冻结的 ARTP-P 配置为：

```text
M=3, tau=.05, reliability strength=2, raw/profile mix=.5
```

source-only 三日期结果：

| 日期 | query trial | TPL-M3 MSE | ARTP-P MSE | 相对改善 |
|---|---:|---:|---:|---:|
| `20210626` | 10 | 0.000350166 | 0.000272345 | +22.22% |
| `20210627` | 28 | 0.000384070 | 0.000366335 | +4.62% |
| `20210628` | 7 | 0.000510542 | 0.000382166 | +25.14% |

日期等权绝对改善为 `+7.4643e-5`，约为等权模板 MSE 的 `18.0%`，3/3 日期为正。它同时
优于 cyclic neural control 和不读 query neural 的 reliability-only control。

这个结果证明：

> B1 query activity 可以作为 content-addressable key，在三条合法 M3 输出模板之间选择和
> 调权。

它没有证明：

> SFC9/Tuning Profile 已经作为 unit-level carrier 改善了共享 SPINT decoder。

ARTP-P 的实际信息流是：

```text
query activity + M3 calibration activity -> 三个 template weights
M3 calibration spectrogram labels       -> 三个 output values
weighted output values                   -> [158,880] prediction
```

所以整体属于 Tier-2 few-shot calibration。query label access 仍为零，但 M3 target labels 是
方法的必要输入。

## 4. 共享网络结构复用证明

使用当前公共实现，纯 CPU、无数据地实例化：

```text
StreamingSpintModel(
    decoder=SpintModel(
        model_dim=128,
        num_covariates=158,
        window_size=64,
        num_heads=8,
        num_layers=1,
    ),
    id_encoder=SideFeatureEarlyPoolEncoder(
        trial_length=900,
        window_size=64,
        hidden_dim=128,
        side_dim=9,
        num_post_layers=2,
    ),
)
```

验证过的张量几何：

```text
current neural window : [2,64,85]
calibration activity  : [2,3,900,85]
SFC9 side features    : [2,85,9]
activity identity     : [2,85,64]
decoder prediction    : [2,64,158]
last-frame prediction : [2,158]
```

结果：

- forward 成功；
- CUDA 未初始化；
- materialized parameter count 为 `785,728`（未计公共 decoder 内未使用的 lazy legacy-ID
  参数）；
- 同时排列 current neural、calibration activity 与 SFC9 的 unit 轴后，identity 的置换等变
  max-abs 为严格 `0`；
- 最终 prediction 的 max-abs 为 `4.76837158203125e-7`，属于 attention 浮点归约次序，
  没有结构性 unit-order 依赖。

这证明 B1 不需要另写 Transformer backbone。

## 5. B1 对齐时仅需改变的外围合同

| 层 | M1/M2 路径 | B1 对齐 |
|---|---|---|
| neural binning | 已准备的 decoder bins | 30 kHz 每 30 sample 合成 1 ms count |
| live window | `W=50/100` 等 | 冻结 `W=64`，或新工单事前冻结另一个单值 |
| calibration trial | M2 约 100 bin、M1 1024 bin | B1 整条 900 ms activity |
| units | session-specific | 85 channels |
| output tokens | 2 velocity / 16 EMG / 7 position | 158 frequency tokens |
| output law | dense or last-bin | `[B,W,158]` 取 last frame |
| carrier | T4/AFC4 等 | SFC9，`side_dim=9` |
| target space | behavior | source-standardized log spectrum |
| package output | streaming bins | 组装为 `[158,880]` |
| governing metric | R2 | B1 trial-wise min-max MSE |

下面的共享计算图保持不变：

```text
calibration activity
  -> shared per-unit pre_pool
  -> mean over available completed trials
  -> concat per-unit carrier
  -> shared post_pool
  -> identity E

current neural window + E
  -> shared fc_in
  -> learned output queries cross-attend to unordered unit tokens
  -> shared fc_out
  -> last timestep
```

## 6. 为什么严格公共 backbone 不等于正结果会自动出现

已有两个 TARM pilot 已经近似测试过这条结构。

### 6.1 Whole-trial TARM

- TPL-M3：`0.0003501655`；
- 四个网络臂的历史最佳约为 `0.00035449`；
- native/J-R1 与 Zero/SFC9 几乎没有可分差异。

### 6.2 Framewise TARM

- TPL-M3：`0.0003501655`；
- epoch 1 已是各臂最好的验证区域；
- `F-TA-NS9` epoch 1：约 `0.00035910`；
- `F-TA-N0` epoch 1：约 `0.00036006`；
- epoch 12 `F-TA-NS9`：约 `0.00036389`；
- epoch 12 `F-TA-N0`：约 `0.00037700`。

SFC9 相对等拓扑 Zero9 有局部正贡献，但两者都没有超过模板。与此同时，SFC9 training
official surrogate 从约 `0.0003965` 下降到 `0.0002902`，跨日期 validation 并未同步下降。

正确解释是：

> 网络具有充分的 source 拟合能力，但每个 LODO fold 只有两个独立 training dates；大量相关
> frame 不能替代 session/date 多样性。主要问题是 date-level generalization，而不是网络容量
> 或输出形状不够。

把近似 FrameTARM 换成字节级公共 SPINT 可能改变小量数值，但当前没有证据支持它会自动消除
跨日期过拟合。

## 7. 不允许作为“小改动”的方案

以下方案均超出 network-alignment cell，当前冻结：

1. 把三张 M3 声谱作为 Transformer value tokens；
2. 增加 query-to-calibration trial attention；
3. 学习三模板 mixture head；
4. 增加 SFC4/PC16/Direct159 rank sweep；
5. 同时展开 native/J-R1 × profile × memory × gate factorial；
6. growing CAP/profile；
7. target optimizer update；
8. 更多 reliability/tau/mix 搜索；
9. 以 ARTP-P 正数字代替共享网络的对照结果。

这些要么改变主架构，要么引入结果驱动选择，不能用于证明“只改细节即可跨数据集复用”。

## 8. 唯一允许的未来 network-alignment cell

若以后明确恢复 B1，只允许运行一次 **ARTP-Anchored Shared-SPINT**：

\[
\widehat Y
=
Y_{\mathrm{ARTP}}
+
\tanh(\beta)\,
\Delta_{\mathrm{SPINT}}(x,E^A,\mathrm{SFC9}).
\]

其中：

- `Y_ARTP` 是冻结的 ARTP-P 强锚点；
- `E^A` 由公共 B3S 从 calibration activity 产生；
- `Delta_SPINT` 由公共 `StreamingSpintModel + SpintModel` 产生；
- `beta` 是 source-only 学习的单一标量；
- `beta` 初始化为 IEEE `+0`；
- `beta=+0` 必须走直接返回原始 ARTP-P bytes 的分支，不能用 log/exp 数值近似冒充；
- 第一格不使用 query-dependent gate MLP。

只允许三个臂：

| Arm | Anchor | Shared SPINT residual | Carrier | 问题 |
|---|---|---|---|---|
| `A0-ARTP` | ARTP-P | 无 | 无 | 当前强锚点 |
| `A-Z9` | ARTP-P | 有 | literal Zero9 | 公共 activity-memory 网络是否增加信息 |
| `A-SFC9` | ARTP-P | 有 | 正式 SFC9 | Tuning Profile 是否有独立内容 |

第一格禁止 J-R1。已有 B1 pilot 中 J-R1 与 native 的差异接近零；把它再加入会增加一个没有决定
价值的因素。只有 `A-SFC9` 已经超过 `A0-ARTP` 后，J-R1 才可作为非 governing 补充。

## 9. 将来的纳入门

如果恢复 §8，只有满足以下门才能把 B1 写入本次工作的主结果。

### 9.1 共享网络门

\[
\mathrm{MSE}(A0\text{-ARTP})-\mathrm{MSE}(A\text{-Z9})>0.
\]

要求：

- 三个 source-LODO dates 同向，或预注册的 date/trial hierarchical bootstrap 下界大于 0；
- paired seeds `{42,43}` 方向一致；
- 统一训练 horizon；不得每臂单独挑最好 epoch；
- query label access 为零；
- `+0` anchor parity 必须精确。

### 9.2 Tuning Profile 内容门

\[
\mathrm{MSE}(A\text{-Z9})-\mathrm{MSE}(A\text{-SFC9})>0.
\]

要求：

- date-equal mean 为正；
- 至少 `2/3` dates 为正；
- 两个 seed 不反号；
- Zero9/SFC9 同宽、同初始化、同 dropout mask、同样本顺序、同训练步数。

判读：

- 两门都过：B1 可作为共享 activity-memory + functional-carrier 的扩展结果；
- 只过共享网络门：只能声称 activity-memory residual 有效，不能声称 Tuning Profile 有效；
- SFC9 相对 Zero9 为正、但所有网络臂仍不如 ARTP：只登记机制读数，B1 不进主结果；
- 两门都不过：B1 从本次工作完全移除，停止后续 B1 网络实验。

## 10. 当前停止边界

自本文档起，除非用户明确恢复 §8：

- 不新增 B1 source/GPU work order；
- 不改 B1 模型或数据代码；
- 不启动 B1 GPU；
- 不继续训练 TARM/SFCJ；
- 不构建新的 B1 Docker；
- 不提交 ARTP-P 或任何 B1 方法到 EvalAI；
- 不在当前论文中把 ARTP-P 写成 T4-like/shared-SPINT 验证。

可以继续做的只有只读引用和论文范围整理；现有结果根保持不变。

## 11. 论文表述

当前允许：

> B1 source-only analysis found that calibration activity can serve as a
> content-addressable key for selecting among labeled acoustic templates.
> Because this route uses a task-specific output-value memory rather than the
> shared SPINT carrier pathway, we did not count it as a cross-dataset validation
> of the main method.

当前禁止：

> Our T4-like network improves B1 by 18%.

也禁止：

> SFC9 caused the ARTP-P gain.

## 12. 证据绑定

| 证据 | 路径 | SHA-256 |
|---|---|---|
| ARTP-P result document | `tfpd_exploration/docs/RESULT_B1_ACTIVITY_RELIABILITY_TEMPLATE_PROFILING_V2_20260903.md` | `e0d5cd5d153d240d7658939f76128e2f318167cddf3eb9292e3d5cad14d83c62` |
| ARTP-P source screen | `tfpd_exploration/results/b1_artp_v2/source_screen.json` | `f8c7fd97b3dbfe1f93221441a3b34c695f83668b4346f4a35a5bddd6cdfd343a` |
| whole-trial TARM terminal | `tfpd_exploration/results/b1_tarm_v1/pilot_fold0_seed42/terminal.json` | `50c27777b644e5ebc9749093c0240328c87527bb3782d4e1301e23f02346f6c0` |
| framewise TARM terminal | `tfpd_exploration/results/b1_tarm_frame_v1/pilot_fold0_seed42/terminal.json` | `3bcbfa55aeccc90cda0237f74b678b25991c586b6f50def734734a1e657e5de1` |
| ARTP-P operator | `tfpd_exploration/src/b1_artp_v2/core.py` | `7bfe968f1bb81e32c32a756e8d4016f9c6bb6a2981a377696e79670b3b52e4f0` |
| framewise near-SPINT model | `tfpd_exploration/src/b1_tarm_frame_v1/model.py` | `7a47aad3e8336c4b7335d22cac3e6f5b200327133846181ef445c40e544da0f5` |
| shared StreamingSpintModel | `streaming_calibration_exp/src/models/components/streaming_spint.py` | `141129622ee2c187c053a0139f3926ab1880871d51dc88d0f8b93cc1b734ad9a` |
| shared B3S encoder | `streaming_calibration_exp/src/models/components/streaming_encoders.py` | `9cbd8f1a3f1acffdeaceea0eefbc20776e2f4156a3c7f43d6c1aab338989d5dc` |

这些 SHA 只绑定本次判断所读取的证据。本文档本身不是执行 closure，也不是 GPU/EvalAI
authorization。
