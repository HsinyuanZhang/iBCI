# 侧信息特征跨 Session 漂移诊断

**状态：诊断完成。纯诊断，未训练任何模型，CPU-only，未用 GPU。2026-07-26。**
**数据隔离：与 `side_feature_ablation_v2` 相同的 27 train / 6 validation sessions。
6 个 test sessions 全程未被打开（连允许的 unit-count 读取都未使用，因为本诊断不需要）。**

## 0. 背景

`side_feature_ablation_v2`（`results/side_feature_ablation_v2/aggregate.json`）跑完后，
F1（SNR/幅度，3 标量）与 F2（+波形形状，6 标量）均判定为 `indeterminate`，但点估计
全部为负，且 shuffled 对照（FS1/FS2）系统性地**优于**真实特征：

| 对比 | mean delta | 6 session 中为正 | 3 seed 均值为正 |
|---|---:|---:|---|
| F1 − F0 | −0.0324 | 1/6 | 0/3 |
| F1 − FS1 | −0.0457 | 0/6 | 0/3 |
| F2 − F0 | −0.0012 | 2/6 | 1/3 |
| F2 − FS2 | −0.0367 | 1/6 | 1/3 |

**待检验假设**：波形/幅度标量跨 recording day 系统性漂移（电极阻抗变化、分选阈值不同
等），且由于它们用**仅 train session** 的统计量做 z-score（`UNIT_SIDE_FEATURE_ABLATION.md`
§6.1），train→validation 的分布偏移会给 identity 向量注入一个**session 相关的偏置**，
主动损害跨 session 泛化——这能解释为什么真实值有害而同维度的随机值反而有利。

本诊断**只**量化这个漂移，不做任何建模、不改 `unit_side_features.py`、不改消融结果、
不碰 `p3_formal_test_816cdd8b*_receipt.json`。

## 1. 方法与复用声明

- 特征计算全部复用 `mc_maze/unit_side_features.py` 的
  `compute_unit_side_features_uncached` / `load_unit_side_features` /
  `fit_side_feature_stats` / `_fit_robust_stats` / `_in_pool_spike_prefix`，
  **未重新实现**任何标量定义或归一化逻辑。
- 参数与 `side_feature_ablation_v2` 训练时完全一致（已用磁盘 cache 命中验证，见下）：
  `feature_group=f2`（p2p, noise_std, snr, pt_width, pt_ratio, repol_slope），
  `pool_size=50`，`bin_size_ms=20`，`window_size=50`，`trial_result_filter="R"`。
- F1 的 3 个标量（p2p, noise_std, snr）已用 `fit_side_feature_stats('f1')` 与
  `fit_side_feature_stats('f2')[:3]` 数值核对**完全相等**（cache 命中，零额外计算），
  因此下面所有关于这 3 个特征的结果对 F1 同样成立，不需要单独跑。
- Null 对照：同一 calibration pool（`pool_size=50`）内每 unit 的 in-pool spike 计数，
  复用 `_in_pool_spike_prefix` 的**同一** pool 定义，但完全不读 waveform 数组——这是一
  个"不该因电极阻抗/分选阈值漂移"的特征，用作漂移量级的参照系。
- 脚本：`scripts/side_feature_drift_diagnostic.py`；结果：
  `results/side_feature_drift_diagnostic.json`（含完整 provenance：session 列表、
  feature_version、cache key、per-session cache 路径、source fingerprint）。
- 全部 33 个 session（27 train + 6 val）的 `f2` 特征均**cache 命中**，未重新打开任何
  NWB 文件读 waveform；null 特征需要读 `units/spike_times`（不读 waveform），对 27+6
  个 session 各打开一次。全程未构造、未打开任何 test session 的文件路径。

## 2. Session 列表与时间跨度

| Split | session 数 | 日期范围 | unit 数范围（均值） |
|---|---:|---|---|
| Train | 27 | 2013-10-03 – 2015-07-16 | 41–91（59.7） |
| Val | 6 | 2015-11-03 – 2015-11-12 | 38–65（54.2） |
| Test（未访问） | 6 | 2015-11-13 – 2015-12-01 | — |

train 与 val 的 unit 数范围相近（均值 59.7 vs 54.2），不存在旧 53-session 分析中出现
的 4× unit-count regime 跳变。但 train 结束（2015-07-16）到 val 开始（2015-11-03）之
间有约 **110 天（~3.5 个月）无 CO session 的记录空档**——val 是"下一次开始记录"，不是
紧邻 train 尾部的平滑延续，解读趋势斜率时需留意这一点。

## 3. 结果 1：6 个 raw 特征的 per-session 分布

Train / val 各 session **均值**的取值范围（完整 33×6×5 统计量在 JSON `task1_...` 中）：

| 特征 | train 均值范围（across-session 均值） | val 均值范围（across-session 均值） |
|---|---|---|
| p2p | 407.15 – 590.85（515.8） | 451.20 – 555.29（493.0） |
| noise_std | 44.17 – 66.76（53.3） | 47.36 – 58.61（50.2） |
| snr | 7.94 – 10.35（9.27） | 8.96 – 10.87（9.41） |
| pt_width | 7.61 – 10.44（9.31） | 8.40 – 9.74（8.88） |
| pt_ratio | 1.82 – 2.65（2.08） | 1.98 – 2.11（2.04） |
| repol_slope | 45.24 – 67.98（58.4） | 51.42 – 64.71（56.7） |

Val 的 across-session 均值全部落在 train 的取值范围内，没有出现"val 完全在 train 分
布之外"的情况。退化 unit（pool 内 ≤1 个 spike）在全部 1938 个 unit 中只有 **1 个**
（`sub-C_ses-CO-20131220`），不足以扭曲任何 session 的统计量。

## 4. 结果 2：train-only z-scoring 后的 train→val 偏移（关键数字）

Train-only 统计量（27 train sessions，clip 1%/99% 分位数后取 mean/std）：

| 特征 | train mean | train std |
|---|---:|---:|
| p2p | 502.54 | 323.38 |
| noise_std | 52.47 | 19.90 |
| snr | 9.18 | 2.77 |
| pt_width | 9.32 | 3.64 |
| pt_ratio | 2.08 | 0.89 |
| repol_slope | 56.73 | 37.77 |

各 val session 的 **mean z**（该 session 全部 unit 的 z 值取均值）：

| Session | p2p | noise_std | snr | pt_width | pt_ratio | repol_slope |
|---|---:|---:|---:|---:|---:|---:|
| 2015-11-03 | +0.163 | +0.309 | +0.006 | +0.115 | +0.031 | +0.211 |
| 2015-11-04 | −0.159 | −0.256 | −0.039 | −0.129 | −0.038 | −0.126 |
| 2015-11-06 | −0.025 | −0.123 | +0.078 | −0.119 | −0.085 | −0.027 |
| 2015-11-09 | −0.135 | −0.188 | −0.063 | −0.113 | −0.119 | −0.140 |
| 2015-11-10 | −0.138 | −0.207 | −0.080 | −0.227 | −0.006 | −0.092 |
| 2015-11-12 | +0.117 | −0.221 | **+0.609** | −0.251 | −0.106 | +0.162 |
| **6-session 均值** | **−0.030** | **−0.114** | **+0.085** | **−0.121** | **−0.054** | **−0.002** |
| 6-session std | 0.128 | 0.193 | 0.240 | 0.118 | 0.054 | 0.139 |
| 参照：27 train session 间的 mean-z 离散度（std） | 0.152 | 0.214 | 0.197 | 0.210 | 0.263 | 0.152 |

**预注册阈值 `|mean z| > 0.5`**：36 个 (feature × val session) 组合中只有 **1 个**超过
——snr 在 `sub-C_ses-CO-20151112` 上为 +0.609，其余 35 个全部在 ±0.26 以内。6 个特征
在 6 个 val session 上的**均值**没有一个超过 0.13（绝对值）。

更重要的对照：val session 之间 mean-z 的离散度（std=0.05–0.24）并不比 27 个 **train**
session 彼此之间本就存在的离散度（std=0.15–0.26，仅由训练集内部正常的 session-to-
session 抽样波动产生）更大——多数特征上前者其实更小。也就是说，val session 表现出的
"偏离 0"程度，落在 train session 集合自身正常波动的范围内，而不是一个显著跳出该范围
的异常区间。

## 5. 结果 3：历史趋势回归

对每个特征，把 33 个 session（27 train + 6 val）的 mean-z 对日期（参照点
2013-10-03）做线性回归：

| 特征 | slope（z/年） | R² | p | 全程漂移量（z，770 天） |
|---|---:|---:|---:|---:|
| p2p | −0.024 | 0.014 | 0.506 | −0.050 |
| noise_std | **−0.153** | **0.288** | **0.001** | **−0.323** |
| snr | +0.080 | 0.089 | 0.091 | +0.170 |
| pt_width | **+0.132** | **0.249** | **0.003** | **+0.277** |
| pt_ratio | −0.014 | 0.002 | 0.804 | −0.030 |
| repol_slope | −0.022 | 0.012 | 0.537 | −0.046 |

`noise_std` 与 `pt_width` 确有统计显著（p<0.01）、中等强度（R²≈0.25–0.29）的历史趋势，
其余 4 个特征趋势很弱或不显著。但即使是这两个最强的趋势，在完整 ~21 个月的跨度上累积
漂移也只有 0.32 / 0.28 个 z 单位——远低于第 4 节的 0.5 阈值。

Val-only（n=6，跨度仅 9 天）回归的斜率在数值上很夸张（如 noise_std 达 −12.76 z/年），
但这是把 9 天窗口线性外推到"每年"尺度的产物，样本量小、置信区间极宽（p 值全部
>0.05，除 pt_width 的 p=0.050 处于边界），**不应作为真实年化速率解读**；完整数字见
JSON `task3_chronological_trend.*.val_only_n6`。

## 6. 结果 4：Null 对照（不该漂移的特征）

Null 特征定义：与真实特征同一 calibration pool（`pool_size=50`）内、每 unit 的
in-pool spike 计数——直觉上这应该主要反映发放率和 session 时长，不该像波形形状那样
随电极阻抗/分选阈值系统性漂移。

| 指标 | Null（pool spike 计数/unit） | 6 个真实特征中最强的（noise_std / pt_width） |
|---|---:|---:|
| Val 6-session mean z（均值） | +0.091 | −0.114 / −0.121 |
| Val 6-session mean z（std） | 0.087 | 0.193 / 0.118 |
| \|mean z\|>0.5 的 val session 数 | 0/6 | 0/6 |
| 历史趋势 R²（33 session） | **0.273** | 0.288 / 0.249 |
| 历史趋势 p | **0.0018** | 0.001 / 0.003 |
| 全程漂移量（z） | **+0.550** | −0.323 / +0.277 |
| Between-session variance fraction | **0.135** | 0.044 / 0.034 |

**Null 特征的历史趋势强度（R²=0.27, p=0.002）与最强的真实特征相当，全程漂移量
（0.55 z）甚至超过全部 6 个真实特征，between-session 方差占比（13.5%）也是全部 7 个
特征（6 真实 + 1 null）中最高的。** 这不支持"这组波形/幅度标量比一个不该漂移的特征更
容易跨 session 漂移"这个具体主张——按这个诊断的衡量方式，pool 内 spike 计数本身的
session 间非平稳性并不比波形标量小，很可能更大。（次要 null——每 session 的 unit 总
数——历史趋势 R²=0.001，几乎无趋势；但它是 session 级标量，不支持第 7 节的方差分解，
仅供参考，见 JSON `task4_null_comparison.secondary_null_unit_count_per_session`。）

## 7. 结果 5：Within-session vs between-session 方差分解

对每个特征，把全部 1938 个 (session, unit) 的 raw 值按 session 做一元方差分解
（该比例在本诊断使用的"全局 train-only 常数"仿射 z-score 下不变——见 JSON 中
`between_within_variance` 的说明与脚本 docstring 的证明）：

| 特征 | Between-session SS 占比 | ICC（unbalanced） |
|---|---:|---:|
| p2p | 0.024 | 0.008 |
| noise_std | 0.044 | 0.029 |
| snr | 0.042 | 0.027 |
| pt_width | 0.034 | 0.019 |
| pt_ratio | 0.038 | 0.022 |
| repol_slope | 0.023 | 0.006 |
| **Null（spike 计数）** | **0.135** | **0.124** |

6 个真实特征的 between-session 方差占比全部在 **2.3%–4.4%** 之间，即 95% 以上的
unit-level 方差来自同一 session 内部的 unit-to-unit 异质性，而不是 session 身份本
身。Null 特征的 between-session 占比（13.5%）反而是这组数字里最高的。

## 8. 结构性论证：即使漂移为真，也无法解释 F_x < FS_x

这是一个与上面的实测数字独立、来自代码结构本身的论证，已用数值验证：

`load_unit_side_features` 构造 FS1/FS2 的方式是对**同一 session 自己的 unit 维度**做
置换（`perm = generator.permutation(normalized.shape[0])`，`normalized` 是该 session
的 `[n_units, k]` 矩阵），从不跨 session 混合数值。置换一个有限集合不会改变它的均值/
方差/分布——因此 **FS1/FS2 与 F1/F2 在每个 session 上的 mean z、方差、
between/within-session 方差分解完全相同**（已用 `np.allclose` 数值验证：
`permutation_preserves_session_mean = True`，见 JSON
`supplementary_checks.shuffled_control_permutation_invariance`）。

推论：任何"session 级偏置"（不管是不是本文假设的漂移）都会**同等地**作用于
`F_x` 和它的 `FS_x` 对照，因为二者在该 session 上是同一组数值、只是换了 unit 归属。
所以 session 级漂移**在原理上**就不可能解释消融观测到的 `F1 < FS1`、`F2 < FS2`——
无论第 4–7 节测出的漂移量级有多大都不能。它至多只能解释 `F_x < F0` 这部分（`F0`
完全不携带侧特征，天然不受此类偏置影响）；而第 4–7 节测出的 `F_x < F0` 相关漂移量本
身也很小（§4：35/36 组合 \|mean z\|<0.5；§6：不比不该漂移的 null 更大）。

## 9. 探索性交叉检验（非预注册，仅供参考）

用每个 val session 的 6 维 mean-z 向量的 L2 范数，与 `side_feature_ablation_v2` 里该
session 实际测到的 `F2−F0` R² delta 做相关（n=6，功效极低，非预注册，不构成假设检验）：

| Session | Drift L2 | F2−F0 R² delta | F1−F0 R² delta |
|---|---:|---:|---:|
| 2015-11-03 | 0.425 | −0.187 | −0.021 |
| 2015-11-04 | 0.356 | −0.021 | −0.071 |
| 2015-11-06 | 0.210 | −0.024 | −0.060 |
| 2015-11-09 | 0.322 | −0.035 | −0.060 |
| 2015-11-10 | 0.358 | +0.090 | +0.035 |
| 2015-11-12 | **0.730** | **+0.170** | −0.017 |

Pearson r(drift L2, F2−F0 delta) = **+0.52**（p=0.29，n=6，不显著）；drift L2 最大的
session（2015-11-12）反而是 `F2−F0` R² delta**最好**的 session，方向与"漂移越大、
越有害"的预测相反。这个探索性检验没有为漂移假设提供支持。

## 10. 结论：漂移假设成立吗

**不成立 / 未被支持。** 依据：

1. Val session 的 train-only mean-z 偏移绝大多数远低于预注册的 `|z|>0.5` 阈值
   （36 组合中 35 个都 <0.26），且不比 train session 集合内部本来就有的正常波动更大。
2. 6 个真实特征里只有 2 个（noise_std、pt_width）有统计显著的历史趋势，且量级不大
   （全程 <0.33 z）；其余 4 个基本无趋势。
3. 用作参照系的 null 特征（不该因电极/分选漂移的 pool spike 计数）在历史趋势强度、
   全程漂移量、between-session 方差占比上都**不比**、甚至**superior于**真实特征，
   这与"这组波形/幅度标量比正常特征更容易漂移"的具体主张相矛盾。
4. 探索性检验中，漂移量级与消融实际测到的 R² 伤害没有正相关（弱、不显著、且部分方
   向相反）。
5. 更根本的是第 8 节的结构性论证：即便漂移量级很大，按 `load_unit_side_features`
   现在构造 FS1/FS2 的方式（session 内置换），这类 session 级偏置在原理上也不可能
   解释消融里 `F_x < FS_x` 这部分结果——而这恰恰是消融最令人意外的部分。

**这不代表侧特征方案没有问题**，只是排除了"train-only z-score 把 session 级漂移偏置
注入 identity"这一具体机制。真实原因更可能在别处：per-unit 层面的置换/身份混淆本身
（即模型是否真的利用了"哪个 unit 对应哪组标量"这个绑定关系）、`ψ` 输入维度变宽带来
的优化/正则化效应、或本诊断未检验的更细致的分布性质（方差、偏度变化，而非均值偏移）。

## 11. 本诊断做不到 / 未做的事

- **只检验了均值偏移**。没有检验 session 间方差、偏度或协方差结构的变化——如果漂移
  主要体现在二阶矩而非一阶矩，本诊断测不出来。
- **Val 只有 6 个 session**，历史趋势回归和 val-only 相关检验的统计功效都很低；第 9
  节的相关系数不构成假设检验。
- **没有跑任何模型**，因此无法直接验证"更小的漂移是否真的对应更好的 R²"这一因果链
  条，第 9 节只是一个基于已有消融结果的观测性交叉检验。
- **未访问任何 test session 数据**（连允许的 unit-count 读取都没用到，因为本诊断不
  需要）；因此完全没有、也不能对 test session 的漂移情况给出任何结论。
- **未检验 MUA**，只用了 SUA（与 `UNIT_SIDE_FEATURE_ABLATION.md` §6.2 一致）。
- **未改动** `unit_side_features.py`、任何消融结果 JSON，或
  `p3_formal_test_816cdd8b*_receipt.json`。
- 第 9 节的探索性相关分析不是本任务预先声明的 5 项之一，n=6，仅供参考，不能当作独
  立证据。

## 12. 产物

| 产物 | 路径 |
|---|---|
| 诊断脚本 | `sua_exploration/scripts/side_feature_drift_diagnostic.py` |
| 机器可读结果（含完整 provenance） | `sua_exploration/results/side_feature_drift_diagnostic.json` |
| 本文档 | `sua_exploration/docs/SIDE_FEATURE_DRIFT_DIAGNOSTIC.md` |
| 复用的特征计算模块（未修改） | `sua_exploration/mc_maze/unit_side_features.py` |
| Split 来源 | `sua_exploration/results/side_feature_ablation_v2/aggregate.json` |
