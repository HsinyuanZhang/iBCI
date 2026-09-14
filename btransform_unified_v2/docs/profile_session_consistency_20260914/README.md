# 跨 session functional profile：一致性与变化（2026-09-14）

已经用实际 support profile 生成 M2 和 DANDI PMUA 的统计与图。数据支持同一电极位置的 profile 保留部分跨 session 结构，同时存在日期间变化；目前不支持将所有变化概括为“轻微漂移”。这些是 profile 的描述性指标，不是解码 R²，也不代表追踪到了同一个神经元。

## 主要数字

各分析均使用冻结的 source-derived profile coordinates；row correspondence 依数据集而异（M2/PMUA 为固定通道/电极，M1/H1 为固定工程输入列）。将适用的四维 profile 展平，计算 Pearson r；同时计算标准化 profile 的 RMSE，以免只看相关而忽略幅度变化。以下均为各比较对象的中位数，方括号为 IQR，不是置信区间。

| 分析对象 | 比较数 | Profile r，中位数 [IQR] | RMSE，fixed source-normalized units |
|---|---:|---:|---:|
| M1：7 个 recording，跨度 30 天 | 21 对 | 0.50 [0.37, 0.54] | 1.00 |
| M2：全部 13 个 session，跨度 36 天 | 78 对 | 0.57 [0.45, 0.67] | 0.88 |
| H1：27 个 recording、13 个 dataset 日期 | 351 对 | 0.31 [0.24, 0.37] | 1.10 |
| M2：仅不同日期 | 73 对 | 0.57 [0.45, 0.65] | 0.89 |
| M2：同日不同 run | 5 对 | 0.90 [0.89, 0.94] | 0.45 |
| DANDI PMUA：24 个日期，跨度 248 天 | 276 对 | 0.47 [0.35, 0.77] | 1.12 |
| DANDI PMUA：每日期奇偶支持试次 split-half | 24 个日期 | 0.90 [0.86, 0.93] | 0.47 |
| DANDI SUA：profile distribution kernel similarity（非 Pearson r） | 276 对 | S=0.99 [见数据] | 不适用 |

各数据集使用各自的 source normalizer 和 profile estimator，观察跨度也不同，因此表中数字不作为数据集之间稳定性高低的公平排名。M2 同日不同 run 与 DANDI 奇偶试次 split-half 是不同的重复性检查，不能互称等价测试或直接扣除其差值作为“真实漂移”。

按每对 session 将一侧电极的完整四维行随机置乱 200 次后，M2 和 DANDI 的 pair-level 平均 null 相关的中位数均约为 **0.00**。实际电极对应的相关高于该参照，说明相关不是任意通道配对都会得到的结果。不同 pair 共享 session，这里不计算将所有 pair 当作独立样本的 p 值。

M2 同日不同 run 的 5 对 r 中位数为 **0.90**（RMSE **0.45**）；其余不同日期的 73 对为 **0.57**（RMSE **0.89**）。H1 的 15 个同 dataset-date 不同 recording pair r 中位数为 **0.47**（RMSE **0.97**）；336 个不同 dataset-date pair 为 **0.30**（RMSE **1.11**）。这些 H1 日期是数据集提供的 session 标签，不能表述为已验证的现实采集年份。

M2 四个分量各自跨电极相关的中位数为：cosine coefficient **0.44**、sine coefficient **0.56**、modulation magnitude **0.58**、baseline **0.77**。因此不应只用相对稳定的 baseline 概括完整 tuning profile。DANDI 对应为 **0.58 / 0.52 / 0.49 / 0.57**。

## 图与可复现数据

### M1：官方 Full 的冻结 muscle-response SVD4 profile

直接使用 official Full 582413 的 `carrier_pack.npz` 中七个 `64×4` T arrays；四个 held-in source recording 与三个 held-out recording 使用同一个 source-only SVD basis 和 normalizer。所有 21 对 recording 以固定输入列 0–63 比较，未进行逐 session 重拟合、旋转或对齐。

- [完整 PDF](m1/m1_profile_session_consistency.pdf)、[热图 PNG](m1/m1_profile_session_consistency.png)
- [完整精度统计](m1/summary.json)、[全部 pair](m1/pairs.csv)、[所有 profile 值](m1/profile_rows.csv)
- [脚本和输入边界](m1/README.md)

这里的固定列是部署时的工程对应条件，不能作为跨 session 生物单元或物理电极身份的验证。

### H1：官方 Full 的冻结 signed-state14 profile

直接使用 official Full 582241 sealed bank 的 27 个 `T/<tag>` arrays；每个均为 `176×4`，在本分析前已由相同的 source plan 投影并标准化。27 个 recording（13 日）均保留，same-date recording 仍作为独立 pair；共 351 对。

- [完整 PDF](h1/h1_profile_session_consistency.pdf)、[热图 PNG](h1/h1_profile_session_consistency.png)
- [完整精度统计](h1/summary.json)、[全部 pair](h1/pairs.csv)、[所有 profile 值](h1/profile_rows.csv)
- [脚本和输入边界](h1/README.md)

固定列 0–175 是部署工程合同，不能证明跨 session 的生物 unit 或物理电极身份。

### M2：实际部署的 profile

直接使用 Full 582481 payload 中的 `bank_by_dataset_tag[tag]["T"]`，13 个 session 各为 `96×4`，所有 mask 均有效。它们与封存 bank source 逐项一致。没有在分析时重新按 session 拟合 normalizer、旋转或对齐 profile。

- [完整三页 PDF](m2/m2_move_t4_profile_session_consistency.pdf)
- [跨 session 相似度热图 PNG](m2/m2_move_t4_profile_session_consistency.png)
- [完整精度统计](m2/summary.json)、[全部 pair 指标](m2/pairs.csv)、[所有 profile 值](m2/profile_rows.csv)
- [脚本和输入追溯](m2/README.md)

热图保留全部 session 和全部通道。矩阵对角线为自相关 1；它不是重复性证据。同日不同 run 的高相关来自非对角元素。日期分组中的 `adjacent date` 严格表示恰好间隔一个日历日，不表示任意两个相邻采集日期。当前结果也不支持将跨日差异画成单调增加的漂移过程。

### DANDI PMUA：完整 source/development 日期

使用 18 个 source 和 6 个 development 日期的现有 support cache，调用正式 MOVE-T4 估计器；采用 PMUA Full 的固定 source normalizer。没有读取 final cache 或 query 标签。各日期观察到的电极不同，276 对日期按电极 ID 求交集；最少共有 14 个电极，全部满足至少 8 个共同电极的分析条件。

- [综合 PDF](dandi_pmua/dandi_pmua_profile_consistency.pdf)
- [综合 PNG](dandi_pmua/dandi_pmua_profile_consistency.png)
- [完整精度统计](dandi_pmua/summary.json)、[全部日期对](dandi_pmua/pairs.csv)、[奇偶试次重复性](dandi_pmua/split_half.csv)
- [脚本和输入追溯](dandi_pmua/README.md)

日历日期已显式解析并核对：2015-03-09 到 2015-11-12 为 248 天。图中短间隔与跨月间隔都保留，不通过挑选日期来获得高相关。

## DANDI SUA：无 unit identity 的 profile-distribution 比较

DANDI SUA 的排序 unit columns 没有可信跨 session identity，因此不做 rowwise trajectory 或 unit-matched null。改以每个 session 在固定 source normalizer 下的 4D unit-profile **分布**计算 Gaussian-kernel mean-embedding cosine similarity `S`。带宽固定为 held-in source rows 的中位非零距离，276 全部日期对都保留；S 中位数为 **0.99**，奇偶 support-trial split-half 为 **1.00**。

- [完整 PDF](dandi_sua/profile_consistency.pdf)、[PNG](dandi_sua/profile_consistency.png)
- [全部 pair](dandi_sua/pairs.csv)、[完整 profile rows](dandi_sua/rows.csv)、[split-half](dandi_sua/split_half.csv)
- [方法、限制及可重算包](dandi_sua/README.md)

S 是 bandwidth-dependent distribution similarity，**不是**其它行匹配分析所用的 Pearson r，不能横向比较数值。接近 1 不代表同一 unit 跨日稳定；split-half 只提供估计噪声背景，不是漂移校正。

## 适合论文使用的解释

可用的结论是：**functional profiles 在固定电极位置上保留了跨 session 的部分结构，同日重复性较高，跨日期则存在异质性变化。** 这为按 session 估计 profile 提供了描述性背景，但不能单独证明校准提高了解码性能；后者仍由解码对照实验回答。

目前不应声称：同一神经元被跨日追踪；所有日期只发生轻微漂移；变化主要由某种特定生物学机制引起；或 profile 相似度等于解码泛化能力。M2 未做 exact T4 split-half，因为现有缓存没有原始逐 trial T4 充分统计量；没有用插值后的 activity encoder 输入冒充该统计量。

论文展示建议选择一张跨 session 相似度热图，并配一张相似度或 RMSE 随实际日历间隔的图。正文同时给出实际相关与置乱参照，并说明电极对应及固定标准化方式。当前产物是完整可审阅分析；尚未替换论文现有图或改变正式主结果。

