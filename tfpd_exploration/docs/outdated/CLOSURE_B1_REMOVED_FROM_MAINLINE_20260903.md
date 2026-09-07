# Closure — B1 Removed from Mainline

日期：2026-09-03  
数据集：FALCON B1 vocal / DANDI `001046`  
状态：**CLOSED。B1 从本次 T4/activity-memory 工作中完全移除。取代
`DECISION_B1_SHARED_SPINT_ALIGNMENT_HOLD_20260903.md` 的 HOLD 状态；§8 network-alignment cell 不再保留。**

## 1. 决定

1. B1 不进入本次工作的主结果、扩展结果或机制结果。
2. 不再运行任何 B1 source/GPU work order、TARM/SFCJ 训练、ARTP-Anchored Shared-SPINT cell、
   Docker 构建或 EvalAI 提交。
3. 现有 B1 结果根（`b1_sfcj_v1 / b1_tarm_v1 / b1_tarm_frame_v1 / b1_artp_v1 / b1_artp_v2`）原样保留，
   仅供只读引用。
4. 论文中 B1 只能作为"模板主导、few-shot 标签价值已被模板穷尽"的数据集描述出现，
   不作为共享 SPINT 的跨数据集验证，也不引用 ARTP-P 的 +18% 数字。

## 2. 勘误：source screen 含模板自检索

三个日期的 `held-in-minival` 两条 trial 与 `held-in-calib` 第 0/1 条**逐字节相同**（`tx` 与
`vocalizations` 均相同，2026-09-03 直接读取 NWB 复核）。calib 第 0/1/2 条正是 ARTP-P 的 M3 模板；
source screen 采用 `calib 4..N -> minival 1..2` 的 query 约定，因此每个日期有 2 条 query 就是模板本身，
ARTP-P 在这些 trial 上 MSE 为 `1e-7 ~ 1e-10`。

以 `results/b1_artp_v2/source_screen.json`
（SHA-256 `f8c7fd97b3dbfe1f93221441a3b34c695f83668b4346f4a35a5bddd6cdfd343a`）内的 per-trial MSE，
去掉每日期末尾两条 minival 后重算：

| 日期 | query | TPL-M3 | ARTP-P | 增益 | vs cyclic | vs neural-free |
|---|---:|---:|---:|---:|---:|---:|
| 20210626 | 8 | 3.4780e-4 | 3.4038e-4 | +2.13% | **−3.13e-5（劣）** | **−2.26e-5（劣）** |
| 20210627 | 26 | 4.0208e-4 | 3.9451e-4 | +1.88% | +4.23e-5 | +1.30e-5 |
| 20210628 | 5 | 5.2828e-4 | 5.2688e-4 | +0.27% | +7.43e-5 | +3.29e-5 |

日期等权增益 **+5.46e-6（+1.28%）**，原报告 `+7.46e-5（18.0%）`；约 93% 的报告增益来自 6 条重复
trial。修正后 ARTP-P 在 20210626 上输给自己的两个机制对照，`gain_vs_tpl / correct_vs_cyclic /
correct_vs_neural_free` 三门不再同时成立。

连带修正：

- `RESULT_B1_ACTIVITY_RELIABILITY_TEMPLATE_PROFILING_V2_20260903.md` 的主数字、bootstrap 区间与
  "content-addressable key" 结论以本文为准；
- TARM 两次 pilot 的 10 条 in-range 验证流同样含这 2 条重复 trial；
- `results/b1_sfcj_v1/BUILD_REPORT.md` 称 TPL-M3-BEST 在每个日期都低于 DR-158-ML，fold 1 不成立
  （DR-158-ML full `3.505e-4` < TPL-M3 median `3.841e-4`）；DR per-trial 未保存，无法去重复核。

## 3. 为什么关闭，而不是继续修

B1 的 gap 结构与 M1/M2/H1 相反：

| | M1/M2/H1 | B1 |
|---|---|---|
| unit 身份 | 跨 session 漂移，是主要 gap | 85 通道索引稳定，3 条 trial 即饱和 |
| 行为方差 | 逐 trial 变化大 | 同日期发声刻板，3 条标签模板接近最优 |
| 剩余 neural 可解释量 | 大 | 去重后约 1%，且不稳 |
| 跨日期映射 | identity 对齐后可复用 | neural→song 映射随日期漂移，2 个训练日期学不到迁移 |

证据：

- SFC9 encoding R² 最佳 `0.036`：tuning profile 近乎无内容；
- DR-158-ML（595 维、带 M3 标签、M3-LOO 选 λ）3 个日期仅 1 个胜 neural-free 模板；
- framewise TARM 训练 surrogate `3.97e-4 -> 2.90e-4`，验证全程劣于模板并从 epoch 1 起恶化；
- whole-trial TARM 四臂彼此差 `1e-9`，GROWING/FIXED3 差 `1e-9`；ARTP-P growing profile 差 `2e-7`：
  activity memory 惰性；
- ARTP-P 去重后 +1.28%。

SPINT 的 activity-identity 与 carrier 都作用于"身份"这一侧；B1 的 gap 在 neural→behavior 映射漂移，
且行为侧几乎没有方差可解。这不是 no-backprop 造成的：允许在 target M3 上反传也只会再记一遍模板。
继续扫 rank、pooling、gate、memory capacity 或训练配方不会改变这个结构。

## 4. 生效边界

自本文起：

- 不新增任何 B1 文档、代码、结果目录；
- `DECISION_B1_SHARED_SPINT_ALIGNMENT_HOLD_20260903.md` §8–§9 作废，仅作历史记录；
- 已构建的 ARTP-P payload（SHA `7c73e803…`）不提交、不删除；
- 若未来任何工作重新使用 B1 minival，必须先剔除与 calib 重复的 trial，否则所有 few-shot 结果无效。

## 5. 允许的论文表述

> On FALCON B1, the three released calibration spectrograms per day form a near-optimal
> date-local template; labelled closed-form readouts, activity-identity networks and
> template re-weighting all remain within ~1% of this template on held-in dates. Because
> channel identity is stable and behaviour is stereotyped, B1 does not exercise the
> identity-drift problem that the shared SPINT pathway addresses, and we exclude it from
> cross-dataset validation.

禁止：任何形式的 "our method improves B1 by 18%"、"SFC9 caused the ARTP-P gain"、
"calibration activity is a content-addressable key on B1"。
