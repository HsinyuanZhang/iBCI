# M2 carrier support stability：描述性汇总

本文件记录已完成的 post-hoc、receipt-bound 描述性汇总。它重读 `results/diagnostics_v1/m2_carrier_support_stability_full_v1/report.json` 的全部 440 条既有记录，SHA-256 为 `573711d9b144b00387e3de07f60c6c89990f81d1681d2a550eb3e394c4443bac`；没有重算 carrier、加载或评分 decoder、打开 NWB/EvalAI，或启动训练。

完成 summary 是 `results/diagnostics_v1/m2_carrier_support_stability_descriptive_v2/summary.json`，SHA-256 `e883763a13f6b8cdfb3938d471e74fe2f9649d4d3481dd48e361646cd0d5b342`；figure data SHA-256 `12f92f2972d896c2b1e1fc95792ffcc4a67af71e49d29635eb5e2e127a44a56d`；validation SHA-256 `2aafb808e9b0a8d36e0436e57aadffc54b2e23fac3b34969eeda5cccabea2801`。已核视图见 [PNG](../results/diagnostics_v1/m2_carrier_support_stability_descriptive_v2/support_stability_ext4.png) 和 [PDF](../results/diagnostics_v1/m2_carrier_support_stability_descriptive_v2/support_stability_ext4.pdf)。

## 主 EXT4 四-session surface

对每个 budget，先在每个 session 内只使用有效 resample 计算 median，再把四个 session median 等权平均。group 数字只在四个 session 都至少有一个有效 resample 时定义；它们不是对 40 个 record 的逐项均值，也不应转为 wall-clock calibration cost。

| Support budget | relative Frobenius vs M33 | cosine vs M33 | 有效记录 |
| --- | ---: | ---: | ---: |
| M8 | 1.06196196377 | 0.7743412 | 26/40 |
| M16 | 0.46468265727 | 0.9076217 | 40/40 |
| M25 | 0.22864267603 | 0.9758220 | 40/40 |

M8 的 14 个无效 resample 全部保留：其中 8 个因不足 3 个 directional trials，6 个因 direction-design rank deficiency。M33 的 canonical 重复是 audit anchor，不是 10 个独立重复，也不进入 independent-resample interpretation。

## 独立的 source7 补充面

七个 source-train session 是独立补充面，不能与主 EXT4 的四个 session 合并成 11-session group。其完整 per-budget/per-session 数值、有效性和 failure records 保留在 summary 中；本文件不将两种 surface 合并、重新加权或比较为一个总体。

这是 descriptive stability evidence：不报告 CI、p 值、training-seed 不确定性、decoder R²、相关性或因果结论。它也不定义 runtime、calibration cost、official test 或 submission performance。
