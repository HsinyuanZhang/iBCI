# 按时间留出最后两个日期的校准消融

主问题改为：仅用较早日期训练与选择模型，活动校准与任务 carrier 能否改善未来日期上的解码。原 leave-one-date-out 扫描已停止，历史结果保留为补充，不作为本轮主线或新模型的初始化。

每个数据集只使用一个固定时间划分。每个 arm 只有一套 source-trained 模型；同一个 source-selected EMA 状态评估两个未来日期。

| 数据集 | Source 日期 | 主目标日期 | 执行 |
| --- | --- | --- | --- |
| M1 | 2012-09-24、09-26 | 2012-09-27、09-28 | Z/B/D 从头训练，24 epochs |
| M2 | 2020-10-19、10-20、10-27、10-28，共 7 sessions | 2020-11-18、11-19 | 复用已完成的 source7 Z/B/D 训练及原 source-only 选择 |
| H1 | 1925-01-01、01-08、01-13、01-15，共 9 sessions | 1925-01-19、01-20，共 4 sessions | Z/B/D 从头训练，32 epochs |

M2 是固定早期 source7 到最后两个可评分日期的迁移。2020-10-30 两条记录未加入 source，保留为补充目标，不能把这一设置写成“所有较早日期训练”。2020-11-24 两条 NWB 各有 33 个 trials，当前 M33 校准预算使用全部 trials，没有剩余 query；因此不作为主目标日期。该条件已从 NWB trial metadata 核对，并写入 `results/chronological_last2_v1/m2_zero_query_date_inventory.json`。

Z 表示无校准，B 表示仅活动校准，D 表示活动校准加任务 carrier。模型臂、网络结构、优化器和校准预算沿用此前主线：M1 为 B3S-family activity 与 rSyn3，M2 为 activity 与 MOVE-T4，H1 为 C2-shaped activity 与 H-C。M1 的独立 residual/carrier refinement pilot 不代替本轮 D，也不继续第四个 LOSO fold。

M1 的 source support/train/validation 分别为 native trial `[0,10)`、`[10,310)`、`[310,end)`；目标为 M10 support 与 `[10,210)` query。H1 使用每个 session 前三个 available eval-valid native trials 作 support，source 的 `[3:-2]` 作训练、最后两个 trials 作验证，目标使用 support 后全部 query；source stride4、target stride1。M2 保持原 M33 support 和 source post-M33 最后 20% whole trials 验证协议。

模型选择只看 source validation。M1 两 source dates 等权；H1 先按 date 平均 session R²，再对四个 source dates 等权；M2 保留原 source7 equal-session 选择结果，复用过程不重新选 epoch。目标的 calibration support 只用于合法的活动记忆或 carrier 估计，目标 query 标签只用于评分与结果复核，target optimizer steps 为零。

主图在每个目标日期内先平均 session R²，再对两个目标日期等权。报告三个 arm 的 source-selected EMA 主终点，以及 M1/M2 e24、H1 e32 的固定终点敏感性结果。活动校准贡献为 `B − Z`，carrier 增量为 `D − B`。单个 seed42 的两个目标日期属于描述性配对观察，不生成显著性或统计非劣结论。

本轮共 9 个模型单元，其中 M2 三项复用，M1/H1 共 6 项重训。旧 M1/H1 LODO checkpoint 的训练 source 包含新目标日期，不能作为新主线 warm start、resume 或 source authority。新入口和结果位于 `scripts/chronological_last2_v1/` 与 `results/chronological_last2_v1/`，具有独立 split ID、source/target roster、数据与代码绑定。

完整结果图、CSV 和论文段落将以这一个时间划分为主线。历史 LODO 与 M1 refinement 数值保持原有实验身份。
