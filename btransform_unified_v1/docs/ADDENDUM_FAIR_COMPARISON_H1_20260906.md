# ADDENDUM — FAIR COMPARISON PROTOCOL（H1 留出日公平对比，2026-09-06 用户裁定）

背景：用户对比表（Original 0.961 / C2 0.888 均为**训练重叠面**；F250 留出日 0.552 为**干净面**）裁定：新训练结果必须有完全公平的对比。公平 = 同一面 + 同等训练暴露；不同暴露必须显式标注，禁止直接比。

## 暴露度分级（当前所有 H1 数字的真实状态）

| 面 | 数字 | 暴露 |
|---|---|---|
| 20,325 完整流 | Original 0.961 / C2 0.888 | **训练重叠**（含 01-20 在内的全部 13 session） |
| sel2908 | F250 0.570 | **大部分重叠**（11/13 训练日） |
| 留出日 exam 2,952（1925-01-20） | F250 0.552 | **干净**（LODO，未训练该日） |
| 官方 HO | C2 0.376 / Original 0.262 | **干净**（隐测 session） |

## 两级公平化

**Level-1（便宜，本轮立即执行）：同面不同暴露，显式标注。**
评测 Original（as-shipped）与 C2（e15）于 **1925-01-20 exam 2,952 窗**（与 F250 逐坐标同 (session,end)）。两者对该日有训练/校准暴露（须从 payload calib 键核验并记录），故结果标注 "exposed"。判读规则：
- 若 exposed-Original 在 exam 面仍 ≫0.552（如 ≥0.85）：差距跨暴露依然真实，F250 的 0.552 是"干净条件下仍与重叠冠军有距离"的诚实读数；
- 若 exposed-Original 在 exam 面大幅缩水（<0.7）：0.961 的很大部分是重叠红利，公平对比的天平向 F250 移动。
任一结果都改变矩阵判读，故 Level-1 是矩阵后续 cell 的前置。

**Level-2（金标准，需用户授权 SPINT 重训）：C2-LOSO。**
重训 C2（SPINT 系，同 prefix-cycle 配方）**排除 01-20**，与 F250 构成训练暴露完全匹配的同面对。Original as-shipped 不可重训（出厂物），其公平对应物只能是官方 HO 隐测（C2 0.376 / Original 0.262 已在，F250 提交需另行授权）。

## 执行

- Level-1 评测（推理 only，CPU/空卡，坐标逐位 join）：`results/h1_holdout_exposed_eval_v1/`，Original + C2 各一份 per-session 报告 + 暴露核验（calib 键含 01-20 与否）。
- Level-2（待授权）：C2-LOSO 重训。
- 本矩阵所有对外表述沿用六行表 + 暴露列；F250 的 0.552 只与 (a) Level-1 exposed 数字、(b) Level-2 matched 数字、(c) 官方 HO 三者之一比较。
