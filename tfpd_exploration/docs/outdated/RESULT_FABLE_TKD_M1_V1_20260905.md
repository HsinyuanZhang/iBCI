# RESULT_FABLE_TKD_M1_V1 — TKD 在 M1 上的双 prong 负结果与结案

Date: 2026-09-05
Type: 结案文档（预注册 ADDENDUM-1 分叉规则执行）
Workorder: `tfpd_exploration/docs/WORKORDER_FABLE_TKD_M1_V1_20260905.md`（含 ADDENDUM-1）
Result root: `tfpd_exploration/results/fable_tkd_m1_v1/`（4 receipts，0444+sidecar）
Status: **M1_LINE_CLOSED_BY_PREREGISTERED_FORK_RULE**（pilot 与 K1/K2 未执行——锚 <0.90 在上游阻断；GPU1 全程未占用）

---

## 0. 一句话结论

TKD 在 M1 上两个 prong 各自在自己的门上失败，因果分离干净：**(1) 共享仿射读入结构上无法表达 rSyn3 的生成式读入映射**（该映射经 (WᵀW+λI)⁻¹ 是载体的二次函数，共享 fold 级读入的天花板 corr 0.51–0.86）；**(2) 生成式基底本身太弱**（闭式合成解码器即使仿射校准后 fold-local R² ≤ 0.26，目标 session 0.257，远低于分叉阈 0.35 与 K1 的 0.5374）。M1 线关闭，H1 预检（父 ridge 解码器诊断）承接。

## 1. 证据

| # | 证据 | 数字 | receipt |
|---|---|---|---|
| 1 | A1-M1 锚（模型 vs 生成式合成 ŷ=diag(scale)·D·(WᵀW/n+λI)⁻¹Wᵀ(r−b)） | **min 0.1488 < 0.90 阻断**（target ses-20120924；源 0.233–0.281；温度扫描 γ∈{1..16} 最优 0.149） | `stage0/a1_m1_anchor.json` |
| 2 | 分解：共享 P̄ 映射 vs 逐 session 精确映射 | corr **0.514–0.858**（结构性天花板） | 同上 `decomposition` |
| 3 | 分解：一阶倾斜机构本身 | vs 其 P̄ 目标 0.93–0.99（机构健康，瓶颈在映射形状） | 同上 |
| 4 | 生成式解码器直接评分（fold-local 面，variance_weighted_last_bin_r2） | target **−19,188**（ρ=1）/ **−2,543**（ρ 加权）；披露口径：仿射校准后 target **0.257**、源 ≤0.279 | `stage0/gen_decoder.json` |
| 5 | 失败机制（量化） | (i) 合成重建的是**整流**缩放 EMG 而目标为**带符号**原始 EMG；(ii) 伪逆放大率噪声（W 列范数 9–47） | 同上 |

REF：Z-Fix 0.6374 / S-Fix 0.6203 / S-Acyc 0.6218。

## 2. 判读

1. M1 的"identity→读入"映射是 **Gram 逆（二次）**，不是 M2/cosine 的仿射——TKD 的共享仿射 Φ_k 设计在 M1 结构性不可达，与训练无关。
2. 即便逐 session 精确实现该映射（fork (a) 的上限），基底也只有 ≤0.26：残差学习需独自补 ≥0.28 R² 才到 K1——没有可建的地面。分叉规则 `fork_pass=false` 判定正确。
3. 与 M2 合并的边界结论：**两个快数据集上，"闭式 identity 作为小 decoder 读入"没有可测正表面**——M2 卡在跨 session 迁移（容量/不变性），M1 卡在映射形状与基底强度。

## 3. 幸存资产与披露

- M1-D5（重要）：fold-local 载体 bank 的 2026-09-02 位级 digest 在当前环境**不可复现**（数值库末位漂移，‖carrier‖_F 相对差 3e-16，特征值 1e-13）；本包改用**数值容差 1e-9 定律**并双 digest 记录，规划侧已追认。此披露对后续任何依赖该 bank 位级密封的线都适用。
- 单位换算披露：neural 为 counts/bin 而载体截距为 Hz（×0.02 换算进 μ/σ_pooled/倾斜常数）。
- 工程资产：M1 面逐字复用（import 复用 `m1_emg_rsyn3_fold_local_v1`）、TKD 类参数化扩展（N=64/W=100/B=16/GRU）、蒸馏管线（未启用）。

## 4. 承接

H1 预检（`fable_tkd_h1_preflight_v1`，进行中）：H-C 载体的父模型（源 session 闭式 ridge 解码器）在 5-date LODO 目标面的等权均值——分叉阈 **0.25**（≥ 则 H1 开生成式读入线，< 则快数据集穷尽、688 决策上交用户）。

## 5. 证据路径

- 工单+ADDENDUM-1：`tfpd_exploration/docs/WORKORDER_FABLE_TKD_M1_V1_20260905.md`
- 代码：`tfpd_exploration/src/fable_tkd_m1_v1/{plan,data,anchor,train}.py`、`scripts/run_fable_tkd_m1_v1.py`
- Receipts：`tfpd_exploration/results/fable_tkd_m1_v1/{attempt.json, stage0/a1_m1_anchor.json, stage0/gen_decoder.json, failure.json}`
- M2 对照：`RESULT_FABLE_TKD_M2_V1_20260905.md`
