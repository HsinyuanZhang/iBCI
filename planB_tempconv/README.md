# Plan B — 时间卷积 student（片上 M2 解码）

> 目标：一个**学习式 depthwise 时间卷积** decoder，比 FALCON flat-ridge(0.11–0.15) 更强、比 BrainDistill 的 CWT+线性注意力更 ASIC-friendly、比 SPINT cross-attn 更小更静态。跨天由**无监督输入侧对齐**(CORAL / SPINT-E) 负责，student 本身固定通道、全静态、无 softmax/LN、量化友好。
> 空位：**没人在 M2 spike 上试过学习式时间卷积 student**（SPINT 无卷积；BrainDistill 明确弃卷积选 CWT，且只测 M1、student 掉点严重 0.44 vs teacher 0.82）。

## 依赖 / 运行方式

复用 `FALCON/m2-research` 的数据加载 / 折分 / CORAL / z-score，**不重造轮子**。用它的 venv 跑（torch 2.12+cu130）：

```bash
cd /path/to/SPINT/planB_tempconv
export M2R_ROOT=/path/to/FALCON/m2-research
VENV="$M2R_ROOT/.venv/bin/python"
"$VENV" _env.py                 # 自检：打印 data_dir / held-in / 6 held-out sessions
"$VENV" models/tcn_student.py   # 自检：TCN student ~4K 参数，输出 (B,2)
```

`_env.py` 已把 m2-research 挂到 path 并 re-export：`load_session / collect_held_in /
load_held_out_sessions / train_frozen_decoder / continuous_folds / eval_r2 /
fit_coral_diag / apply_coral_diag / generate_lagged_matrix`。

## 目录

```
planB_tempconv/
├── README.md              # 本文件：实现路径
├── _env.py                # bootstrap，复用 m2-research 数据/CORAL  ✅已建
├── models/
│   └── tcn_student.py     # DepthwiseTemporalConvStudent + LinearWienerStudent  ✅已建
├── scripts/               # 下面 B1..B5 的脚本，逐步填
└── outputs/{results,figures}/
```

对照数字：SPINT M2=0.26 · ridge/CORAL=0.11–0.15 · BrainDistill IND=30K参数/5.66mW。
验收统一口径：6 held-out sessions 均值 R²(variance_weighted) + 逐 session 分散度 + 参数量 + (量化后) 估算 mW。

---

## 实现路径（B1 → B5，每步一个可跑脚本 + 明确里程碑）

### B1 — 监督上界：TCN student 直接在 M2 上训（不谈跨天，先证结构有效）
**脚本** `scripts/b1_train_tcn_supervised.py`
- 数据：held-in 全量做 5-fold（用 `continuous_folds`），窗口化成 `(B, 96, W)` → 目标 `vel` 最后一帧。
- 训 3 个模型对比：`LinearWienerStudent`(=ridge 结构)、`DepthwiseTemporalConvStudent`(K∈{5,9,15})、(可选) +第二层 depthwise。
- 输出：within-session R²、参数量。
- **里程碑**：depthwise TCN 的 within R² ≥ 线性 Wiener，且参数更少（已知 ~4K vs 9.6K）。若 TCN 打不过线性，说明 M2 时间结构有限→回退线性，止损。

### B2 — CWT vs 学习卷积（on spikes）：复现 BrainDistill 弃卷积的对比，但在 M2 spike 上
**脚本** `scripts/b2_cwt_vs_convlearn.py`
- 三种时间前端喂同一个 mix 头：①固定 CWT 滤波器组(仿 BrainDistill)、②固定平滑核(EMA/高斯)、③**学习 depthwise 卷积**(B1)。
- 同参数预算下比 within/held-out R²。
- **里程碑 / 可发表点**：验证"频带可解释"动机在 spike 上不成立、**学习卷积 > CWT**。这是差异化贡献。

### B3 — 跨天：无监督输入侧对齐 + 冻结 TCN（核心链路）
**脚本** `scripts/b3_tcn_crossday.py`（复用 `15_ridge_fewshot_curve.py` 的评测框架）
- 在 held-in 上训好 TCN → 冻结。held-out 每 session：用 **CORAL-diag**(无监督，`fit_coral_diag`) 对每通道做仿射对齐 → 喂冻结 TCN。
- 扫 calib 预算（4/8/16/32s，同 `15_` 口径），画 few-shot 曲线，叠 SPINT=0.26 / ridge-CORAL=0.11 天花板。
- **里程碑**：跨天 R² 是否超过 flat-ridge+CORAL 的 0.11–0.15。这是"时间卷积到底有没有帮跨天"的判据。

### B4 — 无监督蒸馏拉高上限（接 Plan A/归因结论）
**脚本** `scripts/b4_spint_distill_tcn.py`（**依赖 SPINT M2 checkpoint**）
- SPINT 当无监督 teacher → 在 held-out 无标签数据上出伪标签 → 蒸馏标定 TCN(闭式/少步)。
- 看无监督天花板能否从 0.11–0.15 抬向 SPINT 的 0.26。
- **里程碑**：TCN(蒸馏) 逼近 SPINT，但保持全静态无 attention。⚠ 需要你给 `SPINT-main/logs/train/runs/<run_id>` + epoch。

### B5 — QAT + 功耗核算（tapeout 规格）
**脚本** `scripts/b5_qat_power.py`
- 对 B3/B4 最优 TCN 做 W8A8 QAT（PACT 式可学习 clipping，仿 BrainDistill 2.4）。
- 按 op 数 + 权重 + I/O 估 mW（仿 BrainDistill Appendix A.6.2），对齐 5.66 mW / 15–40 mW 热预算。
- **里程碑**：量化掉点 <3%、估算 mW ≤ BrainDistill 参照、全整数-only。产出 tapeout 规格表。

---

## 依赖关系 & 建议顺序

```
B1(结构有效?) ──► B3(跨天?) ──► B5(tapeout 规格)
   └─► B2(CWT vs conv, 可发表, 可并行)
              B4(蒸馏拉高) 需 SPINT ckpt，可在 B3 后插入
```

先跑 **B1**（不依赖任何 checkpoint，最快证/证伪结构），再 **B3**（复用 `15_` 框架拿跨天判据）。B2 可并行。B4/B5 待 B1/B3 有正信号再上。

## 关键设计约束（贯穿所有步骤）

- student 只用 `Conv1d / Linear / ReLU`，**禁 softmax / LayerNorm / 变长 N**。
- 固定 96 通道；置换不变/跨天全部交给**片外**的对齐（CORAL/SPINT-E）。
- 所有对齐系数无监督、从 calib 协方差算、每天下载（对齐 SPINT 免梯度精神）。
- 全程 6 held-out sessions 报**均值 + 逐 session 分散度**（M2 方差大，别只看均值）。
