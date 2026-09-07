# RESULT_FABLE_TKD_H1_688_PREFLIGHT_V1 — H1/688 预检与 decoder 创新弧线终局

Date: 2026-09-05
Type: 结案文档（预注册分叉规则执行；H1/688 线未开即关，零 GPU 成本）
Workorders: `WORKORDER_FABLE_TKD_M1_V1`（ADDENDUM-1 分叉纪律延续）；Directive 7/8 记录于 receipts
Result roots: `tfpd_exploration/results/fable_tkd_h1_preflight_v1/`、`tfpd_exploration/results/fable_tkd_688_preflight_v1/`
Status: **DECODER_LINE_EXHAUSTED_ON_ALL_FOUR_DATASETS — 688 全线开否上交用户**

---

## 0. 终局结论

**"identity 作为闭式读入权重"（design1 TKD 论题）在全部四个 FALCON 数据集上没有闭式地板。** T4/rSyn3/H-C 的 identity 内容是真实的（688 上 T4 增量 +0.339 为四集之最，逐 session 全正），但**只有在习得的大 decoder 里才可兑现**——同一份 identity 的任何闭式消费者（经典 PV、仿射校准 PV、逐 session 监督 ridge、池化 ridge、生成式 NMF 复合、父 ridge）全部 ≈0 或灾难性负。

## 1. H1 预检（CPU ~8s 计算，receipts 3/3 sealed）

- 面 честности：逐日期 C1 source authority（`C1_AUTHORITIES` sha 锚，`target_recordings_opened==0` 断言）——未用池化五日期的 hu authority（会破坏 date-LOSO）。
- **父 ridge 解码器**（H-C 载体的源模型，逐 session 首 M4 支持 trial 拟合）：5-date LODO 等权 **−13.98**（逐日期 −9.2~−20.3）；披露：预测-目标 Pearson 0.20–0.33（有弱信号），但**自样本 R² 仅 0.045–0.095**——闭式基底连自己的校准块都拟合不动。C1 的 0.406 全部来自 16.9M 习得 decoder。
- 分叉阈 0.25 → `fork_pass=false`，H1 关闭。

## 2. 688 预检（CPU 499s，receipts 3/3 sealed）

- 面：严格 27/6 manifest、首 30 rewarded 校准 trial（密封主线自用律）、trials[30:] 步进 1 评测窗、train-only 行为标准化、SUA view、逐 session 方差加权 R²——全部只读镜像 established machinery。
- 四个闭式基线（dev-6 等权）：PV raw **−0.0069**（最好）/ PV 仿射校准 −21.2 / 逐 session 校准块 ridge −7.66 / 池化 PV 特征 ridge −32.5。
- **同面密封参照**（receipt 内以 aggregate 自身 `load_artifact` 提取）：SPINT+T4 **0.5750** / SPINT+B0（无 T4）0.2364 / SPINT+TS4 **0.2845** → **T4 内容增量 +0.339**（逐 session 0.41–0.71）。
- 判读：identity 内容最大处，闭式地板依然缺席——校准块内拟合都无法越过块边界（逐 bin 尖峰率特征在该稀疏度下噪声过大）。M2 的墙（ridge −0.026 vs champion 0.295）在 T4 增量最大的数据集上复现。
- 分叉阈 0.15 → `fork_pass=false`，按预注册 report-and-stop。

## 3. 四数据集边界总表

| 数据集 | REF（习得 decoder） | 闭式最好读数 | 失败机制（量化） |
|---|---|---|---|
| M2 | 0.2952 external | PV 0.028 / ridge −0.026 / TKD 全配置 **≤+0.074** | 跨 session 迁移墙：习得 decoder 的不变性不可蒸馏 |
| M1 | Z-Fix 0.6374 fold-local | 生成式复合仿射校准 **0.257** | 映射是载体二次函数（Gram 逆），仿射读入结构不可达 + 基底弱（整流 vs 带符号） |
| H1 | C1 0.406 LODO | 父 ridge **−13.98**（自样本 0.045–0.095） | 闭式基底连校准块都不拟合 |
| 688 | SPINT+T4 0.5750（B0 0.2364） | PV **−0.007** | T4 增量 +0.339 真实但只在习得 decoder 里可兑现 |

## 4. 幸存资产（本弧线总计）

1. **成本主张**：TKD 流式 0.41M MAC/bin vs champion 84M/bin（205×/137× 双口径）——架构级事实。
2. **A1 锚方法论**（M2 达 0.9975）+ 三个数据集的锚分析（含 M1 的"映射形状"分解技术）。
3. **Δ_ref 数字族**：M2 0.0878（champion T4 内容增量）、688 0.339（同面复密封）——论文 carrier-gap 表的直接素材。
4. **四负结果本身**：每条线一个量化机制、全部 sealed receipts——论文讨论节"何时 identity 可被闭式消费"的边界证据。
5. 工程：TKD 类（参数化跨数据集）、蒸馏管线、诊断先行纪律（gen-decoder/parent-decoder/闭式地板三连预检模板）。

## 5. 上交用户的决策

剩余可选路线（均涉 688 = 用户计算优先级最后）：
(a) **关闭 decoder 创新线**，整合四负结果入论文讨论节；
(b) **688 学习型小 decoder 线**（蒸馏 TKD-GRU @688，先验低——预检表明 T4 增量在大 decoder 机制里，日级 GPU）；
(c) **review §6 decoder 无关性 2×2 @688**（DeepSets/POYO 对照，测 T4 增量 vs 活动 identity 增量的 decoder 无关性——服务论文设计点 1，~1 天/decoder，不要求小 decoder 达 champion 水平）。

## 6. 证据路径

- H1：`tfpd_exploration/results/fable_tkd_h1_preflight_v1/{attempt,terminal,diagnostics}.json`
- 688：`tfpd_exploration/results/fable_tkd_688_preflight_v1/{attempt,failure,terminal}.json`
- 前序：`RESULT_FABLE_TKD_M2_V1_20260905.md`、`RESULT_FABLE_TKD_M1_V1_20260905.md`
