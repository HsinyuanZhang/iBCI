# DESIGN — 688 本地基准实验协议 V1（无 FALCON 官方面）

> **ADDENDUM-TWO-STAGE（2026-09-09 用户指令：先窄时间跨度后完整）**：实验拆两段——
> **EXP-1 窄窗（M2 同级跨度）**：train = 2015-06-29→07-10 的 9 个 session（12 天，对照 M2 的 10 天 7 session）；**exam = 0713/0714/0715/0716 四个 session**（训练结束后 3–6 天，镜像 M2 的 ext4 "隔天考卷"结构）。块选 2015-06/07 而非 2015-03 的原因：只有它自带"紧随其后的考卷 session"。数据量 ≈ 全协议的 1/3（~9.4k upd/ep，仍为 M2 的 ~3×）。回答：**同 regime 下 RIFT+对齐 carrier 是否复现 M2 量级**（T4 双门 + 0.5–0.6 带锚）。
> **EXP-2 完整**：原 27/6 协议不动（train 跨 2 年，val = 2015-11 的 6 session，距最后 train 约 4 个月）。
> **两段差 = 长期漂移代价的干净测量**（EXP-2 val − EXP-1 exam，同架构同配方仅时间跨度不同）。披露：EXP-1 的 exam session ⊂ EXP-2 的 train 集——两协议独立、不得跨协议比面。融合臂 t4_concat 在 EXP-1 跑（regime 干净，消融解释力最强）；EXP-2 可选。

- 创建：2026-09-09，GLM5.3（规划/文档）。上位：btransform_unified_v2 的 [688 RIFT 执行契约](EXECUTION_688_RIFT_V1_20260907.md)（已冻结数据/模型合同，本文件补"实验设计与对照体系"层）。
- 背景约束（用户）：**688 无 FALCON 官方数据**——数据划分、对照实验、验收门全部本地定义；任何结论标注 local-benchmark 口径。

## 0. 之前的分析盘点（本次找到的三层）

| 层 | 文档 | 关键结论 |
|---|---|---|
| **carrier 对齐** | `sua_exploration/docs/DESIGN_DANDI_000688_SPARSE_EVENT_T4_FILM_V1_20260904.md`（+STAGE0_MASK_AUTHORITY） | 688 carrier = `[a_R, c_R, m_R, δ_b]`，前三列 = **R700 运动窗 T4 列**（与 M2 的 `[a,c,m]` 同估计器族），第四列 δ_b = R700 baseline − H300 hold baseline；R700 从候选时长中选出后冻结（披露为选择证据，禁止再试第三时长）；列序冻结 `[a_R,c_R,m_R,δ_b]`；split-half 信度 0.87–0.94 |
| **方法论先例** | `sua_exploration/docs/PSEUDO_MUA_T4_BRIDGE_48H.md`（COMPLETE） | 27/6/6 严格划分下 F0/T4/TS4 三臂 + 3 种子协议；T4 增益在 pooling 后保留（+0.3177 vs F0）；"formal test 一律不加载"的隔离纪律；score window = e5–12 平均 |
| **数字锚** | FABLE TKD 688 预检（已封存）+ 本次核验 | 封存参照：SPINT+T4 **0.5750** / B0 0.2364 / TS4 0.2845（dev-6 面）→ **T4 增量 +0.339 是 688 上最大的单成分效应** |

## 1. carrier 的 688↔M2 对齐表（实验文档必附）

| 维度 | M2 (MOVE-T4) | 688 (sparse-event directional) | 对齐状态 |
|---|---|---|---|
| 估计器 | 对 `[1,cosθ,sinθ]` 最小二乘 | 同（trig LSQ） | **同族** |
| 前三列 | `[a, c, m=√(a²+c²)]` | `[a_R, c_R, m_R]` | **逐列对应** |
| 第四列 | 绝对 baseline rate | **δ_b = b_R700 − b_H300** | 语义分叉：688 用运动−保持差分隔离运动相关基线漂移（H300 是 M2 没有的相位） |
| 率原语 | 未插值 bin 计数 / 长度 | searchsorted 半开窗计数 / 时长 | 同族不同原语（合同级披露） |
| 支持预算 | M33（全部校准 trial） | **M10**（carrier）/ M30（E0 identity） | 分预算 |
| 归一化 | 7 个 held-in session z-score | **仅 train sessions**（val/test 禁参与） | 688 纪律更严 |
| 置换对照 | —（后续 audit 加过） | TS4 = session 内 channel↔row 置换 | 688 原生带对照 |

## 2. 数据划分（唯一合法切分，SHA 冻结）

`subc_co_27_6_strict_train_val_manifest.json`（SHA `4607e979…`）：**27 train / 6 val / 6 formal-test**。
- **方法选择/开发面 = 6 个 val session**（承担其他任务里 ext6 的角色——"可见官方面"的本地等价物）；
- **formal test 6 session 本地存在但封存**：任何 arm 在 val 面定案前不得加载其 spike/behavior/trial（pseudo-MUA 先例的隔离纪律）；只在全部预注册判读完成后**一次性**解封计分，此后该划分作废（防重复消费）。

## 3. 对照臂体系（本地定义的"官方等价物"）

主臂（RIFT 架构，per 执行合同：50-bin 窗、B3S E0 [N,50]、M2-对齐 carrier、Nmax padding+mask、12ep seed42、e8–e11 权重平均不选点）：

| 臂 | carrier | 回答 |
|---|---|---|
| **T4** | 真值 [a_R,c_R,m_R,δ_b] | 主效应 |
| **F0** | 无 carrier | T4 的总增量 |
| **TS4** | session 内置换 | 增量是否依赖正确 channel↔carrier 对应（内容 vs 容量） |
| **T4-concat**（融合消融，2026-09-09 增补） | 真值原样（与 T4 同字节） | 融合方式消融：concat 前端下主效应是否变化（不进验收门，只报读数） |
| E0-zero | T4 + E0 置零 | identity 贡献分解（可选，M1 RIFT 的 25-condition 方法移植） |
| support-resample | M5/M8/M10 各 n 次重采样 | carrier 支持稳定性（可选，同 M1 协议） |

**融合轴（与 carrier 臂正交，用户裁定 2026-09-09 原文："M1/H1 上 add 是必须的（维度问题）；M2/688 上 concat 和 add（joint）都要试"）**：T4/F0/TS4 全部保持 settled `proj_add` 前端（P: 50→16 加到 local，token_in=20）；T4-concat = 真 carrier 走 matched concat 前端（v2 `concat_model.py`：`[local16 | E0_50 | carrier4]`，token_in=70，init 为 proj_add 的函数保持折叠，几何/种子/数据合同全同）。动因：M2 的先验是 concat 略优（0.4501 vs 0.4016；RIFT 线 concat 0.3901 > joint 变体），688 与 M2 同宽 70，此臂直接检验该先验是否迁移。M1/H1 不开 concat 臂——E0 原始宽度（700/100）无法不经投影加到 16 个 local 通道上，add（proj_add）在该线是维度上的必须选择。

**M2 侧备忘**（不改 M2 代码）：M2 的 concat vs add 并测在 RIFT 主线已有部分证据（concat 0.3901 vs D42 joint 0.3478），完整双臂补测列入 RIFT M2 待办。

**验收门（预注册，val 面 equal-session mean R²）**：
1. T4 − F0 ≥ **+0.03** 且 T4 − TS4 ≥ +0.03（增量成立的双门——比照 pseudo-MUA 先例里两组比较全正的组级判据）；
2. RIFT+T4 对封存参照 **SPINT+T4 0.5750**（dev-6 同域）非劣（≥ −0.01）为"架构可迁移"判据；超过则为新冠军；
3. 过门后 formal test 一次性解封：报 T4/F0/TS4 三臂终局。

## 4. 与外部队友线的边界

`btransform_unified_v1/handoff/dandi688_*`（v1/v2）已交外部队友（EVALAI_RECOMMENDATIONS 标 EXTERNAL_PENDING_INTERFACE）。本协议的本地基准与队友线独立：**不共用选点、不共用 test 消费记录**；若将来对表，双方 split/scoring 口径必须显式并列（688 RIFT 契约开头同款警示）。

## 5. 执行顺序（待用户批准后派单）

P0 数据/合同复核（prepared_cache_v2 已在盘；t4 与 t4_concat 的 CPU preflight 均已 PASSED 存 receipt）→ P1 四臂训练（T4/F0/TS4 主臂 + T4-concat 融合消融臂，GPU 一张，各 ~1h）→ P2 val 面判读（门 1/2；t4_concat 只报读数不进门）→（过门）P3 formal test 一次性解封三主臂 → P4 可选消融（E0-zero / support-resample）。receipt 全 seal；不改 688 RIFT 契约的任何冻结项。
