# 实验一工单 V2：跨 session 模型的 E0 使用依赖与直接 carrier 路径诊断

日期：2026-09-07。状态：**审核意见已收口／文档定稿／待执行授权**。按用户指令完成修订，不再安排一轮文档审核；本文不表示实验已经开跑。

## 0. 定位、意义与三个实验的分工

核心问题：**冻结的解码器是否使用 E0，是否依赖正确的神经单元—E0 对应？**
意义：用零训练的输入干预排查校准信息是否真正进入预测，为论文的 calibration-conditioned 主线提供机制证据；允许发现冗余或负作用。

三份独立工单：

| 实验 | 科学问题 | 不能替代的证据 |
|---|---|---|
| 本实验 EXP1 | 冻结模型对 E0／直接 c 的使用依赖（reliance） | 不能证明移除后重训也会变差 |
| [EXP2：688 配对效用](WORKORDER_EXP2_688_PAIRED_UTILITY_V2_20260907.md) | 重新训练后 E0／显式 carrier 的条件增益（utility） | 不能把 688 结论外推为 M1/H1/M2 的 utility |
| [EXP3：流式结构](WORKORDER_EXP3_STREAMING_STRUCTURE_V2_20260907.md) | depth／历史卷积的质量—运行成本取舍 | 不能证明 E0 必要或 B 架构优于 SPINT |

**全局最高优先级：先锁定数据暴露、bank、观测边界与 oracle；不以先跑出分数代替协议闭合。**
执行顺序建议：EXP1 准备与 EXP2 CPU 审计优先；EXP3 复用既有深度实验线的产物，不重复启动或抢占它。三个实验独立授权、独立失败、独立交付。M1 D4缺失只阻断EXP1-M1主评分和EXP3分支，不阻断其他任务的准备工作。

## 1. 本实验最高优先级与执行边界

P0：把每个任务的候选 checkpoint、EMA/RAW、训练 roster、开发选点面、bank 和 query 坐标绑定成不可变 manifest。**简称如“P16 e24”不构成可执行配置。**

- 负责人只拥有新增 `scripts/exp1_e0_reliance_v2/`、`tests/exp1_e0_reliance_v2/` 与结果目录中的实现；现有公共代码只读。具体文件名可由执行者选择并列入 manifest。
- 你不是工作区唯一执行者：不得回滚他人修改、停止进程、抢占设备、覆盖原 checkpoint/bank/缓存。
- 本文不授权训练、EvalAI、镜像推送、sealed-test、Overleaf 或 git push。获本实验执行授权后仍只允许合法本地推理与独立产物写入。
- 一个任务 P0 不通过不阻止其他任务交付；必须报告缺项，不能用 exposed 模型顶替后不标注。

## 2. 冻结模型与评分面

路径均相对工作区根。以下是**候选定位信息**，执行者必须读 receipt 并解析实际路径/SHA；不得假定正在训练的产物已完成。

| 任务 | 主候选与读点 | 评分面与解释边界 |
|---|---|---|
| M1 | `results/m1_projadd_depth2/20260907_gpu0/depth4/` 下匹配 D4 的固定 e24 EMA，存在且完成才可用 | 26/27/28 训练，20120924 留出；26,496 窗只是历史库存参考，重新核对合法坐标；这是历史开发面，不是新盲测 |
| H1 | `results/h1_c2protocol_l200_p16/20260907T021251Z/` 的固定 `epoch_032.pt` EMA shadow，L200 P16；核验EMA完整性 | C2 HO-M3 的14个public held-out-calib录音，按`c2_protocol.py::grouped_session_metrics`的`key.split("_set_")[0]`归为S6–S12共7组；bootstrap单位为7组，不是14文件 |
| M2 | `results/m2_projadd/20260906_104300/` 的既有 SEL-2 ext4 pick **e9 EMA**，核验 `ext4_scan_receipt.json` 的`picks.SEL`（view: EMA、earliest max）与实际权重 | ext4 是参与过选点的 development 面；不得改用目录里其他 ext6 pick 或根据本次干预另挑权重 |

表内结果路径均以 `btransform_unified_v1/` 为前缀。候选不存在／损坏／epoch 未到：标 `WAITING_FOR_FROZEN_CHECKPOINT`，不得静默改成 latest/best。

本次修订核验快照：匹配M1 `depth4/`尚不存在；D2已有e24评分（20120924 R²=0.688588），但不能替代D4主候选。需要由原深度线确认启动授权并补齐D4训练与评分，本文不自行补跑。实际执行时重新检查产物与资源，不把本快照或审核时GPU空闲状态视为持续事实。

次级诊断：M1 `m1_projadd_series/P16_20260906T150917Z/epoch_024.pt`、H1 full13 e24 仅在核清训练/评分重叠后标 `EXPOSED_DEPENDENCE_DIAGNOSTIC`。M1 depth2 endpoint 可另报结构稳健性，但不是第二 seed，不是主结果替换。
次级诊断保持可选：仅在主矩阵已锁定且剩余预算足够时预声明追加，不延迟主读数、不自动扩预算。并排结果受训练面/权重等差异混杂，不能据此因果宣称“暴露夸大reliance”。

暴露标签分开记录：

1. decoder 梯度训练是否包含评分 session/query；
2. E0 encoder、teacher、basis、normalizer 是否包含评分数据；
3. 是否参与 epoch、架构或历史协议选择。

只有全依赖链闭合才能称 source-only；仅 decoder 留出而 donor 未闭合则 `CONDITIONAL_DECODER_LOSO`。固定 H1 e32 只避免本实验挑 epoch，不抹去历史开发暴露。

## 3. Bank 与输入不变量

- 主体固定 proj_add P16、原 frontend/temporal/window、precision FP32、输出尺度、mask、scorer、EMA 权重；不能用塌缩 concat 判断 E0 无用。
- M1 使用既定 M10、activity-only E0 与直接 rSyn3；H1 **所有臂同一 M3 部署 bank**；M2 support 预算和 bank 以冻结 pick 的已核验配置为准，禁止混用预算。
- M1 E0 来自 B3 Sfix e11；H1 E0 已融合 H-C，M2 E0 可含 T4。H1/M2 的 E0 干预不是“纯活动统计”消融。
- H1/M2 只去直接 c，不等于去除 E0 中的 carrier 信息。
- 冻结 encoder、源统计、basis、bank；不得为任何臂重拟合 normalizer/readout/bias。只在内存副本改输入，原磁盘产物保持不变。

## 4. 干预矩阵：每任务 10 条推理流，零次训练

| arm | 替换 | 不变项 | 流数 |
|---|---|---|---|
| REAL | 原 E0、原 c | 全部参考 | 1 |
| E-ZERO | E0=0 | x/c/mask/顺序 | 1 |
| E-MEAN | 有效单元 E0 的 session 均值，广播至有效行 | x/c/mask/顺序 | 1 |
| E-SHUFFLE | 有效单元内置换 E0 | x/c/mask/顺序 | 3 |
| C-ZERO | **归一化后**直接 c=0 | E0/x/mask/顺序 | 1 |
| C-SHUFFLE | 有效单元内置换直接 c | E0/x/mask/顺序 | 3 |

C-ZERO逐任务解释必须写入manifest，记录normalizer类型、是否中心化、统计量SHA及零值对应的raw系数：

| 任务/直接carrier | 当前归一化口径 | 归一化后全0的含义 |
|---|---|---|
| M1 rSyn3 | 源RMS缩放，无中心化 | raw系数全0；不是源均值carrier |
| H1 H-C | normalized EB；以实际bank/normalizer核验是否中心化 | 若中心化则对应所用源中心；若只缩放则对应raw零，不能预设为均值 |
| M2 MOVE-T4 | 源z-score | 对应源均值carrier |

三者都消除直接c端口的单元间差异，但数值参考点不同；C-ZERO不等于完全移除模型中所有校准信息。若实际artifact与表中口径不符，P0先解决绑定差异，不能擅改normalizer凑表。

Shuffle seeds 固定 101/102/103；每个评分 session/group 的映射整段固定。不得跨 session、逐 bin 重抽、置换特征列或同步置换 x。记录映射和哈希。3 shuffle 不是 3 个训练 seed。

实现首选直接调用训练模型 FP32 `eval()` 的完整窗口 forward，批量 GPU 评分；这不提供 streaming latency 证据。禁止调用异参数化的 `identity_mode=zero`。

P 无 bias 时 raw E0=0 与 P(E0)=0 等价，执行前以实际模块核验。即使直接 forward，若向其传入预折叠 static，也必须按臂重建；不能以“没有 wrapper”推定绝无缓存。

## 5. 开跑门与执行步骤

### P0：只读绑定，先于任何干预分数

生成 `manifest.json`：任务、准确权重路径/SHA、epoch/EMA、代码版本、bank数组/派生项SHA、各层暴露、support/query合同、group映射、scorer配置、query坐标SHA、随机种子、设备/预算。

### P1：参考与干预 smoke

1. REAL 对照同权重现有 scorer；坐标/聚合一致时才比较旧数字，不为配数字更改 mask。
2. 一次 REAL→干预→REAL 往返，恢复参考结果；缓存/static 必须重新生成。
3. 检查置零／均值／映射确实作用于有效行、原数组未改、输出有限、无目标标签进入 bank。
4. FP32 参考容差默认逐输出 `abs(delta) <= 1e-5 + 1e-5*abs(reference)`，记录 native scale 最大误差/RMSE；阈值先封存，不在失败后放宽。

### P2：评分覆盖与预算

- 默认完成各任务**完整合法本地面**；先用不读分数的计时确定可行性，不预设“秒级”。流式窗口不能越过 reset、session 或禁止跨越的 support/query 边界。
- 建议上限：全部任务合计 **1 GPU-hour 推理**，CPU 工程/审计时间单列；预算不继承其他实验。只使用明确分配且空闲的设备。
- 若完整 10 流预计超限，在任何干预分数前冻结降级：每 session/group 最多 2,048 endpoints，按 trial/时间块均匀抽样，所有臂相同。覆盖所有允许 session，不删困难 session。
- 完整面太大时不得只把“看起来效果最好”的对照扩到全量。追加完整评分须对预声明整组对照同面扩展并记录版本；否则保持抽样结论。
- 分批读取/推理，不预展开全部 `[Q,L,N]`；无 profiler 的计时与含诊断的运行分开。

## 6. 指标、判读与不能宣称的内容

主报逐 session/group 的配对 delta、equal-group mean；保留任务原 scorer 的输出维加权、mask和尺度。M1/M2 默认 variance-weighted R²；H1须复用 C2 grouped metric，不能把录音直接当独立 session。pooled R²只辅报。

E-SHUFFLE、C-SHUFFLE先在每组内平均3次，再跨组汇总。多 session 做预固定 seed、2,000 次 paired group bootstrap，区间只描述当前开发样本，不代表训练 seed 方差。H1在7个C2组上重采样。M1只有一个留出session：报单session点估计，主分析预固定按query trial块配对bootstrap，manifest保存trial坐标与块映射；不声称跨session总体置信区间。若trial映射缺失，先标协议缺项，不在见分数后改成时间块；确需5秒块替代时必须在评分前封存，并辅报2/10秒敏感性。

诊断：预测/目标 std 比、dynamic/static RMS、P(E0)范数、第一层预激活 mean/std、NaN/Inf。统计只取固定诊断点，不以诊断数值替换正式 scorer。

| 观察 | 合法解释 |
|---|---|
| REAL优于ZERO、MEAN和SHUFFLE | 支持冻结模型依赖单元特异且正确配对的E0；仍不是重训效用证明 |
| 仅ZERO显著变差 | 分布/幅度冲击、全局偏置与信息缺失都可能；不能唯一归因 |
| MEAN>REAL | 保留均值的平滑可能有利；值得测NOE0，但均值仍含session信息，不等于无E0 |
| ZERO>=REAL | 可能冗余/负作用；不能单凭一次干预宣告删模块更优 |
| 所有臂近常数 | 检查模型/协议，标无辨识力，不报E0无效 |
| C干预改变预测 | 直接c路径有依赖；不能在H1/M2称移除了所有functional信息 |

预激活矩不能分离OOD与信息缺失；E0/C置换也会改变它们与x的联合分布。工程追踪尺度可用均值ΔR²>=0.01且多数group同向，不是显著性门，不因未过而取消EXP2配对。

## 7. 交付与验收

结果根：`btransform_unified_v1/results/exp1_e0_reliance_v2/<UTC>/`。

必须交付：

- `manifest.json`、`provenance_audit.json`、`query_inventory.json`与坐标、`permutations.json`与映射SHA；
- `oracle_parity.json`、`intervention_checks.json`、所有臂原始预测或可验证分片索引；
- `per_group_metrics.csv`、`paired_effects.csv`、`diagnostics.json`、预算实耗；
- `interpretation.md`：每任务证据等级、负结果、局限、是否需要本任务配对重训；
- 实际CLI命令、环境、代码SHA、权重/bank SHA、复现步骤和失败日志。本文不捏造尚未实现的CLI。

验收按 `PROTOCOL_PASS/FAIL`、`REFERENCE_PARITY_PASS/FAIL`、`RELIANCE_DETECTED/SMALL_OR_UNCERTAIN/UNINTERPRETABLE` 分开报告。必须完成所有预声明臂或明确缺失；不存在必须得到正结果的门。

## 8. 执行前最高优先级自检

1. “未参与decoder训练”与“全系统clean/新盲测”是否区分？donor链是否闭合？
2. H1 grouped单位、M3 bank与固定e32是否准确；待产物是否被误当已完成？
3. 十条流是否真正只改指定端口、重建所有派生缓存？
4. 是否把推理干预解释成重训utility，或把均值/std解释成因果分解？
5. 是否存在看分数后改覆盖/权重/主对照的自由度？

来源：[原E0设计](DESIGN_E0_UTILITY_PAIRED_V1_20260907.md)、[初轮审核](REVIEW_E0_UTILITY_AND_688_COSTAWARE_V1_20260907.md)、[V2审核](REVIEW_WORKORDERS_EXP1_EXP2_EXP3_V2_20260907.md)、[H1 C2协议](H1_C2_PROTOCOL_SELECTION_V1.md)。审核原文保留；本V2修订稿作为独立执行合同，不追溯修改其他任务授权。
