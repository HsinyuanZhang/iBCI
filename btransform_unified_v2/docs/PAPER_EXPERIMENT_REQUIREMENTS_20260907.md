# 论文实验需求与可用证据审计（2026-09-07）

**审计对象。** 本文只把 `bci_paper_overleaf/paper_4pp.tex` 中已经写出的实验主张、表格和 `TBD` 转为可执行的证据要求；不修改论文、代码、权重或既有结果。本文的伴随机器可读清单是 [PAPER_EXPERIMENT_REQUIREMENTS_20260907.json](PAPER_EXPERIMENT_REQUIREMENTS_20260907.json)。

**结论。** Published SPINT/zero-shot Wiener 已填充 M1/M2/H1 的外部 BP-free reference；H1 RIFT submission 582073 也已有归档官方 result。它们都不闭合四任务 profile effect。688 现应采用已审计的 `27 train / 6 validation` strict manifest；先前 15-session 三 seed候选的来源/协议不可靠，不能进入本文。M2 RIFT B42 formal 已完成 development score（ext4 `0.266825`）；D/joint 条件仍在运行。M1 RIFT formal v3 已完成，selected EMA epoch 3 的 HO-trio equal-session mean 为 `0.7046861491`；M1 concat 条件仍在运行。尚未形成论文所要求的完整四任务、同一模型族、同一支持/查询与明确 checkpoint 的配对矩阵。

因此，论文中所有红色数值和任何“跨四数据集”“架构贡献”的叙述均保持未证实。当前实验主线是 **RIFT + cached CPU runtime**；旧论文中的 E-ORT/ONNX 描述不是继续补跑的四任务门槛，而是**待按 RIFT 主线改写**。BT-EORT 只保留为必要的架构/运行时对照，不能替代 RIFT 的精度或机制证据。下列工作流将缺口明确分为必须完成、先定协议和若无法证明则删除三类，避免把历史开发分数或不同系统的官方分数误写成新 RIFT/B 论文结果。

## 1. 证据准入规则

一项结果只有同时满足下列条件，才可以进入主文定量表格或支撑对应因果句：

1. **来源绑定：** 有不可变或可复算的 receipt/manifest，记录代码与配置哈希、训练数据、source checkpoint、target support、query、随机 seed、模型选择面和指标实现。
2. **暴露合法：** target carrier 只能读取声明的 support；query 行为只用于打分。源侧训练可反传；目标 session 不可进行网络反传或梯度更新。
3. **比较配对：** 一项差值的双方共享 source 数据、初始化/seed 计划、support、query、有效帧、评分、选择规则和训练预算；唯一变化是被声明的因素。
4. **查询面匹配：** 不能把 M2/H1 organizer score、688 本地 development score、历史 SPINT 分数或不同版本 decoder 分数混在同一列。每个单元格写清官方或本地评分面及聚合方式。
5. **不确定性：** 跨 session/date 的 effect 必须报告单位、点估计、区间/重采样方法、正向单位数和 seed 覆盖。单 seed 训练不能支撑可靠性、显著性或“四任务一致”。
6. **执行模型绑定：** 论文所称 `B` 必须绑定实际网络、窗口、权重和运行后端。BT-EORT、RIFT、旧 SPINT 或只含 carrier 的冻结 consumer 不是可互换名称。

旧结果可作为开发导向、复用权重或协议模板；只要缺任一字段，状态为 `candidate_only`，不是论文数值。

## 2. 论文逐项承诺与关闭条件

| ID | 论文位置 | 承诺/表格单元 | 关闭所需证据 | 当前判断 |
|---|---|---|---|---|
| P01 | 摘要 40–50，结论 537–544 | 688/M2/M1/H1 上 profile 效果和 calibration cost；主线模型为 RIFT | 每任务 `profile+frozen reference − activity-only reference` 的同面配对 effect、区间、support 成本；RIFT checkpoint/runtime binding | **必须完成**；没有跨四任务闭合 |
| P02 | 160–285 | 四种 task-specific estimator 都产出合法四维 carrier，且无 target BP | 每任务 source 参数、target support 输入、4D 输出、normalizer、目标反传步数为零的审计 | 方法可读，实证绑定未完成 |
| P03 | 406–412 | B 的 filterbank/slots/temporal 架构及任务窗口 | 每任务 checkpoint→config→architecture→backend 映射；RIFT 与 BT-EORT 的名称不可混用 | **协议先定**；当前论文未绑定 |
| P04 | 416–434 | 公平 evaluation protocol | 逐单元的 support/query hash、source/eval count、seed、选择面、R² 聚合和 SPINT provenance | **必须完成** |
| P05 | 主表 452–457 | baseline/system × 四任务 | Published SPINT/zero-shot Wiener 可直接引用其原始 FALCON held-out Table 1 值；新 RIFT/本地行仍须各自有 receipt；不适用的 PV 要明确为 N/A | **已部分闭合**；published 数不能被表述为本地配对 effect |
| P06 | 474–491 | profile effect、B effect、carrier stability/basis coverage | 四任务或明确限定任务的配对差、重采样 stability、coverage 定义和关联分析 | **必须完成或删去泛化句** |
| P07 | 480–513 | 代表性 task/split 上的 activity-only、carrier-only、joint、shuffle、spatial、temporal mechanism matrix | **冻结 M2 source-seven/ext4 为代表协议**；同一 decoder/budget/训练方案下的独立 arm；shuffle 在两条 carrier 路径之前 | **必须完成于该代表矩阵**；不把 7 arms 扩写为四任务 28 cells |
| P08 | 515–534 | CPU runtime 与 calibration cost | RIFT cached CPU 的完整调用 parity、median/P95、校准时间、峰值内存、设备/线程/精度/batch；BT-EORT 仅作明确标记的对照 | **论文文字待按 RIFT 主线改写**；不要求重跑 E-ORT/ONNX 四任务矩阵 |
| P09 | 73–80，主表 | BP-free comparator boundary | 每 baseline 的目标校准协议、合法性、query identity 和 score receipt；RIFT 的 target calibration 同样满足零 network BP | ridge 可用候选；Kalman/PV/CEBRA 大部未闭合 |

## 3. 现有资产：可用范围与禁用范围

### 688：已改为 strict manifest 路线；旧候选不可用

688 当前采用已审计的 `27 train / 6 validation` strict manifest。此前 `dandi688_sparse_event_t4_v1` 的 15-session/三-seed 候选不具备可信的论文 provenance，seed44 还曾中断；不得再将其差值、bootstrap 或 `15/15` 写为可复用证据。

不得直接复用为本论文单元格，直到确认：

- `shared_t4` 的 carrier 与论文 Eq. (688) 的 native trial-direction、rate window、equal-direction weight 和 source winsorizer 一致；
- 支持预算、post-support query、session split、R² 聚合和 bootstrap 单位与主表协议一致；
- `shared_zero4` 真正移除了 `e_i` 与 direct `c_i` 两条 profile 路径；`shared_ts4` 在两路径前同一 unit-wise permutation；
- 固定 decoder/checkpoint 是否等于论文 `Profile + frozen SPINT`，而非另一训练家族；
- 688 需要另建 proposed B/RIFT 对照、活动-only SPINT、ridge、PV（如定义）和 Kalman（如保留）。

只复用 strict manifest、已审计数据边界和可验证的接口；在该范围产生新 receipt 前，688 主表为 pending，published SPINT 也为 N/A。

### M2：代表性机制 task；formal24 已完成，joint 条件待闭合

M2 RIFT formal24 已冻结为 R50、local attention、`proj_add` P16、seed42、M33 banks、24 epoch source-only contract，并完成 development scoring：ext4 epoch11 `0.3408126083`、epoch24 `0.3366115095`。这些是 development 诊断，不填 published 主表；当前正在核对 concat/joint 条件。M2 source-seven/ext4 已冻结为论文 Table~\ref{tab:ablation} 的代表性 mechanism protocol，选择不再按模型分数更换任务。

旧 M2 T4 及 official submission 记录可以作为 `Profile + frozen SPINT` 的候选来源，但 `M2_MATCHED_EPOCH34_B0_EVALAI_ELIGIBILITY_20260804.md` 明确说明现有 T4 official epoch-34 与 original-SPINT epoch-27 不匹配。因此官方差值只能称端到端系统差，不能称 profile-only matched effect。复用 M33 cache、frozen mapping 和 source manifest 能节省数据准备；仍需产生配对 B3S、T4、activity-only、carrier-only、shuffle、spatial/temporal B arms 以及清楚的选择和 score receipt。

SPINT 原论文 Table 1 可直接引用 M2 held-out private-split `0.26\pm0.13`（SPINT GF-FSU）和 `0.06\pm0.04`（zero-shot Wiener）：均为跨 held-out session mean$\pm$SD $R^2$。它们是 published FALCON reference，不是本地 M33/RIFT 的 matched delta。

### M1：存在冻结三臂背景对照；RIFT formal 已完成

M1 的 rank-three NMF/NNLS/ridge 公式给出了可审计的四维 carrier 方案；`behavior_autoencoder_v1` 等源侧研究是建立源表示的候选资产。另有同一 HO-trio 下的 FrozenB3、BT/Joint、concat 三臂冻结对照；这些可作为**上下文和支撑性比较**，并能帮助核对 source/HO 边界和静态侧信息注入语义。它们不是 RIFT 训练、也不自动满足本文 A/B/C/D 的成对机制口径，不能直接代填 RIFT 主表。

`m1_r100_recency_s42_formal_v3` 已完成冻结 B3+rSyn3、24 epochs 和完整 HO-trio scan；selected EMA epoch 3 的 equal-session mean 为 `0.7046861491`。先完成 CPU constructibility/provenance receipt（冻结 `H`、source RMS、排序、NNLS 收敛、support/query 和零 target-BP 审计），并将 FrozenB3/BT-Joint/concat 三臂整理为 context bundle。M1 后续是否扩展机制/seed 取决于结果不确定性与最终主张。

### H1：已有冻结 RIFT 官方候选、已完成 selection 与 trained CPU 证据

H1 可复用的严谨候选包括：

- `h1_ridge_baseline_v2r2`：两 fold-0 development recordings、严格 post-support 的 8,965 query windows、固定四 trial/session Ridge50，pooled R² `0.258235`，from-source byte-identical replay 通过；它是描述性 dense-velocity comparator，**不是** matched carrier ablation。
- 同一 H1 发展边界的 H-S、H-SE5、Context Full、H-C 分数和 sparse-support accounting（详见 `HANDOFF_H1_RIDGE_BASELINE_20260812.md`）。这些可作为 frozen-SPINT/profile 候选，但需把本论文 H1 readout carrier Eq. (H1) 的 PCA/q16/lambda100/shrinkage 与实际 carrier 逐字节绑定。
- RIFT R300 recency/flat 两臂 formal 与 selection 已完成；冻结候选为 `h1_rift_r300_recency_e22_cached`（submission `582073`，R300/D4/P16/width256、seed42、EMA epoch22、13 source sessions、cached CPU KV）。`FREEZE_H1_RIFT_20260907.md` 记录用户报告官方 Held-out/held-in/normalized latency 为 `0.403/0.668/0.143`；本地没有官方回执，故可在文稿中标为 **user-reported official result，待 receipt 独立核验**，不可冒充本地复核。
- `cpu_trained_h1_final_20260907T1154Z/benchmark.json` 是训练后 H1 raw-flow B1/B8 end-to-end benchmark；RIFT cached 与 reference 进行了 401 advances 的 `max_abs_error=0` parity，并记录 B1 cached median `2.697 ms`、P95 `2.803 ms`。这是 RIFT cached CPU 的可复用运行时证据；BT-EORT/ORT 同文件中是旧模型对照，不能归为 RIFT。

H1 已完成主线候选冻结，不再用它追逐新分数。submission 582073 的归档官方结果为 held-out mean/std `0.4027782688014744 / 0.14526658466582196`、held-in mean/std `0.6681165838896308 / 0.025051352172649453`、normalized latency `0.14299749625157748`，见 `results/rift_v1/h1_r300_official_receipt_20260907/{submission_get.json,official_result.json,retrieval_receipt.json}`。不得把 development H-C/H-SE5、旧 BT-EORT 或该 official RIFT 结果合并为同一个 profile effect。

SPINT 原论文 Table 1 可直接引用 H1 held-out private-split `0.29\pm0.15`（SPINT GF-FSU）和 `0.16\pm0.03`（zero-shot Wiener），均为跨 held-out session mean$\pm$SD $R^2$；这些不是本文 H1 development 或 RIFT official 的同一查询面。

## 4. 信息与架构机制矩阵

论文的四条 carrier 路径必须采用以下定义。名称 `B3S` 表示活动 identity 的冻结 forward baseline，不能仅凭目录名推断。

| Arm | 保留信息与训练规则 | 必须检查 | 论文回答的问题 | 现状 |
|---|---|---|---|---|
| A / no calibration | 无 activity signature、无 carrier；若作为网络输入，使用固定零/缺失表示且不扩大模型 | 参数数、初始化和有效 unit mask 与其他 arm 兼容 | 校准本身是否有价值 | 各任务待建 |
| B / activity-only B3S | 仅 neural activity-derived identity；carrier 从 `e_i` 和 direct `c_i` 同时移除 | no-carrier parity，support 无标签依赖 | activity statistics 是否足够 | 旧 SPINT/H-S 等候选，不是完整矩阵 |
| C / carrier-only | carrier 保留，activity identity 置零/固定；动态 live neural input 不变 | 静态 path 的零化不改变 live activity | carrier 单独可否工作 | 待建 |
| D / joint T4 | activity identity + task carrier，且联合输入 fusion | 与 B/C/A 同训练预算 | 二者是否互补 | 688 可能有候选；其余待建 |
| D-shuffle | 对 carrier 做固定 unit permutation，再同时送入 embedding 和 direct path；最好独立训练该扰动 arm | permutation SHA、两路径一致、同 seed | unit–carrier 对应性是否重要 | 688 标签置换候选；其余待建 |
| spatial | mean pooling vs learned slots；下游 readout参数应匹配 | 参数/width、输入、训练选择和 query 相同 | 自适应集合聚合的贡献 | 待建 |
| temporal | causal transformer vs matched non-attention causal control，保留 horizon/frontend | receptive field、causality、参数/compute报告 | 时序建模的贡献 | 待建 |

**重要限制：** 同 checkpoint 的 forward shuffle 是有用诊断，却会低估“未曾训练过正确 carrier”的依赖；论文若写训练归因，需要单独训练的 shuffle/no-carrier arms。688 旧 `shared_ts4` 的具体训练语义必须在 provenance 审计中确认后再决定是否足够。

Carrier stability/basis coverage 的最低 protocol：每 task/session 由同一支持集做预注册重采样，保存每 unit 4D vector，比较 Pearson/cosine/Procrustes 或预先选定的坐标一致性；coverage 用方向角/协方差条件数（688/M2）、NNLS basis activation/span（M1）或 H1 projected readout design rank/conditioning 表示。只报告稳定性与 coverage 之间的相关时，必须有独立 session 单位、置信区间和未选择性筛选规则。

## 5. 主表与 baseline 的可执行处置

| Row | 可保留的条件 | 当前处置 |
|---|---|---|
| Published SPINT | 直接引用 SPINT Table 1 的 FALCON held-out private-split结果：M1 `0.66\pm0.07`、M2 `0.26\pm0.13`、H1 `0.29\pm0.15`；目标端只处理未标注 calibration activity、无参数更新 | 可引用；688 为 N/A；不得称作本地 replay 或同面 RIFT effect |
| Published zero-shot Wiener | 直接引用 SPINT Table 1：M1 `0.34\pm0.06`、M2 `0.06\pm0.04`、H1 `0.16\pm0.03`；single held-in-session fit、held-out zero-shot | 可引用；不是 target-prefix ridge，688 为 N/A |
| Population vector | 只在行为可定义的任务实现；写清 tuning/support/readout。M1/H1 若无合理定义应为 N/A | 688/M2 先审计；H1 旧结果已隔离；M1 未定义 |
| Ridge / target-prefix Wiener | target prefix、lambda/selection、dense label accounting、query identity 和 score receipt | H1 v2r2可用候选；不为填表另开实验；只有 RIFT 比较需要该同面 baseline 时才补 |
| Kalman | analytic state fit、state construction、target/query identity和真实数据 receipt | 代码/合成测试不是论文结果；未完成则删数值行，必要时删行 |
| Profile + frozen SPINT | 明确本论文 carrier estimator、frozen consumer、support budget和 matched B baseline | 688/H1/M2各有候选，但均需确认绑定；M1缺失 |
| Proposed set-temporal B | 明确为 BT-EORT 或 RIFT；每任务 checkpoint/config/backend/selection 可追溯 | 四任务均未闭合；M2/H1当前训练记录不等于分数 |

CEBRA `adapt=True` 仍排除；只有冻结 source transform 加指定 non-gradient readout 的 receipt 才能重新考虑。SPINT 原文说明 CycleGAN 在 held-out day 训练 GAN、NoMAD 在 held-out day 训练 alignment network，因此二者不属于本文的 target-BP-free 定量行；NDT2 FSS/OR 也不属于该行。NoMAD、CycleGAN、FA 或其他历史家族不因名称自动成为合格行。

## 6. 运行时与校准成本：RIFT cached CPU 发布门

论文方法 RIFT 表述已在已推送的方法更新 `a186dd9` 中对齐 cached CPU 主线；requirements 仍要求所有数值按 receipt 绑定。RIFT runtime 以 cached CPU 为报告对象；BT-EORT/ORT 仅可作为明确分列的旧架构对照。每个实际报告任务仍应提供：

1. 输入、校准 embedding、stream output 的 numeric parity（阈值、样本数、最大/分位误差）；
2. 完整 decoding call 的 warm-up 后 median/P95，包含 static-cache 查找、frontend、state update、temporal、readout，不以 isolated kernel 替代；
3. one-time calibration wall time，分出 carrier solve、embedding construction、cache/serialization；
4. peak RSS/allocator memory、模型参数、设备、OS/CPU、thread affinity、precision、batch/streams、window、backend version；
5. 同等 batch/precision/thread 的明确 baseline，以及每项是否使用训练后权重或合成输入；
6. 失败/回退行为和 session boundary reset；不得把 logical KV payload 写成总内存。

H1 已有训练后 public raw-flow 的 RIFT cached CPU benchmark：B1/B8 的 reference-cached parity 共核验 401 advances，最大绝对误差为 0；B1 cached steady median `2.697 ms`、P95 `2.803 ms`（300 calls）。该证据可报告为 H1、指定设备/线程/precision 下的 RIFT runtime；仍不得外推为四数据集 speedup、校准总成本或总内存。早期未训练 synthetic probe 仅保留作工程诊断。

## 7. 最小可发表闭合计划与资源排序

### P0：无 GPU 的协议与来源闭合

1. 建每任务唯一 manifest：source sessions/checkpoint、support/query hashes、metric、aggregation、合法性审计、model-selection surface；逐个核验现有 688/H1/M2候选。Published SPINT/zero-shot Wiener 不重跑，单列原文 Table 1 的 FALCON held-out surface。
2. 论文 `B` 已按当前主线指向 RIFT；改写旧 E-ORT/ONNX 文字为 RIFT cached CPU，BT-EORT 仅作为需要时的单独对照。
3. 完成 M1 carrier CPU constructibility、M2/688 directional contract、H1 Eq. (H1) carrier implementation binding；固定 support budget、单位和重采样。
4. 只读复算 688 三 seed汇总、H1 ridge/references，输出与主表完全相同的 score surface。任何 hash 或边界不一致即降为 `candidate_only`。

### P1：最高信息密度的训练与评分

完整 mechanism matrix 先在已冻结的 **M2 source-seven/ext4** 协议上运行：B/activity-only、C/carrier-only、D/joint、D-shuffle、mean-vs-slot、matched non-attention temporal control。A/no-calibration 是有价值的附加诊断，不是当前表格四行的硬门槛。三 seed 优先用于 paired B/D；其他 arms 的 seed42 先作为探索性结果，是否扩展由不确定性和最终主张决定。预先锁定选择策略，official test 不参与选点。

优先顺序是 **M2 representative mechanism closure → M1 formal closure → 688 strict-manifest evidence**。H1 已冻结并只归档。若 GPU 时数不足，宁可缩小跨任务主张，也不要用异构旧分数伪造四任务效应。

### P2：外部评分与运行时

H1 submission `582073` 的官方回执已归档且已验证，不再追逐新 H1 score；M2/M1 的 official 行动在 development closure 后决定。运行时任务与精度任务分开：以实际 RIFT cached CPU 后端做 calibration 和 streaming 测量，BT-EORT 只在明确比较目的下运行。

## 8. 删除条件（不是待补数字）

以下任一项在锁稿前不满足，应删除相应主张，而非保留 `TBD` 或替换为旧数：

- 无四个合格 effect → 删除“matched evaluations across four datasets”和结论中的“verified four-task profile effects”。
- M1 无 paired outcome → 删除 M1 作为实证数据集，只保留且明确标为设计方法。
- B 无 checkpoint/architecture binding → 删除具体 B 结构、架构表和 B-minus-profile claims。
- 无独立训练 arm → 删除“carrier-only/joint/shuffle isolate contribution”及 spatial/temporal 因果归因。
- 无 RIFT cached CPU parity 与所报告任务 benchmark → 删除对应 RIFT runtime 数值；旧 E-ORT/ONNX 表述无论是否补测都应按主线改写。
- Kalman/PV 没有合法真实数据 receipt → 填 N/A 或删除该 baseline 行；不可用红色 TBD 暗示已评估。

## 9. 最终交付检查单

在把数值写入论文前，对每个表格格子勾选：`source manifest`、`support/query hash`、`target BP=0`、`checkpoint/config/backend`、`selection surface`、`metric/aggregation`、`seed/session score arrays`、`effect/interval`、`baseline protocol`、`receipt path`。伴随 JSON 中每项均有 `acceptance` 字段，可由汇总脚本判定；任一必填缺失，状态必须保持 `open` 或 `candidate_only`。
