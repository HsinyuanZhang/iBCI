# 三任务实验：第三方独立审核清单

整理日期：2026-09-06。目的：审核此前实验是否足以支撑质量、网络统一性和 latency 结论，而不是确认现有报告中的 PASS 字样。

## 结论边界与审核方式

建议优先审核下列 P0 四项，再审核 P1 四项。本清单聚焦当前三任务路线及其直接历史对照，不声称覆盖仓库中所有早期探索。当前 **M1/M2/H1 同时正提升或非劣的总目标尚未完成**。

截至 12:37 HKT 的检查，最新 H1 QueryAge 固定 12 轮实验仍在运行，已有 epoch 1–9 记录；它不是已完成的质量结果。此前 H1 CausalPE 正式训练、cold-prefix 和 dense 诊断已完成，必须与这次 fresh QueryAge 分开审核。M1 P1 与 M2 QueryAge+prefix 固定 24 轮及下列归档比较已经完成。

以下文档是实验执行方提供的定位材料，不是第三方结论。请审核者独立实现关键计算，核对原始数组与实际执行路径；仅运行我们的 verifier/pytest 或比较哈希不足以完成科学审核。哈希只能约束文件身份，不能证明文件内容或实验设计正确。

## P0-1 — H1 低 R²：原始数据到分数的完整计算链

**为什么优先：** 已完成旧 H1 同 20,325 点比较的 FLAT/ROUTE pooled R² 为 `.479149/.419929`，as-shipped Original 为 `.960784`。最新 QueryAge 的 epoch-9 selection 为 `.242763/.244583`，同 2,908 点的 Original 子集为 `.963693`；这两种点集不能混减。如此大的差距需要先排除实现/数据错位，不能直接归因于架构。

请独立检查：

- 原始 NWB 时间戳、神经信号分箱、velocity 定义及维度语义，到 cache 的逐段映射；不要只比较两条都依赖同一 cache 的路径。
- W700 的输入 `s..s+699` 是否对应 `velocity[s+699]`；first-three-trial exclusion、eval mask、跨 trial 的历史、cold zero padding 与完整窗口是否符合声明。
- raw/native 单位转换：H1 train target `×20`、score prediction `÷20`，是否只转换一次；M1 divisor 1、M2 divisor 5，不能跨任务套用 H1 常数。
- FP64 SSE/SST、每输出先中心化、pooled 与等 session 平均、缺失/零方差输出处理，以及与实际任务 evaluator 的定义差别；不能用相关系数平方代替 R²。
- EMA/RAW、`train()/eval()`、dropout、prefix、校准 bank、输出 channel 顺序是否一致。用人工可计算输入验证窗口/target 时间关系；用 `prediction=target` 等恒等样例验证评分，再独立复算实际归档。

**验收输出：** 每个边界的实证相等/不相等、首个差异坐标、独立 R² 全表及与原报告差值。若探索 lag/scale，只可列为诊断，不能在此开发集上调到高分后替换原始结果。静态“没看到 bug”不等于完整链路已排除错位。

证据入口：

- [旧 H1 完整质量表](../results/family_runtime_v1/h1_frozen_same20325_quality_v1.json)、[Original 完整归档 receipt](../results/family_runtime_v1/original_h1_frozen_same20325_v1/receipt.json)。
- [cold-prefix 固定诊断](../results/family_runtime_v1/h1_cold_phase2x2_archive_summary_v1.json)、[dense 固定 stage-2 评分](../results/family_runtime_v1/h1_dense_phase2_fixed_e2_score_v1/receipt.json)、[固定 endpoint12 RAW 诊断](../results/family_runtime_v1/h1_fixed_endpoint12_raw_diagnostic_v1/receipt.json)。训练 source208 高拟合不证明 minival 泛化；这些诊断也不单独证明原因。
- [最新 H1 formal scorer](../src/h1_queryage_family_v1/formal_prefix_score.py)、[formal trainer](../src/h1_queryage_family_v1/formal_prefix_train.py)、[预定协议](PROTOCOL_H1_QUERYAGE_FORMAL_PREFIX_V1_20260906.md)、[仍在运行的实验目录](../results/decoder_validation_v2/20260905_190000/h1/queryage_family_v1/formal_prefix_split12_v1/)。最终选中权重审核须等固定实验完成，不提前选择或改动运行。
- 执行方自查：[数值契约](AUDIT_H1_LOW_R2_NUMERICAL_CONTRACT_20260906.md)、[坐标契约](AUDIT_H1_LOW_R2_COORDINATE_ALIGNMENT_20260906.md)、[runtime/direct 对齐](AUDIT_QUERYAGE_LOW_R2_RUNTIME_ALIGNMENT_20260906.md)、[对照点集与校准差异](AUDIT_H1_QUERYAGE_COMPARISON_SURFACES_20260906.md)。

## P0-2 — 基线、镜像、实际加载权重与历史远端分数归属

**已经发现的具体风险：** 历史 M2 submission 581919 的记录镜像内，默认命令读 `/data/decoder.pkl`，声明的 seed44/e8 payload 却在 `/artifacts/t4_m2_seed42_identity.pkl`；两个文件 SHA 不同。本地默认加载不一致已确认，远端是否覆盖命令仍未知，不能据此直接宣布远端用了哪个模型。

请从 image digest → Entrypoint/Cmd → wrapper 参数 → 实际打开路径 → payload/checkpoint SHA → submission 记录/远端执行命令逐一追踪。远端日志需要用户另行提供只读访问或导出，不应为了审核而重提交。对所有重要表格标明 Original as shipped、历史 teacher、Sfix、C2、旧 e8，不能只写“SPINT”。

特别核对：M1 `.597858` 的旧 teacher-overlap 参考不等于 as-shipped `.809289`；H1 C2 `.888499` 不等于 Original `.960784`。这是模型身份/校准和历史训练暴露的区别，不能在证明新模型提升时选择较弱者替代较强者。

**验收输出：** 各分数的唯一模型身份证明；对证据不足的远端归属明确标“未验证”，说明哪些旧结论因此需降级。

证据：[M2 默认 payload 审计](AUDIT_M2_581919_DEFAULT_PAYLOAD_20260906.md)、[机器记录](../results/family_runtime_v1/m2_581919_default_payload_audit_v1.json)、[H1 基线审计](AUDIT_H1_QUERYAGE_COMPARISON_SURFACES_20260906.md)、[M1 原版重放](../results/family_runtime_v1/original_m1_frozen_same31252_v1/receipt.json)。

## P0-3 — M1 的小幅质量提升是否成立，能否称非劣

已完成 same-31,252 source-minival 比较：Original `.80928876`，选中 FLAT `.81165233`、ROUTE `.81219643`。增量只有 `+.00236357/+.00290767`，且 session `20120928` 分别下降 `.01089958/.00652333`。

请独立逐点比对 target/session/start，复算三 session 与 pooled/equal-session 指标，重建 epoch-6 EMA 的预定选择规则，并保留表现较差的 endpoint24。检查当前开发/selection 暴露和历史训练/校准暴露；不能把 31,252 个时间相关点当成同数量独立样本。

**验收输出：** 先判断描述性点估计能否复现，再单独判断泛化/非劣证据是否充分。正式非劣需预先规定容忍界值、统计单位与检验/区间方案；已看过结果后选一个宽松界值不是前瞻性非劣。现有单次训练和三个 source session 不自动证明多 seed 或新 session 的可靠收益，缺证据就列“需要独立确认”，不强行给 PASS。

证据：[完整质量表及绑定归档](../results/family_runtime_v1/m1_p1_frozen_same31252_quality_v1.json)、[全部 epoch 记录](../results/decoder_validation_v2/20260905_190000/m1/family_v1/p1_queryage16_pair_chron80_v2/epoch_metrics.json)、[严格 finalizer](../results/decoder_validation_v2/20260905_190000/m1/family_v1/p1_queryage16_pair_chron80_v2/finalized_p1/receipt.json)。

## P0-4 — M2 pooled 提升是否掩盖严重 session 退化

最新已完成 QueryAge+prefix 的 ext4：FLAT `.14306357`、ROUTE `.29639631`，Original `.22919761`。ROUTE pooled 增加 `.06719870`，但 `ses-2020-11-19-Run1` 为 `−.23816213`，相对 Original 下降 `.25546214`；FLAT pooled 负提升。两臂也都低于历史 packaged e8 `.38750524`。

请复算全部 2,069 点、四个 session 与两个输出的统计，查 support33 bank/session 路由、target 单位及窗口索引；核对 FLAT e2 / ROUTE e20 在看 ext4 前已经 source-only 冻结。区分已完成旧 CausalPE pair 与这次 QueryAge+prefix，二者同时改变 temporal operator 和 prefix recipe，不是单因素消融。

**验收输出：** pooled、逐 session、最差 session、bias/centered-error 全表；判断能支持什么范围的提升，不能称三任务已非劣。现有 Original 同面基线保留标量及预测哈希，但没有物化在当前 ext4 输出目录的 matching prediction NPZ；独立逐点重算基线仍是需补证项，不能拿 1,011 点 source NPZ 代替。

证据：[source-only finalizer](../results/m2/queryage_family_v1/queryage_prefix_pair24_finalized_v1/receipt.json)、[固定 selected ext4](../results/m2/queryage_family_v1/queryage_prefix_pair24_selected_ext4_v1/receipt.json)、[旧 CausalPE ext4](../results/m2/family_v1/frozen24_selected_ext4_native_v1/receipt.json)、[同面基线 addendum](../results/m2/family_v1/ext4_e8_spint_dev_pooled_addendum_v1.json)、[执行方误差分解](DIAGNOSTIC_M2_QUERYAGE_EXT4_ERROR_DECOMPOSITION_20260906.md)。

## P1-5 — 训练、校准与选模的数据暴露和时间顺序

跨三任务重建：calibration/support、source-train、source-minival、external dev、official/hidden 各自的读取与使用角色；特别区分“缓存里有 bytes”“实际 forward 读取”“target 用于评分”“target 用于训练/选型”。查 bank/teacher 的历史暴露、epoch 全记录、seed、初始化、sampler、训练预算、RAW/EMA、最早 tie-break 和选择冻结时间。

**验收输出：** 数据流/访问矩阵和选模时间线；说明每项结果是训练拟合、开发集描述还是未触碰测试确认。既不能从同文件含有两 split 就直接断言泄漏，也不能以无梯度更新就断言完全无选择暴露。完成新模型的 source-only selection 不会抹去旧对照及开发集已知结果带来的历史暴露。

证据：[进展记录及各阶段原始链接](PROGRESS_CRST_FAMILY_20260906.md)、[family freeze](FREEZE_ASTRA_FLAT_ROUTE_OPERATOR_FAMILY_20260906.md)、各实验目录的 `run_meta`/`input_authority`、全 epoch 与 finalizer receipt。审核者须查看这些文件实际约束了什么，不能只看文件名。

## P1-6 — 加速 runtime 是否真的等价于所选模型

复核实际 selected checkpoint 的 independent full-window oracle 对 public stream：gap bin 必须推进状态，reset、cold start、窗口左右边界、partial batch、mask/bank/model 更新后的失效规则都要覆盖；检查输出 native scale、shape/dtype/ownership、参数不变、持久内存。oracle 不能与被测缓存路径共用同一个可能出错的历史状态。

**验收输出：** 明确每条 assertion 是全调用、全评分 endpoint 还是少量抽样；复现完整已选权重结果，并解释误差阈值。toy 测试、初始化模型和 source-capacity 权重通过，不能代替 formal selected 权重证据；实现等价也不能当质量通过。

证据：[M1 selected 完整证明](../results/family_runtime_v1/m1_p1_selected_family_source_complete_v1/receipt.json)、[旧 H1 selected 完整证明](../results/family_runtime_v1/h1_selected_family_source_complete_v1/receipt.json)、[M2 QueryAge uncached](../results/family_runtime_v1/m2_queryage_selected_complete_uncached_ext4_v1/receipt.json)、[M2 QueryAge cached](../results/family_runtime_v1/m2_queryage_selected_complete_cached_ext4_v1/receipt.json)。最新 H1 QueryAge selected 完整证明尚未执行，不能沿用旧 H1 receipt。

## P1-7 — SPINT-relative latency 是否公平、可复现，且与质量是同一权重

逐一绑定同 image、实际 Original payload、selected candidate 权重、设备、PyTorch、线程/CPU affinity、batch、输入流、warm-up 和计时边界；检查 whole public call 是否包含 host/native NumPy 输出。保留全 raw timings、全部 repeats 与配对次序，分别报告 mean/p50/p95/p99、cold/load/reset/首调用和内存；不能只挑最快重复，也不能混 PyTorch 2.12 与 2.5.1 的绝对时间。

需要独立复核的关键现象：M1 本地 P95 speedup 约 `12.9–16.9×`；M2 QueryAge T1 平均较慢，T2 平均及 P95 较快，结论依赖线程。旧 H1 的加速结果对应低质量旧模型，不证明最新 H1 QueryAge 的速度，更不证明已满足质量/速度联合目标。

**验收输出：** 同配置独立重测与 quality/latency 共同权重身份表。当前测量是共享主机本地证据，有 H1 GPU 工作并行等干扰披露；它不自动等于用户部署服务或官方 normalized latency，也不自动解释用户所说的约十倍差距。

证据：[M2 六次选中权重 latency 总表及全部 receipt 路径](RESULTS_M2_QUERYAGE_SELECTED_LATENCY_20260906.md)、[M1 T1 r1](../results/family_runtime_v1/m1_p1_actual_selected_spint_t1_long2048_r1_v1.json)、[M1 T2 r1](../results/family_runtime_v1/m1_p1_actual_selected_spint_t2_long2048_r1_v1.json)（同目录的 r2/r3 也须全部审核）、[旧 H1 T1 首次长测](../results/family_runtime_v1/h1_actual_selected_spint_t1_long2048_v1.json)（同目录 T2/r2/r3 同样保留）。

## P1-8 — “同一或有限可解释网络族”与 ROUTE 机制

从实际构造器、state keys、forward 和 runtime 审计，不凭 QueryAge 名称认定相同：三任务共有 local-k5、八个 256 维 slot、四层/八头 QueryAge16 和 readout 框架；I/O、window、scale、bank 维度不同。H1 还有明确的 unscaled-dot/local-balanced 空间 preset，必须列为实际例外。

检查 ROUTE 是否仅为 calibration-conditioned 空间 attention logit bonus，零 gate 是否与 FLAT 对齐，以及两臂初始化、数据/增强和优化预算是否配对。列出全部运行过的 recipe/例外，避免把失败分支隐去或把仅同 temporal core 称为完全相同网络。跨任务训练差异不能归因于单一模块。

**验收输出：** 共享模块/必要 task adapter/真实额外变体三栏架构表，及当前证据支持的最窄准确表述；网络统一性不能代替每个任务的质量要求。

证据：[三任务网络源代码审计及构造器链接](AUDIT_QUERYAGE_THREE_TASK_NETWORK_FAMILY_20260906.md)、[family freeze](FREEZE_ASTRA_FLAT_ROUTE_OPERATOR_FAMILY_20260906.md)。

## 交付给审核者的最小材料与回报格式

先交本清单、上述原始 receipt/authority、被其引用的归档/源码/运行配置；模型和原始 NWB 在获准环境内只读访问。无需初始就重训，也不要提交账户凭据或为了审核发送新官方任务。首次审阅先定位缺证据项，再安排有资源预算的独立复算/forward；不与当前两 GPU 训练争用资源。

每项回报请含：`支持 / 仅局部支持 / 不支持 / 缺证据`，问题严重度，精确文件/行号或样本坐标，独立复现命令，期望与实际结果，对现有结论的影响，最小补证动作。修复建议与审核结论分开；不修改原始归档，不重选 checkpoint，不用修复后结果覆盖失败记录。

若审核资源有限，建议先做 **P0-1 H1 对齐、P0-2 基线归属、P0-3/P0-4 质量证据**；latency 与网络族审核随后做。内部 subagent 自查只作为辅助材料，不应被计作第三方独立复现。

## 已复核的关键入口文件 SHA-256

下列值是在整理清单时从现有文件重新计算的定位快照，不是对其全部科学主张的重新认证。完整依赖应继续沿各 receipt 的输入绑定追踪。

| 入口 | SHA-256 |
| --- | --- |
| M1 same31252 质量表 | `f481cd4c41f9eb201efd2b15c2e65d764b4f227dcfb087dc009e6fca1436a45e` |
| Original M1 receipt | `7db884108f60c3a7474b82629c18af64eb5c90325c78a6fa4a4ef3efbeb5939d` |
| 旧 H1 same20325 质量表 | `324c0db5a27afe0a26d51c636b5662f7bd2e7ddb41221da1cb546e40675ecef4` |
| Original H1 receipt | `539bfa832c650df51e22be14b5ee0dba0c1f49fbb595f9fb869328a601f30c48` |
| M2 QueryAge finalizer | `57c214b12c2097aa6bff74451169d092296f473c1d544dad53344288c23e9dcc` |
| M2 QueryAge selected ext4 | `0442ed59d740397b621a6c75b53041bca67205ad5e9407096406371d4b966b0e` |
| M2 581919 payload 审计 | `5314f8f5c83b6d6725e36a208ad08dc1923886a7c8879c1b664b581337fd0c28` |
| M1 selected runtime proof | `f2c614f323cd2995f20d5babdfa7dee49a29e02221af465475e93d8a913f5d2a` |
| 旧 H1 selected runtime proof | `974be98cc86541e75839d7e14d8cdceffbdbd412d3ac4aff2f8559f4d53e29cc` |
| M2 QueryAge uncached proof | `93775aa2b2963386681b93c0fb2265f372062f7ae3d9e1dff445fa6335c1fff7` |

## 附录 A — 协调组（GLM5.3）审阅意见（2026-09-06T12:5xZ 追加）

本附录由协调组在清单落盘后追加，只增不删；对正文八项的结构与优先级**整体认可**（P0-2 的条件式结论边界写得尤其准确），以下是逐项增补与三项覆盖缺口。追加时点状态：H1 QueryAge formal12 双臂 epoch9/12 仍在双卡运行，与正文"仍在运行"标注一致。

### 对既有八项的增补

- **P0-1（H1 对齐）**：除正文列出的链路外，必须额外披露两个已知事故/嫌疑，避免第三方把它们当干净输入：(a) `h1_temporal_unit_v2` 的 OVERFIT_WEAK 判定中 pred std/target std ≈ 9e-5 的近常数塌缩——单位契约（×20/÷20）是既有主嫌，独立复算应先验证单位再验 R²；(b) H1 旧 `window_manifest.json` 曾被 worker 误删并重建，标注 `RECONSTRUCTED_NOT_BYTE_VERIFIED`——重建件不得作为字节级权威，需以原始 NWB 重推窗口集合交叉验证。
- **P0-2（581919）**：本地证据链（image digest → 默认 `/data/decoder.pkl` ≠ 声明的 `/artifacts/...`）已闭合；无法本地裁决的部分只有"远端是否用了命令覆盖"。给第三方的提示：581973 对照（默认路径即所选 payload）已在审计文档中，可作为"该打包流程并非系统性出错"的对照点。若需最终归属，唯一干净的路径是在明确授权下以修正镜像重新提交作归属测试——那是一个独立动作，不属于本审核。
- **P0-3（M1 +0.0024/+0.0029）**：除逐点复算外，请按 session/trial 做块自助（block bootstrap）给区间——31,252 个滑窗点高度时间相关，按独立样本算的任何"显著性"都会大幅虚高；ses-20120928 的 −0.0109 应单列。另请核 EMA-e6 选模规则的**授权时间戳**是否早于任何 source-dev 评分时间（authorization-sha 链已在 family_runtime_v1 建立，可直查）。
- **P0-4（M2 QueryAge）**：FLAT 选点 e2 早得可疑——请核冻结时间戳与首次 ext4 评分时间的先后（receipt 均带 unix 时间戳）。另注意血统隔离：QueryAge 家族 0.296 与 S1 小 Transformer 0.4495/0.4584 是**两条不同血统**，任何汇总表不得混排；runtime_v3 的等价性 receipt 只有 v2 有效（v1 已因窗口语义错误显式 INVALIDATE）。

### 覆盖缺口（建议增列 P1-9 至 P1-11）

- **P1-9 M2 SMALL payload（S1 血统）的声明强度**：0.4495 是单种子点估计；N 波（1e-4）seed43 上 cosine 优势未复现（N1/EMA 0.363 vs N0/EMA 0.412），种子间方差 0.05–0.09。selector-v2 已预注册 endpoint24 主统计量，请核其预注册时间早于 seed43 出分。论文若引用 +0.09 必须带种子区间或降级为"单种子开发证据"。
- **P1-10 receipt 封存纪律**：`m2_dual_track_v1/20260905_101500` 全部 JSON 为 0644 可变、无 0444+sidecar（协调组 2026-09-05 审计 F1 项，未修复）；family_runtime_v1 的 authorization-sha 纪律良好但请抽查。若 receipt 可变，本清单第 110-125 行的 SHA 快照对旧根的约束力有限——建议第三方对 P0 级 receipt 先记哈希再读内容。
- **P1-11 提交面选择卫生**：ext6 候选（S1/EMA e8，0.3991）是在官方面上做 epoch 挑选的产物，作 leaderboard 提交可接受，但在任何论文主张中必须标注 dev-on-official-selected（ADDENDUM-7：held-out 面为最高评判，但在其上做选点即消耗其 pristine 性）。待提交清单中该候选与 M1 current_query e6 同属 SELECTED_BEFORE_EVALAI，状态应随提交动作更新。

### 增补材料指针

- 协调组对 dual_track 的实现审核：`results/m2_dual_track_v1/20260905_101500/AUDIT_IMPLEMENTATION_REVIEW_20260905.md`（含 F1/F2/F3 发现项）
- B 臂轨迹不稳定性机制取证：同目录 `ANALYSIS_B_TRAJECTORY_INSTABILITY.md`（权重位移 2–20%/epoch、跨臂 delta 相关 −0.395、minival 10-20 塌陷劫持）
- S0/S1 选择器失效诊断：`results/m2_b_small_stability_v1/20260905_123000/ANALYSIS_SMALL_STABILITY_REVIEW_NOTE.md`
- 581919 机读审计：`results/family_runtime_v1/m2_581919_default_payload_audit_v1.json`
| M2 QueryAge cached proof | `bb3116055f7cd289715b4acadd720c17c5a785d89141eac563df78fd25589701` |
