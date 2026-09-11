# M1 muscle：多 seed 复现与第四列 mean-rate 候选计划

## 决定和边界

用户已认可 M1 `muscle_response16_svd4/global_rms` R100 FULL seed42 的 official `test_split_m1` Held Out R² `0.6259910151098528`（`582205`）。原始 muscle decoder seed43/44 与单独 fourth-column mean-rate candidate seed42 均已完成，以下只保留其历史证据。当前论文 M1 主线固定为该 FULL `582205`；停止进一步 M1 carrier、结构或 seed 搜索。H1 不在本计划内；截至 2026-09-10，M2/H1 后续执行状态见 [M2/H1 ablation handoff](M2_H1_ABLATION_HANDOFF_20260910.md)，后续训练与提交由用户指定的另一 agent 负责。

本计划与后续的正式信息消融分开。M1 `ACTIVITY_ONLY`/`NONE` 的 ablation slot、pack 和 EvalAI official 验收由 [最终信息消融合同](FINAL_ABLATION_OFFICIAL_PLAN_20260909.md) 定义，不计入此处 original-muscle 多 seed 或 fourth-column candidate 的 run。

这不是对原 v4 mixed queue 的恢复。该队列已停止，未开始的 M1 v4 concat/P16/P32 不执行；本计划的 M1 runs 由新的、独立的 runner/artifact identity 承担。original seed42/43/44 和 fourth-column candidate seed42 的正式 24-epoch 训练与 full score 均已完成。候选仍只是一组配对 seed 证据，不能写成非劣、普遍有效，或已解释任何既有分数。

新的 M1 carrier、结构或 seed 想法不再进入本计划；已完成的 mean-rate4 candidate 也不构成新的实验授权或论文主线替换。

seed42 muscle 已有完整公开 HO3 24-epoch scan，并以 e3 选择；其 official 结果见 [OFFICIAL_582205.json](../../tfpd_exploration/submissions/evalai_m1_rift_muscle_r100_v1/artifacts/OFFICIAL_582205.json)。官方 private test 不参与本计划任何 checkpoint、carrier 或 seed 的选择。

同一 original muscle-response/SVD4 recipe 的 official `ACTIVITY_ONLY` `582219` 已完成：`test_split_m1` Held Out R² `0.5709008472436213`、Held In R² `0.793379687455571`、latency `0.1413142769159171`；FULL `582205` 为 HO `0.6259910151098528`、HI `0.7926300410539316`、latency `0.14040520785877766`，FULL−ACT HO 为 `+0.05509016786623144`。两者同属 phase `4599`。只读 [official receipt](../results/final_ablation_official_v1/root_m1_full_activity_official_readonly_20260910T031416Z.json)（SHA-256 `d1bf796f13b752997e5c7ae0a5fdbbeaeb87ff558203c217d59ffb1241e6fe6a`）仅作 direct `GET` 的 result、ACT payload 与 submit-server binding 核验，未 push、register 或 submit，不改变 ROOT 的提交责任边界。FULL→ACT 联合移除 direct T 与 carrier-conditioned E0，故它是行为 carrier 整体信息路径消融，不识别 CARRIER_ONLY 或单一路由的因果作用。mean-rate4 candidate 现有独立 official `582228`：HO `0.6135910493256053`、HI `0.7954639949771956`、latency `0.1395917318632367`，见 [OFFICIAL_582228.json](../../tfpd_exploration/submissions/evalai_m1_rift_mean_rate4_r100_v1/artifacts/OFFICIAL_582228.json)。相对 muscle FULL `582205` delta HO `-0.012399965784247546`，相对 ACT `582219` `+0.042690202081984`；不替换消融 FULL。不得将 public HO3 与 official HO 相减，也不得将 multi-seed sample SD 当作置信区间或显著性结论。M1 `NONE` official `582224` 为 finished，见 [OFFICIAL_582224.json](../../tfpd_exploration/submissions/evalai_m1_rift_none_r100_v1/artifacts/OFFICIAL_582224.json)；v2 未提交。

两个未实施的后续 M1 carrier idea 只保留为历史提案，不 build carrier、query、decoder 或训练。其一已完成 source-only、in-memory centered-SVD geometry 检查，未保存 basis；见 [historical proposal](M1_JOINT_RESPONSE_CARRIER_PROPOSAL_20260910.md)（SHA-256 `ec41d948d6ad95550a1091530d8c164326e1e0fb671f881183acac29e2741c6f`）及 [geometry receipt](../results/m1_muscle_r100_multiseed_v1/root_centered_svd_source_geometry_v1.json)（SHA-256 `49c72aeb9669900a44c406dc69d74977ee45864b6e2ba455357066eb6db90684`）。


M1 三臂的 current public HO3、seed42、channel-variance-weighted R² 审计已完成：FULL e3 `0.5690750181674957`、ACTIVITY_ONLY e3 `0.5618270536263784`，NONE 自身 24-epoch 曲线选 e1 `-1.2152107258637745`，ACTIVITY_ONLY − NONE `+1.7770377794901528` 且三个 session 均为正。三臂各自按完整 24 epochs 选择，见 [formal audit](../results/final_ablation_official_v1/root_m1_full_activity_none_formal_result_audit_v1.json)（SHA-256 `33ba82c6c34512459eb7b39abbcea4e853b61f7689897018d45c732fe5f36b21`，`PASSED_COMPLETE_THREE_ARMS_ALL24_SAME_CURRENT_QUERY`）及[本地摘要](../results/final_ablation_official_v1/m1_ablation_local_summary_v1/summary.json)。该结果支持此配置依赖 activity-derived E0，但只有单 seed 且 selected epoch 不同，不能作普适机制结论。ACT→NONE 移除 activity E0；FULL→ACT 仍联合移除 direct T 与 carrier-conditioned E0。该 local 结果不与 official FULL/ACT 或其 `0.05509016786623144` 差混合或相减；v1 的 host import 失败已由 v2 local package 修复并验证，但 v1 official `582224` 最新观测（`2026-09-10T04:35:18Z`）为 FAILED、没有 scored result，失败原因 `UNDETERMINED_PENDING_PROTOCOL_AND_SCORING_DIAGNOSIS`；v2 未提交；见 [M1 NONE failure evidence](../results/final_ablation_official_v1/root_m1_none_official_failure_evidence_v1.json)。

## A：原始 muscle carrier 的 decoder seed43/44

此部分复现的是已提交的 `muscle_response16_svd4/global_rms` carrier，不改估计器、数据读取、carrier pack、source roster、M10 support、SVD basis、normalizer 或 B3S 初始化来源。新增的变化仅为真实的 decoder initialization seed 与对应 dropout RNG seed：分别为 `43`、`44`。sampler 固定为 `42`。

每个 seed 均固定以下合同：

| 项目 | 固定值 |
| --- | --- |
| task / decoder | M1 RIFT R100/D4 concat，与 seed42 muscle 同一正式 recipe |
| 训练长度 | 24 epochs、同一 updates-per-epoch、optimizer、learning-rate schedule、EMA 与 unit-dropout 规则 |
| sampler | seed `42`，不得随 decoder seed 改变 |
| carrier | 已封存的原 muscle carrier，逐字节绑定既有 pack/fit |
| B3S | 同一预训练初值；训练中仍可训练，不冻结，也不重置为新 encoder |
| 数据与评估 | 同一 source contract、HO3 window/target/start contract 与 channel-centered variance-weighted R² |

每个 seed 必须同时报告：固定 e3 的 HO3 equal-session mean、该 seed 全 24-epoch scan 的 earliest-maximum selected epoch 和 selected mean。三 seed 的汇总只在三个完整 run 都存在后报告 e3 mean/std 与 selected mean/std；不得用 seed42 的 official result、任何 private test 指标或不完整 run 选择 epoch/seed。

## B：单独第四列的 mean-rate 候选，seed42

候选只改变原 muscle carrier 的第四列。对每个 unit，用与原 muscle profile **同一 session、同一 M10 rows、同一对齐 reader**的 arithmetic rate mean 建立

\[
m_u=\frac{1}{n}\sum_{t\in\mathrm{M10}}r_{tu}.
\]

在四个 source session 的 unit rows 上，对该 `m` 使用 source4 mean 和 standard deviation 中心化/缩放，使新第四列的 source-scale 与旧第四列匹配。前三列必须保持原 SVD carrier 的最终 `float32` 值逐字节一致；不得同时改变 reader、SVD basis、前三列 scale、source roster、M10 definition、H1 或 decoder recipe。实现必须把 source4 统计量、rows identity、旧/新前三列 byte-equality 与第四列的 scale binding 写入 receipt。

该已完成候选检验的是一个窄问题：在保持前三列不变时，以同一 M10 的 arithmetic unit mean-rate 取代原 SVD4 的第四坐标，是否改变固定 recipe 的公开 HO3 表现。它不把 rate mean 解释为因果 baseline，也不把结果归因于 SVD、calibration 或任何单独生理机制；它不替换论文 M1 主线 FULL `582205`。

候选只运行 seed42，并同样报告 fixed e3 与该 run 全 24-epoch scan 的 earliest-maximum selected epoch。它与 original-muscle seed42 的比较必须使用相同 HO3 targets、window starts、metric 与 selection contract；official `582205` 只作为原始 muscle seed42 的独立 private-test 记录，不能进入比较或选择。

M1 主比较采用每个方法在相同完整 all-24 公开 HO3 规则下各自 earliest-maximum selected checkpoint；fixed e3 只作补充诊断，不要求同 epoch，也不推翻主比较。

### Candidate seed42 的实际公开 HO3 结果

candidate seed42 已完成 24 epochs / 159,960 updates 和 full HO3 score。成对曲线、逐 epoch CSV、图和摘要见 [candidate_seed42_pair_curves_v1](../results/m1_muscle_r100_multiseed_v1/candidate_seed42_pair_curves_v1/README.md)。同一配对的 original seed42 在 e3 选中，mean 为 `0.569075018167`；candidate 在 e4 选中，mean 为 `0.5876280069351196`，其各自选中 checkpoint 的差为 `+0.0185529887676239`。candidate selected 相对 ACTIVITY_ONLY e3 `0.5618270536263784` 为 `+0.025800953308741215`，fixed e3 candidate `0.5861498316129049` 相对 ACTIVITY_ONLY 为 `+0.02432277798652649`；candidate 仍比历史 rSyn3 FULL `0.654471` 约低 `0.066843`，未解决强基线差距。

| 比较 | candidate − original（公开 HO3 指标） |
| --- | ---: |
| 各自选中 checkpoint（candidate e4，original e3） | `+0.0185529887676239` |
| 固定 e3 | `+0.01707481` |
| session `20121004`（各自选中 checkpoint） | `+0.013209998607635498` |
| session `20121017`（各自选中 checkpoint） | `+0.0036240220069885254` |
| session `20121024`（各自选中 checkpoint） | `+0.03882494568824768` |
| e24（candidate `0.505849440893`，original `0.513130525748`） | `-0.007281084855397468` |

candidate 在相同 epoch 的 e3–e21 共 19 个 epoch 高于 original，但 e24 差为 `-0.007281084855397468`。candidate selected 相对 ACTIVITY_ONLY 的三个 session 均为正；该 3/3 仅指各自 selected，fixed e3 对 ACTIVITY_ONLY 仍有一个 session 下降。original muscle 相对 ACTIVITY_ONLY 则为 2/3 session 负、平均为正，二者不应混称为“当前 muscle”增量。该单一配对 seed 支持在早期选中 checkpoint 保留显式 level 坐标的有限证据；它不证明 SVD 的普遍改进、与历史 rSyn3 的单因素可比性、后期训练稳定性、因果机制或 official 表现。FULL−ACTIVITY_ONLY 联合反映 side/E0 conditioning 与 direct T，且没有已完成 CARRIER_ONLY 结果，不能拆为独立 route 效应。[M1 carrier context comparison audit](../results/m1_muscle_r100_multiseed_v1/root_m1_carrier_context_comparison_v1.json) 已核验既有 receipt 算术、all-24 同 query 与 candidate recipe binding。

## 已知 evidence 与不会越过的解释界限

original muscle 的 seed42 public HO3 直接比较已将 original muscle FULL 与已完成 B ACTIVITY_ONLY 绑定到同一 seed、targets、window starts/count 和完整 24-epoch scan；二者均选 e3。muscle mean 为 `0.5690750181674957`，B 为 `0.5618270536263784`，差为 `+0.0072479645411173505`，见 [root_muscle_vs_activity_seed42.json](../results/m1_muscle_r100_multiseed_v1/root_muscle_vs_activity_seed42.json)。三个 session 的差是 `-0.0020035505294799805`、`+0.032085299491882324`、`-0.008337855339050293`：最差点估计在 `-0.01` 内。original muscle 的公开增量有限；它仍只是三个 session、seed42 的观测点，不是统计非劣证明、official 不劣声明或真实无 shift 的证据，也不应被误指为 mean-rate4 candidate 的结论。

历史 rSyn3 FULL `0.654471` 与 ACTIVITY_ONLY `0.561827` 的 e3 public HO3 差为 `0.092644`，是另一 carrier reference 下的完整证据。它显示 calibration 的端到端作用依赖 carrier 选择，不能被当前 muscle 小增量推广为“calibration 已证明没有意义”。muscle 与 rSyn3 的约 `-0.085396` 同样只属于公开面；缺少同配置 rSyn3 completed official baseline，不能换算成 official 相对结论。

现有 muscle profile 没有硬 identity fallback：它从当前 M10 重新计算 unit mean rate、Poisson scale 和 conditional response，再投影/归一化 carrier。因而不能声称“estimated shift=0 时已有数学无损保证”。`estimated shift=0` 也不等于真实 distribution shift 为零。只有另行实现并审计固定 baseline `f0` 与严格 residual `f(x,c)=f0(x)+g(x,c)`、`g(x,0)=0` 后，才可保证该规则实际取 `c=0` 时逐输入复现 `f0`；这不是本计划的既有功能，也不是本计划要宣称的结果。

## 完成判据

本计划的 original seed42/43/44 与 mean-rate 第四列 candidate seed42 均已具备完整训练、24-epoch score 和汇总证据；结论仍只在同一 public HO3 评分面上书写。任何结果都必须区分“三 seed 原始 muscle 复现”和“单 seed 候选”。

## 执行状态快照（2026-09-09）

候选 carrier 的 CPU build 与 loader 验证已通过。前三列与原始 carrier 的最终 `float32` 字节完全一致；第四列使用同一 source-centered 标准化约束后，旧列的 source standard deviation 为 `0.225242578165926`，新列为 `0.22524257573695483`。证据见 [root_carrier_build_verification.json](../results/m1_muscle_r100_multiseed_v1/root_carrier_build_verification.json)。

前三列在旧 normalized 4D source-centered variance 中占 `98.7316445%`，见 [root_candidate_geometry.json](../results/m1_muscle_r100_multiseed_v1/root_candidate_geometry.json)。该比例只描述这四个旧 normalized 坐标中的 source-centered 方差分配；它不代表 raw 16 维信息保留比例，也不代表预测价值。

ROOT 已完成四组 GPU1 两步 smoke：original seed42、seed43、seed44 与 candidate seed42 均通过。original seed42 的初始化 SHA 与既有正式运行完全相同；seed43/44 的 decoder 初值彼此真实不同；candidate seed42 与 original seed42 使用相同初始化。streaming 误差的最大值小于 `2e-7`。证据见 [root_smoke_verification.json](../results/m1_muscle_r100_multiseed_v1/root_smoke_verification.json)。这些是构建和短程一致性证据，不构成完整训练或性能结论。

正式独立队列现为 `COMPLETED`；变更前的旧队列仍存档为 [root_formal_queue_before_priority_change.json](../results/m1_muscle_r100_multiseed_v1/root_formal_queue_before_priority_change.json)。严格汇总已核对四个 run 的 source、sampler、B3S、HO targets/starts 与 `3881` windows 合同均通过；汇总、逐 epoch CSV、图和说明见 [summary_all4_v1](../results/m1_muscle_r100_multiseed_v1/summary_all4_v1/README.md)。

| run | fixed e3 | selected epoch / mean |
| --- | ---: | ---: |
| original seed42 | `0.5690750181674957` | e3 / `0.5690750181674957` |
| original seed43 | `0.5517233908176422` | e3 / `0.5517233908176422` |
| original seed44 | `0.5443696975708008` | e6 / `0.5669176677862803` |
| candidate seed42 | `0.5861498316129049` | e4 / `0.5876280069351196` |

三个 original seed 的 fixed-e3 mean / sample SD（`ddof=1`）为 `0.5550560355186462` / `0.012685350092674524`，范围为 `0.5443696975708008`–`0.5690750181674957`；各自 selected mean / sample SD 为 `0.5625720255904727` / `0.00945691268292847`，范围为 `0.5517233908176422`–`0.5690750181674957`。candidate 只有一个 seed，虽高于三个 original 的 selected 值，仍只支持保留显式 level 候选，不构成 candidate 三 seed 复现、普遍/因果解释或 official 改进。
