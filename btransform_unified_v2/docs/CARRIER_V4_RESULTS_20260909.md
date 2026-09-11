# Carrier v4：已完成证据与实验进度

本报告保留已经完成的 M1 机制诊断、P0 数值复现、共享 core 的 M1 exact 部署、H1 source fit/27-tag bank、融合和信息路由审计，以及 M1 ACTIVITY_ONLY 全 24 epoch 重评分。H1 FULL/P16 已完成 32 epochs、23,392 updates 并保存 32 个 checkpoint，但用户已停止 H1 v4 新方法：SIGTERM 时 checkpoint score 只完成 28/32，不能产生正式 selected result。训练记录仍见 [metrics.jsonl](../results/carrier_v4/h1_full_proj_add_p16_formal_v1/metrics.jsonl)，停止记录见 [user_stop_h1_v4_20260909.json](../results/carrier_v4/user_stop_h1_v4_20260909.json)。混合队列已经停止，不会执行其余 H1 v4 或未开始的 M1 v4 fusion cells。后续六个信息消融的唯一范围、验收和官方结果要求见 [最终信息消融合同](FINAL_ABLATION_OFFICIAL_PLAN_20260909.md)；其中 H1 的有限 signed-state ACTIVITY_ONLY/NONE 例外不恢复 v4 NMF/fusion path。

当前 [机器汇总](../results/carrier_v4/summary_progress_v2/summary.json) 和 [逐 epoch CSV](../results/carrier_v4/summary_progress_v2/curves.csv) 保留最终 `experiment_manifest_v2` 的历史状态。它们列出的 pending cells 不构成继续执行承诺。完成评分的 M1 两个复用单元和 H1 external baseline 仍可审计；[M1 图](../results/carrier_v4/summary_progress_v2/figures_v3/m1_results.svg) 与 [H1 图](../results/carrier_v4/summary_progress_v2/figures_v3/h1_results.svg) 只表示当时已完成的数值。已完成的 M1 multi-seed/mean-rate4 工作保留在 [multi-seed carrier plan](M1_MUSCLE_MULTISEED_CARRIER_PLAN_20260909.md) 作为历史证据，不复用 stopped queue，也不继续扩大 M1 carrier、结构或 seed 搜索。

## Official FULL vs ACTIVITY_ONLY（EvalAI test）

论文主对比统一为 official FULL 对 official `ACTIVITY_ONLY`；M1 FULL 固定 muscle `582205`。官方差只对同一 task 的 FULL official ID 计算；本地 HO3 / EXT6 / HO-M3 不得与 official HO 相减。FULL→ACT 同时去掉 direct T 与 carrier-conditioned E0，不能据此拆出单独路由效应。对照图见 [official_full_vs_activity.png](../results/final_ablation_official_v1/official_full_vs_activity_v1/official_full_vs_activity.png)。M1 `NONE` official `582224` 现已 finished：HO `-0.6772461160545928`、HI `0.598250857063468`、latency `0.13951016955707982`，相对 FULL `582205` delta HO `-1.3032371311644456`，相对 ACT `582219` delta HO `-1.2481469632982141`。收据 [OFFICIAL_582224.json](../../tfpd_exploration/submissions/evalai_m1_rift_none_r100_v1/artifacts/OFFICIAL_582224.json)。metadata GET `result=null`，官方指标来自 `submission_result_file`。较早将 `582224` 记为 FAILED / 无 scored result 的 ROOT snapshot 为历史并已被取代。local HO3 selected e1 mean `-1.2152107258637745` 不是 official。v2 未提交。M2/H1 `NONE` 仍无 finished official。排除 `582220` / `582221`。

| task | FULL id / HO | ACT id / HO | delta HO |
| --- | --- | --- | ---: |
| M1 | `582205` / `0.6259910151098528` | `582219` / `0.5709008472436213` | `-0.05509016786623144` |
| M2 | `582189` / `0.34654225938843214` | `582217` / `0.09302905280313856` | `-0.2535132065852936` |
| H1 | `582196` / `0.45665779063606854` | `582218` / `0.3445956103610151` | `-0.11206218027505344` |

| task | FULL id / HO | ACT id / HO | NONE id / HO | NONE−FULL delta HO |
| --- | --- | --- | --- | ---: |
| M1 | `582205` / `0.6259910151098528` | `582219` / `0.5709008472436213` | `582224` / `-0.6772461160545928` | `-1.3032371311644456` |

M1 这一对由只读回执 [root_m1_full_activity_official_readonly_20260910T031416Z.json](../results/final_ablation_official_v1/root_m1_full_activity_official_readonly_20260910T031416Z.json) 核验：仅对 phase `4599` 的 `582205`、`582219` 做 direct `GET`，FULL/ACT result、ACT payload 与 submit-server binding 均通过；回执 SHA-256 为 `d1bf796f13b752997e5c7ae0a5fdbbeaeb87ff558203c217d59ffb1241e6fe6a`。核验没有 push、register 或 submit，且不改变 ROOT 的提交责任边界。M1 `NONE` official `582224` 现为 finished，见 [OFFICIAL_582224.json](../../tfpd_exploration/submissions/evalai_m1_rift_none_r100_v1/artifacts/OFFICIAL_582224.json)；较早 FAILED / 无 scored result 的 ROOT snapshot 为历史并已被取代。v2 未提交。mean-rate4 candidate 现有独立 official `582228`：HO `0.6135910493256053`、HI `0.7954639949771956`、latency `0.1395917318632367`，收据 [OFFICIAL_582228.json](../../tfpd_exploration/submissions/evalai_m1_rift_mean_rate4_r100_v1/artifacts/OFFICIAL_582228.json)。相对 muscle FULL official `582205` 的 delta HO 为 `-0.012399965784247546`，相对 ACT official `582219` 为 `+0.042690202081984`。它不替换消融表里的 FULL `582205`。local HO3 selected e4 mean `0.5876280069351196` 不是 official，不得与 official HO 相减。

## M2 RIFT concat 的 official ACTIVITY_ONLY 与本地三臂

M2 `ACTIVITY_ONLY` official `582217` 已 finished：HO `0.09302905280313856`、HI `0.6463020945769177`、latency `0.10743839747764272`。相对 FULL official `582189` HO `0.34654225938843214`，delta HO 为 `-0.2535132065852936`。收据 [OFFICIAL_582217.json](../../tfpd_exploration/submissions/evalai_m2_rift_activity_only_r50_v1/artifacts/OFFICIAL_582217.json)。官方对照图见 [official_full_vs_activity.png](../results/final_ablation_official_v1/official_full_vs_activity_v1/official_full_vs_activity.png)。

M2 FULL、ACTIVITY_ONLY 与 NONE 另有独立训练、24 epochs / 75,960 updates 和全部 24 个 EMA 的同规则 EXT6 选择；三线共享 seed42、sampler、六个 EXT6 session 和 15,403 个 query windows。各自 selected 结果为 FULL e9 `0.3900576650553506`、ACTIVITY_ONLY e9 `0.18402467250439059`、NONE e6 `-0.012659512830027883`。因此 ACTIVITY_ONLY − FULL 为 `-0.20603299255096003`，NONE − FULL 为 `-0.4027171778853785`，ACTIVITY_ONLY − NONE 为 `+0.19668418533441848`。FULL 对 ACTIVITY_ONLY 和 NONE 均为 6/6 session 更高；ACTIVITY_ONLY 对 NONE 为 5/6 更高，例外是 `ses-2020-11-19-Run1`：FULL `0.24492190332616237`、ACTIVITY_ONLY `-0.21598297716573844`、NONE `-0.009663309684460941`，ACTIVITY_ONLY − NONE 为 `-0.2063196674812775`。三臂同 query 的完整审计见 [M2 FULL/ACTIVITY_ONLY/NONE formal audit](../results/final_ablation_official_v1/root_m2_full_activity_none_formal_result_audit_v1.json)。

这些单 seed、本地 public EXT6 结果是 local、非 official。三线各自按同一 all-24 规则选择，selected epoch 可以不同；共同 fixed e9 的 NONE 为 `-0.03969433572921879`，但不替代各自 selected 的主比较，也不得与 official HO `0.346542` / `0.093029` 混口径或相减。M2 NONE 本地 pack 与只读预检已通过、未提交，没有 official；预检收据见 [root_submit_preflight_receipt.json](../../tfpd_exploration/submissions/evalai_m2_rift_none_r50_v1/artifacts/root_submit_preflight_receipt.json)。

## 已知的 M1 性能差异：历史 rSyn3 参考与当前 muscle 对照

主指标均为每个 HO calibration session 的 channel-centered variance-weighted R²，再对三个 session 等权平均。FULL rSyn3、ACTIVITY_ONLY 与 muscle 三条线都完整扫描 24 个 EMA，均选 e3。

| 已完成模型 | e3 / selected mean R² | 相对 FULL rSyn3 |
| --- | ---: | ---: |
| FULL rSyn3，live B3S + direct T | 0.654471 | 0 |
| ACTIVITY_ONLY，live B3S(raw, 0)，direct T=0 | 0.561827 | −0.092644 |
| muscle-response16 / SVD4 / global-RMS | 0.569075 | −0.085396 |

ACTIVITY_ONLY 是原始完整训练的 B arm，不是将 FULL/D checkpoint 的输入清零。训练审计 [root_activity_only_training_audit.json](../results/m1_muscle_r100_multiseed_v1/root_activity_only_training_audit.json) 绑定 fresh `--arm B_ACTIVITY_ONLY --stage train --epochs 24`、无 resume 的 24 epochs/159,960 updates、每 epoch checkpoint SHA 与训练日志、e1/e3/e24 的实际 optimizer steps，以及 B 与 D 各自训练后 state 的差异；两 arm 共享相同 seed42 decoder 初始化和 B3S 预训练初值（initialization SHA `02f056...`），随后独立优化各自的 encoder/decoder。B code 始终同时将 side T 与 direct T 置零。此次 [B replay receipt](../results/carrier_v4/m1_activity_baseline_replay_v1/replay_receipt.json) 重评分的是这个独立训练的 B 的全部 24 个 checkpoint，原 legacy metric 的误差全部为零，并与 [D replay authority](../results/m1_muscle_r100_v1/baseline_replay/replay_receipt.json) 核验相同 source、初始化、HO targets 和窗口。B 仍使用 M10 的无标签神经活动来生成 E0，所以 FULL/B 的准确解释是**行为响应 carrier 的额外作用**，不是“有 calibration”对“完全无 calibration”。B/D 每个 session 分数如下：

| Session | FULL rSyn3 | ACTIVITY_ONLY | ACTIVITY_ONLY − FULL |
| --- | ---: | ---: | ---: |
| 20121004 | 0.749535 | 0.702689 | −0.046846 |
| 20121017 | 0.607642 | 0.506664 | −0.100978 |
| 20121024 | 0.606236 | 0.476128 | −0.130107 |

因此，**对历史 rSyn3 reference**，当前 seed42 evidence 不支持“移除 carrier 后非劣”：三个 session 全部下降，差异大于事前声明的 R² 0.01 容忍度。这个结论不能直接套用到当前 muscle carrier。NONE/CARRIER_ONLY 与 fusion-width cells 不会由原 stopped queue 补跑；rSyn3 FULL/B 公开比较也不能替代未实施信息臂的结论。

对该历史 rSyn3 三个 session 的配对差异进行全部 `3^3=27` 种等概率有放回重采样，均值差为 −0.092644，单侧 95% 下界为 −0.120398，双侧 95% 区间为 [−0.123796, −0.058575]。fixed e3 与 selected-each 两种比较在这组已完成结果上相同；该区间仅描述 rSyn3 reference 下的三个公开校准 session 和 seed42。

### 当前 muscle FULL 对已完成 B ACTIVITY_ONLY

当前 muscle 方案的直接公开对照不是将 muscle 与 rSyn3 的不同 carrier/reader/normalizer 合同相减，而是同 seed42 的 original muscle FULL 与已完成 B ACTIVITY_ONLY。ROOT 已核对同一 HO3 targets、window starts、window counts 和完整 24-epoch scan；两者各自 selected epoch 均为 e3。进一步的 [M1 FULL/ACTIVITY_ONLY reference-binding audit](../results/final_ablation_official_v1/root_m1_full_activity_reference_binding_v1.json) 指定 canonical muscle FULL 为 `m1_muscle_r100_v1/formal_s42_gpu1`，ACTIVITY_ONLY 为独立 B_ACTIVITY_ONLY run `rift_v1/m1_r100_joint_b_s42_formal_v1` 的当前 replay，并重核 48 个 checkpoint 文件 hash、21 个固定 recipe 字段、相同 HO3 三 session / 3,881 query targets、starts/counts 与 B3S source-file SHA。历史 old-D/rSyn replay 仅作为 shared-query provenance，不是 canonical FULL；审计只复核既有 JSON/hash，不运行新 forward 或模型，也不产生 NONE 或 official 结果。

| Session / aggregate | muscle FULL | B ACTIVITY_ONLY | muscle − B |
| --- | ---: | ---: | ---: |
| 20121004 | — | — | −0.0020035505294799805 |
| 20121017 | — | — | +0.032085299491882324 |
| 20121024 | — | — | −0.008337855339050293 |
| HO3 equal-session mean | 0.5690750181674957 | 0.5618270536263784 | +0.0072479645411173505 |

原始 muscle 的公开增量因此有限：三个 session 中最差的点估计为 `-0.008337855339050293`，仍在 `-0.01` 内；但 n=3、单 seed 的点估计不是统计 non-inferiority 证明，不能据此声称 official 不劣或将 M1 定义为真正无 shift。这个结论只适用于 original muscle，不应被误指为后述 mean-rate4 candidate。历史 rSyn3 FULL/B 大差异说明 carrier 选择会改变行为响应 carrier 在端到端输入中的额外作用，不能把 original muscle 的小增量泛化为 calibration 总体无意义。

### 当前 M1 三臂 local HO3：ACTIVITY_ONLY 对 NONE

同一 current public HO3、seed42、channel-variance-weighted R² 评分面上，FULL e3 为 `0.5690750181674957`，ACTIVITY_ONLY e3 为 `0.5618270536263784`，NONE 自身 24-epoch 曲线选 e1 为 `-1.2152107258637745`；ACTIVITY_ONLY − NONE 为 `+1.7770377794901528`，三个 session 均为正。三臂各自按完整 24 epochs 选择，审计 [root_m1_full_activity_none_formal_result_audit_v1.json](../results/final_ablation_official_v1/root_m1_full_activity_none_formal_result_audit_v1.json)（SHA-256 `33ba82c6c34512459eb7b39abbcea4e853b61f7689897018d45c732fe5f36b21`，`PASSED_COMPLETE_THREE_ARMS_ALL24_SAME_CURRENT_QUERY`）及[本地摘要](../results/final_ablation_official_v1/m1_ablation_local_summary_v1/summary.json)保留相同 query 的证据。该严重下降支持此配置依赖 activity-derived E0；它仍是单 seed 且 selected epoch 不同，不能外推为普适机制。FULL→ACT 联合移除 direct T 与 carrier-conditioned E0；ACT→NONE 则移除 activity E0。该 local 结论不与 official FULL `0.6259910151098528`、ACT `0.5709008472436213` 及其 `-0.05509016786623144` 差、或 official NONE `582224` HO `-0.6772461160545928` 混合或相减。v1 的 host import 失败保留为历史；修正后的 v2 local package 为 `tfpd_exploration/submissions/evalai_m1_rift_none_r100_v2`，镜像 `m1-rift-none-r100-e1:v2`（`sha256:1110e2acc27bd41a09c0db5fc086e3bff5625392b55dde29e3f07d5bfb2a2580`）、payload `34083bb09cd4145df17666640edb769641921b99419031668f0b2c002864588f`，host 最大误差 `1.6689300537109375e-6`、container smoke `0`、33 files exact，见 [corrected manifest](../results/final_ablation_official_v1/m1_none_image_metadata_v3/corrected_manifest.json)（SHA-256 `5267ef308adf112be015904c7818dba696443be03225f7a3299c447174b97fc9`）。既有 v1 official `582224` 由 ROOT 在 `2026-09-10T03:58:20Z` 只读 GET 核验为 `running`、无 result，见 [official receipt](../results/final_ablation_official_v1/root_m1_none_official_readonly_20260910T035820Z.json)（SHA-256 `6eb2678ca3d2fe8c6dce9c8983ab884e874fbdcfa9ad11f8333420362ef8c7cd`）；该 snapshot 与其后 FAILED / 无 scored result 的 [failure evidence](../results/final_ablation_official_v1/root_m1_none_official_failure_evidence_v1.json)（SHA-256 `e43ef3618b0d46809cd8845cf7ed112af316e302fbfa6c0d2d1cc9b5c1c0f330`）均为历史并已被取代。当前 official 为 finished，见 [OFFICIAL_582224.json](../../tfpd_exploration/submissions/evalai_m1_rift_none_r100_v1/artifacts/OFFICIAL_582224.json)。它绑定 v1 payload `c572ebc1debefa64cfec74b788c74191f227880219a258dabae80ccac9e9de86`，不是 ROOT 提交。v2 未提交；[recovery binding](../results/final_ablation_official_v1/root_m1_none_pack_recovery_binding_v2.json)（SHA-256 `a8bd4a5d2a65faea7f67581adf8e1b659a013cca83edc91eccc80c91bb3be960`）确认两个 payload 对 selected e1 的内容完全一致，只是 pickle bytes 不同。

### 现行 M1 多-seed 状态

original seed42/43/44 与 mean_rate4 candidate seed42 均已完成 24-epoch / 159,960-update train 和 full 24-epoch public HO3 score。严格汇总确认四个 run 的 source、sampler、B3S、HO targets/starts 与 `3881` windows 合同全部通过；汇总、曲线和逐 epoch数据见 [summary_all4_v1](../results/m1_muscle_r100_multiseed_v1/summary_all4_v1/README.md)。这是独立 multi-seed plan 的完成状态，不恢复 stopped v4 queue。

| run | fixed e3 | selected epoch / mean |
| --- | ---: | ---: |
| original seed42 | `0.5690750181674957` | e3 / `0.5690750181674957` |
| original seed43 | `0.5517233908176422` | e3 / `0.5517233908176422` |
| original seed44 | `0.5443696975708008` | e6 / `0.5669176677862803` |
| candidate seed42 | `0.5861498316129049` | e4 / `0.5876280069351196` |

original 三 seed 的 fixed-e3 mean / sample SD（`ddof=1`）为 `0.5550560355186462` / `0.012685350092674524`，范围为 `0.5443696975708008`–`0.5690750181674957`；各自 selected mean / sample SD 为 `0.5625720255904727` / `0.00945691268292847`，范围为 `0.5517233908176422`–`0.5690750181674957`。

M1 主比较使用每个方法在相同完整 all-24 公开 HO3 规则下各自 earliest-maximum selected checkpoint；fixed e3 保留为补充诊断，不作为推翻主比较或要求同 epoch 的门槛。

| 公开 HO3 比较 | 差值 / 说明 |
| --- | --- |
| candidate selected e4 `0.5876280069351196` − original selected e3 | `+0.0185529887676239` |
| candidate selected e4 − ACTIVITY_ONLY e3 `0.5618270536263784` | `+0.025800953308741215`；三个 session 均为正：`+0.011206448078155518`、`+0.03570932149887085`、`+0.030487090349197388` |
| candidate fixed e3 `0.5861498316129049` − ACTIVITY_ONLY e3 | `+0.02432277798652649` |
| candidate selected e4 − 历史 rSyn3 FULL `0.654471` | 约 `-0.066843`；尚未解决强基线差距 |

candidate 仍只有 single-pair seed42 证据；虽然其 selected 相对 ACTIVITY_ONLY 为 3/3 session 正增量，original muscle 相对 ACTIVITY_ONLY 则为 2/3 session 负、平均值为正，这不是同一 carrier 的结论。candidate e24 相对 original e24 仍为 `-0.007281084855397468`，不能声称后期稳定性。上述 rSyn3 与 SVD carrier 的差异同时包含多个方法因素，不是单因素 carrier 归因；没有已完成的 CARRIER_ONLY 结果，FULL−ACTIVITY_ONLY 是 side/E0 conditioning 与 direct T 联合的端到端信息增量，不能拆作独立 CARRIER_ONLY 或 direct-route 效应。

[M1 carrier context comparison audit](../results/m1_muscle_r100_multiseed_v1/root_m1_carrier_context_comparison_v1.json) 已核验既有 receipt 的算术、all-24 同 query 与 candidate recipe binding。上表的 3/3 正增量仅指各自 selected checkpoint：fixed e3 candidate 相对 ACTIVITY_ONLY e3 的三个 session 差为 `-0.013899922370910645`、`+0.036054253578186035`、`+0.05081400275230408`，平均仍为 `+0.02432277798652649`，即仅 2/3 为正；不能把 selected 结果读为普遍无损或非劣证明。

[用户优先级决定](../results/m1_muscle_r100_multiseed_v1/root_user_priority_change.json) 已明确：seed43 相对 seed42 的变异要求完成 seed44，但不延后用户优先的 candidate。该决定未声明显著性、等价性或预设的数值阈值，不能把该顺序解释为对 seed 差异或 candidate 性能的统计结论。

[训练复用审计](../results/carrier_v4/m1_legacy_concat_reuse_audit_v1.json) 将旧 D/B 两个已完成训练分别绑定为 v4 FULL/ACTIVITY_ONLY concat 的 authority。它依据实际相同的 carrier、初始化、完整 forward、source/sampler/优化与 EMA 合同，以及原 formal run 和全 epoch replay；没有声称在新 wrapper 下重放全部 159,960 个 optimizer updates。新 runner 的 resume 仍拒绝旧 schema。

## M1 官方私有测试证据

muscle-response R100 的 e3 提交 **582205 已完成**，官方 `test_split_m1` 返回 Held Out R² **0.625991**（std **0.089112**）、Held In R² **0.792630**（std **0.021132**）、Normalized Latency **0.140405**。该 e3 在提交前由公开 HO3 的完整 24-epoch 扫描选定，官方结果未参与选择。ROOT 已通过官方 GET 与 `evalai submission 582205 result` 直接核实，完整精度与时间记录见 [OFFICIAL_582205.json](../../tfpd_exploration/submissions/evalai_m1_rift_muscle_r100_v1/artifacts/OFFICIAL_582205.json)。本地 REGISTERED 已同步为 finished，保留原 push/register 的历史 snapshot。

同一 muscle recipe 的 `ACTIVITY_ONLY` official `582219` 已 finished：HO `0.5709008472436213`、HI `0.793379687455571`、latency `0.1413142769159171`，相对 FULL `582205` 的 delta HO 为 `-0.05509016786623144`。收据 [OFFICIAL_582219.json](../../tfpd_exploration/submissions/evalai_m1_rift_activity_only_r100_v1/artifacts/OFFICIAL_582219.json)。FULL→ACT 同时去掉 direct T 与 carrier-conditioned E0，不能据此拆出单独路由效应。local HO3 selected e3 mean `0.5618270536263784` 不是 official，不得与 official HO 相减。M1 `NONE` official `582224` 现为 finished：HO `-0.6772461160545928`、HI `0.598250857063468`、latency `0.13951016955707982`，相对 FULL `582205` delta HO `-1.3032371311644456`。收据 [OFFICIAL_582224.json](../../tfpd_exploration/submissions/evalai_m1_rift_none_r100_v1/artifacts/OFFICIAL_582224.json)。local HO3 selected e1 mean `-1.2152107258637745` 不是 official。较早 FAILED / 无 scored result 的 ROOT snapshot 为历史并已被取代。v2 未提交。

相同 RIFT joint-D R100 concat e3 的 rSyn3 提交 **582150 实际为 cancelled**，没有官方分数，见 [STATUS_582150.json](../../tfpd_exploration/submissions/evalai_m1_rift_joint_d_cached_e3_v1/artifacts/STATUS_582150.json)。ROOT 还读取了 challenge 2319 / phase 4599 的完整 56 条本队提交记录，同名另一条 582133 也为 cancelled，未找到已完成的同配置 joint-D 官方结果。较早的其他 rSyn3 方法具有不同训练/模型合同，不能充当只替换 carrier 的官方配对基线。

因此，本文 muscle 相对 rSyn3 的 **−0.085396** 仅指公开 HO3，同一个 muscle checkpoint 在官方私有 test 上的 **0.625991** 是另一评估集合的实际读数。目前不能声称 muscle 在官方私有测试上劣于 rSyn3，也不能把不同集合分数的差当作方法改进或退化。下述机制诊断针对已观察到的公开 HO3 差异及表示结构，不能外推为已证实的官方性能解释。

## SVD4 公开 HO3 下降的机制证据

不能把整条 muscle 方案的差异都归因于 SVD。新旧方法同时改变了行为压缩位置、单元统计量、rate 去均值/缩放、截距、normalizer、字典的 source roster 和 spike bin 对齐。

[Source 诊断](../results/carrier_v4/diagnosis_m1_source_v1/diagnosis.json) 验证了新旧 movement EMG rows 与 trial IDs 一致。muscle raw SVD4 四轴的方差比例为 90.296%、6.481%、1.954%、1.268%，有效秩约 1.472；除以同一个 global RMS 不会改变这一比例。相比之下，rSyn3 四列通过 source per-column normalization 保持相近尺度。该结果说明当前 SVD4 表示高度集中，不能单独证明改成 per-column 就能恢复 decoder 分数。

旧 rSyn3 第四列是原始 Hz unit ridge 的不受惩罚截距，其 M10/next-M10 的单元间相关性约 0.960–0.986；muscle 第四 SVD 坐标的对应相关性约 −0.379–0.759。二者不具备相同的物理含义。冻结 rSyn3 e3 checkpoint 的 [输入干预](../results/carrier_v4/frozen_m1_e3_probe_v1/receipt.json) 同时处理 B3S side 和 direct T，得到：

| e3 固定模型的 T 干预 | Mean R² |
| --- | ---: |
| 原始 T | 0.654471 |
| 截距列归零 | 0.615020 |
| 三个 slope 归零 | 0.552040 |
| 全 T 归零 | 0.428706 |

原始分支逐字节复现原 e3 predictions。这是固定模型的输入依赖诊断，不是重训练消融；不能把各下降量当作可加的因果贡献。

P0 进一步只在旧 rSyn3 回归中打开当前 muscle 的 rate 标准化：

\[
z_{tu}=\frac{r_{tu}-\bar r_u}{s_u},\qquad
s_u=\sqrt{\max(0.02\bar r_u,1)}/0.02.
\]

由于截距不受惩罚、每个 unit 的 `s_u` 在 support 内恒定，对相同设计矩阵有：

\[
\beta'_u=\beta_u/s_u,\qquad
b'_u=(b_u-\bar r_u)/s_u.
\]

这不是仅改变四列的统一尺度：它明确去掉了 unit mean-rate 项。即使在 source3 上重拟合 per-column normalizer，新的第四列与原截距在所有 7 个 session 的单元间相关性仍为负，范围约 −0.483 至 −0.188；三个 slope 的相关性约 0.924–0.998。这些诊断是当时将 raw-Hz response 与 unpenalized intercept 纳入旧 v4 探索的动机，但还不能单独量化 rate 标准化造成的 decoder 损失。当前 mean_rate4 候选只借鉴显式保留 unit mean-rate 的做法，不恢复旧 raw-Hz ridge/NMF。

另一个独立因素是 spike bin 时钟。旧 reader 与 official end-timestamp reader 的同索引 count correlation 仅约 0.446–0.558；尝试一格平移后约 0.799–0.877，仍非完全一致。因此不能把它描述为已经发现并修复的单纯一格 shift bug。已停止的旧 v4 M1 探索保留旧 reader，避免将时间约定和 estimator 同时改动；已完成的 muscle multi-seed/mean-rate4 历史工作使用 official end-timestamp reader、source4 和原训练配方，见 [multi-seed carrier plan](M1_MUSCLE_MULTISEED_CARRIER_PLAN_20260909.md)。

## 已停止 v4 探索的实现边界

在已停止的 v4 探索中，两任务共享 [common_estimator.py](../scripts/carrier_v4/common_estimator.py) 的 source behavior RMS、非负状态字典、固定字典 NNLS、逐 unit raw-Hz ridge 与 unpenalized intercept、source per-column normalizer。carrier 均为三个 slope 加一个 intercept。任务 adapter 只负责行为映射、source roster、合法 support 与数据读取。

该已停止探索中的 M1 使用 ReLU EMG16，并导入封存 source3（26/27/28）fit；24 仍参与 decoder 训练，但其 carrier 与 HO3 一样由 source3 fit 部署。H1 使用按 source RMS 缩放的 signed-softplus velocity14，在 source13 的训练行为行拟合 NMF3，排除各 source 最后两个 validation trials。H1 新 source fit 于 146 次迭代收敛；四个信息臂共享同一 fit JSON SHA。

当时的 H1 signed-state14 逐列 conditional mean，与有完整交叉项的 joint ridge 是不同估计器。旧 v4 曾在 H1 上尝试 joint ridge，属于实际改变估计器的新实验；不会把改写名称当作算法统一。数学审计见 [CARRIER_V4_REVIEW_AUDIT_20260909.md](CARRIER_V4_REVIEW_AUDIT_20260909.md)。

当时共享 `lambda=1` 不代表两任务具有相同的有效收缩强度。该实现仅将 NMF dictionary 的行做 L2 normalization，没有在 ridge 前将 NNLS activation `Z` 逐列标准化。消去不受惩罚的截距后，令 `C=Z_centered.T @ Z_centered / n`，斜率解为 `(C+I)^-1 @ Z_centered.T @ rate_centered / n`；在 `C` 的特征值为 `d` 的方向，相对 centered least-squares 的收缩乘子为 `d/(d+1)`。最终 raw-carrier 的 source per-column normalizer 位于回归之后，不改变这一收缩。该代数事实限定了当时“共享估计器”的含义；未经实际 source covariance 和 decoder 对照，不能将其当作性能变化的已证实原因。

[P0 receipt](../results/carrier_v4/p0_exact_m1_v2/p0_exact_m1_receipt.json) 显示，该统一模板 raw 与当时旧生产代码在 7 个 tag 全部完全一致。历史 sealed source3 float64 raw 最多相差 `2.132e-13`，单独记录该事实；最终 float32 T 在 7 个 tag 上与原模型输入逐字节一致。随后 [共享 core 的真实部署 pack](../results/carrier_v4/m1_exact_pack_v1/carrier_pack.json) 独立重算并通过同一个 7-tag 硬断言，没有用 P0 预计算 T 替代 core 运算。

当前论文主线固定为 M1 `muscle_response16_svd4/global_rms` FULL、official `582205`，以及 H1 signed14 的 conditional-response + SVD4；允许 normalizer 差异，定义见 [paper narrative](CALIBRATION_ESTIMATOR_PAPER_NARRATIVE_20260909.md)。M1 不再扩大 carrier、结构或 seed 搜索。

## 信息与融合对照的历史执行合同

每任务原计划在 reference fusion 上训练 FULL / ACTIVITY_ONLY / CARRIER_ONLY / NONE；FULL 另比较 concat、proj-add P16、proj-add P32。M1 reference 原为 concat，H1 reference 原为 P16。local convolution width 为 16，temporal width/depth 为 256/D4，P32 只改变 E0 projection 宽度。该合同保留解释已有 audit，不要求其余 cells 继续执行。

四臂分别为 `(f(activity,T),T)`、`(f(activity,0),0)`、`(0,T)`、`(0,0)`。M1 FULL/ACTIVITY_ONLY 保留原 live B3S 训练；另外两臂不调用 encoder。H1 ACTIVITY_ONLY 重新 materialize E0，防止 FULL E0 泄露 carrier。实际 27-tag 四臂 audit 见 [h1_actual27_arm_audit_v1.json](../results/carrier_v4/h1_actual27_arm_audit_v1.json)。

实际模型参数量见 [model_parameter_counts_v1.json](../results/carrier_v4/model_parameter_counts_v1.json)：

| Task | FULL concat | FULL proj-add P16 | FULL proj-add P32 |
| --- | ---: | ---: | ---: |
| M1，含 live B3S | 3,638,292 | 3,614,292 | 3,619,988 |
| H1 | 3,710,055 | 3,542,055 | 3,557,351 |

M1 live B3S 有 80,676 个参数。在 NONE/CARRIER_ONLY 中它不参与 forward，参数从 optimizer/EMA 更新中排除；这两臂的 concat 可训练参数量为 3,557,616，neural decoder 本身不变。H1 的 C2 bank 在模型外构造，P16 四臂的 decoder 可训练参数量均为 3,542,055。不声称 concat、P16、P32 的参数量完全相同。

M1 的旧 v4 合同为 R100、24 epochs；H1 的旧合同为 R300、32 epochs，保留 C2 M7/M5/M4/M3 schedule 及旧 M3/M4 carrier-prefix 分配。初始化、query windows、valid masks、sampler、dropout、optimizer、EMA 与各任务旧 formal recipe 配对；允许的信息路由与 fusion 改动写入 checkpoint 身份。H1 v4 和未开始的 M1 v4 cells 均不再执行。

完成的 M1 结果同时报告 fixed e3 与全曲线 earliest-maximum selected epoch。H1 e16 仅是停止前的 fixed endpoint，score 不完整，不能报告 selected epoch。非劣容忍度事前声明为 R² 0.01；配对区间的范围仍限于 M1 三个 session 与 H1 七个 grouped sessions（S6–S12），不能把 H1 的十四个 tag 当作十四个独立 session。所有完成结果属于 public calibration、seed42 的 development evidence；不由“未观察到增益”推出总体零效应，也不由不显著推出非劣。

H1 的主 external baseline 仍是 signed-state 正式结果：e16、grouped mean R² **0.467651**。旧 H-C/582073 的 0.374823 只作历史背景；runner 中保留的 `versus_582073` 字段不能取代对当前 signed-state baseline 的比较。[实验 manifest](../results/carrier_v4/experiment_manifest_v2.json) 仍可解释 external baseline、同 carrier 信息比较与 fusion 比较的历史定义，但停止后不再产生这些新比较。

H1 `ACTIVITY_ONLY` official `582218` 已 finished：HO `0.3445956103610151`、HI `0.608015116039874`、latency `0.1374597509210613`。相对 FULL official `582196` HO `0.45665779063606854`，delta HO 为 `-0.11206218027505344`。只引用 [OFFICIAL_582218.json](../../tfpd_exploration/submissions/evalai_h1_rift_activity_only_r300_v2/artifacts/OFFICIAL_582218.json)。首次 EvalAI snapshot 为 failed/empty `[]`，随后翻转为 finished；`582220` 不是 official。官方对照图见 [official_full_vs_activity.png](../results/final_ablation_official_v1/official_full_vs_activity_v1/official_full_vs_activity.png)。

H1 FULL 当前 public HO-M3 的 selected e16 已作独立 CPU cached replay：原始 local grouped R² 为 `0.46765143362182887`，重放为 `0.467651490064873`，差为 `+5.6443044127441055e-08`，七组最大绝对差为 `1.7721179190743896e-07`，低于 `1e-5` gate。该 [成功 receipt](../results/final_ablation_official_v1/h1_full_selected_cached_replay_v1/receipt.json) 只复现已有 FULL 的当前公开查询结果。现已完成的 [FULL/ACTIVITY_ONLY formal audit](../results/final_ablation_official_v1/root_h1_activity_only_formal_result_audit_v1.json) 绑定两臂同 seed42、19 个固定 recipe 字段、同一当前 14 recordings / 33,613 queries 聚合为 S6–S12 七组及完整 all-32 curve：FULL selected e16 `0.46765143362182887`，ACTIVITY_ONLY selected e15 `0.2660451704314089`、worst group `0.12474186695147141`，ACTIVITY_ONLY − FULL 为 `-0.20160626319041997`，且 ACTIVITY_ONLY 在 7/7 groups 低于 FULL；[本地图表](../results/final_ablation_official_v1/h1_full_activity_local_summary_v1/curves.png) 展示全部曲线和选中组分数，仅是 local public 曲线。这些 local 数字不得与 official HO 相减。H1 `NONE` 没有 official。M1 `ACTIVITY_ONLY` official 现为 `582219`，见上文三任务 official 表。截至 2026-09-10 的 M2/H1 执行交接状态见 [M2/H1 ablation handoff](M2_H1_ABLATION_HANDOFF_20260910.md)。

H1 停止前的阶段记录：固定 e16 在 [实际评分日志](../results/carrier_v4/h1_full_proj_add_p16_formal_v1.log) 中为 v4 FULL/P16 grouped mean 约 **0.352676**，相对旧 signed-state 同 e16 约低 **0.114975**，worst-session 约 **0.199441**；评分至 e28 时日志中最高 observed 值为 e17 `0.360018`。由于只评分 28/32，二者均不是完整曲线的 selected result。SIGTERM 是用户停止资源投入，不应把 queue 的 `FAILED/-15` 写成方法失败；也不能凭这些未完成的 public endpoint 将差异归因于 NMF、截距或 ridge 尺度。
