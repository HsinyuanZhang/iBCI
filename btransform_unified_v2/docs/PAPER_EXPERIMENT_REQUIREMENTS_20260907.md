# 论文实验需求与证据状态（2026-09-08 刷新）

本清单只记录可进入论文的证据边界、已完成结果及已排队闭合项。它不授权新训练、评分、提交或 688 工作。机器可读副本为 [PAPER_EXPERIMENT_REQUIREMENTS_20260907.json](PAPER_EXPERIMENT_REQUIREMENTS_20260907.json)。

## 当前结论

H1 方法绑定、官方结果和 calibration cost 已闭合：submission 582073 使用冻结 M3 H-C，source-only `q=12`、ridge `lambda=10.0`、source PCA/output-SVD/EB shrinkage 和单一 source-RMS normalizer；replayed plan 的 FP64 PCA/SVD arrays 并非原封存数组字节恢复，但其 27/27 normalized `float32` deployed `T` 已严格复现，且 81 次 warm carrier→E0→bank 路径均与冻结 `T/E0` 字节一致。所有自有 FALCON M1/M2 训练、formal validation、paired/control summary 与已声明的 CPU/support evidence 均已闭合。M1 B/D 是一个 seed42 的 visible-HO3 paired development result；M2 B/D 三 seed 与全部 controls 分别受其 ext4 scope 限制，均不涉及 official test。M2 ext6 audit 只闭合 post-training submission candidate selection。M2 corrected BT direct-carrier reliance v2 已完成；v1 坐标契约无效，禁止用于任何结论。

论文范围已于 2026-09-08 静态审阅并推送到 `origin/main` commit `9b8c7a0d8ba949b2584d2f0072bf34604af1e7cd`；此闭合只确认采用本清单既有证据边界和缩小后的主张，**未编译 TeX，亦未新增数值运行、评分、提交或权限变更**。最终 handoff 文档 SHA-256 为 `3ac7a6e95316fd99ec47fc655d5d00f5c65e1ca52d30538dca798773d04facf9`，inventory SHA-256 为 `d2a34bcbe84d3795c2fad533fff0419a9fbe87a6fbf645f60f759fe81b335706`，completion validation SHA-256 为 `6de941152ff0880c2fdab9410cc345dce8a05866bdbbcf64116860dc1862b778`。四任务 matched profile effect 和全任务架构效应不在最终论文的已主张范围内；M1 仅报告已完成的 single-seed HO3 paired development result，不主张跨 seed 或独立泛化；不得由不同 surface、旧 decoder 或单 seed 数值代填。

## 证据准入

主文的定量单元或差值必须同时具有：(1) receipt/manifest 绑定代码、权重、数据、support/query、seed、选择规则和指标；(2) target query labels 只用于评分、target 无网络反传；(3) 被比较 arms 共享协议且只改变声明因素；(4) 明确评分 surface 和聚合；(5) 与主张相称的 seed/session 不确定性；(6) 网络、窗口和 runtime backend 的实际绑定。缺任一项只能作为开发证据或缩小表述的依据。

## 已闭合证据

| 范围 | 状态与可用结论 | 绑定证据 |
|---|---|---|
| H1 carrier method and cost | **完成。** 582073 的冻结 `T` 属于 all-source M3 `q=12/lambda=10.0` family；不是旧 `q=16/lambda=100` family。replayed plan 的 FP64 `pcs/U/mu` 与 sealed hashes 不同，但 27/27 normalized `float32` `T` 严格复现，不能把它写成原 NPZ 字节恢复。27×3 warm carrier→C2 E0→bank 共 81 次均 `T/E0` 字节一致，total median/P95 为 `1.085079/1.155038 ms`。 | `docs/H1_FROZEN_CARRIER_METHOD_BINDING_20260908.json`；verify27 receipt SHA `5c64b943c2d6758493d88ecaee714edfd1294b9fd28a2eedeec4006fd6461aca`；cost receipt SHA `54d737eac6342f10d0f6517bfe94f48ff6303e35e7a160012727b98b83d53186`。 |
| M1 concat seed42 | **完成，开发面。** selected EMA e3 equal-session mean `0.734192`。runtime B1 median/P95 为 `1.954/1.989 ms`；calibration cost 已完成，约 `50.87 ms`。 | 对应 concat train/runtime/cost receipts；该 runtime/cost 只适用于该权重、主机和 protocol。 |
| M1 joint D42 | **完成，开发面、仅 D seed42。** selected EMA e3 equal-session mean `0.7342600300996954`，相对 frozen concat e3 为 `+0.00006776958380461107`；这个很小的单 seed 差值不构成收益结论。D42/e3 CPU benchmark 已完成：B1 median/P95 `1.98861350145/2.04134084888 ms`，B8 `15.77839250058/16.27837364904 ms`。 | formal validation SHA `7fa2c2092b34670cd368ee1795eb555d5ed7671334ce4b549abeddd5dd41a4f7`；benchmark SHA `8670f2abe91a7e6e4e30d2d3ba2763a4ff423dbec68f8561755634aa640ab8e8`；详见 `docs/M1_JOINT_D42_E3_VALIDATION_AND_CPU_RUNTIME_20260908.md`。 |
| M1 joint B/D pair | **完成，visible HO3 development、一个 paired seed42。** B/D 均 selected EMA e3，B/D mean 为 `0.6630568974542209 / 0.7342600300996954`，selected D−B 为 `+0.07120313264547451`；fixed epoch 24 D−B 为 `+0.08758950421224765`，三 session 均正。session 是同一 seed 内重复测量，不能报告 CI、p 值、多-seed 或独立 generalization。D 相对 frozen concat 的 `+0.00006776958380461107` 不支持 joint-encoder 收益主张。 | pair summary `results/diagnostics_v1/m1_joint_pair_summary_v1.json` SHA `b85d061fca7d13aed62b22be10bdb296428ffcca4c07364e1cf462017572d415`；B audit SHA `a01e762028186891d19e9ec0717c3d752239e8291960f3f360ff8b905e7b3a6e`；`docs/M1_JOINT_PAIR_VALIDATION_20260908.md`。 |
| H1 official | **完成。** 582073 official held-out mean/std 为 `0.4027782688 / 0.1452665847`；它是冻结系统结果，不是 profile-only effect。 | `results/rift_v1/h1_r300_official_receipt_20260907/official_result.json`，SHA `52e88d57b25c47f1648b73d4b8dad431a01fe49c4ea017153652de0266cc54d8`。 |
| M1 baseline | **完成，开发面。** R100 recency seed42 selected EMA e3 equal-session mean `0.7046861491`。 | `results/rift_v1/m1_r100_recency_s42_formal_v3/{train,score}_receipt.json`。 |
| M2 matched seed42 | **完成，开发面、仅 seed42。** B42 selected e7 `0.2668254644`；D42 selected e17 `0.3935032533`；差 `0.1266777889`，不得外推为多-seed效应。 | `results/rift_v1/m2_r50_joint_{b,d}_s42_formal_v1/{train,score}_receipt.json`。 |
| M2 shuffle42 control | **完成，ext4 development、仅 seed42。** selected EMA e10 equal-session mean `0.27836822974156405`；D42 minus shuffle42 selected difference 为 `+0.11513502354880062`。这是单一 seed 的描述性均值，不能写为四个 session 均为正或多-seed 控制效应。 | `results/rift_v1/m2_r50_mechanism_shuffle_s42_formal_v1/{run_meta,train_receipt,score_receipt,validation_audit}.json`；validation SHA `07ea8a52aabcdb8c5ee00390a64013c9407b1f5cc05d6c3fe1d1daed64b4aa62`。 |
| M2 mean42 control | **完成，ext4 development、仅 seed42。** selected EMA e12 equal-session mean `0.2932165628085156`；fixed epoch 24 为 `0.24337902866110805`。结果与空间差值须等待全部 control summary 后按同一规则解释。 | `results/rift_v1/m2_r50_mechanism_mean_s42_formal_v1/{run_meta,train_receipt,score_receipt,validation_audit}.json`；validation SHA `63f74346fbea0e91da66de6fb7d785094831ccc506bf9e1d9b628c908ce28b00`。 |
| M2 nonattention42 control | **完成，ext4 development、仅 seed42。** selected EMA e8 equal-session mean `0.3939113798031849`；fixed epoch 24 为 `0.370724401215297`。 | `results/rift_v1/m2_r50_mechanism_nonattn_s42_formal_v1/{run_meta,train_receipt,score_receipt,validation_audit}.json`；validation SHA `4d8ccde62686df456e9216d1c1c4b871c939907d2a26845e0c071f85cfd2ab27`。 |
| M2 mechanism controls summary | **完成，ext4 development、仅 seed42。** D−mean 为 selected `+0.10028669048184907`、fixed epoch 24 `+0.11764316829207055`；D−nonattention 为 selected `-0.0004081265128202394`、fixed epoch 24 `-0.009702204262118375`。因此这一个 seed42 control set 不支持 temporal attention 优越，也不证明两者等效。D−shuffle 虽为正的 equal-session mean，四个 session 中有两个为负，不能写成全 session 正向。 | `results/diagnostics_v1/m2_mechanism_controls_summary_v1.json`，SHA `b55cb2d71b4b5d704bfce0a3c8a6ae1fe4077b27cbdb3762399ba463d0c71de5`；summary 已验证 144 checkpoints、source/cache 与 selection。 |
| M2 three-seed paired B/D | **完成，ext4 development evidence。** 三对独立 model seed 的 independent-pick D−B mean/sample SD 为 `0.13672691424821373 / 0.03580496790760836`；fixed epoch 24 为 `0.15368779486183196 / 0.03575406559424156`。不报告 CI、p 值或 session-level pseudo-replication，也不外推到 ext6 或 official test。 | `results/diagnostics_v1/m2_joint_paired_seed_effects_v1.json`，SHA `abcba3df8de3601058f758818d37db79fb46d8f1270f248eafa95ca901c2efde`；`docs/M2_THREE_SEED_PAIRED_EFFECTS_20260908.md`。 |
| Support stability | **完成，描述性。** M1 receipt 保持原有边界。M2 的 post-hoc summary 重读全部 440 条既有记录；主 EXT4 四 session 的 M8/16/25 按每 session 有效 resample median 后等权聚合，M8 的有效率为 `26/40`，M16/M25 均为 `40/40`。它不重算 carrier、不训练或评分 decoder。 | M1：`results/diagnostics_v1/m1_carrier_support_stability_v2.json` SHA `c0f7308d686c48d8a0643045b541cd64cae05135be92f4b52eaeeca94286c511`；M2 source report SHA `573711d9b144b00387e3de07f60c6c89990f81d1681d2a550eb3e394c4443bac`；post-hoc summary SHA `e883763a13f6b8cdfb3938d471e74fe2f9649d4d3481dd48e361646cd0d5b342`；详见 `docs/M2_SUPPORT_STABILITY_DESCRIPTIVE_20260908.md`。 |
| M2 BT direct reliance | **完成，仅 corrected v2。** direct/full oracle 与 wrong-start negative control 通过；v1 无效且禁止引用。它不等于 RIFT mechanism effect。 | `results/rift_v1/m2_move_t4_concat_carrier_reliance_v2_20260907T162508Z/report.json` SHA `018793d1b965dc88ea77595f3c9d7ca34534d463bda7e7ba635069d3f0777bec`。 |
| M2 trained CPU D42 e17 | **完成，限定 benchmark。** 2 CPU threads、402 parity advances max error 0；B1 median/P95 `2.702/14.974 ms`，B8 `17.344/18.332 ms`。报告只能说明该 checkpoint、该主机和该 protocol；不代表 M1、全任务 speedup、calibration cost 或整体 memory claim。 | `results/rift_v1/m2_joint_d_s42_trained_cpu_benchmark_v1/benchmark.json` SHA `17a083a0b531d8437bb6d56f24698171bdef724f4077e33551f367bf1148dead`。 |
| M2 D42/e17 finite-only parity | **完成，未重计时。** 该证书只验证原 e17 checkpoint 与 B1/B8 输入的 finite parity，不能替代或更新既有 timing receipt。 | `results/rift_v1/m2_d42_e17_finite_parity_v1/benchmark.json` SHA `6072ef9c2d7397c7fd1ecbcd7e0daf99190855bb31639686cdc36e673207194e`；validation SHA `669f36741f19669f9d687a2cc842321f58c086d021b31a90cd4a45f8b9455e81`。 |
| M2 concat ext6-selected e9 CPU runtime | **完成，限定 benchmark。** 当前 full-concat EMA e9 的 CPU runtime receipt 已由 validation 审计；它只绑定该 checkpoint、主机、ext4 runtime inputs 和 protocol，ext6 只负责 post-training checkpoint selection。 | `results/rift_v1/m2_concat_ext6_e9_cpu_benchmark_v1/benchmark.json` SHA `d7d0743a319b37a004f44e3c27e02ccb14656e0337c4579578c034736d068019`；validation SHA `0a16b5fd4a10b8d5ba8470b2b4842ccc3ebbe108ecd6293d0d0054e925e59725`；`docs/M2_CONCAT_EXT6_E9_CPU_RUNTIME_20260908.md`。 |

Published SPINT/zero-shot Wiener 仍可作为外部 FALCON reference，不是本地 paired effect；688 无 published column。

## 已归档限制与外部接口

**M2 submission epoch-pick policy.** 后续用于提交的 M2 epoch-pick 统一使用
独立的 `ext6` surface：六个 session 是 2020-10-30 Run1/Run2、2020-11-18
Run1、2020-11-19 Run1、**2020-11-24 Run1/Run2**；这里的 `11-24` 是日期，
不是 epoch 范围。submitted weight 必须为 selected **EMA**，候选一律是同一
epoch `1..24`，以六个 session R² 的非加权算术均值取 earliest maximum。封存
submission 582047 是 `S1-SMALL-COS` EMA e8、按 ext6 选择；submission 582128
是另训 RIFT-concat EMA e7、按 ext4 选择。因此两次 submission 的差值不是
matched architecture effect。当前 M2 representative mechanism arms 保持预注册的
ext4（四 session）闭合；未来独立 ext6 后训练 epoch scan 只负责 submission
选模，绝不改写旧 receipt。

| 范围 | 已归档状态 | 论文可用边界或外部接口 |
|---|---|---|
| 688 | **external delegated，唯一外部接口。** 本集群不得审计、运行、排队、编辑或推断 688；仅保留队友结果/receipt 接入接口。 | 等外部团队提供 strict-manifest、arms、score 和 provenance 后再决定表述；本清单没有 688 行动项。 |
| M1 | concat seed42、joint B42/D42 与 B/D paired summary 已完成；CPU runtime 只绑定 D42 checkpoint、主机和 HO3 stream protocol。M1 direct 25 conditions 已完成。 | M1 paired证据仅为一个 seed42、visible HO3 M10/query-trial-0 且可能 support overlap 的 development comparison；不得扩张为多-seed、独立 generalization、official test 或 joint-encoder收益结论。 |
| M2 | B/D 的 42、43、44 ext4 formal receipt 与 three-seed paired summary 已完成。D44 selected EMA e11 `0.42731552501440395`，epoch 24 `0.3646836676840557`，validation SHA `fe03fae483361923a96d80224284b0cc7f62fa1565e75146395f15db316fbfb0`。C/shuffle/mean/nonattention seed42 controls 与 complete control summary 已完成；D−mean selected/fixed 为 `+0.10028669048184907 / +0.11764316829207055`，D−nonattention 为 `-0.0004081265128202394 / -0.009702204262118375`，故不得声称 temporal attention 优越或等效。D−shuffle 的 equal-session mean 为正但有两个负 session，仍不作全 session 正向主张。seed43 含一个负 B/D session delta，仍按 seed-pair 而非 session 解释。四个 ext6 submission candidates 已有完整扫描和 final-package audit：concat e9 `0.3900577`、D42 e13 `0.3478275`、D43 e8 `0.3855227`、D44 e11 `0.3627533`；concat 仍领先。它们只用于提交候选选择，不改写 original ext4 mechanism。 | M2 mechanism controls 已闭合为 seed42 ext4 descriptive evidence；它不构成 multi-seed control/architecture conclusion。D43/D44 ext6 final-package validation 各 279 checks 通过、all-24 scan 和 EMA 121 state-key/alias 绑定均通过；这仍是 ext6 submission selection，不影响 EXT4 paired summary。 |
| H1 | frozen；carrier method、functional replay 与 warm calibration cost complete；不再 score chasing。 | 仅可保持 official-system、method binding 和已定义 timing claim；profile effect 或超出 receipts 的 runtime/generalization 仍须缩小表述。 |

## 删除或缩小表述规则

- 没有四个合格 paired effects：删除“四任务 matched profile effects”及对应架构泛化句。
- 没有 M1 paired outcome：删除 M1 的实证 profile/mechanism claim；但已完成的 concat runtime/calibration-cost receipt 可以按其 checkpoint、主机和 protocol 边界报告。
- M2 的三-seed EXT4 summary 只能报告 seed-paired descriptive mean/sample SD；不得报告 CI、p 值、显著性或将 session 当独立重复。
- 无实际 M1 runtime/calibration-cost receipt：不填 M1 runtime/cost 数字。
- v1 M2 direct-reliance 报告不得用于论文、图表或对照；只可引用 v2。
- 688 的证据和生命周期完全由外部团队管理；未接入 receipt 前保持 pending/N/A。
