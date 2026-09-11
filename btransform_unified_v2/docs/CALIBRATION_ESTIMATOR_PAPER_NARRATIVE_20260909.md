# Calibration estimator：可用于论文的方法与结果叙事

## 主论点

少量 calibration 的作用不应只理解为为每个 session 重新学习一个 decoder。它还可以把同一段短时 neural response 转换为一个静态、可跨 session 比较的 unit-level 描述：某个 unit 在不同非负行为状态下偏离自身平均 firing rate 的条件响应。这个描述先在 source sessions 上确定低维坐标和尺度，再在每个 session 的合法 calibration support 上估计 unit row，因而把“行为状态—神经响应的对应关系”与后续 neural decoder 的时序建模分开。它是一个稳定的表示假设，不是对独立行为变量因果效应的识别。

## 可直接用于方法部分的文字

我们为每个 recorded unit 构造一个 source-frozen calibration carrier。对 support bin `t` 的非负行为状态权重 `w[t,k]`，先在当前合法 calibration support 内将该 unit 的 firing rate 去均值，并按 Poisson noise floor 缩放，记为 `z[t,u]`。随后以

\[
A_{u k}=\frac{\sum_t w_{t k}z_{t u}}{\sum_t w_{t k}+10}
\]

计算 unit `u` 对状态 `k` 的带 pseudo-count 条件响应。分母中的 pseudo-count 降低了稀少状态在短 calibration 中产生不稳定大值的风险；它不是额外的标签，也不改变合法 support 的范围。直观地，这个量是行为权重与同一 support 内去均值 rate 的正则化经验 covariance：若 \(\operatorname{Cov}_n(w,r)=n^{-1}\sum_t(w_t-\bar w)(r_t-\bar r)\)，则 \(A=\operatorname{Cov}_n(w,r)/[q(\bar w+10/n)]\)。等价地，在总权重大于零时，它是 weighted-mean rate 相对同一 support sample mean 的偏离，按 \(\sum w/(\sum w+10)\) 收缩；完整推导与零权重边界见 [moment note](CALIBRATION_ESTIMATOR_MOMENT_NOTE_20260909.md)。我们仅在 source sessions 的 pooled unit rows 上拟合未中心化 SVD4、固定其方向与归一化统计量，然后将每个 session 的 raw conditional-response row 投影到这四维 source 坐标。decoder 接收的 carrier 因而始终是 source-centered、source-scaled 的四维 unit identity，而不是在 target 端重拟合的共享字典或 target-adapted decoder 参数。

这个构造保留两层信息。rate 的 session 内去均值与 noise scaling 使 carrier 关注 unit 对行为状态的相对调制，而不是让高 firing-rate unit 因数值尺度而主导全部轴；source SVD4 再把多个相关状态的 response pattern 压缩为固定、可部署的坐标。由于行为状态可相关，四个坐标也没有单独的生理或因果标签。论文中应把它称为“行为加权的 unit conditional-response representation”，而不应称为单个 muscle 或 velocity axis 的独立效应。

## 两个任务的共同结构及其必要差异

M1 和 H1 共享上述估计顺序，但不要求相同的行为映射或 normalizer。M1 使用 16 路 rectified EMG：每个 muscle state 是 `max(EMG,0)` 除以 source muscle RMS。其 rate 在 20-ms bins 中去均值，以 `sqrt(max(mean_count,1))/0.02` 缩放；source pooled raw16 unit rows 经 SVD4 后，以四个投影列标准差的共同 RMS 作为一个全局尺度。因此，M1 的 `muscle_response16_svd4/global_rms` 让四个 carrier 轴保持共同的数值单位。

H1 使用 7 路 velocity 的 signed-softplus state：每一 velocity axis 产生 `softplus(v/s)` 和 `softplus(-v/s)` 两个非负权重，共 14 个 state。rate 在 100-ms bins 中以对应的 Poisson noise floor 去均值和缩放；13 个 source session 的 pooled raw14 rows 经 SVD4 后，每个投影列分别使用 source RMS scale。signed-softplus 的正负对保留速度方向信息，同时以连续的非负权重支持条件响应统计；它不意味着单个 softplus 项有固定正基线，也不把某个 SVD 轴解释为一个独立方向效应。

两条线路的共同点是“非负行为状态加权的、unit-centered conditional response，再做 source-frozen 低维压缩”；M1 的 global RMS 与 H1 的 per-column scale 是任务特定的数值设计，不能在方法中写成同一个 normalizer。H1 保留已验证的 signed-state 方法作为主线，本工作不再将新的 H1 carrier 变体写成正在进行的实验。

## 可直接用于实验部分的文字

论文主对比统一为各任务 official FULL 对 official `ACTIVITY_ONLY`。M1 FULL 固定为 muscle `582205`（HO `0.6259910151098528`），不使用 mean-rate4 `582228` 作主 FULL。官方差只对同一 task 的 FULL official ID 计算；本地 HO3 / EXT6 / HO-M3 不得与 official HO 相减。FULL→ACT 同时去掉 direct T 与 carrier-conditioned E0，不能据此拆出单独路由效应。对照图见 [official_full_vs_activity.png](../results/final_ablation_official_v1/official_full_vs_activity_v1/official_full_vs_activity.png)。

| task | FULL id / HO | ACT id / HO | FULL−ACT ΔHO |
| --- | --- | --- | ---: |
| M1 | `582205` / `0.6259910151098528` | `582219` / `0.5709008472436213` | `+0.05509016786623144` |
| M2 | `582189` / `0.34654225938843214` | `582217` / `0.09302905280313856` | `+0.2535132065852936` |
| H1 | `582196` / `0.45665779063606854` | `582218` / `0.3445956103610151` | `+0.11206218027505344` |

M1 muscle `ACTIVITY_ONLY` official `582219` 在 `test_split_m1` 上另有 Held In R² mean `0.793379687455571`、Normalized Latency `0.1413142769159171`。收据 [OFFICIAL_582219.json](../../tfpd_exploration/submissions/evalai_m1_rift_activity_only_r100_v1/artifacts/OFFICIAL_582219.json)。其 local HO3 selected e3 mean `0.5618270536263784` 不是 official，不得与 official HO 相减。这对结果由只读 [M1 official receipt](../results/final_ablation_official_v1/root_m1_full_activity_official_readonly_20260910T031416Z.json) 复核（SHA-256 `d1bf796f13b752997e5c7ae0a5fdbbeaeb87ff558203c217d59ffb1241e6fe6a`）：phase `4599` 中仅 direct `GET` `582205`/`582219`，并通过 result、ACT payload 和 submit-server binding；未 push、register 或 submit。它是 original muscle-response/SVD4 的整体信息路径对照。

M2 RIFT concat `ACTIVITY_ONLY` official `582217` 在 `test_split_m2` 上为 Held Out R² mean `0.09302905280313856`、Held In R² mean `0.6463020945769177`、Normalized Latency `0.10743839747764272`。相对 FULL official `582189` HO `0.34654225938843214`，delta HO 为 `-0.2535132065852936`。收据 [OFFICIAL_582217.json](../../tfpd_exploration/submissions/evalai_m2_rift_activity_only_r50_v1/artifacts/OFFICIAL_582217.json)。其 local EXT6 selected e9 mean `0.18402467250439059` 不是 official。

H1 signed-state `ACTIVITY_ONLY` official `582218` 在 `test_split_h1` 上为 Held Out R² mean `0.3445956103610151`、Held In R² mean `0.608015116039874`、Normalized Latency `0.1374597509210613`。相对 FULL official `582196` HO `0.45665779063606854`，delta HO 为 `-0.11206218027505344`。只引用 [OFFICIAL_582218.json](../../tfpd_exploration/submissions/evalai_h1_rift_activity_only_r300_v2/artifacts/OFFICIAL_582218.json)。首次 EvalAI snapshot 为 failed/empty `[]`，随后翻转为 finished；`582220` 不是 official。其 local HO-M3 selected e15 mean `0.2660451704314089` 不是 official。

M1 `NONE` official `582224` 现已 finished：Held Out R² mean `-0.6772461160545928`、Held In R² mean `0.598250857063468`、Normalized Latency `0.13951016955707982`。相对 FULL official `582205` 的 delta HO 为 `-1.3032371311644456`。收据 [OFFICIAL_582224.json](../../tfpd_exploration/submissions/evalai_m1_rift_none_r100_v1/artifacts/OFFICIAL_582224.json)。这条零身份线**不进入主对比**，可留附录作无身份下限：崩塌与 HO 相对 source 的发放率抬升同序，不是 EMG DC 跨日失配。官方 HO 大负与本地 HO3 `-1.2152107258637745` 同结构。metadata `GET` 的 `result=null`；官方指标来自 `submission_result_file`。v2 未提交。M2/H1 `NONE` 仍无 finished official。

公平协议基线记录见 [FAIR_ZERO_BASELINE_DECISION_20260910.md](FAIR_ZERO_BASELINE_DECISION_20260910.md)。M1 记为“使用发放率信息的重新训”（当日 M10 无标签逐单元平均发放率，T=0），尚未开训。M2 last_source 本地 EXT6 等权 R² `0.258669953325137`（`582189` 权重，复制 `ses-2020-10-28-Run1`）已写入该记录。H1 srcbank 已由用户叫停（仅 `own` 收回 `0.467651`，无 last_source）。这些本地数字不是 official，不得与 official HO 相减。

M1 的 muscle-response R100 seed42 模型在 official `test_split_m1` 上取得 Held Out R² mean `0.625991`、Held In R² mean `0.792630`，Normalized Latency 为 `0.140405`。该 checkpoint 在提交前已由公开 HO3 calibration 的完整 24-epoch scan 选定；official 返回不参与 checkpoint 选择。这个结果说明该 source-frozen conditional-response carrier 可以在 M1 的独立 official surface 上部署并产生有效的端到端解码结果。

当前 muscle 方案的直接公开对照是同 seed42、相同 HO3 targets、window starts、window counts 和完整 24-epoch scan 的已完成 ACTIVITY_ONLY B arm。B 是 fresh、无 resume 的独立 24-epoch/159,960-update 训练，不是将 D checkpoint 输入置零；两 arm 共享相同 seed42 decoder 初始化和 B3S 预训练初值（initialization SHA `02f056...`），随后独立优化各自的 encoder/decoder，审计见 [root_activity_only_training_audit.json](../results/m1_muscle_r100_multiseed_v1/root_activity_only_training_audit.json)。B code 持续将 side T 与 direct T 同时置零。两条线均在 e3 selected：muscle mean 为 `0.5690750182`，B mean 为 `0.5618270536`，差为 `+0.0072479645`。三个 session 的差分别为 `-0.0020035505`、`+0.0320852995`、`-0.0083378553`；点估计中最差的下降仍在 `-0.01` 以内。进一步的 [M1 FULL/ACTIVITY_ONLY reference-binding audit](../results/final_ablation_official_v1/root_m1_full_activity_reference_binding_v1.json) 指定 canonical FULL 为 `m1_muscle_r100_v1/formal_s42_gpu1`，ACTIVITY_ONLY 为独立 B_ACTIVITY_ONLY `rift_v1/m1_r100_joint_b_s42_formal_v1` 的当前 replay，并重核 48 个 checkpoint 文件 hash、21 个固定 recipe 字段、同一 HO3 三 session / 3,881 query targets、starts/counts 与 B3S source-file SHA。historical old-D/rSyn replay 仅是 shared-query provenance，不是 canonical FULL；审计只复核既有 JSON/hash，不运行新 forward 或模型，也不产生 NONE 或 official 结果。

B 仍使用 M10 的无标签神经活动生成 E0。这条本地 HO3 对照不能拆成单独的 direct-T 路由或单独的 E0 路由；FULL→ACT 同时去掉 direct T 与 carrier-conditioned E0。它也不是“使用 calibration”相对“完全不使用 calibration”的比较。official FULL `582205` 对 official ACT `582219` 的 delta HO 见上文，不得用 local HO3 selected e3 mean `0.5618270536263784` 去减 official。对 original muscle carrier，更准确的结果叙事是：其公开 HO3 增量有限；这只是三个 session、seed42 的观测点，不是统计 non-inferiority 证明，也不把 M1 定义为真实无 shift。这个 original-muscle 结论不应被误指为后述 mean-rate4 candidate。

同一 current public HO3、seed42、channel-variance-weighted R² 三臂审计中，FULL e3 为 `0.5690750181674957`，ACTIVITY_ONLY e3 为 `0.5618270536263784`，NONE 自身 24-epoch 曲线选 e1 为 `-1.2152107258637745`；ACTIVITY_ONLY − NONE 为 `+1.7770377794901528`，三个 session 均为正。见 [three-arm formal audit](../results/final_ablation_official_v1/root_m1_full_activity_none_formal_result_audit_v1.json)（SHA-256 `33ba82c6c34512459eb7b39abbcea4e853b61f7689897018d45c732fe5f36b21`）与[本地摘要](../results/final_ablation_official_v1/m1_ablation_local_summary_v1/summary.json)。这支持该配置依赖 activity-derived E0；由于单 seed 且各臂 selected epoch 不同，不能外推为普适机制。ACT→NONE 移除 activity E0，而 FULL→ACT 的联合移除仍不能拆为单独 direct-T 或 carrier-conditioned-E0 作用。v1 的 host import 失败已由 v2 local package 修复并验证。既有 v1 official `582224` 现为 finished，见 [OFFICIAL_582224.json](../../tfpd_exploration/submissions/evalai_m1_rift_none_r100_v1/artifacts/OFFICIAL_582224.json)；较早 FAILED / 无 scored result 的 [failure evidence](../results/final_ablation_official_v1/root_m1_none_official_failure_evidence_v1.json) 为历史并已被取代。v2 未提交，见 [corrected manifest](../results/final_ablation_official_v1/m1_none_image_metadata_v3/corrected_manifest.json)。这些 local 数字不得与 official FULL/ACT/NONE 相减。

历史 rSyn3 FULL 与同一独立 ACTIVITY_ONLY B 的完整对照仍是重要证据：FULL rSyn3 为 `0.654471`，ACTIVITY_ONLY 为 `0.561827`，相差 `0.092644`。它说明行为响应 carrier 的端到端额外作用依赖 carrier 选择，不能由当前 muscle 的小增量推广为“calibration 总体无意义”。反过来，muscle 与 rSyn3 的约 `-0.085396` 也只属于公开 HO3 surface。相同 RIFT joint-D rSyn3 配置没有 completed official score，故不能把 M1 muscle 的 official `0.625991` 与这些公开差异相减，也不能写成 muscle 在 official test 上弱于或强于 rSyn3。

M2 的独立 RIFT concat 三臂结果提供另一条、不同任务的本地证据，不能替代上文 official `582217`。FULL、ACTIVITY_ONLY 与 NONE 均以同 seed42、sampler、六个 EXT6 session 和 15,403 个 query windows，独立完成 24 epochs / 75,960 updates 与完整 all-24 EMA scan；各自 selected 为 FULL e9 `0.3900576650553506`、ACTIVITY_ONLY e9 `0.18402467250439059`、NONE e6 `-0.012659512830027883`。ACTIVITY_ONLY − FULL 为 `-0.20603299255096003`，NONE − FULL 为 `-0.4027171778853785`，ACTIVITY_ONLY − NONE 为 `+0.19668418533441848`。FULL 对另外两臂均为 6/6 session 更高；ACTIVITY_ONLY 对 NONE 为 5/6 更高，唯一例外 `ses-2020-11-19-Run1` 的 ACTIVITY_ONLY − NONE 为 `-0.2063196674812775`。三臂 query、seed 与 sampler 的审计见 [formal audit](../results/final_ablation_official_v1/root_m2_full_activity_none_formal_result_audit_v1.json)。这些数字是 local、非 official；NONE 没有 official ID。不得把 local EXT6 与 official HO `0.346542` / `0.093029` 混口径或相减。官方对照只见 [official_full_vs_activity.png](../results/final_ablation_official_v1/official_full_vs_activity_v1/official_full_vs_activity.png)。

H1 official `ACTIVITY_ONLY` 已见上文 `582218`。另有 FULL signed-state 的 current public HO-M3 selected e16 本地基线：原始 grouped R² `0.46765143362182887`，独立 CPU cached replay `0.467651490064873`，差为 `+5.6443044127441055e-08`，七组最大绝对差 `1.7721179190743896e-07`，通过 `1e-5` gate；见 [successful replay receipt](../results/final_ablation_official_v1/h1_full_selected_cached_replay_v1/receipt.json)。现已完成的 [FULL/ACTIVITY_ONLY formal audit](../results/final_ablation_official_v1/root_h1_activity_only_formal_result_audit_v1.json) 以同 seed42、19 个固定 recipe 字段、同一当前 14 recordings / 33,613 queries 的 S6–S12 七组和完整 all-32 curves 对照两臂：FULL selected e16 `0.46765143362182887`，ACTIVITY_ONLY selected e15 `0.2660451704314089`、worst group `0.12474186695147141`，ACTIVITY_ONLY − FULL 为 `-0.20160626319041997`，ACTIVITY_ONLY 在 7/7 groups 低于 FULL；[曲线和组选中分数](../results/final_ablation_official_v1/h1_full_activity_local_summary_v1/curves.png) 已生成，仅是 local public 曲线，官方对照见 [official_full_vs_activity.png](../results/final_ablation_official_v1/official_full_vs_activity_v1/official_full_vs_activity.png)。这些 local 数字不得与 official HO `0.456658` / `0.344596` 相减。H1 `NONE` 没有 official。截至 2026-09-10 的 M2/H1 执行交接状态见 [M2/H1 ablation handoff](M2_H1_ABLATION_HANDOFF_20260910.md)。

## mean-rate 第四列的动机与解释边界

现有 conditional-response carrier 先对每个 unit 的 rate 做 support 内 centering。这是有意设计：前三维及原第四个 SVD 坐标描述的是相对行为调制，而不是 unit 的绝对平均 firing level。与之不同，历史 rSyn3 carrier 的 joint ridge 将原始 Hz rate 回归到 NNLS activation，并保留一个未惩罚 intercept；该 intercept 可携带 unit mean-rate 相关的信息。两种设计都合理，但它们表明“先 centering 的 conditional response”与“显式 raw-rate intercept”编码的信息不完全相同。

因此，M1 的单独 mean-rate 第四列候选只提出一个受限的问题：在保持原 muscle SVD carrier 前三列逐字节不变、使用同一 M10 rows 和同一 reader 的前提下，能否以 arithmetic unit mean rate 替换原第四 SVD 坐标。该替换的动机是恢复一个被 rate centering 刻意移出的 unit-level level statistic，并以 source4 的中心化/标准差缩放使其数值尺度与原第四列匹配。它不等同于恢复“因果 baseline”：mean rate 同时受 unit identity、状态覆盖、recording conditions 和有限-support 噪声影响，也不能与 rSyn intercept 视为同一个统计量。

mean-rate4 另有独立 official `582228`：HO `0.6135910493256053`、HI `0.7954639949771956`、latency `0.1395917318632367`，收据 [OFFICIAL_582228.json](../../tfpd_exploration/submissions/evalai_m1_rift_mean_rate4_r100_v1/artifacts/OFFICIAL_582228.json)。它低于主 FULL `582205` `0.0124`、高于 ACT `582219` `0.0427`，**不替换**主对比的 muscle FULL。local HO3 e4 `0.5876280069351196` 不是 official。

M1 主比较采用各方法在相同完整 all-24 公开 HO3 规则下各自 earliest-maximum selected checkpoint；fixed e3 只作补充诊断，不要求同 epoch。该 candidate 的配对 seed42 已完成：在各自选中 checkpoint，candidate e4 `0.5876280069351196` 相对 original e3 的公开 HO3 差为 `+0.0185529887676239`；相对 ACTIVITY_ONLY e3 `0.5618270536263784` 为 `+0.025800953308741215`，三个 session 都为正。fixed e3 candidate `0.5861498316129049` 相对 ACTIVITY_ONLY 为 `+0.02432277798652649`，但 3/3 仅指各自 selected；fixed e3 仍有一个 session 下降。candidate 仍比历史 rSyn3 FULL `0.6544709205627441` 低 `0.06684291362762451`，未解决强基线差距；rSyn3 与 SVD carrier 的方法差异并非单一因素比较。candidate e24 相对 original e24 为 `-0.007281084855397468`，且在 e22–e24 低于 original42。三个 original seed 的各自选中值为：seed42 e3 `0.5690750181674957`、seed43 e3 `0.5517233908176422`、seed44 e6 `0.5669176677862803`；其 selected mean 为 `0.5625720255904727`，sample SD（ddof=1）为 `0.00945691268292847`。candidate 只有一个 seed，不能据此声称 replicated variance、一般性、因果机制或 official 表现；FULL−ACTIVITY_ONLY 是 side/E0 conditioning 与 direct T 的联合端到端增量，且没有已完成 CARRIER_ONLY 结果。[M1 carrier context comparison audit](../results/m1_muscle_r100_multiseed_v1/root_m1_carrier_context_comparison_v1.json) 已核验这些既有结果的 receipt 算术、all-24 同 query 与 candidate recipe binding。实际四条曲线的摘要见 [summary_all4_v1](../results/m1_muscle_r100_multiseed_v1/summary_all4_v1/README.md)，图见 [curves.png](../results/m1_muscle_r100_multiseed_v1/summary_all4_v1/curves.png)。

## 对“无偏移”的可检验表述

当前证据允许的 no-shift 表述是：若同一可比较 unit 在两个 session 具有相同的行为权重—rate 联合分布，满足有限矩与 LLN 条件，且 source basis/normalization 已冻结，则两个 conditional-response estimate 收敛到同一 population moment，进而收敛到同一 population carrier。该 carrier 通常非零；no-shift 表示估计对象不变，不表示应关闭 carrier。有限 M10/M3 的 pseudo-count、随机分母和噪声仍使两个有限样本 carrier 不必相同，因此不能保证 decoder performance non-inferiority，更不能声称数学上的零损害保证。`estimated shift=0` 也只是有限样本估计或门控事件，不等于真实 distribution shift 为零；任何 empirical non-inferiority 仍需在预先定义的相同评分面、配对比较和不劣容忍度下由数据支持。

## Prior-guided SVD 的术语边界

“prior-guided SVD”意为先由 task-specific behavior conditions 定义 conditional response，再对 source-frozen rows 做普通 SVD 压缩；当前 `np.linalg.svd` 的优化本身没有加入 constrained 或 weighted regularizer。condition mapping、nonnegative weights 与明确的“三维 relative response 加一维 level”分配属于结构/行为假设；pseudo-count 与 noise floor 属于正则化或尺度选择；source-only freeze、sign canonicalization 与 scaling 属于拟合和部署约定。完整区分见 [prior-guided SVD method note](PRIOR_GUIDED_SVD_METHOD_NOTE_20260910.md)。该候选现有一个 candidate seed 与三个 completed original seed；candidate 没有 replicated seed，不能将其写成一般改进或机制结论。
