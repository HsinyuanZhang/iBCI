# Carrier v4：已完成证据与实验进度

本报告随正式实验更新。当前已完成 M1 机制诊断、P0 数值复现、共享 core 的 M1 exact 部署、H1 source fit 与 27-tag 四信息臂 bank、融合和信息路由审计、M1 ACTIVITY_ONLY 全 24 epoch 重评分。H1 FULL/P16 已完成全部 32 个 epoch、23,392 次更新并保存 32 个 checkpoint，正在进行全 checkpoint 的 HO-M3 评分；新 v4 完整正式评分结果尚未产生。训练完成的实际记录见 [metrics.jsonl](../results/carrier_v4/h1_full_proj_add_p16_formal_v1/metrics.jsonl)。后续对照由 [串行队列](../results/carrier_v4/remaining_gpu1_manifest_v1.json) 执行。所有 GPU 运行限制在物理 GPU1。

当前 [机器汇总](../results/carrier_v4/summary_progress_v2/summary.json) 和 [逐 epoch CSV](../results/carrier_v4/summary_progress_v2/curves.csv) 按最终 `experiment_manifest_v2` 生成，包含十二个主实验单元及一个 H1 external baseline；目前完整记录为 M1 两个复用单元和 H1 external baseline，其余十个单元仍为 pending。[M1 图](../results/carrier_v4/summary_progress_v2/figures_v3/m1_results.svg) 与 [H1 图](../results/carrier_v4/summary_progress_v2/figures_v3/h1_results.svg) 只绘制完成评分的数值，并区分 fixed/selected endpoint、双侧差值区间和单侧非劣判据。图可用 [plot_results.py](../scripts/carrier_v4/plot_results.py) 从同一 summary JSON 重建。

## 已知的 M1 性能差异

主指标均为每个 HO calibration session 的 channel-centered variance-weighted R²，再对三个 session 等权平均。FULL rSyn3、ACTIVITY_ONLY 与 muscle 三条线都完整扫描 24 个 EMA，均选 e3。

| 已完成模型 | e3 / selected mean R² | 相对 FULL rSyn3 |
| --- | ---: | ---: |
| FULL rSyn3，live B3S + direct T | 0.654471 | 0 |
| ACTIVITY_ONLY，live B3S(raw, 0)，direct T=0 | 0.561827 | −0.092644 |
| muscle-response16 / SVD4 / global-RMS | 0.569075 | −0.085396 |

ACTIVITY_ONLY 是原始完整训练的 B arm，不是将 FULL checkpoint 的输入清零。此次 [B replay receipt](../results/carrier_v4/m1_activity_baseline_replay_v1/replay_receipt.json) 重评分了全部 24 个 checkpoint，原 legacy metric 的误差全部为零，并与 [D replay authority](../results/m1_muscle_r100_v1/baseline_replay/replay_receipt.json) 核验相同 source、初始化、HO targets 和窗口。B/D 每个 session 分数如下：

| Session | FULL rSyn3 | ACTIVITY_ONLY | ACTIVITY_ONLY − FULL |
| --- | ---: | ---: | ---: |
| 20121004 | 0.749535 | 0.702689 | −0.046846 |
| 20121017 | 0.607642 | 0.506664 | −0.100978 |
| 20121024 | 0.606236 | 0.476128 | −0.130107 |

因此当前 M1 证据不支持“移除 carrier 后非劣”。三个 session 全部下降，差异大于事前声明的 R² 0.01 容忍度。NONE/CARRIER_ONLY 和融合宽度实验仍须完成；不能据此替代其他信息臂的结论。

对三个 session 的配对差异进行全部 `3^3=27` 种等概率有放回重采样，均值差为 −0.092644，单侧 95% 下界为 −0.120398，双侧 95% 区间为 [−0.123796, −0.058575]。fixed e3 与 selected-each 两种比较在这组已完成结果上相同；该区间仅描述当前三个公开校准 session 和 seed42。

[训练复用审计](../results/carrier_v4/m1_legacy_concat_reuse_audit_v1.json) 将旧 D/B 两个已完成训练分别绑定为 v4 FULL/ACTIVITY_ONLY concat 的 authority。它依据实际相同的 carrier、初始化、完整 forward、source/sampler/优化与 EMA 合同，以及原 formal run 和全 epoch replay；没有声称在新 wrapper 下重放全部 159,960 个 optimizer updates。新 runner 的 resume 仍拒绝旧 schema。

## M1 官方私有测试证据

muscle-response R100 的 e3 提交 **582205 已完成**，官方 `test_split_m1` 返回 Held Out R² **0.625991**（std **0.089112**）、Held In R² **0.792630**（std **0.021132**）、Normalized Latency **0.140405**。该 e3 在提交前由公开 HO3 的完整 24-epoch 扫描选定，官方结果未参与选择。ROOT 已通过官方 GET 与 `evalai submission 582205 result` 直接核实，完整精度与时间记录见 [OFFICIAL_582205.json](../../tfpd_exploration/submissions/evalai_m1_rift_muscle_r100_v1/artifacts/OFFICIAL_582205.json)。本地 REGISTERED 已同步为 finished，保留原 push/register 的历史 snapshot。

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

这不是仅改变四列的统一尺度：它明确去掉了 unit mean-rate 项。即使在 source3 上重拟合 per-column normalizer，新的第四列与原截距在所有 7 个 session 的单元间相关性仍为负，范围约 −0.483 至 −0.188；三个 slope 的相关性约 0.924–0.998。该证据支持保留 raw-Hz response 与 unpenalized intercept 作为 v4 的主线，但还不能单独量化 rate 标准化造成的 decoder 损失。

另一个独立因素是 spike bin 时钟。旧 reader 与 official end-timestamp reader 的同索引 count correlation 仅约 0.446–0.558；尝试一格平移后约 0.799–0.877，仍非完全一致。因此不能把它描述为已经发现并修复的单纯一格 shift bug。v4 M1 主线保留旧 reader，避免将时间约定和 estimator 同时改动。

## v4 的实际统一边界

两任务共享 [common_estimator.py](../scripts/carrier_v4/common_estimator.py) 的 source behavior RMS、非负状态字典、固定字典 NNLS、逐 unit raw-Hz ridge 与 unpenalized intercept、source per-column normalizer。carrier 均为三个 slope 加一个 intercept。任务 adapter 只负责行为映射、source roster、合法 support 与数据读取。

M1 使用 ReLU EMG16，并导入封存 source3（26/27/28）fit；24 仍参与 decoder 训练，但其 carrier 与 HO3 一样由 source3 fit 部署。H1 使用按 source RMS 缩放的 signed-softplus velocity14，在 source13 的训练行为行拟合 NMF3，排除各 source 最后两个 validation trials。H1 新 source fit 于 146 次迭代收敛；四个信息臂共享同一 fit JSON SHA。

当前 H1 signed-state14 的逐列 conditional mean，与有完整交叉项的 joint ridge 是不同估计器。v4 是在 H1 实施后一条路径的真实新实验；不会把改写名称当作算法统一。数学审计见 [CARRIER_V4_REVIEW_AUDIT_20260909.md](CARRIER_V4_REVIEW_AUDIT_20260909.md)。

共享 `lambda=1` 不代表两任务具有相同的有效收缩强度。当前实现仅将 NMF dictionary 的行做 L2 normalization，没有在 ridge 前将 NNLS activation `Z` 逐列标准化。消去不受惩罚的截距后，令 `C=Z_centered.T @ Z_centered / n`，斜率解为 `(C+I)^-1 @ Z_centered.T @ rate_centered / n`；在 `C` 的特征值为 `d` 的方向，相对 centered least-squares 的收缩乘子为 `d/(d+1)`。最终 raw-carrier 的 source per-column normalizer 位于回归之后，不改变这一收缩。该代数事实限定了“共享估计器”的含义；未经实际 source covariance 和 decoder 对照，不能将其当作性能变化的已证实原因。

[P0 receipt](../results/carrier_v4/p0_exact_m1_v2/p0_exact_m1_receipt.json) 显示，统一模板 raw 与当前旧生产代码在 7 个 tag 全部完全一致。历史 sealed source3 float64 raw 最多相差 `2.132e-13`，单独记录该事实；最终 float32 T 在 7 个 tag 上与原模型输入逐字节一致。随后 [共享 core 的真实部署 pack](../results/carrier_v4/m1_exact_pack_v1/carrier_pack.json) 独立重算并通过同一个 7-tag 硬断言，没有用 P0 预计算 T 替代 core 运算。

## 信息与融合对照的执行合同

每任务在 reference fusion 上训练 FULL / ACTIVITY_ONLY / CARRIER_ONLY / NONE；FULL 另比较 concat、proj-add P16、proj-add P32。M1 reference 为 concat，H1 reference 为 P16。local convolution width 仍为 16，temporal width/depth 仍为 256/D4。P32 只改变 E0 projection 宽度。

四臂分别为 `(f(activity,T),T)`、`(f(activity,0),0)`、`(0,T)`、`(0,0)`。M1 FULL/ACTIVITY_ONLY 保留原 live B3S 训练；另外两臂不调用 encoder。H1 ACTIVITY_ONLY 重新 materialize E0，防止 FULL E0 泄露 carrier。实际 27-tag 四臂 audit 见 [h1_actual27_arm_audit_v1.json](../results/carrier_v4/h1_actual27_arm_audit_v1.json)。

实际模型参数量见 [model_parameter_counts_v1.json](../results/carrier_v4/model_parameter_counts_v1.json)：

| Task | FULL concat | FULL proj-add P16 | FULL proj-add P32 |
| --- | ---: | ---: | ---: |
| M1，含 live B3S | 3,638,292 | 3,614,292 | 3,619,988 |
| H1 | 3,710,055 | 3,542,055 | 3,557,351 |

M1 live B3S 有 80,676 个参数。在 NONE/CARRIER_ONLY 中它不参与 forward，参数从 optimizer/EMA 更新中排除；这两臂的 concat 可训练参数量为 3,557,616，neural decoder 本身不变。H1 的 C2 bank 在模型外构造，P16 四臂的 decoder 可训练参数量均为 3,542,055。不声称 concat、P16、P32 的参数量完全相同。

M1 完整执行 R100、24 epochs；H1 完整执行 R300、32 epochs，保留 C2 M7/M5/M4/M3 schedule 及旧 M3/M4 carrier-prefix 分配。初始化、query windows、valid masks、sampler、dropout、optimizer、EMA 与各任务旧 formal recipe 配对；允许的信息路由与 fusion 改动明确写入 checkpoint 身份。

评分同时报告固定 e3（M1）/e16（H1）和各自全曲线 earliest-maximum selected epoch。非劣容忍度事前声明为 R² 0.01。配对区间使用 M1 三个 session 与 H1 七个 grouped sessions（S6–S12），不能把 H1 的十四个 tag 当作十四个独立 session。所有结果属于 public calibration、seed42 的 development evidence；不由“未观察到增益”推出总体零效应，也不由不显著推出非劣。

H1 的主 external baseline 是当前 signed-state 正式结果：e16、grouped mean R² **0.467651**。旧 H-C/582073 的 0.374823 只作历史背景；runner 中保留的 `versus_582073` 字段不能取代对当前 signed-state baseline 的比较。[实验 manifest](../results/carrier_v4/experiment_manifest_v2.json) 明确区分 external baseline、同 carrier 的信息比较与融合比较。M1 以 P16−concat 检查融合，以 P32−P16 检查宽度；H1 以 concat−P16 和 P32−P16 做对应比较。

H1 首组评分的阶段记录：固定 e16 已在 [实际评分日志](../results/carrier_v4/h1_full_proj_add_p16_formal_v1.log) 中完成，v4 FULL/P16 grouped mean 为约 **0.352676**，相对旧 signed-state 同 e16 下降约 **0.114975**；v4 worst-session 为约 **0.199441**。这里使用日志保留的六位小数，完整 JSON 与七组逐 session 分数仍须等待全 32 个 checkpoint 扫描结束。该结果已说明固定 endpoint 下的下降，但尚不能替代各自完整曲线的最佳 epoch 比较，也不能单凭它归因于 NMF、截距或 ridge 尺度中的某一项。
