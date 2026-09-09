# M1 muscle carrier：正式四 source 重训结果（2026-09-09）

## 结论

在公开 HO3 calibration 比较中，正式的四 source muscle carrier 重训未超过冻结旧基线 replay。按预先声明的选择规则，两个指标都在 epoch 3 取最早最大值；下表的公开 HO3 新旧差均为负值。这个结论不把公开 HO3 的差异延伸为 official test 的退化。

| 指标（equal-session mean） | 新 muscle carrier | 旧基线 replay | 新 − 旧 |
| --- | ---: | ---: | ---: |
| Channel-centered variance-weighted R² | 0.5690750181674957 | 0.6544709205627441 | -0.08539590239524841 |
| Legacy flattened R² | 0.668062063990566 | 0.7342600300996954 | -0.06619796610912942 |

因此，先前 chronological 实验中的观察不能替代本次正式的四 source 训练与公开 HO3 评估，也不足以支持从该公开比较宣称 muscle carrier 优于旧基线。这里仅报告已观测到的差异，不推测下降机制。

## 正式运行与选择规则

- 新运行目录：[formal_s42_gpu1](../results/m1_muscle_r100_v1/formal_s42_gpu1/)，状态 `COMPLETED`，`seed=42`。
- 训练为 24 epochs、每 epoch 6,665 updates、总计 159,960 updates；具体配置见 [run_meta.json](../results/m1_muscle_r100_v1/formal_s42_gpu1/run_meta.json) 与 [train_receipt.json](../results/m1_muscle_r100_v1/formal_s42_gpu1/train_receipt.json)。
- carrier 为 `muscle_response16_svd4/global_rms`，四个 source session 为 `ses-20120924`、`ses-20120926`、`ses-20120927`、`ses-20120928`；HO session 为 `20121004`、`20121017`、`20121024`。
- 选择只扫描训练后 EMA epoch 1–24，依据可见 HO3 calibration 的 earliest maximum equal-session mean Channel-centered variance-weighted R²；主指标 selection 为 epoch 3。独立的 legacy 选择也为 epoch 3。选择及可重复性记录见 [score_receipt.json](../results/m1_muscle_r100_v1/formal_s42_gpu1/score_receipt.json)。
- 此评估未使用 official test；选择与比较均限于记录中的可见 HO3 calibration surface。

先前 GPU0 队列在 epoch 1 checkpoint 前因用户请求停止（`USER_STOPPED`，exit `-15`）；随后保留中止目录并在物理 GPU1 上以不变 recipe 从头运行。该 restart 记录为 [gpu0_interrupted_queue.json](../results/m1_muscle_r100_v1/gpu0_interrupted_queue.json) 和 [gpu1_restart.json](../results/m1_muscle_r100_v1/gpu1_restart.json)。GPU1 运行在 `CUDA_VISIBLE_DEVICES=1` 下，以逻辑 `cuda:0` 启动，表示唯一可见设备的物理 GPU1。

## EvalAI official `test_split_m1` 结果

submission `582205` 已完成。其提交、开始与完成时间分别为 `2026-09-09T11:26:48.359836Z`、`2026-09-09T12:17:32.721602Z` 和 `2026-09-09T12:17:33.023719Z`。官方结果记录在 [OFFICIAL_582205.json](../../tfpd_exploration/submissions/evalai_m1_rift_muscle_r100_v1/artifacts/OFFICIAL_582205.json)，提交身份同步记录在 [REGISTERED.json](../../tfpd_exploration/submissions/evalai_m1_rift_muscle_r100_v1/artifacts/REGISTERED.json)。

| `test_split_m1` 指标 | Exact official value | Rounded display |
| --- | ---: | ---: |
| Held Out R² mean | 0.6259910151098528 | 0.6260 |
| Held Out R² std. | 0.08911240389082414 | 0.0891 |
| Held In R² mean | 0.7926300410539316 | 0.7926 |
| Held In R² std. | 0.021132442722184144 | 0.0211 |
| Normalized Latency | 0.14040520785877766 | 0.1404 |

EMA e3 在提交前已经由完整的公开 HO3 scan 选定；official result 未参与 checkpoint selection。官方 `test_split_m1` 指标与上文公开 HO3 指标属于不同评估 surface，不能彼此相减或替代。

截至本记录，旧 rSyn3/RIFT joint-D e3 的 submission `582150` 已取消，较早的同配置 `582133` 也已取消；二者均没有 completed official score。因此没有与 `582205` 同配置的 joint-D official baseline，不能从 `582205` 的 official 数字推断相对 joint-D 的 official 改善或退化。尤其，旧 replay 的 channel-centered `0.6544709205627441` 和 legacy `0.7342600300996954` 都只是公开 HO3 数值，不是 official 指标。

## 三个 HO session 的 Channel R²

| HO session | 新 muscle carrier | 旧基线 replay | 新 − 旧 |
| --- | ---: | ---: | ---: |
| `20121004` | 0.7006857991218567 | 0.7495353817939758 | -0.04884958267211914 |
| `20121017` | 0.5387488007545471 | 0.6076418161392212 | -0.06889301538467407 |
| `20121024` | 0.4677904546260834 | 0.6062355637550354 | -0.13844510912895203 |
| equal-session mean | 0.5690750181674957 | 0.6544709205627441 | -0.08539590239524841 |

完整逐 epoch 曲线见 [PNG](../results/m1_muscle_r100_v1/comparison/epoch_curves.png) 与 [PDF](../results/m1_muscle_r100_v1/comparison/epoch_curves.pdf)。机器可读数据为 [comparison.json](../results/m1_muscle_r100_v1/comparison/comparison.json) 和 [comparison.csv](../results/m1_muscle_r100_v1/comparison/comparison.csv)。

## 比较审计

比较结果的 `target_and_starts_exact` 为 `true`。三个 HO session 的 target 与 window starts 在新运行和旧基线 replay 之间逐数组一致；comparison 流程同时审计 dtype、shape、原始数组 SHA-256 与数组相等性。三个 session 的 window 数分别为 1,305、1,295 与 1,281，总数 3,881。

冻结训练脚本 SHA-256 为 `abfa898b6304e6d8a64dabe35d246ba5dd94fcd8e5def5ab486fcacaa1885f61`；carrier 实现 SHA-256 为 `1c23fd0e76303a9f261ebf17d2d9425696562d9606fb3614affe7168400fa96d`。新运行与旧基线的非 carrier 初始化及 source contract 匹配，记录在 [run_meta.json](../results/m1_muscle_r100_v1/formal_s42_gpu1/run_meta.json)。

## Submission preparation、提交与完成状态

submission preparation 阶段已完成：payload 与本地 image 均已生成。live joint raw-M10 与 sealed concat 的逐 session seal validation 最大绝对误差为 `0`；host 120-step 验证中 B1/B3/B4 的最大绝对误差最高为 `4.76837158203125e-07`，full-window streaming 最大绝对误差为 `2.384185791015625e-07`。容器 smoke 已通过；首次仅修正 smoke fixture 的 dataset tag 为完整 Falcon 文件名 stem 后复测通过，payload 与 image 均未改变。只读 preflight 当时为每日 `3/6`、并发 `0/3`。这些是提交前的封包历史记录。

准备记录为 [preparation_receipt.json](../../tfpd_exploration/submissions/evalai_m1_rift_muscle_r100_v1/artifacts/preparation_receipt.json)，候选信息为 [evalai_candidate.json](../../tfpd_exploration/submissions/evalai_m1_rift_muscle_r100_v1/artifacts/evalai_candidate.json)，操作说明见 [submission README](../../tfpd_exploration/submissions/evalai_m1_rift_muscle_r100_v1/README.md)。后续已 push、register 并完成为 submission `582205`；其 final official receipt 见上文 [OFFICIAL_582205.json](../../tfpd_exploration/submissions/evalai_m1_rift_muscle_r100_v1/artifacts/OFFICIAL_582205.json)。这项官方完成不改变公开 HO3 的旧 baseline replay 结论，也不制造缺失的 joint-D official 对照。有关先前 adaptive carrier 研究的范围与受限结论，见 [Carrier Adaptive v3 结果记录](CARRIER_ADAPTIVE_V3_RESULTS_20260909.md)。
