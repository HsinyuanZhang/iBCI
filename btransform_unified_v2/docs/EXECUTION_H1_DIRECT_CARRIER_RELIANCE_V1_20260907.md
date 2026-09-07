# H1 直接 carrier 依赖：BT-EORT / RIFT 并行 CPU 推理

日期：2026-09-07。状态：H1 P0 已固定、BT-EORT / RIFT 两条正式 CPU 评分均运行中；本文不表示正式评分已完成。

## 授权与范围

用户批准先执行零训练的重要性诊断，明确 GPU 忙碌，并要求交给 subagents 并行做 BT-EORT 与 RIFT。两条线仅使用 CPU，不新增训练、不使用官方 test、不提交 EvalAI，不修改已有模型、训练代码、bank、checkpoint 或旧 submission。

共同任务选择当前两家族都有健康冻结权重的 H1。H1 的直接 carrier 是 **H-C**，不是 M2 的 MOVE-T4；结果必须使用准确名称。此处只研究直接 carrier 端口的使用依赖，保留 E0 内已有的 H-C 信息，不将其表述为全系统 functional information 的完全移除，也不作为匹配重训的增量效用证据。

## 并行所有权

| 责任 | Agent | 新文件 / 资源边界 |
|---|---|---|
| RIFT 评分、oracle parity、独立进程与结果 | rift_frontend | scripts/carrier_reliance_v1/run_h1_rift.py、独立 tests/results；CPU 14,15 |
| BT-EORT 评分、oracle parity、独立进程与结果 | rift_training | scripts/carrier_reliance_v1/run_h1_bt_eort.py、独立 tests/results；CPU 12,13 |
| 共用数据面、坐标、置换、组级统计 | rift_temporal | scripts/carrier_reliance_v1/common.py、相关 tests/manifest；CPU 10,11 |
| 协议裁决、资源协调与交付验收 | 主 agent | 本执行记录及整合证据 |

所有路径位于 btransform_unified_v2 下；现有核心和 v1/旧 submission 只读。使用隔离用户 site 的 conda Python、`CUDA_VISIBLE_DEVICES=''`、OMP/MKL 2 threads。正式 CPU 进程优先使用 `nice -n 10`，仅调整自己的新进程，不改变其他任务的资源配置。两条评分线不必串行等待彼此。

## 冻结参考

- RIFT：H1 R300-recency e22 EMA，来自 `results/rift_v1/h1_r300_pair_20260907T075312Z/recency_20260907_155347/formal/epoch_022.pt`。
- BT-EORT：封存 H1 C2-CAL-1 B2、P16、L200、e18 EMA，来自 `../tfpd_exploration/submissions/evalai_h1_c2_cal1_b2_ort_v1/` 的 payload 与 ORT 图。
- 权重、银行数组、normalizer 与源码 SHA 必须在 manifest 中绑定。并行比较每个模型内部的干预差，不把不同模型的差值差直接归因于架构，因为上下文和训练权重不同。

## 五臂合同

每个模型独立执行 REAL、C_ZERO、C_SHUF101、C_SHUF102、C_SHUF103。只有直接 carrier 改变；原 neural、E0、unit mask、normalizer、权重和所有 query 坐标保持不变。

置零指 C2 materializer 输出空间的数值零。现有 payload derivation 有 SHA 绑定，但没有明确的 centered / scale-only 字段，因此不将该操作解释为 raw source mean 或原始系数零。置换只在同一 H1 group 的有效 unit 行内进行，映射整段固定并覆盖其两个录音；不跨 group、不逐 bin 重抽、不同时置换 neural 或 E0。三个 shuffle seed 不是三个训练 seed。

共用 HO-M3 public held-out-calib 的 14 个录音，按既有 C2 scorer 合并为 S6–S12 七组。在查看干预分数前，将每组两个录音的合法 endpoint 列表按固定顺序拼接，以 `linspace` 等间隔选择最多 2048 个合法坐标，再映回各录音。这是合法时间库存覆盖抽样，不是每试次等权抽样。所有模型和所有臂使用同一坐标与置换清单，保留完整合法坐标库存及抽样 SHA；不因结果好坏改变覆盖或权重。

真实 CPU 数据预检已确认 14 个录音、7 组、每组 2048 个坐标，总计 14,336；query inventory SHA-256 为 `f6068c941a8a2c4bdbf32d5fbac798b879d09f8494bc81528c90a732d427d033`。模型 oracle 预检与正式评分的完成状态另以各自 receipt / result 为准。

这是预声明的 CPU 子样本诊断。REAL 不能强行与历史全量 HO-M3 数字配平；完整面历史分数只在同口径全量复现时比较。最终汇总复用 native-scale、variance-weighted R² 的 C2 组级定义，七组等权；shuffle 先在组内平均，再计算组间汇总。配对 bootstrap 以七组为单位，而不是把 14 个文件或相邻 windows 当独立重复。

## 开跑门与产物

1. 先完成共用 manifest / query inventory / permutations，不预展开整个 N×L×U 数据集。
2. 使用少量真实 endpoints 核验完整窗口 FP32 oracle 与加速实现，容差为 `1e-5 + 1e-5 * abs(reference)`；证明干预落在有效行、输出有限、原 bank 未变。
3. 每臂重建 bank snapshot、派生 static、frontend、raw 历史和 KV；REAL→干预→REAL 应恢复参考。不能沿用干预前缓存。
4. 先测无干预分数的 CPU 成本，再正式执行固定五臂；记录实际 PID、argv、环境、heartbeat、失败日志和运行耗时。
5. 输出原预测或可验证分片、坐标 / 置换 SHA、每组 R²、配对差、预测散布和组级 bootstrap。没有必须获得正结果的验收门。

本合同取自既有 EXP1 中的直接 C 路径子矩阵；不自动扩展到 E-ZERO/E-MEAN/E-SHUFFLE、688 或无 carrier 重训。

### 实际启动与预检证据

- 共享真实数据 P0：`results/rift_v1/carrier_reliance_v1_protocol/manifest.json` 与 `provenance.json`，包含逐数组 shape / dtype / bytes SHA、合法与抽样坐标 SHA、group-local 置换和源 payload derivation。完整合法库存 SHA 为 `51dd122236f55749a5679aa177c0be315f614dd81dbe66aa0c729b6c69596078`。
- BT-EORT：PID `1921489`，实际 receipt UTC `2026-09-07T12:29:20+00:00`，CPU 12,13 / nice 10；结果目录 `results/rift_v1/h1_bt_eort_carrier_reliance_v1_20260907T203400Z/`，同级 `.log`。目录名时间不是权威 UTC，以 receipt 为准。评分前已修正 target 使用 raw-timeline `selected_endpoints` 直接索引，不能用合法坐标列表中的 rank 索引完整 target。
- RIFT：`newresults/rift_v1/carrier_reliance_h1_20260907/preflight_20260907T2033Z/preflight.json` 已通过 28 个真实 endpoints 的 FP32 完整窗口对照（最大绝对误差 `4.4238e-9`）、五臂抽查与 fresh REAL 恢复（恢复误差 0）。正式 PID `1921668`，CPU 14,15 / nice 10，结果目录 `newresults/rift_v1/carrier_reliance_h1_20260907/run_20260907T2037Z/`。
- BT-EORT 最终 oracle 证据读取 `report.json` 的逐 group / arm execution 检查（每次要求 32 个实际 endpoint、偏差即退出），启动回执不替代 oracle 通过证明。该已启动进程使用终态 heartbeat 修订前的 runner，最终 heartbeat 可能仍为 RUNNING；成功判定使用 `report.json.status == COMPLETE`、恢复检查和 payload before / after SHA 相等。
- 配对效果统一记为 `REAL − arm`，正值表示干预造成 R² 下降；bootstrap 固定七组、2000 次、seed 20260907；三置换先在每组内平均。

## 后续已授权：M2 concat 的 MOVE-T4

用户于 12:19 UTC 追加要求：H1 对照之后，测试 M2 MOVE-T4 的 BT-EORT **concat** 版本。该项作为下一项排队任务，不混入 H1 的 H-C 表，不替换为 proj_add 或旧 SPINT。候选入口为 `../tfpd_exploration/submissions/evalai_m2_small_concat_ort_v1/`，必须先从其实际 receipt 绑定具体权重、geometry、bank、normalizer、合法本地评分面与训练暴露，再建立独立 M2 manifest / results。

仍为 CPU-only 的五臂冻结模型推理；不重训、不使用官方 test、不提交。H1 共同协议和两条执行线优先完成，M2 可先做只读定位，正式评分在 H1 对照收口后接续。M2 的 E0 若含 T4，直接 carrier 干预同样不能称为全部 T4 信息移除。

M2 后续由 `rift_temporal` 独占新 `scripts/carrier_reliance_v1/run_m2_bt_eort_concat.py`、独立测试、协议和 CPU 接续 launcher；两 H1 worker 只负责各自 H1 收口。接续必须检查两个 H1 成功结果文件的状态，不能把 PID 消失当作成功；M2 使用 CPU 12,13 / two threads / nice 10。已有 M2 submission 文件及其外部更新中的 push-state 保持只读。前置准备和 launcher 在实现期间不等于 M2 正式评分已开跑。
