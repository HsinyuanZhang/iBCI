# pseudo-MUA T4 bridge：48 小时执行与结果日志

**实验 ID：** `pseudomua_t4_bridge_v1`  
**状态：** `COMPLETE`  
**开始时间：** 2026-07-27 21:37 HKT  
**完成时间：** 2026-07-28 04:31 HKT  
**最晚交付时间：** 2026-07-29 21:37 HKT  
**数据范围：** DANDI 000688，sub-C，CO，`27 train / 6 validation / 6 held-out test`  
**证据等级：** validation development evidence；不是 formal held-out test

> **最终结论（2026-07-29 回填）：** 冻结的 9/9 个 pseudo-MUA run 与严格聚合
> 全部完成。T4 在 pseudo-MUA 上相对 F0 为 `+0.3177 ± 0.0239 SE`，相对 TS4
> 为 `+0.3657 ± 0.0308 SE`，两组比较均为 6/6 validation session、3/3 seed
> 全正，组级判定 **`effective`**。因此 T4 增益在 electrode pooling 后明确保留。
> `Gamma=(T4-F0)_SUA-(T4-F0)_pseudo-MUA=-0.0650 ± 0.0307 SE`，按预注册
> `±0.03` interaction tolerance 判为 **`indeterminate`**：点估计偏向
> pseudo-MUA amplification，但不能声称 signal-view-specific amplification。

## 1. 冻结问题

本实验只回答：

> 将同一 electrode 上的 sorted SUA spike counts 求和为 pseudo-MUA 后，T4 的增益是否
> 保留，并且该增益是否依赖正确的 channel–T4 对应关系？

第一轮不加入 B3T、其他 encoder、其他 MUA threshold、unit-count matching 训练、
额外 architecture sweep 或 formal test。

## 2. 冻结数据变换

对 session 内每个 electrode：

```text
pseudo_mua[e, t] = sum(sua[u, t] for electrode(u) == e)
```

electrode-level T4 必须在上述聚合完成后，由 pooled calibration-trial firing rate
重新拟合：

```text
T4[e] = [m_e*cos(phi_e), m_e*sin(phi_e), m_e, b_e]
```

禁止对 unit-level T4 做简单平均。T4 normalization 只允许用 train sessions
拟合；validation/test 不得参与统计量估计。TS4 只在 session 内置换
`channel <-> T4 row`，不得改变特征列的边缘分布。

## 3. 冻结运行矩阵

已有 SUA E3 artifact 只读复用：

| Signal view | Group | Variant | Side feature | Seeds | 本轮训练 |
|---|---|---|---|---|---|
| SUA | F0 | B3 | none | 42,43,44 | 否 |
| SUA | T4 | B3S | t4 | 42,43,44 | 否 |
| SUA | TS4 | B3S | ts4 | 42,43,44 | 否 |
| pseudo-MUA | F0 | B3 | none | 42,43,44 | 是 |
| pseudo-MUA | T4 | B3S | electrode-level t4 | 42,43,44 | 是 |
| pseudo-MUA | TS4 | B3S | shuffled electrode-level t4 | 42,43,44 | 是 |

新增训练总数固定为 9；两张 GPU 使用共享工作队列。不得在看到结果后临时增加 arm。

## 4. 冻结训练与评价协议

- `max_epochs = 12`
- `burn_in = 4`
- score window：epoch 5–12 算术平均
- `no_early_stopping = true`
- `checkpoint_every_epoch = true`
- evaluation：`selection_mode=first`、`calibration_n=30`、`pool_size=50`
- `seeds = 42,43,44`
- `task=CO`
- `max_units_exclusive=100`
- `loss_mode=task_only`
- `identity_mode=calibrated`
- formal test spike、behavior、trial 一律不加载

每个 `(group, seed)` 必须拥有独占 run directory。目标目录若含 checkpoint 或
tfevents，runner 必须拒绝覆盖；只有完整且通过 provenance 校验的 evaluation
artifact 才可跳过。

## 5. 启动前硬 gate

以下项目全部通过才允许启动 9-run：

- [x] 逐 time-bin spike-count conservation
- [x] unit-axis permutation invariance
- [x] singleton-electrode identity
- [x] 每个 sorted unit 恰好映射一个 electrode
- [x] pseudo-MUA T4 row 数等于 pooled channel 数，且全部 finite
- [x] SUA 与 pseudo-MUA feature/cache namespace 不串用
- [x] train-only normalization；validation/test 无统计量泄漏
- [x] TS4 只置换 rows，并保持每列数值集合/统计量
- [x] 现有 SUA 路径/cache key 不变
- [x] 一个真实 train session 与一个真实 validation session smoke 通过
- [x] 两张 GPU 无冲突训练进程且显存足够

任何 gate 失败均 fail-fast，不允许用 warning 绕过。

## 6. 冻结主要估计量

```text
Delta_pMUA         = score(T4_pMUA) - score(F0_pMUA)
Delta_content_pMUA = score(T4_pMUA) - score(TS4_pMUA)
Delta_SUA          = score(T4_SUA)  - score(F0_SUA)
Gamma              = Delta_SUA - Delta_pMUA
```

所有差值在相同 session、相同 seed 上配对。判定使用逐 seed mean delta 的标准误：

```text
sigma_delta_paired = stdev(seed_mean_delta, ddof=1) / sqrt(n_seeds)
```

同时输出 unpaired quadrature 与隐含 seed correlation，不能把两臂当作独立 seed。

单 pair 的 V4 四态判定：

- `effective`：mean delta ≥ `+0.03`，至少 5/6 session 为正，3/3 seed mean 为正；
- `effective_heterogeneous`：`mean - 2*sigma_delta_paired > 0`，3/3 seed mean 为正，
  但 session 一致性不足；
- `ineffective`：`mean + 2*sigma_delta_paired < +0.03`；
- `indeterminate`：以上均不满足。

T4 的组级 `effective` 要求 `T4-F0` 与 `T4-TS4` 都有效；任一 pair 被判为
`ineffective`，组级即为 `ineffective`。

Gamma 作为 view interaction 单独报告：

- `Gamma - 2*SE > +0.03`：SUA-specific amplification；
- 区间完全落在 `[-0.03, +0.03]`：view-invariant within tolerance；
- `Gamma + 2*SE < -0.03`：pseudo-MUA amplification；
- 其余：indeterminate。

## 7. 预定时间线

| 截止点 | 目标 |
|---|---|
| T+2h | 最小实现、缓存隔离与测试就绪 |
| T+4h | 自动测试和真实数据 smoke 通过，启动 9-run |
| T+12h | 预计训练与 validation evaluation 完成 |
| T+18h | paired aggregation、interaction 与诊断表完成 |
| T+24h | 稳定版第一结论 |
| T+48h | 失败重跑或预注册的追加 seed 后最终开发结论 |

如果 3-seed 结果为 `indeterminate`，只有在 9 个原始 cell 全部完整、协议未偏离且
剩余时间允许时，才可追加同三 arm 的 seeds 45–47。不能根据观察到的正负方向选择
新 arm。

## 8. 实时执行日志

### 2026-07-27

- **21:37 HKT — START。** 冻结本文件中的问题、运行矩阵、训练协议和判定规则。
- **21:37 HKT — GPU baseline。**
  - GPU0：RTX 3090，23848 MiB free，1% utilization，44°C。
  - GPU1：RTX 3090，24242 MiB free，0% utilization，33°C。
  - 未发现 `train_variant_dandi688`、epoch-window evaluation 或 pseudo-MUA 训练进程。
- **21:38 HKT — IMPLEMENTING。** Terra 子代理开始实现 signal-view-aware
  pseudo-MUA T4/TS4、自动测试、专用 runner 和 aggregator。全量训练受 §5 gate
  约束。
- **21:42 HKT — SUA REFERENCE PINNED。** 从只读 E3 epoch-window artifacts
  (`results/e3_tuning_ablation/{f0,t4,ts4}_s{42,43,44}.json`) 重新计算 SUA
  reference。F0/T4/TS4 分别为 `0.313987 / 0.566749 / 0.314516`；
  `T4-F0=+0.252761`（paired SE `0.007844`，6/6 sessions、3/3 seeds 为正），
  `T4-TS4=+0.252233`（paired SE `0.012449`，6/6 sessions、3/3 seeds 为正）。
  这些 artifact 只读复用，不重新训练。
- **21:43 HKT — SYNTHETIC GATE PASS。** pseudo-MUA bridge 的 6 个快速测试由
  Terra 与主代理独立各运行一次，均为 `6 passed`。覆盖逐 bin conservation、
  unit-order invariance、singleton identity、单 electrode 映射、T4
  channel shape/finite、TS4 row-only permutation 和 SUA/pseudo-MUA cache 隔离。
- **21:47 HKT — REAL-DATA PARTIAL SMOKE PASS。** 显式 train session
  `sub-C_ses-CO-20131003` 从 71 source units 聚合为 47 channels，T4 shape
  `(47,4)`；`sub-C_ses-CO-20131022` 从 41 units 聚合为 30 channels，T4 shape
  `(30,4)`。两者 neural/calibration/T4 channel 轴一致且全部 finite。注意：
  `20131022` 仍属于 train split，不能作为 validation smoke 证据；已要求补跑冻结
  validation session `sub-C_ses-CO-20151103`，因此真实数据 gate 暂未最终放行。
- **21:49 HKT — FROZEN-VAL SMOKE PASS。** normalization 仍只由显式 train
  `sub-C_ses-CO-20131003` 拟合；冻结 validation session
  `sub-C_ses-CO-20151103` 从 38 source units 聚合为 25 channels，
  `calib=(10,100,25)`、`T4=(25,4)`、validation windows=35032。neural/calibration/T4
  channel 轴一致、T4 finite；seed 42 的 TS4 只置换 rows 且保持每列分布。
  Python status 0，未加载 held-out test。真实数据 gate 放行。
- **21:53 HKT — ADJACENT REGRESSION AUDIT。** bridge tests 与 side-feature
  encoder tests 通过；联合测试中共 100 项通过。另有 19 项失败全部来自仓库既有
  `test_aggregate_e3_tuning_ablation.py` 对旧 helper 参数名和旧三态 API 的断言，
  而当前共享实现已采用 2026-07-27 的 paired-SE/four-state API。相关 E3
  aggregator/test 文件在本实验开始前即存在且本轮未修改；该历史测试不一致不作为
  bridge 阻塞，但新 aggregator 必须用自己的 artifact-level fixtures 固定当前协议。
- **21:56 HKT — SUA ARTIFACT PROVENANCE PASS。** 新 aggregator 的 strict loader
  对现有 SUA E3 的 F0/T4/TS4 × seeds 42/43/44 共 9 个 artifacts 做只读校验，
  signal view、variant/side group、epoch 5–12、metadata SHA-256、fixed 12-epoch
  training 与 no-test provenance 全部通过。正式 source artifacts 未修改。
- **21:57 HKT — LAUNCH REVIEW BLOCKED。** 首版 runner/aggregator 虽通过 syntax
  和基础 dry-run，但主审发现 runner 仍是两任务 lockstep、result validator 读取了
  不存在的顶层字段，以及 Gamma SE/interaction verdict 和 V4
  `effective_heterogeneous` 定义不符合冻结协议。已退回 Terra 修复；GPU 未启动。
- **22:03 HKT — FINAL PREFLIGHT PASS。** 主代理完成 launch 级审查和修复：
  runner 使用两个长期 worker 与原子 claim-next-job 队列；self-test 证明快 worker
  会在 slow job 完成前继续领取后续任务，故障 job 后仍 drain queue，并向
  orchestrator 传播失败。dry-run 精确输出 9 个 train→eval jobs 且不写 result
  目录。bridge + artifact-level aggregator fixtures 共 `14 passed`；aggregator
  严格校验现有 SUA E3 9/9 artifacts。GPU0/GPU1 分别有 23848/24242 MiB free，
  无 compute process。正式 screen ID 固定为 `pseudomua_t4_bridge_v1`，进入 launch。
- **22:03 HKT — FORMAL LAUNCH。** 前台 orchestrator PID `552000`；动态 worker
  PIDs `552076/552077`。首批任务为 GPU0 `F0 seed=42`（train PID `552095`）和
  GPU1 `T4 seed=42`（train PID `552097`）。manifest 已记录完整 train/eval
  命令、GPU、worker PID 和时间。启动后的首阶段正在 CPU 扫描/构建 dedicated
  pseudo-MUA cache，因此 GPU utilization 暂为 0%；两项训练进程均存活且 CPU
  活跃，不是挂起。
- **22:18–22:25 HKT — T4 CACHE ACCELERATION。** train-only T4 stats 原子落盘后，
  首次 session feature 构造原为单进程。使用 4 个 CPU workers 对冻结的 27 train
  + 6 validation sessions 做受限 prefetch；所有 worker 复用同一 train-only
  mean/std 和带锁原子 cache，未打开 test sessions。33/33 features 在约 2 分钟内
  完成，validation shapes 为 `(25/40/39/41/39/33, 4)`，与 electrode counts
  完全一致。
- **22:26 HKT — BOTH GPUS TRAINING。** GPU0 F0/s42 已写 4 个 epoch checkpoints；
  GPU1 T4/s42 已从 cache 构造切换到训练，显存分别约 1086/626 MiB、两卡利用率约
  40–70%。首次 T4 的 CPU-only 等待结束，后续 T4/TS4 jobs 将直接命中 cache。
- **2026-07-28 04:31 HKT — COMPLETE。** F0/T4/TS4 × seeds 42/43/44 共 9/9
  jobs 均以 `rc=0` 完成；每个 run 均有完整的 12 个 epoch checkpoints 与
  epoch-5..12 evaluation artifact。严格 aggregator 校验全部 SUA/pseudo-MUA
  artifact、run metadata SHA-256、split、signal view、训练协议与 no-test
  provenance 后生成
  `results/pseudomua_t4_bridge_v1/summary.json`。manifest 以
  `screen_done/status=completed/rc=0` 收尾。
- **2026-07-29 — POST-COMPLETION AUDIT。** 定向回归
  `test_pseudomua_t4_bridge.py` + `test_aggregate_pseudomua_t4_bridge.py`
  为 `14 passed`。进程与 GPU 复查确认无遗留训练任务。本文档由 `RUNNING`
  回填为 `COMPLETE`；所有数字直接来自冻结 `summary.json`，未重跑或选择性增加
  arm/seed。

## 9. 最终结果

### 9.1 构造诊断

| Session | SUA units | pseudo-MUA electrodes | Singleton electrodes | Multi-unit electrodes | Mean units/electrode |
|---|---:|---:|---:|---:|---:|
| train 27 sessions（range/mean） | 41–91 / 59.74 | 27–56 / 37.63 | 9–31 / 20.41 | 10–26 / 17.22 | 1.59 |
| val: 20151103 | 38 | 25 | 16 | 9 | 1.52 |
| val: 20151104 | 59 | 40 | 26 | 14 | 1.48 |
| val: 20151106 | 60 | 39 | 25 | 14 | 1.54 |
| val: 20151109 | 65 | 41 | 25 | 16 | 1.59 |
| val: 20151110 | 61 | 39 | 24 | 15 | 1.56 |
| val: 20151112 | 42 | 33 | 26 | 7 | 1.27 |

train session 的平均 `channels / SUA units = 0.636`；validation 为 `0.674`。
因此该 bridge 确实进行了实质 channel pooling，而不是几乎全由 singleton electrode
构成的退化 identity view。以上 singleton/multi-unit 分解于 2026-07-29 直接读取
train/validation NWB unit table 的 `units/electrodes` 得到；未读取 held-out test
的 spike、behavior 或 trial 数据。

### 9.2 主结果

| View | F0 | T4 | TS4 | T4-F0 | T4-TS4 | Positive sessions | Positive seeds |
|---|---:|---:|---:|---:|---:|---:|---:|
| SUA | 0.313987 | 0.566749 | 0.314516 | +0.252761 | +0.252233 | 6/6（两组比较） | 3/3（两组比较） |
| pseudo-MUA | 0.208382 | 0.526121 | 0.160447 | +0.317739 | +0.365674 | 6/6（两组比较） | 3/3（两组比较） |

pseudo-MUA 的两个主要配对均为 **`effective`**：

- `T4-F0=+0.317739`，paired SE `0.023856`，2SE 区间
  `[+0.270027,+0.365451]`；
- `T4-TS4=+0.365674`，paired SE `0.030763`，2SE 区间
  `[+0.304148,+0.427200]`。

TS4 低于 F0 不是主要结论；它说明错误的 channel–T4 对应关系不但不能解释
T4 的收益，还可能伤害 pseudo-MUA。冻结的组级判定只要求 T4 同时通过
`T4-F0` 与 `T4-TS4`，本轮两项均通过。

### 9.3 Interaction

| Estimate | Mean | Paired SE | 2SE interval | Verdict |
|---|---:|---:|---:|---|
| Gamma | −0.064978 | 0.030744 | [−0.126466, −0.003489] | `indeterminate` |

三个逐 seed Gamma 均为负（`−0.1034/−0.0042/−0.0873`），说明观察方向一致地
偏向 pseudo-MUA 的 T4 增益更大。但预注册的 pseudo-MUA amplification 条件是
`Gamma + 2SE < −0.03`；本轮上界为 `−0.003489`，未越过 `−0.03`。因此：

- 可以说 **T4 在 SUA 与 pseudo-MUA 上都有效**；
- 可以报告 **pseudo-MUA 的 T4 增益点估计更大 `0.0650`**；
- 不可以把该点估计写成已确认的 pseudo-MUA-specific amplification；
- 更不可以说 T4 是 SUA-specific 机制。

## 10. 证据边界与剩余工作

1. **pseudo-MUA 不等于真实 threshold-crossing MUA。** 本实验只证明在同一
   DANDI session 内按 electrode 汇总 sorted SUA 后，T4 增益仍保留；外部真实
   MUA replication 仍未完成。
2. **全部结果都是 validation development evidence。** `summary.json` 明确记录
   `no_test_files_evaluated=true`；formal held-out test 不在本实验 scope 内。
3. **interaction 仍未定。** 只有在不根据当前正负方向选择 arm 的前提下，按冻结
   规则补充成组 seeds，才可提高 Gamma 的分辨率；不能把当前区间排除 0 偷换成
   越过 `±0.03` 实用容忍带。
4. **T4 是有监督标定特征，且当前 fit 预算是 50 个 trial。** B3 activity
   identity 使用 `calibration_n=30`，但 T4 使用前
   `side_feature_pool_size=50` 个 rewarded trials 的 `target_dir` 与发放率。虽不做
   backward-gradient adaptation，但方法主张不再是“identity 只来自无标签 spike
   statistics”；现有结果也不能证明更少的标签已足够。
5. **工作树版本化仍需收口。** bridge 脚本、文档、缓存和结果当前包含未跟踪资产；
   在对外引用前应提交代码/文档与小型 JSON provenance，缓存和大型 checkpoint
   则继续排除在 Git 之外。
