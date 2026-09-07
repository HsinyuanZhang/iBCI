# sub-M 跨动物三臂 V9 激活审查

状态：`SYNTHETIC_PREEXECUTION_CORE_IMPLEMENTED_BLOCKED_ON_ZERO4_TERMINALS_AND_NATIVE_M2`

## 结论

最终跨动物候选不应是刚封存的两臂 V3R2。V3R2 的 `T4-TS4` 只能检验正确描述子相对错误
row attachment 的机制敏感性，不能证明 T4 相对无描述子有绝对收益。远端正在训练的
`shared_zero4` 提供缺失的 neutral-coordinate 对照；三个 seed 全部闭合后，应构建一个新的
append-only V9 三臂包。V3R2 保留为未授权备用证据，不修改、不签署，也不作为最终 endpoint。

## 冻结科学矩阵

```text
15 sub-M CO sessions
x 2 views: SUA, deterministic pseudo-MUA
x 3 arms: shared_t4, shared_zero4, shared_ts4
x 3 seeds: 42, 43, 44
= 270 cells
```

三臂共享同一 session、query window、seed 和 activity budget：

- activity identity：chronological first 30 rewarded trials；
- T4/TS4 descriptor pool：chronological first 50 rewarded trials；
- query：strictly after rewarded trial 50；
- calibration-time backward/optimizer：0；
- decoder/encoder checkpoint selection：在 sub-M 打开前固定。

## 已实现的 synthetic/pre-execution 配对核心

截至 2026-08-05，V9 已有一个仅用于执行前验证的最小 core：
`mc_maze/subm_co_three_arm_score_only_v9.py` 及其聚焦测试
`tests/test_dandi688_subm_co_three_arm_score_only_v9.py`。它不打开真实 external `sub-M`、NWB、
checkpoint 或 normalizer，不启动 GPU，也不是 formal authorization/runtime/scorer。

该 core 已将下列科学约束做成可执行检查，而非仅文字约定：

- 同一 `(asset_id, session_id)` 的 `2 views × 3 arms × 3 seeds = 18` 个 cells 必须具有
  bitwise-identical ordered behavior targets；
- 每个 commit evidence 都必须有 `query_behavior_trace_sha256`，其绑定 query row order、exact
  `float32` shape/dtype 与 C-order target bytes。writer/re-opener/scan 会重算它，scan 和
  finalizer 还会跨同一 asset/session 直接比较 target bytes；
- helper target 明确不依赖 seed/view/arm。负测证明：即便篡改单个 target 后同时更新其 local
  NPZ SHA/bytes 和自身 trace，跨 18-cell pairing 仍失败；单独篡改 trace 也会被 scan/finalizer
  拒绝；
- direct Zero4 保持 label/rate/normalizer Poison isolation，且当前 direct constructor 只接受
  `0 < channel_count < 100`（`N=1..99`，零与 100 均拒绝）。

已用 synthetic pytest fixtures 完成 `13/13 passed` 的聚焦 suite，并通过 `py_compile` 与
`git diff --check`；其中的 CPU finalizer/R² 仅为 synthetic test helper，**不是**正式 external
metric authority。这一进展不放开 activation gate：真实 sub-M execution 仍必须等待 Native-M2
completion receipt，以及全部 terminal/source/parity/release 审查。

## 两个不可互相替代的估计量

1. `T4-zero4`：主要的 absolute deployable-system contrast。它比较 T4-conditioned source-trained
   system 与 matched-interface neutral-coordinate source-trained system 在 held-out BP-free
   calibration 下的性能。
2. `T4-TS4`：system-level attachment/content sensitivity contrast。它比较 independently trained
   correct-row 与 shuffled-row systems；它不能单独证明 T4 比没有描述子更好，也不是同权重的
   纯 input 因果效应。

SUA 为 primary，pseudo-MUA 为 key secondary。每个 contrast、每个 view 单独报告：paired grand
mean、三个 seed mean、15 个 session 的 cross-seed mean、hierarchical session-by-seed bootstrap
CI，以及 absolute T4 R2。不得跨 view rescue，也不得把两个 contrast 合并成一个平均量。

保留既有预注册门：grand paired mean `>= +0.03 R2`、3/3 seed means positive、至少 12/15
session means positive、hierarchical bootstrap lower 95% bound `>0`、absolute T4 grand/seed means
positive。结果解释按层次进行：

- `T4-zero4` 与 `T4-TS4` 均过：支持 absolute system-level value + attachment sensitivity；
- 仅 `T4-zero4` 过：支持 absolute system-level value，不宣称同权重 row-attachment mechanism；
- 仅 `T4-TS4` 过：只支持 shuffle sensitivity，不能宣称 absolute gain；
- 两者均不过：跨动物 T4 endpoint 阴性，不以 pseudo-MUA 或单 seed rescue。

三臂使用三套独立 source-trained checkpoints，因此上述效应包含 carrier intervention 对 source
training trajectory/final weights 和 held-out input 的共同作用。论文可以写 matched end-to-end
calibrated-system gain，不得写成 “decoder weights held fixed 时 descriptor alone 导致全部增益”。
V9 启动前必须生成 score-blind `source_recipe_equivalence` receipt，比较 architecture、parameter
keys/shapes、teacher family、source cohort/split、loss、view weighting、normalizers、epoch/terminal
rule、seed mapping 和所有已知差异。旧 T4/TS4 initial-state digests 当前不可用；receipt 必须将其
标为 unverified provenance limitation，不能伪造 exact-initialization equivalence。若发现 recipe
存在 carrier treatment 之外的实质差异，formal V9 不启动。

同一 `shared_t4` checkpoint 下分别输入 T4/zero4/TS4 的实验可以隔离 test-time side dependence，
但 zero4/TS4 对 T4-trained model 是 OOD input ablation。默认不把额外 180 cells 加入 formal
primary；先在已消费 sub-C 上作为 non-gating mechanism diagnostic。除非 root 另行明确授权、冻结
独立 ancillary scope，否则 external matrix 保持 270 cells。

## zero4 因果隔离

`shared_zero4` runtime branch 必须与 `load_unit_side_features` 完全分离。它只能从 channel count
构造 positive-bitwise `float32 [N,4]` zeros，并调用已经在 consumed sub-C 上验证的
`attach_standardized_zero4_to_evaluation_record`。必须逐 session/view 封存以下计数：

- `target_direction_label_reads_for_descriptor = 0`；
- `t4_trial_rate_reads_for_descriptor = 0`；
- `target_t4_rate_fit_calls = 0`；
- T4 normalizer reads for zero4 descriptor = 0；
- raw buffer 每个 byte 均为 `0x00`，排除 negative zero。

owner loader 为了确定 rewarded-trial chronology 可以携带 trial metadata，但 zero4 descriptor 和
prediction path 不得读取 target direction/rate。该区别必须在 receipt 中明确，而不是把 owner
metadata presence 错写成 descriptor label use。

## 可继承与必须废弃的实现

可继承：

- V4 的 270-cell contract、两项 comparison gates、30/50/50 budget；
- consumed-sub-C V5R2 T4/TS4 chronology parity；
- consumed-sub-C zero4 parity V1；
- V6/V7 的 safe NPZ、NPZ-level R2 recomputation、exact 270-cell reconstruction 思路；
- V3R2 的 authorization-first import boundary、fresh nonce、短时 detached Ed25519 envelope；
- score-only V2 将 forward device 与 CPU aggregate 分离并绑定物理 GPU identity 的设计。

不得继承：

- V7 caller-supplied `TrustedRoots` / `_ROOT_MINT` capability object；
- V8 blocked anchor 作为 active root；V8 只是 quarantine evidence；
- V1 两臂模板的 activity=50 语义；
- caller-provided R2、aggregate scalar、checkpoint/path/key roots；
- score-driven retry、已 committed cell 重算或混合设备 continuation。允许的恢复仅限 artifact-only
  Phase A 在 clean interruption 后补 exact missing/uncommitted keys，规则见下文。

## V9 控制面

V9 应使用 fresh isolated Python process 和 source-pinned fixed policy。运行前按顺序验证：

1. 九个 epoch-11 checkpoint 的 path/SHA/bytes/mode 与独立 terminal closure；
2. 两个 parity bundles、N=15 cohort、query-count map、normalizers、teacher、source snapshot；
3. 固定 host/Python/runtime 与一个精确的 forward device；若使用 CUDA，必须绑定 physical GPU
   UUID、PCI BDF、logical index、`CUDA_VISIBLE_DEVICES`、Torch/CUDA 版本；
4. Native-M2 completion receipt；
5. fresh output/external roots、canonical envelope、最长 15 分钟 validity、detached signature；
6. atomic single-use nonce claim；
7. 之后才允许导入 NumPy/Torch/NWB owners、打开 checkpoint/NWB 或计算 R2。

模型 forward 不应沿用旧包的无条件 CPU 限制。270 cells 共 `12,758,310` 个 model windows；
强制 CPU 会增加运行时间，却不改变 BP-free 定义。V9 的优先 runtime 是一张精确绑定的 RTX 3090，
所有 270 cells 使用同一 physical device，纯 FP32、`torch.no_grad()`、`eval()`、TF32/autocast
关闭、deterministic algorithms 开启。每个模型按固定 arm/seed 顺序加载、完成全部 session/view 后
释放，避免同时驻留九个模型。若 CUDA forward 在正式运行前未通过 consumed-sub-C CPU/GPU parity，
则整次 formal run 固定退回 CPU；不能在看到 sub-M 结果后切换设备。

CPU/GPU parity 只允许使用已消费的 sub-C fixture 和冻结模型，比较 exact input hashes、prediction
shape/order/finiteness、最大绝对/相对误差以及 variance-weighted R2。建议预冻结门为每个 arm/view
`abs(R2_gpu-R2_cpu) <= 1e-4` 且 prediction `max_abs <= 1e-4`；未通过时不以放宽容差救援 GPU。
无论 forward device 为何，每个 cell 都保存 immutable CPU float32 prediction/target artifact；reported
R2、两项 contrast 与 bootstrap 只从重新打开的 artifact 在 CPU 独立重算。因此 GPU 只减少
forward latency，不成为指标权威。

正式输出分两阶段。Phase A 只产生 270 个 immutable CPU-float32 prediction/target artifacts 和
不含 R2 的 commit receipts；禁止创建 per-cell metric 或 aggregate。只有 exact 270 commits 完整
且重新校验后，Phase B one-time CPU finalizer 才逐 cell 重算 R2、两项 contrast 和 bootstrap。
因此中间运行状态只能公开 key/hash/count，不能公开性能 scalar。

每个 NPZ 必须先写入同 filesystem 的私有 staging file，fsync、重新打开验证、chmod 0444 后，
用不覆盖既有 final path 的原子 hard-link/commit 操作发布；commit JSON 仅在 final NPZ 再次验证后
以 `O_EXCL` 写入。这样 interruption before commit 不会在正式 topology 留下半个 NPZ。orphan
final、unknown file、nonfinite、wrong shape/hash/mode、caught cell exception 或 device/source drift
都是 hard incident，不能删除后继续。

仅 clean interruption between committed cells 可申请一次 fresh recovery authorization。它必须
绑定原 run manifest、contract/source/checkpoint/normalizer/device、original nonce claim、exact
committed key/SHA map 和 exact missing set；只能计算 missing uncommitted keys，不能重新 forward
已 committed cell，且 recovery 前/中不得存在或计算 R2/aggregate。再次中断没有自动循环，必须
重新做一次 score-blind human review。

每个 session/view 的 neural/behavior/query/activity-first30 base 只构造一次；T4 dataset 可跨模型
seed 复用，TS4 按 seed 构造，zero4 直接构造一次并跨三个 zero4 模型复用。模型常驻/重载 schedule
必须在无分数 RAM/device audit 后冻结；不能在 external 运行中动态切换。任何开发/源数据分数都
不能改变 gate。

## 激活条件

在以下全部成立前，V9 只能保持设计状态：

- shared-zero4 seeds42/43/44 的 epoch11 checkpoints 与 terminal closures 完整、hash-verified；
- 九槽 `source_recipe_equivalence` receipt 已封存，所有不可验证项和 carrier 外差异均明确披露；
- 远端训练没有失败、重试混用或 checkpoint selection；
- Native-M2 已完成并有 completion receipt；
- V9 source、policy、writer、aggregate 与 adversarial tests 经过 fresh review；
- 若选择 GPU forward，consumed-sub-C CPU/GPU parity receipt 已封存并绑定 exact physical device；
- artifact-only writer、atomic staging/commit、missing-only recovery 与 one-time finalizer 的
  synthetic/adversarial tests 全部通过；
- external sub-M output root 仍不存在，nonce 为零，R2 为零。

满足条件后只能生成一个新的 append-only V9 prelaunch；不得把 V8 quarantine 或两臂 V3R2
原地改成 active scorer。

远端 score-blind transfer manifest 只作为传输 inventory：它记录 terminal/provenance 文件的
path、bytes、SHA 和 mode，但不验证 closure 内部语义，也不构成 V9 checkpoint authority。文件
传回后，root 必须重新计算 live bytes/SHA/mode，解析每个 complete/closure 的 checkpoint
binding，并与九槽合同精确匹配；只有这个本地独立复核结果可以进入 V9 policy。
