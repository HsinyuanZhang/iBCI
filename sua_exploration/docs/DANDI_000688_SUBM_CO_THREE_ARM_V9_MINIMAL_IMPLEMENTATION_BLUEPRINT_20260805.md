# external sub-M 三臂 V9：最小、score-blind 实现蓝图

**状态：** `SYNTHETIC_PREEXECUTION_CORE_IMPLEMENTED_NOT_AN_AUTHORIZATION_NOT_A_SCORER`

**本文件的作用：** 为 Native-M2 完成、三个 `shared_zero4` terminal closure 已独立核验之后的
V9 实现提供一个最小蓝图。它不创建 capability、nonce、签名、prelaunch、NWB/checkpoint 打开、
model forward 或 R²。它也不把 V3R2、V7 或 V8 原地升级为 V9。

本蓝图只覆盖冻结的 external `sub-M` endpoint：

```text
15 sessions × 2 views × 3 arms × 3 seeds = 270 primary cells

views = {sua, deterministic_pseudo_mua}
arms  = {shared_t4, shared_zero4, shared_ts4}
seeds = {42, 43, 44}
```

它继承的 target-session chronology 是：activity identity 使用 chronological first 30 rewarded
trials；T4/TS4 descriptor 使用 chronological first 50 rewarded trials；所有 query windows
严格在 rewarded trial 50 之后。目标 session calibration 无 optimizer、backward、weight update
或 normalizer fit；这是一项 **supervised, calibration-time backprop-free** endpoint，不是
label-free learning，也不是 offline source training backprop-free。

## 当前 pre-execution 实现状态（2026-08-05）

最小 synthetic core 已落在
`sua_exploration/mc_maze/subm_co_three_arm_score_only_v9.py`，其聚焦测试位于
`sua_exploration/tests/test_dandi688_subm_co_three_arm_score_only_v9.py`。这不是 formal runner：它
不打开 external `sub-M` NWB、checkpoint、normalizer 或真实 prediction，不启动 GPU，也不拥有
authorization、source closure、V5 bridge 或正式 TorchMetrics scorer。它只用 pytest 临时目录中的
synthetic arrays 验证下列正式运行前必须保留的科学/ledger 语义：

* exact `15 × 2 × 3 × 3 = 270` cell contract，以及 chronological first-30 / first-50 / post-50
  declaration；
* artifact-only Phase A、exact cardinality、no-overwrite、missing-only continuation 和 full-matrix
  synthetic CPU finalizer；这里的 synthetic R² 仅是测试辅助，**不是** external endpoint 的权威
  TorchMetrics 数值；
* 对每一个 `(asset_id, session_id)`，所有 18 个
  `2 views × 3 arms × 3 seeds` cells 的 ordered behavior targets 必须 bitwise identical；
* 每份 commit evidence 强制含 `query_behavior_trace_sha256`。它绑定 ordered query row、精确
  `float32` shape/dtype 和 C-order target bytes；writer/re-opener/scan 都重算它，scan/finalizer
  还跨同一 asset/session 直接比较 target bytes；
* 测试 helper 不得使 target 依赖 seed/view/arm；对单一 target 的篡改即使同时更新该 cell 的
  NPZ SHA/bytes 与自身 trace，仍因跨 18-cell 配对失败；单独篡改 trace 也失败；
* direct Zero4 只取 channel count、目标标签/rate/normalizer Poison-isolated，并要求
  `0 < channel_count < 100`（即 `N=1..99`；`0` 和 `100` 均拒绝）。

已执行的聚焦验证是：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/xinyuan/miniconda3/envs/spint/bin/python -m pytest -q \
  sua_exploration/tests/test_dandi688_subm_co_three_arm_score_only_v9.py
# 13 passed
```

同时 `py_compile`、`git diff --check` 与两个新增文件的 no-index whitespace check 均通过。
这只证明 pre-execution synthetic contract；**真实 external sub-M execution 仍必须等待
Native-M2 completion receipt**，再走本蓝图其余 terminal/source/parity/release 审查。

---

## 1. 先做出的实现判断

1. **V9 必须是新包。** V3R2 是 180-cell、CPU-only、`shared_t4/shared_ts4` 两臂包；它保留为
   不可变备用证据。V4/V6 的 scientific/ledger 思路可以参考，V7 的 caller-supplied
   `TrustedRoots` / `_ROOT_MINT` 是 P0，绝不继承；V8 是正确的 blocked quarantine，不是 active
   scorer。
2. **270 cells 是 primary deployable-system matrix。** 它回答的是：在相同 source recipe、
   width、target trial boundary 和 frozen terminal selection 下，带有相应离线 source-trained
   weights 的 T4/zero4/TS4 system 如何在新动物 held-out sessions 上工作。
3. **不能把 270 cells 写成“固定 decoder 权重下纯 T4 descriptor 的因果效应”。** 三个 arm
   使用独立 source-trained checkpoints。`T4-zero4` 和 `T4-TS4` 同时改变 held-out descriptor
   与 source-training trajectory/terminal weights；它们是匹配的 *system-level* contrasts。
4. **GPU 可以只加速 forward，不可以成为 metric authority。** 如果 pre-formal consumed-sub-C
   CPU/GPU parity 过预冻结门，全部 270 cells 固定在同一张绑定身份的 RTX 3090 上做纯 FP32
   forward；prediction/target 立即转为 CPU float32 immutable artifacts；R² 和 aggregate 只从
   重新打开的 CPU artifacts 得出。parity 未过时整次 external run 预先固定回 CPU，不能见到
   sub-M 结果后切换设备。
5. **推荐受限的 artifact-only continuation，而不是“retry”。** 270 个 cells 的单次长运行有
   真实的中断风险。允许的恢复只能续写 *尚未 committed* 的 cells，绝不能重跑已 committed
   cell、不能看/算任何 partial R²、不能换 contract/source/device/checkpoint，也不能借异常做
   result-driven rerun。完整规则见第 9 节。

---

## 2. 冻结科学对象、两项结论与术语纪律

| 项目 | 冻结内容 |
| --- | --- |
| Primary absolute contrast | `R2(shared_t4) - R2(shared_zero4)` |
| Mechanism contrast | `R2(shared_t4) - R2(shared_ts4)` |
| Primary view | SUA |
| Key secondary view | deterministic pseudo-MUA |
| Activity support | rewarded trials `[0, 30)`，chronological |
| T4/TS4 pool | rewarded trials `[0, 50)`，chronological |
| Query | owner valid 50-bin windows strictly after rewarded trial 50 |
| Matrix | exact common N=15 cohort × `{sua,pseudo_mua}` × three arms × seeds 42/43/44 |
| Primary gates per comparison/view | paired grand mean `>= +0.03 R²`；3/3 positive seed means；至少 12/15 positive cross-seed session means；hierarchical bootstrap lower 95% `>0`；absolute T4 grand/three seed means `>0` |
| Forbidden rescue | 不合并两个 contrast；不合并两 view；不删 session/cell；不以单 seed 或 pseudo-MUA 挽救 SUA |

解释必须按 comparison 和 view 分开：

* `T4-zero4` 过而 `T4-TS4` 不过：支持 **absolute system-level descriptor value**；不写
  row-attachment mechanism。
* `T4-TS4` 过而 `T4-zero4` 不过：只支持 correct-vs-shuffled attachment sensitivity；不写
  absolute T4 gain。
* 二者均过：支持 absolute system-level value 和 attachment mechanism；仍不等价于固定权重的
  pure-descriptor causal effect。
* 二者均不过：external cross-animal endpoint 阴性；不能靠 pseudo-MUA、单 seed 或事后改门
  挽救。

`pseudo_mua` 是由同一 SUA recording 作 deterministic electrode pooling 得到的受控 view，
不是独立采集的 threshold-crossing MUA；即便它通过也不能删掉 “pseudo” 或写成独立 modality /
独立实验室泛化。N=15 也是 DANDI 000688 内的 cross-animal result，不是独立 laboratory
replication。

---

## 3. 科学归因 P0：三套独立 source-trained weights

### 3.1 问题是什么

`shared_t4`、`shared_zero4`、`shared_ts4` 不是同一个 frozen decoder 搭配三个 test-time side
input。每一个 arm 有自己的 source-trained terminal checkpoint。因此原始三臂 contrast 同时包含：

```text
held-out side carrier
       +
offline training exposure / optimization trajectory / final weights
```

即使 recipe、architecture、source cohort、teacher、12 epochs、terminal selection、normalizer
policy、paired objective、side width 和 nominal seed 都被匹配，独立 optimization 仍然使它成为
一个 **whole calibrated decoder system** 的对照。这不是缺陷：部署时确实选择的是整个离线训练后
的 system。但它是论文归因 P0：

> 可写：T4-conditioned, source-trained frozen system outperformed a matched direct-zero / shuffled
> system under BP-free held-out calibration.

> 不可写：with decoder weights held fixed, the T4 descriptor alone caused the entire observed gain.

V9 activation 前必须做一个 score-blind `source_recipe_equivalence` receipt，逐 arm/seed 比较：

* architecture、parameter key/shape、side dimension、teacher/initialization family；
* source-session cohort、train/validation split、view weighting、loss definition；
* source normalizers、data preprocessing、batch/order policy；
* epoch count、terminal `epoch_011` rule、seed mapping；
* T4/TS4/zero4 carrier 是唯一预先声明的 treatment，及每个意外差异。

这份 receipt 不打开 sub-M，也不读 R²。若发现 zero4 source recipe 不匹配，primary external V9
不能启动；不能在结果后用文字把差异淡化。

### 3.2 是否应把同权重 input ablation 加入 formal matrix？

推荐是：**保留 270-cell matrix 作为唯一 primary endpoint；不静默扩成 450 cells。**

若正式 scope、计算预算和 root 明确授权，可另建一个预注册、non-gating 的 ancillary
`weight_matched_input_ablation` block：

| diagnostic cell | checkpoint weights | held-out side input | 它检验什么 |
| --- | --- | --- | --- |
| `t4weight_t4input` | `shared_t4`, seed s | correct T4 | primary T4 cell 的同一 prediction，可引用而不重跑 |
| `t4weight_zero4input` | **same** `shared_t4`, seed s | direct zero4 | T4-trained decoder 是否依赖这个 side coordinate |
| `t4weight_ts4input` | **same** `shared_t4`, seed s | seed-specific TS4 | T4-trained decoder 对 row attachment 是否敏感 |

两个新增输入 condition 是 `15 × 2 × 3 = 90` cells each；二者都做时为 **180 extra forwards**，
总数从 270 变为 450。按照已冻结的 N=15 query-count 总和，primary 270 matrix 约为
`12,758,310` model windows；两项 diagnostic 都做会再增加约 `8,505,540`，即约 `1.67×` 的
forward 与 prediction artifact 量。它不需要新训练，但会扩大 formal endpoint scope。

这个 diagnostic 的正确解释同样有限：把 zero4/TS4 输入送给仅在 correct-T4 distribution 上
source-trained 的 `shared_t4` checkpoint 是 **test-time OOD input ablation**。若性能下降，它支持
该 frozen model 使用/依赖 T4 coordinate；若不下降，可能表示冗余、鲁棒性或当前 decoder 不读取
该 coordinate。它不替代 deployable-system contrast，也不修复 independent-training attribution。

因此推荐顺序是：

1. 270 cells 保持 primary、gated、唯一可用于 deployment claim 的矩阵；
2. 若 formal scope 不能扩张，先在已消费的 sub-C fixture 做同权重 diagnostic，作为实现/机制
   evidence，**不**外推为 cross-animal conclusion；
3. 只有 root 明确允许额外 formal data products 时，才把 180 个 cells 作为独立、非主门的
   ancillary transaction，名字中必须含 `OOD_input_ablation`，且任何结果不得改变 270-cell
   gates、候选选择或 terminology。

这比把 450 cells 偷偷混进 primary aggregate 更诚实，也更节省 formal-test scope。

---

## 4. 继承边界：哪些函数可以复用，哪些只能参考

| 来源 | 可以复用/移植的最小部分 | V9 使用条件 | 不可复用的部分 |
| --- | --- | --- | --- |
| V5/V5R2 adapter parity | `bridge_owner_chronology_for_c1_builder` 与 its start/stop-only chronology semantics | 仅 authorization 后；V9 source-pins it；V9 trace 再次证明 first30/first50/post50 不变 | 任何历史 banner 或旧 package authority 不能作为 V9 authority |
| V3R2 executor | `_compact_bridge_trace` 的字段语义；`_build_view_base`/`_dataset_for_cell` 的 **30/50/50 logic** | 建议在 V9 写小的同义函数，并用 synthetic/consumed-sub-C parity 测试；避免把 180-cell claim/CPU policy 隐式带进来 | 直接把 V3R2 executor 当 V9 runtime；V3R2 的 two-arm contract、CPU-only auth、180-cell receipt |
| V1 score runtime | frozen model loader、normalizer loading、paired-view alignment、batch unpacking、runtime safety fence 的 post-auth 监视概念 | 只能 post-auth；全部相关 source 必须进入 V9 source snapshot | `v1._score_one_session_dataset`：它在 forward 时创建/更新/compute R²，违反 V9 score-blind artifact-first design |
| T4/TS4 side loader | `load_unit_side_features`、full realized TS4 permutation semantics | 只可走 T4/TS4 branch；T4 descriptor pool 固定 50、TS4 seed 固定 | zero4 branch 调用 loader，或先做 T4 再 mask |
| direct-zero4 path | `attach_standardized_zero4_to_evaluation_record`、`standardized_zero4`、`require_standardized_zero4` | 仅凭 signal-view channel count 构造 `[N,4]` positive float32 zero；post-auth source pin | raw T4、target label/rate、side normalizer 输入或 pseudo-MUA source-unit count |
| V2 artifact utilities | `_write_immutable_npz_exclusive`、`_write_immutable_json_exclusive`、`_load_prediction_target_npz`、`recompute_torchmetrics_r2_cpu`；GPU UUID/PCI normalisation logic | 只在 post-auth runtime/finalizer；V9 test must cover exact current behavior | `SealedOutputWriterV2` / `aggregate_sealed_v2`：hard-code 180/two arms and per-cell R²；不能直接实例化 |
| V7 numerical statistics | `_bootstrap` / `_reconstruct` 的 hierarchy、PCG64/quantile rule，或等价的小纯统计函数 | 移植后以 synthetic matrix exact-match test 锁定；明确 comparison/view iteration order | `TrustedRoots`、`VerifiedPolicy`、`VerifiedGrant`、`_ROOT_MINT`、generic root/ledger capability classes、resume API |
| V8 | 隔离 process、no-caller-root 的 threat-model lesson | V9 production CLI 必须采用同样的 no mutable in-process root principle | V8 blocked anchor 或其 stubs；它没有 active execution branch |

一个容易漏掉的接口风险是 NPZ member naming：V2 utility 写的是
`predictions`/`targets`，V7 safe parser 的 member names 是 `prediction`/`target`。V9 必须选定其中
一个格式并对 writer/re-opener 一致测试；为了最少改动，推荐用 V2 的 plural format，同时让 V9
re-opener 明确验证它，而不是混用两个 parser。

---

## 5. 必须新写的最小模块

建议只新增以下 V9 文件；不要制造 V7 式可组合 control-plane framework。

| 建议文件 | 责任 | import surface |
| --- | --- | --- |
| `mc_maze/subm_co_three_arm_score_only_v9_control.py` | fixed release anchor、canonical detached signature、source/runtime/device binding、fresh nonce claim、pre-auth audit | module top-level 仅 stdlib + `cryptography`；无 NumPy/Torch/NWB/model/data owner |
| `mc_maze/subm_co_three_arm_score_only_v9_runtime.py` | post-claim owners、30/50/50 base、T4/TS4/zero4 dataset construction、forward-only prediction collector、artifact-only writer/finalizer | 只能由 valid V9 grant 后 import；局部 import NumPy/Torch/data owners |
| `scripts/write_dandi688_subm_co_three_arm_prelaunch_v9.py` | score-blind static policy/prelaunch/closure/source snapshot writer | 不打开 external data/checkpoint，不生成授权 |
| `scripts/prepare_dandi688_subm_co_three_arm_authorization_v9.py` | 将 Native-M2 completion receipt、fresh reviewed V9 policy、exact output/external root 绑定进 **unsigned** envelope；绝不持有/接受 private key | stdlib + control module；不生成签名 |
| `scripts/run_dandi688_subm_co_three_arm_score_only_v9.py` | fresh `python -I` entrypoint；`status` / `score` / narrowly defined `recover` / `finalize` mode | 只接受 authorization 与 detached signature paths；不接受 repo/policy/root/key/output/data/device CLI override |
| `scripts/run_dandi688_subm_co_three_arm_gpu_cpu_parity_v9.py` | consumed-sub-C only CPU/GPU forward parity receipt；不是 formal scorer | explicit separate pre-formal evidence package |
| `tests/test_dandi688_subm_co_three_arm_score_only_v9.py` | synthetic control, writer, aggregate, failure, import-surface tests | no NWB/checkpoint/GPU in ordinary tests |

V9 control 只需要一个 immutable reviewed release policy、一个 source-pinned public-key fingerprint、
一个 detached signed authorization 和一个 derived nonce-claim namespace。它不需要：

* caller-supplied Python root object、`install_trusted_roots()`、private mint symbol、generic
  policy class hierarchy；
* CLI-supplied root/key/policy/device/output/external-NWB locations；
* multiple nested key hierarchies、generic resume framework、or runtime-pluggable filesystem root；
* any way to mutate V3R2/V7/V8 or reuse their authorization objects.

可信边界仍是一个 fresh, isolated `python -I` process 加上 reviewed source、pinned public key 和
external signer。任何拥有同一 host 任意代码执行/修改 trusted source 的攻击者都超出纯 Python
protocol 的可防御范围；V9 应写清这个现实假设，而不要虚构 mutable object 能充当 trust root。

---

## 6. authorization-first、单设备执行的最小顺序

```text
fresh `python -I` V9 runner
        │
        ├─ control-only import check
        │    (no numpy / torch / NWB owner / checkpoint / normalizer)
        │
        ├─ verify fixed release policy + source pins + external public-key pin
        ├─ verify canonical detached authorization, expiry <= 15 min,
        │    Native-M2 completion binding, exact cohort/9 closures/runtime/output/external root
        ├─ verify selected physical 3090 UUID + PCI BDF + host/CVD policy without data/model access
        ├─ atomic nonce claim
        │
        └─ only now import V9 runtime
             ├─ re-verify exact CUDA logical-to-physical mapping before checkpoint/NWB access
             ├─ validate all nine checkpoint pins / terminal closure evidence
             ├─ construct fixed 270-cell plan
             ├─ forward-only artifacts
             └─ after all artifacts: CPU reopen/R²/aggregate
```

必要的不变量：

* control module 在 authorization 前 `sys.modules` 不得已有 NumPy、Torch、data adapter、V5
  runtime、model loader 或 `pynwb`；
* `CUDA_VISIBLE_DEVICES`、host、logical Torch device、physical GPU UUID、PCI BDF、Torch version、
  CUDA version、pure-FP32 flags、batch size、query order 都是 policy-bound；
* 若 GPU path 选择，则 `all_270_cells_same_device = true`。所有 three arms、two views、three
  seeds 都在同一 physical device 上；不能 GPU/CPU 或 two GPUs 混合；
* authorization 中的 output root 与 external-NWB root 是签名内容，不是 runner CLI input；它们
  必须分别位于 reviewed output parent 与 exact canonical external root；
* pre-auth rejection 不创建 output root、不 claim nonce、不打开 checkpoint/NWB/normalizer，也不
  import scoring runtime；
* nonce claim 之后才允许 Torch/CUDA introspection、checkpoint hash check、NWB access。GPU
  mapping mismatch 是 post-claim incident，不能通过换卡后继续同一 grant。

V3R2 `auth_v3` 将 `CUDA_VISIBLE_DEVICES` 固定为空并要求 CPU policy，因此不能直接用于这里。
V9 只借鉴它的 canonical-envelope / detached-signature / fresh-nonce / import-boundary 思想，
而不是导入或扩大其 CPU capability。

---

## 7. post-claim input construction与 zero4 隔离

### 7.1 每个 session/view 的 common base

对每个 `(asset_id, view)`：

1. 使用 source-pinned V5 bridge 获取 rewarded chronology；只允许 start/stop cast；记录 bridge
   trace；
2. owner loader 维持 first-50 query exclusion；独立检查 `valid_starts` 与 frozen N=15
   query-count map 一致；
3. 从 bridge 后 chronology 精确选择 `list(range(30))`，重建 activity calibration；不得复用
   owner 默认 50-trial calibration tensor；
4. 构造 base record 一次，记录 neural/behavior/calibration/valid-starts 的 shape/digest 和
   query count；
5. 在相同 time/behavior/query boundary 上构造两 view，运行 existing paired-view alignment
   check；它只验证允许对齐的 axes，不要求 SUA 与 pooled channel count 相同。

T4 dataset 可在三个 checkpoint seeds 间复用；TS4 dataset 按 seed 构造并封存完整 realized
permutation vector；zero4 dataset 可在三个 zero4 checkpoints 间复用。所有 reuse 都是同一个
session/view 的内存 object 或由 immutable input trace 验证的等价 object，不能跨 asset/view
偷偷复用。

V9 activation draft 中 “base 一次” 与 “每个 model 完成全部 sessions 后释放” 之间存在一个
**工程 P1**：前者需要保留 30 个 bases，后者最小化 checkpoint reopen。V9 开工前应先做无分数的
CPU shape/RAM audit 并冻结其中一个 schedule：

* 若 30 bases 落在预定 RAM cap 内：cache all bases，按 `(arm, seed)` 载入一个 model、完成所有
  session/view 后释放；
* 否则：按 `(session, view)` 构造一次 base，在固定 `(arm, seed)` 顺序内加载/释放 model。

两种 schedule 只要 input/prediction parity 通过就没有科学区别；选项必须 prelaunch 中固定，
不能运行中按表现或内存压力临时切换。

### 7.2 zero4 descriptor 的强隔离

`shared_zero4` 不能调用 `load_unit_side_features`。其 V9 helper 签名应只接收 signal-view
channel count / record 的 `n_units` 与 `neural` shape，立即调用
`attach_standardized_zero4_to_evaluation_record` 或等价 direct constructor：

```text
float32 [N, 4] with every IEEE bit = positive zero
```

当前 synthetic direct constructor 还显式拒绝空/异常宽度：`0 < N < 100`，即只接受 `N=1..99`。
这是已测试的 API/descriptor guard，不是关于 future external `sub-M` channel distribution 的经验
结论；正式 runtime 必须在 source review 中保留同一边界，或在打开任何 external data 前以新的
reviewed policy 明确替代它。

特别是 pseudo-MUA 的 `N` 必须是 pooled channel axis，不是构成 pool 的 sorted source-unit count。
每个 `(asset_id, view)` 的 immutable zero4 trace 必须包含：

| zero4 descriptor counter / proof | required value |
| --- | --- |
| `target_direction_label_reads_for_descriptor` | `0` |
| `t4_trial_rate_reads_for_descriptor` | `0` |
| `target_t4_rate_fit_calls` | `0` |
| `source_t4_normalizer_value_reads` | `0` |
| `source_t4_normalizer_arithmetic_performed` | `false` |
| `raw_t4_constructed` | `false` |
| `side_feature_loader_calls_for_zero4` | `0` |
| `side_normalizer_passed_to_zero4_constructor` | `false` |
| `bitwise_positive_float32_zero` | `true` |
| `side_shape` | `[signal_view_channel_count, 4]` |

owner loader 为 rewarded chronology、neural activity、behavior target 或 query construction 携带
trial metadata 不违反该 descriptor-only counter；但 receipt 必须另列
`owner_metadata_may_contain_target_direction = true`，不能把 metadata presence 误写成 zero4
descriptor label use。zero4 branch 的 function signature 和 Poison-object unit test 是比自报
counter 更强的证据。

为使 “zero4 never reads T4 normalizer” 严谨，base loader 应只接收 behavior normalization；
view-local side mean/std 在 T4/TS4 dataset factory 内才读取/传入。若 global runtime 出于
checkpoint validation 已打开 normalizer file，receipt 必须区分 file existence/open 与
**zero4 descriptor value read/arithmetic**，不能用模糊的全局 counter 伪造零。

---

## 8. GPU forward 与 CPU-artifact R² 的 parity gate

GPU 不是自动许可。正式 external run 前，单独在已经消费的 sub-C fixture 对所有
`{shared_t4, shared_zero4, shared_ts4} × {sua,pseudo_mua}` 执行同一 V9 forward collector 的
CPU/GPU parity；使用固定 terminal checkpoints、全 post-50 windows、相同 input digest、相同
batch order。这个 harness 是 pre-formal evidence，不能接受 sub-M path，也不能成为 external
scorer。

固定 GPU policy：

```text
model.eval()
torch.no_grad()
float32 inputs/weights/outputs
autocast disabled
TF32 disabled for matmul and cuDNN
deterministic algorithms enabled
cudnn.benchmark disabled
no optimizer / backward / gradient accumulation
```

每个 arm/view 的 receipt 至少锁定：input hashes、prediction shape/order/finiteness、CPU/GPU
prediction `max_abs`、相对差摘要、CPU and GPU artifact-recomputed variance-weighted R²、physical
device UUID/PCI BDF、Torch/CUDA versions。通过门预冻结为：

```text
max_abs(pred_gpu - pred_cpu) <= 1e-4
abs(R2_gpu_artifact - R2_cpu_artifact) <= 1e-4
```

其中两个 R² 都从保存后的 CPU float32 arrays 用相同 frozen CPU metric 计算，而不是 GPU-resident
metric scalar。若任一 arm/view 未通过，policy 在打开 external sub-M 前锁定 `device=cpu`；不允许
放宽阈值、挑 batch、挑 view、换 GPU 或查看 sub-M 后才回退。

正式 GPU collector 必须是新函数，例如 `forward_dataset_to_cpu_float32_arrays_v9`：只收集
prediction 与 target，且严格按 dataset order concatenate。**它不能调用**
`_score_one_session_dataset`，因为后者会在 forward phase 提前 `R2Score.update/compute`。

---

## 9. 270-cell artifact writer、finalizer 与 deterministic continuation

### 9.1 为什么 V9 不应直接使用旧 writer

`SealedOutputWriterV2` 和 V7 cell ledger 都会在每个 cell 写完时算/写 R²。那适合一次性
180-cell score，**不适合 result-blind continuation**：若 270-cell run 在第 173 cell 中断，
operator 已能从 partial metric JSON 选择性地决定是否继续。

V9 因而采用两阶段、但仍然很小的 writer：

```text
Phase A: artifact-only forward/commit
  270 × {prediction float32 [Q,2], target float32 [Q,2], no R²}
       ↓ exact 270 commitments
Phase B: one CPU finalizer
  reopen/re-hash/validate all 270 artifacts
       ↓ recompute every R² on CPU
       ↓ construct both contrasts / both views / bootstrap
       ↓ one immutable aggregate
```

### 9.2 V9 writer 的 exact content

新 `ArtifactOnlyWriterV9` 必须在 fresh output root 写 immutable `run_manifest`，其内容包括：

* full ordered cohort、views、arms、seeds、expected 270 keys；
* `30/50/post50` policy、exact query-count map、source snapshot、nine checkpoint pins、two parity
  receipt pins、normalizer pins、terminal closure pins；
* forward device binding、precision flags、frozen batch/schedule policy；
* original authorization hash/nonce claim pin；
* `r2_computations_so_far = 0`、`partial_metrics_visible = false`。

每个 committed cell 只写：

```text
artifacts/<asset>/<view>/<arm>/seed_<s>/predictions_targets.npz
commits/<asset>/<view>/<arm>/seed_<s>.json
```

NPZ 需含 exact C-contiguous finite CPU `float32` `predictions` 和 `targets` arrays，shape
`[frozen_query_count_for_asset, 2]`。commit JSON 只绑定：cell key、NPZ SHA/bytes/mode/shape、
checkpoint SHA、device binding、base/side/TS4-permutation or zero4-trace digest、query count、
contract SHA 和 forward status；它不得含 R²、delta、gate 或任何可读 performance scalar。

此外，commit evidence 必须写入 `query_behavior_trace_sha256`。它包含有 schema 的 ordered query-row
convention、exact `float32` shape/dtype 与 C-order behavior target bytes 的 SHA-256 binding。对于同一
`(asset_id, session_id)`，两个 views、三个 arms、三个 seeds 的全部 18 个 cells 必须拥有同一 trace
且实际 target bytes bitwise identical。writer 在新 cell commit 前检查这个约束；re-opener/scan 在
逐 artifact 重新计算 trace 后，再直接比较 peer target bytes；finalizer 先跑 scan，因此不能绕过。
这条配对规则防止某个 arm/seed 静默换掉 behavior/query sequence 而把非同一 held-out target 当成
paired delta。

commit 顺序固定为 prelaunch 的 canonical key order；同 key 已存在则 fail closed。每次写入都
`O_EXCL`、fsync、chmod `0444`；commit 只在 re-open/validate NPZ 成功后写。所有 observed files
必须属于 manifest 的允许集合；symlink、unknown file、wrong mode、nonfinite array、wrong shape、
重复 key、orphan NPZ/commit 都是 incident。

全 270 个 artifact+commit 存在且被 re-open 验证前：

* **不创建 per-cell metric JSON；**
* **不调用 R²；**
* **不创建 aggregate；**
* status 只能报告 committed/missing count、cell identities、hashes 和 audit counters，不能报告
  output value。

### 9.3 one-time CPU finalizer

finalizer 只可在 exact 270 commits、无 incident 后启动。它逐 cell：

1. 按 manifest re-open NPZ，验证 immutable mode/hash/shape/finiteness/query count；
2. 调用 V2 `recompute_torchmetrics_r2_cpu`（或 byte-identical V9 local equivalent）得到权威
   `torchmetrics==1.5.1`, `R2Score(multioutput="variance_weighted")` CPU float32 value；
3. 用可追溯 metric records 形成 `[seed, session] = [3,15]` grids；
4. 对 `T4-zero4` 与 `T4-TS4`、每个 view 分别执行 V7-equivalent hierarchical bootstrap：
   100,000 draws、`numpy.Generator(numpy.PCG64(68820260805))`、session-then-seed resampling、
   linear percentile。comparison/view iteration order 也进入 policy，禁止依赖 dict order；
5. 在单一 immutable `aggregate/endpoint_aggregate.json` 中写所有 per-cell R²、两个 independent
   comparison blocks、gates、no-cross-view-rescue flag、absolute T4 summary 和 provenance。

aggregate 的 authority 是 reopened artifacts，不是 GPU scalar、streaming metric、caller value 或
previous partial JSON。任何 scalar mismatch 或 one missing cell 都阻止 aggregate；aggregate 已存在
则不可重写。

### 9.4 推荐的 narrow continuation policy

把 V9 activation draft 中笼统的 “no partial resume/retry” 改写为下面的精确规则：

> **No score-driven retry, no cell recomputation, no mixed-device continuation.** A narrowly
> defined artifact-only continuation is allowed only after an interruption, only for missing
> uncommitted keys, and only under a fresh recovery authorization that pins the original run
> manifest, contract/source/checkpoint/normalizer/device bindings, original nonce claim, and exact
> committed-key/commit-SHA map. No R² or aggregate may exist before recovery or during recovery.

实际判断表：

| 事件 | V9 处置 |
| --- | --- |
| pre-claim signature/source/device rejection | 无 output、无 nonce；可修复并重新走 fresh authorization，不涉及 formal prediction |
| clean interruption between fully committed cells（power/queue preemption，且没有 caught failure receipt） | 可申请**一个**独立 recovery authorization；它只允许一次 exact continuation invocation。blind verifier 仅 rehash/shape-check committed artifacts，运行缺失 keys；若再次中断，不存在自动循环，必须由人做一次新的 score-blind review 后才可另签一份 recovery authorization |
| checkpoint/NWB/integrity failure、nonfinite output、cell exception、device mismatch、source drift | 写 immutable incident receipt；**终止**；不允许 retry/continue 同一 output root 或重算失败 cell |
| orphan/partial NPZ、orphan commit、unknown file | incident；终止，不能通过删除/覆盖清理来“恢复” |
| 270 keys complete 后 | forward phase 关闭；仅 one-time CPU finalizer 可开 R²/aggregate |

recovery authorization 不是新实验选择：它不得改变 model、epoch、cohort、data root、source pins、
normalizers、batch size、order、GPU UUID/BDF、CVD mapping 或 missing-key set 之外的任何内容。已
committed cell 永远不能重算，即使 operator 不喜欢其 output。没有 `--retry` / `--resume-any` 循环；
每一份 recovery authorization 只消费一次 continuation opportunity。为防止人为提前得分，output root 的
访问控制和 runner stdout/log 也必须保持 hash/count-only；原始 prediction artifacts 对一个有任意
文件读权限的人理论上仍可被离线评分，纯 Python 不能消除此信任假设。这个 operational boundary
应在 release review 中明确，而不要声称 cryptographic score blindness。

这个受限 continuation 的价值是降低 270/450 long-run 的工程失败风险；它不增加 data-dependent
model selection，也不改变统计样本。它比 V7 的 generic resume/grant framework 小得多，但必须用
synthetic adversarial tests 证明其只补缺失 keys。

---

## 10. 预期测试清单

下列 tests 都要在 external sub-M opening 前通过。除明确标为 consumed-sub-C parity 的一项外，
全部使用 static/synthetic fixtures；普通 test 不打开 NWB/checkpoint、不跑 GPU、不算 formal R²。

**实现状态不能与完整 release checklist 混淆：** 截至 2026-08-05，已实现并复跑的是其中的
minimal synthetic core，聚焦 suite 为 **13/13 passed**；它覆盖 exact-270 topology、30/50/post50
declaration、artifact-only/no-partial-metric、missing-only/no-overwrite、paired target/trace、target/trace
tamper rejection、Zero4 Poison isolation/`N=1..99` boundary，以及 synthetic finalizer 的 two-contrast
shape。它不替代以下 formal authorization/source/parity/runtime tests，也没有消耗 external `sub-M`
scope；真实执行仍等待 Native-M2 completion receipt。

### 10.1 static control / authorization tests

* V9 import AST/runtime test：pre-auth top-level 无 NumPy、Torch、NWB/data owner、model loader；
  no optimizer/backward/fit/training loader call；
* `status` / missing authorization / invalid signature / expired envelope / noncanonical JSON / wrong
  source pin / wrong external-output root 在 nonce/data/model 前 fail closed；
* runner 无 `--repo-root`、`--policy`、`--anchor`、`--key`、`--device`、`--output-root`、
  `--external-nwb-root` bypass；
* caller cannot pass or construct roots/grants/policy objects；static source has no V7
  `TrustedRoots` / `_ROOT_MINT` / `VerifiedGrant`-like capability path；
* duplicate nonce、nonce spelling variant、expired recovery authorization、wrong original-manifest hash
  都 fail closed；
* dry/status run reports zero checkpoint/NWB/normalizer/Torch runtime imports, zero forward, zero R²,
  zero output root, zero nonce claim.

### 10.2 scientific boundary / input tests

* exact 270 cell-key set and canonical order; no arm/view/seed/session omission or duplicate;
* same `(asset_id, session_id)` 的 18 个 `view × arm × seed` cells 必须使用 bitwise-identical
  ordered behavior targets；每个 commit 的 `query_behavior_trace_sha256` 必须由其 artifact target
  重算得到，且跨 peer 相同；
* first30 activity indices, first50 T4 pool, post50 valid-starts and pinned N=15 count map all exact;
* V5 bridge trace proves no chronology mutation and only approved start/stop conversion;
* T4 dataset is seed-reused only where legal; every TS4 vector equals the full legacy seeded
  permutation; zero4 never carries a permutation;
* T4 and pseudo-MUA source normalizers remain view-local; pseudo-MUA side row count equals pooled
  channel count;
* source-recipe equivalence receipt rejects architecture/teacher/source split/epoch/side-dimension
  mismatch among the nine checkpoints.

### 10.3 zero4 tests

* constructor accepts channel count only and currently enforces `N=1..99`; Poison label/rate/normalizer
  objects are not read; zero and `N=100` fail;
* negative zero, float64, wrong `[N,4]` shape, source-unit count in pooled view, nonempty side input,
  and a monkeypatched `load_unit_side_features` zero4 call all fail;
* each session/view zero4 receipt has every required counter equal zero and exact positive-zero bit
  audit;
* target-label shuffle/drop leaves zero4 prediction inputs (neural, activity calibration, valid
  starts, side zero bits) bitwise identical while label evidence changes; owner-metadata distinction
  is explicitly preserved;
* consumed-sub-C zero4 parity V1 remains source/hash-valid before V9 cites it.

### 10.4 writer / metric / aggregate tests

* synthetic 15-session × 2-view × 3-arm × 3-seed writer creates all 270 artifact/commit pairs;
* missing/duplicate/unknown cell, symlink, mutable mode, wrong SHA, altered NPZ, invalid compression,
  wrong dtype/layout/shape, NaN/Inf, wrong query count and premature aggregate all fail;
* a test-only mutation of one target that also updates its local NPZ SHA/bytes and own behavior trace
  still fails the cross-18-cell pairing rule; a standalone trace mutation also fails scan/finalizer;
* Phase A does not import/instantiate R² metric or write a performance scalar; static test specifically
  rejects re-use of `v1._score_one_session_dataset`;
* finalizer re-opens all 270 NPZs and matches frozen TorchMetrics CPU behavior on ordinary, constant,
  near-constant and nonfinite synthetic cases; metric JSON scalar cannot override reopened arrays;
* bootstrap exact-match test against frozen synthetic V7 numerical reference for both comparisons and
  both views; no dict-order dependence;
* aggregate must contain separate `T4-zero4` and `T4-TS4` blocks with no cross-view rescue, and
  aggregate write-once behavior.

### 10.5 continuation / device tests

* interruption after synthetic committed cell K resumes only keys K+1...270; completed artifact SHA
  map is immutable and no existing key is forwarded again;
* caught error/nonfinite/orphan artifact creates an incident and hard-refuses recovery;
* recovery verifier sees only hashes/shapes/counts, not R²; it refuses any metric/aggregate file before
  matrix completion;
* mocked CPU/GPU identity tests reject logical-device, UUID, BDF, host or CVD mismatch and mixed-device
  records;
* consumed-sub-C all-arm/all-view CPU/GPU full-query parity is an explicit pre-formal integration
  receipt. It is the only test allowed to call actual model forward and R², and it cannot accept an
  external sub-M path.

---

## 11. 实现量、顺序与 release gate

在复用 V2 NPZ/R² primitives、V5 chronology bridge、zero4 direct constructor、V7 small numerical
bootstrap logic，而不复制 V7 control plane 的前提下，合理的最小实现量约为：

| 项目 | 估计量 |
| --- | --- |
| V9 control + canonical policy/authorization/nonce code | 250–350 production LOC |
| post-claim runtime、base/dataset factories、forward-only collector | 350–500 production LOC |
| 270 artifact-only writer、re-opener、finalizer/aggregate | 300–450 production LOC |
| writer/preparer/isolated runner/parity CLI glue | 250–350 production LOC |
| total production | **约 1,150–1,650 LOC** |
| synthetic/adversarial/unit tests | **约 600–900 LOC** |
| consumed-sub-C parity harness | 150–250 LOC（可与 runtime test helper 共享） |

这比试图继承 V7 的 generic trusted-root/ledger subsystem 更小，也比仅复制 V3R2 后再修
180→270、CPU→GPU、two-arm→three-arm、partial-metric→score-blind recovery 更可审计。

建议的实现顺序：

1. 编写 V9 static contract/source-recipe auditor、cell-key generator、synthetic artifact-only writer
   和 exact 270 aggregate tests；此阶段零 data/model/GPU；
2. 增加 control-only authorization-first runner，先证明所有失败都在 Torch/NWB/checkpoint 前停止；
3. 接入 30/50/50 V5 bridge、T4/TS4 factory 和 direct-zero4 factory；完成 zero4 Poison tests；
4. 写 forward-only collector，先在 synthetic CPU 测试；之后仅对 consumed sub-C 执行 all-arm/view
   CPU/GPU parity，生成绑定 3090 identity 的 receipt；
5. 等待九个 terminal closures + Native-M2 completion receipt，fresh byte-level review 后才生成
   append-only V9 prelaunch、unsigned envelope 和 detached signature；
6. external score 时先 artifact-only 270 matrix，最后一次 CPU finalizer；optional 450-cell
   weight-matched OOD diagnostic 只能作为另行授权的 ancillary transaction。

### V9 最终启动前的 P0 清单

V9 必须保持 blocked，直到以下每一项有当前字节级证据：

1. `shared_zero4` seeds 42/43/44 的 epoch11 checkpoints、terminal closures、独立 source-recipe
   equivalence receipt 与六个既有 T4/TS4 slots 同时完整；
2. Native-M2 completion receipt 已收到，且 V9 source/policy/closure review 未发现 drift；
3. V5R2 T4/TS4 chronology parity 与 zero4 parity V1 都仍可验证；
4. V9 import/control/writer/aggregate/continuation adversarial suites 全绿；
5. 若使用 RTX 3090，all-arm/all-view consumed-sub-C CPU/GPU parity receipt 已锁定 exact UUID/BDF
   和 `1e-4` 门；否则 policy 在外部 opening 前明确锁 CPU；
6. external output root 不存在、formal nonce 为零、没有 external sub-M forward/R²；
7. release review 写明 system-level claim、label differential、pseudo-MUA scope、N=15 cohort
   boundary 和 optional same-weight diagnostic 的 non-gating/OOD 属性。

在上述条件满足之前，最正确的工作是继续保持 V9 设计包，而不是为了“尽快跑”把 V3R2、V7 或
V8 拼接成一个未审计的 formal scorer。
