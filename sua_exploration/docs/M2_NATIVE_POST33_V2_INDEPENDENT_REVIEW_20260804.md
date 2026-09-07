# M2 native post-33 v2 独立、score-free code review

**Review target:** `M2_NATIVE_T4_SPINT_POST33_CONFIRM_V1_DRAFT_V2_HASH_HARDENED`  
**Date:** 2026-08-04（Asia/Hong_Kong）  
**Review mode:** 独立、只读；未修改任何被审实现、receipt 或既有 `source_map`  
**Scientific access boundary:** 0 GPU；未读取任何新的 post-33 endpoint R²；未打开 formal SUA session；未调用 EvalAI  
**Draft receipt reviewed:** `draft_scorefree_receipt.json`, SHA-256
`943a3eb426e2828e8683a51c071d1c4029ae64163f246ac3201336db75109540`

## 1. Executive verdict

**结论：当前 v2 draft 的数据边界与 50-bin query plumbing 基本正确，但整体仍是明确的
`NO-GO_FOR_PRODUCTION`。** 这不是因为 draft 中已经明确列出的 runner/finalizer 尚未实现，
而是另有三个会让正式运行产生错误或不可解释结果的 production blocker：

1. SPINT 当前实际记录的 `val_heldin/r2_mean` 不是六个 source session 的均值；缺失的 outer
   session 被作为 `-inf` 纳入七 session 均值，故每个 epoch 的 selector metric 都是
   `-inf`。同一问题也使当前 SPINT `test_heldin/r2_mean` 为 `-inf`。
2. T4 的实际 resolved Hydra config 仍指向全局 legacy `epoch_034.ckpt`，并没有指向同
   `fold × seed` 刚选出的 SPINT checkpoint。当前配置因此不满足 draft 所声明的 paired
   decoder substrate 合同。
3. SPINT Hydra run directory 只有秒级时间戳，没有 protocol/fold/seed/arm，也没有训练入口的
   collision assertion。两个 GPU 同秒启动时可写入同一个目录；当前也没有 per-cell `O_EXCL`
   ownership gate。

此外，v2 verifier 对“当前默认 receipt 未漂移”有一定保护，但并不是一个可作为 final launch
authorization 基础的封闭证明：它不要求精确 `source_map` 集合、不限制 receipt 中的路径到规范
root、不强制 receipt sidecar、也不重新验证 old protocol/C1 closure；若 receipt 本身被替换，
可以删除 source rows 或替换 audit/live 路径而仍通过许多结构检查。

| 审计项 | 判定 | 说明 |
| --- | --- | --- |
| outer 不进入 datamodule train dataset | **PASS** | 两条 versioned datamodule 当前均只给 source train 暴露 6 sessions |
| outer 不进入 T4 normalizer | **PASS** | `native_t4_normalization.train_sessions` 为准确的 6 source sessions |
| outer 不进入实际 checkpoint metric | **FAIL** | 数据没有进入，但 SPINT hard-coded 7-session metric 把空 outer 作为 `-inf` 纳入 selector |
| 33-trial support 语义跨实现一致 | **PASS（独立实数据证据）** | 7/7 folds 的 support tensor shape、bytes SHA、trial-start SHA 全部逐 fold 相同 |
| 50-bin full-history q33 boundary 跨实现一致 | **PASS** | 7/7 folds 的完整 `window_indices` SHA、first/last/count 均相同，共 101,171 windows |
| T4 target fit 只用 first 33 labels/rates | **PASS at datamodule level** | query labels 不进入 descriptor；rank-3 audit 已存在；仍需正式 runtime instrumentation |
| paired fold/seed SPINT decoder | **FAIL** | resolved T4 config 使用全局 legacy teacher path |
| 31/31 decoder freeze/update closure | **NOT IMPLEMENTED** | 31-tensor architecture count正确，但尚无 production pre/post runtime closure |
| source-only Hydra/callback contract | **FAIL** | callback 名义监控 source key，但 SPINT key 本身为 `-inf`；contract helper 未接入 callback |
| collision-free paths | **FAIL** | SPINT 默认目录仅秒级时间戳；没有 deterministic per-cell `O_EXCL` gate |
| draft anti-drift verifier | **PARTIAL** | 默认 receipt/data/source SHA 当前通过，但 exact-set/path/closure 防替换不完整 |
| production authorization | **FAIL by design and defects** | draft 自己正确声明未授权，且上述 blocker 尚未修复 |

## 2. Data isolation audit

### 2.1 SPINT versioned datamodule

`SPINT-main/src/data/falcon_post33_confirm_v1_datamodule.py` 没有委托历史 base `setup()`，而是
显式完成以下划分：

- `source_calib_files`：准确的六个 source held-in-calib NWB；
- `source_minival_files`：相同六个 source held-in-minival NWB；
- `outer_files`：唯一 outer 的 held-in-calib NWB；
- `train_dataset = FalconDataset(source_calib, source_calib)`；
- `val_heldin_dataset = FalconDataset(source_minival, source_calib)`；
- `post33_query_dataset = Post33FalconDataset(outer, outer)`。

`load_data()` 只接受 `_approved_nwb_paths`，`_role_files()` 固定规范目录并拒绝文件名中的
`held-out`。在当前 14 个允许 NWB 上，outer 不进入 SPINT train/val dataset。因为
`standardize_covariates=false`，也没有从 outer 拟合行为 normalizer 的路径。

### 2.2 T4 versioned datamodule

`streaming_calibration_exp/src/data/falcon_post33_confirm_v1_datamodule.py` 给历史 base class 的
`heldin_session_names` 只传六个 source sessions，并将内部 delegation 固定为 `minival`。base
setup 因此对六个 source calibration sessions 建立 train dataset，对六个 source minival
sessions 建立 validation dataset，并由六个 source calibration sessions 拟合 T4 z-score
normalizer。之后 subclass 才单独加载 outer calibration NWB，并把 source normalizer 应用到
outer 的 first-33 T4 descriptor。

当前代码在 base setup 后还检查：

```text
set(train_session_names) == set(source_session_names)
set(val_heldin_session_names) == set(source_session_names)
```

当前 live evidence 的 `native_t4_normalization.train_sessions` 也与六 source sessions 一致。
因此，从 datamodule 的 train/normalizer/query 分工看，outer isolation 成立。

### 2.3 重要限定：data isolation 不等于 model isolation

上述 PASS 只证明数据模块没有把 outer window 送入 source fit。它不能弥补：

- SPINT selector 聚合逻辑把缺失 outer 的空 metric 纳入均值；
- T4 当前默认加载的 global teacher 并非该 outer fold 的 source-only SPINT checkpoint。

因此不能从 `get_split_manifest().outer_counts == 0` 推导出整个训练/选择链已经 outer-clean。

## 3. 33-trial support 与 50-bin q33 boundary

### 3.1 两条 query 实现的等价性

SPINT 的 `Post33FalconDataset` 与 T4 base `FalconDataset(query_start_trial=33)` 都先给原始序列
左侧补 `window_size - 1 = 49` 个 bin，然后令：

```text
minimum_window_start_padded_bin = trial_start_indices[33]
retain iff window_start >= minimum_window_start_padded_bin
```

所以第一个合法 50-bin history 的**第一 bin**位于 trial 33 boundary，而不是只要求 prediction
bin 位于 boundary 之后。两者都从同一个 outer held-in-calib NWB 构造 online query，且都从
同一 NWB 的 chronological first 33 trials 构造 calibration activity。

### 3.2 独立 all-fold 字节级检查

本 review 额外运行了 14 次 score-free setup（2 implementations × 7 folds），未调用模型或
scorer。对每个 outer fold 比较：

- `calib_trialized_neural_features[outer][:33]` 的 shape 与完整 bytes SHA-256；
- first-33 `calib_trial_start_indices` 的 bytes SHA-256；
- 完整 `window_indices` JSON SHA-256、首项、末项和数量。

每一 fold 的 SPINT/T4 值均完全相同：

| Fold | Outer session | support tensor SHA-256（两臂相同） | query-index SHA-256（两臂相同） | Windows |
| ---: | --- | --- | --- | ---: |
| 0 | `ses-2020-10-19-Run1` | `ae8a67d36856d304ada8b37b22553d0d94d00085995941eaa923d0b6efd8a04d` | `41aa46b305f182f79a7b6dbf620681bcd03456ae690e489a0b765f8608ea67cd` | 17,854 |
| 1 | `ses-2020-10-19-Run2` | `44a12e9ab869437aed133fe2bb9034db7c63d1c9f19864271c67d25dde1e03df` | `915ab3bc213dc2e0a1d826264158534f608f4da3c537f3cd8823c52be0cd393d` | 18,088 |
| 2 | `ses-2020-10-20-Run1` | `075ca0fe6375485fe2f0ab8266d99f54626b22673f43d6966f532d30ff3297b9` | `71ecbb0f3678627069ac0fe496a25e880ca262b9b6d23b5914f00705b06cc257` | 13,451 |
| 3 | `ses-2020-10-20-Run2` | `967bdc05b71e81fd5286cfb9f1675c0b50a64815797733721b653501e69750b2` | `f27d8e96ca01a904b42d33009a4d3c6d62cdd6d6efa6872f554692077fad9663` | 13,122 |
| 4 | `ses-2020-10-27-Run1` | `33d50819e0bdff702804a6d74f9824bdc568034fc6e6fcffbf2510fd684f968e` | `9dab9f1b5ee273c21b6bc16c976f4b794efef9c61c66550de1ee16636c368b0c` | 10,225 |
| 5 | `ses-2020-10-27-Run2` | `7a3a10b059a15760d7e1c925690596b0146889d1c720a9091aeb3d6106d47bad` | `c63f5fded629c25164dbd6e262b4461f52747f7614937ecf8eb7b99a779add9c` | 12,694 |
| 6 | `ses-2020-10-28-Run1` | `0d5e73711cf221d6389ed13a4980a5afa3ac8c3920b919c337ae8dbf3e349de8` | `9850c4032a6d272b6be90895518f77088d04e51aface3db3f861ef4b57c18f8f` | 15,737 |

所有 support tensors 均为 `[33,100,96]`, `float32`。总 query windows 为 101,171。这个结果
强于只比较 manifest 中的 count，说明当前两条实现的神经 support 和完整 query index 在实数据
上逐字节对齐。

这项检查目前只存在于独立 review 记录中；final score-free receipt 应把同一检查写成不可覆盖的
正式 artifact，而不是依赖本文转述。

## 4. Critical findings

### C1 — SPINT source-only selector metric 恒为 `-inf`

**Severity:** critical; production blocker.

`M2Post33ConfirmSPINTDataModule.val_dataloader()` 正确只返回六 source sessions。但是
`SPINT-main/src/models/falcon_module.py:185-205` 的 validation aggregation 遍历
`DATASET_NAMES['m2']['heldin']` 中全部七个 session。对没有任何 validation sample 的 outer
session，代码显式产生 `-inf`，再将七项全部取均值：

```python
r2 = sess_r2.compute() if sess_r2.total > 2 else torch.tensor(-torch.inf)
val_heldin_r2s.append(r2)
val_heldin_r2_mean = torch.stack(val_heldin_r2s).mean()
```

因此，尽管 outer 的 window count 确实为 0，`val_heldin/r2_mean` 仍然在数学上等于
`-inf`。这不是“manifest 小瑕疵”：ModelCheckpoint 正在监控这个 key，所以无法按六 source
session 表现选择 epoch。draft 中测试的 `contract.select_source_checkpoint()` 从未被 SPINT
训练入口或 callback 调用；它只是一个离线 helper，不能修复当前 runtime selector。

同样，`falcon_module.py:236-257` 的 test aggregation 在只测试唯一 outer 时，把另外六个空
held-in metrics 也作为 `-inf` 纳入均值，故当前 `test_heldin/r2_mean` 也无效。唯一 outer 的
per-session metric 可以是有限值，但 production finalizer 尚未实现，也不能默认 generic mean
正确。

**最小修复合同 M1：**

1. 为本协议使用 versioned SPINT metric module/callback；不要改变历史全局语义。
2. validation epoch end 必须证明 actual finite session set **精确等于** frozen
   `source_session_names`：恰好 6 个、无缺失、无额外项、每项 `total > 2` 且有限。
3. outer session 必须存在于审计状态但 `total == 0`，且绝不能进入 mean。
4. `val_heldin/r2_mean` 必须是上述 6 个 session R² 的 equal-session arithmetic mean；任何
   empty/nonfinite/source-set mismatch 直接 abort，不保存 checkpoint。
5. test 时必须恰好只有 frozen outer session 有样本；只报告该 per-session endpoint，禁止把
   六个 empty placeholders 聚合进去。
6. 记录 epoch 0--34 的全部 source-only metric records，并由实际接线的 selector 选择最大值；
   精确相等时选择更早的 zero-based epoch。不要仅假设 Lightning 对 equal score 的内部行为。

必须添加的负向 test：六个 synthetic source metrics + 一个 empty outer 时 mean 有限且等于六项
均值；outer 获得一个 sample、任一 source 为空、出现第七个实际 source、metric 非有限时均 fail。

### C2 — T4 resolved config 没有使用 paired fold/seed SPINT checkpoint

**Severity:** critical; production blocker and outer-isolation violation risk.

draft receipt 声明：

```text
decoder_source = paired_fold_seed_SPINT_selected_checkpoint_only
```

但实际运行：

```bash
cd streaming_calibration_exp
python src/train.py experiment=m2_native_post33_confirm_v1_t4 --cfg job --resolve
```

解析得到：

```text
model.teacher_ckpt_path =
  ./../SPINT-main/logs/train/runs/2026-07-07-16-05-16/checkpoints/best_ckpt/epoch_034.ckpt
model.require_clean_teacher_receipt = false
model.teacher_receipt_path = null
```

该值来自未进入 draft `source_map` 的
`streaming_calibration_exp/configs/paths/default.yaml`。T4 experiment yaml 没有覆盖它。因而当前
配置对所有 fold/seed 使用同一 global legacy teacher；至少可以确定它不是本次同 fold/seed
source-only SPINT 产生的 checkpoint，也没有 paired outer-exclusion receipt。

**最小修复合同 M2：**

1. T4 production config 不得从 `paths.default.teacher_ckpt_path` 获得 fallback；缺少显式 paired
   path 时必须在 Hydra instantiate 前 fail。
2. 每个 `(fold,seed)` 必须先完成唯一 SPINT cell，写出 write-once completion receipt，其中
   至少包含 protocol、arm=`spint`、fold、seed、outer session、六 source sessions、selected
   epoch、selector records、checkpoint canonical path/size/SHA-256 和 resolved config SHA。
3. runner 只能从该 completion receipt 解析 `teacher_ckpt_path`；fold、seed、outer、source set
   任一不匹配即 fail。不得接受手写路径或 global paths default。
4. T4 optimizer 创建前，比较 paired SPINT `net.state_dict()` 与 T4
   `student.decoder.state_dict()`：精确 31 keys，key set/name/shape/dtype/bytes 全部一致。
5. 31/31 decoder parameter tensors 必须全部 `requires_grad=false`，且 optimizer parameter IDs
   与 decoder parameter IDs 交集为空。
6. 保存 T4 source-train 前的 31-tensor aggregate hash；12 epochs 后、selected T4 checkpoint
   reload 后、outer query 前各重新比较 31/31 bit-exact，并要求 updated tensor count = 0。
7. pre/post receipt 还要记录 teacher checkpoint SHA；只比较 tensor count 而不比较 names/bytes
   不足以放行。

独立 CPU architecture probe 确认当前 M2 SPINT `net.state_dict()` 正好有 31 个 parameter tensors、
0 buffers，所以“31”本身是正确的；缺的是 paired 来源与 runtime pre/post proof。

### C3 — SPINT output/checkpoint directory 可碰撞

**Severity:** critical operational blocker.

`SPINT-main/configs/hydra/default.yaml` 当前是：

```text
${paths.log_dir}/${task_name}/runs/${now:%Y-%m-%d-%H-%M-%S}
```

它没有 protocol/fold/seed/arm，分辨率只有一秒。SPINT `src/train.py` 也没有 streaming 分支的
`assert_run_dir_is_fresh()`。在两张本地 GPU 并行 launch 两个 fold/seed 时，同秒启动会解析到同
一目录，checkpoint、TensorBoard 和 `.hydra` metadata 可混写。即使碰巧错开一秒，也没有
per-cell ownership，重复 launch 会生成另一个目录，违反 no-retry/42 exact-cell contract。

**最小修复合同 M3：**

1. result、checkpoint、Hydra run、status 四类路径都必须含
   `protocol_id / arm / fold / seed`；例如：

   ```text
   .../M2_NATIVE_T4_SPINT_POST33_CONFIRM_V1/arm-spint/fold-3/seed-42/
   ```

2. scheduler 在启动进程前，对 deterministic cell ownership file 使用
   `os.open(path, O_CREAT|O_EXCL|O_WRONLY)`；已存在即 fail，不生成 suffix、不 resume、不 retry。
3. Hydra `run.dir`、ModelCheckpoint `dirpath`、artifact result root 全部显式指向该 cell 私有目录，
   并在进程内再次验证 canonical resolved path。
4. `started`, `completed`, `failed` 各为 write-once；completed/failed 必须引用同一个 started
   receipt SHA。不得用 `mkdir(exist_ok=True)` 作为 ownership gate。
5. 增加一个并发 test：两个进程竞争同一 cell 时恰好一个成功获取 ownership，另一个在任何
   checkpoint 写入前失败；不同 fold/seed/arm 的路径必须两两不相交。

### C4 — v2 verifier 不是 exact-set、canonical-path 的 final authorization verifier

**Severity:** critical for final receipt; current default receipt仍可作为局部 accidental-drift
检查，但不能被“原地提升”。

`verify_m2_native_post33_confirm_v1_draft.py` 对默认 receipt 的 14 个 NWB 和现有 30 个 source
rows 都重新计算 SHA，因此**同路径、同集合下的同大小内容漂移会被正确拒绝**。默认 verifier
本次也确实 PASS。然而它存在以下防替换缺口：

1. `source_map` 只遍历 receipt 中现有 rows，不与代码内的精确 required-path set 比较；删除一
   个或全部 rows 不会触发 `missing source`。
2. source-map key 可以是 absolute path；`ROOT / absolute_path` 会采用 absolute path。verifier
   不要求 row 内 `path == key`，也不要求路径位于 repo root。
3. `verify_pinned_file()` 接受 receipt 自报的任意 absolute NWB path/hash，只检查路径字符串不含
   `held-out`、role 合法、每 role 有 7 个不同 session；它没有要求 exact canonical 14 paths 或
   session set 精确等于 frozen `FOLDS`。
4. receipt 的 `.sha256` sidecar 是可选的；不存在时 verifier 跳过 receipt self-digest。也没有
   一个 verifier 内/外固定的 v2 receipt SHA。
5. verifier 重新检查 old draft SHA，但没有重新检查 old protocol receipt SHA，也没有重新读取
   active C1 receipt 并验证其 34-file source map。receipt 中的
   `active_c1_v3r2_noninterference` 只是未验证字段。
6. verifier 验证 audit/live 文件 SHA 来自当前 receipt 本身；若 receipt 被整体替换，路径和 SHA
   可一起替换。它不把 `audit_payload.folds/endpoint` 与 receipt folds/endpoint 逐项相等比较。
7. 对 live payload，只验证 `status` 和自报 execution scope；`probe_count` 与 window totals 检查
   的是 receipt summary，没有与 `live_payload.probe_count/eligible_window_totals/sides` 交叉比较。
8. `execution_scope` 是文件中的自声明，没有 syscall/import/path access instrumentation。当前
   audit scripts 从静态审查看确实 score-free，但 verifier 本身不能把任意同字段 JSON 证明为
   实际未访问。

这解释了为何 v2 verifier 能 PASS，却没有发现 C1--C3：它验证的是 draft 自报结构、当前列出的
hash 和 data plumbing；它不实例化 SPINT model metric aggregation，不 compose 后断言 paired
teacher path，也不检查 Hydra output-directory topology。

**最小修复合同 M4：** final verifier 内置不可由 receipt 改写的 exact required source/config/
evidence path set；要求 observed keys 与 expected keys 完全相等。所有 path 先 `resolve()`，再
要求位于 frozen canonical root，且 data session/role/path 三元组精确匹配。final authorization
必须从一个外部固定 SHA 的 receipt 入口启动，sidecar 必须存在且路径/文件名/digest 三者精确。
同时重新验证 old protocol、superseded draft、C1 receipt 和 C1 exact source map；逐字段比较
audit/live payload，而非只信 receipt summary。

## 5. High-severity findings

### H1 — T4 base-class delegation 建立了两套互相矛盾的 public/internal hparams

T4 constructor 先验证 public tuple `validation_protocol=loso, loso_fold=f,
heldin_query_start_trial=33`，随后为复用 base setup 把 `self.hparams` 改为：

```text
validation_protocol = minival
loso_fold = None
heldin_query_start_trial = 0
query_start_trial = 0
```

独立实例化确认运行时 `dm.hparams` 的确是上述内部值；public fold/q33 只保存在
`protocol_loso_fold`, `outer_session_name`, `source_session_names` 和自定义
`get_split_manifest()`。当前 query dataset 仍由常量 33 正确构造，所以这不是当前 boundary
失败；风险在于任何 generic callback、checkpoint metadata、resume validator 或未来 closure 若读
`datamodule.hparams`，都会看到 `minival/fold=None/q0`，而不是 public q33 contract。

final 实现应避免覆写 public hparams：优先写 versioned source-only setup，不借用有不同语义的
base hparams；若暂时保留 delegation，则 public contract 与 internal delegation 必须分成明确的
immutable structures，runtime receipt 同时记录并验证二者，任何 generic consumer 禁止从
`hparams` 推断 public endpoint。

### H2 — `select_source_checkpoint()` 的 tie-break 合同没有接到生产路径

contract helper 正确实现了 max metric + earlier zero-based epoch，但当前两个 experiment 都直接
使用 Lightning `ModelCheckpoint(save_top_k=1)`。没有 code path 收集 contract helper 所需的
records，也没有检查 `metric_scope` 或 `outer_session_window_count`。因此 focused unit test 只证明
helper 本身，不证明 runtime checkpoint choice。应按 M1 的要求接入 versioned selector 或在全部
epoch source records 完成后显式选择并核验。

### H3 — draft `source_map` 缺少会改变实际结果/证据的 transitive sources

当前 source map 有 30 rows。两个 model yaml 本身已经包含：

- `SPINT-main/configs/model/falcon_m2.yaml`；
- `streaming_calibration_exp/configs/model/streaming_b3s_t4.yaml`；
- `streaming_calibration_exp/configs/model/_streaming_base.yaml`。

但是至少缺少下列直接影响 runtime 的文件：

| 类别 | 关键遗漏 | 影响 |
| --- | --- | --- |
| SPINT architecture | `SPINT-main/src/models/components/spint.py` | 实际 31-tensor decoder/forward 实现 |
| T4 teacher/decoder architecture | `streaming_calibration_exp/src/models/components/spint.py` | load checkpoint 和 student decoder 的实际类 |
| T4 optimizer/dropout dependency | `streaming_calibration_exp/src/models/components/neuron_dropout.py` | model module 直接 import，虽当前 mode=none仍是运行源 |
| default paths | 两项目的 `configs/paths/default.yaml` | T4 legacy teacher 正是从这里注入；SPINT data/output root 也由此决定 |
| Hydra topology | 两项目的 `configs/hydra/default.yaml` | SPINT 秒级目录碰撞来自此文件 |
| trainer/logger/extras | 两项目 resolved defaults | accelerator、validation cadence、logging/运行目录行为 |
| callback fragments | SPINT `callbacks/{override_epoch_step,lr_monitor,model_summary}.yaml` | `post33_source_only.yaml` 通过 defaults 聚合这些文件 |
| callback implementation | 两项目 `src/callbacks/override_epoch_step.py` | 实际 callback code |
| callback instantiation | 两项目 `src/utils/instantiators.py` 及相关 utils | 决定 YAML 如何转成实际 callbacks |
| T4 artifact/selection metadata | `streaming_calibration_exp/src/metrics/run_artifacts.py` | checkpoint选择记录、run dir、source manifest、metadata |
| T4 model metadata dependency | `streaming_calibration_exp/src/models/falcon_module.py` | `DATASET_NAMES` 和 teacher checkpoint class |
| environment | Python/Lightning/Torch package lock或完整 environment receipt | `ModelCheckpoint` tie/collision行为依赖版本 |

final source map 应从实际 resolved Hydra target + direct/transitive imports + callback defaults 构造，
而不是手写一个方便的 30-file subset。

### H4 — generic T4 artifact schema 与自定义 split manifest 不匹配

versioned T4 `get_split_manifest()` 使用 `loso_fold`, `source_train_sessions`,
`source_checkpoint_selection_sessions`；generic `run_artifacts.py` 则读取 `fold_id`,
`train_sessions`, `validation_sessions`。因此 generic metadata 可写出空 train/validation lists。
更严重的是 `write_run_metadata()` 当前硬编码 `teacher_seen_validation_session=True`，与未来 paired
outer-clean teacher 的 claim 冲突。production closure 必须使用本协议的 exact schema，不得把
generic artifact 成功写出当成 provenance PASS。

### H5 — runtime data/source hash 与校准更新证明尚未接线

当前 draft verifier 会在 prelaunch 时重 hash 约 15 GB 输入，这是好的 accidental-drift gate；但
datamodule 本身只按 canonical path allowlist，不验证 frozen bytes。draft 已把 per-cell data/config/
normalizer/checkpoint hashes、target zero-backprop 和 decoder pre/post equality列为 future closure，
这些是已知 deliberate missing items。它们仍然是 final launch 的硬依赖：最终 runner 必须在实际
cell 使用的数据被加载前绑定 hash，不能把数小时前的 draft verify 当成 runtime proof。

### H6 — 新 source/config 尚未进入 Git；result receipt 被 `.gitignore` 排除

`git ls-files` 对本协议新增的 datamodule/config/test/doc 当前均无输出；它们是 untracked，但并未
被 `.gitignore` 忽略。`sua_exploration/results/` 则按仓库政策整体忽略，所以 v2 draft receipt
不会随 Git 同步。local hashes 在当前机器上仍有用，但发表/远端复现前必须：

- 把 source/config/test/docs 纳入版本控制或不可变 release bundle；
- 把小型 final authorization manifest 放到可同步位置，或在 tracked doc 中固定外部 artifact
  URI + SHA；
- 不要假设 GitHub 已包含当前 local source map。

## 6. Medium findings

1. protocol doc 第 7 节写“23 tests”，当前 focused file 实际收集并通过 26 tests；这是文档陈旧，
   不是科学结果问题。
2. 所谓“exact” calibration count 会接受 `33.0`，因为 tuple equality 中 `33.0 == 33`；如果 final
   schema 要求 integer，应显式拒绝 bool/float/non-Python-int。
3. T4 base setup 仍用 `data_dir.rglob()` 找 source files，然后靠 canonical allowlist 拒绝未批准
   path。当前规范数据树正常，但 symlink/duplicate canonical occurrence 可能重复枚举；versioned
   setup 应像 SPINT path 一样直接用 exact role files。
4. live probe 的 manifest 大部分来自预设 attributes。当前实数据 support/query SHA 对照弥补了
   query 侧证据，但 final live receipt仍应记录 actual dataset session keys、actual loaded canonical
   paths、normalizer input session keys，而不只记录 `get_split_manifest()` 声明。
5. current live payload 记录 `stderr_was_empty`，但 builder/verifier不要求它为 true。若 future
   setup 出现 warning/partial fallback，仍可能生成 marker；final probe 应禁止非空 stderr 或把
  允许的 warning 精确列举。

## 7. Why existing tests and verifier passed

本次运行结果：

```text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 CUDA_VISIBLE_DEVICES='' \
  python -m pytest -q sua_exploration/tests/test_m2_native_post33_confirm_v1_plumbing.py
26 passed

python -m py_compile <7 reviewed Python modules>
PASS

git diff --check -- <reviewed post33 paths>
PASS

python sua_exploration/scripts/verify_m2_native_post33_confirm_v1_draft.py
PASS_DRAFT_V2_HASH_HARDENED_NOT_AUTHORIZED
source_count=30, fold_count=7, terminal_cell_count=42, eligible_windows=101171
```

这些 green checks 没捕获 blocker 的原因很具体：

- tests 检查 contract helper 的 source selector，却没有调用 SPINT
  `FalconLitModule.on_validation_epoch_end()` 的 hard-coded 7-session aggregation；
- tests 只检查 data yaml/experiment 静态字段，没有对 T4 完整 resolved config 的
  `teacher_ckpt_path` 做 paired-cell assertion；
- source map 没有 Hydra default config，所以 verifier 看不到 SPINT run-dir collision，也看不到
  T4 teacher fallback 的来源；
- verifier 遍历 receipt 自带的 source rows，不知道预期还应有上述 transitive files；
- live setup 只实例化 datamodule，不实例化模型、callback、teacher 或 checkpoint selector。

因此，当前 PASS 应准确命名为“data/query plumbing + listed-file accidental-drift PASS”，不能提升为
“production pipeline PASS”。

## 8. Required score-free closure before a final receipt

以下项目全部完成前不得生成 GPU-authorizing receipt：

1. 实现并测试 M1 source-only metric contract；resolved callback 必须实际监控这个有限六-session
   mean。
2. 修复 SPINT test aggregation，只接受唯一 outer per-session endpoint；不允许 empty placeholders
   进入 mean。
3. 实现 M2 paired teacher chain；T4 config 缺显式 paired receipt/path 时 fail，禁止 legacy default。
4. 实现 31/31 named/typed/byte-exact pre/post checks、requires-grad=0、optimizer intersection=0、
   updated tensors=0。
5. 实现 M3 deterministic per-cell paths 与 `O_EXCL` ownership；并发 collision test PASS。
6. 完整 compose 两个 arm 的每个 fold/seed resolved Hydra config；断言 data/model/callback/trainer/
   teacher/output path exact，且 command-line override allowlist 为空或严格固定。
7. 把本 review 的 7-fold support tensor/trial-start/query-index parity 变成 write-once official
   score-free artifact。
8. 将 final source map 改为内置 exact expected set，补齐 default paths、Hydra、model component、
   callback aggregation、artifact writer及环境锁。
9. 新增 anti-tamper tests，至少证明删除 source row、absolute/path-traversal 替换、替换 canonical
   session、缺 sidecar、old protocol drift、C1 receipt/source drift、live payload/summary不一致均 fail。
10. final receipt verifier重新 hash exact 14 data inputs、exact source set、paired SPINT/T4 configs和
    paired checkpoint；receipt 自身 SHA 必须由独立 root authorization 固定。
11. production runner/scheduler、per-cell closure、Stage-A finalizer、full finalizer、cost/latency receipt
    等 draft 已明确列出的 deliberate missing items全部实现并通过独立 review。

只有完成上述 closure，才可以把状态从
`DRAFT_SCORE_FREE_PRELAUNCH_V2_NOT_AUTHORIZED` 推进到一个**新的**、write-once、不可原地修改的
final authorization。现有 v2 draft 不应被修改或原地提升。

## 9. Final review statement

当前最值得保留的成果是：七 fold outer split、first-33 neural support 和 full-history post-33
query 在 SPINT/T4 两条 datamodule 上已经得到很强的 score-free 一致性证据；T4 normalizer 也确实
只使用六 source sessions。当前最需要修复的不是数据切分，而是 **metric aggregation、paired
decoder provenance 和 collision-proof execution**。这三项任何一项未完成，跑出的 42-cell
matrix都不能回答预注册问题，且不应消耗新的 endpoint score scope。
