# HANDOFF：无标定对照修复 + Per-Unit 侧信息消融

**日期：2026-07-25**
**给：下一位接手实现的 AI**
**范围：两个任务，Task A 是 Task B 的硬前置。**

---

## 一句话

先修好 `learned_prior` 无标定对照（它现在退化成了 zero-identity，
使得"calibration 带来多少增益"没有可信下界），再实现 per-unit 侧信息
（SNR / 波形标量 / electrode）消融。章程已经写好并冻结，你只负责实现和
执行，**不要改门槛**。

---

## 0. 开始前必读

| 文档 | 作用 |
|---|---|
| [`UNIT_SIDE_FEATURE_ABLATION.md`](UNIT_SIDE_FEATURE_ABLATION.md) | Task B 的**冻结章程**：特征定义、门槛、数据隔离、不能声称的内容 |
| [`ASIC_DEPLOYMENT_CHARTER.md`](ASIC_DEPLOYMENT_CHARTER.md) | 为什么这些实验值得做（速率域论证） |
| [`CURRENT_RESULTS.md`](CURRENT_RESULTS.md) §F | Task A 要修的缺陷的完整描述 |
| [`ATTENTION_ARCHITECTURE_SCREEN.md`](ATTENTION_ARCHITECTURE_SCREEN.md) | 门槛写法的参照；Task B 沿用同一套阈值 |

### 绝对禁止

1. **不得加载、评估或以任何方式读取 6 个 test sessions 的 spike / behavior /
   trial 数据。** test 列表见
   `results/attention_arch_screen_v3/aggregate.json` 的 `sua.session_splits.test`。
   split 初始化时至多可读 test NWB 的 unit-table 行数以固定 regime。
2. **不得创建、修改、覆盖或删除**
   `results/p3_formal_test_816cdd8b…_receipt.json`。该 scope 的处置是用户决策
   （见 [`../ROADMAP.md`](../ROADMAP.md) G1），不是实现细节。
3. **不得运行 `eval_adaptation_dandi688.py` 的 formal test 路径**（即不得不带
   `--protocol_lock` 调用，也不得新建 protocol lock）。
4. **不得在看到中间结果后增删 seed、特征组或调整阈值。**
5. **不得新增 B17+ 结构或 attention 变体。** P2 已给出阴性结论。
6. 不要删除任何既有产物。被取代的工件标注 `superseded_by`，不要 `rm`。

### 环境

```
python:  /home/xinyuan/miniconda3/envs/spint/bin/python   (conda env: spint)
torch:   2.5.1 + CUDA，2× RTX 3090
data:    sua_exploration/data/dandi_000688/sub-C/   (53 CO NWB，已下载)
cwd:     /home/xinyuan/Work_host/SPINT
```

多数脚本用相对路径，从仓库根目录运行。

---

## Task A：修复 `learned_prior` 无标定对照

### A.1 根因（已定位，不需要你重新排查）

`streaming_calibration_exp/src/models/streaming_calibration_module.py:273-277`：

```python
self.population_identity = nn.Parameter(
    torch.zeros(1, 1, self._window_size, dtype=torch.float32)
)
```

**初始化为全零。** 而
`sua_exploration/scripts/select_gradient_free_protocol_dandi688.py:79` 有
`strict_load = identity_mode != "learned_prior"`，即 learned_prior 模式下用
非严格加载。

于是当 `eval_no_calibration_validation_dandi688.py --control_mode learned_prior`
去加载一个 **calibrated 模式训练的 checkpoint**（其 state_dict 里根本没有
`population_identity` 这个 key）时，该参数静默保持全零，
**learned_prior 与 zero_identity 在数值上完全等价**。

实测证据：

| 文件 | ckpt | 与 zero-identity 的差 |
|---|---|---:|
| `p3_no_calibration_validation_b15_learnedprior_s42.json` | `b15_dandi688_co_oldregime_s42`（calibrated 模式） | `3.97e-07` |
| `p3_no_calibration_validation_b16_learnedprior_s42.json` | `b16_dandi688_co_oldregime_s42`（calibrated 模式） | `0.0` |

`p3_fair_baseline_summary_s42.json` 已记录
`control_mode_learned_prior_equal_zero_identity: true`，却仍把
`delta_vs_learned_prior = +2.6667 / +0.8197` 当作公平余量报出。

### A.2 要做的四件事

**A.2.1 加载守卫（必做，先做）**

在 learned_prior 路径的模型加载处（`eval_no_calibration_validation_dandi688.py`
的 `load_frozen_model`，以及 `select_gradient_free_protocol_dandi688.py` 中
同类逻辑）加入硬性检查，任一不满足就 **raise，不要 warn**：

- checkpoint 的 `state_dict` 必须**包含** `population_identity` 键；
- 其值必须**不是**全零（用 `torch.count_nonzero` 判定）；
- 非严格加载的 `missing_keys` 中不得出现 `population_identity`。

错误信息要指明"该 checkpoint 不是以 `--identity_mode learned_prior` 训练的"。

**A.2.2 诊断挂起（必做，在重跑前）**

`checkpoints/b16_dandi688_co_learnedprior_s42/` 于 2026-07-24 23:20 启动，
23:23 后再无写入（只有 tfevents、hparams、run_metadata，**无 checkpoint**），
进程存活约 18h42m 后消失。先定位原因再重跑，否则会重复浪费 GPU。

排查起点：该 run 用了 `--num_workers` 默认值以外的配置吗；
A.1 提到 learned_prior 模式会 `requires_grad = False` 冻结整个
`id_encoder`（module:269-271），若同时 `--freeze_decoder`，优化器可能
拿到空参数组或只剩 `population_identity`——检查
`configure_optimizers`（module:470-490）在该组合下的行为。

**A.2.3 训练收敛的 learned-prior 基线**

对 **B3 与 B16 各一个**，seed 42，与对应 calibrated run 完全相同的
teacher / split / protocol / epoch 预算：

```bash
/home/xinyuan/miniconda3/envs/spint/bin/python -u \
  sua_exploration/scripts/train_variant_dandi688.py \
  --variant B3 --identity_mode learned_prior \
  --out_name b3_dandi688_co_learnedprior_s42_v6 \
  --task CO --split_counts 27,6,6 --max_units_exclusive 100 \
  --seed 42 --max_epochs 40 --lr 1e-4 \
  --teacher_ckpt sua_exploration/checkpoints/teacher_mc_maze/best-epoch=083-val_heldin/r2_mean=0.9061.ckpt
```

注意：既有的 `b16_dandi688_co_learnedprior_s42_fair` 只跑了 **1 个 epoch**
（`run_metadata.json` 里 `training.max_epochs = 1`），val R² `−0.0191`，
**不是收敛基线**，不要引用。

**A.2.4 重新生成公平摘要**

用新 checkpoint 重跑 `eval_no_calibration_validation_dandi688.py` 的两个
control_mode，然后重生成 `p3_fair_baseline_summary_s42.json`。
同时修改生成脚本：当 `control_mode_learned_prior_equal_zero_identity`
为 true 时，**必须拒绝写出 `delta_vs_learned_prior` 字段**（或写成 `null`
并附 `invalid_reason`），不能像现在这样照常报数。

旧文件加 `superseded_by` 字段指向新文件，不要删除。

### A.3 Task A 验收

- [ ] 用 calibrated 模式 checkpoint 跑 `--control_mode learned_prior` 会**报错退出**（附命令与输出）
- [ ] 挂起原因有书面结论（哪怕结论是"未复现"，也要写明排查了什么）
- [ ] B3 与 B16 各有一个收敛的 learned-prior checkpoint + `run_metadata.json`
- [ ] 新 `p3_fair_baseline_summary_s42.json` 中 learned_prior 与 zero_identity 的
      mean R² **不再相等**
- [ ] 更新 [`CURRENT_RESULTS.md`](CURRENT_RESULTS.md) §F.2 / §F.3 与
      "已知的产物缺陷"表
- [ ] 全部为 validation-only，`no_test_files_evaluated: true`

**Task A 通过我的审核后才能开始 Task B。**

---

## Task B：Per-Unit 侧信息消融

章程是 [`UNIT_SIDE_FEATURE_ABLATION.md`](UNIT_SIDE_FEATURE_ABLATION.md)，
**先完整读一遍**。以下只讲实现。

本轮只做 `F0 / F1 / F2 / FS` 四组，**`F3`（electrode embedding）推迟**到
F1/F2 有结论之后再立项，以免第一版改动面过大。

### B.1 编码器 API 扩展（最小改动方案）

`streaming_encoders.py` 的 `CalibrationEncoder` 用一个显式 `state` dict 在
`reset_stream → push_trial → finalize_identity` 之间传递。所有 `push_trial`
实现都是「原地改 dict 再 return 同一个 dict」，因此**在 `state` 里多塞一个键
可以无损穿过全部现有子类**。

只改基类的 `forward_batch`：

```python
def forward_batch(self, calib_trials, trial_lengths=None, side_features=None):
    ...
    state = self.reset_stream(batch_size, num_neurons, ...)
    if side_features is not None:
        state["side_features"] = side_features      # [B, N, k]
    for trial_idx in range(...):
        state = self.push_trial(...)
    return self.finalize_identity(state)
```

**不要**改 `reset_stream` / `push_trial` / `finalize_identity` 的抽象签名——
那会波及 25 个以上子类。新编码器在自己的 `finalize_identity` 里读
`state.get("side_features")`。

### B.2 新编码器

在 `streaming_encoders.py` 新增 `SideFeatureEarlyPoolEncoder`，
`variant = "B3S"`，并注册进 `build_encoder`（新增 `side_dim: int = 0` 参数）。

结构 = B3 + 在 `ψ` 输入端 concat：

```python
def finalize_identity(self, state):
    mean_feat = state["sum_feat"] / state["trial_count"]        # [B,N,D]
    side = state.get("side_features")
    if self.side_dim > 0:
        if side is None:
            raise ValueError("B3S requires side_features")
        mean_feat = torch.cat([mean_feat, side], dim=-1)         # [B,N,D+k]
    return self.post_pool(mean_feat)
```

`post_pool` 第一层 `Linear` 输入维度改为 `hidden_dim + side_dim`。

**零初始化要求（必做）**：`post_pool[0].weight` 中对应 side 列的部分必须
**零初始化**，使得训练第 0 步 `B3S(side_dim=k)` 与 `B3` 在数值上完全相同。
这与仓库既有的 "B3-preserving zero-init" 约定一致，且让 `F_x vs F0` 的
比较从同一起点出发。

`_mac_per_session` 要相应加上 `num_neurons * side_dim * hidden_dim`。

### B.3 侧信息计算与缓存

新文件 `sua_exploration/mc_maze/unit_side_features.py`。

**数据来源**（已实测，见章程 §3）：`/units/waveforms` 是 `(总样本数, 1)`
int16 的**逐 spike 波形**，每条 48 samples，通过 `waveforms_index`（长度 =
总 spike 数）与 `waveforms_index_index`（长度 = unit 数）双层索引。
**没有** `waveform_mean`，必须自己算。

**泄漏约束（关键）**：per-unit template 只能用 **calibration pool 内**
（即 `trials[0:pool_size]`，`pool_size=50`）时间范围内的 spikes 计算。
必须写一条断言：参与平均的 spike 时间全部 `<= pool 结束时间`。

**特征定义**（须精确实现，注意胞外 spike 通常为负向）：

| 名称 | 定义 | 组 |
|---|---|---|
| `p2p` | `max(template) - min(template)` | F1 |
| `noise_std` | 逐 spike 波形减去 template 后残差的 std | F1 |
| `snr` | `p2p / noise_std` | F1 |
| `pt_width` | `abs(argmax(template) - argmin(template))`，单位 sample | F2 |
| `pt_ratio` | `abs(min(template)) / abs(max(template))` | F2 |
| `repol_slope` | 从 trough（argmin）起固定窗口内的线性拟合斜率 | F2 |

退化情况必须显式处理并记录：spike 数为 0 或 1 的 unit、`noise_std == 0`、
`max(template) == 0`。用固定填充值（如 0）并在产物里记录受影响的 unit 数，
不要静默产生 `inf`/`nan`。

**缓存**：照抄 datamodule 既有模式——`_source_fingerprint` + `_cache_key`
+ `_write_npz_atomically` + `_exclusive_cache_lock`，写到
`sua_exploration/cache/`（已存在）。缓存键必须包含特征版本号与 `pool_size`，
否则改了定义会读到旧缓存。

**归一化**：新增 `fit_side_feature_stats(train_files, ...)`，仿照既有的
`fit_behavior_stats` 与 `_behavior_stats_cache_path`。
**只用 27 个 train sessions 估计 mean/std**，统计量随 checkpoint 保存。
禁止 per-session 重标准化（章程 §6.1）。

**`FS` 对照**：在特征张量准备好之后、送进模型之前，用固定 seed 沿
**unit 维**随机置换。置换 seed 写进 `run_metadata.json`。这是数据侧操作，
不需要新编码器。

### B.4 数据与模块串联

1. `SessionRecord` 增加 `side_features: np.ndarray | None = None`（`[N, k]`）。
2. `Dandi688MultiSessionDataset.__getitem__` 在启用时返回 **5-tuple**：
   `(neural, behavior, calib, session_name, side_features)`。
3. `streaming_calibration_module.py` 的 `training_step` / `validation_step`
   当前是 `neural, behavior_target, calib, session_name = batch`。
   **必须防御式解包**，因为该模块与 MUA 路径共用：

   ```python
   if len(batch) == 5:
       neural, behavior_target, calib, session_name, side = batch
   else:
       neural, behavior_target, calib, session_name = batch
       side = None
   ```

   然后把 `side` 透传到 `self.student(..., calib_trials=calib, side_features=side)`
   直至 `forward_batch`。
4. 注意 `SessionBatchSampler` 保证同一 batch 内所有样本同 session，因此
   `side_features` 在 batch 维上是重复的常量——这是正确的，不要"优化"掉，
   它对应部署时 calibration 后缓存一次的语义。

### B.5 CLI

`train_variant_dandi688.py`：

- `--variant` 的 `choices` 加入 `"B3S"`；
- 新增 `--side_features {none,f1,f2,fs}`（默认 `none`）；
- `fs` 使用 `f2` 的特征集但施加 unit 维置换；
- `run_metadata.json` 必须记录：特征组名、特征版本号、`side_dim`、
  归一化统计量的 SHA-256、缓存键、置换 seed（若适用）、退化 unit 计数。

### B.6 运行矩阵

固定为 8 个 run，不增不减：

| 特征组 | side_dim | seeds |
|---|---:|---|
| `F0`（= B3 基线，复用现有 checkpoint 亦可） | 0 | 42, 43 |
| `F1` | 3 | 42, 43 |
| `F2` | 6 | 42, 43 |
| `FS` | 6 | 42, 43 |

其余超参数与 `attention_arch_screen_v3` 的 SUA 臂完全一致：
`27/6/6` split、`max_units_exclusive 100`、`task_only` loss、20 epoch 上限、
patience 5、固定评价协议 `first / n=30 / pool=50`。

若复用现有 `attention_arch_screen_v3_b3_dandi688_co_s{42,43}` 作为 `F0`，
必须在聚合器中校验其训练超参数与本轮 `F1/F2/FS` 完全一致，否则重跑。

### B.7 聚合与门槛

新脚本 `sua_exploration/scripts/aggregate_side_feature_ablation.py`，
仿照 `aggregate_attention_architecture_screen.py`：

- 汇总前校验全部 8 个工件的 variant / seed / checkpoint SHA / split /
  session 集合 / 固定协议 / `no_test_files_evaluated=true` 一致，任一不符
  **拒绝汇总**；
- 输出 `results/side_feature_ablation_v1/aggregate.json`；
- 门槛**照抄章程 §8**，写死在代码里：

  ```
  变体可用:  mean R² > 0
  内容有效:  (F_x − F0) 与 (F_x − FS) 的 mean delta 均 ≥ +0.005
             且 minimum ≥ −0.03
             且 6 个 session 中至少 4 个为正
  ```

- `gates` 字段的结构与 `attention_arch_screen_v3/aggregate.json` 保持一致，
  便于横向比较。

### B.8 测试（必做）

新增 `streaming_calibration_exp/tests/test_side_feature_encoder.py`：

- [ ] `side_dim=0` 时 `B3S` 与 `B3` 输出逐位相同；
- [ ] 零初始化下，`side_dim=k>0` 的 `B3S` 在**训练前**与 `B3` 输出逐位相同；
- [ ] 输出 shape 恒为 `[B,N,W]`，与 `k` 无关；
- [ ] **permutation 不变性**：同步置换 calib 的 unit 维与 `side_features`
      的 unit 维，输出应相应置换（这是 set-based 语义的核心，必须守住）；
- [ ] 侧信息缓存对同一输入两次调用返回一致结果；
- [ ] 泄漏断言：传入超出 pool 时间范围的 spike 会触发 raise。

---

## 交付物清单

提交时请给出：

1. 改动的文件列表与 diff 摘要；
2. Task A 四项验收的**实际命令与输出**（不要只说"已完成"）；
3. `pytest` 在新增与既有相关测试上的完整输出；
4. Task B 的 `aggregate.json` 与按章程 §9 决策树给出的判读；
5. 对 [`CURRENT_RESULTS.md`](CURRENT_RESULTS.md) 与
   [`../ROADMAP.md`](../ROADMAP.md) 的更新；
6. **明确列出你没做的事和原因。**

## 如果你认为章程有错

章程是冻结的，但不是不可质疑的。如果你在实现中发现章程里的特征定义、
门槛或隔离规则存在技术错误，**停下来写明问题并等待确认**，不要一边实现
一边偷偷改。事后调整过的门槛没有证据价值。

## 已知会绊住你的地方

1. `sua_exploration/mc_maze/multisession_datamodule.py` 已有
   `electrode_ids_from_units` 和 `pool_spikes_by_electrode`（为 pseudo-MUA 写的）。
   F3 阶段可以直接复用，但本轮不要动它们。
2. 训练日志的 model summary 会列出 `test_heldin_*` / `test_heldout_*` metric
   对象——那是 Lightning 预注册的 metric，**不代表执行了 test**。
   训练入口只调 `dm.setup("fit")` 与 `trainer.fit(...)`。
3. `val_heldout/*` 是历史 metric 名，实际对应的仍是那 6 个 validation
   sessions，不是 test sessions。
4. 波形数组很大（单 session 2,300–3,000 万个 int16 样本）。**必须分块读取
   并缓存**，不要整体载入内存，更不要每个 epoch 重算。
