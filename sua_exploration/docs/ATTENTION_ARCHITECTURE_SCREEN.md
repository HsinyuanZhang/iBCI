# Attention NeuronID：SUA/MUA 架构复用验证章程

**状态：开发集筛选，2026-07-25 启动。** 本文固定 B15 relational
attention 的可证伪假设、数据隔离、比较方法和后续模型路线。它检验的是
**同一网络拓扑可在 SUA 与 MUA 上分别训练和部署**；不检验、也不声称
SUA/MUA checkpoint 可以零样本互换。

## 1. 要回答的主张

要支持“attention NeuronID 拓扑可以复用到 SUA/MUA”，至少需要同时满足：

1. 同一输入/输出协议、相同参数预算量级和相同 streaming-calibration
   流程，能在严格 sorted-SUA 与 MUA 上分别训练出正的跨 session 表现；
2. 完整 B15 的收益不能由额外逐 neuron 容量、value/output 变换或
   LayerNorm 单独解释；
3. 结论能在未参与该结构选择的独立数据上复现，且不消耗已锁定的 formal
   test 来调参。

因此，单独观察到 B15 在 SUA 上比 B3 好，或只比较 B15 与零 identity
control，都不足以支持 attention 机制或 SUA/MUA 复用主张。

## 2. 当前开发筛选（`attention_arch_screen_v3`）

### 固定架构

| Variant | 作用 | 可访问跨 neuron 信息 | 参数数（D=64） |
|---|---|---:|---:|
| B3 | 原始共享逐 neuron encoder | 否 | 18,354 |
| B15P | 参数匹配的逐 neuron residual-MLP + LayerNorm | 否 | 34,802 |
| B15D | 参数匹配的 attention 路径，但屏蔽所有非对角 attention | 否 | 34,802 |
| B15 | 完整 relational self-attention + residual + LayerNorm | 是 | 34,802 |

`B15P` 控制“宽度、非线性、残差、归一化”；`B15D` 控制“attention 模块的
value/output 路径与残差/LN”，只移除 neuron-to-neuron 通信。三者同参数数，
故 `B15 − B15P` 和 `B15 − B15D` 是归因于跨 neuron 信息访问的必要比较，
而非把 B3 当作唯一对照。

### SUA：DANDI 000688、sub-C、CO

- session-disjoint `27/6/6` train/validation/test 划分；本筛选仅使用 train/validation；
- variants `B3/B15P/B15D/B15`，seeds `42/43`；
- 训练上限 20 epochs，patience 5，冻结 teacher network；student decoder 按既有
  cross-session SUA 协议与 encoder 联合训练，四个 variant 完全一致，`task_only` loss；
- 所有变体固定同一前向 calibration 评价：前 50 rewarded trials 为 pool，
  按 chronologic `first` 选择 30 个 calibration trials；
- 输出只报告 validation session 的 calibration 后 R²。没有 protocol sweep、
  没有行为标签反传、没有 formal-test session。
- 训练器为兼容既有 Lightning 模块而返回两条**相同的 validation loader**；日志中的
  `val_heldout/*` 因而只是历史 metric 名称，实际仍对应上述 6 个 validation sessions，
  不会读取 6 个 test sessions。每个后续 run metadata 都记录这一 loader-to-session
  契约；真正 formal-test 数据只可由 `eval_adaptation_dandi688.py` 在配置锁定后读取。

### MUA：FALCON M2 internal LOSO

- 三个独立开发单元：`fold1/seed42`、`fold1/seed43`、`fold2/seed42`；
- B3/B15P/B15D/B15 都必须在当前 source tree 下重跑同一 LOSO cell；历史 B3
  artifact 仅作描述性参考，不得进入 paired delta。四个 variant 使用相同 LOSO、
  decoder freeze、任务和 identity loss；
- `include_heldout_in_fit=false` 且 `include_heldout_in_test=false`。指标名中的
  `test_heldin` 是内部 held-in evaluation，**不是** FALCON external formal test；
- 此阶段不将 M2 internal 分数与 SUA 的绝对 R² 横向比较，只比较同一
  fold/seed/cell 的 paired delta。

## 3. 预先固定的筛选门槛

聚合器 `aggregate_attention_architecture_screen.py` 在全体结果齐全后自动写入
`aggregate.json`。设 \(\Delta_{X-C}=R^2_X-R^2_C\)，门槛在运行前固定如下：

在计算分数前，聚合器会拒绝任何 SUA 工件，只要其 variant、seed、checkpoint 或
training-metadata hash、task、signal view、`27/6/6` split、session 集合、fixed
`first/n=30/pool=50` protocol、无梯度/无标签更新声明，或
`no_test_files_evaluated=true` 契约不一致。所有 8 个工件还必须具有完全相同的
train/validation/test session 划分；这防止文件名正确但来源、split 或评估范围错误的
结果进入 paired delta。

| Gate | 条件 |
|---|---|
| SUA usable | B15 mean R² > 0；B15−B3 mean ≥ 0；6 个 session 均值中的最低 delta ≥ −0.03 |
| MUA usable | B15 mean R² > 0；B15−B3 mean ≥ 0；3 个 paired cells 的最低 delta ≥ −0.03 |
| SUA attention | 对 B15P 与 B15D，mean delta 均 ≥ +0.005、minimum ≥ −0.03，且 6 个 session 中至少 4 个为正 |
| MUA attention | 对 B15P 与 B15D，mean delta 均 ≥ +0.005、minimum ≥ −0.03，且 3 个 cells 中至少 2 个为正 |
| 进入下一阶段 | 上述四个 gate 全部为真 |

这些阈值是开发筛选用的“值得投入 replication”的门槛，不是显著性检验或
最终发表结论。汇总前不因中间结果增删 seed、fold、模型或阈值。

## 4. 筛选后的决策树

| 结果模式 | 解释 | 模型/实验动作 |
|---|---|---|
| 两域均通过 attention gate | 有初步因果证据表明跨 neuron 关系有额外价值 | 锁定 B15，扩大 seeds 与 sessions；随后做 pseudo-MUA 和外部 MUA replication |
| B15 ≈ B15P，或 B15 ≈ B15D | B15 的收益可由逐 neuron 容量、LN 或 value/output 路径解释 | 不再把 self-attention 作为核心机制；采用 B15P 或更小的 B3，优先部署友好路线 |
| SUA 通过、MUA 不通过 | relational 结构可能只利用 sorted-SUA 的 population/reliability 模式 | 结论收窄为 SUA-specific；MUA 采用 B3/B15P，做样本量和通道数分层诊断 |
| MUA 通过、SUA 不通过 | 结构可能捕获 electrode population 相关性，而非 sorted-unit 可靠性 | 结论收窄为 MUA-specific；检查 SUA 稀疏性、unit 质量和 calibration length |
| B3 最好或 B15 显著退化 | 当前 relational 模块不具稳定性收益 | 停止 attention 路线；以 B3 为主线，避免为复杂性付费 |
| gate 接近阈值 | 开发噪声下证据不足 | 不修改模型；先按原样增加预定义 seeds/folds，再判断 |

如果 B15 被保留，部署优化只在机制确认后进行：先测 top-k/sparse attention 与
低秩 QKV 是否保持 paired gain，再测将 `finalize` 放在 host、只下发 identity
的端到端时延。不得先压缩再以压缩模型的成功倒推原始 attention 必要。

## 5. 进入 replication 的最小证据链

通过开发 gate 后，后续按以下顺序进行，并冻结本章程中的 B15 配置和 SUA
calibration protocol：

1. **paired expansion**：增加 SUA session/seed 与 MUA LOSO fold/seed；每个
   单元同时训练四个 variant，保留逐单元 R² 和 paired delta；
2. **pseudo-MUA 桥接**：仅由同一 sorted-SUA recording 的 unit pool 构造
   固定、无标签的 pseudo-MUA channel；配对比较 SUA 与 pseudo-MUA，检验
   结构是否耐受 signal aggregation，而不是把不同数据集的绝对分数相减；
3. **外部 MUA replication**：在独立 MC_RTT/M1 或预先声明的 FALCON external
   数据进行一次冻结配置的 confirmation；
4. **SUA confirmation**：在尚未用于结构或 protocol 选择的 subject/session
   block 完成一次正式 evaluation。此前已消费或处于 receipt 状态的 test
   session 只可描述，不能再被重新选择或重跑；
5. **报告**：同时给出每域绝对 R²、B15−B3、B15−B15P、B15−B15D、失败率、
   calibration latency、参数量和估计的 finalize MAC。负 R² 只作绝对差值，
   不报告“提升倍数”或百分比。

### Pseudo-MUA 的固定构造规则

DANDI 000688 的 sorted unit table 为每个 unit 保存一个 electrode region；样例
CO recording 的 71 个 sorted units 对应 47 个唯一 electrode。桥接实验在
**binning 之后、calibration/decoder 之前**按以下规则构造 channel：

1. 以 NWB `units/electrodes` 的唯一 electrode id 作为 channel id；一个 unit
   必须且只能对应一个 electrode，否则该 session 预先排除并记录原因；
2. 同一 electrode 的所有 sorted-unit spike-count bins 作**求和**，不平均、
   不按行为标签归一化、不用 learned pooling；因此它保留 MUA 风格的总事件数；
3. 每个 session 的 SUA 与 pseudo-MUA 使用同一 chronological split、同一
   calibration pool、同一随机 seed 和四个 architecture variants；
4. 主统计量是每个 session/seed 的同源 paired delta，既报告 variant 的域内
   B15−control，也报告该 delta 在 `SUA → pseudo-MUA` 后的变化；
5. 记录 source session、source unit count、resulting electrode-channel count、
   unit-to-electrode mapping hash 和排除列表。

它不等于原始 threshold crossing：它没有包含未排序事件、噪声或跨 unit 的
spike-sorting 误差。其作用是隔离“从单元到电极聚合”这一个分布改变；真实 MUA
结论仍需要第 3 步的外部 MUA replication。

当主 screen 的全部 gate 通过后，`run_pseudo_mua_attention_pilot.sh` 以同一
四个 variants、两个 seed 和固定 protocol 运行该桥接实验；
`aggregate_pseudo_mua_attention_pilot.py` 仅在父 screen 已通过时汇总，并将
`advance_to_external_mua_replication` 作为外部 MUA replication 的运行门槛。

### 外部 MUA replication 的固定执行方式

当前预注册的外部数据为 FALCON **M1**（与 M2 和 DANDI SUA 不同的数据集）。只有
pseudo-MUA gate 通过后，`run_m1_external_attention_replication.sh` 才运行冻结的
`B3/B15P/B15D/B15` 配置，使用 `fold0/seed42`、`fold0/seed43`、
`fold1/seed42` 三个 matched internal-LOSO cells。它与 M2 阶段保持同样的
decoder freeze、`task_plus_y_plus_E` loss 和参数归因控制，并显式固定
`include_heldout_in_fit=false`、`include_heldout_in_test=false`。

因此，M1 的 `test_heldin` 仍是开发用的内部 held-in 指标，**绝不是** FALCON
external formal test。`aggregate_m1_external_attention_replication.py` 会拒绝任何
缺少工件或违反上述隔离、fold、seed、variant、loss 或冻结 decoder 契约的结果；只有
三条 paired deltas 完整且预设门槛通过，才进入独立 formal-confirmation 的规划，
不会自动消费 formal-test receipt。

## 6. 不能由本实验声称的内容

- B15 的 gradient-free calibration 与全零 identity control 的差异，证明的是
  calibration 必要性，不是 attention 必要性；
- training/validation 筛选结果不是 formal held-out test；
- separate training 成功不等于 checkpoint zero-shot transfer；
- 同一结构在两域都可用，不等于两种信号的神经生理机制相同。

## 7. 可复现入口

```bash
bash sua_exploration/scripts/run_attention_architecture_screen.sh \
  --launch --wait --screen-id attention_arch_screen_v3

/home/xinyuan/miniconda3/envs/spint/bin/python \
  sua_exploration/scripts/aggregate_attention_architecture_screen.py \
  --screen-id attention_arch_screen_v3
```

训练运行在命名 `tmux` 会话 `spint_attention_arch_v3` 中；逐任务日志位于
`sua_exploration/results/attention_arch_screen_v3/logs/`。通过主 screen 后的
pseudo-MUA 与 M1 阶段由 watcher 按 aggregate gate 顺序触发；它们不会触碰 formal
held-out sessions。
