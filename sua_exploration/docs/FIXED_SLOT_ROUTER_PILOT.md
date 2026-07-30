# NeuronID Fixed-Slot Router：24 小时可行性 Pilot

**状态：开发集 pilot；不消费 formal held-out test sessions。**

## 问题与假设

当前 DANDI 000688 SUA session 的 sorted-unit 数为 38--91，且 unit identity
仅在各自 session 内有效。现有 SPINT decoder 可以接受变长 unit token 集合，但
其 `fc_in` 为每个 unit token 执行 `50 -> 512 -> 512` read-in。部署时这使
latency、SRAM 与计算图仍随 session unit 数变化。

本 pilot 检验一个受限、可证伪的工程假设：在 calibration 结束后，将任意
`N` 个 session-local unit 通过 NeuronID-conditioned routing 压成固定 `K` 个
virtual neural slot，是否仍可保留有用的跨 session SUA decoding。

它**不**检验 B15 cross-neuron self-attention，也不声称跨日期跟踪了相同的生物
neuron。当前 B15 的公平 controls 和 M2 internal-LOSO 已不支持该 attention
机制主张，因此 pilot 使用较简单的 B3 per-unit NeuronID 作为 router 底座。

## 固定架构

对新 session 的 calibration spikes，B3 encoder 产生每个 unit 的 identity：

`C[M,T,N] -> E[N,W]`。

固定 slot router 从 `E` 导出 session-specific assignment：

`A[N,K] = softmax_k(<W_key E_i, q_k> / temperature)`。

对每个在线 neural window，router 用同一 `A` 投影 live spikes：

`U[K,W] = normalize(A^T X[N,W])`。

并从 pooled identity 与 slot mass 生成固定 FiLM state：

`Z[K,W] = gamma[K,W] * U[K,W] + beta[K,W]`。

decoder 只接收 `Z[K,W]`；`K` 在所有 session 中固定。`A`、`gamma`、`beta`
均只依赖 calibration spikes，部署时可以一次计算并缓存。实现提供了显式的
`derive_fixed_slot_state()` 与 `decode_with_fixed_slot_state()` 接口，以便硬件
路径不必在每个 window 重算 NeuronID 或 routing。若 NeuronID encoder 还产生
per-unit reliability gate，则 gate 同样作为 calibration state 缓存，并在 slot
投影之前应用；因此 cached decode 与普通 forward 保持等价。

导出的 state 可以以 batch size `1` 保存为单一 session artifact，并在该 session 的
多个在线 window 被组成一个 batch 时沿 batch 维广播；因此 window batching 不会重新引入
随 `N` 变化的 decoder shape，也不会重复运行 calibration encoder。

## 固定槽位与 token pruning 的取舍

本轮先验证 **fixed-slot pooling**，而不是直接部署 top-K pruning，原因是它从 router
之后立刻给出完全固定的 `[K,W]` decoder interface；无论 session 有 38、64 或 91 个
SUA units，decoder 的 token 数、activation shape、`fc_in` MAC 与后续 transformer
shape 都固定。slot token 的语义是 session-local virtual channels，不能被解释为跨 session
追踪到的相同 biological neuron。

另一个可行路线是 **NeuronID-scored token pruning**：calibration 从 identity 产生每 unit
score，选择恰好 `K` 个 units，将其原始 `[K,W]` live spikes（可加 identity-conditioned
FiLM）送入 decoder。为保证硬件 shape 固定，必须规定：

- `N >= K`：stable tie-break 的 exact top-K；排序键只能来自 calibration，之后固定；
- `N < K`：以全零 token 与 explicit mask 补到 K，decoder 必须在训练时见过这种 padding；
- score、chosen indices、mask 都在 calibration 后缓存；在线期不执行 NeuronID encoder；
- pruning control 必须匹配 token budget：random-K、activity-K（仅 calibration spikes）、
  learned top-K 和 fixed-slot pooling 使用相同 K、同一训练 schedule 和同一 validation rule。

这两类操作不可混称：pooling 保存所有 N 个 units 的加权信息但混合 token；pruning 保留
selected units 的原始 token 但丢弃其余 units。当前 `soft` router 是 pooling；其 `top1`
模式是每 unit 对单 slot 的 hard assignment，**不是** top-K unit pruning。若 K=32 pooling
通过，本项目的下一条公平分支是同 K 的 cached top-K scorer 与 random/activity controls，
而不是用变长保留数量的 pruning 规避固定-shape 约束。

| 路线 | calibration 后 decoder 输入 | 对 N 的在线依赖 | 主要风险 | 本轮状态 |
|---|---|---|---|---|
| Fixed-slot pooling | `[K,W]` virtual slots | 无 | assignment 近均匀导致信息混合 | 正在验证 |
| Exact top-K pruning | `[K,W]` selected/padded units | 无 | unit 丢弃、N<K padding | `K=32` 后的公平 control |
| Variable-token pruning | `[K'(N),W]` | 仍有 | 无法固定编译图/SRAM | 不作为部署终点 |

## 数据隔离

- 数据：DANDI 000688、sub-C、CO sorted SUA；只保留旧 unit-count regime
  (`N < 100`)；
- split：固定 chronological `27 train / 6 validation / 6 test`；
- 本 pilot 训练只读取 train sessions，模型选择和所有 R² 只读取 6 个 validation
  sessions；
- formal-test session 的 neural、behavior 与 trial data 均不加载；
- 当前 `sub-C/CO/27-6-6` formal-test scope 已有先前 B16 产生的 receipt
  `sua_exploration/results/p3_formal_test_816cdd8bf9f26abd1a3e6251e5fbf8537eb6c6cb4de1e8f3312980ddbf478379_receipt.json`。
  因此本 fixed-slot pilot 无论结果如何都不得创建、覆盖或重跑该 scope 的 formal test；
  它的全部证据严格限定为 validation development evidence；
- 训练完成后只允许使用 validation 的前 50 个 rewarded trials 作 calibration pool，
  用 chronologic `first` 选择 30 个 trials，并只评价 pool 后 trials；
- evaluation 中 encoder、router、decoder 全部冻结，不使用 validation behavior
  labels 更新任何参数。
- 训练日志的 model summary 可能列出 `test_heldin_*` / `test_heldout_*` metric objects；
  它们是 Lightning module 的预注册 metric，并不代表执行了 test。该 pilot 的训练入口
  只调用 `dm.setup("fit")` 与 `trainer.fit(...)`：前者仅构造 train/validation datasets，
  后者不调用 `test_dataloader()` 或 `test_step()`；split 初始化至多读取 test NWB 的
  unit-table row count 以固定 `N < 100` regime，不读取 test spike times、behavior 或 trials。
- 除 R² 外，每个 checkpoint 另执行一次 **spike-only routing diagnostic**：它只读取
  validation session 的 SUA spike times、rewarded-trial order 与 target metadata，以相同
  `first/n=30/pool=50` 规则导出 `A[N,K]`；不读取 cursor behavior、不运行 decoder、
  不计算 R²、不更新权重，也不访问 test sessions。

## 24 小时运行矩阵

主实验先使用 soft routing 与 FiLM fusion：

| Run family | Encoder | Slots | Seeds | 目的 |
|---|---|---:|---|---|
| `fsr_k32_soft` | B3 | 32 | 42, 43 | 主可行性点；固定 decoder interface |
| `fsr_k16_soft` | B3 | 16 | 42, 43 | 约 4x read-in token 压缩点 |

已有 B3/B15P development artifacts 仅作外部 reference，不重新选择 formal-test
protocol。`K=16` 与 `K=32` 都在本轮双 GPU batch 中运行；只有 `K=32` 可行，
才用剩余时间增加 `top1` hard-routing 与量化扩展。

## 预先写明的 pilot gates

这些 gate 只决定是否值得继续该固定接口路线，不是统计显著性或最终性能主张。

1. **接口 gate**：每个 batch 的 decoder 输入必须恒为 `[B,K,W]`，并且对
   calibration/neural 同步 unit permutation 的行为预测不变；导出的 calibration
   state 必须与 end-to-end forward 完全一致。
2. **accuracy feasibility gate (`K=32`)**：两个 seed 的 validation-only、
   fixed `first/n=30/pool=50` 平均 R² 都为正，且 seed 平均结果不低于当前
   B15P reference (`0.3599`) 超过 `0.03 R²`。
3. **compression gate**：只计算 decoder `fc_in` 时，固定 token 数将 dominant
   read-in token count 从 session variable `N` 降到 `K`；对典型 `N=64`，
   `K=32` 至少减少约 2x，`K=16` 至少减少约 4x。
4. **deployment gate**：使用同一 cached calibration state 的 decode 必须与
   normal forward 相同；后续 window 不调用 NeuronID encoder 或 slot-routing
   parameter generation。
5. **mechanism observability gate**：每个 run 必须记录每 unit route 的归一化 entropy、
   最大 assignment 概率、slot mass 的变异系数和有效 slot 数。若 entropy 接近 `1.0`
   且最大概率接近 `1/K`，该 run 只能解释为接近 uniform mean-pooling，不能据此否定
   fixed-slot interface 本身；应触发预注册的低温 follow-up。

即使全部通过，结论也仅为“fixed-slot NeuronID interface 在当前 SUA development
regime 中值得扩展”。它不证明 learned routing 优于 random/hash/top-k、不能证明
SUA/MUA reuse，也不证明 hard routing 或 INT8 已可部署。

## 当前开发集证据（进行中）

截至 `2026-07-25`，`K=32` 的两个 B3/soft/FiLM seed 已完成；`K=16` 的两个
seed 仍在训练，因此以下是预先固定协议下的**部分 validation-only** 结果，而非
pilot 的最终 aggregate 或 formal test 结果：

| Interface | Seed 42 mean R² | Seed 43 mean R² | Seed mean R² | B15P 差值 | K=32 accuracy gate |
|---|---:|---:|---:|---:|---|
| `K=32`, soft FiLM | 0.1571 | 0.2068 | 0.1820 | -0.1779 | false |

两 seed 的 calibrated R² 均为正，但该 pilot 的预注册 threshold 是
`B15P - 0.03 = 0.3299`；当前 `K=32` seed mean 低 `0.1479`。因此现有证据支持
“32-slot 接口仍能进行正值跨 session decoding”，但不支持“在这个训练方案下几乎
无损地压缩到 32 slots”。这是 rate--distortion 的开发集信号，不能归因于 attention
或 routing 本身，尚需等待 `K=16`、routing diagnostic 和可能的低温 follow-up。

同时，两个已完成 `K=32` checkpoint 各在 6 个 validation sessions 上做了
spike-only cached-path verification：每 session 从变长 `[8,50,N]`（这里
`N=38…65`）投影到固定 `[8,32,50]` decoder 输入。两个 session-local
calibration state 的 normal-forward 与 batch-1 cached-state broadcast 在 12 个
session-seed pairs 全部匹配；最大 FP32 output difference 为 `2.62e-6`
（verification tolerance `1e-5`）。该验证读取 spikes 与 rewarded-trial metadata，
不读取 behavior labels、不更新权重、也不访问 formal test sessions。

## 下一步决策

- `K=32` 通过且 `K=16` 接近：加入 `top1` hard-routing、INT8 state export、
  random/hash 与 Top-K controls；
- `K=32` 通过、`K=16` 失败：固定接口可行，但需要至少 32 个 slots；
- `K=32` 失败：先报告 rate--distortion failure，不把性能差归因于 router 注意力；
  先诊断 routing entropy、decoder retraining、sum/mean normalization 与 fusion；
- 任一结论形成前，不运行 formal held-out test。

### 低温稀疏化 follow-up

初始 `K=32` 的两个 seed 均完成后，使用上面的 spike-only diagnostic 作唯一的
触发判定：若两个 seed 的 session-mean normalized assignment entropy 的平均值
**≥ `0.95`**，运行唯一的机制 follow-up。它保持 `B3/K=32/FiLM`、全部训练超参数、
split 和 validation protocol 不变，仅将 router softmax temperature 从 `1.0` 改为
`0.1`。此阈值在初版最终结果出现前写入
`run_fixed_slot_router_followup.sh`，其 decision JSON 会落盘。该 follow-up 直接检验
“fixed slots 不可行”与“router 因近均匀 pooling 而不可辨识”两种解释；仍然只使用
train/validation，且必须单独汇总，不与初始 pilot 合并选择 formal test。

## 当前实现与验证

- router：`streaming_calibration_exp/src/models/components/streaming_spint.py`；
- training arguments 与 provenance：`sua_exploration/scripts/train_variant_dandi688.py`；
- analytic deployment cost：`sua_exploration/scripts/profile_fixed_slot_router_hardware.py`；
- spike-only routing diagnostic：`sua_exploration/scripts/diagnose_fixed_slot_router.py`；
- cached deployment-path verifier：
  `sua_exploration/scripts/verify_fixed_slot_cached_decode.py`；它在 validation
  spikes 上比较普通 forward 与 batch-1 cached state 的多-window broadcast，不读
  behavior labels 或 test files；
- staged follow-up orchestration：`sua_exploration/scripts/run_fixed_slot_router_followup.sh`；
- machine-readable artifacts 的 Markdown handoff：
  `sua_exploration/scripts/write_fixed_slot_router_report.py`；
- unit tests：`streaming_calibration_exp/tests/test_fixed_slot_router.py`；
- 已验证：fixed output shape、含可选 reliability gate 的 cached calibration-state
  equality、single-session state 对多个 online windows 的 batch broadcast、permutation
  invariance、top-1 assignment validity。
