# SPINT Streaming Calibration 实验执行指南（AI Agent）

> 创建时间：2026-07-10 16:22 HKT  
> 适用项目：`/home/xinyuan/Work_host/SPINT`  
> 主要代码库：`SPINT-main/`  
> 目标任务：M2 gradient-free few-shot calibration encoder 的流式化、压缩与硬件友好结构搜索

## 0. 给执行 Agent 的任务定义

你需要在保持 SPINT 无标签、无测试时梯度、神经元置换等变/不变性质的前提下，完成 calibration 路径的实验性重构和结构搜索。

主要优化目标按优先级排序：

1. 最小化 calibration 期间的峰值 live SRAM。
2. 最小化 IDEncoder 的 MAC/session。
3. 最小化 IDEncoder 权重存储。
4. 尽量保持 matched FP32 baseline 的 held-out session R2。
5. 去除完整 support-set 缓存、固定长度 cubic interpolation、通用除法和测试时反向传播依赖。

不要一开始修改 online decoder。第一轮实验必须固定 decoder，只替换 calibration encoder，避免混淆精度来源。

## 1. 非目标

- 不做片上训练器、optimizer 或 backward RTL。
- 不在第一轮缩小 cross-attention decoder。
- 不用 held-out session 标签选择结构、阈值或量化参数。
- 不把 Plan-B 固定 96 通道 TCN 当作原 SPINT 的等价实现。
- 不在没有 PDK、SRAM macro、频率和 activity trace 时声称具体 mW 或 mm2。
- 不覆盖现有 checkpoint、CSV、日志或分析文档。

## 2. 源码事实与权威入口

执行前阅读并引用以下文件：

| 内容 | 文件 |
|---|---|
| SPINT 模型、IDEncoder、cross-attention | `SPINT-main/src/models/components/spint.py` |
| M2 Lightning wrapper、loss、最后一帧选择 | `SPINT-main/src/models/falcon_module.py` |
| calibration trial 构造与采样 | `SPINT-main/src/data/falcon_datamodule.py` |
| M2 模型超参 | `SPINT-main/configs/model/falcon_m2.yaml` |
| M2 数据超参 | `SPINT-main/configs/data/falcon_m2.yaml` |
| packaged streaming decoder | `SPINT-main/third_party/falcon_challenge/spint_decoder.py` |
| 已有 ASIC/成本分析 | `SPINT_deployment_cost_analysis.md` |
| 详细数据流分析 | `SPINT_fewshot_ASIC_dataflow_analysis.md` |

源码优先级高于已有 Markdown。若文档数字与当前源码不一致，以源码重新计算并记录差异。

## 3. 当前 M2 Baseline

固定基准维度：

```text
M = 33 calibration trials
T = 100 normalized samples/trial
N = 96 neurons
W = 50 online history bins
H = 512 ID/decoder hidden dimension
C = 2 behavior outputs
heads = 64
FFN = 2048
bin = 20 ms
```

当前 IDEncoder：

```text
phi, per trial and neuron:
  100 -> 512 -> 512 -> 512

trial aggregation:
  mean over M

psi, per neuron:
  512 -> 512 -> 512 -> 50
```

精确 baseline 数字：

```text
IDEncoder parameters = 1,127,986
IDEncoder MAC(M=33) = 1,875,935,232
raw calibration FP32 = 33*100*96*4 = 1,267,200 B
identity E FP32 = 96*50*4 = 19,200 B
```

MAC 公式：

```text
MAC_ID(M,D) = M*N*(T*D + 2*D^2) + N*(2*D^2 + D*W)
```

## 4. 必须保持的算法不变量

所有候选结构必须通过以下检查：

1. Calibration support 不使用 behavior label。
2. 测试时没有 optimizer、backward 或权重更新。
3. 同一 neuron encoder 权重跨 neuron 共享。
4. 改变 calibration trial 顺序不改变数学输出。
5. 同步置换 query neurons 与 calibration neurons 时，`E` 同步置换，最终 behavior 输出不变。
6. 输出 identity shape 固定为 `[B,N,W]`，第一轮直接接原始 frozen decoder。
7. Streaming state 不保留已经消费的完整历史 trials。

## 5. 统一 Streaming API

为所有 encoder 实现同一逻辑接口。命名可按仓库风格调整，但语义不得改变：

```python
state = encoder.reset_stream(batch_size, num_neurons, device, dtype)
state = encoder.push_trial(state, trial, trial_length=None)
E = encoder.finalize_identity(state)
```

真正 bin-level streaming 的候选额外实现：

```python
state = encoder.start_trial(state)
state = encoder.push_sample(state, neural_bin)
state = encoder.end_trial(state)
E = encoder.finalize_identity(state)
```

要求：

- `push_trial` 后不得在 state 中保留该 trial 的引用。
- `reset_stream` 必须清除 support sum、trial count、trial-local state 和 `E_valid`。
- `finalize_identity` 不得隐式重新读取完整 calibration dataset。
- 支持 `M=1..33`，不得硬编码只支持 33。
- trial count 为 0 时必须显式报错。

## 6. 实验矩阵

### B0：原始 Batch Baseline

保持现有 `SpintModel.forward()`，一次传入 `[B,M,T,N]`。

用途：

- 生成 matched FP32 baseline。
- 生成 teacher identity `E_teacher`。
- 生成 teacher prediction `y_teacher`。
- 作为所有 Delta R2、误差和成本比较的分母。

### B1：Exact Trial Streaming

保持原始权重与网络结构：

```text
SUM_PHI[N,H] = 0
for each trial m:
    phi_m = fc_id_in(trial_m)
    SUM_PHI += phi_m
E = fc_id_out(SUM_PHI / M)
```

预期：

```text
parameters = 1,127,986
MAC = 1,875.94M at M=33
trial buffer = 100*96 values
support state = 96*512 values
```

B1 是精确重排，不允许重新训练。

### B2：LatePool Bottleneck

保持原始 3+3 affine 拓扑，只把 ID hidden dimension 与 decoder hidden dimension 解耦。

扫：

```text
D_id in {256, 128, 64}
```

预估：

| D_id | Parameters | MAC/session, M=33 |
|---:|---:|---:|
| 256 | 301,874 | 510.15M |
| 128 | 85,426 | 148.12M |
| 64 | 26,354 | 47.32M |

输出仍为 `W=50`，接原始 frozen decoder。

### B3：EarlyPool MLP

结构：

```text
per trial/neuron:
  T -> D -> ReLU

support aggregation:
  mean over M

post-pool:
  D -> D -> D -> W
```

扫：

```text
D in {128, 64, 32}
```

已知估算：

| D | Parameters | MAC/session, M=33 |
|---:|---:|---:|
| 128 | 52,402 | 44.31M |
| 64 | 18,034 | 21.37M |

B3 仍需要保存一条 `T*N` trial，但显著缩小 support accumulator。

### B4：Streaming Statistics Lower Bound

每个 neuron、每条 trial 在线维护：

```text
sum(x)
sum(x^2)
max(x)
last(x)
sample_count
```

trial feature：

```text
[mean, variance/std, max, last]
 -> Linear 4->64
 -> ReLU
 -> accumulate over trials
 -> mean over M
 -> 64->64->50
```

不要一开始实现 sqrt。先比较：

```text
[mean, second_moment, max, last]
```

只有确认 std 明显更优时才加入定点 rsqrt/sqrt 复杂度。

预估：

```text
parameters about 7.7K
MAC/session about 1.8M
no trial buffer
```

### B5：Streaming EMA Bank

每个 neuron 维护 R 个 EMA state：

```text
s_r[t] = s_r[t-1] + alpha_r * (x[t] - s_r[t-1])
```

每条 trial 输出：

```text
[EMA final values, EMA trial means]  # 2R features
 -> Linear 2R->64
 -> ReLU
 -> mean over trials
 -> 64->64->50
```

扫：

```text
R in {4, 8}
D = 64
```

第一版 alpha 使用 power-of-two：

```text
R=4: {1/2, 1/4, 1/8, 1/16}
R=8: {1/2, 1/4, ..., 1/256}
```

同时保存一个可学习 alpha 的 FP32 accuracy upper bound，但硬件主候选必须报告 power-of-two 结果。

预估：

| R | Parameters | MAC/session, M=33 |
|---:|---:|---:|
| 4 | 7,990 | 3.59M |
| 8 | 8,506 | 6.48M |

### B6：Streaming FIR Bank

共享 R 个 causal temporal FIR。FIR 权重跨 neuron 共享，不能为每个 neuron 保存独立 filter weights。

结构：

```text
input history K-1
 -> R shared causal FIR outputs
 -> accumulate final and trial mean for each FIR
 -> Linear 2R->64
 -> mean over trials
 -> 64->64->50
```

扫：

```text
R in {4, 8}
K in {5, 9}
D = 64
```

预估参考：

```text
R=8, K=9: about 26.75M MAC/session at 33*100 samples
```

原始 trial 变长时 MAC 必须按实际 valid sample count 统计，不得把 padding 当有效计算。

## 7. 两阶段训练，隔离 Encoder 能力

### 阶段 A：Frozen Decoder

固定一个 baseline checkpoint 和原 decoder。新 encoder 输出 `E_student[N,W]`，损失：

```text
L = L_task
  + lambda_y * MSE(y_student, y_teacher)
  + lambda_E * normalized_MSE(E_student, E_teacher)
```

要求记录：

```text
lambda_y
lambda_E
teacher checkpoint path
teacher checkpoint SHA256
decoder frozen parameter count
```

建议默认起点：

```text
lambda_y = 1.0
lambda_E = 0.1
```

必须做三个 loss ablation：

```text
task only
task + prediction distillation
task + prediction + identity distillation
```

### 阶段 B：End-to-End Fine-tune

只对阶段 A 的 Pareto 候选执行。允许 decoder 以低学习率联合微调，但 decoder shape 不变。

记录 frozen 和 fine-tuned 两套结果，不能只报告 fine-tuned 结果。

## 8. 数据切分与防泄漏规则

主要结构选择使用 held-in session 级交叉验证：

```text
7 held-in sessions
 -> leave-one-session-out 或固定 5 train / 2 validation rotation
```

约束：

- 结构、D/R/K、early-stop threshold、quant scale 策略只看 held-in validation。
- 6 个 held-out sessions 只能在候选锁定后做最终比较。
- Calibration support 的 behavior 不得进入 encoder、停止器或量化校准。
- 允许使用 query behavior 计算离线评测 R2，但不能驱动测试期 adaptation。
- 最终报告必须说明 held-out 是否曾用于开发；若已经使用，不能将其描述为未触碰 test。

随机种子至少：

```text
42, 43, 44
```

M2 session 方差大，禁止只报告单 seed 或总平均。

## 9. Progressive Calibration 与无标签提前停止

所有 streaming encoder 在以下 trial count 生成临时 E：

```text
M_checkpoint = {1, 2, 4, 8, 16, 33}
```

计算三个无标签稳定性指标：

```text
delta_E(m) = ||E_m - E_prev||_2 / (||E_prev||_2 + eps)

delta_y(m) = mean_t ||y_m(t) - y_prev(t)||_2
             on a fixed unlabeled rolling neural buffer

feature_sem(m) = aggregate standard error of support features
```

停止规则阈值只能在 held-in validation 上拟合。

最终报告：

```text
average trials used
median trials used
worst-case trials used
R2 after stopping
R2 loss versus fixed M=33
calibration MAC saved
```

不要每条 trial 都运行昂贵 `psi`；默认只在 `M_checkpoint` 运行，以降低 progressive evaluation 的固定开销。

## 10. 量化实验

只有 FP32 候选通过精度 gate 后才进入量化。

量化顺序：

1. FP32 streaming reference。
2. W8A8，INT32 dot-product accumulator，INT32 support sum。
3. W8A8，INT32 dot-product accumulator，INT16 support sum。
4. 对最佳候选做 QAT。
5. W4 仅作为探索，不得替代 W8 基准。

必须单独处理：

- EMA/FIR state scale；
- support sum overflow；
- `1/M` reciprocal；
- `E` 与 online `X` 相加时的共同 scale；
- cubic interpolation 产生的负值和 overshoot；
- saturation count 与最大 accumulator magnitude。

`1/M` 使用 reciprocal multiplier + shift 或 LUT，不使用通用 divider。

## 11. 必须实现的测试

### 功能测试

- B0 batch 与 B1 streaming FP32 等价。
- `M=1`、`M=33`、非法 `M=0`。
- 交换 trial 顺序。
- 同步置换 neural/calibration neuron 轴。
- 不同 N 的合法前向，若候选宣称 variable-N。
- session reset 后 state 全零且无前 session 残留。
- `push_trial` 后 state 不引用历史 trial tensor。
- padded variable-length trial 不参与统计和 MAC 计数。

### 数值测试

- INT8 accumulator 理论界与实测最大值。
- INT16 support sum 饱和率必须为 0，或显式报告饱和样本。
- streaming accumulation 顺序变化造成的量化误差。
- power-of-two EMA 与 learnable EMA 的误差和 R2 差异。

### 数据泄漏测试

- assert calibration encoder 输入不包含 behavior。
- assert early-stop 函数签名不接受 target behavior。
- assert held-out session 不参与超参搜索表的排名。

## 12. 统一指标与 CSV Schema

每个 run 至少输出：

```text
run_id
variant
seed
split
session
M
R2_variance_weighted
R2_delta_vs_matched_baseline
identity_mse
prediction_distill_mse
parameter_count
weight_bytes
MAC_per_trial
MAC_per_session
peak_live_state_bytes
trial_buffer_bytes
support_state_bytes
requires_cubic_interpolation
requires_general_multiplier
requires_divider
quant_format
saturation_count
```

结果目录建议：

```text
SPINT-main/outputs/streaming_calibration/<run_id>/
  resolved_config.yaml
  environment.txt
  git_state.txt
  train.log
  metrics_summary.csv
  metrics_per_session.csv
  progressive_identity.csv
  hardware_cost.json
  checkpoints/
```

记录 git revision 和 dirty status，但未经用户要求不要自动 commit。

## 13. 成本计算口径

所有成本必须标记为以下之一：

```text
exact_source_formula
measured_runtime
cycle_model_estimate
synthesis_result
assumption
```

峰值 SRAM 必须通过 tensor/state lifetime 统计，而不是把所有张量容量相加。

至少分开报告：

```text
weight storage
trial-local state
support-persistent state
identity E
DMA/FIFO，若有
```

MAC 不包含 add/ReLU 时要明确说明；EMA shift-add 也要单独计数。

## 14. 精度 Gate 与 Pareto 选择

相对 matched baseline 定义两个区域：

| 等级 | 精度条件 | 硬件目标 |
|---|---|---|
| 保真候选 | mean Delta R2 >= -0.01；worst-session >= -0.03 | peak state <64 KiB；MAC <150M |
| 激进候选 | mean Delta R2 >= -0.03；无灾难性 session collapse | peak state <32 KiB；MAC <30M |

不要只用阈值淘汰。最终同时画：

```text
R2 vs log10(MAC/session)
R2 vs peak_live_state_bytes
R2 vs weight_bytes
R2 vs average calibration trials after stopping
```

最终选择两个候选：

1. 一个 high-fidelity Pareto point。
2. 一个 minimum-hardware Pareto point。

若同精度下成本接近，优先级为：

```text
no trial buffer
 > no cubic interpolation
 > no general multiplier
 > fewer state variables
 > fewer MAC
```

## 15. 阶段性 Gate

### Gate 0：Baseline 可复现

要求：

- checkpoint、config、数据 split、seed 固定并记录；
- matched baseline R2 可重复；
- 不强制等于论文 0.26，但后续必须比较同一 baseline。

### Gate 1：Exact Streaming 正确

要求：

```text
max_abs(E_batch - E_stream) < 1e-6，FP32
max_abs(y_batch - y_stream) < 1e-6，FP32
```

若失败，不得进入结构搜索。

### Gate 2：Bottleneck/Persistent Buffer Sweep

比较 B2/B3，保留所有非支配候选。至少一个候选应达到：

```text
Delta R2 >= -0.01
ID MAC reduction >= 10x
```

### Gate 3：No-Trial-Buffer Sweep

比较 B4/B5/B6。至少报告 statistics、EMA、FIR 各一个完整结果，不能只报告最佳结构。

### Gate 4：Quantization

至少一个 W8A8 候选满足：

```text
additional Delta R2 from quantization >= -0.01
zero accumulator overflow
```

### Gate 5：Progressive Stop

停止器不得使用 label，并报告相对固定 M=33 的精度/计算折衷。

## 16. 推荐执行顺序

按以下依赖执行，不要直接全网格长跑：

```text
B0 baseline
 -> B1 exact streaming
 -> B2/B3 one-seed screening
 -> B4/B5/B6 one-seed screening
 -> Pareto candidates, 3 seeds
 -> frozen vs fine-tuned decoder comparison
 -> W8A8/PTQ/QAT
 -> progressive stopping
 -> final held-out evaluation
```

Screening 阶段允许减少 epochs，但必须对所有候选使用相同预算。最终候选必须用完整训练预算重新运行。

## 17. Agent 每次运行后的记录要求

每完成一个实验，立即记录：

```text
改了哪些文件
使用的完整 config
训练是否完成
最佳/最终 checkpoint
per-session 指标
成本计算来源
是否通过当前 Gate
失败原因
下一次运行与本次有何单一变量差异
```

禁止同时改变 encoder、decoder、loss、数据预处理和量化格式后将差异归因于单一模块。

## 18. 常见失败与诊断

### B1 不等价

依次检查：

1. 是否在 `model.eval()` 下比较。
2. calibration trial 是否相同且顺序相同。
3. mean 是否在完整 `fc_id_in` 之后。
4. 是否错误地对原始 trial 先求均值。
5. dropout/dynamic dropout 是否被禁用。
6. batch broadcast 和 neuron axis 是否正确。
7. 浮点累加顺序是否造成仅最低位差异。

### B4/B5/B6 精度崩溃

依次比较：

1. teacher identity matching 是否失败。
2. 是否丢失 trial length/count normalization。
3. raw variable-length 输入与 cubic-normalized teacher 的分布差异。
4. EMA 时间常数是否覆盖有效时间尺度。
5. FIR 是否真正 causal，padding 是否进入 pooling。
6. decoder frozen 是否过于限制，允许低 LR fine-tune 后是否恢复。

### 平均 R2 尚可但个别 session 崩溃

必须保留该结果并报告 worst-session。检查 activation scale、低放电 neuron、calibration trial 长度和 support feature 饱和，不得只用 mean 掩盖。

## 19. 最终交付物

执行 Agent 最终应提交：

1. 可复现的 B0/B1 matched baseline 与等价性测试。
2. B2-B6 的统一配置、代码和 per-session CSV。
3. FP32 与 W8A8 的 Pareto 图。
4. Progressive calibration/early-stop 曲线。
5. 一份硬件状态表，明确 peak live bytes、MAC、weight bytes 和不支持的算子。
6. 一个 high-fidelity 推荐结构。
7. 一个 minimum-hardware 推荐结构。
8. 对每个推荐结构给出仍未验证的风险，不得把估算写成实测结果。

最终结论必须回答：

```text
是否可以完全取消完整 calibration 缓存？
是否可以取消单条 trial buffer？
达到 baseline 90%/95% R2 分别需要多少 trial？
最低可接受结构的参数、MAC 和 peak SRAM 是多少？
精度损失来自 temporal encoder、pooling、量化还是 early stop？
```

