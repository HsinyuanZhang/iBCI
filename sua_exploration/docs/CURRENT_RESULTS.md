# Current Results: SUA/MUA Shared Encoder

**状态：当前结果摘要**  
**更新：2026-07-31**

## FP32 T4 主线补强与 INT8 接续

### M2：修正矩阵已完成

M2 的 all-held-in T4/TS4 三种子已与本地 full-SPINT B0 在同一个
chronological first-33 held-out calibration prefix 下完成配对；T4 只额外使用
这 33 个 trial 的 target labels。严格聚合结果：

| Arm | mean R² |
|---|---:|
| local full SPINT | 0.236041 |
| T4 | **0.301378** |
| TS4 | 0.232119 |

- `T4−SPINT = +0.065338`：3/3 seeds、6/6 sessions 为正，hierarchical
  95% CI `[+0.030874,+0.107409]`，exact paired Wilcoxon `p=.03125`；
- `T4−TS4 = +0.069259`：3/3 seeds、6/6 sessions 为正，hierarchical
  95% CI `[+0.038765,+0.101504]`，exact paired Wilcoxon `p=.03125`。

两项均通过全部预设 gate。该结果仍是本地 M2 held-out-calibration replay，
不是隐藏 EvalAI query/test。权威 artifact：
`results/m2_spint_t4_mainline_fp32_v1/aggregate.json`。

### SUA：修正矩阵 9/9 完成并通过全部 gate

原始 SPINT IDEncoder（B0）/T4/TS4 × seeds 42/43/44 已完成严格配对。训练和
评估的 activity calibration、T4 label pool、evaluation exclusion pool 全部统一为
chronological first-30，只评估 trial 30 之后的窗口；strict 27/6/6 manifest 下
六个 formal-test NWB 保持封存。Fail-closed 聚合接受全部 9 个 artifact：

| Arm | mean R² |
|---|---:|
| original SPINT B0 | 0.236417 |
| T4 | **0.574976** |
| TS4 | 0.284528 |

- `T4−B0 = +0.338559`：3/3 seeds、6/6 sessions 为正，hierarchical
  95% CI `[+0.273083,+0.435412]`，exact paired Wilcoxon `p=.03125`；
- `T4−TS4 = +0.290448`：3/3 seeds、6/6 sessions 为正，hierarchical
  95% CI `[+0.203539,+0.395563]`，exact paired Wilcoxon `p=.03125`。

两项均通过 mean delta `≥+0.03`、3/3 seed、6/6 session、bootstrap-CI 和
exact-Wilcoxon 的全部预设门。权威 artifact：
`results/sua_spint_t4_mainline_fp32_v1/aggregate.json`。

### Confidence-FiLM 五臂 final：T4 content 很强，但 confidence 融合无效

Seed-42 五个 arms 已全部完成 strict validation，并通过逐臂 receipt 与独立
aggregate 复算。以下是 final five-arm aggregate：

| Arm | mean R² |
|---|---:|
| T4 continuation | 0.590273 |
| FiLM(C) | **0.593672** |
| FiLM(shuffle C) | 0.590890 |
| NoFiLM-match(C) | 0.592974 |
| FiLM(TS4) | 0.277239 |

- `FiLM−T4 continuation = +0.003399`：4/6 sessions 为正，95% CI
  `[-0.000141,+0.006661]`，exact paired Wilcoxon `p=.21875`；
- `FiLM−shuffle C = +0.002782`：5/6 sessions 为正，95% CI
  `[-0.004685,+0.008329]`，`p=.4375`；
- `FiLM−NoFiLM-match = +0.000698`：3/6 sessions 为正，95% CI
  `[-0.002204,+0.003696]`，`p=.6875`。
- `FiLM−TS4 = +0.316433`：6/6 sessions 为正，95% CI
  `[+0.221798,+0.410750]`，`p=.03125`，通过 T4-content gate。

前三项必要机制对照都远低于预注册的 `+0.03`，都未达到 6/6 session 为正，
CI/Wilcoxon 门也未通过。因此 overall Stage-0 为 false，不扩展 seeds 43/44。TS4 的大幅下降同时
证明 aligned T4 content 仍然关键；最终结论只否定当前 confidence 融合机制，
不否定 T4 或 residual variance 所含的可靠性信号。五臂均使用同一 strict 27/6
train/validation manifest、`M_activity=30`、`M_T4=50`、共同 eval start 50、
epochs 5–12 score window 和同一个 selected-T4 anchor SHA
`cf533e7c…128273d`；formal held-out 未打开。权威 artifact：
`results/sua_t4_confidence_film_v1/aggregate_m50_seed42.json`。

正式候选启动前只使用 27 个 train sessions 做了 confidence 输入资格审计：

- 原 v1 的 `log residual variance` 与 `0.5 log det(C_ac)` 在 within-session
  centering 后相关达到 `0.975`，第二维几乎是第一维加一个 session 常数；
- covariance-area 在 train sessions 间的标准差只有 `0.0087`；
- scale-free directional uncertainty shape `0.5 log κ([(X'X)^−1]_{a,c})`
  的 train-session 范围为 `0.024–0.180`，标准差 `0.0409`。

因此，在没有查看任何 validation performance、且没有候选 arm 已启动的前提下，
confidence v2 冻结为：

```text
C_i = [
  log(unit_i residual variance + eps),
  0.5 log condition(session a/c design covariance)
]
```

第一维是 unit-specific fit noise，第二维是 session-level directional-balance
geometry；二者不再重复同一噪声尺度。v2 使用独立 feature/cache semantic version，
runner 与 fail-closed aggregator 均要求 `feature_version=2`，旧 v1 cache 不能混入。
这只是输入设计审计，是否有效仍必须由
`FiLM−T4 / FiLM−shuffle-C / FiLM−NoFiLM-match` 的 matched validation 结果证明。

#### Train-only predictive-validity audit：residual variance 确实有信号

为避免把“输入不重复”误当成“输入有用”，随后增加了完全独立于 decoding
validation 的 train-only 外推检查。它只打开 strict manifest 的 27 个 train
sessions；validation/formal NWB 打开数均为 0。对每个 session 和 unit：

1. 用 chronological rewarded trials `[0:M]` 拟合冻结 T4/confidence；
2. 不更新任何权重；
3. 用该 T4 预测同一 train session 的后续 trial rates；
4. 用 leave-one-session-out ridge 检查 confidence 在 T4 四维之外能否预测
   `log future cosine-prediction MSE`。

`M∈{10,15,20,30}`、统一以 `[M:50]` 为未来窗口时，每个预算均覆盖 27 sessions /
1,613 units：

| M_T4 | T4-only LOSO R² | + current C(2) | 增量 |
|---:|---:|---:|---:|
| 10 | 0.5631 | 0.7177 | **+0.1546** |
| 15 | 0.5531 | 0.7277 | **+0.1747** |
| 20 | 0.5411 | 0.7453 | **+0.2042** |
| 30 | 0.5079 | 0.7450 | **+0.2371** |

对当前正在训练的 `M_T4=50`，另用 train-session trials `[50:80]` 检查：
T4-only `R²=0.5324`，加入当前 confidence 后为 `0.8902`，增量
**`+0.3578`**；median per-session R² 为 `0.8525`。但贡献几乎全部来自
unit-specific `log residual variance`：

- residual variance 与未来误差 Spearman `ρ=0.9323`；
- directional geometry 与未来误差只有 `ρ=0.0703`；
- `T4 + residual-only` 的 LOSO `R²=0.8908`，略高于当前两维的 `0.8902`；
- 再加入 exposure、entropy、analytic-SE 的 expanded set 为 `0.8885`，没有增加。

因此，**confidence 本身具有很强的 train-session 外推有效性，但当前证据支持的
核心是 residual variance，不是 geometry**。完整 FiLM 已被三个关键 matched
controls 否定，应解释为融合/优化失败，不能解释为 reliability signal 不存在；
后备优化轮优先使用
参数匹配的 residual-only mask 和对应 residual-shuffle，而不是继续扩展描述量。
这仍不是 decoding R² 结果，不能替代五臂 validation gate。权威本地 artifacts：
`results/sua_t4_confidence_film_v1/t4c_predictive_validity_train_m10_30.json`
（SHA-256 `f2c608f0…5f74`）与
`t4c_predictive_validity_train_m50_future80.json`
（SHA-256 `8d331e47…ded`）。

对应的优化轮已经实现但尚未占用 GPU。`B3SCFR/B3SCFRS/B3SCFRA` 继续读取同一个
六维 `[T4(4), residual, geometry]` v2 artifact，并保留与完整 FiLM 完全相同的
六维 MLP fan-in、rank 和参数量；唯一变化是把 normalized geometry 乘以 0，即
替换为 train mean。三个新臂分别是 residual-only FiLM、只打乱 residual 一列的
content control、以及参数匹配的 residual-only additive NoFiLM。T4 四列和 geometry
列在 residual-shuffle 中逐值不变。所有新臂仍使用同一个 selected T4
`epoch_011` encoder+decoder warm-start、`30/50` 协议和 strict 27/6 manifest。
完整 FiLM 与 shuffle-C 的早期 train/fit-validation 轨迹几乎重合（例如 epoch 3
的 legacy held-in R² 分别为 `0.621197/0.621300`）；这不是跨 session decoding
结论，但预先指出 end-to-end 更新的 4.6M 参数可能淹没 1,208 参数的小头。因此
后备 residual-only 轮冻结 warm-started encoder substrate 和 decoder，只优化四个
`confidence_context/confidence_film` tensors。三个 residual arms 的 optimizer
trainable set 精确相同，均为 1,208 参数；完整五臂当前运行不受此修改影响。
真实 cache 审计覆盖 33 train/validation sessions、1,938 units：T4 changed=0、
geometry changed=0、residual marginal failures=0、residual order unchanged=0，formal
path resolved=0。

Fail-closed aggregate 要求 mask receipt 精确为 `[true,false]`、六维参数匹配收据、
冻结 substrate/decoder、精确四 tensor/1,208 optimizer 参数、相同 anchor SHA 和
formal seal，并将 residual-FiLM 同时与 T4 continuation、
residual-shuffle、residual-NoFiLM 比较。冻结改动的目标回归 `13 passed`，并用
真实 selected-T4 checkpoint 实例化三个 arm，均得到相同的 1,208 参数集合；三臂 dry-run
确认只生成 `calibration_n=30 / pool_size=50` 的 validation 命令，没有启动训练。
自动顺序现为 **完整 FiLM → decoupled K/V → M15 shrinkage → residual-only
优化 → B3TStream+T4**；只有前两项都没有形成有效候选时 M15 shrinkage 才会
使用 GPU，residual 优化位于其后。

### 低标签 T4 shrinkage：train-only 代理结果为正，GPU pilot 已预注册

完整 FiLM 的早期轨迹提示 M50 可能不是 confidence 最有价值的工作点，因此又做了
一个严格 train-only、nested leave-one-session-out audit。对每个单位只收缩 T4 的
`a,c,m`，保留 intercept `b`：

`factor=(a²+c²)/(a²+c²+λ·σ²·trace([(X'X)^-1]ac))`。

每个 fold 只用另外 26 个 training sessions 选择 family/λ，再在被留出的 training
session 后续 trials `[M:50]` 上评分；27 train receipts 与 strict manifest 完全一致，
validation/formal opened 均为 false。结果：

| M_T4 | nested LOSO 选择 | geometric future-rate MSE ratio | 改善 sessions | pilot gate |
|---:|---|---:|---:|---|
| 10 | Wiener λ=3，27/27 folds | **0.9283** | 26/27 | pass |
| 15 | Wiener λ=3，27/27 folds | **0.9537** | 27/27 | pass |
| 20 | Wiener λ=3，27/27 folds | **0.9725** | 23/27 | pass |
| 30 | Wiener λ=1/3 | 0.9899 | 19/27 | fail |

固定、无需 fold tuning 的 Wiener λ=1 在 M10/M15 也分别改善 27/27 sessions，
MSE ratio 为 `0.9441/0.9623`，说明结论不是脆弱的单点超参数效应。权威本地
artifact 是 `results/sua_t4_confidence_shrinkage_audit_v1/train_loso_m10_30.json`
（SHA-256 `71928d62…6e12`）。这仍只是 future-rate proxy，不是 decoding R²。

GPU pilot 选择 **M_T4=15**：相比 M50 减少 70% labelled trials，同时 train-only
proxy 在 27/27 sessions 改善。固定候选 `t4w3`/control `ts4w3` 已实现为四维缓存
特征，Wiener λ=3 在任何 validation 运行前冻结。Stage-0 三臂为 ordinary T4@15、
T4W3@15、TS4W3@15；均保持 `M_activity=30`、共同 eval start 50、12 epochs 和
strict 27/6 manifest，并与现有 T4@50 reference 比较。通过条件同时要求：
T4W3@15 对 T4@15、TS4W3@15 的机制增益通过门槛，且对 T4@50 的平均 R² 差距不超过
0.03。33 个 train/validation sessions 的前 15 rewarded trials 全部 rank-3，
最少覆盖 7 个方向，最大 condition `1.887`；formal path opened=0。

自动顺序更新为 **完整 FiLM → decoupled K/V → M15 shrinkage → residual-head-only
FiLM → B3TStream+T4**。M15 shrinkage 仅在前两项没有验证出有效候选时占用 GPU。

### Decoupled K/V：实现与硬件凭据完成，性能尚未运行

第一阶段已冻结为五个同预算 FP32 arms：

- coupled `readin(x+E)` T4 baseline；
- decoupled `K(E,T4), V(x)`；
- decoupled `K(E,TS4), V(x)`；
- decoupled `K(E), V(x)`；
- activity-key control `K(x), V(x)`。

所有 arm 使用 fresh decoder、同一个 teacher、27/6 train/validation manifest、
`M_activity=30`、`M_T4=50`、共同 `eval_start=50`、12 epochs 和固定
epoch 5–12 checkpoint score。TS4 只打乱 decoder key 的 T4 行；identity
encoder 始终接收对齐的真实 T4，避免把 encoder 内容破坏误算成 key 机制效果。

在实际 teacher 配置（`D=512`、coupled 64 heads；decoupled 2 heads、
`D_k=D_v=32`）和参考 `N=64` 下，静态成本收据为：

| decoder path | configured MAC/frame | persistent session state |
|---|---:|---:|
| coupled T4 | 57,970,688 | 12,800 B (`E[N,50]`) |
| cached decoupled K/V | 4,997,120 | 8,192 B (`K[N,32]`) |

即 decoder-path 配置 MAC 约下降 `91.38%`，持久状态下降 `36%`；这不是整机
实测 latency。缓存只保存 projected key，不保存原始 T4/confidence，也没有
`N²` neuron self-attention。**目前尚无该五臂的 R² 结果**，所以硬件数字不能
单独构成有效性结论。FiLM final aggregate 确认失败后，fresh coupled baseline
已于 2026-07-31 09:53 HKT 在 GPU0 启动；GPU1 在 seed-44 T4 anchor 释放后自动
运行 `e_ts4/x_only`，两条 lane 随后补齐其余 arms。

首轮开始前另做了一个完全不打开 neural dataset 的 teacher-checkpoint 谱审计，
用来约束“若首轮失败，下一轮应该改哪里”。原 attention 投影在 rank 32 时只保留
`Wq=29.82%`、`Wk=39.37%` 的平方奇异值能量；相比之下组合 value map
`Wo@Wv` 保留 `76.94%`。rank 48 时对应为 `38.89%/47.49%/83.78%`。因此
`Dk=32` 是比单独 `Dv=32` 更明确的容量风险。参考 `N=64` 下，
`Dk=48,Dv=64` 的配置 MAC 仍比 coupled path 低 `91.11%`，静态 key width
为 48，不超过现有 `E[N,50]` 状态。该证据只预注册失败后的宽度诊断，不能提前
改写首轮或替代 R²；artifact：
`results/sua_t4_decoupled_kv_v1/teacher_low_rank_audit.json`。

源码审计同时发现，v1 不只改变 K/V 语义：它还移除了 coupled teacher 的
预训练 `fc_in: 50→512→512` activity read-in，并用随机 `50→32` value
projection 直接处理原始活动。因此 v1 若明显掉点，不能直接归因于“静态 key
无效”。预注册的 representation-preserving v2 会分别计算
`h_E=teacher_readin(E)`、`h_x=teacher_readin(x)`，再由 `[h_E,T4]` 生成静态
low-rank key、由 `h_x` 生成在线 value，并用 teacher Q/K bilinear 与
`Wo@Wv` 的低秩分解初始化（或显式声明短 prediction-distillation warm-up）。
在 `Dk=48,Dv=64,N=64` 的同一配置计数下，该路径约为 25,462,784
decoder MAC/frame，相对 coupled 下降 `56.08%`；`K[N,48]` 为 12,288 B，
仍比 `E[N,50]` 少 `4%`。这是失败后下一轮的设计预算，不是当前精度或整机
latency 结果。

另一个 selected-anchor、strict 27-train-cache-only 审计检查了联合
`LayerNorm([E,T4])`。1,613 个 train units 中，T4 占 `4/54=7.41%` 坐标，
归一化后能量占比中位 `8.00%`、平均 `10.00%`，所以没有证据认为 T4 因四维
太少而被直接淹没。但在 `e_only` counterfactual 中把 T4 四维置零，会连带改变
前 50 维 E 的归一化坐标：relative L2 中位 `5.81%`、P90 `17.37%`。因此
`e_only` 不是严格的“只删除 direct T4”控制，`e_t4−e_ts4` 才是主要内容检验。
若首轮需要机制清理，下一轮可用 `W_E LN(E)+W_T LN(T4)`；该改动不能因尺度
假说而自动启动。receipt 为 27 train caches、0 train NWB、0 validation、
0 formal；artifact：
`results/sua_t4_decoupled_kv_v1/key_normalization_audit_train.json`。

未来 formal-held-out 入口也已在不打开 test NWB 的条件下做了兼容性修复：
validation 与 formal 现在共享 checkpoint topology 重建，decoupled candidate 会将
真实/TS4 key feature 送入独立 key path，zero-identity control 则同时清零
`E` 与 direct T4 key；formal 产物会显式披露 calibration target labels/rates
用于冻结 side-feature estimator。相关 CPU/fixture 回归通过，但 formal test
仍未运行、receipt 仍未创建。

### INT8 顺序：等待两个额外 FP32 实验

先前“主线完成即自动量化”的 watcher 已停用。按 2026-07-30 的用户顺序，
当前先运行：

1. confidence-conditioned FiLM / 低标签 T4；
2. T4-conditioned decoupled cross-attention（是否加入 confidence 取决于第 1 项）。

固定顺序为 FiLM → decoupled K/V；仅当两者均未给出有效改进时，才运行已预注册
的 `fresh T4 / B3T+T4 / B3T+TS4` 效率 fallback。该分支不能替代第二个
decoupled-K/V 实验。它目前
实测静态成本目标为相对普通 T4 encoder 参数约 `−30.8%`、session-path MAC
约 `−65.3%`、persistent online state 不增加。2026-07-31 已补齐真正的
bin-streaming 执行：`B3T/B3TS` 现在逐 bin 累积 `[N,K=12]`，不保存完整
`[N,T=100]` trial；参考 `N=64` 下 transient trial state 从 `25,600 B`
降至 `3,072 B`（`−88%`），batch/full/bin、padding 和 unit+side permutation
目标测试为 `19 passed`。这只证明执行等价与成本，不是 R² 结果；最终仍需
matched `B3T+T4` 的 non-inferiority 与 T4-content gates 才能称为 deployment
improvement。

只有最终 FP32 架构冻结后才量化 **identity encoder**；decoder 保持 FP32，复用
其他平台上已有的 decoder QAT 无损证据，避免重复 GPU 消耗：

- `pre_pool`、真实 `post0: 68→64`、`post1`、`post2` 全部 W8A8；
- pooled activity 64 维与 normalized T4 4 维共享同一个 post0-input A8 scale，
  在整数域拼接；禁止 T4 FP bypass；
- INT8×INT8、INT32 accumulator、integer requant，输出 INT8 `E` dequant 后
  送入 FP decoder，按同一六-session fixed protocol 比较端到端 R²；
- PTQ scale/候选选择只使用 27 train sessions 和 identity RMSE，不使用六个
  validation session 选 scale；
- PTQ 需满足 `ΔR²≥−0.01`、每个 activation edge saturation `≤0.5%`、
  INT32 overflow 为 0、STE 与独立 integer engine 的 `E` 逐值一致；
- 任一门失败自动进入固定 8-epoch encoder QAT；QAT 可使用 27 个 train
  sessions 的训练标签，但不用 validation 选 epoch，formal test 继续封存。

量化入口在架构选择前不得启动；未来结果目录为
`results/sua_spint_t4_encoder_int8_v1/`。结论必须写作
“**T4 encoder INT8 + FP decoder**”，不能写作本轮完成了 full-model INT8。

## Native FALCON local held-out-calibration replay

Frozen internal-LOSO F0/T4/TS4 checkpoints were replayed test-only on local
`held-out-calib` files (three matched cells/task), matching the existing
SPINT/B3 local held-out path. This is not a public benchmark or hidden EvalAI
query/test-set result. `trials.tgt_loc` is used from each session's calibration
file for deployment-time support and is not taken from velocity/EMG evaluation
covariates; the same local NWB supplies the neural and behavioural arrays used
for this replay. No backward pass, optimizer update, or held-out checkpoint
selection occurs. M1 uses chronological first 10 trials and M2 first 33;
normalization comes only from that cell's held-in fold-train sessions. Source
SHA, label/trial alignment, rank, support and complete-session checks passed.

| task | F0 | T4 | TS4 | T4−F0 | T4−TS4 |
|---|---:|---:|---:|---:|---:|
| M1 (3 cells × 3 sessions) | .62725 | .62505 | .62879 | −.00221; 1/3 cells positive | −.00374; 0/3 |
| M2 (3 cells × 6 sessions) | .22160 | .29139 | .22938 | +.06979; 3/3, 16/18 sessions | +.06201; 3/3, 17/18 |

T4 is supported on native M2 under this stated label-privileged local
calibration protocol, but does not replicate to M1. The complete audited
per-cell/per-session result is `results/native_mua_heldout_t4_v1/aggregate_heldout.json`.

## 结论摘要

1. **架构迁移成立**：B3 EarlyPool 分别在 MUA 和 SUA 上训练时都能逼近各自 teacher。
2. **直接权重迁移失败**：FALCON M2 的 B3 权重在 MC_Maze 上无法生成有判别力的 identity。
3. **跨 neuron self-attention 是否有效：当前未知（结论已撤回）**。`attention_arch_screen_v3` 的测量不可靠——噪声底是被测效应的 6 倍、选择偏置不均等、且存在 run 目录冲突。此前"attention 已被否定"的表述已撤回。见 §E 撤回说明与 §H 诊断。
4. **calibration 本身是承重部件**：identity 置零后 validation R² 全部转为强负值（B3 `−0.277`、B16 `−0.515`、B15 `−2.298`）。但当前 `learned_prior` 公平对照失效，见 §F.2。
5. **固定 token 接口机械可行、精度未达标**：fixed-slot router 的接口、缓存等价与压缩 gate 全通过，精度 gate 未通过。见 §G。
6. MC_Maze single-session 上的 B15/B16 微小优势（§B）在跨 session 设置下未复现为机制性收益。
7. **方向调谐 T4/T8 是项目首个稳定的 `effective` 结果**：SUA 上
   `T4-F0=+0.2528`、`T8-F0=+0.2567`，6/6 session、3/3 seed 全正；见 §J.5。
8. **T4 增益在 pseudo-MUA 上保留且数值更大**：`T4-F0=+0.3177`、
   `T4-TS4=+0.3657`，两项均为 `effective`；但 SUA-vs-pseudo-MUA interaction
   `Gamma=-0.0650 ± 0.0307 SE` 仍为 `indeterminate`。见 §K.1–K.2。
9. **静态 per-electrode reliability gate 已判无效**：在 T4 substrate 上
   `T4GATE-T4=-0.0108 ± 0.0049 SE`，不是下一轮优化方向。见 §K.3。
10. **原生 MUA held-in internal-LOSO 上的 T4 提升不是跨数据集稳定结论**：完整 18/18 matrix
    中，M2 相对 F0 为 `+0.0963`（3/3 cells 正），但 M1 仅 `+0.0078`
    （1/3 正）；M1 相对 TS4 为 `+0.0248`（3/3 正），M2 相对 TS4 则只有
    2/3 正。见 §K.4。
11. **SUA 同电极 relation 最终为 `ineffective`**：完整
    `4 arms × 3 seeds` 矩阵已严格复核；`REL−T4 =
    −0.001440 ± 0.000876 SE`，3/3 seed 均为负；`REL−REL-NG =
    +0.006496 ± 0.007835 SE` 也排除 `+0.03`。停止该路线，不进入
    relative-amplitude 阶段；formal test 未打开。见 §K.5。
12. **原生 MUA local held-out-calibration replay 同样呈现任务差异**：
    M2 `T4−F0=+0.06979`、`T4−TS4=+0.06201`，两项均 3/3 cells
    为正；M1 分别为 `−0.00221` 与 `−0.00374`。该结果使用 M1 first-10 /
    M2 first-33 chronological target-labeled support、校准时无反向传播，但
    不是隐藏 EvalAI query/test-set 结果。

## 评估口径

现有 MC_Maze 结果使用：

- DANDI 000128 的一个带行为标签 train NWB；
- `heldout == false` 的 sorted units；
- NWB 内部 `train`/`val` trial split；
- 10 个 calibration trials，`T=100`，20 ms bins，`W=50`；
- teacher decoder 冻结，student 学习 identity encoder；
- 比较脚本默认评估完整 validation loader，并用 float64 计算 representation 指标。

这属于 **single-session internal validation**，不是跨 session LOSO。当前 datamodule 还将同一 validation loader 同时标为 `val_heldin` 和 `val_heldout`，因此所有 SUA 文档只采用 “internal validation” 表述。

## A. 架构复用与零样本权重迁移

以下是早期 teacher/checkpoint 组的已报告结果：

| 实验 | MC_Maze internal-val R² | 解释 |
|---|---:|---|
| MC_Maze SPINT teacher | 0.868 | 早期 SUA teacher |
| B3 在 MC_Maze 上重新训练 | 0.831 | 达到 teacher 的约 95.7% |
| FALCON M2 B3 → MC_Maze zero-shot | 0.001 | 直接权重迁移基本失败 |
| zero-shot + per-neuron z-score | -0.18 | 简单输入归一化不能修复迁移 |

对应 representation 分析：

| 指标 | 值 | 解释 |
|---|---:|---|
| M2 identity 与 MC identity cosine | 0.08 | 近似正交 |
| Pearson correlation | 0.01 | 基本无相关性 |
| M2 B3 在 MC_Maze 上的 identity std | 0.25 | neuron 区分度弱 |
| MC_Maze B3 identity std | 0.92 | neuron 区分度明显 |

**支持的结论**：问题结构和网络架构可复用，但已学习 temporal filters 与 identity space 是信号类型相关的。

## B. SUA 结构候选 B15/B16

### 结构

| Variant | 改动 | 参数量 | 主要代价 |
|---|---|---:|---|
| B3 | per-neuron EarlyPool | 18,034 | 基线 |
| B15 | neuron-axis self-attention + residual + LayerNorm | 34,802 | finalize 需要全局 neuron feature，含 O(N²) attention |
| B16 | 跨 trial mean + variance | 22,130 | 需要额外二阶矩 accumulator |

在当前 SUA 配置 `N=137, M=10, T=100` 下，encoder cost profile 为：

| Variant | session MACs | 相对 B3 | support state | peak live state |
|---|---:|---:|---:|---:|
| B3 | 10.33M | 1.00× | 34.3 KiB | 87.8 KiB |
| B15 | 13.93M | 1.35× | 34.3 KiB | 87.8 KiB |
| B16 | 10.89M | 1.05× | 68.5 KiB | 122.0 KiB |

B15 的代价主要是参数量接近翻倍和 `O(N²)` finalize；B16 的计算增量较小，但 support state 翻倍。

### 严格 seed-42 checkpoint validation

| 模型 | checkpoint validation R² | 备注 |
|---|---:|---|
| Teacher epoch 83 | 0.90608 | 当前共享 teacher |
| B15 seed 42 epoch 13 | 0.90977 | 40-epoch 上限、patience 10 |
| B16 seed 42 epoch 35 | 0.90848 | 40-epoch 上限、patience 10 |
| B3 seed 42 epoch 36 | 0.90781 | 40-epoch 上限、patience 10 |

### `compare_neuronid_variants.py` 完整 validation 重评

| Variant | task R² | identity norm MSE | cosine | Pearson |
|---|---:|---:|---:|---:|
| B15 seed 42 | 0.90977 | 0.01584 | 0.99218 | 0.99202 |
| B16 seed 42 | 0.90848 | 0.01835 | 0.99092 | 0.99068 |
| B3 seed 42 | 0.90781 | 0.03397 | 0.98313 | 0.98265 |
| Teacher | 0.90608 | 0.00000 | 1.00000 | 1.00000 |

三种 student 均显式使用 seed 42，且共享 teacher SHA-256、datamodule、loss、decoder、核心超参数、40-epoch 上限、patience 10 和 checkpoint 选择规则。相对 B3，B15 的 task R² 为 `+0.00196`，identity normalized MSE 低约 `53%`；B16 的 task R² 为 `+0.00068`，identity MSE 低约 `46%`。B15 仅比 B16 高 `0.00129`。这些差异在单个 session/seed 上很小，不能视为稳定架构优势。

B15/B16 略高于 teacher 只表示其 identity 与固定 decoder 的组合在这次内部 validation 上更匹配行为目标，不代表外部泛化超过 teacher。旧 B15/B16 的等 epoch 结果保留为历史探索；当前主结论只采用可追踪的 seed-42 重训。

### 结构诊断

在相同前 200 个 validation batches 上进行 inference-only 消融：

| B15 推理路径 | task R² | identity norm MSE |
|---|---:|---:|
| 完整 attention | 0.92009 | 0.01584 |
| 仅保留 self diagonal | 0.91851 | 0.01791 |
| 完全移除 attention 输出，保留 LayerNorm | 0.89754 | 0.07809 |

完整 attention 相对 self-only 只增加 `0.00158` R²，而 attention 路径整体相对 no-attention 增加 `0.02255`。这再次提示 B15 的大部分 attention-path 收益来自每个 neuron 自身的 value/output 变换、残差和 LayerNorm，而不是跨 neuron 关系。该实验是对已训练 B15 的推理消融，不替代独立训练的 capacity/normalization control。

B16 将 variance 输入置零后 R² 从 `0.91902` 降至 `0.51262`，说明当前 B16 强烈依赖 variance 分支；但它在完整 validation 上只比 B3 高 `0.00068`，因此“模型使用了 variance”不等于 variance 带来稳定净收益。机制表使用固定前 200 batches，不能与完整 loader 的绝对 R² 混排。

## C. MUA 参考结果与 B16 首次对照

| 模型 | Task | held-out R² | Paper SPINT | 协议 |
|---|---|---:|---:|---|
| B3 | FALCON M2 | 0.236 ± 0.102 | 0.26 ± 0.13 | LOSO fold 0 / seed 42 + 6 unseen sessions |
| B16 | FALCON M2 | 0.248 ± 0.137 | 0.26 ± 0.13 | 与 B3 完全相同；best epoch 1 |
| B3 | FALCON M1 | 0.630 | 0.66 ± 0.07 | LOSO fold 0 + 3 unseen sessions |
| B3 | FALCON H1 | — | 0.29 ± 0.15 | 尚未完成 |

M2 的 B16-B3 held-out mean delta 为 `+0.0112`，约 `+4.7%` 相对提升；逐 session 为 4 个提升、2 个下降。B16 增加 22.7% encoder 参数，但 session MAC 只增加 1.84%；代价主要是二阶矩 support state 翻倍。由于这里只有一个 fold/seed，当前只能说“B16 在 MUA 也出现正向候选信号”，不能宣称稳定优于 B3，也不能再把 B16 的潜在收益直接归因于 SUA sorting reliability。

这些 MUA 结果不能与 MC_Maze internal validation 的绝对 R² 直接比较。

## D. 跨 session SUA（DANDI 000688 sub-C CO，修正 regime 后）

口径：旧 unit-count regime（`N < 100`）39 个 session，chronological
`27 train / 6 validation / 6 test`；固定前向协议 `first / n=30 / pool=50`；
encoder 与 decoder 冻结。下表全部为 **6 个 validation sessions** 的均值，
**不是** formal held-out test。

**主指标**（固定协议下重新评估的 per-session R²，对 6 session 与 2 seed 取均值）：

| Variant | validation mean R² |
|---|---:|
| B3 | 0.3265 |
| B15D | 0.3534 |
| B15P | 0.3599 |
| B15 | 0.3662 |

**checkpoint 选择依据**（训练期 `val_heldin/r2_mean`，仅用于选 epoch，
**不可与上表混排、不可用于变体排名**）：

| Variant | seed 42 | seed 43 |
|---|---:|---:|
| B3 | 0.2896 (ep 0) | 0.3054 (ep 2) |
| B15D | 0.3050 (ep 0) | 0.3250 (ep 1) |
| B15P | 0.3111 (ep 1) | 0.2696 (ep 1) |
| B15 | 0.3142 (ep 1) | 0.3222 (ep 4) |

两表口径不同：前者是固定 `first/n=30/pool=50` 前向协议下的评估，后者是
训练循环内的 validation metric。所有变体的最佳 epoch 都很早（0–4），
说明在这个跨 session 设置下模型很快过拟合 train sessions。

Step 0 单 session 上界为 `0.6937`（端到端、非冻结），POYO 在同源数据的
single-session CO 参考值为 `≈0.935`。

> 早于 2026-07-24 的 53-session 跨 session 结果（zero-shot `−0.124`、
> few-trial finetune `+0.692`）受 unit-count regime 4× 跳变混淆，且后者使用
> held-out 行为标签与反向梯度，仅为 diagnostic oracle。详见
> [`P3_CROSS_SESSION_ANALYSIS.md`](P3_CROSS_SESSION_ANALYSIS.md)。

## E. Attention 参数匹配对照（`attention_arch_screen_v3`，2026-07-25）

> ## 结论撤回（2026-07-25 晚）
>
> **本节此前记录的「attention 机制被否定、B15 增益约 84% 由 B15P 解释」结论
> 不成立，已撤回。** 该结论所依赖的测量本身不可靠，三条独立缺陷见 §H。
> 下表数字保留为原始记录，但**不得用于任何机制主张**，也不得作为
> B15/B15P/B3 的排名依据。
>
> 需要重跑修正后的筛选才能回答 attention 是否有效。当前状态是**未知**，
> 不是「已否定」。

章程见 [`ATTENTION_ARCHITECTURE_SCREEN.md`](ATTENTION_ARCHITECTURE_SCREEN.md)，
结果见 `results/attention_arch_screen_v3/aggregate.json`。五个 gate 全部为 false，
但见上方撤回说明。

对照的作用：`B15P` = 同参数量、零跨 neuron 通信的 per-neuron residual MLP +
LayerNorm；`B15D` = 保留 attention 的 QKV/输出投影/残差/LN，但 mask 强制对角。

### SUA（6 validation sessions，seeds 42/43）

| Paired delta | mean | median | min | 为正的 session |
|---|---:|---:|---:|---|
| `B15 − B3` | +0.0398 | +0.0468 | −0.0302 | 5/6 |
| `B15 − B15D` | +0.0128 | +0.0164 | −0.0846 | 4/6 |
| `B15 − B15P` | **+0.0064** | **−0.0065** | −0.0394 | **2/6** |

### MUA（FALCON M2 internal LOSO，3 cells）

| Paired delta | mean | min | 为正的 cell |
|---|---:|---:|---|
| `B15 − B3` | **−0.0142** | −0.0159 | **0/3** |
| `B15 − B15D` | −0.0157 | −0.0322 | 0/3 |
| `B15 − B15P` | −0.0190 | −0.0386 | 0/3 |

单元分数：MUA B3 `0.7561`、B15D `0.7576`、B15P `0.7609`、B15 `0.7419`（3 cell 均值）。

### 判读（已撤回）

~~`B15P − B3 = +0.0334`，占 `B15 − B3 = +0.0398` 的约 84%，落在"B15 ≈ B15P"
决策行。~~

该判读不成立。§H 显示这些 delta 全部处于测量噪声之内，且在另一个同样
合理的估计量下 `B15 − B15P` 从 `+0.0064` 变为 `+0.0462`、`B15P − B3`
**符号翻转**。排名由估计量决定，而非由架构决定。

历史 `advance_to_paired_pilot = false` 只约束被撤回的 attention-v3 分支；截至
2026-07-25 当时确无对应产物。后续 T4 驱动的 pseudo-MUA bridge 使用独立冻结
问题与章程，于 2026-07-28 完成（见 §K）；这不恢复 attention-v3 的旧判读，也
不等于真实 M1 external replication 已完成。

## F. 无标定对照（validation-only）

### F.1 zero-identity

identity 置零、不调用 identity encoder，其余协议相同：

| Variant | 有 calibration | 无 calibration | delta |
|---|---:|---:|---:|
| B3 | ~0.29 | `−0.277` | +0.57 |
| B16 | ~0.30 | `−0.515` | +0.82 |
| B15 | ~0.31 | `−2.298` | +2.67 |

三者的 `positive_session_count` 均为 0。**calibration 得到的 identity 是
承重部件，不是装饰。**

### F.2 `learned_prior` 对照当前无效（必须修复）

`p3_no_calibration_validation_b15_learnedprior_s42.json` 与对应 b16 文件所用
checkpoint 为 `b15_dandi688_co_oldregime_s42` / `b16_dandi688_co_oldregime_s42`
——它们是 **calibrated 模式训练的 checkpoint**，其 `population_identity`
参数从未被训练。因此 learned-prior 结果与 zero-identity **逐位相同**
（abs diff 分别为 `3.97e-07` 与 `0.0`）。

`p3_fair_baseline_summary_s42.json` 已记录
`control_mode_learned_prior_equal_zero_identity: true`，但仍把
`delta_vs_learned_prior = +2.67 / +0.82` 作为公平余量报出。**该字段目前不可引用。**

**代码修复（2026-07-25）：** `select_gradient_free_protocol_dandi688.load_frozen_model`
与新建 `fair_baseline_summary.py` 已加入硬性守卫——calibrated checkpoint 跑
`learned_prior` 将 raise；退化对照时 `delta_vs_learned_prior` 写 `null` 并附
`invalid_reason`。

因此当前可写的表述是"calibration ≫ 喂零"，**不是**"calibration ≫ 一个
像样的无标定模型"。唯一真正以 `--identity_mode learned_prior` 训练的 run
（`b16_dandi688_co_learnedprior_s42_fair`）只跑了 **1 个 epoch**，
best val R² = `−0.0191`，不构成收敛基线。

### F.3 待处理的运行事故（代码审查更新 2026-07-25）

`b16_dandi688_co_learnedprior_s42`（40 epoch 正式版）于 2026-07-24 23:20 启动，
输出目录自 23:23 后再无写入（只有 tfevents、hparams、run_metadata，无 checkpoint），
进程持续存在约 18 小时 42 分后消失。

**实现侧排查结论（未在本地复现挂起）：**

1. `configure_optimizers` 在 `learned_prior + freeze_decoder` 下仍保留
   `population_identity` 作为唯一可训练参数，优化器参数组非空。
2. 默认 `num_workers=0`，DataLoader 死锁不是默认路径风险；若手动提高
   `num_workers` 并与 session 缓存锁竞争，存在理论上的 worker 阻塞可能。
3. 已加入 `validate_learned_prior_checkpoint` 加载守卫，防止 calibrated
   checkpoint 被误用于 learned-prior 对照。

**待执行（需 GPU）：** 重跑 B3/B16 learned-prior 收敛基线并重生成
`p3_fair_baseline_summary_s42.json`（见 [`HANDOFF_SIDE_FEATURES.md`](HANDOFF_SIDE_FEATURES.md) Task A.2.3–A.2.4）。

## G. Fixed-Slot Router Pilot（`fixed_slot_router_pilot_v1`，2026-07-25）

章程见 [`FIXED_SLOT_ROUTER_PILOT.md`](FIXED_SLOT_ROUTER_PILOT.md)，
报告见 `results/fixed_slot_router_pilot_v1/REPORT.md`。validation-only。

| Interface | mean R² | Δ vs B15P (`0.3599`) | 为正的 session | 精度 gate |
|---|---:|---:|---:|---|
| `K=32` soft FiLM | 0.1820 | −0.1779 | 5/6 | false |
| `K=16` soft FiLM | 0.0773 | −0.2826 | 4/6 | false |

| Gate | 结果 |
|---|---|
| 固定接口 `[B,K,50]` + permutation 不变 | 通过 |
| cached state 与正常前向等价 | 通过，12/12 session-seed pair，最大差 `2.62e-6`（`K=16` 为 `3.10e-6`） |
| 压缩率（`N=64`） | 通过，`K=32` 2.0×、`K=16` 4.0× |
| 精度（阈值 `B15P − 0.03 = 0.3299`） | **未通过** |

路由诊断：`K=32` 归一化 assignment entropy `0.8568`、最大 assignment 概率
`0.1084`、slot-mass CV `0.7977`、有效 slot 数 `23.61`。预注册的低温 follow-up
触发线是 entropy ≥ `0.95`，**未触发**，因此精度失败不可归因于路由退化为
均匀 mean-pooling。

**判读**：固定 token 接口在机械层面已证明可部署，但当前训练方案下
`K=32` 要付掉约一半 R²。这是 rate–distortion 结果，不是 attention 或
routing 机制的证据。

## H. 测量可靠性诊断（2026-07-25 晚，导致 §E 撤回）

从各 run 的 tfevents 提取逐 epoch validation 曲线后发现三条独立缺陷。
它们共同使 `attention_arch_screen_v3` 无法支持任何机制主张。

### H.1 噪声底高于被测效应

train loss 在所有 run 中单调平滑下降（B3 `0.73→0.50`，B15 `0.67→0.46`），
训练本身正常。但 validation R² 在 epoch 之间剧烈震荡且无趋势：

| 域 | epoch 间 std | 被测效应 | 噪声/效应 |
|---|---:|---|---:|
| SUA（8 runs） | **0.0388** | `B15−B15P = +0.0064` | **6×** |
| SUA | 0.0388 | `B15−B3 = +0.0398` | 1.0× |
| SUA | 0.0388 | 预注册门槛 `+0.005` | **8×** |
| MUA（12 runs） | **0.0245** | `B15−B3 = −0.0142` | 1.7× |

单个 SUA run 的 epoch 间摆幅达 `0.09–0.15`。val loader 使用
`random_calibration=False`（`multisession_datamodule.py:943`），因此该噪声
不是 calibration 重采样造成的，而是权重微小变化在未见 session 上的真实
泛化不稳定。

**预注册的 `+0.005` 门槛比噪声底低约 8 倍，从一开始就不可能被有意义地
通过。** 任何"通过"都会是噪声。

### H.2 不均等的 max-of-N 选择偏置

checkpoint 按 validation 最大值选取，但各变体实际训练的 epoch 数不等：

| 域 | epoch 数 |
|---|---|
| SUA | B15 = 7, 10；B15P = 7, 7；B15D = 6, 7；B3 = 6, 8 |
| MUA | B15 = 16, 13, 20；B15D = 8, 13, 15；B15P = 40, 40, 20；B3 = 17, 17, 20 |

在 `σ = 0.0388` 下，`E[max] − mean` 从 `n=5` 的 `+0.045` 增至 `n=11` 的
`+0.062`。抽样次数多的变体系统性地获得更高的选中值。SUA 上 B15 相对 B3
的这一差额约 `+0.004`（约占观测 `B15−B3` 的 10%）；MUA 上 B15P 以 40 epoch
对 B15D 的 8 epoch，偏置更大。

### H.3 估计量决定结论

改用同样合理的「固定 epoch + 跨 epoch 平均」估计量（消除选择偏置）后，
在训练期 val 指标上：

| 配对 | best-checkpoint 口径 | 固定 epoch 口径 | 逐 epoch 符号翻转 |
|---|---:|---:|---|
| `B15 − B15P` | +0.0064 | **+0.0462** | 1/6 |
| `B15 − B3` | +0.0398 | +0.0218 | 2/6 |
| `B15P − B3` | +0.0334 | **−0.0244**（符号相反） | — |

两个估计量给出相反的机制结论。注意两者所用指标不同（训练期 val vs 固定
`first/n=30/pool=50` 协议），因此绝对值不可直接比较；但排名的不稳定性
足以说明结论不由架构决定。

### H.4 运行目录冲突（独立的完整性 bug）

`attention_arch_screen_v3` 的 MUA 臂中，**两对本应独立的 seed 共用了同一个
Hydra run 目录**：

| 变体 | fold1 seed42 output | fold1 seed43 output | 共用的 hydra 目录 |
|---|---|---|---|
| B15P | `..._s42_20260725_071145` | `..._s43_20260725_071145` | `logs/train/runs/2026-07-25-07-11-45` |
| B3 | `..._s42_20260725_092511` | `..._s43_20260725_092511` | `logs/train/runs/2026-07-25-09-25-11` |

证据：

- 该目录 `.hydra/overrides.yaml` 只记录了 `seed=43`（单一配置）；
- 目录下有**两个 PID**（268035、268037）写入的 tfevents，事件流交错；
- checkpoint 目录含 Lightning 去重后缀：`epoch_016-v1.ckpt`、
  `epoch_017-v1.ckpt`、`epoch_019-v1.ckpt`、`last-v1.ckpt`；
- manifest 把 `epoch_017-v1.ckpt` 归给 seed 42、`epoch_015.ckpt` 归给 seed 43。

**从文件系统无法确定 `-v1` 文件由哪个进程写入**；该 seed 归属是 harness
的假设，不是记录下来的事实。B15 与 B15D 的 fold1 目录互不相同（074850 /
075516、073809 / 073802），未受影响；但 `B15−B15P` 与 `B15−B3` 的 fold1
配对涉及被污染的一侧。

### H.5 结论

`attention_arch_screen_v3` 的 SUA 与 MUA 两臂都不能支持机制主张。
attention 是否有效当前是**未知**。修复方向见 [`../ROADMAP.md`](../ROADMAP.md) M1–M4。

## I. Per-Unit 侧信息消融（`side_feature_ablation_v2`，2026-07-26）

首个使用 [`MEASUREMENT_PROTOCOL_V4.md`](MEASUREMENT_PROTOCOL_V4.md) 的 screen。
15 runs = `F0/F1/F2/FS1/FS2` × seeds `42/43/44`，固定 12 epoch、无 early
stopping、变体分数取 epoch 5–12 的协议指标平均。章程见
[`UNIT_SIDE_FEATURE_ABLATION.md`](UNIT_SIDE_FEATURE_ABLATION.md)。

### 变体分数

| 组 | 内容 | 3-seed 均分 | σ_seed |
|---|---|---:|---:|
| F0 | B3 基线，无侧信息 | 0.3140 | 0.0232 |
| F1 | + p2p/noise_std/snr（3 维） | **0.2816** | 0.0347 |
| F2 | + 波形形状（6 维） | 0.3128 | 0.0481 |
| FS1 | F1 特征的 unit 维置换对照 | 0.3273 | 0.0401 |
| FS2 | F2 特征的 unit 维置换对照 | 0.3494 | 0.0464 |

### 配对 delta（per-session seed-mean，n=6）

| 配对 | mean | 为正 session | 逐 seed |
|---|---:|---:|---|
| `F1 − F0` | −0.0324 | 1/6 | 3/3 为负 |
| **`F1 − FS1`** | **−0.0457** | **0/6** | **3/3 为负** |
| `F2 − F0` | −0.0012 | 2/6 | 2/3 为负 |
| `F2 − FS2` | −0.0367 | 1/6 | 2/3 为负 |
| `FS2 − F0` | +0.0355 | 3/6 | 3/3 为正 |

### 判定

**`F1: indeterminate`、`F2: indeterminate`**（聚合器输出与独立复算一致）。

四个主配对**全部为负**——真实特征从未赢过基线或其同维度置换对照。最一致
的是 `F1 − FS1`：6 个 session 全负、3 个 seed 全负。但 `2σ_delta` 为
`0.048–0.077`，`|delta|` 均未超过，故按预注册规则只能判 `indeterminate`，
**不得写成阴性结论**。

### 反常现象：置换对照赢过基线

`FS2 − F0 = +0.0355`（3/3 seed 为正）、`FS1 − F0 = +0.0133`。即：给 `ψ`
加宽度（喂打乱的无意义数值）略有帮助，而喂**真实特征值**反而有害。

曾提出的机制假设：波形标量随记录日系统性漂移（电极阻抗、sorting 阈值），
而归一化只用 train session 统计量（章程 §6.1），于是漂移变成注入 identity
的 session 相关偏置。

> ### 该假设已被证伪（2026-07-26），且它本来就解释不了主要观测
>
> 诊断见 [`SIDE_FEATURE_DRIFT_DIAGNOSTIC.md`](SIDE_FEATURE_DRIFT_DIAGNOSTIC.md)。
>
> **实测层面：**
>
> - train→val 的 mean-z 偏移，36 个（特征 × val session）组合中只有 **1 个**
>   越过预注册的 `|mean z| > 0.5`；val session 的离散度**不大于** 27 个
>   train session 自身的内部离散度；
> - 6 个特征中只有 2 个（`noise_std`、`pt_width`）有统计上真实但**温和**的
>   时间趋势（R² ≈ 0.25–0.29，21 个月累计漂移 < 0.33 z）；
> - **零假设对照更致命**：故意选的"不该漂移"的特征（每 unit 的 pool 内 spike
>   计数）在每一项漂移指标上都**不低于**真实波形特征——between-session 方差
>   占比 `0.135` vs 真实特征的 `0.023–0.044`，累计漂移 `+0.550 z` 也大于任一
>   真实特征；
> - 每个真实特征 **95%+ 的 unit 级方差是 session 内**的，不是 session 间的。
>
> **更根本的结构性理由（这是我提假设时的分析错误）：**
>
> `load_unit_side_features` 构造 FS1/FS2 的方式是对**该 session 自身**的特征
> 矩阵做 **unit 维行置换**（`unit_side_features.py` 中
> `normalized = normalized[perm]`，`perm` 的长度是本 session 的 unit 数）。
> 行置换保持列统计量不变，因此 **FS1 与 F1 在每个 session 上的均值和方差
> 逐位相同**，只有 unit↔数值的配对被打乱。
>
> 于是任何 **session 级偏置在 `F1 − FS1` 中是共模的、精确抵消**。漂移假设
> 至多能解释 `F_x < F0`，**在结构上不可能解释 `F1 < FS1` / `F2 < FS2`**——
> 而后者恰恰是最一致的观测（`F1−FS1` 6/6 session 为负）。
>
> 教训：提出机制假设前，必须先检查**对照设计本身是否已经排除了该机制**。
>
> **诊断的局限**：只检验了一阶矩（均值）漂移，未检验方差/形状/协方差漂移，
> 且 val 只有 6 个 session（低功效）。因此不能完全排除 session 非平稳性作为
> **某一个**贡献因素，但所提的具体机制不被支持。

### 追加限制：所有 15 个 run 都在收敛前被截停（2026-07-26 发现）

E2 的 40-epoch 长 run（同 seed、同超参数、仅 `max_epochs` 不同）给出 B3 基线
在协议 epoch 5/8/11/14 上的分数为 `0.3110 / 0.3100 / 0.3497 / 0.3781`，
**斜率 `+0.00804/epoch`**——是 12-epoch 窗口内所测 `+0.0024` 的 3.3 倍，且
epoch 14 已高于 12-epoch 窗口内的任何值（最高 `0.3604`）。

> 顺带得到一个确定性验证：该 40-epoch run 在 epoch 5/8/11 的分数与
> `side_feature_ablation_v2` 的 `f0_s42` **逐位相同**，证实 M1/M2/M3 的
> 可复现性契约成立。

因此 `E=12` 的预算**明显偏短**，15 个 run 全部停在模型仍在快速改善的位置。
这给 §I 的 `indeterminate` 增加了第二重原因：不只是噪声受限，**还是预算受限**。
在基线仍以 `+0.008/epoch` 上升处比较架构，比较的成分里混入了"谁收敛得快"，
而不是纯粹的"谁最终更好"。

E2 完成后确定的新预算将写入
[`E3_E4_ENCODER_PROGRAM.md`](E3_E4_ENCODER_PROGRAM.md) §0，且**侧信息结论若要
翻案，必须在新预算下重跑**——现有 `indeterminate` 判定在旧预算下仍然有效，
但其证据强度比原先记录的更弱。

### 当前对侧信息结果的最简解读

漂移被排除后，最简约的解释是：**这些效应就是噪声**。`F1 − FS1 = −0.0457`
在 6/6 session 为负确实醒目，但 `2σ_delta = 0.061` 仍高于它。在 3 个 seed 下
无法进一步分辨；要分辨需要约 13 个 seed。

**结论：侧信息方向没有正向证据，表观的负向不能归因于漂移，且当前设计无力
再说更多。不建议继续追（约需 65 个 run），应转向降 `σ_seed`（E1）。**

### 方差分解（本 screen 最重要的方法学产出）

| 分量 | 实测 |
|---|---:|
| 窗口内 std（15 runs 均值） | 0.0345 |
| → `σ_run`（8-epoch 平均后） | **0.0122** |
| 跨 seed std（5 组均值） | **0.0385** |
| → `σ_delta`（3 seeds） | 0.024–0.039 |
| → `2σ_delta` | **0.048–0.077** |

**`σ_seed` 是主导项，且 M3 完全没有触及它。** M3 把 epoch 分量从 v3 的
`0.0388` 压到 `0.0122`（burn-in 排除贡献了约一半，超出预期），但 seed 间
方差更大且不受平均影响。

后果：**`+0.03` 的门槛再次落在噪声底之下**。要在 2σ 水平分辨 `+0.03`
需要约 **13 个 seed**，不是 3 个。这与 v3 的 `+0.005` 是同一类错误，
只是没那么极端。详见 [`MEASUREMENT_PROTOCOL_V4.md`](MEASUREMENT_PROTOCOL_V4.md) §4.1 的修订。

### 已知缺陷

聚合器的 `sigma_delta = sqrt(σ_A² + σ_B²)` **漏了 `/√3`**——那是单 seed
delta 的标准差，不是 3-seed 均值 delta 的标准误，偏保守约 1.73 倍
（`F1−F0` 阈值报 `0.0836`，正确值 `0.0482`）。今日判定不受影响（两种算法
都给 `indeterminate`），但它会永久性偏向 `indeterminate`，必须修。

## J. E1/E2/E4（2026-07-26 完成）

估计量与判定遵循 [`MEASUREMENT_PROTOCOL_V4.md`](MEASUREMENT_PROTOCOL_V4.md)，
章程见 [`E3_E4_ENCODER_PROGRAM.md`](E3_E4_ENCODER_PROGRAM.md)。

### J.0 流水线确定性验证（通过）

E4 的 B3 臂与 `side_feature_ablation_v2` 的 `f0` 在三个 seed 上分别相差
`1.7e-7 / 4.6e-6 / 3.4e-5`——**两个独立 screen、同 seed 同配置，结果一致**。
M1/M2/M3 的可复现契约成立。

### J.1 E2 收敛性：epoch 预算不是杠杆

40 epoch × 3 seed。**不存在跨 seed 共享的收敛点**：seed 42 在 epoch 14 后进入
`0.357` 平台，seed 43 在 epoch 8 达峰后一路下滑（同区段均值 `0.265`）。

| | epoch ≤11 | epoch ≥20 |
|---|---:|---:|
| 2-seed 平均 R² | 0.3131 | 0.3000 |
| seed 间差 | 0.0436 | **0.0749（1.7×）** |

```
2-seed 均值斜率 = −0.00072/ep   训练更久，平均无收益
seed 间差斜率   = +0.00047/ep   训练更久，seed 越发散
```

**结论**：延长训练不改善平均表现，反而放大 seed 分歧。E3/E4 因此沿用
`E=12, burn_in=4`，与既有 screen 完全可比。

### J.2 E1 SWA（次要，仅作记录）

> **SWA 是通用训练技巧，不构成本项目的科学贡献，此处仅记录数值，不作为
> headline。** 它对信号类型无关，与 SUA/MUA、架构复用等主张均无关系。

3 seeds 实测：SWA last20 相对旧协议 `+0.0223`（逐 seed `+0.0391/+0.0256/+0.0023`，
全为正，约 2 SE）。**但它不降低 `σ_seed`**（`0.032` vs 旧协议 `0.023`）——它只压
轨迹内的 epoch 抖动，而 `σ_seed` 是**轨迹之间**的差异。

对统计功效**无改善**；分辨 `+0.03` 仍需约 13 个 seed。

> **唯一值得保留的方法学区分**：SWA 的效应可测而架构效应难测，是因为前者是
> **轨迹内配对**（`σ_seed` 抵消），后者是**轨迹间**比较（`σ_seed` 全额进入）。
> 今后设计对照应优先构造轨迹内配对。

### J.3a `B3T` 六 seed 确认（2026-07-27）

E4 的 seeds `42/43/44` 之外补跑 `45/46/47`（`b3t_confirmation`，同配置同协议）：

```
seed:   42      43      44      45      46      47
B3:   0.3341  0.2886  0.3193  0.3093  0.2654  0.2830    均值 0.2999  σ_seed 0.0255
B3T:  0.3857  0.3170  0.3364  0.3314  0.3337  0.3463    均值 0.3418  σ_seed 0.0235
Δ:   +0.0516 +0.0284 +0.0172 +0.0222 +0.0683 +0.0633
```

| | 3 seeds | **6 seeds** |
|---|---:|---:|
| 配对 delta | +0.0324 | **+0.0418** |
| 配对 SE | 0.0101 | **0.0090** |
| 显著性 | 3.2 SE | **4.6 SE** |
| seed 为正 | 3/3 | **6/6** |
| per-session 为正 | 3/6 | **4/6** |

**加倍 seed 后效应量与显著性都上升**——与 SWA 的情形相反（那里 seed 44 把
`+0.030 ± 0.002` 打回 `+0.022 ± 0.019`）。这是本项目少见的、随样本增加而**加强**
的结果，支持 B3T 是真效应。

判定仍为 **`effective_heterogeneous`**：per-session 4/6 未达 `effective` 所需的
5/6，且**最差的一个 validation session 为 `−0.1344`**。seed 层面完全一致、
session 层面存在一个持续反向的 session——两点都是真实观测，不得只报其一。

### J.3b `B3T + SWA` 叠加（次要）

`B3T` 与 SWA 的增益近似可叠加（`B3T+SWA = 0.3719`，相对 B3 基线 `+0.0579`，
`+0.0324 + 0.0256 = +0.058`）。**但 SWA 部分不作为项目结论**（见 §J.2），
架构侧的可引用结果是 `B3T` 本身的 `+0.0418`（6 seeds，§J.3a）。

### J.3c 聚合器的 `σ_delta` 忽略了 seed 配对（待修）

聚合器用 `sqrt(σ_A²+σ_B²)/√n` 由两臂**独立**的跨 seed std 合成。该式假设两臂
的 seed 效应互不相关——**但它们用的是同一批 seed**，seed 难度是共享的、在差值中
大部分抵消。

以 `B3T − B3` 实测：

```
quadrature σ_delta                 = 0.0244
逐 seed 配对差 +0.0516/+0.0284/+0.0171
配对 std 0.0175  →  配对 SE        = 0.0101
隐含的两臂 seed 相关              ≈ 0.90
```

**quadrature 估计偏大约 2.4 倍**，会系统性地把真实效应误判为
`indeterminate`。按配对口径，`B3T − B3 = +0.0324 ± 0.0101`（3.2 SE），
`B3T+SWA vs B3 = +0.0579 ± 0.0095`（约 6 SE）。

这是同一个估计量的**第二个**缺陷（M7 修的是漏掉 `/√n`，这个是独立性假设）。
修复中：四个聚合器统一改用配对估计量，并同时输出两个值与隐含相关系数。

> 注意：`effective` 判定还要求 per-session 一致性（6 个 session 中至少 5 个
> 为正），而 `B3T − B3` 只有 3/6。**即使 σ 收紧，它也可能仍不满足该条款**——
> seed 层面高度一致、session 层面不一致，这两点都是真实观测。

### J.5 E3 方向调谐特征：项目首个 `effective` 判定（2026-07-27）

| 组 | s42 | s43 | s44 | 均值 | σ_seed |
|---|---:|---:|---:|---:|---:|
| F0（B3 基线） | 0.3341 | 0.2886 | 0.3193 | 0.3140 | 0.0232 |
| **T4**（余弦调谐拟合，4 维） | 0.5728 | 0.5544 | 0.5730 | **0.5667** | **0.0107** |
| **T8**（逐方向发放率，8 维） | 0.5795 | 0.5604 | 0.5724 | **0.5707** | **0.0097** |
| TS4（T4 的置换对照） | 0.3405 | 0.3052 | 0.2978 | 0.3145 | 0.0228 |
| TS8（T8 的置换对照） | 0.3105 | 0.3108 | 0.2843 | 0.3018 | 0.0152 |

| 配对 | mean | 2σ_paired | 为正 session | 逐 seed | 判定 |
|---|---:|---:|---:|---|---|
| T4 − F0 | **+0.2528** | 0.0157 | **6/6** | 全正 | **`effective`** |
| T4 − TS4 | **+0.2522** | 0.0249 | **6/6** | 全正 | **`effective`** |
| T8 − F0 | **+0.2567** | 0.0157 | **6/6** | 全正 | **`effective`** |
| T8 − TS8 | **+0.2689** | 0.0222 | **6/6** | 全正 | **`effective`** |

**这是本项目第一个 `effective`，且效应量是噪声底的约 16 倍。**

四条支撑：

1. **归因干净**：两个置换对照都落在基线上（`0.3145` / `0.3018` vs F0 `0.3140`），
   说明增益**完全来自特征内容**，与 `ψ` 变宽、参数增加无关。
2. **一致性完整**：6/6 session、3/3 seed 全部为正，无异质性问题（与 B3T 不同）。
3. **`σ_seed` 减半**（`0.0232` → `0.0107`/`0.0097`）。调谐特征不只提升性能，
   **还把一直卡住所有比较的主导方差源压下去了**。
4. **T8 ≈ T4**（`0.5707` vs `0.5667`）：4 参数余弦拟合已捕获几乎全部信号，
   与 M1 余弦调谐的经典认识一致。

#### 这验证了章程的事前判据

[`UNIT_SIDE_FEATURE_ABLATION.md`](UNIT_SIDE_FEATURE_ABLATION.md) §2 在实验前写明
`E_i` 需要编码**功能属性（tuning）**而非**解剖属性**。结果完全对上：**解剖属性
（波形）失败，功能属性（调谐）以 8 倍于其他所有效应的幅度成功。** 这是预注册
推理被证实，不是事后解释。

#### 强制对照之一：单 session 经典线性解码（2026-07-27 完成）

`scripts/linear_decoder_control_dandi688.py` → `results/linear_decoder_control.json`。
闭式拟合，仅用该 session 的 30 个标定 trial，评估窗口与神经路径完全一致
（复用 `load_session_with_trials` / `select_calibration_trial_indices` /
`_compute_valid_starts` 与同一 `R2Score`）。

| 解码器 | mean R² | vs F0 | vs T4 |
|---|---:|---:|---:|
| F0（神经，无侧信息） | 0.3140 | — | −0.2527 |
| **T4（神经 + 调谐）** | **0.5667** | +0.2528 | — |
| `ridge_raw_window`（经典，最好） | **0.3078** | −0.0062 | **−0.2590** |
| `ridge_pooled_rate` | 0.0887 | −0.2253 | −0.4780 |
| `population_vector` | 0.0870 | −0.2270 | −0.4798 |

**§1.3 字面担心的情形没有发生**：经典解码器拿不到 T4 的水平，最好的一个恰好
落在 F0 附近（Δ = −0.006）。

> ##### ⚠️ 但这个对照**信息不对等**，必须写明
>
> ```
> F0 神经网络          ←  27 个 train session 训练 + 该 session 30 个 activity trials
> T4/T8 神经网络       ←  同上 + 前 50 个 pool trials 的 target_dir/rate 侧信息
> 经典解码器           ←  仅该 session 30 个标定 trial
> ```
>
> 它证明的是「相对**真实 BCI 部署做法**（短标定块拟合线性解码器）我们大幅更好」，
> **不是**「增益来自架构而非调谐信息」。后者需要下面这个对照。

#### 强制对照之二：跨 session 调谐对齐的线性解码（2026-07-27 完成）

`scripts/tuning_aligned_linear_control_dandi688.py` →
`results/tuning_aligned_linear_control.json`。用各 session 标定期实测的调谐
（复用 E3 同一份调谐代码）把 unit 映射到 session 不变的固定维表示，在
**全部 27 个 train session 上池化拟合一个共享 ridge**（alpha 由
leave-one-training-session-out 选），应用到 6 个 validation session。

| 方法 | mean R² |
|---|---:|
| **T4（神经 + 调谐）** | **0.5667** |
| 单 session 全 N ridge（对照之一） | 0.3078 |
| **`dirbin_16_mweighted`（最好的跨 session 调谐对齐）** | **0.0679** |
| `dirbin_8` | 0.0607 |
| `tuning_proj`（权重线性依赖调谐参数） | 0.0289 |
| `pop_rate_only`（无调谐，仅跨 session 池化） | 0.0118 |

##### ⚠️ 该对照本身太弱，不能单独下结论

**所有跨 session 池化的调谐对齐表示都低于单 session 拟合的 ridge（0.3078）**，
而后者根本没有跨 session 池化。原因是把 ~50 个 unit 按偏好方向压进 8–16 个箱，
**丢掉的信息多于跨 session 池化换来的**。这检验的是"固定维瓶颈有多有损"，
不是"线性方法能不能做到"。

##### 但失败的方式本身有信息：经典线性方法的结构性两难

| 路线 | 保留 per-unit 分辨率 | 能跨 session 池化 | R² |
|---|---|---|---:|
| 单 session 拟合全 N | ✅ | ❌ | 0.308 |
| 投影到固定维再池化 | ❌ | ✅ | 0.068 |

**线性解码器需要固定输入维度，而 sorted unit 跨 session 不对应。** 任何固定维
投影都有损，任何单 session 拟合都无法池化；两条路都到不了 `0.567`。

**set-based 架构从构造上跳出该两难**：保留 `N` 个 unit token（不压维），
同时全部权重跨 session 共享（完全池化）。这是本对照对"架构贡献"给出的
**机制性陈述**，比单纯的数字差更有说服力。

顺带：`dirbin_16_mweighted` 相对 `pop_rate_only` 的 `+0.056` 说明该表示里
约 80% 的增益确实来自**调谐对齐**而非单纯的跨 session 池化——只是绝对量很小。

##### 必须写明的保留

最富表达力的经典替代是**每个 unit 的解码权重为其调谐的学习函数**
`w_i = f(tuning_i)`，对活动仍是线性的。本轮只测了最受限的形式
（`tuning_proj`，`f` 线性于 `[cosφ, sinφ, m, 1]`），得 `0.0289`。
**若 `f` 换成 MLP，它同样跳出两难——但那时它本质上就是本方法**，只是读出更简单。

**因此「经典 vs 本方法」的边界实际上就是 `f` 的表达力。** 不得声称跑赢了
"所有线性方法"；可声称的是：**该模式的自然线性退化形式全部远远不及。**

#### ~~强制对照之二（运行中）~~

最强的经典对手是「**我们的方法，但线性版**」：用各 session 自己标定期实测的调谐
把 unit 映射到**共同的功能坐标系**（按偏好方向分箱 / 投影到调谐参数），再在
**全部 27 个 train session 上拟合一个共享线性解码器**，应用到 6 个 validation
session。

- 若它也达到 ~0.55 → transformer 相对"线性调谐对齐"增量有限，E3 的表述必须改写；
- 若明显不及 → 差额即架构的贡献。

`scripts/tuning_aligned_linear_control_dandi688.py` 运行中。**在它出结果前，
E3 只能声称"调谐信息是关键，且需要超出单 session 经典解码的东西才能利用它"，
不能声称"架构本身是关键"。**

#### 神经家族内部的归因（独立于上述对照，已成立）

`F0 = 0.3140`、`TS4 = 0.3145`、`T4 = 0.5667`——同架构、同训练、唯一差别是调谐值
真假。因此 **`+0.25` 确定来自调谐内容**，这一条不依赖任何经典对照。

#### ~~⚠️ 强制未决对照：经典线性解码~~（已拆分为上面两条）

[`E3_E4_ENCODER_PROGRAM.md`](E3_E4_ENCODER_PROGRAM.md) §1.3 预先写死：余弦调谐拟合
本质上就是经典 population-vector / OLE 在算的东西，**若 E3 有效，必须做「直接用
调谐做线性解码」的对照**。

**在该对照完成前，E3 只能声称"调谐信息有用"（早已知），不能声称"本架构有用"。**
对照正在运行（`scripts/linear_decoder_control_dandi688.py`，闭式最小二乘，
只用标定 trial 拟合，评估窗口与神经网络路径完全一致）。

#### 方法主张的变化（必须显式声明）

加入调谐特征使方法主张从「identity **只从 spike 统计**得出」变为
「identity 来自一个**有监督的**标定块」。这仍然是 gradient-free、部署现实的
（标定本来就是让受试者做已知方向的运动），但**假设更强**，任何引用 E3 的
文档都必须写明，不得含糊带过。

### J.4 本轮最重要的方法学教训：小样本反复骗人

本次会话**三次**小样本读数被增加一个 seed 推翻：

| # | 被推翻的结论 | 推翻者 |
|---|---|---|
| 1 | epoch 预算 `E≈26`（基于 seed 42 的"平台"） | seed 43 |
| 2 | SWA 增益 `+0.030 ± 0.002`（2 seeds"高度一致"） | seed 44（真值 `+0.022 ± 0.019`） |
| 3 | "attention 已被否定"（v3） | 噪声底诊断 |

**在 `σ_seed ≈ 0.023–0.047` 的量级下，n=1 和 n=2 的读数没有信息量，
且会系统性地给出过度自信的结论。** 任何低于 3 个 seed 的方向性判断都不应
写入文档。

## K. SUA → pseudo-MUA T4 bridge 与 T4 gate 收口（2026-07-29）

### K.1 pseudo-MUA 冻结矩阵：9/9 完成

`pseudomua_t4_bridge_v1` 将同一 electrode 上的 sorted SUA spike counts 逐 bin
求和，并在聚合后重新拟合 electrode-level T4；不是对 unit-level T4 求平均。
冻结矩阵为 F0/T4/TS4 × seeds 42/43/44，训练协议与 SUA E3 完全相同：
`E=12`、`burn_in=4`、epoch 5–12 平均、`first/n=30/pool=50`、27/6/6
session-disjoint split。

9/9 个 run 于 2026-07-28 04:31 HKT 完成，严格聚合写入
`results/pseudomua_t4_bridge_v1/summary.json`。每个 artifact 都通过
signal-view、split、epoch-window、run-metadata SHA-256 与 no-test provenance
校验；`no_test_files_evaluated=true`。

| View | F0 | T4 | TS4 | T4 − F0 | paired SE | T4 − TS4 | paired SE | 组级判定 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| SUA | 0.313987 | **0.566749** | 0.314516 | **+0.252761** | 0.007844 | **+0.252233** | 0.012449 | **`effective`** |
| pseudo-MUA | 0.208382 | **0.526121** | 0.160447 | **+0.317739** | 0.023856 | **+0.365674** | 0.030763 | **`effective`** |

四个主要配对均为 6/6 validation session、3/3 seed 全正。由此可支持：

- T4 的价值不是 sorted-unit view 独有；electrode pooling 后增益明确保留；
- 正确的 channel–T4 对应关系是承重信息，TS4 不能解释 T4 的收益；
- pseudo-MUA 是由同一批 SUA 构造的受控 bridge，**不是**真实 threshold-crossing
  MUA 外部复验。

### K.2 SUA 是否显著优于 pseudo-MUA：绝对分数与增益必须分开

以下 `SUA − pseudo-MUA` 是基于同 seed、同 session 的**事后配对诊断**，不是
bridge 章程预注册的主要终点。两种 view 使用相同 session、行为目标与评价协议，
因此绝对 R² 可在本 bridge 内比较；但不得与 FALCON M2 等不同数据集横向比较。

| Arm | SUA − pseudo-MUA | paired SE | 2SE 区间 | 正 seed | 正 session |
|---|---:|---:|---:|---:|---:|
| F0 | **+0.105605** | 0.019942 | `[+0.065720,+0.145490]` | 3/3 | 5/6 |
| T4 | **+0.040627** | 0.015256 | `[+0.010116,+0.071139]` | 3/3 | 5/6 |
| TS4 | +0.154069 | 0.020408 | `[+0.113252,+0.194885]` | 3/3 | 6/6 |

因此问题“**SUA 是否相对 pseudo-MUA 有很大的提升**”的准确回答是：

1. **无 T4 时有清楚的 SUA 优势**：F0 高 `+0.1056 R²`；
2. **有 T4 后只剩较小的 residual SUA advantage**：T4 高 `+0.0406 R²`，
   数值约为 F0 view gap 的 38.5%；
3. T4 把观察到的 SUA–pseudo-MUA gap 缩小 `0.0650 R²`（约 61.5%），因为
   pseudo-MUA 的 T4 增益比 SUA 大 `0.0650`；
4. `+0.0406` 的 2SE 区间仍在 0 以上，说明不能说两种 view 完全等价；但它远小于
   T4 自身的 `+0.25–0.32` 增益，**不构成“很大提升”**。

预注册 interaction：

```text
Gamma = (T4-F0)_SUA - (T4-F0)_pseudo-MUA
      = -0.064978 ± 0.030744 SE
2SE   = [-0.126466, -0.003489]
```

三个逐 seed Gamma 均为负，但 pseudo-MUA amplification 的冻结条件要求
`Gamma + 2SE < -0.03`；当前上界 `−0.003489` 未越过实用容忍带，故正式 verdict
仍是 **`indeterminate`**。这一区分防止把“区间排除 0”偷换为“view interaction
大到具有预注册的部署相关性”。

### K.3 T4 上的静态 electrode reliability gate：`ineffective`

`t4_gate_screen` 已在 T4 substrate 上完成 T4/T4GATE/T4GATE_SHUFFLED ×
seeds 42/43/44 共 9 个 run。2026-07-29 用冻结 aggregator 补写
`results/t4_gate_screen/aggregate.json`：

| Group | mean R² |
|---|---:|
| T4 | **0.566749** |
| T4GATE | 0.555931 |
| T4GATE_SHUFFLED | 0.561363 |

| 配对 | mean delta | paired SE | 正 session | 判定 |
|---|---:|---:|---:|---|
| T4GATE − T4 | **−0.010817** | 0.004873 | 1/6 | **`ineffective`** |
| T4GATE − T4GATE_SHUFFLED | **−0.005432** | 0.005779 | 1/6 | **`ineffective`** |

两个 pair 的 `mean+2SE` 都低于 `+0.03` 门槛，组级 `T4GATE` 因而为
**`ineffective`**。这排除了“用一个跨 session 固定的 electrode-index scalar
对 T4 identity 做乘法可靠性门控”作为当前优化路线；design C/A 虽已实现但未跑，
不应在没有新机制理由时继续堆叠 array-specific electrode tables。

更重要的诊断是：T4 的巨大收益来自**每个新 session 重新测得的功能属性**，
而静态 electrode identity 没有提供额外信息。下一轮应优先优化功能表示、估计
置信度与 T4–activity 交互，而非继续扩展固定 electrode lookup。

### K.4 原生 FALCON MUA 的 T4 验证：M1/M2 18/18 完成

为避免把 pseudo-MUA bridge 当作原生 MUA 证据，已单独实现并冻结
M1/000941 与 M2/000953 的 calibration-only T4。这里的输入是 FALCON NWB
`units` 经 20 ms bin 得到的 64/96 路 per-electrode/Units spike counts，不是
由 DANDI SUA 合成的 pseudo-MUA。NWB 没有一个显式字段把 Units 命名为
“MUA”，故论文应使用上述精确表述。

标签 gate 已通过：M1 的 `trials.tgt_loc` 为逐 trial 方位角；M2 为相对
`(0.5,0.5)` 的目标坐标，中心/rest trial 不拟方向。所有 held-in calibration
session 的 NWB trial 数与 loader `trial_change` 对齐；固定首支持窗口
M1 `N=10`、M2 `N=33` 的 `[1,cos(theta),sin(theta)]` 设计均为 rank 3。
T4 数据模块只读对应 calibration NWB 的 target metadata，并在任何 held-out
加载请求下 fail closed。

冻结矩阵为每任务 F0/T4/TS4 ×
`(fold1,seed42)/(fold1,seed43)/(fold2,seed42)`，三组统一首支持、12 epochs、
无 early stopping、frozen decoder、internal LOSO 且 held-out fit/test 均为
false。真实 M1/M2 GPU forward/backward smoke 均 exit 0；正式
`native_mua_t4_v1` 于 2026-07-29 14:06 HKT 在两张 RTX 3090 上运行。
GPU0 运行 M1、GPU1 运行 M2，各自优先完成首个 F0→T4→TS4 配对。

M2 已于 2026-07-29 完成 9/9 并通过 strict aggregator：

| Task | F0 | T4 | TS4 | T4−F0 | 正 cell | T4−TS4 | 正 cell |
|---|---:|---:|---:|---:|---:|---:|---:|
| M2 | 0.527791 | **0.624124** | 0.588357 | **+0.096334** | 3/3 | +0.035767 | 2/3 |

M2 的 `T4−F0` 三个 cell 为
`+0.073464/+0.094278/+0.121259`，说明 side-feature network path 相对无
side-feature F0 在 M2 上一致有效。但 `T4−TS4` 为
`−0.057007/+0.028527/+0.135781`，正确 T4 内容相对 shuffled T4 的机制优势
并不稳定。

M1 也已完成 9/9 并通过 strict aggregator：

| Task | F0 | T4 | TS4 | T4−F0 | 正 cell | T4−TS4 | 正 cell |
|---|---:|---:|---:|---:|---:|---:|---:|
| M1 | 0.591662 | **0.599500** | 0.574703 | +0.007837 | 1/3 | +0.024797 | 3/3 |

M1 的 `T4−F0` 逐 cell 为
`+0.039790/−0.003559/−0.012720`，没有稳定净提升；`T4−TS4` 为
`+0.049554/+0.024130/+0.000707`，虽然 3/3 为正，但 fold-2 的幅度几乎为零。

所以当前结论不是“T4 对 MUA 稳定”或“T4 对 MUA 无效”，而是：
**M2 相对 F0 明显且一致，M1 相对 F0 很小且不一致；正确 T4 内容相对 TS4
在 M1 一致但幅度偏小，在 M2 则异质。** 两个任务没有给出统一的性能/机制图景。
这属于 native FALCON internal-LOSO development evidence，不是 formal
held-out/few-shot 结论。协议、label provenance、PIDs、日志位置与故障恢复记录见
[`NATIVE_MUA_T4_M1_M2_PROGRAM.md`](NATIVE_MUA_T4_M1_M2_PROGRAM.md)。

### K.5 SUA 同电极 relation：最终 verdict 为 ineffective

`sub-C/CO/units<100/27-6-6` 的 calibration-only audit 已完成，formal test
未打开。六个 validation session 中，落在 multi-unit electrode 上的 unit 比例为
`57.9/55.9/58.3/61.5/60.7/38.1%`，因此 5/6 ≥50%；全部 session 的 T4
方向设计 rank 3、condition `sqrt(2)`，同电极 T4 与 50-trial activity
heterogeneity 均非零。故“同电极 relation 在该数据上是否有对象可建模”的
eligibility gate 通过。

但这**不解释**当前 SUA–pseudo-MUA gap：n=6 的描述性 Pearson 相关中，
T4 view gap 对 multi-unit unit share 仅 `r=+0.036`，对 T4 heterogeneity
仅 `r=−0.047`。当前 artifact 也没有 per-electrode decoder R²，不能制造
“按电极分层的性能归因”。SNR、waveform residual CV 和 template drift 已降为
read-only negative diagnostics，不是默认模型输入或 headline。

CPU 组件与真实 cache smoke、扩展隔离/数据路径测试及 relation encoder 测试通过；它们验证 zero-init FiLM、
同电极 segmented-mean relation、membership shuffle、no-group 和近参数匹配
concat 的 tensor/gradient contract，不含 behavior label 或 R²。主审删除了
concat 对照中仅为凑参数数而加入的冗余 offset 参数；当前真实 MLP 为 1168
参数，对应 FiLM 1232 参数（`−5.2%`），差异显式记录而非伪装成完全相等。

端到端 fresh-token 路径已经完成：`B3S/t4`、`B3SER/t4rel`、
`B3SER/t4rel_membership_shuffled`、`B3SERN/t4rel_nogroup`。冻结
train/validation manifest 只解析 27+6 个允许 session 的路径，formal-test 六项
仅保留名字 receipt；训练与评分均校验 manifest SHA-256。训练 activity
calibration 固定为 10 trials，评分 forward calibration 固定为 30 trials。

完整 `T4/REL/REL-MS/REL-NG × seeds 42/43/44` validation-only 矩阵已完成，
并通过原始 artifact 重读、manifest/hash、calibration、epoch-window 与
no-formal-test 守卫：

| Seed | T4 | REL | REL-MS | REL-NG | REL−T4 |
|---:|---:|---:|---:|---:|---:|
| 42 | 0.572817 | 0.572654 | 0.569188 | **0.579878** | −0.000162 |
| 43 | 0.554434 | 0.553393 | 0.530710 | 0.546593 | −0.001041 |
| 44 | 0.572995 | 0.569879 | 0.545061 | 0.549966 | −0.003116 |

seed44 的 fresh T4 与历史 E3 artifact 的全部 epoch 分数和总分逐值一致，
排除了基线实现漂移。最终跨 seed 严格结果为：

| 必要比较 | mean ΔR² ± paired SE | 正 seed | 正 session | 上两-SE 界 | 判定 |
|---|---:|---:|---:|---:|---|
| REL−T4 | **−0.001440 ± 0.000876** | 0/3 | 1/6 | +0.000311 | **ineffective** |
| REL−REL-MS | +0.016989 ± 0.006790 | 3/3 | 5/6 | +0.030568 | indeterminate |
| REL−REL-NG | +0.006496 ± 0.007835 | 2/3 | 3/6 | +0.022166 | **ineffective** |

三个必要比较必须全部通过；其中 REL−T4 与 REL−REL-NG 已各自排除
`+0.03`，且 REL−T4 在三个 seed 上全部为负。因此 final group verdict 为
`ineffective`。不补 seeds 45–47，不进入 relative-amplitude 阶段，也不为这条
失败的 relation 路径继续做硬件优化。当前结果不支持把同电极 membership
作为 T4 后 SUA 优化方向。
详见 [`SUA_AUXILIARY_STAGE0.md`](SUA_AUXILIARY_STAGE0.md)。

## 当前证据边界

### 可以写入摘要

- 同一轻量架构可分别适配 SUA 和 MUA。
- 当前 B3 权重不支持 MUA→SUA 零样本迁移。
- **calibration identity 是承重部件**：置零后 R² 全部转为强负值（§F.1）。
- **标定期 T4/T8 功能特征有效**：SUA 上约 `+0.25 R²`，正确内容对照为
  6/6 session、3/3 seed 全正（§J.5）。
- **T4 增益在 pseudo-MUA 上保留**：相对 F0 `+0.3177`、相对 TS4
  `+0.3657`，均为 `effective`（§K.1）。
- **T4 后的 SUA residual advantage 较小而非“很大”**：同数据同协议的
  exploratory paired gap 为 `+0.0406 ± 0.0153 SE`（§K.2）。
- **静态 electrode-index reliability gate 无效**：T4GATE 组级
  `ineffective`（§K.3）。
- **当前同电极 relation 不改善 T4 后的 SUA**：final strict
  `REL−T4=−0.001440`，3/3 seed 为负，且未达到相对参数匹配 no-group 的
  `+0.03` 门槛（§K.5）。
- 固定 `K` slot 接口的可部署性（固定 shape、缓存等价、压缩率）已验证（§G）。
- B15 需拆分跨 neuron attention 与额外 per-neuron capacity/LayerNorm 的贡献
  ——**该拆分已完成，结论为后者**。

### 暂时不能写入摘要

- B15 或 B16 已经稳定优于 B3 或 teacher。
- **attention 已被否定**（§E 已撤回；当前是未知，不是阴性）。
- **`attention_arch_screen_v3` 的任何变体排名**（§H）。
- B15 的增益来自 spike sorting split/merge。
- B16 的微小增益是 MUA/SUA 通用机制。
- 18K 参数模型绝对无法通过联合训练获得跨信号权重共享。
- T4 的收益已经证明来自“方向”本身：当前 T4 同时含 `[a,c,m,b]`，还需分量
  消融区分 preferred direction、modulation depth 与 baseline rate。
- pseudo-MUA 的 T4 增益已被证明显著大于 SUA：预注册 Gamma 仍为
  `indeterminate`（§K.2）。
- pseudo-MUA bridge 等同于真实 threshold-crossing MUA 外部复验。
- T4 已经在原生 FALCON **M1/M2 都稳定有效**：M2 相对 F0 一致为正，但 M1
  相对 F0 只有 1/3 cell 为正，且 M2 相对 TS4 也异质（§K.4）。
- 同电极 relation 已提升 SUA，或已解释 SUA–pseudo-MUA gap：现有端到端
  validation R² 证据方向相反（§K.5）。
- **任何 formal held-out test 结论**：`sub-C/CO/27-6-6` 的全部数字都是
  validation development evidence（§D、§E、§G、§J、§K）。
- **`calibration ≫ 无标定模型`**：当前只有 `calibration ≫ 喂零`，
  learned-prior 对照失效（§F.2）。
- 固定 slot 接口的精度可用性（§G 精度 gate 未过）。

## 已知的产物缺陷（引用前必读）

| 缺陷 | 位置 | 影响 |
|---|---|---|
| `learned_prior` 对照退化为 zero-identity | `p3_no_calibration_validation_{b15,b16}_learnedprior_s42.json`、`p3_fair_baseline_summary_s42.json` | `delta_vs_learned_prior` 字段不可引用（§F.2） |
| formal-test receipt 悬空 | `p3_formal_test_816cdd8b…_receipt.json` | `status="started"`（2026-07-24T22:52），无对应结果文件；该 scope 已被占用且按预注册规则不可重跑 |
| 40-epoch learned-prior run 挂起 | `checkpoints/b16_dandi688_co_learnedprior_s42/` | 无 checkpoint 产出（§F.3） |
| **测量噪声底 > 被测效应** | 全部 `attention_arch_screen_v3` 工件 | §E 结论撤回；`+0.005` 门槛不可达（§H.1） |
| **不均等 max-of-N 选择偏置** | 同上 | 训练更久的变体被系统性高估（§H.2） |
| **MUA fold1 两个 seed 共用 run 目录** | `logs/train/runs/2026-07-25-07-11-45`、`...-09-25-11` | B15P/B3 的 seed 归属不可考（§H.4） |
| **数据隔离违规（已纠正）** | `UNIT_SIDE_FEATURE_ABLATION.md` §3 曾使用 test session `20151119` 的 spike/waveform/trial 数据 | 仅用于成本量级估算，未进入训练/选择/评估/门槛；共享缓存经枚举核验无 test 派生条目（0/48）。已替换为 train session 并留存违规记录 |
| **E3/E4 实现期违规（已纠正）** | 实现 subagent 曾在一个临时脚本与一个测试中使用 `20151119` | 自查发现并全部替换为 validation session；所有临时检查使用自动删除的 `TemporaryDirectory`，未写入共享缓存（已独立核验） |

## 关键产物

- Teacher：`sua_exploration/checkpoints/teacher_mc_maze/best-epoch=083-val_heldin/r2_mean=0.9061.ckpt`
- 严格 B3 seed-42：`sua_exploration/checkpoints/b3_mc_maze_v2_s42/best-epoch=036-val_heldin/r2_mean=0.9078.ckpt`
- 严格 B15 seed-42：`sua_exploration/checkpoints/b15_mc_maze_v2_s42/best-epoch=013-val_heldin/r2_mean=0.9098.ckpt`
- 严格 B16 seed-42：`sua_exploration/checkpoints/b16_mc_maze_v2_s42/best-epoch=035-val_heldin/r2_mean=0.9085.ckpt`
- 旧 B3：`sua_exploration/checkpoints/b3_mc_maze/best-epoch=019-val_heldin/r2_mean=0.8387.ckpt`
- 旧 B15：`sua_exploration/checkpoints/b15_mc_maze/best-epoch=016-val_heldin/r2_mean=0.9092.ckpt`
- 旧 B16：`sua_exploration/checkpoints/b16_mc_maze/best-epoch=017-val_heldin/r2_mean=0.9020.ckpt`
- 完整比较 JSON：`sua_exploration/results/p0_matched_e18_comparison.json`
- 严格 seed-42 完整比较：`sua_exploration/results/p0_s42_full_comparison.json`
- seed-42 机制诊断：`sua_exploration/results/p0_s42_mechanism_diagnostics.json`
- 训练：`sua_exploration/scripts/train_variant_mc_maze.py`
- 比较：`sua_exploration/scripts/compare_neuronid_variants.py`
- 数据：`sua_exploration/mc_maze/datamodule.py`
- Encoder：`streaming_calibration_exp/src/models/components/streaming_encoders.py`
- MUA B16：`streaming_calibration_exp/outputs/streaming_calibration/b16_m2_loso_f0_s42_20260722_160720/`

下一步及验收标准见 [`../ROADMAP.md`](../ROADMAP.md)。
