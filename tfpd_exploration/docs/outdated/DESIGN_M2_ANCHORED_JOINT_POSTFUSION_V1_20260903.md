# M2 Anchored Joint Post-Fusion（AJPF）V1 设计

日期：2026-09-03  
状态：**设计候选；尚未授权执行**  
设备边界：只允许 GPU0；GPU1 属于并行 M1 工作，禁止查询、占用、发信号或改变其进程  
网络边界：不包含 EvalAI、ECR、镜像推送或任何网络提交

## 1. 研究问题

已有 Post-Fusion（PF）结果不能支持“晚池化本身没有能力”这一广义结论。
现有 12-epoch PF screen 从 native early-pooling 初始化出发，但训练时 decoder
保持冻结，只更新 identity path（以及 PF-R1/PF-R50 的残差参数）。因此它真正回答的是：

> 在为 early pooling 训练好的冻结 decoder 上，只重新训练 identity path 12 epoch，
> 无法让 late pooling 逼近或超过强 POOLED 解。

它没有测试 decoder 是否能与 pooling 位置共同适配。AJPF 的问题是：

> 从同一个强 Selected-T4 checkpoint 出发，在 native 和 PF 臂中解冻完全相同的
> student decoder/encoder 参数后，锚定 late-pooling residual 能否产生至少
> `+0.010` external R2 的匹配增益？

本实验是 warm-start joint adaptation，不得称为“从头训练”。真正的随机初始化
late-pooling 需要另一个 matched native-from-scratch 对照和更长训练预算，V1 不做。

## 2. 现有证据与为什么还值得做

### 2.1 已知负结果

- 纠正后的 PF-R1 external R2 为 `0.2383703`。
- PF-R1 相对 PF-MEAN 为 `+0.051466`，external `4/6` 为正。
- 但 PF-R1 相对强匹配 POOLED 仍为 `-0.060687`，只有 `1/6` 为正。
- PF-R50 相对 POOLED 为 `-0.065510`；PF-MEAN 相对 POOLED 为 `-0.112153`。

这些数字证明 late branch 不是零信息，也证明 scalar anchored residual 明显优于
完全替换或 50-bin gate；但在既有的 **teacher-initialized、decoder-frozen、
`task_plus_y_plus_E`** 合同中，信息没有被有效吸收。AJPF 是新的 matched joint
adaptation 可行性实验，不把与旧实验的差异因果归结为 decoder 解冻这一项。

### 2.2 小而稳定的互补信号

- AOF-S source OOF：`+0.006358371947682961`，`6/7` 为正，最差
  `-0.0021069852144761647`。
- AOF-M matrix-vs-native：`+0.005617074728110476`，`5/7` 为正；但相对
  matched scalar 为 `-0.0007412972195724851`，所以更复杂矩阵门无价值。
- APFG external：`+0.00461224`，`4/6` 为正，但未达 `+0.010`。

这组证据支持一个窄假设：late branch 有少量互补信息，但低容量后处理/单参数门
只能取回约 `+0.005`。若 decoder 共同适配仍不能把它放大到 `+0.010`，PF 在当前
数据与架构上的实用路线即可关闭。

## 3. 方法与三臂矩阵

统一方法名：**AJPF（Anchored Joint Post-Fusion）**。

| 臂 | 训练 identity 算子 | 初始预测 | trainable 参数 | 作用 |
|---|---|---:|---|---|
| `J-NATIVE` | 原生 `post_pool(mean(pre_pool(trial)))` | Selected-T4 native | 全部 student encoder + decoder | 匹配联合微调对照 |
| `J-R1` | `h_n + tanh(alpha) * (h_p - h_n)` | `alpha=+0`，与 native bitwise 相同 | 全部 student encoder + decoder + 1 scalar | **主方法** |
| `J-MEAN` | `mean(post_pool(pre_pool(trial), T4))` | 纯 late pooling | 全部 student encoder + decoder | 机制臂：联合适配能否吸收 placement |

其中：

```text
h_n = native.forward_batch(ordered_activity_stack, side)
h_p = arrival_order_float32_mean(
        post_pool(concat(pre_pool(each_trial), expanded_side)))
```

`J-R1` 必须以 IEEE `+0` 初始化；`-0` 拒绝。第一步前，对完全相同的 calibration
tensor、side tensor 和 neural window，`J-R1` 与 `J-NATIVE` 的 identity、prediction
和 task loss 必须 bitwise 相同。`J-MEAN` 不要求相同，它就是完整 placement 替换。

不再运行 PF-R50：现有证据显示 50 参数门不优于 scalar，继续增加 gate 自由度没有依据。

`J-R1` 比 `J-NATIVE` 多一个可训练 scalar `alpha`，因此三臂不是严格 equal-parameter
capacity。这个差异在 V1 中作为方法定义的一部分预先披露；不得把 `J-R1` 的胜出描述为
“完全同参数量”的胜出。由于只差一个标量，V1 不另设 inert-scalar placebo；若结果仅在
`+0.005～+0.010` 的灰区，必须把这一容量差异列为限制，而不能据此升级方法主张。

## 4. 共同初始化

三个臂都必须从同一个完整 Selected-T4 Lightning checkpoint 加载：

```text
path:
  streaming_calibration_exp/outputs/streaming_calibration/
  m2_spint_t4_mainline_fp32_v1_t4_m2_s42_20260730_131806/checkpoints/best.ckpt
sha256:
  25d7bc72b4d440004b58f1beaeadb7e15565a43e83dd1eadd160374270ec1d3e
student-state authority:
  2a340745e2e1b4c7eecb4b187c9b061548bf3686f12389aae76bf53e4f3acc20
```

执行顺序固定为：

1. source-only PIT DataModule 构造一次；不得 materialize held-out/target。
2. 构造一个 native module，no-follow 读取并 SHA 验证 checkpoint。
3. 对完整 Lightning `state_dict` 做 strict load。
4. 从 strict-loaded native module 深拷贝三份。
5. `J-NATIVE` 保持原 encoder；`J-R1/J-MEAN` 在已加载的 native encoder 外安装
   route-owned PF adapter。不得先用随机 PF wrapper 接收 historical checkpoint。
6. 逐臂验证安装 wrapper 前后的 native 子状态 SHA 完全一致。
7. 只解冻 `student.id_encoder` 与 `student.decoder`；teacher、数据统计量、T4、normalizer
   和任何 target 量保持冻结。
8. 解冻不能只改 `requires_grad`。strict-loaded 模型继承
   `student._decoder_frozen=True`；route 必须在参数检查之后显式把它改成 `False`，再调用
   `student.train(True)`，并证明 `student.decoder.training=True`。否则
   `StreamingSpintModel.train()` 会把 decoder 强制留在 eval，且
   `decode_with_identity()` 会跳过训练期 dynamic/unit dropout，这不属于 AJPF。
9. 不得调用继承的 `LightningModule.configure_optimizers()`：其 `_freeze_decoder` 配置仍是
   historical frozen 值。AJPF 使用 route-owned fresh Adam，并逐名验证两个互斥参数组：
   `student.id_encoder.*`（含 `J-R1` 的 `alpha`）恰好属于 `1e-4` 组，
   `student.decoder.*` 恰好属于 `1e-5` 组；两组交集为空，其并集等于全部且仅全部
   AJPF trainable student 参数，teacher 参数一个也不能进入。

现有 PF screen 从 teacher-initialized PIT base 开始，而本设计从强 `25d7...` 解开始。
这一区别必须写进结果，不能把 AJPF 与旧 PF screen 的训练曲线直接作因果比较。

## 5. 训练合同

### 5.1 固定预算

- seed：`42`
- epoch：三个臂都严格 `12`
- batch size：`32`
- calibration exposure：复用 reviewed G00m/source-replay 的因果原始活动池，沿用 APFG
  已执行并落收据的 frozen cycle `M in (4, 10, 30)`，精确 controller 为
  `POOL_CYCLE[(epoch_one_indexed-1 + canonical_batch_ordinal) % 3]`；每个 state 以 D-opt4
  support 为不可驱逐前缀，再加入该 query
  之前已完成的 non-support trials。请求 M10/M30 时分别取最新 6/26 个 eligible completion，
  同一 global step 三臂取完全相同的 ordered member indices。不得退回 chronological
  `range(M)` 静态 calibration prefix。canonical stream 在 materialization 阶段只收录已经
  M30-ready 的 coordinates，因此进入训练的每个 state 都必须同时可构造 M4/M10/M30；若
  runtime 仍出现不足，应 fail-closed，不能跳过。禁止 clamp、pad、repeat、future-trial
  substitution 或跨 trial history。
- source roster 固定为全部七个 source session，按 lexical session、query trial、window-start
  排序；每个 epoch 恰好 `3455` 个 shared batch group，12 epoch 合计 `41460` 个 shared
  groups。该计数来自已封 APFG source authority 的 `91717` 个 M30-causal coordinates；
  live materialization 必须重算并精确匹配 coordinate/batch digests，任何缺失或漂移都停止。
- optimizer：fresh Adam；weight decay `0`；无 scheduler。
- encoder/alpha LR：`1e-4`
- decoder LR：`1e-5`
- checkpoint：固定 epoch 12；不得按 source 或 external 数字挑 epoch。
- loss：route-owned M2 task-only、last-bin、behavior scale 5.0；不加入 distillation、对比损失、
  output smoothing 或 target adaptation。

decoder 使用比 encoder 小 10 倍的学习率，是因为起点已经是强部署 checkpoint；V1 的
目标是允许 pooling co-adaptation，同时避免把整个 decoder 以 `1e-4` 快速洗掉。三个臂
必须使用完全相同的两组 LR 与参数归组规则。不得看到结果后另开 LR sweep。

这不是旧 PF screen 的 loss 复现：旧 resolved config 的 governing loss 是
`task_plus_y_plus_E`（含 prediction/identity distillation），而 AJPF 有意使用不调用 teacher
的 task-only loss，使 late branch 有机会和 decoder 一起移动。因此 AJPF 只能支持
“warm-start joint placement adaptation 是否有效”，不能单独支持“旧 PF 仅因 decoder
冻结而失败”的因果结论。route 不能调用 inherited `model_step`，训练和 descriptive monitor
都必须直接使用 student prediction 与真实 source target；training/monitor 的
`teacher_forward_calls` 分别记录且都必须为 0。

### 5.1.1 部署匹配的 activity / carrier 语义

训练与评分都固定使用 POOLED/G00m 的 linear raw-activity authority；不得把 PIT DataLoader
内的 cubic/interpolated calibration tensor 当成另一种活动表示。每个 session 先在 first-30
中做一次 frozen D-opt4，然后只用这四个 selected support trials 拟合固定 ridge T4。
该 selected-support4 carrier 在 M4/M10/M30 三种 activity exposure 中都不更新，经过固定
`0.020` Hz→counts/bin 换算和 source normalizer 后送入 B3S。

`raw_m30_t4_*` 仍必须记录，但它只是 first-30 审计对照，**不是**本实验送入 decoder 的
carrier。不得沿用 DataLoader 随原始 `calib_n_trials=33` 生成的 side，也不得随 M 重估
carrier。收据必须分列：D-opt4 indices、ordered activity member indices、selected-support4
raw T4、counts-per-20ms-bin T4、normalized side、raw-M30 audit T4 与 normalizer SHA。

source 训练坐标必须来自 reviewed causal replay：窗口 endpoint 只能读取严格早于它完成的
trial；同一 query-trial / pool-state 的窗口组成最多 32 个样本的 resident batch。所有
source state、target window 和 activity 都在 attempt 后一次物化，三个 arm 共享；不得调用
普通随机 calibration DataLoader 来悄悄恢复旧 static-prefix 分布。

### 5.2 配对随机性

每个 resident batch 只从 loader 读取并转移到 GPU0 一次。三个臂按固定顺序
`J-NATIVE -> J-R1 -> J-MEAN` 执行；每臂 step 前恢复完全相同的 Python、NumPy、Torch
CPU 和 Torch CUDA RNG state，三臂完成后只推进一次 canonical RNG state。

因为 decoder 已解冻，其 dynamic/unit dropout 重新进入训练路径。为避免三种 identity 图
在 decoder 前消耗 RNG 的差异，route 在每个 resident batch 上只生成一次 governing
dynamic-dropout probability 与 `[B,N]` scaled unit mask，然后让三臂复用同一 mask。三个
模型实例内建的 `decoder.dynamic_dropout/dropout_rate` 路径必须在 route clone 上禁用，
但 train mode 保持开启；显式 mask 在原 unit-dropout 乘法位置注入，数值语义与原算子一致。
任何 transformer 内部随机性继续通过 arm 前 RNG restore 配对。收据必须证明：

- 三臂每一步的 dropout 概率与 mask digest 相同；
- 三臂的 `student._decoder_frozen` 均为 `False`，decoder 均处于 train mode；每个
  shared step group 恰好发生一次 governing dropout draw，每臂恰好应用一次该显式 mask，
  每臂内建 unit-dropout draw 必须为 0；
- arm 内 forward/backward/Adam 各恰好一次；
- 每步两个参数组的所有已物化 gradient 均 finite；12-step smoke 和每个完整 epoch 内，
  encoder/alpha 组与 decoder 组都至少一次出现非零 group gradient 和非零 update；不得因
  单个参数或单个 batch 合法零梯度而误杀整次实验；
- teacher forward calls 为 0；
- 同一批次只 materialize/transfer 一次。

显式 mask 生成之后、每个 arm forward 之前恢复相同的 post-mask RNG state；三臂结束 RNG
必须完全一致。不能把 dropout 彻底关闭来换取表面配对。

### 5.3 source 使用与 checkpoint 选择

`25d7...` checkpoint 已见过七个 source session，因此任何这七个 session 的 monitor
都不是 OOF。V1 不得称其为 source OOF，也不得用它选择 arm、epoch 或 LR。

训练期间只记录（descriptive monitor 必须为 student-only eval，不得调用 inherited
`model_step` 或 teacher）：

- per-arm source task loss；
- 七个 source session 的 descriptive R2；
- epoch 11→12 loss/R2 slope；
- alpha trajectory（只对 `J-R1`）；
- encoder/decoder update norm 与 state digest。

三个 epoch-12 checkpoint 全部进入一次预先固定的 local external score；source 数字
不承担选择权。

## 6. GPU0 利用率方案

本实验不通过增大 batch 改变优化统计，也不启动多个互相竞争的训练进程。提高利用率的
方式是：

1. 三个完整模型同时常驻 GPU0；
2. 一个 source-only DataModule、一个 canonical causal-coordinate stream、一个 pinned
   resident batch；
3. batch 只做一次 H2D，随后三臂连续 forward/backward；
4. ordinary step 不做全模型 CPU hash，不逐步 `cuda.synchronize()`；
5. 只在 12-step smoke、epoch 边界和固定 sentinel step 做完整 digest/同步；
6. `num_workers`、prefetch 与 pinned-memory 只能在 smoke 中按预先固定值使用，不能依据
   external 结果调参。

这会显著减少 loader 空隙并让 GPU0 连续工作，同时保持三臂配对。禁止 CUDA streams
并行三个臂：共享 RNG/dropout 语义和显存峰值更难审计，收益不抵风险。

启动前 12-step smoke 必须记录 steps/s、GPU0 peak allocated/reserved memory 与 projected
12-epoch wall time。三臂版本的 hard cap 固定为 `7200 s`；若三模型不能在 24 GiB 内常驻，
或投影超过 hard cap，本次运行 fail-closed。不得在看到 feasibility 或任何训练数字后删除
`J-MEAN`，把预注册的三臂 estimand 动态改成两臂。

## 7. 评分矩阵

训练结束后，三个 epoch-12 checkpoint 在完全相同的 13 个 local session、相同 starts、
targets、D-opt4 activity support、selected-support4 fixed T4 和 query activity 上评分：

| checkpoint | FIXED30/static | UNCAPPED continual |
|---|---:|---:|
| historical POOLED `25d7...` | **必须重用 sealed `m4_activity_only` row** | 不存在；不得伪造或改名 |
| `J-NATIVE` | required | required |
| `J-R1` | required | **primary candidate** |
| `J-MEAN` | required | mechanism-only |

`J-R1/J-MEAN` 必须使用已纠正的 whole ordered activity-stack operator，不能恢复旧的
“每个 trial 先 M=1 计算 residual identity，再在 identity space 平均”的错误 scorer。
每次 decode 均为 commit-before/after 纪律中固定的 decode-before-commit；support 永不被驱逐；
UNCAPPED 不得偷偷变成 cap30。

现有 corrected scorer 只接受旧 PF checkpoint schema，不能直接被当作 AJPF scorer。
需要 route-owned generalized strict loader：先构造对应的 native/PF arm，再 strict-load
AJPF epoch-12 student state；score 前强制 `eval()/inference_mode()`，score 前后 state SHA
不变。13 个 session 的 starts、targets、D-opt4 activity support、selected-support4
raw/normalized T4、raw-M30 audit T4 与 query-activity digest 必须三臂完全一致。historical
POOLED comparator 固定为同一
`25d7...` checkpoint 的 sealed `m4_activity_only` rows；不得用 c51 cached-identity payload
替代这一 local matched lineage，也不得重跑 support selection。

评分开始后、读取任何新臂 R2 作为结果之前，必须从 `25d7...` strict-load 一个未训练的
`J-NATIVE` reference，并用同一 materialized 13-session input 运行 native G00m replay。
它的 **FIXED30** law 必须逐 session 精确复现 sealed `m4_activity_only` 的 prediction SHA、
R2、starts、targets 和 window count。任何一行不等即停止，不能用“R2 很接近”替代同输入
证明。generalized scorer 还必须从同一 held source graph 重建并绑定 historical selected-
support4 carrier、normalized side 与 normalizer authority；prediction parity 不能替代 carrier
lineage。该 reference 只承担 lineage/same-input sentinel，不进入 optimizer，也不成为动态
选择的新 comparator。

必须输出 external 6-session 与 within 7-session 的逐 session paired delta、equal-session
mean/median、positive count 与 fixed-seed session bootstrap CI。

## 8. 预注册主估计量与成功门

主候选为 `J-R1 / UNCAPPED`。预注册两个固定的配对对比，不按 target session
动态挑选或拼接参考：

```text
method_delta(session) =
    J-R1 / UNCAPPED - J-NATIVE / UNCAPPED

practical_delta(session) =
    J-R1 / UNCAPPED - historical POOLED / FIXED30
```

第二项是预先声明的 **total deployed-effect** 对比：它同时包含 AJPF route 和 uncapped
continual-memory 的产品效果；不得把它解释为纯 pooling-placement 效应。纯 route 效应只由
第一项同为 UNCAPPED 的 `J-R1 - J-NATIVE` 判断。历史 sealed graph 没有 POOLED/UNCAPPED
row，因此任何 receipt 都不得生成或使用这一虚构名称。

论文级成功必须同时满足：

1. 两个 external equal-session mean 都 `>= +0.010`；
2. 两个对比都至少 `4/6` session 为正；
3. 两个对比的 external worst session 都 `>= -0.015`；
4. 没有 target update、target selection 或 checkpoint selection；
5. `J-R1` 不是仅靠 `J-NATIVE` 共同微调整体变好：必须同时胜过 historical POOLED；
6. 所有 same-input、state、RNG、whole-stack scorer 和 receipt gate 通过。

`+0.005 <= mean < +0.010` 只能登记为小机制信号，不得称为有意义提升。小于 `+0.005`
或 positive count `<4/6` 为 null。任何 matched reference 下降造成的表面胜利均无效。

## 9. 机制读数

- `J-R1 - J-NATIVE`：锚定 late branch 在 decoder 共同适配后是否增值。
- `J-MEAN / UNCAPPED - historical POOLED / FIXED30`：placement replacement 加 continual
  memory 的 total deployed-effect；它不能单独识别纯 placement 效应。
- `J-R1 - J-MEAN`：锚定路径是否仍是必要保护。
- `UNCAPPED - FIXED30`：持续 activity memory 是否与 joint PF 正交增益。
- `J-R1 alpha`：是否离开零且方向/幅度稳定；alpha 本身不作为成功门。
- decoder update cosine/norm：PF 是否通过改变 decoder，而非只在 identity encoder 内重演旧路线。

可能结论预注册如下：

- `J-MEAN` 接近 POOLED，`J-R1` 再胜：joint adaptation 吸收 placement，anchored residual
  提取互补信息。
- `J-MEAN` 仍差但 `J-R1` 胜：完全替换仍 OOD，锚定联合适配是有效产品。
- 三个新臂共同变好但 `J-R1 <= J-NATIVE`：收益来自一般 decoder fine-tuning，不是 PF。
- `J-R1` 仅在 within 正、external 负：source overfit，PF 路线停止。
- 全部不胜 POOLED：warm-start joint PF 在当前架构/数据上关闭；只有真正 from-scratch
  matched architecture study 尚未测试，但不因本结果自动放行。

## 10. 停止与后续规则

V1 只有一次三臂 12-epoch matched run 和一次固定 local external score。不得依据 external
结果调整 LR、epoch、pool cycle、support 或 gate。

只有在**不看 external**的 epoch-12 收据中同时出现以下条件，才允许另写 work order
讨论 24 epoch continuation：

- `J-R1` source loss 在 epoch 11→12 仍下降至少 1%；
- train/monitor finite，无爆炸；
- alpha 没有饱和到数值边界；
- decoder update norm 非零且稳定。

即使满足也不自动继续。若 V1 external 已打开，它不能再用于批准同一开发面的 continuation。

真正 from-scratch PF 必须另行满足：随机初始化的 matched native 与 late-pool 两臂、相同
长预算、独立 source selection、不能继承 `25d7...` 的 encoder/decoder 权重。它成本更高，
不是 AJPF V1 的补跑或换名。

## 11. 实现与审计边界

- 新增 additive package、work order、runner、inert CLI 和 tests；不得修改旧 PF result root。
- 复用 selected checkpoint strict loader、PostFusionIdentityAdapter 的 whole-stack 算法与
  corrected whole-stack scorer；实现必须是新的 route-owned `AJPFIdentityAdapter`，不得调用
  旧的 `install_variant()` 或复用旧 frozen-decoder optimizer contract。
- 训练入口必须为 opaque one-shot capability；public CLI 默认 inert。
- attempt 必须早于 checkpoint tensor、data、CUDA；失败保留 immutable prefix，terminal/failure XOR。
- 闭包显式绑定 checkpoint loader、model/encoder/decoder、DataModule、PF variant、runner、scorer、
  design/work order 与测试外的所有 production transitive source。
- 执行前须有独立 no-CUDA 单测：strict load、三臂初值、`J-R1 +0` parity、参数归组、paired
  dropout、whole-stack score、failure topology 和 closure drift。
- 当前文档本身不授权 GPU；必须在实现与独立审计通过后由单独 work order 授权 GPU0。
