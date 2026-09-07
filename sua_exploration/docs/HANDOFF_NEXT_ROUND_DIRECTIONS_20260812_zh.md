# HANDOFF：功能载体项目的下一轮方向

**日期：** 2026-08-12
**状态：** ROOT claim/implementation 审计已完成；gated execution 正在进行。下方原始队列
仅为保留 provenance，**不得按原样执行**；每个相关合同与真实数据适配器被替换并复审前，
以第 -1 节的审计覆盖说明为准。本文件自身不授权任何 GPU 运行。
**范围：** 在融合/解码器搜索全部落空、且相对 SPINT 的原生 benchmark 增益偏小之后，下一步该做什么。
**与其他文档的关系：** 不覆盖 `HANDOFF_MAINLINE_CLOSURE_20260811.md`。本文只增加一个诊断和一份排序菜单。该 handoff 中列出的所有已关闭杠杆保持关闭，除非本文明确提出了新的 estimand。

> 本文是 `HANDOFF_NEXT_ROUND_DIRECTIONS_20260812.md` 的中文翻译，内容一一对应。以英文版为准。

---

## 轮次边界（2026-08-13）

由本 handoff 启动的上一轮验证现以 A2 terminal 结果正式收口。权威终点是
`a2_matched_subject_shift_v2/terminal_aggregate.json`（SHA-256 `5b1459df...`）：同一组三 seed
source-trained T4/Z4 checkpoint 在 sub-C development 与 external sub-M 上原样评分，所有冻结
interaction gate 通过，六个 formal sub-C test session 始终封存。A4 与 A11 是该轮支持性的 CPU
诊断，不扩大 A2 的 claim。

下一轮包含三个相互独立的范围：B1 carrier-by-distillation Stage P、A12 描述性 attention audit，
以及 A1 hidden-space carrier interface。B1 的部分 cell 不是 A2 replication；A12 不产生因果结论。
此边界禁止跨轮静默沿用 evidence、gate 或 checkpoint reuse 规则。C1 teacher-domain 与 C2 sampling 仍为 held，
不属于当前 GPU queue；A1/B1 的终局 routing 结果只在下方审计执行账本中记录一次。

---

## -1. Root 审计覆盖说明（2026-08-12）

在执行前，我们独立复查了 handoff 及同一晚生成的 scaffolding。结论不是“tests
没通过”，而是多处 **estimand 与真实实现错位**：合成测试合计 `102 passed, 3 skipped`，
但被跳过的正是真实数据路径，若干真实入口仍缺失或错误。因此 synthetic green 不等于
实验可运行。

### -1.1 实质性修正

| 项目 | 审计结论 | 使用前必须修正 |
|---|---|---|
| 1.2 节 / A2 | **交互方向与门控写反。** 历史数值预测 external subject M 上 carrier gain 更大，合同却要求 within-subject 更大；相同 source arm 又被为两个 scoring domain 重训两次。3 个 seed 上 exact two-sided Wilcoxon 要求 `p <= 0.05` 在数学上不可达。 | 把因子改名为 **target-subject distribution shift**。每个 seed 只训练一个 T4 和一个 Z4 checkpoint，原样评分 C、M，并在可行时加入 J；主统计量改为 `gain_external - gain_within`。一个 external subject 不能支撑普遍 cross-subject claim，删除不可达的 p-value 门。 |
| A3 / B4 | **并未破坏 correspondence。** neural、calibration、carrier 同步 permutation 对 permutation-invariant 模型是严格空操作；同步 subset/pooling 改的是 population size 或 signal view。 | 只保留为 **channel-loss / population-degradation robustness**；permutation 只能作为实现零对照，不能作为机制或有效 augmentation。 |
| B1 | **T4-only sweep 无法识别 carrier x distillation 交互。** 4 个 session mean 上 exact two-sided Wilcoxon `p <= 0.05` 同样不可达；B2/B3 也不以 B1 成功为逻辑前提。 | 至少使用 `{T4,Z4} x {with-E-loss,no-E-loss}`，以交互为主；`task_only` 后置。采用 effect size、符号一致性和 cluster-aware interval，不使用不可达门。 |
| A12 | **原始 scaffold 不可运行且绑定错误。** 它使用错误的 model/data path、转置 session batch、把 whole-identity zeroing 错称为 carrier control，并把 K 当成 V。这些缺陷在真实审计前已修复；后续审核还冻结并记录 CPU batch size，因为 BLAS partition 会改变 raw float32 digest。 | 当前 bounded partial aggregate 覆盖 seed 42、epoch 5--7（`3/24` 个 seed-epoch pair；SHA `2bf287...`）。六个 development session 上，T4-minus-Z4 的 normalized entropy、effective attended units 与 pairwise head cosine 均为 `6/6` 负。这只是未完成矩阵上的描述性 frozen-checkpoint observation；没有 inference、causal intervention 或 gate。 |
| A4 | **原始 scaffold 不可运行；该问题已解决。** 修复后的 probe 绑定 checkpoint、M30、train-only authority、固定 epoch 5--12 与 session-LOSO。 | 三个 seed 均已完成且正向。只能表述 AC4 比 Z4 保留了显著更多可跨 session 线性恢复的 tuning phase；这不是 nonlinear information 定理，也不是 decoder gain。 |
| A13 | **推断设计不成立。** window bootstrap/sign flip 忽略时间与 session 聚类；外部 JSON prediction 未证明来自所写 checkpoint；coverage 的大多数格子无法满足两个 session；“coverage 越差 gain 越大”还与估计噪声相冲突。 | 删除 30-stratum falsification gate。若保留，先做 checkpoint-bound exporter，再用 trial/session block resampling，只做小规模描述性分析。 |
| A5 | **SUA 上无法解释 aged drift。** 排序后 SUA 跨日没有稳定 unit identity，row prefix 不是生物学对应。 | 不跑 SUA aged claim。若仍需要，移到显式固定 channel 对应的 native MUA；否则已有 row/label shuffle 已回答错误 attachment。 |
| B9 | **主分数无量纲一致性。** `median cosine - median RMSE` 把无量纲相似度与尺度相关误差直接相减；从前 50 trial 选 M 个还保留 50-trial 等待时间。 | 分开报告 phase cosine 与归一化 held-out encoding error，使用 disjoint reference，并明确测试的是 label count、等待时间还是可预排 target geometry。 |
| B8 | **没有真实 gate。** runner 对 real data 明确停止，只能生成 synthetic correct/shuffled；same-M 静态 refit 也不是 prefix→future RLS。 | 必须使用 frozen decoder 在 post-prefix trial 上的真实 pseudo-label，回顾性真标签只用于审计，并报告 freeze frequency/drift；当前后置。 |
| A10 | **“same labels”未定义。** 用 dense per-bin velocity 训练 readout/fine-tune，比 T4 的每 trial 一个 direction label 多得多。 | 分成显式 dense-label supervised upper bound，以及真正 equal-annotation 的 sparse target；不得把两者差值称为独立的 backprop cost。 |
| B14 / B15 | **结构论证按当前写法错误。** softmax 权重虽非负，signed V/out projection 仍能产生负贡献；只有独立 cross-attention + per-token FFN 时，新增 latent query 无法影响 behavioural query。 | 删除 B14；B15 只有增加明确 query mixing/self-attention 后才成为另一个高成本假设，本 handoff 不授权。 |

所有表述还必须遵守两个 scope 修正：

1. H1 oracle swap 只测量**一个冻结 H1 consumer**的局部敏感性；它不是 estimator
   ceiling，也不能把 SUA/M2/RT 的 estimator 工作全局降级。
2. H1 H-SE5 不支持跨日期 sparse-label 正面结论：date 1 相对独立 Zero5 为
   `+0.028469`，date 2 为 `-0.022590`。H1 只保留 dense H-C compact-consumer 主线，
   不得写 H1 sparse-label claim。

### -1.2 审计后的执行顺序

1. **A4 已完成。** 保留 immutable 三 seed 结果，不扩展成 GPU 分支。
2. **把 A12 修成描述性诊断。** 它不再 gate A10，也不是因果 head-collapse test。
3. **把 A2 改写为 target-subject-shift interaction。** 每个 source checkpoint 跨 scoring
   domain 复用；预计 fresh training 是 2 arms x 3 seeds，而不是 12 个独立 cell。
4. **把 B1 改写为 carrier-content x distillation。** B2 与 estimator-noise augmentation
   是独立假设，不因 B1 null 自动关闭。
5. **保留 A11 baseline schedule caveat。** 已完成 replay，不支持恢复 B0 训练。
6. **先修正 A10 的 supervision accounting，再启动任何 gradient upper bound。**

A13、SUA A5、B8、B9、B14、B15 暂停。A14 是独立低优先级 efficiency branch：三格 fresh
matched run 只能重建 one-seed pilot，三 seed 论文矩阵需要九格；它不在当前科学关键路径。
相关合同重写、真实数据 preflight 与独立复审通过前，
不得依据旧第 5 节启动 GPU。

### -1.3 独立代码审计后的执行台账

| 项目 | 审计状态 | 当前决定 |
|---|---|---|
| A4 token-content probe | 已改写为 paired v10 AC4/Z4、M30、固定 epoch 5--12、六 session LOSO、training-session 内 pairing null、mean-normalized ridge、固定 executable 超参数与 immutable receipt。Focused suite 为 `17 passed`。三个 seed 的 AC4 raw/null/advantage 为 `0.749669/0.232369/+0.517299`，Z4 为 `0.319513/0.229369/+0.090144`，AC4-Z4 delta `+0.430155`，`18/18` 个 seed-session 为正。Multi-seed summary SHA 为 `6e0ead18...`。prose protocol 晚于 receipts，不能作为独立 preregistration。 | **正向诊断已完成。** Activity-only token 是 phase-poor 而不是 phase-free；AC4 的 null-relative phase signal 大 `5.74x`。这不是 nonlinear structural necessity 或 decoder gain。 |
| A11 B0 convergence | 独立权威 CPU replay **已完成**。全部 36 个 pinned checkpoint（3 seeds x 12 epochs）在严格六个 development sessions 上重放，gradient 关闭，model-state hash 不变，且未访问 training/formal-test NWB。Epoch-5--12 mean 为 `0.236417`，三个 seed 的 late-minus-early 为 `-0.01095/-0.01595/+0.00187`。第一个 checkpoint 的三 seed mean 为 `0.269176`，比固定窗口高 `0.032759`，但曲线明显非单调。Immutable full-receipt SHA 为 `955ebaf8...`。 | **已关闭：无 late-curve 改善信号。** Frozen continuation flag 为 false，mean `-0.00834`，不支持恢复 B0 训练。早期数值说明分解对 schedule 敏感，但不能事后选择 checkpoint。 |
| A2 target-subject shift | v2 已得到**终局正向**结果。三个 seed 的同一 source-trained checkpoint bundle 均在 sub-C development 与 external sub-M 上打分。sub-C 的 mean `T4-Z4=+0.248968`，sub-M 为 `+0.484766`；subject-shift interaction 为 `+0.235799`，三个 seed interaction 为 `+0.230085/+0.187631/+0.289681`，crossed seed-by-session bootstrap interval 为 `[+0.100852,+0.371768]`。四个 absolute means 为 within `Z4=+0.326008`、`T4=+0.574976`；external `Z4=-0.143399`、`T4=+0.341367`。Immutable aggregate SHA 为 `5b1459df...`；六个 formal sub-C test session 保持封存。 | **已完成。** 结果支持在观察到的 C-to-M subject shift 下，carrier 的**相对价值**增加；它不是 T4 绝对提高 `+0.235799`，也不是 biological unit-correspondence test。报告 interaction 时必须同时报告四个 domain/arm absolute means。 |
| B1 carrier-content x distillation | Stage-P routing 结果 | Mean interaction `+0.0137`；符号混合；低于 `+0.03`：**`STOP_B1_NO_STAGE_F`**。 |
| A1 hidden-space adapter | Seed42 routing pilot | Primary interaction `+0.0128 R²`；低于 prereg `+0.03`：**`PILOT_ROUTING_STOP`**。Attachment control 确认使用但没有 useful lift。 |

---

## 0. 为什么写这份文档

两个事实驱动它。

1. 在原生数据集上量到的、相对训练充分的 SPINT 的增益很小：matched M2 `+0.089796`，H1 organizer-held `+0.013447`，M1 `-0.00652`。
2. 一组异质的架构/融合干预在 matched control 下大多只有小 delta，而两个替换骨干的
   方案明显失败。它们**不是**一个八臂预注册 lattice：一个 logit 结果被用两个名字重复列出，
   B15 是 activity-only，CI64 也略超出 `±0.02`。

自然反应是再试第九种融合机制。证据说不要。本文先解释证据到底意味着什么，然后把剩余工作分成两类：

- **A 类 —— 证明、验证、表述。** 不引入新机制。基于现有设计补充实验，以及改变结果的描述方式。
- **B 类 —— 再设计。** 新的网络接口、新的训练方法、新的估计器。

---

## 1. 诊断

### 1.1 已测试的 fusion/interface 变体没有超过 matched control

| 干预 | 数据集 | 结果 | 门槛 | 出处 |
|---|---|---|---|---|
| 载体 concat 到 `post_pool` 输入（现选定） | 全部 | 基准 | - | `streaming_encoders.py::SideFeatureEarlyPoolEncoder` |
| Confidence-FiLM 作用于 pooled `h` | SUA | 相对 T4 `+0.003399` | 失败 | `DECODER_SIDE_DESIGN_SPACE_20260809.md` |
| 活动路径乘性 gain（L-D option 1） | RT | `G-Full - A0 = -0.002068` | `+0.03`，失败 | `HANDOFF_MAINLINE_CLOSURE_20260811.md` |
| T4 carrier-biased logit residual，rank 8，48 参数 | SUA | 相对 T4 continuation `-0.003142`，3/6 session | 失败 | `results/sua_t4_factorized_logit_residual_v1/aggregate.json`；design-space 文档引用的是同一运行 |
| Electrode gate `t4gate` | SUA，3 seed | 相对 T4 `-0.010817`，1/6 session | `ineffective` | `results/t4_gate_screen/aggregate.json` |
| Same-electrode relation `t4rel` | SUA，3 seed | 相对 T4 `-0.001440`，1/6 session | `ineffective` | `results/sua_electrode_relation_full_v1_scheduler/` |
| Activity-only 跨 neuron relational attention B15 | M2 | `B15-B3 = -0.014218`，0/3 cell | 失败 | `results/attention_arch_screen_v3/aggregate.json`；不是 carrier-fusion arm |
| 同一 activity-only 机制，容量对照 | SUA | `B15-B15P = +0.006354`，2/6 session | 失败 | 同上 |
| 联合载体接口宽度 32→64（CI64） | H1 | `-0.020130`，2/5 date | 失败 | `HANDOFF_MAINLINE_CLOSURE_20260811.md` |
| **Decoupled cross-attention `K(E,T4), V(x)` v1** | SUA | **`-0.444658`** 相对 coupled，0/6 session | 失败 | `results/sua_t4_decoupled_kv_v1/aggregate_seed42.json` |
| **Fixed slot router，K=32 soft** | SUA，2 seed | **`-0.177935`** 相对 B15P | 失败 | `results/fixed_slot_router_pilot_v1/aggregate.json` |

`attention_arch_screen_v3` 里四个 gate 全为 `false`。在 SUA 上表面的 `B15-B3 = +0.039770`，一旦对上参数匹配的对照 `B15P` 就塌到 `+0.006354`，所以那点增益来自容量，不是跨 neuron 通信。

**解读——是两个不同的类别，不是一个。**

1. **保留骨干的 carrier-path addition 都没有打赢 matched control。** FiLM、live-activity
   gain、logit residual、electrode gate、relation 与 CI64 都很小（CI64 是
   `-0.020130`，所以“位于 `±0.02` 内”只能是近似说法）。B15 是 activity-only，不能
   归入这类证据。结果足以停止该实现家族，但不能证明 concat 能表达所有有用函数。
2. **"替换"骨干一部分的干预会灾难性失败。** Decoupled K/V v1 的 `-0.444658` 是因为它移除了预训练的 `fc_in: 50->512->512` 读入，换成随机的 `50->32` value 投影，所以它根本没有检验 key/value 分解。Fixed-slot router 的 `-0.177935` 说明该压缩干预失败，但机制仍未解决：普通温度运行的 entropy 为 `0.856824`（约 `23.61` 个有效 slot），预先规定的低温 follow-up 从未启动。因此不得把 rate-distortion 或 uniform routing collapse 写成既有发现。

**反复出现的签名及其边界。** 所列候选相对**参数匹配但无机制**的对照均输或平：FiLM
对 `nofilm_match` `+0.000698`；relation 对 `no_group` `+0.006496`；logit residual 对
`additive_control` `+0.003756`；L-D 对 `G-XLS` `+0.006606`。若干 content control
明显更差，但并非全部“大幅”：membership shuffle 只有 `+0.016989`，判定为
indeterminate。因此这里只能支持“已测 matched mechanism 未超过选定接口”，不能声称
所有 content control 或全部 fusion family 都已解决。

**从未真正运行，因此是"未测量"而非"被否证"：** residual-FiLM、full-64-head oracle、T4 key-residual adapter、electrode anchor 与 embedding 设计。这四项都有通过的 CPU 测试套件、没有任何数值结果。Decoupled K/V v2 在 epoch 0 内被杀，held-in 读数约 `0.06` 对 coupled T4 的 `0.58`——那是 kill signal，绝不是正式比较。closure board 禁止全部这些。不要仅因为代码存在就复活其中任何一个。

### 1.2 未匹配的线索提示 target-subject shift 下 carrier value 可能更大

| 体制 | 历史 comparator（并不统一） | carrier system | 历史 delta / 线索 |
|---|---:|---:|---|
| 同被试跨 session（sub-C validation，B3） | `0.326479` | - | - |
| **跨被试（subject-M，Zero4 / T4）** | **`-0.057766`** | **`0.356828`** | **从不可用到可用** |
| native M2，同被试跨天（matched SPINT） | `0.293110` | `0.382906` | `+0.089796` |
| H1，同被试跨日期（organizer-held） | `0.261492` | `0.274939` | `+0.013447` |
| M1，同被试 | `0.648591` | `0.644766` | `-0.003825` |

Zero4 本身不是“零内容地板”；它是等宽度的 **activity-only identity encoder**，只把
标准化 carrier 置零（`mask_standardized_t4`，arm `z4`）。但上表只有 subject-M 行是
T4/Zero4 对比；M2 用 SPINT/B0 类 comparator，H1 用 organizer system submission，M1
又是另一种 system contrast。因此该表尚不能证明 activity-only 在 within-subject 可用而
cross-subject 崩塌。

M2、H1、M1 都是同被试、同阵列、跨天，而 subject-M 对比使用 external subject。
由于 protocol 与 comparator 不同，这张表只能生成假设；它不能证明“小 delta 是 benchmark
的性质”，也不能证明 carrier 修复 unit correspondence。审计后的 A2 estimand 是
target-subject distribution shift。

**必须遵守的边界。**

*协议不匹配。* `0.326479` 来自 20-epoch 的 `attention_arch_screen_v3` sub-C 协议；`-0.057766` 与 `0.356828` 来自 12-epoch 的 V9 subject-M 协议。这不是 matched run。上表是**假设**。A2 才是让它可以印出来的实验。注意更好的同被试比较对象是 SUA lattice 的 `Z4 = 0.326008`——那是同一个臂在 sub-C M30 基底上的分数，而不是 `B3 = 0.326479`。

*SUA headline 中有很大一块是架构性的，不是信息性的。* 在 M30 基底上，`Z4 = 0.326008` 对 `B0 = 0.236417`，其中 B0 是 SPINT 原始 `fc_id_in/fc_id_out` 的可训练、非别名副本，**side 宽度为零**。因此：

```text
T4 - B0 = +0.338559      headline
Z4 - B0 = +0.089591      装着全零向量的 side 通路本身
T4 - Z4 = +0.248968      真正的载体内容
```

**SUA headline 差距的 26.5% 是由一条装着零向量的四宽 side 通路提供的。** 任何"载体比
SPINT 高 +0.34"的表述都必须换成 T4 减 Z4。A11 已重放 36 个 B0 checkpoint：没有 late
upward trend，但第一个 checkpoint 的三 seed mean 为 `0.269176`，比固定 epoch-5--12 mean
`0.236417` 高 `0.032759`，而整条曲线明显非单调。因此该分解只在预定 epoch rule 下成立，
不是 architecture-invariant 数量；不得事后挑选 early checkpoint。见 A11。

*M2 的 delta 在 epoch 匹配后进一步缩小。* 官方 SPINT 镜像打包的是 epoch-27 的 decoder，而 T4 提交打包的是 epoch-34。在匹配 epoch-34 的回放下，`T4 - B0 = +0.06420`（3/4 session）、`T4 - TS4 = +0.09558`（4/4），而官方系统级差距是 `+0.116765`。匹配 epoch 后的载体增益大约只有 headline 的一半。

*在 H1 上，正确 attachment 占一个小分解的大约三分之二。* `H-C = 0.525511`、
`H-LS = 0.499895`（label-rotated carrier）、`H-S = 0.496833`、`H-C0 = 0.486616`。
因此 `H-C-H-LS = +0.0256`，而 `H-LS-H-C0 = +0.0133`。H-LS 在旋转 attachment
之前仍使用 behavioral labels，所以后一项**不能**叫 label-free；它是错误 attachment 下
的 nonspecific carrier-path content。compact rotated-label model 仍追平 H-S，这说明要把
compression 与正确 functional content 分开。

### 1.3 为什么 concat 可能已经够了，以及 phase-loss 假设

在理想线性 cosine encoding 且方向严格均衡时，`E[cos theta] = E[sin theta] = 0`，
所以**原始平均响应**会消掉 first-harmonic phase。这为无标签 trial 平均的 activity-only
identity 提供 phase-loss 假设，但不是 SPINT 的结构定理：网络先做 learned nonlinear
per-trial projection，再平均；有限方向只近似均衡；temporal response 也可能保留与 phase
相关的统计量。因此 A4 必须直接量跨 session 线性可恢复性。

这是假设，不足以单独解释所有结果；尤其不能从现有 null 推出“缺的信息只有 phase”或
“concat 已表达所有有用函数”。

**三个现成测量只是在动机层面支持该图景，并非一个 matched proof。** 它们混合 M2 的
baseline redundancy、H1 非 T4 carrier residual 与 H1 token angle，不能共同证明
center-out phase loss。

1. **基线坐标冗余，而 carrier content 只能被 activity 线性恢复一部分。** 平均发放率对 T4 的 `b`：M2 原生上 Pearson `r = 0.9960089736`，残差 R² `0.002697`（`results/n4_cpu_precheck_v1/audit.json`）。被同 pipeline 精确校准结果取代后，H1 carrier residual R² 为 `0.8048187892`，即 pooled-activity probe 线性解释约 `19.5%`（`SPINT-main/src/data/h1_overlap_gate_decoder_calibration_receipts/h1_overlap_gate_decoder_calibration_root_review_v2.json`）。这是跨数据集的动机，不是对 T4 phase 的证明。
2. **把冗余坐标还回去是有害的。** `B4 = 0.287273` 比零向量 `Z4 = 0.326008` 低 `0.038735`。给编码器一个它已经握有的标量，会消耗 side 通路容量而不提供任何东西。CPU 螺旋从那个 `0.001` 残差事先预测了这个符号。
3. **两条路径写入近似正交的方向。** 在 H1 identity-token 审计中，`E_i` 的活动驱动分量与载体驱动分量的中位夹角是 `83.483` 度，载体分量幅度约为三分之一（RMS 比 `0.332486`）。

这组现象与 baseline-rate 冗余、carrier content 互补一致，但不能证明 nonlinear activity
token 中不存在 `[a,c]`。A4 才是直接检查。

**直接 token 测量。** A4 现已在六 session LOSO 下从 learned identity token 读出 tuning
phase。三个 seed 的 AC4 raw/null/advantage 为 `0.749669/0.232369/+0.517299`，Z4 为
`0.319513/0.229369/+0.090144`；paired delta `+0.430155`，`18/18` 个 seed-session 为正。
Activity-only pooling 是 phase-poor 而不是 phase-free，AC4 的 null-relative linear phase
signal 大 `5.74x`；这不说明 decoder gain 或 Z4 的 nonlinear absence。

**一个警告。** overlap-residual 统计量曾被显式地对解码增益做过校准，并且**失败了**：三个 H1 臂上 Spearman `rho = -0.5`，精确 `p = 1.0`。把 `0.804819` 当作该校准 probe 下的线性可恢复性事实，不要当作关于有用性的证据。

### 1.4 未匹配诊断提示可能有 headroom；matched headroom 尚未测量

来自 `P3_CROSS_SESSION_ANALYSIS.md`：

| 量 | 值 |
|---|---:|
| 单 session 端到端上界（B3，单 session 内 80/20 chronological） | `0.6937` |
| POYO 单 session CO 参考（13M 参数，5-10 ms bin） | `~0.935` |
| 带标签 encoder-only finetune oracle，K=20 校准试次 | `+0.692` |
| 同一 oracle，K=10 / K=5 | `+0.614` / `+0.408` |
| 同设置下的 zero-shot | `-0.122` |
| 当前最好的无梯度跨 session 载体结果（subject-M） | `0.356828` |

**这些数字与主线不可比，绝不能当作可比数字引用。** 该 oracle 跨越了 4× 的 unit 数体制跳变（train `~60` units，test `~245` units），使用了 held-out 行为标签和反向梯度，且 split 与 V9 不同。

即便如此，它们是目前关于"禁用反传的代价"的唯一线索，而这个线索指向代价可能很大。这个测量的 matched 版本是 `HANDOFF_BASELINE_GAP_ANALYSIS_20260812.md` 的 3.4 节，本文列为 A10，从未跑过。**在它出结果之前，不应启动任何解码器再设计。**

### 1.5 一个冻结 H1 consumer 对 query-fitted leakage perturbation 局部不敏感

`pilot_artifacts/h1_carrierid_quality/H1_CARRIERID_QUERY_ORACLE_LEAKAGE_DIAGNOSTIC_v1.json`：

| 冻结 H-C 消费者上的条件 | 池化 R² |
|---|---:|
| 诚实的 support 拟合载体 | `0.5255107931` |
| **query 拟合的 oracle 载体** | **`0.5226522069`** |
| oracle 减 support | **`-0.0028585862`** |

这次替换确实改变了载体——相对 Frobenius 差 `0.827` / `1.051`，行余弦均值 `0.537` / `0.359`——而消费者几乎没动。receipt 记录 `consumer_local_insensitivity_supported: true` 与 `consumer_response_locally_attenuated: true`，预测位移 RMS 只有目标中心化 RMS 的 `0.0606`。

这明确是 `LEAKAGE_DIAGNOSTIC_ONLY` perturbation，不是严格 oracle ceiling。约
`4716/8965` 个评估输出（`52.6%`）与用于拟合 query-local carrier 的 trial 重叠；替换后的
carrier 并未被证明是更好的 estimator；冻结 consumer 训练时看到的 carrier 分布也不同。
因此它只支持**这个冻结 H1 consumer 对该 perturbation 局部不敏感**，不能证明 carrier
estimation 已饱和。

同一 fold/seed receipt 中，同 checkpoint 置零的差值是 `0.103871`，而独立重训 H-C0 的
差值是 `0.038895`，比值 `2.67×`。这说明该 receipt 中同 checkpoint 置零会夸大 carrier
效应，但不能把这个倍数推广到其他设置。

这对下面的菜单有直接后果，也是本文档优先级与直觉排序不同的主要原因：

- **只对这个冻结 H1 consumer，估计器改进必须配套重训后才可判断。** 该 oracle swap
  不能把 SUA/M2/RT 的 D-optimal、temporal kernel、covariance 或 shrinkage 全局降级。
  另一个独立事实是 SUA M15 W3 无效：`T4W3@15 - ordinary T4@15 = +0.000753`，尽管
  train-only proxy 改善 27/27 session。`T4@15 - T4@50 = -0.058842` 是普通预算差，
  不是 shrinkage contrast。
- **consumer-side 与 estimator-side 仍是独立假设。** 这一诊断既不能提升所有 consumer
  redesign 的优先级，也不能把 estimator work 全局降级。

该 receipt 自己点名了所需的后续：在 oracle 载体分布上从零训练一个源消费者，再用匹配的 query-local 载体评估。它同时记录 `estimator_saturation_proven: false`。在那个实验跑之前，"载体估计不是瓶颈"只对一个从未有机会使用更好载体的模型成立。

---

## 2. A 类 —— 证明、验证、表述

不引入新机制。按单位投入的收益排序。

### A1. 把 Zero4 改称 "activity-only identity"

**做什么。** 改描述，不改实验。说明 Zero4 臂与载体臂架构完全相同，只是把标准化后的载体置零。
**为什么。** 防止把 zero-vector control 误解成 zero-capacity model。subject-M 的分数仍然
是特定 protocol 下的结果，不能单独推出“baseline 普遍失效”。
**成本。** 仅写作。
**依据。** `unit_side_features.py::mask_standardized_t4`，arm `z4` 返回 `zeros_like`。
**否决判据。** 无。这是对一个 under-claim 的事实性更正。

### A2. Matched target-subject-shift × carrier-content 实验

**做什么。** 每个 seed 只在 sub-C 训练 `{activity-only, carrier}` 两个 arm，再把每个未改
checkpoint 分别打分于 sub-C development sessions 与 external subject M。主统计量是
`T4-Z4` gain 的 external-minus-within interaction。
**为什么。** 它用一个 target-subject-shift estimand 替换不匹配的历史线索；不再声称
检验 stable unit-index correspondence，因为 B3S/T4 都不要求这一点。
**成本。** `2 arm × 3 seed = 6` 个 fresh source-training cell，随后每个 checkpoint 两个
score domain。
**否决判据。** 若 `A2_MATCHED_CORRESPONDENCE_CONTRACT_20260812.md` 的 interaction gate
失败，删除 subject-shift 叙事；不得做 correspondence claim。

### A3. unit availability 变化下的稳健性（暂停，待重写）

原先的 consistent permutation 在 activity 与 carrier 同步移动时，对 permutation-invariant
模型是 exact null；跨日期按 row prefix 也不能建立 biological unit correspondence。在围绕
missing-unit robustness 等明确定义的部署扰动重写之前，A3 不授权任何实验，也不得使用
sealed sessions。

### A4. 跨 session phase-content probe

**做什么。** 使用配对 Z4/AC4 checkpoint、六 session LOSO 与 training-session 内 pairing
null，训练线性探针从 learned identity token `E_i` 恢复 `[a_i,c_i]` 的方向。
**为什么。** 它直接测量 phase 能否从 learned token 跨 session 线性恢复，不能证明 nonlinear
absence 或结构必然性。
**成本。** 已封存 checkpoint 上 forward-only，CPU。
**结果。** Seeds 42/43/44 已全部完成：AC4 raw/null/advantage
`0.749669/0.232369/+0.517299`，Z4 `0.319513/0.229369/+0.090144`，paired delta
`+0.430155`，`18/18` 个 seed-session 为正。Activity-only 是 phase-poor 而不是
phase-free；只支持 differential linear recoverability。Prose protocol 晚于 receipts，不能作为独立 preregistration。

### A5. 迁移/陈化 carrier 对照——因不可识别而暂停

**原提案。** 用 session A 的 per-unit carrier 给 session B 打分，并扫 date gap。
**为什么当前无效。** Session 间没有经验证的 biological unit correspondence，而 B3S/T4
自身 permutation invariant。把 A 的 carrier row `i` 装到 B 的 activity row `i`，会人为制造
row-prefix/electrode correspondence；任何退化都会把 carrier aging 与 wrong attachment 混在
一起。这与原 A3 cross-date 提案的缺陷相同。
**处置。** 不运行 A5。论文中声称已经测得 drift advantage 的句子应删除或弱化。未来 stale
carrier 实验需要独立验证的 unit-tracking map 或 correspondence-free estimand；当前都没有。

### A6. 固定 pipeline 内的适用性诊断

**做什么。** 在同一个 carrier/decoder/comparator protocol 内，把各 calibration budget 的
paired session-level gain 与 train-support design rank、condition number、directional balance
关联。任何 predictive rule 只能在 source/development cell 上拟合，并通过 held-out-session
cross-validation 与仅使用预算 `M` 的简单基线比较。
**旧 cross-task rule 为什么无效。** SUA、pseudo-MUA、M2、RT、H1、M1 使用不同 carrier、
output、supervision density、consumer 与 comparator。Task identity 同时改变 predictor 与 gain，
六任务相关不能识别 applicability law，也不能把 M1 变成 confirmatory evidence。
**成本。** Exact matched-cell inventory 后做 CPU。Cosine-fit path 已有 `design_rank` 与
`design_condition`；非 cosine carrier 需另行定义统计量。
**否决判据。** 若 fixed-pipeline rule 在 held-out-session 上不能超过 budget-only baseline，
则不发布 predictive rule。Cross-task 图只能描述，不能设 deployment threshold。

### A7. 用掉已识别但闲置的免费资产

organizer 测得的延迟 `0.113919` vs paper-LR SPINT 的 `0.129075`；dense H1 H-C 的 organizer-held 结果；compact-consumer 压缩；以及 M1 作为主动申报的负边界。H-SE5 只作为 sparse-label 边界，不是正向跨日期结果；CI64 只否掉所测的 widening 配置，不能推出 consumer 容量普遍不受限。仅写作。见 `PAPER_FRAMING_ANALYSIS_20260812.md` part 2。

### A8. 把标签预算曲线作为 empirical sample complexity 呈现

**做什么。** `M10 = 0.304264`、`M15 = 0.338115`、`M20 = 0.351767`、`M30 = 0.358154`、
`M50 = 0.356828`。配合 session dispersion/sign count 与固定 evaluation boundary 报告 matched
budget curve。
**为什么。** 它支持该 protocol 在 M20--M30 附近经验饱和。在 A6 独立地比 budget-only
baseline 更好预测 held-out session 之前，不把曲线形状归因于 design conditioning。
**成本。** 仅写作；A6 是可选机制证据，不是报告这条曲线的前置。

### A9. 把已测试的 fusion slice 作为历史消融族发表

**做什么。** 把 1.1 节表格报告为已测试 fusion slice 的历史地图，并保持两个失败类别
分开：保留骨干的 matched additive pathway 是平的，backbone-replacing variant 则在不同且
有记录的 confound 下失败。4b 中从未运行的项目必须写成"从未运行"，不得让它们读成 null
或已覆盖的 design space。
**为什么。** 它把已经花掉的 GPU 转成诚实的 coverage map：在各自所选接口上真正 matched
的 additive variant 是平的，两个 backbone replacement 则在不同且有记录的 confound 下失败。
这缩小了已观察的设计空间，但不证明整个 fusion space 都平，也不证明 descriptor content 是
唯一因果因素。
**成本。** 仅写作。
**风险。** 审稿人可能读成钓鱼式实验。应称为 heterogeneous historical ablation family，
只报告真正 matched 的 contrast；不得称为预注册 lattice，也不得用它预先否掉未测试方案。

### A10. 设计 matched no-backprop constraint 审计

**问题。** 禁止 target-session weight update 究竟牺牲多少性能？
**当前 design blocker。** 现有提案并不 matched。T4 每个 calibration trial 只消费一个方向
target，而普通 decoder fine-tuning 消费 dense neural--velocity pairs；同时改变 update rule、
label density 与 optimization target，不能识别 backpropagation 的纯代价。“同一 calibration
prefix”不等于“同一 supervision”。
**任何 GPU 前的必要重写。** 冻结 supervision ledger，逐 arm 列出可见的每个 target value、
temporal multiplicity/weighting、trainable parameters、optimizer steps、saved state，以及
calibration/query window。Primary contrast 必须固定 target observations 与 objective，只改变
closed-form/readout update 和 backward gradient。如果无法构造与 carrier arm 科学上等价的
sparse-label gradient objective，则 dense-label fine-tuning 只能作为明确 label-richer 的 upper
bound，不能称为 no-backprop constraint 的代价。
**状态与成本。** 仅设计，**GPU-blocked**。1.4 节不匹配的 oracle 不能设 numeric gate，也不能
推出 `0.3 R2` headroom。A10 当前不 gate B1，也不能建立 consumer saturation；未来 gate 必须
在 supervision contract 可审计之后冻结。

### A11. B0 的收敛诊断

**做什么。** 重放全部 36 个已有 B0 checkpoint，检查固定 12-epoch 曲线的后半段是否仍在
改善；本诊断不包含 epoch 12 之后的新训练。
**为什么。** `Z4 - B0 = +0.089591` 说明 headline 差距的一部分来自 side 通路；已有曲线可
直接检验 late upward trend 是否足以支持任何续训。
**成本。** 仅 forward replay，无新训练。
**权威结果。** Immutable CPU full replay（SHA-256 `955ebaf8...`）覆盖全部 36 个 pinned
checkpoint；epoch-5--12 mean 为 `0.236417`，三个 seed 的 late-minus-early 为
`-0.01095/-0.01595/+0.00187`，aggregate 为 `-0.008342`。Frozen continuation flag 为
false，因此 A11 关闭，不支持扩展 B0 训练。CPU smoke 与 full replay 的最大数值差小于
`3e-8`，checkpoint、query、normalizer、model-state 与 scope binding 全部通过。

### A12. 描述性 attention-path 审计

**做什么。** 在 canonical SUA v10 M30 paired T4/Z4 checkpoint 上，截取 frozen decoder 的
精确 Q/K/V tensor 与 attention weights。报告 per-head normalized entropy、effective support、
跨 window/covariate variability，以及 paired T4-Z4 变化。每个 dump 必须绑定 checkpoint、
run metadata、normalizer、support/query policy、session roster 与 arm；不得用 whole-token zeroing
冒充 carrier intervention。
**为什么。** SUA consumer 是一层 64-head cross-attention 和两个 output-query token。严格绑定
的 dump 可以描述 explicit carrier 是否改变 attention geometry，以及许多 head 是否呈描述性
冗余；但它不能单独解释 accuracy saturation，因为高/低 entropy 不是 causal head importance，
这里也没有 pruning 或 intervention。
**成本。** Metadata-only preflight 与 root 独立审核通过后，做 forward-only CPU。
**状态。** v4 preflight SHA `ce9fcfc3...` 绑定 `B_cpu=512`。Bounded partial aggregate 已覆盖
seed 42、epoch 5--7（计划 `3/24` 个 seed-epoch pair），immutable aggregate SHA 为
`2bf287307595e5b58e10b3ea00ff082de9873d50421ccd67c9b099764d95eb30`。六个
development session 上，mean T4-minus-Z4 delta 分别为 normalized entropy `-0.041562`、effective
attended units `-2.612026`、pairwise head cosine `-0.035522`，三者均为 `6/6` session 负；剩余
`21/24` pair 尚未完成。
**解释边界。** 无 pass/fail gate，不宣称 causal “head collapse”。A12 不 gate A10、不授权
decoder redesign；在另行绑定 dataset-specific checkpoint path 前，不推广到 M2/H1。该 partial
aggregate 只是 frozen-checkpoint 描述性 observation，不是 inferential 或 causal result。

### A13. 误差结构分解——等待 provenance/statistical redesign

**原问题。** Paired T4-Z4 gain 是否随 movement direction、speed、magnitude 或 calibration
coverage 改变？
**当前 scaffold 为什么无效。** 它读取外部 prediction JSON，却不能证明 prediction 来自所写
checkpoint、query windows、normalizer 或 arm；bootstrap/sign-flip 还把相关 window 当 IID。
Calibration coverage 是 session/support-level 量，而 speed/magnitude/direction 是 trial/window-
level 量，把它们混进同一个 strata gate 会产生 pseudoreplication 和 level mixing。H1 recording
差异与 A2b ridge instability 属于不同 carrier/protocol，不能由这个 SUA analysis “解释”。
**必要重写。** 先构建 checkpoint-bound exporter，在同一 forward pass 写出 paired T4/Z4
predictions、targets、trial/session IDs、window hashes、normalizer、support policy 与 arm
provenance；export 前冻结 strata。Within-session contrast 用 trial 或 contiguous-block resampling，
再在 session 层聚合；coverage effect 需要 session-level replication，不能拿 window 当样本量。
**处置。** 暂停；当前无 numeric gate、无 mechanism claim。未来即使有效，也只能先作为
descriptive heterogeneity，除非 interaction 与 resampling hierarchy 独立预注册。

### A14. 收尾 B3TStream + T4 效率分支

**做什么。** `results/sua_b3t_t4_efficiency_v1/` 里有 `t4_s42 = 0.585301` 和 `b3t_t4_s42 = 0.597073`，delta `+0.011773`——在 `-0.03` 非劣容差内——并记录了 30.79% 参数缩减、65.29% session-MAC 缩减、88% 瞬态状态缩减。所需的同 seed `B3TStream + TS4` 内容对照从未运行，也没有 aggregate。
**为什么。** 一个结果有利的效率分支，被 2026-07-31 的收缩当作队列第 5 位取消，不是因为失败。
**成本——2026-08-13 更正。** 只读检查发现**零个 checkpoint**：只有 `t4_s42.json`、
`b3t_t4_s42.json` 和三个日志，其中 `b3t_ts4_s42.log` 表明 control 在 receipt 前被杀。
没有原 checkpoint，历史 `0.585301/0.597073` 不能和新 control 配对。重建 seed42 是**三个
fresh cell**，但只构成 pilot；论文级三 seed matrix 是**九个 fresh cell**。在恢复 checkpoint
或明确冻结完整矩阵前，preflight 必须 fail closed。
**否决判据。** Pilot 中若 B3TStream+T4 不超过 TS4 content control，或相对 T4 超出冻结
non-inferiority margin，就停止。正向 one-seed pilot 最多授权剩余六格，本身不是论文 claim。

---

## 3. B 类 —— 再设计

### 3a. 训练方法（从未尝试；本类中期望价值最高）

### B1. 移除或改换 identity 蒸馏项

**做什么。** M2 与 streaming 线默认 `loss_mode = task_plus_y_plus_E`，`lambda_y=1.0`，
`lambda_E=0.1`。`lambda_E` 把 student identity token 拉向 **activity-only teacher**。
 A4 已说明 activity-only representation 是 phase-poor 而不是 phase-free，而 AC4 保留了
显著更多 null-relative linear phase signal。因此剩余假设更窄：identity distillation 可能
选择性压制额外的 carrier-specific content。B1 必须识别 carrier-content × distillation
interaction。
**为什么重要。** SUA 线的默认已经是 `task_only`（`train_variant_dandi688.py`，默认 `task_only`），而 SUA 恰恰是载体表现最强的地方。M2 用 `task_plus_y_plus_E`，而 M2 恰恰是 delta 最小的地方。这个共现从未被当作因果检验过。
**实验。** 在 M2 上做 primary `2 × 2` factorial：

```text
carrier content:       T4 versus Z4
identity distillation: task_plus_y versus task_plus_y_plus_E
```

固定 `lambda_y`、architecture、seed、split、calibration/query policy、epoch rule 与其余训练
选择。Primary estimand 是 difference-in-differences：
`(T4-Z4)_without_E - (T4-Z4)_with_E`。T4-only 的三 loss sweep 不能识别 interaction，现撤回；
`task_only` 只能进入另行冻结的 secondary experiment。
**成本与分阶段范围。** 当前 M2 `loso_fold=0` 只有一个 validation session，因此
`4 arm × 3 seed = 12` 个 fresh GPU cell 只能构成**单 session 的 Stage-P routing pilot**，不能作为
跨 session 结果。Stage P 启动前固定 folds 1--3 作为唯一可扩展范围；只有 practical
interaction threshold 与 3/3 seed 同号门同时通过时，Stage F 才追加
`3 fold × 4 arm × 3 seed = 36` 个 cell。**只有 Stage F 是 confirmatory primary**：其 terminal
rule 只作用于三个 fresh、预先指定的 LOSO session × 三个 seed。P+F 只能作为明确标注的
descriptive sensitivity，绝不能接受 terminal rule。
**否决判据。** 启动前冻结 interaction threshold、Stage-P seed rule 与最终 session/seed rule。
Stage P 未通过即停止；confirmatory Stage F 后若移除 `lambda_E` 仍不能按最终规则增大 `T4-Z4`，则放弃
selective-suppression 假设；不得用 T4 main effect、事后 loss mode 或更换 fold 挽救。
**Provenance，现已核查（2026-08-12）。** 最初的 R1 loss-mode 选择是在**无载体**的学生上做的。Gate-2 的 R1 消融配置（`b3_d64_anchor`、`b3_d64_task_only`、`b3_d64_task_plus_y`）全部解析到 `streaming_b3`，即变体 **B3、`side_dim = 0`**，而 `gate2_matrix.py::evaluate_r1` / `choose_winning_loss` 正是从这些行里选的。没有任何 receipt 记录过在 B3S 或 T4 这种带载体的学生上做过 loss-mode 选择。

两个后果。其一，M2 carrier arm 从一个**在没有载体时**做出的决定继承了
`task_plus_y_plus_E`，所以 `lambda_E` 与 carrier content 的 interaction 从未被检验。其二，
上面的 matched `2 × 2` factorial 是**新的 estimand**，不是 sealed decision 的 replay；它需要
自己的 source-only selection，不能继承 carrier-free R1 sweep 的 gate。

### B2. Session-consistent carrier-reliance corruption

**做什么。** 架构和部署路径保持不变。源训练时，每个 source session 在每个 logical epoch
只使用一种 carrier 状态，并按精确十二 epoch 日程分配 `T4 x 6 / RS4 x 3 / Z4 x 3`。
同一 session 的所有 window 在该 epoch 内共享状态；RS4 对每个 run seed/session 使用一个固定的
complete row derangement。问题是：错误或缺失 carrier 出现时，consumer 能否退回 activity-derived
identity，同时仍保留正确 T4 的收益。
**为什么。** 历史 P3 从未训练模型：其 sampler 未接入 datamodule，唯一 receipt 只是重放旧的
RS4/LS4/Z4 数字。Activity-path dropout 会强迫模型**更依赖** carrier，方向正好相反，因此从本
estimand 删除。Per-window 随机 carrier noise 也被拒绝，因为 T4 是 session identity，不应高速变化。
**Primary estimand。** 对 clean 与 corruption-trained consumer 使用匹配的
`{T4,Z4,RS4,LS4}` forward views，difference-in-differences 检验 RS4/LS4 是否向 Z4 靠拢；普通
regularization 的共同增益会抵消。Anti-triviality gate 同时要求 corruption-trained
`T4-Z4 >= +0.03`，且 clean-T4 deployment non-inferior。
**分阶段与状态。** CPU contract v2 已完成（`31` 个 focused tests），但没有 runtime hook 或 GPU
结果。Stage P 为 `2 arm x 3 fold x seed42 = 6` cells；只有预冻的机制、content-retention 与
deployment gate 全过，才追加 seeds 43/44 的 Stage F（`12` cells）。只有 Stage F 是
confirmatory，P+F 仅为 descriptive。机制上 B2 独立于 B1，但实现必须等待 B1：B1 选出的 loss
定义 B2 substrate，而且 B1 streaming tree 已 hash-lock。
**否决判据。** Stage P 中错误内容没有向 Z4 靠拢、正确 T4 content 消失，或 clean-T4 deployment
低于 `-0.03` margin 时立即停止；不得用 probability、M、loss、fold 或 epoch 挽救。

### B3. 估计器噪声增广

**做什么。** 源训练时给载体加噪，噪声取自解析已知的小 M 估计器协方差 `sigma^2 (X'X)^-1`，`tuning_fit_confidence_descriptor` 已经在算它。
**为什么。** 它训练消费者对部署时真正会遇到的估计误差保持鲁棒，直接针对 `M10 = 0.304264` 到 `M50 = 0.356828` 的缺口。
**成本。** 少量代码改动，一次噪声尺度扫描。
**否决判据。** 低 M 预算点没有恢复。
**备注。** 这与已失败的 `t4c` / confidence-FiLM 路线不同。那条路线把不确定性作为**额外输入**喂给模型；这一条用不确定性塑造**训练分布**，不增加任何输入宽度。它仍是低优先级、暂停中的假设，不是已排队 GPU 实验；若以后测试，必须重训 consumer，并加入 matched T4/Z4 sibling interaction。冻结 H1 query-oracle 诊断不能作为放行依据。

### B4. Population-degradation augmentation——等待 non-null 合同

**做什么。** 未来的 source-training 稳健性实验可以采用固定强度的 unit dropout、unit
subsetting 或 pseudo-electrode pooling，并在评估时使用同一定义的 degradation。排除同时作用于
activity、calibration 与 carrier 行的一致 permutation：对当前 shared-unit permutation-invariant
模型，它是 exact implementation null，不是有效增广。
**为什么。** 可辩护的目标是对 unit 数减少或 population observation 变粗的稳健性，而不是修复
biological unit correspondence。Subsetting 与 pooling 保持 activity-carrier attachment，只改变
可用信号，因此无法隔离 correspondence。
**状态与成本。** 暂停。Pseudo-MUA pooling 虽已有 evaluation view，但当前没有 matched A3
degradation curve、冻结 severity grid、T4/Z4 anti-triviality control 或 source-only selection rule。
补齐这些属于新的中等成本实验，不是免费复用 A2 checkpoint。
**否决判据。** 启动前冻结 clean-performance 非劣 margin 与 population-degradation endpoint；
若增广不能改善后者，或 clean T4 低于冻结 margin，则停止。普通 clean-score 上升不得解释为
correspondence evidence。

### B5. 情节式、与部署对齐的目标函数

**做什么。** 优化模拟 session 上"校准之后在留出 query 分片上的表现"，而不是配 `random_calibration: true` 的逐 batch MSE。
**为什么。** 当前目标只是部署指标的弱近似。显式的情节式目标（先校准前缀，再 query）优化的才是真正被打分的东西。
**成本。** 训练循环重写。3a 中工程成本最高。
**否决判据。** 只在 B1-B3 之后运行，且至少其中一项有效果。

### B6. 多被试、多数据集联合源训练

**做什么。** 以载体作为共享接口，在 sub-C + M2 + RT 上训练一个消费者。
**为什么。** 如果载体是一个被试无关的坐标系，联合训练应当在每个域都有提升。`ARCHITECTURE_ANALYSIS.md` 的"方案 B"，从未运行。
**成本。** 数据管线 + 一次大训练。
**否决判据。** 任何一个域退化超过冻结容差。

### B7. 用掉无监督的输出 bin

**做什么。** `decode_last_timestep_only = True` 意味着解码器吐 `W` 个 bin 而只有最后一个被监督——50 个输出 bin 中有 49 个不携带梯度。加一个覆盖其余 bin 的辅助损失，或在 `E` 上加一个载体重建头。
**成本。** 小。
**优先级。** 低。列出以求完整。

### 3b. 估计器与载体内容

> **受 1.5 节限域，而非全局降级。** query-fitted oracle 只让一个冻结 H1 consumer
> 移动 `-0.002859`；该 receipt 是 leakage diagnostic，不是严格上限，而且 consumer
> 没有在 query-carrier 分布上训练。因此 standalone H1 estimator swap 价值较低，但
> SUA/M2/RT 必须各自在 matched substrate 上判断。W3 的正确警告是
> `T4W3@15 - ordinary T4@15 = +0.000753`；`-0.058842` 是普通 M15→M50 预算差。
>
> **因此 3b 中任何一项都不应作为独立的精度臂启动。** 每一项都必须与"在改进后的载体分布上重训的消费者"配对，这正是 oracle receipt 自己点名的后续。B8 是例外：它改变的是载体**何时**被估计，而不是它有多准，所以不受此论证约束。

### B8. 通过递归最小二乘实现自一致载体

**做什么。** 照旧用 M 个带标签试次拟合载体。之后在后续的**无标签**试次上，用解码器自身的输出作为伪目标，通过带遗忘因子的递归最小二乘更新 `[a, c, b]`。闭式、每 unit `O(1)` 状态、无反向传播、无优化器。
**为什么。** 它把载体从静态描述子变成自适应状态。标签只在每个 session 付一次，而不是每次重校准都付一次，并且它处理慢性漂移——那才是真正的临床问题。它直接回应了"校准并非 label-free"这个长期反对。
**为什么 N4 的失败不封它。** N4 是 label-free 的**静态**描述子（rate、Fano、autocorrelation、population coupling），得分 `+0.001588`，6 个 session 中 3 个为正。RLS 载体从带标签前缀继承了任务对齐；这是不同的机制。
**GPU 之前的廉价 CPU 判据。** 在已封存 checkpoint 上，用解码器输出的伪标签重拟合载体，量 `cos([a,c]_pseudo, [a,c]_true)` 对比 shuffle 基线。这与 RT 的 split-half 可构造性判据（`0.787119`）是同一模式，工具链已经存在。
**最强反对。** 自训练会放大自身误差。**回应。** 更新只触碰每 unit 三个参数，不动任何网络权重；加遗忘因子，并加一条 fail-closed 规则：载体偏离初始拟合超过固定阈值就冻结。A5 提供了无更新时的衰减基线，RLS 需要证明它把那条曲线压平。

### B9. D-optimal 校准设计

**做什么。** 选择校准期指示哪些目标方向以最小化载体协方差，而不是取按时间顺序的前 M 个 rewarded trial。
**为什么。** 估计器是设计矩阵已知的 OLS，所以设计是可控的。`M10 = 0.304264` 的缺口可能主要是条件数问题而非数据量问题，因为 chronological-first-M 给出的方向覆盖是随机的。
**成本。** CPU，回溯性。在同一批试次上用 D-optimal 选择重算预算曲线。
**若成立的收益。** "八个带标签试次就够"比"五十个"锐利得多。
**否决判据。** 低 M 下没有恢复 M50 的性能。
**当前 correctness blocker。** Scaffold 私自复制了 canonical-direction 与 cosine-fit
estimator，而不是调用主线实现，也没有 equivalence test 绑定二者。B9 继续暂停，直到把
primitive 抽成 framework-free shared module，或用 parameterized randomized test 证明 bit-level
一致。在此之前产生的 B9 数字不得使用。

### B10. 时序核载体

**做什么。** 把单一的、源上选定的延迟 `tau` 换成每 unit 的时序核：`r_i(t) = sum_tau w_i(tau) y(t - tau)`。载体变成 `[N, d_y × n_lags]`。
**为什么。** 这是论文自身 general form 的严格推广，仍然闭式，而且是唯一明确增加信息量（而非重排信息）的内容扩展。它也天然适合 H1 与 RT，那里单一延迟不太可能覆盖运动的每个阶段。
**成本。** 先 CPU 可构造性判据，再一个 GPU 臂。
**风险。** 加宽描述子会在 `M = 30-50` 下恶化条件数。与 B9 以及已有的收缩机制（`uncertainty_wiener_shrink_t4`）配对使用。

### B11. 群体载体：加入 label-free 的协方差摘要

**做什么。** 逐 unit 载体给出 `W` 和 `b`，但丢弃了噪声协方差 `Sigma`。最优线性解码两者都需要。加入一个紧凑的、闭式的 `Sigma` 摘要（例如其 top-k 特征基）作为 session 级侧输入。
**为什么。** `Sigma` **不需要标签**。这是唯一一条能加信息而不加监督的路径，从而保持监督密度主张不变。H1 估计器已经在做 project、regress、back-project、shrink，先例存在。
**成本。** CPU 可构造性判据，再一个 GPU 臂。

### B12. 载体群体的 Procrustes 对齐

**做什么。** 用闭式 SVD 把目标 session 的载体群体对齐到源定义的正则框架。
**为什么。** 跨被试或跨阵列时，整个 tuning 分布可能被旋转。用源统计量做的逐列 z-score 修不了旋转。电极与空间先验已经试过；载体群体的 Procrustes 没有。
**成本。** CPU，便宜。
**关联。** 若 A2 显示存在跨被试交互，这是第一个值得用来加宽它的手段。

### 3c. 网络接口与 target-time adaptation（边界分开）

**2026-08-13 hidden-space adapter 提案的边界。** A1 只改变 source training；目标 session 部署时仍只做
解析 T4 拟合与 forward pass，不更新任何网络权重。因此它**不被 A10 阻塞**；A10 可约束的是
target-session weight-update headroom。当前实现是 additive path，adapter 与 sealed shared backbone 分开持有。

可归因的 H-add 保留 matched activity-only identity：

`h_i = fc_in(x_i + E_i^A) + P(T4_i)`，

其中 `E_i^A` 来自 matched Z4/activity route，`P` 无 bias 且 `P(0)=0`。H-add/Z4 是 W-add/Z4 的
exact structural alias。最小 logical 2x2 为 sealed W/Z4、sealed W/T4、aliased H/Z4，以及唯一需要
fresh training 的 H/T4。原始裸公式 `fc_in(x_i)+P(T4_i)` 同时删除了
activity identity，只能称 identity-route replacement，不能隔离 add site。Contract 还必须冻结
zero-initialization parity、teacher/decoder state 不变、support/query provenance 相同、
parameter/MAC/state accounting，以及可达到的 synthetic pass/fail test。Primary estimand 必须消去
generic adapter gain：

`(H-add(T4)-H-add(Z4)) - (W-add(T4)-W-add(Z4))`。

只有 source roster、normalizer、M30、loss、schedule、epoch bundle、teacher、query 与 implementation
binding 全部精确一致时，才可复用 A2 的 W-add checkpoint。这些检查以及 optimizer/parity proof 已在
preflight SHA `812d426ef73464ce4f6cb933856ad39035ffbe95c95114e38fc8596f125cd7b1` 下通过。
A1 的终局 routing 结果已在上方执行账本记录。

### B13. 闭式末层适配

**做什么。** 不再只在特征空间适配每 unit 四个数，而是在**参数空间**做闭式适配：在校准前缀上对 `psi` 的最后一层线性（或解码器 readout）做 ridge 求解。
**为什么。** 它仍然是无反传、无优化器的，与论文已有的主张完全同一含义，但它适配的是数千个参数而不是 `4N` 个输入。linear probing 众所周知能捕获微调收益的很大一部分，而 1.4 节提示微调收益很大。**这是在尊重部署约束的前提下，最大的未尝试容量提升。**
**成本。** 中等。需要仔细说明闭式参数适配在设备上的代价（一次 ridge 求解，加上存储求解后的层）。
**否决判据。** 只在 A10 显示存在实质 headroom 时运行。

### B14. 带符号读出 / 线性注意力

> **Root 审计已撤回。** attention probability 非负不代表 unit contribution 非负，因为
> value 与 output projection 都是 signed。下方结构前提不成立，本项不可执行，也没有实验排队。

### B15. 潜 query token

> **Root 审计已撤回。** 当前 decoder 的 cross-attention 与 FFN 对每个 query token
> 独立工作；若没有额外 query mixing/self-attention，未评分 latent query 不可能影响
> behavioural query。加入该路径后已是另一个假设。本轮没有实验排队。

### B16. 载体驱动的 read-in rank-1 修正

**做什么。** 让载体为每个 unit 生成对该 unit 有效 `fc_in` 行的 rank-1 修正。
**为什么。** 加性 `x_i + E_i` 与乘性 `x_i (1 + g)` 能平移和缩放一个 unit 的贡献，但结构上无法在输出空间**旋转**它的方向。这是经典编码模型所蕴含的机制，也是唯一尚不可表达的机制。
**否决判据。** 若它依赖 target-session 参数适配或以 adaptation headroom 为前提，必须先重写 A10
的监督量 contract；若是 source-trained forward-only 版本，则须另做 T4/Z4 factorial，并不自动被
A10 阻塞。本文档中风险最高的一项。

---

## 4. 不要重复

这里有两种不同的状态，务必区分。

### 4a. 已测量并被否证

- 任何在现有骨干上"增加"载体通路的融合机制（1.1 节第 1 类）。
- 把置信度或后验作为消费者的额外**输入**：`t4c`、confidence-FiLM `+0.003399`。
- 波形与 SNR 侧特征：`F1 - F0 = -0.0324`、`F2 - F0 = -0.0012`，对各自等维对照均为 indeterminate。
- 静态 electrode gate `-0.010817` 与 same-electrode relation `-0.001440`，均 `ineffective`。
- 加宽 identity 接口：CI64 `-0.020130` 已 terminal；H64 被禁止。
- N4 一族的 label-free **静态**描述子：`N4 - NS4 = +0.001588`，3/6 session。
- Fixed-K temporal prototype：Gate A2 输给了更简单的顺序无关边缘分布基线，`P20 - B20 = -0.041963`，0/4 session。
- 低标签预算下的 Wiener 收缩：`T4W3@15 - ordinary T4@15 = +0.000753`，基本为零，
  尽管 train-only proxy 在 27/27 session 改善。普通 `T4@15 - T4@50 = -0.058842`
  是预算差，不是 shrinkage effect。**W3 才是 train-only proxy 未必迁移的警告。**
- 二阶或 gain-field 载体项：从未运行，但它增加估计器参数、在 `M = 30-50` 下恶化条件数，而且按 1.5 节，冻结的消费者本来也不会花掉这个改进。

### 4b. 从未测量，但被 closure board 禁止

以下各项都有已实现的代码和通过的 CPU 测试套件，**完全没有任何数值结果**。它们是未测量，不是被否证。`ACTIVE_EXPERIMENT_CONTROL_BOARD.md` 第 7 节禁止它们。不要仅因为代码存在就复活其中任何一个。

- Residual-FiLM（`B3SCFR` / `B3SCFRS` / `B3SCFRA`）——在队列第 4 位被取消。
- Full-64-head oracle——从未启动；其结果目录只有一个 watcher 日志。
- T4 key-residual adapter——从未写过 runner。注意与之密切相关的 logit-residual 变体**跑过**并失败于 `-0.003142`，设计笔记把两者描述为同一想法的等价低状态形式。
- Electrode anchor（`t4anchor`）与 electrode embedding（`t4e`）——在设计 D 报告 `ineffective` 后从未排期。
- Decoupled K/V v2——在 epoch 0 内被杀；`~0.06` 对 `~0.58` 的读数是 kill signal，绝不是正式比较。

### 4c. 未决而非失败——值得收尾

`B3TStream + T4`（A14）有有利的历史 seed42 receipts，但没有保留 checkpoint，因此不能只补
content control。它是低优先级三格 reconstruction pilot，或九格三 seed matrix，不是仓库里
最便宜的未认领结果。

---

## 5. 推荐顺序

Root 审计后的顺序。旧的宽队列已经撤回：其中若干诊断是 exact null、规格不完整、只有
synthetic path，或者在统计上不可能通过自己写下的 gate。

| 步骤 | 项目 | 成本 | 阻塞什么 |
|---|---|---|---|
| 已完成 | v10 M30 AC4/Z4、固定 epoch 5–12、session-LOSO 的 A4 | CPU/forward-only | 正向 linear token-content 结果；不自动触发后续 |
| 已完成 | A11：权威 CPU 重放全部 B0 checkpoint | CPU/forward-only | 无 late continuation signal；continuation=false；SUA 分解已加入 schedule caveat |
| 已完成 | A2 v2 target-subject-shift interaction | 6 个 fresh source-training GPU cell | 终局正向 interaction `+0.235799`；同时报告 external T4/Z4 absolute means |
| 描述性 partial | A12 v4 attention audit | CPU/forward-only；`3/24` pair，aggregate SHA `2bf287...`；三个 attention-geometry delta 均为 6/6 session 负 | 无 causal saturation claim，也不 gate 任何实验 |
| 4 | 保留 B2 v2 作为独立 training-side contract | CPU contract 已完成；B1 终局且未选择 substrate；6-cell routing + 12-cell confirmation | 不自动启动；检验能否减少盲信而不丢失正确 T4 |
| 5 | 先重写 A10 的 supervision accounting | 先设计 | matched no-backprop-cost statement |
| 6 | A14 仅作为独立 efficiency branch | 三格 pilot；三 seed evidence 需九格 | B3TStream efficiency hypothesis |

**暂停：** A13、SUA A5、B8、B9、B14、B15。A3 必须先改成非 exact-null 的稳健性
干预。B3/B4 与后续 decoder redesign 不在本 handoff 队列中；B2 已形成 contract，但在操作上排在
B1 之后。弱或阴性的实验简洁收口，
不再据此扩张分支。

**当前策略边界。** A1/B1 的终局 routing 结果已在上方执行账本记录。固定 W-add 的
C1 `{MC-Maze,CO-native} x {T4,Z4}`、C2 `{legacy,equal-session} x {T4,Z4}` sampling，以及
CF1 activity-path dropout 仍为 held candidate；C1/C2 仍保持 held，不会自动进入队列。CF1 使用
`{p=0,p>0} x {T4,Z4}`。C2 第一阶段只表示 ordinary session-balanced MSE，不是 R2-native
loss，也不是后续 B5 episodic calibration-prefix/query objective。当前只有
`streaming_calibration_exp` 的 M2 sampler 实现了 `balance_sessions`；ordinary M2 config 默认关闭，
而且该开关只作用于 source-training sampler，不作用于 validation sampling；fixed per-session window
budget 也没有通过该 DataModule 暴露，而 SUA/A2 与 legacy SPINT sampler
没有同样的开关。CF1 要求在 activity identity 不可靠时更多依赖 T4；B2 carrier corruption
要求 carrier 错误时回退到 activity。两者方向相反，不能互称替代。每条暂停候选都必须先有独立
CPU contract 与 source-only selection，之后才可考虑 GPU。

---

## 6. 关于哪些是假设的诚实陈述

为使本文档日后仍可作为证据使用，各条主张的状态：

| 主张 | 状态 |
|---|---|
| 已测的 matched add-on mechanism 没有改善所选接口 | **在 heterogeneous protocol 下测得。** 只保留 1.1 节真正 matched 的 contrast；B15 是 activity-only，重复 logit 行属于同一运行。 |
| 两个替换骨干的变体灾难性失败 | **效应已测、原因未解决。** Decoupled K/V v1 `-0.444658`、fixed-slot router `-0.177935`，均有记录在案的混淆。 |
| Zero4 就是 activity-only identity 编码器 | **代码已核实。** `mask_standardized_t4`，arm `z4`。 |
| SUA headline 有四分之一来自 side 通路而非载体 | **已测量。** `Z4 - B0 = +0.089591`；`T4 - Z4 = +0.248968`。 |
| B0 在 epoch 12 仍有向上趋势 | **A11 权威 CPU full replay 不支持。** 两个 seed 的 late-minus-early 为负，第三个仅 `+0.00187`，mean `-0.00834`；frozen continuation flag 为 false。 |
| T4 的 `b` 与 activity 冗余 | **在 M2 上已测量。** `r = 0.996`。A4 另行证明 AC4 token 比 Z4 含有显著更多可跨 session 线性恢复的 `[a,c]` phase；H1 overlap residual `0.804819` 不是 gain predictor。 |
| AC4 比 Z4 保留更多线性可恢复 tuning phase | **A4 已测量。** Null-relative advantage `+0.517299` 对 `+0.090144`（`5.74x`），raw delta `+0.430155`，`18/18` 为正。Z4 是 phase-poor 而非 phase-free；不证明 decoder benefit。 |
| 在观察到的 C-to-M subject shift 下 relative carrier value 增加 | **A2 v2 已测量。** Interaction `+0.235799`，三个 seed interaction 均为正，crossed seed-by-session interval `[+0.100852,+0.371768]`。External absolute `T4=+0.341367`、`Z4=-0.143399`；因此 interaction 不是 absolute T4 lift，也不是 correspondence test。 |
| 一个冻结 H1 consumer 对 query-fitted leakage perturbation 局部不敏感 | **局部已测量。** 差值 `-0.002859`；存在 fit/eval overlap 与 distribution mismatch，不是严格 ceiling；`estimator_saturation_proven: false`。 |
| 同 checkpoint 的载体置零会高估载体 | **仅在一个 H1 fold/seed receipt 上测得。** 置零 `0.103871` 对重训 `0.038895`，倍数 `2.67`；不得推广。 |
| 无梯度结果之上的 headroom 很大 | **仅为不匹配的线索。** 体制混淆、使用标签的 oracle。A10 测量它。 |
| `lambda_E` 蒸馏项在 M2 上压住了载体 | **尚未建立；B1 routing 结果见第 -1.3 节。** |
| head concentration 解释局部不敏感 | **尚未建立。** A12 的 `3/24` 描述性 partial aggregate 有三个 6/6-negative geometry delta，但没有 intervention、inference 或 causal gate。 |
| hidden-space carrier addition 能改善 content use | **Routing pilot 低于 preregistered gate 后停止；见第 -1.3 节。** |
| 当前 MC-Maze teacher 造成 domain-mismatch penalty | **合理但尚未隔离。** C1 必须固定 add site、改变 teacher domain，并包含 T4/Z4 siblings。 |
| equal-session sampling 改善 carrier interaction | **未测试。** M2 仅有部分 sampler infrastructure；C2 不得与 R2-native 或 episodic training 混为一谈。 |
| softmax 的非负性限制带符号群体读出 | **已撤回。** signed value/output projection 使该前提不成立。 |
| 载体增益落在理论预测的分层上 | **未测试，且当前 A13 scaffold 无效。** 缺 checkpoint/query provenance 与 cluster-aware resampling；不能形成 mechanism claim。 |

本文档不授权任何 GPU 运行，不重开任何已关闭的杠杆，不改变任何已封存的结果。

## 7. 修订说明

本文档于 2026-08-12 在三次独立的只读仓库审计之后修订。修订内容：新增 1.5 节、A11 至 A14、以及 4a/4b/4c；把 1.1 节的表述从"八个平的 null"更正为两个不同的失败类别；更正 1.2 节 headline 算术中的 `Z4 - B0` side 通路混淆与 M2 epoch 不匹配；把 A4 改成 matched-null-relative 报告并记录 protocol timestamp 限制；加入 B9 estimator equivalence blocker；并重排第 5 节，使 consumer diagnostics 先于 carrier 改进。英文原件为 `HANDOFF_NEXT_ROUND_DIRECTIONS_20260812.md`，以英文版为准。

**2026-08-13 补充。** 加入 A2 三 seed 终局 interaction 与四个 absolute means；记录 A12 的
`3/24` 描述性 partial aggregate；并在上方执行账本中汇总 A1/B1 的终局 routing 结果。C1/C2 保持 held。
另外，`/mnt/data/SPINT_cold_archive/2026-08-13/` 现包含初始 `8.90 GiB`，以及 sealed batch 的
source-unique `81.9967 GiB` 迁移；由于 hard-link 去重，新增归档占用为 `74.5878 GiB`。随后，
`49.940 GiB` 的 `streaming_calibration_exp/logs/train` 在 path-compatibility smoke test 通过后也完成归档，
原绝对路径保留为 symlink。根盘剩余空间约 `208 GB`；pointer-plus-SHA manifest 支持恢复且不改变
authoritative artifact。
