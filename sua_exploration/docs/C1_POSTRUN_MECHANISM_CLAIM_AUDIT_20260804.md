# C1 paired SUA / pseudo-MUA：post-run 机制与发表主张审计

**日期：** 2026-08-04（Asia/Hong_Kong）  
**性质：** 独立、score-blind、预结果审计  
**适用程序：** `t4_paired_view_c1_fresh_prelaunch_v3r2_remote_gate_allowlist`

## 0. 审计边界和首要结论

本审计只读取冻结协议、聚合器、数据/训练实现、score-free CPU gate，以及已经终止的
Path 1 / Path 2 审计。它没有读取当前 C1 任一中间或最终 R²，没有打开或解析 6 个
formal SUA test session，没有运行 GPU，也没有修改 v3r2 的 `source_map`、receipt、remote
handoff 或任何运行文件。本文件是新的外部解释文档，不属于正在运行的冻结程序。

首要结论有四条：

1. **五个冻结 gate 全过，支持的是“双视图非劣 + 两视图正确 T4 行附着有效 +
   cross-view gap 不恶化”。它本身不等价于“paired-view co-training 提高了 R²”。**
   原因是两个 shared-vs-separate 主 gate 的阈值是 `lower >= -0.03`，不是
   `lower > 0`。
2. **仅凭五个 gate 的任意布尔组合，都不能推出严格 accuracy improvement。** 如需写
   “improves”，必须另外检查 shared-minus-separate 的正向强证据；即使该证据出现，因
   它不是冻结主 gate，也只能作为本批 development evidence 的探索性二级结论，独立
   confirmatory endpoint 才能把它升级为确认性主张。
3. **C1 的因果处理是整个离线 paired-view training package。** 它不能把增益进一步归因
   于 consistency、manifold alignment、T4 的某一个分量，或真实 MUA 泛化；这些因素在
   C1 中没有被分别随机化/消融。
4. **只有五 gate 全过才许可 C2；任何 gate 失败都应按冻结规则停止 C-family，不做
   loss-weight、宽度、epoch、seed 或方向性补救。** 若全过且已经出现双视图严格正向
   证据，最节省 GPU 的路线反而是停止 C2，冻结候选并转向一次独立确认。

在查看任何 aggregate 之前，还存在一个更高优先级的 evidence gate：外部 finalizer 必须
先证明 12 个 cell、remote transfer、source hashes、runtime/data attestations 和 TS4
microfit closure 完整。若 finalizer 失败，结论是 **evidence incomplete**，不是 accuracy
positive 或 negative。

## 1. 冻结问题到底是什么

### 1.1 四个 arm 和共同评价边界

冻结矩阵是 4 arms × seeds `{42,43,44}`：

| Arm | 离线 source training | Development 时输出的视图 |
| --- | --- | --- |
| `separate_sua_t4` | 只用 SUA，task-only | SUA |
| `separate_pseudo_mua_t4` | 只用 pseudo-MUA，task-only | pseudo-MUA |
| `shared_t4` | 同一权重，`0.5 L_SUA + 0.5 L_pseudo` | SUA、pseudo-MUA |
| `shared_ts4` | 同一权重，同一 paired objective，T4 行被打乱 | SUA、pseudo-MUA |

所有 arm 都固定：

- 27 个 source-train session、6 个 reused-development held-out session；6 个 formal session
  仍 sealed；
- 12 epochs，无 early stopping，固定平均 epochs 5–12；
- source activity calibration 为 chronological first 10 trials；
- held-out forward activity calibration 为 chronological first 30 trials；
- T4 使用 first 50 rewarded trials 的 direction labels 和 per-unit/per-channel rates；
- query scoring 从 trial 50 开始；
- seed × held-out-session 的每个比较值先对 8 个固定 epoch 求平均；
- held-out calibration/evaluation 无 optimizer step、无 backward pass；
- encoder 和 decoder 在 27 个 source session 上**共同离线训练**，`freeze_decoder=false`。

因此准确措辞是 **supervised, calibration-time backprop-free held-out-session adaptation**。
它不是 label-free，也不是 backprop-free training。部署时模型权重冻结；离线 source
training 仍正常使用反向传播。

### 1.2 SUA 与 pseudo-MUA 的关系

pseudo-MUA 不是独立采集的 threshold-crossing MUA。它由同一份 SUA 记录按照 electrode
把 spike counts 作确定性求和得到。两视图具有相同的 session、时间窗和 behavior target；
pooling 只改变 neural channel axis。pseudo-MUA 的 T4 必须先逐 trial 合并 channel rate，
然后重新拟合 T4，绝不是把 SUA unit T4 行取平均。

这个构造给 C1 一个真实的机制动机：在共享 cosine design 下，`a`、`c`、`b` 对求和是
线性的，`m` 从合并后的 `a,c` 重新计算。C1 因而检验的是一套权重能否适应一个已知的
unit-merge transform，而不是两个独立 modality 之间的普遍迁移。

### 1.3 Paired objective 的精确含义

每个 aligned microbatch 先对 SUA 做 `0.5 * L_task` backward，再对 pseudo-MUA 做
`0.5 * L_task` backward，最后只执行一个 shared optimizer step：

```text
L_C1 = 0.5 L_task(SUA) + 0.5 L_task(pseudo-MUA)
lambda_consistency = 0
view_specific_heads = false
```

顺序 backward 对无状态 task loss 产生与相加 loss 相同的累计梯度，同时避免同时保留
两张大图。它不包含 prediction matching、embedding matching、teacher-student
distillation 或 view classifier。因此，C1 的任何正向结果都不能写成“consistency
regularization 有效”。

## 2. 五个冻结 gate 的数学定义

令

```text
R[a,v,s,j] = arm a、view v、seed s、development session j
             在 epochs 5..12 上的平均 R²
```

其中 `s` 有 3 个，`j` 有 6 个。聚合器对每个 paired delta 保存完整 `3 × 6` 矩阵、3 个
seed mean、6 个 session mean、基于 seed means 的 `mean ± 2 SE`，以及同时重采样 seed
和 session index 的 hierarchical bootstrap interval。

定义五个布尔 gate：

### N_S：SUA non-inferiority

```text
D_NS = R[shared_t4, SUA] - R[separate_sua_t4, SUA]
N_S  = (two-SE lower >= -0.03) AND (bootstrap lower >= -0.03)
```

### N_P：pseudo-MUA non-inferiority

```text
D_NP = R[shared_t4, pseudo] - R[separate_pseudo_mua_t4, pseudo]
N_P  = (two-SE lower >= -0.03) AND (bootstrap lower >= -0.03)
```

### A_S / A_P：correct-content attachment

```text
D_AS = R[shared_t4, SUA]    - R[shared_ts4, SUA]
D_AP = R[shared_t4, pseudo] - R[shared_ts4, pseudo]

A_v = (two-SE lower > 0)
      AND (bootstrap lower > 0)
      AND (positive seed means = 3/3)
      AND (positive session means >= 5/6)
```

### G：cross-view gap 不增加

先在每个 seed/session cell 内定义：

```text
gap_separate = abs(R[separate_sua_t4,SUA]
                   - R[separate_pseudo_mua_t4,pseudo])
gap_shared   = abs(R[shared_t4,SUA]
                   - R[shared_t4,pseudo])
D_G          = gap_shared - gap_separate

G = (two-SE upper <= +0.03) AND (bootstrap upper <= +0.03)
```

冻结聚合器只在

```text
C1_PASS = N_S AND N_P AND A_S AND A_P AND G
```

时给出 `pass_enter_conditional_c2`；其余所有组合都是 `stop_c_no_rescue`。

### 2.1 五个 gate 为什么没有“improvement”布尔量

`N_S` 和 `N_P` 允许 shared model 最差到 practical margin `-0.03`。所以以下三种数值情形
都可能得到 `N_v=true`：

1. shared 明显高于 separate；
2. shared 与 separate 几乎相同；
3. shared 略低于 separate，但不劣于 `-0.03`。

为了 score 出来后不发生措辞漂移，定义一个**只用于解释、未预注册为 C1 主 gate**的
严格正向诊断：

```text
I_v = (D_Nv two-SE lower > 0)
      AND (D_Nv bootstrap lower > 0)
      AND (positive seed means = 3/3)
      AND (positive session means >= 5/6)
```

解释层级必须是：

| 观察 | 可以写什么 |
| --- | --- |
| `N_v=true`，但 `I_v=false` | shared 在该视图满足预注册非劣界；不能写 improves |
| mean delta `>0`，但 `I_v=false` | directionally higher / descriptive positive mean；不能写 robust improvement |
| `I_v=true` | development held-out 上有强探索性 improvement evidence；仍非预注册主 gate |
| 独立 endpoint 上预先冻结并通过 `I_v` 类门 | 才能升级为 confirmatory improvement claim |

`I_v` 只能从最终一次 aggregate 已经保存的 paired summary 读取，不触发新训练，也不能用来
选择 epoch、seed、session 或下一组超参数。

## 3. 32 种 gate 组合的穷尽 decision matrix

下表按互斥优先级压缩了全部 `2^5 = 32` 种组合；`*` 表示任意值。各行覆盖数相加严格为
32。表中的决定假定 publication-readiness finalizer 已先通过。

| 优先级 | `N_S` | `N_P` | `A_S` | `A_P` | `G` | 覆盖组合数 | 冻结决定 | 最强可支持结论 |
| ---: | :---: | :---: | :---: | :---: | :---: | ---: | --- | --- |
| 1 | 0 | * | * | * | * | 16 | stop | shared training 对 SUA 未达非劣；paired dual-view 主假设为 negative，其他 gate 不能救回 |
| 2 | 1 | 0 | * | * | * | 8 | stop | SUA 可非劣，但 pseudo-MUA 未达非劣；只能作 view-asymmetric negative |
| 3 | 1 | 1 | * | * | 0 | 4 | stop | 两视图分别非劣，但 sharing 使绝对 performance gap 恶化超过 margin；不支持 granularity preservation |
| 4 | 1 | 1 | 0 | 0 | 1 | 1 | stop | one-model engineering non-inferiority/gap preservation；没有正确 T4 行附着证据 |
| 5 | 1 | 1 | 1 | 0 | 1 | 1 | stop | 仅 SUA 有正确附着证据；不能作双视图 T4 mechanism claim |
| 6 | 1 | 1 | 0 | 1 | 1 | 1 | stop | 仅 pseudo-MUA 有正确附着证据；不能作双视图 T4 mechanism claim |
| 7 | 1 | 1 | 1 | 1 | 1 | 1 | pass | 一套权重双视图非劣、两视图正确行附着有效、gap 不恶化；许可但不强制 C2 |

这张表的两个重要读法是：

- `A_S/A_P/G` 无论多好，都不能补偿任一 view 的 non-inferiority 失败；
- `N_S=N_P=A_S=A_P=G=1` 仍然只保证 **non-inferiority**，是否有 accuracy improvement
  必须另看 `I_S/I_P`。

### 3.1 Full C1 pass 后的二级 accuracy matrix

仅当五 gate 全过时，再使用以下矩阵解释 shared-vs-separate accuracy：

| `I_S` | `I_P` | 可发表的 accuracy 层级 | GPU 决策 |
| :---: | :---: | --- | --- |
| 0 | 0 | 双视图 non-inferior；价值是 one-weight-set consolidation，不是更高 R² | 原则上停止；只有 source-only C2 gate 明确时才跑 3 个 C2 cell |
| 1 | 0 | SUA exploratory improvement + pseudo-MUA non-inferiority | 只有预先指定的 C2 方向确实针对 pseudo gap 时才考虑，否则停止 |
| 0 | 1 | pseudo-MUA exploratory improvement + SUA non-inferiority | 当前 one-way C2 方向不针对 SUA 修复；不要看结果后反转 teacher 方向 |
| 1 | 1 | 双视图 exploratory improvement，且全部主机制 gate 通过 | 不跑 C2；冻结候选，优先做一次独立/formal/native confirmation |

如果只是 mean delta 正而 `I_v` 不过，表中仍按 `I_v=0` 处理。不能为了更好看的措辞临时放宽
正向判据。

## 4. 每个对照实际识别什么

| 对照 | 保持不变 | 改变 | 能识别 | 不能识别 |
| --- | --- | --- | --- | --- |
| `shared_t4:SUA - separate_sua_t4:SUA` | source sessions、SUA input、T4、architecture、seed、epoch/eval 边界 | single-view task training → equal-weight paired training | 整个 paired-training package 对 SUA 的总效应 | consistency、某个 T4 分量、配对本身 vs 额外 view exposure、native generalization |
| `shared_t4:pseudo - separate_pseudo_t4:pseudo` | source sessions、pseudo input、pooled-rate T4、architecture、seed、eval | single-view → paired training | 整个 paired-training package 对 pseudo view 的总效应 | native MUA、独立 modality transfer、SUA teacher mediation |
| `shared_t4 - shared_ts4` | paired objective、四维宽度、T4 row multiset、label budget、normalizer、architecture | T4 row 与真实 neural row 的对应关系 | 正确 row attachment 是否比错误 attachment 有用 | T4 vs no descriptor、phase vs magnitude/rate、label necessity、paired interaction |
| `gap_shared - gap_separate` | seed/session/eval | shared 与 separate 的 absolute inter-view performance disparity | sharing 是否把两视图 performance disparity 扩大超过 margin | prediction agreement、latent alignment、哪个 view 更好、per-sample invariance |

### 4.1 SUA 的因果对照

SUA 对照中，两模型在同一 held-out SUA view 上评价，也使用同样的 50-trial unit-level
T4。唯一预期 treatment 是 source training 时是否同时接收 aligned pseudo-MUA 梯度。因此
delta 可以归因于“paired-view offline training package 相对 single-view training”的效果。

但这个 package 同时改变了多件事：

- 每个 optimizer step 包含两个 view forward/backward，而 separate 只有一个；
- SUA task gradient 在 shared objective 中乘 `0.5`；
- 额外 pseudo-MUA view 是同一 recording 的确定性聚合，起 data augmentation / gradient
  regularization 的作用；
- 没有 doubled-SUA exposure control，也没有 unpaired-two-view control。
- shared 与 separate 使用两个不同的 trainer entrypoint；协议冻结了相同 nominal seed 和同一
  architecture，但没有保存一个证明两条入口初始 `state_dict` bit-exact 的 receipt。因此
  不能把 matched seed 进一步夸大为“除 objective 外所有随机状态逐位相同”。

所以若 SUA 提升，最严谨的解释是 **multi-granularity paired training package helped SUA**，
不能单独归因于“pair alignment”或“更多数据”中的任何一个因素。

### 4.2 pseudo-MUA 的因果对照

pseudo-MUA 对照保持相同的 electrode-pooled neural stream，并从 pooled trial rates 重新拟合
T4。shared 模型相对 separate 模型的差值同样识别整个 paired training package 对这个
受控 pooled view 的效果。

它不识别真实 threshold-crossing MUA。两视图来自同一 SUA recording，噪声、session、
behavior targets 和 spike events 高度共享；pseudo-MUA 只是一个确定性 granularity transform。
因此即使 `N_P` 和 `I_P` 都为真，也只能先写 pseudo-MUA / electrode-pooled SUA，不能把
“pseudo”从方法名中删除。

### 4.3 TS4 row-shuffle 的识别力

TS4 在每个 view 内：

1. 先加载与 T4 完全相同的 raw descriptor；
2. 使用该 view 的 source-train T4 normalizer 标准化；
3. 对完整四维 row `[a,c,m,b]` 沿 unit/channel axis 作 deterministic permutation；
4. neural activity row 不动，因而打断 descriptor-to-neural-row attachment；
5. 每个 session 内的完整 row multiset、四维宽度和 label-derived 数值全部保留。

冻结的 score-free microfit audit 已检查 seeds 42/43/44、两视图、全部 33 个 permitted
source/development session 的 permutation 均为 nonidentity，并保留完整 row multiset。
因此本 C1 不存在“TS4 恰好全局 identity”这一静默失效。

若 `A_v=true`，可以排除的解释包括：

- 仅仅多了 4 个 input coordinates；
- 仅仅有一个 session-level T4 值分布；
- T4/TS4 parameter count 或 normalizer 不同。

它支持“正确 attached functional descriptor 对该 view 有用”。但局限必须同时写出：

- TS4 不是 label shuffle；direction labels 产生的 descriptor multiset 仍存在；
- full row 一起移动，不能区分 `[a,c]`、`m` 或 `b` 哪部分造成效果；
- C1 没有 shared-F0，所以不能仅靠 C1 声称 T4 相对 no-descriptor 的绝对增益；
- C1 没有 separate-TS4，所以不能做 difference-in-differences 来证明 paired training
  特别增强了 attachment；
- permutation 由 seed 和 row count 决定，不是一个新的随机 biological replicate；相同
  row count 会重复同一 index map，训练也可能学习任何可复现的排序残余；
- nonidentity index permutation 不保证每一行的数值都改变；相同/近相同 descriptor rows 会
  让局部 attachment 残留，所以 TS4 是破坏 correspondence 的实用 control，不是数学上的
  零信息 oracle；
- T4 与 TS4 是分别从头离线训练的两个模型，训练和 evaluation 始终使用各自的正确或错误
  attachment；它检验“可学习的正确 attachment”，不是对同一冻结 checkpoint 作
  inference-only row ablation；
- `T4 > TS4` 证明 attachment，不直接证明 direction/phase mechanism。SUA `[a,c]`
  component 机制来自历史 AC4/RS4 实验，不能未经 pseudo-MUA component gate 自动移植到
  pseudo view。

### 4.4 Cross-view gap 的含义

`G` 比较的是两个 scalar accuracy 之间的**绝对差**，不是两组 prediction 的差。它回答：

> 相比两个分别优化的模型，使用一套 shared weights 是否把 SUA 与 pseudo-MUA 的 R²
> disparity 增加超过 `0.03`？

它不回答：

- 两个 view 对同一时间窗是否给出相同 prediction；
- latent embedding 是否对齐；
- SUA 与 pseudo-MUA 哪一个更强，因为绝对值移除了方向；
- 一个 view 是否改善、另一个是否退化；
- gap 缩小是否来自两者一起变差。

最后一点由 `N_S/N_P` 部分约束，但 non-inferiority margin 仍允许两者各自略降。因此 `G`
只能与两个 non-inferiority gate 联合解释，绝不能单独写成 view invariance。

## 5. Full pass 时能说什么、不能说什么

### 5.1 五 gate 全过后的最强主结论

推荐主句：

> On the six frozen development-held-out sessions of DANDI 000688 sub-C/CO, one
> source-trained B3S/T4 weight set met the predeclared non-inferiority margins in both sorted-SUA
> and deterministic electrode-pooled pseudo-MUA views. Correct per-channel T4 attachment provided
> a reproducible advantage over row-shuffled attachment in both views, and sharing did not increase
> the absolute cross-view performance gap
> beyond the frozen margin. New-session calibration used labelled T4 fitting and forward neural
> computation without calibration-time backpropagation.

这句话没有把 non-inferiority 偷换成 improvement，也准确披露了 development、pseudo-MUA、
labels、offline training 和 calibration-time no-backward 边界。

### 5.2 只有在 `I_S/I_P` 成立时才可追加的句子

- `I_S=true`：可追加“shared training showed a robust exploratory positive delta over the
  separate SUA model on this development scope”。
- `I_P=true`：同理，但必须保留 pseudo-MUA 限定。
- 两者都真：可写“exploratory accuracy improvement in both controlled views”，同时明确它
  不是冻结 primary superiority gate，且需要 independent confirmation。

不要用“statistically significant”。`mean ± 2 SE` 在 3 个 seed clusters 下不是常规 95%
t interval；hierarchical bootstrap 也只有 3 个 seed clusters，是稳定性 gate，不是广泛
population inference。

### 5.3 无论结果多好都不能声称

1. formal、sealed 或 independent final SUA test 已通过；
2. native MUA 泛化已由 C1 证明；
3. T4 在 M1/M2 或其他 subject 上普遍有效；
4. calibration 是 label-free 或 unsupervised；
5. 整个训练过程 backprop-free，或 decoder 从未被训练；
6. consistency loss、FiLM、dynamic weights、subspace alignment 或 memory 起了作用；
7. SUA/pseudo latent manifolds 已对齐，或两视图 predictions 一致；
8. 增益由 `[a,c]`、preferred direction 或 pooling homomorphism 单独因果中介；
9. 正确 T4 attachment 是 shared model 达到 non-inferiority 的必要条件；C1 只证明它优于
   shared TS4，并没有检验 shared TS4 是否也可能落在 non-inferiority margin 内；
10. paired training 比 separate training 更省单次在线 MAC；正确的是 dual-view deployment
   只需要一份 weights、相对两套 separate deployment 少一份模型存储，单 view 在线 decoder
   计算图并未变轻；
11. 一个混合 3090/5070 Ti 的 pooled latency/memory 数值。运行成本必须按 hardware group
    分开报告。

## 6. Partial / negative 结果如何写

### 6.1 任一 non-inferiority gate 失败

这是 paired shared-weight 主假设的直接 negative：共享训练使至少一个 view 的不确定性界越过
`-0.03`。即使该 view 的 T4 attachment gate 通过，也只能说 shared architecture 内正确
descriptor 比 shuffled descriptor 好，不能说 one shared model 可以替代 separate models。

推荐措辞：

> Correct T4 attachment remained detectable in [view, if applicable], but equal-weight paired
> training did not meet the predeclared non-inferiority margin in [failed view]. The shared-weight
> deployment hypothesis was therefore rejected under this protocol.

### 6.2 两个 non-inferiority 过、`G` 失败

这是一项有信息量的 partial result：one-model 在每个 view 单独看都没有超过 practical loss
margin，但 sharing 改变了两视图相对表现，使 gap 恶化超过允许值。可报告“view-wise
non-inferiority without granularity-gap preservation”，不能称 robust/invariant carrier。

### 6.3 Non-inferiority 和 gap 过，但 attachment 失败

这是 engineering consolidation 与 functional-identity mechanism 的分离：一套模型可能仍能
服务两视图，但 C1 没能证明它依赖正确的 T4-to-neural-row correspondence。

- 两个 attachment 都失败：只剩 non-inferior shared training；可能来自 activity path、
  descriptor multiset、generic regularization 或 shared capacity。
- 只一个 attachment 通过：只能作 view-specific attachment 结论。

根据冻结协议，这三种情况都必须停止，不能在看到结果后补 F0、AC4、另一种 shuffle 或额外
seed 来救 gate。

### 6.4 Provenance/finalizer 失败

不要读取/解释 aggregate。先做不涉及 score 的 evidence repair；若能证明只是 transfer/path
closure 问题，可保持模型和指标不变修复 finalization。若 run identity、source hash、artifact
hash 或数据边界无法证明，则该 cell 不可用于科学结论。不得把 provenance failure 当成
accuracy failure，也不得用新训练结果覆盖旧证据而不重新冻结完整协议。

## 7. Path 1 / Path 2 对 C1 的约束

历史 terminal audits 给出的是两条明确的停止边界：

- Path 1 的 generic cross-budget reliability correction 在 target-free development 上没有
  transferable incremental signal，EB/second-harmonic/Poisson analytic upgrades 也没有 source-only
  winner；
- Path 2 的 P20 temporal prototype 被更简单的 order-invariant B20 marginal carrier 在 4/4
  source sessions 上击败，不能升级为 K/V memory 或 cross-attention。

这些 negative 不否定 C1，因为 C1 没有更改 T4 estimator，也没有加入 temporal memory；它
只检验已有 T4 substrate 在确定性 granularity transform 下能否共用离线权重。

反过来，C1 失败也不能作为重新打开 Path 1/2 的理由。失败说明 shared-weight training family
没有达到自己的 gate，而不是 estimator correction 或 temporal memory 突然获得了新证据。
任何重开都需要独立数据和全新、预先冻结的机制假设。

## 8. 最小 C2 或停止路线

### 8.1 C1 任一 gate 失败：停止

不运行 C2，不改 `lambda`，不增加 seed，不改变 0.5/0.5 权重，不扫 epoch，不反转 teacher
方向，不补新的 fusion path。保留：

- pooling homomorphism 和 33-session data audit；
- separate-view T4 的既有正向证据；
- C1 对 shared-weight 假设的明确 negative/partial evidence；
- Path 1/2 的 terminal mechanism findings。

这是最高信息/最低 GPU 浪费的收口。

### 8.2 Full C1 pass 且 `I_S=I_P=1`：停止 C2，转 independent confirmation

C2 的主要作用是争取额外 accuracy。如果 C1 已经在两视图产生强正向 exploratory delta，
再做 consistency 很可能只增加一个难归因的训练变体。应冻结 C1 candidate，优先选择一个：

1. 经单独授权的一次 formal SUA endpoint；或
2. external subject 的同类 paired-granularity test；或
3. 为 native MUA 单独建立 compatibility/provenance gate 后的一次固定 M2 confirmation。

其中任何 native-MUA 测试都必须另立协议；不能把 pseudo-MUA 的 pass 直接搬过去。

### 8.3 Full C1 pass 但至少一个 `I_v=0`：最多一个 C2 candidate

只有在 source-only、development-blind 的 gate 能冻结一个 `lambda` 时，才运行最小 C2：

这里的“许可”只是 post-run 路线建议，**不构成运行授权**。当前 C1 v3r2 receipt 没有授权
C2；若满足触发条件，必须先另立并冻结一个新的 C2 prelaunch receipt，绑定 source-only
`lambda` 选择证据、代码/数据 hashes、三 seed 计划、停止门和新的 result/checkpoint roots。
在该 receipt 通过独立 verifier 之前，C2 的 GPU run 数仍为零。

```text
L_C2 = 0.5 L_task(SUA)
     + 0.5 L_task(pseudo-MUA)
     + lambda * D(y_pseudo, stopgrad(y_SUA))
```

约束：

- 只允许一个预选 `lambda`，不得用 6 个 C1 development sessions 调参；
- architecture、T4 M=50、Q=30、trial-50 query、epochs 5–12、seeds 42/43/44 全部不变；
- 只新增 `C2_shared_t4` 三个 run，复用冻结 C1 `shared_t4` 作为 matched comparator；
- 不新增 TS4/F0/宽度 arm；C1 已完成 attachment gate，C2 的问题只应是 objective；
- deployment architecture/state/MAC 不变；consistency 只存在于 source training；
- 若 source-only `lambda` gate 选不出唯一候选，则 0 GPU 停止。

建议 C2 success rule 预先冻结为：

```text
pseudo-MUA: C2 - C1 shared_t4 满足 I_P 型严格正向门
SUA:        C2 - C1 shared_t4 两种 lower bound 均 >= -0.03
gap:        C2 absolute cross-view gap - C1 gap 的两种 upper bound均 <= +0.03
```

这是因为指定方向是 pseudo prediction 向 stop-gradient SUA prediction 靠拢。若 C1 结果显示
真正需要修复的是 SUA，而 pseudo 已强正向，则这个 C2 方向没有先验针对性；不能看完结果再
把 teacher/student 反转，应该停止或在独立数据上另立新计划。

C2 任一门失败即终止，不做第二个 `lambda`。C2 若只达到 non-inferiority 而未取得指定
pseudo-MUA strict-positive gate，也没有增加足够的 accuracy 发表价值，应保留 C1。

## 9. Post-run 执行清单

结果完成后只按以下顺序执行一次：

1. 等待全部 12 个 cell terminal-complete；不提前读取单 cell/单 seed R²。
2. 完成 score-blind seed-44 transfer manifest，并运行冻结 external finalizer。
3. 只有 `publication_readiness.status=passed` 且 aggregate SHA 被绑定后，读取唯一 aggregate。
4. 原样记录 `N_S,N_P,A_S,A_P,G`、完整 seed/session sign counts、两种 bounds；不删 outlier。
5. 用第 3 节的 32-combination matrix 给出 primary decision。
6. 五 gate 全过时才计算解释性 `I_S/I_P`；它们不改变 frozen `c1_pass`。
7. 选择第 8 节唯一允许的路径：停止、一次 independent confirmation，或最多 3-run C2。
8. 不访问 formal SUA，除非另有明确授权和候选冻结 receipt。
9. 发表表中同时披露：M_activity=30、M_T4=50 labelled trials、score start=50、27/6/6、
   fixed epochs 5–12、offline decoder training、calibration-time no-backward、pseudo-MUA
   construction、one-weight-copy deployment 和分 hardware 的 cost。

## 10. Evidence anchors

本审计使用下列 score-blind/frozen evidence anchor；SHA 是审计时当前字节的标识：

| Evidence | SHA-256 | 用途 |
| --- | --- | --- |
| `results/t4_paired_view_c1_fresh_prelaunch_v3r2_remote_gate_allowlist/receipt.json` | `8b17c19515fa0e6cc122233fd20cf7e87a616287fa247e5eacb136d6a56c9d85` | v3r2 matrix、protocol、cost/data gates |
| `scripts/aggregate_t4_paired_view_c1.py` | `428982de0de7b72446583547ce8402211f7230cc155577af1cde5a252cace9e3` | 五个 gate 的实际实现 |
| `mc_maze/paired_view_c1.py` | `0978643d1ce90bb66733610ca0132548bfe89a5ee0a98828885a9c3a489e9c0a` | paired axes/targets/count conservation contract |
| `streaming_calibration_exp/src/models/paired_view_c1_module.py` | `16d3c318a51c8b34e1a44f725e03460446f9dee4b3130ba39d4439613211efa9` | 0.5/0.5 sequential backward、lambda=0 |
| `mc_maze/unit_side_features.py` | `059faefcd766dfc8e25253d9ded2b619a46dea408e6f00a30cfa5b2ecd185ab6` | pooled-rate T4 与 TS4 row shuffle semantics |
| `docs/PATH12_TERMINAL_GATE_AUDIT_20260804.md` | `cbb660fdfe1cb083f6b30c4fb86395ace6e475434b9c78dbd1eda84e2c9e061a` | Path 1/2 terminal boundaries |
| `docs/HELDOUT_BP_FREE_CALIBRATION_METHOD_CLOSURE_20260804.md` | `424becb0b0f2eac4f6e270974016ab3dd1f53ac753bc602b7c4a3ea300676afe` | offline/held-out boundary及跨 SUA/MUA claim discipline |

## 11. 一句话决策规则

```text
finalizer fail -> evidence incomplete, no score decision
any C1 gate fail -> stop C-family, report exact partial/negative mechanism
all C1 gates pass, I_S/I_P not both true -> one-weight-set noninferiority;
                                         at most one source-selected 3-run C2
all C1 gates pass, I_S=I_P=true -> freeze C1, stop GPU tuning, seek independent confirmation
```
