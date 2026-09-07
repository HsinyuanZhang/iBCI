# C1 到 native MUA 的主张桥接审计

**日期：** 2026-08-04（Asia/Hong_Kong）  
**性质：** 独立、score-blind、只读证据审计  
**面向主线：** held-out-session supervised backprop-free calibration  
**当前 C1：** `t4_paired_view_c1_fresh_prelaunch_v3r2_remote_gate_allowlist`

## 0. 审计边界

本审计没有读取正在运行的 C1 任一中间或最终 R²，没有解析或打开 6 个 formal SUA
session，没有运行 GPU，没有调用或提交 EvalAI，也没有修改 v3r2 `source_map`、receipt、
handoff、runner、checkpoint 或运行结果。它只读取已经存在的 native-MUA、SPINT
replication、support/query-disjoint replay、官方提交 receipt、method-closure 和 C1 冻结协议。

本文件是运行程序之外的新解释文档。它不是 GPU launch receipt，也不授权 formal 数据访问、
EvalAI 行为或一个新的 native-MUA score。

## 1. 结论先行

1. **native M2 已有可信且重要的 official system-level positive result。** T4 submission
   `578221` 的 organizer-hidden held-out R² 为 `0.30324395`，原始 SPINT submission
   `578218` 为 `0.18647872`，系统差为 `+0.11676523`。这足以支持“提交的 T4 系统对
   native M2 held-out session 有用”，但不能支持“这 `+0.1168` 全部由 T4 descriptor
   单独造成”，因为两次 submission 使用 epoch-34 与 epoch-27 decoder。
2. **native M2 的 local causal evidence 是正向但还未完成稳定性闭环。** 正确的 M33/q33
   replay 在四个真正有 future query 的 session 上得到 `T4-F0=+0.071980`、
   `T4-TS4=+0.066484`，4/4 session 均正；但是 `n=4` 的最小双侧精确 p 值为 `.125`。
   M24 的 matched clean-SPINT 三 seed 均值为 `+0.036604`，却只有 2/3 seed 和 4/6
   session mean 为正，预注册稳定性门失败。二者不能被合并成一个“稳定纯 T4 效应”。
3. **native M1 不支持 T4 有效性主张。** official T4 相对 original 的 held-out 差为
   `-0.00382498`；正确 post-support local replay 也只有 `T4-F0=+0.008187`，且只覆盖
   两个 distinct left-out session、checkpoint selection 仍受旧 support-internal minival
   污染。M1 不应作为 C1 正向后的 rescue endpoint。
4. **即使 C1 full-pass，也不能把 pseudo-MUA 自动升级成 native-MUA evidence。** C1 的
   pseudo-MUA 是同一 DANDI SUA spike events 按 electrode 的确定性求和；native FALCON
   MUA 是 threshold-crossing acquisition view。二者的 pooling law 有明确代数联系，但
   噪声、阈值、未排序 spikes、session/task、label geometry 和 calibration budget 并不相同。
5. **不论 C1 最终 pass 或 fail，当前 workspace 中唯一尚未读过 R²、又能合法预注册的 local
   native-MUA endpoint，是 M2 七个 held-in session 的 outer-LOSO post-M33 future-query
   endpoint。** 它可以做一次 prospective internal confirmation，并与已有 official M2
   system result 形成“external deployment + local matched attribution”的发表组合；它不能
   被称为新的独立 external test。
6. 旧 `m2_heldin_postsupport_endpoint_v1` receipt **不能直接用于 effectiveness 判决**：
   它明确封存了 `ordinary-T4-versus-clean-SPINT`，且规定 `no_effective_verdict=true`；其
   Stage-B 还允许用 outer-left-out selection window 选择 checkpoint，不符合最严格的
   held-out-session deployment claim。必须在访问任何 post-M33 R² 前写一个新、独立、
   fail-closed 的 v2 receipt，固定 source-only selection、两臂、三 seed 和一次性门。
7. **当前没有一个仍然 pristine 的 external native-MUA endpoint。** M2 与 M1 的 EvalAI
   hidden scores 均已暴露。将来提交 matched epoch-34 B0 可以关闭 decoder-version confound，
   但只能称为 matched legacy system ablation，不能称为独立 confirmation。若论文要求第二个
   independent native-MUA test，必须获得未曾用于本项目开发的新 session、subject、dataset
   或 organizer phase；在现有数据上切新窗口或再投一个候选不能恢复独立性。
8. **C1 与 native M2 是正交证据链，不存在科学触发关系。** native M2 可以现在完成
   0-GPU plumbing、tests 和 write-once prelaunch receipt 的准备；等当前 C1 释放 GPU 后，
   只由 native-M2 自身的 score-free prelaunch gate 与独立 root authorization 决定是否启动。
   等待 GPU 是 operational scheduling，不是“C1 pass 才许可 M2”。C1 fail 不会抹掉
   official M2 或 corrected M33 的正向证据。
9. **三 seed 的 42 terminal evidence cells 仍是任何 positive claim 的必要条件，但允许一个
   negative-only futility stage。** 先完整执行 seed42 的 `7 folds × 2 arms = 14 cells`；只有
   `mean delta<=-0.03` 或 `<=1/7` session 为正时才可 severe-negative stop。否则自动补完
   seeds43/44 的 28 cells。Stage A 不产生 positive claim、hyperparameter selection 或缩减
   full matrix 的权力。

## 2. native-MUA 证据账本

### 2.1 可以可信引用的证据

| 证据 | 结果 | 可以支持的最强结论 | 必须同时披露的边界 |
| --- | ---: | --- | --- |
| M2 EvalAI T4 `578221` vs original `578218` | held-out `+0.116765`; held-in `+0.019366`; online normalized latency `-0.069458` | submitted supervised T4 system 在 organizer-hidden native M2 held-out 上有效，cached identity 没有造成 online latency penalty | epoch-34 vs epoch-27 decoder；T4 使用 support direction labels；offline T4 fit/export 不在 online latency 内 |
| M2 corrected M33/q33 local replay | `T4-F0=+0.071980`; `T4-TS4=+0.066484`; 两者 4/4 session 正 | 在四个 M33-eligible visible session 上，chronological future-query signal 与 correct channel attachment 均正 | local development；2 个 zero-query session 合法排除；`n=4` 不能形成六-session significance claim |
| M2 M24/q24 source-exact replay | `T4-F0=+0.052885`; 5/6 session 正；common epoch 9 `+0.046628` | 正号不由 independent best-checkpoint selection 单独造成；M24 future-query 有候选效应 | 单一 screened cell；六 session 已被多次用于 development；不能称 formal |
| M1 official three-way | T4-original held-out `-0.003825`; D4-original `-0.004598` | 当前 T4/D4 系统不改善 official native M1 | official 是 aggregate，不提供 per-session paired inference；不能据此证明任意 T4-like 方法永远无效 |
| M1 post-support local replay | `T4-F0=+0.008187`; `T4-TS4=+0.010814` | 修正 support/query overlap 后，已有 frozen checkpoint 只显示很小 descriptive delta | 2 distinct left-out sessions；checkpoint 曾由 contaminated minival 选择；不是 effectiveness test |

以上第一行是 strongest external deployment evidence；第二行是目前最干净的 local
T4-content evidence。它们回答不同问题，不应把两行数值平均或当作同一个 replicate。

### 2.2 正向均值存在，但稳定性主张未成立

#### M24 T4 versus matched clean SPINT

冻结三 seed 结果为：

| Seed | T4-clean SPINT |
| ---: | ---: |
| 42 | `+0.043984` |
| 43 | `+0.071220` |
| 44 | `-0.005392` |

三 seed mean 为 `+0.036604`，但预注册门要求 3/3 seed 正、至少 5/6 seed-averaged
session 正；实际为 2/3 和 4/6，`primary_effective=false`。session-block bootstrap interval
`[-0.019261,+0.102021]` 跨零。这是“positive mean candidate”，不是 stable improvement。

该比较只匹配 support trial 数量：两臂均见 chronological M24 neural support；T4 额外见
24 个 trial-level target-direction labels，clean SPINT 不见 target labels。因此它测量
“有标签 calibration package 的端到端价值”，而不是 equal-information carrier ablation。

#### decoder 与 T4 联合 source training

`m2_joint_t4_upperbound_v2` 在一个 seed 上让 encoder 与 decoder 共同 source-train，得到：

- `T4-zero4=+0.018389`，4/6 session 正；
- `T4-TS4=+0.057094`，4/6 session 正；
- `zero4-TS4=+0.038705`，4/6 session 正。

第一项低于 `+0.03` practical margin；较大的 T4-TS4 差有约 68% 来自 TS4 低于 zero4，
不能用一个有害 shuffle control 替代绝对收益证据。该 branch 的 seeds 43/44 已按冻结规则
停止。它证明“decoder co-training 未被遗漏”这一假设已经做过一次有意义的 upper-bound
screen，但未给出新的 stable native-MUA gain。

### 2.3 已污染、撤回或只能诊断使用的证据

| 证据 | 状态 | 原因 |
| --- | --- | --- |
| historical M33 `~+0.06979` | 永久撤回 | scoring 从 trial 0 开始，与 33-trial support 重叠，并给两个没有 post-M33 query 的 session 人工端点 |
| historical M1 local held-out effect | 永久作废 | 3/3 held-out-calib session 只有 10 trials；`query_start_trial=10` 无 future query |
| historical M1 internal T4-TS4 | 撤回为 content evidence | minival 是 calibration 的 bit-exact prefix，T4 descriptor 在被评分 trial 上 in-sample fit |
| historical M2 internal minival | diagnostic only | 2-trial minival 位于 calibration prefix；不能代表 chronological post-support held-out calibration |
| M24 internal `-0.1502` vs held-out `+0.0529` | 未解释的符号矛盾 | domain、file composition、trial count 与 future horizon 没有正交交叉，不能归因于单一机制 |
| official M2 T4-original difference | system effect，不是 pure T4 ablation | decoder epoch 与 calibration label information 均不匹配 |

## 3. C1 到 native MUA：能桥接什么，不能桥接什么

### 3.1 成立的代数桥

对共享 cosine design 下、同一 electrode 内的 SUA units 集合 `G_k`，least-squares
response coefficients 满足：

```text
a_k = sum_{i in G_k} a_i
c_k = sum_{i in G_k} c_i
b_k = sum_{i in G_k} b_i
m_k = sqrt(a_k^2 + c_k^2)
```

因此 `[a,c,b]` 对 deterministic unit merge 是线性的，`m` 必须从 pooled `a,c` 重算。
C1 的 33-session score-free audit 已验证 online count conservation、calibration count、
pooled-rate T4、singleton electrode 和两视图行为/索引一致性。这个桥支持的结论是：

> T4 carrier 在“已排序 spike events 的已知 electrode pooling”下有明确 transformation law，
> 而不是依赖跨 session unit ID 对齐。

### 3.2 任何 C1 verdict 都不能替代 native-MUA endpoint

即使五个 C1 gate 全过，也只说明：

- 一套 source-trained weights 在 sorted SUA 与 deterministic pseudo-MUA 两视图分别满足
  预注册 non-inferiority；
- correctly attached T4 优于 shared TS4，且 cross-view R² gap 没有恶化超过 `0.03`；
- target-session calibration 仍为 forward/closed-form、无 backprop。

它不能推出：

1. C1 weights 在 FALCON native MUA 上可直接使用；C1 没有评价该输入域。
2. pseudo-MUA 等于 threshold-crossing MUA。后者含未被 SUA sorter 接收的 spikes、阈值
   与 waveform-dependent detection、背景噪声、碰撞和不同 refractory/quality process。
3. C1 shared training 是 strict accuracy improvement。两个 shared-vs-separate 主 gate 是
   lower bound `>=-0.03` 的 non-inferiority gate，允许一个小的负均值。
4. DANDI M50/M30 calibration 能外推到 FALCON M2 M33 或 M1 M10。label geometry 与
   coefficient reliability 不同。
5. native M2 official `+0.1168` 是 C1 paired training 导致。官方 T4 image 不是 C1 model。

反向也同样成立：C1 若失败，否定的是 one-weight paired-view training package，不会抹掉
official M2 T4 system result；native M2 若新确认失败，也不能反推 SUA T4 attachment effect
不存在。

### 3.3 C1 full-pass 时可联合呈现，但不是启动依赖

若 C1 最终 full-pass，最严谨的论文结构是三条彼此限定的证据链：

1. **SUA mechanism：** correctly attached first-harmonic calibration descriptor 在 sorted SUA
   development held-out 上有效。
2. **controlled granularity：** 同一 T4 carrier 服从已知 unit-merge law；C1 判断一个 weight
   set 是否能同时服务 SUA 与 deterministic pseudo-MUA。
3. **native deployment：** organizer-hidden M2 显示 submitted T4 system 的 held-out utility；
   新的 local matched endpoint只负责补 pure-method/clean-SPINT attribution。

不能把三条压缩成“C1 model 在 SUA 和 native MUA 上都显著提高 R²”。

若 C1 fail，第一条历史 SUA T4 mechanism evidence 与第三条 native M2 evidence 仍各自存在；
失败只终止 paired one-weight family。反过来，native M2 新 endpoint 也不以 C1 aggregate、
其数值大小或 strict-improvement diagnostic 作为 arm/seed/budget/epoch 的选择依据。两条程序
可以在同一天并行准备，只有 GPU occupancy 可以影响启动先后。

## 4. formal scope 不是“文件是否打开”的同义词

### 4.1 SUA formal scope

C1 v3r2 receipt 正确证明本批 6 个 formal SUA path 未解析、文件未打开。但项目中已有：

```text
results/p3_formal_test_816cdd8b..._receipt.json
status = "started"
```

该 formal scope ID 按既有预注册纪律已被占用，且没有完成结果。因而：

- “C1 没打开 formal 文件”是正确的 data-access statement；
- “C1 full-pass 后仍可把同一个 formal scope 当作新的一次确认”是错误的 statistical
  authorization statement。

不得因 receipt 悬空而重新运行、重命名 scope 或把文件未读当成恢复独立性的理由。

### 4.2 FALCON EvalAI scope

M2 original/T4 和 M1 original/T4/D4 的 hidden aggregate 均已返回。这里的“scope consumed”
是科学候选选择意义，不是断言 API quota 已耗尽：

- 不得根据已见 hidden score 调模型后再称同一 phase 的结果为 independent confirmation；
- 不得重复提交 T4 `578221`；
- matched epoch-34 B0 若完成 packaging、得到新的显式用户授权并只提交一次，可以作为
  decoder-matched **legacy system ablation**；
- B0 结果仍不能把 epoch-34 legacy held-out-selected teacher 变成 clean teacher，也不能
  证明 C1 model 的 native generalization。

本审计没有查询 live phase、quota、team permission，也不授权提交。

## 5. 独立授权的最小 native-MUA confirmatory endpoint

### 5.1 推荐名称与主张层级

建议新协议名：

```text
M2_NATIVE_T4_SPINT_POST33_CONFIRM_V1
```

建议主张层级：

> prospective, native-MUA, outer-session-held-out, chronological future-query internal
> confirmation of supervised backprop-free calibration against a clean local SPINT baseline.

它与已有 official M2 result 联合后具备发表价值，但单独不能叫 external/organizer-hidden
confirmation。

该协议的科学授权只来自自身的 score-free prelaunch gate：endpoint 未评分证明、plumbing
tests、outer-fold isolation、source/data hash closure、label disclosure、cost contract 和独立
root authorization。它不读取 C1 verdict，也不要求 C1 pass。当前即可完成全部 0-GPU
plumbing/receipt 准备；GPU launch 可以为避免资源竞争等 C1 释放设备，但这只是调度顺序。

### 5.2 为什么这是唯一合理的 local 新端点

`m2_heldin_postsupport_endpoint_v1/audit.json` 只做 structural audit，没有在该端点运行
F0/T4/SPINT R²。七个 M2 held-in-calib session 各有 204--339 trials；first 33 之后均有
大量 future query。以每个 session 轮流作 outer-left-out，可以得到 7 个 session cluster，
而不是 M33 local held-out-calib 的 4 个 eligible cluster。

但是现有 v1 protocol 只能作为输入清单：

- 它封存了 ordinary-T4-versus-clean-SPINT branch；
- 它声明 `no_effective_verdict=true`；
- 它的 Stage-B checkpoint selection 使用 outer-left-out post-support first half，等于让
  target-session query behavior 参与模型选择，不符合 deployment-pure held-out claim。

新 v2 receipt 必须明确 supersede **协议用途**，不能覆盖、改写或删掉 v1。

### 5.3 数据与 outer-LOSO split

固定七个 outer fold：

| Fold | 唯一 outer-left-out native M2 session |
| ---: | --- |
| 0 | `ses-2020-10-19-Run1` |
| 1 | `ses-2020-10-19-Run2` |
| 2 | `ses-2020-10-20-Run1` |
| 3 | `ses-2020-10-20-Run2` |
| 4 | `ses-2020-10-27-Run1` |
| 5 | `ses-2020-10-27-Run2` |
| 6 | `ses-2020-10-28-Run1` |

对 fold `j`：

- source training、normalizer、checkpoint rule 只能使用其余 6 个 session；
- outer-left-out session 不得进入 dataloader、loss、optimizer、early stopping、epoch choice、
  normalizer、teacher selection 或 hyperparameter selection；
- target support 固定为 chronological trials `[0,33)`；
- target query 固定为 trials `[33,end)`；
- 每个 50-bin scored history 必须完全位于 post-support 区间，不能仅要求 prediction time
  大于 boundary；
- target query behavior 只在所有 checkpoint/identity/receipt 锁定后用于一次评分。

不再把 post-support first half 用作 selection。checkpoint selection 只能使用六个
outer-train source session 的 source-held-in validation：clean SPINT 固定最多 35 epochs，
T4 encoder 固定 12 epochs；两者都只按 source-only `val_heldin/r2_mean` 选一个
checkpoint，tie-break 为较早 epoch。T4 的 decoder 是该 fold/seed 已锁定的 SPINT decoder，
不参与这 12 epochs 的更新。exact metric key、missing-metric failure、epoch index convention
和 tie-break 必须在任何 post-M33 score 出现前写入 receipt；不得根据 outer score 改成
final epoch、另一个 best epoch 或 epoch-window estimator。

当前 datamodule guard 会明确拒绝 M2 `heldin_query_start_trial=33`，因为现有允许分支只为
M1 contamination-correction replay 开放。新协议在 GPU launch 前必须先做一个独立、版本化的
plumbing repair：只允许
`task=m2, validation_protocol=loso, calibration_n_trials=33,
heldin_query_start_trial=33, random_calibration=false, include_heldout_in_fit=false`，并让 query
来源显式切到 outer-left-out `train_calib_heldin_session` 而不是 2-trial minival。focused tests
必须证明其他组合仍 fail-closed、七 fold query session 各自唯一、101,171 个已审计 eligible
windows 的 full-history boundary 语义可复现；preflight 只检查形状/索引，不能读取 R²。

### 5.4 最小两臂

#### Arm S：clean local SPINT

- 每 fold/seed 只用 6 个 outer-train session 训练 clean SPINT；
- 该次训练选出的唯一 SPINT checkpoint 同时承担两个角色：直接产生 Arm S 结果，并提供
  Arm T 的 decoder substrate；不得为 Arm T 再训练第二套“matched decoder”；
- target session calibration 只读相同 first-33 neural trials；
- identity 为 original-SPINT activity identity，不是 zero identity；
- 不读取 `tgt_loc` 或 T4 tensor；
- target calibration 无 optimizer/backward/refit；只有固定 forward identity computation；
- decoder state、preprocessing、behavior scale、50-bin online window 和 runtime batching 均写入
  receipt。

#### Arm T：ordinary T4/B3S

- 从同一 fold/seed、也就是 Arm S 实际使用的 clean SPINT checkpoint 初始化；
- T4 student 的 query decoder 必须与 Arm S decoder 31/31 tensor bit-exact，并在 source
  T4 training 中冻结；否则该 cell fail-closed；
- target support 同样使用 first-33 neural trials；
- 额外从同一 33 trials 读取 trial-level target direction，centre/rest trial 明确 excluded；
- fit `[a,c,m,b]`，要求 design rank 3、label/rate trial alignment 完整；
- target session 上只做 closed-form fit、forward identity 和 cached decoder inference，无
  parameter update。

这两臂最小化的是**发表主问题**：相同 native session、相同 chronological neural exposure、
相同 decoder 下，增加 supervised T4 calibration 是否优于 clean local SPINT。

### 5.5 “matched budget”的准确措辞

SPINT 与 T4 可以匹配：

- 33 个 chronological calibration trials；
- neural exposure、preprocessing、query start、online history、decoder tensor、behavior scale；
- source split、seed、checkpoint rule、session aggregation。

它们**不能同时匹配 label information**，因为 original SPINT 的定义就是不消费 direction
labels。正确 disclosure 是：

```text
SPINT: 33 neural support trials, 0 target-direction labels
T4:    the same 33 neural support trials, plus one direction label for each eligible
       directional/rewarded support trial (M2 first-33 normally contains 16 directional trials)
```

所以 `T4-SPINT` 是 supervised-calibration package 的 utility estimand，不是 equal-information
carrier estimand。若论文还要求在这个新端点重新确认 correct attachment，必须**在第一次
score 前**把第三个 separately trained TS4 arm 加入 receipt；不能在 T4-SPINT 结果出来后
追加。最小路线不加 TS4，因为已有 corrected M33 `T4-TS4` development control 和 C1 两视图
attachment gate；相应地，新端点不得单独声称 direction/attachment mechanism 被重新确认。

### 5.6 seeds、计算矩阵、negative-only futility 与最终主门

使用 seeds `{42,43,44}`，形成：

```text
2 arms × 7 outer sessions × 3 seeds = 42 terminal arm-cells
21 paired T4-SPINT deltas
21 clean-SPINT trainings + 21 T4 trainings
```

这里的 42 是 terminal evidence-cell 数，不是 42 套独立 decoder training。每个
`seed × fold` 只训练一套 clean SPINT；它既是 Arm S，又是随后 T4 training 的冻结 decoder
substrate。于是完整矩阵实际是 21 次 SPINT training 加 21 次 T4 training。

不能只跑 seed 42：既有 clean-SPINT replication 的 seed 44 已经反号，训练随机性正是当前
未闭合的问题。三 seed 是“最小可发表”而不是可删的冗余。

对每个 seed `s` 和 outer session `j`，先得到 common query 上的：

```text
D[s,j] = R2_T4[s,j] - R2_SPINT[s,j]
```

#### Stage A：seed 42 的一次 negative-only futility screen

为了允许非常差的候选尽早停止，同时不让一个 seed 产生选择性正向主张，先完整运行：

```text
seed 42 × 7 folds × 2 arms = 14 terminal arm-cells
7 paired deltas D[42,j]
7 SPINT trainings + 7 T4 trainings
```

必须等 14/14 completion/hash closure 后一次性读取七个 delta；禁止按 fold 逐个偷看或提前
改变队列。定义：

```text
mean42 = mean_j D[42,j]
pos42  = count_j(D[42,j] > 0)
```

冻结 futility stop：

```text
STOP_NEGATIVE_FUTILITY = (mean42 <= -0.03 R²) OR (pos42 <= 1/7)
```

两个分支分别覆盖不同的“极差”：

- `mean42 <= -0.03` 表示平均 harm 已达到最终正向 SESOI 的相反幅度；继续寻找 `+0.03`
  会成为明显的 seed rescue。
- `pos42 <= 1/7` 表示至少 6/7 outer sessions 不正，恰好是最终 `>=6/7` 正 session 门的
  镜像反面。若每个 sign 在零假设下等概率，一侧 `P(X<=1; n=7,p=.5)=.0625`；这不是
  significance test，只说明它是接近全 session 一致的方向性失败，而不是两个 session 的
  噪声。

若任一条件触发，程序写为 `seed42_severe_negative_futility_stop` 并终止；必须完整报告这
14 个 cell，但不得外推成“三 seed 已证明 ineffective”。若两项都未触发，唯一许可动作是
按原 receipt 无条件完成 seeds 43/44 的另外 28 个 arm-cells。**未触发 futility 不叫 pass、
promising、trend 或 GO evidence**；即使 seed42 7/7 大幅正，也不能作 positive claim、不能
改超参、不能减少后续 seed、不能增加 arm。

这个 futility boundary 对 efficacy 是 non-binding：它只有 severe-negative 资源停止功能，
不提供任何 selective-positive authorization。最终 positive claim 仍必须包含 Stage A 的全部
数据与完整 42 cells，并通过原先冻结的三 seed gate。由于存在条件性停止，two-SE/bootstrap
bounds 是预注册的工程 decision bounds，不能在论文中冒充未经 sequential-design 修正的
nominal 95% confirmatory CI 或 p-value；无论 stop/continue 都必须入 evidence ledger，避免
只发表 survival branch。

#### Stage B 与最终三 seed effectiveness gate

Stage A 未触发 futility 时，完成 seeds 43/44：

```text
2 seeds × 7 folds × 2 arms = 28 additional terminal arm-cells
full evidence = 42 arm-cells / 21 paired deltas
```

最终 effectiveness gate 保持不变：

1. `mean_{s,j} D[s,j] >= +0.03 R²`；
2. 3/3 seed means `> 0`；
3. 至少 6/7 seed-averaged session means `> 0`；
4. paired two-SE lower bound across the three seed means `> 0`；
5. two-way seed/session hierarchical-bootstrap lower bound `> 0`；
6. clean SPINT 与 T4 的 absolute equal-session mean 均有限，且 T4 absolute mean `>0`。

六项必须全部满足，只执行一次 full aggregate。exact Wilcoxon/sign statistics可同时报告为
descriptive inference，但不能替代上述 gate，也不能用于结果后修改门槛。

若担心 `+0.03` 在 7 session 下不可识别，只能在 score 前用 source-only/已封存历史残差做
MDE sensitivity；不能从这 7 个 target delta 先估 noise 再改 SESOI。`+0.03` 是 deployment
SESOI，不应伪装成 power-derived significance threshold。

### 5.7 required receipts

#### A. prelaunch protocol receipt（任何 score 前）

必须冻结：

1. native-M2 自身的 score-free authorization chain：structural endpoint audit、未产生该端点
   R² 的 inventory、plumbing tests、outer-fold exclusion、source/data closure 和独立 root
   authorization。C1 finalizer/aggregate 不得出现在 native launch condition 中；它只在最终
   论文 evidence ledger 中证明并行 C1 程序是否完整。
2. 新 protocol ID、O_EXCL write-once path，以及对
   `m2_heldin_postsupport_endpoint_v1` branch seal 的显式、非覆盖式 supersession。
3. 七个 NWB 的 canonical path、bytes、SHA-256、trial count、first-33 direction count、
   cosine-design rank/condition 和 post-M33 query count。
4. exact outer-fold manifest；每 cell outer-left-out exclusion 的 expected data trace。
5. seeds 42/43/44、两臂、固定 epoch budgets/source-only checkpoint rule、optimizer
   coverage、source-only normalizer rule；这里应写成上节的 35-epoch SPINT / 12-epoch T4、
   `val_heldin/r2_mean` 和 earlier-epoch tie-break，而不是留下运行时可选项。
6. SPINT/T4 source code、configs、runner、scorer、aggregator、finalizer 的 complete transitive
   source-map hashes。
7. label disclosure、query-boundary/full-history rule、no target-query selection rule。
8. decoder bit-equality contract、teacher provenance和“不允许 legacy held-out-selected teacher
   替代 clean outer-fold teacher”的 guard。
9. Stage A exactly 14 terminal cells、固定 OR futility rule、未触发时 Stage B exactly 28
   additional cells，以及 positive claim 必须 exactly 42 terminal cells 的声明；同时冻结
   no-retry/no-rescue/no-extra-arm/no-EvalAI/no-formal-SUA。
10. 参数、source-training MAC、target calibration MAC/state、online decoder MAC/state 和 latency
    measurement protocol。

#### B. per-cell start/completion closure

每个 cell 至少证明：

- receipt/source-map/data-manifest hash 未漂移；
- outer-left-out session 在 train/validation/normalizer/checkpoint selection trace 中为零；
- exact support labels/rates 与 trial boundary 一一对齐；
- query 的 minimum history start 完全在 trial-33 boundary 后；
- target calibration optimizer steps `0`、backward calls `0`、updated parameter tensors `0`；
- decoder equality、checkpoint SHA、normalizer SHA、identity shape/state hash；
- R² artifact 与 metadata/cost receipt hash-bound，失败 cell 不得静默重跑。

#### C. 两级 score-blind finalizer

Stage-A futility finalizer 在读 R² 前必须要求：

- exactly 14 starts + 14 completes；0 failed、0 extra、0 overwritten；
- exactly 7 matched seed42/fold pairs；
- all source/data/runtime/closure hashes valid；
- only then read all seven deltas together, write `mean42/pos42`、两个 futility predicates 和
  immutable `stop` 或 `continue_without_positive_claim` decision；
- 不得输出 efficacy pass、candidate ranking 或任一 hyperparameter recommendation。

若 decision 为 continue，full finalizer 在读全矩阵 R² 前必须要求：

- exactly 42 starts + 42 completes；0 failed、0 extra、0 overwritten；
- 21 matched seed/session pairs；
- Stage-A 的七个 pair 原样包含，hash 与 futility receipt 一致，不能重跑或替换；
- all source/data/runtime/closure hashes valid；
- no EvalAI call、no formal SUA path、no visible local held-out-calib six-session endpoint混入；
- only then invoke aggregator once and O_EXCL-write aggregate、gate decision与 publication
  readiness receipt。

### 5.8 futility/pass/fail 后允许写什么

若 Stage A 触发 futility：

> At the preregistered seed-42 seven-session stage, the candidate crossed a severe-negative
> futility boundary and the remaining seeds were not launched.

必须附完整七 session delta，并写“这是预注册的资源停止，不是三 seed effectiveness
inference”。不得把未运行的 seeds 当作零、不得重新选择 seed43 作为新 Stage A，也不得以
official M2 positive score覆盖这次 negative stop。

若 full gate pass：

> On a prospectively locked seven-session outer-LOSO native M2 endpoint, T4 used the same
> chronological 33-trial neural calibration exposure as a clean local SPINT baseline, plus
> disclosed trial-direction labels, and improved fully post-calibration query R² without
> target-session backpropagation. Together with the existing organizer-hidden system result,
> this supports native-M2 deployment utility and a clean local attribution result.

仍不得写：

- C1 shared weights 直接 generalize 到 native MUA；
- T4 label-free；
- T4-SPINT 是 equal-label-information ablation；
- native M1 同样有效；
- 七折内部确认是第二个 independent external dataset。

若完整 42-cell final gate 任一项 fail：

- 报告完整 delta、signs、bounds 与 cost；
- 结论为 pure/native matched attribution 未稳定通过；
- official `578221` 仍保留为 system-level result，但不能再升级成 pure T4 effect；
- 不加 M1、TS4、F0、seed45、M24、epoch scan、decoder-unfreeze、C2、INT8 或 EvalAI rescue。

## 6. matched epoch-34 B0 EvalAI 的准确位置

现有 audit 已证明 matched B0 technically constructible：submitted T4 decoder 与 epoch-34
teacher 的 31 个 state tensors bit-exact，B0 可以从相同 first-33 neural support 计算 original
SPINT identity。但是目前没有 B0 payload、独立 image、container parity receipt 或 guarded
submit helper。

若以后明确授权并完成一次 B0 submission，它能回答：

> 在同一个 legacy epoch-34 decoder 下，submitted T4 system 是否优于 original-SPINT
> activity identity？

它不能回答：

- 一个未看过 hidden result 的候选是否独立复现；
- clean teacher 下是否稳定；
- C1 paired training 是否适用于 native MUA。

因此 B0 submission 是可发表的 formal ablation，但不是本文件所说的 fresh local
confirmatory endpoint，也不是当前任务的必要动作。没有新的显式用户授权时不得 push、
register 或 submit。

## 7. 硬停止条件

以下任一成立，native-MUA 新 endpoint 都不启动：

1. native-M2 自身的 score-free prelaunch verifier、plumbing tests、data/source closure 或独立
   root authorization 任一不通过。
2. post-M33 endpoint 的任一 arm R² 已在新 protocol freeze 前被读取、聚合或用于选择。
3. 无法证明 outer-left-out session 对 source training、normalizer、checkpoint rule 完全隔离。
4. Arm S 不是 clean local SPINT activity identity，或被替换成 zero identity/F0 而仍叫 SPINT。
5. 两臂 decoder 不 bit-exact，或 T4 source training 更新了 decoder。
6. target calibration 出现任何 optimizer/backward/parameter update。
7. support/query full 50-bin history disjointness、label alignment 或 rank-3 fit任一失败。
8. 三 seed/七 fold/两臂 full matrix 与 14-cell Stage-A futility rule 不能在第一次 score 前
   一次完整冻结；不得先跑一个 fold，再按结果决定是否扩展。
9. 想把同一 EvalAI phase 的新候选称为 independent confirmation，或想复用已占用的 SUA
   formal scope。

C1 finalizer incomplete 或 C1 five-gate fail **不是** native-M2 的停止条件。它只限制 C1
paired-family claim。native-M2 的 official `578221`、corrected M33 evidence 与新 local
attribution program 均由自己的 provenance/gates 维护。

native program 启动后，另有两个预注册停止出口：Stage A 跨 severe-negative futility boundary
时在 14 cells 终止；Stage A 继续后，完整 42-cell final gate失败时终止。两者都不授权新
arm、seed replacement 或 endpoint rescue。

若项目要求的是**第二个 independent external native-MUA confirmation**，而不是上述 prospective
internal confirmation，那么当前 workspace 的正确动作是停止：先获得新的、未参与开发的
native-MUA sessions/subject/dataset/organizer phase，再在任何 score 前冻结相同 M33、SPINT、
label disclosure 和 bp-free gate。现有数据不能通过重新切窗口恢复外部独立性。

## 8. 证据锚点

| Artifact | SHA-256 | 用途 |
| --- | --- | --- |
| `docs/NATIVE_MUA_T4_M1_M2_PROGRAM.md` | `c67d5ea27df234b101d7cd246bf52a5c5636e2161cc1250016cce65a3e8496a0` | native M1/M2 implementation、污染修订与 claim history |
| `docs/M2_CPU_CORRECTION_AND_CARRIER_RESULTS_20260801.md` | `d812cf1031d43028526f4adb1c4a4e52f7f580d81569d0708223b1e00ea33a7f` | corrected M33、M24 sign audit、uncertainty |
| `results/m2_m33_disjoint_replay_correction_v1/aggregate_heldout.json` | `8d56e60825cc3c017b17764146118a0e6f04a33f273f0c6c9191e14de6da70fb` | four-session M33/q33 result |
| `results/m2_m24_disjoint_heldout_v1/aggregate_heldout.json` | `1db3477417ffefa358a432404d1223cb4ea25a31667b922d9fd2f254c84ecca7` | six-session M24/q24 visible development result |
| `results/m2_t4_clean_spint_replication_v1/aggregate_3seed_final.json` | `b3cace02693bd556f69282a2352c661c34773e828fd915f65a81aa784bff0e49` | matched clean-SPINT failed stability gate |
| `results/m2_joint_t4_upperbound_v2/aggregate_local_heldout_seed42.json` | `ef1e46c95ab5beecc2dfc792876e1de597f9ae5e9a2829ba393ef8faa46f0487` | decoder-joint upper-bound screen |
| `evalai_t4_m2/SUBMISSION_RECEIPT.md` | `af0ccbdb8efb76e0c1a1b1f1a7a667acb4ffa7c6f7adbede702e5f6d456a5736` | official M2 T4/original metrics与confound |
| `docs/M2_MATCHED_EPOCH34_B0_EVALAI_ELIGIBILITY_20260804.md` | `ee26160160f76dd13e3abbd294154b95e421a5242ba51294063c920d5dbaf31b` | matched B0 constructibility和submission boundary |
| `evalai_m1_threeway/SUBMISSION_RECEIPT.md` | `14eae99b10a66cb5a7de62849515100c44d69bc36828ff88356e2a657c4f249f` | official M1 negative comparison |
| `results/m1_heldin_disjoint_replay_v1/aggregate_heldin.json` | `5ec3da0b93fc6cdd2ab3883c6e6ce32f6de17e9ee9c9ef74f4f08624094d5dfa` | corrected M1 post-support descriptive endpoint |
| `results/m2_heldin_postsupport_endpoint_v1/audit.json` | `d6940d156d05cd1cbd43bac95220c1d1db370c50d841de8a51d49a53c9ac70c5` | unused seven-session endpoint structural evidence |
| `results/m2_heldin_postsupport_endpoint_v1/protocol_receipt.json` | `7673d360099775e37b199621956d099a9a2a7bafcbe7b074c3146978300f1c49` | old branch seal；不能直接作新判决 |
| `results/p3_formal_test_816cdd8bf9f26abd1a3e6251e5fbf8537eb6c6cb4de1e8f3312980ddbf478379_receipt.json` | `013d21a738dc604071f276f3b04362d178e8fa8d9cb1801b28c989a91610e07a` | consumed SUA formal scope evidence |
| `docs/HELDOUT_BP_FREE_CALIBRATION_METHOD_CLOSURE_20260804.md` | `424becb0b0f2eac4f6e270974016ab3dd1f53ac753bc602b7c4a3ea300676afe` | cross-domain method/evidence boundary |
| `docs/C1_POSTRUN_MECHANISM_CLAIM_AUDIT_20260804.md` | `ce959af02bf59e4ddbbe3d14b0fc59d2b1266f47b0e20a6abb7763e480f5c4df` | C1 gate semantics与claim matrix |
| `results/t4_paired_view_c1_fresh_prelaunch_v3r2_remote_gate_allowlist/receipt.json` | `8b17c19515fa0e6cc122233fd20cf7e87a616287fa247e5eacb136d6a56c9d85` | active score-blind C1 frozen contract |

## 9. 最小路线摘要

```text
C1 evidence chain (independent)
    -> complete 12-cell finalizer
       |-- fail/incomplete -> stop or qualify paired-view claim only
       '-- pass -> report controlled SUA/pseudo-MUA one-weight evidence only

Native-M2 evidence chain (independent; CPU preparation may start now)
    -> versioned M2 held-in q33 plumbing + score-free tests
    -> freeze M2_NATIVE_T4_SPINT_POST33_CONFIRM_V1 before any post-M33 R2
    -> native-only prelaunch verifier + independent root authorization
    -> when GPU capacity is available: seed42, 7 folds × {one SPINT substrate, one T4}
       -> Stage-A finalizer reads all 14 cells once
          |-- mean42 <= -0.03 OR pos42 <= 1/7
          |      '-- severe-negative futility stop; no three-seed claim
          '-- otherwise continue_without_positive_claim
                 -> seeds43/44, 28 additional arm-cells
                 -> full 42-cell finalizer + unchanged six-part effectiveness gate
                       |-- fail -> retain official system result, deny stable local attribution
                       '-- pass -> combine official M2 deployment with prospective local attribution

Independent external replication still requires genuinely new native-MUA data/phase.
```
