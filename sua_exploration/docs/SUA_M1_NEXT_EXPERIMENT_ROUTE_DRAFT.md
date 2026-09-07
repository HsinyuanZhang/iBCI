# DRAFT — SUA / FALCON M1 下一轮实验路线（ROOT REVIEW REQUIRED）

> **Root review notice（2026-08-02）：** 本文件保留为 Terra 的原始路线草案，
> 不可直接作为执行章程。Root 已完成审核并在
> [`SUA_M1_NEXT_EXPERIMENT_ROUTE_ROOT_REVIEW.md`](SUA_M1_NEXT_EXPERIMENT_ROUTE_ROOT_REVIEW.md)
> 中修正了 held-in endpoint readiness、PCA split、D4 pass 后的 official-test 顺序、
> support-set Gate-A 标签披露，以及 dynamic residual 的目标域和硬件状态。与本草案
> 冲突时，以 root review 为准。

**状态：仅供 root 审核的证据路线草案；不授权训练、datamodule 改动、官方评测或
EvalAI 提交。**

**撰写日期：2026-08-02（Asia/Hong_Kong）**

本草案的目的不是把“functional embedding / population manifold / FiLM”并行变成
三条 GPU 队列，而是在当前 D4 首格结果未知时，规定一个可证伪、低重复、避免反复
查看同一 held-out 的顺序。所有数值均为已存在 artifact 的 development evidence；
不读取本轮 D4 的任何未封存结果，也不改变正在运行的 `m1_d4_pilot_v1`。

## 0. 决策原则与域边界

### 0.1 三个数据域绝不互换

| 域 | 信号/数据 | 当前可支持的结论 | 本路线中的作用 | 不能外推成 |
|---|---|---|---|---|
| **sorted SUA** | DANDI 000688 / `sub-C/CO` 的 sorted units | T4/TS4 在 27/6 validation development 上很强；T4 对 F0 为 `+0.252761`，对 TS4 为 `+0.252233`，均 6/6 session、3/3 seed 正向 | 提供“功能性、正确 channel attachment 可以有巨大收益”的机制先验及融合失败的反例 | FALCON M1 的 M=10、native-MUA 或 official few-shot 成功 |
| **derived pseudo-MUA** | 从同一 DANDI SUA 按 electrode 聚合得到的通道 | T4 对 F0 为 `+0.317739`、对 TS4 为 `+0.365674`；它是受控 pooling bridge | 检验某个 SUA 机制是否对 pooling 鲁棒；只能作辅助边界 | 真实 threshold-crossing/native MUA 外部复验 |
| **FALCON M1 native-MUA** | DANDI 000941 的原生 NWB Units/electrode-channel count，M=10 | clean-selection report 上 T4 对 F0 `+0.008410`、对 TS4 `+0.007696`（3 cells、但仅 2 个 left-out session）；这是小而一致的 development signal | **本路线唯一的 M1 efficacy 域**；D4 是其最小下一步 | SUA/pseudo-MUA 中 T4 的大效应必然重现，或 old support-overlap result 是正式 held-out 证据 |

因此，SUA 的强 T4 结果只说明“functional carrier 值得在 M1 进行受控检验”；它
不降低 M1 对其自己的 F0、T4、row-shuffle 对照和 future-query / official endpoint
的要求。pseudo-MUA 绝不替代 FALCON M1 native-MUA。

### 0.2 当前 M1 事实与不可撤销边界

1. 历史 M1 internal-LOSO minival 是 corresponding calibration 的 bit-exact two-trial
   prefix；历史 `T4−TS4` content claim 已撤回，旧 local held-out-calib 三个 session
   又各只有十个 trial，`query_start_trial=10` 无 query。因此这些数值不能做本路线的
   efficacy 终点。
2. 可构造的本地 **development** endpoint 是 held-in-calib session 的 post-support
   trials：四个 session 各约 366–404 future trials。该端点尚未由 production
   datamodule 支持，未来若启用，必须先有独立的 fail-closed boundary implementation
   与 root 授权；它不等于 official held-out。
3. `m1_clean_selection_v1` 使用 support `[0,10)`、checkpoint selection `[10,210)`、
   sealed report `[210,end)`。报告显示 M1 target directions 为八个方向、跨一个
   `157.5°` 半平面，故 T4 的 `[a,c]` 在未采样半圆上确有外推风险；这是一条合理
   机制假设，尚不是已隔离的因果结论。
4. `m1_d4_pilot_v1` 的 Phase-B receipt 记录的唯一授权臂为 D4 / DS4、fold 1 / seed
   42；D4 是 `[obj_id=1,2,3,4]` 的 exposure-corrected channel rate profile，DS4
   在 train-only normalization 后完整 row permutation。receipt 中 report access 为
   training/checkpoint selection 前 forbidden。本文不假定该训练的结果。

## 1. 对四类启发的审核

| 启发 | 现有证据 | 审核结论 | 可接受的下一步 / 明确禁止的偷换 |
|---|---|---|---|
| **Data-driven functional embedding** | SUA T4/T8 的正确 attachment 强；M1 clean T4 只有约 `+0.008`；D4 Gate-G audit 的 all-future neural-rate proxy 有 `D4−T4=+0.383313`、4/4 held-in 正，但这是 future-label oracle diagnostic | **plausible but unproven for M1 decoding**。D4 是最简单的非-cosine categorical functional carrier，已经具有恰当 DS4 对照 | 先按冻结 protocol 判断 D4/DS4。若需要学习式方案，采用跨 session meta-trained、new session 只 forward 的 support-set encoder；不直接启动 generic autoencoder 或无定义 contrastive learning |
| **Population subspace / PCA** | 尚无 M1 M=10 split-half stability、behavioral incremental value或跨 session alignment证据；SUA spatial prior Stage 0 显示真实邻域向零收缩更差（M10/15/20 neighbor/zero MSE `1.145/1.119/1.104`，0/27 改善） | **plausible but unproven；裸 PCA-loading 作为 unit identity 目前不成立** | 只可先做 CPU split-half projection audit。PCA score/loading 的符号、近简并旋转、variable-N 与 unit-order/permutation 问题必须在 Gate A 先解决；不把 loadings 直接 concat 进 decoder |
| **Dynamic fusion / FiLM** | SUA confidence-FiLM 相对 T4 continuation `+0.003399`，相对 shuffled C `+0.002782`，相对 matched NoFiLM `+0.000698`，均未过机制门；decoupled K/V v1 重构 activity path 后严重失配 | **“一般 FiLM 会修复 concat”被现有证据反驳；baseline-preserving attention residual 仍是窄的未检验假设** | 只有在 D4 content 已通过后，允许 selected-D4 checkpoint 上的 zero-init key/logit residual；必须保留 activity read-in、value 与 decoder，且与 DS4-residual、continuation、additive NoFiLM 三配对 |
| **停止 physical waveform/SNR** | SUA F1/F2 raw concat 均为 indeterminate 而非逻辑上的“无效”；但真实值没有赢过 F0 或 row-shuffle。T4GATE `−0.010817` 已 ineffective；REL `−0.001440`、3/3 seed 负；spatial prior Stage 0 NO-GO | **停止把 waveform/SNR、absolute electrode、REL 或 spatial prior 当作 M1 identity 主线** | 保留 waveform/SNR 作为 QC、sorting/坏通道和 calibration-uncertainty 诊断。除非它们先通过独立 CPU reliability-calibration gate，且有新的交互机制，不再给它们单独 GPU 融合实验 |

### 1.1 为什么不是直接做 autoencoder / contrastive learning

M1 deployment 支持集只有十个 trial。逐新 session 训练 autoencoder 既增加 calibration
backprop，也最可能压缩总体 firing rate、trial exposure 或 session nuisance；这些都不
保证形成行为/功能相关的 channel carrier。contrastive learning 同样需要定义
positive/negative：若利用 condition，它并非无监督；若不利用 condition，则没有机制
保证它服务运动解码。

较合适的形式是 **support-set functional encoder**：在 source training sessions 上
meta-train，输入是每个 channel 的 `{support spike/rate summary, exposure, known support
condition}`，输出固定 q 维 static carrier；新 session 只做一次前向/streaming sufficient
statistic finalization，无 optimizer step、无 query label。它把 D4 视为 q=4 的闭式
baseline，而不是绕过 D4。

## 2. 当前 D4 的条件决策树

### A. D4 首格通过（仅按冻结 clean-best report contrast）

```
D4-F0 >= +0.015  AND  D4-T4 > 0  AND  D4-DS4 >= +0.010
        |
        +-- yes --> 只扩到预声明 fold1/seed43、fold2/seed42；不改 carrier、M、
        |           model、optimizer、selection/report boundary 或 controls。
        |             |
        |             +-- 三 cell mean(D4-F0)>+0.03 且 mean(D4-DS4)>0
        |                   --> D4 content candidate 通过；可由 root 另行授权候选 #3 的
        |                       predeclared local development screen，或直接冻结 D4。
        |                   --> 只在最终唯一 candidate freeze 后，才进行一次官方 few-shot
        |                       held-out；official result 不用于再开启候选 #3。
        |
        +-- no --> 走 B。
```

“通过”也不是 M1 effectiveness 或 object semantics claim：Gate S 已失败，因此名称始终
是 *categorical calibration profile (D4)*；三 cells 仍只有两个 left-out sessions。

### B. D4 首格失败（任一门失败）

```
立即停止 D4 decoder extension
  ├─ 不扩 seed/fold；不做 D4+T4、D4 FiLM、PCA concat、support-budget scan 或 EvalAI
  ├─ 不把 Gate-G neural-rate proxy 当作 decoder positive result
  └─ 先完成一项只读/CPU development-endpoint readiness gate：
       fixed M=10 support, explicit [support|selection|report] query boundary,
       full-history disjointness, no query labels, source/checkpoint isolation.
       |
       +-- readiness 未过 --> 暂停所有 M1 model search；修正评测基础设施后由 root 重审
       +-- readiness 过 --> 仅候选 #2 的 CPU Gate A 可开始；Gate A 不足则不启动 GPU
```

失败不证明所有 functional carrier 都无效，但它否定“以 D4 的 profile 内容和当前
B3S concat 方式能产生实用 M1 decoder 增益”。此时先开 autoencoder、PCA 或 hypernetwork
会同时改变 carrier、容量与融合，无法解释失败。

## 3. 排名候选主线（最多三条）

共通规则：最终 task endpoint 是 **unseen M1 session、chronological M=10 calibration
后的 official few-shot held-out**。下面 Gate A 的 rate/stability proxy 与 Gate B 的
internal development screen 都不能替代该终点。每个新 GPU proposal 都须先由 root 生成
result-before-run receipt，固定 source manifest / train-only normalizer SHA / source
checkpoint rule / label scope / query boundary / seed block，并确认不会读取已封存的 official
test 结果。

### 候选 #1（当前唯一运行中的主线）：D4 categorical calibration profile

| 项目 | Gate A（已完成、只读/CPU） | Gate B（当前冻结的最小 GPU） |
|---|---|---|
| 假设 | 不要求 cosine 在半平面外推；channel 的四类别 exposure-corrected rates 是比 T4 更有用的 functional carrier | D4 的 channel attachment 能转化成真正 decoder gain，而不只是 neural-rate proxy |
| 固定预算/边界 | M=10；support `[0,10)`；Gate-G proxy 的 future labels 仅供 oracle diagnostic，不能部署 | support `[0,10)`；selection `[10,210)`；sealed report `[210,end)`；所有 scored history full-window disjoint |
| 载体/状态/成本 | 每 channel 4 float rate means + four count/exposure sufficient statistics；无 generic fitting/backprop | existing B3S, side dim=4；无网络/optimizer/loss/decoder 改动；online 仅缓存 D4，额外 online MAC 约零 |
| 匹配 controls | T4 作为 cosine baseline；D4 semantic coverage / train-only normalization audit | F0、T4 frozen clean anchors；DS4 为 post-normalization nonidentity complete-row shuffle |
| seed/session 阶段 | Gate-G 4 held-in sessions的 rate proxy（不是 decoding endpoint） | 首格 `fold1_seed42`；过门才扩 `fold1_seed43`、`fold2_seed42`，两 distinct left-out sessions |
| 效应门 | Gate-G 仅授权最小 falsification，不构成 R2 effect gate | 首格：`D4-F0>=.015`、`D4-T4>0`、`D4-DS4>=.010`；三 cell：mean `D4-F0>.03`、mean `D4-DS4>0` |
| 停止 | semantic gate failure 已限制名称和 claim | 任一首格门失败即停止该 decoder branch；不扫描 M 或补充融合 |

### 候选 #2：meta-trained support-set functional encoder（仅在 #1 fail 后）

**最小形式。** 令 (S_i\) 为第 i 个 channel 在 `[0,10)` 的 activity/exposure/known
condition sufficient statistics。训练期只在 source sessions 学一个共享、permutation-
equivariant map `g(S_i) -> z_i in R^4`；新 session 只前向/stream-finalize。不得把
query behavior/condition 输入 `g`，不得在新 session 做 gradient step。D4 是同样四维、
same-budget 的闭式 primary baseline。

| 项目 | Gate A：0-GPU / CPU，先于任何模型训练 | Gate B：最小 GPU decoder screen，须另行授权 |
|---|---|---|
| 数据与边界 | **仅 source training sessions**，nested leave-one-source-session-out。每 source eval session 固定 support `[0,10)`；proxy query 使用预先固定的 future neural activity segment（建议 `[210,end)`，若不足则在 receipt 中声明唯一可用的 `[10,210)`），且 encoder feature 不索引 query labels | 与 #1 相同的 M1 development protocol，固定 M=10、support `[0,10)`、selection `[10,210)`、sealed report `[210,end)`；D4、encoder、row-shuffle 共用完全相同 query windows |
| Gate A 工作 | train on other source sessions，比较 D4、support encoder、encoder row-shuffle、support-condition-label shuffle、rate/exposure-only input。评估预注册 future **neural-rate** prediction，不作为 R2 claim；记录 M=10 condition coverage、per-channel support split stability、train-only normalization | existing B3S concat only，side dim=4；fresh E4/ES4 arms，D4 continuation/anchor 复用必须经过 exact provenance check。严禁同时加入 FiLM、PCA、T4 mixture 或更宽 decoder |
| Gate A 明确放行门 | 被留出的 source sessions上：geometric mean future-rate MSE `E4/D4 <=0.95`，且 `E4/ES4 <=0.90`；两项均至少 3/4 session 改善。若有效 source session 少于四个，转为 feasibility-only，**不放行 GPU** | 首格 fold1/seed42：`E4-F0>=+.015`、`E4-D4>=+.015`、`E4-ES4>=+.010`。若 D4 首格失败，D4 对照仍保留为 frozen closed-form baseline，不得替换为较弱 T4 |
| seeds/session 与扩展 | nested 4 source session reports；所有 hyperparameters（q=4、encoder width、condition representation）只可由 nested source folds选择 | 首格过门才扩预声明 3 cells；三 cell要求 mean `E4-D4>=+.03`、mean `E4-ES4>0`，且不能由 single left-out session 独占 |
| 成本/在线状态 | 输入必须可由 per-condition rate/count/exposure streaming sums构成，不保存 raw support trials；没有 calibration backprop | static `N x 4` cache，加一次小 shared per-channel map；online decoder MAC/state与 D4近似相同，额外仅为 cached carrier finalization |
| 提前停止 | Gate A 任一机制门未过，停止；不“用更大 autoencoder/更多 augmentation 救回” | 首格任一门失败即停；不得在同一 report endpoint 扫 latent dimension、loss、contrastive pair定义或 seed |

这个路线优于 autoencoder/contrastive 的前提恰是：它把 supervised support condition 作为
明确输入、把跨 session 学习留在 source training、把 deployment adaptation 限为
前向计算。若这仍不能超过 D4，结论应是“更复杂的 support-to-carrier mapping 没有
转化为 M1 decoder benefit”，而不是“需要更大 representation learner”。

### 候选 #3：baseline-preserving D4 attention-selection residual（仅在 #1 三-cell content 通过且 official endpoint 未被查看前）

**最小形式。** 从选定 D4 full checkpoint warm-start，严格保持现有 activity read-in、
value path、decoder、normalization和 D4 carrier。只增加 zero-init static residual：

```text
K_i = K_teacher(x_i + E_i(D4)) + DeltaK(z_i(D4))
V_i = V_teacher(x_i + E_i(D4))                         # exactly unchanged
```

等价的 cached low-rank attention-logit residual 也可接受，但只能二选一并在 Gate A
前冻结。与已有 confidence-FiLM 的区别必须可机械验证：FiLM 修改 pooled activity-to-E
融合，而此处 step 0 精确为 selected D4、只测试 static functional carrier 是否额外改变
attention selection，且不替换 activity/value route。

| 项目 | Gate A：0-GPU / CPU contract gate | Gate B：最小 GPU causal screen，须另行授权 |
|---|---|---|
| 前提 | D4 已完成三-cell content replication；official M1 held-out 尚未打开 | 只允许 selected-D4 candidate；不能回退用历史 T4、F0 或 fresh teacher 来规避 baseline |
| 固定 budget/boundary | M=10；source selected checkpoint所绑定的 support/selection windows；不访问 report/official labels | M=10，support `[0,10)`、selection `[10,210)`、report `[210,end)`；full-history disjoint |
| 必须验证 | zero-init output 与 selected-D4 checkpoint bitwise equal；cached/on-the-fly equal；D4 versus DS4 只改变 new residual input row attachment；frozen tensor/optimizer whitelist；incremental state/MAC receipt | 4 arms: D4 continuation、aligned D4 residual、DS4 residual（只 permute new residual input）、parameter-matched additive NoFiLM residual。普通 full fine-tune、generic dynamic-weight generator 一律不是此测试 |
| 效应门 | 任一 exact-baseline / attachment isolation / cached-state contract失败即不训练 | 首格：aligned 同时满足 `>=+.015` vs continuation、DS4 residual、additive NoFiLM；三 cell才可要求 mean `>=+.03`、所有 cell均正、no single session domination |
| 可允许的优化 | 只冻结的 residual tensors；如果 first screen 显示 aligned-vs-DS4 内容效应正但 `<.03`，一次性预注册“residual + existing attention out projection”策略可由 root 重审 | 不得依结果再解冻 read-in、value、decoder或增加 confidence/waveform/electrode branch |
| 状态/硬件 | rank-8、N=64参考：约 264,192 calibration-only MAC；可缓存 full-width key residual约 131,072 B FP32；online仅 key addition | 这是 accuracy candidate，非 efficiency claim；若目标硬件不能容纳该 cache，GPU positive也不能自动转部署 |
| 停止 | Gate A fail 或 D4 content未复制，停止 | 首格任一比较未过即停止；不以“FiLM 先前只差一点”或更多 rank/seed复活 |

## 4. Population subspace：允许的 CPU 审计，不是当前 GPU 主线

“population manifold 比 single unit 更稳定”在本项目尚只是文献启发。尤其在 M=10，
**PCA loading 不能直接作为 channel identity input**：

- PCA 最大化总神经方差，不是运动相关方差，主成分可主要编码 shared rate/drift；
- 同一 session 的 split PCA 有 sign ambiguity；近似重复 eigenvalue 时还有 arbitrary
  rotation，逐 loading 坐标没有固定语义；
- variable N 使不同 session 的 loading vector 不同维，不能跨 session直接比较或
  channelwise concat；
- channel order permutation 会改变 naive PCA implementation 的 row attachment unless
  the carrier is explicitly equivariant/invariant；
- population coupling若每 frame重算 covariance/SVD，会违背当前静态 calibration
  carrier和近线性在线路径。

唯一建议的 Gate A 是小型、CPU-only、source-only audit：在每 source session 的
`[0,10)` 用固定的 chronological first-five/last-five 与 odd/even 两个 split，先对
train-only chosen rank `r∈{1,2}` 的 rate-residual matrix计算 projection matrices。报告：

1. principal-angle / trace-overlap `tr(P_A P_B)/r`；
2. eigen-gap、rank degeneracy、channel-order permutation exact invariance；
3. 相同 rate/exposure 但 channel row-shuffled 的 null；
4. 只由 support 得出的 rotation-invariant quantities（例如 projection energy、
   Gram-spectrum summary）对预固定 future neural-rate proxy 的增量，且不取用 query label。

建议放行条件为所有可用 source session均 `trace-overlap >= .70`，并比 shuffled/null
高至少 `.15`，且一个固定 r 在 nested source selection 中赢得未来-rate MSE `>=5%`。
否则 **no GPU**。即使通过，它也只允许设计一个 permutation-equivariant population
context carrier；不授权 feeding PCA loadings，也不授权 PCA+FiLM。这个 Gate A 的
目的是驳倒/约束假设，不是制造另一个 feature sweep。

## 5. Waveform/SNR 的合理终点：QC，而不是另一个 auxiliary branch

可以保留的 CPU-only 问题是：waveform/SNR 是否预测 *calibration estimator reliability*，
例如 D4/E4 的 odd/even support-profile disagreement、zero-exposure比例、dead/noisy
channel flag 或 forecast error。该诊断必须 source-only、leave-one-session-out、并与
rate/exposure baseline 比较；结果仅用于数据质量报告/flag，不直接改变 decoder。

没有建议的 GPU Gate B。理由不是 F1/F2 逻辑上已证明“物理特征不存在任何信息”，而是：

1. raw static concat 在相关的 SUA protocol 中没有正向受控证据；
2. current M1 是 native-MUA，未必有同等 waveform semantics；
3. confidence-FiLM 已经不能把强 train-only reliability proxy 变成 decoder gain；
4. 把 QC signal 再作 feature fusion会重演已失败的机制，而不是检验 QC。

若未来要把 QC 用于硬 filter/abstention，必须是独立的 safety/coverage proposal，先
定义不使用 behavior R2 调出的 threshold、zero/false-rejection cost和online fallback；
它不应挤占当前 carrier search 的 GPU。

## 6. Development selection 与 official held-out 的授权顺序

| 阶段 | 可以做 | 禁止做 | 开门条件 |
|---|---|---|---|
| 现阶段 | 等待并审计当前 D4/DS4 的冻结首格结果；读取其 receipt/contract | 任何新 M1 arm、D4 report re-run、official/EvalAI | 当前 pilot自身完成 |
| Development selection | 仅按相应 receipt 在未见 session的**本地 development** report/query上比较完整预先声明的 candidate/control block；source train sessions用于 Gate A | 根据已看到的 report 选 latent/rank/M、epoch、loss或再开“补救”臂；把 local development写成 official result | D4 pass/fail decision树的对应节点 + root新增receipt |
| Candidate freeze | 固定唯一 architecture、support M=10、feature semantics、normalization、checkpoint-selection rule、seeds/sessions、submission image/hash；对本地 development endpoint封口 | 再看同一 local held-out/development结果来调候选；多个candidate sequential official试投 | 完整 development effect/control gate过关 |
| Official M1 few-shot held-out | 对冻结的唯一候选进行一次正式测试/提交；报告 calibration scope、online state与 latency | 用 official score进行再训练、再选 checkpoint、调门槛、或重复试多个nearby candidate | root明确授权并确认测试范围未被此前开发占用 |

若 M1 的 local `held-out-calib` 没有 M=10 后 query，official held-out 更应被视为
稀缺终点；不能因为本地 endpoint不方便而跳过 D4/controls，也不能把 historical
overlap replay 当作节省它的替代品。future held-in-calib post-support endpoint 可用于
development，但需要一个新、独立、fail-closed plumbing receipt，并在第一次结果出现
前固定它的 selection/report split。

## 7. 明确的现在不做清单

- 不在当前 D4 首格结果前启动 autoencoder、contrastive learner、PCA carrier、D4+T4
  mixture、FiLM、hypernetwork、support sweep或 quantization。
- 不恢复 raw waveform/SNR concat、absolute-electrode lookup、T4GATE、same-electrode
  REL 或 CMP spatial-neighbor route。后两者已有更直接的 negative / NO-GO 证据。
- 不把 FALCON M2 K4、SUA T4 或 pseudo-MUA的大小直接移植为 M1 threshold；label
  information、signal view、support budget和endpoint不同。
- 不在同一已查看 M1 report/development endpoint上逐轮调 carrier dimension、PCA rank、
  contrastive定义、FiLM rank或checkpoint。

## 8. 引用 artifact（阅读/实现前的最低证据集）

| Artifact | 本草案使用的关键事实 |
|---|---|
| `sua_exploration/docs/CURRENT_RESULTS.md` | SUA F1/F2、T4/T8、pseudo-MUA、T4GATE、REL、FiLM、M1 endpoint correction及当前证据边界 |
| `sua_exploration/docs/T4_SUA_AUXILIARY_EXPERIMENT_PROGRAM.md` | raw physical feature的狭义结论、matched controls、label accounting与原始 C1–C4 kill logic |
| `sua_exploration/docs/T4_NEXT_ROUND_IDEATION.md` | zero-init coupled key-residual的非重复实现边界、cost以及已有 contract readiness |
| `sua_exploration/docs/HANDOFF_ELECTRODE_SPATIAL_PRIOR.md` | spatial prior Stage 0 真实 geometry / low-budget shrinkage NO-GO，不能将 handoff当作待执行命令 |
| `sua_exploration/results/m1_clean_selection_v1/report.md` | M1 clean T4/F0/TS4 numbers、半平面覆盖和仅两个 left-out sessions的限制 |
| `sua_exploration/docs/M1_D4_MINIMAL_GPU_PILOT_PROTOCOL.md` | D4/DS4 semantics、exact first-cell thresholds、frozen boundaries与非目标 |
| `sua_exploration/results/m1_d4_semantics_v2/ROOT_GATE_G_REVIEW.md` | D4 的 semantic limitation、rate-proxy evidence及one-minimal-pilot-only authorization |
| `sua_exploration/results/m1_d4_pilot_v1/phase_b_receipt.json` | current authorized arms/cell、frozen `[0,10)/[10,210)/[210,end)` windows、train-only normalization和no-hidden access receipt |

## 9. Root 决策请求

建议 root 在 D4 首格完成后只作一个二元选择：按 §2 机械执行 **pass extension** 或
**fail stop + endpoint readiness**。不建议在该节点同时批准候选 #2/#3 GPU。候选 #2
首先只需要 CPU Gate A；候选 #3 的先决条件更高，必须等 D4 three-cell content和
candidate-freeze顺序成立后再重新授权。
