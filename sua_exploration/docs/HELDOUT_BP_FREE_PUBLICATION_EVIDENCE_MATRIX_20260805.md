# Held-out BP-free calibration：publication evidence matrix

**审计日期：** 2026-08-06  
**审计性质：** 结果与边界汇总；C1 shared-zero4、external sub-M V9、传统对照与 external label-budget 行为曲线已完成并纳入；Native-M2 fresh matched Stage A 已完成并强正，full three-seed gate 尚待完成。  
**目的：** 把“已经有数字”“可以发表的主张”“仍然缺失的因果对照”分开，防止把开发集、代理任务、原生 MUA、pseudo-MUA 和外部动物混为一谈。

---

## 1. 必须先固定的方法定义

T4 对 unit/channel `i` 的校准块 firing rate 拟合

```text
r_i(theta) = b_i + a_i cos(theta) + c_i sin(theta)
m_i = sqrt(a_i^2 + c_i^2)
T4_i = [a_i, c_i, m_i, b_i]
```

项目中通常所说的 **T4 系统** 不是“把四维特征临时接到任意现成 frozen decoder 上”。正确口径是：

1. 在 source sessions 上，网络已见过 T4 路径；主 SUA/C1 路径的 encoder 与 decoder 是共同训练/适配的；
2. 到新的 held-out target session 后，仅用 chronological calibration block 解析拟合描述子，并一次性生成/缓存 identity；
3. target-session calibration 和后续 query forward 均无 optimizer step、无 backward、无权重更新；
4. 因此可称为 **deployment-time / held-out-session backpropagation-free calibration**，不可省略限定语而笼统称“整个 T4 训练无需反向传播”。

Native-M2 Phase C 是一个特意构造的、更严格的因果变体：T4 臂在 source training 中锁定同 fold、同 seed 的已选 SPINT decoder 31 个 tensors，只训练 T4-aware identity path；target calibration 仍然 BP-free。它不能反过来改写主 SUA/C1 模型的训练事实。

T4 还是一个 **supervised calibration** 方法：它使用 support trials 的 target-direction labels 与 trial rates；SPINT/B3 基线只获得同一批 neural support。故 `T4-SPINT` 是“有标签的部署包相对 neural-only 部署包”的效用比较，而不是 equal-label estimator 比较。

### 证据等级

| 等级 | 含义 |
|---|---|
| A — external hidden | organizer-hidden/official endpoint；系统级效用最强，但仍受 comparator matching 约束 |
| B — strict development | held-out session、chronological support/query disjoint，完整行为 R²；开发证据，不等于一次性 formal confirmation |
| C — source/proxy | source-only、LOSO、描述子重建或 later-neural proxy；可筛选机制，不能声称 held-out behavior gain |
| D — diagnostic/withdrawn | 无未来 query、support/query overlap、checkpoint/confound 或仅工程诊断；不能作为有效性证据 |
| P — pending/sealed | 协议或运行存在，但结果尚未合法封口；当前 R²/delta 必须写 `N/A` |

---

## 2. Native MUA evidence matrix

### 2.1 Native M2

| 方法 | 数据与 endpoint | held-in / held-out | chronological support → query | 标签预算 | target calib BP | decoder 来源 | comparator | 已知 R² / delta | 稳定性 | 等级 | 可发表主张 | 禁用主张 |
|---|---|---|---|---:|---|---|---|---|---|---|---|---|
| Submitted T4 system | FALCON native M2 EvalAI `578221` | organizer-hidden held-out；另报 held-in | 官方容器的 few-shot protocol；offline fit 后 cached identity | support direction labels；具体容器预算按 frozen submission | 无 | epoch-34 teacher decoder；source-trained T4 path | original SPINT `578218`，epoch-27 decoder | held-out `0.30324395` vs `0.18647872`, **Δ `+0.11676523`**；held-in Δ `+0.01936560`；normalized latency Δ `-0.06945843` | 只有 submission aggregate；无 paired session inference | **A，system-level** | 已提交的 supervised T4 package 在 organizer-hidden native-M2 held-out 上优于原始 SPINT package；cached identity 未造成官方 online-latency penalty | 不得把 `+0.1168` 全归因于四维 T4；不得称 matched-decoder causal effect；不得声称 equal-label fairness |
| Corrected T4 local replay | native M2 M33/q33，4 个有未来 query 的开发 sessions | outer session held-out development | neural support `[0,33)`；query `[33,end)`，完整 history 在边界后 | 33 neural trials；其中 16 个有限方向标签 + 17 centre/rest | 无 | legacy selected source model | F0 / TS4 | `T4-F0=+0.071980`，`T4-TS4=+0.066484` | 两者 4/4 session 正；`n=4` exact sign `p=0.125` | **B-，小样本开发** | corrected post-support M33 replay 与官方系统正号一致 | 不得称 formal、显著性确认或新动物；不得复用已作废的旧 `+0.06979` |
| T4 local replay | native M2 M24/q24，6 个开发 sessions | outer session held-out development | support `[0,24)`；未来 query `[24,end)` | 24 neural trials与相应 support direction labels | 无 | legacy selected source model | F0 | `+0.052885`；common epoch-9 `+0.046628` | 5/6 sessions 正；仅一个 audited cell | **B-/D+** | 在该单 cell 的 post-support endpoint 上存在候选正效应 | 不得把单 cell 当作 three-seed 稳定结论；不得与 internal minival 混合 |
| **Ridge24-W50 closure comparator** | native M2 M24/q24，同 6 个开发 sessions | outer session held-out development | first24 support；query W50 history 完整 post-boundary | first24 的逐 bin continuous velocity；固定 `lambda=1` | 无；per-session dual closed form | 无 source checkpoint；target-session direct linear W50 readout | F0 / T4 / K4 | Ridge `0.113916`；F0/T4/K4 `0.173901/0.226786/0.245774`；K4−Ridge `+0.131857`，T4−Ridge `+0.112870`，K4−T4 `+0.018987` | K4−Ridge 4/6 正；一次性 local descriptive closure | **B-/classical boundary** | source-pretrained T4/K4 systems 在该固定 dense-label ridge 之上；K4 pipeline 高于 direct ridge | 不得把 K4−Ridge 全归因 K4；不得称 direct-decoder 上界、formal confirmation 或 K4 主线复活；固定 λ 不代表所有 ridge |
| **K4/KS4 P2 M33 eligible subset** | native M2 M33/q33；仅 4 个有 future query 的 sessions，7 source LOSO folds | held-out development subset | support first33；另外 2 sessions zero-query/ineligible | 逐 bin continuous velocity | 无 target-session BP | source-trained B3S K4/KS4 | K4−KS4 row attachment | K4 `0.308510`，KS4 `0.196836`，Δ `+0.111673` | 24/28 fold×session pairs、7/7 fold means 正；一个 session mean 略负 | **B-/mechanism subset** | K4 内容在 eligible M33 subset 中优于 row-shuffle | 不得称 6-session、K4>T4/SPINT/F0 或 `primary_effective`; 两个 zero-query sessions 不得按零分纳入 |
| Matched M24 candidate | native M2，3 seeds × 6 sessions | held-out development | chronological M24/q24 | 24 support labels | 无 | paired local source models | matched comparator | mean Δ `+0.036604` | 2/3 seed means、4/6 session means 正；bootstrap 跨 0；frozen gate fail | **B-，negative stability gate** | 平均值为正但不稳定 | 不得称“通过”“显著”或稳定泛化 |
| Joint source-trained screen | native M2 `m2_joint_t4_upperbound_v2` | held-out development screen | post-support query | support labels | 无 | encoder+decoder source-time joint train | independently source-trained zero4 / TS4 | `T4-zero4=+0.018389`；`T4-TS4=+0.057094`；`zero4-TS4=+0.038705` | 前两项均仅 4/6 sessions 正；绝对增益低于 `+0.03` | **B-/diagnostic** | TS4 很可能有害，因此 T4-TS4 不能单独证明 T4 的绝对价值 | 不得把 `+0.0571` 解释为纯 descriptor gain；不得声称 joint training 已证明上限 |
| **Fresh matched Phase C retraining** | native M2，7 outer folds × 3 seeds × 2 arms | 预注册 outer-session held-out development | support `[0,33)`；query `[33,end)`；同 neural exposure | T4：same support 中 16 个有效方向标签；SPINT：0 target labels | **无** | T4 臂锁定 same-fold/seed selected SPINT decoder；source 侧只训练 identity path | matched SPINT | **Stage A seed42：SPINT `0.293110`，T4 `0.382906`，Δ `+0.089796`** | exact 14/14；7/7 outer sessions 正；delta 范围 `+0.016567` 至 `+0.187562`；frozen severe-negative stop 未触发 | **B-/Stage A** | matched decoder 后仍有强、一致的 T4 development signal；official positive delta 不像是纯 decoder-epoch artifact | Stage A 明确是 `continue_without_positive_claim`；不得替代 seeds42/43/44 的 full 42-cell gate，不得称 equal-label comparison |

Native-M2 deployment-cost receipt还支持一个独立的工程事实：在该 Phase-C 规格下，两臂 online 均为 `84,021,248` MAC/query；SPINT support calibration 约 `1,875,935,232` MAC，T4 support encoder + analytic fit 约 `21,404,100` MAC；peak FP32 calibration state 为 `235,008` vs `64,552` bytes，final cached identity 均为 `19,200` bytes。这个结果支持“解析 T4 calibration 不必很重”，不支持“任何 T4 实现都更快”。

### 2.2 Native M1

| 方法 | 数据与 endpoint | held-in / held-out | chronological support → query | 标签预算 | target calib BP | decoder 来源 | comparator | 已知 R² / delta | 稳定性 | 等级 | 可发表主张 | 禁用主张 |
|---|---|---|---|---:|---|---|---|---|---|---|---|---|
| Submitted T4 system | FALCON native M1 EvalAI `578245` | organizer-hidden held-out + held-in | official few-shot endpoint | first-10 support direction labels | 无 | submitted T4 system | original `578244` | held-out `0.6447659719` vs `0.6485909555`, **Δ `-0.0038249836`**；held-in Δ `-0.0062515696` | aggregate only；无 per-session paired inference | **A，negative boundary** | 当前 T4 submission 不改善 official native-M1 accuracy | 不得声称 T4 普遍改善 native MUA；也不得由一个系统断言所有 T4-like estimator 永远无效 |
| Historical local M1 T4 | 3 held-out-calib files / internal minival | nominal held-out，但 endpoint 无效 | 每文件恰好 10 trials；support `[0,10)` 后无 future query；internal minival 落在 support 内 | 10 labels，且方向仅半平面 | 无 | legacy source model | F0 / TS4 | internal `T4-F0=+0.007837`（1/3 正）；旧 `T4-TS4=+0.024797` | support/query overlap 或 query empty | **D — withdrawn** | 可作为“为什么 M1 不能复用旧 local claim”的审计证据 | 不得称 held-out、chronological future-query 或内容因果证据 |
| **Fixed-K temporal prototype P20** | M1 四个 source sessions，later-neural proxy；无 behavior decoder | source-LOSO proxy，不是 target held-out R² | support-only descriptor → later-neural reconstruction proxy | 无 target labels | 不适用；CPU-only | **无 decoder / 无 K-V consumer** | rate-only；matched-width order-invariant B20 | 初始 `P20-rate-only=+0.100526`；决定性 `P20-B20=-0.041963`；P20 `0.862794` vs B20 `0.904758` proxy R² | 对 B20 为 4/4 sessions 负，paired CI `[-0.076004,-0.007923]` | **C，terminal negative** | temporal ordering 未超过更简单的 marginal rate/count distribution；该 carrier 分支应停止 | 不得称 fixed-K memory 改善 behavior、held-out decoding；不得声称已测试 cross-attention/K-V memory |

---

## 3. Sorted SUA evidence matrix

这里的 6 个 sub-C sessions 是反复使用的 **held-out development sessions**；形式上保持 activity first-30、T4 first-50、query strictly after trial 50，但不能重新包装成一次新的 formal test。既有 P3 receipt 已占用 formal scope。

| 方法 | 数据与 endpoint | held-in / held-out | chronological support → query | 标签预算 | target calib BP | decoder 来源 | comparator | 已知 R² / delta | 稳定性 | 等级 | 可发表主张 | 禁用主张 |
|---|---|---|---|---:|---|---|---|---|---|---|---|---|
| Ordinary T4 | sorted SUA，27 source / 6 reused sub-C development | session-held-out development | activity `[0,30)`；T4 `[0,50)`；query `>50` | 50 rewarded-trial direction labels；B3 only neural first 30 | 无 | encoder+decoder 在 source 侧 joint-trained / teacher initialized | F0 / TS4 | absolute `0.574976`；`T4-F0≈+0.2528`；`T4-TS4≈+0.2522` | 3/3 seeds、6/6 sessions 正 | **B** | supervised BP-free T4 system 在 reused held-out SUA development sessions 上有强且稳定增益 | 不得称 formal、新 subject、无标签或 arbitrary frozen-decoder plug-in |
| AC4 component attribution | 同上 | session-held-out development | 同上 | 50 labels | 无 | source joint-trained AC4-capable system | full T4 / AC4-RS4 / zero4 | AC4 `[a,c,0,0]` absolute `0.562753`；`T4-AC4=+0.012222`；`AC4-AC4RS4=+0.294163`；zero4 `0.326008` | row attachment contrast 3/3 seeds、6/6 sessions 正；AC4 within `0.03` of full | **B，mechanism development** | 正确附着的一阶谐波 `[a,c]` 保留了 full T4 的主要效果；full `[m,b]` 不是必要条件 | 不得把 full T4 gain 全说成 baseline rate 或 modulation magnitude；不得把 independently-trained systems 当 fixed-weight causal ablation |
| Cross-budget correction Step2A | SUA source nested LOSO + 6 reused dev descriptor audit | source/proxy；非 behavior R² | calibration descriptors only；无 query decoder | M as input；利用 fit statistics | 无 deployment BP；CPU-only estimator | 无 behavior decoder | `q_unit+M` vs `M-only` | source LOSO descriptor MSE Δ `-0.7671`（越低越好）；target-free dev Δ `-0.0226` | source 21/27 favorable、CI `[-1.3035,-0.2307]`；dev 3/6、CI `[-0.3268,+0.2817]` | **C，non-transfer** | source 内可学习校正，但没有稳定 transfer 到 reused held-out sessions | 不得称低预算 T4 behavior 提升或 held-out R² improvement |
| **Experiment B v7 estimator upgrade** | SUA source-only CPU audit | source-only；formal/dev target unopened | estimator predictive/reliability audit only | same source labels | 无 | 无 GPU decoder run | EB ridge / second harmonic / Poisson IRLS vs base | EB best prospective deviance ratio `0.976601`，directional reliability Δ `+0.000124` | 远低于 frozen `+0.02` gate；all candidates fail；`no_winner_no_gpu` | **C，terminal negative** | 当前候选 closed-form estimator upgrade 未达到进入行为 GPU test 的先验门 | 不得称 R² 下降/上升；这里根本没有 behavioral decoder endpoint |
| **C1 shared zero4 three-arm control** | reused sub-C，shared SUA+pseudo model family | held-out development | frozen 30/50/>50 chronology | T4/TS4 50 labels；zero4 0 labels | 无 | 三臂各自 source-trained；不是同权重替换 descriptor | shared T4 / shared TS4 / shared zero4 | **SUA:** T4 `0.574378`，zero4 `0.311042`，TS4 `0.288904`；T4-zero4 `+0.263335`，T4-TS4 `+0.285474`。**pseudo-MUA:** T4 `0.546109`，zero4 `0.225019`，TS4 `0.176641`；T4-zero4 `+0.321090`，T4-TS4 `+0.369468` | 四个 contrast 均 3/3 seed means、6/6 session means 正；paired-2SE 与 two-way bootstrap 95% lower 均 >0；全部 frozen gates pass | **B，strong reused-development system contrast** | 在 shared-weight family 中，T4 相对无 descriptor 的绝对系统价值和相对错误 row attachment 的敏感性均强且跨 seed/session 一致 | 不得称 fixed-weight descriptor causality、formal/new-subject confirmation 或 native MUA；T4-zero4 同时包含 source-trained system 与 target-label availability 差异 |

---

## 4. Deterministic pseudo-MUA evidence matrix

Pseudo-MUA 是把 sorted SUA spikes 按 electrode 做确定性求和得到的 channel view；它不是 native threshold-MUA。其价值是检验 granularity transform 与 carrier 合并律，不是替代 FALCON native-MUA 外部验证。

| 方法 | 数据与 endpoint | held-in / held-out | chronological support → query | 标签预算 | target calib BP | decoder 来源 | comparator | 已知 R² / delta | 稳定性 | 等级 | 可发表主张 | 禁用主张 |
|---|---|---|---|---:|---|---|---|---|---|---|---|---|
| Separately trained pseudo-MUA T4 | deterministic electrode-sum view，6 reused sub-C dev sessions | session-held-out development | activity first 30；T4 first 50；query after 50 | 50 support direction labels | 无 | pseudo-view model source-time joint-trained | F0 / TS4 | `T4-F0=+0.317739`；`T4-TS4=+0.365674` | aggregate positive；开发 endpoint | **B** | T4 carrier 在 deterministic pooled view 上仍有强 development utility | 不得称 native MUA、formal 或跨动物 |
| Algebraic SUA→pseudo carrier audit | 33 sessions，count/rate identity audit | 非 decoder endpoint | same trial bins | same calibration labels | 不适用 | 无 decoder | direct pooled fit vs merged SUA carrier | exact binned count max error `0`；calibration count `4.77e-7`；pooled-rate T4 `4.68e-6`；693 singleton checks | 33 sessions 全审计 | **B/C，algebraic** | `a,c,b` 可加、`m` 应在 pooled coefficients 上重算；实现与代数一致 | 不得由代数一致性推导 behavior R² 必然提高 |
| **Paired SUA/pseudo-MUA co-training C1** | 一套 shared weights；随机 view batch；无 consistency loss | 6 reused held-out development sessions | 30/50/>50 | 每 view 正确层级的 50 T4 labels | 无 | encoder+decoder source-time jointly trained；两个 view 共享全部权重 | separately trained T4；shared TS4 | shared absolute：SUA `0.574378`，pseudo `0.546109`；shared-separate：SUA `+0.008351`，pseudo `+0.012543`；shared T4-TS4：`+0.285474/+0.369468` | non-inferiority：SUA 3/3 seed、4/6 session 正；pseudo 2/3、5/6；T4-TS4 两 view 均 3/3、6/6 | **B** | 一套共享权重可覆盖 SUA 与其 deterministic pooled view，且相对 separate models 在 `0.03` margin 内非劣 | 不得称 co-training 带来显著 accuracy gain；不得声称用了 consistency loss、teacher-student、view-specific head 或 native-MUA 泛化 |

---

## 5. External subject sub-M evidence matrix

| 方法 | 数据与 endpoint | held-in / held-out | chronological support → query | 标签预算 | target calib BP | decoder 来源 | comparator | 已知 R² / delta | 稳定性 | 等级 | 可发表主张 | 禁用主张 |
|---|---|---|---|---:|---|---|---|---|---|---|---|---|
| **V9 shared three-arm external-subject confirmation** | DANDI 000688 subject sub-M；22 assets 中按 frozen `N<100` 得 15 eligible；SUA primary、pseudo secondary | 新动物 / external-session held-out | activity first 30；T4/TS4 first 50；query strictly after 50 | T4/TS4：50 direction labels；zero4：0 | 无 | `shared_t4/shared_zero4/shared_ts4` 三臂各自从 source 训练；非 checkpoint continuation | T4-zero4（absolute system value）与 T4-TS4（attachment）分开 | **SUA:** T4 `0.356828`，zero4 `-0.057766`，TS4 `-0.115319`；T4-zero4 `+0.414594`，T4-TS4 `+0.472147`。**pseudo-MUA:** T4 `0.306073`，zero4 `-0.086878`，TS4 `-0.164314`；T4-zero4 `+0.392951`，T4-TS4 `+0.470387` | 四个 contrast 均 3/3 seed means、15/15 session means 正，exact two-sided sign `p=6.1035e-5`；100k hierarchical-bootstrap lower 95% 分别为 `0.340273/0.395148/0.314165/0.388574`；全部 frozen gates pass | **B+，same-Dandiset external subject** | supervised、target-session BP-free T4 system 的 absolute utility 与 row attachment sensitivity 跨到同一 DANDI 的新动物；SUA primary 独立通过 | 不得称 independent-lab/dataset replication、fixed-weight descriptor causality 或 equal-label comparison；不得以 pseudo view rescue SUA |
| **External sub-M classical controls** | 与 V9 相同的 15 sessions、两个 view、逐字节相同 query targets；V9 完成后的 post-hoc additive reuse | 新动物 / external-session held-out | first 50 calibration trials；query strictly after 50 | F0：0 target labels；PV50：50 个方向标签并用逐 bin 连续速度拟合 affine gain；Ridge50：first 50 的逐 bin 连续速度；均比 T4 的每 trial 一个方向标签信息更丰富 | 无；PV/Ridge 为闭式拟合 | F0 为 historical independently trained B3；PV/Ridge 为 target-session analytic direct decoder | V9 T4；shared-zero4 仍是严格匹配无标签 control | **SUA:** F0/PV/Ridge/T4 `-0.196077/0.115374/0.417922/0.356828`；T4−PV `+0.241454`，T4−Ridge `-0.061094`。**pseudo:** `-0.204725/0.104219/0.410193/0.306073`；T4−PV `+0.201854`，T4−Ridge `-0.104120` | T4−PV 两 view 均 14/15 正且 bootstrap CI 下界为正；T4−Ridge 仅 3/15 与 2/15 正，bootstrap CI 均跨 0 | **B，标签效率/经验边界对照** | T4 显著胜过传统 PV；固定超参数 Ridge50 是本 cohort 的 dense-label 线性参考，均值高于 T4 | 不得称 equal-label、matched-architecture、监督解码上界，或 T4 打败 Ridge；F0 仅为 system-level ancillary comparator；不增加独立确认次数 |
| **External sub-M frozen-model T4 label-budget curve** | 同一 V9 `shared_t4` checkpoints、15 sessions、两个 view；450 个新 M10/15/20/30/40 cells，M50 复用 | 新动物 / external-session held-out；post-hoc endpoint reuse | activity first 30 固定；所有 M 的 query 都 strictly after 50，target 逐字节相同 | `M={10,15,20,30,40,50}` 个 trial-direction labels | 无 | source 在 M50 descriptor distribution 上训练；target 只重算闭式 T4 | 每个 M 对同 seed/session M50；辅助对照 zero4/TS4/PV/Ridge | **SUA M10/15/20/30/40/50:** `.304264/.338115/.351767/.358154/.351648/.356828`；**pseudo:** `.259185/.288234/.298447/.305291/.301075/.306073`。M15−M50 `-.018713/-.017839`；M30−M50 `+.001326/-.000782` | M15 为两个 view 最小 grand-mean + 3-seed `-.03` 操作门；但 bootstrap lower 为 `-.04758/-.05112`，不是 CI non-inferiority。M30 intervals `[-.00978,.01257]/[-.01288,.00966]` 完全高于 `-.03` | **B，exploratory fixed-model robustness** | M50-trained T4 在少至 15 labels 时保留 group-level/seed-stable adequacy；M30 在同 first-30 activity block 内近乎复现 M50 | 不得称 M15 是理论最小、正式 non-inferior、新独立 external confirmation，或 trial15/30 已验证 early-start；行为 R² 不逐 M 单调 |
| **External sub-M M30 true-early-start** | 同一 V9 三臂 checkpoints、同 15 sessions、两个 view；270 个新 forward cells | 新动物 / external-session held-out；同一 external cohort 的 post-hoc latency characterization | activity 与 T4/TS4 均 first30；所有 W50 query history fully post-trial30；all-post30 与 trials31--50-only 分开 | T4/TS4：30 direction labels；Zero4：0 | 无 | frozen `shared_t4/shared_zero4/shared_ts4` systems | T4−Zero4 absolute value；T4−TS4 attachment | **all-post30 SUA/pseudo:** T4 `.360607/.307641`，T4−Zero4 `+.419778/+.394296`，T4−TS4 `+.474675/+.470110`。**trials31--50 only:** T4 `.393664/.339323`，T4−Zero4 `+.483843/+.439688`，T4−TS4 `+.508258/+.475302` | 四个 endpoint×control contrast 在每个 view 均 3/3 seed、15/15 session 正；SUA early-only bootstrap lower `+.420583/+.453254`；pseudo `+.379167/+.410997`；T4 absolute seed means 全正 | **B+，strong latency characterization** | first30 neural activity 与30个 direction labels 后可立即开始正 absolute decoding；新增 trial31--50 window 上 T4 的 absolute utility 与 row attachment sensitivity 均强且跨 session 一致 | 不得称第二次 external confirmation、independent dataset、fixed-weight/equal-label causal effect，或 early-only R² 高于 late R² 代表性能随时间下降；同 cohort/post-hoc scope 必须披露 |

V9 表中数字全部来自本地 TorchMetrics `1.5.1` 对 270 个封存 prediction/target artifacts 的
完整重算。远端 NumPy approximation 与权威值的 cell-level 最大绝对偏差为 `5.71e-4`，超过其
实现注释中的 `2e-6` 经验容差；尽管四个 grand contrast 仅相差 `<=2.42e-5` 且不影响判决，
远端 aggregate 仍不得作为论文数字来源。

相对 reused sub-C C1，external sub-M 上 absolute T4 仍下降：SUA `-0.217550`、pseudo-MUA
`-0.240036`；只是 zero4/TS4 的下降更大（SUA `-0.368808/-0.404223`，pseudo-MUA
`-0.311897/-0.340955`）。因此跨动物结论是 **relative robustness and retained positive
absolute decoding**，不是 domain shift 无损或 T4 absolute score 提升。SUA T4 的 15/15
session means 为正，pseudo-MUA 为 14/15；15/15 正的是四个 paired contrast。

sub-M 的 50 labels 占每个 session rewarded trials 的 `7.46%--28.90%`，equal-session mean
`22.28%`，pooled `750/4297=17.45%`；这排除了通过更大 fractional calibration block 获得
external gain 的解释。

传统对照进一步限定了这个结论。T4−PV50 在 SUA/pseudo-MUA 分别为 `+0.241454`
（95% CI `[+0.173660,+0.300498]`）与 `+0.201854`
（`[+0.126247,+0.265518]`），两者均 14/15 sessions 正。Ridge50 直接用 first-50 内逐 bin
连续速度拟合 50-bin history，均值则比 T4 高 `0.061094/0.104120`；T4−Ridge50 的 CI
`[-0.175229,+0.069881]` 与 `[-0.211620,+0.004589]` 都跨零。故最强可守主张是 T4 相对 PV
有明确优势，并以稀疏 trial-level direction labels 接近 dense-label ridge，而不是在同等标签下
优于 ridge。historical F0-B3 与 V9 T4 并非同一 source-trained 架构，不能替代 shared-zero4。

固定模型标签预算曲线补上了此前只有 descriptor-reliability proxy、没有行为 R² 的缺口。
M15 把 pooled 标签占比从 `17.45%` 降至 `5.24%`，并在两个 view 同时满足 grand mean 与
三枚 seed mean 均不低于 M50 `0.03 R²` 的操作门；但其 hierarchical intervals 下界仍低于
`-0.03`，且只有 10/15 SUA、9/15 pseudo sessions 各自落在该 margin 内，故不能包装成
formal non-inferiority。M30 的两个 interval 完整落在 margin 以上，并与 B3 activity 所需的
first-30 support 对齐，是更稳妥的部署点。由于 evaluation start 仍固定为 trial 50，这证明
label-count robustness；随后独立、完整的 M30 true-early-start replay 把 query 严格移到
trial30 后，并将 trial31--50 单独评分。该 early-only endpoint 在 SUA/pseudo 上的 T4
absolute R² 为 `0.393664/0.339323`，相对 Zero4 为 `+0.483843/+0.439688`，相对 TS4
为 `+0.508258/+0.475302`，四项均 15/15 sessions 正且 bootstrap lower >0。因此
“first30 后即可开始解码”现有行为 R² 支持；仍不得把同一 cohort 的 latency replay 计作
新的外部确认。

---

## 6. 三条新方向的统一判断

| 方向 | 当前真实状态 | 是否还缺一个 GPU 实验 | 判决 |
|---|---|---|---|
## 7. 方法收口前的必要实验状态

### 必要实验 1：完成 Native-M2 fresh matched Phase C

这是当前唯一能直接解除官方 M2 `epoch-34 T4 vs epoch-27 original` 混淆的本地因果桥。seed42 Stage A 已以 exact 14/14 完成：`T4-SPINT=+0.089796`，7/7 sessions 正，未触发 severe-negative stop。receiver-lineage 审计进一步确认，这 14 格属于只允许 seed42 且无 Stage-B/full-opening capability 的 r9 终止分支；不能与新训 seeds43/44 拼接，也不能导入 r10。下一步必须在 fresh r10 static lineage 中保持 same fold/seed、same selected SPINT decoder tensors、same neural support `[0,33)`、query `[33,end)`，从头产生完整 42 个 sealed cells 并一次性形成 full aggregate。该运行仍由其他工作流负责，本审计不接管其训练。

### 必要实验 2：external sub-M V9（已完成）

V9 已回答此前最重要的 SUA 外推缺口：强 development gain 能跨到同一 DANDI 的新动物，而且 shared model family 在 SUA/pseudo 两个 view 上的 absolute T4 value 与 row-attachment sensitivity 同时成立。SUA primary 自身 15/15 session 正并通过所有门，不依赖 pseudo view rescue。保留的边界是 same-Dandiset external subject，而不是 independent-lab/dataset replication。

除此以外，当前主论文唯一未闭合的核心证据缺口已从“matched Native-M2 是否有正号”收缩为“seed42 的强正号能否在 fresh r10 全 42-cell、三-seed full gate 中复现”。新的 FiLM、dynamic fusion、cross-budget correction、fixed-K memory、M1 tuning repair 或 quantization accuracy search 都不应替代这个稳定性收口。P2/P3 属于另一工作流的预先冻结后续计划，不由本审计启动或修改。

---

## 8. 已发现的数字冲突、过度主张与处理规则

1. **官方 M2 system gain 与 matched T4 effect 不可混为同一个数。** `+0.11676523` 是 submitted end-to-end system delta；Stage-A matched decoder delta 为 `+0.089796`，说明正效应不完全由 decoder epoch 解释，但 full three-seed gate 前仍不得叫稳定的 T4-only causal effect。
2. **两个 M33 数字族不可平均。** corrected M33/q33 是 `T4-F0=+0.071980`、`T4-TS4=+0.066484`；另有 matched epoch-34 replay 摘要写 `+0.06420/+0.09558`。它们的 comparator、checkpoint/window provenance 不同；在 exact endpoint 对齐前只能分列，不能叫重复实验。本文矩阵采用有明确 corrected aggregate receipt 的前一组。
3. **旧 M33 `+0.06979` 已作废。** 两个 zero-query sessions 曾污染聚合；不得再用它证明 M33 已“立住”。
4. **M24 internal 与 held-out 反号未解释。** internal LOSO 曾报 `T4-F0=-0.1502`，strict disjoint held-out 为 `+0.052885`。这说明 internal screen 不能替代 future-query held-out evidence；不能选择性只报正值，也不能把反号归因给单一机制。
5. **Native M1 旧 local gain 无效。** support 后没有 future query，或 minival 位于 support 内；只保留 official aggregate negative boundary。
6. **`T4-TS4` 不是 absolute value。** 错行 descriptor 可以主动伤害网络；必须另看 zero4/F0。C1 zero4 现已给出强正 `T4-zero4`，但因三臂分别 source-trained，仍只能称 system-level absolute value，不能称 fixed-weight descriptor causal effect。
7. **C1 不是 consistency-learning 实验。** 实际 `lambda_consistency=0`，是两 view task loss 的 shared-weight training；“consistency loss 提升泛化”属于未做实验。
8. **Pseudo-MUA 不能替代 native MUA。** 它是 deterministic SUA electrode pooling，而 FALCON M1/M2 是 native threshold MUA。
9. **Proxy metric 不是 behavior R²。** Experiment B 的 deviance/reliability、fixed-K 的 later-neural reconstruction 都不能进入 held-out decoding accuracy 表述。
10. **“T4 无需反向传播”必须加作用域。** target-session calibration 确实无 backward；source training 通常有反向传播，且主 SUA/C1 decoder 与 T4-aware path 共同适配。
11. **开发 formal scope 已消耗。** C1 没有打开那 6 个 formal paths 是 data-access 事实，但既有 P3 scope receipt 使同一 formal scope 不再是 pristine one-shot confirmation。
12. **pending 就是无数字。** Native-M2 fresh retraining 在完整终态 receipt 之前一律写 `N/A`，不得从 log、partial checkpoint 或运行中 cell 预报趋势。C1 zero4 与 sub-M V9 已有终态 aggregate，不再属于 pending。

---

## 9. Authoritative local evidence ledger（当前文件 SHA-256）

| 本地文件 | SHA-256 | 用途 |
|---|---|---|
| `docs/EXTERNAL_AI_REVIEW_CURRENT_STATUS_20260805.md` | `42bee9a37ce4e440b53722b8746b2dfcaf8a2707a3ddf7967cb4c2ae112ce8eb` | 跨域当前数字、口径和 pending 边界 |
| `docs/HELDOUT_BP_FREE_CALIBRATION_METHOD_CLOSURE_20260804.md` | `645fe780336cdcc90bb60fc6d1f23ca9a6e306a7778489ca0812ceb2bb303ee7` | T4 方法定义、成本、跨域 closure |
| `results/e8_falcon_evalai_m2_t4_seed42_v1/submission_578221_receipt.json` | `90692d4d1f61c7d15b78da4af8ac0ffa0ee6e22f8aca67a37b9561189c092c44` | official M2 T4 score |
| `results/e8_falcon_evalai_m2_spint_epoch27_v1/submission_578218_receipt.json` | `5f4499810758078dff0020ae1c3bfb14fe404d30446a077048febe620111e8ee` | official M2 original score |
| `results/evalai_m1_threeway_v1/threeway_summary.json` | `fc6be339b0db98879e897a9186380b5483b082d42aa9937e1ab347a6ca25f010` | official M1 original/T4/D4 |
| `results/t4_paired_view_c1_data_audit_v1_20260804/receipt.json` | `9542a393aa58176357730cb6c565241ba6696c160ad6b336d6df748c7ac8acdb` | SUA→pseudo pooling exactness |
| `results/t4_paired_view_c1_fresh_v3r2_remote_gate_allowlist/finalization/aggregate.json` | `32ebde0b145c63c09c1bdbc67e48582db3e2ad588e70cb5a1d52914352607e31` | C1 12-cell paired-view aggregate |
| `results/t4_paired_view_c1_fresh_v3r2_remote_gate_allowlist/finalization/publication_readiness.json` | `fcd1e45702640178efd0c351f435406916f94bc51fa0da294480f2a5c3d31bfd` | C1 gate/claim readiness |
| `docs/SUA_STEP2A_INDEPENDENT_RESULT_AUDIT.md` | `b1db9db300fdc50d4b501a251903e4045093aaf2a8d8cb0e33c503076ec38644` | cross-budget Step2A source/dev transfer audit |
| `results/t4_estimator_b_v7_source_only_cpu_audit_v1/aggregate.json` | `2909f9f7658cca4fc8d5edeb5eacbd7cb4224bc744e8d97d61d8556b1bdfbcb1` | Experiment B terminal no-winner result |
| `results/m1_fixed_k_temporal_prototype_gate_a2_v1/source_gate_a2.json` | `379f3c3b85fe120735802af887d8668fc4d84c3e89cb77acdaabecd485565981` | P20 vs matched B20 terminal control |
| `docs/M1_FIXED_K_TEMPORAL_PROTOTYPE_GATE_A2_RESULT_AUDIT.md` | `2ec9c564865a164b41f281f3c5ce01c94d9094184d3b28a415286679f1c19d4d` | fixed-K interpretation boundary |
| `docs/PATH12_TERMINAL_GATE_AUDIT_20260804.md` | `cbb660fdfe1cb083f6b30c4fb86395ace6e475434b9c78dbd1eda84e2c9e061a` | estimator/fixed-K independent closure |
| `results/m2_native_post33_phase_b_v3_scorefree_20260804/phase_b_scorefree_receipt.json` | `130af6fa2829122ed77127bf01571e0507520a15bec1a524a1c7df138bcdb39e` | Native-M2 fresh Phase-C score-free protocol ancestry |
| `docs/M2_NATIVE_POST33_PHASE_C_ROOT_REVIEW_ADDENDUM_20260804.md` | `2e3e3165adabbcdb036ddd98887c91ef5afbff9d41135b424f86d8517c54c3fc` | exact 42-cell/matched-decoder requirements |
| `results/t4_paired_view_c1_shared_zero4_terminal_eval_direct_recovery_v1_20260805/aggregate/c1_three_arm_terminal.json` | `e945ac11a2d60aee5fb2ebfc8df1129e9ddb8fd5ca5c7039bb6970652b5b3ac2` | C1 shared T4/zero4/TS4 terminal three-arm aggregate |
| `docs/DANDI_000688_SUBM_CO_THREE_ARM_V9_ACTIVATION_REVIEW_20260805.md` | `48d3c3d91c8342bc529bb7daf858cba821149be7ffc1ad42b6e22b7d6cf0f372` | external sub-M V9 frozen scope |
| `docs/DANDI_000688_SUBM_CO_THREE_ARM_V9_MINIMAL_IMPLEMENTATION_BLUEPRINT_20260805.md` | `fe8e0b86127ca7b1fcb44897ed9c5355d0874b0e849acea9d4ea693f45f39f31` | external sub-M V9 implementation boundary |
| `results/dandi_000688_subm_co_three_arm_v9_formal_20260805/aggregate/endpoint_aggregate_torchmetrics151.json` | `8a5ba373169cc28237666917f21fc893003afcc9ca486aa8f4b78d3f65aca7f4` | external sub-M V9 270-cell local TorchMetrics 1.5.1 authoritative endpoint |
| `results/dandi_000688_subm_f0_pv_ridge_v1_full_20260805/aggregate/endpoint_aggregate_torchmetrics151.json` | `ffccd91fc128edb7ad6199671f2e32d0c6c450cdff4f2b3d734a1815b176ebc0` | external sub-M F0/PV50/Ridge50 150-cell local TorchMetrics 1.5.1 endpoint |

SHA 是本审计时刻的当前工作区内容哈希；若权威文档后续被维护，应重新计算，不能把旧 SHA 继续当作当前文件身份。

---

## 10. Publication-ready one-paragraph conclusion

现有证据支持一种窄而有价值的结论：T4 是一个在 source 阶段由网络学习如何消费、在新 session 上利用少量方向标签作解析拟合且无需 target-session backpropagation 的 functional carrier。它在 reused held-out SUA development sessions 上有强且一致的增益，主要信息可归因于正确附着的一阶谐波 `[a,c]`；C1 shared 三臂对照给出 SUA `T4-zero4=+0.263335`、pseudo-MUA `+0.321090`。更重要的是，external subject sub-M V9 在 15 个 session 上复现并增强了该系统级信号：SUA `T4-zero4=+0.414594`、`T4-TS4=+0.472147`，pseudo-MUA 分别为 `+0.392951/+0.470387`，四项均 3/3 seeds、15/15 sessions 正且 bootstrap 下界为正。传统对照显示 T4 明显胜过 PV50（`+0.241454/+0.201854`），但平均低于使用逐 bin 连续速度监督的 Ridge50（`-0.061094/-0.104120`），明确给出了标签效率与高监督直接读出的边界。该 carrier 对确定性的 SUA→electrode pooling 具有精确合并律，一套 shared weights 也能在 SUA 与 pseudo-MUA 两种 granularity 下工作；新结果把这一证据扩到同一 DANDI 的外部动物，但不是 independent-lab/dataset replication。organizer-hidden native M2 显示 submitted T4 system 的正向部署效用；现在 matched-decoder Stage A 也给出 `+0.089796` 且 7/7 sessions 正，显著削弱了“official gain 只是 decoder epoch 不同”的解释。full seeds43/44 稳定性 gate 仍未闭合，native M1 则是明确的 negative boundary。cross-budget correction 与 fixed-K temporal carrier 均未通过进入 held-out behavior GPU test 的先验门；方法收口前的核心缺口现只剩 matched Native-M2 的 full three-seed confirmation。
