# 两条主线长程执行工单：M2 / M1 / H1

日期：2026-09-05  
作者/研究审核：Astra；执行：独立 execution coordinator 及其 worker。  
状态：**BOUNDED_EXECUTION_WORKORDER__CONDITIONAL_AUTO_ADVANCE__NO_JOBS_LAUNCHED_BY_AUTHOR**。  
设备：两张独立 RTX3090 24GB，主机内存64GB。  
首个执行周期：最多7个自然日、72 GPU-device-hours；先到者终止新增作业准入。

**2026-09-05提交优先级补充（R2）：** 用户随后要求今天8小时内安排6个新候选、分配M2×2/M1×2/H1×2，并确认H1 C2＋随机M3 activity等已提交。见 [今日六席提交补充](ADDENDUM_SIX_EVALAI_SUBMISSION_POINTS_8H_V1_20260905.md)。H1今日两席改为新时间Transformer FLAT/ROUTE，撤销FF/RF重训重投；本文 §6 暂保留定义但取消自动执行，避免另一agent重新开启同类路线。仅补充列出的产品训练、早期epoch快照与正式提交覆盖本文“暂不提交/all-source”的限制；P正确性门及其余长程科学权威不变。今日队列优先且成本计入本总预算。

## 0. 授权解释：有边界的自主推进，不再每格请示

用户要求把 M1/H1 训练纳入长程任务，并预先决定什么情况自行继续、什么情况找 Astra。
本工单交接后，执行者可以完成本文列出的实现、正确性修复、有限 profiling、正式配对训练、明确列出的条件复制及本地评分。
**作者此轮只写工单，不启动上述作业。**

本文是新的前瞻性任务授权，不改写历史决定：

- `REVIEW_TWO_MAINLINES_M2_SMALL_AND_M1_P_PREFLIGHT_20260905.md` 中对现有 P preflight 的拒绝仍然有效。
- 新的 P 正式12epoch权限，仅在本文 §5 的全部修复门有真实证据后自动生效；不需要再问一次“可以训练吗”。不允许原有错误实现绕过该门。
- 原 M2 S1/v1、S1/v2 和 N 格的选择规则、结果、终端不变。本工单不把另一格追认为旧主候选。
- H1 已关闭的 FiLM 内容、C3、q3-AFC4、top-K/D-opt、加宽接口等路线不重开；本文 H1 新训练格有独立名字和有限问题。
- 只授权本地研究；不自动提交 EvalAI、不上传镜像、不耗提交额度、不打开 hidden eval/test、不替换产品文件。
- Full permission 指系统操作不反复询问，不是允许修改科学比较、数据分割和预算。不得调用交互式权限申请或在脚本内等待人工输入。

**默认沟通方式：正常推进写进度文件，全部队列终结后集中汇报；只有 §11 红色事件需要中途发审查包。普通 null 不属于必须找 Astra 的事件。**

## 1. 研究目标必须一直保持为两条

### D：decoder 的创新设计探索

研究神经单元集合如何组织成可迁移的功能表示，再进行时间解码。强参照仍是 SPINT-like 系统；新架构应保留 activity signature 与 tuning profile 的全部合法信息。

压缩、cosine、EMA、epoch-pick 是可信比较的训练条件，不单独算网络创新。本周期只加入一个明确结构变量，拒绝再开无界 LR/seed 搜索。

### C：T4 carrier / tuning profile 的跨数据集解释与性能提升

统一的是作用关系：

`activity signature A = Phi(X_support)`

`tuning profile T = C_task(X_support, Y_support)`

`prediction = D(query neural history, A, T)`。

不要求 M2、688、M1、H1 使用相同的四个生理坐标，不把 backward H-C 错叫逐单元 forward tuning，不把 FiLM 的容量收益错叫 profile 内容收益。

本周期必须同时交付：

1. 跨数据集估计器/消费者差异的证据化解释；
2. M1 的可微短校准估计器配对训练；
3. H1 保留已完成的匹配M3校准证据，新的训练优先投入时间decoder及校准路由；
4. 新 decoder 在 M2 和 H1/M1 的明确训练/迁移记录。

“解释讲通了”不能替代训练；“跑出一个正数”不能替代同面比较。不能以某个实验格失败为由关闭整条主线。

解释任务也有固定问题，不再展开新的CPU搜索：复用已完成E1/E2/coverage收据，回答“估计的是什么”“哪些信息可能未由activity提供”“估计可靠但consumer为什么可能不用”“预算/窗口/parent与选点错配在哪里”。在平方损失下，增加条件变量可降低Bayes最优风险，但有限模型和有限训练不保证非劣；这是经典条件期望性质，不是本项目的新定理。不能用它代替P或H1的实测。

## 2. 当前事实与新的假设分开

| 事实/限制 | 对本周期的影响 |
|---|---|
| M2 小 S1/EMA source-pick 两seed未过；endpoint24为0.449450/0.337340 | 不追逐0.45，不追加原网格；新结构从匹配初始化做新对照 |
| cosine−legacy 的方向在42/43之间反转 | 不把cosine当已证明迁移增益；固定一个共同训练制度 |
| M1 P factory共享可训练对象、profile处于eval、resume未真恢复 | 先修复和实测，旧100step不能当正式两臂初值 |
| M1 rSyn3稳定，但fold0 S-Fix低于Z-Fix | P-CA胜P-FIX仅证明改进carrier系统，未必证明标签净增量 |
| H1 activity重采样主要增益来自既有LP实验 | EP上的迁移尚是新问题，不能把LP的+0.021744直接贴到EP |
| H1 EMPTY与真实profile近似；M2最终内容对照也不支持真实profile优势 | 本周期不救FiLM语义；可保留既有静态adapter为完整参照的一部分 |
| H1 H-C受其他通道组成影响 | 解释其backward含义，不直接用E1干预当新训练增强 |

本工单提出的 D-ROUTE 是待检验假设，不声称已证明新颖性。集合注意力/latent-slot压缩已有成熟先例，见 [Set Transformer](https://proceedings.mlr.press/v97/lee19d.html) 和 [Perceiver](https://proceedings.mlr.press/v139/jaegle21a.html)。潜在研究价值必须来自校准驱动的功能组织、配对收益与迁移证据，不能来自更换模块名字。

## 3. 数据与评测契约

| 数据集 | 固定校准/输出 | 本地研究面 | 禁止替代 |
|---|---|---|---|
| M2 | M33；MOVE-T4；完整绑定的E0/已有静态head；输出2维 | 7 source；既有M33-disjoint ext-4；共2069窗 | 历史6-session分数、官方HO不能代替clean ext-4 |
| M1 | M10；rSyn3 `[w1,w2,w3,b]`；完整16维signed EMG | 首个outer为20120924；source仅26/27/28；支持后query | later-day M4/6不是M10；all-source模型不是clean fold模型 |
| H1 | first-M3；H-C；176通道，7输出；W=700 | 既有5日期LODO，按相同manifest评分 | 训练日期minival约0.97、M3拟合面均不是outer部署证据 |

H1日期次序固定为 `19250108, 19250113, 19250115, 19250119, 19250120`。
M2唯一历史clean参照均值为 `0.3582396424175502`，必须同时绑定逐session分数/窗口ID；不能只保存一个常数。

Stage A 的 `dataset_contracts.json` 必须逐数据集绑定：

- 实际文件allowlist、parent/teacher/encoder/normalizer字节SHA和排除谱系；
- activity与label支持ID、候选池、query整段历史边界、eval mask、padding与时间单位；
- 训练、source选择、outer报告三张window-ID表；
- 指标权重、输出缩放、训练stride和实际每epoch样本/更新数。

这些是从指定权威工单/代码提取的绑定工作，不授权执行者选择更好看的日期、窗口、normalizer或指标。缺失且无法唯一重建时，按 §11 报契约缺口。

所有query训练loss仍可用合法source标签；禁止只给候选多时刻监督、更多query窗或更多校准标签。source选择文件如果被parent/representation历史接触，明确称source-development，不称全程未见日期验证。

H1 continual查询不提供合法trial完成事件；本周期始终静态M3，不做query记忆增长、不猜边界。H1流式历史不能在eval-mask间隙或推断trial边界重置。

## 4. 统一训练、epoch-pick与比较纪律

### 4.1 选点

**允许epoch-pick，并且正式表必须有选点后的数字。** 合法使用公开数据与是否构成独立泛化证据是两回事。

每个新臂同时保存三个视图，不得互相顶替：

1. **Governing source-pick**：按冻结source-development面与原任务指标，在固定候选epoch集合中取最高；差≤1e-10取早epoch。选点JSON先封存，再评该checkpoint的outer数据。
2. **固定端点与轨迹诊断**：P/H1校准为epoch12，M2新decoder为epoch24，跨任务新decoder为epoch12。报last4/last8均值、总体std、逐session变化；分数均值不是权重平均或可部署模型。
3. **Visible-development product pick**：可对同一候选epoch集合做公开outer选点，所有臂享有相同权利。单独列“已使用该面选点”；不能用其触发原source-pick主张或称独立确认。

source-pick与visible-pick分裂时标 `SELECTION_TRANSFER_GAP`，按冻结队列继续，不临时删除难session、改median或用endpoint替换主表。

若某数据集确实没有可绑定的source选择集合，不能运行后临时用outer顶上；先按 §11 报契约问题。允许已见source日期上的预先隔离query-trial集合，并披露parent历史曝光；不要求凭空存在新日期。

### 4.2 对所有新配对臂的约束

- 每臂独立进程或fresh对象；共同基础权重从同一只读模板复制，新增参数独立RNG域。必须检查真实step0状态，不只比较构造函数里的克隆。
- session-pure shuffled batch manifest；同seed内query、unit dropout及activity子集清单配对。不得在worker中用全局NumPy RNG隐式抽样。
- effective batch32；尾batch按实际样本数计loss。可microbatch/梯度累积，不改变有效batch、loss权重、optimizer更新次数。
- source训练支持与query完整输入历史分离；单个样本不能通过历史窗口读到其被排除的support/query目标。
- 每epoch完整checkpoint：model、basis、normalizer实际数组、optimizer、scheduler、EMA（有则存）、CPU/CUDA RNG、sampler cursor、global step。保存全部epoch，不删除不利点。
- source选择按各epoch**完整候选集合**预写；禁止某臂只评挑出的几个epoch。若计算全部outer轨迹成本高，先完成selected/endpoint治理评分，再按相同规则补齐两臂的visible与轨迹诊断，不能只补候选的好点。
- 配对run可顺序执行；无需为了“同时启动”空耗GPU。基于不可变manifest与RNG配对，而不是进程时钟。
- 不以train loss下降自动续训。本周期没有未经下文命名的12→24/24→48续训权。

所有均值门槛只是计算资源路由，不是显著性、MDE或非劣性证明。日期/session是报告单位，不能把窗口、通道、epoch或seed当独立生物样本膨胀样本量。

## 5. C-M1：修复后正式 P-FIX / P-CA，并自动做一次复制

### 5.1 固定科学定义

保留已命名的 S-Fix parent `epoch_011.pt`，SHA：
`7976e0b064fc4d92396b38a8e385aaa78379f0330c52ba7244bb45831b72178a`。

保留 `row_normalized_nnmf_nnls_v1`、D∈R^(3×16)、ReLU+行L2、active-set NNLS、float64 ridge `(X'X/n + diag[0,1,1,1])`、不罚截距。保持4列接口、完整16维输出及共同可训consumer；不加入E2 lag、FiLM、rank变化或输出硬投影。

### 5.2 自动正式训练前的硬门

以下全部为可执行断言，由实现者之外的shared/audit worker复核，不需要Astra再逐项口头批准：

1. 修复global `_PAIR`：两臂model/basis/optimizer无可写共享。先更新A，未启动B的权重hash保持不变；两真实step0共同参数完全相同。
2. 显式train/eval模式切换；记录训练期实际dropout、requires_grad和AdamW decay分组。合法训练RNG真正设置，不只是写进spec。
3. 真正磁盘save/load与下一步恢复：非空AdamW状态、真实sampler、CPU/CUDA RNG、固定normalizer数组。恢复后下一batch、mask、LR、loss、梯度与更新一致；误差合同提前冻结。
4. 独立旧字典/逐unit carrier/同输入预测parity。不能继续用“新raw+旧均值”冒充旧carrier参考。找不回独立权威时停止P分支；不能放宽成“两个新臂一样就行”。
5. 真实source neural/EMG query loss反传到D；真实window-ID证明支持/query分离；target前后learned-state hash不变且无optimizer/backward。
6. source、source-pick、outer窗口表完整；teacher和新basis均排除20120924。source内部选择历史曝光如实记载。
7. 新根20 warmup + 100个**配对有效更新**的disposable profile，轮转覆盖全部3 source session；计时可靠同步CUDA；正式前重置全部状态。

测试必须能在旧错误实现上失败；不得用固定返回 `true` 作为证据。旧 `20260905_123100` 保留只读；旧profile不是正式warm-start。

### 5.3 自动执行格

| ID | 内容 | 启动条件 |
|---|---|---|
| C-M1-42 | P-FIX/P-CA，fold0，seed42，各12epoch | §5.2全过；整对预计≤12 GPU-h；总/分支预算够 |
| C-M1-43 | 同fold、fresh paired seed43，各12epoch | 42的source-picked P-CA−P-FIX≥+0.005，且P-CA不低同面parent超过0.010 |

训练制度继承已冻结P：AdamW1e-4、wd1e-2（bias/norm排除）、clip1、1epoch warmup后恒定、FP32 consumer / float64估计器、不加EMA。seed43复用同一个合法parent是优化seed复制，不是独立parent-pretraining复制。

独立旧carrier参考的查找限于本地仓库/已有备份清单，不自动下载替代parent或重拟一个新D0后继续同名。如果最后只能提出数值重物化的新锚点，这属于红色的named scientific revision，不是normalizer测试放宽。

P-FIX的固定D/support/normalizer允许缓存z和carrier；P-CA仍随当前D重算，不能跨update使用stale embedding。允许有parity的active-set分组求解；不允许改NNLS目标来省时间。

42为null：封存这对，自动转H1/decoder队列，不找新rank/lag或补seed。43不复制：报告seed不稳定，结束该格，不开44救结果。

同时评分S-Fix及可用的同面Z-Fix，但分别列parent context与活动-only context。P-CA超过P-FIX却低Z-Fix，不能写“tuning profile净增量已证实”。

### 5.4 第二个M1 outer的明确边界

下一outer固定 `20120926`，其合法sources应为24/27/28。
本周期**只在本地已有、且全链证实排除该outer的强carrier parent/teacher字节可用时**，允许正向复制后的同配方seed42配对12epoch；不得用all-source或fold0 parent顶替。

若缺失，这一项记 `BLOCKED_CLEAN_PARENT`，不阻塞其他队列；给Astra列出重建该teacher+S-Fix的真实成本和准确recipe。全新parent预训练不自动挤进72 GPU-h或被叫作“normalizer重算”。这是明确的下一周期依赖，不是本周期必须伪造完成的格。

## 6. C-H1原FF/RF方案：保留解释，撤销自动重跑

**R2执行状态：USER_CONFIRMED_PRIOR_SUBMISSION__NO_AUTOMATIC_RERUN。** 用户确认C2＋随机M3 activity等路线已经提交；不再以换parent/重新命名为由自动重开本节FF/RF训练、复制或提交。以下仅保留原设计供对照解释，所有 §6.3 自动分支失效。H1新的GPU任务按 §7 和今日补充的Transformer FLAT/ROUTE执行；如将来发现具体的未测科学问题，再另立工单，而不是本次重新查记录要求用户证明。

### 6.1 问题与边界

新问题是：**在early-pool、相同固定H-C、匹配M3训练下，随机activity支持能否改善整个双校准系统？**
既有正结果主要来自late-pool；本格明确不把它当成EP已成功。它属于校准系统性能优化，不是H-C新估计器，也不能独自证明T4内容。

使用每fold既有clean C1 early-pool parent；不取LP权重换算成EP，不新加FiLM/EMPTY头，不加入MAT7。decoder/body冻结，仅原identity分支可训；carrier固定仅指其输入与估计器固定，post_pool消费carrier的权重仍按identity分支训练。

### 6.2 唯一配对

| 臂 | Source activity | Source H-C | Outer部署 |
|---|---|---|---|
| C-H1-FF | 固定canonical first3 | 固定canonical first3 | first3 activity + first3 H-C |
| C-H1-RF | 在canonical first7中每样本均匀无放回选3、排序输入 | 与FF完全相同的first3 H-C | 同FF |

所有source query历史须在first7候选支持之后，两臂共用同一query面。first7在source可见不意味着target可取7条；target仍严格只有3。若与历史C1训练面不同，历史性能只作context，新FF才是采样效应anchor。

训练12完整eligible-window passes、stride4、batch32、Adam5e-5、wd0、FP32、native last-bin MSE。两臂均fixed-M3，不再混prefix-cycle或carrier随机重拟。模型尺度/输出`/20`/mask法则继承实际H1实现并写进contract；不硬拷M2的`/5`。

实际source候选池不足7或clean fold parent缺失：不能变成自选M5/M4或all-source替代；按契约缺口处理。训练支持的随机性必须通过subset频数与不同step索引证明，不接受no-op。

### 6.3 R2处置

原先的前2日期筛选、余下3日期训练及seed43复制权限全部撤销，不进入job ledger待执行清单。此前的预算不是允许再做20个同类identity-branch作业。H1的训练名额改投 §7 的decoder结构对照；本节仅说明FF/RF估计量，不再构成启动命令。

输出至少包含：FF−C1（预算匹配微调）、RF−FF（activity采样制度）、RF−C1（系统效应）。RF与FF训练中总activity覆盖不同，不能把全部差值称作纯正则化效应。

H-C全程保持；本周期后置完整label/null训练消融。如果RF正，只支持“更好的activity估计与既有task profile兼容”，不支持“新profile信息被证明”。

## 7. D：一个明确的新decoder结构变量，而不是再找一组LR

### 7.1 D-FLAT / D-ROUTE

两臂都使用小B的完整新训练consumer：per-unit causal Conv(1→16,k5)、token MLP、8个256维slot、4层256维causal Transformer（8heads，FFN512）、256→128→原任务输出。不是只训几千参数的head。

保留所有合法E0及四列T输入。当前E0可能已经融合T；不能称它为纯activity-only表示。

- **D-FLAT**：复用当前small Transformer的动态slot-to-unit注意力。
- **D-ROUTE**：在同一注意力logit上增加一个由静态校准决定、跨query时刻共享的functional-routing项；动态token、value、时间主干和readout不变。

对head h、slot k、unit i、时刻t：

`v_i = LayerNorm_without_affine(concat(E0_i,T_i))`

`p_hi = Linear_h(v_i), p_hi ∈ R^32`

`r_hki = <q_cal_hk,p_hi>/sqrt(32)`

`logit_hkti = logit_flat_hkti + tanh(g_h) * r_hki`。

`q_cal`为每head每slot的32维source-trained参数，`g_h=0`初始化；其余新增参数独立域初始化。保留原unit mask，softmax仍沿unit轴。校准不能读取query标签、预测或未来activity。

零g时必须回到同实例D-FLAT数值合同；至少两次真实更新验证g以及随后q_cal/p收到梯度。零门不是训后非劣保证。共同基础权重应先初始化D-FLAT模板再拷入D-ROUTE，不能因新增参数改变排序而扰动其余权重RNG。

假设：把慢变的功能身份明确放到集合路由上，可能降低全部身份/瞬时活动混合后反复重学对应的负担。**这只是参数化假设**；现有concat网络也可能表达同类函数。首轮收益不能排除额外参数或优化偏置，机制对照后置。

它不属于PV，不把Mamba改名，也不是对identity做FiLM。总active/stored/trainable参数必须实测，预期仍在小B同量级，不能写成已知精确数。最终新颖性主张要另做最近邻文献与必要对照核验。

### 7.2 共同训练制度

- 每dataset一个独立模型，不共享跨任务checkpoint。
- 新decoder从头训，原calibration encoder/normalizer/T估计器冻结。固定calibration law，不同时引入C分支的新carrier或随机activity。
- AdamW peak1e-4、wd1e-2（bias/norm及已声明特殊参数排除）、clip1、effective batch32。
- 1完整epoch warmup，之后恒定LR到该格既定终点；EMA decay0.9995为governing view，RAW为诊断。选定此制度是限制自由度，不声称它普适最优。
- 配对unit dropout0.10；同一window整段共用mask；E0/T/token同步mask。FP32起步，若profile表明只能BF16才满足预算，允许在正式前共同冻结BF16并通过source-only数值对照，之后不得按分数切换。
- M2各24epoch；H1/M1迁移各12epoch。不允许看到train下降自动加倍。

### 7.3 M2与跨任务作业表

| ID | 正式作业 | 先决条件 |
|---|---|---|
| D-M2-42 | FLAT/ROUTE，各24epoch | 新结构正确性、资源profile通过 |
| D-M2-43 | 同配对，各24epoch | 42正常完成；无论42正负都执行一次，避免单seed决定结构命运 |
| D-H1-42-A | 前两个固定LODO日期，各FLAT/ROUTE12epoch | H1适配器/clean parents通过；**不要求M2过性能门** |
| D-M1-42 | fold0，各FLAT/ROUTE12epoch | M1数据/parent/冻结calibration合同通过；**不要求P-CA或M2过性能门** |
| D-H1-42-B | 其余3日期同配对 | 前2日期ROUTE−FLAT平均≥+0.003、worst≥−0.015 |
| D-H1-43 | 完整5日期同配对 | 42完整5日期达到下文结构门与产品安全门 |
| D-M1-43 | fold0 fresh配对seed43 | 42达到结构门且ROUTE不低强parent超过0.010 |

先兑现M2双seed、H1首2日期、M1首fold三个基本问题，再开额外复制。不能让M2所有后继占满预算，使H1/M1永远只是计划。三项基本问题的成本应在正式前整体测算；超预算则先汇报资源取舍，不能静默取消某数据集。

**结构门**：source-picked ROUTE−FLAT平均≥+0.005，逐报告单元worst≥−0.020；多日期时至少半数严格为正。M1一个outer只具pilot资格。

**产品门**：相对完整同面强SPINT-like parent平均≥+0.005；M2≥3/4session非负且worst≥−0.050；完整H1≥4/5日期非负且worst≥−0.010；M1单outer不能确证跨日期安全性。

结构门过但产品门不过，记 `STRUCTURAL_SIGNAL_BELOW_STRONG_REFERENCE`，不宣称性能创新已成立。M2两seed结论分裂则保留全部，不能用平均掩盖；无第三seed自动救援。

### 7.4 跨任务实现不允许偷换的问题

- M2当前token硬编码identity50/T4四列、输出2、PE max_len256。H1/M1必须改为由contract读取真实E0维度、输出7/16及真实history长度；H1 W700需扩PE容量。不是截到256以免报错。
- 保留每任务原始采样间隔和完整历史；不把H1的700bin硬缩成M2的50bin，不增加对照没有的未来context。
- H1 N176和长历史可能主要占用set-token激活内存。允许microbatch、activation checkpointing、固定时间tile精确分块set frontend；不可downsample/改slot数冒充同结构实现。
- 时间tile只能切分逐时刻独立的set计算；共享causal Conv必须保留kernel左侧overlap，时间Transformer不能随tile重置。用完整窗口前向/梯度做parity再启用节省显存路径。
- 重叠滑窗训练和官方逐bin运行不是天然同一个cache算子。禁止跨滑窗复用不满足截窗/PE法则的Transformer KV cache；先做与逐窗reference的一致性。部署速度不足时记工程限制，不改变评分协议。
- H1 backward H-C仍可能依赖整体roster。给定已经一起置换的X/E0/T，decoder集合不变性应通过；不能据此声称整个PCA/H-C materializer对任意roster不变。
- 不把部署静态carrier换成query oracle；M1仍预测16维signed EMG，不能只给协同坐标打分。

### 7.5 SSM保留为有条件的有限后继

不是宣布SSM无效，也不再自动重开旧Mamba。仅当D-ROUTE在M2两seed都过结构门、且至少一个跨任务完整比较有正结构信号，同时D剩余预算≥6 GPU-h时，允许一格 `D-ROUTE-MAMBA-M2-42`：

- 同ROUTE frontend/输入/训练manifest，时间主干换为4层Mamba2，d_model256、d_state64、expand2、headdim64、ngroups1、d_conv4、chunk_size64；24epoch，当前共同训练制度；
- 从头初始化，参考dynamics初始化及no-weight-decay规则保留；不是从Transformer权重续训；
- 与本周期D-ROUTE-Transformer42比较，报告参数/计算差异；无进一步宽度/state sweep、无自动seed43或48epoch；
- 仅复用已验证的独立环境。kernel/reference/step数值或环境不通，在2小时工程限额后记blocked，不换成手写DiagSSM继续同名。

这是时间模型后继，不能倒置为主线最先耗时的任务。PV始终不进入本工单。

## 8. 两条线什么时候可以合并

本周期默认分别固定另一条轴：C使用原强decoder，D使用原固定calibration。

只有两条线在同一数据集各自完成指定复制，才形成下一周期 `C×D` 四格建议：原/新carrier制度 × 原/新decoder。**不自动运行该组合**，以免两个独立小增益拼接后无法归因。

完整内容消融、matched-capacity routing-null、task-label shuffle、SUA/pMUA、长历史扫描均后置。正数首先叫候选/配对性能信号；没有这些控制不把第二创新点视为已经完成。

没有dense-label FiLM后继，没有DANDI688 GPU训练。688仅用于已有forward-T4/SUA-pMUA证据整理；不改其历史结果。

## 9. 双GPU调度与资源预算

### 9.1 硬预算

- **D线32 GPU-h；C线32 GPU-h；正确性profile/评分/本地artifact整理8 GPU-h，共72。** 预算按每进程使用的device wall time保守累加；同卡两个进程重叠也分别计。同时另报按GPU时间区间并集计算的实际device占用，两者不得混为同一个硬件利用率数字。
- 执行coordinator可在D/C之间转移最多4 GPU-h一次，写ledger，且不取消另一线基本问题；总额不变。更大转移需要Astra裁决。
- 预留最近profile估计的全队列评分时间再加20%余量。没有完整配对训练及其评分预算，不准启动其第一臂。
- 若72GPU-h或7日边界到达：不再开新格；已有作业按已预留预算结束或安全checkpoint，最后报告 `RESOURCE_CAPPED`。不能把未训练的格写成null或完成。
- 两GPU不是48GB共享显存。不用DDP。读取lease/PID确认没有用户/队友作业，禁止清理不属于本队列的进程。

### 9.2 队列原则

GPU-A优先D-M2；GPU-B优先修复后C-M1。H1的CPU物化与loader适配同时进行。

正式首轮前必须完成D-H1真实W700/N176及D-M1真实形状的成本探针，并为D-H1首2日期、D-M1首fold预留整对预算；C-H1的FF/RF重复训练不再占用预留。今日H1 all-source产品对按补充另计、不能填入LODO格。不能只profile便宜的M2然后才发现H1无法运行；profile阶段不读取outer成绩。

第一张卡释放后优先给尚未开始的H1配对，不得继续排M2“再试一个seed”。D-M1在C-M1真实数据合同通过后复用只读数据，但不使用P-CA训练后的可变carrier。

单卡可以并行两臂，但先做有限探针：合计峰值显存≤20GiB、MemAvailable≥12GiB，较顺序执行aggregate updates/s提升≥15%才采用。否则顺序或分卡；配对正确性不依赖同卡常驻。

主机目标总PSS≤44GiB；loader workers初始0，每job最多2；BLAS/Torch线程每job2–4。共享只读memmap，避免复制全部重叠窗口。OOM时先降microbatch/增accumulation或启用已允许的checkpointing，不能削减query面或改变effective batch。

### 9.3 预计时间，而不是保证

| 阶段 | 建议墙钟窗口 | 可并行工作 |
|---|---|---|
| A：authority、P修复、D-ROUTE与H1适配 | 前6–12h | CPU实现/测试；只做有限disposable profiling |
| B：首个正式问题 | 约第1–2天 | D-M2与C-M1分卡，H1准备；两边互不等正数 |
| C：跨任务与有限复制 | 约第2–5天 | H1 decoder、D-M1、已触发的seed复制；不重开H1 FF/RF |
| D：收口 | 最晚第7天或预算先用完 | 补齐已训checkpoint评分、解释表、核验与最终审查包 |

P旧估计约7GPU-h只是粗估；修复后train-mode和固定臂缓存的profile才支配ETA。H1整decoder与identity-branch费用差异很大，不能用FiLM-head速度估整decoder训练。

## 10. 执行subagent分工：1 coordinator + 3 workers

这是交给执行agent的组织方案；Astra作者此轮不启动这些agent，也不承担后台训练。

| 角色 | 独占所有权 | 并行职责 |
|---|---|---|
| Coordinator | 根contracts、job ledger、调度/runner、决策状态机、总报告 | 冻结配置，发GPU lease，验证条件分支，不写另一worker模型代码 |
| D worker | 新D decoder/routing/SSM适配及D单测 | 做M2/H1/M1同结构consumer；不得修改旧B sealed namespace |
| C worker | P修复wrapper、M1校准训练及C单测 | 保持原估计器定义；H1只复用已有校准证据，不重训FF/RF，不自开lag/FiLM |
| Shared/audit worker | 数据适配、训练/评分/resume公共层与其测试 | 独立复核真实step0/数据边界/恢复；运行统一只读监视器 |

建议新namespace：`tfpd_exploration/src/two_mainlines_long_v1/`，子目录 `decoder/`、`calibration/`、`pipeline/`；对应新tests与单一runner。现有canonical算子只读import；确需改其bug，另存版本wrapper或局部副本并加回归测试，不能覆盖旧运行的科学权威。

结果只写 `tfpd_exploration/results/two_mainlines_long_v1/<timestamp>/<cell>/`。共享文件由coordinator合并，worker不越权编辑，不要每个seed开一批新agent。

每个worker任务必须写：

> You are not alone in the codebase. Modify only your owned files. Do not revert other agents' edits. Preserve old result roots and shared contracts. Do not independently claim a GPU or launch an unregistered experiment. Use the inherited full-access, non-interactive environment; do not ask routine permissions.

一个独立监视进程每60s记GPU/进程/吞吐/ETA；每10min更新可读status。agent无需每分钟完整重读日志，也不能把监视器关闭当训练完成。

## 11. 自动推进、局部停止、立即找Astra：确定状态机

### 11.1 绿色：不用问，执行到队列完成

| 事件 | 允许的动作 |
|---|---|
| 绑定完成、实现与真实preflight全过、预算准入成功 | 自动启动本文命名的第一配对，不另求“正式训练批准” |
| 达到本文指定复制/跨fold门槛 | 自动启动指定seed/日期，不增加第三种自由选择 |
| 普通负结果或复制失败 | 封存当前格；取消其有条件后继；继续另一数据集/主线 |
| source-pick与visible分裂 | 两者都报，主判定不变；按既定队列继续 |
| GPU利用率不高 | 在已允许microbatch、worker、只读cache范围优化，并记录变化 |
| 依赖晚到或某张GPU被队友占用 | 让可运行分支先走；不抢卡、不全队列等待 |
| 已通过resume测试的作业遭一次进程/设备中断 | 从完整checkpoint自动恢复一次；保留incident与实际曝光 |
| 未见成绩前的普通实现bug | 最多两次局部修复、总2h；加失败回归测试并重做相关preflight |

不用把“null”“某epoch掉分”“首次OOM”“下一格已经在工单里”发成权限问题。

### 11.2 黄色：关闭受影响格，最终报告，其他队列继续

- 该配方未达到门槛，或预注册候选集合耗尽；
- 缺第二个M1 clean parent但首fold可用；
- 未触发的SSM、seed43、后3日期或all-source阶段；
- 小型工程问题两次修复仍未解决，且与其他分支隔离；
- 某optional作业没有剩余预算。

用明确状态：`COMPLETE_NEGATIVE`、`REPLICATION_FAILED`、`NOT_TRIGGERED`、`BLOCKED_LOCAL_DEPENDENCY`、`RESOURCE_CAPPED`。不把它们统称NO-GO，不宣称整个任务成功。

### 11.3 红色：立即停止受影响分支并给Astra中期审查包

1. 发现forbidden/hidden数据访问、target query标签拟合、parent outer泄漏或sealed文件被改。
2. 正式结果已出现后发现两臂共享状态、不同样本/预算、单位/mask/缩放/选点错误，结果不能比较。不得修好后覆盖同根继续当原实验。
3. 必须改变科学定义才能推进：basis/ridge目标、支持容量/窗口、输出空间、parent谱系、主指标、epoch预算、结构/训练allowlist。
4. P独立parent-carrier parity不成立，或恢复/因果性/真实数据断言需放宽容差才能“过”。
5. 三个基本跨任务decoder问题无法在预算内入场，或需要超过总预算/跨线转移上限；不能偷偷取消H1/M1。
6. 共享pipeline/数据污染可能影响两条线，或重复非有限梯度、异常磁盘写入、内存压力威胁队友任务。

红色事件只暂停其依赖闭包；无依赖的队列继续。只有共享数据完整性、安全或全局预算问题才暂停全队列。

报告发给Astra/用户的内容必须是具体证据与所需决定，不是“请给full permission”。若Astra当时不可达，留下包并把受影响job标BLOCKED；其他合法作业照跑，不循环刷请求。

### 11.4 中期审查包最小内容

`INCIDENT_OR_DECISION.md`：问题一句话、影响哪些cell、发现时间、旧根是否完整、已停止PID、能继续哪些任务。

附：相关SHA/可复现失败断言、已经观察的成绩范围、当前累计GPU-h/剩余预算、两个以内解决方案及各自代价、建议。涉及成绩污染时，分清哪些旧分可保留、哪些需要撤回或另立新格。

没有红色事件时，不要求人工看每个中间分数；coordinator可自主执行到全队列终结。

## 12. 正确性基线与一次性技术修订范围

下列检查是强制，不能因“消融后做”跳过：

- 原parent相同输入重放；D零routing还原FLAT；原calibration数据/权重路径无漂移；
- 实际source数据梯度、独立初始化、train/eval、optimizer/EMA更新计数；
- future-input扰动、unit permutation、padding/all-masked fail-closed、输出尺度；
- H1 W700与M1真实W全长测试；滑窗与所声称部署算子一致；
- 真磁盘resume、source/outer隔离、target learned-state不变；
- parameter/VRAM/内存/速度实测，而不是只从模型名称推断。

FP64线性/NNLS参考按原P合同；不同GEMM形状的FP32 forward暂定atol1e-5/rtol1e-4，历史指标重放另按其既有严门，不混用。相同实例/同算子重复必须确定性；若环境不支持bitwise要在看成绩前说明原因并触发契约裁决，而非看分后放宽。

允许正式前最多一次 `TECHNICAL_BINDING_REVISION` 来填实际路径/SHA/设备、精度与microbatch，并修正不改算子的API/shape问题；新hash与全部尝试收据保留。它不是改变本文科学假设或门槛的许可证。后续普通bug修复适用 §11 的限次规则。

## 13. 最终产物与“全部完成”的定义

所有计划格必须有terminal状态；已启动的有效训练必须有对应评分或明确评分阻塞；不留无人监视的后台作业。条件未触发可正常收口，资源限制/阻塞必须显式留下，不能把未完成实验计为完成。

最终包：

1. `HANDOFF_FOR_ASTRA.md`：两条主线分别的结论、最重要的下一步，不能只列最佳数。
2. `comparison.csv`：dataset/fold/seed/arm、parent与manifest SHA、governing epoch、source/outer、endpoint、visible-pick、逐session、参数与GPU-h。
3. `decision_ledger.json`：每个命名格的准入/门槛/状态、未运行原因、预算转移；包含失败尝试。
4. `calibration_generalization.md`：forward/backward、标签类型/预算、activity与profile各自的证据、估计可靠性/consumer可用性/选择迁移三者区别。
5. `decoder_architecture_review.md`：FLAT/ROUTE/强SPINT-like同面比较、跨seed/跨任务、容量与速度、最接近的既有结构；哪些机制仍未证明。
6. `artifact_inventory.json`：可重评分的checkpoint实际字节、normalizer、配置、恢复状态、预测/target/窗口digest。

可以准备本地deployment candidate清单，但**本周期不自动all-source再训、Docker上传或EvalAI**。先把真正通过的跨数据集/配对结果交给Astra，避免只凭开发面选点消耗提交额度。all-source和正式主张消融属于下一次有限任务，不是当前无限尾巴。

最终答复必须明确：

- D线究竟获得结构收益、只有优化收益，还是这一个结构假设失败；
- C线究竟改善了carrier估计、activity支持训练，还是只有解释证据；
- M1/H1分别实际训了什么，而不只给M2数字；
- 哪些正数只是同一外部开发面反复看过的探索证据；
- 是否存在需用户/Astra决定的下一周期事项。

## 14. 权威引用与历史保护

- `docs/REVIEW_TWO_MAINLINES_M2_SMALL_AND_M1_P_PREFLIGHT_20260905.md`：当前审计与两主线边界。
- `docs/WORKORDER_CROSS_DATASET_FUNCTIONAL_CALIBRATION_V1_20260905.md`、`docs/REVISION_CROSS_DATASET_P_OPERATOR_V1_20260905.md`：P估计器科学定义。
- `results/m2_b_small_stability_v1/astra_pack_v1/`：旧M2矩阵、P与H1/M1资料入口；不能覆盖。
- `docs/RESULT_ID_ENCODER_CONTENT_AND_CARRIER_CLOSURE_V1_20260905.md`：FiLM内容结论，优先于旧FABLE表述。
- `h1_series_20260830/docs/RESULT_H1_ACTIVITY_CARRIER_RESAMPLING_FACTORIAL_V1_20260904.md`：LP activity多样性证据，不是EP已证实结果。
- `h1_series_20260830/docs/HANDOFF_H1_SUCCESSOR_AGENT_20260903.md`：H1接口/预算；较晚结果以closure为准。
- `src/m2_b_small_stability_v1/decoder.py`：D-FLAT算子来源，只读复用/局部新版本，避免全局常量monkeypatch。

原cross-dataset Stage0 `20260905_113700/terminal.json` SHA必须仍为：
`ea4c3227ad09d09610a8d78a38dbe400bf03ada0893f6058f35594073b36f875`。

本工单没有创建result root、修改训练代码或启动GPU；它为后续execution agent提供有条件的、可机器记录的本地执行范围。
