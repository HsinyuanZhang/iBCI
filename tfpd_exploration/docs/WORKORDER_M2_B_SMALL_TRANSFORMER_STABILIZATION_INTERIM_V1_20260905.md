# M2 B 中间工单：小型 Transformer + 可归因的训练稳化

日期：2026-09-05  
状态：**READY_FOR_EXECUTOR__TWO_SEED42_RUNS_AFTER_TECHNICAL_PREFLIGHT__NO_JOBS_LAUNCHED_BY_AUTHOR**  
职责：Astra 规划/审核；执行 agent 实现、训练、评分和交付。本文创建本身不运行 GPU、不创建 result root。  
范围：保留历史 A/B；新增两个同规模 Transformer 训练 cell；跨数据集 E/P 线独立继续。不是第三轮 Mamba 搜索，也不是 EvalAI 工单。

## 0. 直接决定

1. 不仅补 last-k 报告：在新 namespace/root 实施一次有界的小模型稳化比较，**两臂、seed42、各24完整 epoch**，通过下述技术检查后执行，无需再申请普通文件/命令权限。
2. 两臂均为 **3,543,010 decoder 参数**：集合宽度256、8 slots不变；时间宽度512→256、FFN1024→512、4层不变。保留现有 activity identity、MOVE-T4、50-bin历史和因果法则。
3. `S0-SMALL-LEGACY` 复用历史 LR 轨迹；`S1-SMALL-COS` 改为1 epoch warmup后持续 cosine。**峰值 LR均保留3e-4**，不同时降至1e-4。
4. 两臂均维护同规则的 EMA shadow，并保存 RAW/EMA 两个权重视图。EMA不参与反传或teacher loss，因此仍然只有两次优化轨迹，而不是四次训练。
5. 主候选预先固定为 `S1-SMALL-COS / EMA`；`S0 / EMA`是调度对照，RAW用于归因/历史对接。不得评分后在四格中任选最大值替代主候选。
6. 旧12/24 epoch、sequential/shuffled Mamba和A记录保持不变。**不能从旧e12/e24续训**这次小模型；不自动开seed43、48 epoch、LR/width sweep、SWA、蒸馏、A+B或新Mamba。
7. 跨数据集线继续；最新named operator revision已接受进入共同normalizer/consumer检查和一次disposable profile，见[独立有限冻结记录](REVIEW_CROSS_DATASET_P_OPERATOR_PREFLIGHT_FREEZE_V1_20260905.md)。P正式12epoch仍需后续冻结。M2结果不构成跨数据集线的科学门槛，M2调度也不自动替换P的训练制度。

选择两个小模型的原因：只运行“小模型+cosine+EMA”一格可探产品，但不能区分缩小网络与稳化配方。本方案用一个同规模训练对照和低成本shadow视图解决主要归因，不扩成完整消融矩阵。

## 1. 证据来源与审核修正

父记录：

- [原双线工单](WORKORDER_CALIBRATION_MEMORY_AND_TEMPORAL_DECODER_PARALLEL_V1_20260905.md)。
- [12→24决策](REVIEW_M2_DUAL_TRACK_SHUFFLED_12EP_AND_24EP_DECISION_20260905.md)。
- 旧根 `tfpd_exploration/results/m2_dual_track_v1/20260905_101500/`。
- 旧根 `ANALYSIS_B_TRAJECTORY_INSTABILITY.md`，审阅SHA256：`af64ee40fa5dc3c0a4036f1d67c2ea48e92866b33d5803d1f48309d2ce780d60`。
- 旧根 `comparison.csv`，审阅SHA256：`94ab3271896a01feb68b090c1896e4a893c92ad46556d803b4b122965977a95a`。
- 两个shuffled根及`seed42_shuffled_e13_24`下的`metrics.jsonl`、`ext4_epoch_scan.json`、交接包。

作者只读重新计算了以下统计；没有重新进行decoder评分、读取新NWB或修改旧receipt。

| 需采用的解释 | 限定 |
|---|---|
| Transformer train MSE与source-minival R²相关接近零 | e1–12为+0.02504，e13–24为+0.06988；e1→e24 loss约降9.14倍。不能推广为两种架构均“完全解耦”。 |
| Mamba首段存在明显学习信号 | train MSE与source-minival R²相关−0.78617，尾段−0.31872；更符合伴随高方差的学习，不能由此证明或否定SSM能力。 |
| Mamba尾段波动减小 | source-minival总体std从0.11095降至0.03694；但训练阶段和LR同时变化，不是随机化的调度因果实验。 |
| 实际优化器是AdamW | 不写“SGD checkpoint”；“平稳分布抽样”“过拟合游走”只能作假设，当前非平稳曲线没有证明这些机制。 |
| A的稳定不能全部归因于LR | A只训练约2.3万残差参数并冻结冠军；B从零优化约860–1040万decoder参数。 |
| session异质性不是振荡的充分解释 | 第一波B的平均两两相关约0.14–0.27；Mamba尾段为−0.00820。A也能低相关而均值稳定。 |
| EMA/SWA缺失是原工单规定 | 不称遗漏实现；加入是新实验制度。额外RAW/EMA评分有成本，不称“零额外评测成本”。 |

**额外重要纠正：不得将Mamba e24的source-minival 0.309186与ext-4 REF 0.358240相减。**
两者不是同一评价面。“稳住后仍差约0.06”的这项计算无效。相同ext-4面上，Mamba e24/source-pick为0.343016，差值是−0.015224。
source-minival只用于source选点/学习诊断；最终迁移比较在固定ext-4面完成。

## 2. 历史基准与last-k补丁

固定 `R_REF_session=0.3582396424175502`，`R_REF_date=0.3279795072`；同M33-disjoint ext-4，共4session/3date，query窗口519/490/425/635，共2069个。重新绑定实际window-ID digest，不用仅计数证明相同。

| 历史B | Source-pick 1–24的ext-4 | Endpoint24的ext-4 | e21–24 mean ± population std | e17–24 mean ± population std |
|---|---:|---:|---:|---:|
| Transformer | e9: 0.360008（+0.001768） | 0.321388 | 0.332195 ± 0.008984 | 0.340155 ± 0.024152 |
| Mamba | e24: 0.343016（−0.015224） | 0.343016 | 0.339657 ± 0.015601 | 0.348886 ± 0.016601 |

可见ext-4选点另列：Transformer e20=0.371954；Mamba e18=0.377224。它们是合法的visible-development候选，不是独立测试，也不改写旧source-picked路由失败。

执行者先将上述派生统计写到**新根**的`historical_trajectory_summary.json/.csv`，绑定输入SHA；无需新训练或重新推理。

- 全部std固定`ddof=0`；mean|Δ|/max|Δ|使用所声明连续区间内的相邻epoch差；范围是min/max，不是CI。
- last-k是**k个模型得分的算术均值**，不是权重平均、预测集成或可部署模型的R²；不得把它作为新的checkpoint结果。
- 时间相关的24个epoch不是24个独立实验；这些std不是seed/session不确定度或显著性检验。
- 合法epoch-pick继续保留；禁止的是只展示偶然端点/峰值的稳定性叙事，不是禁止选点。

## 3. 唯一的小模型结构与参数预算

| 字段 | 历史B-Transformer | 新S0/S1 |
|---|---:|---:|
| 每unit causal conv | 1→16，kernel5 | 相同 |
| token输入 | local16 + E0 50 + T4 4 | 相同70维 |
| 集合token宽度 / slots / heads | 256 / 8 / 8 | 相同 |
| slot投影 | 2048→512 | 2048→256 |
| 时间宽度 / heads / FFN / 层数 | 512 / 8 / 1024 / 4 | 256 / 8 / 512 / 4 |
| 输出头 | 512→128→2 | 256→128→2 |
| decoder参数 | 10,403,554 | **3,543,010** |
| 加共享校准分支19,514 | 10,423,068 | **3,562,524** |

当前SPINT REF有效decoder=3,466,902；有效系统=3,486,416。新decoder约多2.20%，不是严格字节级等容量，也不代表FLOPs或延迟相等。
REF对象闲置的原`fc_id_in/fc_id_out`不计入有效路径；训练teacher、优化器和会话E0/T缓存不是decoder参数。

按现层定义，时间部分每层参数为`4d² + 2df + 9d + f`。新时间stack=2,108,416，其他前端/读出=1,434,594。Stage0用真实`named_parameters()`核对精确值，偏离即阻塞，不靠手算掩盖实现偏差。

不减历史长度、不减slots、不删T4/E0、不改head数、不加新归一化/位置编码/输出滤波。这里检验较窄的同类decoder，不把压缩本身写成创新。

## 4. 两个训练cell、四个已声明权重视图

| Cell | 初始化 / sampler / architecture | LR | 权重视图 |
|---|---|---|---|
| S0-SMALL-LEGACY | 同一个seed42小模型初始化；同24epoch manifest | 原1epoch warmup + 至e12恒定 + e13–24旧cosine尾 | RAW + EMA shadow |
| S1-SMALL-COS | 与S0逐字节相同初始化 | 同warmup，之后按update持续cosine至e24 | RAW + EMA shadow |

主候选唯一为S1/EMA。S0/EMA作为同规模、同平均规则的调度对照；两条RAW是归因诊断。四视图都需保留，即使EMA更差。

### 4.1 必须保持的训练合同

- AdamW，peak LR=3e-4，weight_decay=1e-2；betas=(0.9,0.999)，eps=1e-8；bias/norm排除weight decay沿用实际旧规则；grad clip=1.0。
- decoder从零训练；校准encoder和MOVE-T4相关状态冻结，沿用相同P0/EMPTY head、normalizer和session-bank。无A-QMEM残差、teacher loss或target更新。
- BF16 autocast、FP32参数/AdamW状态和MSE；先FP32 disposable smoke。unit dropout=0.10，任务loss/行为缩放和last-target定义不变；不加Huber、loss重加权或新正则。
- 有效batch32，session-pure shuffled batches；包含尾batch，loss/gradient accumulation按真实样本数计。若microbatch因显存需变，两臂共同冻结相同值；完整query曝光不变。
- 同24epoch sampler manifest：digest `a95255fa339ea06f1e1cc3ef9f53f3d49d15799bf95e4579f672417fd878d79a`；首12与旧digest `0c1808baa4d6722742a3705963c60b55fd7fcc7f3b2d03c2c416e3e0d2937a95`关联。
- 当前每epoch3165次optimizer update，24epoch共75960。以manifest重算并断言；不能把wall-time budget截断或FiLM子采样称完整epoch。
- S0/S1专用dropout域以(seed,epoch,batch ID)确定，相同batch同unit mask；EMA、评估、monitor不消耗训练RNG。旧大模型与新小模型不声称跨形状初始化/归约位精确配对。

### 4.2 LR的唯一公式

设t为1-based **optimizer update**，W为一个epoch的updates，T为24epoch总updates；lr_max=3e-4，lr_min=3e-5。

两臂在`1≤t≤W`使用`lr(t)=lr_max*t/W`，不存在warmup结束的突跳。

S0在epoch2–12使用lr_max；epoch13–24逐epoch使用旧实现：

`lr(e)=lr_max*[0.1+0.45*(1+cos(pi*(e-13)/11))]`。

S1在`t>W`使用：

`p=(t-W)/(T-W)`，`lr(t)=lr_min+0.5*(lr_max-lr_min)*(1+cos(pi*p))`。

无restart、无第二段热启动。每次optimizer.step前设置本步LR。Stage0核对t=1/W/W+1/T，S0核对e12/13/24和整条旧LR序列。暂不检验1e-4；若失败，不现场换LR。

### 4.3 EMA的唯一公式与保存合同

冻结`decay=0.9995`，按**每次成功optimizer update**更新，不按microbatch或epoch更新，不扫描0.999/0.9995。

- 第一次成功step后`theta_EMA=theta_RAW`；以后`theta_EMA=d*theta_EMA+(1-d)*theta_RAW`，FP32、no_grad。
- 这是约1386 updates半衰期、约1999 updates平均年龄；当前相当于0.44/0.63 epoch。不是“最后4epoch平均”，不能保证消除所有跨epoch漂移。
- EMA只跟踪decoder参数；校准分支与非训练buffer保持固定。当前架构为LayerNorm，无BatchNorm重新估计，也不通过target数据更新任何buffer。
- 两臂训练loss、梯度和optimizer始终来自RAW，EMA从不反馈入训练。开启shadow不能改变RAW下一步；须测试。
- 每epoch保存RAW、EMA、AdamW、LR公式/config、成功step数、EMA更新数、RNG、sampler位置、初始化/数据/校准SHA。不得只保存EMA或只保存选中点。
- 若使用PyTorch AveragedModel，明确首次copy语义；其平均函数不随state_dict保存，必须单独序列化decay/算法版本并测试恢复。部署只需要选中的一套权重，不能把RAW+EMA两套存储量称模型参数翻倍。[PyTorch AveragedModel说明](https://docs.pytorch.org/docs/2.9/generated/torch.optim.swa_utils.AveragedModel.html)

## 5. 哪些差值能回答哪些问题

记`R(cell,view,e)`为相同ext-4面、指定epoch的session-equal R²。

1. **主系统候选**：S1/EMA按source规则选点后，减同面REF。它是“小模型+新训练制度”的产品表现，不自动等于机制收益。
2. **调度效应**：同epoch或固定末段，对比S1/EMA与S0/EMA；RAW对RAW也报告。两者只改LR轨迹，包括其与AdamW更新的相互作用。
3. **平均权重效应**：同cell、同epoch，对比EMA−RAW；两条轨迹均可测。不能用两个不同epoch的各自最大值代替同点效果。
4. **缩窄的历史比较**：S0/RAW对旧大Transformer/RAW，按相同epoch/相同1–24选点预算比较。此为单seed、跨模型宽度的历史参照，不声称严格匹配所有初始化/数值路径，也不证明其他宽度都无必要。
5. **稳定性与性能分开**：较小std而平均R²下降为“稳而差”；不能靠平滑后的std宣布泛化提高。

不要求本轮补RAW/EMA以外的完整机制消融。与SPINT的参数接近只能控制规模量级；前端、时间组织、注入方式、预训练谱系仍不同。

## 6. 评分、合法epoch-pick与防止挑结果

### 6.1 数据与执行顺序

训练仅现有7 held-in源session及原训练窗口。source-minival保留原7session定义，不重命名为ext-4。ext-4保持M33后整窗与support不交；目标query标签不进入bank、训练、normalizer或EMA。合法M33 calibration标签仍按既有MOVE-T4协议使用，不能把它们与query标签混为一谈。

每epoch对RAW/EMA分别做source-minival，保存native MSE、session-equal/per-session R²。两次评分有实际成本；与历史相同精度、行为缩放和metric。

**两条24epoch都结束、全部候选和source选择manifest封存后，再统一跑ext-4扫描。** 训练/调度/停止不读取本次ext-4结果。一个cell工程失败时记录INCOMPLETE，不能根据另一个外部成绩补配方。

### 6.2 固定的选择与展示

- 主候选：仅S1/EMA的epoch1–24，以source-minival session-equal R²最大选点，ext-4仅报告其迁移表现。
- S0/EMA、S0/RAW、S1/RAW各按同规则保存source-picked诊断；这些不参与主候选视图切换。
- tie统一：先取有限score全局最大m，在`m-score≤1e-10`的候选中选最早epoch；不能使用依赖遍历顺序的两两近似并列。非有限值记录且不候选；全部非有限为失败。
- 另报epoch12、epoch24；e21–24/e17–24两个固定区间的mean、population std、mean|Δ|、max|Δ|和range，附per-session/date。
- 可见ext-4 epoch-pick合法：对两臂**EMA**各扫描同24个候选，单列visible-development结果、全部分数与相同tie法则。不是clean未见测试；“clean”只指support/query不交。
- RAW的ext-4全24评分用于轨迹诊断，不能评分后把RAW峰值升级为本轮EMA产品候选。总计2cell×2view×24=96次候选评分，固定并披露；可复用相同checkpoint评分不重复跑。
- 模型峰值、末段均值、EMA权重结果、预测集成不是同一统计量。禁止以last-k均值推断EMA必然改善。

若96次评分超出下述冻结资源预算，应在**正式训练前**提交资源计划变更并保留更少但对称的固定epoch集合；不得看到分数后删难看epoch或改候选数。

### 6.3 资源路由判定，非显著性承诺

沿用B旧路由：主候选source-picked ext-4 Δ≥+0.005，且无session Δ<−0.05，才记`PERFORMANCE_CANDIDATE`。同时报告3date-equal差值/4session符号，不把门当作统计显著或非劣性证明。

- 主候选过门且末8epoch EMA均值≥REF：可称“具有末段支持的候选”，提出seed43确认，**本工单不自动执行**。
- 主候选过门但末段均值低于REF：记`PICK_SENSITIVE_CANDIDATE`，不能宣布稳定胜出。
- 主候选未过门而S0/EMA更好：报告小模型锚点结果，说明full-cosine没有显示优势；不暗中切换本轮主假设。
- Δ∈[−0.03,+0.005)：完成但未过门；不叫非劣，不自动延长。Δ<−0.03：该小模型配方存在迁移缺口。结构检查失败与科学null分别标注。
- visible-only过门：保留合法产品发展线索，但不改写source-picked路由或立即提交EvalAI。

稳定性诊断固定使用相同e17–24：source和ext-4分别比较RAW/EMA及S0/S1的std与mean|Δ|。可描述比例变化，不设置“比A稳定”等跨训练范围伪等价门。

## 7. Stage0技术门与实现隔离

新namespace：`tfpd_exploration/src/m2_b_small_stability_v1/`；新tests：`tfpd_exploration/tests/test_m2_b_small_stability_*.py`；新runner：`tfpd_exploration/scripts/run_m2_b_small_stability_v1.py`。

新根仅执行时创建：`tfpd_exploration/results/m2_b_small_stability_v1/<timestamp>/`，含`S0_SMALL_LEGACY/seed42/`与`S1_SMALL_COS/seed42/`。旧根只读，连parent pointer也写入新根，不再往旧根追加。

允许导入旧数据/metric等纯函数，但**不得进程内修改旧plan全局常量**来改宽度或LR。新decoder使用显式不可变config；必要时在新namespace实现局部参数化类，注明来源，不复制修改整个旧实验系统。历史代码重放和跨数据集进程不应受新配置影响。

训练前全部通过并出receipt：

1. 绑定新config/code、sampler、校准encoder/P0/EMPTY head/normalizer/bank、旧REF窗口与score SHA；确认没有新增训练session、target更新或carrier重拟。
2. 实例化小模型核对精确参数量、张量shape、因果mask、输出单位；不得意外回落512维。
3. 保持共同输入窗和纯set性质：unit共同置换对应neural/E0/T保持输出；修改未来bin不影响更早输出。保持旧窗口内位置编码，不开跨窗口持续KV-cache新法则。
4. S0/S1初始化SHA、manifest与unit-dropout mask一致；可处置smoke不推进正式RNG；FP32/BF16均finite，梯度可达所有预定模块。
5. S0 LR逐update复现旧1–24规则；S1端点及连续衰减正确；EMA数值对小张量手算、首次copy、n_updates均正确。
6. RAW单步在shadow on/off和evaluation on/off时相同；checkpoint中两种权重不能别名共享存储，评分后不得把EMA拷回RAW继续优化。
7. disposable interruption/resume：RAW/EMA/optimizer/LR/global step/sampler/RNG下一个batch与下一步一致。相同3090与软件栈优先位级；若内核非确定，正式训练前冻结FP32 `atol=1e-6, rtol=1e-5`的状态/输出检查并披露，不能看结果后放宽。
8. 100步disposable profile记录单臂与可选同卡双臂的吞吐、VRAM/PSS、minival和ext-4评分成本。参数少66%不等于时间少66%；前端每bin执行和kernel效率仍会占成本。

代码/debug预算先限2小时。遇到具体技术失败记录；不把同配置bug修复变成LR/宽度试验。

## 8. 两张3090与subagent并行安排

硬件：两张独立24GB 3090、64GB RAM，无DDP。审阅瞬时两卡利用率0%，不是永久空闲承诺；每次启动查UUID/进程和共享lease。

- 两卡都空闲且跨数据集P未进入GPU阶段：S0/S1各一张，完成后立即释放。不得因此取消其他用户任务。
- 跨数据集P已技术就绪或GPU1有既有任务：B只租GPU0；先测同卡双臂，若aggregate throughput相对单进程顺序提升≥15%、峰值VRAM≤20GiB且无显著host压力，则并行。否则顺序执行S1、S0，不强行同卡。
- 若跨数据集原匹配pilot需要两卡但尚未启动，和它的coordinator约定短作业先完成/单卡策略；不抢占运行中PID、不更改其选点/epoch制度。
- Host MemAvailable≥12GiB，工作集目标≤44GiB；共享只读memmap/session-bank，不重复物化重叠窗口。每训练进程BLAS/Torch先2线程、loader workers0；最多2workers需profile支持。
- 当前swap已有占用，不等于正在抖动；监控新的swap-in/out与MemAvailable，禁止swapoff或清用户缓存来“提速”。

执行team最多coordinator+3个活跃worker，复用既有agent，不另开一套满员团队：

| Owner | 独占文件/职责 | 并行边界 |
|---|---|---|
| Coordinator | 新`config.py`、runner、authority/selection/job manifests、最终报告 | 冻结接口、GPU lease、启动两个已登记cell |
| Model worker | 新`decoder.py`、模型shape/因果/参数量tests | 不改旧decoder，不独立启动实验 |
| Training worker | 新`training.py`、`ema.py`、LR/resume tests | 不决定候选或读取target分数调参 |
| Audit/monitor worker | 历史last-k派生、新`report.py`、score/receipt核验、monitor | 只读已封存旧根；模型工作结束可复用其slot做跨数据集CPU工作 |

如果跨数据集E/P已有活跃worker，计入同一个并发/内存预算；合并Model+Training职责，不复制整套worker池。每个worker说明：**你不是唯一在代码库工作的agent；只修改自己的文件，不回滚他人改动，配合共享接口，不独立claim GPU、不启动未登记cell。**

后台训练使用持久launcher和PID/UUID ledger，禁止依赖聊天turn存活。监控每60秒记录step/epoch/LR、RAW loss、gradient norm及clip比例、EMA更新数、VRAM/PSS、吞吐/ETA；启动后10分钟必须一次主动汇报，之后至少每10分钟汇报。

非finite/异常梯度、EMA不更新、无step增长或设备丢失：保存可恢复状态，停止本工单明确PID及其已核查子进程；禁止宽泛pkill。有限但R²差不是即时修改LR或删臂的理由。只允许一次同配置完整恢复；改代码的失败attempt记录保留。

资源上限：首轮训练+所有评分合计不超过**2 GPU-hours**（按两卡实际占用相加），不含CPU实现；新根磁盘≤12GiB。profiling显示无法完成则在正式训练前报告具体预算，不先跑一半再改epoch。预计一轮工作时段内完成，实际ETA必须据实测，不承诺参数缩减线性加速。

## 9. 跨数据集线继续：哪些不受影响，哪些门仍存在

原工单：[跨数据集functional calibration](WORKORDER_CROSS_DATASET_FUNCTIONAL_CALIBRATION_V1_20260905.md)，SHA256 `1acd9f2372fa1c819648581c54dfb01e79821bd4299a02e46b33427ab0589eb7`。
Stage0独立根：`tfpd_exploration/results/cross_dataset_functional_calibration_v1/20260905_113700/`；最新named revision根：`.../20260905_122000/`。

只读交接包已包含H1 population assay（date-equal raw/EB相对变化约0.533/0.521）及M1 dynamic held-trial诊断约+0.03865；这些是该执行者报告的CPU结果，本工单未重新跑或完整审计。最新revision交接包已将E0–E3标为COMPLETE；旧Stage0 READY字样是冻结历史，不自动重跑或改写。

- E线继续完成证据解释/复核。M2网络缩窄不改变H1 backward carrier或M1 forward rSyn3的定义，也不否定CPU诊断。
- P已绑定S-Fix `7976e0b0…`、source-only teacher `f2921cab…`、`row_normalized_nnmf_nnls_v1`与float64可微ridge。Astra复跑25测试通过，接受`CLEAN_OUTER_SESSION_FILE_EXCLUSION`窄声明；具体证据和限制见独立有限冻结记录。
- **新P阶段从确定权重重新开始优化，不天然要求旧阶段所有epoch/Adam状态齐全**；旧epoch缺失只限制旧阶段回溯选点。共同normalizer在D0下重算一次后固定；consumer检查与100配对step disposable profile现在可继续，新阶段必须自行保存完整轨迹。
- 原owner无需等待B评分或重复系统权限确认；不得选all-source teacher伪装clean LOSO、加入lags/FiLM/新rank。P正式12epoch仍需其完整source选点/consumer/资源spec通过后续冻结；不是看到可用GPU便自动开跑。
- P按其原12epoch匹配制度执行；B的24epoch/cosine/EMA不自动迁入。解释线与decoder优化是两种问题，不绑定输赢。

## 10. 交付、结论和禁止扩张

新根交付一个`HANDOFF_FOR_ASTRA_REVIEW.md`，链接：

1. 新旧authority与code/config/manifest/初始化/校准SHA，全部attempt和只读旧根证明。
2. 参数量、LR/EMA/因果/resume/pairing技术门结果及资源profile。
3. 历史last-k补丁；新2×2视图全部source曲线、source-picked/endpoint/last-k/ext-4 per-session/date。
4. 单独visible-development EMA选点与候选数；主候选与对照不可交换。
5. 性能、轨迹稳定性、容量三条独立结论；GPU分钟/host峰值/训练与评分耗时分开。
6. 跨数据集E/P继续状态、实际技术缺口及资源交接；不报告未启动的GPU训练为正在运行。

本轮可证明的最好结论是“接近SPINT参数预算的一种显式时空decoder，在冻结配方下得到一个待复制的性能候选”。不能由单seed小幅胜出宣布第二个机制创新、两种校准信息的独立价值、SSM无效或跨数据集普适。

权重平均是已有优化方法，其收益依赖任务与配方；本工单不把EMA写成新方法，也不保证其改善AdamW下的M2迁移。[EMA研究](https://arxiv.org/abs/2411.18704)

最终动作只允许：提交这轮审查包，提出一个最小确认建议，或关闭该冻结配方。没有自动seed43/48epoch/第三个LR cell/新Mamba/新数据集/镜像/EvalAI。
