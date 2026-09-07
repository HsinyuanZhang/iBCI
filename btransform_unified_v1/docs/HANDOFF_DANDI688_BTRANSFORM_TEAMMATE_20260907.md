# DANDI 000688：B-transformer 远端研究交接指南

日期：2026-09-07；修订：V2，按负责人最新指令开放实验设计权限。接收者已有 SPINT-like decoder + T4 工作，但未做 B-transformer。本文是唯一阅读入口；路径相对仓库根。

**任务：接管688系列研究，自主设计、实现、诊断和迭代；不是机械执行固定48k三臂工单。只有 B-transformer 核心网络结构和 M2-like T4 的计算方法固定。** 其他网络参数、训练方法、支持预算、数据集划分、实验顺序、资源分配及 PMUA 对照，均由接收方 agent 根据结果判断。无需每改一个超参数就回原作者审批。

本指南取代旧《实验二V2工单》中与上述自由度冲突的限制。旧工单作为起点与审计检查清单保留，不再是强制执行合同。其他任务的工单不因此变化。

## 1. 固定的两个方法核心

### 1.1 B-transformer 核心结构

这是与 SPINT-like 不同的 decoder 系统；不得把旧 SPINT 分数当成新 B 分数，也不得换回 SPINT decoder 后仍称 B-transformer。

```text
支持trial活动 → 冻结的activity encoder → session-static E0 → 学习投影
因果神经窗 → 每单元local causal conv → local表征 + projected E0
                                        ↓ 拼接M2-like T4
                                  per-unit token MLP
                                        ↓
                              learned-slot set frontend
                                        ↓
                              causal temporal Transformer
                                        ↓
                                  2D behavior readout
```

保留上述信息流、E0投影加法融合、显式四维carrier输入、slot聚合与因果时序主干。不能把主模型改为E0直接拼接、移除slot聚合或换成另一种decoder再归入同一主结构。为检验作用而设的NOE0/NOC等消融当然可以关闭相应输入。

“固定结构”不等于锁死所有尺寸超参数。宽度、层数、head/slot数、投影维度、卷积核/窗口长度、dropout等可调，前提是仍属于上述同一结构、代码确实支持、因果性与端口通过测试，并记录准确配置。初始参考是 P16、k5/16 local conv、token width256、8 slots、4层 temporal Transformer、W50；它们是便于起跑的配置，不是不可改动的科学常量。

### 1.2 M2-like T4 的计算方法

688的主carrier采用已确定的MOVE700事件定义，固定估计方法而不固定支持trial数量：

- 原始字段为 `go_cue_time_array`、`target_dir`、`num_targets`。参考 `streaming_calibration_exp/src/data/rt_sparse_endpoint_loader.py::_go_cues` 的二维ragged解析。
- 单目标trial：num_targets必须为有限整数且=1；第一个声明cue有限，其他位置全NaN；方向合法。Inf不是合法缺位，不取多cue的首cue伪造单目标。
- 每trial、每unit按原始spike半开区间 `[go, go+0.700)` 计数；除35，得到counts/20ms-bin。不得以近似dense grid替代原始事件对齐。
- 设计矩阵 `[1, cos(theta), sin(theta)]`；trial等权、无ridge、float64 ordinary least squares；必须rank=3。
- 若系数为截距b、cos系数a、sin系数d，则输出 `[a,d,sqrt(a*a+d*d),b]`，保存float32。
- 无效cue/方向、越trial终点或侵入下一trial：排除并记原因；不夹短窗口后仍除35，不用query速度推断方向。不足满秩则该条件不可估计，不静默改成另一种estimator。
- 输入标准化可按网络训练需要设计，但与上述raw T4定义分开记录；统计仅由当前划分的训练数据拟合，不能借验证/query标签拟合。原始四维T4及其哈希应保留。

可复用 `streaming_calibration_exp/src/data/falcon_t4_features.py::t4_from_trial_sums` 的数值核心，lengths全35，不复用FALCON读取器。这是“M2-like”的trial等权OLS/四维映射，不声称688与M2的事件锚点完全相同。改变窗口、ridge、trial权重或映射属于改变固定方法，不能不说明地替换主版本。

支持预算M、支持trial选取策略、缺失时的预算方案可以探索，但需在使用query成绩前定义清楚，且不能跨越所声明的支持/query边界补取标签。E0与carrier预算可以不同，只要真实披露；做条件效用对照时须匹配可比的信息预算。

## 2. 交给接收方agent的决策空间

| 领域 | 可自行决定 | 需要留下的证据 |
|---|---|---|
| 网络参数 | 维度、depth、heads/slots、窗口、卷积核、dropout等同结构配置 | 准确结构配置、参数量、因果/padding检查 |
| 训练方法 | optimizer、lr/schedule、precision、batch、采样、loss、EMA、训练长度、早停/续训 | 曲线、资源消耗、checkpoint/选点规则、变更原因 |
| 校准输入 | 支持预算、选trial策略、合适的合法activity donor、标准化策略 | 来源与标签使用范围、预算与query隔离、donor审计 |
| 数据划分 | 重新划分可开发session、按时间/era分组、LOSO、交叉验证 | 版本化roster、选择依据、train/val角色及历史暴露 |
| 对照 | 三臂优先顺序、追加/删减有理由的诊断、matched SPINT、PMUA | 对应假设、可比性、哪些结论由哪些对照支持 |
| 研究推进 | pilot→扩展、多seed、停止无效支线、调整计算分配 | 不只保留成功结果；变更和失败记录可追溯 |

允许根据已有结果改方案，不要求所有选择在第一次val读分前永久封死。关键是**把探索与确认分开**：调参用过的数据是开发集；如果用结果选择了配置，就如实称为探索结果，之后用未参与该选择的session/fold或独立重复确认。不能见分后换划分、藏掉不利seed，却仍把汇总写成预注册结论。

原建议6 GPU-h不是硬性总额，48k不是硬性终点，seed42/43不是唯一合法seed。资源由队友结合其机器与实际授权安排，不抢占他人任务；预计显著超出其可用资源时再协调。网络参数或训练配方改动不需要逐项向原作者申请许可。

## 3. 推荐起点：先得到可解释的结果

最容易起跑的是：复用队友已有688/SPINT数据管线与T4经验，接入B模型；先做小规模真实数据smoke与吞吐测试，再决定正式矩阵。

参考起点而非硬约束：

- sub-C/CO/SUA，20ms bin，W50，原生2D behavior；训练raw target=5×native，评分pred/5。
- 前30个按时间排序的奖励trial作为支持，M_activity=M_carrier=30，query置于支持区之后，整个输入历史不得跨边界。
- P16/D4/8slot、batch32、AdamW lr1e-4、warmup200、cosine、wd0.01、clip1、unit dropout0.1、EMA0.9995。
- 先短跑验证学习能力与三条输入路径，再根据曲线、成本及差异选择步数、seed和读点评估。48k仅是前稿候选。

必要科学问题是：

| 问题 | 推荐对照 | 如何解释 |
|---|---|---|
| 有carrier时E0有无额外效用？ | FULL vs NOE0，同形同图匹配重训 | 条件E0效用，不靠推理时清零代替 |
| 有E0时显式carrier有无效用？ | FULL vs NOC，同形同图匹配重训 | 直接carrier效用；E0本身仍可能含方向信息 |
| B系统是否有竞争力？ | 同面B0与队友已有T4重放；必要时matched SPINT重训 | 旧系统重放是系统比较；架构归因需共享输入和训练条件 |
| 模型是否真的学到了东西？ | support constant、训练曲线、pred/target尺度与方差 | 低成本诊断，不用query均值构造预测基线 |
| 在未分离单元观测上是否仍成立？ | 合理设计的PMUA与对应参考/消融 | 是pseudo-MUA证据，不直接等同真实硬件MUA |

以上是推荐的证据结构，不强制一开始跑完所有seed×arm。如果只完成FULL/NOE0，就先回答E0问题，不提前宣布carrier有增益；若诊断发现某支线无意义，可停并解释。若要发表某个机制结论，最终必须有支撑该结论的公平对照。

同一条件效用比较尽量共享初始化字节、数据流、支持/query点与随机mask，或用清楚的多seed设计处理不可匹配部分。若为每个系统单独调参，那回答的是系统竞争力，不是严格单因素效应。允许两种研究，但不要混称。

## 4. 数据划分可以改，数据角色不能含糊

已有起点manifest：
`sua_exploration/configs/subc_co_27_6_strict_train_val_manifest.json`

SHA256：
`4607e979c6c2ff451c147a8d9878fe1080b9d3e9bbc7304b559616eb2a13a0c9`

原分配为27 train、6 development val、6 sealed-test。27+6开发session可以按研究需要重划分或做交叉验证，不必沿用旧27/6。旧6val已有研究暴露，不能通过改名变成blind test。

原6 sealed-test仍属于保留的最终评估资产；“可以改划分”不默认等于现在就打开全部保留集。可以先在已开放session内自由设计；若研究确需改变保留集用途，明确协调并记录资产角色变化，不能悄悄读取后继续声称sealed。sub-M/RT等扩展由队友按科学价值设计，但须核对数据权限与独立任务边界，不能把不同任务分数混在同一指标中。

每个新划分建立自己的manifest和哈希，按允许路径构造读取列表，记录实际打开的文件。避免旧脚本全目录自动发现把保留数据顺带读入。原始train/val NWB约3.69 GB，不在权重包中；队友可复用已有数据，核对session/file哈希。惯用路径：
`sua_exploration/data/dandi_000688/sub-C/{session}_behavior+ecephys.nwb`。

改变W、M、trial选择或划分后，重新生成相应support/query和缓存receipt；旧M10 calibration cache不能改标签称M30。合法窗、mask、target列序、单位顺序必须核对；完整输入历史不得穿越支持/query或不合法trial边界。

**重新划分时还要重新审计预训练来源。** 当前B0 donor在旧27train上训练过；若把其中session当新的held-out，它已经通过donor影响模型。可将其明确标为预训练暴露，或选择/训练与新划分兼容的donor，而不是只重训decoder就称干净LOSO。队友可根据资源与研究问题决定方案，不受旧“不得新训donor”条款约束。

## 5. PMUA：负责人重点推荐的研究主线，不需要等待SUA正结果

**负责人特别推荐PMUA实验，应列为高优先级研究主线，而不是有空才做的附录。** 建议在SUA数据/模型管线通过基本检查后尽早开展，不必等待SUA出现正增益或全部seed结束。队友自主决定构造方式、pool大小、重复次数、训练/迁移设置与控制矩阵；若实际数据或资源使其不可行，应给出具体依据与替代方案，而不是静默略过。

建议首先回答：聚合观测下B-transformer能否保持可用解码；E0/显式T4的相对作用是否改变；与队友已有SPINT-like+T4在相同PMUA输入下相比如何。先用小规模pool设置摸清趋势，再由接收方决定扩展。不是要求一开始跑庞大的全因子网格，也不要求结果必须优于SUA。

建议保留以下可解释性约束：

- 清楚说明PMUA是怎样从SUA构造的：按真实电极/位置分组、随机分组或其他规则；没有空间信息时不要虚构电极关系。若随机，记录seed与成员映射；不按query性能挑pool。
- PMUA先按定义形成聚合spike/count观测，再用固定MOVE700方法重新估计四维T4。尤其不要平均SUA的幅度列来冒充聚合后的幅度，因为幅度映射是非线性的。
- E0也来自PMUA的支持活动；不要默认让PMUA模型额外获得部署时不存在的SUA身份信息。使用SUA预训练donor是可研究的迁移设置，但要明确训练来源与观测假设。
- 各系统使用对应的相同PMUA观测、支持标签范围与评分点；报告有效通道数、pool大小、尺度变化和成本。通道归一化可探索，但不能偷偷改变raw T4定义。
- 可以设计train-PMUA/test-PMUA、SUA→PMUA迁移、多pool尺度或跨session鲁棒性；分别命名，不混成一项“PMUA结果”。真实MUA仍需要真实MUA数据验证。

## 6. 已提供的donor：可用的起点，不是强制唯一来源

B0原文件：
`sua_exploration/checkpoints/sua_spint_t4_mainline_fp32_v1_b0_dandi688_co_s42/epoch_ckpts/epoch_011.ckpt`

原SHA：
`dc1c8e512cb1731154243918cda457bae2ad55da8cb97aac442ae662f36ce2cc`。

实际 `student.id_encoder.` 下12个张量，`fc_id_out.4.weight=[50,512]`，**E0输出宽度50**，不是配置hidden_dim=128。side_dim=0、trial_length=100、window_size=50。epoch_011是第12个完成epoch，不是48k。

此B0是 `BatchReferenceEncoder`：输入[B,M,T,N]转[B,M,N,T]，逐trial fc_id_in，再在M维mean，再fc_id_out；不是B3 early pooling。原生E0宽50与decoder的投影宽度是两个不同参数；更换donor或修改窗口时要做真实端口验证，不假定随意reshape就成立。

Teacher初始化依赖已确认：`StreamingCalibrationModule.setup` 载入teacher、复制decoder，`build_encoder` 与 `copy_teacher_id_weights` 使用teacher id权重。task_only不消除初始化影响。Teacher原SHA：
`9b4a94ca890042ca3570ec2fceedcc7597a64bc42a70d87182739d0aa9ee831d`。

teacher的训练/验证/选点roster尚需补齐；不能从mc_maze文件名推断无688影响。队友可以补审计、换合法donor或重训兼容当前划分的donor；在来源不明时不要给出source-only或干净held-out结论。NPZ含权重不等于含完整来源证明。

有效N先产生E0，再按模型需要pad；pad行不参与attention、source统计或校准。encoder的冻结/预训练成本与decoder训练成本分开报告；同一donor上的多个decoder seed不等于独立预训练复现。

## 7. 代码端口、评分与实现状态

通用模型已有；**688专用真实数据适配、缓存与配对训练runner仍需接收方实现**，可以充分复用其已有SPINT/T4工作。本包不是一条命令开训的独立发行版。

起跑构造：

```python
geometry = dict(task="dandi688", window=50, prefix=0, units=100,
                e0_dim=50, carrier_dim=4, out_dim=2, target_scale=5)
model = BTransformerUnifiedDecoderIdentity(
    geometry, seed=42, identity_mode="proj_add", proj_dim=16, temporal_layers=4)
```

不要传字符串m2借默认配置，也不要省略identity_mode（默认并非proj_add）。调整网络参数时检查实现可支持的范围；需要适配时隔离修改并补测试，不用静默fallback。

TaskBank包含E0[N,d]、carrier[N,4]、unit_mask[N]、X_store[S,L,N]、target_store[S,2] native scale、window_ids[S]；meta至少有shape/trial_count/estimator/array_sha256/budget。live-batch可用S=0空stores。输出[B,2]；尺度变换由harness负责。

评分参考 `sua_exploration/scripts/eval_epoch_window_dandi688.py` 与generic版本调用链，但旧脚本有12epoch/e5..12聚合，不能直接当任意新训练终点评分器。底层 `evaluate_fixed_protocol_over_validation_sessions` 需显式传manifest；不要照跑自动discover全部NWB的旧顶层选择CLI。

以相同support/query点重放B0，做逐预测与指标parity；FP32可从 `1e-5+1e-5*abs(reference)` 容差起步，必要时根据硬件/数值路径解释误差。支持预算或query变了，旧分数不再是parity目标。建议主报每session variance-weighted R²再等权平均，同时给逐session值；其他指标可补充，不能通过改聚合藏掉失败session。

结果相反或近零允许形成结论；曲线还在改善可延长，稳定后无效可停止。不要预设48k必收敛，也不要把无效唯一解释成训练不足。根据结果改变实验设计时，把先前结果和改变理由保留在research log。

## 8. GitHub资产与接收命令

**当前开放研究版包：** `btransform_unified_v1/handoff/dandi688_20260907_v2/`。
v1是被本修订取代的本地历史包，不作为当前交接版本发布。

| 文件 | 用途 |
|---|---|
| b0_s42_e11_encoder.npz + JSON | 最小E0 donor；12个tensor，去掉student.id_encoder.前缀 |
| b0_s42_e11_full.npz + JSON | B0原state_dict，74个tensor，含teacher/student |
| t4_s42_e11_full.npz + JSON | T4/B3S side_dim4，seed42 e11参考；70个tensor |
| source_snapshot.zip | 537个明确列出的源码/指南/工单/历史688资料/manifest/receipt文件 |
| source_manifest.json | 每个源码文件的路径、字节数、SHA256 |
| bundle_manifest.json / SHA256SUMS | 包校验、原ckpt来源、逐张量往返结果与排除项 |

三个NPZ约38.33/4.19/34.22 MB，均原精度、无pickle；未量化。完整NPZ已保留teacher.*，不再另复制原teacher文件。没有optimizer/scheduler/RNG等恢复状态，不能用NPZ恢复旧训练进度；新实验应另存自己的可恢复checkpoint。JSON原绝对路径只是provenance，远端不能直接照用。

源代码快照用于补齐当前共享GitHub上尚未跟踪的B代码及其依赖；是明确文件内容的工作树快照，不冒充干净commit。先解压到新目录/独立worktree，与队友已有代码比对后合并，不覆盖其工作。只保证manifest所列资产；旧文档内其他结果路径未必随包提供。第三方环境、NWB、dense caches不在包中。

在合并源码后的仓库根，选择队友可用的Python环境运行：

```bash
python btransform_unified_v1/scripts/pack_688_handoff.py verify btransform_unified_v1/handoff/dandi688_20260907_v2
python btransform_unified_v1/scripts/smoke_688_handoff.py btransform_unified_v1/handoff/dandi688_20260907_v2
```

第一条验文件/逐tensor；第二条合成数据CPU检查E0、B端口、pad不变性与NOE0/NOC零数据梯度，不证明真实数据协议通过。加载函数为scripts下 `pack_688_handoff.load_state(npz_path)`，构造对应类后 `load_state_dict(state, strict=True)`。最小encoder见smoke。完整B0/T4若旧Lightning setup仍要求原teacher路径，需要NPZ-aware适配，不能改后缀冒充ckpt。

记录实际PyTorch/CUDA及科学计算依赖版本；源码有SPINT与streaming两套顶层src，避免namespace串包，记录模块__file__和B init_meta。本地CPU测试不构成远端GPU环境认证；不要机械沿用旧environment.yaml。

GitHub常规Git对>50MiB文件警告、>100MiB阻止，浏览器单文件上限25MiB，见[官方限制](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)。本包用Git CLI或Release共享；不要把整个工作区、NWB和缓存一起提交。当前尚未自动commit/push。

## 9. 历史工作：可以借鉴，不要求全部重跑

| 分支 | 材料与含义 |
|---|---|
| B0/T4/TS4 FP32 | FP32_T4_MAINLINE_PROTOCOL.md；历史三seed e5..12均分.236417/.574976/.284528，仅其原support/query合同有效 |
| Calibration-profile FiLM | RESULT_DANDI_000688_CALIBRATION_PROFILE_FILM_V1_20260904.md；三seed profile相对capacity control无稳定收益 |
| CP postpool/coadapt | WORKORDER_DANDI_000688_CP_FILM_POSTPOOL_COADAPT_V2_20260904.md及incident；旧独立分支 |
| Sparse-event T4/FiLM | RESULT_DANDI_000688_SPARSE_EVENT_T4_FILM_V1_20260904.md是标注未终结的历史快照，使用数字前查最终receipt |
| 活动选择/TC-AS | WORKORDER_DANDI_000688_TC_AS_EP_V1_20260904.md及STAGE0_ATTEMPT2，供输入设计参考 |
| sub-M、MUA、量化 | DANDI_000688_SUBM_*、CURRENT_RESULTS.md、FP32主线后续段；不同评分面需单独命名 |

上述材料位于sua_exploration/docs，相关688文件已随快照提供。队友可依据结果复用、扩展或跳过，尤其PMUA已明确开放，不受旧文档“先等某gate”的顺序约束。其他人的活跃任务、其GPU和外部提交权限并未因此转移。

## 10. 研究交付而非固定工单验收

推荐接管顺序：验包和模型smoke → 明确数据/预训练来源 → 接真实数据和评分 → 小规模学习与成本诊断 → 尽早启动重点推荐的PMUA主线，并自主安排SUA/PMUA矩阵 → 根据结果迭代 → 分清探索与确认后总结。

回传至少包括：当前问题与假设、结构/固定T4实现版本、划分与support/query manifest、数据访问和donor来源、实际训练配置、每次改动的依据、实际成本、逐session/seed结果、必要对照、失败/停止原因及结论适用范围。可用轻量research log持续GitHub沟通，不必等完整大矩阵结束才报告。

分开回答“实现/数据是否可信”“是否学会”“E0条件效用”“直接carrier效用”“系统竞争力”“PMUA鲁棒性”。没有跑到的对照标未回答，不把所有状态合并成一个有效/无效。科学上可信的负结果和边界结果，同样是完成任务。
