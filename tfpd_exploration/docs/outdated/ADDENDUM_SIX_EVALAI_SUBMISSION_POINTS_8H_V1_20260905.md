# 今日8小时提交插队：M2×2 / M1×2 / H1×2

日期：2026-09-05，Asia/Hong_Kong。  
状态：**EXECUTION_HANDOFF_READY__SIX_NEW_CANDIDATES__AUTHOR_DID_NOT_SUBMIT**。  
父工单：`WORKORDER_TWO_MAINLINES_M2_M1_H1_LONG_HORIZON_V1_20260905.md`。

**修订R2：用户确认C2＋随机M3 activity等路线已经提交；不再查记录争论，不再安排其重训重投。原H1 FF/RF两席撤销，替换为新的时间Transformer D-FLAT / D-ROUTE。以下H1时间点已重排，原“2小时轻量identity训练可提交”估计作废。**

## 0. 最新用户决定与权限优先级

今天不等长程实验全部结束。目标是在8小时内完成6个**尚未提交的新候选**的EvalAI注册：默认M2两次、M1两次、H1两次；M1仅一个候选及时合格时，允许M1一次、H1三次、M2两次。

优先新decoder和M1新设计；不重复既有581919/581920及历史M1/FiLM产品，不用改名、重打包或无变化权重凑次数。

本补充对父工单“本周期不提交/不做all-source”的条款作**仅限下述六席与H1新decoder产品配对**的例外。执行者通过硬验收即可上传并注册，不再请求逐次确认。其余新结构、数据边界和长程预算限制不变。

作者此轮只做只读额度/资产核查并写工单，没有启动GPU、上传镜像或调用注册POST。

**完成定义是收到6个不同submission ID，不是docker push完成，也不是承诺8小时内6个都出finished分数。** 服务器排队、并发上限、断网及正确性失败不在本地可保证范围内。不能为实现“6/6”绕过验收；预计不足时在下文早期关口明确上报。

## 1. 实时额度与截止时间

2026-09-05 `07:35:46 UTC`（15:35:46 UTC+8）只读authenticated GET：

| 字段 | 当时值 |
|---|---:|
| challenge / phase | 2319 / 4599，few-shot-test-2319 |
| is_active / is_submission_paused | true / false |
| max_submissions_per_day | 6 |
| max_concurrent_submissions_allowed | 3 |
| max_submissions_per_month / max_submissions | 50 / 100 |
| 当前team该phase总submission | 30；全部30条已读，无下一页 |
| submitted_at在UTC 2026-09-05的条数 | 0 |
| submitted/queued/running | 0 |

最近581919/581920分别注册于UTC 09-04 18:30/18:31，因此不计入上述UTC日期计数。不能按本地文件名中的“09-05”推断它们已用掉今天额度。

API计数是时点快照，实际服务端准入为准；执行者在启动及每次注册前重新读phase、完整分页submission列表。全队共用一个quota ledger，不能每dataset各自假设有6次。

设T0为执行者实际接单时间，今天软截止 `D=min(T0+8h, 2026-09-05 23:30 UTC+8)`。23:30是用户本地“今天”的保守交付目标，**不声称服务端额度在UTC+8零点重置**。如T0已使可用窗口明显短于本计划，应立即重估而不沿用陈旧时刻。

官方README要求先本地验证容器，并说明每次评测有6小时运行限制；这不等于排队+注册具有6小时SLA：[FALCON官方提交说明](https://github.com/snel-repo/falcon-challenge#evalai-submission)。

## 2. 六个提交点：按就绪即提交，不等上一单出分

以下时间是**取得submission ID的目标时间**，不是才开始build的时间。镜像/候选就绪可以提前提交；若并发已满，保持已上传状态并等合法空位。

| 席位 | 目标时间 | 候选 | 它回答什么 | 是否需要新训练 |
|---|---|---|---|---|
| S1 | T0+45min | **M2-SMALL-TRF-PICK** | 约3.54M新时间decoder的官方迁移 | 否，已有未提交checkpoint |
| S2 | T0+90min | **M2-LARGE-TRF-PICK** | 较大时间decoder与压缩版本的产品比较 | 否，已有未提交checkpoint |
| S3 | T0+4h | **H1-TEMPORAL-TRF-FLAT** | 固定现有校准，显式时间Transformer能否迁移 | 是，新的完整decoder |
| S4 | T0+4h30min | **H1-TEMPORAL-TRF-ROUTE** | 同一时间主干，校准引导的功能路由是否增加价值 | 是，新的完整decoder |
| S5 | T0+5h30min | **M1-P-FIX-QUICK** | 新M10匹配consumer训练的官方参照 | 是，修复后的P-FIX |
| S6 | T0+6h30min | **M1-P-CA-QUICK** | 通过短校准估计器优化行为basis的官方探针 | 是，修复后的P-CA |

T0+7h为最后补位/上传/注册缓冲；截止D前必须核实server ID，不能只记录“后台上传中”。如果服务器并发尚未释放，继续合法轮询，不制造重复请求或新账号绕限制。

H1全新decoder及H1 D-ROUTE现在进入今天队列；M1全新decoder、M2新的D-ROUTE配对仍在长程队列。今天的M2是“已训练但尚未官方提交的新decoder”，H1是新训练，二者不能混称已经拥有可部署权重。H1的S3/S4时间须经真实profile确认，不再沿用identity-only的速度。

## 3. M2两席的精确选择与部署

### S1：small Transformer

使用现有 `m2_b_small_stability_v1/20260905_123000` 与 `20260905_131500` 的S0/S1/N0/N1、seed42/43、epoch1..24；**只允许EMA视图参与今日S1产品选点**，不临时扩RAW/新seed。

产品选择面固定M33-disjoint clean ext-4，依次按：equal-session mean高、worst-session高、epoch早、seed小、cell字典序。完整8×24候选表预先封存，优先复用astra_pack内已经评分的CSV；确认所选epoch的真实EMA字节存在且重放一致。

S1可以合法使用这个可见开发面选点，但不改写旧source-pick的失败/分裂结论。不能把可见选出的最高点叫预注册S1跨seed复制成功。

### S2：large Transformer

只用旧B-TRANSFORMER的 `seed42_shuffled` epoch1..12与`seed42_shuffled_e13_24` epoch13..24，RAW视图。按与S1相同ext-4产品规则选点；不使用原顺序训练或把Mamba混入此候选集合。

small与large的历史训练/容量不同；官方并列表是产品比较，不是已经严格控制优化制度的容量消融。未达到长程产品门仍可作为明确标注的合法architecture probe提交，不能隐藏本地worst-session退化。

### 共同硬验收

- 保留训练时完整E0、MOVE-T4、已有静态head/normalizer的权威；导出全部held-in与held-out dataset tag所需的合法M33 bank，不只导出ext-4。
- **不能只换旧静态SPINT的identity payload。** 新B的frontend、temporal stack、readout都必须装入新runtime；旧SPINT decoder不能代替新B执行。
- 50-bin因果窗口、单位/`/5`位置、mask、batch<=官方上限、reset/observe/predict与逐窗reference一致。禁止跨窗口无证明复用KV状态。
- CPU目标环境与容器真实流测试；本地CUDA训练成功不证明远端可运行。速度按实际评测batch与流长估计，提前发现6小时运行风险。
- 模型、payload、预测指纹与已提交30条清单去重；方法不同但偶然打包字节相同也要排除no-op。

Mamba不占“有把握的必达席”：当前依赖Mamba2专用backend，尚无本轮远端目标环境合格证据。若另一个agent已经完成相同目标环境的算子/速度验证，可以作为S2技术阻塞时的未提交结构替补；否则不在今天临时手写scan或装错CUDA来抢名额。

## 4. H1两席：真正的新时间Transformer，不再做FF/RF重投

### 4.1 固定的校准底座与新的可训练decoder

依据用户确认，C2＋随机M3 activity等已有提交的路线从本次队列移除；不要求用户提供submission ID，也不为排重重新打开其记录。

当前C2只提供两臂共同冻结的calibration encoder/H-C物化及完整原模型评分参照。可使用已绑定的C2 epoch15字节（SHA `ce46267eb220142b8ef1f2e5acf05194650796ac2752594995b40d2ad0950215`）提取校准底座；**不是将C2 decoder权重当成新Transformer的初始化**。

固定first-M3 activity和H-C，按实际C2算子获得E0。若E0已融合H-C，明确称fused identity，不叫纯activity。两臂都不重新训练校准分支，不改变pooling、不增加随机activity、不加FiLM。

新decoder从头训练，结构继承父工单 §7：

`neural history [B,700,176] -> shared causal Conv(1→16,k5)`

`-> concat(local,E0,H-C) -> per-bin 8-slot set attention (d=256)`

`-> 4-layer causal temporal Transformer (d=256, heads=8, FFN=512)`

`-> 256→128→7 readout -> last-bin native output`。

- **H1-TEMPORAL-TRF-FLAT**：普通动态token到slot的注意力。
- **H1-TEMPORAL-TRF-ROUTE**：相同动态路径/时间主干，在slot-to-unit logit上加入父工单定义的静态calibration routing；32维key、每head零初始化gate，原unit mask不变。

这两臂是完整frontend＋时间主干＋readout训练，不是head-only。共同参数由FLAT模板逐字拷贝，新增routing用独立RNG；zero-route还原FLAT，并证明真实更新后路由梯度有效。

原SPINT-like本来含Transformer/注意力；这里测试的是**显式时间轴解码与校准路由组织**，不能宣传为“H1首次使用Transformer”。ROUTE比FLAT多参数，首轮只测试候选价值，不自动证明新的信息语义或独立机制。

### 4.2 数据、训练和选点

训练只用公开held-in source：canonical first3用于校准，source query的完整W700历史须位于其后。source选择用预先冻结held-in-minival集合。校准底座/其预训练可能接触过source选择日期，称known-source development，不称clean LODO。

held-out-calib的M3只用于合法离线校准，不增加query监督，不把同M3拟合成绩当独立验证。今天是all-source decoder产品探针，与后续五折结构比较另立根 `h1_temporal_decoder_quick_product_v1`，不可填入LODO表。

同seed42、同query/update/dropout manifest、目标12完整passes、stride4、effective batch32、AdamW1e-4、wd1e-2（bias/norm排除）、clip1、1epoch warmup后恒定；EMA0.9995为选点视图，RAW诊断。精度在source-only profile后按父工单冻结。

target始终M3、7维输出、完整W700、原输出单位与`/20`合同、无MAT7/readout拟合、无growing memory。PE长度须覆盖700；不硬拷M2的identity50、输出2或max_len256。

T0+3h冻结一次时间驱动快照：`E_H=min(12, 两臂都完成的epoch数)`，要求E_H>=4；两臂同在epoch1..E_H中按source选择器选点。若不足4，可等待至T0+4h的唯一后备快照，再不足则H1席标训练未及时就绪，不降低最低量或回填C2/FF/RF旧路线。

达到快照后优先评分/打包，剩余训练到12可以让位并从完整状态恢复。早期快照按实际epoch披露，不保证从头训练4epoch就接近强C2；只通过finite/合法性不能声称性能已成立。禁止根据本轮官方分数改变候选集合或延长选择窗口。

### 4.3 今天首先要确认的成本与部署风险

T0+45min前完成H1真实W700/N176的forward/backward和目标环境流式推理探针；测完整effective-batch update，而不是microbatch一小步或FiLM-head耗时。若需更长接口实现，立即更新ETA，不能继续报旧“2小时必达”。

允许microbatch+梯度累积、activation checkpointing、保持causal Conv overlap的精确set-frontend时间tile；不允许把700截成50/256、downsample或改slot数后还叫同一个配对。时间Transformer不能随tile重置。

新H1 runtime要真正执行新decoder，可复用既有合法M3缓存和Docker基底，**不能仅在581920 payload里换一组identity后继续跑旧C2 decoder**。覆盖全部dataset tags，验证逐窗reference、continual reset/no-op边界、eval-mask暖启动、host/container预测parity和实际CPU运行成本。无证明的跨滑窗KV缓存不得启用。

## 5. M1两席：修复后的P对，不把旧冠军换标签

保留长程工单的fold0 S-Fix parent、排除20120924的26/27/28 source、D0/NNLS/ridge、共同normalizer和P-FIX/P-CA trainability定义。

**P1–P5真实验收仍为硬门，今日额度不构成豁免。** 尤其要解决fresh对象隔离、train模式、真实resume、独立旧carrier参考、真实source-query梯度和window-ID审计。不能把原有有污染的100-step实例直接打包。

两卡时，P-CA占较长时间的一卡；P-FIX使用固定carrier缓存，完成后把卡交给仍未完成的打包/校准任务。两臂时间可不同，但正式样本/更新与选择候选epoch权利匹配。

目标仍为12epoch，使用已冻结P制度。为今天登记，在T0+4h30min冻结 `E_P=min(12,两臂都完成的epoch数)`，至少4；每臂只从1..E_P按同source选择法则选checkpoint。不足4则继续到T0+5h30min再判一次，不能事后降低最低训练量。

要求明确披露：

- 今日是**three-source/fold-parent官方探针**，不是all-source最优M1，更不是完整新parent预训练。
- P-FIX用来解释consumer再训练效应；P-CA与之比较才是basis学习的增量。无需两个都超过历史0.6396才允许作为预先指定的合法比较提交。
- 官方held-in也包括本地outer20120924；适配必须只用其合法M10，导出所需全部dataset tag，不能因本地训练排除它而缺payload。
- 未使用公开later-day M4/6代理冒充M10选点；不得在hidden输出上重拟basis/normalizer/输出affine。
- 目标物化只使用冻结的D与闭式M10拟合，零query标签/optimizer；缓存E0/identity允许，完整16维signed EMG输出不变。

P-CA以当前D重算所需target carrier；P-FIX不能误用候选D。两个部署包的parent/calibration/consumer/basis摘要必须可比且真实不同。

只有一个M1候选在正确性与时间门下可用时，允许提交这一席，另一席按 §7 换给H1；**不把缺失的paired结果伪装为已测的性能增量**。

## 6. 八小时资源调度

开局就暂停长程队列新增GPU作业；已有其他用户/队友作业不抢占。今日作业计入长程72GPU-h总账，不在账外另花。

| 时间 | GPU-A | GPU-B | CPU/上传 |
|---|---|---|---|
| T0..约1h | H1真实形状profile，过门后FLAT/ROUTE训练 | P整改/profile，过门后FIX/CA训练 | M2两runtime、镜像、去重与早期上传；H1/M1基础镜像并行准备 |
| 约1..3h | H1配对：合卡并行仅限实测有吞吐收益，否则交错完整epoch | P配对：FIX缓存固定carrier，匹配进度轮转 | 两M2就绪即注册；H1/M1 runtime和smoke准备 |
| 约3..5h30 | H1共同快照、评分与打包优先 | M1共同快照及评分 | H1两席就绪即注册；M1等待最后权重小层 |
| 约5h30..D | 未完成候选的验收/导出优先 | 同左，尚未到12的长程更新让位 | M1两席及ID核对；不等成绩决定下一席 |

旧H1“≤1.5GPU-h”估计撤销。暂给H1新decoder配对6 GPU-h、P快照配对6 GPU-h、profile/评分2 GPU-h的准入预算，其余至8小时双卡容量留缓冲，仍记长程总账。这里是资源上限，不是实测运行时间。两任务都须预测可达到共同epoch>=4且留至少60min打包/注册；不满足就即时上报资源冲突。不能把H1全decoder与P两完整12epoch同时写成未经profile的8小时保证。

两卡分别优先H1和M1，可按最新ETA借卡追平较慢的配对臂；不能某臂连续跑满12而另一臂还停在0。隔离的per-arm完整checkpoint支持交错，不共享可写model/optimizer。合卡并行仍要求显存/内存安全且实测aggregate吞吐更高。

T0+30min前完成：六席owner、所有parent实际字节、P失败项现状、目标环境runtime风险、镜像增量大小/上传带宽估计。若一眼可知候选无法赶上，立刻发红色短报，不等到最后一小时。

先争取两M2在前90min注册；第三单取最先合格的H1/M1新候选，不死等表格顺序。不能再承诺第三单必在2h出现。服务端并发为3，**不能假设6单可同时排队注册**。只在active<3时发新注册；若前两单长时间占槽，应提前上报截止风险。运行时限不含未知排队延迟，因此“今天6个ID”仍不能给服务器层面的无条件保证。

## 7. 替补法则：保留新方法含义，不重复旧包

1. S2若large Transformer目标环境/时限不合格：先检查已经有合格目标环境证据的未提交Mamba。没有，则用**small Transformer两seed等权预测ensemble**：每seed在S1相同cell/epoch集合内独立产品选点，固定0.5/0.5，不拟合输出融合权重。它是新decoder稳健化探针，不是第三种架构创新；须另验速度。
2. M1仅一个候选及时合格：空出的一席可给**H1新Transformer FLAT/ROUTE等权预测ensemble**，两者都必须是今日新训练且通过验收，集成也须通过额外推理成本检查。固定0.5/0.5，不按本轮官方分数选择成员或权重。若两预测等价/no-op或ensemble不满足时限，则不占这一席。原FF/RF集成替补撤销。
3. 如果M1两者均因共同正确性问题阻塞：不自动把两席都换成H1，也不把尚未审核的新M1变体临时拼上。即时给Astra/用户报告“6席目标受到正确性阻塞”，保留可运行4席。
4. 不重复581919、581920、581727、581736及用户确认已经提交的H1 C2＋随机M3/FF-RF类系统，不把旧产品seed/epoch的同一功能包换tag当新实验。H1新decoder不就绪也不回填这些旧路线。

替补只依据本地资格、截止时刻与目标环境，不使用**本轮新产生的hidden结果**。六席候选与替补规则先封存；远端状态可以读来释放并发槽，本轮分数不进入尚未提交候选的训练或选择。

## 8. 构建、上传和注册：三个不同状态

- 复用已在ECR存在的基底层，依赖尽量放稳定base层；COPY只含必要源码、manifest与权重。禁止COPY仓库、数据集、训练optimizer全状态、虚拟环境和日志。
- 本地image完整尺寸仍受phase/challenge限制；“新增层很小”不代表总镜像可无限大。M1既有镜像约9.64GB，必须实时检查而非再次上传整个环境。
- 每镜像先做host-vs-container预测parity、合法全tag覆盖、remote-path模拟、finite输出与状态不变；不能只跑一条全零输入。
- 同时最多一个带宽主上传进程，CPU构建可与其并行。实测带宽低于计划时减少重复层，不重建CUDA大环境；已上传层复用而不是从头重试。
- 单一持久registrar持有quota ledger与锁。每候选状态：`FROZEN -> CONTAINER_PASS -> IMAGE_PUSHED -> REGISTERED(id) -> REMOTE_TERMINAL`。
- POST超时先按team/phase/候选method与digest查询是否已经注册，未知时不盲目重发。不能把短暂failed/空result自动当应再占一席。
- private提交。TTA/外部预训练等声明依据实际实现，不机械照抄旧标签。本文所有候选设计为静态合法prefix适配，无query端更新；若实际不符停止注册。
- 每60s记录后台上传字节/速度、build状态、remote active数；每10min汇总。不需要用户坐在电脑前批准每次动作。

## 9. Executor分工与异常汇报

延续最多4agent：coordinator负责quota/冻结表/注册和公共容器验收；D worker负责两M2 runtime；C worker负责P整改与训练；H1 worker负责新Transformer FLAT/ROUTE适配、训练及H1 runtime。M2 runtime完成后D worker转为交叉审计/容器集成，不再另开一组worker。不能由P作者自己用固定true放行P。

只有独占owner可修改本次新namespace；所有worker明确“不独自抢GPU、不改旧根、不向用户反复请求系统permission”。H1/P训练run与每个submission artifacts各自新建根，引用而不覆盖长程结果。

需要立即汇报：额度不足6、服务端暂停、parent缺失、P硬门未解、H1真实形状成本无法达标、runtime无法兼容、并发槽预计不能及时周转、预计截止前不足6个合格新候选。报告已完成ID、各席障碍、预计可交付数量与唯一建议；其他可行席继续。

普通local负分并不自动取消本次预先指定的合法方法对照；但必须完整披露，不能描述为“最优产品已证明”。

## 10. 今天交付什么

`SIX_SLOT_LEDGER.json`：候选、task、选点面/候选集合、实际completed epoch、训练authority、payload/model/image digest、上传时间、server submission ID、官方UTC提交时间、重复检查与状态。

最终表的六行必须是实际新ID或明确 `NOT_REGISTERED + reason`，不能把READY/UPLOADING当提交完成。今天形成的early-snapshot、fold-parent和visible-product结果不改写长程配对/12epoch/LODO统计权威。

本补充编制时只读确认：phase每日6/并发3、当前UTC日0条、历史30条无分页遗漏，M2 small/large checkpoint与H1 C2字节存在。**尚未证明新的B容器已能运行、P整改已通过或H1新训练已完成。** 这些是执行队列的工作，不是文档已经完成的事实。
