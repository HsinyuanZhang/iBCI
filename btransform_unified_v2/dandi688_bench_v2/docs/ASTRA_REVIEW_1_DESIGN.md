# Astra 第一轮独立设计与实现审核

日期：2026-09-11。结论：**PASS_WITH_FIXES**。

**2015-only 的两套表示独立 encoder 预训练及六个 seed42 decoder，在 source-only B32/CUDA bf16 检查通过后可以正式运行。** 未发现需要重建现有 source/dev 缓存、改变 MOVE-T4 或增加主矩阵格子的原因。CPU baseline 校准神经历史边界已在本轮复核修复通过，现可启动正式网格；首次 final 前须补足执行代码、选择规则与硬件坐标的校验。以下条件均为明确的软件或记录修复，不要求用户再次批准已授权实验。

本轮审核只读设计、实现、测试、官方 MOVE-T4 核验材料和现有 24-session source/dev 缓存。未运行训练，未打开任何真实 final NWB 或 final 缓存。既有 37 tests 和 CPU smoke 是工程证据，不替代本轮判断；B32/CUDA 检查由运行负责人另行执行，本报告不将尚未收到的结果视为通过。

## 审核范围与可解释的结论

主设计为 `../../docs/DESIGN_DANDI688_SUA_PMUA_V2_20260911.md`；实现审阅覆盖 `protocol.py`、`data.py`、`carrier.py`、`model.py`、`training.py`、`common.py`、`baselines.py`、`baseline_runner.py`、`finalize.py`、`final_access.py`、README 和相应测试，以及实际调用的 causal numerics。

15 格主表可以回答每个神经臂内 SUA 与 pooled sorted-unit PMUA 的配对差异，并比较所声明的部署方法。它不能把 Full−ACT 解释成目标角度标签的纯因果贡献：Full 额外接受 source encoder 预训练，随后冻结；ACT 联合训练。WF-FSS 读取 dense velocity，而 Full 读取 trial angle，不能解释成等标签信息预算胜负。单个 seed42 主表也不能代表训练随机性的稳健性。这些界限已在主设计披露，均不要求扩增主矩阵。

## 已通过的关键检查

| 检查 | 判断与证据 |
| --- | --- |
| 2015-only 划分 | `protocol.py:26` 校验 manifest SHA；`protocol.py:38` 固定 18/6/6。prepared receipt 实际含 24 sessions、18 train/6 dev、24 条访问记录、`final_sessions_opened=0`。普通 loader 的 source/dev 权限在 `protocol.py:56`、`common.py:98`；正式训练禁止注入缩减名单在 `training.py:179`。访问收据证明正常程序路径的声明，不是对所有外部进程的监控。 |
| SUA/PMUA 公平性 | `data.py:61` 根据 M1 electrode metadata 选 units；`data.py:228` 把这些相同 spikes 按硬件电极合并，并在 `data.py:240` 验证逐 bin 总数守恒。两表示共享时间、速度、query；padding 在 `data.py:272`，无截断。PMUA 含义是 sorted-unit pooling，不能称 threshold-crossing MUA。 |
| M33 与 Q50 隔离 | `data.py:199` 从 rewarded index50 生成 R50 query；`data.py:202` 只取前33个合法 rewarded trials 活动。额外只读检查24份 SUA缓存的索引，Q50 的完整 R50 历史与 M33 support 的交集总计为0。 |
| 正式 MOVE-T4 | `carrier.py:42` 排除非有限角度，`carrier.py:46` 检查至少3行、rank3，`carrier.py:53` 用 spike sums/25 做 OLS 并输出 `[a,c,hypot,baseline]`。与 `streaming_calibration_exp/src/data/falcon_t4_features.py:76` 和 `tfpd_exploration/src/m2_dual_track_v1/champion.py:360` 实际调用一致；审阅 `results/smoke_2015_m33_v2/MOVE_T4_AUDIT.json` 的582292链路。DANDI `[GO+.1,GO+.6)` 为明确任务锚映射，非跨数据集绝对时间等同。 |
| Full 源预训练 | `training.py:169` 明确以 source velocity 联合训练临时 decoder+encoder，再固定末段 EMA 导出；`training.py:193` pretrain 不载入 dev；`training.py:269` 只导出末段；`training.py:70` 验证表示、source统计、18source与预算。主 Full 在 `model.py:137` 冻结 encoder。SUA/PMUA 各自训练对应 encoder，这一设计通过。 |
| 三臂目标信息 | `training.py:49` Raw 不建立校准输入；ACT仅活动；Full另生成T。`model.py:232` 目标校准在no-grad下前向计算；Raw使用共享g16，`model.py:277` 无unit identity table。 |
| 采样与监督 | `common.py:193` 的session-balanced批次在每段最多差1，endpoint各session独立打乱循环且不依赖表示；`training.py:233` 使用同seed流。velocity统计仅18source Q50，`common.py:132`；carrier分别用表示自身source拟合。统计按query加权，训练按session平衡：两者作用不同且不造成target泄漏。 |
| 网络和正式预算 | `model.py:126` 复用M2 learned R50/D4、P16与local temporal；`common.py:20` 固定24×3165、B32、AdamW/EMA等；`training.py:230` 执行完整预算，24段全dev earliest-max在 `training.py:274`。参数初始化、mask、梯度和数值oracle已有针对性测试。 |
| 分数及失败策略 | `common.py:172` 原物理速度variance-weighted R²，`common.py:185` 等日期均值；FA秩与收敛门在 `baselines.py:192`、`:234`，整dev可评才入选。final FA若有日期失败，`finalize.py:302` 整行UNAVAILABLE，不用剩余日期均值代替。 |

## 必要修复与执行门槛

### R1 — 校准的因果历史超出严格 M33 neural 支持（P1，已修复并复核通过）

本条保留初始发现作为审计轨迹；下列初始行号对应修复前实现。负责人明确采用strict M33，不扩展预算。修复后 `baselines.py:354` 的 `_calibration_stream` 在平滑前先将support外raw置零，平滑后再次置零；`:417` target adapter与`:453` reference均使用该stream；`:501` WF-FSS标准化后再次mask再构造lag。仅读query时保留full causal stream。

本审核已独立阅读修复实现及新增支持外±1e6扰动测试；运行负责人报告12 tests passed。已读取 `results/baseline_smoke_m33_maskfix_two_source/receipt.json` 和 `selection.json`，确认SMOKE/no-final、7methods全部eligible=True。**R1闭合：允许CPU正式144候选网格，无需重cache或重神经训练。**

证据：`baselines.py:342` 先对完整session调用`smooth_raw`，再在`:347`选择支持。`external_baselines_v1/fair_v2/numerics.py:23` 的12tap平滑不重置trial，因此M33支持行可以吸收trial间隙/先前trial的神经输入。`baselines.py:477` 的WF-FSS对carrier端点调用H50历史；`numerics.py:315` 同样按整条stream取历史。

直接只读24份source/dev SUA NPZ的 `support_indices`、`carrier_indices`、`query_indices`：共有19,800个MOVE标签行；H50校准历史中有 **223个输入位置（按重复使用计数）不属于M33 support_indices**。该数字未计平滑额外11bin历史。标签本身仍严格在MOVE窗口，且未发现未来query标签泄漏；问题是实际校准neural范围比“仅M33 neural”宽。

必要修复：在拟合校准统计和WF-FSS特征时只允许M33支持内的neural输入，明确支持外历史的补零/截断政策；同一政策用于source参考与target。保持query推理的既定因果流与共同Q50端点。不得只改文字继续宣称严格M33-only；若负责人选择保留完整causal prehistory，应在正式网格前明确将其冻结为额外neural预算，并说明这比原严格定义宽。

验证：构造支持外neural扰动，检查拟合的统计/adapter/WF-FSS系数不变；跑真实source上的七baseline定向smoke。**无需重cache；不阻止两encoder及六decoder；需在正式baseline网格前闭合。**

### R2 — final seal 未绑定完整执行代码（P1，首次 final 前）

证据：`common.py:75` 已能记录所有直接及共享实现的源码SHA；训练和baseline receipts保存了这些SHA。但 `finalize.py:225` 在seal中仅把`protocol.py`作为执行源码artifact；`final_access.py:40` 只检查seal列出的artifact。修改`model.py`、`training.py`、`baselines.py`、`data.py`或共享numerics后，旧seal仍可通过，即使checkpoint字节未变。

必要修复：seal绑定完整最终执行源码集合；评分前核对当前代码与seal。把训练/选择时的代码provenance与当前实现差异显式检查，允许有记录的纯运行支持修复，不得让改变模型/特征/评分语义的漂移静默通过。新增代码漂移拒绝测试。

**无需重cache或重复CPU训练smoke；不阻止正式训练；首次final前必须通过封存定向测试。**

### R3 — static-controls seal 不验证完整候选与选优（P1，首次 final 前）

证据：`training.py:327` 正常runner确实跑diag1/coral4配置和完整dev；但 `finalize.py:196` 仅检查index合法、checkpoint一致与存在shrinkage，未检查候选数量和值、每候选完整6dev、分数有限、earliest-max。baseline seal在`:133`也主要验证artifact绑定，不复核所声明候选grid完整性和winner是否等于合法最大值。神经封存在`:108`已具有更强的完整曲线检查。

必要修复：对static-controls校验预定1/4配置、每候选6日期、有限指标和earliest-max；对baseline校验各预定网格数量/配置及被选项等于全6dev合法最大值，仅FA可明确UNAVAILABLE。对selection、prepared cache、source/dev binding做一致性关联，防止混用不同缓存版本的完整名单receipt。

**无需重cache、重神经训练；已有正确runner产物可直接通过新增检查；首次final前做缺日期/删候选/错winner拒绝测试。**

### R4 — final canonical table 未与 source 参考表比较（P1，首次 final 前）

证据：`data.py:61` 保证每session有96个唯一M1硬件key，但不保证key集合跨日相同。`common.py:112` 只验证被同次载入的source/dev records一致。`finalize.py:332` 载入final pair后立即进入评分，未将final keys与已封存prepared canonical table比较。PMUA adapter在`baselines.py:54`依据整数坐标求交；若final table变化，整数相同可能代表不同硬件。

必要修复：final载入后、任何预测前，以完整canonical keys或prepared中canonical table SHA检查全部final表一致；失配必须明确失败，不能继续按旧整数对齐。可以用合成SessionData测试，不提前打开真实final。

**无需重cache或重训练；只在首次final入口增加必要的物理坐标验证。**

### R5 — GPU 正式规模尚待实测（P2，正式神经训练启动条件）

证据：`training.py:248` 实现CUDA bf16，既有smoke走CPU、batch2；它不能证明B32的显存、autocast梯度或吞吐。正式dev推理在`training.py:127`中fp32执行，与训练autocast不同但选择/最终预测应保持同一数值策略。

必要执行检查：仅source数据，实际B32在CUDA下覆盖pretrain、Full冻结交接、ACT和Raw的前后向，记录loss/grad finite、峰值显存、耗时、checkpoint重载；短检查标SMOKE不可选点。无需等待144个CPU候选完成才开始神经训练。运行负责人已有独立GPU验证任务，本报告等其证据闭合。

**不需重cache；这是新的GPU检查，不要求重复全部CPU smoke。**

## 不阻止开跑的运行事项

`finalize.py:248` 的`load_trained_model`使用默认CPU，`run.py`的score-final目前没有device入口。完整神经final和两个adapter可耗费较长时间；建议在final前增加固定device参数与明确fp32推理路径，用已选dev checkpoint检验CPU/GPU预测及分数容差，然后封存。设备选择不应根据final表现调整。此项不改变模型、预算或主矩阵，不阻止GPU训练。

`training.py:263`保存EMA模型但没有optimizer/sampler/RNG恢复状态，`fresh_directory`也禁止覆盖。正式训练中断需重启该任务；应按运行稳定性决定是否实现严格等价resume。它是效率限制，不是设计有效性失败，不要求为此推迟已可运行的完整任务。

Full预训练与主训练合计监督更新多于ACT/Raw，且full encoder是有source velocity监督的，不应称无监督或全部神经臂同总计算量。主比较保持每臂内SUA−PMUA；共享source encoder的多decoder seed不会覆盖encoder预训练随机性。

## 多 seed 计划判断

负责人已在本轮明确固定采用主设计允许的Full成对seed43/44补充：**该预先固定计划通过，首次final前完成全部4个附加decoder，不根据某个seed的dev结果决定是否继续。** seed42保持15格主表；Full另报SUA/PMUA各seed42/43/44的值、均值与样本SD，以及每个相同seed的SUA−PMUA差值。不要选最好seed；每seed仍按完整24段dev earliest-max选checkpoint。三seed共享表示自身seed42 encoder，SD仅描述decoder训练随机性。

封存必须显式表示这4个补充decoder及其独立checkpoint/完整dev曲线；在同一次final访问中评分并归入补充结果，不改写seed42主15格。当前`finalize.py:102`和`final_access.py:14`仅支持固定seed42主矩阵，扩展需在首次final前通过合成测试。这是既定可选计划的固定化，不要求增加ACT/Raw或baseline seed矩阵。

## 第二轮复审输入与完成标准

第二轮只需提供：R1–R5闭合证据；两encoder末段导出receipt与表示/source绑定；各已完成段的完整dev-6分数、loss、吞吐/显存与采样hash；CPU候选资格和失败原因；Full补充seed固定计划。复审检查是否偏离设计、配对结论是否由个别日期驱动、是否有数值或覆盖异常，不据中途结果扩增超参搜索或改变15格主表。

本轮结论的边界是：**神经设计通过；GPU检查通过即可启动正式训练。baseline支持边界已修复通过，现可跑全网格。R2–R4及补充seed封存扩展未完成前，不可开启真实final。**
