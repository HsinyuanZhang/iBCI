# Astra 第三轮：concat 迁移、GPU 执行资格与 final 前审核

日期：2026-09-12。**新 concat 定义及正在进行的正式训练：GO。真实 final：尚未具备整体进入条件。** 两个 encoder 正式任务已各完成至少 3 段、9,495 updates，不能将它们称为已完成预训练或新神经泛化结果。本轮发现补充 SUA seed 的 final 表示解析错误，已立即通知负责人；该问题不影响训练，须在首次 final 前修复并用默认 scorer 路径验证。

本轮独立只读源码、source/dev smoke 与 GPU profile receipts、正式 pretrain protocol/progress 和 checkpoint 文件哈希；仅新增本审核文档。未运行训练、未打开真实 final NWB/缓存、未修改 Python 实现。父执行负责人报告的 CPU 56 tests 和 final 定向 24 tests 不记为本审核重新运行；本轮直接检查其相关测试源码及实际产物。用户最新授权继续 GPU 全实验，覆盖第二轮报告中的暂停安排；旧 FiLM 片段仍不具备新 concat 资格。

## 科学定义与训练合同

`model.py:B3SIdentityEncoder` 现为 `Linear(100,64)+ReLU → M33 trial mean → concatenate MOVE-T4 → Linear(68,64)+ReLU → Linear(64,64)+ReLU → Linear(64,50)`。Full 的 side 列初始化为零，但可在 source 预训练中学习；无 FiLM。ACT 使用 side_dim=0，拒绝 T 输入。Raw 不构造 encoder，使用共享可学习 g16。Full decoder 另接收 T4；零初始化 side 列只保证 encoder E0 的初始 side 独立性，不意味着 Full 总预测初始不依赖 T。

DANDI 的本地 B3S wrapper 保留 T100/E50 几何，decoder 继承 M2 learned-RIFT、P16、proj_add、R50/D4，并安装 local temporal。当前代码没有将 M1 的 T1024/N64 硬编码 wrapper 生搬到 DANDI。三个神经臂的信息定义仍与 18source/6dev/6final、2015-only、strict M33 的既定设计一致；本轮没有依据改动缓存、矩阵、选参网格或训练预算。

`training.py` 在 pretrain 阶段不加载 dev，encoder 和临时 decoder 均可训练；只在第 24 段 EMA 应用期间导出 encoder。`DecoderEMA` 跟踪 requires_grad 参数，pretrain 因而包含 encoder；Full 主训练将 encoder 所有参数冻结，EMA 和 optimizer 不再更新它。Full 主 decoder 重新按自身 seed 初始化，各表示的 seed42/43/44 共用自身 seed42 source encoder。ACT 联合训练自身 encoder，Raw 从头训练。每项正式预算为 24×3165、B32，decoder 每段完整 dev-6，严格 `>` 更新 winner，平分保留 earliest maximum。

`run_campaign.py:plan` 每表示固定 6 项：pretrain42、Full42、ACT42、Raw42、Full43、Full44；两表示共 12 项正式任务，包含 10 个可评分 decoder。最终必须保留 15 格 seed42 主表及 4 个 Full 补充格，共 19 格；两个 encoder 本身不是 final 格。Full 的三 seed 均应报告，均值与样本 SD，以及每个配对 seed 的 SUA−PMUA；三 seed SD 仅描述 decoder 随机性，不包含 encoder 预训练随机性。Full−ACT 也仍混合预训练监督量及冻结策略差异，不是目标角度标签的纯因果消融。

## 当前执行证据

审核读取的 `results/gpu_profile_concat_20260912_{sua,pmua}/receipt.json` 均为 source-only、PROFILE_ONLY_INELIGIBLE_FOR_SELECTION、final_sessions_opened=0。每表示覆盖 full_pretrain、full_frozen、activity、raw_set，每臂 B32、8 steps，所有 loss/gradient finite。

| 表示 | pretrain updates/s | Full frozen | ACT | Raw | 各臂峰值 allocated MiB 范围 |
| --- | ---: | ---: | ---: | ---: | ---: |
| SUA | 35.81 | 37.49 | 37.06 | 37.10 | 1055.58–1080.69 |
| PMUA | 35.73 | 37.57 | 37.31 | 38.50 | 1055.58–1080.69 |

这些数值是短训练步性能，不能推算包含完整 dev 推理的总工期。profile 的 Full frozen 使用短预训练对象，是可运行性检查，不能替代正式 encoder 末段交接。`results/smoke_concat_2015_m33_v1` 六个 decoder receipt 均记录 in-memory EMA 与 reload 的 8-query 预测逐值相等、max_abs_delta=0；这补足 CPU 序列化检查，不等于 GPU→CPU 全 dev 数值等价已验证。

正式目录 `results/formal_campaign_concat_2015_m33_allseeds_20260912` 的两套 pretrain 在本轮采样时均为 3 段、9,495 updates，所有 development=null；独立重算全部 6 个 segment checkpoint SHA 均等于 progress 记录。SUA 三段 mean_loss 为 0.540640、0.295596、0.498939；PMUA 为 0.568233、0.319395、0.521006。有限但非单调的预训练 loss 不构成中途改变预算或选择 encoder 的理由。尚无可供本轮解读的正式新 decoder dev 曲线或 final 表现。

## 代码绑定与并行修改

逐文件比较两套正式 pretrain `protocol.json.code_hashes` 与当前文件：仅 `final_access.py`、`finalize.py`、`run.py` 已变更；本地 `model.py`、`training.py`、`common.py` 及记录的共享 M1/M2/learnable_recency 文件全部匹配。因此未发现已启动训练的数值路径漂移，无需因 final 支持修复重启本轮训练。

`provenance.py:execution_dependencies` 对 neural 纳入共享三个目录的全部顶层 `.py`，范围大于实际 import 闭包，包括独立 joint/cross_session 模块。之后并行修改任何这些文件，会使当前严格哈希核验拒绝封存，即便改变的是 DANDI 未调用的模块。应保存/冻结当前共享实现；若确有变化，只能逐文件确认实际依赖后记录明确的非数值迁移，不能删掉所有共享依赖检查或重写旧 receipt 哈希来使其通过。现有正常 Python 进程不自动重载模块，后续 campaign 子进程却会重新 import，尤其不能让真正依赖的 model/temporal/schedule/EMA 文件在各任务之间漂移。

## R2、R3、R4 状态与本轮发现

**R2：执行源码封存主体已实现，encoder 全链路证据仍待正式产物完成。** `bind_execution_sources` 将当前全部执行源码放入 seal artifacts；FinalAccess 在首个 raw open 前及评分间复验 artifact 字节。训练依赖按阶段核验；旧 CPU common 的例外仅接受固定旧 SHA/snapshot 且剔除 SCHEMA/RECIPE 后 AST 完全一致，不能掩盖 metric/helper 改动。当前 CPU 144 网格不需因 concat 元数据迁移重跑。

但 `load_encoder` 尚未自动验证 encoder receipt 中的 code_hashes、selection 字符串及 encoder_seed=42，`_neural_entry` 也未把对应 encoder.pt/json 加入专门的预训练证据链。正常 campaign 确实走预定路径，尚无错误交接证据。首次 final 前应独立校验两份完整 pretrain receipt、75960 updates、24 段 dev=null、source/表示/代码绑定和固定末段 EMA，并检查所有 6 个 Full checkpoint 的 encoder state 与其对应 export 逐张量相等、记录的 encoder SHA 一致；不能把仅加载成功当作全部 provenance 已验证。

**R3：baseline/static 的原缺口已在实现中闭合；最终神经产物仍需全曲线审核。** baseline seal 重建预定网格，检查完整候选、全 6 日期有限分数、等日期均值、earliest-max、prepared binding，只有整行 FA 可以有明确 UNAVAILABLE。static seal 固定 diag 1 配置/CORAL 4 配置、全 dev 指标、earliest-max 及所用 Raw PMUA checkpoint。tests 包含缺网格、错 winner、cache binding 的拒绝检查。真实 CPU 结果沿用第二轮独立预测重算结论。

神经 `_neural_entry` 的曲线检查仍较 baseline/static 弱：它检查日期集合和 n_sessions，但不逐日期验证有限数值、query 数量、均值重算及 source_binding 对当前 prepared 的一致性，也不直接核对已选 payload 的 arm/representation/seed 与 receipt。这不证明正常 runner 产物有错，但首次 final 前必须对 10 个实际 decoder 完成这些独立核对；允许在不改变训练数值路径的封存代码中补强自动校验。

**R4：坐标检查实现正确，负例证据待补。** `score_final` 在任何 score_cell 调用前，将所有 final 表示的 canonical electrode keys 与 sealed prepared source 参考逐一比较，不一致即失败。当前所读测试只见等值占位表的 happy path，未见 canonical mismatch 导致 scorer 从未调用的独立负例；已要求负责人补合成测试，不提前读取真实 final。

**新发现 F1（P1，首次 final 前必须修复）：补充 SUA seed 被错误识别为 PMUA。** 本轮发现时 `_default_score_cell` 以 `cell.endswith('_sua')` 判断表示，因此 `full_sua_s43` 和 `full_sua_s44` 被分配 PMUA records/source。`score_final` 写预测的 query/velocity SHA 也存在同样判断。轻则 source stats 验证拒绝、浪费一次 final claim，若未拒绝则造成科学上无效的表示评分。应统一按明确格子映射/去掉合法 seed 后缀解析表示，并验证默认 scorer 确实给 SUA 补充 checkpoint 传入 SUA source/final，同时保存正确表示的预测绑定。仅检查 callback 收到 19 个名字无法覆盖该错误。发现已即时发送父负责人；本报告不将正在安排的修复提前记为已通过。

## 进入 final 的剩余条件

1. 固定 12 项正式任务全部完成；10 个 decoder 各自完成 24 段全 dev 选择，两个 Raw PMUA static adapter 完成既定 dev 候选；不能依据途中分数取消 seed43/44。
2. 闭合 F1 及 canonical mismatch 合成负例；19 格必须无缺失、无重复，seed42 主表与 Full 三 seed 补充都按既定规则报告。
3. 对正式 encoder/decoder 执行上面的 source、code、EMA、冻结和完整曲线核验，复查 campaign 全部神经任务真正数值依赖未漂移。只有最终完整产物通过后才封存。
4. 固定评分设备与 fp32 推理策略。若正式选择在 CUDA 而 final 拟换 CPU，先用已选 dev checkpoint 核验数值容差并记录，不能根据 final 表现选择设备。

上述工作不需要改变科学定义、重建 source/dev 缓存、重跑已合格 CPU 144 候选或停止当前合法训练；它们是首次真实 final 前的具体闭合事项。本轮 GO 仅覆盖已审核的 concat 定义和正式训练执行资格。

补充归档核验：本报告写入期间，负责人新增 `EXECUTION_SOURCE_SNAPSHOT.json` 和 `START_VERIFICATION.json`。本审核独立重算 snapshot 中全部 48 个源码文件 SHA，并逐项与 SUA/PMUA 两份 pretrain protocol 对照，全部吻合。START_VERIFICATION 明确记录 12 个正式任务、15+4=19 格及每任务完整预算。归档提高可追溯性，不代表后续任务自动从快照目录执行；后续仍应核验实际 import 文件。
