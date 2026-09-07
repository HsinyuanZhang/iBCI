# 实验二工单 V2：688低成本跨session B-transformer与校准输入配对效用

> **2026-09-07最新授权覆盖：本稿降级为历史起跑方案，不再是688的强制执行合同。** 负责人已明确：只固定B-transformer核心结构与M2-like T4计算方法，网络参数、训练方法、数据划分、支持预算与PMUA对照由接收方agent根据结果自主设计；**PMUA是负责人重点推荐的高优先级主线，不是可有可无的附录，不依赖SUA正结果才启动。** 下文关于固定48k/seed/配方、6 GPU-h、禁止PMUA、禁止重划分或新训donor、以及逐项需再授权的限制，不约束当前688研究。最新权威入口为[远端研究交接指南](HANDOFF_DANDI688_BTRANSFORM_TEAMMATE_20260907.md)。固定方法、诚实报告与数据角色透明仍适用；其他任务权限不变。下文保留是为追溯前稿，不应用其限制覆盖最新授权。

日期：2026-09-07。状态：**审核意见已收口／文档定稿／待执行授权**。按用户指令完成修订，不再安排一轮文档审核；本工单不启动训练、不解封测试集。

## 0. 意义、独立问题与优先级

本实验回答：**在DANDI 000688上从头训练匹配decoder后，E0和显式方向carrier分别是否提供条件增益？B系统是否具有本地跨session竞争力？**

- FULL−NOE0：有carrier时E0的增量效用，承接[EXP1](WORKORDER_EXP1_E0_RELIANCE_V2_20260907.md)不能回答的重训问题。
- FULL−NOC：有activity-only E0时显式carrier的增量效用。
- 与旧SPINT/B0重放：系统级竞争力，不是共享输入的架构因果对照。
- 不依赖EXP1是否出现正结果；EXP1的干预跌分不用于有利选择NOC是否训练。
- 不继承[EXP3](WORKORDER_EXP3_STREAMING_STRUCTURE_V2_20260907.md)的depth/卷积变体；本轮固定D4与窗内补零，避免三个问题混在一起。

**最高优先级P0：M30支持/query合同、donor实际影响链、raw事件对齐及padding先闭合，再用真实688 D4步时封存预算。**
48k是候选正式horizon，不是已证明足够收敛，也不是“少于一epoch无效”的推论。

## 1. 所有权与授权边界

- 负责人拥有新增 `scripts/exp2_688_paired_v2/`、`tests/exp2_688_paired_v2/` 和独立结果/缓存目录。共享loader/模型默认行为只读；必要适配先在本实验模块隔离。
- 你不是工作区唯一执行者；不撤销他人改动、不停止活跃任务、不抢GPU、不覆盖旧缓存/normalizer/checkpoint。
- 待用户授权后按所分配设备执行；**训练/验证GPU预算建议合计6 GPU-h**，是本实验独立总量，不是每卡/每臂6小时。CPU预处理耗时另报。
- 不读取6个sealed-test文件，不做其神经输入预处理或扫描形状；不EvalAI、不推镜像、不修改论文、不git push。
- 将来独立测试需要冻结方案后另获一次明确授权；当前只有development结论。

## 2. 数据合同与新缓存

manifest：`sua_exploration/configs/subc_co_27_6_strict_train_val_manifest.json`。
SHA256：`4607e979c6c2ff451c147a8d9878fe1080b9d3e9bbc7304b559616eb2a13a0c9`。

- 仅 sub-C / CO / SUA；27 train、6 val、6 sealed-test；合同max_units_exclusive=100。
- 从manifest构造精确train/val路径allowlist后读取；禁止先glob全部NWB后过滤。日志记录每个实际打开的数据文件。
- 6 val有历史研究暴露，标visible development，不命名blind test。
- 20ms bin、W=50、2D behavior、训练raw=5y/评分pred/5，列序和scorer沿用并核验。

评分合同入口：`sua_exploration/scripts/eval_epoch_window_dandi688.py`（B0 receipt指定的确定性窗口评分入口）。复用其合法窗、mask、last-bin、target尺度及聚合规则，并记录代码SHA；若使用新harness，须以B0 e11在相同support/query坐标上做逐预测及指标一致性核验，默认FP32逐输出容差`abs(delta)<=1e-5+1e-5*abs(reference)`。对不上先解决协议差异，不直接套用历史分数。“M3”若出现在旧评分入口的协议名中，不是本实验的3-trial支持预算；本实验固定M30。

远端实现补充：旧epoch-window脚本含12epoch/e5..12聚合，不能直接充当48k固定终点评分器；也参考实际generic入口的调用链。底层 `evaluate_fixed_protocol_over_validation_sessions` 必须显式传严格 `train_val_manifest`，不得运行会自动discover全部NWB的旧顶层选择CLI。旧pool50/q50坐标不能冒充本轮M30/q30。

### M30/q30

按既定奖励trial筛选和时间顺序取前30个trial：M_activity=M_carrier=30。所有臂相同；不能因rank/缺cue到第31个trial补标签。

query位于第30个支持trial之后；必须核验**整个W50输入历史**不跨支持区及禁止边界。保留合法trial/mask/last-bin规则，冻结endpoint stream，不按误差、幅度或“好单元”挑选。

可复用 `sua_exploration/cache/dandi688_subc_co_v1` 的dense neural/behavior，但必须核对其bin起点、target对齐和valid_starts合同；同session多版本缓存按receipt SHA选定，不能随便取第一个。
已发现的 `calib_trials (10,100,N)` 是M10，不得充当M30。**新建M30 activity/E0与carrier缓存**；活动trial长度100与encoder输入整理沿用donor合同，不对其插值语义自行改造。为构建/验证新M30可读allowlist内NWB；不重建不必要的全部dense序列。

## 3. M2-like carrier：固定MOVE700，不新增GPU窗口网格

native字段为 `go_cue_time_array`、`target_dir`、`num_targets`。cue解析直接参考 `streaming_calibration_exp/src/data/rt_sparse_endpoint_loader.py::_go_cues`；`sua_exploration/mc_maze/rt_classical_comparators.py`为使用示例。`_go_cues`返回每trial一行的二维cue矩阵（可由ragged数组补齐，缺位NaN），**不会自动选择一个cue**。

封存 `event_contract.json`：cue矩阵/缺位语义、方向单位、trial起止边界及以下固定单目标规则：

- 每个参与MOVE700拟合的支持trial必须`num_targets`为有限整数且恰好等于1；第一个声明cue有限，其余位置全为NaN，恰有一个有效cue。Inf或其他非法缺位值不视为合法NaN。
- `num_targets!=1`、有效cue数不等于1、cue位置与声明不一致、缺cue/缺方向或方向非法：仅从前30支持trial的carrier拟合中排除并记录原因，**不取首cue凑单目标、不用速度推断方向、不读取第31个补足**。
- 按session报告前30trial总数、各排除原因及最终拟合数；CO单目标比例由数据审计给出，不先假定全部为1。
- 这项排除只改变carrier拟合有效行，不改变已冻结的M30 activity输入和q30边界；E0仍使用原前30活动trial。记录两条输入路径的实际有效数，不把不同路径偷偷重新对齐到后续trial。

每个前30支持trial m：

`S_mi = count(spikes_i in [go_m, go_m+0.700))`

`R_mi = S_mi / 35`，单位counts/20ms-bin；原始spike半开区间计数，不能用以首spike起始的dense grid假装精确事件对齐。

`A_m = [1, cos(theta_m), sin(theta_m)]`

`B = lstsq(A, R)`；每trial一行、trial等权、无ridge、float64求解、rank(A)=3。
映射为 `[a_i,d_i,sqrt(a_i²+d_i²),b_i]`，存float32。可复用 `streaming_calibration_exp/src/data/falcon_t4_features.py::t4_from_trial_sums` 数值核心，lengths全35；不复用FALCON读取器。

source27的raw profile有效行拟合feature-wise mean/std；std<=1e-6置1；val使用冻结统计；pad行不参与统计。归一化算法也随哈希封存，不覆写旧winsorized normalizer。

非法cue/方向、区间越出所属trial或侵入下一trial时，在前30内排除并披露；不夹短窗口后仍除35、不扩大support。不足满秩的session导致该主版本P0失败，不静默fallback到whole。

### 仅source的CPU sanity

比较旧WHOLE、新MOVE700、[go+0.100,go+0.600) counts/25三种descriptor：方向覆盖、有效trial、秩、condition number、尺度、split-half相关性与符号一致率。
split-half规则训练前固定，rank不足标unavailable；近零系数的符号一致率单列，幅度/截距的符号不能混作方向稳定性。
报告700ms窗超trial/与下一trial重叠比例。不得根据val R²选窗口、ridge或归一化；本轮主版本预定MOVE700。若source审计要求改定义，先形成新manifest再训练。

与M2共同点是trial等权OLS、四维映射和counts/bin尺度；事件锚点/时段不相同，不宣称完全相同carrier。与历史688比較是estimator bundle变化，不是只改窗口。

## 4. Activity-only E0 donor与影响链审计

候选：`sua_exploration/checkpoints/sua_spint_t4_mainline_fp32_v1_b0_dandi688_co_s42/epoch_ckpts/epoch_011.ckpt`。

核验权重SHA、实际config/state、E0输出宽度（不从hidden_dim推断）、side_dim=0、M30、trial_length100、W50、训练/val/test roster。只提取冻结id/calibration encoder；新B decoder从头初始化，不载入旧decoder。

teacher审计必须覆盖**实际计算影响**：是否提供初始化、蒸馏loss、target、预计算embedding、normalizer或选点依据。`task_only`不是完整证明；若teacher确实完全未影响所提取encoder，可用代码/运行记录闭合无依赖，无需凭一个无效路径强造泄漏结论。存在依赖则审计对应roster；无法闭合时停止source-only效用主实验，报告需要的合法donor，不自动新训。

源encoder用过27训练session属于预训练成本；两个decoder seed共享donor，只证明decoder seed稳定性。
activity-only指输入没有显式方向carrier，不表示E0不含方向相关信息；NOC只能称“移除显式carrier输入”。

## 5. 固定几何与三臂

固定：proj_add P16、local k5/16、token width256、8 slots、temporal_layers=4、CausalPE、conv_pad=in_window_zero、window=50。

**units=100**，所有session按native列序pad至100；x/E0/归一化后c的pad行=0、unit_mask=False。pad行不进attention，但仍可能消耗token MLP/KV计算。
E0优先按donor原生有效N生成后再pad；如果donor自身固定宽度，必须使用其经过验证的mask/padding流程，不能先塞假单元改变encoder输出。
unit dropout仅作用有效行，每窗至少1个有效单元；source统计与MEAN等操作排除pad。

| arm | 训练与推理E0 | 训练与推理直接c |
|---|---|---|
| FULL | activity-only真实E0 | MOVE700真实c |
| NOE0 | 全0 | 同一真实c |
| NOC | 同一真实E0 | 归一化后全0 |

保持图/参数形状，使用输入gate而非换identity_mode。同seed共享**相同初始state字节**、session/trial/endpoint流、dropout RNG、优化器、步数、EMA；encoder/bank冻结。
记录两种零输入分支的梯度检查。零输入导致对应数据梯度为零，但AdamW的decoupled weight decay仍可能改变参数；不得写“权重必定停在初始化”。

三臂给条件增益，不估计完整E0×c交互；不默认增加NOE0+NOC第四臂，不加RT/sub-M/MUA/P32/其他时间核网格。

## 6. 先测预算，再封存唯一训练矩阵

### S0：吞吐与协议门（不读val分数）

在真实688 D4/units100/B32、拟采用的loader和mask上，以低/中/高有效N源session测warmup后至少100步，重复3段；记录forward/backward/optimizer/EMA/数据加载、显存与最慢稳态步时。FP32 smoke通过后统一bf16训练，loss用float预测。

候选horizon=48,000、warmup=200、AdamW lr1e-4 cosine→1e-5、wd0.01、clip1、unit dropout0.1、EMA0.9995、effective batch32；这些是封存配方，不承诺最优。
每step单session；session等概率，内部trial尽量等概率、抽取分散的合法endpoints；各臂同stream。更新数不是独立样本数，报告实际trial/endpoint覆盖。

预算两档在任何val分数前选定：

| 档位 | 同预算终点矩阵 | 总训练updates |
|---|---|---|
| A（优先） | 三臂×seed42/43，全部48k | 288,000 |
| B（资源受限） | seed42三臂48k；seed43仅FULL/NOE0各48k | 240,000 |

费用估算：`1.30 * (训练总updates×实测最慢步时 + 验证/保存/数据GPU占用秒数) <= 21,600秒`。此处余量定义是估算成本乘1.30，不是把总预算先乘0.70；两种口径不得混用。A档仅训练时的必要步时上限约57.69ms，**不是充分条件**。
两档都不满足：在val读分前提交较小共同horizon/预算新manifest供确认；未确认不进入正式矩阵，不见结果后随意扩步。
不再按NOC pilot是否有利选择晋级；B档的carrier结论明确只有一个decoder seed。

### S1：8k健康检查（包含在终点预算内）

seed42三臂均到8k RAW检查。各val group最多预固定2,048点，覆盖全部6session。常数基线固定为该session前30支持trial内有效原生20ms行为bin的逐输出均值（bin等权、排除padding/无效值、不用插值重复帧加权）；保存使用坐标/有效数/均值SHA，在query全部点输出该native-scale均值。无有效支持行为则标baseline不可用并诊断，不回退query均值。各模型与基线共享query坐标和scorer；该均值拟合不使用query标签。
仅排查NaN、端口未接通、尺度错误、异常mask或近常数输出；pred std<1% target std且无改善作为诊断信号，不是自动科学否定。
出现实现错误：暂停整组，修复形成新版本，必要时所有臂同起点重跑；不能仅修补失败臂并沿用其他臂旧结果。
仅R²低/三臂接近不自动停一个臂或取消seed；健康模型按已封存矩阵继续。

### S2：固定终点比较

16k/32k保留固定开发读点，主结果只用48k endpoint EMA；RAW辅报，不按val选epoch/seed。续跑保留optimizer、schedule、EMA、RNG/stream offset，不重置。
最后六个val全合法query评分；中途只用固定诊断子集，不每500步全val。
若实际资源超限，停在可恢复checkpoint，标 `BUDGET_INCOMPLETE`；不能靠少跑某个不利臂凑完整结论。

## 7. 指标、参考与裁决

- 主指标：每session variance-weighted R²，再6session等权平均；pooled辅报、全部session/seed原值公开。
- 配对差值：FULL−NOE0、FULL−NOC，严格同seed、同horizon、同points；多seed先各自报再平均。
- 同一对照两个seed的equal-session效应符号相反时，主定性结论为`UNCERTAIN`（可注明近零），不以平均后的正号覆盖方向冲突；原值与区间仍完整报告。
- 2,000次预固定seed的paired session bootstrap，区间只描述开发session不确定性；2个decoder seed不足以估充分训练方差。
- 分别报告训练曲线、val固定读点、pred/target std、实际预训练/缓存/decoder训练/推理成本。
- 同面重放B0 e11原decoder与常数参考；B0接入支持/query不匹配则不能直接引用旧分数，修正合同或标不可比较。
- E0增益均值>=0.01且两seed/多数session同向是实用支持；±0.005量级为小效应参考，不是统计等效证明，区间与逐组结果优先。
- 48k仍明显进步：预算下证据有限；曲线稳定而差异小：报告小效应/不确定。**不能把零差异唯一解释成未收敛，也不能认为48k必然收敛。**
- NOE0/NOC更好照报，不以得到正增益为验收条件；不自动延长到正结果。
- B相对历史系统的差异混有encoder/估计器/训练预算，不将其归因全给架构。标题若强调新架构优越性，还需另立共享校准输入的matched decoder对照，不在此偷偷扩scope。

## 8. 交付与执行前验收门

结果根：`btransform_unified_v1/results/exp2_688_paired_v2/<UTC>/`；新缓存在该根或独立同名version目录。

交付：`manifest.json`（档位/步数/读点/随机流）、`data_access_audit.json`、`support_query_audit.json`、`event_contract.json`（逐session排除计数）、`donor_dependency_audit.json`、`carrier_sanity.json`、`scorer_parity.json`、`support_constant_baseline.json`、`cache_index.json`/SHA、`throughput_budget.json`、每臂初始化与最终权重SHA/可恢复状态、逐group/seed表、`decision.md`、完整实际CLI/环境/代码版本及失败日志。

执行前最高优先级自检：

1. test是否从未打开，M30与全输入历史隔离是否可证，M10缓存是否排除？
2. donor/teacher实际影响链、归一化来源、cue多值规则是否闭合？
3. padded单位有没有影响encoder/source统计或泄入attention？
4. 预算是否真在688 D4上测；矩阵是否在val读分前固定？
5. 三臂是否匹配初始化/RNG/步数；零效应是否被如实接受？

验收分 `PROTOCOL/DONOR/BUDGET_PASS`、`TRAINING_COMPLETE`、`E0_EFFECT`、`DIRECT_C_EFFECT`、`SYSTEM_COMPARISON`，不合并为笼统“有效”。B档不能称两seed carrier复现。

来源：[原688设计](DESIGN_688_BTRANSFORM_COST_AWARE_V1_20260907.md)、[初轮审核](REVIEW_E0_UTILITY_AND_688_COSTAWARE_V1_20260907.md)、[V2审核](REVIEW_WORKORDERS_EXP1_EXP2_EXP3_V2_20260907.md)。审核原文保留；本V2修订稿作为独立执行合同，不修改其他活跃任务权限。
