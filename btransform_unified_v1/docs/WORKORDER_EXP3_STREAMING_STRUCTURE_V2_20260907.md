# 实验三工单 V2：流式前向的深度与历史卷积结构取舍

日期：2026-09-07。状态：**审核意见已收口／文档定稿／新增cell待授权**。按用户指令完成修订，不再安排一轮文档审核；本工单不是启动新训练或正式提交的许可，既有深度线仍按其原授权处理。

## 0. 科学意义、分工与最高优先级

问题：**保持proj_add校准接口与时间token窗口，在可接受质量变化内，能否通过较浅时间核和历史卷积降低流式解码成本？**

意义：把“解码器能离线算分”推进到“流式成本可解释、可验证”，并区分参数深度改变、观测历史改变和纯执行优化。
本实验不证明E0效用（见[EXP1](WORKORDER_EXP1_E0_RELIANCE_V2_20260907.md)、[EXP2](WORKORDER_EXP2_688_PAIRED_UTILITY_V2_20260907.md)），也不替代共享输入的B-vs-SPINT架构对照。

**最高优先级：先定义history-conv的合法输入与reset语义，建立新模型的批量oracle，再证明缓存等价；先测成本，后花新增训练预算。**
同时复用既有深度线的D2结果并等待匹配D4基线，不重复训练、不在其他任务root原地加代码。**当前共同依赖是D4-ZP产物缺失**：须原深度线确认补跑授权、训练与评分；这不是本文默认启动动作。

## 1. 已知事实与不能当作事实的预测

可核验源：

- `src/btransform_unified_v1/model.py::SharedCausalConv.forward` 当前窗内left-pad4；滑窗移动需重算左4位置。
- 同文件 `_temporal` 加 `pe[:L]`，PE是窗相对；普通逐层KV复用不能保持旧模型等价。
- `results/m1_projadd_runtime_v1/20260907T020150Z/scripts/m1_exacte_fast.py` 已将set frontend每步重算从9位置减少为5位置（left4+new1）。
- 同root的 `results/runtime_budget.json`、`results/m1_projadd_depth2/20260907_gpu0/temporal_timing.json` 提供本机参考；准确数字/代码版本需在执行时重新绑定。

以上路径以 `btransform_unified_v1/` 为前缀。

历史探针提示M1 B4/t1 frontend约8.1ms、D4 temporal约13.4ms，D2与单位置前端有潜在收益。但未训练权重、共享机器和分项计时只能排序；**不是实际D2-VC端到端性能/P95，也不是官方CPU保证**。
不接受“同算子优化已到理论地板”“VC无质量代价”“只要两线程就稳过”等表述。更换后端可能有剩余收益，但此实验不无限搜索执行后端。

## 2. 所有权、资源与非目标

- 新增实现归本执行者：`scripts/exp3_streaming_structure_v2/`、`tests/exp3_streaming_structure_v2/` 与新结果root；可隔离适配公共模型，但不改其他任务默认行为、不编辑旧提交adapter。
- 你不是工作区唯一执行者：不撤销他人修改、不停进程、不抢GPU；源root与旧receipt只读。发现冲突先隔离，无法隔离则报告。
- 当前请求只生成工单；新增VC训练须单独获本工单执行授权。建议新增预算：**最多一个M1 VC cell、3 GPU-h上限**；CPU实现/预检建议2小时，正式长流回放另列耗时并在授权预算内安排。
- 现有D4/D2成本归旧工单，同时在总报告中披露复用成本；新工单不扩张其原预算。不可继承EXP2的6GPU-h。
- 第二seed、额外fold、H1/M2 VC、全held-in重训、镜像push/EvalAI均需新增授权；本轮不启动。
- 不做concat/P32/量化/改PE/KV/stride/Mamba/GRU网格；这些仅后备假设，不是本工单执行阶段。

## 3. 主任务、数据与冻结项

先仅做M1：26/27/28的既有chron-80训练面，20120924留出开发判决，source-minival作诊断。
沿用 `results/m1_projadd_depth2/20260907_gpu0/` 的session inventory、训练配方、支持M10、seed42、固定e24 EMA、更新数和sampler；执行时核SHA与完成状态，不根据目录存在推定成功。
当前depth2 run_meta记录92,016 updates、24epochs、batch32、LR1e-4→1e-5、warmup3834、EMA0.9995；**采用实际已匹配D4/D2的封存receipt为准**，若互相不一致先报告，不能任意选择有利配方。

固定64units、16EMG、divisor1、P16、8slots、conv k5/16、temporal宽256/原FFN、sinusoidal窗相对PE、相同readout/normalizer/bank。
encoder、teacher、rSyn3 basis与normalizer必须审计outer session暴露；不闭合时只称conditional decoder-LOSO。该单session是历史开发面，不是独立blind test。

## 4. 最小矩阵：复用两个cell，最多新增一个

本次修订核验快照：`results/m1_projadd_depth2/20260907_gpu0/depth2/score_receipt.json`已落盘，固定e24 EMA的20120924 R²=**0.6885881561**，source-minival session等权=**0.8351426336**、pooled=**0.8353485492**；同root `depth4/`尚不存在。下表D4是待补的匹配基线，不是已完成事实；执行时重新绑定最新产物/SHA。不得用QueryAge-family约0.658或Original约0.798替代D4读取深度门。

| cell | temporal depth | 卷积/原始历史 | 来源与作用 |
|---|---|---|---|
| D4-ZP | 4 | L=100 raw，窗内左补0 | 待原深度线补齐并评分的匹配基线；不可用全4session旧e24顶替 |
| D2-ZP | 2 | 同上 | 已完成并有上述评分；相对D4的质量门仍待D4结果 |
| D2-HC（历史名称D2-VC） | 2 | 104 raw→100 frontend tokens | 仅D2质量过门时首选新增；与D2-ZP比较history-conv效应 |
| D4-HC（历史名称D4-VC） | 4 | 同上 | D2质量失门时替代新增；与D4-ZP比较，不同时新增两个HC |

HC=history-context convolution，数学实现为valid conv；命名同时记录`conv_pad=history`，避免把VC理解为减少时间token数量。
选择规则只依赖已声明D2质量门与未训练HC成本预检；不能训练两个HC再只报赢家。
只有D4/D2配对完备才能决定分支；等待时可以完成HC实现/parity/无标签计时，不启动新质量训练。

### 初始化与匹配

- HC与同depth ZP必须共享**从头初始化**权重字节及训练随机流，不从已训练ZP微调后称matched retraining。
- state_dict形状相同、seed相同本身不构成字节匹配证明；保存初始化SHA，确保HC构造不额外消耗影响后续初始化的RNG。
- D4与D2维度不同，不能要求完整state相同；记录共同模块初始SHA与匹配方式。若既有线仅同seed而后续模块受层数影响初始化不同，照报，不擅改运行中基线。
- 若既有同depth ZP初始化/随机流无法复现，则不能声称严格paired HC比较；申请补基线或保持探索性标签，不自动增加第二训练。
- 训练匹配按`(session, end)` endpoint流、batch分组和dropout RNG定义；HC仅将同一end的raw切片向前扩4bin，target/end/顺序不变。保存`endpoint_stream_sha256`及坐标映射作为主要匹配证明。
- 保留`sampler_batch_sha256`，另存输入构造配置/诊断张量SHA。当前`m1_projadd.py::sampler_digest`只哈希batch序号和样本索引，不哈希raw张量；若索引映射未改，该哈希可保持相同，**不能宣称HC使其必然不同**。仅索引哈希相同也不能替代session/end映射核验。

## 5. History-conv完整语义（实现前封存）

令L=100、K=5。当前输出时刻t：

- ZP读取`x[t-L+1..t]`，在该窗左补4个0；
- HC读取`x[t-L-3..t]`共L+4个真实历史bin，valid conv产生L个tokens；
- token j 对应时刻`t-L+1+j`，卷积依赖该时刻及此前4bin；temporal仍接L个token和`pe[0..L-1]`，读最后时刻输出。

**这改变了原始观测历史100→104（M1增加80ms过去信息），不是旧模型等价加速，不保证质量不降。** 使用更多历史仍然因果，但公平比较须披露。

数据门：额外4bin必须存在于同session、同允许连续流，不能越过reset/禁止trial边界、不能回读禁止的support区。共享合法endpoint集合；若HC需要删点，则两个比较臂都按同集合评分。若删点改变训练stream，既有ZP不再严格匹配，先报告/申请新的配对设计，不能偷偷变采样。
不把每个trial默认当reset；边界由既有任务合同决定。只有真实流/reset起点之前可按规则补零；其他历史缺失不得随意伪造。
不采用无条件“end<103就补零／每session最多4个”的规则：这只适用于零基流坐标下、原ZP已有完整100bin且真实reset前缺历史的特定情况。support/query分界与chron-80 split分界不自动成为reset；逐session记录受影响endpoint、补零原因/数量、purge或排除结果。真实cold-start不足100bin的情况按完整reset合同处理，不套用该4点计数。

实现注意：当前 `SharedCausalConv.forward` reshape按输入width写死，不能仅改left_pad=0；需要独立计算conv输出长度、reference path、输入检查、位置索引、causal_check、训练采样与wrapper全链路。
不能放开任意输入长度：本cell固定raw=104/token=100。泛化到有prefix任务须新协议，本轮不自动支持。

## 6. 先验收语义与成本，再训练

### P0：新批量oracle

实现HC完整窗口forward，并用显式每tap局部计算参考核对长度、卷积对齐、尺度和输出时间戳。
扰动未来输入验证过去输出不变；对首bin、L前后、多个窗口周期分别测试。原始坐标与标签索引写入可人工检查的小样例。

### P1：单位置缓存与parity

初始/reset时按完整HC oracle构建raw/frontend缓存；随后每bin仅算一个新的conv/set token，保留其余L-1个，temporal仍按当前窗相对PE重算。
wrapper raw环形历史缓冲从100扩至104，frontend token缓冲仍为100；`observe()`/`predict()`的既有调用与计时合同不变，不新增或漏掉状态推进。实际adapter若有observe实现，保持其观测语义并单测，与raw缓冲/reset细节一并写入`history_contract.json`。
bank/static/keep必须在段内固定；reset、bank或mask切换触发全重建。不得沿用旧bank缓存，也不得因异步结束漏算其他活跃流。
**不能简单把frontend缓存清零当oracle冷启动**：零raw经bias、E0/c和set网络后通常不是零token。

FP32工程parity门：`abs(delta)<=1e-5+1e-5*abs(reference)`，native输出比较；最大误差、RMSE逐session报。该门与既有runtime工单一致，不采用未验证的绝对1e-6承诺。测试失败不能事后放宽。
覆盖训练模型→oracle、oracle→cached两层；B1..4、reset/cold-start、长流、多次reset/bank切换、有效mask、异步结束与所有可访问合法bank/tag。容器输入转换单测保留。
同模型parity不等于HC与ZP相同，后者须配对训练后比较质量。

### P2：无标签CPU成本门

固定实际CPU型号、cpuset/quota、后端版本、threads/interop/BLAS；t1/t2主测，t4仅环境明确允许时辅报。与baseline交错至少3轮，报告全部轮次mean/median/P95，不挑最佳一次。
t2必须单列，与运行时线 `p1_timing_fast_t2.json` 的历史B4约12.41ms核对配置/代码SHA后作参考；不能把其已优化baseline替换成更慢旧提交制造加速比。正式比较重测同depth ZP与HC，历史D4/t2数值不是D2同depth控制，也不能和t1混减。
分别测B1/B2/B3/B4；无profiler正式计时，启动/加载/reset/转换和稳态拆分。共享机器干扰或CPU资源不可控时标低可信，不作通过证明。
最多复用一个已验证精确后端与torch参考；新后端不替代结构配对，不把量化收益混入HC效应。
新增HC训练的最低成本门：相对同depth ZP，在同资源配置B4 mean至少降低15%，重复轮次方向一致；同时给B3和总预算场景。该15%为预注册工程筛选，不保证最终权重性能。
若不足或语义失败，不训练HC；提交负结果。不以理论P=5→1比例代替计时。

### P3：最多一个新增M1 cell

经授权且P0/P1/P2过门后，按§4规则选D2-HC或D4-HC，封存manifest，复用既有同depth ZP训练配方，从头匹配训练固定e24 EMA。
训练前用实际HC trainer预检总新增成本（含最终评分/保存+30%余量）<=3GPU-h；不足则申请，不缩某一臂步数。
训练中只看预声明健康诊断，不在20120924挑epoch/LR；endpoint固定；保留RNG/optimizer/EMA与失败日志。

## 7. 质量门与运行门分别裁决

### 质量（开发工程门，不是统计非劣证明）

- D2-ZP相对D4-ZP：20120924同面ΔR²>=-0.01才作为D2-HC首选；同源minival辅报。
- HC相对同depth ZP：ΔR²>=-0.01。
- **最终候选相对D4-ZP也须ΔR²>=-0.01**，避免depth和HC分别损失0.01、累计损失0.02仍被错误判通过。
- 单session逐输出质量、预测方差、时间块敏感性照报；不把单session点门称跨session普遍非劣。
- 不与all-session exposed 0.979或不同协议历史官方分混用，不以runtime过门代表超过Original。

### Runtime（训练后重新测）

`T_est = start/load + resets + sum_B(n_calls_B * mean_predict_cost_B) + observe_only + other_fixed`。

用实际adapter调用清单，B3/B4各自测量；不跳过mask=False bin、不漏observe或reset、不把B4摊薄值当B1调用成本。
保留7200s历史评测上限与内部5400s工程目标的区别。隐藏长度和CPU倍率未知时报告多场景敏感性，**只可称CONDITIONAL_RUNTIME_READY**。
timeout反推CPU倍率依赖隐藏长度/固定开销假设，不是唯一确定的硬件测量；t1加速比例不得直接套到t2或官方机器。
最终无profiler完整允许本地流回放，记录start/reset/compute/wall及normalized latency。短流外推不得写为完整官方回放；median预测不能作为P95门已过的证据。

## 8. 交付、后续与执行前自检

新结果根：`btransform_unified_v1/results/exp3_streaming_structure_v2/<UTC>/`，旧D4/D2只以路径/SHA引用。

交付：`manifest.json`、`provenance_audit.json`、`history_contract.json`、`endpoint_audit.json`（边界影响与补零统计）、`init_pairing.json`（endpoint/sampler/RNG匹配及输入构造SHA）、`causal_checks.json`、`oracle_cache_parity.json`、训练前/后的`cpu_timing.json`（t1/t2分列）、`call_inventory.json`、`runtime_budget.json`、训练可恢复产物、`quality_paired.csv`、`decision.md`、CLI/环境/代码SHA/完整失败日志。

最终表分列：结构、raw历史/token长度、depth、训练成本、B3/B4mean/P95、同depth/D4质量差、parity与总预算假设。
状态必须拆为`SEMANTICS/PARITY/COST_SCREEN/TRAINING/QUALITY/RUNTIME`；不得只交一个“PASS”。

后续仅建议：M1通过后申请第二seed或合法额外fold；H1可能因大N受益，但须独立定义L200 raw204、C2 M3/reset合同并重新评估，不能搬M1增益。若最终发布模型换成HC，EXP1需要在最终模型上重读；EXP2保留原D4-ZP合同并如实说明任务实例差异。

执行前最高优先级自检：

1. 额外4bin是否改变/违反观测合同？reset与非零静态frontend缓存是否正确？
2. oracle是否是新HC模型，是否误把state_dict相同当函数等价？
3. 既有D4/D2与新增HC是否真正匹配训练数据/初始化，是否错误重复启动旧线？
4. 是否检查相对D4总质量损失，而不仅逐步损失？
5. 实测、估算、官方保证是否严格区分？B3/B4和冷启动是否齐全？

来源：[前向时间分析](ANALYSIS_FORWARD_TIME_STRUCTURAL_LEVERS_V1_20260907.md)、[原M1 runtime/quality工单](WORKORDER_M1_PROJADD_RUNTIME_QUALITY_V1_20260907.md)、[V2审核](REVIEW_WORKORDERS_EXP1_EXP2_EXP3_V2_20260907.md)。审核原文与活跃任务不改；本V2修订稿仅在获独立授权后作为新增结构实验合同。
