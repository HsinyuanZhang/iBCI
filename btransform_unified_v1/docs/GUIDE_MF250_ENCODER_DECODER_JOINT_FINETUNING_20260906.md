# M-F250：冻结编码器与分阶段联合微调——后续 agent 工作指南

- 日期：2026-09-06。
- 状态：`DESIGN_GUIDE / NOT_TRAINING_AUTHORIZATION`。
- 用户请求：将关于 M-F250 encoder 是否联合训练的讨论整理成后续工作指导。
- 本文只新增指导文档；不授权训练、修改模型、停止其他任务、占用 GPU、全数据重训、打包或 EvalAI 提交。后续实现和运行按用户授权执行。
- 适用范围：H1、M-F250、FLAT、proj_add、CausalPE4。不是 ROUTE 实验，不扩展到 M1/M2 或窗口搜索。

## 1. 结论与研究问题

建议保留冻结 encoder 基线，优先比较“继续只训练 decoder”与“解冻 encoder 的 post-pool 后联合微调”。完整 encoder 解冻放在下一步；暂不优先做全随机端到端训练。

**两阶段与联合训练不是对立选项。** 推荐的是：先冻结 encoder 建立稳定 decoder，再以较小 encoder 学习率联合微调。永久冻结并非已证最优，联合训练也并非已证有效。

唯一主问题：

> 从同一 M-F250 起点出发、在相同新增训练预算下，允许校准表示适应新 decoder，是否改善留出日期开发集表现？

不能用训练 loss 或同源 minival 的提升替代跨会话结果；不能把“联合微调后超过旧 checkpoint”当成联合训练收益，必须超过同期继续训练的冻结对照。

## 2. 已知事实与假设边界

### 2.1 当前参考点

参考运行：`results/h1_matrix/M_F250_lr1e4_20260906T094633Z/`。

- 新 decoder：3,542,055 参数，seed 42 随机初始化；不是继承旧 decoder 权重。
- 包含新训 local Conv、P:700→16（无 bias）、token MLP、8-slot attention、四层 CausalPE、readout。
- 输入 L=250 bins（5 秒），prefix=0；训练 20y，native 评分 pred/20。
- 校准：固定 M3；冻结 C2 epoch15 encoder 生成 E0 [176,700]，H-C carrier [176,4] 沿用现有校准流程。
- decoder 在其他 11 session 训练，留出 `ses-19250120T115044`、`ses-19250120T115537`。
- epoch24 EMA：留出日期 2,952 点 pooled R²=0.5492458863，session-mean=0.5522278177。
- 2,908 点全日期 query-grid 辅助面 pooled=0.5695226421，不是独立留出成绩。
- 上述是已存记录，不是本文重新运行所得；接手者应先检查是否已有后续版本，不能默默换父点。

来源：[cell receipt](../results/h1_matrix/M_F250_lr1e4_20260906T094633Z/cell_receipt.json)、[run meta](../results/h1_matrix/M_F250_lr1e4_20260906T094633Z/run_meta.json)。

### 2.2 为什么考虑解冻

C2 encoder 原来生成 E0，与 700-bin 原始神经活动窗口相加后交给 SPINT。M-F250 改为 P(E0) 加到逐时间点 local16，再交给集合与时间模块。因此旧表示的训练用途和新 decoder 的消费方式不一致。

这是“表示适配可能成为限制”的机制假设，不是已经找到根因。P 已能学习部分适配；当前分数也可能受 decoder 优化、历史建模或数据分布限制。

源码：[C2 原模型](../../SPINT-main/src/models/components/h1_carrierid_spint.py)、[M-F250 identity 实现](../src/btransform_unified_v1/identity_variant.py)。

## 3. 比较分支：名称不能与既有矩阵字母混用

| 新分支 | pre-pool | post-pool | P + 全部 decoder | 用途 |
|---|---|---|---|---|
| `JF-FROZEN` | 冻结 | 冻结 | 继续训练 | 必要同期对照 |
| `JF-POST` | 冻结 | 训练 | 继续训练 | 第一优先实验 |
| `JF-ALL` | 训练 | 训练 | 继续训练 | 后续有条件实验 |

- pre-pool：`Linear(1024,32)+ReLU`，逐 trial 编码后均值聚合。
- post-pool：`[pooled32; H-C4] → 32 → 32 → 700`。
- H-C 的估计过程、源先验、normalizer 与支持集保持固定，不将 carrier 拟合也改成可训练变量。
- 新分支名只属于本文；不要叫 M-A/M-B，也不要把 `JF-POST` 称为 ROUTE。
- 第一轮只比较 `JF-FROZEN` / `JF-POST`。不得同时切换 L、proj_add 为 joined、slot 数、dropout、CAL-1 或时间核。

## 4. 两种证据级别：必须先选并披露

### 4.1 继承 C2 的配对诊断

从当前父 checkpoint 及同一冻结 C2 出发，研究冻结/解冻策略的相对效果。可复用现有日期划分，但必须记录 C2 encoder 的训练暴露。

**decoder 留出不自动等于整个系统留出。** 若 C2、carrier 源先验/行为基或 normalizer 曾用留出日期的数据，这个实验只能支持“继承表示条件下的 decoder 留出诊断”，不能宣称完整系统从未见过该日期。

### 4.2 全流程日期留出验证

若目标是严格跨日期泛化，则 encoder 的预训练/选点、carrier 源统计、normalizer、decoder 训练均须排除外层测试日期；测试 session 只使用协议允许的校准支持数据。

若旧 C2 不满足，则需要 fold-local encoder 与匹配 decoder 父点，属于新的工作量，需单独授权。解冻时不使用留出数据不能“洗掉”已有预训练暴露。

### 4.3 开发集不是测试集

当前 01-20 两 session 已用于父模型 epoch 选择并被反复分析。后续继续用于选点时，明确称为“留出日期开发集”，不称独立 test。

同源 2,908 辅助面不得用于选点。正式泛化结论需要未参与调参的外层日期/会话验证；官方隐藏结果不参与选点。

## 5. 配对初始化和训练合同

以下为建议方案；执行者应在读新结果之前生成不可变 run manifest，核实并冻结。本文不自动启动这些步骤。

1. **父权重：** 优先使用当前 epoch24 的 EMA decoder 权重作为全部分支共同起点；C2 pre/post 权重完全相同。记录文件 SHA、state key、shape、参数 digest。不得某臂从 RAW、某臂从 EMA 开始。
2. **新阶段优化器：** 建议两臂均新建 AdamW 状态，不继承父 RAW 优化器状态去优化 EMA 起点。所有 shared decoder 参数分组及设置相同，只有解冻分支增加 encoder 参数组。
3. **学习率：** 建议 decoder peak=1e-5，encoder peak=1e-6（0.1 倍）作为首轮候选，形状采用共同 warmup+cosine。此数值是待验证建议，不是已证最优，也不是必须照搬的历史训练合同。
4. **小规模预算：** 建议新增 8 epoch 的成对诊断；按实测 optimizer updates 冻结总预算与 warmup，不能拿旧全数据 731 upd/ep 代替该划分。若实测每 epoch597，则总4,776步、warmup597步；计数不同必须在启动前修正 manifest。
5. **EMA：** 两臂同一衰减、同一更新频率，新阶段计数同时归零。建议保存整个组合模型的原始/EMA 参数，明确包含新解冻 encoder；冻结参数保持不变。评价 EMA 时 encoder、P、decoder 必须来自同一 EMA 视图，不能混用 RAW encoder。
6. **数据与随机性：** 同 query 坐标、同 M3 支持、同 session/batch 顺序、同 whole-unit dropout mask。使用隔离随机流，不能因新增模块初始化消耗 RNG 改变 decoder 的随机掩码。
7. **共同训练设置：** effective batch32、AdamW wd0.01、unit dropout0.10、既有 x20/native 桥和精度设置保持一致；microbatch允许因显存不同，但需验证 loss/梯度累积按样本数正确加权。
8. **梯度裁剪：** 建议 decoder 参数组独立 clip1，解冻 encoder 组独立 clip1；两臂 decoder 相同。不要让新增 encoder 梯度通过全局 norm 额外缩放 decoder 而不披露。记录裁剪规则和两组梯度范数。
9. **变化记录：** 保存新阶段起点、每 epoch RAW/EMA、optimizer/scheduler/EMA 状态和最终 receipt。用新增 optimizer updates 比较，不只看 epoch 或 GPU 小时。

若配方需要调整，两臂成对重启或共同延长；禁止仅为较差臂临时加预算并仍称严格配对。初始 pilot 可以单一 seed；推广为机制结论需独立重复及更多留出日期，不能把相邻 epoch 当独立重复。

## 6. 实现要点：不仅是 requires_grad=True

当前 [FrozenC2Materializer](../../tfpd_exploration/src/two_mainlines_long_v1/decoder/h1_calibration.py) 使用 `torch.no_grad()`，参数被显式冻结；[adapter](../src/btransform_unified_v1/adapters.py) 产生 NumPy/CPU 静态 bank。直接解冻参数并继续读取现有 E0，梯度不会到达 encoder。

若获得实现授权：

- 新建可训练 encoder 路径，建议继承 C2 模块结构并严格加载同一 pre/post state；不要破坏现有 FrozenC2Materializer 或覆盖旧 bank。
- encoder 注册为组合模型的 `nn.Module`，进入正确 optimizer 参数组；移除新路径中的 no_grad、detach、NumPy 往返。
- 同一 session 的 M3 支持生成 E0，通过 P、decoder 和源 query loss 回传。支持数据不允许混入 query 标签。
- `JF-POST` 可以缓存冻结 pre-pool 的 pooled32，post-pool 每次随当前权重重新计算；`JF-ALL` 必须连 pre-pool 一起进入训练图。
- 每次参数更新后旧 E0 失效。不能长期缓存 detached E0，也不能复用已反向释放或跨 optimizer step 的旧计算图。梯度累积优先每个 microbatch 重算可训练部分，按 effective batch 加权。
- EMA 评分时从 EMA encoder 重建 bank，结束后正确恢复 RAW；缓存 key 应包含权重版本/RAW或EMA、session、支持集digest、预算和mask。
- 推理时所有权重冻结，使用合法 M3 校准计算一次 E0，后续可缓存。源侧联合训练不等于部署侧梯度更新。
- 保留“无目标 session 梯度”的部署约束，但不能称“无标签”：H-C 仍使用规定的校准标签。

## 7. 训练前必须通过的验收

1. **起点一致：** 组合模型 encoder+decoder 在更新前，`JF-FROZEN` 与 `JF-POST` 的相同真实输入预测一致；对旧 M-F250 native reference 建议使用元素容差 `1e-5+1e-5*abs(reference)`，记录最大误差。若父权重或bank映射不一致，停止配对实验。
2. **图连通与冻结：** 非退化真实小批量上，JF-POST 的 post-pool、P、decoder 有有限且非零的梯度；pre-pool无梯度，更新后digest不变。JF-FROZEN encoder全部不变。JF-ALL如执行，pre/post均验证。
3. **更新与缓存：** post-pool更新后对应E0发生变化；同权重缓存/重算一致；从RAW切EMA后不会命中旧bank。
4. **支持边界：** 每 session 支持 trial ID、M3、query ID 和时间坐标可追溯，按既有协议隔离。源统计仅来自允许的数据。
5. **尺度与数值：** 验证 `MSE(raw,20y)=400*MSE(raw/20,y)`；预测、loss、梯度有限，记录 pred_std/target_std，识别均值预测坍缩。
6. **训练步骤：** warmup/cosine 使用跨 epoch 的 global optimizer step，不能每 epoch 重置；EMA只在 optimizer 更新后更新，累积步不重复算作更新。
7. **无副作用：** 不覆盖旧checkpoint/receipt，不改官方包，不改历史成绩，不自动启动其他分支。

## 8. 指标、选点和继续/停止

### 8.1 主统计量

- 主比较：相同新增预算 endpoint 的留出日期开发集 **session-mean variance-weighted R²**，报告 `JF-POST − JF-FROZEN`。
- 辅报：pooled、两session逐一分数及差值、RAW/EMA、last4均值/范围、源训练loss及同源辅助面、梯度范数、参数位移和耗时。
- earliest-max EMA 可作为预声明的次级选点规则；不能事后在 endpoint/RAW/EMA/最佳epoch中择优改变主统计量。
- 两个session来自同一天；不能按数千相邻窗口计算虚假的独立样本置信度。此pilot不提供跨日期统计显著性结论。

### 8.2 判读

- JF-POST 只胜过旧父点、未胜过继续训练的 JF-FROZEN：**没有联合微调增益证据**。
- source指标提升而留出开发集下降：泛化风险，不能晋级。
- 小差异或只改善一个session：标记不确定，不包装为机制成立。
- 两session一致改善且endpoint/末段趋势一致：可进入独立seed/外层日期复核；若需要操作阈值，建议将session-mean增益≥0.01作为复核触发候选，启动前确认，而不是当统计显著性门槛。
- JF-POST无改善不自动否定全部联合训练；也不自动授权JF-ALL。应先判断优化是否健康、预算是否足够，再申请下一步。
- 非有限值、图断开、支持污染、起点不一致、错误缓存、尺度失败：技术停止并出receipt，不能以重训掩盖。
- 不能仅凭早期EMA低分杀臂；M-F250已有较晚才显著上升的记录。性能提前停止规则必须预注册并作用于配对双方，技术错误除外。

### 8.3 部署门独立

联合训练不改变在线decoder结构，不应把离线训练开销直接当在线开销。但encoder重建校准bank的耗时、内存仍需记录。

当前M-F250本地短程synthetic B1 exact-E median约20.31ms、P95约20.66ms，尚未通过自身本地速度门。即使联合微调提高精度，也不能据此宣称官方全流程部署通过。

## 9. 不继承旧报告中的过度解释

- 五臂L100的36-d结果是PROXY，不是真pre-pool36；不能作为本实验改encoder结构的既定依据。
- 当前五臂脚本存在epoch内step归零、add-tail仍保留concat E0的审查问题；实跑快照未核实前不能作干净机制结论。
- 固定E0特征列置换可被可训练线性层吸收，不是“破坏identity信息”的通用负对照。
- Original长窗遮挡崩溃不能证明H1任务必需14秒，不能推导必须历史互相attention。
- M2复现路径的“ROUTE-B”不是calibration routing；本文所有分支继续FLAT。
- 历史review可能是较早快照，run receipt也可能有旧模板残留；冲突时核对实际权重、manifest、metrics与源码，不按PASS/FAIL标题推断事实。

## 10. 后续 agent 交付清单与执行顺序

1. 先只读审计父点、C2来源、数据划分与现有任务状态，报告“继承诊断”或“全流程留出”的证据级别。
2. 实现/训练前确认授权范围；文档创建不等于训练授权。
3. 获准实现后新增独立路径和测试；不得改写原M-F250权重、历史记录或其他agent改动。
4. 提交预训练验收结果及冻结manifest，列出父点SHA、encoderSHA、支持/评分坐标digest、参数组、预算、LR、EMA、clip、随机流、选择/停止规则。
5. 获准运行后优先执行JF-FROZEN/JF-POST配对；资源是否并行由实际授权和可用性决定，不主动派生额外agent。
6. 最终交付一份paired receipt：同面同预算表、逐session差值、曲线、起点/终点参数变化、技术验收、失败或不确定性、可复现命令及全部产物路径。
7. 明确给出“保留冻结 / 进入复核 / 结果不确定 / 技术失败”之一；不得自动启动JF-ALL、全数据重训或提交。

**交接一句话：先证明让旧C2校准表示适应新M-F250 decoder有额外价值，再决定是否扩大到完整联合训练；所有收益必须相对于同期冻结对照，并在正确的跨会话证据边界内表述。**
