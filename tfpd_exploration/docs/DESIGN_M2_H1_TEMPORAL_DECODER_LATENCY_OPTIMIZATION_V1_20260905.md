# M2 / H1 时间解码器：计算延迟优化草案 V1

Date: 2026-09-05, UTC+8  
Status: DRAFT_FOR_REVIEW__NO_EXECUTION_AUTHORITY  
Scope: decoder 推理成本；M2 / H1 优先，M1 接口迁移后置。

## 0. 决策摘要与边界

建议保留现有 decoder 创新探索，但将“近参数量”与“可实时部署”分开设门。第一优先是 **同权重、同历史窗口、同预测定义的重复计算消除**；只有这条路线仍无法达到部署预算，才进入新的时间读出结构。

两级路线，不是同时扩张两个新 backbone：

1. **E 路：保留原算子。** 静态项缓存、逐时刻集合前端缓存并修复窗口左边界、最后一层只计算最后一个 query；时间主干的前几层仍完整计算。
2. **T 路：改变算子并匹配训练。** 首选“有限历史、仅更新当前 query 的时间读出”；真正的逐层 KV 流式 Transformer 是后备，不与首选一起开扫参。

这份草案只新增文档，不启动 CPU 性能实验、GPU 训练、result root、镜像或提交；不暂停、覆盖或抢占已经在执行的 M2/H1 提交和 M1 carrier 工单。后续执行必须使用新 revision/root，继承原数据和预算边界。用户的 full permission 不等于本次“给草案”请求要求立即重训，也不需要逐命令询问权限。

**不能承诺的事情：** 当前网络经缓存必达 20 ms；小模型一定更快；KV cache 对原滑窗模型天然等价；优化后的非劣性已经证明；本地速度等于 EvalAI 速度。

## 1. 已有证据，及其测量局限

### 1.1 当前收据

| 对象 | 输入 / 活跃 decoder 参数 | 本地 CPU 时间 | 收据性质 |
|---|---|---|---|
| M2 SMALL | W=50，N=96；3,543,010 | 12.3848 ms/predict | batch=1，单个 50-bin replay 的均值，非正式延迟基准 |
| M2 LARGE | W=50，N=96；10,403,554 | 13.2134 ms/predict | 同类短探针；不能仅凭二者相近就断言前端占比 |
| H1 FLAT | W=700，N=176；3,710,055 | mean 482.0288 ms；P50 483.3494；P95 498.3021 | batch=1，随机输入，4 次预热后 24 次；未形成填满 700 bins 的真实流 |
| H1 ROUTE | W=700，N=176；3,892,591 | **未由上述 CPU 数字单独测量** | 不得把 FLAT 的 482 ms 填给 ROUTE |

M2 的当前 REF 有效 decoder 为 3,466,902 参数；SMALL 约多 2.2%。同参数量只约束权重规模，不能约束权重复用次数、访存或延迟。

本次读取的证据快照（上游文件仍可能被执行 agent 正常更新）：

| 文件，相对仓库根 | SHA256 |
|---|---|
| `tfpd_exploration/results/six_evalai_slots_v1/20260905_155800/slot_S1.json` | `3a5a7dd7ee194540e419c62aadfd55c13983ba8459ac91d55998e1589c7f3c0b` |
| `tfpd_exploration/results/six_evalai_slots_v1/20260905_155800/slot_S2.json` | `9d924f16f11dd9b2f18c64b55959d1d5252bf104711fe54b9f2c5e2c8e7a7db9` |
| `tfpd_exploration/results/h1_temporal_decoder_quick_product_v1/h1_cost_profile.json` | `b030c0748ef5252773ced45c76067469a169c74f29f18fcc9b0434b4fb708274` |

### 1.2 代码证据

本小节的 `src/` 与 `submissions/` 路径均相对 `tfpd_exploration/`。

- M2 提交运行时 `submissions/evalai_m2_small_trf_pick_v1/trf_falcon_decoder.py:1` 明确每次重算 50-bin whole window，不跨窗口复用 KV；`predict` 对活跃 session 逐个 forward。不能用单路时间估计整个官方多路调用而不计 session 数。
- M2 测速 `src/two_mainlines_long_v1/m2_runtime/parity.py:60` 是 batch=1、50 次调用；其中 `estimate_cpu_hours` 仅按给定 bins/waves 乘单路时间，尚不足以证明完整评估最坏耗时。
- H1 `src/h1_temporal_decoder_quick_product_v1/streaming.py:47` 每个新 bin 重算 W=700；其 `probe_cpu_stream` 使用随机输入和短预热。
- H1 `src/two_mainlines_long_v1/decoder/h1_temporal.py:122` **已经实现 FactoredTokenMLP**：静态 E0/H-C 在线性层中与动态 local 分开投影。剩余机会是跨 predict 缓存静态投影，不是首次消除沿 700 个 bins 的 E0 投影。
- H1 E0 为 C2 fused identity 的 **700 维**，不是 M2 的 50 维。H1 当前新增 ROUTE 为 182,536 参数；不得套用旧 FiLM 的 1,224 参数数字。
- 两者每个 token 的归一化和 slot attention 都不跨时间归一；前端只有 kernel=5 的局部因果卷积引入时间依赖。这是 E 路前端缓存的关键前提。

代码本次快照：H1 temporal SHA `1ca2cc5da5e96c1f87f409d26225360482000c3132094b1c58bee5e5f9b7ca04`；M2 提交 decoder SHA `b1c8079e51b918570b4b9ff1bc8a00a4643945f6ef1f63456e48bbb760d7bb36`。

### 1.3 延迟的三个不同问题

1. **算法等待：**是否依赖未来 bins。E/T 主案都不引入未来等待，不为凑 batch 等未来输入。
2. **计算延迟：**输入已到达后，直到完整输出返回的 wall time。
3. **吞吐：**一次调用多个独立 session 的整体时间，以及每个 session 的实际响应时间。不能把 batch wall time 除以 batch size 当成闭环延迟。

本地安装版 FalconConfig 默认 bin_size_ms=20。以下目标按 20 ms 更新周期制定；接口探针仍须记录每个任务的实际配置，不把其当作新的官方合格规则。

## 2. 保持两条科学主线

这不是 FiLM 复活，也不改变 carrier 的标签预算。

- **decoder 主线：**单元集合读出与时间建模分离；检验怎样的结构可以在预测性能、单元置换对称性和实时预算之间取得有效折中。
- **跨数据集 calibration 主线：**activity signature + tuning profile 提供会话/单元描述，source-trained consumer 使用该描述。M2 的 T4、H1 的 H-C、M1 的 rSyn3 不被强行解释为相同生理坐标。
- 静态 calibration 可以在 reset 时编译成投影或 routing logits，这是有用的部署分工，**不是其内容有效或结构新颖的证据**。
- decoder 输入 unit 行与 E0/carrier/mask 同步置换应保持输出近似不变；时间次序仍有意义。不得通过固定 channel-index 权重来偷换速度。

两句话研究定位：先把支持集产生的静态单元描述，与连续到达的活动和时间查询拆开，使重复历史不必重复编码。然后检验保留校准条件化集合读出的轻量时间计算，能否接近强 decoder 的精度，并满足真正的逐 bin 延迟预算。

## 3. 候选登记与收敛

构思采用“失败边界分析 + 计算图分解 + 简单方案检验”。先登记候选，再按算子一致性、实现风险、可测性和主线影响收敛，避免看到某个速度数字后不断加结构。

| # | 候选 | 分类 / 优先级 | 本轮处理 |
|---|---|---|---|
| 1 | 静态 E0/carrier 线性投影缓存 | E，优先 | 做；H1 已有因式分解，只增加跨调用缓存 |
| 2 | 静态 slots 的 Q 投影、ROUTE bonus 缓存 | E，优先 | 做；不缓存依赖活动的整个注意力权重 |
| 3 | 前端逐时刻 z 缓存，重算左边界 | E，最高 | 核心，无需重训，但须完整 parity |
| 4 | 最后一层只计算当前 query 与 FFN | E，优先 | 做；前 L−1 层仍全算 |
| 5 | 只对最后 hidden 做最终 LN/readout | E，简单 | 做；预期很小，不当核心贡献 |
| 6 | 多 session 向量化、环形 buffer | E，次优先 | 不等未来、不混 session，不把吞吐当延迟 |
| 7 | 编译 / SDPA backend / 线程配置 | 数值实现优化 | 单独测速 cell，兼容已固定容器；不保证提速 |
| 8 | INT8 / BF16 / FP16 | 近似数值路线 | 后置；需要精度门，不叫 exact |
| 9 | 有限历史 current-query temporal reader | T，首选结构 | E 不够快时进入，必须匹配训练 |
| 10 | 原生流式逐层 KV Transformer | T，后备结构 | 不是原模型的开关；独立 memory/position 契约 |
| 11 | 时间 patch / 稀疏输出 / 缩短历史 | 改信息和时序 | 不进首轮，尤其禁止积攒未来 patch 再预测 |
| 12 | 更窄网络 / 静态-only routing / TCN或SSM替换 | 改容量或 backbone | 后置；不扩张现有 B 的搜索面 |

收敛到四个可交付组件：E-static、E-front-cache、E-last-query、T-current-query。T-KV 是前三者及 T-current-query 均不足时的 successor，不并行扫两个新时间 backbone。

## 4. M0：先建立同机基准

### 4.1 对照与输入

- 每个任务测当前实际可部署 SPINT reference、未优化新 decoder、优化新 decoder；同权重比较固定同一 weight SHA、E0/carrier SHA、归一化、输出比例、dtype、历史长度。
- 首轮 M2 SMALL + H1 FLAT；M2 LARGE、H1 ROUTE 在主方法通过后做共享实现验证，不再次搜索最优算法。
- 只用 source/public-development 流；不打开 hidden/test 文件，不读新 EvalAI 分数来选择优化。随机张量只做形状/性能微探针，不代替真实完整接口。
- batch=1 为闭环主统计；M2 另测当前运行时允许的 7 路，H1/M1 按实际 adapter 的合法最大 batch 单独测。所有 session 身份、mask、reset/gap 行为必须保留。

### 4.2 测量层级

1. `cold/reset`：模型载入、编译、calibration materialization/cache 构建时间单列，不能通过移入 reset 隐藏成本。
2. `startup`：零填充阶段单列；历史中“尚无观测”的零 bins 经过有 bias 的网络后未必是零 embedding。
3. `steady predict`：先真实推进至少 W bins；正式测量建议 2,048 个连续 calls × 3 次独立进程重放，记录 P50/P95/P99/max、deadline-miss 数及比例。
4. `end-to-end`：输入预处理、buffer、搬运、模型、输出转换全部包括；GPU 必须正确同步，不能只报异步 launch。
5. `component`：分解 local conv、unit MLP、set K/V+attention、slot FFN/projection、temporal各层、readout、Python/buffer。分段 profiler 可能扰动绝对延迟，必须另跑无 instrumentation 的总时间。

先用 128 次微探针确认无灾难，再做长测；若长测不完整，报告 provisional，不用小样本 P99 宣布实时合格。

记录 CPU 型号、线程/亲和性、PyTorch/MKL版本、容器 digest、GPU型号与dtype、并发负载。CPU-only 是当前提交运行时的主口径；3090 单路速度仅为 GPU 部署另表，不替代 CPU 门。CPU benchmark 不与其他 benchmark 并行，且不得抢占正在供训练的 dataloader 核心。

### 4.3 草案门槛：目标而非已达事实

| 门 | 判据 |
|---|---|
| E 数值正确性 | §8 的逐步输出 parity 全通过；未通过不进入提速产品 |
| 有意义工程增益 | 同机相对原 B 的 P95 至少降低 20%，同时不恶化 P99；这是实现采用门，不是论文创新门 |
| 单路实时候选 | 20 ms 更新下 P95≤15 ms、P99≤20 ms，miss-rate 如实报告；不等同 hard-real-time guarantee |
| H1 中间里程碑 | ≥5× 同机 P95 提速可保留为工程进展，但若仍 >20 ms 就仍未实时合格 |
| T 结构筛选 | 精度—延迟 Pareto 表，不能只看参数或平均分；见 §9 |

用实测热点比例做 Amdahl 分析：若可优化部分占 f、该部分加速 s，则总加速上限估计为 `1 / ((1−f)+f/s)`。不能把“前端 700→5 个时间位置”直接写成端到端 140×。

## 5. E-static：数学等价的静态部分与小型执行优化

对于第一层线性：`W[local; E0; c]+b = W_local local + (W_E E0 + W_c c + b)`。

在冻结 checkpoint 的 inference/reset 中缓存括号内项；梯度训练期不得跨 optimizer update 复用。H1 当前代码已经做了分块线性投影，但每次 forward 仍计算静态项；M2 则要先实现分解。浮点归约次序变化，属于实数代数等价，**不保证 FP32 逐位一致**。

同时可缓存规范化后的固定 slots 和 Q 投影、ROUTE 的静态 bonus。`softmax(dynamic_logits + static_bonus)` 仍依赖每个 bin 的活动；把它换成固定 attention weights 属于新模型。

最终 readout 只消费最后 hidden，则先切 `hidden[:, -1:]` 再做最终逐 token LayerNorm/MLP；不改训练期所有目标的 loss。

环形 buffer 与 session 向量化只按实测热点实施。栈成 batch 后 GEMM kernel 可能变化，仍跑 parity；不为提速人为合并不同 session 的 calibration。

缓存键至少包含：weight/EMA-view、E0/carrier、normalizer、session、unit roster/order、unit mask、dtype/device、预处理配置和代码版本。任何一项变化全失效；不能只按 dataset tag 永久缓存。

## 6. E-front-cache：保留滑窗算子的主要机会

### 6.1 可复用的量

定义原前端为：

`z_i = SetFrontend( CausalConv_5(x)_i, E0, c, mask )`。

前端没有时间 PE、跨时间 normalization 或时间 attention。固定推理参数及 bank 后，它仅依赖该位置及前四个原始输入。因此，窗口向右移动一个 bin 时，大多数 `z_i` 保持不变。还须确认预处理不会在新bin到来后修改已存历史值；若未来实现改为每窗重新平滑或归一，则本缓存证明不再自动适用。

**但是左边界必须重算。** 原 whole-window forward 在每个窗口左侧补四个零；窗口一移动，最左四个 token 的上下文就变了。不能把连续流上的卷积状态直接当成原滑窗输出。

### 6.2 精确定义

窗口 W，kernel k=5；新窗口位置为 0…W−1：

1. 首次调用按原窗口和相同零填充完整构造 `z`。
2. 每次仅推进一个 bin，保留上一窗口位置 5…W−1 的 `z`，映射到新位置 4…W−2。
3. 用**新窗口的左零填充**重算新位置 0…3；用新窗口最后五个输入重算位置 W−1。
4. 按新时间顺序拼回完整 `[B,W,d]`；再运行原有 PE 与时间主干。
5. 大跳步、mask变化、bank变化、reset：先走完整重建，不猜测缓存有效性。小跳步优化后置。

因此，稳态每次前端只需算 5 个位置，而不是 M2 的 50 或 H1 的 700。这是前端位置数的 **10× / 140× 减少**，不是整个 decoder 的速度承诺。

缓存最终 z 而不是所有 unit tokens：FP32下，W×256 分别约 50 KiB / 700 KiB 每 session，另加原始输入 buffer、静态投影和临时边界张量。不能把新增状态说成零内存。

零活动、不同合法unit数、极端有效输入、长序列、窗口 rollover 都必须逐 bin 与原实现对齐。相同算法的不同 GEMM batch shape 也可能有数值差，按预冻结容差测。

### 6.3 特别警告

不直接把现有 `forward_tiled` 当缓存正确性的权威。必须新写或复用经测试的“指定窗口位置”primitive，核对原始区间、卷积输出长度及 overlap 裁切，不能同时重复剔除左重叠。

缓存在 `observe` 与 `predict` 混合调用、eval-mask gaps、不同batch活跃数时仍须严格跟随 adapter 的输入历史。`on_done` 不得擅自改为清空 continual state；只有原契约允许的 reset 才清空。

## 7. E-last-query：最后一层可以裁剪，前面不能一起裁

原模型只输出最后时刻预测。第 L 层的最后一行依赖第 L−1 层所有历史 hidden，但不依赖第 L 层其他行的输出。因此可：

- 前 L−1 层保持完整；
- 第 L 层计算所有历史 K/V，但只计算最后一行 Q、attention output、residual、FFN；
- 最终只对这一行做 readout。

当前 L=4。它可消除最后一层大部分逐时刻输出与 FFN 计算，不能消除前三层的 W² attention，也不能将“四层所有历史行都删掉”称为等价。

**非方形 causal mask 陷阱：**`Q_len=1, K_len=W` 时不能盲用 `is_causal=True`。PyTorch 文档将非方形 causal bias 定义为 upper-left 对齐；最后 query 本应看到全部合法过去 keys。应使用明确最后位置的 allow-mask，或在 keys 已全为合法过去时用 `is_causal=False` 并另加有效位 mask。[PyTorch SDPA 文档](https://docs.pytorch.org/docs/main/generated/torch.nn.functional.scaled_dot_product_attention.html)

该裁剪只用于当前 inference 返回最后预测的路径；若训练 loss 使用所有时刻输出，不能顺手改变训练目标。

## 8. E 路验收与禁止项

### 8.1 预冻结 parity

采用当前任务返回空间，先对每元素要求：

`abs(y_opt − y_ref) ≤ 1e−5 + 1e−5 * abs(y_ref)`。

同一设备/dtype分别评估，报告 max/mean/分位差和第一失败步；精度 gate 失败不能在 decoder R² 后放宽容差。位级一致可额外报告，但非 E-static 分块GEMM的预设承诺。完整 public-development scoring 再验平均 R² 绝对差≤1e−5，不借“数值误差”保留实质性能漂移。

必须覆盖：startup/零填充、W−1/W/W+1、3W以上真实连续流、反复reset、不同session交错、mask及bank切换、gap/on_done、单路/多路、同步unit置换、未来输入干预不影响之前输出、所有cache失效条件。CPU是必测，GPU若宣称支持则另测。

### 8.2 不属于 E 路的操作

- 跨滑窗缓存普通多层 temporal KV：历史 token 的位置PE重置、左边界及上层可见上下文变化，不能照搬旧KV。
- 只换相对PE也不自动修复上层 stale KV：被淘汰历史已经影响了保留token的深层表示。
- 截断 W、降低输出频率、采用双向窗口、把未来输入凑batch、换SSM/TCN、量化、蒸馏，都不得伪装成无损实现优化。
- 只缩训练 microbatch/做 activation checkpoint 不等于降低部署延迟；现有训练OOM绕过办法不应被写进推理增益。

## 9. T-current-query：若 E 不够，首选的新结构

### 9.1 设计动机

当前任务每个 bin 只需一个输出，原时间主干却反复更新窗口内全部历史 token。候选改为 **历史内容固定编码、只对当前输出 query 做多层时间读取**；保留 calibration-conditioned unit-set 前端，不同时换 carrier。

查询式读出本身不是新发明；Perceiver IO 已使用 query 机制支持灵活输出。本草案的迁移点是 calibration-conditioned 神经集合 + 有限历史的在线时间查询，不能把“使用 cross-attention”本身当独立创新。[Perceiver IO](https://arxiv.org/abs/2107.14795)

### 9.2 最小可执行算子候选（仍待冻结成工单）

保留 W、N、前端、8 slots、d=256、4 个读取 block、FFN=512、原输出头维度。区别在于每层历史 memory 不再来自该层更新过的历史 hidden：

`q_t^0 = z_t`

`K_i^l = W_K^l LN_mem^l(z_i), V_i^l = W_V^l LN_mem^l(z_i)`

`a_ti^l = softmax_i( (W_Q^l LN_q^l(q_t^{l−1}))·K_i^l / sqrt(d_head) + b_h^l(t−i) )`

`u_t^l = q_t^{l−1} + W_O^l Σ_i a_ti^l V_i^l`

`q_t^l = u_t^l + FFN_l(LN_ffn^l(u_t^l))`。

仅 i∈[t−W+1,t] 的合法窗口位置参与，零填充政策与parent显式对齐。年龄 bias 首案用逐 head、16 桶、零初始化的可学习标量；桶 `min(15, floor(16*age/W))`，不引入dense行为或session ID。零bias起点不意味着该新网络与旧网络相同。

历史 z 和逐层 K/V 可缓存；窗口左侧的前端边界修复沿用 §6，同时刷新被修复位置的全部层 K/V。每个旧 memory token 不消费其他历史 memory，故没有普通多层 self-attention 的上层 stale-history 问题。

其每步时间读取成本约 `O(L*(W*d + d² + d*f))`，外加至多 k 个前端位置及其 KV 投影；缓存 `O(L*W*d)`。对增长的 W 仍是线性，**不是 O(1) attention**。M2/H1保留W50/W700，不能通过换短窗虚报结构优势。

### 9.3 明确代价与主线边界

- 失去了多层历史 token 之间的 contextualization；可能损失预测性能，不能由“只输出一行”推导其它历史行都没用。
- 它是新结构，不能给现有 checkpoint 打一个快路径开关直接上产品；需要自己的训练、离线/在线等价测试和初始化收据。
- 不加入 FiLM、新的标签窗口、额外activity支持或q-memory；carrier优化仍在独立C路线开展。
- 首轮保持 FLAT 前端；H1 ROUTE 在共享结构可用之后作为单独容量/条件化轴，不在首个速度单元里混合解释。

### 9.4 最小训练/选择方案

H1优先：同一个合法source parent的兼容权重初始化两臂（原full-window时间主干与T-current-query），共同前端/输出可复用；新norm/age-bias的初始化另记SHA。权重兼容不等于算子相同，继续训练的历史暴露如实披露。不得一边从零、一边用选过点的强parent后称结构公平。

正式GPU前，先冻结source数据、窗口/采样manifest、parent来源、common参数映射、归一化、优化器/调度/EMA、epoch-pick面和工作量。首个配对试验建议沿用12 epoch、seed42；seed43只作为预定义成功后的确认，不能挑掉失败seed。

建议产品筛选门：同一public-development面上，相对**匹配训练的full-window对照**平均 ΔR²≥−0.005，worst-session Δ≥−0.020，且P95至少快3倍或达到§4实时门；单seed只称探索性近似保持，不称统计非劣。正式非劣需要预先给定margin和独立配对不确定性分析。

H1通过后移植到M2，不以H1收益替代M2数字。M1只复用已验证计算接口，不改P-FIX/P-CA内容或预算，也不因本草案自动启动M1新decoder训练。每个数据集都与其自己的SPINT基准和原B做Pareto比较。

现有可见面已多次查看；新选择使用source-development并同时报endpoint/选点规则，visible-product结果不升级为全新clean泛化证据。不得利用今天新EvalAI分数来决定哪个速度分支值得保留。

## 10. 后备 T-KV：真正流式 Transformer，而非错误缓存旧模型

若 T-current-query 快但精度损失过大，后备是原生逐bin更新的多层 causal Transformer，每层维护有限KV、用明确的相对时间表示、按同一流式状态法则训练。

它需要另行冻结：全局时间计数、每层memory长度、reset时机、source连续片段和burn-in、梯度截断、跨optimizer-step缓存重算/失效。不能继续随机独立窗口训练，却在部署中无限累积state。

每层memory存W个token不等于有效原始历史严格W：深层token可间接带入更早上下文。若要与原W模型公平，须定义总感受野或把额外历史列为单独信息预算；不能只比较cache大小。

这类 recurrence/position 联合设计有既有先例，见 [Transformer-XL](https://arxiv.org/abs/1901.02860)。文献速度数字不是本BCI任务的速度预测，不能借用其倍率。

## 11. 三个验证问题与最强反对意见

1. **是不是单纯重复计算？** E前后同权重逐步parity + 模块耗时/Amdahl。若前端提速而总速度不变，应转向时间主干，不再细抠前端。
2. **当前query足不足够？** 匹配full-window与T-current-query，报告R²—P95—P99—state bytes；若速度好但精度差，承认历史contextualization有用，进入T-KV后备。
3. **跨数据集是否一致？** H1到M2，同样的静态描述/动态活动分解，保留各自carrier和时间跨度；unit同步置换与状态连续性测试共享。carrier信息价值和ROUTE因果归因的完整消融后置，不把未完成写成已证实。

最强反对意见：“这是把已有缓存和query attention拼起来，既不保证精度，也不是新机制。”

回应：E只作为工程贡献，不占两个科学创新点之一；T也必须以跨任务的精度—延迟折中及校准条件化的必要性证据来定位。若只是工程加速但保持了准确率，就按工程结果报告；不强行升级成新的神经机制。

## 12. 分阶段交付与资源方案

以下是交接规划，不是本次已执行事项，也不改变原8小时六提交目标。

| 阶段 | 建议时间盒 | 交付 / 停止规则 |
|---|---|---|
| E0 基准冻结 | 执行开始后0–2小时 | 固定parent及CPU环境，组件热点、真实长测ETA；无同机baseline不报倍数 |
| E1 无重训优化 | 当天优先，目标2–8小时内首份证据 | E-static→E-front-cache→E-last-query逐项收据；这是开发目标不是保证 |
| E2 完整验证 | 第1–2天 | source完整流parity、warm/cold、多batch、实际容器长测；仅本地快不足以替换产品 |
| T0 结构CPU预检 | 第2–3天且E未达预算 | 冻结current-query算子、offline/online parity、单步成本；没速度空间则不训练 |
| T1 配对训练 | 第3–7天 | H1 seed42两臂，12ep建议，按重新profile成本纳入既有总预算 |
| T2 确认/迁移 | 第8–14天可选 | seed43确认、M2迁移或关闭；不因日历安排自动扩到M1 |

GPU训练不得挤占今天已排队的H1/M1。既有总GPU预算不增加：待GPU空闲且本草案冻结后，最多占用一张3090做T首轮；另一张留给原carrier路线。64GB主存下只读数据共享/惰性加载，训练不与benchmark竞争CPU线程。

交给执行协调员的推荐分工（本轮未spawn任何agent）：

- Runtime worker：仅新runtime目录及E-static/E-front-cache实现，不动训练数据或旧提交目录。
- Operator/QA worker：测试、独立whole-window oracle、长流状态与因果/parity，不调优化器、不读hidden结果。
- Coordinator/benchmark owner：唯一的CPU正式benchmark执行者、环境锁/收据、候选采用决策；协调其它工作的CPU负载。
- T worker（仅后续启用）：新时间读出和匹配训练工单，不覆写E/旧B。所有worker须知道并行人员存在，不能回退他人修改。

同一时刻最多一项正式CPUbenchmark；agent可以并行写代码和测试，但两个测速并跑只会污染数字。不要因full permission不停询问常规命令权限；遇到下面实质边界时一次性报告。

## 13. 自动继续 / 必须报告

**草案被正式冻结执行后可自行继续：**冻结范围内的E项逐项实现与parity，缓存bug修复，重复同机测速，失败优化回退到baseline，新root保存全部attempt；不需要每一步找Astra确认。

**需停对应分支并报告：**

- 想改窗口、更新频率、label/activity预算、carrier、时间memory语义；
- parity未过而准备放宽容差、跳过startup/gaps、用R²“差不多”代替算子核对；
- 想让新KV运行时直接使用旧checkpoint提交；
- 要进入T训练、改变既有GPU总预算，或替换已注册镜像/使用新的EvalAI slot；
- 工程提速仍不达实时门，准备将“可离线跑完”写成“可闭环实时”。

正常无收益不阻塞其它路线。E失败可保持原产品；T失败可关闭结构路线，不据此否定activity/tuning profile。报告需给出第一个失败case、同机耗时、精度变化、成本与建议，不只写FAIL。

## 14. 最小收据与论文措辞

每个候选记录：parent/weight/code/container SHA；支持集及normalizer摘要；窗口/步长/输出频率；cache键/失效法则；理论等价或新算子标记；逐步parity；同机cold/steady P50/P95/P99/max及miss-rate；合法batch wall time；CPU/GPU环境；状态内存/峰值内存；R²与选点面；所有尝试及时间成本。

允许的结论：“保持原预测定义与精度容差的实现优化，在指定硬件上将P95从X降到Y”；“新时间读出在指定精度预算下改善了延迟—准确率折中”。

不允许：“参数一样所以同计算量”；“缓存让模型无损实时化”但未做长流parity；“按batch均摊为2ms所以单用户延迟2ms”；“KV为O(1)”但扫描W个keys；“统一calibration叙事证明了两个创新点”；“所有实验完成”但GPU/长流测试尚未执行。

**当前建议：先做E，无需为速度立即再训一批模型。H1若在E后仍超过20ms，再审T-current-query；M2保留为短窗参照，M1 carrier线继续独立推进。**
