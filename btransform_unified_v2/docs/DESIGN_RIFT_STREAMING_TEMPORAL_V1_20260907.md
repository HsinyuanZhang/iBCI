# RIFT V1：近时优先的增量有限上下文 Transformer

日期：2026-09-07。状态：**H1-R300 已实现，CPU/CUDA 预检通过，recency／flat 正式训练进行中；尚无精度/收益结论**。实际验收、运行进度与实现边界见[执行记录](EXECUTION_H1_RIFT_R300_V1_20260907.md#7-实际验收与启动记录)。

方案名：**RIFT — Recency-biased Incremental Finite-context Transformer**。
中文名：近时优先的增量有限上下文 Transformer。代码/结果族标识：`rift_v1`。
项目根目录：`btransform_unified_v2/`，与只读基线 `btransform_unified_v1/` 隔离。目录的 v2 是项目代际，本文的 V1 是 RIFT 设计修订号。

首个扩展候选：**`H1-RIFT-D4-R300`**；`D4` 指四层 temporal，`R300` 指最多 300 个 H1 原始神经 bin 的动态感受野，不能用来表示每层窗口。**200 只是当前 H1 经人工筛选的基线窗口，不是 RIFT 的通用上限。** 其他数据集采用各自完整任务上下文，不统一套用 H1 的 bin 数；具体口径见 §6。

## 1. 决策与边界

RIFT 将网络改为：每个新神经 bin 生成一个定义不变的 frontend token；每层仅计算当前 token 的 Q/K/V、注意力输出和 FFN；过去的 K/V 在有限窗口内复用。新位置机制直接表达“近时优先”，不再对随窗口移动的 token 反复添加窗相对 sinusoidal PE。

这是新模型与新运算流，不是旧权重上的等价加速补丁。目标是探索端到端成本的数倍下降，并把释放的计算预算用于更充分的任务历史：H1 优先考虑 300 bin，其他数据集不再为沿用 H1 的延迟限制而额外截短。实际倍数、长历史的精度收益或代价均待测量。保留 D4、宽度和前端主体，不同时探索 HC、减深、减宽、stride 或新执行后端。

命名边界（2026-09-07 用户锁定，详见 [BT-EORT / RIFT 命名](../../btransform_unified_v1/docs/NAMING_BT_EORT_RIFT_20260907.md)）：

- **BT-EORT**：旧 B-transformer 系列；其 fast exact-E + ONNX 推理合同曾简称 E-ORT，此后统一以 BT-EORT 标识该旧系列。BT-EORT 不代表新的 temporal 结构。
- **HC / 深度线**：由另一 agent 负责；本设计不更改其工单、代码、训练任务或判决门。
- **RIFT**：新系列，重新定义时间位置、逐层依赖和状态的长期路线；`RIFT-flat` 是该新系列内的 flat-bias 对照，仍属 RIFT。将来使用 ORT 时也应分列 `architecture=RIFT` 与 `backend=ORT`，不能自动称作 BT-EORT 的数学等价版本。

执行授权更新（2026-09-07）：用户要求计划完成后由主 agent 驱动 agents 搭建代码、完成 GPU 训练前测试，并在 GPU 空闲后立即开始训练。按[独立执行计划](EXECUTION_H1_RIFT_R300_V1_20260907.md)推进；不继承其他工单的预算或既有结果，不包含外部提交/发布。

## 2. 当前锚点与必须改变的语义

系统锚点为 H1 C2-CAL-1 B2 **BT-EORT**，官方 submission **582044**：HO R² **0.3751451553**、HI R² **0.5822121260**、Normalized Latency **0.7834237955**；本地 HO-M3 选模约 0.3784。来源为[官方结果本地记录](../../tfpd_exploration/submissions/evalai_h1_c2_cal1_b2_ort_v1/artifacts/OFFICIAL_582044.json)。其中 **B2 是协议名，不是两层网络**。

对应[封装模型配置](../../tfpd_exploration/submissions/evalai_h1_c2_cal1_b2_ort_v1/h1_trf_falcon_decoder.py)为：176 units、20 ms/bin、原始窗口 200、conv k5/C16、identity P16、8 slots、temporal D4/width256/8 heads/head_dim32/FFN512、7 维输出；官方最大 batch 为 8。训练/校准基线见 [H1 B2 协议](../../btransform_unified_v1/docs/superpowers/specs/2026-09-07-h1-c2-cal1-b2-design.md)。不同 checkpoint/协议之间的 HO 差异不能归因为 E-ORT；EvalAI 的 0.13s 字段不作为完整评测耗时。

当前[模型定义](../../btransform_unified_v1/src/btransform_unified_v1/model.py)有两个普通 KV 缓存障碍：

1. `CausalTransformerStack` / `_temporal` 每窗重新添加 `pe[:L]`。同一物理时刻的 token 随滑窗改变位置编号，其逐层表示不能直接沿用。
2. `SharedCausalConv.forward` 在每个输入窗口左侧补四个零。窗口移动后，左边界四个 frontend token 的卷积上下文改变；只替换 PE 仍不能让所有旧 token 保持不变。

所以 RIFT 不采用“保留旧前向、换成 RoPE、直接缓存 K/V”的实现。即使修正 PE，持续重绘的边界 token 仍会使依赖它们的上层状态失效。

## 3. 稳定 token 接口：与 HC 实验分开管理

用 `x_t` 表示当前已收到、经过本任务既定预处理的 decoder 输入神经 bin，固定 session bank 为 `B=(E0, carrier, unit_keep)`。H1 本文配置下直接对应原始分箱计数；其他任务的预处理历史要另行计入，见 §6。首版前端合同为：

\[
z_t = F_{\mathrm{last}}(x_{t-4:t}; B),\qquad z_t\in\mathbb R^{256}.
\]

`F_last` 沿用 k5/C16 卷积、P16 融合、set encoder、slots 和投影的运算定义，只取真实时刻 t 的 token。在同一个模型版本、bank 和 unit mask 下，`z_t` 计算一次后不随未来查询窗口改变。

这不要求先实现高性能 HC：CPU oracle 可以逐次调用现有前端处理五点输入并只保留最后一个 token，或者用相同局部算子批量生成完整 token 序列。首要任务是验证 temporal 的新定义；单点前端的优化实现由接口对接，避免另起竞争的 HC 分支。

但必须如实承认：**连续流中保存历史 token 的方式不同于旧模型每窗左边界补零**。前端参数/局部算子不变，不等于旧完整模型函数不变。RIFT 与无 recency 对照必须共享这一接口；582044 仅作系统锚点，不能据此把所有差异单独归因于 PE 或 KV。

边界约定：

- 只在真实 reset 之前，按任务合同为局部卷积补缺失的四点原始历史；不得在普通 chunk 起点重新补零并给后续目标计分。
- reset 后只有已发生的神经 bin 才生成有效 token；不能把若干零向量当作有效的历史 KV。零 raw 经 bias、E0/carrier 和 set encoder 后也不一定产生零 token。
- 原始历史不得越过不允许的 session/trial/校准边界；trial 或 support/query 分界是否 reset 由数据合同确定，不自行假设。
- bank、unit mask 或模型权重变更后，所有依赖它们的 frontend/KV 状态失效。允许历史内可重建，否则按真实冷启动处理。

若其他 agent 的 HC 实现已经提供该稳定接口，可只读复用已验收部分；其 HC 质量结论不自动转移给 RIFT。若坚持保留旧窗内重绘语义，本路线的“旧 token 不变”前提不成立，不能继续声称普通 KV 复用是精确实现。

## 4. 位置机制：把近时优先做成软先验

### 4.1 首版选择

首版采用 **ALiBi-style 相对时间线性偏置**：移除输入上的窗相对 sinusoidal PE，在注意力 logits 上按历史距离减去偏置。ALiBi 原论文给出了距离线性偏置及其 recency 倾向；这里的时间单位和多尺度设置是本方案的设计选择，不是已验证的运动解码结论。[ALiBi 原论文](https://arxiv.org/abs/2108.12409)

不默认同时叠加 RoPE。RoPE 用旋转在 Q/K 交互中表达相对位置；单凭这种位置机制并不保证实际注意力随时间距离严格单调下降。因此本任务先采用直接可解释、与缓存兼容的时间衰减偏置；RoPE 留作后续有必要时的独立变体。[RoFormer 原论文](https://arxiv.org/abs/2104.09864)

### 4.2 定义

第 l 层第 h 个头，在允许的历史 j 上计算：

\[
s^{l,h}_{t,j}
=\frac{q^{l,h}_t\cdot k^{l,h}_j}{\sqrt{d_h}}
-\ln 2\;\frac{(t-j)\Delta t}{\tau_h},
\qquad \Delta t=\Delta t_{\mathrm{task}}.
\]

H1 的 `Delta t=0.02 s`；其他任务读取各自的 bin 宽度，不硬编码 H1 的采样率。`tau_h` 是该头的先验半衰期；`tau_h=∞` 约定为零偏置。在内容 logits 相同的条件下，未归一化权重随年龄增加一个 `tau_h` 乘以 1/2。它不是神经信号的生理半衰期，也不要求最终注意力权重单调。

八个头的初始固定配置为：

```text
half_life_seconds = [0.08, 0.16, 0.32, 0.64, 1.28, 2.56, infinity, infinity]
```

每层先用同一组尺度：六个头提供由强到弱的近时偏好，两个零偏置头保留对较远历史的访问能力。数值是可审计的起始超参数，不代表已做生理拟合。首版固定斜率；以后若需要学习正斜率，另立变体，不混入第一个对照。

“运动 decoding 具有强因果约束”支持只用当前及过去输入；“更近通常更有用”作为本项目的结构假设进一步加入。二者不是同一个逻辑命题：内容项仍可克服距离惩罚，保留非零时滞、持续状态和运动转折前的有效证据。

还要区分 **token 的时间戳** 与其内部信息的原始年龄：上层的较近 token 已聚合更早历史。此偏置直接约束的是各层对 summary token 的偏好，不能据此声称最终输出对所有原始神经 bin 呈相同指数衰减。是否真正更依赖近期原始输入，后续用时间块扰动/遮挡与时滞诊断检验。

偏置只依赖相对年龄，对整段时间坐标的共同平移不变；不需要对缓存 token 重新编号、旋转或重加绝对 PE。时间戳使用整数 bin 坐标，计算小范围差值后才转浮点。

## 5. 逐层滑窗与 KV 的精确定义

定义 `b_l` 为第 l 层一次注意力最多访问的 token 数，**包含当前 token**。真实流起点为 `t_reset`：

\[
\mathcal A_l(t)=\{j:\max(t_{\mathrm{reset}},t-b_l+1)\le j\le t\}.
\]

首版保持四层 pre-LN Transformer，位置处理之外的 block/readout 形式保持：

\[
\begin{aligned}
h^0_t &= z_t,\\
u^l_t &= \mathrm{LN}_{1,l}(h^{l-1}_t),\\
(q^l_t,k^l_t,v^l_t) &= W^{QKV}_l u^l_t + b^{QKV}_l,\\
\tilde h^l_t &= h^{l-1}_t + W^O_l\,
\operatorname{Concat}_h\left[
\sum_{j\in\mathcal A_l(t)}\operatorname{softmax}_{j}(s^{l,h}_{t,j})v^{l,h}_j
\right]+b^O_l,\\
h^l_t &= \tilde h^l_t+\mathrm{FFN}_l(\mathrm{LN}_{2,l}(\tilde h^l_t)),\\
\hat y_t &= \mathrm{Readout}(\mathrm{FinalLN}(h^4_t)).
\end{aligned}
\]

每层历史缓存保存的是 **该层输入 `LN(h^{l-1}_j)` 投影得到的 `k^l_j, v^l_j`**，不是把 `h^l_j` 当成同一层 K/V。历史 K/V 在 j 到来时算好，以后只读取或淘汰；不根据当前窗口重新更新旧 `h_j`。

该图没有 `h^l_{t-1} → h^l_t` 的同层循环状态转移；时间依赖来自下层表示的局部注意力。不要额外引入 EMA summary、递归 memory token 或跨层共享状态后仍沿用下节的有限感受野结论。

单步伪代码：

```text
advance(stream_id, t, x_t):
    check stream identity, next timestamp, model/bank/mask versions
    z_t = frontend_last(previous_raw_4, x_t, fixed_bank)
    h = z_t
    for l in 1..4:
        q, k, v = QKV_l(LN1_l(h))
        keys, values, times = concat(valid_cache_l, (k, v, t))
        # Before this step, cache contains at most b_l - 1 previous entries.
        logits = dot(q, keys) / sqrt(head_dim) + recency_bias(t - times)
        h = h + output_projection_l(softmax(logits) @ values)
        h = h + FFN_l(LN2_l(h))
        cache_l = keep_last(keys, values, times, max_count=b_l - 1)
    update previous_raw_4, timestamp, last_prediction
    return readout(final_norm(h))
```

上面假设连续、规则 bin 且缓存已按时间窗口过滤。`b_l=1` 时缓存为空；冷启动 `valid_len` 可能小于容量。padding 槽位始终在 attention mask 外，不能因 batch 对齐成为历史证据。

逐层局部注意力与定长 rolling cache 有成熟先例；叠层后的历史范围会大于单层窗口。[Mistral 7B §2](https://arxiv.org/html/2310.06825v1#S2) 本文采用“包含当前 token”的 `b_l` 计数，下面的精确 off-by-one 推导以本文定义为准，不照搬其他论文的 W 约定。

## 6. 按任务配置感受野：H1-R300，其他任务使用完整上下文

### 6.1 通用定义

设 `R_z(l)` 是第 l 层一个 token 最多依赖的 frontend token 连续跨度。因为每层向前最多扩展 `b_l-1` 个位置：

\[
R_z(0)=1,\qquad
R_z(l)=R_z(l-1)+(b_l-1).
\]

前端卷积核为 k 时，最终动态 decoder 输入跨度为：

\[
\boxed{R_{x}=k+\sum_{l=1}^{D}(b_l-1)}.
\]

这是固定 bank、固定 mask、RIFT 内无额外时序统计/递归状态条件下的结构上界。H1 对原始分箱计数直接建模时 `R_raw=R_x`；若某任务先用 K_pre 点因果滤波，未滤波原始观测跨度还需增加 `K_pre-1`，若使用无限递归滤波则不能声称整个系统有限 raw 感受野。必须根据真实 payload 的 preprocessing 开关与实现判断，不能仅凭 task 名猜测。校准 E0/carrier 本身也可包含更早的允许信息。

`R_task` 是每个任务独立的配置量。给定 `R_task>=k`，只需分配整数 `b_l>=1` 满足 `sum(b_l-1)=R_task-k`；初版采用近似均分的分配，后续可独立研究不同层的窗口分配。不存在跨任务固定 `R=200` 的结构约束。

### 6.2 任务历史政策

| 配置 | 目标 decoder 输入跨度 | D4 / k5 的初始逐层 b | 角色 |
|---|---:|---|---|
| **`H1-RIFT-D4-R300`** | **300 bin / 6 秒** | **[75,75,75,74]** | 成本预检通过后的主要扩展候选 |
| `H1-RIFT-D4-R200` | 200 bin / 4 秒 | [50,50,50,49] | 同历史诊断配置，不作为新路线的容量上限 |
| `M1-RIFT-D4-R100` | 100 bin | [25,25,25,24] | 按当前封存部署完整任务窗，不另做缩窗 |
| `M2-RIFT-D4-R50` | 50 bin | [13,12,12,12] | 按当前封存部署完整任务窗，不另做缩窗 |
| 其他数据集 | 各自 `R_full_task` | 由上述公式生成 | 读取该数据集完整上下文合同，不继承 H1 数字 |

M1=100、M2=50 的依据是当前封存 [M1 adapter](../../tfpd_exploration/submissions/evalai_m1_projadd_depth2_ort_v1/trf_falcon_decoder.py) / [M2 adapter](../../tfpd_exploration/submissions/evalai_m2_small_concat_ort_v1/trf_falcon_decoder.py) 的 `WINDOW`，不是把 M1 的 D2 权重作为 RIFT 的 D4 匹配基线。这里“使用满”先按各任务原始完整窗口解释，不指把整个 session 无界放进缓存。

旧 [v1 plan](../../btransform_unified_v1/src/btransform_unified_v1/plan.py) 还列有 M1 `W100+P100`、M2 `W50+P50` 的额外输入 prefix；这是不同的输入合同，不能与已部署的 100/50 混称同一“全窗”。若具体后续任务选定完整 `W+P` 合同，就把 `R_full_task` 分别解析为 200/100 并重新生成 b；不修改 v1 常数，也不把校准 trial 的 prefix-cycle 加到时间窗口上。实现时 manifest 必须记载完整上下文数值及其来源。

H1 的历史原始任务窗曾为 700 bin，200 是后续人为筛选结果；本轮按用户方向先设 300，既不声称 300 已最优，也不自动启动 200/300/700 全套训练。更大历史可以在确认收益后再考虑，且不要求为形式对齐而先完成一个新的 R200 训练。

### 6.3 H1-R300 的逐层依赖

H1 扩展候选固定 `k=5, D=4, b=[75,75,75,74]`：

| 层 | 单层可见数 b（含当前） | 输出的 frontend token 跨度 | 输出的 raw bin 跨度 | 步后持久 KV 条目/层 |
|---|---:|---:|---:|---:|
| 前端 | — | 1 | 5 | — |
| 1 | 75 | 75 | 79 | 74 |
| 2 | 75 | 149 | 153 | 74 |
| 3 | 75 | 223 | 227 | 74 |
| 4 | 74 | 296 | **300** | 73 |

最终最多依赖 `x[t-299:t]`：300 个 20 ms bin，按 bin 宽度是 6 秒历史；首末 bin 时间戳相差 5.98 秒。残差不扩大此跨度。R200 的对应计算是 `5+49+49+49+48=200`，只保留为历史匹配示例。

计数陷阱仍需避免：四层都用 `b=75` 得到 **301** raw bins；四层都用 `b=300` 得到 **1201** raw bins，而不是 300。逐层 b 和总感受野 R 必须分别报告。

即使选 R200 与旧 L200 保持最大动态 raw 历史相同，两者 token 构造、每层访问路径和边界语义仍不同：RIFT-R200 最终相关的是 196 个完整历史卷积 token，旧模型有 200 个窗内 token、其中左侧四个使用窗内补零。**历史上界一致不等于函数等价。** R300 对 anchor 的比较还显式增加 100 bin 历史；这是有意的质量—速度设计选择，不藏在“等价加速”里。

## 7. 状态、计算量和在线调用合同

### 7.1 状态预算

每个独立 stream 保存：四个过去的 raw bins；各层最多 `b_l-1` 个 K/V；各层时间戳、`valid_len` 和 ring 指针；`stream_id/model_version/bank_version/mask_version`；最后输出和最后已消费时间戳。静态 bank 可以复用，但不能跨 stream 混用动态 KV。

H1-R300 的 FP32、width256 配置，四层持久 KV 总条目为 295。仅 K/V 张量：

\[
M_{KV}=2\times256\times295\times4
=604{,}160\;\mathrm{bytes}=590\;\mathrm{KiB/stream}.
\]

B8 合计约 **4.61 MiB**；R200 对照为 390 KiB/stream、B8 约 3.05 MiB。均不含 raw、元数据、临时当前 token、attention workspace、bank 与模型权重。若具体实现持久保存 `b_l` 而不是 `b_l-1` 个槽位，应按真实容量重算并披露，不能沿用上述数字。内存对流时长有界，不是与模型规模无关。

令 temporal 宽度为 d、FFN 宽度为 m。稳态单步 temporal 运算规模为：

\[
O(Dd^2+Ddm+d\sum_l b_l),
\]

加一次当前 frontend token。关键收益来自过去 token 的 QKV/FFN 不再随窗口重做，注意力也只为当前 query 计算；不是只减少一个算子的常数。新结构可能让前端或固定调用开销成为主导，因此不能从 token 数减少直接推导端到端加速倍数。

H1 从 R200 扩到 R300 时，当前 token 的 frontend、四层 QKV/FFN 和 readout 次数不变；各层 attention 可见条目总数由 199 增至 299，该部分运算/读取量约增加 50%，持久 KV 从 195 增至 295 条目。**这不意味着端到端时间也增加 50%**，更不是回到整窗重算的二次增长。另一方面，warm-up、批量训练上下文和缓存重建成本会增加，仍须实测。更长历史不是免费，也不保证提高精度。

### 7.2 在线合同

- 每个已发生且被提供的神经 bin 恰好推进一次。`valid_prediction_mask=False` 通常只代表该点不计分，不代表可以跳过观测或 KV 更新。
- 用逻辑 `stream_id` 绑定状态；batch 行号不是 stream 身份。B1/B8、动态重排、部分 stream 结束和单独 reset 均不能串状态。
- 把消费观测与读取预测分开定义：新 bin 进入 `advance` 后，同一时间戳重复 `predict` 应幂等；adapter 如在 `predict` 内消费输入，也只能调用一次 `advance`。不能 `observe` 和 `predict` 各推进一遍。
- V1 使用本任务规则时间网格（H1 为 20 ms），不把“上次输出之后经过几次 predict”当作时间。缺 bin/乱序必须按数据合同显式处理；不静默压缩时间，不把缺观测自动当作真实零放电。若另立支持稀疏时间戳的变体，偏置用实际时间差且 local mask 同时执行真实年龄上限。
- reset 只清理目标 stream；bank/mask/model 变更触发所有依赖层失效并重新 warm-up。不同 bank 的训练样本不能共享 KV。
- 首版 attention dropout 为 0，保持原 feature-wise LayerNorm；不添加跨时间/跨 batch 统计。训练 unit dropout 在一个完整目标及其 warm-up 上固定，同一缓存生命周期不得重采 unit mask。

## 8. 训练与推理：同一张依赖图

### 8.1 新批量 oracle

训练/参考推理在完整 token 序列上执行与 §5 完全相同的 **causal + 每层 local mask + recency bias**。序列可以并行计算，但任何目标不访问未来，也不因处于 batch/chunk 中心而获得更长上下文。在线版本只改变执行顺序与复用方式，不改变这张图。

数值等价的对象是 **同一 RIFT 权重的 batch oracle 与 streaming forward**，不是 RIFT 与旧 CausalPE4。旧 checkpoint 可以单独探索 warm-start，但换 PE/local mask 后不再是免训练部署；首个质量对照建议从共同初始化训练，避免初始化历史成为额外因素。

### 8.2 Chunk 与 warm-up

令计分目标区间为 `a..a+T-1`。对 k5 的 RIFT，通常需 `R_task-1` 个过去 decoder 输入 bin、其中 `R_task-5` 个成为 temporal warm-up tokens；若有预处理，原始数据还要提供它所需的前史。H1-R300 在正常连续流中的精确训练样本为：

```text
raw 输入：       a-299 .. a+T-1      （299 个过去 bin + T 个目标 bin）
frontend tokens：a-295 .. a+T-1      （295 个 warm-up tokens + T 个目标 tokens）
loss：           仅 a .. a+T-1
```

前四个 raw bin 给最早 frontend token 提供卷积历史。使用当前权重重新计算全部需要的上下文，保留从目标 loss 到这些上下文运算的梯度。足够 warm-up 后，目标区间的输出及其梯度与相同固定权重的完整连续序列一致；最前面的 warm-up 高层输出不要求与完整前缀一致，也不参与 loss。

H1-R200 的对应数值是 199 raw 前史、195 temporal warm-up。不得扩大推理窗口到 300 却仍按 R200 的 warm-up/loss 合同训练。

如果 a 接近真实 reset，只取 reset 之后真实存在的 tokens，局部卷积缺少的原始前史按 §3 处理；batch oracle、训练样本和在线冷启动必须相同。普通 chunk 切分、训练随机采样起点不是新的真实 reset。

不能跨 optimizer step 复用旧权重算出的 KV。`detach(KV)` 的训练方式改变梯度；即使固定权重下前向可能一致，也不等于完整上下文训练，首版不默认采用。unit dropout 在一次目标/warm-up 内固定；两条对照共享对应 mask 与目标时间戳，重新采样 mask 时重新计算上下文。

首个 H1 pilot 可继续用 `T=1` 的 endpoint 采样来匹配 B2 的目标流与更新预算；不要求一开始改成长 chunk 多目标损失。以后若用 `T>1` 提高训练利用率，需独立记录每次更新的目标数、loss 归一化及 sampling 差异。

## 9. 最小对照：架构、recency、历史预算与缓存分别归因

冻结 H1 的 units、P16、slots、D4/width/FFN、readout、输出尺度、允许数据与 C2-CAL-1 B2 校准/选模合同。首版不更改 carrier estimator，不加入 HC/depth 的质量实验，也不增加其他数据集网格。

| 标识 | 定义 | 回答的问题 |
|---|---|---|
| `ANCHOR-582044` | 已封存旧 CausalPE4 + E-ORT | 新方案相对当前可用系统的质量/成本位置；不是单因素对照 |
| `H1-RIFT-D4-R300-R0` | 稳定前端、R300/同四层窗口，八头偏置全为 0 | 新架构内的无 recency 控制 |
| `H1-RIFT-D4-R300` | 与 R0 相同，仅用 §4 固定半衰期 | 近时先验在这张新依赖图中是否有益 |
| `RIFT-oracle / RIFT-stream` | 同一份 RIFT 权重，两种执行方式 | 缓存实现是否正确，以及增量复用减少了多少成本 |
| 可选 `H1-RIFT-D4-R200` | 相同 recency/训练协议，缩回 200 动态历史 | 仅当需要分解收益来源时，检验 200→300 的历史增益 |

R0/RIFT 配对需同初始化字节、目标 endpoint 流、bank 序列、unit-dropout 流、更新数、优化器/EMA、选模规则和评分坐标。共同 seed 不是全部匹配证据，保留共同参数的初始化摘要及实际目标/bank/mask 映射。

主 pilot 是 R300 上的 R0/RIFT 配对，不把新增 R200 训练当作前置门。无标签成本预检可直接覆盖 R200/R300；同历史的 R200 质量 cell 只有在需要声称“收益来自扩历史”时才补充。R300 vs 582044 是有意允许历史增加的系统质量—速度比较，不能只称同历史结构收益。

这组最小对照不单独归因“局部注意力”“位置机制”和“前端边界变化”各自对旧模型质量的影响。若之后要提出这些单因素结论，才补充对应控制；不要为了一个初步成本判断一次展开大规模 PE × depth × HC 矩阵。

质量看同一合法开发面的 session 等权 R²、逐输出 R²、相对 anchor 与 R0 的差值，并检查转折/起止运动、预测时滞及各时间尺度的敏感性。强 recency 若压掉有用延迟，不能只凭总体均值掩盖问题。官方 test 只作最终报告，不用其分数反复调半衰期。

速度同时列出新结构 oracle→stream 的增量收益与当前 anchor→新方案的端到端收益。比较相同 CPU 资源、batch/call inventory 和后端；首轮可统一 Torch 参考后端，正式系统结论再与实际 E-ORT 对齐。不能用 `RIFT/ORT` 对 `旧模型/慢 Torch` 得出结构加速比。

## 10. 实施顺序与验收出口

### P0：定义和低成本实现验证

在新隔离目录实现稳定 token reference、RIFT batch oracle、单步 KV 和最小调用 adapter；不改旧 submission 或公共模型默认分支。先用固定/合成 tokens 核对 temporal，不依赖 HC 优化落地。

验证以下与架构直接相关的性质：

1. 因果与范围：扰动未来不影响过去；固定 bank 后扰动 `x[t-R_task]` 及更早 decoder 输入不改变 `y_t`。在 `t-R_task+1` 的扰动允许产生影响，不把随机权重下“必须显著影响”作为测试门。H1-R300 对应边界为 `t-300` / `t-299`；有预处理的任务须另测完整 raw 依赖。
2. batch/stream：FP32 工程门 `abs(delta) <= 1e-5 + 1e-5*abs(reference)`，报告最大误差和 RMSE。覆盖满窗、冷启动、ring 多次回绕；门只用于同模型实现，不用于新旧架构比较。
3. chunk：固定权重、相同 mask 下，有足量 warm-up 的分块目标输出与整流 oracle 一致；用小规模参考同时检查目标 loss 的梯度。
4. 状态生命周期：重复 predict、独立 reset、bank/mask 切换、B1/B8 异步流和行重排，结果与逐流 oracle 一致。
5. 有界性：长流推进时各层 `valid_len` 不超过容量，KV 存储不随时长增长；时间坐标整体平移不改变输出。

数值门失败先定位实现/精度，不为通过而事后放宽。这里是未来实现的验证计划，本次文档交付不虚构 PASS。

### P1：H1 成本预检与一个匹配 pilot

P0 通过后先测 H1 R200/R300 在固定形状 B1/B8 下的 frontend、temporal 和整个 adapter，量出扩历史的实际增量成本；拆分冷启动与稳态，报告 mean/median/P95 与实际全流 wall。不只看单算子、不以 mask=False 跳步制造加速。

建议工程目标是 **同资源端到端至少约 2× 加速**，体现长期路线区别于再省三成；这是待检验目标，不是预测结论。若基于同口径官方 NL 作尺度参考，0.7834 减半约为 0.3917，但本机测速不能证明已达到该官方 NL。

成本方向成立后，才开展 R300 上 R0/RIFT 一对 H1 pilot。目标是在保有显著加速的同时用更多历史争取质量，不要求所有比较都缩回 200。具体 GPU 时间根据实施后的单步成本确定，不自动挪用 HC 或深度线预算。首轮报告完整质量—速度取舍，不把其他任务历史的 -0.01/-0.03 门直接当作本路线已获接受的精度代价。

### P2：确认与推广

若 pilot 显示有用的质量—速度折中，再做独立 seed/合适的开发验证，确认 recency 的方向是否稳定，然后将 M1/M2 及其他数据集按各自完整上下文合同接入；不要求各任务共享 bin 数或套用 H1 缩窗策略。只有 RIFT 自身结论成立，才另立组合实验考察 HC 实现复用、减深或其他压缩。

失败也形成明确出口：parity 失败是实现未闭合；成本不足说明总瓶颈不在本路线释放的计算；RIFT 弱于 R0 提示当前 recency 尺度不适配；R0/RIFT 均明显弱于 anchor 则需检查历史表示与训练合同。不得把这些不同原因合并成“KV 不适合神经解码”。

## 11. 后续交付标识与推荐布局

用户指定独立项目根：**`btransform_unified_v2/`**。核心模块放在 `src/btransform_unified_v2/`，实施/训练入口放在 `scripts/rift_v1/`，结果放在 `results/rift_v1/` 的新带时戳目录，均相对此 v2 根目录。v1 的模型、工单、训练产物与旧 submission 只读引用，不覆盖其默认路径。实现和自动开训按独立执行计划推进。

最小 manifest 示例：

```yaml
architecture: rift_v1
project: btransform_unified_v2
variant: H1-RIFT-D4-R300
status: proposed
anchor_submission: 582044
frontend_contract: causal_k5_event_v1
calibration_protocol: h1_c2_cal1_b2
depth: 4
temporal_width: 256
heads: 8
ffn_width: 512
local_window_including_current: [75, 75, 75, 74]
context_policy: task_specific
context_source: user_h1_300_candidate_20260907
decoder_input_receptive_field_bins: 300
raw_receptive_field_bins: 300  # H1 raw-count input; audit preprocessing for other tasks.
bin_seconds: 0.02
position_mode: relative_linear_recency
half_life_seconds: [0.08, 0.16, 0.32, 0.64, 1.28, 2.56, null, null]
# null means exactly zero bias; R0 uses null for all eight heads.
attention_dropout: 0.0
cache_previous_capacity: [74, 74, 74, 73]
cache_dtype: float32
backend: to_be_recorded
```

实际实施还需记录：代码与 checkpoint 标识、bank/mask/reset 合同、训练/评分 endpoint 配对、warm-up 和训练损失定义、oracle/cache 验证、CPU 配置与调用清单、完整质量/成本结果及失败记录。方案名不代表公共命名唯一性，也不代表已有实验性能。

最终对外表述应回答四件事：**近时先验是否有用；新历史表示是否保住任务质量；避免重算过去 token 是否带来端到端数倍降本；释放的预算能否支持 H1 更长历史及其他任务的完整上下文。** 各项分别给证据，才能决定 RIFT 是否成为下一代主线。
