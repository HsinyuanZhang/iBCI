# Astra 结构冻结：统一架构族，不要求 FLAT 与 ROUTE 是同一个网络

状态：`DESIGN_REVIEW_FREEZE`；冻结架构族、FLAT/ROUTE 差分及配对要求，**不冻结所有数据集使用唯一时间核**。不授权训练、迁移权重、打包或提交。

依据：[原审查包](HANDOFF_ASTRA_UNIFIED_TEMPORAL_OPERATOR_20260906.md)，以及用户补充：若网络差异由同一套 FLAT/ROUTE 定义解释、且每个数据集都有对照，则不必全统一。本文不修改原审查包或历史结果。

## 1. 决定

正式 B 族命名为 **CRST-B4：Calibration-Routed Set Temporal Transformer**，中文为“校准路由集合时序 Transformer”。这是工程/实验合同名称，不宣称发明了新的 Transformer。

- 保留两个正式成员：`CRST-B4-FLAT`、`CRST-B4-ROUTE`。
- 两个成员在 M1、M2、H1 上都必须存在；FLAT/ROUTE 的结构差分必须相同。
- 允许有限的显式时间主干：`FW-CausalPE4` 与 `FW-QueryAge16`。数据集可以使用不同成员，但**每个数据集内 FLAT/ROUTE 必须使用同一个主干**；完整名称应记录为 `CRST-B4[temporal=..., routing=...]`，不能隐藏开关。
- 新主干的工程默认仍建议 M2-like 因果 PE（保留 contextualization，且有 M1 FULL 本地证据与 exact-E 合同），但它不是论文必须唯一采用的结构，也不是按官方 HO 选出的赢家。既有 M1 query/M2 causal 可以保留为明确命名的主干，不为“整齐”自动重训。
- `FLAT/ROUTE` 只改变 **slot→unit attention 的校准路由偏置**，不同时切换时间核、位置编码、前端类型或训练目标。
- M1 current-query 不改名为 ROUTE、不删结果。保留不同时间主干时，可以研究“同一 routing 机制在不同主干上的效果”；不能声称整网差异只有 FLAT/ROUTE。只有要归因时间核优劣、交互或跨任务通用性时，才要求每个数据集共同的 `routing × temporal` 2×2。

两句话定位：先用同一套校准条件化的单元集合前端把不同神经元集合变成时序 token，再由显式命名的有限窗主干读出当前行为。ROUTE 只检验“校准信息能否改善单元选择”，不把“历史是否互相更新”混进这个结论。

**主张边界：**三任务各自有可信 FLAT/ROUTE 对照，可以支持 routing 机制在这些任务/主干组合上的效果；即使时间主干不同，这个结论仍然成立。但它不能分离“数据集效应”和“主干效应”，不能证明一个时间核跨任务最优，也不能把它写成所有数据集运行完全同一网络。

## 2. 为什么现有差异不能直接叫 FLAT/ROUTE

实际代码中的定义如下，不能靠包装重命名消除：

| 差异 | 是否属于现有 FLAT/ROUTE 轴 | 审查结论 |
|---|---|---|
| M1 factored Linear vs M2 concat Linear | 否，首先是等价计算候选 | 第一层满足 `W[local;E0;T]+b = Wl·local + We·E0 + Wt·T + b`；非线性前再相加。需严格权重映射和数值验证，不必因此重训。 |
| M1 手写多头 slot attention vs M2 `MultiheadAttention` | 否，首先是等价实现候选 | 在投影、head 排列、缩放、mask、dropout=0 相同且至少一有效单元时，可映射；不能只因名字不同就认定新架构，也不能未验证就称已等价。 |
| FLAT 无 calibration logit bias，ROUTE 有 | **是** | 这是每对 FLAT/ROUTE 比较中唯一允许的结构开关。 |
| M1 `QueryTemporalStack` vs M2 causal stack | **不是** | 前者各层读原始 frontend memory，只有当前 query 更新；后者历史表示逐层互读并更新。 |
| H1 signed mixing vs 8-slot set frontend | **不是** | H1 V4/V6/V7 已经不是原审查包所概括的 8-slot 前端，不能把它作为 CRST-B4 的任务适配器悄悄混入。 |

源码入口：[factored/slot/routing 定义](../src/two_mainlines_long_v1/decoder/h1_temporal.py)、[M2 正式包算子](../submissions/evalai_m2_small_trf_ext6_epochpick_v1/trf_falcon_decoder.py)、[current-query 合同](../src/two_mainlines_long_v1/current_query_v2/core.py)、[H1 signed 前端](../src/h1_optimized_v4/model.py)。

## 3. FLAT/ROUTE 的严格差分与共享模块

对 head `h`、slot `s`、unit `n`，设校准载体 `c_n = LN([E0_n; T_n])`：

```text
FLAT:  logits[h,s,n] = q[h,s] · k[h,n] / sqrt(d_head)
ROUTE: logits[h,s,n] = FLAT_logits[h,s,n]
                        + tanh(g[h]) * q_cal[h,s] · P[h](c_n) / sqrt(32)
```

ROUTE 的 `g=0` 时应退化为同权重 FLAT；`q_cal`、`P` 不能同时做成令 gate 永久无梯度的死分支。现有初始化是 gate 归零、其余 routing 参数独立初始化，可作为实现参考。ROUTE 额外参数/缓存/延迟必须单列，不能称参数量完全相等。

`shared_must`（共享算子定义与代码，不要求三数据集共享训练权重）：

1. 逐单元共享 causal Conv `1→16, k=5` + SiLU。
2. `concat(local16,E0,T4) → Linear256 → GELU → Linear256 → LayerNorm`；允许已证明等价的静态分块投影。
3. 8 个 learned slots、8-head/width256 的 slot→unit attention、相同有效单元 mask；slot residual FFN `256→1024→256`，flatten `8×256→256`。
4. 时间模块共同宽度256、8 heads、总深度4、FFN512；每个命名时间成员在所有任务上有同一精确定义。`FW-CausalPE4` 固定四层 pre-LN 因果更新与窗口内正弦 PE；`FW-QueryAge16` 固定原始 memory、四层当前 query 更新及16个线性 age buckets。不能在同名成员中私换 log-age/recency τ。
5. final LayerNorm、当前时刻 `256→128→Dout` 的 GELU readout。
6. 同一 FLAT/ROUTE bias 定义、gate 初始化、训练时 whole-unit mask 语义；同一有限窗、启动、缓存失效和 native-output 契约。

| `per_dataset_may` | M1 | M2 SMALL | H1 |
|---|---:|---:|---:|
| 神经单元数 N | 64 | 96 | 176 |
| 原始窗长 W | 100 | 50 | 700 |
| E0 维度 | 100 | 50 | 700 |
| 功能载体 T 维度 | 4 | 4 | 4 |
| 输出维度 | 16 | 2 | 7 |
| 训练目标倍数 / native 除数 | 1 / 1 | 5 / 5 | 20 / 20 |
| 官方 batch 上限 | 4 | 7 | 按 H1 当前协议绑定，不从其他任务复制 |

E0/T 的冻结来源、合法 calibration support、原始输入预处理和固定输出校准按任务记录，属于输入/输出适配，不允许藏新的 trainable 主干。学习率/批量/预算等训练数字可以按任务预注册，但每个任务内 FLAT/ROUTE 必须配对一致。允许预声明从上述有限集合选择时间成员；位置规则随成员绑定，不能再自由逐任务调换。深度、宽度、slot 数本版不按任务改变；M2 LARGE 单列为 width 消融，不充当 SMALL 家族的无标注替身。

## 4. 三条时间路线的并列审查

| 路线 | 命名/归纳偏置 | 真正收益与限制 | 需要的新证据 / 旧点如何处理 | 第一结构消融 |
|---|---|---|---|---|
| **A：允许成员/新主干工程默认，M2-like** | `FW-CausalPE4`：有限窗历史逐层 contextualize | 保留完整因果交互；exact-E 不保留跨窗 temporal KV。W700 成本仍高，不能因 M2 跑完就保证 H1。 | M2 581973 保持历史点；M1 已有 FULL 对照，并非从零开始；H1 signed 不能冒充共同前端。 | 同一 `FW-CausalPE4` 上只打开 calibration routing：FLAT↔ROUTE。 |
| **B：允许成员，M1-like** | `FW-QueryAge16`：四次 query 更新、memory token 不相互 contextualize | 可合法复用各层独立 memory KV；不能假定它跨任务不损精度。M1 T≈FULL 只在固定 source-dev 成立。 | M1 既有 T 可保留；M2 只有决定采用 query 时才需新训；H1 必须先检查 common-frontend 学习能力，不能用 signed 数字充数。 | 同一 `FW-QueryAge16` 上只打开 routing；如果另做 A↔B，只能归因“核整体”，不能单独归因 PE 或历史互读。 |
| **C：不作默认的真融合** | `FW-C1Q3-PE`：1 层 causal context + 3 层当前 query 读取同一 contextual memory | “浅层历史交互 + 深层当前读出”是可解释假说。仍要重算窗口 contextual memory，不获得 B 的跨窗 KV 合同。 | 三数据集都需要新训/配对；A/B 的旧点都不能当作 C 的成绩。 | 在相同 PE、Norm 绑定、总深度4下，比较 `contextual_depth=1` 对 `0` 和 `3`；三格都用同一 routing 设定。 |

两个重要边界：

- **3 层 causal + 第4层只算最后一个 query，在同权重/Norm/PE/残差条件下，就是 A 的 exact-E 输出，不是新的融合模型。** 代码中的 `_ExactEEngine._temporal_last` 已采用该等价关系。
- C 的受控 `depth=0` 使用相同 PE 和同层共享 query/memory Norm，**不等于**现有无 PE、带 age bias、分离 query/memory Norm 的 M1 T；它需要新训，不能偷用 T e6。如此才能把第一格归因于 contextual depth，而不是同时改位置规则。

没有把 A 或 B 宣告为三任务通用赢家：现有证据不足以把 B 的“不让历史互读”当作三任务共同主句，也不足以保证 A 的 W700 工程预算；C 则尚无独立质量或速度证据。B 的约十倍容器收益主要是对旧 naive 整窗适配器，**不是**对合理 exact-E 控制的十倍：M1 已记录的单流 T P95 约3.8–4.1 ms，E 约4.8–4.9 ms，差距约1.2–1.3倍，见 [M1 验收](../results/decoder_validation_v2/20260905_190000/m1/ACCEPTANCE.md)。

## 5. 重训、保留与不再混用的点

- **M1：** T e6 不能迁移成因果核。已有 `full_window` FULL e5 则来自 `M1TemporalFlatDecoder`，与 A 同构族，可在严格权重映射、单位及完整 parity 核验后作为 FLAT 迁移候选；不能笼统说选 A 必须重新训练所有 M1 权重。新的 FLAT/ROUTE 因果比较仍需匹配训练证据，不能拿一个旧已选 FLAT 加一个新 ROUTE 称完全配对。见 [factory](../src/m1_optimized_v2/source_dev.py)。
- **M2：** 581973 的因果 SMALL 无需因接口/等价投影改名而重训；它只代表相应 FLAT 历史点，不证明 ROUTE 或 B/C。新 ROUTE 需要配对训练；官方分不能选择新点。
- **H1：** V4/V6/V7 signed 前端与本版共同前端不同，权重不能迁移成 CRST-B4。旧 causal e5 的 naive 超时不是时间核失败证明；若尝试继承须先审查真实输入/输出单位及算子。同族 H1 新训必须先过真实 source 学习门槛，不能直接开完整训练，也不能靠缩 W 或换回 signed 前端规避。
- **C2：** 始终保留 SPINT 对照地位；581920 不纳入 B-transformer，也不把读取 C2 的冻结 identity 误写成运行 C2 主干。
- **M1 T e6 / 活 V3 包装：** 保留 source-dev 与运行时对照，不注册、不更名成 ROUTE；已删除的 naive 图不恢复。

更新 H1 事实：原审查包的“V6≈0.195”不应继续引用。当前 V6 T 完整 pooled R² 是0.45403，固定 M3 后0.55847；V7 同轮固定 M3 后 T=0.56576、FULL=0.61362。两者已通过20325有效帧实际 API 重放；T 仍未达到原先 FULL 保真边界。它们使用 signed 前端，**不能证明共同 8-slot 前端上的 A/B 对比**。见 [V6](../results/decoder_validation_v2/20260905_190000/shared/h1_v6_staticquery_m3_selected_complete_api_v1.json)、[V7 T](../results/decoder_validation_v2/20260905_190000/shared/h1_v7_staticquery_m3_selected_complete_api_v1.json)、[V7 FULL](../results/decoder_validation_v2/20260905_190000/shared/h1_v7_staticfull_m3_selected_complete_api_v1.json)。

旧点“退出统一主线证据”不等于删除或宣告其原实验无效。所有原权重、receipt、官方记录保留原标签与适用范围。

## 6. 最小验证表与执行边界

主表固定为三数据集 × `{FLAT, ROUTE}`，并在每一行明列所用时间主干；完整报告每一格，不能只展示各任务较优者。每对固定时间主干、source/dev、carrier、初始化共用部分、mask RNG、训练预算、loss/native 单位、epoch-pick 方法和输出校准。允许另外报告预注册的本地选择结果，但不把三个不同成员的最佳分拼成“同一模型”一行。仅研究 routing 机制时不强制额外2×2；研究时间核/交互时才补齐交叉表。

三个必要验证，不由本文自动启动：

1. **合同测试：** 同权重 `ROUTE(g=0)=FLAT`；routing gate 可学习；MHA/factored 映射；raw-zero startup、W边界、mask/roster/session/inactive lane、所有合法 batch；固定 native 元素容差 `1e-5+1e-5*abs(reference)` 与全量 R² 差 `<=1e-5`。
2. **配对质量：** 第一结构变量仅 routing bias 开关，三任务均有 FLAT/ROUTE。本地选择规则先冻结，C2/历史 T/官方点只作对照。H1 共同前端的历史学习失败是实质风险，必须先解决；不能凭这份结构冻结声称已解决。
3. **完整运行时：** 同 payload 的 reference/optimized public API，正式 batch、3×2048 持续调用、cold/reset、峰值与常驻内存、全序列外推并最终全流程验证。不能用 B1×B 推算整批，不能由局部快推定2h必过。

exact-E 允许静态 bank 投影、k5 左边界修复、最后层 last-query、异构 bank batching；不允许复用已 contextualize 的跨窗 temporal KV。正弦 PE 随窗口滑动重新编号，历史 token 的有限窗祖先也会变化。采用跨段 recurrence 会定义另一种模型；例如 [Transformer-XL 原论文](https://arxiv.org/abs/1901.02860) 明确引入跨段 recurrence 和新的位置机制，不能拿它的缓存思想当作现有有限窗 A 的等价证明。

本版明确拒绝：按数据集隐藏切换 causal/query、把 signed 前端称作 ROUTE、把 C2 称作 B、只统一 `predict` 的包装式主张、以官方 HO 选新点、未配对的“融合更强”、复活已关闭 FiLM/C3/q3-AFC4 路线。结构决定采用“拆分独立因素、再比较可组合因素”的审查方法；不扩大为新训练授权。
