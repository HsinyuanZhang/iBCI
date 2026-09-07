# 实际执行的有限网络族与主张边界

记录时间：2026-09-06 05:32 HKT。本文描述已经执行的代码和仍在运行的固定实验，
不改写原始[架构冻结](FREEZE_ASTRA_FLAT_ROUTE_OPERATOR_FAMILY_20260906.md)，
不把尚未完成的质量目标标成通过，也不授权新的训练或提交。

## 实际有哪些网络差异

当前执行的不是“三任务逐算子完全相同的一个网络”。它是一个显式的有限族：
两个时间主干、两个固定空间注意力温度规则，以及每个任务内一对 FLAT/ROUTE。

| 任务 | 时间主干 | 空间 logit 倍数 α | 额外初始化规则 | 成对结构差分 |
| --- | --- | ---: | --- | --- |
| M1 | FW-QueryAge16，W100 | 1 | 原 M1 constructor 与 QueryTemporalStack 映射 | 仅 calibration routing bias |
| M2 | FW-CausalPE4，W50 | 1 | 同一次 seed42 初始化；共享权重逐项复制 | 仅 calibration routing bias |
| H1 | FW-CausalPE4，W700 | √32 | local FC1 列初始化乘 √45≈6.7082；两臂相同 | 仅 calibration routing bias |

每一行的 FLAT/ROUTE 都有相同的时间主干、温度、共享初始化、源数据、批次、
单位掩码、训练预算和选择规则。ROUTE gate 从零开始，但其余 routing 参数不全为零；
已有逐臂配对初始化、g=0 数值退化和 gate 非零梯度证据。

空间注意力的共同写法是：

\[
L_{h,s,n}=\alpha\left(\frac{q_{h,s}^{\mathsf T}k_{h,n}}{\sqrt{32}}
+ I_{\mathrm{ROUTE}}\tanh(g_h)
\frac{q^{\mathrm{cal}\,\mathsf T}_{h,s}P_h(c_n)}{\sqrt{32}}\right).
\]

H1 的 `UnscaledDotSlotUnitAttention` 将**整个括号**乘 √32，包括已定义的 routing
bonus；不是只取消 q·k 的缩放而保持 bonus 不变。α=√32 是 H1 为通过真实 source
学习门槛而采用的、两臂共享的已声明 logit 规则，和原冻结中的 α=1 并不完全相同。
不能把这一差别藏进“数据适配器”，也不能声称已经证实 α=1 在三任务上都足够。

H1 的 √45 来自 √((16+700+4)/16)，只在开始训练前调整 local16 对应的 FC1 权重列；
它是初始化规则，不是推理时新增一层，也不是从其他任务迁移来的训练权重。
两臂的 `activity_scale=1`，目标仍为 native velocity×20，推理仍除20。

代码依据：[M1 factory](../src/m1_optimized_v2/model.py)、
[M2 explicit attention](../src/m2_family_v1/decoder.py)、
[H1 温度与初始化](../src/h1_family_v1/model.py)。

## 保持相同的部分与允许的任务尺寸

三条线保留逐单位 k5 causal Conv+SiLU、local/E0/T4 组成的 token、256 宽两层 token
MLP、8 个 learned slots、8-head slot→unit attention、slot residual FFN、256 宽四层
时间读出、final norm 和当前时刻 readout。N/W/E0/out_dim 依任务变化；M1、M2、H1
分别是 N64/96/176、W100/50/700、输出16/2/7。M2 与 H1 的 native 除数分别为5与20。
源校准载体的实际文件和可用支持区间仍由各自输入 authority 绑定。

`FW-QueryAge16` 让四层 query 读取原始 frontend memory；`FW-CausalPE4` 的历史表示
逐层互相 contextualize。这是实质的主干差异，不是缓存实现的重命名。
H1 历史 signed mixing、SPINT C2、M1 Sfix 仍是各自明确标注的对照，不能加入共同
8-slot 族来凑齐成功结果。

因此，可信的直接归因是“同一任务、固定主干与温度下的 routing 对照”。当前设计
不能独立区分跨任务的数据效应、时间主干效应和注意力温度效应，也不是完整的
`routing × temporal × temperature` 交叉实验。

## 等价加速与网络变化分开记账

静态 E0/T 投影、固定 slot-query 投影、五个边界 token 的 k5 修复、grouped value
投影和最后时间层只计算所需 query，均是同权重下的等价执行候选，不另算网络成员。
四层 causal 主干仍对前三层的完整有限窗重算；没有偷用跨窗口 contextual KV。
M1 QueryAge16 的缓存是其自身定义允许的原始 memory 投影缓存，不与 causal 缓存混同。

实际训练的 M2 selected F2/R6 已通过全部1,011源查询及 ext4 的全部2,069计分点
流式/原生等价核验。后者每臂推进10,839个真实输入 bin，最大 native 误差分别为
`4.191e-8` / `5.961e-8`。H1 目前只有初始化权重的完整生命周期与速度测试；
其训练后的全量 proof 必须等待12轮选点与严格导出完成。

## 截至本记录的质量证据

M2 的原始24轮 source-only 选点已经固定为 FLAT e2 / ROUTE e6。下表报告两臂，
不以 ext4 结果重新挑 epoch，也不混入随后冷启动诊断的候选。

| M2 固定面 | FLAT selected | ROUTE selected | 历史 Original-SPINT |
| --- | ---: | ---: | ---: |
| 源会话1,011查询，等会话 R²（该面主指标） | .24442853 | .28045316 | .64101315 |
| 本地 ext4 2,069查询，pooled R² | .23054801 | .35783687 | .22919761 |
| 同 ext4，等会话 R² | .24995125 | .35753978 | .23759323 |

ROUTE 在四个 ext4 会话上均高于历史 SPINT，pooled 提升 .12863927；但低于历史 e8
的 pooled .38750524。FLAT 的 ext4 pooled 只略高于 SPINT，且在11-18会话下降。
源会话1,011查询仍有明显负差距。不能挑有利的一行概括为“所有场景都非劣”。

ext4 是已经使用过的本地开发面；历史 e8 与 SPINT 的选择/训练暴露条件并不完全
匹配。该结果既不是未触碰的 heldout 确认，也没有预设非劣界值及相应统计检验。
M1 和 H1 的新成对训练仍未完成，三任务总目标尚未通过。

证据：[M2 source strict finalizer](../results/m2/family_v1/finalize_pair_v1/receipt.json)、
[固定 ext4 原生结果](../results/m2/family_v1/frozen24_selected_ext4_native_v1/receipt.json)、
[固定 ext4 全流式等价](../results/family_runtime_v1/m2_frozen24_ext4_complete_v1/receipt.json)。
