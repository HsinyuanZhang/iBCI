# M1 proj_add 优化工单 V1

日期：2026-09-07。性质：给执行 agent 的路线与验收合同；本文件的创建不表示优化、训练或提交已经执行。

> **ADDENDUM-DEPTH2-PROMOTED（2026-09-07 用户指令）**：§7 的优先方案（temporal depth 4→2）**从后备闸提前为正式前排实验**，与 P1 runtime 优化线并行（一条动网络、一条纯工具）。执行规格：3-session 训练面（26/27/28）+ 20120924 留出判决面；**同配方 depth-4 基线配对**（单变量 = depth）；门 = ΔR² ≥ −0.01（LOSO 面主判）；训练前后各做一次 depth-2 的纯 CPU temporal 计时验证（§7 的"先测轻量 runtime"要求保留）。GPU1 排队仲裁。结果根 `results/m1_projadd_depth2/`。§7 其余条款（W=50 后备、申请制）不变。

## 0. 一句话任务

保留 **proj_add P16 + 8-slot frontend + CausalPE4 + W=100**，先让现有 endpoint24 EMA 权重在完整 M1 推理协议下有足够运行余量，再单独审查质量证据。优先消除 CPU 时间核的重复执行成本，不做 concat，不把加宽 P32 当作超时修复。

你不是工作区中唯一的执行者。不得撤销他人修改、停止其他任务、占用已在训练的 GPU、覆盖历史 checkpoint/镜像/receipt。发现重叠修改时先隔离自己的文件；无法隔离则报告。

默认执行范围：本地只读诊断、独立目录中的运行时实现/测试、独立本地镜像、有限候选性能与数值评估。**不自动启动神经网络重训，不 push 镜像，不注册/重交 EvalAI，不动 Overleaf。** 需要重训的后续阶段见 §7，先提出申请。

## 1. 起点与冻结项

所有路径相对 `/home/xinyuan/Work_host/SPINT`。

### 1.1 必须读入并复核的历史产物

- checkpoint：`btransform_unified_v1/results/m1_projadd_series/P16_20260906T150917Z/epoch_024.pt`
- 训练与评估记录：上述目录的 `run_meta.json`、`train_receipt.json`、`score_receipt.json`
- 已提交实现：`tfpd_exploration/submissions/evalai_m1_projadd_exacte_v1/trf_falcon_decoder.py`
- 已提交权重：该提交目录下 `artifacts/m1_projadd_p16_s42_ema_e24.pkl`
- 权重包 SHA256：`3e260ebdda7d0482094366a53daf6068ca48be9dd6660bd8b1e8f8738251dbfa`
- 提交描述与失败记录：`artifacts/evalai_candidate.json`、`artifacts/TIMEOUT_DIAGNOSIS.json`
- 训练模型：`btransform_unified_v1/src/btransform_unified_v1/identity_variant.py`
- 打包参考：`btransform_unified_v1/scripts/pack_m1_projadd_exacte_v1.py`，**不得原地运行覆盖旧包**。

提交 582019 的记录为 `Execution time limit exceeded`。本地历史 B4/t2 约 14.7 ms/call，记录的总时限为 7200 秒。8.3 ms 是旧 current-query/V3 的参考值，既不是当前 CausalPE 的性能，也不是官方每次调用阈值。

### 1.2 第一阶段不能变化

- P16，identity 100→16、无 bias，和 local16 相加；四维 rSyn3 carrier 保留独立拼接；token_in=20。
- W=100、kernel=5、8 slots、原 4 层 causal temporal Transformer、原窗口内 sinusoidal PE。
- EMA e24，不能换 RAW，不能按本地分数另选 checkpoint。
- 64 neural units、16 EMG outputs、divisor=1；保持列序、掩码、平滑策略与已有观测合同。
- 固定 M10 校准与 source-frozen basis/normalizer；不能借用 query 标签更新 bank。
- M1 的 E0 由 B3 Sfix e11 activity-only encoder 产生，rSyn3 **不进入 E0**；它在下游直接进入融合层。不要按泛化示意图静默改成 joint E0。
- 七个 public calibration tag 的 bank 必须逐个保持一致；同时校验标签映射和校准输入 provenance。

第一层概念式：`a = W_local (u + P E0) + W_carrier c + b`。缓存静态部分合法；普通跨窗口 temporal KV 复用未经等价证明不合法。

### 1.3 新工作目录与资源

建议新建 `btransform_unified_v1/results/m1_projadd_runtime_v1/<UTC_timestamp>/`。
实现放在独立模块/脚本/测试中，不修改 H1/M2 公共默认行为，也不直接改旧提交目录。新文件的具体名字由执行者确定，必须在 receipt 中列明。

先读取当前 CPU/GPU 使用情况。测试默认 CPU、官方 B≤4、workers=0；限制线程，不开并发性能测试。不要把本机 GPU 推理速度当成官方 CPU 的证据。

## 2. P0：冻结参考实现与建立真实预算

### 做什么

1. 校验 checkpoint/EMA/payload/banks 哈希，记录依赖版本、镜像 ID、CPU 型号、容器 cpuset/CPU quota、线程与环境变量。
2. 在现有或复刻的同依赖容器中跑原始 P16 exact-E，保存参考预测与实际调用序列。
3. 分别测 B1/B2/B3/B4，尤其包含异步结束导致的 active batch 变化。正式耗时测试不要开 profiler；用独立短段 profiler 做归因。
4. 拆分：数据读取/启动、reset/bank、observe/转换、frontend 更新、temporal blocks 1–3、最后一层+readout。
5. 记录 `observe` 和 `predict` 实际调用次数，保留 mask=False bin 的处理，不许为了加速漏掉协议调用。
6. 做完整本地可访问流的 wall-clock 回放。若没有等同隐藏评测规模的可访问数据，给出带假设的预算外推，不能把短 calibration 文件回放说成完整官方模拟。

### 预算公式与门

`T_est = T_start + T_load + sum_resets(T_reset) + sum_B(N_predict,B * mean_cost_predict,B) + observe_only_cost + T_other`。

- `N_predict,B` 按真实协议统计；未知隐藏长度/批次构成必须标注，不得臆造。
- 工程目标建议预注册为 **总预算 ≤ 5400 秒（7200 的 75%）**，给不可控环境留余量；这是内部工程门，不是官方保证。
- B4 mean ≤8 ms/call 可作为开发目标，**不能替代总预算门**。报告 mean/median/P95 和重复运行范围。
- 若估算依赖未知 hidden 长度或机器差异，只能给 `CONDITIONAL_RUNTIME_READY`，不能给无条件可通过承诺。
- P0 总墙钟预算建议 45 分钟，含一个完整本地回放所需时间时可单列实际成本；不为凑预算截断后冒称完整回放。

### 交付

`baseline_manifest.json`、`profile_breakdown.json`、`call_inventory.json`、`runtime_budget.json`。

若哈希/EMA/官方数据转换对不上，先报告实现错误，禁止直接进入加速比比较。

## 3. P1：相同权重、相同算子的执行优化

按顺序推进，**最多两个执行后端候选**，不无限试库版本。建议本阶段最多 3 小时工程预算（最终全流验收耗时另记）。

### P1-A：整理现有缓存路径，一次完成

- reset 时缓存每个实际 batch 的 P(E0)、静态融合项、bank/mask 与常用常量。
- 复用窗口/输出缓冲，减少不必要 clone、numpy↔torch 转换和小张量拼装。
- 保持 k=5 frontend 的左边界重算语义；不得把“多数位置可缓存”误写成整个窗口永久缓存。
- 保留最后一层 last-query 优化；不要给前三层套普通跨窗口 KV cache。

已知这类小优化历史节省不足 0.3 ms。因此最多投入约 30 分钟，不把它反复作为新实验主线。

### P1-B：优先优化 temporal core 的 CPU 执行

1. 根据 P0 占比，优先针对固定 W=100 的 temporal_last 做图编译/冻结/融合，降低三层全窗重复调用的框架开销；若 profiler 指向别处，按实际占比调整并解释。
2. 先查部署环境已安装、可离线打包且支持所需算子的后端，再选最多两个候选，例如可用的 PyTorch 编译/冻结路径，或独立 ONNX Runtime CPU 路径。不要假定某个 attention 一定会被自动融合。
3. B=1..4 各自验证；静态特化可按 active batch 维护，不能把 padded batch 的耗时/语义隐去。
4. 在同样 CPU quota 下筛 threads=1/2/4（4 不可用则跳过）；固定 interop 与外部 BLAS 线程，防止超额并行。每种短筛至少三次交错运行，避免用一次最佳耗时挑点。
5. 编译/导出/加载的成本必须披露；如果容器每次冷启动需要 JIT 编译，计入总时限，避免只报稳态加速。
6. 转换器改变 attention/PE/LayerNorm/mask 语义或者存在不支持算子时，判该候选不可用，不靠删除节点“修通”。

### FP32 等价门

- 先与训练模型的正确 EMA oracle 对齐，再与冻结旧 adapter 对齐；二者分别出报告。
- 默认逐输出 `abs(error) <= 1e-5 + 1e-5*abs(reference)`，以 native M1 输出比较；最大误差、RMSE、每 session 误差均报告。超门不得事后放宽阈值。
- 覆盖七种 bank/tag、B1..B4、uint8/float32/float64/非连续输入、reset 冷启动、前 100 bins、连续长流、done 与多次 reset、active batch 变化。
- 长流比较要包含多个窗口滑动周期，不能仅检查单窗 smoke。
- 当前 receipt 中原 exact-E 与训练 EMA 的 max_abs≈1.8e-7 是参考，不是新后端已经达标的证据。
- 同算子候选应预测不变；同-session 数据可用于数值等价验证，但不得借此宣称跨 session 泛化提升。

### 淘汰规则

等价失败立即淘汰；性能增益落在重复测试噪声内或不足 10% 的执行后端不继续重投入。能过总预算门则跳到 P3，不继续追逐更多变体。

## 4. P2：有限 INT8 路线（仅 P1 仍不够快时）

目标仍是 P16 + CausalPE4，不改网络拓扑、不重训。**量化不是 bit-exact 等价**，必须单列 numerical-approximation candidate。

最多两个递进候选：

1. **Q1：前三层 temporal FFN 的 Linear 权重量化**。LayerNorm、softmax、PE、residual、最终 readout 保留 FP32。
2. **Q2：只有 Q1 速度不足且误差合格时，扩大到 temporal attention 的 QKV/output Linear**；其余仍按 Q1 保持 FP32。

先确认所选 CPU 后端真正执行 INT8 kernel、且不会因小矩阵量化开销变慢。可用动态权重量化路径避免额外 activation calibration；若需校准激活，只用预先固定的训练源数据，不用 hidden query 或 held-out label 选择量化尺度。

不对 Conv/frontend 一开始就全量量化；不一次同时改 INT8、窗口、P 宽度和训练配方。阶段墙钟预算建议 2 小时，最终回放另记。

### 预先固定的近似误差门（工程容差，不是统计非劣证明）

- 相对同一个 FP32 checkpoint：pooled R² 下降不得超过 0.003，各 session 下降不得超过 0.01；同列序、同点、同评分聚合。
- 再报告按 FP32 源目标标准差归一化的预测 RMSE，建议整体 ≤0.01、每输出 ≤0.02；尺度 floor 固定并报告。
- 不发生 NaN/Inf、输出塌缩、reset 错误或 mask 错误。阈值附近的结果应复测，不按四舍五入判 PASS。
- 训练覆盖面上通过这些门，只能说明当前权重的近似误差较小。它不证明量化模型的新 session 质量过线；若没有未参与训练的配对评估证据，标 `APPROXIMATION_PASS_GENERALIZATION_UNVERIFIED`。
- 用允许访问的较晚日 calibration 神经输入检查 FP32/INT8 输出差异，不引入额外标签选择；短支持序列无法代替完整 unseen-query 质量测试。

任何 Q 候选只有同时明显加速并过近似门才进入 P3；否则保留 P1 最佳者并判运行门未过，不重复官方提交。

## 5. P3：全链路确认与交付包

1. 把唯一候选打到**新目录、新标签**的本地镜像，记录 base/candidate image ID、payload SHA、代码 SHA、依赖与 engine config。
2. 同容器资源约束下，完整协议回放；启动、载入、reset、observe、predict、输出转换与评分串联。最终性能回放不打开 profiler。
3. 重新执行数值/协议门，不接受“开发进程快、镜像没测”。不同 B、不同 session 长度和异步结束要覆盖。
4. 汇报 runtime 与 quality 两条独立状态：
   - `EXACT_RUNTIME_READY_QUALITY_UNVERIFIED`
   - `APPROX_RUNTIME_READY_QUALITY_UNVERIFIED`
   - `CONDITIONAL_RUNTIME_READY`（预算存在未闭合假设）
   - `RUNTIME_FAIL`
5. 输出新 candidate manifest，固定 `register:false`。给用户建议最多一次新提交及其依据，但**不自动提交或重试 582019**。

最终报告一页主表：baseline / 每个实际尝试候选的 CPU 配置、启动成本、B1/B4 mean/P95、总回放时间、预算估计及假设、数值误差、每 session ΔR²、裁决。失败候选和日志保留。

## 6. 质量判定：与运行验收完全分开

- 新 P16/P32/concat 的约 0.979 分数来自 all-session 重训覆盖面，不能作为“明显超过 Original”的证据；这条路线不再训练 concat。
- 新 P16 582019 超时，因此没有可用的新官方质量结论；旧 QueryAge/V3 0.574 不能冒充新 P16 分数。
- 既定目标“Original +0.03”需要在同一合法评分面落实。历史官方 Original≈0.649 时对应约 0.679，但先核对官方 scoring surface/聚合和基线 provenance，不把本地 0.839 门混入官方门。
- 只换精确执行实现时，不需要为了“重新证明能拟合”重训已有 e24；其尚未知的泛化能力仍未知。
- 若要进一步选择模型，必须先做 session 留出第一阶段，再冻结配方进行全 held-in 重训。不得通过再打训练覆盖面来补资格。

## 7. 若 P1/P2 都无法过运行门：申请一个受限重训阶段

本节是后续路线，**不是默认启动训练的授权**。提交 runtime 证据后，申请有限训练预算，不无限微优化同一慢模型。

### 优先方案：仍为 proj_add + CausalPE，先改变一个成本轴

推荐首个结构候选为 **temporal depth 4→2**，保留 P16、W100、8 slots 和前端，其余训练合同不动。原因：直接减少每 bin 必须全窗执行的时间层数，同时不先丢掉 M1 的 2 秒历史。该项改变了固定四层配置，需要用户接受“同一家族、深度可调”；不能称权重等价，也不能直接截掉 e24 的层就报正式质量。

若用户要求所有任务严格同样四层，跳过此方案，先申请一个 W=50 的 P16/CausalPE4 独立 cell；必须从匹配配方重训并测历史损失。不能从 Original 的 history-mask 诊断推定新 B 的短窗性能。

### 最小质量设计

- 首个研究面：20120924 留出、26/27/28 训练；此面有历史暴露，是开发验证面而非全新测试。
- 从头训练 decoder，不从见过 20120924 的全四 session e24 继续训练后冒称 clean LOSO。
- 审核 encoder、NMF dictionary、normalizer 和任何 teacher 的训练 provenance，必须排除留出 session；若无法满足，标 conditional decoder-LOSO，不能用 clean 全系统 LOSO 命名。
- 第一批只跑匹配的 **P16/CausalPE4 基线 + 一个轻量候选**，固定同 seed、同源数据、同更新预算与 endpoint EMA；不得在留出分数上反复挑 epoch。
- 先测未训练轻量模型的真实 CPU runtime，若成本不足以越过时限门，不花完整训练预算。
- 轻量候选门：相对匹配四层基线 equal-session ΔR² ≥−0.01，且满足运行预算；这是折中门，不等于已超过 Original。
- 首批通过后，再申请第二 seed 或额外 held-in fold 做稳定性确认。多 fold 必须重新审查 source-frozen encoder/basis 的外层暴露，不复用不合法 source artifact。
- 冻结方案后，按已有四 held-in session 协议重训、重新做 P3，最后由用户决定正式提交。

## 8. 不允许用来“过线”的捷径

- 少调用 predict、跳过协议 bin、只算 eval_mask=True、跨 session 共享错 bank。
- 使用未来 neural bin 缓存、跨窗口不等价 KV、将 query labels 引入 calibration。
- 用 P32 替代 P16 然后把新成绩称为纯运行时优化。
- 将量化称 bit-exact，或只凭 pooled 分数掩盖某 session 明显退化。
- 用本机更强硬件/更大 CPU quota 或 GPU 速度证明官方 CPU 能跑完。
- 把编译成本排除出冷启动总预算而不披露。
- 根据官方 held-out 分反复挑 checkpoint/调结构，并称其为未触碰测试证据。

## 9. 最终给用户的回答必须包含

1. 主要耗时在哪里，有什么实际测量支持。
2. 改了什么，属于同算子执行优化、数值近似还是新架构。
3. 原权重/bank/观测合同是否保持，有什么校验结果。
4. runtime 门与 quality 门各自是什么状态；不能只写一个笼统 PASS。
5. 是否值得消耗一次官方提交，以及仍有哪些不确定性。
6. 全部文件位置、哈希、复现命令、失败候选、资源消耗；明确未自动提交。
