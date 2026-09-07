# 前向时间优化：在 A 级之后还剩什么（计算技术 vs 小幅结构调整）

日期：2026-09-07。性质：分析与建议，不是训练授权。承接 [REFERENCE_TASK_DIFFERENCES_CHERRYPICK_AND_SPEED_20260906.md](REFERENCE_TASK_DIFFERENCES_CHERRYPICK_AND_SPEED_20260906.md) §5 与 [WORKORDER_M1_PROJADD_RUNTIME_QUALITY_V1_20260907.md](WORKORDER_M1_PROJADD_RUNTIME_QUALITY_V1_20260907.md)。本机数字只作排序（工单 §1.3），不是官方 CPU 证据。

## 1. 当前实测状态（运行时线 `results/m1_projadd_runtime_v1/20260907T020150Z/`，M1 P16 e24，容器）

| 实现 | B4 ms/call | 说明 |
|---|---|---|
| 提交版 582019（torch, t2） | **15.07**（P95 worst 17.6） | `runtime_budget.json`：中心场景 ≈5.2×10³ s（S=1），582019 超时 ⇒ 官方 CPU 减速因子 S ≥ 1.42 ⇒ **RUNTIME_FAIL** |
| fast engine（torch, t2 / t1） | 12.41 / 21.65 | 同权重同算子；9→5 位置、静态折叠、末层 Q-only、预分配 |
| ORT（intra2 / intra4） | 9.69 / 5.41 | 容器 minival 回放 max\|Δ\|=2.5e-6，R² 不变；**加速依赖线程数，官方 CPU 线程数未知** |

P0 归因（提交版，B4）：frontend 边界重算 **51%**（left4 + right5 共 9 个位置的 set-attention），temporal L1–3 **42%**，末层 4.5%，其余 <2%。

## 2. fast 之后还剩什么——两块 FLOP-bound，A 级已到地板

本机单线程探针（`/tmp/fwd_time_probe.py`，未训练权重，机器同时有两个训练在跑，取中位数；与运行时线 fast/t1 B4 = 21.65 对照：8.1 + 13.4 = 21.5 ✓）：

| 任务 | frontend 每步重算位置数 P=9 / **P=5 (fast)** / **P=1** | temporal 全窗 L1..L(d−1)+末层 Q-only：**depth4** / **depth2** |
|---|---|---|
| M1 B1（N=64, L=100） | 5.25 / 3.07 / 0.62 | 3.71 / 1.45 |
| M1 **B4** | 16.15 / **8.08** / 2.16 | **13.44** / 5.28 |
| H1 L200 B1（N=176） | 12.74 / 6.93 / 1.23 | 7.85 / 3.00 |
| H1 L250 B1 | 同上 | 9.83 / 3.88 |
| M2 B1（N=96, L=50） | 6.16 / 4.03 / 0.71 | 2.25 / 0.82 |

两个事实：

1. **temporal L1–3 不能缓存**（REF §5.3；代码确认：`_temporal` 先 `fused + pe[:L]` 再 LN，PE 是窗相对的，每步所有位置的 LN 输入都变）。
2. **left-4 边界必须重算**是训练约定的后果：`SharedCausalConv.forward` 在**窗内**左零填充 4，位置 0–3 的 conv 输出依赖窗起点；窗每前进一步，这 4 个"边缘 token"都是新值。fast engine 把 right 侧浪费的 4 个位置去掉了，left-4 是真的。

因此同权重、同算子的 A 级还能做的只剩后端（ORT 同线程 −22%）与线程数；**再往下必须动模型**。H1 尤其：N=176 使 frontend 每位置 ≈1.2 ms，在 L=200 时 5 位置 frontend（6.9）几乎等于全部 temporal（7.9）。

## 3. 结构杠杆（需重训），按"省下的 ms / 模型改动量"排序

### S1. 有效卷积前端（valid-conv, VC）— 首推，改动最小

- 做法：输入 `L+4` 个原始 bin，conv 不做窗内填充（`left_pad=0`，输入长 L+K−1，输出长 L）。位置 t 的 conv 输出只依赖 `raw[t−4..t]` ⇒ 前端**平移等变** ⇒ 每步只需算 **1 个新位置**；边界 4 位置的重算消失。
- 参数与 state_dict **完全不变**（同 seed ⇒ 与现 cell 同初始化，可配对）。训练侧只改采样器多取 4 个历史 bin（session 起点零填充）；流式 reset 时 buffer 零初始化 = 现行 cold-start 语义。
- 收益：M1 B4 8.1→2.2（**−5.9 ms，≈ fast 的 −27%**）；H1 L200 B1 6.9→1.2（**−5.7，≈ −39%**）；M2 4.0→0.7。
- 质量风险：极低——位置 0–3 的 token 从"看截断上下文"变为"看真实历史"，信息严格不减；同时消除了每窗 4 个人造边缘 token 这一训练伪影。
- 附带：exact-E wrapper 简化为 `conv(raw[-5:])[-1] → 1 位置 set-attention`，等价证明更简单（不再有 left/right 两段 cat）。

### S2. temporal depth 4→2（已在跑：`results/m1_projadd_depth2/`，LOSO 20120924 判决面，门 ΔR² ≥ −0.01）

- 收益：M1 B4 13.4→5.3（−8.1）；H1 L200 7.9→3.0；M2 2.3→0.8。运行时线的 `temporal_timing.json` 实测比值 0.50–0.55 ✓。
- 质量风险：真实存在，须由门裁决（不能截层）。

### S1+S2 合并 cell「P16-D2-VC」

- M1 B4 t1：21.5 → **7.5 ms**（2.9×；t2 约 4.3 ms）。用 `runtime_budget.json` 的公式外推：计算部分 4850 s × 0.35 ≈ 1.7×10³ s；S=1.42 时 ≈ 2.4×10³ + 350 固定 ≈ **2.7×10³ s**，S=2 仍 ≈ 3.7×10³ < 5400 工程门。
- H1 L200 B1：14.8 → **4.2 ms**（清过 `latency_cpu_v1` 的本地门 median<10 / p95<14；即使只做 S1 也到 9.1，勉强清 median 门）。
- 建议：在 depth2 root 同一 LOSO 面加第三个 cell（同 seed42、同 3-session 面、同配方），形成 D4 / D2 / D2-VC 三向配对；GPU0 每 cell ≈45 min。VC 与 D2 互相独立，若 D2 失门，VC 单独也值得（S1 的 −27%/−39% 无质量代价）。

### S3–S6：次序在后

| 编号 | 做法 | 收益 | 为什么排后 |
|---|---|---|---|
| S3 | temporal FFN 512→256 或宽 256→192 | temporal −25–40% | 质量风险与 S2 同级、收益更小；仅在 S2 失门时作替代 |
| S4 | 进 temporal 前时间步长 2（SPD-C3） | temporal −50–75% | 改分辨率；末 bin 需单独保留；S1+S2 达标则不必 |
| S5 | 滑窗注意力 + 相对 PE（RoPE/ALiBi）+ 逐层 KV 缓存（SPD-C1/C2） | temporal → ~0.1 ms/步；总 ≈ 1 位置 frontend ≈ **1–2 ms** = SPINT 量级 | 新模型：感受野 = depth×b（b=25 才保住 100 bin；或接受更长）；训练须同语义（滑窗 mask）；重定义 exact-E 与 `causal_check`。只在 H1 回到 W=700 语义、或官方 CPU ≫1.4× 慢时才值得 |
| S6 | 递归核（Mamba2/GRU） | O(1)/步 | 家族已有 B-MAMBA 先例：M2 12 ep −0.058、e24 ext4 −0.015、ext4 选点 0.377 vs 0.372（[REVIEW_M2_DUAL_TRACK_SHUFFLED_12EP_AND_24EP_DECISION_20260905](../../tfpd_exploration/docs/REVIEW_M2_DUAL_TRACK_SHUFFLED_12EP_AND_24EP_DECISION_20260905.md)）；官方 CPU 上 scan 核无成熟快路径。现阶段不推 |

不做：INT8/FP16 当"加速"（工单 P2 仅作数值近似后备）；剪枝；跨窗普通 KV（非等价，M2 wrapper 里显式 `raise`）。

## 4. 计算技术侧（不改模型）还值得做的小项

- ORT 已给 −22%（同线程）。若官方容器允许 ≥2 线程，intra2 是稳的；intra4 的 5.4 ms 不能当预算依据（工单 §8）。
- frontend 微项（≤0.1 ms，可做可不做）：`slot_norm(slots)` 与 Q in-proj 预计算（8×256² 极小）；all-true `key_padding_mask` 去掉（fast 已做）。
- `torch.compile`/TorchScript on CPU：对 3 层 L=100 的 GEMM-bound 图预期 ≤10%，且冷启动编译计入 7200 s——排在 ORT 之后。
- 官方口径提醒：`normalized_latency = Σcompute/Σneural`，另有 7200 s 墙钟；B4 批处理把每次调用摊到 4 个文件，所以 B4 的 ms/call 才是预算变量，不是 B1。

## 5. 与另两份设计的关系

- 实验一（E0 干预）在现有 d4/窗内填充模型上做，不受本文影响；若家族转到 D2-VC，reliance 结论需在新模型上复读（零训练）。
- 实验二（688）无延迟约束，首轮保持 CausalPE4 + 窗内填充以便与 M1/H1 既有 cell 可比；manifest 必须记 `temporal_layers`、`conv_pad` 两个字段，方便日后对齐。

## 6. 建议动作（申请制，不自动启动）

1. 在 `m1_projadd_depth2` 同 root 增加 cell `P16-D2-VC`（seed42、3-session 面、20120924 判决）；D4 基线跑完后排队 GPU0。
2. 实现层：`SharedCausalConv` 加 `pad_mode ∈ {in_window_zero, history}`；`_check_input` 允许 `l_in + K − 1` 原始长度；采样器多取 4 bin；`causal_check` 同步；exact-E wrapper 单位置路径 + 1e-6 parity。
3. H1：L200 P16 若 C2-protocol 过线，下一 cell 直接用 VC（H1 收益最大）。
