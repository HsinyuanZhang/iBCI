# ASIC 部署章程：速率域与可重构主张

**状态：定位文档，2026-07-25 建立。**
**作用：说明本项目的实验取舍为什么这样排序。发生冲突时，科学结论仍以 [`CURRENT_RESULTS.md`](CURRENT_RESULTS.md) 为准。**

本文补上此前只存在于口头讨论、未进入仓库的一层背景：`sua_exploration/`
的最终目标不是发表一个更准的 SUA decoder，而是一颗**可重构的
intracortical 运动解码芯片**。所有算法选择都应按"它落在哪个速率域、
它是否改变实时路径的形状"来评估，而不是只看 R²。

## 1. 论文定位与它带来的举证责任

主张是：**芯片架构的可重构能力可跨应用复用；性能不是核心诉求。**

这个定位比"SUA 解码更准"好辩护，但它把举证责任换了个位置。要证明的
不再是精度，而是**可重构的代价小于收益**。因此：

- SUA/MUA 二选一**不足以**支撑可重构主张。SPINT 是 set-based、
  permutation-invariant 的，`N` 本来就是变量；MUA 只是"`N` 恰好等于
  通道数且固定"的特例。decoder 侧的模态切换几乎不需要 dataflow 重构，
  审稿人会正确地指出这只是把 sorter power-gate 掉。
- 真正值得写的可重构建立在**形状差异**上：同一片 MAC 阵列要在
  "逐通道短抽头流式滤波"和"规整瘦 GEMM"之间切换，归约树切分点、
  驻留方向、数据供给模式都不同。

因此本项目的算法实验只有在能回答"它对哪个速率域的形状做了什么"时，
才对芯片主张有贡献。

## 2. 速率域划分

| Block | 触发频率 | 计算形状 | 代价来源 |
|---|---|---|---|
| 带通滤波 | 30 kHz × N_ch 连续流 | 每通道独立、短抽头 | 采样率 |
| 检测 / 模板匹配 | 事件驱动 | 小矩阵 | 事件率 |
| **NeuronID encoder** | **每 session 一次** | 规整 GEMM，`N×T×D` | **摊薄后可忽略** |
| **Cross-attention decode** | **50 Hz** | 每 token `W→512→512` read-in | **实时路径主项** |

前两行属于 sorter 前端，本项目未实现，其数字尚未测量；本文不引用未测量值。
后两行有本仓库产出的成本工件，见下。

## 3. 不对称性（已用本仓库工件量化）

这是整个设计里最该讲清楚的一件事：**identity 路径与实时路径相差约五个
数量级**。

来源：`sua_exploration/results/fixed_slot_router_pilot_v1/hardware_profile.json`
（analytic，2026-07-25）与各 run 的 `hardware_cost.json`。

### 实时路径（每个解码窗口都要付）

`readin_mac_per_token = 287,744`（= `50×512 + 512×512`，measured artifact）。

| 配置 | read-in MAC / window | @50 Hz |
|---|---:|---:|
| 变长 `N=38` | 10,934,272 | 547 MMAC/s |
| 变长 `N=64` | 18,415,616 | **921 MMAC/s** |
| 变长 `N=91` | 26,184,704 | 1,309 MMAC/s |
| 固定 `K=32`（含 router） | 9,310,208 | 466 MMAC/s |
| 固定 `K=16`（含 router） | 4,655,104 | 233 MMAC/s |

MAC/window 为 measured artifact；`@50 Hz` 由 20 ms bin +
`decode_last_timestep_only` 推得，标记为 **assumed decode rate**，若实际
下采样解码需按比例修正。

### Session 路径（每 session 只付一次）

B3 encoder 在 `N=64, T=100, M=30` 下约 `1.30e7 MAC/session`（formula-based，
按 `_mac_per_session` 推导；measured 同源工件为 FALCON M2 `N=96` 的
`21,368,832`）。按实测 session 时长 ~1,200 s 摊薄：

- **≈ 0.011 MMAC/s**
- 相对实时路径 `921 MMAC/s` 约 **1/85,000**

B15 的 `O(N²)` attention 项 `N²·D·3` 在 `N=64` 下为 `786,432 MAC/session`，
摊薄后 ≈ 0.00066 MMAC/s。

**结论：在 session 路径上增加复杂度，成本上确实近似免费。**

## 4. 但"免费"不等于"有用"

`attention_arch_screen_v3`（2026-07-25，见
[`ATTENTION_ARCHITECTURE_SCREEN.md`](ATTENTION_ARCHITECTURE_SCREEN.md)）
用参数匹配对照否定了"既然免费就加 self-attention"这条推论：

- SUA：`B15 − B15P = +0.0064`，6 个 session 中仅 2 个为正；
- MUA：`B15 − B3 = −0.0142`，0/3 为正。

`B15P` 是同参数量、**零跨 neuron 通信**的 per-neuron residual MLP + LayerNorm。

因此本章程第 3 节的不对称性论证应由 **B15P 承载**：它同样只在 session
路径、同样约 35K 参数，但没有 `N²` 项，也不需要为 identity 路径引入全局
neuron 缓冲。attention 阴性结果本身是可发表的，但不能作为架构卖点。

## 5. 实时路径才是真问题

第 3 节的表说明一件此前被低估的事：**实时代价由 per-token read-in 支配，
而 token 数就是 `N`**。变长 `N` 不只是软件上的便利问题，它直接让
`fc_in` MAC、activation SRAM 与编译图随 session 变化。

`fixed_slot_router_pilot_v1` 正是针对这一点：calibration 结束后把 `N` 个
session-local unit 路由压成固定 `K` 个 virtual slot，`A`/`gamma`/`beta`
一次算好缓存，在线窗口不再运行 NeuronID encoder。

当前状态（见 [`FIXED_SLOT_ROUTER_PILOT.md`](FIXED_SLOT_ROUTER_PILOT.md)）：

| Gate | 结果 |
|---|---|
| 固定接口 `[B,K,50]` + permutation 不变 | 通过 |
| cached state 与正常前向等价 | 通过，最大差 `2.62e-6` |
| 压缩率 | 通过，`N=64` 下 `K=32` 2.0×、`K=16` 4.0× |
| **精度** | **未通过**，`K=32` 0.1820 / `K=16` 0.0773，阈值 0.3299 |

路由熵 `0.857 < 0.95`，低温 follow-up 未触发，因此失败不可归因于路由退化
为均匀 pooling。这是真实的 rate–distortion 代价。

**固定 shape 接口在机械层面已证明可部署，代价是当前训练方案下约一半 R²。**
下一条公平分支是同 `K` 的 cached top-K scorer 与 random/activity control，
而不是放宽固定 shape 约束。

## 6. 支撑可重构主张所需的数据（尚未产出）

以下三项都还没有工件，任何对外主张都需要它们：

1. **可重构面积代价 vs 两套专用硬件。** 关键论据是两套专用硬件都要漏电
   而只有一套在跑，但必须给数字。
2. **每种模式的利用率与 energy per inference。** 利用率低本身不可怕
   （阵列按最大形状 size，其他模式必然低），但必须报 energy/inference
   而不是峰值算力。
3. **分模式端到端延迟拆解。** SUA 模式多出检测 + 匹配延迟，MUA 模式跳过。
   闭环 BMI 通常要求 < 50 ms。

第 3 节的表只覆盖了 decoder read-in 与 NeuronID encoder 两项，
不足以回答上述任何一条。

## 7. 本章程不能声称的内容

- 第 3 节的 `@50 Hz` 是假设的解码率，不是实测时序；
- 滤波与模板匹配 block 的数字未测量，本文未给出；
- analytic MAC 不等于面积、功耗或延迟；
- 固定 slot 接口的可部署性已验证，但其精度代价尚未解决；
- session 路径"近似免费"只针对 MAC，未考虑该路径的峰值 SRAM
  （`peak_live_state_bytes`）对面积的影响。

## 8. 相关文档

- [`ATTENTION_ARCHITECTURE_SCREEN.md`](ATTENTION_ARCHITECTURE_SCREEN.md) — attention 机制的参数匹配对照
- [`FIXED_SLOT_ROUTER_PILOT.md`](FIXED_SLOT_ROUTER_PILOT.md) — 固定 token 接口
- [`UNIT_SIDE_FEATURE_ABLATION.md`](UNIT_SIDE_FEATURE_ABLATION.md) — session 路径侧信息消融
- [`CURRENT_RESULTS.md`](CURRENT_RESULTS.md) — 结果口径
- `software-to-hardware/` — B3 INT8/QAT 与硬件交接支线
