# 05 — SPINT Dim256 / TopK64 A8W8 部署目标（尺寸参数化）

来源：`hardware_pe_sram/source/SPINT_A8W8_MIXED_QAT_HARDWARE_SPEC.md`（v0.1）+ `src/models/components/spint.py`、`quantization.py`。

**本文是第一版真正的部署目标**，取代早期 B3-only 视角（`01_OPERATORS.md` 的 B3 仍可作为最简整数 bring-up 子集）。核心策略：**冻结设计方法与量化合同，把所有尺寸降级为可加载 descriptor，尺寸由 DSE 后再冻结**——因此本文把 spec 里"流片固定常量"改标为 `PARAM`。

---

## 1. 三个精度域（务必区分）

| 域 | 覆盖模块 | 权重 | 激活 | 累加 |
|---|---|---|---|---|
| **W8A16** | ID 分支 `fc_id_in`(3)、`fc_id_out`(3) | INT8 per-channel | `P_FP` | FP（P_FP） |
| **A8W8** | `fc_in`(2)、Q/K/V/O(4)、FFN(2) | INT8 per-channel | INT8 per-tensor | **INT32** |
| **P_FP** | RMS/TopK、LN1/LN2、softmax、残差、`fc_out` | — | `P_FP` | FP |

`P_FP` 建议 FP16，但格式（FP16/BF16/FP32）、scale/bias/LN 位宽、DQ 舍入/饱和/denormal 均属 tape-out blocker（见 §7），流片前须由 Q/DQ reference 冻结。

**累加器结论**：A8W8 是 INT8×INT8、最大 reduction 维 = `D`（默认 256），`127²×256 ≈ 2²²`，**INT32 足够**，不需加宽。W8A16 走 FP 累加，无整数溢出问题。（此前"整条 A16W8 需 INT48"的担心不适用于本 spec。）

---

## 2. 网络尺寸（全部 `PARAM`，不写死）

| 符号 | spec 默认 | 建议 DSE 范围 | 含义 |
|---|---:|---|---|
| `N` | 96 | 16–160 | 输入 neuron/token 数 |
| `T` | 50 | — | 解码窗长 |
| `M` | 33 | 1–33 | calibration trials |
| `Tc` | 100 | — | 每 trial 时间长度 |
| `C` | 2 | 1–16 | query/协变量 token |
| `D` | 256 | 128/256 | 模型隐藏宽度 |
| `H` heads | 32 | DSE | attention heads |
| `Dh` | 8 | `D/H` | 单头宽度 |
| `K` | 64 | 32/64/96 | RMS Top-K 保留 token |
| `L` | 1 | 1 | cross-attention 层数 |
| `F` | 256 | `D`/`2D` | FFN 隐层宽度 |

除 `B`（batch）外，语义在给定 model image 内固定；RTL 用 layer descriptor 承载这些值，非法 shape（如 `K>N`）→ `error`，不静默截断。

---

## 3. 数据流（顺序冻结，尺寸参数化）

```text
CALIB_IN[B,M,Tc,N] ─► ID Engine(W8A16) ─► ID_MAP[B,N,T] ─┐
NEURAL_IN[B,T,N] ─► transpose ─► SRC[B,N,T] ─────────────┴─► X = SRC + ID_MAP
   X ─► RMS score ─► TopK(K) ─► gather ─► S[B,K,T]
   REP[1,C,T] ─┐
               ├─► fc_in(A8W8, 共享) ─► KVTOK[B,K,D] / QTOK[B,C,D]
   ┌───────────┘
   LN1 ─► Q/K/V(A8W8) ─► Softmax(P_FP) ─► O ─► out_proj(A8W8) ─► +残差
       ─► LN2 ─► FFN(A8W8, 2层) ─► +残差 ─► fc_out(P_FP) ─► BEHAVIOR_OUT[B,T,C]
```

**顺序不变量**：`ID_MAP` 加到 `SRC` 后先算 RMS/TopK/gather，再对 retained token 跑 `fc_in`；`fc_in` 权重对 `K` 个 KV token 与 `C` 个 query token 复用（两次调用，同权重）。

---

## 4. 两种控制模式（冻结）

| 模式 | 条件 | 行为 |
|---|---|---|
| 完整 | `id_map_valid=0` | 读 `CALIB_IN` → 生成并缓存 `ID_MAP` → 解码一帧 |
| 缓存 | `id_map_valid=1` | 忽略 `CALIB_IN`，复用 `ID_MAP` → 只解码 `NEURAL_IN` |

缓存模式仅在 calibration/权重/`P_FP` 格式均未变时有效；calibration 更新、模型切换、复位后软件须清 `id_map_valid`。`ID_MAP` 容量 `B×N×T` 个 `P_FP`（FP16、B=1、默认尺寸 = 9,600 B）。

状态机：`IDLE → (LOAD_CALIB→ID_ENCODE→ID_CACHE_WRITE | ID_CACHE_READ) → LOAD_NEURAL → ADD_RMS → TOPK_GATHER → READIN → ATTENTION → FFN_OUT → WRITE_OUTPUT → DONE`。错误寄存器：`busy/done/error/id_map_valid/model_crc_match/scale_crc_match/topk_tie_seen`；非法 `K>N`、CRC 不符、缓存模式未载 `ID_MAP`、NaN/Inf、DMA 越界 → 置 `error`、屏蔽 `done`。

---

## 5. 量化合同（方法冻结，数值待签核）

- **权重**（W8，全部量化 Linear）：per-output-channel 对称，`s_w,o = max_i|W_o,i|/127`，`q = clip(round(W/s_w), -127,127)`。
- **激活**（仅 8 个 A8 Linear）：per-tensor 对称，`s_a = max|x|/127`。observer 取观测期历史最大绝对值，**epoch 10 后冻结**；推理**不得**重新 observe 或自适应更新 scale。
- **A8W8 计算**：`y_o = (s_a·s_w,o)·Σ q_a·q_w + b_o`，`Σ` 在 INT32，`s_a/s_w/b/y` 在 `P_FP`。
- **W8A16 计算**：`y_o = Σ x·(q_w·s_w,o) + b_o`，`x/b/s_w/y` 在 `P_FP`（INT8×FP MAC 或等价）。
- `fc_out` 完全不量化（权重/bias/输入/输出全 `P_FP`）。

**Top-K tie-break**：`torch.topk` 分数相等时不保证稳定；流片前必须固定 tie-break（推荐小 neuron index 优先）并软硬一致，否则逐 token 输出不比特一致。

---

## 6. 算力分布与调度（决定 PE 定尺方向）

`B=1` MAC（spec §8，默认尺寸）：

| 阶段 | MAC | 占比 | 频率 |
|---|---:|---:|---|
| `fc_id_in` | 496,336,896 | 94.66% | 每次校准（可缓存/摊销） |
| `fc_id_out` | 13,811,712 | 2.63% | 每次校准 |
| `fc_in`(K+C) | 5,170,176 | 0.99% | 每帧 |
| Q/K/V/O | 8,650,752 | 1.65% | 每帧 |
| attn `QKᵀ`+`PV` | 65,536 | 0.01% | 每帧 |
| FFN | 262,144 | 0.05% | 每帧 |
| `fc_out` | 25,600 | <0.01% | 每帧 |
| **完整模式合计** | **524,322,816** | 100% | |
| **缓存模式（每帧）** | **14,174,208** | — | |

**关键判断**：算力大头在 `fc_id_in`（session-amortized），**实时帧 deadline 由 14.17M 的小路径决定**。因此 PE 阵列同时服务两条路径，但 20 ms deadline 主要压力来自缓存模式；`fc_id` 可低速复用同一阵列。

调度：M0(ID) 作低频 calibration task 并缓存；M1–M4 作每 `T`-bin 实时 task；14 个量化 Linear 共享 INT8 GEMM 阵列 + scale/bias DQ 后处理；LN/RMS/TopK/Softmax/残差/`fc_out` 走独立 `P_FP` 向量单元；attention 按 head/query 分块流式，不要求整 score 矩阵落 SRAM；Top-K 维持 `K` 个 `(score,index)` streaming select。

延迟须分别签核：`t_decode`（缓存模式 p50/p95）、`t_calib`（一次 ID 生成）、`t_amortized = t_decode + t_calib/R`（每 R 帧更新一次校准）。

---

## 7. 尺寸不定下的落地策略

| 现在可做（方法冻结） | 待 DSE/签核（勿写死） |
|---|---|
| 模块拓扑与数据流顺序 | `D/F/K/H/N/C` 最终值 |
| 三精度域边界 + Q/DQ 数学 | `P_FP` 具体格式与各位宽 |
| 两模式 FSM + 错误/CRC 寄存器 | 精度模式：`w8a16`(已验无损) vs `a8w8_mixed` |
| Top-K streaming + 确定性 tie-break | 逐层 scale/bias 数值、最终 checkpoint |
| 共享 INT8 GEMM（A8W8→INT32 / W8A16→FP）+ P_FP 向量单元 | Q/DQ golden、误差阈值、PPA 目标、复位安全 |
| 参数化 layer descriptor + model image 格式 | `B_MAX`、时钟、SRAM 容量、校准周期 R |

**精度模式当一个轴**：`quantization.py` 支持 `w8a16_qat`（8 个 Linear 也走 FP 激活，**已验证无损**）与 `a8w8_mixed_qat`（spec 目标，更小更快）。**同一 datapath 都能跑**——mixed 只是在那 8 个 Linear 上额外开激活量化。建议先用 w8a16 无损基线 bring-up，再切 a8w8 做面积/能耗收益，单独 arm 对照。

---

## 8. Tape-out blocker（spec §10，签核前不可逆决策别做）

1. 最终模型版本 + checkpoint SHA-256；
2. 量化参数包（14 组 INT8 weight、per-channel `s_w`、8 个 `s_a`、bias、LN、`REP`、`fc_out`）+ 各项 CRC；
3. 精确数值格式（`P_FP`、各 scale/bias/LN 位宽、INT32 acc 宽、DQ 舍入/饱和/denormal）；
4. Q/DQ reference（覆盖 14 个量化 Linear，且与硬件 Top-K tie-break 一致）；
5. 数值验证阈值（逐输出 `max_abs/mean_abs`、R² 退化，跨 PyTorch-QAT / Q-DQ / RTL）；
6. 输入前处理归属（芯片接收已备好的 `NEURAL_IN/CALIB_IN`，raw spike 处理不在范围）；
7. 性能/功耗目标（`B_MAX`、时钟、`t_decode/t_calib`、SRAM、带宽、PPA、R）；
8. 复位与安全（`ID_MAP` 掉电是否保留、热切换协议、CRC/NaN 故障上报与输出屏蔽）。

---

## 9. Golden 现状

`golden/` 当前只覆盖 **B3-only(D=64)**，不是本目标。本网络的逐层 golden 必须来自 **实际 checkpoint 的 Q/DQ reference**（§8 item 4），当前拿不到——待那台机器导出 release 后，接入 `golden/` 并补齐 M0–M4 中间张量 dump（D0–D12 对应 attention/FFN 各 stage）。在此之前，本文档 + `tools/energy_model.py` 的参数化 cost model 用于架构探索与面积/带宽估算。
