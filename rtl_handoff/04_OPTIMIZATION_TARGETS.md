# 04 — 值得优化的点（收益 / 风险 / kill 标准）

优化分两档：**A. 现在就能在 B3 RTL 上做的微架构优化**（不动模型、零算法风险）；**B. 需要软件侧实验先关掉 `OPEN` 才能落到 RTL 的方向**（不要提前写死）。

> 面向完整 SPINT Dim256/TopK64 A8W8 目标的优化机会：`fc_id_in` 占 94.66% MAC 但属低频 session 任务，可与实时解码分离调度；实时 frame 截止时间由 ~14.17M-MAC 的 cached 路径决定（`tools/energy_model.py --preset spint_dim256_cached`）。因此“共享 INT8 GEMM 阵列 + 独立 P_FP 向量单元 + ID_MAP 缓存复用”是最高杠杆方向。尺寸/精度细节见 `05_SPINT_DIM256_SPEC.md`。

---

## A. 现在可做（B3，微架构级）

| 点 | 做法 | 预期收益 | 注意 |
|---|---|---|---|
| A1 双 accumulator bank | bank A 计算 / bank B requant+writeback 交换 | 隐藏 requant 延迟，避免阻塞下一 tile | 需验证 tail tile 时序 |
| A2 requant lane 复用 | 8/16 lane 共享，不做 64 份 | 省面积 | 短层（in_dim=8/12/16）要验 requant 不成瓶颈 |
| A3 `1/M` 融进 post0 requant | 固定 M 时把 mean 除法并入 scale | 省一次 reciprocal 路径 | 多 M 仍需 reciprocal LUT |
| A4 Mode A weight 复用 | `8×8` 让 weight 被 8 token 复用 | 省 weight SRAM 读 | Mode B 仅小 token/bring-up 用 |
| A5 clock/power gating | 未用 PE lane、encoder 完成后关 coeff bank | 降 50 Hz duty-cycle 下漏电 | 需 bank 级 gating 支持 |
| A6 partial-word bank 读 | Mode A 只读 8-output slice | 省读能量 | 要确认 512-bit macro 部分读是否仍全读能耗 |

这些都不改数值合同，可直接进 RTL 并用 golden 回归。

---

## B. 需软件实验先行（勿现在写死 RTL）

来自 `hardware_pe_sram/08` 候选池，收敛为四条主线：

| 主线 | 内容 | 预期杠杆 | 风险 / kill 标准 |
|---|---|---|---|
| B-静态 query 编译 | cache query + last-row readout + QK folding | M2 约 **-23% MAC**、去掉 K projection/K SRAM | 若 FP32 无法稳定等价、compiled INT8 task delta 不可恢复、或只单任务有收益 → 降级 |
| B-backbone H/F scaling | `H:512→256/128`、`F/H:2/4` + distillation | 权重/MAC 最大杠杆 | worst-session 灾难性下降、distill/QAT 无法恢复、或 H256 已够则不再缩 |
| B-双数据流 PE 重构 | `8×8 ↔ 1×64` 可重构 | 提升 token-rich / query-poor 利用率 | 重构互连/控制面积过大则退回固定 + 共享 reduction tree |
| B-双速率 memory gating | encoder/decoder bank 互斥生命周期 + bank leakage gating | 降平均功耗（低 duty-cycle 神经解码主项） | 需真实 SRAM macro 才能定量 |

其它候选（online integer softmax、low-precision LN、FFN low-rank、structured pruning、identity top-K、sparse read-in、E→bias folding、linear attention、RMSNorm）优先级更低或创新性弱，见 `08 §2` 候选表与 `§5` kill criteria。

---

## C. 关键判断（避免走偏）

1. **SRAM-first**：对低频（50 Hz）神经解码，coefficient leakage / bank activity / FFN 权重驻留 很可能比 softmax 算术更决定平均功耗。优化优先看存储，不是先换 attention。
2. **attention score 不是主项**：M2 里 `QK + attn×V` 仅约 `0.20M MAC/frame`，主项是 read-in 和 projection。别把工程量投在 score 本身。
3. **H512 是上界不是默认**：不要把 H512/F2048 当 tapeout 规格；它是 Pareto 上端。
4. **每个优化单独 arm**：不要把 quant error、压缩 error、近似 error 混在一个实验里，避免错误归因。

---

## D. 报告纪律（写论文/汇报时）

- `indeterminate` 不写成阴性；validation 不写成 formal test。
- analytic MAC 不写成实测能耗；文献功耗不写成本芯片 PPA。
- query folding 必须声明「一层 / 常量 query」边界。
- 量化后 compiled graph 作为独立部署图，不要求与独立 WQ/WK 路径 bit-exact。
