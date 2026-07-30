# 决策记录与开放问题

状态日期：2026-07-27

## 1. 已决策

| ID | 决策 | 状态 | 理由 |
|---|---|---|---|
| D-001 | 新硬件文档独立放在 `hardware_pe_sram/` | `FROZEN` | 不干扰现有实验树 |
| D-002 | 显式拆分 session encode 与 frame decode | `FROZEN` | 两个速率域相差巨大 |
| D-003 | 原始 B3 作为第一套整数 bring-up | `FROZEN` | 已有 W8A8/golden |
| D-004 | weights/scales/mult/shift 可加载 | `FROZEN` | 当前 checkpoint 未最终 sign-off |
| D-005 | INT8 PE、INT32 acc、INT64 requant | `FROZEN` for B3 | 当前整数合同 |
| D-006 | B3T 通过 descriptor 兼容 | `PROVISIONAL` | 成本更优，尚缺量化 release |
| D-007 | B3A 不进入第一版 | `DEFERRED` | 零净信号、约 30× state |
| D-008 | fixed-slot 不作为第一版依赖 | `DEFERRED` | 当前精度 gate 失败 |
| D-009 | decoder 第一版按一层设计 | `PROVISIONAL` | 当前 M1/M2/H1 均一层 |
| D-010 | 64 PE 为起点，128 PE 为对照 | `PROVISIONAL` | H512 32 PE 无实时余量 |
| D-011 | PE 支持 `8×8` 与 `1×64` | `PROVISIONAL` | token-rich/query-poor形状 |
| D-012 | 先保留 accurate softmax/LN reference | `FROZEN` | 近似需要独立消融 |
| D-013 | static query folding 进入 pilot | `PROVISIONAL` | M2 有明显解析收益 |
| D-014 | H/F 必须在 SRAM 定型前扫描 | `FROZEN` as process | 宽度是最大面积杠杆 |

## 2. 开放硬件问题

| ID | 问题 | 需要的证据 | 决策截止点 |
|---|---|---|---|
| H-001 | 64 还是 128 PE | cycle model + synthesis | top-level RTL freeze |
| H-002 | activation SRAM 128/192/256 KiB | liveness + SRAM compiler | memory macro selection |
| H-003 | coefficient SRAM 总容量 | H/F Pareto + compiler | floorplan |
| H-004 | 8×64b 或其他 banking | macro energy/area/timing | SRAM integration |
| H-005 | materialized or online softmax | cycle/energy/accuracy | NLU freeze |
| H-006 | LayerNorm sumsq/rsqrt 位宽 | trace + range analysis | numeric contract |
| H-007 | 是否保留 standard K RTL 路径 | area/debug value | decoder RTL freeze |
| H-008 | trial buffer full or tiled | DMA/control/energy | encoder controller |
| H-009 | ECC/parity | reliability/area | sign-off |
| H-010 | max runtime N | dataset + memory/timing | contract v1 |

## 3. 开放模型问题

| ID | 问题 | 需要的实验 |
|---|---|---|
| M-001 | 最终 encoder 是 B3 还是 B3T | B3T W8A8 + paired validation |
| M-002 | H/F 最小可接受工作点 | multi-seed distillation/QAT |
| M-003 | C16 任务是否适合 QK folding | M1 shape/model experiment |
| M-004 | heads/head_dim | H/F/head Pareto |
| M-005 | X+E scale | end-to-end QAT |
| M-006 | softmax/LN approximation | paired task R² |
| M-007 | FFN 是否可低秩/缩宽 | memory-first ablation |
| M-008 | top-K neuron 是否恢复精度 | top-K/random/activity controls |
| M-009 | formal test scope | 项目治理决策，不由硬件自行选择 |

## 4. 明确不做

第一版不实现：

- on-chip backprop；
- arbitrary Transformer ISA；
- B3A retained-trial buffer；
- mandatory fixed-slot interface；
- general divider；
- global FP32 datapath；
- automatic session-specific scale fitting from test data；
- multi-layer static-query folding；
- 以单 checkpoint 常数综合成不可更新 ROM。

## 5. 决策依赖图

```text
H/F accuracy sweep
  ├─> coefficient SRAM capacity
  ├─> 64/128 PE deadline
  └─> floorplan feasibility

static-query FP32/QAT
  ├─> K/R weight image
  ├─> attention schedule
  └─> score SRAM

B3T W8A8
  ├─> final encoder descriptor
  ├─> fixed basis format
  └─> encoder model release

SRAM compiler
  ├─> bank topology
  ├─> activation capacity
  ├─> leakage argument
  └─> unified-vs-dedicated comparison
```

## 6. 决策模板

新增决策时记录：

```text
ID:
date:
owner:
question:
options:
evidence:
decision:
hardware impact:
software impact:
verification impact:
contract version impact:
revisit condition:
```

## 7. 版本记录

### 2026-07-27 / v0.1

- 建立总体与模块级文档；
- 固定双 API 与状态标记；
- 记录 64-PE 双模式起点；
- 纳入 B3T、static query folding 和 H/F DSE；
- 明确 B3A/fixed-slot 第一版不做；
- 将 softmax/LN、SRAM 容量和最终 backbone 标记为开放项。

