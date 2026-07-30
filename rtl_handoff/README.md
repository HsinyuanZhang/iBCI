# RTL Handoff — SPINT 神经解码加速器（第一版）

状态日期：2026-07-27  
适用对象：在**非 EDA 机器**上做 RTL 设计/仿真的同事。  
本文件夹是**自包含**的：不依赖仓库其余部分即可读文档、跑 golden、算能效。

---

## 0. 一页速览

- **真实部署目标是完整 SPINT Dim256/TopK64 A8W8 网络**，规格见 `05_SPINT_DIM256_SPEC.md`；B3 encoder 只是其中最简、已 `FROZEN` 的子情形。
- **现在能立刻做 RTL 的只有 B3 encoder 算子**（W8A8 整数合同已 `FROZEN`，有可运行 golden）。完整 SPINT 的 decoder/attention 数值尚未 sign-off。
- **完整 SPINT 三个精度域**：W8A16（ID 分支）、A8W8（8 个高算力 Linear）、P_FP（非线性 + fc_out，建议 FP16）。尺寸按 `05` 文档保持**参数化**（loadable descriptor），不写死。
- **MAC 分布关键点**：`fc_id_in` 占 94.66%，但属低频 session/校准更新任务；实时 frame 截止时间由约 14.17M-MAC 的 cached 路径决定（见 `05` 与 `spint_dim256_cached` preset）。
- 权重/scale/requant 都是**可加载 image**，不要固化进 RTL 常数（当前 checkpoint 只过了 LOSO0/seed42，未跨 fold sign-off）。
- 先按 `01_OPERATORS.md` 的算子顺序做，用 `golden/` 产生逐层向量对齐（**目前仅 B3**），用 `tools/energy_model.py` 做 pre-PDK 能效估算。

---

## 1. 文件夹内容

```text
rtl_handoff/
├── README.md                 # 本文件：总览 + 冻结/可配置边界
├── 01_OPERATORS.md           # 要实现的算子、整数合同、位宽、RTL 实现顺序
├── 02_ENERGY_MODEL.md        # 能效计算方法（session/frame 分开，duty-cycle 加权）
├── 03_VERIFICATION.md        # 检测点 S0–S7、metamorphic、验收顺序、容差
├── 04_OPTIMIZATION_TARGETS.md# 值得优化的点 + 预期收益/风险/kill 标准
├── 05_SPINT_DIM256_SPEC.md   # 完整 SPINT Dim256/TopK64 A8W8 目标（参数化尺寸）
├── golden/                   # 可运行的 golden model（当前仅 B3，FP32 + INT8）
│   ├── README.md
│   ├── b3_hw_golden.py       # FP32 分层参考
│   ├── b3_quant_engine.py    # INT8 W8A8 整数引擎（合同语义以此为准）
│   ├── early_pool_encoder.py # PyTorch 参考（仅可选交叉验证）
│   └── export_golden.py      # 一键导出 stage 向量 + testbench hex/dec
└── tools/
    └── energy_model.py       # 解析能效估算器（系数为占位，需替换 PDK 数据）
```

---

## 2. 冻结 vs 可配置边界（务必遵守）

| 类别 | 内容 | 状态 |
|---|---|---|
| 网络图 | `Linear(100→64)-ReLU-mean(M)-Linear(64→64)-ReLU-Linear(64→64)-ReLU-Linear(64→50)` | `FROZEN` |
| 数值合同 | INT8 w/a、INT32 acc、INT64 requant product、reciprocal mean、算术右移 rounding | `FROZEN` |
| 权重/bias | 四层 INT8 W + INT32 bias | 可加载 image |
| requant | per-output-channel INT32 mult + shift（默认 shift=31） | 可加载 image |
| activation scale | 六个 per-tensor 标量 | 可加载 image |
| reciprocal | `1/M` → `recip`/shift（默认 recip_shift=20） | 可加载 image |
| `M`、`N` | trial 数 / neuron 数（DSE：M 1–33，N 16–160） | 运行时可配 |
| decoder 全部 | `H/F/C/heads`、`X+E`、LN、softmax、静态 query 编译 | `OPEN`，勿写死 |

> 上表针对 B3 encoder 子情形。完整 SPINT Dim256/TopK64 A8W8 目标的三精度域、参数化尺寸与 8 项 tape-out blocker 见 `05_SPINT_DIM256_SPEC.md`。

硬件现在可做：数据流、存储层次、PE 数、带宽、控制 FSM、DMA/CSR、tiny/full RTL 骨架、随机 golden bring-up。  
硬件现在不做：把 epoch-14 权重/scale 固化成常数；用单 fold 结果宣布 sign-off。

---

## 3. 环境需求

- Python 3 + NumPy（跑 golden 与能效）。
- 可选 PyTorch（仅 `--torch-check` 交叉验证）。
- 无需 EDA / 综合工具即可完成本 handoff 的所有对照与估算。

---

## 4. 事实来源（仓库内，供追溯）

- 分层设计：`hardware_pe_sram/00..09*.md`
- B3 合同：`software-to-hardware/B3_QAT_B_hardware_handoff.md`、`B3_EarlyPool_network_spec.md`
- 整数语义：`software-to-hardware/b3_quant_engine.py`

当这些来源变化时，先在 `hardware_pe_sram/09_DECISIONS_OPEN_QUESTIONS.md` 记录影响，再更新本 handoff，避免数字互相矛盾。
