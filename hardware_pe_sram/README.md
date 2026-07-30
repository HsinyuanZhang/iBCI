# PE–SRAM 神经解码硬件文档

状态日期：2026-07-27

## 1. 目录目标

本目录把当前 SPINT 硬件化讨论拆成两层：

1. **总体层**：系统目标、速率域、模块边界、第一版芯片范围与总体验收；
2. **模块层**：PE、SRAM、NeuronID encoder、cross-attention decoder、量化、
   验证/PPA 和科研实验各自独立描述。

当前项目对象是 **intracortical spike-count motor decoding**，主要包括 sorted
SUA 与 threshold-crossing/MUA，不是传统 scalp EEG。参考论文中的 EEG/ECoG
硬件方法只作为算子、量化和存储设计参考，不能直接替代本项目的任务验证。

## 2. 核心结论

第一版硬件不等待所有模型超参数定型，但也不把当前 checkpoint 或 `H=512`
写死在 RTL 中。建议冻结：

- session/online 两条显式 API；
- INT8 affine、INT32 accumulate、INT64 requant、reduction、LayerNorm、
  softmax 等算子语义；
- runtime shape descriptor 与 memory image 格式；
- 分层 golden、debug dump 和验收方法。

建议保留可配置：

- `N`、`M`；
- encoder 选择（B3/B3T）；
- decoder 的 `H/F/C/heads/head_dim`；
- weights、bias、scale、multiplier、shift、LUT 版本；
- coefficient/activation SRAM 的最终容量。

建议第一版不支持：

- B3A trial-axis attention；
- 以 fixed-slot router 作为必需接口；
- 通用 FP32 datapath 或通用除法器；
- 多层 cross-attention 的完整优化路径；
- 把单 fold/seed 的 QAT checkpoint 固化为 ROM。

## 3. 状态标记

本目录统一使用以下四种状态：

| 状态 | 含义 | 硬件动作 |
|---|---|---|
| `FROZEN` | 已有明确软件语义或整数合同 | 可进入 RTL/接口验收 |
| `PROVISIONAL` | 方向明确，但尺寸、实现或模型尚需 DSE | 做参数化实现，不固化常数 |
| `OPEN` | 需要实验、PDK、SRAM compiler 或用户决策 | 不据此做不可逆决策 |
| `DEFERRED` | 第一版明确不做 | 不为其预留专用面积 |

文档中的“冻结”只约束硬件语义，不表示某个模型 checkpoint 已完成算法
sign-off。

## 4. 推荐阅读顺序

1. [`00_SYSTEM_OVERVIEW.md`](00_SYSTEM_OVERVIEW.md)  
   总体目标、双速率架构、第一版范围和系统数据流。
2. [`01_HW_SW_CONTRACT.md`](01_HW_SW_CONTRACT.md)  
   tensor、API、layer descriptor、模型包与软硬件责任边界。
3. [`02_PE_ARRAY.md`](02_PE_ARRAY.md)  
   64-PE 双数据流阵列、执行模式、带宽与周期模型。
4. [`03_SRAM_MEMORY.md`](03_SRAM_MEMORY.md)  
   coefficient/activation/session SRAM、banking、生命周期与容量 DSE。
5. [`04_NEURONID_ENCODER.md`](04_NEURONID_ENCODER.md)  
   原始 B3、B3T、mean reducer、session state 和量化交接。
6. [`05_CROSS_ATTENTION_DECODER.md`](05_CROSS_ATTENTION_DECODER.md)  
   在线主路径、静态 query 编译、LayerNorm/softmax 与 backbone 扫描。
7. [`06_QUANTIZATION_NUMERICS.md`](06_QUANTIZATION_NUMERICS.md)  
   已冻结 B3 整数合同与 decoder 待冻结数值格式。
8. [`07_VERIFICATION_PPA.md`](07_VERIFICATION_PPA.md)  
   软件、整数模型、RTL、综合网表和 PPA 验收矩阵。
9. [`08_RESEARCH_ROADMAP.md`](08_RESEARCH_ROADMAP.md)  
   科研假设、实验矩阵、两周 pilot 和论文叙事。
10. [`09_DECISIONS_OPEN_QUESTIONS.md`](09_DECISIONS_OPEN_QUESTIONS.md)  
    当前已决策、待决策、弃用项和版本记录。
11. [`GLOSSARY.md`](GLOSSARY.md)  
    统一符号、信号类型、attention 类型和硬件术语。

## 5. 文档与现有仓库的关系

本目录不替代以下事实来源：

- `software-to-hardware/B3_EarlyPool_network_spec.md`：原始 B3 计算图；
- `software-to-hardware/B3_QAT_B_hardware_handoff.md`：B3 W8A8 合同；
- `sua_exploration/docs/CURRENT_RESULTS.md`：当前算法结果及证据边界；
- `sua_exploration/docs/ASIC_DEPLOYMENT_CHARTER.md`：可重构芯片定位；
- `SPINT-main/src/models/components/spint.py`：原始 decoder 语义。

当这些来源更新时，应先在
[`09_DECISIONS_OPEN_QUESTIONS.md`](09_DECISIONS_OPEN_QUESTIONS.md)
记录影响，再更新对应模块文档；不得只改一处数字而让总体与模块文档互相矛盾。

## 6. 当前一页式状态

| 部分 | 状态 | 当前判断 |
|---|---|---|
| 系统双 API | `FROZEN` | `encode_session` 与 `decode_frame` 必须分离 |
| 原始 B3 算子 | `FROZEN` | 可作为第一套 bit-exact RTL bring-up |
| 原始 B3 最终权重 | `OPEN` | fold/seed sign-off 未关闭，权重必须可加载 |
| B3T 数据流支持 | `PROVISIONAL` | 建议支持 `100→12→64`，尚缺 W8A8 release |
| B3A | `DEFERRED` | 零净信号且 support state 约 30× |
| cross-attention 层数 | `PROVISIONAL` | 第一版以一层为目标 |
| decoder 宽度 `H/F` | `OPEN` | 必须由精度–SRAM Pareto 决定 |
| 静态 query 编译 | `PROVISIONAL` | FP32 应精确等价，INT8 需 compiled-graph QAT |
| PE 数 | `PROVISIONAL` | 64 PE 起点，128 PE 作为 DSE 对照 |
| SRAM 容量 | `OPEN` | 等 SRAM compiler 与 activation liveness |
| fixed-slot | `DEFERRED` | 当前精度 gate 未通过 |
| softmax/LN 格式 | `OPEN` | 先保留准确基线，再评估近似 |
| PPA 结论 | `OPEN` | 未有工艺、宏、activity 前不报告可信 mW |

