# B3 QAT-B 软件结果与硬件并行交接

状态日期：2026-07-14  
当前范围：B3-D64 EarlyPool IDEncoder，LOSO fold=0，seed=42  
第一版硬件范围：不包含 cross-attention；只实现本目录定义的 B3 W8A8 identity encoder。

## 1. 当前结论

当前可供硬件架构设计使用的软件候选是：

```text
runs/b3_qat_b_loso0/checkpoints/
  qat-b-epoch=14-val_integer_engine/r2_mean=0.6536.ckpt
```

独立四路径重评：

| 路径 | R² | ΔR² vs anchor |
|------|---:|---------------:|
| anchor FP32 | 0.63024885 | — |
| shadow FP32 | 0.62401214 | -0.00623670 |
| fake quant | 0.65356012 | +0.02331128 |
| NumPy integer engine | 0.65356012 | +0.02331128 |

最佳 checkpoint 的 fake/integer identity 完全一致：

```text
E max_abs = 0
E mean_abs = 0
E RMSE = 0
```

独立报告：`runs/b3_qat_b_loso0/eval_best/eval_paths_report.json`。

判定：

- LOSO0 integer 部署门槛 `ΔR² >= -0.01`：**PASS**。
- 最佳 artifact 的 fake/integer bit-exact：**PASS**。
- shadow 稳定性原门槛 `ΔR² >= -0.005`：**AMBER**，实际为 `-0.00624`。
- 跨 fold/seed 部署结论：**尚未关闭**。

## 2. 收益应如何归因

QAT-B 从修复 STE backward 后的 QAT-A checkpoint warm-start：

```text
runs/b3_qat_a_v2_ste/checkpoints/
  qat-epoch=01-val_integer_engine/r2_mean=0.6375.ckpt
```

R² 演进：

| 阶段 | Integer R² | 相对上一阶段 |
|------|------------:|-------------:|
| fixed-scale epoch=-1 | 0.58665389 | — |
| corrected-STE QAT-A | 0.63752937 | +0.05088 |
| QAT-B epoch 14 | 0.65356012 | +0.01603 |

QAT-B 同时更新 weights 和 scales。因此 scale 明显变化证明 learnable-scale 路径被使用，但不能把 `+0.01603` 全部归因于 scales。严格因果结论仍需要一个相同初始化、epoch、seed、weight LR 和 scheduler 的 fixed-scale continuation control。

## 3. 最佳点的 scale 与 saturation

六个 activation scale 都是全张量共享、可导出并冻结的标量：

| Edge | scale 相对初始化 | heldout saturation |
|------|-----------------:|-------------------:|
| input | 0.739× | 0.158% |
| pre-out | 1.042× | 0.026% |
| mean | 0.667× | 0.423% |
| post0-out | 0.710× | 0.146% |
| post1-out | 0.979× | 0.130% |
| E | 1.082× | 0.042% |

所有 edge 的 saturation 均低于 0.5%。当前优化结果是在少量 outlier clipping 与主体量化分辨率之间取得折中，不需要为这些 scale 增加 per-channel activation scale 硬件。

## 4. 已冻结的硬件计算合同

以下内容可以立即用于微架构设计，并且不依赖后续选择哪一个 QAT checkpoint：

| 项目 | 合同 |
|------|------|
| 网络 | `Linear(100→64)-ReLU-mean(M)-Linear(64→64)-ReLU-Linear(64→64)-ReLU-Linear(64→50)` |
| 输入 | signed INT8，`[M,T,N]`，默认 `M=33,T=100,N=96` |
| 权重 | signed INT8，per-output-channel symmetric scale |
| bias | INT32 accumulator domain |
| dot accumulator | INT32；验证过的当前工作点无 overflow |
| activation | 每个 edge 一个 shared per-tensor INT8 scale |
| trial sum | `SUM_feat[N,64]`，INT32 |
| mean | reciprocal multiply + arithmetic right shift，默认 reciprocal shift=20 |
| requant | per-output-channel integer multiplier，shift=31 |
| requant product | 必须用至少 INT64 保存 `acc_i32 × mult_i32` |
| rounding | `(product + 2^(shift-1)) >>> shift`，与软件 arithmetic shift 一致 |
| 输出 | signed INT8 `E_q[N,50]` + 标量 `E_scale`；需要时再 dequant |

层与张量布局详见 `B3_EarlyPool_network_spec.md`；整数运算语义以 `b3_quant_engine.py` 为准。

## 5. 暂不冻结的模型内容

以下内容必须设计成可加载配置，不能硬编码在 RTL：

- 四层 INT8 weights；
- 四层 INT32 bias；
- per-output-channel weight scales 对应的 requant multipliers；
- 六个 activation scales；
- 每层 requant multiplier/shift；
- reciprocal value 和有效 trial count `M`；
- 当前候选 checkpoint 的版本与 manifest hash。

建议硬件使用 coefficient SRAM/ROM image + CSR 配置。软件继续优化时，只替换模型包，不改 datapath。

## 6. 软件与硬件并行边界

```text
软件持续优化                         硬件立即开始
-------------------------------      --------------------------------
QAT weights / shared scales           INT8 MAC array / PE mapping
fixed-scale continuation control      INT32 accumulator and SUM buffer
更多 fold / seed                      INT64 requant product + shifter
artifact export + manifest            reciprocal mean datapath
最终 golden release                   programmable weight/config memory
                                       layered dump and compare interface
```

硬件现在可以完成：数据流、存储层次、PE 数量、带宽、控制 FSM、DMA/CSR 接口、tiny/full-shape RTL 骨架和随机/旧 golden bring-up。

硬件现在不应完成：把 epoch-14 权重或 scale 固化为不可更新常数，以及以当前单 fold 结果宣布最终模型 sign-off。

## 7. 仍需关闭的风险

1. **验证选择范围**：epoch 14 是在同一个 LOSO0 heldout session 上选出的最佳 epoch；当前结论只覆盖 fold=0 / seed=42。
2. **scale 因果性**：缺少 matched fixed-scale continuation control。
3. **shadow gate**：最佳点 `-0.00624`，略低于原 `-0.005` 健康门槛；integer-only 部署可接受，但状态保持 AMBER。
4. **训练期跨引擎偶发一-code mismatch**：epoch 9、13 的训练日志出现 `E_exact=0`，`E_max_abs≈0.04`；冻结 epoch 14 的独立 CPU engine 评估为 exact。最终 golden 必须来自冻结后的统一 exporter，而不能直接把训练期 CUDA fake path 当合同。
5. **后期漂移**：epoch 14 后 integer R² 回退，epoch 24 为约 0.623。当前必须使用 best checkpoint，不能使用 last checkpoint。

## 8. 最终模型包应包含

在 RTL model-specific sign-off 前，软件侧应导出一个不可变目录：

```text
model_release/
├── manifest.json                 # checkpoint/hash/protocol/shapes/bit widths
├── weights_int8/                 # 四层 Wq
├── bias_int32/                   # 四层 bias accumulator values
├── requant/                      # per-channel mult + shift
├── scales.json                   # 六个 activation scales + weight scales
├── reciprocal.json              # M/recip/shift
├── calib_int8.npy                # 固定验收输入
├── stages_integer/               # 每层整数 golden
└── eval_paths_report.json        # anchor/shadow/fake/integer R²
```

当前脚本已经具备 integer engine 和分层随机 golden，但 QAT best-checkpoint 的完整 release exporter 仍需单独冻结。硬件 bring-up 可以先使用 `b3_int8_validate.py` 的 tiny golden；model-specific sign-off 必须切换到上述 release 包。

## 9. 验收顺序

1. tiny shape：所有整数 stage 逐元素相等；
2. D64/M=1：单 trial、单 neuron/多 neuron；
3. D64/M=33：SUM、reciprocal mean、post MLP；
4. epoch-14 frozen release：完整 `[33,100,96] → [96,50]` bit-exact；
5. 多 calibration draw：检查状态机与 reciprocal，而不是重新拟合 scale；
6. 最终网络 release：重复 full-shape exact compare 后再做 RTL/综合网表 sign-off。

