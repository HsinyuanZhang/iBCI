# B3 INT8 量化基准（RTL 前必验）

> 本文件记录第一轮 **硬件友好基准**，用于随机/tiny RTL bring-up，不代表当前模型精度上限。  
> 实现：`b3_quant.py`；导出与验收：`b3_int8_validate.py`。

> 2026-07-14 状态：本文件的随机/round-1 baseline 继续用于 RTL bring-up；模型精度候选已经更新为 QAT-B epoch 14。当前 LOSO0 integer `ΔR²=+0.02331`，详见 `B3_QAT_B_hardware_handoff.md`。

## 量化策略

| 张量/计算 | 第一候选 |
|-----------|----------|
| cubic 输入（trial 激活） | signed INT8，**per-tensor symmetric** |
| weights | signed INT8，**per-output-channel symmetric** |
| bias | **INT32**（累加器域：`round(b_fp / (s_x * s_w[o])`） |
| dot-product accumulator | **INT32** |
| ReLU 输出 | **INT8 requant**（先乘有效 scale、ReLU、再按 `output_scale` 量化） |
| 跨 trial `SUM_feat` | **INT32**（对 `feat_i8` 按 trial 累加） |
| `SUM/M` | **定点 reciprocal**：`mean = (SUM * round(2^s/M) + 2^(s-1)) >> s`，默认 `s=20` |
| post MLP accumulator | **INT32** |
| 最终 `E` | 同时导出 **INT8** 与 **INT16**（各自 symmetric scale） |

## 数据流（整数域）

```text
calib_fp  --quant-->  calib_q [M,T,N] int8
  per (m,n):  acc_i32 = calib_q @ W_pre^T + bias_i32
              feat_i8 = requant_int8(ReLU(acc * s_x * s_w))

SUM_feat_i32[n,d] = Σ_m feat_i8[m,n,d]
mean_i32 = reciprocal_mean(SUM_feat_i32, M)
mean_fp_est = mean_i32 * s_pre_out
mean_q = quant_int8(mean_fp_est)

post0/1: 同 pre（int8 乘 + int32 累加 + ReLU + int8 requant）
post2:    无 ReLU → E_fp → E_int8 / E_int16
```

## 导出目录结构

```text
runs/<name>/
├── quant_meta.json          # scales, bias_i32, reciprocal, vs FP32 误差
├── quant_summary.txt
├── quant_weights/
│   ├── pre_pool_w_int8.npy
│   ├── pre_pool_w_scale.npy
│   ├── pre_pool_bias_i32.npy
│   ├── post0_... post1_... post2_...
├── stages_int8/
│   ├── calib_q.npy
│   ├── feat.npy              # int8
│   ├── sum_feat_i32.npy
│   ├── mean_i32.npy
│   ├── mean_q.npy
│   ├── post0_acc.npy         # int32
│   ├── post0_relu.npy        # int8
│   ├── ...
│   ├── E_int8.npy
│   └── E_int16.npy
├── E_int8.npy
├── E_int16.npy
├── E_int8_dequant.npy        # 与 FP32 比精度用
└── E_int16_dequant.npy
```

## 用法

```bash
cd /path/to/software-to-hardware

# 1) 先生成 FP32 golden（可选，但推荐同一 calib/权重）
python b3_hw_golden.py --profile tiny --out runs/tiny_sw

# 2) INT8 验证 + 导出（复用 FP32 的 calib/weights）
python b3_int8_validate.py --profile tiny --fp-ref runs/tiny_sw --out runs/tiny_int8

# 3) 或一步随机权重（快速冒烟）
python b3_int8_validate.py --profile tiny --out runs/tiny_int8

# 4) RTL/DUT 整数 dump 对齐
python b3_int8_validate.py --compare runs/tiny_int8 runs/tiny_int8_rtl
```

## 验收建议（RTL 前）

1. **整数 exact**：`stages_int8/*.npy`、`E_int8`/`E_int16` 与软件 golden **逐元素相等**（`--compare`）。
2. **相对 FP32**：查看 `quant_meta.json` → `validation_vs_fp32`：
   - `E_int16_dequant` 通常优于 `E_int8_dequant`
   - 全规格目标可先以 `E_int16 max_abs` 相对 FP32 为参考（随机权重仅用于连线；**训练权重**需单独跑）
3. **饱和**：随机 baseline 的 `b3_quant.py` 在量化时执行 clip；QAT 候选的逐 edge saturation 由 `b3_qat_encoder.py`/`b3_qat_module.py` 记录，epoch-14 实测均 `<0.5%`。
4. **M sweep**：对 `M∈{1,2,4,7,33}` 重复 `--fp-ref` 流程，确认 reciprocal LUT 覆盖。

## 与 RTL 的接口

硬件应实现与 `b3_quant.py` 相同的：

- `weight_scale[o]`、`bias_i32[o]`  per layer
- `input_scale` / 各层 `output_scale`（存 CSR 或 header）
- `reciprocal.recip`、`reciprocal.shift`  per session（或 LUT 索引 `M`）
- 累加器位宽：**INT32**（本基准已验证不溢出：T=100,D=128,M=33）

硬件架构和通用 RTL 可以现在开始，但必须保持 weights/scales/mult/shift 可加载。最终 model-specific 综合验收仍要求冻结 QAT release、生成对应分层 integer golden，并与 RTL `--compare` PASS。

## B3-D64 anchor 实测（round-1 基准，M=33，cubic calib）

checkpoint：`b3_d64_anchor_s42_20260711_011020/checkpoints/best.ckpt`  
7 个 held-in session（`--validation-protocol minival`）汇总量级：

| 指标 | E INT8 dequant | E INT16 dequant |
|------|----------------|-----------------|
| mean RMSE vs FP32 E | ~0.15 | ~0.15 |
| mean cosine | ~0.95 | ~0.95 |
| median abs err (单 session) | ~0.05 | ~0.05 |
| p95 abs err | ~0.3 | ~0.3 |
| max abs err（少数神经元 outlier） | ~1.5–1.6 | ~1.5–1.6 |

解读：
- 该 round-1 identity-only 测试表明整体形状保留较好，但少数神经元会拉高 `max_abs`。
- 后续 frozen-decoder 决定性实验已经完成：单独把 `E` 提升到 INT16 收益很小，端到端判据必须使用 R²，而不能只看 identity RMSE/cosine。
- corrected-STE QAT-B 已使 W8A8 在 LOSO0 达到 `ΔR²=+0.02331`；当前不采用 E INT16 或 activation per-channel scale。下面保留 round-1 数字仅用于历史对照。

复现：`python b3_ckpt_quant_validate.py`（见 `README.md`）。

## 当前 QAT-B 候选（2026-07-14）

round-1 结果只描述未经 QAT 的 anchor，不代表当前 W8A8 上限。修复 scale、equalization reference 和 STE backward 后：

| Artifact | Integer R² | ΔR² vs FP32 anchor | E exact |
|----------|------------:|--------------------:|:-------:|
| QAT-A corrected-STE best | 0.63752937 | +0.00728052 | yes |
| QAT-B epoch 14 | **0.65356012** | **+0.02331128** | yes |

QAT-B 最佳点所有 activation edge saturation `<0.5%`，因此当前硬件合同仍是 W8A8 + INT32 accumulator + shared per-tensor activation scales；不需要因为旧 PTQ 失败而升级到全局 INT16。
