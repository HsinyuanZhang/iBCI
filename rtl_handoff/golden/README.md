# golden/ — 可运行的 B3 golden model

自包含：只需 Python3 + NumPy。用于给 RTL 产生逐层对照向量。

> **范围限定**：本目录目前只覆盖 B3 EarlyPool encoder（W8A8）。完整 SPINT Dim256/TopK64 A8W8 目标（见 `../05_SPINT_DIM256_SPEC.md`）尚无 bit-accurate golden——它需要软件侧先交付冻结 checkpoint 与 Q/DQ reference（`05` 文档 tape-out blocker #2/#4）后才能生成。

## 文件

| 文件 | 作用 |
|---|---|
| `b3_hw_golden.py` | FP32 分层参考（S0–S7）；也可 `--compare` 两个 dump |
| `b3_quant_engine.py` | INT8 W8A8 整数引擎，**整数合同语义以此为准** |
| `early_pool_encoder.py` | PyTorch 参考，仅用于可选 `--torch-check` |
| `export_golden.py` | 一键导出 FP32 + INT8 stage 向量 + testbench hex/dec |

## 快速使用

```bash
# 1) 导出 bring-up 向量（随机权重，输入标定 scale）
python3 export_golden.py --profile tiny --out vectors/tiny
python3 export_golden.py --profile d64  --out vectors/d64  --seed 42
python3 export_golden.py --profile full_m33_d64 --out vectors/full --seed 42

# 2) 纯 FP32 分层参考 + 可选 torch 交叉验证
python3 b3_hw_golden.py --profile d64 --out runs/d64_sw --torch-check

# 3) 对照 RTL dump（把 DUT 的 stages/*.npy 放到 dut 目录）
python3 b3_hw_golden.py --compare runs/d64_sw runs/d64_rtl
```

## 输出布局（`export_golden.py`）

```text
vectors/<name>/
├── manifest.json          # shapes / scales / recip / requant shift / rounding / 诊断
├── calib_i8.{npy,dec,hex} # 输入 INT8
├── E_i8.{dec,hex}         # 最终 identity INT8
├── fp32_stages/*.npy      # pre_linear/feat/sum_feat/mean_feat/post0/post1/E
├── int8_stages/*.npy      # feat_i8/sum_feat_i32/mean_i32/E_i8/E_dequant
└── coeff/<layer>/         # 每层 weight_i8 / bias_i32 / requant_mult_i32 (.npy/.dec/.hex)
```

- `.dec`：一值一行十进制，适合 `$readmemh` 前处理或脚本比对。
- `.hex`：二进制补码定宽（weight/E=2 位、bias/mult=8 位），适合 `$readmemh`。
- 逐元素对照优先用 `int8_stages/`（整数 exact）；`fp32_stages/` 给容差参考。

## 重要边界

- `export_golden.py` 的 scale 是**对当前输入现标定**的，仅供 bring-up；**不是 model sign-off release**。
- model-specific sign-off 必须切换到冻结的 `model_release/` 包（见 `../01_OPERATORS.md` 与 `software-to-hardware/B3_QAT_B_hardware_handoff.md`）。
- 整数 rounding：`(product + 2^(shift-1)) >>> shift`，算术右移；requant shift 默认 31，reciprocal shift 默认 20。
