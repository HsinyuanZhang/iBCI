# software-to-hardware

可拷贝目录：B3 EarlyPool IDEncoder 的网络说明 + **FP32/INT8 分层 golden**，用于软件/硬件结果对照。  
`b3_hw_golden.py`、`b3_quant.py` 和随机 INT8 bring-up 可独立运行；checkpoint、LOSO 与 QAT 流程需要原 SPINT 数据、checkpoint 和 `streaming_calibration_exp`。

## 当前状态（2026-07-14）

- QAT-B epoch 14 在 LOSO fold=0 / seed=42 上，独立 integer engine `R²=0.65356012`，相对 anchor `ΔR²=+0.02331128`。
- 最佳 artifact 的 fake/integer E 完全 bit-exact，所有 activation edge saturation `<0.5%`。
- 当前 checkpoint 可作为 **LOSO0 硬件候选**；跨 fold/seed、matched fixed-scale control 和最终 model-release exporter 尚未关闭。
- 网络优化与硬件架构可以并行：立即冻结算子/位宽/接口，weights/scales 保持可加载，不在 RTL 中硬编码。

详细状态、软硬件边界和 sign-off 清单见 `B3_QAT_B_hardware_handoff.md`。

## 目录

```text
software-to-hardware/
├── README.md
├── requirements.txt
├── B3_EarlyPool_network_spec.md       # 网络计算说明
├── B3_INT8_quantization_baseline.md   # INT8 基准策略（RTL 前必验）
├── B3_QAT_B_hardware_handoff.md       # 当前 QAT 候选与软硬件并行合同
├── early_pool_encoder.py
├── b3_hw_golden.py                    # FP32 分层 golden
├── b3_quant.py                        # INT8 定点前向 + 量化打包
├── b3_int8_validate.py                # 随机权重 INT8 golden / DUT 比对
├── b3_ckpt_loader.py                  # 从 Lightning ckpt 提取 B3 权重
├── b3_ckpt_quant_validate.py        # checkpoint + 真实 calib（oracle scale，feasibility）
├── b3_eval_protocol.py               # LOSO split / R² / calib draw 协议
├── b3_ptq.py                         # PTQ scale strategies + equalization
├── b3_fake_quant.py                  # STE fake-quant ops (W8A8 / INT32 MAC / requant)
├── b3_qat_encoder.py                 # QATEarlyPoolEncoder
├── b3_qat_module.py                  # Lightning QAT training module
├── b3_qat_train.py                   # **W8A8 QAT training**
├── b3_qat_train_b.py                 # learnable shared-scale QAT-B
├── b3_qat_eval_paths.py              # anchor/shadow/fake/integer 独立四路径评估
├── test_b3_qat_backward.py           # STE backward coverage + exact-forward 测试
├── b3_qat_validate.py                # fake-quant + integer engine eval
├── b3_quant_engine.py               # 固定 scale / 整数 requant / 消融引擎
├── b3_decisive_validate.py          # **决定性实验（RTL/PTQ 前必跑）**
├── B3_decisive_quant_experiments.md # 决定性实验说明与判定标准
└── runs/
```

## 依赖

```bash
pip install -r requirements.txt
# 可选交叉验证：
pip install torch
```

## 用法（在本目录内执行）

把本文件夹拷到目标机后，进入拷贝后的绝对路径，例如：

```bash
cd /path/to/software-to-hardware
```

导出 tiny 用例（推荐 RTL 首轮）：

```bash
python b3_hw_golden.py --profile tiny --out runs/tiny_sw
```

导出接近部署规格：

```bash
python b3_hw_golden.py --profile d64 --out runs/d64_sw
```

与同目录内的 PyTorch 实现交叉验证（需 torch）：

```bash
python b3_hw_golden.py --profile tiny --out runs/tiny_sw --torch-check
```

硬件侧把同名 `stages/*.npy`（或至少 `E.npy`）放到 DUT 目录后比对：

```bash
python b3_hw_golden.py --compare runs/tiny_sw runs/tiny_rtl
```

使用已有权重 / 输入：

```bash
python b3_hw_golden.py --profile d64 --out runs/from_ckpt \
  --weights-dir runs/tiny_sw/weights \
  --calib /path/to/calib_MTN.npy
```

`--calib` 形状必须为 `[M, T, N]`，与 `--profile` / `--M --T --N` 一致。

## B3 checkpoint 量化验证（训练权重 + 真实 calib）

**这才是 RTL 前应跑的验证**，不是随机权重冒烟。

```bash
cd /path/to/software-to-hardware

python b3_ckpt_quant_validate.py \
  --ckpt /path/to/b3_d64_anchor/.../checkpoints/best.ckpt \
  --exp-root /path/to/streaming_calibration_exp \
  --data-dir /path/to/SPINT-main/data/000953 \
  --validation-protocol minival \
  --out runs/b3_d64_anchor_ckpt_quant
```

输出：
- `ckpt_quant_report.json`：各 session 的 FP32 vs INT8/INT16 误差
- `ckpt_quant_summary.txt`：汇总
- `<session>/fp32/` 与 `<session>/int8/`：分层 golden（可给 RTL 对照）

仓库内已跑通的参考命令（相对 SPINT 根目录）：

```bash
cd software-to-hardware
python b3_ckpt_quant_validate.py \
  --ckpt ../streaming_calibration_exp/outputs/streaming_calibration/b3_d64_anchor_s42_20260711_011020/checkpoints/best.ckpt \
  --exp-root ../streaming_calibration_exp \
  --data-dir ../SPINT-main/data/000953 \
  --validation-protocol minival \
  --out runs/b3_d64_anchor_ckpt_quant
```

## INT8 量化验证（随机权重 / RTL 连线）

详见 `B3_INT8_quantization_baseline.md`。

```bash
# FP32 参考
python b3_hw_golden.py --profile tiny --out runs/tiny_sw

# INT8 基准：导出整数 golden + 打印相对 FP32 误差
python b3_int8_validate.py --profile tiny --fp-ref runs/tiny_sw --out runs/tiny_int8

# 硬件整数 dump 逐元素比对
python b3_int8_validate.py --compare runs/tiny_int8 runs/tiny_int8_rtl
```

策略摘要：输入 per-tensor INT8；权重 per-channel INT8；bias/dot/SUM INT32；ReLU 后 INT8 requant；`SUM/M` 定点 reciprocal；`E` 同时测 INT8 与 INT16。

## 决定性量化基线与协议回归（LOSO matched）

用于复现 PTQ 基线、检查 LOSO 协议和防止量化引擎回归；当前模型候选与硬件 sign-off 状态以 `B3_QAT_B_hardware_handoff.md` 为准。历史实验与判据详见 `B3_decisive_quant_experiments.md`。

```bash
cd software-to-hardware

python b3_decisive_validate.py \
  --ckpt ../streaming_calibration_exp/outputs/streaming_calibration/b3_d64_anchor_s42_20260711_011020/checkpoints/best.ckpt \
  --exp-root ../streaming_calibration_exp \
  --data-dir ../SPINT-main/data/000953 \
  --split-manifest ../streaming_calibration_exp/outputs/streaming_calibration/b3_d64_anchor_s42_20260711_011020/split_manifest.json \
  --expected-heldout-r2 0.63024879 \
  --out runs/b3_decisive_loso0_corrected
```

硬门槛：FP32 heldout R² 与 checkpoint artifact 差 ≤1e-5；否则整个实验 FAIL。

## 有限 PTQ 搜索（第二阶段）

```bash
python b3_ptq_search.py \
  --ckpt ../streaming_calibration_exp/outputs/streaming_calibration/b3_d64_anchor_s42_20260711_011020/checkpoints/best.ckpt \
  --exp-root ../streaming_calibration_exp \
  --data-dir ../SPINT-main/data/000953 \
  --split-manifest ../streaming_calibration_exp/outputs/streaming_calibration/b3_d64_anchor_s42_20260711_011020/split_manifest.json \
  --out runs/b3_ptq_search_loso0
```

Scale 仅在 6 个训练 session 上搜索；heldout 仅用于最终 ΔR²。停止标准：最优 heldout ΔR² < -0.01 → 进入 QAT。

## W8A8 QAT-A / QAT-B（第三、四阶段）

模拟硬件数值路径训练 B3 encoder（decoder 冻结）：

```bash
# QAT-A：固定 p9999 scales，修复后的 exact-forward / surrogate-backward
python b3_qat_train.py \
  --ckpt ../streaming_calibration_exp/outputs/streaming_calibration/b3_d64_anchor_s42_20260711_011020/checkpoints/best.ckpt \
  --exp-root ../streaming_calibration_exp \
  --data-dir ../SPINT-main/data/000953 \
  --epochs 3 \
  --equalize \
  --scale-method p9999 \
  --lr 5e-6 \
  --out runs/b3_qat_a_v2_ste

# QAT-B：从 corrected-STE QAT-A warm-start，共享 learnable scales
python b3_qat_train_b.py \
  --ckpt ../streaming_calibration_exp/outputs/streaming_calibration/b3_d64_anchor_s42_20260711_011020/checkpoints/best.ckpt \
  --init-qat-ckpt 'runs/b3_qat_a_v2_ste/checkpoints/qat-epoch=01-val_integer_engine/r2_mean=0.6375.ckpt' \
  --exp-root ../streaming_calibration_exp \
  --data-dir ../SPINT-main/data/000953 \
  --epochs 25 \
  --lr-weight 5e-6 \
  --lr-scale 1e-5 \
  --out runs/b3_qat_b_loso0

# 对最佳 QAT-B artifact 做独立四路径评估
python b3_qat_eval_paths.py \
  --qat-ckpt 'runs/b3_qat_b_loso0/checkpoints/qat-b-epoch=14-val_integer_engine/r2_mean=0.6536.ckpt' \
  --anchor-ckpt ../streaming_calibration_exp/outputs/streaming_calibration/b3_d64_anchor_s42_20260711_011020/checkpoints/best.ckpt \
  --exp-root ../streaming_calibration_exp \
  --data-dir ../SPINT-main/data/000953 \
  --out runs/b3_qat_b_loso0/eval_best
```

QAT 路径要点：
- W: fake INT8 per-output-channel（shadow FP32 权重更新）
- A: fake INT8 shared per-tensor scale（QAT-A 固定；QAT-B 六个 log-scale 可学习）
- MAC / requant / SUM/M：与 `b3_quant_engine` 对齐的整数仿真
- Forward：bit-exact integer branch；Backward：独立 float surrogate branch
- Loss：`task + λ_y·distill + λ_E·identity + λ_weight·relative_MSE`
- 成功标准：heldout `ΔR² ≥ -0.01`（integer engine 为准）

当前最佳点：integer `R²=0.65356012`、`ΔR²=+0.02331128`，LOSO0 gate PASS。该 gate 不是跨 fold/seed 的最终 sign-off。

STE 修改后先运行 backward/exact-forward 回归：

```bash
python test_b3_qat_backward.py
```

## 软件优化与硬件架构并行

硬件现在可以基于固定的 D64/W8A8/ACC32/integer-requant 合同开始架构与 RTL；模型权重、bias、mult/shift 和六个 activation scales 必须通过可加载 memory/CSR 提供。详细冻结边界见 `B3_QAT_B_hardware_handoff.md`。

## 对照约定

- Linear：`y = x @ W.T + b`，`W` shape = `[out, in]`（与 PyTorch `nn.Linear` 一致）
- 检查点名称见 `B3_EarlyPool_network_spec.md` 与 `runs/*/meta.json`
- 浮点默认 `atol=1e-5`、`rtol=1e-5`（可用 `--atol` / `--rtol` 改）
