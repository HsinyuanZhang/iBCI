# B3 决定性量化实验说明（修正版）

对应脚本：`b3_decisive_validate.py` + `b3_eval_protocol.py` + `b3_quant_engine.py`

## 与旧版 oracle 测试的区别

| | `b3_ckpt_quant_validate.py`（旧） | `b3_decisive_validate.py`（修正） |
|--|-----------------------------------|-----------------------------------|
| Scale 来源 | 每个 eval session 自己的 FP32 中间激活 | **6 个 LOSO 训练 session 上冻结的全局 scale** |
| Requant | 浮点 `acc × scale` 再 round | **integer multiplier + shift** |
| 任务指标 | 仅 E cosine/RMSE | **frozen decoder variance-weighted R²（last timestep）** |
| FP32 自检 | 无 | **heldout R² 与 checkpoint artifact 差 >1e-5 则 FAIL** |
| 诊断 | 少量 | **逐层 saturation / INT32 overflow（全 session × 消融）** |
| 稳定性 | 单次 draw（实际相同） | **从完整 calib pool 抽样不同 33-trial index set** |
| 消融 | 无 | **W8A32 / W32A8 / W8A8 / W8A16 / W16A8 / W16A16** |

## 运行

```bash
conda activate ks4
cd software-to-hardware

python b3_decisive_validate.py \
  --ckpt ../streaming_calibration_exp/outputs/streaming_calibration/b3_d64_anchor_s42_20260711_011020/checkpoints/best.ckpt \
  --exp-root ../streaming_calibration_exp \
  --data-dir ../SPINT-main/data/000953 \
  --split-manifest ../streaming_calibration_exp/outputs/streaming_calibration/b3_d64_anchor_s42_20260711_011020/split_manifest.json \
  --expected-heldout-r2 0.63024879 \
  --calib-seeds 42,43,44,45,46 \
  --out runs/b3_decisive_loso0_corrected
```

输出：`runs/.../decisive_quant_report.json`

## 评估协议（必须与 checkpoint 一致）

### LOSO fold=0

| 角色 | Session |
|------|---------|
| Scale 校准（6） | Run2, 10-20-Run1/2, 10-27-Run1/2, 10-28-Run1 |
| 决定性 ΔR²（heldout） | `ses-2020-10-19-Run1` |

- Scale **不得**使用 heldout session（防泄漏）。
- Heldout 评估 calib = val pool 前 33 trials（`random_calibration=False`）。
- R² 窗口 = Lightning `SessionBatchSampler` 的**完整 batch**（batch_size=32，丢弃末尾不足 batch），与 `metrics_per_session.csv` 一致。

### 端到端 R²

1. 收集整个 session 所有 `y_true` / `y_pred`（last timestep only）
2. 按两个行为输出做 **variance-weighted R²**（`torchmetrics.R2Score(multioutput='variance_weighted')`）
3. FP32 基线必须与 artifact 一致：`ses-2020-10-19-Run1` → **R² ≈ 0.63024879**

### 位宽消融定义

| 配置 | 定义 |
|------|------|
| W8A32 | FP32 activation × dequant INT8 weight，FP32 accumulator/output |
| W32A8 | INT8 activation dequant × FP32 weight |
| W8A8 | INT8×INT8、INT32 accumulator、integer requant |
| W8A16 | INT16 activation 使用按 qmax=32767 重新校准的 scale |
| W16A8 | 判断 weight INT8 是否为主要瓶颈 |
| W16A16 | 判断 16-bit fixed point 保底方案 |

A16 **不得**复用 INT8 的 max/127 scale；不能把 FP32 activation 直接 cast 成 INT32。

### Multi-draw 有效性

- 必须从完整 `calib_trialized_neural_features` pool 抽样不同 33-trial index set
- 检查：不同 seed 的 trial indices 不同、calib SHA256 不同；否则实验无效

## 修正后实测摘要（2026-07-13）

checkpoint: `b3_d64_anchor_s42_20260711_011020`，LOSO fold=0

| 检查项 | 结果 |
|--------|------|
| FP32 baseline self-check | **PASS**（measured 0.63024885，Δ=5.6e-08） |
| Multi-draw valid | **PASS**（5 seeds，indices + SHA256 均不同） |
| INT32 overflow（W8A8） | **0** |
| W8A8 heldout ΔR² | **-0.035**（未达 -0.01 门槛） |

### 位宽消融（heldout，frozen scales）

| Preset | Identity cosine | ΔR² |
|--------|-----------------|-----|
| fp32（engine 对照） | 1.000 | ~0 |
| w8_a32 | 0.9996 | **-0.0017** |
| w32_a8 | 0.931 | **-0.048** |
| w8_a8_e8 | 0.935 | **-0.035** |
| w8_a8_e16 | 0.935 | -0.031 |
| w8_a16_e16 | 0.956 | -0.056 |
| w16_a8_e8 | 0.930 | -0.048 |
| w16_a16_e16 | — | —（部分 session INT32 acc overflow） |

**解读：**

- **Activation 量化是主瓶颈**（W32A8 ≈ W8A8 损失量级；W8A32 几乎无损）。
- **E 升到 INT16 收益很小**（W8A8E16 vs W8A8E8）。
- **W8A16 未改善端到端**（identity 更好但 ΔR² 更差，说明 decoder 敏感区域未对齐）。
- 当时的 round-1 frozen max-abs PTQ **未通过** ΔR² ≥ -0.01，因此进入有限 PTQ scale 搜索，并最终进入 W8A8 QAT；后续结果见本文件第四阶段。

## 当时使用的决策树

```
修正验证器 → FP32 baseline 复现？
    ├─ NO  → 实验无效
    └─ YES → PTQ ΔR² ≥ -0.01？
              ├─ YES → W8A8 可接受
              └─ NO  → QAT W8A8 → 仍失败 → 选择性 INT16
```

## 第二阶段 PTQ 搜索（2026-07-13）

脚本：`b3_ptq_search.py`（22 个有界候选，W8A8，scale 仅在 6 训练 session 搜索）

| 策略 | heldout ΔR² |
|------|-------------|
| raw / max_abs（基线） | -0.114 |
| **equalized / max_abs ×0.8（搜索最优）** | **-0.026** |
| equalized / p9999（train 代理最优） | -0.044 |

要点：
- **Cross-layer equalization 显著有效**（-0.114 → -0.026）
- **Percentile scale 在 train identity 更优，但 heldout 泛化差**
- **仍未达 -0.01** → 进入 **W8A8 QAT**（第三阶段）

Scale 校准 trial 抽样现已使用 `stable_session_seed()`（可复现；不再用 Python `hash()`）。

## 第三阶段 QAT 基础设施（2026-07-13）

| 文件 | 作用 |
|------|------|
| `b3_fake_quant.py` | STE、INT8 linear、定点 mean |
| `b3_qat_encoder.py` | `QATEarlyPoolEncoder` |
| `b3_qat_module.py` | Lightning 模块（冻结 decoder） |
| `b3_qat_train.py` | 训练入口 |
| `b3_qat_validate.py` | fake-quant + integer engine 决定性评估 |

早期基础设施使用 anchor ckpt + cross-layer equalization + max_abs×0.8 scales。该初始化后来被 train-only `p9999` 和 corrected-STE QAT-A 取代；最终有效流程与结果见下一节。  
当时已修复 checkpoint `student.*` 前缀加载，确保 encoder 权重与 anchor 一致。

该阶段训练结果使用 `b3_qat_validate.py` 验证 integer engine ΔR²，而不是只看 fake-quant；当前 QAT-B artifact 统一使用 `b3_qat_eval_paths.py` 做四路径独立评估。

## 第四阶段：corrected-STE QAT-A + learnable-scale QAT-B（2026-07-14）

### 关键修复

- exact integer forward 与 differentiable surrogate backward 分离；不再通过 `int64` MAC 张量反传。
- requant surrogate 使用 `acc_sur × eff_scale / out_scale`，移除重复的第二次 `/out_scale`。
- backward coverage 测试确认四层 weight matrix 都有有限、非零梯度，且 optimizer step 后全部发生更新。
- equalization 后 snapshot shadow weight reference；weight penalty 使用归一化 relative MSE。
- `mse_opt` 改为在 quantization-scale 空间搜索；QAT-A 默认使用 train-only `p9999`。
- QAT-B 将 weights 与六个 log-scale 放入独立 optimizer parameter groups。

### 结果演进

| 阶段 | Integer R² | ΔR² vs anchor |
|------|------------:|----------------:|
| fixed-scale epoch=-1 | 0.58665389 | -0.04359496 |
| corrected-STE QAT-A best | 0.63752937 | +0.00728052 |
| QAT-B best（epoch 14） | **0.65356012** | **+0.02331128** |
| QAT-B last（epoch 24） | ~0.62347 | ~-0.00678 |

最佳 artifact：

```text
runs/b3_qat_b_loso0/checkpoints/
  qat-b-epoch=14-val_integer_engine/r2_mean=0.6536.ckpt
```

独立四路径评估：anchor `0.63024885`、shadow `0.62401214`、fake/integer `0.65356012`；最佳点 E bit-exact。

### 判定与保留项

| 检查项 | 状态 |
|--------|------|
| LOSO0 integer `ΔR² >= -0.01` | **PASS** |
| epoch-14 fake/integer exact | **PASS** |
| saturation `<0.5%` on all edges | **PASS** |
| shadow `ΔR² >= -0.005` | **AMBER**（实际 -0.00624） |
| learnable-scale 因果收益 | **OPEN**（缺 matched fixed-scale continuation） |
| 训练全 epoch bit-exact | **FAIL as stated**（epoch 9/13 有一-code mismatch；best artifact exact） |
| 跨 fold/seed sign-off | **OPEN** |

QAT-B 同时继续更新 weights 与 scales，因此不能把 QAT-B 的 `+0.01603` 净增益全部归因于 learnable scales。硬件可以按当前固定算子/位宽合同并行开始，但 weights/scales 必须可加载。详见 `B3_QAT_B_hardware_handoff.md`。
