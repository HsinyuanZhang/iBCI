# M1 / M2 / H1 source-only static decoder

这条路线与当前 learned-recency 的 FULL、ACT 和 NORM 位于同一子项目。它从头训练完整 decoder，通道表示、前端、temporal 和 readout 都只接受 source 监督；推理时使用冻结权重，不构建目标 calibration bank。

## 模型与信息边界

用任务内共享的 `static_identity[N,16]` 替代 `e0_proj(E0)`。M1 为 `64×16`，M2 为 `96×16`，H1 为 `176×16`；每个任务只有一张表，所有 source / target session 共用。该表由独立 seed 域初始化，并与完整 decoder 一起训练、进入 EMA 和 checkpoint。`e0_proj` 被移除，carrier 为模型内部恒零的 `[N,4]` buffer。

前端仍为 `raw spikes → causal k5 convolution → local + static_identity → token MLP → 8-slot attention`，后接当前 learned-recency temporal 与 readout。保留原始 spike 输入，本版不额外拟合或应用归一化统计。`input_valid_mask` 标识时间左侧 padding，真实的全零 spike bin 仍是有效观察。

输入接口只接受 raw neural、unit mask、dropout mask 和时间有效性 mask；不接受 `TaskBank`、E0 或外部 carrier。训练加载 source 的 query neural / supervised targets / endpoint indices；评分加载目标 query neural / scoring labels / masks。标签只用于 source loss 或独立评分，不输入模型。评分不打开独立 support 张量，也不计算目标发放率、目标身份或目标归一化统计。

**通道对应假设：** 每个任务的数据加载器在各 session 保持固定列索引，M1 为 64 列、M2 为 96 列、H1 为 176 列。`static_identity[u]` 对应输入第 `u` 列，不能单独打乱输入列。这里依赖工程上的通道索引对应，不声称跨 session 是同一个生物学神经元。

## 全训配方与 checkpoint 选择

| 项目 | M1 | M2 | H1 |
|---|---|---|---|
| Context | 100 bins | 50 bins | 300 bins |
| Temporal | D4, width 256, 8 heads | D4, width 256, 8 heads | D4, width 256, 8 heads |
| Layer windows | 25 / 25 / 25 / 24 | 13 / 12 / 12 / 12 | 75 / 75 / 75 / 74 |
| Identity | `64×16` source-trained table | `96×16` source-trained table | `176×16` source-trained table |
| 参数张量元素总数 | 3,533,072（相对 FULL −576） | 3,531,778（相对 FULL +736） | 3,533,703（相对 FULL −8,384） |
| 全训预算 | 24 × 6,665 = 159,960 updates | 24 × 3,165 = 75,960 updates | 32 × 731 = 23,392 updates |
| 默认 checkpoint | HO3 all24 earliest-max（末轮 e24 仅 sidecar） | EXT6 all24 earliest-max（末轮 e24 仅 sidecar） | HO-M3 all32 earliest-max（末轮 e32 仅 sidecar） |
| 本地评分面 | HO3 query | EXT6 query | 当前 H1 HO development query，沿用 grouped-seven 指标 |

三条入口均固定 `seed=42`、P16、`learned_slope`、per-layer、D4，以及半衰期 `4,8,16,32,64,128` bins 和两个固定 flat heads。沿用各任务现有的采样、目标缩放、AdamW、warmup-cosine、EMA `0.9995` 和 whole-unit dropout `0.1`。通道表使用普通 decoder 参数组；新增 recency 参数沿用 `weight_decay=0` 参数组。M1 另外保留当前主线按参数名称划分的 norm/bias 免衰减组，监督目标使用原生 EMG 数值。

选轮与同任务 FULL 相同：目标 development 面完整扫描 + earliest-max（M1 HO3 all24、M2 EXT6 all24、H1 HO-M3 all32）。末轮 EMA 只作 sidecar 对照，**不得作为此后 EvalAI 选点**。入口：`m1_static_train.py --stage pick`、`m2_static_score.py --pick`、`h1_static_train.py --stage pick`。提交必须使用扫描收据中的 `selected_epoch`。已交 582290 是末轮 e32 例外，扫描后若不是 32 另交。

M1 / H1 当前本地 query 面的 NWB 文件名含 `held-out-calib`；这里沿用原评分代码的 raw neural、endpoint 和 eval mask，不构建独立 support，也不调用 calibration materializer。常规 causal 输入历史是 decoder 的观察，不用于更新权重或校准统计。

## 验收范围

Smoke 使用真实本地 source / query 数据，缩小更新数或 batch，保留模型尺寸与 context。它验证训练更新、有限 loss、通道表梯度和更新、EMA / checkpoint 保存与重载、推理一致性以及有限的 query 评分。Smoke 的分数不代表收敛效果；正式训练的 24 / 32 轮仍需另行运行。

## 运行命令

以下命令从 workspace 根目录 `/home/xinyuan/Work_host/SPINT` 执行。结果目录必须为空；复测时更换 `--dest`，正式训练不传 smoke 参数。

```bash
STATIC_PY=/home/xinyuan/miniconda3/envs/spint/bin/python
RECENCY_PKG=btransform_unified_v2/learnable_recency_v1

# M1 训练与 HO3 query smoke：2 次 B32 更新，评分每 session 1 个 batch。
env -u PYTHONPATH PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES='' \
  "$STATIC_PY" "$RECENCY_PKG/scripts/m1_static_train.py" \
  --stage train --device cpu --cpu-threads 2 --max-updates-smoke 2 \
  --dest "$RECENCY_PKG/results/smoke/m1_static_s42_recheck"
env -u PYTHONPATH PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES='' \
  "$STATIC_PY" "$RECENCY_PKG/scripts/m1_static_train.py" \
  --stage score --device cpu --cpu-threads 2 --allow-smoke --max-batches 1 \
  --dest "$RECENCY_PKG/results/smoke/m1_static_s42_recheck"

# M2 训练与 EXT6 query smoke：2 次更新，评分每 session 1 个 batch。
env -u PYTHONPATH PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES='' \
  "$STATIC_PY" "$RECENCY_PKG/scripts/m2_static_train.py" \
  --device cpu --cpu-threads 2 --max-updates-smoke 2 \
  --dest "$RECENCY_PKG/results/smoke/m2_static_learned_slope_s42_recheck"
env -u PYTHONPATH PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES='' \
  "$STATIC_PY" "$RECENCY_PKG/scripts/m2_static_score.py" \
  --run-dir "$RECENCY_PKG/results/smoke/m2_static_learned_slope_s42_recheck" \
  --dest "$RECENCY_PKG/results/smoke/selection_m2_static_ext6_s42_recheck" \
  --device cpu --cpu-threads 2 --allow-smoke --max-batches 1

# H1 训练 smoke：2 次更新，包含流式检查、重载检查与有限 query 评分。
env -u PYTHONPATH PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES='' \
  "$STATIC_PY" "$RECENCY_PKG/scripts/h1_static_train.py" \
  --device cpu --max-updates-smoke 2 \
  --dest "$RECENCY_PKG/results/smoke/h1_static_s42_recheck"
```

正式训练命令已准备好，本次未运行：

```bash
# M1：24 epochs，之后 --stage score 报末轮，--stage pick 做 HO3 earliest-max。
env -u PYTHONPATH PYTHONNOUSERSITE=1 \
  "$STATIC_PY" "$RECENCY_PKG/scripts/m1_static_train.py" \
  --stage train --device cuda:0 --dest "$RECENCY_PKG/results/m1_static_s42"
env -u PYTHONPATH PYTHONNOUSERSITE=1 \
  "$STATIC_PY" "$RECENCY_PKG/scripts/m1_static_train.py" \
  --stage score --device cuda:0 --dest "$RECENCY_PKG/results/m1_static_s42"

# M2：24 epochs，之后 m2_static_score.py 报末轮，加 --pick 做 EXT6 earliest-max。
env -u PYTHONPATH PYTHONNOUSERSITE=1 \
  "$STATIC_PY" "$RECENCY_PKG/scripts/m2_static_train.py" \
  --device cuda:0 --dest "$RECENCY_PKG/results/m2_static_learned_slope_s42"
env -u PYTHONPATH PYTHONNOUSERSITE=1 \
  "$STATIC_PY" "$RECENCY_PKG/scripts/m2_static_score.py" \
  --run-dir "$RECENCY_PKG/results/m2_static_learned_slope_s42" \
  --dest "$RECENCY_PKG/results/selection_m2_static_final_ema_ext6_s42" \
  --device cuda:0

# H1：32 epochs，训练完成后 --stage score 报末轮，--stage pick 做 HO-M3 earliest-max。
env -u PYTHONPATH PYTHONNOUSERSITE=1 \
  "$STATIC_PY" "$RECENCY_PKG/scripts/h1_static_train.py" \
  --device cuda:0 --dest "$RECENCY_PKG/results/h1_static_s42"
```

若 H1 仅需重启评分，沿用同一 `--dest` 并添加 `--stage score`。M2 / H1 训练入口的 `--resume` 只接受所属运行目录下的完整正式 epoch checkpoint，恢复 optimizer、EMA 与 RNG；smoke checkpoint 不用于正式续训。M1 沿用当前主线的从头训练方式，未提供 `--resume`。

## 2026-09-11 M2 / H1 原始验收结果

最终验收通过。两个任务均使用完整模型、原 context 和 `batch=32`，在 CPU 上执行 2 次 optimizer update / 2 次 EMA update。

| 检查 | M2 | H1 |
|---|---|---|
| 真实 source 训练 smoke | 2 updates，约 16.50 秒 | 2 updates，约 63.62 秒 |
| 静态通道表确实更新 | 通过 | 通过 |
| EMA 含静态通道表且执行第二次滑动更新 | 通过 | 通过 |
| loss / gradient 有限 | 通过 | 通过 |
| checkpoint 重载预测最大差值 | 0 | 0 |
| 真实 query 评分 smoke | EXT6：6 sessions × 32 = 192 windows | 14 个文件 × 2 = 28 windows，聚合为 7 session groups |
| 实际 R300 输入的 stream / offline 最大差值 | 另由跨窗口模型测试覆盖 | `3.43e-7` |
| 运行记录中的代码哈希与当次验收实现一致 | 通过 | 通过 |

新增 static 测试 **16 项通过**；既有 NORM / ACT、初始化和 learned-recency streaming 回归 **55 项通过**。静态模型测试覆盖 M2 55 bins、H1 305 bins 的跨窗口流式一致性、不同层 slope、batch 重排、reset、调用方复用输入张量，以及通道表、前端、temporal 和 slope 的梯度。

独立对照还确认 H1 的 13 个 source session、23,212 个端点及其 valid masks 的 SHA-256 与当前 NORM 完全一致，每轮均为 731 updates。M2 沿用并验证原 24-epoch sampler manifest、source session 长度和每轮 3,165 batches。

产物：

- [汇总验收收据](../results/smoke/static_smoke_audit.json)：checkpoint 哈希、代码哈希、参数组、通道表实际更新、EMA 状态及测试汇总。
- [M2 训练 smoke](../results/smoke/m2_static_learned_slope_s42_final/smoke_receipt.json) 与 [M2 EXT6 评分 smoke](../results/smoke/selection_m2_static_ext6_s42_final/score_receipt.json)。
- [H1 训练、重载、流式及 query smoke](../results/smoke/h1_static_s42_final_v2/smoke_receipt.json)。
- [H1 与 NORM 的 source 端点 / mask 对照](../results/smoke/static_h1_source_contract_audit.json)。

这些 smoke checkpoint 仍处于训练最初两步，有限 query 的 R² 只用于检查评分路径，不能用于判断收敛或比较 NORM。正式 24 / 32 轮训练和最终性能评估尚未运行。

## 2026-09-11 M1 扩展验收

M1 已接入同一 STATIC 模型与流式接口。真实 CPU smoke 使用完整 `R100 / D4 / P16 / B32`，完成 2 次 optimizer update 和 2 次 EMA update。静态通道表确实更新，loss 与 learned slope 梯度有限，slope 梯度非零；raw / EMA checkpoint 均从磁盘重载并与保存前状态核对。HO3 有限评分覆盖 3 个 session、每个 32 个窗口，共 96 个窗口，结果标记为 `SMOKE / partial`。

集成测试 **34 项通过**，覆盖三个任务的 STATIC 模型、M1 数据与训练接口、M2 / H1 原入口及 M1 既有主线接线。真实 M1 R100 输入的 stream / offline 最大差值为 `2.98e-7`。最终 checkpoint、源码哈希、三组 optimizer 参数及两次 EMA 更新均已核对。

数据路径直接读取 raw neural、原生 16 维 EMG 和 eval mask。4 个 source session 共 213,336 个候选窗口，沿用现有固定 sampler 和不足 B32 的尾 batch 丢弃规则，每轮实际使用 213,280 个窗口，丢弃 56 个，形成 6,665 次更新。source 端点、eval mask 与 sampler 摘要，以及 HO3 的 3,881 个完整 query 窗口和 target 摘要，均与当前 P16 FULL 记录一致。

M1 扩展只在共享模型的任务允许列表中加入 `m1`，M2 / H1 的初始化权重逐字节相同，原 smoke checkpoint 均可严格加载，加载后的预测最大差值为 0。上节历史验收收据保留原始代码哈希；新增兼容性收据记录本次模型代码哈希及前后对照。

产物：

- [M1 汇总验收收据](../results/smoke/static_m1_smoke_audit.json)。
- [M1 训练、重载与流式 smoke](../results/smoke/m1_static_s42_verified/smoke_receipt.json) 和 [HO3 评分 smoke](../results/smoke/m1_static_s42_verified/score_receipt.json)。
- [M2 / H1 扩展前后兼容性检查](../results/smoke/static_m1_extension_compatibility.json)。

M1 的正式 24 轮、159,960 次更新和完整 HO3 性能评估尚未运行。
