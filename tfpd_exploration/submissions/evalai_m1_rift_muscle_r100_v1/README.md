# M1 muscle-response R100：正式产物与提交准备

正式训练、24-epoch 评分和本地提交准备已完成，但镜像尚未 push，EvalAI 尚未登记。训练在物理 GPU 1 上完成 24 epoch、159960 updates；public held-out calibration 的主指标选择 epoch 为 e3。隐藏 official test 没有用于选择，也没有官方隐藏测试分数。

主指标 `channel_variance_weighted_r2` 的 HO3 等权均值为 0.569075；与可复现 baseline 的 0.654471 相比下降 0.085396。legacy flattened 均值为 0.668062；相对 baseline 的 0.734260 下降 0.066198。详细的 per-session 对照、完整曲线和选择规则见[comparison.json](/home/xinyuan/Work_host/SPINT/btransform_unified_v2/results/m1_muscle_r100_v1/comparison/comparison.json)。

## 已完成产物

- [EvalAI candidate](/home/xinyuan/Work_host/SPINT/tfpd_exploration/submissions/evalai_m1_rift_muscle_r100_v1/artifacts/evalai_candidate.json) 与 [preparation receipt](/home/xinyuan/Work_host/SPINT/tfpd_exploration/submissions/evalai_m1_rift_muscle_r100_v1/artifacts/preparation_receipt.json) 已生成。
- payload SHA-256：`b40ff5c5d63f0b54e31fae0b506a0393050b26cf9be8111952f528df8e1d26e1`。
- 本地 image：`m1-rift-muscle-r100-cached:v1`；image ID：`sha256:7f8c750b4be163f886007fe184f7d26afaaa6c9a3fbc674abf9bdf3ff26c34d4`。
- host B1、B3、B4 cached 验证及 W100 full-window/streaming parity 均通过；容器 smoke 也通过。receipt 记录 B1/B4 最大误差为 0，B3 为 `4.768e-07`，W100 为 `2.384e-07`。
- 当时的只读 preflight 为 `today=3/max_per_day=6`、`active=0/max_concurrent=3`。该结果是当时快照，实际 push/登记前脚本会重新读取额度。
- 状态为 prepared、not pushed、not registered；后续 `submit.py` 的 `--execute` 仍需要上述完整 image ID 与 payload SHA-256。

该候选使用 `muscle_response16_svd4/global_rms` carrier、M1 RIFT R100/D4 与 live B3S identity。训练只使用四个 official held-in source session 的梯度；每个七标签 bank 只使用 M10 校准数据。

## 路径与执行环境

正式 run root、carrier pack 和 Python 解释器固定为：

```bash
RUN_ROOT=/home/xinyuan/Work_host/SPINT/btransform_unified_v2/results/m1_muscle_r100_v1/formal_s42_gpu1
CARRIER_PACK=/home/xinyuan/Work_host/SPINT/btransform_unified_v2/results/m1_muscle_r100_v1/carrier_official4/carrier_pack.npz
PY=/home/xinyuan/miniconda3/envs/spint/bin/python
TRAIN=/home/xinyuan/Work_host/SPINT/btransform_unified_v2/scripts/m1_muscle_r100_v1/train.py
PACK=/home/xinyuan/Work_host/SPINT/tfpd_exploration/submissions/evalai_m1_rift_muscle_r100_v1/pack_and_verify.py
```

正式训练使用单一物理 GPU 1。通过 `CUDA_VISIBLE_DEVICES=1` 仅暴露这一张卡，因此训练进程中的设备名是 logical `cuda:0`。计划中的 CPU 线程数为 4。

## 1. 正式训练

以下命令保留为复现记录。当前 `formal_s42_gpu1` 已含正式产物，不能在该目录重跑；训练脚本会拒绝非空的新 destination。任何复现必须使用新的空 run root，并相应更新后续 receipt/payload 绑定。

在空的正式 run root 上启动 24 epoch 训练：

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  "$PY" "$TRAIN" \
  --dest "$RUN_ROOT" \
  --arm D_JOINT \
  --seed 42 \
  --carrier-pack "$CARRIER_PACK" \
  --device cuda:0 \
  --cpu-threads 4 \
  --stage train
```

执行计划规定 24 个 epoch、每 epoch 6665 updates，共 159960 updates。训练完成后，`train_receipt.json`、24 个 checkpoint 和 `run_meta.json` 是后续评分与打包的输入合同。

## 2. 完整 24 epoch 的 held-out-calibration 选择

以下命令同样只保留为复现记录。当前 run root 已有完成的 score receipt，不能对既有正式产物重跑并覆盖其评分记录。训练 receipt 完成后，对全部 24 个 EMA checkpoint 执行 public held-out calibration 扫描：

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  "$PY" "$TRAIN" \
  --dest "$RUN_ROOT" \
  --arm D_JOINT \
  --seed 42 \
  --carrier-pack "$CARRIER_PACK" \
  --device cuda:0 \
  --cpu-threads 4 \
  --stage score
```

主选择指标为 `channel_variance_weighted_r2`：先在每个 HO session 的 channel-centered variance-weighted R² 上计算分数，再对三个 public held-out calibration session `20121004`、`20121017`、`20121024` 作等权算术平均。若存在并列最大值，选择最早 epoch。

旧的 flattened 指标仍会保存在每个 epoch 的 `equal_session_mean`，用于历史兼容和报告；它不参与主选择。完整 receipt 必须包含 24 行 `ema_by_epoch`、三 session 的窗口数（1305、1295、1281）、总计 3881 窗口，以及主指标对应的 earliest-max `selection.epoch`。

## 3. CPU 本地封包与验证

以下命令保留为复现记录。当前提交目录已有不可覆盖的 payload 和 candidate，不能对现有产物再次运行。评分 receipt 完成后，在 CPU 上重建 selected EMA 的七标签 M10 bank、重新导出 sealed B3S `E0`，完成主机验证，并构建本地 Docker image。默认应一次性执行以下封包命令：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONNOUSERSITE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  "$PY" "$PACK" \
  --run-root "$RUN_ROOT" \
  --carrier-pack "$CARRIER_PACK"
```

封包程序会 fail closed 地核验：当前代码/输入与训练元数据、carrier binding 和 fit SHA、completed train receipt、完整 score receipt，以及全部 24 个 checkpoint SHA。它还会重新计算每个 session 的 live-joint 与 sealed-concat W100 前向，并记录 source4/HO3 的 `E0`、`T` SHA 和误差。运行时验证覆盖 B1、B3、B4 的 cached CPU 路径以及 W100 full-window/streaming parity。

`--skip-docker` 只适用于第一次封包时明确只做主机验证、不构建 Docker 的需求。它不能作为生成 payload 后再以默认命令补建 Docker 的路径，因为 `pack_and_verify.py` 会拒绝覆盖既有 payload。

> `pack_and_verify.py` 拒绝覆盖既有 payload。若已有有效 `artifacts/m1_rift_muscle_r100.pkl`，请先保留其 receipt、payload SHA 和验证结果；不要把后续打包当作静默重跑。

## 4. 提交前准备与显式提交

封包后会写入 `artifacts/evalai_candidate.json`。该文件应包含动态选出的 epoch、主选择均值、legacy flattened 均值、payload SHA、method metadata 和本地验证信息；没有 Docker build 时 image 字段为 `null`。

[`submit.py`](submit.py) 的无 `--execute` 调用只执行提交前检查，包括本地 image/label、phase 状态和实时日/并发额度检查：

```bash
"$PY" /home/xinyuan/Work_host/SPINT/tfpd_exploration/submissions/evalai_m1_rift_muscle_r100_v1/submit.py \
  --manifest /home/xinyuan/Work_host/SPINT/tfpd_exploration/submissions/evalai_m1_rift_muscle_r100_v1/artifacts/evalai_candidate.json
```

实际 push 与登记需要显式传入 `--execute`，并提供完整且不可变的 image ID 与 payload SHA-256。提交脚本会在 push 后读取 ECR remote manifest，将 manifest digest 与 config digest 分开保存，并在登记前再次读取 phase 和 submissions 以检查实时额度。本文不提供实际提交命令中的确认值，也不主张有任何提交已经发生。

## 静态依据

- `btransform_unified_v2/results/m1_muscle_r100_v1/execution_policy.json`：锁定的方法、source/HO session、24 epoch、选择指标和 earliest-max 规则。
- `btransform_unified_v2/results/m1_muscle_r100_v1/formal_pipeline_gpu1_plan.json`：`formal_s42_gpu1` 路径、GPU 1 可见性、logical `cuda:0`、正式训练/评分命令和解释器路径。
- [`pack_and_verify.py`](pack_and_verify.py)：24-row score validation、carrier/fit/checkpoint binding、CPU sealed payload 与 B1/B3/B4/W100 验证。
- [`submit.py`](submit.py)：显式执行门槛、动态 phase quota、remote ECR manifest/config digest binding 和登记前复查。
