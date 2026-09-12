# ACT-only v2：M2 实验交接

本文件提供给负责 GPU 调度的 agent。代码已接入 M1、M2、H1；当前实验范围仅 M2，先检查 M2 结果再决定后续任务。实现方只执行 CPU 检查，没有启动 GPU 训练、GPU smoke 或 EvalAI 提交。

## 已确定的实现

沿用现有 B3/B3S/C2 的 activity 主干：一层 `pre_pool`，跨 trial 均值，三层 `post_pool`。M1/M2/H1 的隐藏宽度分别为 64/64/32，支持集形状分别为 `[10,1024,64]`、`[33,100,96]`、`[3,1024,176]`。纯 ACT 路径删除 descriptor 拼接与依赖 descriptor 的 FiLM。

Encoder 和现有 P16/D4 learned-recency decoder 从头联合训练，保留 decoder 的初始化与 zero carrier；target calibration 只接受 activity，冻结权重后以 `eval/no_grad` 生成 identity。每次更换 checkpoint 都重新校准，同一 checkpoint/session 的 query 复用同一个 identity。

模型与权重 schema 是 `existing_early_pool_activity_trunk_v2` / `activity_joint_early_pool_epoch_checkpoint_v2`。训练和评分必须使用此版本。旧 frozen ACT、FULL 及早期深层 `phi` 的 checkpoint 不能直接作为本路线的权重。

完整机制和数据说明见 [ACTIVITY_ONLY_V2.md](ACTIVITY_ONLY_V2.md)。

## CPU 验收

模型、数据及运行器联合测试：**22 passed**。命令：

```bash
env -u PYTHONPATH PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES='' \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  /home/xinyuan/miniconda3/envs/spint/bin/python -m pytest \
  btransform_unified_v2/learnable_recency_v1/tests/test_activity_model.py \
  btransform_unified_v2/learnable_recency_v1/tests/test_activity_data.py \
  btransform_unified_v2/learnable_recency_v1/tests/test_activity_runners.py -q
```

检查覆盖 encoder/decoder 梯度、与原 `EarlyPoolEncoder` 的同权重数值一致性、trial 重排、同步通道重排、mask、零 carrier、真实数据形状及 source support/query 边界。M1 的 4 个 source 和 3 个 HO support 张量与独立 Falcon 路径逐元素完全一致。

逐任务 CLI smoke 和评分回执保存在以下目录；`score_receipt.json` 的 smoke 分数只验证执行，不作为效果结果：

| Task | CPU train smoke | CPU score smoke |
|---|---|---|
| M1 | `results/smoke/activity_v2_early_pool_m1_cpu/smoke_receipt.json` | `results/smoke/activity_v2_early_pool_m1_cpu_score/score_receipt.json` |
| M2 | `results/smoke/activity_v2_early_pool_m2_cpu/smoke_receipt.json` | `results/smoke/activity_v2_early_pool_m2_cpu_score/score_receipt.json` |
| H1 | `results/smoke/activity_v2_early_pool_h1_cpu/smoke_receipt.json` | `results/smoke/activity_v2_early_pool_h1_cpu_score/score_receipt.json` |

M2 已通过两个完整 B32 source optimizer steps，`pre_pool`/`post_pool` 梯度有限且非零，encoder 参数已更新，raw 和 EMA checkpoint 重载预测均一致。公开 EXT6 smoke 覆盖六个 session，每个 session 一个 batch。三任务的实际默认 context 前向、反向及 checkpoint 数值检查另见 `results/smoke/activity_v2_model_cpu/receipt.json`。

三任务 CLI smoke 均已完成：M1/H1 各一个 optimizer step，M2 两个；三者的 raw/EMA 重载预测全部一致。评分 smoke 分别覆盖 M1 的 96、M2 的 192、H1 的 448 个窗口。H1 实际经过 14 份记录到 S6–S12 七组的汇总。完整验收状态、代码指纹和六份 smoke 回执索引见 [验收记录](../results/activity_v2_handoff/receipt.json)。这些都是 CPU 执行结果，正式训练和评分尚未执行。

## M2 正式命令，由调度 agent 执行

工作目录为 `/home/xinyuan/Work_host/SPINT`。由调度 agent 设置 `ACTIVITY_GPU`；以下命令没有在本次实现工作中执行。正式输出目录必须为空。

```bash
cd /home/xinyuan/Work_host/SPINT
: "${ACTIVITY_GPU:?Set ACTIVITY_GPU to the GPU assigned by the scheduling agent}"
env -u PYTHONPATH PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES="$ACTIVITY_GPU" \
  /home/xinyuan/miniconda3/envs/spint/bin/python \
  btransform_unified_v2/learnable_recency_v1/scripts/activity_train.py \
  --task m2 \
  --dest btransform_unified_v2/learnable_recency_v1/results/m2_activity_only_v2_early_pool_s42 \
  --device cuda:0 --cpu-threads 4 \
  --tier learned_slope --ladder default --layers 4 \
  --seed 42 --proj-dim 16 --identity-hidden 64 --support-bins 100 --epochs 24
```

训练沿用 M2 的既有固定 manifest：7 个 source sessions、101,171 个窗口、B32、每轮 3165 updates，共 24 轮/75,960 updates。LR `3e-4`、一轮 warmup、cosine floor `3e-5`、AdamW WD `0.01`、clip `1.0`、EMA `0.9995`、unit dropout `0.1`、行为缩放 `5`。元数据和每轮 checkpoint 包含代码/输入 hash、support 来源以及梯度记录。

完成训练后，按 FULL 的本地口径扫描 24 个 EMA checkpoints：

```bash
env -u PYTHONPATH PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES="$ACTIVITY_GPU" \
  /home/xinyuan/miniconda3/envs/spint/bin/python \
  btransform_unified_v2/learnable_recency_v1/scripts/activity_score.py \
  --run-dir btransform_unified_v2/learnable_recency_v1/results/m2_activity_only_v2_early_pool_s42 \
  --dest btransform_unified_v2/learnable_recency_v1/results/selection_m2_activity_only_v2_early_pool_s42_ext6 \
  --device cuda:0 --cpu-threads 4 --save-predictions
```

默认使用 FULL 的 EXT6 全量 15,403 个窗口、相同 padding/mask 和 variance-weighted R²。以六个 session 的等权平均选择最高分，分数相同取最早 epoch。`ema_by_epoch` 保留完整曲线和各 epoch checkpoint SHA；`selection.epoch` 给出选中的 epoch。保存预测时会对所选 checkpoint 进行一次额外的校准和评分，回执中有明确记录。

## 结果用途与后续交付

本地分数用于开发和选轮，**最终数字来自 EvalAI 提交结果**。M1 的本地入口同 FULL 扫描 24 轮 HO3；H1 同 FULL 扫描 32 轮，并将 14 份公开记录按 grouped-seven 汇总。当前不启动这两个任务的正式实验。

M2 完成后交回：训练完成回执、本地 EMA 曲线、所选 checkpoint 路径及 SHA、逐 session 指标，以及后续 EvalAI 提交返回的结果。本实现阶段没有创建或启动 EvalAI 提交；届时提交包应包含所选 EMA 的 encoder 和 decoder 全部权重，并使用本版纯 activity 校准接口，不能沿用只识别旧 frozen encoder 的加载方式。

判断校准是否有用时，除本地 FULL 同口径指标，还应检查训练后替换 calibration activity 对预测的影响和各 session 的表现。STATIC 使用固定通道表；本版 ACT 使用共享函数从当前 session 的 activity 生成 identity，二者仍有明确机制区别。
