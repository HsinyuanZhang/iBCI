# M1 Carrier Refinement V1：冻结 B 的探索性候选

本文记录一个独立于 M1 Z/B/D 主实验的探索性候选流程。它不改写既有 `CrossSessionM1Decoder`、原训练程序或原 33 项 Z/B/D 主图；主图仍保留旧 D 结果。本流程不用于论文主结论、提交或隐藏测试声明。

## 方法边界

基准是已完成的 `B_ACTIVITY_ONLY` 的 selected-EMA checkpoint。加载后，B 的全部参数冻结并保持 `eval()`；残差训练仅更新新 head。B 的原始 `forward(x, bank, input_valid_mask=...)` 先产生 16 维预测，候选输出为该预测加上 carrier 条件残差。head 采用 `Linear(16 + F, 64) → GELU → Linear(64, 16)`，最后一层 weight 和 bias 都严格零初始化。因此，初始化时的可见 carrier 路径与 B 相同；显式全 false 的 `carrier_visible` 会返回原 B prediction 的硬回退。

carrier 由官方 Falcon `bin_units` 的 **end-at-t** timestamp 语义对齐。源侧 NNMF basis 只拟合三个 source session 的 EMG `[0, 310)`；每个 source carrier 的 unit ridge 只使用其 M10 neural `[0, 10)`。目标侧只读取目标 M10，并以冻结的 source basis 和 source normalizer 投影 carrier；不会重新拟合 basis 或 normalizer。

每个预测窗口使用 24 维外部 carrier feature。它在 FalconDataset 的 padded neural stream 上，以输出 endpoint `window_start_padded + 99` 取值。对宽度 `(1, 5, 10, 25, 50, 100)`，每一宽度计算当前及过去 counts 的因果均值，再与保持 unit 行对应关系的 `[64, 4]` normalized carrier 相乘并除以 64，得到四维投影。source train/validation 的独立范围仅在各自起点左零 padding；不在 trial 边界重置。

这是一项网络加 carrier 的联合改动：head 可以以 B prediction 为条件。即使某一折观察到正增量，也不能将其单独归因于 carrier。

## Source-only 训练与选择

训练只使用三个 source session：

- train cache 以每 session 原窗口顺序的 stride 8 取样；每个 session 用相同更新次数，较短 session 仅为平衡更新数而有确定性的有放回索引。
- carrier feature mean/scale 只从该 source train cache 拟合，按 session 等权；不会从 target 或 forward batch 隐式拟合。
- head 在 CPU 上训练 20 epochs，batch size 为 512，learning rate 为 `0.001`，weight decay 为 `0.001`。训练时每个 row 的完整 carrier 通道以概率 `0.5` 不可见；false row 不通过 residual head 获得梯度。
- 每个 epoch 在完整 source validation 上评估 `alpha ∈ {0, 0.25, 0.5, 1.0}`。候选必须在每个 source session 上都满足相对于 B 的 gain `>= -1e-6`，然后在合格候选中选择最高的 equal-session R²；并列按较早 epoch、再按 alpha 顺序处理。`alpha=0` 是精确 B 回退。
- 保存并恢复与所选 source-validation 状态一致的 `residual_head`，再复评分；target 只会在 `sealed_selection.json` 写入之后打开。

这里的 source 逐 session 条件仅是 source-validation 的选择约束。它不构成 target 非劣保证。

## 缓存、绑定与复核

source cache 保存 B prediction、source target、24D feature、padded window start、valid mask 和原始窗口索引。cache metadata 绑定协议、B receipt/selected EMA/resume checkpoint、runtime source SHA、base state hash、source evidence、feature-grid SHA、stride 和运行时信息；NPZ 本身也由 SHA 绑定。

preflight 在 source-only 状态下执行：检查 raw/cached identity 和 B forward 的字节一致性、重复 session stack、正 scale 可安装与零 scale 被拒、zero-init/false visibility 的精确 B 回退、训练后可见 carrier 特征会改变输出、冻结 B 无梯度且 hash 不变。target M10、query 坐标、target labels 和 strict B target NPZ 都不在此阶段读取。

`fit_score` 先校验 preflight binding，再冻结并保存 source carrier、cache、selection 和 residual checkpoint；随后才校验 strict B receipt 绑定的 target NPZ filename/SHA 与坐标，投影目标 M10 carrier，并以保存的原始 strict B target prediction 为基准。target optimizer steps 为零，target labels 不参与 fit 或 selection，只用于最终指标和坐标门控。

独立 audit 会复核协议和各 SHA、source cache 的坐标/valid mask/feature/target、source B 的代表性 replay、等 session moments、完整 source 选择曲线及恢复后的所选 head；也会复核目标物理 query、target M10 carrier、因果 feature、candidate prediction 与零可见性 B 回退。它不把 target 指标用于重新选择状态。

## 执行入口

以下命令以 `ses-20120924` 为例；其他 fold 替换 `--target` 和对应目录。先完成且严格评分原 B run，再执行本候选。命令本身不改变原 B 目录。

本次已验证的本机环境如下。先在同一 shell 设置这些变量，随后三条命令都使用 `/home/xinyuan/miniconda3/envs/spint/bin/python`，以避免默认 `python3` 缺少依赖或混入 user-site；CPU 线程数与 runner 的 `--threads 4` 保持一致。

```bash
export PYTHONNOUSERSITE=1
export PYTHONWARNINGS=ignore
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
```

```bash
/home/xinyuan/miniconda3/envs/spint/bin/python btransform_unified_v2/scripts/m1_carrier_refinement_v1/run_pilot.py \
  --target ses-20120924 \
  --dest btransform_unified_v2/results/m1_carrier_refinement_v1/loso_20120924 \
  --b-dest btransform_unified_v2/results/cross_session_v1/m1_loso_20120924/B_ACTIVITY_ONLY/s42 \
  --stage preflight --threads 4 --train-stride 8
```

```bash
/home/xinyuan/miniconda3/envs/spint/bin/python btransform_unified_v2/scripts/m1_carrier_refinement_v1/run_pilot.py \
  --target ses-20120924 \
  --dest btransform_unified_v2/results/m1_carrier_refinement_v1/loso_20120924 \
  --b-dest btransform_unified_v2/results/cross_session_v1/m1_loso_20120924/B_ACTIVITY_ONLY/s42 \
  --stage fit_score --threads 4 --train-stride 8
```

```bash
/home/xinyuan/miniconda3/envs/spint/bin/python btransform_unified_v2/scripts/m1_carrier_refinement_v1/audit_pilot_result.py \
  --target ses-20120924 \
  --dest btransform_unified_v2/results/m1_carrier_refinement_v1/loso_20120924 \
  --b-dest btransform_unified_v2/results/cross_session_v1/m1_loso_20120924/B_ACTIVITY_ONLY/s42
```

## 当前已完成的探索性三折

以下仅转录各折 `fit_score_receipt.json` 与 `result_replay_audit.json` 已记录的最终 target delta R²；不作平均或跨折汇总。

| LOSO target fold | audit 状态 | target candidate − B 的 R² delta |
| --- | --- | ---: |
| `ses-20120924` | `PASSED` | `-0.00026882130859529063` |
| `ses-20120926` | `PASSED` | `+0.0005367000063748462` |
| `ses-20120927` | `PASSED` | `-0.0006682594007311193` |

这三折均为单 seed 的 exploratory public LOSO 结果。方向并不一致，且候选同时改变网络残差与 carrier 条件路径；因此不能声称 carrier 的独立效果、所有 fold 都不下降，或已建立统计非劣性。`ses-20120928` 仍待其原 B 训练完成后才可运行本候选流程。原 33 项 Z/B/D 主图继续保留旧 D。

## 关联文件

- Runner：`btransform_unified_v2/scripts/m1_carrier_refinement_v1/run_pilot.py`
- 对齐与投影：`btransform_unified_v2/scripts/m1_carrier_refinement_v1/aligned_carrier.py`
- 独立 replay audit：`btransform_unified_v2/scripts/m1_carrier_refinement_v1/audit_pilot_result.py`
- 候选 head：`btransform_unified_v2/src/btransform_unified_v2/m1_carrier_residual_candidate.py`
- 已完成 receipt/audit：`btransform_unified_v2/results/m1_carrier_refinement_v1/loso_20120924/`、`btransform_unified_v2/results/m1_carrier_refinement_v1/loso_20120926/` 与 `btransform_unified_v2/results/m1_carrier_refinement_v1/loso_20120927/`
