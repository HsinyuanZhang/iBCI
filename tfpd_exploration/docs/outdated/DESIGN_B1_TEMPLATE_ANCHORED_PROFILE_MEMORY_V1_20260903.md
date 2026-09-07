# B1 Template-Anchored Residual Memory V1

日期：2026-09-03  
工作名：**B1-TARM**（Template-Anchored Residual Memory）  
数据：FALCON B1 / DANDI 001046  
状态：source-only pilot 可执行；不授权读取 hidden query label 或提交 EvalAI

## 1. 为什么重开 B1

旧 B1-SFCJ Stage 0B 只证明原六臂计划的冻结门失败，没有训练任何神经模型。失败由
SFC4/fold-1 的一个近零反转触发：correct R² `0.0223789`，derangement
`0.0224115`。SFC9 的 correct R² 则在三个 fold 全部高于 derangement，差值分别为
`+0.005337/+0.002256/+0.025684`。因此关闭 SFC4 与“通道重识别”故事是合理的，关闭
全部 B1 activity/profile 学习则证据不足。

CPU ceiling 同时表明 B1 的主要 target-specific 信息是三条 calibration 声谱模板；神经
信号在最佳模板之上的线性净增益较小。V1 因此不再从零预测整张声谱，而把 target M3
逐坐标 median 当作显式输出锚，只训练 neural-conditioned residual。

## 2. 假设

1. `activity signature` 的作用不是重新识别通道，而是从稳定索引的 85 个通道估计
   session neural state。
2. `tuning profile` 使用 source-frozen PCA8 声学场中的 SFC9，但 target M3 系数向
   outer-training dates 的同索引 channel prior 收缩；不做 channel permutation。
3. M2 的可迁移部分是 native/post identity anchored fusion、matched causal memory、
   decode-before-commit 和 paired dropout，而不是 M2 的 chunk100e 能量门。
4. 模板锚定让模型只学习较小的、真正需要 neural input 的残差；成功必须超过
   neural-free `TPL-M3-median`。

模型输出为：

```text
pred_stdlog = target_M3_median_stdlog + residual_decoder(current_activity,
                                                         completed_activity_pool,
                                                         tuning_profile)
```

residual head 全零初始化，所以四臂第一步前都精确等于 TPL-M3。

## 3. Source-prior tuning profile

每个 fold 只用两个 training dates 拟合 acoustic standardizer/PCA8 和 lag。每个 date 用
first M3 拟合 `[intercept, w1..w8]`。稳定索引 source prior 是两个 training-date profile
的逐 channel 均值。收缩为：

```text
profile_date(gamma) = (1-gamma) * M3_fit_date + gamma * source_prior
gamma in {0,.25,.5,.75,1}
```

`gamma` 只根据 training dates 的 calib 4..N encoding R² 选择；validation date 不参与。
最终 9 维 profile 的 mean/std 也只由 training dates 冻结。target/validation 只用公开 M3
标签拟合一次，query 标签永不进入 profile。

## 4. 最小 2×2

| Arm | Activity pooling | Profile |
|---|---|---|
| `TA-N0` | native `post(mean(pre))` | Zero9 |
| `TA-NS9` | native | source-prior SFC9 |
| `TA-J0` | native/post anchored J-R1 | Zero9 |
| `TA-JS9` | native/post anchored J-R1 | source-prior SFC9 |

四臂共享模型宽度、模板、样本顺序、whole-unit dropout mask、loss、optimizer 和训练步数。
这允许分别计算 profile gain、J-R1 gain 和 interaction，不用恢复 SFC4。

## 5. Causal memory

每个 date 的 first M3 是 seed。训练 sample 的 history 只含当前 trial 之前已完成的
activity；validation 使用 `calib 4..N -> minival 1..2`，先 decode 再 commit。query
spectrogram 只用于 source loss/score，不进入 memory 或 profile。

训练使用 whole-stack 重算以保持梯度；冻结部署可维护 `sum(pre_pool)`、
`sum(post_pool)` 与 K。已修复旧实现中 `if mode == whole_stack or True` 导致
`running_sum` 仍保存全部 raw trial 的问题。

## 6. Pilot

第一格固定：

- fold 0：train `20210627/20210628`，validation `20210626`；
- seed 42；
- 四臂共同训练 12 data epochs；
- batch 4，Adam，LR `5e-5`，weight decay 0；
- whole-unit dropout `p=0.10`，四臂共用 mask；
- model width 256、16 heads、1 cross-attention layer；
- loss = official raw min-max MSE surrogate + `1e-4 × SmoothL1(stdlog residual)`；
- 每 epoch 同 checkpoint 同时报 GROWING 与 FIXED3；
- 保存 final checkpoints，不按 pilot 单格数字删除任何臂。

12 epoch 是快速可行性点，不宣称收敛。若末段训练 loss 仍明显下降或所有神经臂仍等于
初始化模板，后续扩展使用统一的固定 step horizon；不得让各臂单独选 checkpoint。

## 7. 判读

Pilot 只回答代码/优化是否可用，以及效应方向：

- `TPL-M3 - TA-N0`：activity residual 是否提供信息；
- `TA-N0 - TA-NS9`：native topology 中 profile content；
- `TA-J0 - TA-JS9`：J-R1 topology 中 profile content；
- `TA-N0 - TA-J0` 与 `TA-NS9 - TA-JS9`：M2 式 fusion 的贡献；
- 每臂 `FIXED3 - GROWING`：已完成 query activity 是否改善 session state。

所有差值用 MSE(reference) − MSE(candidate)，正值为改善。只有三折 source-grouped OOF
和第二 paired seed 完成后，才可选择 final arm 或形成一般化声明。

## 8. 停止与扩展

Pilot 不作方法淘汰。若任一 neural arm 优于 TPL-M3，优先扩三折；若 profile contrast 为正，
保留 source-prior SFC9；若只有 activity arm 为正，转为 activity-only TARM。若模板锚定四臂
全部明显差于 TPL-M3，先检查 residual horizon/loss，不回头扫 SFC rank、energy gate 或
target-label fine-tuning。

