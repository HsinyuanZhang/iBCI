# M2 joint B44 formal ext4 validation（2026-09-08）

`m2_r50_joint_b_s44_formal_v1` 已终端闭合。guard state 为 `COMPLETE`；本审计只读检查 receipt、24 个 checkpoint、记录的 source/cache 文件与 ext4 score rows，没有训练、评分、预检、benchmark、绘图或 paired 推断。

审计对象为 seed44 的 `B_ACTIVITY_ONLY` arm，sampler seed42。正式训练为 24 epochs、每 epoch 3,165 updates、共 75,960 updates。source gradients 仅在 source-train surface 上使用；ext4 是已声明的 visible development selection surface，`official_test_used=false`。

## 验证结果

[validation_audit.json](../results/rift_v1/m2_r50_joint_b_s44_formal_v1/validation_audit.json) 为完整机器可读结果，SHA-256 `3c7350709707dfa155db179a90b802c9bc7c34ddf11377feb6041c8870570f12`。

审计通过 106 个检查：

- formal `run_meta`、completed train receipt、completed score receipt 与 guard terminal state 一致；
- 记录的全部 source 和三种 surface cache 文件与 receipt hash 一致；
- 24 个 checkpoint 文件 SHA 均与各自 EMA row 一致；每个 checkpoint 在 CPU `torch.load` 后均满足 M2 joint v2 schema、cell、B arm、seed44、epoch、`global_step=epoch×3165`、source/cache binding；model 与 EMA shadow 非空且 finite；
- 24 个 ext4 rows 都为 full scan：四个 session、2,069 windows、有限 per-session R² 与 equal-session mean；每行 mean 都由四个 session R² 重新核对；
- selected EMA 是全 24 行的 earliest maximum，且 endpoint24 与 receipt `endpoint24` 完全一致。

## 可报告的 B44 development 数值

selected EMA epoch 为 e16，equal-session mean 为 `0.2508328368160401`：

| ext4 session | windows | R² |
|---|---:|---:|
| ses-2020-10-30-Run1 | 519 | 0.3614619267668836 |
| ses-2020-10-30-Run2 | 490 | 0.2838217573542904 |
| ses-2020-11-18-Run1 | 425 | 0.2414776563510439 |
| ses-2020-11-19-Run1 | 635 | 0.1165700067919424 |

固定 endpoint e24 的 equal-session mean 为 `0.1791421755945572`；per-session R² 为 `0.3783076521729588`、`0.3386674493446291`、`0.06610747634843717`、`-0.0665138754877963`，顺序同上。

这只闭合 B44 单 arm 的 seed44 ext4 development evidence。D44 的正式训练已启动，但 train/score receipts 尚未完成，因此本文件不作 B44/D44 paired effect、跨 seed uncertainty、CI 或显著性推断。
