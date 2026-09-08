# M2 三个 seed 的 B/D 配对效应（EXT4）

本文件记录 `results/diagnostics_v1/m2_joint_paired_seed_effects_v1.json` 的完成态 summary，SHA-256 为 `abcba3df8de3601058f758818d37db79fb46d8f1270f248eafa95ca901c2efde`。

比较单元是同一 model seed 内、在同一四-session EXT4 development surface 上分别独立选取 EMA epoch 的 B activity-only 与 D joint arms。三个独立配对 seed 是 42、43、44；四个 session 是每个 seed 内的重复测量，不能作为 12 个独立样本。

| 预先定义的比较 | 三个 seed 的 D−B mean | sample SD | 独立配对数 |
| --- | ---: | ---: | ---: |
| 各 arm 独立 EMA epoch pick | 0.13672691424821373 | 0.03580496790760836 | 3 |
| 固定 epoch 24 | 0.15368779486183196 | 0.03575406559424156 | 3 |

seed43 的 session-level delta 中保留一个负值（`ses-2020-10-30-Run2`）；summary 没有删除、重加权或将该 session 解释为独立 seed。

这些数值是 EXT4 visible development evidence 的描述性 seed-paired summary。它们不提供置信区间、p 值或显著性结论；不支持 session-level pseudo-replication；不涉及 EXT6 checkpoint selection，也不涉及任何 official test 或 submission 结果。
