# M1 joint D42 EMA e3 验证与 CPU runtime

M1 joint D42 的 formal 24-epoch validation audit 已通过：`results/rift_v1/m1_r100_joint_d_s42_formal_v1/validation_audit.json`，SHA-256 `7fa2c2092b34670cd368ee1795eb555d5ed7671334ce4b549abeddd5dd41a4f7`。该 run 的 selected EMA e3 equal-session mean 是 `0.7342600300996954`；与 frozen concat seed42 EMA e3 的差为 `+0.00006776958380461107`。这是一个单 seed、development-surface 的很小差值，不构成收益、profile effect 或 mechanism effect 结论。

selected checkpoint 是 `epoch_003.pt`，SHA-256 `d7eba8ba9cea88b1290a68ed86fa3a75d51a11256747c6232fcb1c17b341efba`。CPU benchmark receipt 是 `results/rift_v1/m1_joint_d42_e3_cpu_benchmark_v1/benchmark.json`，SHA-256 `8670f2abe91a7e6e4e30d2d3ba2763a4ff423dbec68f8561755634aa640ab8e8`；其 validation audit SHA-256 为 `9a2399ae8d24b263c9a88f0bb240ec52e285e502552ad88c4623fbd8457b6e4e`。

## Runtime 结果

在 AMD Ryzen 9 7950X、2 CPU threads 的该 benchmark protocol 下，selected EMA checkpoint（包括 live B3S encoder）先 materialize HO3 E0，再输入 static decoder；D arm 使用 direct carrier。HO3 runtime streams 与早期 frozen-concat benchmark 的 source4 inputs 不同，二者不能用于速度比较或 speedup claim。

| Case | aggregate median | aggregate P95 | 三轮 median | 三轮 P95 |
| --- | ---: | ---: | --- | --- |
| B1 | 1.98861350145 ms | 2.04134084888 ms | 1.9834 / 1.9905 / 1.9896 ms | 5.2426 / 2.0349 / 2.0406 ms |
| B8 | 15.77839250058 ms | 16.27837364904 ms | 7.1797 / 15.7993 / 15.7729 ms | 48.2959 / 16.0347 / 15.8867 ms |

B8 首轮的 median `7.1797 ms` 和 P95 `48.2959 ms` 是实际观测值，保留在记录中；本文件不重测、不覆盖，也不从 aggregate 数字推断速度提升。RSS/HWM 是 whole-process 值，不是模型增量 memory。

## 验证边界

9 条 live/static 比较记录对应 6 个独立 session/window 组合，均为 exact zero。92 个 runtime source SHA 分别与其对应记录和现有文件匹配；formal source SHA 也分别与其对应记录和现有文件匹配。B1/B8 各 402 个 cached/state/reference advances 均 finite 且 exact zero error。first/full-valid windows 重合，因此不会把重合窗口重复计为独立 endpoint。full-window oracle 在 end bins 99、200、400 的最大绝对误差为 `2.3841858e-7`。

该 benchmark 只验证 D model。B flag 为 vacuous，不能描述为实际 B-side benchmark。旧脚本曾因 backend 问题失败；该失败记录不构成当前 receipt 的失败，也没有被重测结果覆盖。结果仅限本地 development evidence，不涉及 official test、submission 或最终候选排序。
