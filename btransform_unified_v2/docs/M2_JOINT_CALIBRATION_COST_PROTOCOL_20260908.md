# M2 trained-joint D42/e17 calibration-cost protocol — 2026-09-08

此审计只测实际 `D_JOINT` seed42、EMA epoch 17 的 M33 calibration 到静态 bank。它不用旧 support-stability component timer，也不用未训练 baseline encoder。T 从 raw ext4 M33 support 通过生产 MOVE-T4/source normalizer 重建；E0 由 `benchmark_cpu_m2` 绑定的真实 trained joint encoder materialize；随后构建静态 `TaskBank`。不运行 query score 或 target BP。

每个 reportable wall 从 raw M33、normalizer、trained encoder 等 source objects 已在内存时开始，到 T、E0、静态 bank 构成后立即结束。`TaskBank` 构造内部的 metadata hashing 在 boundary 内；timer 后的 parity verification、provenance file hashing、receipt JSON 和报告 serialization 不在 boundary 内。预检使用一个 ext4 session、一轮、CPU 12,13、nice 10、OMP/MKL/OPENBLAS 两线程；其 raw T 必须 byte-equal production cache。正式运行只能由协调者执行三轮：

```bash
nice -n 10 taskset -c 12,13 /home/xinyuan/miniconda3/envs/spint/bin/python scripts/diagnostics_v1/m2_joint_calibration_cost.py --rounds 3 --output results/diagnostics_v1/m2_joint_calibration_cost_formal_20260908.json
```

正式时可逐一指定每个 `--session` 运行，或由协调者编排全部 ext4 sessions；此脚本绝不写 training/core/queue 或既有 receipts。
