# Work Order — B1-TARM V1 Fold-0 Pilot

日期：2026-09-03  
Governing design：`DESIGN_B1_TEMPLATE_ANCHORED_PROFILE_MEMORY_V1_20260903.md`

授权：使用 GPU0 运行一次 source-only `fold=0, seed=42, epochs=12` 四臂配对 pilot；GPU1
不使用。允许读取 held-in-calib/minival B1 数据和写入新根
`tfpd_exploration/results/b1_tarm_v1/pilot_fold0_seed42`。禁止读取 hidden query label、禁止
EvalAI、禁止覆盖旧 B1-SFCJ receipt。

运行前要求：

1. `CUDA_VISIBLE_DEVICES=0`，进程内只可见一张 GPU；
2. 新结果根不存在；
3. focused CPU tests 通过；
4. 四臂第一步前 prediction 等于 M3 template；
5. GROWING history 对当前 trial 严格 decode-before-commit；
6. profile lag/gamma/normalizer 只由 outer-training dates 选择。

命令：

```bash
env CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
  PYTHONPATH=/home/xinyuan/Work_host/SPINT/tfpd_exploration:/home/xinyuan/Work_host/SPINT \
  /home/xinyuan/miniconda3/envs/spint/bin/python \
  tfpd_exploration/scripts/run_b1_tarm_pilot_v1.py --execute --fold 0 --seed 42 --epochs 12
```

Pilot 无科学淘汰权；完成后按 epoch 报告四臂 GROWING/FIXED3、TPL-M3、profile gamma/lag、
训练曲线和运行成本，再决定是否扩三折。

