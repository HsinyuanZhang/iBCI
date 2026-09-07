# Matched M2 RIFT concat execution plan

`m2_concat_train.py` is a new M2 cell, separate from the frozen proj_add
baseline. It keeps the exact source-seven M33 cache, frozen 24-epoch manifest,
R50/D4 `(13,12,12,12)` local recency temporal stack, seed 42, batch 32,
AdamW 3e-4, 3,165 updates per epoch, 24 epochs, EMA .9995, unit dropout .10,
raw-target `*5` training bridge, and post-training ext4 scan.

The one architectural change is the static identity interface. `RiftConcatDecoder`
uses the existing v1 concat frontend: each unit token is
`[local16 | E0_50 | MOVE-T4_4]`, so the token MLP input is **70**. It uses the
full frozen M33 B3S `E0 [96,50]`; it does not project E0 to P16. The concat model starts from a fresh paired baseline at seed 42: all 76
shared equal-shape parameters are copied byte-for-byte, and the first token
MLP layer is folded exactly from proj_add as
`[W_local | W_local @ e0_proj.weight | W_carrier]`. Its parameter count is
therefore baseline `3,531,010 + 12,000 = 3,543,010`; the active owner aliases
the same final norm and readout, so no unused trainable decoder path is
registered.

The runner carries the baseline's manifest, cache, source-only-gradient,
padded-timeline, checkpoint, resume, and ext4 score-progress guards under a
new cell/schema and binds `concat_model.py` in source hashes. It neither reads
additional external dates nor official test data.

CPU smoke:

```bash
PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 /home/xinyuan/miniconda3/envs/spint/bin/python \
  btransform_unified_v2/scripts/rift_v1/m2_concat_train.py \
  --dest btransform_unified_v2/results/rift_v1/m2_concat_cpu_smoke_v7 \
  --stage train --device cpu --epochs 1 --max-updates-smoke 1 --cpu-threads 4
```

No GPU launch is part of this document. Queue only after the CPU smoke,
full/stream parity, native-scale evaluation, and score-progress behavior
checks are all recorded.
