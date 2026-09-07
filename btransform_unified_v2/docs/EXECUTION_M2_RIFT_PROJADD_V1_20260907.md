# M2 RIFT proj_add execution record

The formal runner is `scripts/rift_v1/m2_train.py`.  It is a single M2 cell:
RIFT `recency`, context 50 bins (one second at 20 ms/bin), D4 windows
`(13,12,12,12)`, width 256, local attention, and the existing `proj_add`
P16 frontend.  It does not expose flat or concat variants.

It uses seed 42, effective batch 32, M33 banks from frozen B3S + MOVE-T4,
the frozen source-seven 24-epoch manifest
`a95255fa339ea06f1e1cc3ef9f53f3d49d15799bf95e4579f672417fd878d79a`, AdamW,
the legacy M2 x5 target contract, LR 3e-4 warmup for 3165 updates then cosine
to 3e-5, clipping at 1, unit dropout 0.10, and EMA 0.9995.  Training performs
75,960 source-only updates.  `source_minival` is scored after every epoch;
EMA ext4 scoring is resumable from epoch checkpoints and selects the earliest
maximum equal-session mean.  Neither the excluded external sessions nor an
official test surface is opened.

The raw query cache uses a 49-bin leading timeline pad.  For minival/ext4 the
runner reads each frozen `mapping.json` and derives a valid mask from padded
window coordinates.  It never decides validity from neural values, so a true
all-zero neural bin remains evidence.  Source-train windows are all post-M33
legal windows and get an explicit all-true mask.

Each epoch checkpoint is atomically published and includes raw weights,
optimizer, EMA, epoch/global step, and Python/NumPy/Torch CPU/CUDA RNG state.
`run_meta.json` freezes code, manifest, and cache-array hashes.  `heartbeat`
and per-epoch minival metrics are updated during training; ext4 progress is
atomically recorded one epoch at a time for restartable scoring.

The comparison reading is the p1a_v2b M2 concat run at ext4 epoch 19,
`0.452135`; this is a historical development reference only, not a source of
training data or a selection override.

Formal launch after CUDA preflight:

```bash
PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 \
  /home/xinyuan/miniconda3/envs/spint/bin/python \
  btransform_unified_v2/scripts/rift_v1/m2_train.py \
  --dest btransform_unified_v2/results/rift_v1/m2_r50_recency_s42_formal_v1 \
  --stage all --device cuda:0 --epochs 24
```

Resume a completed-epoch checkpoint with this replacement command:

```bash
PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 \
  /home/xinyuan/miniconda3/envs/spint/bin/python \
  btransform_unified_v2/scripts/rift_v1/m2_train.py \
  --dest btransform_unified_v2/results/rift_v1/m2_r50_recency_s42_formal_v1 \
  --stage train --device cuda:0 --epochs 24 \
  --resume btransform_unified_v2/results/rift_v1/m2_r50_recency_s42_formal_v1/epoch_00N.pt
```

Then run `--stage score` with the same destination to continue any interrupted
ext4 scan.
