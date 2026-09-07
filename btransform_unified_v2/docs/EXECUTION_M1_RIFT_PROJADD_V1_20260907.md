# M1 RIFT proj_add execution record

The formal runner is `scripts/rift_v1/m1_train.py`. It trains exactly one cell:
RIFT recency, 100-bin raw context, D4 windows `(25,25,25,24)`, width 256,
local attention, and the established P16 `proj_add` frontend. It uses seed 42,
batch 32, M10 frozen banks, AdamW at 1e-4 with one-epoch warmup, clipping 1,
whole-unit dropout 0.10, and EMA 0.9995.

The source train face is the native four-session full-timeline universe:
`ses-20120924/26/27/28`, 213,336 windows (54,849 / 54,476 / 49,228 / 54,783),
6,665 batches per epoch, and 159,960 formal updates. The frozen sampler SHA256
is `afce94aeaf359734faa06f115a2ee5b62011a9b1df5a99b11790d53e4b5df796`.
The source padded representation uses W100 and a 99-bin prefix. Source gradient
windows are post-M10 legal and receive an explicit all-true validity mask.

Held-out-calib M10 sessions `20121004`, `20121017`, and `20121024` are opened
only after training, during `--stage score`. Each held-out window retains its
original padded coordinate and receives `valid[start + offset >= 99]`; zero
neural values never define validity. The runner evaluates EMA weights under
`model.eval()`, performs a small repeated deterministic evaluation check, then
scans each of the 24 checkpoints once. It records restartable progress with the
SHA256 of every checkpoint and writes `score_receipt.json` only after the
complete scan. Selection is the earliest epoch at the maximum equal-session
mean. No official test surface is opened.

Training and scoring are deliberately separate commands; `--stage all` is
rejected so held-out evaluation cannot be interleaved with training. A formal
run requires exactly 24 epochs. Resume accepts only a checkpoint in the
matching destination and verifies its formal status, full configuration,
source/bank/query hashes, initialization hash, and checkpoint ownership.
A completed epoch-24 checkpoint must proceed through `--stage score`.

CPU smoke performs training only:

```bash
PYTHONNOUSERSITE=1 /home/xinyuan/miniconda3/envs/spint/bin/python \
  btransform_unified_v2/scripts/rift_v1/m1_train.py \
  --dest btransform_unified_v2/results/rift_v1/m1_r100_cpu_smoke \
  --stage train --device cpu --epochs 1 --max-updates-smoke 1
```

Formal launch after GPU arbitration:

```bash
PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 \
  /home/xinyuan/miniconda3/envs/spint/bin/python \
  btransform_unified_v2/scripts/rift_v1/m1_train.py \
  --dest btransform_unified_v2/results/rift_v1/m1_r100_recency_s42_formal_v2 \
  --stage train --device cuda:0 --epochs 24 --cpu-threads 4
```

After `train_receipt.json`, scan the checkpoints:

```bash
PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 \
  /home/xinyuan/miniconda3/envs/spint/bin/python \
  btransform_unified_v2/scripts/rift_v1/m1_train.py \
  --dest btransform_unified_v2/results/rift_v1/m1_r100_recency_s42_formal_v2 \
  --stage score --device cuda:0 --epochs 24 --cpu-threads 4
```
