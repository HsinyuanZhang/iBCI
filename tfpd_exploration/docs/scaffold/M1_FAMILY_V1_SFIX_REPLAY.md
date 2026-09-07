# M1 family V1: historical S-Fix comparator

`pilot_r3/s_fix/epoch_011.pt` is a usable frozen historical checkpoint, not a
source-dev generalization baseline. It was trained for 12 epochs on full-window
behavior targets from `ses-20120926`, `ses-20120927`, and `ses-20120928`; those
are also the three sessions in the current chronological source-dev tail.

The source-only replay therefore answers a narrow question: what does that
historical B3/S-Fix system output on the exact current 31,252-row manifest?
It must be labelled **training-overlap comparator only**. It cannot establish
SPINT non-inferiority, a fair FLAT/ROUTE effect, or held-out generalization.

The exact frozen artifact is
`tfpd_exploration/results/m1_emg_rsyn3_fold_local_v1/pilot_r3/s_fix/epoch_011.pt`
with SHA-256 `7976e0b064fc4d92396b38a8e385aaa78379f0330c52ba7244bb45831b72178a`.
Its checkpoint has epoch `11`, global step `59412`, uses the B3 architecture,
and was initialized from teacher checkpoint epoch 018 recorded in its launch
receipt. Its M10 rSyn3 source carriers are exactly compatible after float32
conversion with the sealed current source-only carrier bank; that was checked
independently before this runner was added.

The replay command is source-only and performs no optimization:

```bash
taskset -c 4-7 env PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /home/xinyuan/miniconda3/envs/spint/bin/python -m tfpd_exploration.src.m1_family_v1.sfix_replay --device cpu
```

For the one-time GPU replay, after the scheduler grants GPU 1:

```bash
taskset -c 4-7 env PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 /home/xinyuan/miniconda3/envs/spint/bin/python -m tfpd_exploration.src.m1_family_v1.sfix_replay --device cuda:0
```

If a genuine same-manifest SPINT comparator is required, train a new historical
system only on the frozen chronological train rows from 26/27/28, with the
sealed source-only D0/rSyn3 carrier and fixed M10 calibration. Score its fixed
epoch/EMA choice once on the same 31,252 tail. For a CRST-B4 claim, make this a
paired `FW-CausalPE4` FLAT/ROUTE run: identical initialization of shared
weights, split, carrier, whole-unit mask RNG, budget, and selection rule; the
only structural switch is the zero-initialized routing logit bias. Do not mix
the historical B3 checkpoint into that pair.
