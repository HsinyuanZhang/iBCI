# Work order: M1 EMG-rSyn3 fold-local Stage-0 audit V1

**Date:** 2026-09-02  
**Status:** frozen implementation contract for the additive fold-local
Stage-0 successor  
**Design authority:**
`tfpd_exploration/docs/DESIGN_M1_EMG_RSYN3_FOLD_LOCAL_STAGE0_20260902.md`

This work order authorizes CPU Stage 0 of the fold-local audit. It does
not rewrite sealed EMG-Syn3 FAIL or EMG-rSyn3 V1 PASS. It does not by
itself authorize a GPU process.

## 1. Additive ownership

```text
tfpd_exploration/src/m1_emg_rsyn3_fold_local_v1/**
tfpd_exploration/scripts/run_m1_emg_rsyn3_fold_local_stage0.py
tfpd_exploration/tests/test_m1_emg_rsyn3_fold_local_v1.py
tfpd_exploration/results/m1_emg_rsyn3_fold_local_v1/**
tfpd_exploration/docs/DESIGN_M1_EMG_RSYN3_FOLD_LOCAL_STAGE0_20260902.md
tfpd_exploration/docs/WORKORDER_M1_EMG_RSYN3_FOLD_LOCAL_STAGE0_20260902.md
```

Read-only sealed evidence:

```text
tfpd_exploration/results/m1_emg_syn3_fcm_v1/stage0/**
tfpd_exploration/results/m1_emg_rsyn3_fcm_v1/stage0/**
```

Any write, chmod, truncation, or re-reserve of those two roots is a
contract failure.

## 2. Per-fold load law

For each fold in `{0,1,2,3}`:

1. Source sessions: full EMG; neural trials `[0,10)`.
2. Target session: EMG trials `[0,10)` and neural trials `[0,10)` only.
3. M10/M6/M4/M2 target carriers are fit only on that target-support view.
4. Dictionary immutability uses the real target-support EMG, not a source
   prefix probe.
5. Receipts record exact source and target trial ranges and
   `target_query_values_read = false`.
6. Reproduce the 16-row constructibility/reliability table. Compare
   carrier, dictionary, coverage, and split-half digests with the sealed
   V1 PASS.
7. Name the PCA diagnostic `rectified trial-mean PCA3`. Do not call it
   historical signed PCA.
8. Keep `R(x)=max(x,0)` exactly: float64 `np.maximum(x_raw, 0.0)` before
   RMS and NNMF. No rectifier sweep.

## 3. LS4

Disclose that unequal-length LS4 currently cyclic-repeats or truncates.
`enabled_in_stage1_pilot = false`. Repair is mandatory before Stage-2
`S-LS4`. The first GPU pilot remains Z-Fix / S-Fix / S-Acyc only.

## 4. GPU

Public CLI cannot mint a GPU capability. After an accepted fold-local
Stage-0 terminal, GPU 0 and GPU 1 are both eligible when idle. Occupied
cards are polled; no occupant may be signaled or reprioritized.
