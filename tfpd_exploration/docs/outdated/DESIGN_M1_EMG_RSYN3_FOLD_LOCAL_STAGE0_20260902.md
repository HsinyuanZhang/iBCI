# M1 EMG-rSyn3 fold-local Stage-0 audit successor

**Date:** 2026-09-02  
**Status:** additive successor; does not amend sealed EMG-Syn3 FAIL or
EMG-rSyn3 V1 PASS  
**Parent rSyn3 design:**
`tfpd_exploration/docs/DESIGN_M1_EMG_RSYN3_SUCCESSOR_20260902.md`  
**Parent rSyn3 design SHA-256:**
`b5d12b51e74a70d22f50d84ab7078fd0592ba19aae637ed0c193f57497f97cd1`

This document repairs the Stage-0 *scope audit* of EMG-rSyn3. It does not
change the rectifier, rank, ridge, injection law, or the predeclared
`S-Fix − Z-Fix` gate. It does not rewrite either sealed result root.

## 1. What was wrong in V1 Stage 0

The V1 CPU Stage 0 loaded every allow-listed session once with
`emg_trial_stop=None` and `neural_trial_stop=10`. Neural support was
correctly truncated. EMG was not: target-session query trials were read
into the stored signal view even though M10/M6/M4/M2 *fits* later masked
to `trial_id < budget`. The dictionary-immutability probe used eight bins
from a source session, not target-support EMG. The PCA disclosure was
mislabeled as historical signed PCA.

Those defects do not, by themselves, prove that V1 carrier bytes are
wrong. This successor rebuilds the 16-row table under fold-local loads
and compares carrier, dictionary, coverage, and split-half digests with
the sealed V1 PASS.

## 2. Fold-local load law

For each outer fold independently:

1. Read **full EMG** only from the three source sessions.
2. Read **only trials `[0,10)`** from the target session for both EMG and
   neural activity.
3. Build M10/M6/M4/M2 target carriers only from that target-support view.
4. Use the real target-support EMG for the dictionary immutability test.
5. Record exact source and target trial ranges and prove target query
   values were not read.

Source neural encoding remains chronological M10. Source EMG used for the
dictionary is the full source-session movement-window series. Target query
EMG and target query spikes are not opened.

## 3. Unchanged scientific operators

Rectifier, identical to V1:

```text
x_rect = np.maximum(x_raw, 0.0)   # float64, threshold exactly 0.0
```

before RMS scaling and before NNMF. No abs/offset/envelope/smoother, no
per-session threshold, no sweep.

NNMF, ridge, Zero4, budgets, and the independent injection law are
unchanged. The PCA *diagnostic* on this table is named

> rectified trial-mean PCA3

It is not “historical signed PCA”. It remains disclosure, not a decoder
R².

## 4. LS4 disclosure

The current unequal-length LS4 helper uses cyclic repeat/truncation
(`take = arange(dest_n) % source_n`), not true resampling of EMG onto
the destination trial’s time base. LS4 stays **disabled** for the first
Z-Fix / S-Fix / S-Acyc Stage-1 pilot. It must be repaired before Stage-2
mechanism controls (`S-LS4`).

## 5. GPU

This document does not issue a GPU capability. After an accepted
fold-local Stage-0 terminal, Stage-1 may use GPU 0 or GPU 1 when the
chosen card is idle. GPU launch still requires a separately issued
capability and must not overwrite sealed V1 roots.
