# M2 native T4 versus SPINT post-33 confirmation — draft protocol

**Protocol ID:** `M2_NATIVE_T4_SPINT_POST33_CONFIRM_V1`  
**Date:** 2026-08-04 (Asia/Hong_Kong)  
**Status:** score-free plumbing draft; **GPU launch is not authorized**  
**Endpoint:** native M2, outer-session-left-out, chronological trial-33 future query  
**Claim class:** prospective internal confirmation of supervised backprop-free calibration

## 1. Scope and immutability

This protocol prepares a new local native-MUA endpoint without reading any R² from that
endpoint. It does not open formal SUA sessions, FALCON held-out-calib/test files, or EvalAI. It
does not authorize a GPU run. A later root authorization must bind a separately finalized
prelaunch receipt before the first cell starts.

The protocol supersedes only the **effectiveness use** of
`m2_heldin_postsupport_endpoint_v1/protocol_receipt.json`. The old receipt remains unchanged as
historical evidence. Its target-session selection window and `no_effective_verdict=true` branch
must not be reused.

The active C1 v3r2 receipt contains a 34-file `source_map`. Those files are immutable while C1 is
running. This program uses new, versioned modules and configs; neither historical
`falcon_datamodule.py` is edited.

## 2. Exact data contract

Only the following public data-module tuple is legal:

```text
task                         = m2
validation_protocol          = loso
calibration_n_trials          = 33
heldin_query_start_trial      = 33
heldin_query_end_trial        = null
random_calibration            = false
include_heldout_in_fit        = false
include_heldout_in_test       = false
query_start_trial             = 0
window_size                   = 50
```

Every other tuple fails closed. The seven frozen folds are the seven held-in M2 calibration
sessions in chronological filename order. In fold `j`:

- six sessions supply source training;
- the same six sessions supply the source-only `val_heldin/r2_mean` checkpoint metric;
- all feature normalizers are fit from those six source sessions only;
- exactly one outer-left-out `train_calib_heldin_session` supplies target support `[0,33)` and
  query `[33,end)`;
- the outer session contributes zero train, normalizer, or checkpoint-selection windows.

The query filter is based on the **start** of the 50-bin window. A window is legal only when its
first bin is at or after the trial-33 boundary. Checking only the prediction bin is forbidden.
The frozen structural audit contains 101,171 eligible windows over seven sessions; the new
score-free audit must reproduce that sum and the seven per-session counts.

## 3. Arms and information disclosure

There are exactly two arms.

### Arm S — clean local SPINT

For each `seed × fold`, SPINT is trained once for a fixed maximum of 35 epochs on the six source
sessions. The selected checkpoint maximizes source-only `val_heldin/r2_mean`; ties choose the
earlier zero-based epoch. Missing, nonfinite, differently named, or target-session metrics fail
closed. The same selected SPINT checkpoint produces Arm S and is the sole decoder substrate for
Arm T.

At target calibration, Arm S consumes the first 33 neural trials and zero target-direction
labels. It computes ordinary SPINT activity identity by forward operations only. The target
session performs zero optimizer steps, zero backward calls, and zero parameter updates.

### Arm T — ordinary T4/B3S

Arm T source-trains for exactly 12 epochs. Its decoder must match the paired Arm-S checkpoint in
all 31 decoder tensors, bit for bit. All 31 decoder tensors remain frozen; zero decoder tensors
may require gradients or change value.

At target calibration, Arm T consumes the same 33 neural trials plus one target-direction label
for each eligible directional trial. M2 normally contributes 16 directional labels and 17
centre/rest trials in the first 33; centre/rest trials are excluded rather than assigned an
angle. The cosine design must have rank 3. The descriptor `[a,c,m,b]` is fitted in closed form and
cached. Target query behavior contributes zero labels, rates, normalizer samples, or selection
values. Target calibration has zero optimizer steps, backward calls, and updated parameter
tensors.

Therefore `T4-SPINT` estimates the deployment utility of an explicitly **supervised** calibration
package. Neural exposure is matched; label information is not. This protocol does not support an
equal-label-information or label-free claim, and it contains no TS4 mechanism arm.

## 4. Matrix, staging, and no-rescue rule

Seeds are fixed to `{42,43,44}` and folds to `{0,...,6}`:

```text
2 arms × 7 folds × 3 seeds = 42 terminal arm-cells
21 paired T4-SPINT deltas
21 SPINT trainings + 21 T4 trainings
```

Each paired `seed × fold` uses one SPINT training, not separate baseline and decoder trainings.

Stage A runs all 14 seed-42 arm-cells before any Stage-A score is read. Only after 14 starts,
14 completions, zero failures, zero extras, and complete hash closure may a finalizer jointly read
the seven paired deltas. The frozen severe-negative futility condition is:

```text
(mean42 <= -0.03 R²) OR (pos42 <= 1)
```

If true, the program stops as `seed42_severe_negative_futility_stop`; this is not a three-seed
ineffectiveness conclusion. If false, the only allowed action is automatic Stage B: all 28
seed-43/44 arm-cells. Non-triggering is named `continue_without_positive_claim`; it is not a pass,
trend, or candidate-selection signal.

There is no retry, seed replacement, arm addition, hyperparameter change, epoch scan, M24/M1
fallback, TS4/F0 substitution, decoder unfreezing, INT8 rescue, formal SUA access, or EvalAI
submission. Failed cells remain failed and prevent aggregation.

## 5. Full-matrix effectiveness gate

Only a 42/42 complete, hash-closed matrix can support a positive claim. All six conditions must
hold:

1. mean of the 21 paired deltas is at least `+0.03 R²`;
2. all three seed means are positive;
3. at least six of seven seed-averaged session means are positive;
4. the paired two-standard-error lower bound across the three seed means is positive;
5. the two-way seed/session hierarchical-bootstrap lower bound is positive;
6. equal-session absolute SPINT and T4 means are finite, and absolute T4 mean is positive.

The sequential two-SE/bootstrap quantities are engineering decision bounds, not unadjusted 95%
confirmatory confidence intervals. One immutable full aggregate is permitted.

## 6. Runtime evidence required later

Every future cell must bind the final receipt, source map, data manifest, config, seed, fold,
outer session, checkpoint, and normalizer hashes. Runtime closure must prove:

- outer-session counts are zero in train, normalizer, and checkpoint selection traces;
- support labels and rates align one-to-one and the design is rank 3;
- the minimum query-window start is at or after the trial-33 boundary;
- target calibration has zero optimizer steps, backward calls, and updated tensors;
- decoder comparison is 31/31 bit-exact and the decoder stays frozen;
- parameter count, source-training MACs, target-calibration MAC/state, online MAC/state, and
  latency protocol are present;
- no forbidden file, scorer, formal SUA path, EvalAI call, or extra arm appears.

This draft freezes the intended evidence schema but is deliberately not a launch receipt. Root
must review the score-free audit, focused tests, configs, and verifier, then issue a new explicit
authorization without modifying this draft or the old v1 receipt.

## 7. Score-free plumbing evidence completed on 2026-08-04

- all 14 native-M2 held-in calibration/minival NWBs were byte-size and SHA-256 verified;
- all seven first-33 supports contained 16 directional trials, rank-3 cosine designs, and finite
  2-norm condition numbers from `1.5837` to `2.0900`;
- the frozen per-session window counts summed to 101,171;
- 14 live data-module setups (`2 sides × 7 folds`) reproduced 101,171 windows independently on
  both the SPINT and T4 paths, with outer train/normalizer/checkpoint-selection counts all zero;
- the focused score-free suite passed 23 tests, and both Hydra configs composed with the intended
  specialized modules, source-only callback metric, and 35/12 epoch budgets;
- the active C1 v3r2 34-file source map still matched 34/34 hashes and was not edited.

These are endpoint and plumbing facts only. They provide no R², efficacy trend, launch
authorization, or positive/negative scientific verdict.
