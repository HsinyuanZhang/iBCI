# M1 D4 minimal decoder pilot protocol

**Frozen:** 2026-08-01 (Asia/Hong_Kong)  
**Status:** implementation authorized; launch waits for `m1_clean_selection_v1` completion.  
**Primary question:** Does the M=10 categorical calibration profile improve the real frozen-SPINT
decoder without exposing any query label?

## 1. Candidate and controls

### D4

For neural channel `i` and the chronological calibration support trials `[0,10)`, compute the
four exposure-corrected mean firing rates indexed by the observed `obj_id` levels `{1,2,3,4}`:

```text
D4_i = [mu_i,1, mu_i,2, mu_i,3, mu_i,4]
```

The label is used only in the support block to fit this static per-channel vector.  The decoder
receives D4 and online neural activity; it never receives a query-trial `obj_id`, `tgt_loc`,
`tgt_obj`, `condition_id`, EMG target, or hidden label.

The method name is **categorical calibration profile (D4)**.  The terms O4, object descriptor,
object-aligned identity, and dominant object factor are prohibited because Gate S failed.

### DS4

Fit and train-normalize D4 exactly as above, then deterministically permute the complete D4 rows
across channels within each session.  This preserves feature values and their distribution while
breaking channel attachment.  It is the mechanism-matched analogue of TS4.

Support-label permutation is not DS4 and is not part of this pilot.

## 2. Frozen implementation contract

- Native-MUA M1 only; `calibration_n_trials=10`, `random_calibration=false`.
- Reuse the exact T4 calibration trial exposure/counting contract.  Native-MUA concatenated spike
  rows must use explicit boolean interval counts or the already validated trial-sum tensors; never
  apply `searchsorted` to a globally non-monotonic concatenated row.
- All four `obj_id` levels must be present.  Missing or duplicated level semantics fail closed.
- Side dimension is four and the model is the existing B3S path; no architecture, loss, optimizer,
  teacher, dropout, or decoder change.
- Fit normalization on training-session D4 rows only.  Validation/report/held-out sessions cannot
  contribute normalization statistics.
- DS4 is applied after fitting and normalization by a deterministic non-identity channel-row
  permutation keyed by session and run seed.
- Unit tests must bind exact hand-computed D4 values, exposure handling, train-only normalization,
  DS4 distribution parity/non-identity, chronological support, and fail-closed label coverage.

## 3. Exact pilot cell

Train only the two new arms below for `fold1_seed42`:

| Arm | Architecture | Side feature |
|---|---|---|
| D4 | existing B3S | aligned categorical profile |
| DS4 | existing B3S | channel-row-shuffled D4 |

Reuse the already frozen `m1_clean_selection_v1` F0 and T4 `fold1_seed42` arms as comparison
anchors.  Do not retrain them for this pilot.

All training and evaluation settings must match `m1_clean_selection_v1`:

- support `[0,10)`;
- checkpoint-selection window `[10,210)`;
- sealed report window `[210,end)`;
- LOSO fold 1, seed 42;
- maximum 12 epochs;
- primary checkpoint `clean_best`; fixed `last` is diagnostic only;
- no report-window access until both D4 and DS4 training/checkpoint selection finish.

The two new jobs may run simultaneously, one per GPU, only after all pre-existing M1
clean-selection training and report work has finished.

## 4. Frozen early-stop decision

Primary report-window contrasts for `clean_best` are:

```text
D4 - F0
D4 - T4
D4 - DS4
```

Advance beyond the single cell only if all three conditions hold:

1. `D4 - F0 >= +0.015 R2`;
2. `D4 - T4 > 0`;
3. `D4 - DS4 >= +0.010 R2`.

Otherwise stop D4 decoder work immediately.  The thresholds are an early screen, not a
task-wide effectiveness claim.

If the cell passes, a separate frozen extension may add only `fold1_seed43` and
`fold2_seed42`.  The three-cell target is mean `D4 - F0 > +0.03`, positive mean
`D4 - DS4`, and no post-hoc architecture or endpoint changes.  Because those cells cover only
two distinct held-in sessions, even a positive three-cell result remains development evidence;
the decisive deployment endpoint would be a separately authorized official M1 held-out test.

## 5. Explicit non-goals

- No M={40,90,200} decoder sweep.
- No O4/object claim.
- No query labels at inference.
- No D4+T4 mixture, FiLM, confidence MLP, cross-attention, extra harmonics, or electrode lookup.
- No PTQ/QAT until the unquantized D4 decoder pilot is positive.
- No EvalAI submission from a one-cell result.
