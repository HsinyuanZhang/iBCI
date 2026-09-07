# Design: H1 Support-Resampled Post-Pool V1

Date: 2026-09-03  
Status: `FROZEN_SOURCE_ONLY_DATE_LODO_THREE_ARM`

## 1. Question

The existing H1 evidence does not test the narrow middle point we need:

- frozen C1 with a direct MLP-after-pooling substitution was nearly neutral
  (`-0.001379 R2`, 4/5 dates nonnegative);
- a 700-parameter post-pool profile had positive mean (`+0.007046`) but only
  2/5 dates nonnegative;
- updating the complete 10.95M-parameter model for 12 epochs made both the
  native and joint post-pool arms worse than frozen C1 (about `-0.008` to
  `-0.009`).

This experiment tests whether the missing solution is to preserve the sealed
C1 decoder and train only its 58,140-parameter identity branch under the exact
official M3 support cardinality.

## 2. Frozen operator

For one ordered M3 support set, let

`e_i = carrier_pre_pool(activity_i)`

and let `c_S` be the frozen H-C carrier fitted from the same three trials.  The
late-pool identity is

`h_LP = mean_i carrier_post_pool(concat(e_i, c_S))`.

This is genuinely MLP-after-pooling.  It is not the native C1 operator

`h_C1 = carrier_post_pool(concat(mean_i e_i, c_S))`.

Activity and carrier always use exactly the same three trials.  The deployment
law is the chronological first three calibration trials because all 14 official
H1 held-out-calibration recordings contain exactly M3.

## 3. Arms

All trained arms start from the same sealed date-LODO C1 epoch-49 checkpoint.
The decoder (`fc_in`, transformer, `fc_out`) and every non-identity parameter
remain frozen and in evaluation mode.  Only `carrier_pre_pool` and
`carrier_post_pool` are optimized.

1. `LP-F3`: late-pool identity; every source update uses chronological first M3.
2. `LP-R3`: late-pool identity; deterministic 50/50 anchored support law:
   even support events use first M3, odd events choose uniformly from the
   session's non-first contiguous M3 blocks.  The three activity trials and
   carrier trials are identical.
3. `SRPD`: the exact LP-R3 support schedule plus native-identity distillation.
   The loss is

   `L = MSE(y_LP, y) + 1e-5 * MSE(h_LP, stopgrad(h_C1)) / (mean(h_C1^2)+1e-8)`.

The `1e-5` weight was frozen from one source-only preflight before any outer
date was opened: task MSE `5.04e-7`, normalized identity MSE `1.50e-2`, so the
initial distillation contribution is about 30% of the task term.  There is no
weight sweep.

`FROZEN-C1` is the untouched evaluation anchor, not a fourth training process.

## 4. Source protocol

- five source-grouped date-LODO folds: 19250108, 19250113, 19250115,
  19250119, 19250120;
- training inputs: source-date held-in-calib plus the distinct held-in-minival
  recording;
- query endpoints: the inherited stride-4 source surface;
- 12 fixed epochs, batch 32, Adam `5e-5`, weight decay 0, seed 42;
- no epoch selection, seed selection, hyperparameter sweep, formal held-out
  access, or EvalAI access;
- outer-date calibration/minival is opened only after all three epoch-12
  branch checkpoints for that fold have been sealed;
- all arms see identical query batches; LP-R3 and SRPD see the identical M3
  support event on every update.

The random support bank contains all legal contiguous M3 blocks.  This is a
source-side augmentation only.  It does not claim that official deployment
contains more than three calibration trials.

## 5. Main readings

For each outer date report equal-recording R2 for `FROZEN-C1`, `LP-F3`,
`LP-R3`, and `SRPD`.

Primary SRPD gate:

- mean `SRPD - FROZEN-C1 >= +0.005`;
- at least 4/5 dates nonnegative;
- worst date at least `-0.010`.

Support-diversity reading:

- `LP-R3 - LP-F3` mean and per-date paired deltas;
- diversity is considered consistently directional only when the absolute
  mean is at least `0.003` and at least 4/5 dates have the same sign.

Distillation reading:

- `SRPD - LP-R3` mean and per-date paired deltas;
- SRPD may advance only if its primary gate passes and the distillation
  increment has mean at least `-0.002` and worst at least `-0.010`.

If SRPD fails, a non-distilled arm may advance only if its own primary gate
passes.  A passing result authorizes a separate all-source/package work order;
it does not authorize an automatic EvalAI submission.

## 6. Carrier extension trigger

No carrier ablation is bundled into the main cell.  Exactly one source-only
extension may be proposed afterward only if either support resampling or
distillation has absolute mean effect at least `0.003` with a common sign on
at least 4/5 dates.  The extension must decompose matched activity support from
carrier sensitivity on the frozen winning checkpoint and cannot change the
main result.

## 7. Stop rule and allowed claim

If no arm passes, close H1 late-pool+C1 training rather than increasing epochs,
selecting a favorable date, or sweeping distillation weights.  The allowed
negative conclusion is narrow: under exact-M3 source-grouped training, a
frozen C1 decoder could not turn the late-pool identity branch into a stable
cross-record improvement.

