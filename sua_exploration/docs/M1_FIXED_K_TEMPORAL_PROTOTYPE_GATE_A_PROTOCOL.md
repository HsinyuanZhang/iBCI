# M1 fixed-K temporal prototypes — source-only CPU Gate A v2

**Status:** prelaunch only.  Step 2's immutable NO-GO unblocks this CPU gate, not a decoder,
GPU, formal evaluation, or EvalAI submission.

## Scope and source manifest

The only admissible inputs are the four exact native-M1 `held-in-calib` NWBs named in
`m1_fixed_k_temporal_prototype_gate_a_v2_source_manifest.json`: `ses-20120924`, `20120926`,
`20120927`, and `20120928`.  The runner verifies their full SHA256 values.  It does not discover
paths recursively, and rejects any minival, held-out, test, EvalAI, or formal path.

Calibration support is chronological trials `[0,10)`.  The raw production `FalconDataset` view is
configured with no smoothing, no interpolation, `max_trial_length=1024`, and `pad_value=-1`.  For
every trial it verifies: valid-prefix values are finite nonnegative integers; the entire tail after
the true `calib_trial_lengths[t]` is exactly `-1`; and valid-prefix sums equal raw
`calib_trial_spike_sums`.  It then uses only `trial[:calib_trial_lengths[t]]`; padded/interpolated
values never reach a carrier or target.  The temporal filter resets before every trial.

## Frozen representation and outer LOSO

The representation remains fixed: `K=4`, rank `r=4`, fixed causal EWMA bank
`(0.5, .25, .125, .0625)`, deterministic source/unit-order-invariant anchor fitting, hard nearest
shared-anchor routing, and ridge `lambda=1`.  In every outer LOSO fold, anchors, normalization,
and the ridge proxy readout are fit only from the other three sessions' complete raw first-ten
supports.  The left-out session does not affect those objects.

All four arms have width 20: `D4`, neural-only `rate_only`, neural-only `prototype`, and
`slot_shuffle`.  D4 alone receives first-ten `obj_id` labels.  Slot shuffle applies a deterministic
session-keyed nonidentity permutation to complete `(count, rank-vector)` blocks for every train and
left-out session in the fold, shared across units within a session.

## Primary scorer-only endpoint

For each session, use trials `[210,end)` only to score a later-neural oracle target:

```text
Y[channel, obj_id] = log1p(mean_over_trials_with_that_obj_id(raw_trial_count / (valid_bins * 0.020 s)))
```

All four future `obj_id` levels and their exact expected counts are frozen in the manifest.  These
future labels are scorer-only: they cannot enter an anchor, route, carrier, readout input, or any
deployment claim.  This is a neural-rate proxy, not behavioural decoder R².

## Repeatability and Gate A

For every outer left-out session, perform 256 within-trial disjoint `Binomial(count, 0.5)` spike
partitions of its first-ten support bins.  Full outer-train anchors and ridge readout remain frozen;
there is no refit per seed.  Summarize each seed by median per-unit cosine between the two flattened
slot prototype values (counts excluded).  Seeds characterize measurement repeatability and are not
biological N; sessions (`n=4`) remain the inference unit.

For each contrast `prototype-rate_only` and `prototype-slot_shuffle`, report four outer-session
paired R² deltas, Student-t 95% CI, and

```text
MDE_80 = (t_0.975,df=3 + t_0.80,df=3) * SD_session / sqrt(4).
```

The CPU gate passes only if prototype exceeds both controls with CI95 lower bound above zero and
mean delta at least the measured MDE, all repeatability partitions are defined with each session's
2.5% similarity quantile above zero, and all raw/state/no-oracle contracts pass.  A failure stops
this branch; it cannot justify a K/r sweep, learned router, decoder fitting, GPU run, or formal
test.

## Execution guard

The script defaults to a no-NWB prelaunch receipt.  Real source data can be opened only when both
an independent root-review file binds its SHA to the immutable prelaunch receipt and the exact
source-only confirmation phrase is supplied.  Output directories are fresh-only.
