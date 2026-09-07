# M1 fixed-K temporal prototypes — source-only CPU Gate A v3

**Status:** immutable-prelaunch review only. The v2 prelaunch is superseded because its use of
`FalconDataModule.setup()` could resolve minival files. Step 2's NO-GO permits this CPU source
gate only; it does not permit a decoder, GPU, formal evaluation, or EvalAI.

## Exact scope and raw production trialization

The sole input scope is the four hashed `held-in-calib` NWBs in
`m1_fixed_k_temporal_prototype_gate_a_v2_source_manifest.json`: `ses-20120924`, `20120926`,
`20120927`, and `20120928`. The runner never uses `FalconDataModule.setup()` or a recursive file
search. Instead, it calls `load_nwb` once per exact manifest path, constructs one production
`FalconDataset` from only those four objects, and fails if an unlisted path is requested.

The calibration view is raw and non-interpolated: `smooth_calibration=False`,
`interpolate_trials=False`, `max_trial_length=1024`, `pad_value=-1`, and
`use_calib_intertrials=False`, matching production D4/support semantics. Only the first ten
variable-length valid prefixes are materialized as raw bins. For each, values must be finite,
nonnegative integer counts; every tail after the true `calib_trial_lengths[t]` must be exactly
`-1`; and its valid-prefix sum must exactly equal `calib_trial_spike_sums[t]`. No padded or cubic
value is passed onward. The temporal filter resets at every trial boundary.

For trials `[210,end)`, no raw trial matrices are retained. The scorer target uses only production
`calib_trial_spike_sums` and valid `calib_trial_lengths`, paired with production
`calib_trial_obj_ids`. The frozen target is

```text
Y[channel, obj_id] = log1p(mean_over_trials_with_obj_id(raw_count / (valid_bins * 0.020 s))).
```

Future `obj_id` is scorer-only. It cannot enter a carrier, anchor, route, readout input, or
deployment interpretation. All support and future label counts are hash-bound in the manifest and
must pass exactly.

## Fold, controls, and proxy

The fixed representation is `K=4`, rank `r=4`, causal EWMA alphas `(0.5, .25, .125, .0625)`,
shared deterministic anchors, hard routing, and ridge `lambda=1`. Each outer source-LOSO fold
fits anchors and ridge only from the other three sessions' complete raw supports `[0,10)`. All
arms have width 20: D4 (first-ten support `obj_id` only), rate-only, prototype, and slot shuffle.
Every train and left-out session receives a deterministic session-keyed nonidentity shuffle of whole
`(count, rank-vector)` blocks for the slot-shuffle arm.

## Repeatability and strict stop rule

For each left-out support, 256 disjoint within-trial `Binomial(count,.5)` partitions are formed.
The two halves are multiplied by two before routing against full-outer-train frozen anchors, making
their rate expectation match full-count calibration. No anchor or readout is refit per seed.

Each repeat separately measures median per-unit cosine for (a) normalized four-slot count
distributions and (b) flattened slot prototype values. Each session passes repeatability only if
both metric lower 2.5% seed quantiles are at least `0.5` and at least 90% of units are
prototype-value-defined in every repeat. Seeds measure technical repeatability only; outer session
is the biological/statistical inference unit (`n=4`).

For both `prototype-rate_only` and `prototype-slot_shuffle`, report paired outer-session R² deltas,
two-sided Student-t 95% CI, and

```text
MDE80 = (t(0.975, df=3) + t(0.80, df=3)) * SD_session / sqrt(4).
```

Gate A passes only if both contrasts have CI95 lower bound above zero and mean delta at least their
observed MDE80, all raw/state/oracle receipts pass, and repeatability passes. Otherwise it stops;
there is no K/r/router/decoder/GPU/formal expansion.

## Double guard

Default execution writes only a prelaunch receipt and opens no NWB. A real source run requires an
independent root-review authorization bound to the exact prelaunch SHA plus the exact v3
source-only confirmation string. Outputs are fresh-only. No formal, held-out, minival, decoder,
EvalAI, or CUDA code path exists.
