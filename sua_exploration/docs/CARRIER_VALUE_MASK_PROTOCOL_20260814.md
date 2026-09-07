# Carrier value-weighted mask protocol

**Status:** queued after SetKV-delta; additive forward-only mechanism diagnostic.

## Question

Do weakly tuned unit tokens consume meaningful decoder routing capacity, and can excluding them improve use of the
same sparse T4 carrier without changing any trained weight?  The diagnostic targets SUA/sub-M only.  It does not
reopen the lower-priority H1 carrier-architecture branch.

## Frozen source screen

Use the sealed A2 seed-42 T4 bundle, its fixed first reporting epoch 5, and the first 128 post-30 query windows from
each of the strict 27 source sessions.  This cap is an engineering budget fixed before the diagnostic, not a
score-selected subset.  T4 is `[a,c,m,b]`; the mask ranks units by the within-session gain coordinate `m`.  For each
source query and each attention head, compute

```text
importance_i = mean_(head,query) attention_(head,query,i) * ||V_(head,i)||_2 .
```

This is a nonnegative routing proxy, not an exact signed output decomposition.  Attention mass alone is not a valid
gate.  The frozen candidate masks the bottom 25% of units by `m`, always leaving at least one unit.  If the median
source-session share of value-weighted mass in that quartile is below 0.10, stop before target scoring: those units
already have too little routed value for pruning to be an informative mechanism test.

The source probe follows the sealed A2 production source path, not the stricter target-session label gate.  Its
query windows remain strictly after usable rewarded trial 30, and `attach_side_features` constructs T4 through the
unchanged A2 production helper.  The production rewarded-trial loader normalizes a missing or non-finite direction
to `None`; `unit_side_features` retains that trial in chronological order and in the rate matrix, represents its
direction by `-1`, and excludes it from the present-direction set and all direction-conditioned means.  No label is
imputed and no session is removed.  The source receipt reports the observed finite and missing direction counts and
usable indices for both the first 30 and first 50 trials; it must not claim that all source directions are finite.

## Forward controls if the source gate passes

Reuse the sealed A2 T4 and Z4 checkpoint bundles, normalizers, M=30 support, post-30 query sets, epoch mean 5--12,
and seed 42.  No model is retrained.

1. matched A2 no-mask T4/Z4 receipts;
2. bottom-25%-`m` mask on T4 and Z4;
3. deterministic random same-count mask on T4 and Z4;
4. top-25%-`m` T4 negative control.

Z4 still constructs the ordinary target-session T4 substrate before its final standardized carrier is zeroed, so
the same label-derived mask is available to both siblings.  It must not be described as label-free.  The primary
read is the external-sub-M bottom-mask interaction relative to matched no-mask parents; random same-count masking
prices generic token removal, and the top-mask arm verifies that the gain ranking is behaviorally ordered.

## Boundaries

No target optimizer/backward step, normalizer refit, target-selected quantile, checkpoint selection, formal sub-C
session, new parameter, or production-model edit is permitted. The boolean mask is materialized once per
`(checkpoint, target session)` and reused across that session's query batches. It is one byte per unit in the
current implementation and must be reported as persistent state; one-time mask-setup wall time is reported
separately from query-forward wall time. Zero new parameters must not be called zero cost.
PyTorch `key_padding_mask` still projects and multiplies the full token tensor, so the current dense-MHA analytic
MAC delta is zero; only a future compacted-token implementation could convert a positive mechanism result into
compute savings. A flat result closes only this fixed forward mask, not trained sparse routing. Any positive result
is diagnostic and requires a separate training/deployment contract.
