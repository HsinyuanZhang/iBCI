# CDM-D pure-speed evaluator successor V1

Status: no-data/no-CUDA implementation work order.  This route is an
additive accelerator for the sealed Cell-D V8 and Precision-V2 evaluator
stacks.  It does not replace, modify, retry, or reinterpret any historical
CDM-D receipt, score, transition, model, normalizer, input authority, or
result root.

## Scope

The only intended production behavior is an opt-in composition over the
existing V8/Precision runtimes.  It may change evaluation work accounting and
add speed evidence, but must preserve the concatenated prediction bytes,
prediction SHA-256, last-bin R2, state-transition decisions, target boundary,
and sealed model state exactly.

### O1: per-state identity cache

For an exact activity stack, active normalized T4, and (for a held-group
forward) exact held-unit view, derive B3S identity once with batch size one and
reuse it via `StreamingSpintModel.forward(..., identity=...)`.  This is allowed
only when the live identity encoder has no `forward_batch_with_gate` semantics,
the model is the ordinary coupled decoder, and direct cached decoding is
bitwise equal to the eager route.  Any gate-bearing encoder, shape drift,
nonfinite identity, or eager/cached parity failure is a hard failure; bypassing
an encoder-provided gate is never allowed.

### O2: deterministic sampled repeat audit

The existing eval/no-grad/dropout/RNG invariants remain mandatory.  Rather than
calling a duplicate forward for every logical chunk, repeat only at two
predeclared coordinate classes:

1. the first chunk of the **canonical full-system** path on first use of each
   exact full trial state after a state transition; and
2. exactly one held-path coverage proof: the first chunk of `held_group_0` at
   deterministic `floor(n_query_trials / 2)`, even if that held-state cache
   was already warm.

All other full/held chunks, including first uses of `held_group_1` through
`held_group_3`, run once.  This deliberately avoids a repeat per
`(path, group, state)`: the full proof covers each causal state transition,
and the one predeclared group proof establishes held-path dispatch coverage.
The sealed Cell-D comparator has no held-group path, so it requires only the
canonical full-system proof; the held-group coordinate is mandatory only for
the dynamic CDM-D/Precision trajectories.

Receipt-ready evidence records exact query trial ID, state/cache digest,
path/group, chunk range, duplicate-output digest, bitwise-equality result, and
reasons.  The accelerator holds only compact coordinates/digests, never target
data or retained prediction tensors.

### Batch-size boundary

Logical evaluation batching remains **128** in this pure-speed version.
Real CPU Cell-D parity found a changed prediction SHA and max absolute drift
`2.60770320892334e-07` between chunks 128 and 2048.  Therefore 1024/2048 are
explicitly deferred numeric variants that need their own re-anchored successor;
they are not a selectable setting in this route.  The existing CPU microbench
is descriptive only (roughly 2.5x for cached identity plus sampled repeats),
not a GPU throughput claim.

O3 subsampling/interval gates, O5 concurrency scheduling, O6 disk caching,
and O7 TF32 are outside this work order.

## Required tests and future GPU0 gate

CPU synthetic/real-model tests must cover cached-vs-eager identity, variable
prefix held masks, chunk-partition prediction digest parity at B128, sampled
repeat coordinates, state/RNG/dropout invariants, forward-call accounting,
closure drift, and dry CLI inertness.  A later independently reviewed GPU0
smoke may run the sealed V8/Precision stack only after proving exact per-session
prediction SHA and R2 anchors against the applicable historical route,
recording speed evidence, and preserving all current authority/capability
boundaries.  This work order authorizes neither that smoke nor any result root
operation.
