# DANDI 000688 sparse-event T4/FiLM Stage-2 V2 single-GPU reallocation incident

**Status:** frozen additive-successor authority.

The user requested a single-GPU allocation, retaining GPU 1 for separate work.
The managed Stage-2 V2 estimator seed 44 process was therefore interrupted by
the controller during CPU preparation, before CUDA, decoder loading, NWB/data
opening, training, scoring, or checkpoint publication.  This incident is
`USER_REQUESTED_SINGLE_GPU_REALLOCATION__PRE_CUDA`.  It is not a model result,
model failure, data failure, training failure, or scientific gate result.

The held V2 estimator root is exactly
`stage2_v2/estimator/sua/seed44/`, with these immutable leaves:

| Leaf | SHA-256 |
| --- | --- |
| `attempt.json` | `c6b85f67d7029d6a338b0966d77bc6d1bffa1479dd3cc0066275547411e6cf85` |
| `failure.json` | `95103d16b9b0013b8e6df4f446e84f5835647d5baaf7b6707c0b1e1bd67e3477` |

At validation, it contained exactly `attempt.json`, `attempt.json.sha256`,
`failure.json`, and `failure.json.sha256`; every leaf was regular, `0444`, and
single-link.  The attempt records `gpu_initialized: false`.  The failure is a
`KeyboardInterrupt` whose published prefix is the V2 attempt pair.  No leaf
may be added to, changed in, or reused from this root.

The sole authorized correction is the additive `stage2_v3` namespace, using
only GPU 0.  It reruns estimator seed 44 only at
`stage2_v3/estimator/sua/seed44/`, after held validation of this document and
all four V2 leaves.  It does not rerun estimator seeds 42 or 43: their
successful V1 terminals remain authoritative.  FiLM is unaffected: all three
FiLM terminals remain exclusively under `stage2_v2/film/sua/seed42..44/`.
The sole final aggregate must explicitly bind estimator authorities V1/V1/V3
for seeds 42/43/44 and FiLM authorities V2/V2/V2, and Stage 3 must admit only
that newest aggregate.
