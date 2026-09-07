# Work Order — M2 Post-Fusion Operator-Corrected Checkpoint Score V1

Status: **implementation authorization; CPU-only; no training**.

This work order corrects one already demonstrated train/score operator mismatch
for the immutable PF-R1 and PF-R50 checkpoints.  It does not revise, retry, or
reinterpret the immutable V1/V2 checkpoint-score roots.

## 1. Question

At every causal decode-before-commit point, does a residual Post-Fusion
checkpoint perform differently when it is evaluated with the same whole-pool
operator used by training?

For a current ordered activity pool `A = [a_1, ..., a_M]`, the only admissible
identity is:

```text
adapter.forward_batch(A[None, ...], side_features=T4_M30_normalized)
```

For PF-R1/PF-R50, it is prohibited to compute one residual identity per trial
and average those identities.  That prohibited expression changes the native
pre-pool/nonlinear branch and is not the training graph.

PF-MEAN is a mandatory control.  Its exact 26 rows (`13 sessions × 2 laws`),
including every prediction SHA-256 and governed R2, plus its summaries and
contrasts, must reproduce the successful V2 score exactly.  The control must
reuse V2's literal singleton-identity/incremental-pool arithmetic: after a
FIFO eviction, re-summing an equivalent whole stack can change float32
addition order and is not an admissible replacement for a SHA-exact control.
Numeric tolerance is not an acceptable replacement for this control.  Any
mismatch is fail-closed and no residual result is interpretable.

## 2. Additive scope and immutable predecessors

New code is limited to:

```text
tfpd_exploration/src/m2_postfusion_operator_corrected_score_v1/
tfpd_exploration/scripts/run_m2_postfusion_operator_corrected_score_v1.py
tfpd_exploration/tests/test_m2_postfusion_operator_corrected_score_v1.py
```

New canonical result root, reserved only after opaque live admission:

```text
tfpd_exploration/results/m2_postfusion_operator_corrected_score_v1
```

The route descriptor-reads with held parent directory FDs and `O_NOFOLLOW` the
successful V2 graph at:

```text
tfpd_exploration/results/m2_postfusion_checkpoint_score_v2
```

It requires exactly the five immutable JSON receipt pairs, no failure or extra
leaf, `0444` regular body/sidecar files, link count one, basename-bound
sidecars, and these body SHA-256 values:

| Body | SHA-256 |
|---|---|
| `attempt.json` | `6bada93b3f6273a04866c0878f5367b9a9fcae4e5040e2a4ab2fcee7bb059bc5` |
| `launch.json` | `4e88d2b2df57430e36fcb15a5ab948ddb267534599f7507c2b53f65773a44e70` |
| `input_authority.json` | `92e85b8d0c1b27b4b40c857f4459969c76907e4a630057c5e697792f49cb657e` |
| `score.json` | `9dce9042360a4d835859532bd3ad97c0d586d74e25499a11f2898e2fce6adfef` |
| `terminal.json` | `75b5719e9cb46fa2c50e45131049ba1903a70a8923849f354a78b2f39eefd2f1` |

The V2 closure is historical evidence (`68d489aacd23611c625f98fee919430451fad251d3b681e0d01771bf546f982d`),
not this successor's closure.  V2 supplies the locked 13 input authorities,
the three screen checkpoint descriptors, the sealed POOLED comparator, and
the literal PF-MEAN control rows.

The screen checkpoint parameters are frozen descriptive evidence, not a
selection condition: PF-R1 has `tanh(alpha)=-0.6469457746`; PF-R50 has
`tanh(alpha)` mean `-0.42333123`, minimum `-0.61189920`, and maximum
`-0.17585394`.  Thus the correction is a material extrapolation toward the
native branch, not a near-zero numerical perturbation.

## 3. Fixed science contract

All of the following are inherited exactly from V2:

- three screen checkpoints: PF-MEAN, PF-R1, PF-R50;
- B30, D-opt-k4 activity support and M30 ridge carrier/T4;
- external-post30-local (6 sessions) and within-post30 (7 sessions);
- governed post30 query starts, targets, validity masks, metric, and order;
- the support-never-evicted law, completed-query FIFO at capacity 30, and
  UNCAPPED law;
- decode-before-commit ordering;
- CPU-only frozen decoder, no optimizer/backward/parameter update/target
  update; and
- the historical sealed `m4_activity_only` POOLED comparator.

The route produces exactly `3 arms × 2 laws × 13 sessions = 78` canonical
rows, in the V2 order.  Each input record and every PF-MEAN row must bind to
the corresponding V2 authority.  PF-R1/PF-R50 are the only changed operator
rows.

For every law, the ordered raw activity pool starts with the four selected
support activity rows.  Each completed query activity is appended only after
both law decodes.  FIXED30 evicts exactly the oldest completed activity at
index `support_count`; support rows are never evicted.  The current *whole*
ordered activity stack is passed once into the trained public adapter for the
corresponding arm.  The resulting `[1,96,50]` identity is used for the current
query windows only.

## 4. Admission, lifecycle, and CPU boundary

Public CLI is inert and cannot mint a capability.  A root-only opaque one-shot
capability binds the canonical fresh successor root, parent/named inode
identity, exact CPU environment (`CUDA_VISIBLE_DEVICES=''`,
`PYTHONNOUSERSITE='1'`), current explicit closure, and V2 predecessor witness.
It is checked at issue, attempt, before physical materialization, and final or
failure publication.

The route is CPU-only.  It must assert `torch.cuda.is_initialized() is False`
before and after physical work; it may not enumerate, initialize, or fallback
to either physical GPU.

The immutable lifecycle is:

```text
attempt -> launch -> input_authority -> score -> terminal
```

or a stage-honest prefix-preserving `failure` with no terminal.  The attempt
precedes every V2 descriptor, checkpoint, Torch, source, or target read.
Terminal/failure revalidates the V2 graph, closure, CPU environment, and root
identity.  Failure records a bounded exception class/message and published
prefix; no published immutable body is removed.

## 5. Mandatory pre-launch tests

No-data/CPU tests must prove:

1. the whole-stack PF-MEAN identity equals the legacy sequential singleton
   mean at M=4, 10, and 30, while PF-R1/PF-R50 do not silently use that
   singleton decomposition on nondegenerate inputs;
2. exact activity-pool order, decode-before-commit, support retention, FIFO
   eviction, and FIXED30/UNCAPPED equality before the first eviction;
3. per-row PF-MEAN control comparison to V2 prediction SHA/R2, with a
   fail-closed mismatch adversary;
4. V2 held graph body/sidecar/mode/topology/closure/link adversaries;
5. exact 78-row ordering/cardinality and independent metric/summary/contrast
   recomputation; and
6. inert CLI, capability forgery/reuse/root/closure/environment drift, and
   terminal XOR failure behavior.

An optional real source-only algebra test may strict-load the three immutable
screen checkpoint bytes on CPU, but it may not score target rows or write a
successor result root during construction.

## 6. Interpretation gate

- PF-MEAN mismatch: successor is invalid; stop.  This controls the inherited
  literal scorer, not a re-associated floating-point reduction after FIFO
  eviction.
- PF-R1 and PF-R50 both remain materially below POOLED on the locked external
  UNCAPPED contrast, i.e. neither meets `mean delta >= +0.010` and at least
  `4/6` positive sessions: close this Post-Fusion residual architecture axis;
  no extra GPU training.
- A residual arm recovers external performance: report only the fixed
  operator-correction result.  It does not establish placement causality;
  a separately authorized matched pre-fusion control would still be required.

No checkpoint, support, memory law, epoch, width, target signal, or threshold
may be selected after viewing this score.
