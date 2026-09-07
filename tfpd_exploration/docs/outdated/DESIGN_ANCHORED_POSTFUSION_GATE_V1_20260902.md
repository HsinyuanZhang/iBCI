# Design — APFG: Anchored Post-Fusion Gate V1

Date: 2026-09-02. Status: **frozen design; execution authority is carried only
by the separate live work order**.

## 1. Motivation

The first Post-Fusion screen jointly trained the entire identity encoder and a
residual gate from a teacher initialization.  Its PF-MEAN checkpoint improved
the within continual-memory surface but lost `0.1122` external R2 versus the
sealed POOLED champion.  That result mixes two questions:

1. whether a post-fusion statistic contains useful complementary information;
2. whether full source-side fine-tuning preserves the already strong native
   cross-session solution.

The observed pattern is the familiar source-overfit signature, so another
full-network Post-Fusion retrain is not justified.  APFG instead nests the
sealed POOLED/native route as an exact safety anchor and learns only the
smallest possible residual gate.

The operator-corrected checkpoint score is running under a separate frozen
work order.  Its outcome cannot change the APFG definition below:

- if corrected PF-R1/PF-R50 recover, APFG is the clean anchored confirmation;
- if they remain negative, APFG is the last low-capacity test of whether their
  failure was caused by joint encoder drift rather than absence of a useful
  post-fusion statistic.

## 2. Method

For the same ordered causal activity pool and the same frozen historical
POOLED carrier, maintain two identities:

```text
h_pre  = native B3S pre-fusion identity
h_post = arrival-order mean of per-trial post-fusion identities
g      = tanh(alpha)
h_APFG = h_pre + g * (h_post - h_pre)
```

`alpha` is one scalar float32 parameter initialized to exact positive zero.
The decoder, native B3S `pre_pool`, native `post_pool`, T4 normalizer, and all
other parameters are frozen.  No target parameter, pseudo-label, carrier
update, attention, per-bin gate, or additional MLP is allowed.

This is a residual gate, not necessarily a convex interpolation: `g` may be
negative.  The already frozen PF-R1 checkpoint learned
`tanh(alpha) = -0.6469458`, which is descriptive evidence that the source loss
preferred extrapolation away from PF-MEAN.  That value is not reused as an
initializer, bound, or selection rule for APFG.

## 3. Exact anchor

APFG starts from the immutable selected-T4 POOLED checkpoint, not the
Post-Fusion teacher initialization.  At `alpha = 0`:

```text
h_APFG == h_pre
```

must be bitwise true before training.  The deployment scorer must use the
existing POOLED/native stream arithmetic for `h_pre`, so the zero-gate score
reproduces all 13 sealed POOLED comparator rows exactly, including prediction
SHA, R2, governed starts, targets, and window counts.  A merely close R2 is not
an acceptable control.

Because every inherited weight stays frozen, this zero-gate counterfactual is
available from the same trained artifact.  A separately retrained T0 network
is therefore unnecessary for the gate-effect estimate; there is no backbone
state that can differ between the control and intervention.

## 4. Training exposure

Only source sessions may affect `alpha`.  Training windows are restricted to
governed source coordinates whose endpoint is strictly after completion of the
first 30 calibration trials.  No window inside those first 30 trials is
eligible.  Each source session constructs the same family of causal activity
pools used at deployment:

1. D-opt selects the fixed four support trials from the first 30 calibration
   trials;
2. the frozen carrier is the exact historical `m4_activity_only` carrier: a
   fixed-ridge fit from those four selected support trials.  The separately
   recorded first-30 ridge array is audit evidence, not the carrier supplied
   to the decoder;
3. for each supervised window, only trials completed strictly before that
   window's query endpoint are eligible; future, current, and partially
   completed trials are forbidden;
4. the pool-size cycle is exactly `(4, 10, 30)`: M4 is the support four; M10
   is the support four plus the six most recent eligible completed
   non-support trials; M30 is the support four plus the 26 most recent
   eligible completed non-support trials, in chronological arrival order;
5. a requested cardinality is emitted only when that exact cardinality is
   available.  There is no clamp, pad, repeat, or future-trial substitution;
6. the same pool and selected-support carrier are used to compute `h_pre` and `h_post` before the
   supervised source window is decoded.

This replaces the prior mismatch of chronological `range(M)` activity with a
fixed inherited side tensor.  It does not use target sessions or target
labels.  Pool membership, D-opt indices, T4 digests, query coordinates, and
the `(4,10,30)` controller digest must be receipt-bound.

Every receipt binds the source `(session, window_start)`, its query-trial
position, the ordered completed-trial IDs, the selected activity digests, and
the proof that each member completed before the endpoint.  If the existing
batch contains more than one pool state, the route must group examples by
identical state or construct per-example calibration tensors; it may not give
the whole batch one member's pool.

Training remains exactly 12 epochs, Adam, constant learning rate `1e-4`, seed
42, and batch size 32.  Twelve epochs are a fixed matched exposure, not a
convergence claim.  Milestones are epochs 1, 3, 6, and 12.  Only `alpha` may
have a gradient or optimizer state; a single changed inherited parameter is a
hard failure.

## 5. Source-only selection without the old in-sample monitor

The old seven-session monitor was fully contained in training and cannot
select APFG.  Use a pre-declared grouped source split:

- selection fit: the first five session names in lexical order;
- selection validation: the final two session names in lexical order;
- select the earliest epoch attaining the best equal-session validation R2;
- after selecting the epoch, refit one scalar from exact zero on all seven
  source sessions for exactly that many epochs.

The split, roster, window disjointness, and selected epoch must be fixed before
opening any target surface.  Training batches are restricted to M30-ready
coordinates because their controller cycles through M4/M10/M30.  Selection
validation is deliberately a different, deployment-matched population: the
complete governed post-30 query stream beginning at trial 30 under UNCAPPED
decode-before-commit replay.  Receipts must bind separate counts and digests
for these two populations and must not describe validation as M30-ready.
The all-source refit may not inspect its in-sample metric to extend or shorten
the horizon.

The governing validation R2 must call the exact historical
`variance_weighted_r2` implementation and therefore preserve its per-output
float64 reduction order.  An algebraically equivalent aggregate reduction is
not acceptable where exact-zero comparisons are required.  A FIXED30
descriptive replay is computed only for the selected epoch, for both learned
alpha and exact positive zero; it is recorded but cannot select the epoch or
change the safety gate.

The source validation gate is intentionally asymmetric:

- finite, connected, and no worse than zero-gate by more than `0.002` in
  equal-session mean;
- at least one of the two validation sessions nonnegative versus zero-gate;
- otherwise stop before target scoring.

This is a safety/constructibility gate, not the paper result.

## 6. Deployment and sufficient statistics

The scorer uses the locked M2 surface from the completed POOLED/Post-Fusion
program:

- six `external_post30_local` sessions and seven `within_post30` sessions;
- B30/D-opt-k4 support and the exact selected-support four-trial fixed-ridge
  historical POOLED carrier;
- `FIXED30` and `UNCAPPED` memory;
- decode before committing the completed query trial;
- support rows are never evicted; FIXED30 removes only the oldest completed
  row.

GPU0 is used only for the source alpha fit and all-source refit.  Before any
target row is decoded, synchronize the source run, freeze the selected alpha,
move both ZERO and LEARNED models to CPU, and perform all target scoring on
CPU with the historical POOLED decode-batch/numeric authority.  This device
split is required because the immutable POOLED prediction digests were minted
on CPU; a CUDA recomputation would not be a valid bitwise zero-anchor test.
Receipts distinguish `source_training_device=cuda:0` from
`target_scoring_device=cpu`.

V1 retains the existing ordered raw-activity memory and calls the native and
post-fusion branches from that authority.  An incremental sufficient-statistic
cache is deferred: it is an optimization, not part of this experiment, and
could change float32 association around FIFO eviction.  The production
prediction uses the historical POOLED arithmetic for the native branch, not an
algebraically similar rewrite that changes the zero-gate prediction SHA.

## 7. Required comparisons

The single APFG artifact yields a real two-by-two decomposition:

1. `APFG-ZERO/FIXED30`: set `g=0`; this must be the exact sealed
   POOLED/native anchor;
2. `APFG-ZERO/UNCAPPED`: keep `g=0` but remove FIFO eviction, measuring the
   activity-memory-law effect without a learned gate;
3. `APFG-LEARNED/FIXED30`: use the frozen source-selected scalar under the
   historical bounded memory law;
4. `APFG-LEARNED/UNCAPPED`: use the same frozen scalar under uncapped causal
   activity memory.

Report FIXED30 and UNCAPPED on both surfaces.  The historical POOLED
comparator is a FIXED30 system.  It is therefore incorrect to require
`APFG-ZERO/UNCAPPED` to reproduce that row after the first FIFO eviction.
The required hard sentinel and attributable contrasts are:

```text
APFG-ZERO/FIXED30 == sealed POOLED/FIXED30        [bitwise hard sentinel]
APFG-ZERO/UNCAPPED - APFG-ZERO/FIXED30            [memory-only effect]
APFG-LEARNED/FIXED30 - APFG-ZERO/FIXED30          [gate-only, bounded]
APFG-LEARNED/UNCAPPED - APFG-ZERO/UNCAPPED        [gate-only, uncapped]
APFG-LEARNED/UNCAPPED - sealed POOLED/FIXED30     [total deployed effect]
```

Only the first equality must be bitwise exact, for all 13 sessions, before
any learned row is interpreted.  The UNCAPPED contrasts may differ because
an uncapped native B3S pool is not the historical POOLED memory law.
For every paired contrast, report the paired per-session deltas, mean,
median, positive-session count, worst-session delta, and a deterministic
10,000-resample ordinary session-bootstrap 95% interval with seed 42.  The
interval is descriptive and does not replace the preregistered promotion
gate.

Promotion uses the **total deployed effect** on locked external/UNCAPPED,
with sealed POOLED/FIXED30 as the comparator:

```text
mean paired delta >= +0.010 R2
positive sessions >= 4/6
```

`0 <= mean < 0.005` is null.  A within gain with external loss is the same
overfit signature as the first screen and stops this PF axis.  No epoch,
learning rate, pool law, support, gate clipping, or second seed is selected
from target results.

## 8. Compute plan

This is one scalar parameter over a frozen network.  GPU0 may be used only
after the current operator-corrected CPU score terminates and a separate work
order passes no-CUDA review.  GPU1 remains outside scope.

To avoid low utilization, one source batch may be reused for the fit/control
forwards, and the two identities may be computed from shared frozen branch
features.  No asynchronous stream, multi-process model replica, AMP,
`torch.compile`, batch-size change, or concurrent target scorer is authorized
in V1.  The public CLI must remain inert and launch through a root-owned opaque
capability.

## 9. Interpretation

A positive result supports a narrow but clean claim:

> A source-learned scalar gate over complementary pre- and post-fusion
> calibration statistics improves causal continual decoding while exactly
> nesting the original strong decoder as its zero-gate state.

A null or negative result is also decisive: after correcting the residual
operator and protecting the native backbone, Post-Fusion contributes no
transferable value on this M2 surface.  In that case the architecture axis is
closed; only the already validated pre-fusion CDM/T4 route remains.
