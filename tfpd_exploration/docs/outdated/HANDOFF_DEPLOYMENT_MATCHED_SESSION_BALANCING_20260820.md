# Handoff: Deployment-Matched Source-Session Balancing

Date: 2026-08-20  
Scientific owner and reviewer: root / Sol  
Implementation owner: Terra  
Runtime monitor: Luna  
Status: proposal only; no implementation, data access, GPU launch, or score authorization

## 0. Decision first

If the operator chooses to occupy GPU0 with a second performance experiment while TF-SR trains on
GPU1, the preferred cell is:

```text
CELL_D_EQUAL_SESSION_SEED42
```

This is a training-method experiment, not a decoder-architecture claim. It retains the complete
sealed Cell-D graph and changes only the distribution of source-session exposure. It must not be
launched until the operator gives GO, Terra completes an additive implementation, and root performs
an independent no-target review.

Do not replace this cell with GroupDRO, REx, IRM, a source-session classifier, a worst-k loss, a
temperature-weighted objective, or a sampler sweep. Those are different experiments with additional
hyperparameters.

## 1. Motivation

The current Cell-D trainer uses `SessionBatchSampler`:

```text
sua_exploration/mc_maze/multisession_datamodule.py:805+
```

It forms every full B32 batch within each session and then shuffles the union of those batches.
Consequently, a source session contributes in proportion to its number of eligible windows.

The governing deployment estimator is different:

```text
compute one variance-weighted last-bin R2 per session
then average sessions with equal weight
```

The scientific question is therefore:

> Does matching source optimization exposure to the equal-session deployment estimand improve
> zero-target-update transfer, without changing the decoder or increasing compute?

This is independent of the TF-SR question. TF-SR changes causal architecture and information flow;
this cell changes only source-domain weighting on the already sealed Cell-D system.

## 2. One changed factor

Held exactly equal to sealed Cell D seed 42:

```text
model graph                    exact Cell D, two heads, B3S+normalized T4
whole-token dropout            per-step p sampled from U(0,1), unchanged
initial state                  exact canonical Cell-D initial state
source roster                  exact strict-27
calibration                    exact M30
behavior_scaling_factor        None
batch size                     32
optimizer                      exact Adam authority
learning-rate schedule         exact 48-epoch warmup/cosine authority
epochs                         48
steps per epoch                33,925
total optimizer steps          1,628,400
SWA                            final epochs 44,45,46,47
seed                           42
teacher                        none
target optimizer/backward      forbidden
within/external/formal access  forbidden during training
```

Changed exactly once:

```text
source batch exposure:
  window-proportional SessionBatchSampler
  -> deterministic equal-session batch schedule
```

No other model, loss, mask, optimizer, scheduler, normalization, checkpoint, or metric field may
change in the first cell.

## 3. Frozen equal-session schedule

Let the canonical strict-27 roster order be fixed by the sealed manifest. Every epoch contains
exactly 33,925 B32 batches.

```text
33,925 = 27 * 1,256 + 13
```

For epoch `e`:

1. every session receives exactly 1,256 batch positions;
2. thirteen sessions receive one additional position; in the canonical strict-27 roster order,
   their zero-based indices are exactly `(13 * e + j) mod 27` for `j = 0..12`;
3. the thirteen extra positions rotate deterministically across epochs, so cumulative exposure over
   48 epochs is exactly 60,311 or 60,312 batches per session and differs by at most one batch between
   any two sessions;
4. the resulting session-position multiset is shuffled only by a local RNG whose integer seed is
   derived from SHA-256 over the ASCII domain string
   `CELL_D_EQUAL_SESSION_SEED42|global-order|42|{epoch}`;
5. within each session, eligible window indices are deterministically reshuffled in cycles;
6. a cycle consumes only full B32 batches; when exhausted, the next cycle uses a new deterministic
   permutation and continues until that session's epoch quota is filled; its local RNG seed is
   derived from SHA-256 over
   `CELL_D_EQUAL_SESSION_SEED42|session-cycle|42|{epoch}|{session}|{cycle}`;
7. no global Python, NumPy, Torch, sampler, model, or dropout RNG state may be consumed by schedule
   construction.

SHA-256 seed derivation means interpreting the full lowercase hexadecimal digest as an unsigned
big-endian integer. Python's process-randomized `hash()` is forbidden.

The implementation must publish, before training:

- exact eligible-window and full-batch counts for all 27 source sessions;
- original window-proportional exposure fractions;
- proposed per-epoch and full-run equal-session exposure counts;
- over/under-sampling ratios per session;
- schedule SHA-256 for every epoch and for the complete 48-epoch plan;
- proof that every batch contains exactly one session and 32 valid dataset indices;
- proof that the complete plan has 1,628,400 batches and no forbidden session.

Do not describe the current imbalance magnitude until this source-only audit has measured it.

## 4. Required Stage-0 and source-only gates

Before any GPU request:

1. synthetic unequal-session fixture proves exact equal exposure and deterministic replay;
2. repeated indices are allowed only after a session's current deterministic cycle is exhausted;
3. no duplicate index occurs within a B32 batch;
4. per-epoch extra-session rotation and full-run max-minus-min exposure are exact;
5. schedule construction does not change global RNG states;
6. real strict-27 source-only audit binds the sealed roster and exact window indices;
7. Cell-D graph, initial-state, parameter count, B3S/T4 path, dropout law, optimizer, scheduler, loss,
   total steps, and SWA epochs exact-match the sealed Cell-D authorities;
8. no within, external, formal, target, or organizer-held session is resolved or opened;
9. a short source-only GPU smoke proves one optimizer step, complete critical gradients, finite
   model/Adam state, and exact schedule accounting;
10. a measured memory/throughput receipt confirms GPU0 feasibility without changing B32.

All code and result roots must be additive. Existing Cell-D, TF-SR, scorer, and sealed result bytes
must not be edited.

## 5. Full run and evaluation

The only first run is seed 42 on physical GPU0. The sealed Cell-D seed-42 predecessor ran from
2026-08-17T17:24:26Z through 22:32:58Z, or approximately 5 h 08 min for the same 48 epochs and
1,628,400 optimizer steps. Use 5--6 hours as the planning estimate for this compute-matched cell;
it remains an estimate until the source-only GPU smoke measures the successor route.

Training must finish and seal the final-four SWA before any evaluation data is resolved. Evaluation
then uses the existing matched convention:

```text
governing metric       variance-weighted R2
governing query        last bin of each valid 50-bin window
aggregation            equal weight per session
within sessions        exact sealed six
external sessions      exact sealed sub-M fifteen
target updates         zero
```

Primary contrast:

```text
CELL_D_EQUAL_SESSION_SEED42 minus sealed Cell D seed42
```

Predeclared screen:

```text
CLEAR GO:
  external mean delta >= +0.03
  within mean delta >= -0.03
  external median delta > 0
  external positive sessions >= 9/15

HOLD:
  external mean delta in [0,+0.03), unless a STOP condition applies

STOP:
  external mean delta < 0
  or within mean delta < -0.03
```

This single seed cannot establish multi-seed superiority. If it clears the screen, seeds 43 and 44
may be proposed later. If it does not clear, stop this route; do not rescue it with GroupDRO or a
sampler/weighting sweep.

## 6. Licensed interpretation

Positive result:

> Matching source-session exposure to the equal-session deployment estimand improves zero-target-
> update transfer of a population-robust decoder.

Negative result:

> Source-window exposure imbalance is not the main remaining transfer bottleneck under the matched
> Cell-D protocol.

This cell is not licensed as a new decoder architecture, a new representation, or a replacement for
the TF-SR contribution. Uniform source-domain sampling is a standard idea; its value here is a
clean, compute-matched performance test of an estimand mismatch. The larger architecture successor
remains a carrier-generated observation/innovation filter (Route C), which requires a separate
handoff and Stage-0.

## 7. Scheduling boundary

```text
GPU1: keep the live TF-SR seed42 run unchanged.
GPU0: this cell only after explicit operator GO and root code/preflight acceptance.
Terra: implement only after the current Phase-E repair is frozen.
Luna: monitor any authorized GPU0 run independently at approximately ten-minute intervals.
Root: owns the contract, independent review, launch decision, result interpretation, and ledger.
```

This document itself authorizes no code implementation, source-data access, GPU smoke, full
training, target access, scoring, or result publication.
