# Frozen Protocol: H1 Tag-Free Position-Context Carrier (PCTX)

**Date predeclared:** 2026-08-12, before any candidate was executed
**Status:** CPU-only source screen. No GPU authorization is granted in advance.
**Scope:** the 13 public `sub-HumanPitt-held-in-calib` H1 recordings only.

---

## 1. Governance statement — read first

The sealed fold-0 Context Full cell has predeclared status `STOP_CONTEXT` because its
per-recording deltas against H-SE5 were `-0.004610/+0.077458`, failing the
both-recordings-positive clause.

**This protocol does not reopen, rescore, reinterpret, or rescue that cell.** `STOP_CONTEXT`
stands. The sealed receipts are inputs and comparators only.

What this protocol does instead: it predeclares a **new** family of candidates that were never
tested, screens them on CPU under gates fixed in advance, and — only if they pass — earns them
their **own fresh GPU cell** under a separate multi-date protocol. A passing candidate is a new
result with its own lineage, not a revived Context Full.

No number in this document may be changed after the first candidate is executed.

---

## 2. Motivation — what Context Full's gain is actually made of

Bound sealed reference: `sua_exploration/results/h1_event_carrier_design_screen_v1/source_screen.json`,
SHA `74bbc01490432794546e7ca2fd4242fbed6f2a7ebdd65786f56035eaa49bfeb3`.

Contrast against `pca_delta_q4` (= H-SE5):

| Candidate | features | M3 mean | M4 mean | positive |
|---|---|---:|---:|---:|
| `pca_delta_q4` (H-SE5) | `delta(7)`, PCA basis | 0 | 0 | - |
| `ser_delta_q4` | `delta(7)`, supervised basis | +0.005213 | +0.005051 | 9/13 |
| `ser_context_q4` (Context Full) | `delta(7), midpoint(7), onehot(8)`, supervised basis | +0.015390 | +0.014438 | 13/13 |

Decomposition at M4: the supervised source-encoding basis alone contributes `+0.005051` (35%);
adding midpoint and tags contributes a further `+0.009387` (65%).

At the decoder, the sealed same-checkpoint tag shuffle costs only `0.516518 - 0.513960 = 0.002558`
out of Context Full's `+0.016481` gain over H-SE5, i.e. about 16%.

**Inference to be tested: the absolute workspace position carried by `midpoint` supplies most of
the enrichment, and the event-tag one-hot supplies little.**

Two independent reasons this matters:

1. **Publishability.** The paper review sealed a prohibition on "H1 Context Full, event-tag,
   tag-shuffle, or the 0.5165 sparse-context claim". A carrier with no tag input is outside that
   prohibition by construction. It also removes the tag-shuffle control burden entirely.
2. **Variance dilution.** The K-point screen (2026-08-12) demonstrated that at fixed `q=4`,
   widening the raw feature space reduces retained variance and can cost accuracy
   (`0.7190 -> 0.6024` as raw dimensionality grew). Context Full carries 8 tag columns, of which
   only 6 are ever active (`SnapTo` and `Release` have zero parsed events across all 13
   recordings). Dropping them narrows the active raw space from about 20 to 14 dimensions and
   should raise retained energy at `q=4`.

**Critical gap confirmed by code inspection:** every context family in
`h1_event_carrier_design_screen.py` (`context`, `context_mid`, `context_start`, `tag_delta`,
`tag_context`) concatenates the tag one-hot. **No tag-free position-context family has ever been
constructed or evaluated.**

---

## 3. Invariants held fixed across every arm

- `q = 4`; carrier is `[176, 5]` ordered `[w1,w2,w3,w4,b]`. The decoder interface is unchanged, so
  a passing candidate needs no architecture change to translate to GPU.
- Target estimator: closed-form ridge, intercept unpenalised, `lambda = 3.0`, normalised by event
  count. Zero target-session optimizer or backward steps.
- Source basis fitted leave-one-date-out; the evaluated session's own date is never in its basis.
- Support = first M trials; forward-transfer evaluation on later events, exactly as the sealed
  design screen defines it.
- Budgets M3 and M4, both reported. A candidate must pass at **both**.
- Native `OpenLoopKinematicsVelocity` is never read.

---

## 4. Candidate set — frozen

Reference arms, which must reproduce the sealed receipt exactly:
- `pca_delta_q4` — H-SE5.
- `ser_context_q4` — Context Full.

New tag-free arms:
- `ser_poscontext_q4` — supervised basis, features `[delta(7), midpoint(7)]`. **Primary candidate.**
- `pca_poscontext_q4` — PCA basis, same features. Isolates whether the supervised basis is required.
- `ser_startstop_q4` — supervised basis, features `[start(7), stop(7)]`. Spans the same space as
  `[delta, midpoint]` but standardises different columns, so the basis can differ.
- `ser_deltastart_q4` — supervised basis, features `[delta(7), start(7)]`. Third reparameterisation.

Diagnostic arm, not gate-eligible:
- `ser_context_notag_energy` — reports retained energy for `ser_poscontext_q4` against
  `ser_context_q4` to test the variance-dilution prediction directly.

---

## 5. Predeclared gates — frozen before execution

Numeric thresholds are computed from the bound sealed values and fixed here.

### 5.1 Primary gate (advances a tag-free candidate to a GPU proposal)

At **both** M3 and M4, contrast against `pca_delta_q4`:

1. mean `>=` 80% of the sealed `ser_context_q4` mean at that budget:
   - M3: mean `>= +0.012312`
   - M4: mean `>= +0.011550`
2. at least **11/13** positive sessions (deliberately stricter than the historical 10/13, because
   credibility is the objective of this round);
3. positive leave-largest-absolute-session-out mean;
4. `correct - label_shuffle` and `correct - intercept` both satisfy the standard positive control:
   13/13 defined, mean > 0, median > 0, at least 11/13 positive, positive leave-largest-out.

Rationale for 80%: the objective is not to beat Context Full but to reach a comparable operating
point **without tags**, which converts a prohibited result into a permissible one. An 80%
pre-commitment prevents post-hoc rationalisation of a weak result.

### 5.2 Secondary stretch gate (recorded, not required)

mean `>=` the sealed `ser_context_q4` mean at both budgets (M3 `+0.015390`, M4 `+0.014438`), i.e.
the tag-free candidate matches or beats Context Full outright.

### 5.3 Failure handling

If no candidate passes the primary gate, the terminal state is
`STOP_CPU_PCTX_NOT_MATERIAL` and **no GPU arm is launched**. Thresholds must not be lowered, and
the candidate set must not be extended, to manufacture a pass.

---

## 6. If the primary gate passes — GPU credibility protocol

The whole point of this round is credibility, so the GPU phase is specified now, in advance, and
fixes the three weaknesses that produced `STOP_CONTEXT`.

1. **Multi-date from the start.** At least two outer dates, not fold-0 alone. Fold-0 uses the
   existing schedule for comparability; the second date is chosen from the established five-date
   LODO set.
2. **Separately trained corruption controls**, not same-checkpoint only. Context Full's controls
   were same-checkpoint, which is weaker than the separately trained TS4/LS4/XLSv2 arms every other
   dataset in the paper uses.
3. **Both-recordings-positive declared as an outcome clause up front**, with the same status
   semantics that produced `STOP_CONTEXT`. If it fails again, the result stops again.

No GPU arm may launch before the CPU receipt and an independent verifier both exist.

---

## 7. Required evidence

- One immutable CPU receipt, mode `0444`, canonical sorted JSON, binding the sealed design-screen
  receipt SHA, all input file SHAs, and the implementation SHAs.
- Exact reproduction of the sealed `pca_delta_q4` and `ser_context_q4` values at both budgets,
  within `1e-10`, proving the reimplementation is faithful.
- An **independent** second implementation that recomputes the headline contrasts from scratch
  without reading the primary implementation, matching to `1e-10`.
- Focused tests covering the new feature families, the LODO no-leakage boundary, the tag-free
  property (assert the tag one-hot never enters a tag-free arm's raw features), and determinism.

---

## 8. RESULTS — executed 2026-08-12. Terminal status `STOP_CPU_PCTX_NOT_MATERIAL`

Receipts:
- primary `sua_exploration/results/h1_tagfree_position_context/source_screen.json`, SHA `d86f7f8179b4f1b00b04f6ea77c6407c63d73d87378e97ddb5b56791d12f3a8d`
- independent `sua_exploration/results/h1_tagfree_position_context/independent_crosscheck.json`, SHA `8f6b684917eb9af48181c51b4b1902345b094dd44fa169ca4e651fecb5379bee`

Verification: the two implementations are **bit-identical** across 156 values on both
`median_r2_correct` and `median_delta_intercept` (`max|diff| = 0.000e+00`), and both reproduce the
sealed design-screen receipt for `pca_delta_q4` and `ser_context_q4` at exactly `0.0`.

### 8.1 Outcome — every tag-free candidate fails

Contrast against `pca_delta_q4` (= H-SE5):

| Candidate | tags | M3 mean | M3 pos | M4 mean | M4 pos | retained energy @4 |
|---|---|---:|---:|---:|---:|---:|
| `pca_delta_q4` (H-SE5) | no | 0 | - | 0 | - | 0.7188 |
| `ser_context_q4` (Context Full) | **yes** | +0.015390 | 13/13 | +0.014438 | 13/13 | 0.5063 |
| `ser_poscontext_q4` | no | +0.007229 | 12/13 | +0.004720 | 10/13 | 0.5237 |
| `pca_poscontext_q4` | no | -0.001472 | 7/13 | -0.005781 | 2/13 | 0.4179 |
| `ser_startstop_q4` | no | -0.000455 | 8/13 | +0.000261 | 6/13 | 0.5036 |
| `ser_deltastart_q4` | no | +0.007442 | 11/13 | +0.003049 | 9/13 | 0.5258 |

Required at M4: mean `>= +0.011550` and `>= 11/13` positive. The best tag-free candidate reaches
`+0.004720` and `10/13`. **No candidate passes. No GPU arm is launched.** Thresholds were not
lowered and the candidate set was not extended.

### 8.2 The motivating inference was wrong, and the correct decomposition is the opposite

Section 2 inferred from the decoder tag shuffle that tags contribute about 16% and the midpoint
carries the rest. The measured decomposition is the reverse:

| Step (M4) | mean vs H-SE5 | positive | increment |
|---|---:|---:|---:|
| supervised basis, `delta` only, no tags (`ser_delta_q4`) | +0.005051 | 9/13 | - |
| add midpoint, still no tags (`ser_poscontext_q4`) | +0.004720 | 10/13 | **-0.000331** |
| add event tags (`ser_context_q4`) | +0.014438 | 13/13 | **+0.009719** |

**The midpoint adds nothing on its own. The event tag supplies 67% of Context Full's gain at M4
and 53% at M3.**

### 8.3 A methodological finding the paper must not get wrong

The sealed same-checkpoint within-trial tag shuffle costs only `0.002558` of Context Full's
`+0.016481` decoder gain, i.e. about 16%. The ablation measured here puts the tag's true
contribution at 67%. **That control understates tag dependence by roughly 4x.**

The reason is that a within-trial tag rotation preserves the tag distribution and is applied to a
checkpoint that was trained with correct tags, whereas removing tag columns changes the supervised
source basis itself. Anyone citing the tag-shuffle number as evidence that "tags contribute little"
would be wrong by a factor of four. This applies to the sealed Context Full record.

### 8.4 A third proxy statistic fails to predict forward transfer

Dropping tags **raised** retained energy at rank 4 (`0.5063 -> 0.5237`), confirming the
variance-dilution prediction about energy, yet forward transfer **fell**. Together with the
condition-number result (C2) and the retained-variance result (K-point screen), three separate
proxy statistics — design condition number, retained variance, retained energy — have now each
failed to predict forward transfer. Do not use any of them as a selection criterion.

### 8.5 Consequence

There is no tag-free path to Context-level performance on H1. Any H1 sparse carrier at
`~0.5165` must consume native event tags. The choice is therefore binary: either the paper carries
Context Full with its tag dependence, or the H1 sparse arm stays at H-SE5 `0.500037`.

Paradoxically this round **increases** Context Full's scientific credibility without touching the
sealed cell: the tag content is load-bearing at 67%, not decorative; its own control understated
that by 4x; and no simpler tag-free variant does the same job, so the design is not gratuitously
complex.

---

## 9. Interpretation limits fixed in advance

- A CPU pass is a forward-transfer estimator result. It is not a decoder R-squared result and must
  never be reported as one.
- The CPU-to-decoder relationship is not a law. Observed precedents: H-SE5 CPU `+0.0121` produced
  decoder `+0.0285`; Context Full CPU `+0.0144` produced decoder `+0.0165` over H-SE5. Any
  extrapolation is launch rationale only, never a forecast or an acceptance criterion.
- Nothing in this protocol licenses a claim that a tag-free carrier is superior to the dense H-C
  arm, which remains `0.525511` on the sealed fold-0 query.
