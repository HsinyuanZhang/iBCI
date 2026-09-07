# Frozen Protocol: Context Full Unseal — Multi-Date GPU Validation (CTXV2)

**Date predeclared:** 2026-08-12, before any GPU arm was launched
**Status:** predeclared. Phase 1 is CPU infrastructure only. No GPU arm may launch before the
Phase 1 preflight receipts exist and are independently verified.
**Scope:** the 13 public `sub-HumanPitt-held-in-calib` H1 recordings only.

---

## 1. Governance — explicit unseal decision

The sealed fold-0 Context Full cell carries predeclared status `STOP_CONTEXT`, which forbade
expansion. **The project owner has explicitly decided to unseal and run a fresh multi-date
validation.** This document records that decision and the protocol that replaces the old stop.

Rules that still bind:

- The sealed fold-0 receipts are **immutable inputs**. Nothing in this program rewrites, rescores,
  or reinterprets them.
- The old `STOP_CONTEXT` status remains the correct historical description of the fold-0 cell.
  CTXV2 does not retroactively convert it into a pass.
- CTXV2 is a **new lineage** with its own arms, its own gates, and its own terminal status.
- Every threshold in Section 5 is frozen now. If CTXV2 fails, it fails; thresholds must not be
  relaxed afterwards and arms must not be added to manufacture a pass.

## 2. Why this is worth GPU time

The tag-free screen (`H1_TAGFREE_POSITION_CONTEXT_CARRIER_PROTOCOL_20260812.md`, terminal
`STOP_CPU_PCTX_NOT_MATERIAL`) established:

- there is **no tag-free path** to Context-level performance; the best tag-free candidate reached
  `+0.004720` against a `+0.011550` requirement;
- the event tag supplies **67%** of Context Full's gain at M4 and 53% at M3;
- the sealed same-checkpoint within-trial tag shuffle **understates tag dependence by about 4x**.

So Context Full's content is genuinely load-bearing and irreplaceable by a simpler variant. What it
lacks is not mechanism evidence but **generalisation and control rigour**. That is exactly what GPU
time can buy.

## 3. The three weaknesses CTXV2 must fix

| # | Weakness of the sealed fold-0 cell | CTXV2 remedy |
|---|---|---|
| 1 | Single fold (date `19250101`, 2 recordings) | Add outer date `19250108`, which has **3** target recordings |
| 2 | Corruption controls were same-checkpoint only | Add **separately trained** Context-LS and Context-RS arms |
| 3 | Recording-uniformity clause failed and was discovered after the fact | Declare the uniformity clause up front, now, with its failure semantics |

Second date choice: `19250108` is selected because it carries three target recordings, making the
uniformity clause a strictly harder 3/3 test than fold-0's 2/2. It is an established outer date in
the five-date LODO programme.

## 4. Arms — frozen

Five new GPU runs. Each is 50 fixed epochs, seed 42, FP32, batch 32, no validation-based
checkpoint selection, identical topology across arms.

| # | Arm | Outer date | Purpose |
|---|---|---|---|
| 1 | Context Full | 19250108 | candidate on a new date |
| 2 | H-SE5 Full | 19250108 | the endpoint-only reference for the Context-minus-H-SE5 clause |
| 3 | Zero5 | 19250108 | independently trained no-carrier null |
| 4 | Context-LS, separately trained | 19250101 | label-content control at full rigour |
| 5 | Context-RS, separately trained | 19250101 | channel-attachment control at full rigour |

One Zero5 per date is sufficient and must be justified by a bound parity preflight proving
identical source windows, trial identities, batch order, calibration schedule, topology, optimiser,
seed, initial state, and carrier shape, with `zero_carrier=True` replacing the carrier by a literal
zero tensor at the model boundary. This mirrors the audited fold-0 parity argument.

Measured cost: about 2.1 hours and 126 MB per run. Two GPUs are free, so three waves, roughly
6.3 hours wall clock, about 630 MB of the 79 GB available.

## 5. Predeclared outcome clauses — frozen

Let `Context` be arm 1 and `HSE5` be arm 2, scored on the identical strict post-support query for
date `19250108`.

### 5.1 Primary clause — generalisation
`Context - HSE5 >= +0.010` pooled on date `19250108`, **and positive on all 3 target recordings.**

### 5.2 Secondary clause — content and attachment, at full rigour
On fold-0, `Context Full` must exceed **separately trained** Context-LS and Context-RS. These are
stronger controls than the sealed same-checkpoint versions and may give smaller margins; both
margins must nevertheless be positive.

### 5.3 Tertiary clause — null separation
On date `19250108`, `Context - Zero5 > 0` pooled and on all 3 recordings.

### 5.4 Terminal status rules, fixed now

- All three clauses pass -> `PASS_CTXV2_MULTIDATE` . Context Full becomes a two-date,
  full-rigour result and is eligible for the paper subject to the tag prohibition being revisited
  separately.
- Primary clause passes pooled but fails recording uniformity -> `STOP_CTXV2_NONUNIFORM`. This is
  the same failure mode as fold-0 and would then be observed twice, which is itself a reportable
  and important finding: it would indicate the effect is real but not recording-uniform.
- Primary clause fails pooled -> `STOP_CTXV2_NOT_REPLICATED`. The fold-0 gain would not have
  generalised, and Context Full must not enter the paper.
- Any required control inverts -> `STOP_CTXV2_CONTROL_FAILURE`.

No proportional extrapolation from fold-0 is a forecast or an acceptance criterion.

## 5.5 Staging amendment — recorded 2026-08-12, before any GPU launch

A code-feasibility review after Section 4 was written established that the carrier corruption
machinery lives in `H1ContextStrictTargetDataset.INTERVENTIONS`
(`full`, `zero`, `row`, `label`, `tag`, `midpoint`) and is applied to the **target** carrier at
**evaluation** time on a single checkpoint. **No source-side carrier corruption exists**, so a
*separately trained* LS or RS arm requires new source-path code. Separately, the target dataset
pins `H1_M4_FOLD0_TARGET`, so a second date also requires parameterising the target side.

The programme is therefore split into two stages. **No gate, threshold, or outcome clause in
Section 5 is changed by this amendment.** Only the order and the build cost change.

- **Stage A — arms 1, 2, 3 on date `19250108`.** Answers the make-or-break primary clause
  (Section 5.1) and the tertiary clause (Section 5.3). Requires a date-parameterised snapshot
  builder and a date-parameterised target dataset, but **no** source-side corruption.
- **Stage B — arms 4 and 5 on fold-0.** Answers the secondary clause (Section 5.2). Requires new
  source-side carrier corruption. **Built only if Stage A passes**, because control rigour on a
  result that did not replicate would be wasted effort.

If Stage A fails, the terminal status is decided by Section 5.4 on Stage A evidence alone and
Stage B is never built.

## 5.6 Feasibility findings from Phase 1 — the staging in 5.5 is INVERTED

Phase 1 was executed on CPU and established that Section 5.5 had the relative costs backwards.

**Stage A (second date) is expensive.** The entire context data path is pinned to fold-0:

| Sealed file | Pin |
|---|---|
| `h1_context_event_source_snapshot.py:45` | binds `basis_by_candidate_and_outer_date["ser_context_q4"]["19250101"]` |
| `h1_context_event_source_snapshot.py:62,81` | hardcodes `normalizer_shape [116,176,5]`, `source_carriers (116,176,5)`, `active_mask (22,)`, `projection (20,4)` |
| `h1_context_event_carrier.py:54,106-108,137-143` | source assets pinned to `H1_M4_FOLD0_SOURCE`, `FOLD0_DATE` |
| `h1_context_event_carrier.py:249,251,291` | target dataset pinned to `H1_M4_FOLD0_TARGET` |

Date `19250108` has **109** source supports, not 116, so the snapshot shape validation fails closed
by construction. Running a second date therefore requires **forking the whole sealed context data
pipeline**, not parameterising one script. A fork also carries divergence risk that would
undermine the very comparability the multi-date test exists to establish.

**Mandatory gate if Stage A proceeds:** the forked pipeline must first reproduce fold-0
**bit-identically on CPU** — same source manifest SHA, same `window_indices_sha256`, same
per-intervention `carrier_sha256` — before it is trusted on any new date. Phase 1 already
demonstrated part of this: the dated target dataset reproduces fold-0's target order, window count,
`window_indices_sha256`, and every intervention `carrier_sha256`.

**Stage B (separately trained LS/RS on fold-0) is cheap.** It stays on fold-0, so every hardcoded
constant in the sealed snapshot module is satisfied: same map, 116 supports, `(22,)` active mask,
`(20,4)` projection. Only the source carrier cache contents differ.
`build_context_source_assets` returns the cache as a plain object, and
`H1ContextSourceDataset.__getitem__` reads carriers from it, so a deterministic corruption can be
applied by a thin wrapper using the existing `event_v2.row_shuffle` and
`event_v1.within_trial_label_shuffle`. No fork is required.

Revised cost:

| Stage | New code | GPU runs | Wall clock |
|---|---|---:|---:|
| B — separately trained LS/RS, fold-0 | thin cache-corruption wrapper + 2 configs | 2 | ~4.2 h |
| A — second date `19250108` | fork of source + target + snapshot modules, plus a fold-0 bit-identical reproduction gate | 3 | ~6.3 h plus build |

Scientific priority is unchanged: Stage A tests generalisation and is the more important clause;
Stage B tests control rigour. The gates in Section 5 remain frozen and unmodified.

### Phase 1 measured quantities for date `19250108`

| Quantity | 19250108 | fold-0 |
|---|---:|---:|
| target recordings | 3 | 2 |
| source sessions | 10 | 11 |
| source supports | 109 | 116 |
| batches per epoch | 3,356 | 3,610 |
| query windows | 13,107 | 8,965 |
| `window_indices_sha256` | `b0cd1537...da7f6e` | - |

The date-2 schedule differs from fold-0 by construction. This is acceptable because every declared
contrast is **within** a date, but no cross-date comparison of absolute R-squared is licensed.

### Outstanding defect to fix before launch

The Phase 1 preflight receipt was written as
`sua_exploration/results/ctxv2_stage_a_preflight/CTXV2_STAGE_A_PREFLIGHT_19250108.json` while the
launcher requires `CTXV2_STAGE_A_PREFLIGHT_v1.json`. The launcher would refuse to start. Reconcile
the path before any launch.

## 6. Phase 1 — CPU infrastructure, required before any GPU launch

The existing Context path is hardcoded to fold-0 in at least two places and must be generalised
**without editing any sealed file**:

- `SPINT-main/scripts/build_h1_context_event_source_snapshot.py` pins `outer_date="19250101"` and
  pins that date's `FIXED_MAP_SHA` and `FIXED_ARRAYS`.
- `SPINT-main/configs/experiment/h1_context_event_carrier_{full,zero}.yaml` pin
  `fold_date: "19250101"`.

Crucially, the sealed design-screen receipt already contains the leave-one-date-out
`ser_context_q4` map for **every** date under
`basis_by_candidate_and_outer_date["ser_context_q4"][date]`. Therefore no new basis fitting is
required: a date-parameterised builder must **bind** the sealed map for the requested date and fail
closed on any drift, exactly as the fold-0 builder does for its own date.

Phase 1 deliverables:

1. A date-parameterised snapshot builder (new file) that binds the sealed receipt map for the
   requested outer date and fails closed on any SHA drift.
2. New experiment configs for all five arms.
3. A parity preflight receipt proving arm-to-arm equivalence on date `19250108`.
4. An independent verifier for the preflight.
5. Focused tests, including a leave-one-date-out no-leakage test proving the date-`19250108`
   source pool excludes all three date-`19250108` recordings.

## 7. Phase 2 — GPU execution

Launch only after Phase 1 receipts exist and are independently verified. Three waves across the two
RTX 3090s. A watcher records terminal checkpoints and runs the strict evaluator only after both
training processes for a wave have exited. Evaluation must use byte-identical query windows across
arms within a date, with the query SHA recorded.

## 8. Phase 3 — evaluation and analysis

Required evidence per date: pooled R-squared per arm, per-recording R-squared per arm, the three
clause evaluations, the query window SHA, and an independent recomputation of the headline
contrasts by a separate prediction implementation.

## 9. Interpretation limits fixed in advance

- CTXV2 evaluates development folds. It opens no minival, formal held-out, EvalAI, or
  organizer-hidden endpoint.
- A pass makes Context Full a two-date result. It does **not** make it an organizer-hidden result,
  and it does not license any comparison to the dense H-C arm beyond the descriptive gap already
  recorded (`0.525511` on the sealed fold-0 query).
- The paper's event-tag prohibition is a separate editorial decision. A CTXV2 pass supplies
  evidence relevant to that decision but does not itself overturn it.
