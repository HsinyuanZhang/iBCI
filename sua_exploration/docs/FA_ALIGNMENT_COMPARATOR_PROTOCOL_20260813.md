# Frozen Protocol: FA-Based Alignment Comparator (FAA)

**Date predeclared:** 2026-08-13, before any arm was executed
**Status:** CPU-only. Skeleton build first, gated behind a constructibility audit.
**Scope:** subject-M (DANDI 000688, SUA and pseudo-MUA views) and RT. **Not** H1 or M2 — those are
already covered by citable FALCON Table 1 baselines.

---

## 1. Why this comparator exists

Unsupervised subspace alignment is the closest competing family to our method: it adapts to a new
session using **unlabelled** target data, where we use **sparse labelled** target data. It is the one
comparator family we lack on subject-M and RT, because FALCON Table 1 only covers H1 and M2.

**Why FA stabilization rather than NoMAD.** The PRI-T study (Nature BioMed Eng 2025) reports that
*"PRI-T, FA stabilization and ADAN all perform equivalently when applied offline to pairs of days."*
That published equivalence lets us implement the simplest member — factor analysis plus a linear
subspace alignment, with no dynamics model and no adversarial training — and defend it as
representative of the family. Reimplementing NoMAD faithfully carries far more risk for the same
comparison point.

**Prior expectation, from FALCON Table 1.** On H1, `NoMAD + WF` scores 0.13 and `CycleGAN + WF` 0.12,
both *below* zero-shot WF at 0.16. On M2 they score 0.20 and 0.22, well *above* zero-shot WF at 0.06.
Alignment is therefore strongly dataset-dependent, and we should not assume either outcome on
subject-M or RT.

---

## 2. Correction to an earlier claim

An earlier note suggested FA alignment might be structurally undefined on subject-M because the
source-pooled ridge audit found **zero correspondent channels** in both views.

**That reasoning applies to only one of two variants.** Alignment methods match *latent spaces*, not
channel indices, so a distribution-alignment variant needs no channel correspondence at all. This is
precisely what distinguishes alignment from a direct readout, and it must not be prejudged.

The audit in Part A therefore tests both variants separately.

---

## 3. Part A — constructibility audit, run FIRST

For each dataset and view, measure and report per session, **without fitting any alignment**:

1. **Channel correspondence** — reuse the finding already established for subject-M (zero
   correspondent channels in both views); confirm independently and measure it for RT.
2. **Latent fit feasibility** — can a factor analysis model be fitted on the target session's
   unlabelled calibration block? Report channel count, samples, and the samples-per-parameter ratio.
3. **Latent dimensionality stability** — the shared dimensionality `d` must be fixed across sessions
   for an alignment to be defined. Report the per-session variance explained curve and whether a
   single `d` is defensible for all sessions.
4. **Alignment estimability** — can an alignment transform be estimated from unlabelled target
   statistics alone?

Emit a verdict per variant:

| Variant | Requires channel correspondence | Expected verdict |
|---|---|---|
| **A1 channel-anchored** | yes | likely `FAA_UNDEFINED_NO_CHANNEL_CORRESPONDENCE` on subject-M |
| **A2 latent-distribution alignment** | no | expected definable |

If A1 is undefined, that is a **finding**, not a failure: it would be the second independent method
family blocked by the same structural fact, after the source-pooled ridge. Record it as such and do
not manufacture a correspondence.

---

## 4. Part B — the arms, only where Part A says definable

### Design, following FALCON's own convention
FALCON evaluates alignment as `NoMAD + WF` and `CycleGAN + WF`: an unsupervised alignment front-end
feeding a linear readout. Mirror that exactly, so our comparator sits in the same class:

```
source:  fit FA on pooled source sessions      -> latent space L_src, dimensionality d
source:  fit a linear readout on source latents using SOURCE labels
target:  fit FA on the target calibration block using UNLABELLED activity only
target:  estimate an alignment T mapping L_tgt onto L_src from latent statistics only
deploy:  y_hat = readout( T( FA_tgt(x) ) )
```

**The alignment must consume zero target labels.** That is the defining property of this family and
the axis on which it differs from us. Any use of a target label invalidates the arm.

### Arms
- `faa_a2_align` — latent-distribution alignment, the primary arm.
- `faa_a1_anchored` — channel-anchored alignment, only if Part A says definable.
- `faa_no_align` — the same source readout applied with no alignment. **Required.** This is the
  zero-shot floor and isolates what the alignment itself contributes.
- Sealed references, not re-run: subject-M carrier 0.3568 / 0.3061, dense ridge 0.4179 / 0.4102;
  RT T4d 0.448176, ridge 0.200202, Zero4 0.179272.

### Fixed conventions
Same calibration budgets, query boundaries and **sealed query identities** as the existing
comparator arms for each dataset, with query identity hashes recorded. Same metric per dataset —
note RT uses variance-weighted sum-over-dimensions R-squared. Latent dimensionality `d` and any
regularisation selected on **source sessions only**; no target query window may influence anything.
Per-session fitting; no transfer of target-fitted parameters across sessions.

---

## 5. What each outcome licenses — fixed in advance

Primary contrast per dataset and view: `carrier − faa_a2_align`, reported as mean, median, per-session
sign count, and a paired bootstrap interval. Also report `faa_a2_align − faa_no_align`, which
isolates the alignment's own contribution.

- **Carrier exceeds alignment.** The paper may state that on these datasets a sparse labelled
  calibration outperforms an unsupervised alignment of the same readout family, consistent with what
  FALCON Table 1 shows on H1. Remains a system-level statement: the architectures differ.
- **Alignment matches or exceeds the carrier.** Report it plainly. The supervision-cost axis still
  differs — alignment uses zero target labels, we use sparse ones — and that trade-off becomes the
  honest framing rather than an accuracy claim.
- **`faa_a2_align − faa_no_align` is near zero.** The alignment contributes nothing on this dataset
  and the arm is a floor measurement, not a competitive baseline. Say so.
- **A1 undefined.** Report as a structural finding per Section 3.

Do not tune the alignment on datasets where tuning helps and leave it fixed elsewhere. Whatever
selection rule is used must be applied identically across subject-M and RT.

---

## 5.5 PART A RESULTS — executed 2026-08-13

Skeleton built, 9/9 unit tests pass, Part A audit run on all three targets. **No scoring arm was
executed.** Receipts under `sua_exploration/results/fa_alignment_comparator/`.

| Dataset | A1 channel-anchored | A2 latent-distribution |
|---|---|---|
| subject-M SUA | `FAA_UNDEFINED_NO_CHANNEL_CORRESPONDENCE` | **`FAA_DEFINABLE`** |
| subject-M pseudo-MUA | `FAA_UNDEFINED_NO_CHANNEL_CORRESPONDENCE` | **`FAA_DEFINABLE`** |
| RT | `FAA_UNDEFINED_NO_CHANNEL_CORRESPONDENCE` | **`FAA_DEFINABLE`** |

The Section 2 correction is confirmed: A2 is definable everywhere despite zero correspondent
channels, because it aligns latent spaces rather than channel indices. **The comparator is runnable.**

### 5.5.1 The "zero correspondence" figure is definition-dependent — a claim is qualified here

The strict rule counts a correspondence only when identifier **sequences** are identical and widths
match. Under that rule all three return zero. The underlying overlap differs greatly:

| Dataset | identifier type | set intersection | pairwise overlap | channels/session |
|---|---|---:|---:|---:|
| subject-M SUA | positional `0..n-1` only | 25 | 25-90 | 25-92 |
| subject-M pseudo-MUA | electrode ids | **2** of union 129 | 3-40 | 19-66 |
| RT | per-unit NWB electrode ids | 20 | **23-47** | 57-88 |

**This qualifies an earlier statement.** It was previously argued that zero correspondent channels
means a direct readout structurally cannot exploit source data, and that this is a mechanism
argument in the paper's favour. That holds for **subject-M pseudo-MUA**, where the intersection is 2
channels out of a union of 129 — genuinely nothing to anchor on. It does **not** hold for **RT**,
where 23-47 electrodes genuinely persist between session pairs; a partial-overlap anchored method is
constructible there and the zero is an artifact of requiring identical sequences and widths.

For subject-M **SUA** the identifiers are positional only, so the question is unanswerable from
identifiers alone: we cannot tell whether sorted units correspond physically.

**Consequence:** the structural argument may be made for subject-M pseudo-MUA, must be stated
carefully for SUA, and must **not** be made for RT.

### 5.5.2 A shared latent dimensionality is not defensible on any dataset

`d80` is the smallest `d` reaching 80% cumulative variance explained.

| Dataset | `d80` range | spread | feasible shared grid value |
|---|---:|---:|---:|
| subject-M SUA | 13-47 | 34 | 16 |
| subject-M pseudo-MUA | 10-31 | 21 | 16 |
| RT | 15-45 | 30 | 8 (`d=16` infeasible on 3 sessions) |

Every dataset fails the predeclared spread criterion. A2 remains `FAA_DEFINABLE` because a grid
value is feasible on every session, but forcing a shared `d` means some sessions are heavily
under-modelled and others over-modelled, and alignment quality will vary accordingly. **Report this
alongside any FAA number**; it is a real limitation of the comparator, not a tuning detail.

### 5.5.3 Scope-rule false positive, third occurrence

subject-M sealed query identities could **not** be bound. The implementation raises
`NotImplementedError` rather than guessing, which is correct behaviour. The cause is that the sealed
V9 query lives under
`sua_exploration/results/dandi_000688_subm_co_three_arm_v9_formal_20260805`, whose path contains the
token `formal`, which a blanket scope rule forbids.

That block is a **false positive**. DANDI 000688 is not a FALCON dataset and has no organizer-held
evaluation endpoint; `formal` here is our own results-directory naming, and this is the very query
set our own sealed subject-M arms use.

This is the third false block from the same over-broad rule, after FALCON M2 `held-out-calib` and
the H1 Kalman adapter. **The corrected rule is in `HANDOFF_COMPARATORS_20260812.md` Section 7.5.1
and is per dataset:** forbid only paths that are genuinely private evaluation data for that dataset,
and always allow the split the sealed arms already consume. RT query identities bound correctly
against the sealed Stage-2 triple on all 15 folds.

### 5.5.4 Build status

`faa_a2_align` and `faa_no_align` are built and runnable on all three targets once subject-M query
binding is unblocked. `faa_a1_anchored` is built but correctly refuses to run wherever A1 is
undefined. The zero-target-label guard is implemented and unit-tested.

---

## 5.6 FIRST SCORING RUN 2026-08-13 — ALL NUMBERS VOID, DO NOT CITE

Three arms were executed on RT and H1. **Every number produced is void.** The run was useful as
debugging, not as measurement. No result from it may enter the paper or any comparison table.

| Arm | Result | Status |
|---|---:|---|
| FA `faa_a2_align`, RT | mean −0.001229 | **VOID** |
| FA `faa_no_align`, RT | mean −0.001818 | **VOID** |
| Kalman `velocity_only`, RT | mean −1.019 | **VOID** |
| Kalman `position_velocity`, RT | mean −1.052 | **VOID** |
| Kalman, H1, both | no score, filter failed at query t=0 | **VOID** |

Sealed references for scale: RT T4d `0.448176`, RT target-fit ridge `0.200202`.

### 5.6.1 Root cause, FA arms

The receipt records the source construction as *"per-session FA on each source calibration block,
then concatenate latents."*

**Factor analysis is fitted separately on each of 14 source sessions, producing 14 latent spaces
each with its own arbitrary rotation, sign and scale. Those latents are then concatenated and a
single readout is fitted across the pile.** The readout is therefore trained on 14 mutually
inconsistent coordinate systems, which yields approximately zero predictive power by construction.

This also explains why `faa_a2_align` and `faa_no_align` are *both* at zero: they share the same
broken source readout, so the contrast between them measures nothing.

The correct construction aligns all source sessions into a **common** latent space first — using the
same alignment machinery already implemented — and only then fits the readout. The observation that
observation-space pooling is impossible with differing channel counts is correct; the error was
skipping alignment on the source side rather than applying it there too.

### 5.6.2 Contributing problem: the latent dimensionality is far too small

Scoring used `d = 8`, the largest grid value feasible on every RT session. Section 5.5.2 measured
RT's `d80` at **15-45**. The latent space therefore captures well under 80% of variance on every
session, and on some sessions a small fraction. Even after the source-alignment fix, `d = 8` is
likely to under-model RT badly. The feasibility grid `{4, 8, 16}` needs revisiting against the
measured `d80`, not chosen for convenience.

### 5.6.3 Root cause, Kalman arms

A per-session Kalman filter fitted on the target's own calibration block should land in the
neighbourhood of the target-fit ridge (`0.200` on RT), not at `−1.0`. It is numerically broken:

- raw `W` is **infinite on every RT and H1 session** before the regularisation floor, from the
  constant-state row, so the floor is load-bearing on 15/15 sessions rather than being a safeguard;
- after the floor, RT `position_velocity` keeps `cond(W) ~ 1e17-1e18` — this is the synthesised-position
  failure mode predicted before the run, now confirmed;
- RT `velocity_only` is less broken (`cond(W) ~ 1e6-1e8`) but still far from sound;
- H1 failed outright at the first query update with innovation `cond(S) ~ 1e14-1e15`.

The constant term should be handled outside the stochastic state rather than as a state row with
zero process noise, which is what forces the infinite condition number.

### 5.6.4 Missing diagnostic that would have caught this immediately

Neither receipt records **any source-side fit quality** — no readout training score, no variance
explained. With a source-side R2 recorded, a broken readout would have been obvious before any
target number was produced.

**Requirement for the next attempt:** every arm must report its source-side fit quality first, and
a target score may not be interpreted unless the source fit is sound. Add this to the integrity
gate alongside the existing query-identity check.

### 5.6.5 Process note

Five changes were made to existing modules during execution, including wiring an FA scoring path
that the runner had previously refused. Writing new code paths inside an execution run is how
untested code produces confident-looking numbers. Build and test first, then run.

---

## 6. Interpretation limits

- FA stabilization stands in for the alignment family on the strength of a published equivalence
  finding, not because it is the strongest member. State this whenever the arm is cited.
- This is an offline, pairwise recalibration. PRI-T's central result is that all such methods
  degrade over long inter-session gaps and require chaining; nothing here tests that regime.
- Our carrier consumes sparse target labels and the alignment consumes none. Accuracy and
  supervision cost must be reported side by side and never collapsed.
