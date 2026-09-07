# M1 D4 semantic and identifiability audit protocol

**Frozen:** 2026-08-01 (Asia/Hong_Kong)  
**Status:** CPU audit authorized; decoder training is not authorized by this protocol.  
**Scope:** FALCON native-MUA M1 (`000941`) calibration files only.

## 1. Why this audit exists

`M1_OBJECT_DESCRIPTOR_PROPOSAL.md` proposes four per-channel means indexed by `obj_id` and
interprets them as an object-aligned descriptor, O4.  Direct inspection that triggered this
audit found that every official M1 held-out-calibration file contains only `tgt_obj=Sphere`
in its ten deployment trials, while the four `obj_id` levels are locked to four target
locations.  Therefore an O4 effect at `M=10` may be a non-parametric four-direction tuning
profile rather than independently identifiable object information.

This audit does not assume that either interpretation is correct.  It freezes the semantic,
predictive, and advancement tests needed to distinguish them before any O4/D4 decoder is
implemented.

## 2. Isolation and legality

- CPU only; no decoder training, GPU process, EvalAI submission, or hidden query.
- Read only from the seven M1 `held-in-calib` / `held-out-calib` NWB files and existing audit
  utilities.  Do not alter any NWB, checkpoint, scored result, or running clean-selection job.
- Write only a new immutable result tree under
  `sua_exploration/results/m1_d4_semantics_v1/`; refuse to overwrite it.
- Record SHA-256 for every NWB input and for the final JSON artifact.
- Use strict JSON: undefined estimands are explicit records with reasons, never NaN/Infinity.

## 3. Frozen questions and estimands

### 3.1 Calibration-design semantics

For every held-in and held-out calibration session and prefix `M in {10,40,90,200}`, report:

- level counts for `obj_id`, `tgt_obj`, `tgt_loc`, and `condition_id`;
- observed joint cells and their trial counts;
- whether `tgt_obj` is constant;
- whether `obj_id -> tgt_loc` and `tgt_loc -> obj_id` are deterministic;
- rank and condition of the relevant cosine and categorical designs.

Prefixes longer than a file are unavailable, not truncated silently.  The exact M=10 joint
tables for all three held-out-calibration sessions are mandatory.

### 3.2 Full-session factor audit

On each of the four held-in-calibration sessions, compute movement-window trial-mean EMG and
neural-rate estimands using the definitions already frozen by `audit_m1_t4_mechanism.py`.
Compare at least:

1. cosine direction `[1, cos(theta), sin(theta)]`;
2. categorical direction;
3. categorical `tgt_obj`;
4. categorical `obj_id`;
5. direction plus `tgt_obj`;
6. direction plus `obj_id`;
7. descriptive `condition_id` and selected interactions when rank permits.

Report both in-sample fit and chronological/block-held-out prediction.  Model comparisons must
state rank/degrees of freedom.  The primary evidence is held-out incremental predictive R2;
in-sample marginal variance is descriptive and cannot establish a dominant causal factor.

### 3.3 Deployment-support neural prediction

For each held-in session, fit per-channel trial-rate models on the chronological support prefix
and predict strictly later trials:

- T4 predictor: cosine direction basis;
- D4 predictor: categorical `obj_id` cell means.

Primary support is `M=10`; `M=40` is secondary.  Report pooled and per-session neural trial-rate
R2/error for:

- all later trials;
- later trials whose observed label/location combinations were represented in support.

This is a descriptor-identifiability diagnostic, not decoder R2 and not a hidden-test result.

## 4. Frozen decisions

### Gate S — object semantics

`object_factor_identifiable_at_deployment_m10=true` only if every held-out-calibration M=10
prefix contains at least two `tgt_obj` levels and does not make `obj_id` perfectly deterministic
with `tgt_loc`.  Otherwise O4 is rejected **as an object descriptor**, irrespective of any
downstream predictive score.

### Gate P — categorical-profile predictive value

D4 is eligible for one minimal decoder pilot only if, at M=10 on the four held-in sessions:

- mean later-trial neural-prediction `D4 - T4` is strictly positive; and
- at least three of four session deltas are strictly positive.

The all-later-trial endpoint is primary.  The support-combination-matched endpoint is diagnostic.
Undefined or rank-deficient evidence yields `indeterminate`, never a pass.

### Gate G — GPU authorization

This CPU protocol cannot itself launch a GPU job.  If Gate S fails but Gate P passes, the only
authorized interpretation is a **categorical calibration profile (D4)**, not an object descriptor.
The root reviewer must approve a separate frozen pilot protocol before any GPU work.

If Gate P fails, O4/D4 decoder work stops.  Larger support budgets cannot rescue the deployment
claim; they may be reported only as identifiability diagnostics.

## 5. Candidate register and convergence

The following candidates are retained in the written record so that rejected variants are not
quietly reopened after observing results:

1. raw O4 `[mu_obj1, ..., mu_obj4]` — reject if Gate S fails;
2. centered O4 — same semantic dependency as raw O4;
3. physical `tgt_obj` means — unavailable at M=10 if only one level is exposed;
4. 24-way `condition_id` means — unidentifiable at M=10;
5. raw D4 categorical profile — current minimal candidate;
6. D4 as `[baseline, three Helmert contrasts]` — invertible, better-conditioned form;
7. count-aware shrinkage D4 — permitted only after raw D4 passes Gate P;
8. confidence-augmented D4 — deferred; adds complexity before base identifiability;
9. second-harmonic/Fourier tuning — plausible geometry fix but wider than the minimal test;
10. spline tuning profile — too flexible for ten trials;
11. T4 plus residual-energy scalar — does not preserve the categorical tuning pattern;
12. T4/D4 learned mixture — invalid before a calibration-only selector is validated;
13. design-geometry selector — retained as the higher-level method if both task-specific bases work;
14. behavior-target-based factor selector — not deployment-legal without exposed target behavior;
15. condition-interaction descriptor — deferred until condition coverage is adequate.

The simplicity and failure-analysis filters select raw D4 for the first CPU comparison.  No FiLM,
attention, confidence MLP, or additional decoder pathway is justified before Gate P passes.

## 6. Required artifact contract

The final `audit.json` must include:

- schema version, generation command, environment, input paths and SHA-256;
- per-session prefix joint tables and design diagnostics;
- factor-model ranks, folds, fit and prediction metrics with undefined reasons;
- M=10 and M=40 T4/D4 later-trial prediction contrasts;
- Gate S, Gate P, and the resulting disposition;
- limitations and explicit prohibited claims.

Focused tests must bind the semantic gate, strict-JSON contract, support/query chronology,
decision logic, and artifact hash.  The implementation must pass `py_compile`, focused pytest,
and `git diff --check` before review.
