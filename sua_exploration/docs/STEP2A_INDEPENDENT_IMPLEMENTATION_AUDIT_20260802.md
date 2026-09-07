# Step-2A independent implementation audit — 2026-08-02

**Scope.** Read-only review of the isolated cross-budget preparation files:
`mc_maze/t4_cross_budget_protocol.py`, `mc_maze/t4_cross_budget_features.py`,
`scripts/audit_sua_t4_cross_budget.py`, their focused tests, and the Step-2/3 preparation
document.  No formal SUA test was opened, no GPU was used, and no active residual or shared
training/datamodule file was modified.

**Initial verdict:** block, pending the four implementation corrections below.

**Remediation review (same date): APPROVE source-only CPU-audit *readiness*, not a data audit or
a GPU pilot.**  The corrections now have synthetic/readiness coverage (`11 passed`), the cache
namespace/semantics are bumped to v2, and the command remains dry-run/manifest-validation only.
No actual train/validation NWB has been opened under this revision; formal tests and GPUs remain
out of scope.

## What is already sound

- The feature builder calls `list_datamodule_rewarded_trials()` rather than recreating the
  reward/duration filter, and slices the resulting chronological list before rate extraction.
- It reuses the ordinary per-direction-mean cosine helper.  It does not synthesize coefficient
  noise.
- `rank != 3` is represented by NaN descriptors/reliability and `fit_defined=False`, so an
  undefined low-budget prefix cannot silently become a finite zero carrier.
- The cache uses a new namespace and includes budget vector, signal view, bin/window, reward
  filter, source fingerprint, and fit-semantics version in its *key*.  It does not register a
  feature token or touch the active residual path.
- The current executable entrypoint only has `--dry-run`; it neither opens data nor enables CUDA.
  Focused tests passed: `5 passed in 2.03s` under the `spint` environment.

## Initial blocking corrections and remediation

1. **Missing target directions contaminate reliability fields.**  The T4 fit itself correctly
   ignores direction index `-1` through `_unit_tuning_features`, but the residual design in
   `fit_t4_prefix()` constructs `CANONICAL_DIRECTIONS_RAD[int(index)]` for *every* prefix row.
   Python maps `-1` to canonical direction 7.  Thus an unlabeled trial is spuriously evaluated as
   direction 7 in residual variance; the denominator also uses `budget - 3` rather than
   `n_valid_labelled - 3`.  This can change the proposed `q` predictor while leaving T4 unchanged.
   Fix by applying the residual design and degrees of freedom only to `direction_indices >= 0`.
   State and test explicitly whether exposure is labelled-fit exposure (recommended) or the
   separately named total first-M causal exposure.  Add a regression test with an extreme-rate
   `-1` direction row: it must leave every labelled-fit reliability coordinate unchanged.

2. **The cache cannot satisfy the mandatory chronology receipt.**  The bundle persists only
   descriptors and summary counts.  It does not store each selected trial's chronological ordinal,
   start/stop time, target direction/direction index, or the complete source/cache payload.
   Therefore a later audit cannot prove the exact prefix nesting described in Step 2.4, and a
   cache file copied into the expected path cannot be semantically checked beyond namespace/
   version.  Persist a compact first-50 trial receipt (not raw rates): ordinal, start, stop,
   finite/raw direction plus snapped index, selection settings, source fingerprint, and full
   canonical payload.  On load, recompute expected payload/key and fail closed if it differs.
   Include a test that a mismatched cached payload/key is rejected.

3. **Thinning is only a pure helper, not an executable estimator-stability procedure.**  The
   cache deliberately omits raw M-by-T arrays, and the NWB boundary currently returns rates only.
   No implemented source-only audit can recover integer trial spike counts for repeated thinning,
   form independent/matched split replicates, calculate repeatability or `T@M -> T@50` errors,
   or produce paired uncertainty/MDE.  This is fine for readiness code, but not for the required
   CPU audit.  Before opening source data, pre-register an isolated count-extraction/audit path
   that operates on the same first-M trial receipt, keeps raw counts in RAM only, and writes only
   summaries.  Specify a valid replicate construction (for example a multinomial partition of
   each trial's counts into two disjoint halves, or independent thinnings with the dependence
   correctly accounted for), the number/seeds of replicates, undefined-fit handling, session-level
   aggregation, and the MDE estimator.  Do not call one thinned realization a repeatability
   statistic.

4. **The requested source-only predictive audit is not implemented.**  The present command is
   intentionally dry-run-only.  It has no reviewed train/validation manifest, no nested source
   folds, no `q@M -> T@50` constant/M baseline, no metric/uncertainty receipt, no deterministic
   row-shuffle contract, and no target-free validation application.  Those are required before a
   CPU decision or any GPU authorization; they should remain a new isolated script rather than a
   hidden expansion of the current dry-run entrypoint.

### Remediation receipt

All four initial blockers are resolved at **implementation/readiness** level, not by running
data:

1. residual rows and degrees of freedom now use only valid labelled directions and
   `n_labelled - 3`; the unambiguous 7-D reliability vector records labelled/total spike counts
   and labelled/total time exposure separately.  An extreme-rate unlabeled-trial regression test
   confirms that only the explicitly total-prefix fields change.
2. cache namespace and semantics are now v2.  It writes the compact first-50 trial receipt
   (ordinal, start/stop, raw and snapped direction) and full canonical payload; expected payload
   mismatch fails closed on load.
3. the new isolated `t4_cross_budget_audit.py` can extract only a receipted source prefix's
   integer counts into RAM, uses eight fixed-seed disjoint two-way multinomial partitions with
   rate rescaling, reports undefined ranks, and returns summary-only repeatability/MDE inputs.
4. the same pure module validates the frozen development manifest without resolving NWB paths,
   implements nested source-LOSO `q@M -> T4@50` descriptor-error prediction versus constant and
   M-only controls, and provides deterministic nonidentity complete-row shuffle.  Its left-out
   application is explicitly target-free; `T4@50` is offline scoring only.

Focused synthetic/readiness test result after remediation: **11 passed**.  The command supports
only `--dry-run` and manifest-only validation; it has not opened real source/validation data.

### Guarded execution extension

The runner now implements, but has **not invoked**, the real source-only CPU path.  Invocation
requires both `--execute-source-audit` and `STEP2A_REVIEWED_SOURCE_AUDIT=YES`, an explicit frozen
manifest, data/cache roots and a previously nonexistent result directory.  It resolves exactly
the manifest's 27 train and six development-validation names, never glob-discovers sessions and
never resolves the six formal-test names.  It writes strict JSON with no raw counts; cache/input
hashes, per-session first-50 chronology, rank/condition, descriptor error, eight-seed
repeatability, nested source-LOSO and target-free validation summaries are included.  Complete-row
shuffle is recorded only as an attachment diagnostic, not evaluated as a decoder arm.

## Non-blocking clarifications to lock before the real audit

- The code's rank-three gate is stricter than ordinary legacy `t4`, which can emit an
  underdetermined least-squares vector for two observed directions.  This stricter gate is the
  right Step-2A policy, but the wording "reproduces existing ... T4 semantics" should say
  "reuses the fit semantics *conditional on an auditable rank-three gate*".
- `design_condition` is reported but no condition threshold is pre-registered.  Report it
  continuously first; do not retrospectively convert it to an eligibility cutoff after looking at
  decoder scores.
- The `reliability[..., 6]` field is a redundant all-ones rank-valid flag for every defined
  session.  Retaining it is harmless only if the planned learner receives the session-level rank
  field separately; otherwise it adds no per-unit information.
- The document correctly keeps evaluation start at trial 50 and fixes activity support at 30.  A
  future runner must receipt both boundaries and ensure no windows originating in trials `<50`
  enter its query endpoint.

## Approval condition

The source-only CPU audit now has implementation/readiness approval: the cache is verifiable and
the manifest validator does not resolve formal-test paths.  Executing it on real development NWBs
still needs a separate authorization and must produce chronology, identifiability, repeatability,
q-predictability and paired-precision receipts.  This does not approve a GPU pilot; a pilot stays
conditional on the predeclared out-of-fold reliability/error and precision gates.
