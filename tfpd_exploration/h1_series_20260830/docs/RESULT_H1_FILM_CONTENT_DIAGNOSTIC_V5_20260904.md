# H1 FiLM Content Diagnostic V5 — Result

Date: 2026-09-04
Status: `COMPLETE_BRANCH_A_BUDGET_ADAPTATION`

Preregistration: `WORKORDER_H1_FILM_CONTENT_DIAGNOSTIC_V5_20260904.md`
(same directory).  Result root:
`results/h1_calibration_profile_film_content_diag_v1/` (terminal.json 0444 +
sidecar).  Code: `src/h1_calibration_profile_film_content_diag_v1/` +
`scripts/run_h1_film_content_diag_v1.py`, run on GPU1 (GPU0 was occupied by
another agent's dandi688 jobs; both GPUs are RTX 3090).

## Decision

**Branch A — the H1 FiLM gain is M3-budget adaptation, not calibration-profile
content.  C3 is permanently cancelled** under the preregistered decision law.

## Grid results (EP-FILM-minus-EP-ZERO, sealed V2 LODO runner re-executed)

| Arm (mode) | Per-date gain | Mean | Gate |
|---|---|---:|---|
| full (drift check) | +.0219 +.0640 +.0163 −.0080 +.0254 | **+0.0239** | PASS (4/5) |
| **empty** (`carrier4 ‖ 0`) | +.0194 +.0544 +.0164 +.0161 +.0209 | **+0.0254** | **PASS (5/5)** |
| rowshuffle (`carrier4 ‖ P(profile)`) | +.0228 +.0589 +.0088 +.0104 +.0198 | **+0.0241** | PASS (5/5) |

Sealed V2 reference: EP-FILM mean `+0.023929…` (4/5).  Zeroing the entire
profile input **does not reduce the gain — it slightly increases it**
(empty/full ratio `1.063`; empty is 5/5 dates nonnegative where full is 4/5,
and empty's worst date is `+0.0161` where full's is `−0.0080`).  Row-shuffling
the profile across units likewise leaves the gain intact.  The 648-parameter
FiLM acquires nothing from the speed-contrast profile content: its entire
+0.024 is the zero-init adapter adapting a non-M3-aware frozen substrate to
per-block conditioning.

## Integrity canaries (all PASS)

1. **Bit-exact trajectory reproduction**: the `full` run's first-fold final
   film states equal the sealed V2 values exactly — EP
   `60eb1161d9621b0830d226e0eaf07aefb0ee1f56eb72d42880e95aa518bc6b58`,
   LP `fba420f403b248ee0a6fe6c1898f704dc6ae4be0be03d3a92324771afe979736` —
   so the re-executed runner is the same computation as V2, bit for bit.
2. **Score drift**: full mean vs sealed `|Δ| = 2e-6` (gate ±0.005).
3. **LP canary**: every mode-run's LP arm (untouched by the mode patch)
   reproduces the sealed LP gains within the preregistered tolerances.
4. **Init binding**: the train-time `build_film` init equals the sealed
   `ff2263b5705a499caeb5fe807f9889479fdbc31f806b42de82b11d351ba210cf` once per
   fold in every run (the scratch builds inside `_load_film` are recorded in
   the receipts and unconstrained by design).

## Consequences (preregistered in WORKORDER V5 §5, Branch A)

1. **C3 (CAL-AUG C2 recipe + FiLM joint training) is permanently cancelled.**
   The frozen C3 documents are not rewritten; this receipt governs.  The
   mechanism is now measured: on an already-M3-aware substrate (C2 recipe,
   like LP-R3 before it), FiLM has nothing left to adapt — exactly the
   LP-R3 −0.0002 precedent.
2. **Paper H1 narrative rewrite** (replaces the M2→H1 profile-replication
   story): *"On H1, a 648-parameter zero-init FiLM adapter recovers the gain
   of full M3-aware decoder retraining (~+0.02 R2 official) at ~1/90 of the
   parameters; the gain is entirely budget adaptation — an EMPTY control with
   the profile input zeroed preserves it (+0.0254, 5/5 dates), and a
   row-shuffled profile control confirms the unit-level content is unused."*
3. FiLM's remaining honest role is **deployment efficiency** (adapter vs
   retraining cost), not information injection.  No H1 FiLM submission is
   justified by these results; the profile claim is withdrawn everywhere it
   appeared.

## Scope boundaries

- All runs are source-only (held-in); the diagnostic opens no held-out
  recordings, fits no targets, and makes no submission.
- The claim covers H1 EP-FiLM in the sealed V2 framework.  M2-side profile
  claims are not addressed by this receipt; the M2/H1 capacity-control table
  alignment note in WORKORDER V5 applies at writing time.
