# Work Order: H1 FiLM Content Diagnostic — EP-EMPTY / EP-ROWSHUFFLE (V5)

Date: 2026-09-04
Status: **preregistered, pending design review + GPU ownership check.  No data
access, no CUDA init before approval.**
Lineage: V1 (FiLM machinery) → V2 (LODO 2×2, EP-FILM +0.0239 PASS / LP-FILM
−0.0002 FAIL, sealed) → V3 (581866, frozen-C1 additive stack, official HO
0.2675) → **V5 (this): a cheap diagnostic grid that decides whether H1's FiLM
gain is M3-budget adaptation or calibration-profile content.**
C3 (`WORKORDER_H1_C3_FILM_CALAUG_V4_20260904.md`, review doc
`REVIEW_H1_C3_FILM_CALAUG_V4_20260904.md`) stays **frozen and unlaunched**;
this diagnostic gates any C3 reopen.

## 1. The question

All H1 FiLM evidence to date is consistent with one mechanism: **FiLM's gain =
adapting a non-M3-aware frozen substrate to the M3 deployment budget.**

| Substrate | Training-time identity budget | FiLM-only training (M3 blocks) | Delta |
|---|---|---|---|
| Frozen C1 (EP) | not M3-aware | ✓ | **+0.024** LODO / +0.027 official-chain |
| Frozen LP-R3 | M3-resampled; carrier fit from same M3 | ✓ | **−0.0002** |
| C2 (cal-aug line) | M7/M5/M4/M3 cycle | no FiLM | **+0.021** official, C2−C1, no selection |

Under this reading, C3 (already-M3-aware C2 recipe + FiLM) has expected
C3−C2 ≈ 0 — the LP-R3 situation — and the 648-parameter FiLM bought the same
+0.02 that a full decoder retrain bought.  The only open alternative is that
**profile content** (speed-contrast modulation depth per unit), not budget
adaptation, drives V2's +0.024.  V2 has no arm that separates the two.  This
diagnostic is that arm.

## 2. Grid (frozen C1 substrate, V2 framework verbatim)

Sealed and reused without re-running (primary numbers): **EP-ZERO**, **EP-FILM**
(V2 receipts; FILM +0.0239, gate PASS).

New arms — identical to EP-FILM in architecture (648 params), optimizer
(12 epochs, lr 3e-4, batch 32, seed 42), M3 block schedule, and scoring;
differing **only** in the FiLM context transform:

- **EP-EMPTY**: context = `carrier4 ‖ 0` — the `[176,4]` profile input is
  replaced by exact zeros after masking.  Tests: with profile content removed
  and per-block conditioning (budget adaptation) kept, does the gain survive?
- **EP-ROWSHUFFLE**: context = `carrier4 ‖ P(profile)`, where `P` is ONE fixed
  permutation of the 176 unit rows (seed-fixed, SHA-recorded), applied
  identically at training and scoring.  Marginal profile distribution
  preserved; unit correspondence destroyed.  H1 counterpart of M2's C2
  capacity-control arm, so the paper tables align across datasets.

## 3. Implementation surface (sealed files untouched)

- New package `SPINT-main/../src/h1_calibration_profile_film_content_diag_v1/`
  (in the SPINT repo) importing `h1_calibration_profile_film_v1.plan/core` and
  `h1_calibration_profile_film_v2.evaluate` internals verbatim; it adds
  `film_identity_with_mode(net, activity, carrier, profile, film, mode, perm)`
  replicating the v1 early-branch body with the context transform applied at
  the single injection point (`context = torch.cat((effective, profile), -1)`).
- Rationale: V1/V2 sealed receipts pin their code closure; a wrapper keeps the
  sealed authority intact while reusing 100% of the validated path.
- **Pairing binding**: each new arm's initial film state SHA must equal
  `V2_INITIAL_FILM_STATE_SHA256 = ff2263b5705a499caeb5fe807f9889479fdbc31f806b42de82b11d351ba210cf`
  (same seed, same builder).  All batch schedules are deterministic, so new
  arms are exactly paired against the sealed V2 arms — no V2 re-run required
  for the primary contrast.
- Receipts: 0444 + sidecar under fresh
  `results/h1_calibration_profile_film_content_diag_v1/`.

## 4. Anchors and per-arm assertions

1. Zero-init anchor (every arm, every fold): at init the arm's forward equals
   the EP-ZERO identity bit-exact (`torch.equal`; film_out zero-init makes all
   context modes the identity at step 0).
2. EP-EMPTY: the fed context's profile half is exactly zero and its carrier
   half equals the batch carrier (guards against silent passthrough).
3. EP-ROWSHUFFLE: `P(profile)` rows sort-equal `profile` rows (true
   permutation); `P` SHA recorded once; train and score paths use the same
   recorded `P` (receipt cross-check).

## 5. Scoring and preregistered decision law

- Score each new arm per LODO fold exactly as V2 did (same evaluator, same
  held-out date).  Primary contrast: `arm − EP-ZERO` per date, evaluated with
  V2's own gate machinery (`_gate`: mean ≥ +0.005, ≥ 4/5 dates nonnegative,
  worst ≥ −0.010).
- Optional default-on drift check: re-run EP-FILM same-session (cheap); if its
  mean deviates from the sealed +0.0239 by more than ±0.005, STOP — the
  environment drifted and nothing is interpretable (sealed numbers still
  govern the record).

Decision branches (preregistered):

- **Branch A — budget adaptation**: EMPTY passes the V2 gate (and FILM
  reproduces within ±0.005).  → C3 permanently cancelled.  Paper H1 narrative
  becomes: *"a 648-parameter zero-init adapter reproduces the gain of
  M3-aware decoder retraining at ~1/90 of the parameters"* — a deployment
  adapter story; the profile-content claim is withdrawn.
- **Branch B — profile content**: EMPTY mean < +0.002 (gate fails) AND FILM
  reproduces.  → profile content is real; C3 may then be opened, and must add
  a same-machine **C3-ZERO** control arm at that point.
- **Branch C — mixed**: neither A nor B.  → attribute proportionally by the
  EMPTY/FILM delta ratio; write both stories one notch weaker.

Threshold note: `+0.005` reuses V2's own gate constant; the `+0.002`
content-floor is proposed here and flagged for reviewer confirmation.

## 6. Compute and environment

- 10–15 FiLM-only trainings × 12 epochs, minutes each — **≤ ~1 h total**,
  versus C3's 3–5 h joint run with zero expected effect.
- GPU by ownership preflight (GPU0 was occupied by another agent's dandi688
  jobs at preregistration time; expected GPU1; any deviation recorded).
  CPU thread discipline; `PYTHONNOUSERSITE=1` spint env.
- **No dependency on the iBCI predecessor artifacts** (no npz, no C2
  checkpoints): substrate is the local sealed C1 checkpoint, data is local
  held-in NWBs.  The pending 0444 permission fix in the iBCI checkout does
  not block this diagnostic.
- No Docker build, no EvalAI submission in this workorder.

## 7. Boundaries

- The two C3 documents remain frozen; this workorder does not modify them.
- Deployment-side top-k epoch prediction averaging on the HO face with the
  existing C2 checkpoints (zero training) is explicitly **out of scope** here —
  a separate deployment decision for the line owner.
- No paper claims from this diagnostic beyond the two narrative rewrites in
  §5; M2/H1 capacity-control symmetry is a presentation note, not a new claim.

## 8. Reviewer checklist

1. Injection point: is `context = cat(effective, profile)` in v1 `core.py`
   the complete FiLM-context surface (no other place profile enters)?
2. EP-EMPTY semantics: zeros applied to the masked `[176,4]` profile input —
   is `carrier4 ‖ 0` the intended control (vs zeroing the carrier half)?
3. EP-ROWSHUFFLE: one fixed permutation per arm (not per sample)?  Permutation
   seed/proposal acceptable?
4. Decision thresholds: gate reuse (+0.005) and content floor (+0.002)
   acceptable?
5. FILM rerun-for-pairing default-on acceptable?
6. Branch A paper rewrite wording acceptable?
