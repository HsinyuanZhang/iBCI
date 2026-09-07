# Comparators

## A. What this folder is

A consolidated working area for classical and published comparators (ridge family, population
vector, Kalman, source-pooled ridge).

Everything under `docs/` and `receipts/` is a **copy**, not the authoritative original. The
originals remain the receipt-bound artifacts under `sua_exploration/docs/` and
`sua_exploration/results/`. See `PROVENANCE.md` for destination/original paths, SHA256 of both
sides, and the rule that published numbers must cite the original path and SHA recorded inside the
receipt itself.

`core/`, `runners/`, and `tests/` (if present) are owned by a separate agent. Those copies have
deliberately edited imports and are not byte-identical to the originals.

## B. Status matrix

| Comparator | Dataset(s) | Status | Executed? | Receipt |
|---|---|---|---|---|
| Ridge family (fixed lambda, PCA, lambda-CV, W sweep) | H1 | complete | yes | `h1_ridge_family_receipt.json` |
| RT classical (ridge fixed + lambda-CV, PV audit) | RT | complete | yes | `rt_classical_comparators_receipt.json` |
| Population vector | M2 | audit only, `PV_DEFINABLE` 6/6 sessions | audit only | `population_vector_comparator_m2_audit_receipt.json` |
| Population vector | H1 | **QUARANTINED**, scope violation | audit ran on wrong files | quarantined, must be re-derived |
| Kalman filter | subject-M, M2, RT, H1 | skeleton, unit tests pass | **NOT run** | none |
| Source-pooled ridge | subject-M | Part A only; Part B integrity gate was failing | Part A only | **none** |

Receipt copies live under `receipts/`, preserving the original subdirectory names. The H1 PV
receipt is under `receipts/population_vector_comparator/QUARANTINE_scope_violation/` and must stay
there.

## C. Key results already banked

From `docs/HANDOFF_COMPARATORS_20260812.md`:

- H1 best tuned ridge `0.288861` against carrier `0.500037`; sealed fixed-lambda `0.258235` reproduced at delta exactly `0.0`
- RT ridge `0.200202` against T4d `0.448176`, carrier wins 15/15 folds
- RT population vector `PV_INAPPLICABLE_NATIVE_FIELD_DEGENERATE`, one unique direction across all 15 sessions
- subject-M source-pooled ridge: `0` correspondent channels in **both** views, so it is structurally undefined there

## D. Known issues that block running anything

From Section 7.5 of the handoff:

1. Kalman position is synthesised as `cumsum(velocity) * bin_size`, not measured position — the most consequential unreviewed design decision
2. subject-M Kalman query identity is a Kalman-specific composite, not the sealed ridge fields; equality must be proven before reporting any contrast
3. Kalman M2 is marked `blocked_data_access` due to an over-broad scope rule; for FALCON M2 the `held-out-calib` split is legitimate few-shot calibration data and should be unblocked
4. H1 population vector audit must be re-derived through `h1_sparse_event_endpoint.index_heldin_calib` on the 13 held-in sessions rather than a raw glob

## E. The scope rule

The scope rule is per dataset, not global. `held-out-calib` is in scope for FALCON M2 because the
sealed M2 arms use it as the M24 budget, and out of scope for H1 because the sealed loader requires
`sub-HumanPitt-held-in-calib`. Neither is the private evaluation set, which is never in scope.

## F. How to run

Always from the repository root:

- `PYTHONNOUSERSITE=1` on every Python invocation
- `PYTHONPATH=SPINT-main:.`

Per-comparator recipes are in `docs/HANDOFF_COMPARATORS_20260812.md` Section 8. Reporting rules are
in Section 11 of the same document. The source-pooled ridge protocol is
`docs/SOURCE_POOLED_RIDGE_PROTOCOL_20260812.md`.
