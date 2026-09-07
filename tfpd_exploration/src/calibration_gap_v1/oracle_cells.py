"""Z1 — the missing honest-M oracle cells (RUNBOOK; IMPLEMENTED 2026-08-25).

Status: RUN COMPLETE (CPU-only).  Implementation: ``z1_oracle_cells.py``
(same package); receipt:
``results/calibration_gap_v1/z1_oracle_cells.json`` (+ .sha256, 0444).
Result summary: honest-M oracle external-15 = 0.2413 (M4) / 0.3885 (M10) /
0.4449 (M30 = the C3 anchor, reproduced to 1.19e-07); the pre-registered
reading landed on the "~0.25 -> activity pathway breaks, P4 becomes the main
line" branch.  Cell matrix note: {M4,M10,M30} x {external-15, within-6} x the
single C3 support = SIX executed cells (the "9 cells" phrasing below
over-counts the single-level support axis); the historical runbook text is
kept verbatim for traceability.

Original runbook (spec only). Implementation must reuse the low-cost diagnostics v3
machinery (support-set materialization + frozen Cell-D forward + governing
scorer) with ONE new factor: the B3S calibration ACTIVITY tensor restricted
to the first M trials, while the carrier support stays C3 (full-session
labels). Everything else (sealed SWA `626f65d8...`, source-only OLS
normalizer, strict surfaces, last-bin/equal-session/variance-weighted
convention, zero target optimizer/backward/update) is copied verbatim.

Cell matrix (9 cells; none exist in any sealed receipt — ledger.missing_cells
registers them):

    for budget in (4, 10, 30):
        for surface in (external-15, within-6):
            support   = C3_full_session_oracle   (leakage-labelled diagnostic)
            activity  = first-M-trials only      (the new factor)
            estimator = OLS point carrier (deterministic, deployment-grade)

Readings (pre-registered):
  - honest_M4_oracle ~= 0.4449  -> carrier term is real at M4 activity;
    P1 (subspace restriction) is aimed correctly.
  - honest_M4_oracle ~= 0.25    -> the activity pathway, not the carrier,
    breaks at M4; P4 becomes the main line and P1 demotes.
  - also emit the full ceiling curve honest(M4, M10, M30) + the existing
    C3(M30-activity) rung so the paper has one ladder.

Non-negotiables:
  - C2/C3-style support sets are leakage/oracle diagnostics, labelled as
    such in every receipt (never deployable systems) — same convention as
    low_cost_calibration_diagnostics_v3.
  - receipt via tfpd_lane/receipt.py transactional write; body/sidecar 0444;
    bind the ledger SHAs of every consumed receipt.
  - no training, no target updates, formal data never opened.
"""
