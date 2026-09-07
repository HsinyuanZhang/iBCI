# Work Order — Support-Anchored T4 Memory, Stage O (oracle headroom)

Authority: `DESIGN_SUPPORT_ANCHORED_CAUSAL_T4_MEMORY_20260830.md` §3/§7.1/§13.
Operator goal 2026-08-30. Score-only successor; NO training, NO model change.

## Cells (within-6 + external-15 × M4/M10/M30, frozen activity-only runtime)

- **O0**: activity-only CDM, frozen support T4 (reference; must reproduce the
  sealed activity-only rows bit-exactly — V8 quick_v2 / score_v8 anchors).
- **O1**: same activity state; TRUE completed-trial direction (leakage-labelled)
  into the EXISTING recursive incremental carrier estimator (diagnostic: does
  the old estimator fail even with correct directions).
- **O2**: same activity state; TRUE direction into the support-anchored block
  refit (§4.1/§4.2) with trust-region projection (§4.3).

## Operator amendment 1 (binding, pre-registered)

O2's commit law = **always-commit + trust-region projection only**. The
three-factor gate's rejection semantics belong to Stage P; a gated O2 could
turn gate miscalibration into a false STOP. Fixed hyperparameters for Stage O:
`rho_M = 1.0` (true directions need no shrinkage), block size = 1 completed
trial, `alpha_M ∈ {0.5}` primary with {0.125, 0.25, 1.0} as non-governing
sensitivity rows, `c_M` = source-calibrated on within-6 folds ONLY
(external-15 never selects).

## Cross-reference anchor (receipt-required)

Include the sealed P2' oracle rows read-only (stage_cop: true-direction +
future-reading oracle accept, external M4 ≈ +0.092 vs O0-class) as the upper
bound bounding the deterministic commit law's loss.

## Gate (design §3.5, verbatim)

GO: O2−O0 ≥ +0.03 external, ≥10/15, other low budget ≥ O0−0.01, within all
≥ O0−0.02, zero target updates. HOLD: +0.01..0.03 & ≥9/15 → ONE
source-selected sensitivity only. STOP: < +0.01 or <9/15 or leak/safety fail.
Interpretation rows per §3.5 (O2>O1≤O0 ⇒ recursive estimator at fault; both
≤O0 ⇒ close continuous T4; M4-only ⇒ M4-only extension; …).

## Process

Additive `src/support_anchored_t4_v1/` + driver + no-data/no-CUDA synthetic
tests (causality: no update touches its own supplying trial; anchor
reconstruction: A0/b0 digests immutable; block-refit recomputation from the
evidence bank, never from a pseudo-updated carrier; trust-region projection
math; O0 bit-anchor logic; gate boundaries, epsilon 1e-12). Receipt
discipline: attempt before data/model access; 0444+sidecar; atomic
terminal-or-failure; state digests per trial (design §9); fresh root
`results/support_anchored_t4_stage_o_v1/`. GPU scoring launch only after code
review, on an idle card (C2/C3 mainline has priority; do not share a card
with an active training).
