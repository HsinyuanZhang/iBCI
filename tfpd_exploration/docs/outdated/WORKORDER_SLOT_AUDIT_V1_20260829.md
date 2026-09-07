# Work Order — SLOT-AUDIT V1: the 50 already-supervised output slots, no training

Date: 2026-08-29
Authority: `docs/EXECUTION_GUIDANCE_AFTER_AC3_AND_CONTINUITY_AUDIT_20260829.md` §5 (queue item 2), §13.
Authorization: operator goal directive 2026-08-29. One process, CPU-only.

## 1. Question

The sealed Cell-D producer already supervises all `W=50` output positions densely.
The continuity probe's trajectory-alignment gain (best within-M10 ≈ `+0.0418` at
K=16, incurring K−1 bins of delay) must be explained mechanically:

> Are some already-supervised slots systematically better calibrated than the
last bin, or is the gain only covariance reduction from averaging equivalent
slots?

## 2. Immutable predecessors

| Artifact | Role |
|---|---|
| `results/continuity_probe_v1/continuity_probe_v1.json` (sha `8afaa910…`) | frozen baseline + trajalign arm numbers (anchor) |
| sealed Cell-D SWA `626f65d8…` + terminal `b3431db…` | decoder weights, read-only |
| `src/continuity_probe_v1.py` | the exact decode/trajalign law to reuse (never edit) |
| `src/calibration_gap_v1/{z1_oracle_cells,p4_stream_stats}.py` | the frozen materialization path |
| `src/tfpd_lane/matched_scorer.py` | house variance-weighted R², paired stats |

## 3. Materialization

One CPU decode per (surface, session, budget) through the frozen static
deployment recipe (identical to the probe's baseline row):
surfaces = within-6 + external-15; budgets = M4/M10/M30; full prediction
tensors `[n_windows, 50, 2]` + targets + window starts + valid masks cached once
under `cache/slot_audit_v1/` with sha256-bound manifest. Baseline parity: each
session's last-bin R² and full-tensor prediction SHA must reproduce the sealed
continuity-probe baseline rows exactly (they were bit-reproduced once already in
the filter-line static cache; reuse that anchor law).

The strict-27 source-roster variant is a separate additive stage, only if the
frozen runtime exposes a source roster without new parsing (decision recorded in
the receipt either way).

## 4. Audit rows (all numpy on the cache; deterministic)

1. **Per-slot calibration**: for each slot s ∈ 0..49 and each (surface, budget):
   per-slot variance-weighted R², MSE, bias (mean pred − target), residual
   variance, n. Absolute-bin alignment law: window with start `w` speaks about
   bin `w+s`; the same absolute bin `b` is estimated by every window
   `b−49 ≤ w ≤ b` at slot `b−w`. Slot comparisons are within-absolute-bin only.
2. **Redundant-estimate covariance**: for absolute bins covered by ≥2 windows,
   residual (pred−target) pairs grouped by slot separation δ = |s_i − s_j|;
   per-δ residual covariance, correlation, and common-mode fraction
   (ρ times the product of residual std). Report also the disjoint-group
   residual correlation for context (prior `rho_group=0.9171`) with the
   hierarchical-variance caveat the guidance mandates: never phrase it as
   "92% common-mode variance".
3. **Slot-subset stability across sessions** (§5.2): equal-weight average of
   slot subsets {49}, {45..49}, best-fixed-earlier-slot, all-50; per subset the
   equal-session mean R² and the per-session rank stability of the best slot
   (Spearman-style rank overlap between sessions). A fixed source-selected slot
   ensemble may only be *considered* if a stable subset beats slot 49 across
   sessions; target-selection is forbidden.
4. **Trajectory-alignment re-measurement**: the exact frozen probe law,
   K ∈ {2,4,8,16}, mean weighting, per (surface, budget): gain vs baseline,
   per-session paired deltas — must reproduce the sealed probe numbers
   (tolerance: float equality on recomputed values, since inputs are
   bit-identical; any drift is a hard failure).
5. **Latency accounting**: per K, output delay = K−1 bins; convert to ms using
   the sealed loader's bin size (read the constant from the frozen datamodule /
   loader; do not guess). The phrase "zero-lag" is banned.
6. **Digests**: last-bin prediction SHA and full-window SHA per (surface,
   session, budget); model-state digest before/after (unchanged).

## 5. Reading discipline (pre-registered)

- Primary reading is descriptive-mechanistic, not a gate: classify the
  trajalign gain as (a) redundancy ensembling (earlier slots individually worse,
  average helps), (b) stable better-calibrated subset (a fixed slot subset
  beats slot 49 consistently across sessions), or (c) session-dependent
  (best slot changes per session → no deployable fixed choice).
- This audit cannot authorize a training run by itself (§5.2). Any
  source-selected slot ensemble is a *candidate for a future work order*, not a
  deployment change.
- No target-selected quantity may appear without `target_label_leakage: true`,
  `checkpoint_selection_eligible: false`, `deployment_eligible: false`.

## 6. Process

Fresh root `results/slot_audit_v1/`; additive package `src/slot_audit_v1/` +
`scripts/run_slot_audit_v1.py` + tests. Attempt receipt → materialization
receipt (with cache manifest) → audit terminal receipt (atomic, 0444 + sidecar).
No CUDA anywhere in this line. Runtime bound: ≤30 min total CPU (decode ~63
sessions-budgets, seconds each; statistics numpy).
