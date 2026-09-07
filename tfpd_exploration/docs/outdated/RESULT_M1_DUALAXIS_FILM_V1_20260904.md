# Result — M1 Dual-Axis FiLM V1 (Rev3): both context layouts null-to-negative

Date: 2026-09-04 (updated after the t4_plus_contrast main-arm run)
Status: **completed; static-anchor PASS (bit-exact); all gates FAILED; no submission**
Receipts: `results/m1_dualaxis_film_v1/receipt.json` (t4_plus_contrast main arm, 0444+sidecar);
preserved predecessors: `…_contrast_only/` (weaker-layout negative),
`…_anchor_failed/`, `…_crash2/`, `…_typed_tensor_crash/` (fail-closed infrastructure runs).
Workorder: `docs/WORKORDER_M1_DUALAXIS_FILM_V1_20260904.md` (Rev3).

## What was run

The M2-validated recipe (per-unit phase contrast → zero-init FiLM → cached
identity) ported to M1.  Phase contrast from RAW 20 ms counts + the trials
table (hold `[start, gocue)` vs reach `[gocue, contact]` per-trial means →
delta + log-ratio, robust-z) — Task 0.5 loader, never the interpolated
[trials,1024,N] encoder array.  FiLM-only training (Adam 1e-4, 12 epochs, 256
windows/session, batch 32, seed 42 — the M2 `means_ep12_lr1e4` cell; not
3e-4).  Context layouts, both tested:

- `contrast_only` (archived `…_contrast_only/`): context = phase(2) only.
- `t4_plus_contrast` (main arm, 581801 layout): context = carrier(4) ‖ phase(2),
  carrier also raw-concat into post_pool.

Champion: 581727 weights `c5672a2b`-lineage SHA `237de3b9…` (strict-load).
Surface B: 3 later-day sessions, M4 identity / last-6-trial query
(`carrier_k_later_day` convention), same windows as the sealed chrono-k4 rows.

## Anchor precondition (gate precondition) — PASS

Zero-init FiLM (γ=β=0) reproduces the sealed chrono-k4 baseline **bit-exactly**
(ΔR2 = 0.0 on all three sessions vs 0.694172 / 0.589838 / 0.493788).  The first
run of this cell FAILED this anchor because the optimizer's parameter list
included the wrapped champion encoder — the precondition caught a real
"FiLM-only" contract violation, exactly as designed.  (Fail-closed roots
`…_crash2`, `…_typed_tensor_crash` are infrastructure-only.)

## Results (Surface B, ΔR2 vs static champion)

| Context layout | mean Δ | per-session | gates |
|---|---:|---|---|
| contrast_only (archived) | +0.0002 | +0.0019 / +0.0013 / +0.0015 | full vs static FAIL (2/3); zero-contrast 8× full |
| **t4_plus_contrast (main)** | **−0.0011** | −0.0063 / +0.0017 / +0.0012 | FAIL (2/3); all three null arms above it |

t4_plus_contrast null arms (mean Δ vs static): zero-contrast −0.00053,
session-constant −0.00030, row-shuffle −0.00098 — every null arm is *less
negative* than the full FiLM.  The phase contrast adds nothing; if anything it
is the worst component.

## Interpretation (audit-adjusted)

1. **No contrast increment on M1, in either context layout.**  The full FiLM is
   the worst arm of its own comparison set; null checks confirm the tiny
   positive in the archived contrast_only run was the f(0) affine path.
2. **Scope limits (disclosed):** single seed (the pre-registered confirmation
   seed was not run — moot given the negative); Surface B is an M4-carrier/6-
   trial proxy with an h-distribution shift the FiLM never saw (training used
   M10-pool features); the Surface B phase contrast was computed over the
   first 10 trials while the deployment horizon there is 4 — this makes the
   evaluated contrast *cleaner than deployable*, i.e. it biases toward the
   null's opposite, so the negative is if anything understated.
3. **"Consistent with" framing:** a 3-session M4 proxy cannot prove or refute
   cross-dataset generalization in general; what it does establish is that the
   specific M2 recipe, executed faithfully (bit-exact anchor, FiLM-only
   training, both context layouts), carries no measurable increment here.

## Session-level conclusion (across M2 + M1)

| Question | Answer |
|---|---|
| Does the contrast→FiLM recipe work at all? | On M2, officially yes (581801, +0.025) |
| Does it transfer to M1 under matched contract? | No (this cell: main arm −0.0011, 2/3) |
| Does any online row-update work on these encoders? | No (three row shapes × two regimes, all null/negative; trial-shaped −1.69) |
| Where is the value? | Static identity construction: M2 0.3203 (hold/reach FiLM), M1 champion 0.6396 (rSyn3 carrier) — both **static** |

The difference between the two datasets' outcomes is now well-localized: M2's
contrast axis (reach-vs-hold, from public target angles) carried per-unit
information that M2's decoder could use; M1's phase axis (same construction,
better timestamps) carries none beyond what the rSyn3 carrier already encodes.
