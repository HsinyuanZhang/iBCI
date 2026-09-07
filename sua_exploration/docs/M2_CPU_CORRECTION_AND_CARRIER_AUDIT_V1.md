# M2 CPU correction and carrier audit v1

**Frozen:** 2026-08-01 HKT  
**Owner:** root reviewer; execution delegated to Terra agents  
**Primary deployment endpoint:** unseen-session chronological calibration support followed by a fully disjoint future query

**Completed 2026-08-01.** Authoritative results and the no-GPU-continuation
decision are in
[`M2_CPU_CORRECTION_AND_CARRIER_RESULTS_20260801.md`](M2_CPU_CORRECTION_AND_CARRIER_RESULTS_20260801.md).

This charter separates protocol validity, uncertainty, estimator reliability,
and decoding efficacy. Passing a support-only reliability diagnostic does not
authorize a decoding claim. A local held-out development replay is not a hidden
EvalAI result.

## Global invariants

- This entire batch is CPU-only. It must not initialize CUDA or occupy either
  RTX 3090.
- Historical result directories and JSON files are immutable provenance. Every
  correction writes to a new result directory.
- Calibration support is a chronological prefix. Every future-query result
  must machine-check that a 50-bin input history lies wholly after the query
  boundary.
- Query labels may be used only for scoring, never for descriptor fitting,
  hyperparameter selection, checkpoint selection, fallback selection, or
  session exclusion.
- F0/T4/K4/KS4 comparisons must disclose label access. Equal trial counts do
  not imply equal label information.
- Undefined rank-deficient fits remain `undefined`; no pseudoinverse value,
  zero fill, or silent session/channel deletion is allowed.
- Results are aggregated first within session/cell as declared below. Repeated
  cells on the same session are not independent sessions.
- Each workflow must write a pre-compute protocol receipt, source hashes,
  machine-readable raw rows, an aggregate, and focused contract tests.

## Endpoint and comparison roles

- `K4-F0`: end-to-end utility including the benefit and cost of dense behavior
  labels and the side-feature path.
- `K4-KS4`: label- and width-matched mechanism test for correct
  channel-attached content.
- `K4-T4`: descriptive operational comparison only; it is not a carrier-form
  mechanism contrast because K4 consumes dense per-bin velocity while T4
  consumes one target direction per trial.

## 0e — Correct the historical M33 replay

**Purpose:** determine whether the historical M33 direction/magnitude survives
after removing support/evaluation overlap. This is a correction audit, not a
new six-session efficacy test.

- Arms/cells: frozen F0/T4/TS4 checkpoints for
  `fold1_seed42`, `fold1_seed43`, and `fold2_seed42`.
- Support: trials `[0:33]`; query starts at trial 33.
- Re-run inference. Deleting rows from historical metrics is forbidden because
  the four nonempty sessions were also scored from trial 0 historically.
- The two sessions with exactly 33 trials are `ineligible_zero_query` and
  receive no R2.
- The four eligible sessions must have newly generated per-session scores and
  full-window-disjoint audits.
- Aggregate equal-session means within cell and equal-cell means across the
  three cells. Preserve all cell/session paired deltas.
- Name the endpoint `M33-eligible four-session subset`.
- With four independent session clusters, an exact two-sided signed-rank test
  cannot attain `p <= .05` (minimum `.125`). A positive result cannot reinstate
  the old six-session significance claim.

## 0f — Explain the M24 internal/held-out sign reversal

The frozen observed endpoints are:

- held-in internal `T4-F0 = -0.15019443`;
- strict local held-out future-query `T4-F0 = +0.0528854117`.

Run an audit-only `domain x query_start` factorial for F0/T4:

- domain: fold-1 held-in validation session versus six local held-out sessions;
- query start: 0 versus 24;
- M=24 and window size 50 throughout;
- use the already frozen source checkpoints;
- report each session, never just a pooled mean.

**Post-freeze feasibility correction:** the held-in minival session contains
only two query trial boundaries, so held-in `query_start=24` is structurally
ineligible. It is recorded as missing-by-design rather than scored or replaced.
The completed audit therefore partially identifies the planned factorial but
cannot uniquely separate domain, horizon, and their interaction.

`query_start=0` is a diagnostic resubstitution endpoint and must never be
called future-query performance. Repeat the factorial with the existing common
epoch-9 F0/T4 checkpoints as a single predeclared checkpoint-selection
diagnostic; do not scan epochs.

Interpretation is frozen before computation:

- sign mainly follows query start: support/resubstitution or temporal segment
  is the main explanation;
- sign mainly follows domain: session/domain shift is the main explanation;
- both change with a nontrivial interaction: both mechanisms matter;
- a material best-versus-common-epoch change implicates independent
  validation checkpoint selection.

The stored internal `all_three_pass=true` remains historical arithmetic. It is
not a release gate for future-query evaluation.

## 0a — Uncertainty and identifiability audit

The available cells (`f1s42`, `f1s43`, `f2s42`) are not a complete
fold-by-seed factorial. The audit must not report separately identified
`sigma_seed`, `sigma_fold`, or their interaction.

Report only quantities supported by available evidence:

- paired cell/session dispersion;
- epoch-window fluctuation when raw epoch curves remain available, explicitly
  labeled as autocorrelated within-run fluctuation rather than run variance;
- cluster-aware MDE sensitivity over plausible dispersion values;
- the distinction between statistical uncertainty and a deployment SESOI;
- a machine-readable list of non-identifiable or missing quantities.

The invalid historical M33 held-out effect is excluded from future-query MDE.
The 18 cell-session rows are not treated as 18 independent observations.

## 0b/0c — Lag by split factorial for K4

- Data: seven M2 held-in-calibration sessions only.
- M grid: `8,10,12,16,20,24,28,30,32,33`.
- Behavior-lead grid: integer 20-ms bins from `-5` through `+10`
  (`-100` through `+200` ms), including the current `+2`/40-ms setting.
- Blocks and shifted behavior must remain inside the same raw trial.
- Split types:
  - chronological trial-half: primary;
  - odd/even trial: drift diagnostic;
  - odd/even block: optimistic autocorrelation upper bound.
- For each audited session, a global lag may be selected only using the other
  six held-in sessions and chronological future-rate MSE.
- Per-channel optimal lag is diagnostic only. Within each half and lag, use
  trial-level leave-one-trial-out OLS prediction MSE for each channel; do not
  select lag by in-sample residual. Invalid-rank folds are undefined and each
  half must meet a frozen valid-fold minimum. Ties are broken by MSE, distance
  to the current `+2`-bin lag, then numerical lag. Report half-to-half agreement;
  a per-channel lag cannot become a deployed lookup.

## 0d — T4 split-half curve

Use the same M grid. T4 uses chronological trial-half and odd/even-trial splits;
odd/even-block is explicitly `N/A` because partial-bin exposure would define a
different estimator. Report separately:

- the vector/direction component `[a,c]`;
- modulation magnitude `m`;
- baseline rate `b`;
- fit rank, usable directional trial count, condition/balance, and defined
  channel/session coverage.

Rank-deficient halves are undefined. A low `[a,c]` cosine does not by itself
invalidate T4 decoding if `m` or `b` remains reliable; component reliability is
a mechanism diagnostic, not a decoder gate.

## K4 components and balanced-design controls

Report split-half reliability for `Wx/Wy`, `||W||`, `b`, and standardized
`[||W||,b]`. Normalization is frozen from the appropriate LOSO train sessions
and shared across halves; halves must not estimate separate normalization.

For direction-balanced selection:

- use eight fixed 45-degree velocity-angle bins; if any bin is empty, mark that
  half's balanced arm undefined rather than merge bins;
- use a fixed number of blocks equal to eight times the minimum bin occupancy;
- select each half independently;
- compare balanced OLS with 50 fixed-seed equal-N random OLS subsamples, the
  all-block OLS reference, and an all-block ridge control with fit-local
  standardized velocity and fixed `alpha=1` mapped back to raw coordinates;
- report covariance eigen-ratio, raw condition number, direction balance,
  future-rate MSE, and component reliability;
- do not predeclare or discover a `0.45` pass threshold from these seven
  development sessions.

## Component naming rule

- If full K4 is interval-equivalent to standardized `[||W||,b]`, the claim is
  `modulation-depth + baseline-rate identity`, not a directional signature.
- If `[||W||,b]` is interval-equivalent to `b` alone, reduce the claim to
  `baseline-rate identity`.
- Equivalence requires a frozen practical margin and an interval-based
  decision. Failure to reject a difference is not equivalence.
- If neither superiority nor equivalence is resolved, use the neutral name
  `K4 calibration feature vector`.

## GPU gate after the CPU batch

No M33 training run is authorized. The only possible GPU continuation is an
M24 chronological-disjoint precision replication for the two missing cells,
reusing the existing `fold1_seed42` result. It may begin only after root has:

1. independently reproduced the CPU aggregate or core statistics;
2. confirmed all focused tests and provenance contracts;
3. classified 0e and 0f without unresolved protocol drift;
4. frozen the exact arms, checkpoints, support/query contract, label
   disclosure, aggregation rule, and stopping rule.

Because the six local held-out sessions have already been viewed, any such GPU
block is a precision replication, not a clean confirmation or hidden formal
test.

**Final decision:** the conditional K4/KS4 block was not launched. Existing
strict M24 `K4-T4=+0.018987` is below the practical margin, and the CPU
factorial shows that the network-attached directional component is unreliable
(`.1837` median per-channel cosine at M24); balancing and ridge do not rescue
it. The immutable decision receipt is
`results/m2_cpu_correction_and_carrier_audit_v1/gpu_no_launch_decision_v1.json`.
