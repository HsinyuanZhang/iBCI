# D-optimal calibration design replay protocol (B9)

**Frozen:** 2026-08-12 (Asia/Hong_Kong)  
**Status:** CPU scaffolding and pre-registered gate only. Authorizes no GPU run, no decoder
R² evaluation, and no training by itself.  
**Parent:** `HANDOFF_NEXT_ROUND_DIRECTIONS_20260812.md` item B9.

## 1. Hypothesis

The carrier `T4_i = [a_i, c_i, m_i, b_i]` is fit by ordinary least squares over a
chronological prefix of `M` rewarded calibration trials with design
`X = [1, cos(theta), sin(theta)]`. Today's deployment rule takes the **first** `M`
trials, which yields random direction coverage. The measured label-budget curve
(`M10 = 0.304264`, `M15 = 0.338115`, `M20 = 0.351767`, `M30 = 0.358154`,
`M50 = 0.356828`) saturates early. The hypothesis is that the low-`M` deficit is
largely a **design-conditioning** problem rather than a pure data-quantity problem:
better trial subsets should yield **better carriers**, not merely better-conditioned
design matrices.

**Scientific question:** does better geometry buy a better carrier? That can fail:
D-optimal selects an extreme, spread-out subset that under nonstationary firing rates
may be less representative than a chronological prefix. Better geometry, worse estimate.
That negative result is informative and can close the branch.

This replay measures **carrier estimation quality only**. Decoder R² is a separately
authorized endpoint.

## 2. Algorithm

### 2.1 Estimator (unchanged)

Per unit, fit `rate(theta) = b + a cos(theta) + c sin(theta)` from per-direction means
over the **selected** labelled trials — same as `unit_side_features.py::_fit_cosine_tuning`.

### 2.2 Trial-selection arms (per session, per budget `M`)

Candidate pool: chronological prefix of `K = 50` rewarded trials. Arms:

1. **chronological_first_m** — trials `[0, M)` (current rule).
2. **d_optimal_prefix_k** — greedy forward D-optimal from `[0, K)`; algorithm
   `greedy_forward_d_optimal_v1`.
3. **random_m_prefix_k** — uniform random `M`-subset, seeds `{42, 43, 44}`.
4. **random_m_span_matched_k** — random `M`-subset whose **index span** matches the
   D-optimal selection span within tolerance (see section 2.7).

### 2.3 Primary endpoint: held-out carrier fidelity

For each arm:

1. Fit carrier on the selected `M` trials only.
2. Reference carrier: fit on **all `K` pool trials** (best available labelled reference).
3. Evaluate on **unselected pool trials** `[0, K) \ selected` and separately on
   **post-pool trials** `[K, …)` when present.
4. Report per-unit held-out RMSE and per-unit `cos([a,c]_M, [a,c]_reference)`.
5. Scalar **carrier fidelity score** (primary): `median_cosine_ac_vs_reference − median_held_out_rmse`.

**Primary statistic:** paired per-session difference at each `M`:

```text
delta(M) = fidelity_score(d_optimal) − fidelity_score(chronological)
```

### 2.4 Frozen primary pass/fail gate

Before any GPU or decoder work, on validation sessions (or synthetic CI fixtures):

| Budget | Gate |
|---|---|
| `M = 10` | median across sessions of `delta(10) ≥ +0.03` |
| `M = 15` | median across sessions of `delta(15) ≥ +0.03` |

**Negative result closes the branch:** if both medians are `≤ 0`, report "better geometry,
no better carrier" and do not authorize decoder or GPU follow-up from B9 alone.

A pass authorizes separately reviewed real-data replay on validation sessions only.

### 2.5 Implementation sanity checks (not primary gates)

Greedy D-optimal **maximizes `det(X'X)` by construction**. The following are recorded as
`implementation_sanity_checks` / `sanity_only_not_primary_gate` — they catch bugs but
cannot falsify the hypothesis:

- `det_xtx(d_optimal) − det_xtx(chronological) > 0`
- `cond(chronological) / cond(d_optimal) > 1`
- D-optimal `det_xtx` ≥ mean random-null `det_xtx`

### 2.6 Temporal coverage confound control

D-optimal tends to span a wider trial-index window than chronological-first-M. Add
**span-matched random null**: random `M` with the same index span as D-optimal (tolerance:
`max(1, ceil(0.05 × target_span))` trials).

**Interpretation rule (mandatory):** if D-optimal beats chronological on fidelity but **not**
the mean span-matched null, attribute the effect to **temporal coverage**, not design geometry.

Receipts must record `selected_indices`, `min_index`, `max_index`, `index_span`, and
`time_span` when trial times are available, for both D-optimal and chronological arms.

### 2.7 Leakage boundary (encoded in code)

Selection uses only target-direction labels and fixed geometry. Rates, decoder outputs,
and query data are forbidden. Enforcement: `assert_leakage_boundary()` in
`d_optimal_calibration_design.py`.

### 2.8 Deployment trade-off

Non-prefix D-optimal selections change the "time until calibrated" story. The prefix-bounded
pool (`K = 50`) is the deployment-realistic variant; chronological-first-M remains the
baseline comparator.

## 3. Frozen parameters

```text
K = 50
M ∈ {10, 15, 20, 30, 50}
random_null_seeds = [42, 43, 44]
algorithm_id = greedy_forward_d_optimal_v1
span_match_tolerance_fraction = 0.05
span_match_min_trials_tolerance = 1
primary_gate.minimum_median_session_delta = 0.03
primary_gate.budgets = [10, 15]
```

## 4. Sessions and sealed-test isolation

Six SUA validation CO sessions. Sealed formal-test sessions refused in code.
`sealed_test_sessions_opened: false` on every receipt.

## 5. What would falsify the hypothesis

- Primary gate fails at `M = 10` and `M = 15` (better geometry does not improve carrier).
- D-optimal beats chronological but not span-matched null → coverage confound, not geometry.
- Primary gate passes but split-half reproducibility does not improve (noise-dominated).

## 6. Scope limits

CPU only; no training; no decoder R²; no modification of sealed results.

## 7. Artifacts

| Artifact | Path |
|---|---|
| Protocol | `sua_exploration/docs/D_OPTIMAL_CALIBRATION_DESIGN_PROTOCOL_20260812.md` |
| Runner | `sua_exploration/scripts/d_optimal_calibration_replay.py` |
| Selection + fidelity | `sua_exploration/mc_maze/d_optimal_calibration_design.py` |
| Replay receipts | `sua_exploration/mc_maze/d_optimal_calibration_replay.py` |
| Aggregator | `sua_exploration/scripts/aggregate_d_optimal_calibration_replay.py` |
| Tests | `sua_exploration/tests/test_d_optimal_calibration_replay.py` |

Receipt schema: `d_optimal_calibration_replay_v2`.
