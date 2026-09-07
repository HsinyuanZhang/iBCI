# Work Order — AC3-U M4-Only Zero-Learning Utility Bridge (V1)

Date: 2026-08-29
Authority: `docs/EXECUTION_GUIDANCE_AFTER_AC3_AND_CONTINUITY_AUDIT_20260829.md` §4 (queue item 1), §13 launch boundary.
Authorization: operator goal directive 2026-08-29 ("实现 … 设计的实验，得到有意义的结果，尽量优化性能"). One launch, one process.

## 1. Question

Does the zero-parameter group direction ensemble (R-GE) improve **coherent carrier
utility** under the exact P2' replay, even though it missed the AC3-0
representation threshold? Rows U0 / UGE / U2, frozen everything else.

## 2. Immutable predecessors (bound by SHA-256 at attempt time)

| Artifact | Role |
|---|---|
| `results/ac3_action_continuity_v0/trajectories.npz` (+ sidecar) | frozen four-group trajectories + R0/R-GE pseudo-direction evidence |
| `results/ac3_action_continuity_v0/screen.json` | frozen row estimate digests (R0/R0.5/R2 `theta_sha256`), trial set |
| `results/ac3_action_continuity_v0/selection.json` | frozen R2 selected hyperparameters |
| `results/ac3_action_continuity_v0/materialize.json` | trial_id order binding (1,206 trials) |
| `results/learned_gate_p2prime_v1/stage_cop.json` | U0 anchor: within-M4 `O0` rows (bit-exact reproduction required) |
| `results/learned_gate_p2prime_v1/stage_a.json` | A0 within-M4 context |
| sealed Cell-D SWA `626f65d8…` | decoder weights (read-only) |

## 3. Surface, budget, roster

- budget = **M4 only**; surface = **within (source) only**; sessions = the frozen
  AC3 six: `sub-C_ses-CO-20151103/04/06/09/10/12` (1,206 completed trials).
- `sub-C_ses-CO-20151103` is a predeclared high-error stratum, NOT excludable: it
  stays in the governing equal-session mean and the 4/6 breadth denominator. A
  five-session sensitivity summary may be shown only as explicitly non-governing.
- Trial order = `materialize.json` `trial_ids`, which must equal the P2' runtime's
  `query_trial_ids[4]` per session in replay order (asserted at runtime).

## 4. Rows

| Row | Direction input | Construction |
|---|---|---|
| U0 | R0 raw single-view pseudo direction | frozen `raw` construction — MUST reproduce sealed stage-cop within-M4 `O0` bit-exactly (prediction SHA + matrix R2 + oracle decision counts) |
| UGE | R0.5 four-group circular mean (+ resultant length as reported credibility) | **rotation injection**: each group's velocity trajectory rigidly rotated so its net displacement aligns with the row's ensemble angle |
| U2 | R2 supervised-head grouped-OOF direction | rotation injection, same law |

Rotation law: for trial row `i`, group `g`, trajectory `v_g` with net
displacement `d_g`: rotate by `Δ = wrap(θ̂_i − atan2(d_g))`. Rotation preserves
every norm, the validity evidence object, and the movement mask, so the frozen
acceptance pattern (magnitude/movement gates) is unchanged; only the direction
content changes. If `θ̂_i` is undefined for a trial (no accepted view), that
trial's views pass through unrotated (U0-equivalent evidence). This fallback is
predeclared, per row, before replay.

## 5. Replay contract (frozen)

- exact P2' oracle law: `_rollout_oracle` with `horizon_H=5`, coherent greedy
  `u_j > 0` accept rule, `branch_states_for_counterfactual` parity checks;
- identical trial order and proposal-generating law across rows; rows differ
  ONLY through the complementary-prediction construction;
- no target-session data, no decoder training, no threshold sweep, **no output
  filter** (raw predictions scored), no model mutation (state digest asserted
  unchanged before/after);
- source behavior labels are read only where the frozen replay law already reads
  them (scoring + u_j), exactly as in sealed stage-cop.

## 6. Injection seam (implementation constraint)

The frozen `src/learned_gate_p2prime_v1/{physical,policy}.py` and
`src/causal_dual_memory_cell_d_v1/core.py` files are NOT edited. The driver
installs a process-local wrapper around `policy.build_construction_predictions`
that handles the two new construction names (`group_ensemble`, `r2_head`) by
pre-rotating `group_predictions` and delegating the frozen `raw` transform; all
frozen constructions pass through untouched. The U0 bit-exact anchor against
sealed stage-cop is the no-op proof of wrapper purity. The wrapper source digest
is recorded in the receipt.

## 7. Direction-estimate provenance (CPU, before GPU)

- R-GE: rebuilt by calling the frozen screen's own `static_row(trial_set,'R0.5')`;
  its `theta_sha256` must equal the sealed screen digest.
- R2: rebuilt by `learned_row_estimates(row='R2', …)` with the frozen selected
  hyperparameters; `theta_sha256` must equal the sealed screen digest.
- R0 sanity: `static_row(trial_set,'R0')` digest equals sealed screen digest.

## 8. Gates (pre-registered, §4.4)

```text
ADVANCE_SMALL_M4_CARRIER_STUDY requires BOTH:
  UGE − U0 equal-session mean R2 ≥ +0.01
  positive sessions (UGE > U0 paired per session) ≥ 4/6
U2 rescue clause: U2 may substitute only if U2 − UGE ≥ +0.01 under the same replay.
If UGE and U2 both fail: AC3_U_UTILITY_NULL__CLOSE_AC3_ON_FROZEN_M4_SURFACE
```

Boundary comparisons use the program epsilon 1e-12. No M10/M30 claim is made or
implied by any outcome (M10/M30 materializations were not present).

## 9. Pre-registered expectation (recorded before replay)

Sealed stage-cop within-M4: A0 0.46559 / C0 0.45072 / O0 0.48869 / O1 0.48885 /
O2 0.49327. The true-direction ceiling over raw on this surface is
O2 − O0 = +0.0046, and the UGE gate demands +0.01. UGE is a direction-content
intervention bounded in expectation by the true-direction construction, so the
expected outcome is NULL; a pass would require the ensemble direction to beat
the true direction by more than the entire true-direction advantage. This
expectation is recorded to prevent post-hoc reinterpretation in either
direction; the run is still mandatory because it converts the §4.4 closure
condition from an inference into a measurement.

## 10. Process

1. additive package `src/ac3_utility_bridge_v1/` + tests; no edits under
   frozen packages; fresh root `results/ac3_utility_bridge_v1/`;
2. CPU stages (no CUDA): attempt receipt → direction rebuild + digest anchors;
3. focused tests green (no data, no CUDA) before any GPU process;
4. replay under one process on one GPU (3090, GPU 1; GPU 0 reserved for the
   parallel SLOT-AUDIT line), projected ≈ 15–30 min from the sealed stage-cop
   throughput (47 s/session-rollout × 18); hard timeout 90 min;
5. terminal receipt atomic (terminal-or-failure); per-session paired deltas,
   per-row oracle decision counts, prediction digests, model-state-digest
   equality, and the §8 disposition strings.

## 11. Out of scope

R3/R4/R5/RS rows; new encoders; target-selected arms; M10/M30; carrier threshold
sweeps; output filters; any decoder training; rewriting the outcome as an
all-budget AC3 closure.
