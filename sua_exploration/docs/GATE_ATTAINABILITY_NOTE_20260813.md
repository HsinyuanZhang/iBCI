# Gate-attainability inventory (2026-08-13)

Shared harness: `sua_exploration/mc_maze/gate_attainability.py`.
Applied as read-only tests to the frozen A2 v2 and B1 v2 aggregators.
No frozen gate, threshold, contract, receipt, or log was changed.

Section 4 of `HANDOFF_FOUR_LANE_BRAINSTORM_SYNTHESIS_20260813.md` requires a
constructed must-pass input and a constructed must-fail input with opposite
verdicts before a gate is trusted.  This note lists every **existing** A2/B1
aggregator gate that was checked under that rule.

## A2 matched subject-shift v2

| Gate | Role | Attainable? |
|---|---|---|
| `mean_interaction_at_least_0p03` | primary | yes |
| `all_three_seed_interactions_strictly_positive` | primary | yes |
| `hierarchical_bootstrap_95ci_lower_positive` | primary | yes |
| `receipt_pairing_and_reuse_invariants_pass` | primary (fail-closed) | yes |
| `interaction.passes_all_gates` / `subject_shift_interaction_effective` | primary compound | yes |
| `interaction_ineffective` (`mean + 2σ < +0.03`) | kill rule | yes |
| within `t4-z4`: mean `>= +0.03` | secondary | yes |
| within `t4-z4`: all 3 seed means `> 0` | secondary | yes |
| within `t4-z4`: all 6 session means `> 0` | secondary | yes |
| within `t4-z4`: crossed bootstrap 95% CI lower `> 0` | secondary | yes |
| within `t4-z4`: exact Wilcoxon `p <= 0.05` at `n = 6` | secondary | yes (min p = `0.03125`) |
| within `t4-z4` `passes_all_gates` | secondary compound | yes |
| external `t4-z4`: mean `>= +0.03` | secondary | yes |
| external `t4-z4`: all 3 seed means `> 0` | secondary | yes |
| external `t4-z4`: all 15 session means `> 0` | secondary | yes |
| external `t4-z4`: crossed bootstrap 95% CI lower `> 0` | secondary | yes |
| external `t4-z4`: exact Wilcoxon `p <= 0.05` at `n = 15` | secondary | yes (min p = `2/32768`) |
| external `t4-z4` `passes_all_gates` | secondary compound | yes |
| seed-level exact Wilcoxon `p <= 0.05` at `n = 3` | withdrawn | **no** (min p = `0.25`); aggregator still sets `computed=False` |

## B1 M2 carrier × distillation factorial v2

| Gate | Role | Attainable? |
|---|---|---|
| Stage-P `mean_interaction_at_least_threshold` (`>= +0.03`) | routing | yes |
| Stage-P `all_three_seed_interactions_positive` | routing | yes |
| Stage-P `stage_f_authorized_by_evidence` | routing compound | yes |
| Stage-F `mean_interaction_at_least_threshold` (`>= +0.03`) | terminal | yes |
| Stage-F `all_three_session_means_positive` | terminal | yes |
| Stage-F `all_three_seed_means_positive` | terminal | yes |
| Stage-F `terminal_pass` | terminal compound | yes |
| P+F `terminal_gate_applied` | not a gate | always `False` by contract (descriptive sensitivity only) |
| Stage-F crossed bootstrap 95% CI | descriptive only | no pass/fail threshold; not a decision gate |

## Historical defects this harness is meant to catch

These are not current A2/B1 decision gates.  They are the four cases named in
section 4, recorded so the same shape is not reintroduced:

| Historical rule | Defect | Harness signal |
|---|---|---|
| D-optimal `det(X'X)` gate | could only pass | both constructed inputs pass |
| A4 falsification rule | could never fire | both constructed inputs fail |
| exact Wilcoxon `p <= 0.05` at `n = 3` | unattainable | `wilcoxon_threshold_attainability(3, 0.05).attainable is False` |
| identity-rank diagnostic | could not discriminate | pass and fail inputs yield the same verdict |

New aggregators (A1/C1/C2 included) should call `assert_gate_can_act` on every
decision clause before freeze, and `assert_wilcoxon_threshold_attainable` before
any p-value cutoff is written down.
