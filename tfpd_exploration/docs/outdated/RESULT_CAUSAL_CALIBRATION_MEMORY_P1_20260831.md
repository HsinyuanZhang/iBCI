# Result: Causal Calibration Memory P1

Date: 2026-08-31

Status: completed DANDI 000688 Stage-P result, completed SUA replication, and
completed LOCAL-M2 protocol test. The official M2 deployment-contract audit
rules the method not evaluable under the continual interface. SUA is
directionally positive at M4 but misses the preregistered effect-size floor;
LOCAL-M2 is exactly null. Neither outcome changes the DANDI promotion
decision.

## Executive interpretation

This is a real but narrow positive result. On the matched DANDI 000688
surface, P1 improves the activity-only reference at M4 external by
`+0.020463` R2, with positive paired deltas in `12/15` sessions and a median
delta of `+0.016048`. It also improves all `6/6` within sessions.

The result hierarchy for the manuscript is:

| Evidence tier | Result | What it permits us to say |
|---|---|---|
| Promoted primary carrier result | DANDI external M4 `+0.020463`, `12/15`; within M4 `+0.022081`, `6/6` | support-anchored carrier memory adds information beyond activity-only CCM on this matched M4 surface |
| Directional cross-dataset evidence | SUA external M4 `+0.008083`, `10/15` | the sign and M4 localization are consistent, but the unchanged `+0.01` gate was not crossed |
| Applicability boundary | official M2 completed-trial transition is unavailable | no official M2 performance claim, positive or null, is defined for this state machine |
| Local transport test | LOCAL-M2 external M4 `0`, `0/6` | the DANDI estimator does not transfer automatically even when local trial boundaries are supplied |
| Budget boundary | DANDI M10 `+0.002013`; M30 exact `0` | no broad M10 claim; M30 is a safety no-op, not an improvement cell |

Thus the positive result is scientifically useful because it is isolated by
matched records and controls, not because `+0.020463` is a large universal
effect. Relative to its activity-only DANDI M4 reference (`0.245904`), the
equal-session mean rises to `0.266368`; this is an approximately `8.3%`
relative increase in R2, but the paper should report the absolute R2 delta as
the primary effect size.

The result supports one specific mechanism: an immutable support anchor plus a
bounded, cross-fitted, next-trial-only carrier refit. It does **not** support a
general claim that online T4 updates, temporal smoothing, learned confidence,
or C1 training weights improve deployment. M10 is effectively null, M30 is a
designed exact no-op, and the valid SUA replication is a directionally
consistent near miss rather than a second promoted dataset.

## 1. Scope and naming

This document refers to **Stage-P P1**, the deployable carrier state in
`support_anchored_t4_stage_p_v1`:

```text
completed-trial complementary-group circular direction
    + immutable labeled-support carrier anchor
    + bounded block refit
    + next-trial-only causal commit
```

It does not refer to the unrelated `PACD-P1` training arm. PACD-P1 remains
pending the PACD-P0 full-training terminal gate.

The primary promoted result applies only to the matched DANDI 000688
sub-C/sub-M center-out surface with explicit completed-trial boundaries. SUA
now has a separate completed replication result in Section 8. FALCON M2 has
no official-contract P1 performance result; its separate LOCAL-protocol null
is diagnostic only. M1 and H1 do not have a P1 performance result.

## 2. Immutable evidence

The completed root is:

`tfpd_exploration/results/support_anchored_t4_stage_p_v1`

All six body/sidecar leaves are regular mode-`0444` files. Recomputed body
digests equal their canonical sidecars:

| Body | SHA-256 |
|---|---|
| `attempt.json` | `18f1dc7267b6f9e2a25349f9f23ef4e83825a5aa536ccdeed2c581f3087e0f0c` |
| `replay.json` | `3d4759ce059f8a5d21bae672dfdb7a09340bb996ef2039667d2f74ed5c17277a` |
| `terminal.json` | `6f2929f6dd2eebaae9a0ef1fefbd36417ad0d4b0fe1bb04125f9c4aa3b93e5a6` |

The terminal status is `TERMINAL`; its decision is
`PROMOTE_DEPLOYABLE_CARRIER_STATE`; the driving cell is `P1@m4`.

## 3. Primary matched result

P0 is the activity-only Causal Calibration Memory reference on this exact
Stage-P scoring surface. P1 changes only the carrier state.

| Budget / surface | P0 activity-only | P1 carrier | P1 - P0 | Positive sessions |
|---|---:|---:|---:|---:|
| M4 external | 0.245904 | 0.266368 | **+0.020463** | **12/15** |
| M4 within | 0.421605 | 0.443686 | **+0.022081** | **6/6** |
| M10 external | 0.412169 | 0.414183 | +0.002013 | 10/15 |
| M10 within | 0.493055 | 0.499537 | +0.006482 | 5/6 |
| M30 external | 0.467105 | 0.467105 | 0 | exact no-op |
| M30 within | 0.527738 | 0.527738 | 0 | exact no-op |

For M4 external, the paired median delta is `+0.016048`; the session bootstrap
95% interval is `[-0.002470, +0.043600]`. The interval crosses zero, so the
result should be described as passing the preregistered effect-size, breadth,
control, and safety gate, not as an uncertainty-free population claim.

For M4 within, the paired bootstrap interval is
`[+0.014762, +0.031595]` and all six sessions are positive.

## 4. Controls and safety

On the M4 external driving surface, P1 beats all required same-record controls:

| Contrast | Equal-session delta |
|---|---:|
| P1 minus no pseudo evidence / P0 | +0.020463 |
| P1 minus constant-confidence control | +0.006328 |
| P1 minus deterministic shuffle control | +0.013318 |

The terminal also verifies:

- exact per-trial causal state chains;
- a source-selected-only hyperparameter law;
- zero target optimizer, backward, parameter, model, or normalizer updates;
- support trust-region compliance;
- an exact M30 carrier no-op;
- exact P0 prediction anchoring;
- within-budget regression bounds.

The M30 no-op is a substantive design result. Earlier unrestricted carrier
updates damaged the already-strong M30 state. P1 removes that failure mode by
construction instead of hoping a learned or fixed gate discovers it.

## 5. Supported and unsupported claims

Supported:

> A completed-trial, support-anchored, bounded carrier refit can add useful
> functional-tuning information beyond activity-only Causal Calibration Memory
> in the extreme M4 regime.

Within this project, this is the first preregistered matched result in which a
deployable causal T4 carrier state beats its activity-only reference. This
wording is deliberately scoped to the matched DANDI 000688 experiment; it is
not a cross-dataset or population-wide first claim.

Not supported:

- broad M4/M10/M30 improvement;
- an M10 carrier benefit beyond noise;
- an M30 carrier update benefit;
- a trajectory-continuity benefit;
- semantic value from the P2 confidence mapping;
- a learned-gate claim;
- cross-dataset generality or a second-dataset promotion.

P2, the trajectory-smoothed variant, is worse than P1 by `-0.011087` at M4
external and `-0.003333` at M10 external. It does not beat both constant and
shuffle controls. Therefore the positive result belongs to the simple
cross-group direction plus anchored block-refit mechanism, not to temporal
smoothing or learned confidence.

## 6. Paper position

Causal Calibration Memory should be presented as a two-state family:

```text
activity memory
    validated broad positive core

support-anchored carrier memory
    validated conditional M4 extension
```

The effect size is modest, but the mechanism-level result is important: it
changes the conclusion from "continuous T4 updates have no value" to
"recursive self-updating T4 was the wrong estimator; support-anchored causal
block refit can be useful when labels are extremely scarce."

Do not add the Stage-P `+0.020463` mechanically to activity-memory numbers from
another historical scoring harness. The final paper comparison must use the
same-input six-system matched scorer.

## 7. Completed DANDI factorial

The follow-up DANDI factorial has terminalized:

| Cell | Training weights | Carrier | External M4 delta vs F00 | Positive sessions | Decision |
|---|---|---|---:|---:|---|
| F00 | sealed | frozen | 0 | anchor | matched baseline |
| F01 | sealed | P1 | **+0.020463** | **12/15** | passes |
| F10 | C1 | frozen | -0.021584 | 3/15 | fails |
| F11 | C1 | P1 | -0.024468 | 6/15 | fails |

The preregistered additivity condition was:

```text
F11 >= max(F10, F01)
```

It failed: F11 scored `0.221437`, below F01 at `0.266368`. The correct
interpretation is not that C1 and P1 are two improvements waiting to be
stacked. On this matched surface, C1 changes the conditions under which P1 was
validated and destroys the P1 benefit. F01 therefore remains the promoted
configuration; F10 and F11 must not be promoted or averaged into the P1 claim.

This negative interaction improves attribution. The positive result belongs
to the support-anchored carrier update under the sealed producer, not to a
generic training-weight change. It also means that future cross-dataset P1
tests should first preserve the validated producer and change only the data
surface.

### Factorial provenance caveat

The final factorial root is internally terminal and its reported F01 row
reproduces the sealed Stage-P result. However, the first local attempt failed
before terminal with `KeyError: 'sealed_model'`; the canonical root was then
recreated for the repaired run. A separately named abort-disclosure directory
exists at `cdm_p1_cross_v1_attempt1_swap_keyerror_abort`, but it is not a
complete immutable predecessor failure graph and must not be described as one.
The retry history is documented in
`AUDIT_CDM_P1_CROSS_V1_RETRY_20260831.md`.

Therefore the factorial is useful as a matched mechanistic ablation, but it is
not a pristine one-attempt confirmatory replication. The original immutable
Stage-P P1 root remains the primary evidence for the `+0.020463` claim.

## 8. Cross-dataset replication contract and current status

The SUA route has now published a valid immutable replay and terminal. It is a
completed preregistered null by the promotion rule, although its estimate is
directionally positive. Cross-dataset generality therefore remains
unconfirmed. Any further transfer must keep the following fixed:

- the P1 proposal, support anchor, confidence law, trust region, and block
  commit thresholds;
- activity-only CCM as the matched reference;
- target updates equal to zero and next-trial-only causal commits;
- M4 and M10 reported separately, with M30 retained only as an exact-no-op
  safety cell;
- the official trial-boundary contract of the target dataset.

The execution order is:

1. test the unchanged P1 law on SUA, where trial boundaries and the T4 machinery
   are closest to the validated setting;
2. test FALCON M2 only if the official deployment contract exposes or permits a
   preregistered causal completed-trial boundary reconstruction;
3. do not start a learned gate or another smoothed P2 successor before a new
   deterministic carrier proposal beats P1 on a matched utility surface.

Current evidence is:

| Dataset / route | Current state | Scientific interpretation |
|---|---|---|
| DANDI 000688 | completed positive Stage-P terminal | one matched M4-positive result |
| FALCON M2 | `AUDIT_TERMINAL`; `P1-M2 NOT_EVALUABLE_OFFICIAL_CONTRACT` | no P1 performance claim is possible under the official continual interface |
| M2 LOCAL protocol | completed `TERMINAL`; external M4 delta `0`, `0/6`, `promoted=false` | exact local-protocol null; not an official-contract result |
| SUA | completed `TERMINAL`; external M4 `+0.008083`, `10/15`, `promoted=false` | directionally positive near miss; valid preregistered null, not a second promoted dataset |

The M2 audit is recorded at
`tfpd_exploration/results/cdm_p1_m2_v1/audit.json` with body SHA-256
`8ee508e6973da25dc27c29f7c245c265ea3231d8579e68de2d091a5c4b0055a3`.
The official evaluator exposes neural observations to `predict` but does not
call the completed-trial hook on the continual M2 path. The three
pre-registered causal neural-only boundary statistics achieved best source
session AUCs of only `0.4866`--`0.5306`, below the fixed `0.65` floor in every
source session. This is an applicability boundary, not evidence that P1 has a
zero or negative R2 effect.

A separate explicitly LOCAL M2 route subsequently used the locally available
completed-trial boundaries.  Its immutable six-leaf root is
`tfpd_exploration/results/cdm_p1_m2_local_v1`:

| Body | SHA-256 |
|---|---|
| `attempt.json` | `28440774a260d6f9ee916b2533bc4be892b7e5f2d8c2a68eb500b6d2606c33b2` |
| `replay.json` | `2dcd92fed7ebbf544f903c250758281e0f42e259f9bec5439a290696843be0b7` |
| `terminal.json` | `414f1e825a0ac0067e7c44cb2b34ea3c024381def041de77c45fc5d7d6ec3c82` |

All bodies/sidecars are mode `0444`, digest-valid and exactly linked.  The
local external-M4 F01m-minus-F00m delta is exactly `0` with `0/6` positive
sessions; within M4, M10 and M30 deltas are also exactly `0`, and external M30
is an exact no-op.  The gate records `CELL_FAILS_THE_GATE`, `promoted=false`,
while all anchors, causality, source-only selection, no-target-update and
safety checks pass.  It is an inference-only local-protocol null with no
decoder training, model/checkpoint update or target backward/update.  It does
not override the official-interface `NOT_EVALUABLE` verdict and must not be
reported as an official M2 score.

### 8.1 Completed SUA replication

The completed SUA root is
`tfpd_exploration/results/cdm_p1_sua_v1`. Its exact topology contains only
three immutable mode-`0444` body/sidecar pairs:

| Body | SHA-256 |
|---|---|
| `attempt.json` | `e28aa50c776553313ba9e42ffcd19ca8319cc8d747ec985e27798e18bc0b7dcc` |
| `replay.json` | `f8d3dca108de173d4f27c7d1948c5c11c75807719cf24df7b1374fbfad044179` |
| `terminal.json` | `00a3c409965de95f1ef778f44263a6527f6de74710ae3f4cd844806aec4e91a1` |

The body digests, canonical sidecars, attempt-to-replay link and
attempt/replay-to-terminal links all verify. The replay is inference-only:
`decoder_training=false`, `model_or_checkpoint_updated=false`, and all target
backward, optimizer/backward/update and parameter-update counts are zero.

The preregistered SUA result is:

| Surface | F00s activity-only | F01s P1 | Delta | Positive sessions | Decision |
|---|---:|---:|---:|---:|---|
| M4 external | 0.160935 | 0.169018 | **+0.008083** | **10/15** | fails `+0.01` effect floor; not promoted |
| M4 within | 0.419663 | 0.431160 | +0.011496 | 5/6 | safety passes |
| M30 external | 0.348035 | 0.348035 | 0 | exact no-op | safety passes |
| M30 within | 0.575024 | 0.575024 | 0 | exact no-op | safety passes |

The external breadth requirement is met exactly at `10/15`, but the mean
effect misses the `+0.01` floor by `0.001917`. The terminal therefore records
`CELL_FAILS_THE_F00S_GATE`, `promoted=false`, with
`no_candidate_beats_f00s` as the only fired stop condition. Anchors, causal
chains, source-only hyperparameter reselection, zero target updates and both
M30 no-op checks all pass. SUA M10 was not evaluated because its sealed M10
carrier uses grouped-direction OLS and the required exact anchored mirror was
not constructible under this Stage-P solver; no M10 inference should be made
from this run.

This result is more informative than an undirected null: DANDI and SUA both
move M4 in the positive direction and both preserve the designed M30 no-op.
It is nevertheless below the registered SUA promotion threshold. The honest
interpretation is **mechanism portability is plausible, but cross-dataset
performance generality is not established**.

The first SUA physical invocation stopped before scientific replay because
the configured streaming source root was absent. That invocation did not
publish a complete immutable failure graph; the canonical attempt was then
recreated for the repaired execution. This is a provenance caveat analogous
to, but separate from, the DANDI factorial retry caveat. The repaired
execution is internally terminal and scientifically usable, but it is not a
pristine one-attempt confirmatory lineage.

The paper must say "one promoted DANDI result plus one directionally positive
but non-promoted SUA replication," not "generalizes across datasets."

## 9. Recommended paper wording

### Result sentence

> On DANDI 000688, a support-anchored causal carrier memory improved the
> activity-only CCM reference by `+0.020` R2 at M4 external (`12/15` sessions;
> median `+0.016`) while preserving an exact M30 no-op. The session-bootstrap
> interval crossed zero, so we treat this as a preregistered matched positive
> result requiring cross-dataset replication.

### Mechanism sentence

> The gain came from cross-fitted complementary-group direction estimates and
> bounded block refits to an immutable labeled-support anchor; trajectory
> smoothing was worse, and combining the carrier with C1 training weights
> removed the gain.

### Cross-dataset sentence

> On SUA, the unchanged mechanism produced a directionally positive
> `+0.008` R2 at M4 external (`10/15` sessions) and an exact M30 no-op, but it
> missed the preregistered `+0.01` effect floor and was not promoted.

### Claim that should not appear yet

> Causal carrier memory consistently improves low-label decoding across
> datasets and calibration budgets.

That sentence is not supported by the current evidence.

## 10. Independent review assessment

The result is meaningful enough to retain as the second CCM mechanism, but it
should not displace activity memory as the broad primary result. Its strongest
value is causal attribution rather than raw effect size:

1. activity-only and P1 use the same matched prediction records;
2. P1 changes only the carrier state;
3. constant-confidence and deterministic-shuffle controls are weaker;
4. M30 is protected by an exact designed no-op;
5. trajectory smoothing is worse than P1; and
6. the C1 factorial removes rather than amplifies the gain.

Together these facts support a narrow statement that **anchored block refit is
a better carrier estimator than unrestricted recursive T4 updating in the M4
regime**. They do not support a statement that carrier memory is broadly
beneficial.

For manuscript structure, use the result in this order:

```text
primary CCM contribution
    activity memory across the validated low-budget surfaces

conditional extension
    support-anchored carrier memory at DANDI M4

applicability boundary
    requires a causal completed-trial boundary and a compatible T4 carrier
```

The completed SUA near miss and exact LOCAL-M2 null leave the promoted claim
explicitly dataset-specific while SUA still supplies a useful consistency
signal. The next
scientific upgrade is not another DANDI gate or smoother. It is either a
second compatible dataset that crosses the unchanged preregistered gate, or a
predeclared pooled/hierarchical analysis across compatible datasets. The
current SUA result must not be retroactively promoted by relaxing its effect
floor. The M2 audit cannot serve as a positive or null replication because the
official interface prevents the state transition from being defined.
