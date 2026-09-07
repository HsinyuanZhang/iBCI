# Handoff: M4/M10 Comparison and Performance Route

Date: 2026-08-24  
Status: Phase 1 and Phase 2 terminal; complete 204-cell comparison closed.

## Decision first

1. Adopt fixed normalized ridge lambda 0.1 as the current low-cost T4
   estimator for short label prefixes.  Do not use per-session GCV in the
   main route.
2. Reject the tested Calibration-Budget-Marginalized Cell D (CBM-D) training
   recipe.  It improves held-in M4 but harms cross-subject transfer at M10/M30;
   do not spend another round on its schedule, loss, or sampling ablations.
3. Keep M4-only causal D-optimal selection and fixed ridge 0.1 as low-cost
   deployment options.  Turn D-opt off by M10.  These are useful protocol and
   estimator results, not a new decoder breakthrough.
4. Keep A2 only as a matched M30 anchor because no honest M4/M10 A2 row exists.
   Original SPINT B0, Arm A, Cell D, classical baselines, protocol controls,
   lambda controls, and CBM-D are now all scored.

## Phase-1 terminal authority

- Result: `results/calibration_budget_comparators_v1`
- Receipt SHA-256:
  `0ec107cc95cfb806336e6859e55fb8d7d30c365c2ff829757e100045cfb5c1cc`
- Matrix: 48 cells, within-6 and external-15, last-bin variance-weighted R2,
  equal session weighting, paired session comparisons.

## Mean R2 comparison

### Within-6

| System | Regime | M4 | M10 | M30 |
|---|---|---:|---:|---:|
| Cell D, OLS | T4 label-limited; B3S M30 | 0.3815 | 0.5231 | 0.5697 |
| Cell D, OLS | total calibration limited | 0.2620 | 0.4516 | 0.5697 |
| Cell D, fixed ridge 0.1 | T4 label-limited; B3S M30 | **0.4537** | **0.5369** | 0.5665 |
| Cell D, GCV ridge | T4 label-limited; B3S M30 | 0.2869 | 0.4837 | 0.5665 |
| Arm A, OLS | T4 label-limited; B3S M30 | 0.3044 | 0.4964 | 0.5538 |
| Dense W50 ridge | classical prefix fit | 0.0853 | 0.1829 | 0.3063 |
| Trial-rate ridge | classical prefix fit | -0.0362 | 0.0339 | 0.0835 |
| Population vector | classical prefix fit | 0.0452 | 0.0642 | 0.0915 |

### External-15

| System | Regime | M4 | M10 | M30 |
|---|---|---:|---:|---:|
| Cell D, OLS | T4 label-limited; B3S M30 | 0.1147 | 0.3583 | 0.4179 |
| Cell D, OLS | total calibration limited | 0.0458 | 0.2872 | 0.4179 |
| Cell D, fixed ridge 0.1 | T4 label-limited; B3S M30 | **0.1663** | **0.3703** | **0.4286** |
| Cell D, GCV ridge | T4 label-limited; B3S M30 | 0.0844 | 0.2893 | 0.4184 |
| Arm A, OLS | T4 label-limited; B3S M30 | 0.0170 | 0.2370 | 0.2604 |
| Dense W50 ridge | classical prefix fit | -0.6684 | -0.1446 | 0.3308 |
| Trial-rate ridge | classical prefix fit | -0.0791 | 0.0205 | 0.0933 |
| Population vector | classical prefix fit | 0.0224 | 0.0724 | 0.1104 |

## Paired findings

- Fixed ridge minus OLS at M4:
  - within: +0.0722, 6/6 positive, bootstrap 95% CI [+0.0260, +0.1396];
  - external: +0.0516, 11/15 positive, CI [-0.0169, +0.1081].
  The external uncertainty is dominated by one session
  (`sub-M_ses-CO-20150617`, delta -0.3135); the other 14 sessions average
  +0.0777.  This is a real robustness warning, not a reason to delete the
  session or claim a clean universal win.
- Fixed ridge minus OLS at M10:
  - within: +0.0138, 4/6 positive, CI [+0.0006, +0.0255];
  - external: +0.0119, 9/15 positive, CI [-0.0060, +0.0329].
- Total-calibration limit minus label-only limit:
  - M4: within -0.1195 and external -0.0689;
  - M10: within -0.0714 and external -0.0711.
  These paired losses show that short-prefix failure is not only noisy T4.  A
  decoder trained only with M30 B3S identity is also exposed to an unseen
  calibration-activity regime.
- GCV is not a safe short-prefix selector.  It often chooses lambda 1--100
  from only 4 or 10 trials and underperforms fixed lambda 0.1.  External GCV
  minus OLS is -0.0303 at M4 and -0.0690 at M10.  Relative to fixed ridge,
  GCV loses 0.0819 at external M4 and 0.0809 at external M10.  Across external
  sessions, correlation between selected log10(lambda) and GCV-minus-fixed
  performance is -0.659 at M4 and -0.702 at M10: the overly strong choices
  are directly associated with worse decoding.
- Dense W50 ridge has a positive external median at M4/M10 but a negative
  mean because one session catastrophically fails.  It is an informative
  high-variance baseline, not a robust deployment candidate.

## M30 anchors

- Cell D seed42: within 0.5697, external 0.4179.
- A2 matched scores:
  - seed42: within 0.5849, external 0.3178;
  - seed43: within 0.5725, external 0.3367;
  - seed44: within 0.5755, external 0.3837.
- Historical original SPINT B0 has no T4 label carrier and is not an honest
  M4/M10 row until its calibration activity is actually truncated.  That
  evaluation is running separately.

## Existing selection/oracle comparison

The earlier low-cost diagnostic used the same sealed Cell-D model and provides
the following additional M4 carrier-support rows:

| Support rule | Within-6 M4 | External-15 M4 | Deployable? |
|---|---:|---:|---|
| C0 first four trials, OLS | 0.3815 | 0.1147 | yes |
| C1 cue-balanced early D-optimal | 0.4403 | 0.1858 | yes, if cue metadata is counted |
| C2 cue-matched scattered | 0.3600 | 0.2375 | diagnostic; uses later-session labels |
| C3 full-session label oracle | 0.6082 | 0.4449 | no |

This resolves two questions.  First, the M4 carrier ceiling is genuinely
large: the full-label oracle gains +0.2266 within and +0.3303 external over
C0.  Second, four scattered labels do not approach the oracle, so the gap is
not explained by temporal placement alone.  Cue-balanced early selection is
a useful protocol baseline and is numerically competitive with fixed ridge,
but its earlier carrier-fidelity gate failed and its M30 behavior regressed;
it is not promoted to the main learned method.

Fixed ridge and cue-balanced selection answer different questions and should
both remain in the paper table:

- fixed ridge changes estimator regularization while keeping the first-M
  support fixed;
- C1 changes which labeled trials are acquired while retaining ordinary OLS;
- CBM-D changes the source training distribution while keeping deployment
  inference ordinary and closed-form.

## Phase-2 experiment

CBM-D keeps the exact sealed Cell-D graph and 3,510,842 live parameters.  The
only main treatment is source training over joint prefix budgets: for every
batch, both B3S calibration activity and ordinary OLS T4 use the same M, with
M deterministically covering every integer from 4 through 30 per session.

The first two smoke attempts failed before backward or update.  They exposed
an exact producer-contract edge case rather than a scientific result: one
source session has a calibration row with missing target direction encoded as
`-1`; the generic D-opt fitting helper treated it as a real last-direction
cue, while the sealed T4 producer excludes it.  V3 now calls the exact shared
ordinary-T4 producer used by the sealed lane.  A focused replay of the failing
session is bitwise equal to the sealed producer (86 units, maximum absolute
difference zero).  V3 binds both immutable pre-update failures and must pass
the source-only smoke before full training.

Primary Phase-2 gate: external total-calibration M4 improves by at least 0.05
with at least 10/15 sessions positive; M10 must be non-negative, and M30 may
not regress by more than 0.03.  If the main system is not positive, do not
spend another round on schedule/loss ablations.

The final performance gate is recipe-matched rather than cherry-picked:
M4 uses D-opt-first30 + ridge 0.1, M10 uses chronological + ridge 0.1, and M30
uses chronological + ridge 0.1.  Each CBM-D row is paired against the exact
same inference recipe on sealed Cell D; therefore the delta isolates source
budget marginalization.  A second OLS/chronological gate is retained as the
mechanistic attribution check.

## Execution status

- Complete: original SPINT B0 M4/M10/M30 with actual truncated calibration
  activity, plus Arm A total-calibration M4/M10/M30.
- Complete: causal first-30 protocol factorial at M4/M10: chronological vs D-optimal
  support, OLS vs fixed ridge, and B3S=M30 vs B3S using the exact selected M
  trials.  The first-30 cap prevents calibration activity from crossing the
  fixed query boundary.
- Complete: CBM-D 48-epoch source training and final-four SWA.  Its final score
  covers label-limited and total-calibration M4/M10/M30.
- Complete: for every CBM-D evaluation cell, ordinary OLS and fixed ridge 0.1
  are both evaluated.  The result shows negative rather than additive
  composition with the Phase-1 winning short-prefix estimator.
- At M4/M10, the same scorer also evaluates chronological versus causal
  first-30 D-opt support.  M30 is not duplicated because selecting all first
  30 trials is exactly chronological.  This tests composition with the only
  positive protocol treatment without inventing another training arm.
- Complete: final paired CBM-D deltas against Cell D OLS, fixed-ridge Cell D,
  Arm A, original SPINT, dense ridge, trial ridge, and population vector.
- Complete: a frozen normalized-lambda curve at M4/M10.  Lambda selection
  uses within means only; external is locked and never participates in
  selection.  This determines whether the single external M4 ridge failure is
  a global regularization issue or irreducible session heterogeneity.

## Complete comparison ledger

| Axis | Cells / evidence | Status |
|---|---|---|
| Estimator | ordinary OLS, fixed ridge 0.1, prefix GCV ridge | complete |
| Classical decoder | trial-rate ridge, dense W50 ridge, population vector | complete |
| Neural baseline | Cell D, Arm A, original SPINT B0 | complete |
| Information accounting | T4-only label limit vs B3S+T4 total-calibration limit | complete for Cell D and Arm A |
| Acquisition protocol | chronological vs cue D-optimal within first 30 trials | complete |
| Protocol factorial | support x estimator x B3S regime at M4/M10 | complete, 32 cells |
| Ridge sensitivity | nine fixed lambdas at M4/M10; within selects, external locked | complete, 72 cells |
| Training distribution | fixed M30 Cell D vs CBM-D M4..M30 | complete; governing STOP |
| CBM inference composition | chronological/D-opt x OLS/ridge x B3S regime; M30 deduplicated | complete, 40 cells |
| Non-deployable ceiling | full-session label oracle | complete, diagnostic only |

No A2 M4/M10 point will be fabricated.  A2 remains an M30 anchor until an
actual matched short-calibration input contract exists.

## Causal protocol factorial terminal result

- Receipt: `results/calibration_budget_protocol_factorial_v2/receipt.json`
- Receipt SHA-256:
  `ce283aa7d046d575ed50860090633a1cab9448ccfba51472ff7694812c7594b2`
- All selected trials are among the first 30 and precede the fixed query
  boundary.  The failed v1 predecessor was a float64-side engineering error
  before any update; v2 casts the side tensor to the sealed float32 contract.

### M4 mean R2

| B3S regime | Support | Estimator | Within | External |
|---|---|---|---:|---:|
| M30 activity | chronological | OLS | 0.3815 | 0.1147 |
| M30 activity | chronological | ridge 0.1 | 0.4537 | 0.1663 |
| M30 activity | D-opt first30 | OLS | 0.4403 | 0.1799 |
| M30 activity | D-opt first30 | ridge 0.1 | **0.4636** | **0.1902** |
| selected M activity | chronological | OLS | 0.2620 | 0.0458 |
| selected M activity | chronological | ridge 0.1 | 0.2749 | 0.0665 |
| selected M activity | D-opt first30 | OLS | 0.3011 | 0.1142 |
| selected M activity | D-opt first30 | ridge 0.1 | **0.3089** | **0.1197** |

For the deployable total-calibration M4 regime, D-opt minus chronological is
external +0.0684 for OLS (11/15 positive, CI [+0.0298,+0.1084]) and +0.0532
for ridge (9/15, CI [+0.0091,+0.1008]).  Ridge adds only +0.0055 on top of
D-opt, so the two treatments are not additive; they primarily repair the same
short-prefix conditioning problem.

The causal first-30 D-opt row also reproduces almost all of the earlier
first-50 diagnostic gain: external M4 label-limited OLS is 0.1799 here versus
0.1858 in the first-50 diagnostic.  Therefore the gain is not an artifact of
selecting calibration trials after the fixed query boundary.

### M10 mean R2

| B3S regime | Support | Estimator | Within | External |
|---|---|---|---:|---:|
| M30 activity | chronological | OLS / ridge | 0.5231 / 0.5369 | 0.3583 / 0.3703 |
| M30 activity | D-opt first30 | OLS / ridge | 0.5002 / 0.5217 | 0.3418 / 0.3590 |
| selected M activity | chronological | OLS / ridge | 0.4516 / 0.4677 | 0.2872 / 0.2955 |
| selected M activity | D-opt first30 | OLS / ridge | 0.4309 / 0.4406 | 0.2922 / 0.2983 |

D-opt has no useful M10 effect: total-calibration external changes by only
+0.0049 (OLS) or +0.0028 (ridge), while within changes by -0.0207/-0.0271.
The acquisition policy should therefore be budget-conditional: useful at M4,
off by M10.

At this point in the execution order, calibration-activity distribution shift
was the dominant unresolved short-budget hypothesis: M4 D-opt+ridge
total-calibration external (0.1197) was far below the same carrier treatment
with M30 B3S activity (0.1902).  Phase 2 tested that hypothesis directly;
the terminal result below rejects the CBM-D solution rather than leaving it
as the main performance experiment.

## Original SPINT and Arm-A matched short-budget result

- Receipt: `results/original_spint_short_budget_v2/receipt.json`
- Receipt SHA-256:
  `526dc11460f44d67674287762273da74265fe05fa5aa0c10bab953b65fb06a68`

| Surface | System | M4 | M10 | M30 |
|---|---|---:|---:|---:|
| within | original SPINT B0, activity only | 0.1007 | 0.2752 | 0.3198 |
| within | Arm A, total calibration | 0.2443 | 0.4622 | 0.5538 |
| within | Cell D, total calibration | 0.2620 | 0.4516 | 0.5697 |
| external | original SPINT B0, activity only | -0.1232 | -0.1349 | -0.0934 |
| external | Arm A, total calibration | 0.0050 | 0.2054 | 0.2604 |
| external | Cell D, total calibration | **0.0458** | **0.2872** | **0.4179** |

Original SPINT improves within as more calibration activity is supplied, but
external stays negative at every budget.  This is direct evidence that an
activity-derived identity can fit a session without forming a transferable
cross-subject identity; T4 is load-bearing rather than decorative.

Cell D minus Arm A on external is +0.0408 at M4 (11/15 positive, CI
[+0.0061,+0.0753]), +0.0819 at M10 (13/15, CI [+0.0335,+0.1240]), and
+0.1576 at M30 (14/15, CI [+0.1014,+0.2224]).  Within differences are small
and uncertain.  Whole-unit dropout therefore contributes predominantly to
cross-subject robustness, and its benefit grows as the identity estimate
becomes more stable.

The v1 route failed an exact historical M30 comparison.  The corrected v2
shows that current matched and historical M30 values differ only at roughly
1e-5 to 1e-4 from scorer/rounding details; the historical row is diagnostic,
while the current matched row is governing.  There is no substantive science
disagreement.

## Fixed-lambda curve terminal result

- Receipt: `results/ridge_t4_lambda_curve_v1/receipt.json`
- Receipt SHA-256:
  `e82d917b348be09bd3888a924d8523e8c88dc7794d1a8d273aa3b333b725f257`
- Grid: 0.0001, 0.001, 0.01, 0.03, 0.1, 0.3, 1, 3, 10.
  Selection uses within means only; external is locked.

| Budget / regime | within-selected lambda | Within R2 | Locked external R2 |
|---|---:|---:|---:|
| M4, B3S M30 | 0.3 | 0.4573 | 0.1883 |
| M4, total calibration | **0.1** | 0.2749 | **0.0665** |
| M10, B3S M30 | **0.1** | 0.5369 | **0.3703** |
| M10, total calibration | **0.1** | 0.4677 | **0.2955** |

For the deployment-relevant total-calibration regimes, lambda 0.1 is selected
at both M4 and M10 and also has the highest locked external mean on the grid.
The more permissive M4 label-limited regime selects 0.3; it adds external
+0.0220 over 0.1 (9/15 positive, CI [+0.0011,+0.0429]) but only +0.0035
within.  Lambda >=1 rapidly collapses all curves.  This confirms that GCV's
short-prefix failure is over-regularization, while preserving 0.1 as the
single deployment default.  The lambda line is closed; no further tuning run
is justified.

## Phase-2 terminal result

### Authority and execution integrity

- Training root: `results/calibration_budget_marginalized_cell_d_seed42_v3`.
- Training terminal SHA-256:
  `8e5c807dbf961cbda673d355b81ae530f6078b02b7aadd06deae92e95d95c3a1`.
- Final-four SWA SHA-256:
  `51c38cb0cda5c520993eed05a2ff4f178ef9f577cc5fd0a49a72a1e007ba7642`.
- SWA state SHA-256:
  `162bf60dedf08ec4c72fd4b53599b8cc9455d5a81b7ab3eab13af21945c32b62`.
- Training completed all 48 epochs and 1,628,400 source optimizer steps.  All
  48 epoch receipts report finite model/optimizer state and all eight critical
  gradient groups.  Mean source loss fell from 0.60150 to 0.31747.  Final
  budget exposure was balanced: 59,514--60,946 batches per M across M4..M30.
- Matched-score root: `results/calibration_budget_marginalized_score_v2`.
- Score receipt SHA-256:
  `09ffbd837be8529ecd0a3cca60bbe010ebb989137346ddb92cf8023f98756f51`.
- Score terminal SHA-256:
  `2d90cca19ff60a072929d4bbd97c96182162af33981fe83f987aa74d5ef695f0`.
- The first score attempt is preserved at
  `results/calibration_budget_marginalized_score_v1`: all 40 forwards completed,
  then aggregation failed because the same ridge estimator was named
  `ridge_fixed_0p1` in the protocol receipt and `fixed_ridge_0p1` in the new
  scorer.  V2 exact-binds that no-result/no-update failure and adds a live
  receipt-topology regression test.  It is an engineering predecessor, not a
  scientific result.

### Predeclared recipe-matched training effect

The deployment recipe is M4 D-opt-first30 + ridge 0.1, M10 chronological +
ridge 0.1, and M30 chronological + ridge 0.1.  Each row below compares CBM-D
with sealed Cell D using the exact same inference recipe; only trained weights
differ.

| Surface | Budget | Cell D R2 | CBM-D R2 | Delta | Positive | Bootstrap 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| within | M4 | 0.3089 | 0.2972 | -0.0117 | 3/6 | [-0.0844,+0.0649] |
| within | M10 | 0.4677 | 0.4638 | -0.0039 | 2/6 | [-0.0472,+0.0448] |
| within | M30 | 0.5665 | 0.5396 | -0.0270 | 3/6 | [-0.0848,+0.0121] |
| external | M4 | 0.1197 | 0.0050 | **-0.1147** | 3/15 | [-0.1782,-0.0645] |
| external | M10 | 0.2955 | 0.2060 | **-0.0894** | 2/15 | [-0.1383,-0.0429] |
| external | M30 | 0.4286 | 0.2949 | **-0.1337** | 0/15 | [-0.1734,-0.0978] |

Every predeclared check fails: M4 does not gain +0.05, breadth is 3/15 rather
than 10/15, M10 is negative, and M30 regresses far beyond -0.03.  The governing
decision is `STOP`.

### Mechanistic OLS/chronological contrast

The simpler ordinary-OLS contrast exposes the important within/external split:

| Surface | Budget | CBM-D minus Cell D | Positive | Bootstrap 95% CI |
|---|---:|---:|---:|---:|
| within, total calibration | M4 | **+0.1121** | 6/6 | [+0.0653,+0.1530] |
| within, total calibration | M10 | +0.0216 | 3/6 | [-0.0191,+0.0698] |
| within, total calibration | M30 | -0.0171 | 3/6 | [-0.0697,+0.0227] |
| external, total calibration | M4 | -0.0104 | 5/15 | [-0.0639,+0.0504] |
| external, total calibration | M10 | **-0.0706** | 3/15 | [-0.1158,-0.0280] |
| external, total calibration | M30 | **-0.1195** | 0/15 | [-0.1549,-0.0858] |

CBM-D therefore learned to handle short calibration prefixes in held-in source
geometry, but it did not learn a transferable calibration-budget invariance.
The M4 within gain is real and broad; its external counterpart is null.  At
M10/M30 the same training treatment actively damages external transfer.  This
is the same warning shape seen elsewhere in the project: held-in improvements
can come from session-specific fingerprints rather than subject-invariant
identity.

### Final deployment comparison

For total calibration, the best CBM-D rows are external 0.0353 at M4
(chronological OLS), 0.2221 at M10 (D-opt OLS), and 0.2984 at M30
(chronological OLS).  They do not beat the low-cost sealed-Cell-D recipes:

| Budget | Recommended low-cost recipe | Within R2 | External R2 |
|---:|---|---:|---:|
| M4 | Cell D + causal D-opt-first30 + ridge 0.1 | 0.3089 | **0.1197** |
| M10 | Cell D + chronological first10 + ridge 0.1 | **0.4677** | 0.2955 |
| M30 | Cell D + chronological first30 + ridge 0.1 | 0.5665 | **0.4286** |

The alternative M4 label-limited setting (B3S keeps M30 activity) reaches
0.4636 within / 0.1902 external with D-opt + ridge, but it costs 30 calibration
trials and is not an honest four-trial deployment row.  The distinction must
remain explicit.

### Final interpretation

This campaign supports a protocol/estimation result, not a new decoder claim.
Use fixed ridge 0.1; use causal D-opt only at M4; keep ordinary Cell D weights.
Do not continue CBM-D schedule/loss/sampling ablations.  The remaining large
M4 oracle gap cannot be recovered by merely exposing the source decoder to a
uniform prefix-budget distribution.  A future performance route must add a
genuinely different cross-session invariance mechanism or carefully gated
unlabelled target adaptation, and it must predict external rather than only
held-in gains before receiving another full training budget.
