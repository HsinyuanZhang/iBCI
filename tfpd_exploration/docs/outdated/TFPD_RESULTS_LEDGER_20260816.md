# TFPD results ledger

**Updated:** 2026-08-19 (sub-population round closed, §3D/§3E; cell E closed unstarted; successor
round `HANDOFF_ACTIVITY_VS_IDENTITY_MASK_20260819.md`)
**Status:** development numbers only. sub-M is a repeatedly used development external
subject; nothing here is an official or cross-subject claim. Every number cites its
immutable receipt.

**Correction notice (11:30 HKT).** Three readings previously recorded in §3 are withdrawn, and
the `pv` row is void. See §3 "Withdrawn readings" and §6. The diagnosis and its receipts are in
`HANDOFF_LARGE_ROUTE_OPTIMIZATION_20260816.md` §2A, §5. Numbers in the tables are unchanged and
remain what the receipts say; only their interpretation changes, except for `pv`, whose numbers
are now marked invalid.

## 1. Stage-0 synthetic gates

| rung | params | receipt | G5 aligned/zero/wrong (query-mean R²) |
|---|---:|---|---|
| bilinear (frozen P1) | 22,434 | `results/stage0_synthetic_gates_v1` (`9455805d…`) | 0.803 / −0.285 / −0.604 |
| population vector | 17,908 | same | 0.575 / −0.010 / −0.591 (**not transferable, see §6**) |
| bilinear low-rank B_i (§5.3) | ~23k | `results/stage0_synthetic_gates_v2` (`78581e62…`) | 0.742 / −0.218 / −0.501 |
| bilinear state-gain (§5.4) | ~23k | same | 0.765 / −0.119 / −0.673 |
| task-frame set attention (5M) | 5.02M | `results/stage0_large_gates_v3` | 0.688 / −0.003 / −0.007 |

Notes: widening the static class (low-rank) does not beat the diagonal model on the
synthetic task; the large rung required four predeclared engineering fixes
(axis-scramble, causal mass channel, eval-mode gates, gradient clipping 1.0 — first
step grad norm 122 otherwise lands a dead plateau).

**Gate-validity caveat (added 11:30 HKT).** `synth.py:95-108` emits the carrier as
`[a_noisy, c_noisy, m_noisy, b_noisy]` in raw rate units, with no normalization, whereas the
real datamodule z-scores every side-feature column (`unit_side_features.py:1539`). G5 therefore
validated the models against a carrier parameterization the real pipeline never delivers. This
is harmless for any model that feeds the carrier into a learned affine map (`bl`, `large`,
`spintshape`) and fatal for `pv`, which decomposes the carrier arithmetically (§6).

## 2. Time-varying weighting pre-gate (§5.5)

`results/pregate_timevarying_v1` — **overall FAIL (honest verdict, non-vacuous)**:
isotropic cosine population fisher−static = −0.0007 (FAIL; even an oracle Fisher
weight at the true state does not beat static carrier ridge); state-matched
population align−static = +0.0796 (PASS). Time-varying read-in weighting is
first-order only under sparse-instantaneous-information populations.

## 3. Stage-1 within-validation (6 strict-27 val sessions, pooled per-session R², epochs 5–12)

| model | params | T4 native | Z4/zero | content effect | wrong-pair | destroyed-act |
|---|---:|---:|---:|---:|---:|---:|
| bl (probe) | 22K | 0.0767 | 0.0216 (z4-arm) / −0.329 (zero diag) | +0.41 | −0.303 | −0.167 |
| ~~pv (transparent baseline)~~ **VOID (§6)** | 18K | ~~0.0012~~ | ~~0.0037~~ | — | — | — |
| large (task-frame set attention) | 5.02M | 0.2562 | 0.0059 (z4-arm) | +1.02 | −0.219 | −0.022 |
| **spintshape (comparator)** | ~3.5M | **0.4574** | 0.1985 (z4-arm) / 0.0006 (zero diag) | +0.26 | −0.0822 | 0.0505 |

**Within verdict (development):** under identical standard-init task-only training, the
SPINT-shaped decoder BEATS the 5M task-frame set-attention model by +0.20 within-val
(0.457 vs 0.256).  Teacher initialization is worth a further +0.12 within over standard
init (A2 teacher-init within T4 = 0.575).  The contract's primary decision is EXTERNAL
absolute T4 (the task-frame premise is a transfer claim; A2 within 0.575 → external
0.341 while SPINT-shaped activity systems collapse externally) — the external scorer is
prepared and user-gated.

**Two qualifications on that verdict (11:30 HKT).** First, `spintshape` consumes strictly more
deployment-time information than `large` — 30 calibration trials × 100 bins × N units of raw
activity, against 4 numbers per unit — so the +0.20 is not an architecture-only comparison and
must never be reported without that disclosure. Second, the epoch-5–12 window mean is not a
deployable model's score and is most misleading where the curves move: `bl_t4` runs
`0.0576 0.1204 0.1089 0.1042 0.0856 0.0682 0.0493 0.0195` while `bl_z4` runs
`−0.0185 −0.0009 0.0169 0.0273 0.0316 0.0357 0.0400 0.0407`, so by epoch 11 the bilinear's
carrier arm is **below its own zero arm** and its `+0.055` window-mean advantage is an
early-epoch effect that reverses. `large_t4` peaks at epoch 6 and drifts flat-to-down
(`0.2626 0.2704 0.2760 0.2325 0.2461 0.2623 0.2546 0.2447`) — it converged and plateaued, so the
gap is not an epoch-budget artifact.

Receipts: `results/stage1_source_cells_v1/{bl,pv,large}_*/within_val_epoch_window_eval.json`
(pv_z4 in `stage1_source_cells_v1_r1`).

Readings that stand:
- capacity × task-frame constraint: large 3.3× bl in absolute within-val;
- destroyed-activity with an aligned carrier: large `−0.022` genuinely fuses
  activity × carrier rather than reading a carrier-only session signature. Note the
  contrast with `spintshape` `+0.0505`, which *does* retain a carrier/calibration-only
  shortcut — this is the one diagnostic on which large is cleaner;
- the honest per-unit-correspondence evidence is the T4→wrong-pair drop on the trained
  T4 model: large `0.2562 → −0.2189` (−0.475), spintshape `0.4574 → −0.0822` (−0.540).

**Withdrawn readings (11:30 HKT).** All three are superseded by
`HANDOFF_LARGE_ROUTE_OPTIMIZATION_20260816.md`; they are retained here struck through rather
than deleted, since they were cited elsewhere while in force.

- ~~"the carrier is the information channel at every scale (zero-carrier arms ≈ 0)"~~ —
  **does not follow.** `bl`, `pv`, and `large` have no identity channel *other than* the
  carrier, and their pooling is permutation-invariant, so under a uniform gate the movement
  direction is not identifiable from one trial's activity multiset. Z4 ≈ 0 is close to a
  tautology for those arms. `spintshape_z4 = 0.1985` is the control that shows it: supply a
  second identity channel and the zero-carrier arm stops being zero. (Handoff §5.)
- ~~"wrong-pair > zero at the large scale (−0.22 vs −0.77): the slot attention attenuates
  mis-paired carriers"~~ — **not attenuation.** `carrier_map` is affine *with bias*
  (`tfpd_large.py:83`), so a zero carrier gives every unit the identical gate `W_c · bias_g`:
  identity is erased *and* the token geometry leaves the trained manifold. The wrong-pair arm
  keeps the carrier distribution intact and merely mis-assigns it, so it stays on-manifold.
  `zero < wrong_pair` is explained by off-manifold-ness alone, and the zero arm of this
  architecture is not a usable baseline. No attenuation claim can rest on it. (Handoff §5.)
- ~~"the transparent PV fails on real SUA (0.001) — 'isn't this just a learned population
  vector?' is answered negatively by construction"~~ — **the question was never asked.** PV is
  broken by a carrier-normalization bug (§6). This was the reading that most flattered the lane,
  since it removed the strongest simple baseline on the strength of a scaling error.

## 3A. Gate 1–3: Z4 boundary pilot + three admission arms (2026-08-16/17)

Authority: `HANDOFF_CARRIER_ADMISSION_CURRICULUM_20260816.md` (§5 pilot, §6 arms, §11 gates).

**Gate 1 — source-only Z4 boundary pilot** (`results/z4_boundary_pilot_v1`, terminal
`1372887ac976f219…`).  Canonical Z4, seed-42 spintshape standard init, constant Adam 1e-4, 5%
deterministic sha256-frozen source-audit split (54,287 of 1,086,007 windows, excluded from
gradients), dual-track judgement: the **source-audit column alone** froze
`selected_epoch=14 → T_pre=15, E_t4=33` (S_max 0.700245 at ep15, earliest within 0.005 = ep14).
The within-dev report-only column peaked at epoch 1 (0.165) and decayed to ~0.11 — the two
columns diverged in opposite directions, which Gate 3 below resolves decisively.

**Gate 2 — three arms, one canonical initial state** (state sha `65bacb85447df40e…`,
artifact `b0a340fe4d09eac1…`), strict-27 full window set (33,925 steps/epoch, the official
cells' sampler), GPU1 sequential chain 13:12Z–01:40Z, all `ARM_TERMINAL`, zero invariant
failures, closure equal, final-four SWA each:

| arm | epochs | optimizer steps (=exact) | final train loss | final-step LR | ‖W_side‖ | SWA sha256 |
|---|---|---:|---:|---:|---:|---|
| A `direct_t4_48` | 48 | 1,628,400 (=48×33,925) | 0.163724 | 1.000000e-06 | 1.3992 | `920eb4c9827dbe66…` |
| B `z4_pretrain_then_t4` | 15 Z4 + 33 T4 | 1,119,525 (=33×33,925) | 0.174894 | 1.000000e-06 | 0.7606 | `6db6c437d82a062a…` |
| C `direct_t4_exposure_matched` | 33 | 1,119,525 (=33×33,925) | 0.184273 | 1.000000e-06 | 1.3475 | `2811c7da0997e9e2…` |

Contract receipts: B's Z4 phase held W_side/exp_avg/exp_avg_sq elementwise exactly zero all 15
epochs with visible side bitwise equal to canonical Z4; the admission switch verified no weight
reset (state SHA equal), fresh empty Adam, finite first T4 forward, W_side exactly +0 before the
first T4 step; B-phase-2 vs C LR equal at every T4 step (33,925×33), same constructor/batch
order/step count.  Receipts: `results/admission_arms_v1/arm{A,B,C}_*/{launch,terminal}_receipt.json`
(A `6cf7d317…`, B `a6b42fbf…`, C `e08cd735…`).

**Gate 3 — within-dev screen** (`results/gate3_within_screen_v1/within_screen_receipt.json`,
0444; all four SWAs rescored on the same 184,146-window six-session surface, same matched
scorer, equal session weight; baseline re-score 0.47344 vs sealed 0.47272, max per-session
diff 0.0014):

| system (SWA) | within mean R² |
|---|---:|
| A `direct_t4_48` | **0.516273** |
| C `direct_t4_33` | 0.513452 |
| baseline spintshape 12-epoch | 0.47344 (sealed 0.47272) |
| B `z4_pretrain_then_t4` | 0.339155 |

| contrast | mean | median | n+ /6 | bootstrap 95% | sign test p (exact, 2-sided) |
|---|---:|---:|---:|---|---:|
| B − A (gate) | **−0.17712** | −0.15192 | 0/6 | [−0.23202, −0.12583] | 0.0312 |
| B − C (gate) | **−0.17430** | −0.14091 | 0/6 | [−0.22937, −0.12016] | 0.0312 |
| A − baseline12 | **+0.04283** | +0.05355 | 5/6 | [+0.02042, +0.06290] | 0.2188 |
| C − baseline12 | +0.04001 | +0.03973 | 6/6 | [+0.02463, +0.05444] | 0.0312 |
| B − baseline12 | −0.13429 | −0.09801 | 0/6 | [−0.19023, −0.08463] | 0.0312 |

**Verdict (frozen §11 table): curriculum gate FAILS — B−A and B−C each 0/6 sessions positive,
means ≈ −0.17.  A improves over the 12-epoch baseline by +0.0428 (5/6 positive) → "adopt the
long direct-T4 recipe for performance".  Z4 pretraining at the source-audit-frozen T_pre=15 is
strongly harmful to within transfer at matched T4 exposure.**  This closes the concrete
curriculum recipe (per the handoff: a negative result closes *this* recipe, not all ordering
hypotheses) and confirms the pilot's dual-track warning that the source-audit column is a weak
proxy for transfer.  Best new arm (A, 0.5163) remains −0.0587 below the A2 teacher-initialized
cited reference (0.5750, mean-level only, not session-paired).  A2 stays the performance
reference; the long-schedule finding (12→48 epochs, +0.043) is the surviving positive result.

**External matrix (descriptive, from the sealed GPU-0 audit receipts; compile receipt
`results/gate3_within_screen_v1/external_matrix_receipt.json`)**: native sub-M external R² —
spintshape_t4 SWA `+0.0902`, pv_t4 `−0.0164`, spintshape_z4 SWA `−0.2936`, large_z4 `−0.7367`,
bl_t4 `−0.9597`; `large_t4` pending (pipeline incomplete at compile time, marker absent).
Gate-4 external scoring of the new arms is a separate, later step.

**Gate 4 — performance-route external (sub-M) for arms A and C**
(`results/gate4_arm_external_v1/`, schema `tfpd_gate4_arm_external`, same authorization-gated
15-session external path and source-only normalizer injection as the Stage-1 extswa receipts;
matrix `final_external_matrix_receipt.json`):

| system | external native R² (15 sessions) | zero | wrong-pair | destroyed-act | n+ sessions |
|---|---:|---:|---:|---:|---:|
| **A `direct_t4_48` SWA** | **0.1610** | −0.0362 | −0.1293 | −0.1425 | 8/15 |
| C `direct_t4_33` SWA | 0.0775 | −0.2559 | −0.3353 | −0.2400 | 5/15 |
| spintshape 12-epoch T4 SWA (sealed) | 0.0902 | −0.2775 | −0.2498 | −0.2449 | — |
| spintshape 12-epoch Z4 SWA (sealed) | −0.2936 | — | — | — | — |
| large_z4 / pv_t4 / bl_t4 (sealed) | −0.7367 / −0.0164 / −0.9597 | | | | |
| A2 teacher-init (cited only) | 0.341 | | | | — |

External paired contrasts (15 sessions, matched scorer): **A − spintshape-12ep-SWA = +0.0708**
(13/15 positive, bootstrap 95% [+0.046, +0.094], exact sign p=0.0074) and **A − C = +0.0835**
(15/15 positive, [+0.057, +0.113], p=0.0001).  **Three-row external conclusion: the long
direct-T4 recipe (A, 0.1610) is the best teacher-free system externally and improves the
current spintshape SWA baseline (0.0902) by +0.071; A2 teacher initialization (0.341, cited
only, own authority) remains +0.180 ahead — teacher/source pretraining stays the performance
frontier, now labeled initialization rather than distillation (§12).**  Arm B was not scored
externally: Gate 3 failed the curriculum gate 0/6, so Gate-4 curriculum external scoring is
moot; B remains a sealed negative-result artifact.  `large_t4` external has since landed (GPU-0 pipeline marker present; native **−0.0848**, zero −2.1477 / wrong-pair −0.1018 / destroyed −0.2855): the complete six-cell matrix is recompiled in `results/gate4_arm_external_v1/final_external_matrix_v2_receipt.json` (sha `8cbc4a88…`). Ordering is unchanged: `large_t4` sits between `large_z4` and `spintshape_t4`, so no ledger reading above is affected.

**New reading admitted by the completed matrix — identity-source transfer asymmetry.** With
`large_t4` external now landed, the two single-source arms can be contrasted directly:
`large_t4` carries **carrier-only** identity, `spintshape_z4` carries **calibration-activity-only**
identity. Within, large leads by `0.2562 − 0.1985 = +0.058`; externally it leads by
`−0.0848 − (−0.2936) = +0.209`. **The gap widens ≈3.6× under subject shift**, and note the sign:
activity-derived identity is not merely weaker externally, it is *negative* (−0.2936), whereas a
markedly weaker topology carrying only the carrier loses less. Two constraints on citing this:
(i) it is **topology-confounded** — the arms differ in per-unit read-in, fusion, temporal topology
and parameter allocation — so it is a system contrast, not a causal estimate of identity source, and
(ii) the within legs come from the superseded epoch-window estimator while the external legs come
from the matched scorer, so the `+0.058` and `+0.209` are not on one estimator and their ratio is
indicative only. Report it as a direction, not a magnitude. Practical consequence: adding capacity
to the calibration-activity encoder carries a session-fingerprint risk and cannot be adopted on a
within-only gain.

## 3B. TFAP round: 000128 whole-model pretraining (2026-08-17, closed)

Authority: `docs/TFAP_CONTRACT_20260817.md` (+ sealed execution addenda; Stage-0 receipt
`8c789b3d…`, payload `7e60654c…`).  Arms: P-T4 (aligned-T4 pretrain) and P-Z4
(zero-masked mechanism control), both 48ep x 5,372 steps on the sealed 128 payload from
the canonical initial state (state sha `65bacb85…`), then strict-27 whole-model
strict=True fine-tune with arm A's exact recipe (1,628,400 steps each).  All four runs
TERMINAL, zero invariant failures, closures equal; P-Z4's pretrain phase held W_side/
exp_avg/exp_avg_sq elementwise exactly zero for all 48 epochs.

Stage-3 one-shot matrix (`results/tfap_stage3_v1/stage3_receipt.json`, sha `23666f3e…`;
arm A re-scored in the same engine — within 0.516273 and external 0.16105 match the
sealed Gate-3/4 values):

| system (SWA) | within (6) | external (15) |
|---|---:|---:|
| **arm A scratch 48ep** | **0.516273** | **0.161050** |
| P-T4 (000128 aligned-T4 pretrain) | 0.450562 | 0.153692 |
| P-Z4 (000128 zero-mask control) | 0.485213 | 0.098359 |

Paired contrasts: P-T4 − A external **−0.00736** (11/15 positive, CI [−0.1178, +0.0774]);
P-T4 − A within −0.06571 (0/6 positive); P-Z4 − A external −0.06269 (4/15);
**P-T4 − P-Z4 external +0.05533 (10/15 positive)** — the task-frame mechanism gap.

**Verdict (frozen dual hard gates): ENGINEERING GATE FAIL (mean −0.007 < +0.03);
MECHANISM GATE PASS (+0.055, 10/15); within noninferiority FAIL (−0.066 < −0.03);
diagnostics/g5/g6 pass → `BOTH_GATES_FAIL__STOP_ROUTE`: no seed expansion, no further
TFAP arms.**  Reading: 000128 whole-model pretraining does not transfer to the 688
external surface under this protocol (both pretrain arms land below the scratch
48-epoch baseline, and within-dev degrades too); the one surviving positive signal is
the controlled mechanism gap — aligned task-frame supervision, not generic pretraining,
separates P-T4 from P-Z4 (+0.055 external) — but that signal lives on a route that lost
to scratch overall, so it is recorded as mechanism evidence only.  Arm A remains the
deployed best teacher-free system.  (Execution note: the first P-T4 pretrain run was
terminated at epoch 32/48 by external process-group kill — infrastructure event, logged
in `execution_addendum_pt4_relaunch.json`, relaunch reproduced its trajectory
bit-for-bit.)

**A2 matched rescoring (2026-08-17, read-only; `results/a2_matched_rescore_v1_r1/a2_rescore_receipt.json`).**
The sealed A2 teacher-initialized references (T4/Z4, seeds 42/43/44, final epoch-12 checkpoints,
A2's own `load_frozen_model` reconstruction) were rescored under the TFPD matched scorer at A2's
deployment granularity (last timestep, /5.0 scaling): per-session values match the sealed A2
receipts with max |diff| = 0.00000 on all 12 cell/surface pairs.  A2_T4 means: within
0.585/0.572/0.575, external 0.318/0.337/0.384 (seeds 42/43/44); A2_Z4: within 0.283–0.336,
external −0.111 to −0.146.  At the matched last-bin granularity arm A scores within 0.5538 /
external 0.2604 (its full-window deployable numbers 0.5163/0.1610 remain the Gate-3/4 reference),
so paired A2_T4 − armA(last-bin): external **+0.0857** (8/15 positive, bootstrap 95%
[+0.0196, +0.1651]), within +0.0238 (5/6, [+0.0073, +0.0423]) — teacher initialization retains
a real external edge over the long scratch recipe, now under one scorer with per-session pairing.
A superseded buggy first compile (missing /5 scaling) is voided inside the r1 receipt.

## 3C. Population-sparsification round R / S2 (2026-08-18, closed)

Authority: `docs/WORKORDER_BEAT_A2_SPARSIFICATION_20260818.md`; Step-0 receipt `92b5cef8…`;
theta authority `83d799b1…`; score receipt `results/sparsification_score_v1/
sparsification_score_receipt.json` (sha `583b899b…`, post-audit scorer with receipt-loaded
governing bars).  Governing granularity: last-bin, variance-weighted, equal session weight.

| system (SWA, seed 42) | within | external | external (full-window, diagnostic) |
|---|---:|---:|---:|
| **D (2 heads + iid whole-unit dropout, sealed)** | 0.5697 | **0.4179** | 0.3086 |
| S2 (carrier-ordered sector dropout) | 0.5723 | 0.3767 | 0.2619 |
| arm A (scratch 48ep) | 0.5538 | 0.2604 | 0.1611 |
| R (elementwise control) | 0.5082 | 0.1877 | 0.1351 |

**Verdicts (frozen §7 rules):** `R_recovery = −0.461` → **whole-unit structure is materially
load-bearing** (R recovers none of D's governing gain; R−armA external −0.0726, 2/15 positive).
**S2-over-D gate FAIL**: S2−D external −0.0413 (5/15 positive, CI [−0.0731, −0.0089]) →
"sector-gap training recipe rejected"; S_perm is NOT run (conditional on the gate) and the
sector line stops.  S2 still beats arm A decisively (+0.1163, 13/15, CI [+0.0624, +0.1779]) —
that is D-family whole-unit-dropout value, not a carrier-sector mechanism.  A2 development
screen: S2 external 0.3767 clears the 0.3461 bar but within 0.5723 misses 0.5776 by 0.0053 →
screen fails (both absolute bars required); R fails both.  Honest boundary per §11: D's
mechanism is the existing SPINT `dynamic_dropout`, not a new invention; the one-axis table is
closed (D > S2 > armA > R externally).  DH at governing granularity collapsed (external
−1.1373, Step 0), so no S64 proposal.  Execution notes: both cells ran the exact shared
p-stream (identical `p_sequence_sha256` `e62fc92f…`), zero invariant failures; five user-audit
scorer gaps were fixed atomically before scoring opened.  Receipt caveat (disclosed, file
immutable): the `window_share_of_external` block field read the wrong per-session key and
reads 0.0 for all blocks — display-only; the window-weighted means themselves use the correct
key and stand.

## 4. Defect: the PV arm is invalid (carrier normalization)

PV's frozen math (`src/tfpd/population_vector.py:7`) computes
`dir_i = [a_i, c_i] / max(m_i, 1e-6)`, which is the unit preferred direction **only in raw rate
units**, since T4 is `[m·cos φ, m·sin φ, m, b]` (`unit_side_features.py:127, 926`). The real
datamodule z-scores each side-feature column (`unit_side_features.py:1539`, wired at
`multisession_datamodule.py:943, 984-995`), and `unit_side_features.py:916-925` explicitly
documents that per-column z-scoring "would rescale it into an ellipse and destroy the 'this pair
is a single angle' geometry" — which is exactly the decomposition PV performs.

`m` is a non-negative right-skewed modulation depth with source stats
`mean_m = 1.3432`, `std_m = 1.2352`
(`sua_exploration/cache/dandi688_subc_co_v1/side_feature_stats/3c08fc40fc0dd1a32ac5.npz`), so
`m_z < 0` for every below-average unit and `clamp_min(m_z, 1e-6)` maps them all to `1e-6`.
Measured over all 7,752 cached units in `.../dandi688_subc_co_v1/side_features/`:

| quantity | value |
|---|---:|
| units with `m_z ≤ 1e-6` (clamped) | 5,093 / 7,752 = **65.7%** |
| PV `\|dir_i\|` median | **3.24 × 10⁵** |
| PV `\|dir_i\|` p90 / p99 / max | 8.53 × 10⁵ / 1.28 × 10⁶ / 1.41 × 10⁶ |
| units with `\|dir_i\| > 10⁵` | 64.5% |

`basis` (`nn.Linear(2, 16)`) and the GRU therefore receive 10⁵–10⁶-scale inputs and saturate.
This also explains `pv_t4 (0.0012) < pv_z4 (0.0037)`: under the zero carrier
`dir = 0/max(0, 1e-6) = 0` exactly, so `pv = 0` and the model falls back to the well-scaled
`log1p(mass)` rate channel. **A carrier-driven model scoring below its own zero-carrier control
is a bug signature.**

Status: the transparent-baseline control is **missing, not negative**. Fix by inverting the
z-score inside PV (carry source `mean`/`std` as buffers) — do not remove the datamodule
normalization, which is correct for `bl`, `large`, and `spintshape`, all of which feed the
carrier into a learned affine map. Re-run required before any successor cell; see
`HANDOFF_LARGE_ROUTE_OPTIMIZATION_20260816.md` §2A and §8 #1.

## 5. Incident log (all disclosed in receipts)

- pv_z4 first run killed at 11/12 epochs by an OOM event under multi-job memory
  pressure; terminal re-issued `SOURCE_CELL_INCOMPLETE__ABNORMAL_TERMINATION`,
  superseded receipt retained; clean retrain in `_v1_r1` (terminal 12/12,
  closure drift disclosed: `src/tfpd/spintshape_module.py` was added during the
  retrain inside its source-closure glob).
- The first within-val evaluator artifact (per-batch R² averaging, −1e5-scale
  numbers) was deleted and re-minted with pooled per-session R²; the batch
  materialization OOM (exit 137) was fixed by trimming the calib tensor.

## 6. Pending

Not user-gated, no authorization needed:

1. **PV normalization fix and re-run** (§4). 18K parameters, CPU-feasible. This is the lane's
   missing simple-baseline control and the pre-gate for the analytic-cosine family. A corrected
   PV near `0.25` would invalidate `large`'s `0.2562`; near `0.05` it would establish the route's
   non-triviality for the first time.
2. **Checkpoint policy** — replace the epoch-5–12 window mean with a declared single-checkpoint
   policy, then re-mint the affected numbers. Re-evaluation only, no training.

User-gated:

2A. **D seeds 43/44 — the single blocker on any A2-superiority claim** (A2 is a three-seed reference
   with an external spread of 0.066: 0.3178 / 0.3367 / 0.3837). Deliberately deferred twice by the
   operator, most recently in favour of the AM/IM round. Deferring the seeds defers the *claim*, not
   the requirement; every teacher-free number in §3C–§3E is seed-42-only.

3. swap-v2 formal aggregate blocked by the four-terminal `torch_runtime`
   common-binding check (only difference: swap_t4 ran on CUDA_VISIBLE_DEVICES=1).
   Options prepared: amend the validator (torch/python common, CUDA index
   per-cell disclosed) vs rerun swap_t4 on GPU0.
4. TFPD external sub-M scoring: `scripts/run_stage1_external_eval.py` dry-run
   passed; awaiting `--authorize-target I_AUTHORIZE_TFPD_DEVELOPMENT_TARGET_ACCESS`.
   The first use should be the **degradation contrast** of handoff §8 #2 — `large_t4`
   (carrier-only identity) against `spintshape_z4` (calibration-only identity) — pre-registered
   as a mechanism contrast, since it is confounded by topology and cannot become an adoption
   claim.

Closed:

5. ~~spintshape cells (7/12 epochs at 09:21)~~ — terminal 02:43/02:54 UTC, 12/12 epochs,
   `closure_equal=True`. Five-arm within matrix complete (four valid arms plus the void `pv`).

## 3D. Sub-population invariance decomposition round T / C / G / W (2026-08-18, closed — see §3E)

Handoff: `HANDOFF_SUBPOPULATION_INVARIANCE_DECOMPOSITION_20260818.md` (supersedes the forward
parts of the R/S2 workorder). D bundles three factors — constant placeholder tokens (fc_in bias),
random global gain 1/(1-p) on survivors, varying surviving count; this round decomposes them.

**Step 0C (zero training; `results/subpop_step0c_v1/step0c_receipt.json`, SCORED).**
Native cross-check reproduced the sealed governing numbers with delta exactly 0.0 (path
bit-identical to the sparsification scorer).

Unit-loss robustness curve, external governing, seed-averaged, PRIMARY `zero_nogain`
(deployment semantics: units silently lost, survivors untouched):

| removed | arm A | D |
|---|---:|---:|
| 0 | 0.2604 | 0.4179 |
| 0.10 | 0.2158 | 0.3940 |
| 0.25 | 0.1349 | 0.3505 |
| 0.50 | −0.0777 | 0.2496 |
| 0.75 | −0.7940 | +0.0905 |

D's curve is far flatter: trained dropout bought genuine test-time unit-loss robustness. This is
the baseline curve C must beat ("flatter than D" is the pre-registered design claim).

Padding diagnostic (true absence via key_padding_mask) vs zero_nogain at f=0.75: D drops to
0.0002 (padding) vs 0.0905 (placeholder); arm A the reverse (−0.539 padding vs −0.794
placeholder). Each model degrades least under its own training-matched intervention — D's
robustness may be partly placeholder/sink-mediated. Cell T adjudicates.

Test-time subset ensembling (0C ii, the dangerous cheap baseline): D masked-member ensembles
K=4/K=8 reach only 0.366/0.386 (zero_gain) and 0.377/0.373 (zero_nogain) — all BELOW native
0.4179; arm A ensembles also all below native. Ensembling recovers masked-input performance but
never reaches native, so it cannot deliver what C promises. C's premise survives its cheapest
competitor.

**Cells in flight (Arm A contract verbatim, one change each, seed 42, 48 ep):**
- C paired-subset consistency (GPU0): two independent D-law whole-unit draws per step, both
  branches supervised + λ=0.1 behaviour-prediction MSE consistency (R-Drop over whole-unit
  perturbation); compute option "two-view same-batch" (2 forwards/step, step count identical to
  Arm A/D; pre-registered: small C−D requires a 96-epoch compute-matched D control before any
  claim). Skip floor 1 bites at E[p^N]=1/(N+1) ≈ 1.5%/branch (roster units 41–91); Jaccard mean
  ≈ 0.29 — the constraint bites.
- T true removal via `key_padding_mask` (GPU1 after 0C): no placeholder, no gain, min_keep 4;
  launch-time bitwise proof that an excluded unit's input is invisible to the output.
- G random global gain (queued after T on GPU1): p clamped [0, 0.95], all units ×1/(1-p),
  no masking — the claim-killer for "it's just input-scale augmentation".

Scoring: `scripts/run_subpop_score.py` (57 tests; verdicts machine-checkable: T trichotomy
|Δ|<0.01≈equal, G bands +0.08/+0.03, C conjunctive gate +0.03 & ≥10/15 & within ≥−0.03 with a
15-session fourth conjunct; references pointer-loaded from SHA-verified receipts; no bootstrap
escape). Curves for T/C/G reuse the 0C machinery with paired masks. Prohibitions honored: no
seed 43/44, no dropout-strength sweep, no 64-head revival, no S2 revival, no width change, no
pretraining/teacher, no silent behavior_scaling change.

**T and C landed and scored (receipt `results/subpop_score_v1/subpop_score_receipt.json`, scorer
rc=0; G/W pending at scoring time).**

Cell T — true removal via `key_padding_mask` (no placeholder, no gain, min_keep 4, zero invariant
failures, SWA `eb323aa2…`, p-stream SHA identical to R/S2 `e62fc92f…`):

- **Pre-registered trichotomy verdict: T_APPROX_EQUAL** (rule |T−D| < 0.01). T−D external
  governing −0.0099 (7/15), within −0.0035 (3/6); full-window +0.0026 (8/15).
- T−ArmA external **+0.1477 (15/15)**, within +0.0124 — true removal carries essentially D's
  entire gain (+0.1576, 14/15). **Placeholders are harmless: the effect is subset variation per
  se**, and the mechanism sentence upgrades to genuine set-invariance (handoff §3 T: "T ≈ D"
  branch). This also unblocks C's mechanism interpretation (§3 C item 5): the consistency
  constraint operates over sub-populations, not placeholder configurations.
- A2 screen: external 0.4080 clears 0.3461; within 0.5662 misses 0.5776 by 0.0114 → screen fails
  (as with D; A2 remains within-superior; seeds 43/44 still mandatory for any superiority claim).
- Curve asymmetry (recorded, deployment-relevant): T under the PLACEHOLDER intervention
  (zero_nogain) collapses at high removal (f=0.75 → −0.0404; zero_gain −1.027) while D holds
  +0.0905; under padding T is the robust one. Each training form buys invariance to its own
  ablation semantics. Since the deployed pipeline would deliver absent units as placeholder
  tokens unless the mask plumbing is shipped, **D remains the deployment-matched system; T is the
  mechanism adjudicator, not the replacement.**

Cell C — paired-subset consistency (λ=0.1 frozen, Jaccard mean ≈ 0.29, skip floor ~1.6%/branch):

- **Pre-registered conjunctive gate: FAIL** — C−D external governing **−0.0357** (4/15, CI
  [−0.061, −0.010]); within +0.0029 (4/6) noninferior; 15-session conjunct satisfied. Conditions
  failed: external mean ≥ +0.03 and ≥10/15 positive. Per the work order, no λ sweep on a failure;
  line closed as a negative result.
- C−ArmA external +0.1219 (13/15) — C still carries most of the dropout-family gain; the
  consistency term costs ~0.036 versus plain D. C−T external −0.0258 (6/15).
- A2 screen: external 0.3823 clears; within 0.5726 misses by 0.0050 → fails.
- Curve not flatter than D in any meaningful sense (f=0.75: C 0.0741 vs D 0.0905, and C starts
  0.036 lower at native).

Incident (fixed same hour): Cell W's first launch crashed at the launch proof —
`_raw_bytes_equal` called `.numpy()` on a CUDA tensor (CPU smoke could not catch it). Fixed by
`.cpu()` before `.numpy()` (py_compile + 13 tests re-run green); failed dir preserved as
`cellW_temporal_residual_launchfail_v0` (CELL_FAILED receipt); relaunched clean `CELL_LAUNCHED`.

Cell G — random global gain, the claim-killer control (receipt
`results/subpop_score_v1_r1/subpop_score_receipt.json`; T/C re-scored in the same pass, values
identical to v1):

- **Pre-registered band verdict: GAIN_NOT_THE_MECHANISM** (G−ArmA external < +0.03). G−ArmA
  external governing **+0.0095** (10/15); within **−0.0200** (1/6); G−D external **−0.1481**
  (1/15). G native external 0.2699 / within 0.5338 — gain augmentation recovers ~6% of D's
  +0.1576 external gain and slightly hurts within.
- **The population claim is clean.** With T≈D (placeholders harmless) and G≪D (gain not the
  mechanism), the decomposition of D's three bundled factors closes: the load-bearing factor is
  subset variation itself — training against randomly thinning the unit population — not the
  placeholder token mass and not input-scale augmentation.
- Realized gain stats across training: clamp rate ≈ 5.0% (theoretical U(0,1)>0.95), max gain
  19.999999999999982 = 1/(1−0.95) exactly; p-stream SHA identical to R/S2/T (`e62fc92f…`);
  zero invariant failures; SWA `b4545a43…`.

Round status after G: T (T_APPROX_EQUAL), C (gate FAIL, closed), G (GAIN_NOT_THE_MECHANISM)
adjudicated; W (temporal residual) in flight, auto-scores into `subpop_score_v1_r2` on landing.

Cell W — zero-initialized temporal latent residual head, K=8 (receipt
`results/subpop_score_v1_r2/subpop_score_receipt.json`; T/C/G re-scored, values identical):

- **Exploratory cell (no pre-registered gate): clearly negative.** W−ArmA external governing
  **−0.0902** (3/15); full-window external −0.1289 (3/15) — worse at BOTH granularities, so this
  is not a within-window-profile trade; W−ArmA within −0.0151 (3/6); W−D external −0.2478 (1/15).
  W native external 0.1702 / within 0.5387, both far below the A2 bars.
- The residual woke from exact zero as designed (|delta| mean ≈ 0.11, value_head max-abs ≈ 0.079
  by epoch 47; bitwise-equal to Arm A at initialization, proven at launch) — it learned
  something, but that something does not transfer across subjects. Per the directions handoff
  ("only if it works is an SSM refinement worth considering"), the temporal-head line stops
  here. Non-causal caveat stands as recorded.
- Run integrity: CELL_TERMINAL, 48 epochs, zero invariant failures, closure equal, SWA
  `e0822dd7…`. First-launch CUDA `.numpy()` crash fixed same hour and disclosed above.

## 3E. Round closure (2026-08-19)

`HANDOFF_SUBPOPULATION_INVARIANCE_DECOMPOSITION_20260818.md` is fully executed: Step 0C + cells
T, C, G, W all landed CELL_TERMINAL with zero invariant failures; three scoring receipts sealed
(`subpop_score_v1` T+C, `_r1` +G, `_r2` all four — cumulative, not corrective: every shared contrast
is bitwise identical across the three, though none carries a `supersedes` field); seeds 43/44 not run;
no sweeps. **Cell E is now closed unstarted** rather than held (below, item 5). The decomposition is
complete and one-sided:

1. **D's mechanism is subset variation itself.** Placeholder mass is harmless (T≈D, −0.0099),
   input-scale augmentation is not the mechanism (G, +0.0095 vs D's +0.1576), elementwise
   scattering is catastrophic (R, 0.1877). The paper's mechanism sentence: training against
   random whole-unit population thinning.
2. **Nothing beats D in this round.** C (consistency) −0.0357, W (temporal head) −0.0902 vs
   Arm A, S2 (prior round) −0.0413 vs D. D remains the best teacher-free system (external 0.4179
   > every A2 seed; within 0.5697 inside A2's seed spread). Test-time ensembling cannot replace
   any of it (0C ii: K=8 reaches only 0.386 < native 0.418).
3. **Deployment note:** each training form is robust only to its own ablation semantics
   (D/placeholder vs T/padding at f=0.75). The deployed pipeline delivers absent units as
   placeholder tokens, so D stays the deployment-matched system; T is the mechanism adjudicator.
4. A2 superiority remains claimable only after D seeds 43/44 (deliberately deferred; A2
   development screens: T 0.4080/0.5662, C 0.3823/0.5726, G 0.2699/0.5338, W 0.1702/0.5387 —
   none pass both bars, as with D).
5. **Cell E closed unstarted (2026-08-19), on C's evidence.** C made subset-invariance an explicit
   objective and it *cost* 0.0357 external (4/15, CI excluding zero), while 0C showed D's degradation
   curve is already flat. The reading: D buys robustness without invariance, and demanding invariance
   is harmful — plausibly because which units are present carries usable information, so forcing
   f(subset₁)=f(subset₂) forbids subset-specific evidence. E was the architectural form of that same
   demand (an aggregation whose expectation is independent of which units are present), so C is direct
   evidence against its premise. Reopening it requires explaining why architectural invariance should
   succeed where objective-level invariance measurably failed.
6. **Four design additions were attempted across this and the R/S2 round — S2, C, W, DH — and all
   four failed.** That is a pattern, not noise: the design space around the perturbation and around
   decoder capacity is exhausted at current evidence. The lane's asset is a mechanism result of
   unusual completeness (five converging controls), not a module.

Successor round: `HANDOFF_ACTIVITY_VS_IDENTITY_MASK_20260819.md`. Its question is the one
decomposition no cell has touched — every perturbation so far was applied to `src = x + id`
(`spint.py:445-455`), so activity and identity were always ablated *together*, and the identity in
question is calibration-activity-derived (`subpop_cells.py:322-325`), i.e. a session fingerprint, not
the analytic carrier. Cells AM (mask activity only) and IM (mask identity only) discriminate whether
D's mechanism is population thinning or session-fingerprint suppression. If IM ≈ D, item 1's mechanism
sentence must be rewritten.

## 3E. Activity-vs-identity mask decomposition AM / IM (2026-08-19, closed)

Handoff: `HANDOFF_ACTIVITY_VS_IDENTITY_MASK_20260819.md`. Question: is D's whole-unit ablation
effect activity ablation, identity (calibration-activity session-fingerprint) ablation, or joint?
Both cells share the exact D mask law and code path (`F.dropout(ones[B,N], p)` — gain inherited
from the kernel, global torch RNG; p stream PCG64(42), SHA `e62fc92f…` identical to R/S2/T/G) and
bit-identical mask sequences to each other; the ONLY difference is the application site
(`activity*mask + identity` vs `activity + identity*mask`). Sibling module
(`src/tfpd_lane/activity_identity_cells.py`); sealed files untouched.

| cell | forward | external | vs D | vs Arm A | within |
|---|---|---:|---:|---:|---:|
| AM — activity masked | `activity*mask + identity` | 0.2996 | **−0.1183** (2/15) | +0.0393 (8/15) | 0.4841 |
| IM — identity masked | `activity + identity*mask` | 0.1146 | **−0.3033** | **−0.1458** (2/15) | 0.4919 |

**Pre-registered matrix row: `JOINT_ABLATION_REQUIRED`** (both ≪ D, band |Δ|<0.03 not met by
either). Consequence per the handoff: the mechanism requires the unit token to be fully replaced
— neither component alone carries it. The population story stands in a stronger form; this is the
decomposition's floor. The identity-suppression headline hypothesis is REFUTED: stochastic
identity suppression without activity masking is actively harmful (IM below even Arm A), the
opposite of the fingerprint-ablation reading. The mechanism sentence "training on random
sub-populations" is unchanged and now twice-confirmed (T last round, joint requirement this
round).

Paper-facing wording must remain narrower than the shorthand above. The result shows that
identity-only masking is not a sufficient explanation or replacement for Cell D; it does **not**
show that calibration-derived identity contains no session fingerprint, and IM's larger failure does
not establish that identity is intrinsically more important than activity. Both AM and IM break the
attachment between the two halves of a unit representation. The licensed claim is:

> Whole-unit sparsification treats each fused activity-plus-identity token as an atomic population
> element. Removing both halves together is required for the observed transfer gain; perturbing only
> activity or only identity does not reproduce it.

This result directly constrains successors: any carried-over D law must be applied after complete
unit-token fusion and shared across time. It does not license independent activity-mask or
identity-mask branches.

Run integrity: both cells CELL_TERMINAL, zero invariant failures, closure equal (AM relaunched
once mid-run after binding a pre-edit module version — behaviorally identical bytes, disclosed;
~35 min GPU cost, restart reproduced the killed run's epochs bitwise). Receipts:
`results/aimask_score_v1/subpop_score_receipt.json` (supersedes `_r2` recorded; scorer 169
tests green). A2 screens: AM/IM fail both bars. Curves: AM moderately unit-loss robust
(0.2996→−0.0509 at f=0.75), IM flat but from a collapsed native (0.1146→−0.0307).

Standing constraint unchanged and now three times deferred: no A2 superiority claim until the
headline system has seeds 43/44.

## 7. TF-SR causal state-decoder round (2026-08-19/21, in flight)

The performance-first successor is `TFSR_B3ST4_DDROP_SEED42`: shared B3S plus normalized T4,
complete fused-token U(0,1) whole-unit dropout, two state-conditioned queries, and a causal
per-time set reader with a window-local GRU. It has 1,651,380 parameters versus Cell D's
3,510,842, but its analytic B=1/N=128 workload is approximately 1.710B MAC versus Cell D's
matched approximately 0.136B MAC. The model is therefore parameter-efficient but compute-heavy;
it reuses its weights over 50 serial time steps rather than increasing parameter scale.

Phase-D v1 failed before its first accepted checkpoint because the Adam `step` state is a scalar
tensor and the original digest path assumed a non-scalar tensor. That attempt is immutable and
non-resumable (`failure.json` SHA `997be82b...`). Phase-D v2 fixes only the tensor-digest/provenance
route. Its live 48-epoch source-only run began on GPU1 at 2026-08-19 22:24 HKT. Epoch 0 is accepted:
33,925 cumulative steps, 5,638.24 s, 6.01695 steps/s, all eleven critical-gradient blocks nonzero,
finite model/Adam states, exact launch closure, and no target/formal access (`epoch-00.json` SHA
`fa0a1535...`). Epoch 1 is also accepted (`epoch-01.json` SHA `7a5baa93...`): 67,850 cumulative
steps, 5,643.65 s for the epoch, 6.01118 steps/s, exact warmup learning-rate endpoints, all eleven
critical-gradient blocks nonzero, finite model/Adam states, and no target/formal access. Its observed
dropout stream remains consistent with the frozen law (`p` mean 0.49997; kept fraction 0.50026).
The run remains in progress; it has completed 37/48 epochs and has no scientific score yet. The
latest accepted receipt is epoch 36 (`epoch-36.json`, SHA `c6bc20f3...`): 1,255,225 cumulative steps,
5,943.54 s for the epoch, 5.70787 steps/s, mean source loss 0.35430, exact scheduled learning-rate
endpoints, all eleven critical-gradient blocks nonzero, and finite model/Adam state. Root
descriptor-reloaded the
0444 body/sidecar pair and reran the native frozen identity plus `validate_epoch_receipt` chain with
exit status zero; the same durable artifact also passed the prospective Phase-E read-only attach
seam, and the complete epoch 0--36 chain revalidated. Across epochs 0--36, source loss fell from
0.77017 to 0.35430 (−54.00%); epoch 36 is the current minimum, and the last-five mean is 0.36018
versus 0.66994 for the first five (−46.2%). Epochs 29--36 fall another 5.49%, with only the tiny
epoch-33 increase (0.36381 to 0.36418). This is healthy but noisy source optimization
evidence, not a transfer score or a
checkpoint-selection signal. A predeclared-receipt-only trend audit likewise provides no early-stop
case: epochs 32--36 have a 2.62% net reduction, and eight of the last nine transitions are negative.
The run
must therefore continue to its frozen 48-epoch terminal rather than export an in-memory state or
select an intermediate epoch. GPU1 continues the same original process with no restart or
target/formal access.

A separately authorized seed-43 replication began on physical GPU0 at 2026-08-21 12:45 HKT as
`TFSR_B3ST4_DDROP_SEED43`. Its immutable attempt, launch, and 100-step throughput pairs are 0444,
sidecar-valid, and all launch-bound Stage-0/Phase-C/Phase-D/seed-43 source bytes still match. The
measured 100-step rate is 8.95032 steps/s (11.1728 s), approximately 1.49x the seed-42 live rate;
the engineering ETA is approximately 50.5 h for 1,628,400 steps. Its first complete epoch is now
accepted (`epoch-00.json`, SHA `b689a14a...`): 33,925 cumulative steps in 3,868.71 s, 8.76908
steps/s, or 1.457x seed42 epoch-0 throughput. The measured full-epoch projection is approximately
51.6 h for 48 epochs. Mean source loss is 0.76845 versus seed42 epoch 0's 0.77017 (−0.22%); this is
strong early evidence against an acceleration-induced optimization discontinuity, not a transfer
score. LR endpoints are exact, all eleven critical-gradient blocks are true, model/Adam states are
finite, dropout accounting closes over 1,085,600 examples, and the frozen seed43 validator accepts
the complete receipt against a freshly rebuilt identity. Epoch 1 is also accepted
(`epoch-01.json`, SHA `af12afdb...`): 67,850 cumulative steps in 3,883.31 s for the epoch,
8.73611 steps/s, mean source loss 0.69344, exact LR endpoints, all eleven critical-gradient blocks
true, finite model/Adam state, and closed dropout accounting over 1,085,600 examples. Root reran the
native seed-43 frozen-route, throughput-authority, artifact-root, and `validate_epoch_receipt`
chain against the durable pair with exit status zero; this validation did not use the prospective
scorer. Epoch 2 is accepted (`epoch-02.json`, SHA `5fbb7400...`): 101,775 cumulative steps in
3,891.46 s, 8.71780 steps/s, mean source loss 0.69097, exact LR endpoints, all eleven gradients
true, and finite model/Adam state. Root independently accepted the same durable bytes through both
the native training artifact and the repaired prospective scorer attach seam. Epoch 3 is accepted
(`epoch-03.json`, SHA `ada01051...`): 135,700 cumulative steps, 8.75346 steps/s, mean source loss
0.61392, exact LR endpoints, all eleven gradients true, finite model/Adam state, and the same native
plus scorer-attach validation. Epoch 4 is accepted (`epoch-04.json`, SHA `91e9809b...`): 169,625
cumulative steps in 3,879.57 s, 8.74454 steps/s, mean source loss 0.56846, exact LR endpoints, all
eleven gradients true, and finite model/Adam state. Root descriptor-reloaded the same immutable pair
through both the native seed-43 identity/receipt validator and the prospective scorer's repaired
artifact-attach seam. Epoch 5 is accepted (`epoch-05.json`, SHA `8acbbe26...`): 203,550 cumulative
steps in 3,873.80 s, 8.75755 steps/s, mean source loss 0.62870, exact LR endpoints, all eleven
gradients true, and finite model/Adam state. The mean loss is 10.60% above epoch 4, but this is not
an anomaly by itself: seed42 rose 20.64% at the same epoch boundary under the same shuffled source
and dropout regime. Root revalidated the complete seed43 epoch 0--5 chain through both native and
prospective-scorer artifact paths. Epoch 6 is accepted (`epoch-06.json`, SHA `03babc1c...`): 237,475
cumulative steps in 3,884.41 s, 8.73363 steps/s, mean source loss 0.54803, exact LR endpoints, all
eleven gradients true, and finite model/Adam state. It reverses the epoch-5 rebound and establishes
a new seed43 low; the complete epoch 0--6 chain passes both native and prospective-scorer artifact
validation. Epoch 7 is accepted (`epoch-07.json`, SHA `5868858a...`): 271,400 cumulative steps in
3,875.85 s, 8.75291 steps/s, mean source loss 0.53773, exact LR endpoints, all eleven gradients true,
and finite model/Adam state. It is a second consecutive reduction and another new seed43 low; the
complete epoch 0--7 chain passes both artifact paths. Epoch 8 is accepted (`epoch-08.json`, SHA
`eb2a87b6...`): 305,325 cumulative steps in 3,880.80 s, 8.74175 steps/s, mean source loss 0.56587,
exact LR endpoints, all eleven gradients true, and finite model/Adam state. Its 5.23% rebound is
ordinary epoch-level noise under the shuffled source/dropout regime, not an anomaly gate; the full
epoch 0--8 chain passes both artifact paths. The overnight continuation through epoch 19 is also
accepted without a gap. The latest receipt (`epoch-19.json`, SHA `09f8ec29...`) records 678,500
cumulative steps in 3,865.91 s, 8.77543 steps/s, mean source loss 0.41992, exact LR endpoints, all
eleven gradients true, finite model/Adam state, and source-only boundaries. Epoch 19 is the current
seed43 minimum: loss is 45.35% below epoch 0 and 25.79% below epoch 8; the last-five mean is 0.43741,
with only the epoch-18 rebound among the last nine transitions. The complete epoch 0--19 chain
passes both native and prospective-scorer artifact paths. The run has completed 20/48 epochs with
no failure, checkpoint, terminal, SWA, target/formal access, or scientific score. GPU0 and GPU1 are
simultaneously occupied by seed 43 and seed 42 respectively, with no device overlap.

The dual-run engineering tradeoff remains favorable and stable. Seed42 epochs 24--26 average
5,873.08 s versus 5,685.28 s for epochs 0--23 (+3.30% duration after the overlap began; association
does not establish causality), while the current mean seed42 plus seed43 throughput is approximately
14.49 steps/s versus 5.97 steps/s for the earlier single-seed42 period (2.43x aggregate). Across all
accepted receipts, each route's RSS, peak CUDA allocation, and peak CUDA reservation are exactly
stable, providing no evidence of a slow memory leak. Continue both authorized runs; do not reclaim
the small seed42 slowdown by idling GPU0.

The early loss trajectories are not identical and must not be described as such. At epoch 0,
seed43's mean loss was 0.22% below seed42; at epoch 1 it was 5.42% above seed42 (0.69344 versus
0.65777); at epoch 2 it was 3.52% below seed42 (0.69097 versus 0.71615); at epoch 3 it was 1.19%
above seed42 (0.61392 versus 0.60671). Seed43 fell 11.15% from epoch 2 to 3 and 20.11% from epoch 0
to 3. At epoch 4 seed43 is 5.08% below seed42 (0.56846 versus 0.59891); it fell another 7.40% from
epoch 3 and 26.02% from epoch 0. At epoch 5 both routes rebound: seed43 rises 10.60% to 0.62870,
whereas seed42 rose 20.64% to 0.72249; seed43 is 12.98% lower at that epoch. Across epochs 0--5,
the mean losses are 0.66066 for seed43 and 0.67870 for seed42 (seed43 2.66% lower). The alternating
early cross-run order and common epoch-5 rebound are followed by seed43 epoch 6 falling 12.83% to
0.54803, 23.89% below seed42's same-epoch 0.72006. Across epochs 0--6, seed43's mean is 0.64457
versus seed42's 0.68461 (seed43 5.85% lower), and seed43 has fallen 28.68% from its epoch 0. These
seven points remain evidence against an obvious acceleration-induced optimization discontinuity,
not evidence of bitwise trajectory equivalence or superior transfer. At epoch 7 seed43 falls another
1.88% to 0.53773, 12.00% below seed42's same-epoch 0.61108; epochs 0--7 average 0.63121 for seed43
versus 0.67542 for seed42 (seed43 6.54% lower), and seed43 has fallen 30.02% from epoch 0. This
eight-point comparison remains optimization monitoring only:
seed and recurrent execution build changed
together, so it cannot isolate an acceleration effect and supplies neither a stop rule nor a
transfer claim. It reinforces the requirement to retain both seed and accelerated-build labels
through scoring. At epoch 8 seed43 rebounds 5.23% to 0.56587 and is 5.56% above seed42's same-epoch
0.53607; across epochs 0--8, however, the means are 0.62395 versus 0.65993 (seed43 5.45% lower), and
seed43 remains 26.36% below its epoch 0. The cross-run ordering therefore continues to alternate;
this ninth point changes neither the healthy-optimization judgment nor the non-causal boundary.

Root's cumulative receipt audit at this boundary passed all 37 seed42 epoch pairs and all 20 current
seed43 epoch pairs. Each sequence is gap-free from epoch 0, every body/sidecar is regular,
non-symlink, mode 0444 with exact digest content, cumulative/global steps equal
`(epoch + 1) * 33,925`, all model/Adam states are finite, all eleven critical-gradient groups are
true, and every boundary remains source-only with target/formal/scientific score false. This audit
does not replace the frozen per-route native validators at future natural gates. The Luna monitor's
response stream disconnected after the 2026-08-21 22:33 HKT check; this was a monitoring-channel
failure, not a training failure. At 2026-08-22 10:32 HKT Root reconstructed and natively validated
every intervening immutable receipt before resuming the monitor.

Seed 43 must be reported as **accelerated build v2**, not as a strict seed-only byte/execution
replicate of seed 42. It preserves the frozen model graph, B3S+normalized-T4 inputs, Cell-D dropout,
loss, Adam, LR schedule, 48-epoch budget, checkpoint policy, and final-four SWA, while replacing the
50-step recurrent inner loop with the throughput-v2 JIT-scripted implementation. The sealed
engineering receipt (SHA `3438b9a5...`) proves bitwise-equal first-step forward/loss and first-step
gradient inside the frozen `1e-6` band, but honestly records 20-step FP32 trajectory drift
(forward approximately `1.0e-4`--`1.3e-4`, gradient approximately `5e-5`). Consequently the
contract's inherited phrase “Only the seed differs” is too strong relative to its own build
disclosure; do not use that phrase in scientific reporting. Seeds 43/44 can be same-build v2
replicates if seed 44 uses v2. A seed42-v2 bridge is optional only if the final paper requires a
strict three-seed same-build estimator. The benchmark receipt itself is engineering-only and says
`authorizes_training=false`; it is acceleration evidence, not the source of the operator's launch
authorization.

Operational disclosure: the shared seed-43 log contains one pre-root fresh-root rejection at byte
offsets 0--570, followed by an exact 5,372-byte sparse NUL hole at offsets 571--5,942 and then the
live process's benign warning stream. The log was born at 12:45:52.816 HKT, the canonical root at
12:45:52.854, and the winning attempt receipt at 12:45:52.867. PID 13331 still owns stdout and
stderr descriptors on that inode at a later shared offset; it could not have resumed after the
other process's uncaught `RuntimeError`. This proves a duplicate pre-root contender lost the
canonical-root race while the current PID retained the immutable root. The losing contender did
not reach a result root, CUDA, optimizer step, or failure-receipt boundary. The live run remains
healthy; preserve the sparse log unchanged and report the duplicate attempt at terminal review.

The previous seed-43 Phase-E replication-addendum candidate (source/CLI/test
`4f16bc81...` / `5af8aa97...` / `bda6f3e8...`, closure `d6ed5dfb...`) is **superseded**. Root's
first real metadata-only attach against the live seed-43 artifact root exposed a material untested
seam: `_attach_readonly_seed43_artifact` called `train_43._directory_identity`, but `train_43`
does not export that helper. The attach therefore failed before terminal validation even though
17/17 mock-focused tests had passed. This was a future scorer implementation defect, not a defect
in the live training or its epoch receipts.

The bounded replacement is independently accepted as **code-preservation GO only**: source/CLI/test
SHAs `b4259919...` / `5af8aa97...` / `fae2e996...`, explicit 35-path closure
`ab1a200d7df4bdcc4cee564cd2de08368962bf4d5131ded6a6c523fd846a24fa`. It imports the exact
closure-bound frozen train-42 identity primitive, proves the seed-43 aliases refer to that same
module, and has no caller fallback. Root independently reproduced 19/19 no-user-site/no-CUDA tests,
py_compile, the closure, dry CLI, correct fresh roots, alias/symlink/inode-replacement rejection,
and the real metadata-only live epoch-01 attach plus native receipt validation. No seed-43 Phase-E
authority, evaluation NWB access, CUDA scoring, or result publication is authorized until both
training runs satisfy the ordered terminal prerequisites and root issues a later execution
capability.

The separate GPU0 synthetic throughput diagnosis is terminal and engineering-only (`results/
tfsr_b3st4_ddrop_throughput_engineering_v1/receipt.json`, SHA `c87444be...`; launch=final closure
`a5f56ddd...`; no data/checkpoint/target/formal access and `authorizes_training=false`):

| fixed N=128 cell | steps/s | samples/s | change from production B32 |
|---|---:|---:|---:|
| production eager B32 | 4.1397 | 132.47 | baseline |
| deferred receipt conversion B32 | 4.1553 | 132.97 | +0.38% steps/s |
| validation-hoisted B32 | 4.1766 | 133.65 | +0.89% steps/s |
| production eager B64 | 2.2149 | 141.76 | +7.01% samples/s |
| production eager B128 | 1.1470 | 146.82 | +10.83% samples/s |
| GRUCell diagnostic B32 | 4.1890 | 134.05 | +1.19% steps/s |

`torch.compile` is unavailable in the installed Torch/Inductor runtime (`builtins.TypeError` during
variant construction) and is recorded as such rather than treated as a model failure. The GRUCell
path is diagnostic/non-authorizing and is not numerically identical to the production GRU. The
fixed-N synthetic B32 projection (109.3 h) is deliberately not substituted for the measured live
projection (approximately 75--76.5 h), because real sessions have a different unit-count mix.

Decision: keep the live B32 scientific run unchanged. Receipt conversion, validation hoisting, and
GRUCell each recover only about 0.4--1.2%; B128 gains only 10.8% sample throughput while changing
batch/optimizer semantics. Restarting would discard valid progress and introduce a second scientific
factor. The observed cost is therefore architectural (50 serial attention/FFN/GRU updates), not a
parameter-count, memory, or bookkeeping bottleneck. Phase-E matched scoring remains blocked until a
valid Phase-D v2 terminal/SWA pair exists.

The Phase-E matched scorer is frozen at a no-data review boundary. Its current explicit 26-path
closure is `2772ec4b...`; the focused no-CUDA suite passes 71/71, and both canonical Phase-E
authority/result roots remain absent. Root independently reproduced the closure and 71-test result,
confirmed that the dry import does not load Torch, and verified that the execution route stops at the
missing Phase-D terminal without creating an authority or score root. The final pre-terminal
hardening removes all caller control over the within-6 asset map: the scorer descriptor-reads and
rechecks the fixed mode-0600 paired-view C1 manifest (`bb3440b6...`) and its exact six `val` rows.
Each physical input record also proves the raw M30 T4 row-to-SUA channel-axis correspondence and
binds the governing last-bin target, query-level validity mask, and valid count; both Cell-D and
TF-SR score paths recheck those arrays immediately before the governing R2 calculation.

A root pre-terminal readiness replay descriptor-verified all eight fixed authorities plus the
mode-0600 paired-view manifest, recovered exactly 6 within and 15 external sessions, and reproduced
the corresponding 6/15 A2 reference rosters. It imported no Torch, opened no NWB, and left both
Phase-E result roots absent. This is upstream-integrity evidence only; it does not substitute for the
still-missing Phase-D terminal/SWA validation.

Root also independently verified the actual sealed Cell-D SWA with the scoped
`weights_only=True` lazy-parameter allowlist, exact strict load, the two retained dead lazy keys,
and lazy-safe initialized-trainable accounting (Cell D 3,510,842; TF-SR 1,651,380). A separate CPU
synthetic audit loaded the scorer through the real CLI's `_tfsr_phase_e_static` package, constructed
its matching `NormalizedT4Batch` capability, and completed real Cell-D and TF-SR within-surface
forwards over six synthetic sessions with bitwise repeat, eval/no-mask, no-gradient, and
state-invariance proofs; CUDA remained uninitialized. This clears the implementation review
boundary only. It does not authorize authority publication, target access, or scoring before the
Phase-D v2 terminal/SWA exists.

## 8. Equal-session Cell-D source-sampling control (2026-08-20, closed)

The optional GPU0 successor `CELL_D_EQUAL_SESSION_SEED42`, proposed in
`HANDOFF_DEPLOYMENT_MATCHED_SESSION_BALANCING_20260820.md`, completed its full one-factor chain. Exact
Cell D was held and only source-session exposure changed from window-proportional to deterministic
equal-session sampling. Training completed 48/48 epochs and 1,628,400 steps; terminal SHA is
`cb0bc529...`, SWA artifact SHA is `512af5a7...`, and SWA state SHA is `e7f8959d...`.

The dedicated matched scorer first reproduced the full sealed Cell-D last-bin table exactly, then
scored both SWAs on identical no-cache inputs. Its immutable score SHA is `0c752bb4...` and terminal
SHA is `933b1ab9...`; verdict `STOP`:

| surface | sealed Cell D | equal-session | delta | paired signs |
|---|---:|---:|---:|---|
| within | 0.569685 | 0.566522 | -0.003163 | 2/6 positive; median -0.007223 |
| external | 0.417936 | 0.415124 | -0.002812 | 9/15 positive; median +0.008660 |

The external effect is heterogeneous rather than uniformly negative. Two sessions dominate the
negative mean: 20140627 (-0.223514) and 20150512 (-0.116988), while 20150626 improves by +0.121676.
Nevertheless, source-session balancing does not improve the governing average zero-shot score. Its
lower source training loss is not comparable evidence because the source objective itself was
reweighted.

Decision: close this route, do not replicate seeds 43/44, and do not spend another round tuning
session-sampling weights. The result is usable as a negative control showing that Cell D's transfer
gain is not explained by correcting source-session window-count imbalance. Full handoff:
`HANDOFF_CELL_D_EQUAL_SESSION_RESULT_20260820.md`.

## 3F. SO(2) round: rotation-augmentation arm B, equivariant arm C, matched control C′ (2026-08-25, closed)

Roadmap family B executed as three arms against sealed Cell D (arm A) and Arm A baselines,
same-engine live scoring, governing last-bin/equal-session convention (receipts:
`results/equivariant_v1/score_b_c/` and `score_cprime/`).

| arm | external | within | vs A (external) |
|---|---:|---:|---:|
| A (no perturbation) | 0.2604 | 0.5538 | — |
| **B = D + rotation augmentation** | **0.4140** | 0.5301 | **+0.1536 (13/15) PASS** |
| D (whole-unit dropout, sealed) | 0.4179 | 0.5697 | +0.1576 (14/15) |
| C (equivariant consumer, 492,951 trainable) | 0.0955 | ~0.159 | −0.3224 (1/15) FAIL |
| C′ (matched-capacity non-equivariant control) | −0.1733 | ~0.194 | −0.4336 (1/15) FAIL |

Findings (pre-registered gates + the review-mandated C′ decomposition):
1. **Augmentation factor redundancy**: rotation augmentation recovers ~97% of the dropout
   gain; B≈D (−0.004 external) means subset-thinning and frame-rotation marginalizations
   buy the SAME thing and saturate at the same external ceiling (~0.41–0.42, consistent
   with the frozen-D carrier oracle 0.4449). D stays Pareto-dominant (within +0.04 over B).
2. **C′ − A = −0.4336**: the disconnected small-consumer architecture is catastrophic
   regardless of symmetry — the capacity confound the operator review flagged is the
   dominant cause of the C family's failure.
3. **C − C′ = +0.2688** (exact mean-difference arithmetic over the two same-engine
   paired-vs-D receipts): **at matched capacity the SO(2)-equivariance constraint is worth
   +0.27 external** — the symmetry inductive bias is strongly positive, not the cost.
4. System verdict: soft symmetry via augmentation + full graph (B/D) strictly dominates
   hard equivariance + capacity cut (C). The equivariant-consumer architecture line is
   closed; the +0.27 matched-capacity effect is recorded as the honest positive residue.

Run integrity: B/C/C′ all CELL_TERMINAL, zero invariant failures, closure equal; C's
equivariance held at ~5e-8 through all epochs and post-SWA; C′ parameter parity with C
exact (185,793/492,951), non-equivariance proven (violations 0.17–0.24). Scorer 26/26
tests; engine-parity proofs recorded. Aux-head line closed the same day by probe+E8
(see sua_exploration/docs/RESULTS_BEHAVIOR_MANIFOLD_V2_20260825.md).
