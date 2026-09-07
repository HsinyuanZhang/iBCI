# Design: Common-Mode Bound on Output-Side Adaptation, and the Soft-Posterior Carrier Successor

Date: 2026-08-29
Status: DESIGN v1 — for independent review
Authorization: no training, scoring, GPU launch, external-roster access, result-root
creation, or modification of any already dispatched experiment
Primary datasets: M2 and SUA/pseudo-MUA internal trial protocols
Working name: CM3 — Common-Mode Ceiling, Soft Carrier, Carrier-Normalized Credibility

---

## 0. One-sentence claim

The Cell-D output error at low label budget is approximately **92% common-mode**
across disjoint unit subsets and across adjacent time windows; this single
measured quantity bounds the entire output-side adaptation family (smoothing,
ensembling, per-bin reweighting, output-derived credibility) at roughly the
+0.042 zero-lag ceiling already observed, and it relocates the remaining
+0.176 M4-external oracle headroom to a **carrier estimation-under-label-uncertainty**
problem that cannot be solved by any function of this decoder's own output stream.

## 1. Why this document exists

Three separate lines converged on the same wall, and each was recorded as its own
local negative rather than as one mechanism:

1. the frozen-output causal filter line (LINE FROZEN, α=0.7 retained as a
   secondary configuration only);
2. the P2-prime smoothed-pseudo-label carrier line
   (`SMOOTHING_NO_CARRIER_VALUE_KEEP_OUTPUT_FILTER_ONLY`);
3. the continuity-probe within-surface null, whose own pre-registered verdict
   string already states that it "bounds the CEBRA-style training-time
   continuity prior".

This document does four things:

- states the common mechanism and the arithmetic that makes it a *bound*
  (section 3);
- fully discloses the unsealed probes that measured it (section 4);
- pre-registers the successor program CM3-0/1/2 with its gates (section 6);
- fixes the paper positioning so the contribution does not depend on any
  contrastive method winning (section 8).

It does **not** authorize any run. It does not edit, stop, restart, or reinterpret
any dispatched cell. AC3-0 remains dispatched under its own addendum authority;
section 4.3 of this document is a *preview* of AC3-0's zero-parameter rows and
changes none of AC3-0's arms, thresholds, or pre-registration.

## 2. Immutable evidence anchors

| Body | SHA256 |
|---|---|
| `results/continuity_probe_v1/continuity_probe_v1.json` | `8afaa9109f2dbeb1fb68e1044c4e7ae275771113c4cb3daf5c5565ea77c588b1` |
| `results/learned_gate_p2prime_v1/result.json` | `0b1264318c37b31fa34d655c1bf89989c6c97090c17821a092f2c4daa1971303` |
| `results/learned_gate_p2prime_v1/terminal.json` | `0ab2c2fef8c0c2e880ae47e57567921e1d5dd95aae721169cc64369d3f115b36` |
| `results/causal_dual_memory_cell_d_activity_only_quick_v2/result.json` | `48d658c866b0bcb45a9f03c6e0c84204ecb87286e940d850c846a46f67b33bb8` |
| `results/precision_aware_causal_dual_memory_cell_d_matched_score_v2/score.json` | `4e06ffc544210fbb7cba68c21fd86831c7d0a0407d43ef379a33b9a05bb2bcb1` |
| `results/causal_dual_memory_cell_d_matched_score_v8/score.json` | `98ea2bca22b4dbce6ac96b9b517a3774262115b62b6b0e15a1c242191633f77e` |
| `results/learnable_output_filter_v1_p5/r3_external.json` | `0c9d6c7b04181cf109d72e0bbad88b931187323c38c2b5d17ea71c7d921b5e04` |
| `results/learnable_output_filter_v1/terminal.json` | `16a3e932897b4f3398e901538e5d6150db2ed06aa3e28102d11bdfe69aaec2c8` |
| `results/calibration_gap_v1/p4_stream_stats.json` | `31fdd8cfd048849aa314a98f5b91d935c12e147073aca8018cab690479a1ae73` |
| `results/ac3_action_continuity_v0/trajectories.npz` | `e3512c9760b72af015ce6ae2ddd79fce79f2aa51e8c93b2240d438b67a5f3eb8` |
| `results/ac3_action_continuity_v0/materialize.json` | `7abd9e8ac128fe03361231151d27242140ad39a9039ef367db7da29d7ee3adde` |
| `docs/ADDENDUM_ACTION_CONTINUITY_CONTRASTIVE_CARRIER_20260829.md` | `1af34afc591f48ac8c514a9edbe5c209b2a9831a19c97c185280c43f47e9ff9d` |
| `docs/DESIGN_LEARNABLE_CAUSAL_OUTPUT_FILTER_20260829.md` | `314912233015579c9824695766049f1446befb2cc61f732f7a2e311a141c1ed4` |
| `scripts/probe_common_mode_unsealed_20260829.py` (section 4 authority) | `1c2d548d4defad50dbb62a520bc65ba41c247c54aed06500d1294ece1ebe3cf9` |

**Disclosed anchor drift.** The AC3 addendum cites the output-filter design at
`32508a28...`. That body is now `314912...` because section 17 (the operator
LINE FROZEN disposition) was appended after the addendum was written. The
addendum's own body is unchanged and still matches the SHA recorded in
`ac3_action_continuity_v0/attempt.json`. No result body is affected.

## 3. The mechanism

### 3.1 Continuity decomposes into zero-lag variance reduction and lag bias

The continuity probe ran two families over the identical frozen prediction
bodies: `causal_window` (trailing windows' own last-bin outputs; deployment-legal;
incurs lag) and `trajalign` (the K window-estimates of the *same absolute bin*;
zero lag; non-causal because only windows `t..t+K-1` contain bin `b_t`).
`W=50`, `stride=1 bin`, so `trajalign` at K costs `K-1` bins of latency.

Paired equal-session governing deltas over baseline:

| K (latency K−1 bin) | within-M10 zero-lag | within-M10 causal | within-M30 zero-lag | within-M30 causal |
|---|---:|---:|---:|---:|
| 2 | +0.007581 (6/6) | −0.000016 (4/6) | +0.009117 (6/6) | +0.004846 (5/6) |
| 4 | +0.016721 (6/6) | −0.016321 (0/6) | +0.019332 (6/6) | −0.006768 (1/6) |
| 8 | +0.028996 (6/6) | −0.085524 (0/6) | +0.031195 (6/6) | −0.083206 (0/6) |
| 16 | **+0.041768 (6/6)** | **−0.240889 (0/6)** | **+0.046009 (6/6)** | **−0.277287 (0/6)** |

Baselines: within-M10 `0.4677178959051768`, within-M30 `0.5665177305539449`.

Reading: averaging redundant estimates of one time point is **monotonically
beneficial and 6/6 positive at every K**; averaging estimates of *different*
time points is **monotonically harmful**. The two effects are close to additive.

Two facts constrain the whole family:

- the zero-lag family never exceeds **+0.046009**, and needs 15 bins of latency
  to get there;
- at **within-M4** — the budget with the largest headroom — the zero-lag family
  reaches only **+0.004591** (K16, 4/6) and its best arm is
  `trajalign_exp_a0.25` at **+0.006158** (4/6, CI `[-0.000242,+0.012145]`).

On external the ordering inverts (`causal_window_mean_K4` +0.019060 vs
`trajalign_mean_K4` +0.008447 at M10): where the decoder is already poor, paying
lag is cheap. That is the entire content of the sealed output-filter Pareto
result and it is why the operator retained α=0.7 as a configuration, not a method.

### 3.2 Why smoothing has exactly zero carrier value — an identity, not a failure

The completed-trial direction is `θ̂ = atan2(Σ_t v_t Δt)`. For any causal filter
`h` with unit DC gain (`Σ_k h_k = 1`):

```
Σ_t (h*v)_t = Σ_k h_k Σ_t v_{t-k} ≈ (Σ_k h_k)(Σ_t v_t) = Σ_t v_t
```

The filter and the integral **commute up to boundary terms of the filter support**.
P2-prime's `O1 − O0 = −0.000236` (M4 external) and `−0.000343` (M10 external) are
that boundary residual.

Consequence for interpretation: KC2 did **not** refute continuity priors in
general. It refuted one composition — a DC-preserving linear filter in front of a
linear integral. Any continuity prior that hopes to reach the carrier must fail to
commute with the direction functional. Section 4.3 tests the most natural such
construction and it also fails, for a different and deeper reason.

### 3.3 The measured common-mode fraction

The four complementary unit-group views share **no units**. Their velocity errors
are nevertheless correlated at

```
rho_group = 0.9171        (x: 0.9052 [0.8885, 0.9206];  y: 0.9291 [0.9230, 0.9429])
```

which is indistinguishable from the correlation implied for adjacent overlapping
time windows. The usable independent fraction is `1 - rho = 0.0829`.

**Therefore the error is not unit-sampling noise.** It is a shared/common-mode
component: model bias against the behavior it cannot represent, plus shared input
and shared readout structure. Disjoint neural inputs reproduce it.

### 3.4 What the mechanism bounds

1. **Output averaging is capped.** Only ~8% of the error variance is available to
   any averaging scheme. The observed zero-lag ceiling (+0.0418 within-M10) is
   consistent with that order of magnitude; the causally reachable part within
   session is ≈ 0.
2. **Group disagreement is structurally blind to it.** A disagreement statistic
   can only see the non-common component. This is a mechanistic prediction, and
   sections 4.2 and 4.4 confirm both of its consequences.
3. **Every output-derived credibility signal shares that blindness.** Trial-level
   ranking from the four group trajectories saturates near AUC 0.67 (section 4.4).
4. **Hence the +0.084 admission and +0.092 direction headroom at M4 external
   cannot be recovered from the decoder's own output closure.** Breaking it
   requires conditioning on something outside that closure.

**Caveat, binding on all citation.** `rho_group` was measured on per-bin velocity
pooled over trials on 6 sub-C source sessions at M4; the `trajalign` ceiling is a
governing last-bin windowed R2 on the within-6/external-15 surfaces. These are
different surfaces. The naive product `(1-rho)(1-R2)` matches within-M10
(predicts +0.0441 vs observed +0.0418) but fails at within-M4 (predicts +0.0443
vs observed +0.0046). `rho_group` is therefore admitted as a **mechanistic
explanation only** and must never be published as a quantitative predictor of the
ceiling. CM3-0 exists to replace it with a properly surfaced measurement.

## 4. Unsealed exploratory probes — full disclosure

All numbers in this section come from
`scripts/probe_common_mode_unsealed_20260829.py`
(SHA `1c2d548d4defad50dbb62a520bc65ba41c247c54aed06500d1294ece1ebe3cf9`)
reading only `results/ac3_action_continuity_v0/trajectories.npz`
(6 sub-C within source sessions, 1206 completed trials, 4 complementary group
views, M4 activity-only never-commit parent line).

**Status: UNSEALED, NON-GOVERNING, NON-CITABLE.** No external roster opened, no
result root created, no artifact written, zero target parameter updates, no
dispatched experiment touched. Every number must be re-derived under a sealed
work order before it may enter a paper, and each is reported with its weighting
convention because pooled and equal-session numbers differ.

### 4.1 Group output ensemble (Probe A/B)

| Quantity | Value |
|---|---:|
| single-group mean R2 | 0.413210 |
| best single group R2 | 0.470628 |
| 4-group ensemble R2 | 0.453121 |
| ensemble − mean group | +0.039911 (6/6 sessions) |
| ensemble − **best** group | **−0.017507 (2/6 sessions)** |

Zero-lag redundancy ensembling over disjoint unit sets does not beat simply using
the best single group, let alone all units. The predicted advantage of unit
redundancy over temporal redundancy **does not exist**.

### 4.2 Direction estimator: per-bin credibility weighting (Probe C)

Pooled over 1206 trials. Uniform integration is the best arm and every sharpening
of the weighting is monotonically worse:

| Estimator | mean\|err\| (rad) | 8-way snap mismatch |
|---|---:|---:|
| **uniform (plain integral)** | **0.5075** | **46.19%** |
| concentration¹ | 0.5098 | 46.19% |
| concentration² | 0.5122 | 46.52% |
| concentration⁴ | 0.5166 | 47.10% |
| concentration⁸ | 0.5225 | 47.68% |
| concentration ≥0.50 hard | 0.5082 | 46.52% |
| concentration ≥0.80 hard | 0.5173 | 47.10% |
| concentration ≥0.90 hard | 0.5267 | 47.84% |
| concentration ≥0.95 hard | 0.5419 | 48.26% |
| speed top 50% bins | 0.5079 | 46.19% |
| speed top 25% bins | 0.5113 | 46.43% |

This is the non-commuting construction demanded by section 3.2, and it fails.
Reason: down-weighting by group disagreement discards signal faster than error,
because the error being targeted is common-mode and disagreement cannot see it.
**Per-bin reweighting of the direction integral is closed.**

### 4.3 Direction baselines: R0 vs R-GE (Probe D) — preview of AC3-0 rows R0/R0.5

Pooled over 1205 usable trials:

| Row | mean\|err\| (rad) | snap mismatch |
|---|---:|---:|
| R0 pooled single group | 0.5581 | 48.55% |
| R0 best single group | 0.5049 | 44.48% |
| **R-GE** 4-group circular mean | **0.5116** | **46.89%** |
| R-GE − R0 | **+0.0465 rad** | **+1.66 pp** |
| R-GE − best single group | −0.0067 rad | — |

AC3-0's `REPRESENTATION_GATE` requires **+0.10 rad and +10 pp** over R0. The
mandated zero-parameter primary baseline reaches **less than half the rad
threshold and one sixth of the pp threshold**. Any contrastive or supervised arm
must roughly double the best zero-parameter improvement to clear the gate.

This is decision-relevant before spending a GPU cell, and it compromises nothing:
R0 and R-GE are deterministic zero-parameter estimators with no selection, so
previewing them cannot bias AC3-0's pre-registered comparisons.

### 4.4 Trial-level credibility ranking (Probe E)

Target: is this trial's snapped pseudo-direction correct? Base correct rate 0.5311.
Leave-one-source-session-out for the learned arm.

| Scorer | AUC | err@top-50% | err@top-25% | snap mismatch @top-25% |
|---|---:|---:|---:|---:|
| ρ_GE (zero parameter) | **0.6678** | 0.3524 | 0.2991 | **26.91%** |
| logistic, 14 label-free features (LOSO OOF) | 0.6659 | 0.3491 | 0.3261 | 29.24% |
| ORACLE ranking (leakage) | 0.9829 | 0.1657 | 0.0799 | 0.66% |

ρ_GE is genuinely and monotonically informative (quintiles 0.8372 → 0.5847 →
0.4741 → 0.3652 → 0.2967 rad; snap mismatch 66.8% → 56.4% → 48.5% → 36.9% →
25.7%; Spearman(ρ_GE, |err|) = −0.3494, p = 6.44e-36). But **13 additional
hand-crafted trial summaries add nothing** (AUC 0.6659 < 0.6678; per-session LOSO
AUC is better in 3/6 and worse in 3/6). ρ_GE also **saturates**: top-15% gives
26.1% and top-10% rebounds to 27.5%.

Two consequences, both load-bearing:

- even the best-ranked quartile carries a **26.91% wrong-direction rate**, so hard
  pseudo-label commitment is a mis-specified estimator at M4 — this is the
  argument for CM3-1;
- the oracle-to-deployable AUC gap (0.9829 vs 0.6678) is large, but the oracle
  ranks by *true* error and is therefore **not an attainable ceiling**. The
  attainable ceiling is the Bayes-optimal predictor of correctness given the
  observation, which is unknown and — by section 3.4(2) — may be close to 0.67.
  **The oracle gap must never be quoted as recoverable headroom.**

### 4.5 Common-mode rotation and cross-trial structure (Probe F)

Per-session circular mean direction error: `−0.1214, +0.1197, +0.3792, −0.0576,
+0.1202, +0.2020`; pooled `+0.1343 rad (7.7 deg)`; circular SD `0.3886–1.2319`.
The shared bias is real but small relative to dispersion.

| Arm (equal-session snap mismatch) | Value |
|---|---:|
| R-GE argmax | 45.38% |
| + label-free 8-fold grid alignment φ̂ | 45.31% (**+0.07 pp — fails**) |
| + Sinkhorn balanced assignment to the uniform 8-target marginal | **41.71% (+3.68 pp, 5/6 sessions)** |

Label-free rotation estimation fails (φ̂ has the wrong sign in 3/6 sessions).
Balanced assignment works because it exploits a constraint outside the output
closure — the known task marginal — and is the first construction in this document
that touches the common-mode component. It is nonetheless far below the 10 pp
gate and is admitted as an **ablation row only**, never as a headline.

Note this arm as specified is block-level and non-causal; a causal prefix variant
is required before it can be deployed, and the causal prefix estimate of φ was
measured as unusable (48.45% / 52.70% / 49.44% / 44.18% for prefixes of 8 / 16 /
32 / 64 trials against a 45.38% baseline).

## 5. Where the headroom is

M4 external ladder, all from sealed receipts:

| Row | Equal-session mean R2 | Delta vs predecessor |
|---|---:|---:|
| sealed static Cell-D | 0.11968413268526396 | — |
| activity-only CDM (A0) | 0.22649496145825285 | **+0.10681081 (14/15)** |
| + coherent oracle admission of raw pseudo (O0) | 0.3106506535531867 | +0.08415569 |
| + true completed-trial direction (O2) | 0.40261723531466814 | +0.09196658 |

M10 external: A0 `0.38880874036168755`, O0 `0.4726642707831387` (+0.08385553),
O2 `0.5043261323679273` (+0.03166186).

Budget-conditional allocation follows directly: **at M4 direction quality is the
binding constraint (+0.0920 vs +0.0842); at M10 admission is (+0.0839 vs +0.0317).**

Against that, the deployed carrier branch is currently worth **zero or less on the
governing equal-session mean**. Stated in each receipt's own sign convention:

| Contrast | Field / source | M4 external | M10 external |
|---|---|---:|---:|
| Precision-CDM V2 − activity-only | handoff audit §2.4 (derived; **no such field in `score.json`**) | −0.003865 (3/15) | −0.037807 (4/15) |
| activity-only − V8 full CDM | `activity_only_minus_v8_full_cdm` | **+0.035369 (3/15)** | **+0.048726 (9/15)** |
| Precision-CDM V2 − sealed | `paired_precision_v2_minus_v8_sealed` | +0.102946 (14/15) | +0.055546 (12/15) |
| V8 − sealed, M30 external | V8 `score.json` | — | −0.059092 (3/15) |

CM3-0 must re-derive the Precision-versus-activity-only contrast inside a sealed
cell, because it is currently a doc-level derivation rather than a receipt field
and it is the single most decision-relevant number in the program.

**Outlier nuance, load-bearing.** `activity_only_minus_v8_full_cdm` is positive in
the mean but positive in only **3/15** sessions at M4 external, because one session
contributes `+0.7699422016739845` (M10: `+0.6791759431362152`). For 12/15 sessions
V8 is the better arm; the mean favours activity-only purely through robustness to a
single catastrophic V8 session. "Activity-only beats full CDM" is therefore an
outlier-robustness statement, not a per-session one, and Precision-CDM's real
contribution was eliminating that outlier rather than raising the median. Any paper
claim here must report mean, median, and sign count together.

Precision gate external decisions: **M4 accept 19 / reject 276 / 295 total
(6.44%)**; M10 accept 532 / reject 282 / 814 (65.4%). At M4 the gate is
effectively reject-all, so Precision-CDM ≈ activity-only and the +0.0842 admission
value is **entirely unrealized**.

## 6. The CM3 program

### CM3-0 — Seal the common-mode bound (CPU, no GPU, no external roster)

**Purpose.** Convert section 3/4 from an unsealed diagnostic into a citable
mechanism result on the governing surface.

**Required outputs.**
1. `rho_group` on the **governing last-bin windowed** surface, per budget
   (M4/M10/M30) and per surface (within-6, external-15 sealed replay only —
   no new selection), with per-session values and session-resampled CIs.
2. The zero-lag/causal decomposition table of section 3.1 restated as
   `V(K) = trajalign(K)` and `L(K) = trajalign(K) − causal(K)`, with the latency
   cost of each K in bins **and in milliseconds** for each dataset.
3. The single-group / best-group / ensemble table of section 4.1 on the governing
   surface.
4. An explicit statement of the surface-mismatch caveat of section 3.4.

**Pre-registered gates.** CM3-0 is a measurement, not a horse race. It has no
performance gate. It **fails** if any of: `rho_group` on the governing surface
falls below 0.70 (which would invalidate the common-mode reading and reopen the
ensembling family); the sealed baseline rows fail bit-exact reproduction; or the
causal/zero-lag decomposition does not reproduce the anchored probe body.

### CM3-1 — Soft-posterior carrier update (CPU, primary performance cell)

**Purpose.** Attack the +0.0842 (M4) / +0.0839 (M10) admission headroom without
requiring any improvement in ranking, by replacing hard admission with a
correctly specified estimator under label uncertainty.

**Motivation from measured numbers.** The best deployable ranking admits a
quartile with a **26.91%** wrong-direction rate, and the current gate at M4
admits 6.44% of proposals. A binary accept/reject rule is mis-specified in both
directions: it either commits to a 27%-wrong label or discards the trial's
information entirely.

**Exact intervention.** T4 is an *encoding* fit (firing rate regressed on
direction). Under a posterior `p(c | trial)` over the K discrete reach
directions, accumulate expected sufficient statistics rather than hard-assigned
ones:

```
X'X  +=  Σ_c p(c|trial) · x_c x_cᵀ
X'r  +=  Σ_c p(c|trial) · x_c r
```

closed form, no backprop, no target gradient, no optimizer step. The existing
fixed-ridge solve and analytical parameter covariance are retained unchanged; the
Mahalanobis statistic is retained as a **diagnostic and safety rail**, not as the
admission decision.

**What must stay frozen.** Decoder graph and checkpoint; all decoder parameters
and buffers; source-only behavior and T4 normalizers; support selection and label
budget; B3S activity-memory input and transition sequence; neural query windows
and target arrays; last-bin governing metric and equal-session weighting; target
optimizer / backward / update counts all zero.

**Arm ladder.** Each arm differs from its predecessor in exactly one component.

| Arm | Posterior source | Update rule |
|---|---|---|
| S0 | — | activity-only, carrier never commits (the accepted parent) |
| S1 | — | current hard Mahalanobis gate (Precision-CDM V2 replay) |
| S2 | one-hot at the ρ_GE-ranked argmax | hard, ρ_GE-thresholded admission |
| S3 | softmax over K directions from ρ_GE-derived concentration | **soft EM accumulation** |
| S4 | S3 + outlier component with fixed prior mass | robust soft EM |
| S5 | S3 + causal balanced-assignment marginal constraint | soft EM + task structure |

S3 is the primary cell. S2 isolates whether the gain comes from soft weighting or
merely from a different threshold. S5 carries the +3.68 pp ablation of section 4.5
and requires the causal prefix variant.

**Pre-registered gates.** Primary estimand is `S3 − S0` on realized governing
equal-session R2 under exact coherent causal replay. Advance only if, on the
primary budget:

- source grouped held-session `S3 − S0 >= +0.010`;
- `S3 − S1 >= +0.005` (soft must beat the current gate, not only reject-all);
- `S3 − S2 >= +0.005` (softness must beat a re-thresholded hard rule, else retain
  the simpler hard rule and report that as the finding);
- no session regresses worse than −0.030;
- M30 does not regress below −0.010 (the V8 failure mode: `−0.059092`);
- target optimizer / backward / update counts remain zero and state digests
  reproduce.

Every CM3-1 delta reports **mean, median, and sign count together**, for the reason
given in section 5: a single session can carry an equal-session mean by a margin
larger than the entire effect being measured. A gate satisfied by the mean while
the median and sign count disagree is recorded as a null.

Any `0`-to-threshold outcome is recorded as a null **relative to the hard gate**,
even if both beat reject-all.

### CM3-2 — Carrier-normalized credibility encoder (conditional; 1 GPU cell)

**Entry condition.** CM3-1 S3 passes its source gates. CM3-2 is not authorized by
this document.

**Purpose.** Supply a `p(c | trial)` that is **not a function of this decoder's
output on this session**, which section 3.4 identifies as the only way to reach
the common-mode component.

**Why this is the only defensible CEBRA composition here.** CEBRA natively
requires a per-session encoder (`MultiSessionSolver`: 22,790 parameters across two
sessions, zero shared tensors; `UnifiedSolver`: input width fixed to the sum of
training unit counts and therefore unable to serve an unseen session). Both
violate the no-target-backprop constraint. The T4 carrier is exactly the missing
piece: it is a closed-form, session-invariant interface over a changing unit set.
**T4 makes a source-trained contrastive encoder deployable on unseen unit sets;
the encoder gives T4 a credibility estimate that is not output-derived.** Neither
component supplies both properties alone.

The AC3 v1 input contract (addendum §8.1) deliberately chose decoded trajectories
and excluded raw neural vectors. By section 3.4 that choice guarantees blindness
to common-mode error. CM3-2 therefore takes **carrier-normalized activity** as
input, not the decoded trajectory. This is the substantive design change.

**Feasibility already measured elsewhere in this repository.** Cross-session
behavior matching residual ratio for sub-M center-out: `1.069` (vel-2D) and
`1.067` (window) — matching is nearly free. Behavior-match residual by dimension:
`0.026` at d=2 versus `0.368` at d=7 — viable for M2/SUA 2D velocity, **not** for
H1 7-DoF. Discrete direction keying is `2.8×` worse than continuous
(`0.0051 → 0.0144`), so positives must be constructed on continuous
`[cos θ, sin θ]`.

**Mandatory counter-evidence, pre-registered.** The contrastive prior in this
repository is negative: InfoNCE monotonically degrades linear decodability
(`0.8084 → 0.7125` at 240 samples; `0.7522 → 0.6924` at 400); the cross-session
consistency arm is a **TERMINAL KILL** (frozen headroom `0.006188 < 0.10`); the
single real Track-B pilot cell scores `0.18901` (SUA, linear ridge) and `0.19844`
(pseudo-MUA, linear ridge) against a sealed carrier reference of `0.3061`; and
section 4.4 of this document measured a learned credibility model **losing** to
zero-parameter ρ_GE. Any contrastive loss must therefore sit on a projection head,
never directly on the readout coordinates.

**Arm ladder.** G0 zero-parameter ρ_GE (the incumbent to beat); G1 supervised
circular head on carrier-normalized activity; G2 C-Time positives; G3 C-Action
cross-session positives; GS shuffled-label control.

**Pre-registered gates.** The primary contrast is **ranking utility, not
representation quality**: realized `S3` R2 when `p(c|trial)` comes from the arm
versus from ρ_GE. Advance only if:

- the arm improves source held-session realized R2 by `>= +0.005` over G0;
- the arm beats `max(G0, G1)` — a contrastive arm that fails to beat the
  supervised head terminates the contrastive claim;
- GS degrades as expected;
- the improvement survives dropping any single source session;
- target update counts remain zero.

**Pre-registered null outcome.** If G1 beats G0 and G2/G3 fail to beat G1, the
disposition is `SUPERVISED_ROUTE_KEEP_CONTRASTIVE_CLAIM_TERMINATED` — a success of
the route, not of the contrastive mechanism, and it must be reported as exactly
that. This mirrors the AC3 addendum §23 amendment 3 and is the expected outcome
given the counter-evidence above.

## 7. What this document closes

| Route | Disposition |
|---|---|
| Output-side causal filtering as a main method | **CLOSED** (already LINE FROZEN; α=0.7 secondary configuration retained) |
| F3/F4/F5 filter ladders, adaptive gain | **CLOSED** (P4 OOF gain `+5.31e-05`, CI crosses 0) |
| Training-integrated filter route J0′/J1/J2/J3 | **CLOSED** as a performance route; spec retained as a document |
| Pseudo-label smoothing as a carrier mechanism | **CLOSED** by the identity of section 3.2 |
| Zero-lag redundancy ensembling over unit groups | **CLOSED** by section 4.1 (`−0.017507` vs best group) |
| Per-bin credibility weighting of the direction integral | **CLOSED** by section 4.2 (monotone harm) |
| Adding hand-crafted trial summaries to ρ_GE | **CLOSED** by section 4.4 (AUC `0.6659 < 0.6678`) |
| Label-free 8-fold grid rotation estimation | **CLOSED** by section 4.5 (`+0.07 pp`, wrong sign 3/6) |
| Balanced assignment | **ABLATION ROW ONLY** (`+3.68 pp`, below gate, needs causal variant) |
| Quoting the oracle ranking AUC as recoverable headroom | **FORBIDDEN** (section 4.4) |

Do not respond to any of these closures by adding a GRU/TCN/Transformer filter,
lengthening K, enlarging the feature search, or re-scoring external-15.

## 8. Paper positioning

The contribution is the **mechanism decomposition**, and it does not depend on any
learned component winning.

1. **Primary claim.** Low-budget cross-session decoder error is dominated by a
   common-mode component (`rho_group ≈ 0.92` across disjoint unit subsets), which
   bounds the entire output-side adaptation family. Presented with the zero-lag
   versus causal decomposition (+0.0418 zero-lag at 15 bins of latency versus
   −0.2409 causal at the same K, within-M10) and the analytical commuting-filter
   identity that explains the measured `−0.000236` carrier null. This reframes a
   widely assumed benefit — temporal smoothing of BCI decoder output — as
   redundancy averaging with a hard ceiling, and it is supported by 6/6 and 15/15
   sign counts.
2. **Secondary claim.** The relocated problem is carrier estimation under label
   uncertainty. Activity memory already delivers `+0.10681081` (14/15) at M4
   external; the residual `+0.084` admission and `+0.092` direction headroom is a
   semi-supervised estimation problem, and the correctly specified estimator is
   the soft-posterior accumulation of CM3-1, not a hard gate that admits 6.44% of
   proposals.
3. **Benchmarked arm, not headline.** The carrier-normalized contrastive encoder
   appears as one arm against ρ_GE and a supervised head, with its negative prior
   evidence stated and its null outcome pre-registered.
4. **Honest disclosure required in the paper.** The budget-conditional policy
   (carrier updates frozen at M30 after the `−0.059092` V8 regression); the
   external-versus-within Pareto structure of the output filter; the trial-structure
   dependence of every trial-level mechanism and its consequence for
   official/stream deployment where trial boundaries are not exposed; and the
   surface-mismatch caveat on `rho_group`.

Item 1 is publishable whether or not CM3-1 and CM3-2 succeed. That is the point of
this ordering.

## 9. Receipt requirements

Every CM3 cell records: parent work order and evidence SHAs; source session/fold
identities; exact trial and row membership digests; per-session raw and adapted
governing R2; paired deltas, sign counts, and session-resampled CIs; posterior
and responsibility digests; carrier proposal and commit counts; carrier state
before and after; Mahalanobis diagnostics retained even when not gating;
activity-memory transition digests; target optimizer / backward / update counts;
determinism evidence and repeated-run output digests; wall time, throughput, and
added latency; oracle and leakage flags where applicable; attempt-before-data /
model and atomic terminal-or-failure lineage.

CM3-2 additionally records encoder architecture and parameter digests, positive
and negative construction counts, shuffled-control results, and proof that no
target parameter was updated.

## 10. Stop conditions

Stop the CM3 program if any of the following occurs:

1. governing-surface `rho_group` falls below 0.70 (reopen the ensembling family
   instead);
2. CM3-0 fails bit-exact reproduction of the sealed baseline rows;
3. `S3 − S0` is null or negative on source grouped held sessions;
4. `S3` fails to beat `S1` or `S2` by the declared margins;
5. the soft update degrades M30 below −0.010;
6. any arm requires a target parameter update, target-selected hyperparameter, or
   external label to select a component;
7. trial and reset chronology cannot be proved for a trial-level mechanism;
8. an improvement cannot be separated from output filtering;
9. a larger model is required before a small one shows any downstream utility;
10. filter or carrier state becomes nondeterministic.

## 11. Interpretation map

| Outcome | Interpretation | Next action |
|---|---|---|
| CM3-0 confirms high `rho_group`; CM3-1 null | the bound is the whole result; carrier is unreachable from available signals | publish the mechanism; close the carrier branch |
| CM3-1 S3 beats S1 and S2 | admission was mis-specified, not merely mis-thresholded | replicate seeds/folds, then one external factorial |
| S3 beats S0 but not S2 | thresholding was the issue; softness adds nothing | retain the hard rule; report as a simplicity result |
| S5 beats S3 | task structure carries information the output closure lacks | develop the causal prefix variant |
| CM3-2 G1 beats G0; G2/G3 fail | `SUPERVISED_ROUTE_KEEP_CONTRASTIVE_CLAIM_TERMINATED` | keep the supervised route; terminate the contrastive claim |
| CM3-2 G3 beats max(G0,G1) | cross-session action correspondence carries non-output-derived information | one frozen external score |
| Everything null | the M4 headroom is not deployably reachable | publish the bound and the negative boundary |

## 12. Independent-review questions

1. Is `rho_group` on the governing last-bin surface the right formalization of
   "common-mode", or should it be a variance decomposition into shared and
   group-specific components with explicit confidence intervals?
2. Should the CM3-1 posterior be over the K discrete reach directions, or a
   continuous von Mises over direction with the discrete snap retained only for
   reporting?
3. Is `S3 − S2 >= +0.005` the right way to isolate softness from thresholding, or
   should S2 sweep its threshold on source folds first so the comparison is
   against the best hard rule rather than a single one?
4. Does the robust outlier component in S4 need a source-fitted mixing weight, or
   is a fixed pre-registered prior mass sufficient at M4?
5. For CM3-2, what exactly is "carrier-normalized activity" — the T4 projection
   of binned rates, the carrier residual, or a per-unit standardized rate in
   carrier coordinates? This is the main unresolved design question.
6. Is the balanced-assignment marginal defensible at deployment, given that the
   uniform-8-target assumption is a property of the internal center-out protocol
   and not of an official stream?
7. Should CM3-0 be published as a standalone mechanism note before CM3-1
   completes, given that its value does not depend on CM3-1?

## 13. Current disposition

```
GO:
    independent review of this document
    CM3-0 sealed measurement work order (CPU, no external selection)

HOLD:
    CM3-1 until CM3-0 passes and a separate work order exists
    CM3-2 until CM3-1 passes its source gates

STOP:
    output-side filtering, ensembling, and per-bin reweighting as performance routes
    hand-crafted feature expansion on top of rho_GE
    any citation of section 4 numbers without sealed re-derivation
    any modification of the dispatched AC3-0 cell
```

Final principle:

> Measure the bound before paying for a method. The common-mode fraction decides
> which mechanisms are reachable at all; only then does it make sense to ask
> whether a learned representation beats a zero-parameter statistic.
