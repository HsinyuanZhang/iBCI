# Revised work order: beat A2 with matched population sparsification

Date: 2026-08-18

Status: **EXECUTED AND CLOSED (2026-08-19).** Cells R and S2 landed and Step 0 completed; the
forward-looking parts were superseded by `HANDOFF_SUBPOPULATION_INVARIANCE_DECOMPOSITION_20260818.md`,
which has itself now executed and closed. Current open document:
`HANDOFF_ACTIVITY_VS_IDENTITY_MASK_20260819.md`. This file is retained for its Step 0 audit and its
R/S2 pre-registration; it authorizes nothing further.

Outcome of the one axis this work order tested: **R 0.1877, S2 0.3767, D 0.4179** external at the
governing granularity (last-bin, equal-session, variance-weighted, seed 42). R came in *below* Arm A
(0.2604), so coordinate-level corruption is not merely weaker than whole-unit thinning but harmful;
S2's gate was rejected, so carrier-ordered contiguity adds nothing over iid whole-unit removal. The
successor round added three more controls (G, T, C) that together isolate the mechanism to whole-unit
structure alone — see the decomposition handoff's §0.

Original status line, retained: authoritative revision. This file supersedes the earlier version at
the same path. Implementation and source-only preflight are authorized. Launch only the cells and
follow-up conditions defined here. Do not open formal/organizer-held data, select checkpoints on
target scores, or modify sealed Arm A, D, DH, or A2 artifacts.

## 1. Objective

Performance is primary. Test one clean training axis:

| cell | heads | removal structure | robustness being trained |
|---|---:|---|---|
| R | 2 | elementwise over `B x N x W` | coordinate-level corruption; no whole-unit invariance |
| D | 2 | existing iid whole-unit dropout | random population thinning/cardinality |
| S2 | 2 | carrier-ordered contiguous whole-unit dropout | task-space coverage gaps |

D is already sealed. R and S2 are the only new primary GPU cells. S2 must use the 2-head D graph;
target scores may not select a 64-head base. A later S64 may be proposed as a performance
combination, but it is not part of this one-axis mechanism comparison.

The intended paper contribution is not the existing SPINT `dynamic_dropout` flag. The possible new
method is carrier-ordered population sparsification with gain compensation. A carrier-specific
claim additionally requires the conditional shuffled-order control defined below.

## 2. Fixed authorities and bars

Bind exact immutable bodies and sidecars, not rounded values:

- A2 matched rescore:
  `results/a2_matched_rescore_v1_r1/a2_rescore_receipt.json`, body SHA
  `0ae0f74c6b5d606599d1108378836c8cbb7d1e5df05a453aa83f682255734d4e`.
- The old `results/a2_matched_rescore_v1/` is void and must be rejected.
- Arm A SWA:
  `results/admission_arms_v1/armA_direct_t4_48/swa_final4.pt`, SHA
  `920eb4c9827dbe66f98e110d9e4880d6f2ea4180fe9acc8ac30cacf3073f6ad0`.
- D/DH full-window score receipt:
  `results/pop_robust_v1/matched_score_receipt.json`, body SHA
  `91bf81f9bc33c32bb1e6a2472d465a6e3c8eb50fd36c3b249dfea4af27652aaf`.

The governing benchmark is variance-weighted R2 at the last timestep of each 50-bin window, with
equal weight per session:

```text
A2 pooled external = 0.3460880616472827
A2 pooled within   = 0.5776186750994788
Arm A external     = 0.2603564786414305
Arm A within       = 0.5538396437962850
```

Full-window and window-weighted results are required diagnostics but cannot rescue a failed
governing mean.

A seed-42 candidate cannot be declared superior to the three-seed A2 reference. Seed 42 is a
development screen. A final claim requires the same frozen cell at seeds 42, 43, and 44. For the
three-seed comparison, first average each session over seeds, then weight sessions equally. Also
report the three seed-matched candidate-minus-A2 deltas separately.

## 3. Step 0: zero-GPU work before launch

### 3.1 Rescore D and DH at last-bin

Read only:

```text
results/pop_robust_v1/cellD_2heads_dynamic_dropout/swa_final4.pt
results/pop_robust_v1/cellDH_64heads_dynamic_dropout/swa_final4.pt
```

Use the exact A2 `_r1` scorer convention: `matched_scorer.session_r2`, variance-weighted, equal
session weight, last bin, strict-27 source-only normalizers, semantic SHA
`f062506cb1db65e2a0872c55af2b542a9e3638fc5588e86735cd293dc890a391`.

This rescore is diagnostic and may guide a later S64 proposal. It may not change S2 from 2 heads to
64 heads.

### 3.2 Audit behavior-scaling parity

The A2 teacher uses `behavior_scaling_factor=5.0` and `predict_scaled_behavior=true`; deployment
divides its output by five. Determine and record the Arm A/D/DH convention. Do not silently change
R or S2: both must remain exact Arm A/D training-convention replicas so that R-D-S2 stays clean.

A mismatch does not block the internal R-D-S2 comparison. It remains a disclosed A2-versus-
teacher-free training-parameterization confound and prevents attributing the entire remaining A2
gap to sparsification.

## 4. Common training contract

R and S2 must otherwise reproduce Arm A/D exactly:

- standard initialization; no teacher checkpoint, distillation, feature matching, or pretraining;
- exact canonical Arm A initial tensor bytes, loaded with `strict=True`;
- 2 attention heads, identical state keys/shapes and active parameter count;
- strict-27 source roster, M30, normalized visible T4, task-only loss;
- 48 epochs, identical batch order and optimizer-step budget;
- warmup `1e-5 -> 1e-4` over two epochs, then cosine to `1e-6`;
- no gradient clipping; final-four epoch 44--47 SWA; seed 42;
- no target-based checkpoint, architecture, hyperparameter, or mask selection.

The sparsification site is the combined unit token after `src = activity + identity` and before
`fc_in`, matching the existing D implementation.

For every training forward:

- sample exactly one shared `p ~ Uniform(0,1)`;
- use gain compensation `1 / (1 - p)`, matching D;
- do not clamp `p` and do not impose `min_keep`;
- all-zero populations are allowed, because they are part of sealed D's treatment;
- when evaluation/scoring is active, the mask must be exact ones and the forward must be bitwise
  equal to the unsparsified path.

R and S2 must share an exact `p` stream and record its SHA. Additional mask/sector randomness must
use route-owned generators and must not perturb data ordering or unrelated model RNG. Because
sealed D did not record every random draw, describe the new cells as distribution-matched to D,
not as bitwise random-path-matched to D.

## 5. Cell R: elementwise matched control

R uses the D architecture with an elementwise Bernoulli mask over `B x N x W`:

```python
p = common_p_stream.next()
mask = elementwise_bernoulli_keep(src.shape, keep_probability=1.0 - p,
                                  generator=mask_generator)
src = src * mask / (1.0 - p)
```

Handle the exact `p == 1` edge as an all-zero tensor without division. Do not use a different
dropout location or a second probability draw.

R is a strong control, not a no-invariance strawman. It trains coordinate-level and temporal-token
corruption robustness but does not train removal of an entire unit token.

## 6. Cell S2: carrier-ordered sector sparsification

S2 always uses 2 heads and the D graph. It changes only which whole units are removed.

### 6.1 Raw carrier direction

T4 is `[a, c, m, b] = [m*cos(phi), m*sin(phi), m, b]`. Therefore:

```text
theta = atan2(raw_c, raw_a)
```

Never compute theta from z-scored `[a,c]`, and do not use `atan2(a,c)`. Prefer carrying an
immutable source-only theta authority aligned to canonical unit order. If normalization is
inverted instead, prove exact reconstruction from the frozen mean/std authority. Theta is a
masking authority only and is never concatenated into the model-visible token.

### 6.2 Undefined directions

Units with `raw_m <= MODULATION_EPS` have no meaningful preferred direction. Freeze this policy:

1. Let `V` be direction-valid units and `U` be undefined units.
2. For each batch example, draw total `k_drop ~ Binomial(N,p)`, matching D's realized count law.
3. Draw `k_valid ~ Hypergeometric(N, len(V), k_drop)`.
4. Draw an independent sector centre `phi_b ~ Uniform(-pi,pi)` for that batch example.
5. Drop the `k_valid` valid units with smallest circular distance to `phi_b`; break equal-distance
   ties by canonical unit index.
6. Drop the remaining `k_drop-k_valid` undefined units uniformly without replacement.

This preserves the total count distribution and avoids assigning all zero-modulation units the
fake direction zero.

### 6.3 Batch and gain semantics

`p` is shared across the forward, as in D. Sector centres and masks are independent across batch
examples, matching D's `B x N` mask independence. Survivors are scaled by `1/(1-p)`, not by
`1/(1-p_effective)`. No floor or clamp is allowed.

## 7. Quantitative decision rules

### 7.1 R interpretation

After Step 0, compute all quantities at governing last-bin/equal-session granularity:

```text
D_gain = D_external - ArmA_external
R_recovery = (R_external - ArmA_external) / D_gain
```

- `R_recovery >= 0.75`: generic elementwise regularization explains most of D; do not claim a
  population-sparsification mechanism.
- `R_recovery <= 0.50`: whole-unit structure is materially load-bearing.
- `0.50 < R_recovery < 0.75`: mechanism result is inconclusive.

Always report paired R-ArmA and R-D session deltas. The old full-window D gain `+0.1474` is a
diagnostic, not the governing denominator.

### 7.2 S2 interpretation

S2 is a material improvement over D only if all hold:

```text
external mean(S2-D) >= +0.03
positive external S2-D sessions >= 10/15
within mean(S2-D) >= -0.03
```

If `abs(external mean(S2-D)) < 0.01` and within is noninferior, treat S2 and D as practically
equivalent: cardinality invariance is sufficient. A positive delta below `+0.03` is promising but
does not establish the carrier-sector mechanism. A negative delta below `-0.01` rejects the
sector-gap training recipe.

### 7.3 A2 development screen and final claim

A seed-42 cell clears the A2 development screen only if both absolute means meet the exact pooled
bars in Section 2. This is not a superiority claim. Seeds 43/44 remain mandatory before saying the
frozen method exceeds A2.

## 8. Conditional controls and performance combinations

### 8.1 Shuffled-order sector control

Run `S_perm` only if S2 clears the S2-over-D gate. S_perm must match S2 exactly except that its
circular unit order comes from an immutable source-only random permutation keyed by session and
unit identity, not carrier angle.

- S2 > S_perm supports a carrier-specific geometry claim.
- S2 approximately equal to S_perm supports only grouped/sector corruption, not carrier geometry.

Do not claim that the carrier is load-bearing in the training procedure before this control.

### 8.2 S64 performance arm

S64 is not automatic. It may be proposed after S2 scoring if the 64-head last-bin diagnostic or S2
result makes the combination promising. S64 is a performance combination and may not be used in
the R-D-S2 one-axis causal table.

## 9. Required tests and receipts

Before GPU launch, tests must prove:

- R/S2 exact Arm A initial-state parity, graph parity, 2 heads, and parameter-count parity;
- exact sparsification site after identity addition and before `fc_in`;
- one shared `p` per forward and exact R/S2 `p_sequence_sha256` equality;
- R elementwise mask shape and S2 whole-unit mask shape;
- S2 per-example sector independence and seeded determinism;
- S2 total count follows the frozen Binomial construction in exact seeded fixtures;
- `theta=atan2(c,a)` from raw T4, normalizer invariance, canonical unit alignment, undefined-angle
  handling, and deterministic ties;
- gain is `1/(1-p)` and no floor/clamp exists;
- evaluation masks are exact ones and outputs equal the unsparsified parent bitwise;
- no target/formal data, gradients, optimizer steps, or checkpoint selection occur during scoring.

Every launch, terminal, score, and aggregate receipt must bind source closure and exact parent
artifacts and include:

- `num_heads`, mask structure, p distribution/stream SHA, generator namespaces, gain rule, and
  floor/clamp fields in the integrity block;
- per epoch p min/max/quantiles, realized drop/keep distributions, all-zero counts, maximum gain,
  post-mask token norms, gradient finiteness, and loss;
- for S2, valid/undefined direction counts and realized angular-sector width statistics per
  session;
- `n_windows` for every scored session;
- last-bin/full-window and equal-session/window-weighted results, clearly labelling the governing
  pair;
- 2014/2015 date-block results and `sub-M_ses-CO-20141203` separately;
- paired session deltas, sign counts, and descriptive bootstrap intervals;
- O_EXCL publication, regular non-symlink 0444 body/sidecar pairs, and launch-final-live closure
  equality.

No confidence interval or secondary aggregation may rescue a failed governing mean.

## 10. Execution order

1. CPU: rescore sealed D and DH at last-bin.
2. CPU: record behavior-scaling parity; do not modify R/S2 convention.
3. Implement route-owned R/S2 modules, tests, preflight, runner, scorer, and receipts. Do not edit
   shared SPINT or sealed Arm A/D/DH files.
4. Independent no-data/source-only review.
5. Launch seed-42 R and S2 in parallel on the two GPUs only after preflight GO.
6. Score both once on the frozen within/external development sets under both granularities and
   aggregations.
7. If S2 clears its gate, run conditional S_perm; otherwise stop the sector line.
8. Choose any seeds 43/44 replication only under the frozen rules above; do not tune the recipe.
9. Consider S64 separately as a performance combination, never as a substitute for S2.

## 11. Honest write-up boundary

- If R matches D, the result is strong regularization, not unit-cardinality invariance.
- If D beats R, whole-unit removal is load-bearing, but `dynamic_dropout` remains an existing SPINT
  mechanism rather than a new invention.
- If S2 beats D and S_perm, carrier-ordered population sparsification is the new training method.
- If S2 beats D but not S_perm, claim grouped-sector augmentation, not carrier geometry.
- The carrier remains necessary: A2 Z4 external is negative across all three seeds.
- The A2-versus-Arm A matched pooled external gap is `+0.0857315830`, not the old mismatched
  `+0.180` narrative.

Performance results come first. Mechanism cells remain bounded to R, S2, and the conditional
S_perm; do not expand into a dropout-strength sweep, width sweep, new pretraining route, or teacher
checkpoint training path under this work order.
