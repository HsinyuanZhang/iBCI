# HANDOFF: the post-bilinear consumer extension (dynamic population observation)

**Date:** 2026-08-15
**Status:** research-direction handoff. Authorizes no code change, no data access, no GPU
launch, no target scoring, no formal evaluation. It does not amend the Stage-0 gate
contract, the Stage-1 matched-pair contract draft, or any sealed A2/RT/H1 endpoint.
**Parent:**
[`HANDOFF_TEACHER_FREE_TASK_FRAME_DIRECTIONS_20260815.md`](../../sua_exploration/docs/HANDOFF_TEACHER_FREE_TASK_FRAME_DIRECTIONS_20260815.md)
§5.1 (#1, #4, #6), §6 Priority 3, §7 kill criteria, §8.6.
**Precondition:** Stage-1 seed-42 BL/PV cells are still training at the time of writing.
Nothing in this file may be used to interpret, select, or rescue a Stage-1 outcome.

> ## Outcome note (2026-08-17): the Stage-1 cells this file was written ahead of have landed, and they do not support proceeding
>
> This document's roadmap — bilinear as the minimal H0, then dynamic attention as the
> "higher-capacity consumer", then shared SSM dynamics — was written before the arms were scored.
> Three results now bear on it directly.
>
> **1. The bilinear H0 did not establish a floor worth building on.** `bl_t4` reaches within 0.0767
> on the superseded epoch-window estimator and **external −0.9597**, by far the worst arm in the
> matrix. It is not a weak-but-sound baseline; it does not transfer at all.
>
> **2. The transparent PV comparator is void, so the "static vs dynamic aggregation" contrast this
> file is built around was never actually measured.** `population_vector.py` decomposes
> `[a, c] / max(m, eps)` on a **z-scored** carrier: 65.7% of units have `m_z` clamped to `1e-6`,
> giving median direction magnitude 3.24×10⁵. It passed Stage-0 only because `tfpd/synth.py` emits
> raw-unit carriers, so G5 does not transfer to real data. See §4 of
> `TFPD_RESULTS_LEDGER_20260816.md`. **The lane has no valid transparent baseline**, so no claim of
> the form "dynamic selection beats static pooling" can currently be made — in either direction.
>
> **3. The attention arm this file argues toward lost to a plainer alternative.** `large_t4`
> (carrier-conditioned set attention, static learned query slots) reaches within 0.2562 and external
> **−0.0848**, while teacher-free `spintshape` — shared per-unit read-in, calibration-derived
> identity, coupled additive identity port — reaches 0.4574 within and **+0.0902** external. Under
> the long-schedule recipe the same backbone (Arm A, 48 epochs + final-four SWA) reaches within
> 0.5163 / external 0.1610. So the ordering assumed here is inverted empirically: extra dynamic
> selection capacity did not beat a simpler consumer with a better identity path and a longer
> schedule.
>
> **What survives.** The framing sentence about task-frame observation is still the lane's best paper
> framing, and §1's point that bilinear is a static operator remains structurally true. But the
> specific progression — BL → dynamic attention → shared SSM — is **not** the route to schedule. The
> nearest live descendant of the "shared dynamics" idea is the temporal-latent-residual decoder in
> `HANDOFF_SPINT_DECODER_DIRECTIONS_20260817.md` §5 Priority 2, which targets the fact that the
> current decoder has no temporal model at all (two slots emit a whole 50-bin trajectory in one
> shot) while staying inside the topology that actually works. An SSM refinement is worth revisiting
> only if that residual pays.
>
> Also note: any successor here framed around decoupling attention keys from values must first read
> `sua_exploration/results/sua_t4_decoupled_kv_v1/aggregate_seed42.json` — key replacement measures
> `0.1392` against `0.5838` coupled.
**Review:** revised after an adversarial `cursor-grok-4.6-high-fast` review that found
four substantive errors in the first draft (value-mask mis-summary, PV-as-attention
category error, BL-vs-PV described as an isolating contrast, a wrong-pair kill rule that
contradicted parent §8.6) and a mis-specified zero-GPU pre-gate. All are corrected below.

## 1. What the frozen Priority-1 model actually commits to

The frozen bilinear read-in (`src/tfpd/bilinear_readin.py:110`) is

```
z_t = (1/N) * sum_i (U f(x_i[t-W:t])) (.) (V g(c_i)),     y_hat = D_psi(z)
```

Because the Hadamard product with a per-unit vector is a diagonal matrix, this is exactly

```
z_t = (1/N) * sum_i  diag(V g(c_i)) * U * a_it        where a_it = f(x_i[t-W:t])
```

i.e. a carrier-determined **linear operator applied to the per-unit activity features**.
`C(c)` is constant in `t` because the carrier tensor has no time axis
(`bilinear_readin.py:95`). Note the scope precisely: `f` is nonlinear
(Linear→ReLU→Linear→Tanh, `bilinear_readin.py:38-43`), so the map from *spikes* to `z` is
not linear. The linearity claim is about the map from **per-unit features** to the pooled
latent, which is the object this document is about.

Three properties follow, and they are structural rather than incidental:

1. **The pooling weights do not depend on activity.** `V g(c_i)` is a function of the
   carrier alone.
2. **The pooling weights do not depend on time or on decoder state.** `C(c)` is constant
   across `t`; `h_{t-1}` cannot influence what is read in at `t`.
3. **All cross-unit nonlinearity happens after pooling.** Each unit contributes
   additively and independently; no unit's contribution can be conditioned on another's.

Widening the frozen hyperparameters does not touch any of the three: `r` enlarges the
domain of `C`, `d` its codomain, and `k` the family of carrier-indexed maps. A larger
family of *static linear* pooling operators is still a static linear pooling operator.
The reviewer's natural objection — "just increase the rank" — is answered by this
argument and not by a capacity argument.

**Two things this section must not be read as claiming.**

- The frozen model is the *diagonal* special case `B_i = diag(V g(c_i)) U` of the parent
  handoff's generic per-unit read-in `B_i = H(c_i, stats(S_i))`
  (parent §3, lines 93-113). A full low-rank `B_i` is still static and still multilinear,
  and it is exactly parent Priority 3. **It is a rung this document must not skip.**
  Property 3 bounds the diagonal class; it does not show the static class is exhausted.
- The frozen `f` also drops the parent's support-set argument: parent §3 writes
  `a_it = f(x_i[t-W:t], S_i)`, the implementation uses `f(x_i[t-W:t])` alone. Support
  statistics reach the model only through the 4-dimensional carrier.

Finally, note what the frozen model already *is*: **a carrier-generated read-in operator,
not fitted per session, followed by shared temporal dynamics.** The "carrier defines the
read-in, dynamics stay shared" architecture is therefore already instantiated at
Priority 1. Any successor must state what it adds beyond that.

## 2. The SSM question

### 2.1 An SSM placed after the existing aggregation is a weak scientific increment

- By the time `D_psi` runs, unit-resolved information is already destroyed. The temporal
  model cannot recover selection, competition, or correspondence it never received.
- The T4/Z4 and aligned/zero/wrong-pair contrasts are all determined inside `C(c)`. An
  SSM cannot move them by construction.
- Novelty consisting only of which standard block occupies a fixed slot matches kill
  criterion parent §7-1 in spirit.

**But mechanism-immobility is not the same as worthlessness.** The house primary endpoint
is *absolute* external T4, not a mechanism delta. If the read-in is adequate and the GRU
is the binding constraint (long-context drift, trial-reset artifacts), an SSM could move
the absolute endpoint while moving no contrast. That would be a real engineering result
and should be reported as one, not as the method.

Two arguments can justify an SSM on its own terms:

- **Deployment horizon.** Continuous streaming without trial resets, where a stable
  linear recurrence beats a GRU over very long contexts. Engineering claim.
- **Analytic composability.** With linear-Gaussian dynamics and a *linear* carrier-generated
  observation, a Kalman/ridge target-prefix solve exists in closed form — parent
  Priority 5 / the A10 lane.

### 2.2 Three routes, not two

The first draft posed a binary between state-conditioned cross-attention and a linear
SSM. That was a false dichotomy: it erased the route that most literally implements
"the carrier defines the observation".

| route | read-in | dynamics | Tier-2 closed form | read-in depends on state |
|---|---|---|---|---|
| A: discriminative attention | `softmax_i(q(h_{t-1}) . k(c_i, a_it))` | any shared nonlinear (GRU suffices) | no | yes |
| B: static carrier-linear | `C(c)`, up to full low-rank `B_i` | linear SSM | yes | no |
| C: generative task-frame | `x_i(t) | z_t ~ Pois(softplus(b_i + m_i <u(theta_i), z_t>))`, carrier supplies `(theta_i, m_i, b_i)`; read-in is the Bayesian inverse | shared (switching-)linear or exponential-family SSM | yes, piecewise | **yes, and derived rather than learned** |

Route C deserves emphasis because it is the one the program's own synthetic generator
already assumes (`TFPD_STAGE0_SYNTHETIC_GATE_CONTRACT_20260815.md:55-56`). Under a
Poisson cosine observation model, a unit's informativeness is automatically state-dependent
through its instantaneous rate and tuning slope; reliability weighting emerges as inverse
noise variance instead of being bolted on as a learned gate; population mass enters as
total Fisher information rather than as an ad-hoc extra channel; and per-unit
correspondence is definitional. It also retains the closed-form Tier-2 adapter.

The genuine tension is that **Route A forfeits the closed-form adapter that is the
cleanest technical reason to want an SSM at all.** "Cross-attention observation operator
followed by shared SSM dynamics" is Route A with a block substitution downstream. That
should be a deliberate choice, not a phrase.

## 3. What a dynamic read-in would actually add, and what is already being tested

Against the frozen model, attention over the unit axis adds four separable things. They
are separable, so they should not be bought as one bundle.

1. **Activity-dependent weights.** The score depends on `a_it`, not only `c_i`.
2. **Normalized competition.** A softmax over units makes contributions compete for a
   fixed budget; the bilinear mean has no competition at all.
3. **State-dependence.** A query derived from `h_{t-1}` makes the read-in
   `C(c, x_t, h_{t-1})` and leaves the multilinear class entirely.
4. **Multiple read-out slots.** Several heads probe the population in parallel instead of
   one fixed `d`-dimensional projection.

Factor 3 is the only one with no cheaper implementation — but note Route C obtains it
*generatively*, and a state-conditioned diagonal gain obtains it *without competition*
(§5.4). Attention is not the unique carrier of factor 3.

### 3.1 The PV baseline is activity-dependent pooling — it is not attention

`population_vector.py:72-79` computes

```
score_i(t) = conf(c_i) * softplus(f(x_i));   w_i(t) = score_i / (sum_i score_i + eps)
value_i    = [a_i, c_i] / max(m_i, eps)
```

This has activity-dependent weights (factor 1) and a normalized budget (factor 2). It
has **no query**: there is no `q . k` and no state input, so it cannot be written as
cross-attention, and describing it that way would manufacture a false continuity with the
attention ladder. It is a Georgopoulos population vector with a learned amplitude and a
learned scalar confidence.

### 3.2 BL vs PV is not an isolating contrast

The Stage-1 contracted primary is `TFPD-BL-T4` minus `SPINT-shape-T4`; PV is the
simple-baseline floor (`TFPD_STAGE1_MATCHED_PAIR_CONTRACT_DRAFT_20260815.md:36-39`). BL
and PV also differ in feature dimension (`r=16` Tanh vs `r=1` softplus), carrier map
(learned `V g(c)` vs analytic direction plus scalar confidence), value space, GRU input
width (32 vs 17), and the presence of the `log1p(mass)` channel. A PV win therefore does
**not** isolate "activity-dependent pooling beats static pooling". Read Stage 1 for what
it was contracted to answer; the isolating arm is §5.4's rung, which is precisely why
that rung exists.

## 4. Prior negative evidence: what it says, and how far it transfers

Three results are routinely cited against "attention consumes carrier information
better". They must be quoted accurately, and their transfer boundary stated.

### 4.1 Value-mask (`results/carrier_value_mask_v1/terminal_mask_aggregate.json`)

| arm (external sub-M) | delta from matched parent |
|---|---:|
| `low_t4` (prune bottom 25% by `m`) | `+0.002082` |
| `random_t4` (prune random 25%) | `-0.062221` |
| `high_t4` (prune top 25% by `m`) | `-0.278777` |

Official verdict: `VALUE_WEIGHTED_LOW_GAIN_MASK_POSITIVE__DESIGN_SEPARATE_TRAINED_ROUTE`.

The correct reading is **the opposite of "selection is worthless"**: the `m`-ranking is
strongly load-bearing (`-0.279` to remove the top quartile, `-0.062` for a random
quartile of the same size), and only the *residual* bottom quartile is free to discard.
A2 already prices static quality ranking through its T4-painted identity. That is exactly
why entmax was demoted (`HANDOFF_DECODER_DIRECTIONS_20260815.md:128-133`): the mass a
learned veto could reallocate is small **because the ranking is already applied**, not
because ranking does not matter.

### 4.2 SetKV (`sua_exploration/results/setkv_delta_forward_v1/terminal_forward_aggregate.json`)

Two distinct facts, often conflated:

- **Pairing null:** T4 versus RS4 (row-permuted) attachment differed by `2.9e-9` external.
  That consumer used the carrier as a population codebook, not per-unit routing.
- **Crash versus parent:** `setkv_t4` lost `-0.738673` within and `-0.638865` externally.

### 4.3 Transfer boundary (binding)

All three probes were **forward-only interventions on the sealed A2/SPINT consumer**,
which still carries an additive identity port and a teacher-initialized decoder. TFPD has
no teacher, no additive port, and a read-in that *structurally* pairs `c_i` with unit `i`
— and its Stage-0 G5 pairing effect already passed on synthetic data (bilinear query-mean
aligned `+0.803`, zero `-0.285`, wrong-pair `-0.604`; aligned minus wrong-pair `+1.407`,
`tfpd_exploration/results/stage0_synthetic_gates_v1/receipt.json`). On that synthetic
population a mis-paired carrier is actively *worse* than no carrier. These results are
therefore **a prior to disclose, not a gate to pass**. In particular, SetKV's pairing null
is a fact about a codebook consumer and must not be used to constrain a model whose
architecture makes pairing structural. The first draft of this document let these three
results select TFPD's next architecture; that was a methodological error.

What survives as an honest caution is narrower: on this data, *static* unit ranking is
already largely exploited, so a new consumer that only re-derives static ranking should
not be expected to clear `+0.03`.

## 5. Proposed sequencing: cheap kills before capacity, and do not assume the aggregator

The first draft assumed the binding constraint is the aggregator. Parent §4.3 (lines
167-169) already names the competitors — neural nonstationarity, normalizer transfer, T4
estimation noise, task ambiguity — and they must be priced before an aggregator arm.

### 5.1 Read Stage 1 for its contracted question (zero cost)

Report BL-T4 against SPINT-shape-T4 and the sealed A2 reference, with PV as the floor.
Do not report BL-vs-PV as a static-versus-dynamic pooling experiment (§3.2).

### 5.2 Is the aggregator even the bottleneck? (cheap, mostly CPU)

Three diagnostics, any of which can invert the whole ladder:

- **Activity encoder.** Replace `f` with a capacity-matched per-unit temporal conv, or
  restore the parent's dropped support-set argument `f(x, S_i)`, with pooling frozen. A
  gain here means the bottleneck is per-unit encoding, not pooling.
- **Carrier quality.** Score the frozen consumer with a bootstrap or longer-prefix T4
  against the production 30-trial T4. A gain means the bottleneck is estimation noise.
- **Interface shift.** The MATCH session-moment pre-gates
  (`HANDOFF_DECODER_DIRECTIONS_20260815.md:58-88`) act *before* any aggregator and are
  already specified as zero-GPU.

### 5.3 Next static rung before any attention arm

Full low-rank `B_i = H(c_i)` (parent Priority 3), still static, still multilinear, no
softmax, no state-dependence. This is the rung the first draft skipped. If the diagonal
model's ceiling is really a pooling-class ceiling, this arm should not close the gap; if
it does close the gap, the attention lane was never needed.

### 5.4 Rungs for state-dependence, cheapest first

- **State-conditioned diagonal gain (FiLM on `h`).**
  `z_t = (1/N) sum_i (U a_it) (.) (V g(c_i)) (.) sigma(W h_{t-1})`. Buys factor 3 with no
  competition, no softmax, no variable-`N` normalization hazard, and a one-line ablation
  against frozen BL.
- **Generative Route C.** Carrier supplies `(theta_i, m_i, b_i)`; inference under a
  Poisson cosine observation model with shared dynamics. Buys factor 3 through the
  likelihood, plus derived reliability weighting and a Tier-2 closed form.
- **Dynamic selection without state-dependence.**
  `w_it = softmax_i(q . k(c_i, a_it))` with `q` learned and fixed, values `W_v a_it`, plus
  an explicit unnormalized mass channel. This is the arm that actually isolates factors
  1+2 with a learned value space — the thing §3.2 says Stage 1 cannot tell us.
- **State-conditioned cross-attention.** Only after the above, and only with the Route A
  cost of §2.2 accepted explicitly.

### 5.5 Zero-GPU pre-gate for the time-varying premise (respecified)

The first draft proposed comparing the best static per-unit weighting against an oracle
time-varying weighting proportional to the alignment between each unit's preferred
direction and the instantaneous behavior direction. **That pre-gate was wrong in both
directions and is withdrawn.**

- It **leaks**: weights built from the true `y_t` put the target inside the feature, so a
  large gap is unattainable by any causal model and cannot license a lane.
- It points the **wrong way**: for a Poisson cosine unit the Fisher information about
  movement direction is `~ m^2 sin^2(theta - theta_pref) / lambda`, maximal at
  `+/- 90` degrees from the preferred direction, minimal at alignment. An
  alignment-weighted oracle upweights the locally least informative units, so it can fail
  while the real mechanism succeeds.

Respecified pre-gate:

1. **No target in the weights.** Time-varying weights come from the frozen model's own
   causal estimate `y_hat_{t-1}` — either a Fisher/slope weight evaluated at
   `y_hat_{t-1}`, or a learned key. A true-`y` oracle may be reported separately as a
   leaked upper bound and can never be the kill.
2. **Pre-register both estimators.** Static: nonnegative ridge (or the frozen `V g(c)`
   refit) on source, frozen, then scored on query. Time-varying: identical value map and
   head, only the weights change.
3. **Score the Stage-1 surfaces, including external sub-M.** If target scoring is
   refused here, the `+0.03` house threshold may not be quoted.
4. **Synthetic pass/fail contract, two constructions** (house discipline requires a gate
   demonstrably able to both pass and fail): (i) an isotropic cosine population, where a
   Fisher-weighted oracle must beat static and an alignment-weighted oracle should be
   weak or null; (ii) a "only the state-matched unit is informative" population, where the
   alignment oracle must beat static.
5. **Not a sole kill.** PV is already a better-specified activity-dependent baseline and
   is already training.

### 5.6 Architecture gates for any new arm

Re-pass the Stage-0 family before a Stage-1 contract: G1 permutation invariance, G2
variable-`N` stability, G3 finite live gradients on both branches, G4 exact query/support
separation, G5 attainable carrier effect. Two notes:

- **G2 measures output variance, not latent scale** (`src/tfpd/gates.py:98-107`,
  `o.var()` over model outputs), and it draws a different synthetic session per `N`
  (`seed + n`, line 95), so `N` is confounded with session content. Quote it accordingly.
  Recorded ratios: bilinear `1.0714`, PV `1.294`, threshold `< 2.0`.
- **Softmax pooling changes every weight when `N` changes**, so its `N`-stability is a
  design question that **must be measured**, not assumed either way; the required
  unnormalized mass channel may well restore it. Normalization also erases absolute
  population confidence (parent §3, lines 110-113), which is why PV carries
  `pop_conf = log1p(mass)`.

### 5.7 Mechanism diagnostics

Same-checkpoint, no retraining: aligned T4 / exact zero / wrong-pair (row-permuted) /
activity-destroyed.

T4 minus Z4 remains **necessary** for TFPD: A2's carrier-content result was measured on a
different consumer and does not establish that this model uses content. The wrong-pair
contrast is the **distinctive** addition, because per-unit task correspondence is the
claim in the title and the row-permuted carrier is the only diagnostic that isolates it.
Both frozen models already implement `wrong_pair_carrier`.

## 6. Kill criteria specific to this lane

Beyond parent §7, reject the extension if:

- it is promoted to rescue a failed Priority-1 result by capacity alone. The parent's
  conditional is **pass → extend**. A bilinear failure showing the carrier is not used at
  all (flat aligned/zero/wrong-pair) falsifies the premise and does not license a bigger
  consumer;
- it skips the full-`B_i` static rung (§5.3) and therefore cannot attribute its gain to
  dynamic rather than static read-in;
- its advantage over PV disappears once PV is given a learned value space;
- softmax pooling is adopted without an explicit population-mass channel and a measured
  variable-`N` result;
- an SSM is introduced without either the streaming-horizon or the closed-form-adapter
  argument of §2 being made explicitly and tested.

**Deliberately not a kill criterion:** requiring the new model to show a *larger*
wrong-pair penalty than the bilinear model. Parent §8.6 states that swap-worse-than-zero
is strong evidence but **not a necessary condition**, because a good model may
conservatively attenuate a corrupt carrier. A first draft of this document imposed the
stronger rule; it would have killed the attenuation success mode the parent reserved.

## 7. Framing note: get the direction of the observation model right

In standard state-space notation the observation model maps latent to measurement,
`x_i(t) = C_i z_t + noise`. A network mapping measurement to latent is the inverse, so
calling it "the observation operator" inverts the convention and neurophysiology
reviewers will notice.

- **Amortized-inverse framing (Routes A/B).** Call it a carrier-generated read-in that
  amortizes the inverse of an implicit observation model. Honest, minimal, matches the code.
- **Generative framing (Route C).** Posit `x_i(t) | z_t` with carrier-supplied tuning and
  *derive* the read-in. This is the framing that literally earns the phrase "the carrier
  defines the observation model", and it is the version a neuroscience reviewer will find
  most familiar — which is both its strength and the reason its novelty must rest on the
  carrier-generated, no-per-session-fitting property rather than on the model class.

## 8. Corrected prospective statement

The working two-sentence version must not claim a limitation Stage 1 has not yet
measured, and must not call a lossy summary a sufficient statistic.

> If the Priority-1 result holds, task-frame decoding will have been established with a
> read-in whose pooling weights depend on the carrier alone — a static, carrier-weighted
> population average that fixes each unit's contribution before any behavior-relevant
> state is inferred, and that admits no cross-unit interaction prior to pooling.
> We would then ask whether a shared dynamical state should instead determine how a
> variable population is read in, through calibration-derived unit coordinates, making
> the read-in state-dependent while remaining permutation-invariant and free of any
> persistent NeuronID.

Note that the second sentence deliberately does **not** say "queries", because
state-dependence can be obtained generatively (Route C) or by a state-conditioned gain,
not only by attention.

The contribution sentence, which must survive independently of which block implements it:

> The claim is the pipeline **per-unit task correspondence → state-dependent population
> read-in → shared dynamics**, evidenced by a wrong-pair penalty and held-subject
> transfer. Attention and state-space models are implementation choices inside that
> pipeline, not the contribution.

This claim currently has **no Stage-1 number** behind it. Synthetic G5 shows only that a
pairing effect is *attainable* on toy cosine data. Until Stage 1 exists, the pipeline
statement is a hypothesis, and treating it as established would license exactly the
module shopping it is meant to prevent.

## 9. Prospective-boundary reminder

sub-M is a repeatedly used development external subject. Every result in this lane is
development evidence. A general cross-subject claim still requires a never-used
subject/dataset/task frozen before evaluation, with subject/session as the inferential
unit.
