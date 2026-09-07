# HANDOFF: where the next contribution comes from, if this round moves no number

**Date:** 2026-08-12
**Question addressed:** given the measured evidence, where can accuracy still come from, and what contributions
are available that do not require accuracy to move at all.
**Authorizes:** nothing. This is a strategy analysis. Every experiment named here needs its own pre-registered
contract before it runs.
**Companion documents:** `HANDOFF_NEXT_ROUND_DIRECTIONS_20260812.md` (the A/B menu),
`HANDOFF_TO_ROOT_POST_A4_REVIEW_20260812.md` (open corrections).

**Round boundary (2026-08-13):** A2 is the terminal endpoint of the preceding validation round. B1 and A1
are now terminal **routing** pilots in the new carrier/consumer round; neither is confirmatory evidence and
neither is an unfinished A2 arm. A12 remains a `3/24` descriptive partial and gates nothing.

**Root audit status (2026-08-13):** this document is a candidate-generation record, not an execution queue.
Several claims in the original strategy did not survive code/receipt inspection. In particular, latent decoder
queries and the proposed signed-readout limitation are withdrawn; B1's matched `{T4,Z4} x {with-E,no-E}`
factorial and A1's hidden-space adapter both stopped below their practical gates; and decoder depth or 10-ms
binning require a new teacher/source-training contract rather than a small configuration change.

**Terminal evidence update (2026-08-13):** A2 is terminal positive, with mean subject-shift interaction
`+0.235799` and crossed seed-by-session bootstrap interval `[+0.100852,+0.371768]`; this is increased relative
`T4-Z4` value, not an absolute T4 lift. B1 then completed all 12 Stage-P cells under the v10 binding: seed
interactions were `+0.059153/+0.004307/-0.022215`, mean `+0.013748`, so both the `+0.03` practical gate and
3/3-positive gate failed (`STOP_B1_NO_STAGE_F`; aggregate SHA starts `0ff38c4b`). A1's six-session seed-42
interaction was `+0.012833`, with 4/6 positive but below its `+0.03` mean gate (`PILOT_ROUTING_STOP`; aggregate
SHA starts `fe8b3114`). These are terminal routing decisions: do not resume B1 Stage F or expand A1. A12 has
only three of 24 planned seed-epoch pairs and is descriptive, non-causal, and non-gating.

**Terminal H-U/M1 boundary update (2026-08-14):** The development fold-0 seed-42 H-U arm ended at epoch 49 /
step 180,500. On the frozen `665fe535...` query (8,965 windows), four label-free per-unit statistics score
`0.4829429873` pooled (`0.5284200128/0.3507516311` per session), versus matched H-C `0.5255108533` and H-C0
`0.4866156747`; H-U−H-C0 is `−0.0036726874`, H-U−H-C is `−0.0425678660`, and recovered fraction is `−9.44%`.
There were zero target updates and no formal access (`formal=false`). Thus this tested descriptor does not
recover the H1 carrier increment: H-U is terminal with no expansion, without claiming universal equivalence or
nonequivalence. Receipt:
`SPINT-main/pilot_artifacts/h1_carrierid_hu/gpu_runs/h32_fold0_hu_v2_envfix/hu/H1_CARRIERID_HU_TERMINAL_EVAL_v9.json`,
SHA `f2f101888125a80dd0e44fb26b048197238fcced1fcf8d924d870cd8eaae9ef4`.

For M1, the authoritative matched compact EMG-AFC4 Full−Zero4 carrier increment is `−0.00652463` in one fold-0
seed-42 development cell; receipt SHA `c68f38822fc35414d570f2b932b16491fed763379325ad7efd0529f460e8e035`.
The official unmatched T4−Original `−0.003825` and same-checkpoint whole-identity Full−Zero `+1.8485` are a
system comparison and identity-reliance ablation, respectively, not carrier increments.

**Terminal H1 SPINT-capacity update (2026-08-14):** An architecture-preserving, activity-only H-S width
experiment now separates NeuronID encoder capacity from carrier content. Fold-0 fresh H-S-1024/W224/W32 score
`0.496833/0.432845/0.520634`; W224 fails the frozen `-0.03` non-inferiority margin, while W32 passes at
`+0.023801` and alone advances. Across the four predeclared remaining development dates, W32−H-S is
`+0.053540/+0.034049/+0.058984/-0.001049`, with mean `+0.036381` and outer-date bootstrap interval
`[+0.012598,+0.056262]`. W32 uses `60,124` rather than `5,965,500` identity parameters (`99.22x` fewer)
and `28.813M` rather than `2.710B` identity MACs (`94.05x` fewer). This supports substantial **extra
NeuronID-capacity redundancy on H1**, not a no-identity claim and not a cross-dataset capacity bound. Aggregate
SHA: `e65461e89c7d5df2d22a36307248d849534ebc7e9aa4dc09ca520e08aa1ccbb0`. Do not widen H1 NeuronID again;
future H1 work must target information, objectives, or generalization rather than raw identity-MLP capacity.

---

## 1. Expect little from more of the same T4 OLS under a frozen consumer

**Corrected 2026-08-12 after an independent audit.** An earlier draft of this section claimed "the carrier side
is finished" and closed D-optimal design, temporal-kernel carriers, population-covariance carriers, and Wiener
shrinkage as standalone accuracy arms. **That claim was wrong and is withdrawn.** It over-read the evidence and
contradicted `HANDOFF_NEXT_ROUND_DIRECTIONS_20260812.md` section 1.5, which is the more careful document and
states `estimator_saturation_proven: false`. What the evidence actually supports is narrower.

| Observation | Measurement | What it does support | What it does **not** close |
|---|---|---|---|
| Chronological label budget plateaus | `M10/M15/M20/M30/M50 = 0.304264 / 0.338115 / 0.351767 / 0.358154 / 0.356828` | Taking **more chronological-first-M trials** buys nothing past `M30` | Better **design** at fixed `M`, more informative labels, or non-OLS estimators |
| One frozen consumer is locally insensitive to a target-time carrier substitution | Query-fitted oracle carrier moves frozen H-C by `-0.002859`; `consumer_local_insensitivity_supported: true` | This checkpoint did not benefit from this leaky query-fitted substitution | Whether another estimator helps a retrained consumer. `estimator_saturation_proven: false` |
| Within the T4 four-dimensional family, `[a,c]` carries the content | `AC4 - T4 = -0.012222`; `m = hypot(a,c)` by construction; `b` versus mean rate `r = 0.9960089736` | `[m,b]` add nothing **inside that family** | Descriptors that are different content rather than wider `[a,c]`: temporal kernels, population covariance |

Three further caveats that the earlier draft omitted and that an adversarial reviewer will raise.

**The oracle receipt is a leakage diagnostic, not an upper bound.** It is tagged `LEAKAGE_DIAGNOSTIC_ONLY` with
an explicit `intentional_leakage` block: the oracle carrier is fitted on query trials, and for one recording all
`2230` of its ordinary query windows fall inside the oracle fit trials, with `query_label_reuse_present: true`.
The receipt states it is `not_a_strict_upper_bound` on an honest source-trained feature. Read correctly this
makes the frozen-consumer insensitivity **stronger** — even a leaky oracle does not move it — while making the
statement about estimation quality in general **weaker**, not stronger.

**The three rows are not independent.** The budget plateau and the oracle result are the same family: both ask
how much a better-fitted T4 buys. The third row is a within-family coordinate result. Together they argue
against more of the same, not against carrier research.

**The M15 citation in the earlier draft was wrong.** `-0.058842` with CI `[-0.095206, -0.023628]` is
`t4_m15_vs_t4_m50_noninferiority` in `aggregate_ordinary_pair_s42.json`, i.e. the **budget gap** between
ordinary T4@15 and T4@50. It is not the Wiener shrinkage effect; the shrinkage contrast is a separate arm
(`t4w3_m15_s42.json`) and is reported elsewhere as approximately flat. The earlier text conflated "Wiener did
not help at fixed M" with "low M loses to high M" and then used the mash-up to close several estimator arms.
Do not repeat that.

**Revised redirect.** Consumer, decoder, and data-side levers can be *prioritized* over further carrier
estimation work, because the frozen consumer demonstrably will not spend a better carrier and the chronological
budget has plateaued. That is a priority argument, not a closure argument. The carrier-side arms remain open and
must be paired with a retrained consumer, which is the follow-up the oracle receipt itself names.

---

## 2. Performance side: what is genuinely untried

The logged architecture/fusion evidence is repeatedly flat or worse, but it must **not** be counted as nine
independent carrier-fusion tests. The two `-0.003142` logit entries refer to the same aggregate, and B15 is an
activity-only self-attention encoder rather than a carrier-fusion arm. Read as a design-space audit, the results
cover interface width (CI64 `-0.020130`), identity-path interventions (FiLM `+0.003399`, live-activity gain
`-0.002068`, logit residual `-0.003142`, electrode gate `-0.010817`, relation `-0.001440`), cross-neuron
communication inside the encoder (B15, gain explained by capacity: `B15 - B15P = +0.006354`), and backbone
replacement (decoupled K/V `-0.444658`, slot router `-0.177935`). This supports a broad empirical boundary;
it does not license an exact count of independent negative carrier mechanisms.

**None of them changed the decoder's depth or the input time resolution.** Those axes remain untried but are
new training regimes, not cheap toggles. The apparently untried latent-token idea is structurally inert in the
current decoder and is withdrawn in section 2.3.

### 2.1 Decoder depth — no bound multi-layer mainline receipt was found

`MultiLayerCrossAttention` accepts `num_layers > 1`; the audited mainline uses one layer. However, the streaming
student does not own an independently swappable decoder: it constructs its decoder with
`num_layers=self.teacher.num_layers` and loads the teacher state with `strict=True`. A depth-2 or depth-3 arm
therefore needs a newly trained compatible teacher/decoder and fresh source training. It is **not** a small
configuration-only sweep. Several existing adapters also assume one layer by construction.

This remains an untried capacity lever, and the existing interface-width and encoder experiments do not settle
it. But it is a new pretrained-decoder regime, not a low-cost diagnostic. It should be attempted only after a
source-only screen and an explicit accuracy/compute contract justify the extra training and online depth.

**Honest cost.** Depth multiplies the online path and changes the pretrained decoder. Any gain must be reported
as an accuracy/compute trade-off and cannot be folded into the existing bit-identical-online-path story. A flat
result would be informative, but it would not by itself prove that every consumer is not depth-limited.

### 2.2 Input time resolution — `bin_size_ms = 20`, hardcoded, never swept

POYO reaches roughly `0.935` single-session on the same DANDI 000688 substrate at 5--10 ms bins with about 13M
parameters. This program's within-session ceiling is `0.6937`, and `P3_CROSS_SESSION_ANALYSIS.md` attributes
the gap to two causes: capacity and binning. Capacity has been probed repeatedly. **Binning has never been
touched.**

At 10 ms with `W = 100` the temporal span remains 1 s, but the computational and statistical regime does not:
binning/smoothing, `fc_in`/`fc_id_out`/`fc_out`, representation dimensions, normalizers, cached windows, and the
teacher checkpoint all change. A W50 checkpoint is not shape-compatible with W100. This therefore requires a
new W100 teacher, source training, carrier/query contracts, and a revised hardware-cost accounting before any
matched comparison can be made.

**Do not call this a large expected win.** `P3_CROSS_SESSION_ANALYSIS.md` lists capacity and binning as two
*speculative* attributions for a gap measured against a non-matched POYO figure under a different split. The
defensible statement is that binning is the one untried data-side knob, not that it is the lever that explains
the POYO gap.

### 2.3 Decoder token count — latent-query proposal withdrawn

The proposed addition of unscored latent query tokens does not work in the current decoder topology. The
cross-attention layer maps each query independently over neural keys/values, and the following feed-forward path
is also per query; there is no query-to-query mixing by which an unscored latent query could influence the
behavior queries. Making it useful would require a new query-mixing block and a new architecture, not a cheap
token-count sweep. **Drop this branch for the current program.**

### 2.4 Closed-form parameter-space adaptation

Every mainline carrier adaptation happens in feature space: four numbers per unit. An alternative is to adapt
**parameters** in closed form — for example, ridge-solve a final linear layer or decoder readout on the
calibration prefix. This can remain optimizer-free and backward-pass-free, but it is still target-label-using
parameter adaptation. It must not be described as the same deployment contract as a cached T4 carrier.

The unmatched labelled-oracle numbers are not an effect-size forecast: they cross a roughly 4x unit-count
regime and use gradients. Before any run, freeze a supervision ledger covering target type, number and density
of labels, updated state, query boundary, and online cost. Dense per-bin ridge is a label-richer upper bound,
not an equal-information T4 comparator.

### 2.5 Training method — B1 is terminal negative, not paused

The M2 streaming line defaults to `loss_mode = task_plus_y_plus_E` with `lambda_E = 0.1`, inherited from an
R1 selection made on carrier-free `streaming_b3`. That provenance made B1 a clean hypothesis, but the matched
test has now answered its routing question. The complete 12-cell `{T4,Z4} x {task_plus_y,
task_plus_y_plus_E}` Stage P produced seed interactions `+0.059153/+0.004307/-0.022215`, mean `+0.013748`.
The mean missed `+0.03` and the signs were not 3/3 positive, so the frozen rule requires
`STOP_B1_NO_STAGE_F`.

Drop the claim that identity distillation is the carrier-specific limiter on this substrate. Do not run Stage F,
add `task_only`, or tune `lambda_E` from these results. This does not close distinct training interventions:
activity-path dropout asks whether the consumer should practise using T4 when activity identity is unreliable;
estimator-noise augmentation asks whether it should tolerate uncertain T4; carrier corruption asks a safety
rather than an accuracy question. Each still requires a T4/Z4 sibling and a separate contract.

### 2.6 Terminal boundaries and converged ranking

| Evidence | Decision boundary |
|---|---|
| A2 subject-shift interaction `+0.235799` | Survives as the positive premise: T4 has increased **relative** value under the observed shift; it is not an absolute T4 lift |
| B1 mean interaction `+0.013748`, mixed seed signs | Terminal routing STOP; no Stage F and no carrier-specific distillation claim |
| A1 hidden-space interaction `+0.012833`, 4/6 positive | Terminal routing STOP below `+0.03`; attachment worked, useful lift did not; no expansion |
| A12 `3/24` seed-epoch pairs | Descriptive partial only; no causal inference, no gate, and no completion merely for matrix closure |

The remaining recommendations are deliberately simple and non-compositional:

| Rank | Lever | First gate and kill rule | Why it survives |
|---|---|---|---|
| 1 | **Activity-path dropout with a T4/Z4 sibling** | Freeze one nonzero dropout level, then a 12-cell fold-0 `{T4,Z4} x {p=0,p>0} x 3 seeds` routing pilot. Kill if mean interaction `< +0.03` or any seed interaction is non-positive; do not sweep `p` after seeing scores. | B1 changed identity distillation and A1 changed the add site; neither trained the consumer to rely on T4 when activity identity is missing. Z4 separates carrier-specific use from generic regularization. |
| 2 | **Source-session exposure reweighting with T4/Z4** | CPU first: bind the actual streaming M2 route, prove disabled mode reproduces the legacy sampler, keep validation sampling unchanged, and quantify the source-session exposure change. If promoted, retain the frozen C2 kill rules: T4 lift and T4-minus-Z4 interaction both `>= +0.03`, with all three fold means and all three seed means of the T4 lift positive. | It changes the training objective distribution without adding a module. It is cheaper to falsify structurally than another architecture arm. |
| 3 | **Estimator-noise-aware consumer training** | CPU contract must freeze one noise law from source-only estimator residuals and prove exact T4/Z4/no-noise controls. A later routing pilot stops below `+0.03` interaction or on mixed seed signs; no post-result noise-scale sweep. | The frozen-consumer oracle does not close a jointly retrained consumer, but the M30 plateau makes another deterministic OLS refit a weak bet. |
| 4 | **Evaluation standard for backprop-free adaptation** | Writing/receipt audit only; ship only if every checklist claim links to an immutable worked example and negative boundaries are labeled by scope. | Converts the existing failures into a reusable contribution without depending on another accuracy gain. |

**Sampler wording.** In the current repository snapshot, only
`streaming_calibration_exp/src/data/falcon_datamodule.py` exposes the default-off `balance_sessions` /
`balance_session_batches` path. It interpolates **training batch exposure** toward equal session counts while
preserving the epoch batch count and deterministically cycling shorter sessions. It is not an exact equal-window
dataset, is not routed to validation, and is not present in the other two `SessionBatchSampler` implementations.
Call this *source-session exposure reweighting*, not a generic or already-used "equal-session M2 mechanism."

Hold session-consistent carrier corruption as a safety study; it applies the opposite training signal from
activity forcing and should not be sold as the next accuracy arm. Hold decoder depth, W100/10-ms input, and
closed-form target-parameter adaptation because each changes the regime or supervision contract. Drop repeats
of B1/A1, latent queries, signed-readout claims, widening, and further additive fusion search. Do not finish A12
unless a specific paper diagnostic is declared in advance; its current partial does not select a mechanism.

No row authorizes execution. Every performance row still requires a settled implementation, frozen contract,
attainable gate, and independent review.

### 2.7 Convergence record

The audit reconsidered fourteen raw directions spanning loss, add site, source-training incentives, sampling,
estimator uncertainty, carrier safety, decoder/data regimes, target-parameter adaptation, controls, diagnostics,
and standards. After applying the current terminal results, mechanism separation, composition risk, and minimum
implementation complexity, the set converges to the four rows above.

The strongest remaining performance pitch is:

> A2 establishes that T4-specific information becomes more valuable under the observed subject shift, while A1
> and B1 show that moving the add site or removing identity distillation does not deliver a practical interaction.
> The next clean test is whether source training can make the consumer practise using T4 when activity-derived
> identity is unavailable, with Z4 detecting ordinary dropout regularization.

The strongest objection is still generic regularization. The Z4 sibling and interaction gate are therefore
mandatory. One predeclared dropout level gets one routing pilot; failure ends the branch rather than starting a
dropout-rate search.

---

## 3. Contributions that do not require accuracy to move

Ranked by novelty per unit of cost. These are the fallback if section 2 returns flat, and several are worth
doing regardless.

### 3.1 Attribute A4's residual phase leakage, rather than assert its source

**Corrected 2026-08-12.** An earlier draft claimed the residual "can only come from" activation curvature. That
is false and is withdrawn. It also contradicted `HANDOFF_NEXT_ROUND_DIRECTIONS_20260812.md` section 1.3, which
had already been corrected to list three candidate mechanisms. An adversarial reviewer would quote one document
against the other.

A4 seed 42 measured AC4 raw/null/advantage `0.740706 / 0.235635 / +0.505071` and Z4
`0.312414 / 0.223577 / +0.088837`. These are **arm-specific nulls**, not one common null. Across three seeds,
AC4 advantage is `+0.517299`, Z4 advantage `+0.090144`, and AC4-Z4 `+0.430155` with 18/18 session comparisons
positive; the authoritative multi-seed receipt is cited in `HANDOFF_TO_ROOT_POST_A4_REVIEW_20260812.md`.

Exactly balanced linear averaging would remove the first harmonic. The residual therefore has **at least three**
candidate sources, and they are not mutually exclusive:

1. the learned nonlinear per-trial projection, of which `pre_pool`'s `Linear -> ReLU` curvature is one part;
2. **the finite direction set being only approximately balanced**, which leaves first-harmonic residue even
   under strictly linear averaging;
3. temporal response structure that preserves phase-correlated statistics.

Source 2 alone is sufficient to produce the observed residual, so no curvature claim is entailed.

**What is still worth doing, correctly scoped.** An activation ablation is a cheap *attribution* experiment, not
a derivation. It cannot isolate source 1 on its own, because changing the activation requires retraining, which
moves several things at once, and because sources 2 and 3 can keep leakage alive under an identity activation.
To make it informative, pair it with a measurement of the actual direction-balance of each calibration block, so
source 2 can be quantified and subtracted rather than assumed away. Report it as an attribution decomposition.

**Consequence for priority.** This is no longer the strongest item in section 3 and should not lead the plan.

### 3.2 The proposed dead-port effect is not isolated by the existing contrast

`Z4 - B0 = +0.089591` is **not** an isolated dead-port effect. Z4 uses the B3S early-pool encoder with an
all-zero side input, while B0 uses the teacher-style batch-reference identity encoder. The contrast therefore
mixes encoder topology, parameterization, initialization, optimization, schedule, and port width. It cannot
support the claim that a permanently zero port is worth `0.09` R-squared.

A11's completed authoritative CPU replay additionally shows schedule sensitivity, but schedule correction cannot turn this unmatched
architecture contrast into a port-isolation result. A valid test must hold the B3S encoder, data, loss, schedule,
normalizers, and seed fixed while comparing a true `side_dim=0` path against zero-valued side widths. Until that
matched control exists, keep the dead-port mechanism as a held hypothesis rather than a contribution.

### 3.3 Frozen-consumer saturation as a general claim

The oracle-carrier result plus the flat fusion arms point toward a statement with reach beyond this dataset:
**for a saturated frozen consumer, even a leaky oracle feature is worth nothing.** That would be a constraint on
the whole family of feature-side improvements to frozen models, not just on carriers.

**As of now this is an H1-local fact, not a general claim.** It rests on one dataset, one frozen checkpoint, and
a receipt explicitly tagged `LEAKAGE_DIAGNOSTIC_ONLY` and `not_a_strict_upper_bound`. Do not publish the general
form until the retrained-consumer follow-up named in that receipt has run, and until at least one further
substrate reproduces it.

### 3.4 Control bias, quantified in both directions

Same-checkpoint carrier zeroing **overstates** the carrier by `2.67x` (`0.103871` versus `0.038895` against a
retrained control). Same-checkpoint tag shuffling **understates** dependence by about `4x` (16% versus a true
67% contribution). Two worked examples, opposite directions, one mechanism-level explanation: a
same-checkpoint intervention on a model trained with correct content measures something different from
retraining without it.

### 3.5 What else occupies the identity token

One H1 token audit measured carrier/activity RMS about `0.3325`, an angle near `83.483` degrees, and carrier
participation ratio near `2.08`. This does **not** prove that the carrier literally occupies a rank-2 subspace
of the full identity token, nor does A4 answer token occupancy. A separate forward-only residual-token analysis
could ask whether remaining variation is unused, session-linked, or higher-order activity structure, but this is
an unrun descriptive study and must not be presented as an existing result. A12's current `3/24` descriptive
attention partial does not answer this question and supplies no causal reason to prioritize it.

### 3.6 Signed-readout limitation — withdrawn

The original argument incorrectly inferred a nonnegative population readout from nonnegative softmax weights.
Attention values and the output projection are signed, so a unit's net contribution can be positive or negative
even when attention weights are nonnegative. The current architecture therefore does not have the claimed
structural inability to subtract evidence. **Do not run or cite this branch.**

### 3.7 An evaluation standard for backprop-free session adaptation

The program already holds the material: gates frozen before results, dimension-matched content controls,
fail-closed aggregators, hashed immutable receipts, volunteered negative boundaries, four proxy statistics that
failed to predict performance, an overlap statistic that failed its own calibration against decoder gain
(`rho = -0.5`, `p = 1.0`), and a train-only proxy that improved 27/27 sessions before failing the real
endpoint. Assembled as a checklist with worked negative examples, this converts spent compute into a
contribution.

---

## 4. What not to do

- Do not search for another fusion mechanism merely to increase the arm count. Repeated additive-pathway
  interventions are flat and the two backbone-replacing arms fail for documented and different reasons, but
  the logged rows are not all statistically or mechanistically independent.
- Do not widen anything. CI64 is `-0.020130` and terminal; H64 is prohibited.
- Do not add a module for the sake of having a contribution. Added structure has repeatedly bought capacity,
  not mechanism, and the parameter-matched controls detect it every time.
- Do not revive the A13 30-stratum falsification design. Its inferential problems are real: window-level
  resampling treats autocorrelated windows as independent, most design-coverage cells cannot hold two
  sessions, the runner accepts externally prepared predictions without checkpoint binding, and the prediction
  that poorer design yields larger carrier gain is confounded with worse carrier estimation in the same cells.
- Do not present any never-run item as a null. Residual-FiLM, the 64-head oracle, the key-residual adapter,
  and the electrode anchor and embedding designs have no numeric outcome at all.
- Do not revive B1 or expand A1 after their frozen routing gates failed. Their negative boundaries are useful
  precisely because the stop rules are honored.
- Do not complete A12 merely to reach `24/24`. Its current `3/24` aggregate is descriptive and non-gating; resume
  only for a pre-specified figure or diagnostic that changes a concrete decision.

---

## 5. Priority

Reordered after the terminal A1/B1 receipts. The plan now addresses the failure mode first and refuses to compose
multiple speculative changes.

| Step | Item | Decision |
|---|---|---|
| Done | A2 terminal positive | Retain `+0.235799` only as a relative subject-shift interaction, reported with its absolute arm/domain means |
| Stop | B1 carrier x distillation | Stage-P mean `+0.013748`, mixed signs: no Stage F, no loss sweep |
| Stop | A1 hidden-space adapter | Routing mean `+0.012833 < +0.03`: no expansion or alternate add-site tuning |
| 1, strongest performance lane | Activity-path dropout with T4/Z4 | One frozen level, one 12-cell routing pilot; kill below `+0.03` interaction or with any non-positive seed |
| 2, cheapest CPU gate | Source-session exposure reweighting audit | Verify the streaming-only default-off interpolation, exact legacy null, unchanged validation sampler, and actual exposure shift before considering the already contracted matched matrix |
| 3, lower performance priority | Estimator-noise-aware consumer training | Source-only noise law and matched controls first; do not reopen deterministic OLS tuning |
| 4, no-accuracy contribution | Evaluation standard | Assemble immutable worked examples after the experimental boundary is stable |
| Hold | Carrier corruption; decoder depth; W100/10 ms; target-parameter adaptation; zero-port/token diagnostics | Safety-only, new-regime, new-supervision, or non-blocking descriptive work |
| Drop | B1/A1 repeats; A12 completion without a declared use; latent queries; signed-readout limitation; widening/additive-fusion search | Failed gate, non-gating compute, invalid mechanism, or saturated design family |

No new GPU arm is authorized by this strategy file. Each surviving item still requires a settled implementation,
frozen receipt, and independent review.

### 5.1 Active non-H1 queue after the 2026-08-14 CEBRA review

The H1 compact-SPINT result changes priority, not scope.  Its architecture-preserving W32 identity path is
`99.22x` smaller than H-S-1024 and is non-inferior across the four confirmed development dates, so more H1
NeuronID capacity is low priority.  This does **not** prove that every H1 carrier, objective, normalizer, or
phase-conditioned representation is exhausted.  H1 remains open behind the monkey 2-D program.

The active queue is continuous and ordered.  A stopped item advances to the next item without tuning the stopped
item from target scores:

1. **CEBRA-inspired size-matched pseudo-session mixing, SUA/sub-M: terminal Stage-P stop.**  Seed 42 gave
   external `mix-T4 - A2-T4 = +0.022609` and within-sub-C `+0.009485`; the absolute external gate was `+0.03`,
   so seeds 43/44 and tuning were not run.  Aggregate SHA-256 is `16b83e5e4251fe31cbd72d6902fbd8300eaff9217e9c5a9a358020e2c01f7334`.
2. **Bias-cancelled carrier tokens (`SetKV-delta`): terminal forward-only stop.**  The immutable aggregate
   SHA-256 is `1d6ea6f9fd12623094341b41157ca18c6be26bf134554a939ce53ce7b4009802`.  SetKV-T4 lost `-0.738673`
   within subject and `-0.638865` externally relative to its matched A2 parent, and the T4-versus-row-shuffle
   attachment contrast was effectively zero.  Do not launch joint training from this diagnostic; it does not
   prove that joint training could never work.
3. **Value-weighted carrier-mask diagnostic: current internal-mechanism gate.**  The source-only probe relates T4 tuning magnitude to
   attention-weighted value contribution, not attention mass alone.  A weak association stops the mask route
   before performance scoring.  The low-level `key_padding_mask` support is not yet a routed SUA experiment.
4. **Session-consistent misleading-identity swap v2.**  Because pseudo-session Stage P stopped, this remains a
   standalone A2-parent experiment.  Swap the B3S pooled activity `mean_feat` before
   `post_pool`, while keeping query activity and the unit's T4 row attached.  Donor maps are fixed per source
   session/epoch (never per query window) and are matched on source-only rate and T4 statistics.  The logical
   matrix is `{clean,swap} x {T4,Z4}` with external T4 lift `>= +0.03`, within floor `-0.03`, and a separately
   reported interaction.  A clean-domain gain is not required; clean-domain non-inferiority and robustness to a
   deliberate swapped-input diagnostic are required.
5. **Fair Track-B CEBRA comparator if the internal mechanism queue stops.**  Repair independent-query scoring,
   source-only lambda selection, exact reference authority, positive-control iteration parity, and receipt
   provenance before any real comparison.  All-bin supervision and rate-gauge augmentation are not automatic
   fallbacks; the former changes temporal causality and the latter did not pass its frozen magnitude trigger.

The additive Track-B v2 route now hard-excludes H1/M2 and admits only subject-M SUA/pseudo-MUA and RT.  Its
strict-27 sub-C source-only SUA and pseudo-MUA loader/authority stage is complete, including exact replay of
all 27 pseudo-MUA pooling transforms; it still contains no target access, CEBRA fit, checkpoint, GPU, or score.
It records one important solver boundary before target data are opened: CEBRA's sklearn-accessible
`MultiSessionSolver` is the accuracy-comparator family, whereas `UnifiedSolver` concatenates all training-session
units into a fixed-width input and cannot serve a standalone unseen target session.  The latter may support only
an explicit `UNIFIED_UNSEEN_SESSION_UNSERVABLE` structural receipt, never a fabricated few-shot accuracy arm.
No real CEBRA score exists yet; live scoring remains gated on canonical support/query extraction, source-only
geometry selection, and exact-geometry positive/negative controls.  Matching M50/M24 fixes the number of target
trials, not the amount of supervision inside them: the primary CEBRA-Behavior arm uses dense bin-level velocity
on the prefix, while T4 uses sparse trial-level labels.  Every future receipt/table must therefore print both
label counts and mark the contrast `information_matched=false`; this supervision mismatch favours CEBRA on
accuracy and must not be hidden behind the phrase "matched calibration budget."

Two interpretation corrections are binding for this queue.  First, the external A2 Z4 mean `-0.143399` shows
that the no-carrier system fails under subject shift; because Z4 still contains activity-derived identity, it
does not by itself prove that activity identity is worse than a matched no-identity system.  The swap experiment
therefore tests a hypothesis rather than repairs an already-proven cause.  Second, appending carrier tokens keeps
the original read-in structurally intact but changes the attention set and softmax immediately; Z4, duplicate,
and row-shuffle controls are mandatory even though SetKV-delta introduces no trainable parameters.

Only a three-seed absolute external-T4 gain promotes a mechanism to M2 and then RT.  No H1 migration, descriptor
sweep, gauge arm, or all-bin arm may jump ahead of this queue.

---

## 6. Revision note

Revised 2026-08-12 after an independent audit of this file. The first audit found three substantive errors, all in the
direction of claiming more than the receipts support: section 1 declared the carrier side finished when
`estimator_saturation_proven: false` and the oracle receipt is a leakage diagnostic with intentional query
overlap; the M15 number was attributed to Wiener shrinkage when it is the ordinary T4@15-versus-T4@50 budget gap;
and section 3.1 asserted that the residual phase leakage "can only" come from activation curvature when
approximate direction imbalance alone suffices. All three are corrected in place and marked.

A second root audit on 2026-08-13 found further material problems: decoder depth and W100 require new teacher
contracts; latent queries have no route to affect behavior queries; a T4-only loss sweep cannot identify a
carrier-specific interaction; Z4-B0 is not a matched zero-port contrast; token participation was overstated as
literal rank; and the signed-readout argument ignored signed values/output projections. These are corrected in
place. The strategy now retains a narrow, testable near-term program rather than presenting every untried axis
as a cheap opportunity.

A third evidence update on 2026-08-13 removes the stale execution narrative. B1 is a complete terminal Stage-P
STOP, A1 is a terminal routing STOP, and A12 is only a non-causal `3/24` descriptive partial. The former top two
execution rows are therefore retired. The remaining plan prioritizes a single matched activity-forcing test,
puts source-session exposure reweighting behind a CPU audit, and leaves expensive decoder/data-regime changes on
hold.
