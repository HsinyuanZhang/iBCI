# Handoff: decomposing the sub-population invariance effect, and turning it into a design

Date: 2026-08-18
Status: **EXECUTED AND CLOSED (2026-08-19).** Every cell this document authorized has landed except E,
which is closed unstarted on evidence produced by C (see §0). No cell in this document remains open.
Successor: `HANDOFF_ACTIVITY_VS_IDENTITY_MASK_20260819.md`.

Supersedes the forward-looking parts of `WORKORDER_BEAT_A2_SPARSIFICATION_20260818.md`; that
document's Step 0, cell R and cell S2 have all executed.

---

## 0. Closing verdict (2026-08-19)

Authoritative receipt: `results/subpop_score_v1_r2/subpop_score_receipt.json` (finished
2026-08-19T00:23:04Z, `not_landed: {}`, cells C/G/T/W). The earlier `subpop_score_v1` and
`subpop_score_v1_r1` are **cumulative, not corrective** — every contrast they share with `_r2` is
bitwise identical; `_r2` simply adds the cells that had not yet landed. None of the three carries a
`supersedes` field, which is a receipt-discipline gap worth fixing given the
`a2_matched_rescore_v1` precedent, but there is no substantive conflict between them.

| cell | verdict | governing numbers (last-bin, equal-session, variance-weighted, seed 42) |
|---|---|---|
| Step 0C | D's degradation curve is far flatter than Arm A's; ensembling excluded | at f=0.75 D **+0.09** vs Arm A **−0.79**; K=8 subset ensemble 0.386 < native 0.418 |
| **T** — true removal via `key_padding_mask` | **T_APPROX_EQUAL to D** | T−D −0.0099 (7/15, CI [−0.028, +0.008]); T−Arm A **+0.1477 (15/15)** |
| **C** — paired-subset consistency, λ=0.1 | **gate FAIL**, closed without a λ sweep | C−D external **−0.0357** (4/15, CI [−0.061, −0.010]); C−D within +0.0029 |
| **G** — random global gain | **GAIN_NOT_THE_MECHANISM** | G−Arm A +0.0095 (10/15, CI [−0.043, +0.055]); G−D −0.1481 (1/15) |
| **W** — K=8 temporal residual | negative, exploratory, no gate | W−Arm A −0.0902 (3/15); full-window −0.1289 (3/15) — bad at both granularities |
| **E** — unbiased-estimator read-out | **CLOSED UNSTARTED** on C's counter-evidence (§0.2) | — |

### 0.1 The absolute ladder this round completes

| system | external | vs Arm A |
|---|---:|---:|
| **D** — whole-unit ablation | **0.4179** | +0.1576 (14/15) |
| T — true removal, no gain | 0.4081 | +0.1477 (15/15) |
| C — D + explicit consistency | 0.3823 | +0.1219 (13/15) |
| S2 — carrier-sector | 0.3767 | — |
| A2 pooled (teacher-initialized) | 0.3461 | +0.0857 |
| G — gain only, nothing dropped | 0.2699 | +0.0095 |
| Arm A | 0.2604 | — |
| R — elementwise, matched amount | 0.1877 | −0.0727 |
| W — temporal residual | 0.1702 | −0.0902 |
| DH — 64 heads | −1.1373 | — |

**The mechanism is now isolated by five independent controls, all converging on one sentence.**

| what was varied | result | what it rules out |
|---|---|---|
| R: same amount, scattered elementwise | 0.1877, *below* baseline | structure over whole units is required |
| G: gain only, nothing dropped | +0.0095 | the `1/(1-p)` gain is not the mechanism |
| T: true removal, no placeholder, no gain | ≈ D | placeholder-vs-absence semantics do not matter, and the gain is ruled out a second time |
| S2: carrier-contiguous subsets | < D | task-space structure of the subset adds nothing |
| C: explicit invariance objective | < D | making the invariance explicit actively hurts |

Surviving claim: *training-time random ablation of whole units is necessary and sufficient; none of
its implementation details, its geometric structure, or its explicit formulation matter.* §6's
interim framing restriction is now **lifted** — T settles it. Because T ≈ D with no gain and no
constant-token placeholder, the mechanism sentence may be written as "training on random
sub-populations".

**But T does not replace D as the system.** The ledger's curve asymmetry
(`TFPD_RESULTS_LEDGER_20260816.md` §3D) records that each training form is robust only to its *own*
ablation semantics: at f=0.75 under the placeholder intervention T collapses to −0.0404 while D holds
+0.0905, and under true padding the ordering reverses. Since a deployed pipeline delivers absent units
as placeholder tokens unless the `key_padding_mask` plumbing is shipped, **D remains the
deployment-matched system and T is the mechanism adjudicator.** Write T into the mechanism section,
not the results headline.

### 0.2 Why E is closed unstarted

C made subset-invariance an explicit objective and it **cost** 0.0357 external (4/15, CI excludes
zero) while Step 0C independently showed D's degradation curve is already flat. Read together:

> **D buys robustness without invariance, and demanding invariance destroys performance.**

The likely reason is that *which units are present carries information*. Forcing
f(subset₁) = f(subset₂) forbids the model from using subset-specific evidence. The correct objective
is "be accurate under any subset", which D's per-branch task loss already supplies; C adds "give the
same answer under every subset" and pays for it.

E was the architectural form of exactly that second demand — an aggregation whose expectation does
not depend on which units are present. C is direct evidence against its premise, so E is closed
rather than held. Anyone reopening it must first explain why architectural invariance should succeed
where objective-level invariance measurably failed; "the two are not isomorphic" is not sufficient on
its own, because the mechanism of C's loss is the same demand.

### 0.3 What this round says about design additions

Four design additions were attempted — S2 (subset geometry), C (explicit objective), W (temporal
capacity), DH (head capacity) — and **all four failed**. That is a pattern, not noise: the design
space immediately around the perturbation and around decoder capacity is exhausted at current
evidence. The asset this lane holds is a mechanism result of unusual completeness, not a module.

Consequences carried into the successor document:

1. The only remaining blocker on any A2-superiority claim is still **D seeds 43/44** (A2 is a
   three-seed reference with 0.066 external spread). Deferred twice; still required.
2. The `behavior_scaling_factor` asymmetry (5.0 for A2, none for the teacher-free family) remains
   disclosed but uncontrolled.
3. Step 0C's degradation curve is the strongest figure the lane owns and already exists.

---

## 1. What is established

All numbers below are the governing convention: last-timestep-of-window, variance-weighted R²,
equal weight per session, final-four SWA, seed 42.

| system | external (15 sub-M) | within (6 sub-C) |
|---|---:|---:|
| **D** — Arm A + `dynamic_dropout(0,1)` | **0.4179** | 0.5697 |
| S2 — carrier-sector sparsification | 0.3767 | 0.5723 |
| A2 pooled (teacher-initialized) | 0.3461 | 0.5776 |
| A2 per seed 42/43/44 | 0.3178 / 0.3367 / 0.3837 | 0.5849 / 0.5725 / 0.5755 |
| Arm A (no dropout) | 0.2604 | 0.5538 |
| R — elementwise, matched amount | 0.1877 | 0.5082 |
| DH — 64 heads + dropout | −1.1373 | −0.5114 |

Paired contrasts: `D − Arm A` external **+0.1576** (14/15, CI [+0.101, +0.222]), within +0.0158
(5/6, CI [−0.006, +0.038]). `R − D` external **−0.2302** (0/15, CI [−0.277, −0.182]), within
−0.0615 (0/6). S2-over-D gate: **rejected**. D on `sub-M_ses-CO-20141203` is **+0.4863** over Arm A;
D on the 2015 block is +0.1538 (11/11).

Three readings that should now be treated as settled:

1. **D exceeds A2 on external, above every A2 seed.** 0.4179 against A2's best seed 0.3837 (+0.034)
   and pooled 0.3461 (+0.072). On within, D is 0.5697 against A2's *worst* seed 0.5725, i.e. short
   by 0.0028 — inside A2's own seed spread of 0.0124. The correct statement is "exceeds A2 on
   cross-subject transfer, indistinguishable on in-distribution fit", not "reaches A2".
2. **The effect is not generic regularization.** At matched site and matched removal amount,
   scattering the removal elementwise gives 0.1877 — *below* the no-removal baseline of 0.2604.
   Same quantity of destroyed input, opposite sign of effect, 0/15 sessions favouring the scattered
   form. This is the strongest asset the lane owns and it cannot be described as restoring a flag,
   because the flag's strength is matched in R.
3. **Carrier-structured gaps add nothing over random gaps.** S2 works (0.3767, also above A2) but
   does not beat D. So the invariance that matters is over *unit subsets*, not over *task-space
   coverage*. The carrier is load-bearing as input and not as training structure. Record this as a
   real negative that narrows the claim.

Disclosed confound to keep attached to every A2 comparison: the A2/teacher lineage uses
`behavior_scaling_factor = 5.0` with `predict_scaled_behavior = true`, while the
Arm A / D / DH / R / S2 lineage has no scaling factor at all
(`sparsification_step0_v1/step0_receipt.json`, `behavior_scaling_parity_audit`). This is a
parameterization difference between the two families, not a sparsification effect, and it has never
been controlled.

---

## 2. The blocking problem: we are describing D incorrectly

`spint.py:449-455` does not remove units. It does this:

```python
dropout_mask = torch.ones(batch_size, num_neurons).to(src)          # BxN
p = random.uniform(self.dynamic_dropout_low, self.dynamic_dropout_high)
dropout_mask = torch.nn.functional.dropout(dropout_mask, p=p, training=self.training)
src = src * dropout_mask.unsqueeze(-1)                               # BxNxW
```

Three consequences, all verified in code:

- **`fc_in` has biases.** It is `Linear(50,512) → ReLU → Linear(512,512)` (`spint.py:401-403`), so
  `fc_in(0)` is a non-zero constant. A "dropped" unit is therefore **not absent** — it becomes a
  *shared constant token* that still contributes a key and a value to the attention softmax
  (`spint.py:464`). D trains against a variable-size block of identical placeholder tokens, which is
  a learnable null/sink mass, not a smaller population.
- **Survivors are amplified.** `F.dropout` on a ones-tensor returns `1/(1-p)` for survivors, and the
  multiply happens *before* `fc_in`, so surviving units' input windows are randomly gain-scaled per
  forward.
- **`p` is drawn once per forward** with `random.uniform`, shared across the batch, while the
  Bernoulli mask varies per (batch, unit).

So D bundles at least three separable factors: (a) a random subset replaced by a shared constant
placeholder, (b) a random global gain `1/(1-p)` on survivors, (c) a varying count of non-placeholder
units. "Trained on random sub-populations" is not what the code does, and any reviewer reading
`spint.py` will find this.

This matters for two reasons. It is a correctness problem for the paper's mechanism sentence, and it
is a **prerequisite for design**: you cannot build a better version of an effect whose active
ingredient is unidentified. Section 3 is therefore a decomposition before an invention.

Useful fact for implementation: `CrossAttentionLayer.forward` **already accepts `key_padding_mask`**
and forwards it to `nn.MultiheadAttention` (`spint.py:26, 35`). The call site at `spint.py:464`
simply never passes one. True removal is a small change against existing plumbing.

---

## 3. Cells, ranked

Every cell is the **Arm A recipe verbatim** (48 epochs, warmup 1e-5→1e-4 over two epochs, cosine to
1e-6, strict-27, final-four SWA, seed 42, `num_heads=2`, no clipping, no pretraining, no teacher
checkpoint) with exactly one change, scored on the governing convention and reported against both
Arm A and D.

### Cell G — random global gain, nothing dropped (claim-killer control, run first)

Keep every unit. Per forward draw `p ~ U(0,1)` from the same RNG source and multiply **all** unit
windows by `1/(1-p)`. No masking.

This isolates factor (b). It is the control that R does not provide: R contains the same `1/(1-p)`
rescaling *plus* element scattering, so R's failure cannot separate "scattering is harmful" from
"gain augmentation is helpful". If random input-scale augmentation alone recovers a large share of
D's `+0.1576`, then the finding is input-scale augmentation, not population structure, and the
paper's central sentence changes.

Pre-registered reading — `G − Arm A` external:

| outcome | consequence |
|---|---|
| `≥ +0.08` (over half of D's gain) | the population story is largely wrong; reframe around scale augmentation and re-derive everything |
| `+0.03` to `+0.08` | gain is a real secondary term; D's claim must be stated as net of it |
| `< +0.03` | factor (b) is not the mechanism; D's population claim is clean |

Clamp `p` to `[0, 0.95]` to keep the gain finite, and record the realized gain distribution.

### Cell T — true removal via `key_padding_mask`

Build the boolean mask from the same Bernoulli draw and pass it as `key_padding_mask` so dropped
units are genuinely absent from the softmax. Do **not** zero `src`, so no constant placeholder is
created, and do not apply the `1/(1-p)` pre-`fc_in` rescaling (attention already renormalizes over
surviving keys).

This isolates factor (a) and answers what the mechanism is:

| outcome | consequence |
|---|---|
| `T > D` | the placeholder was a handicap; genuine set-invariance is both a better method and a clean, honest mechanism sentence. This becomes the headline system. |
| `T ≈ D` | placeholders are harmless; the effect is subset variation per se. Describe it that way. |
| `T < D` | **the constant placeholder / attention-sink mass is doing the work.** Surprising, and a more interesting claim than the current one — but it must then be stated as such, and the "population invariance" framing is wrong. |

Implementation traps: with `high = 1.0` a draw near 1 can mask every key and produce NaN from a
fully-masked softmax, which D never hits because its placeholders always exist. Enforce
`min_keep ≥ 1` (suggest 4) and record it. Also assert the mask is all-`False` at eval so scoring is
unmasked.

### Cell C — paired-subset consistency (the design step)

Runs in parallel with T. **T does not block C's execution, only C's interpretation** — see item 5.

Per step, draw **two independent whole-unit subsets** of the same sample using the *same* sampling
process as D, decode each with the same model, supervise **both** against the behaviour target, and
additionally require the two predictions to agree:

```text
mask_1, mask_2 ~ independent draws, each identical in law to D's mask
pred_1 = f(sample, mask_1)
pred_2 = f(sample, mask_2)

loss = behavior_loss(pred_1, target)
     + behavior_loss(pred_2, target)
     + lambda * consistency(pred_1, pred_2)
```

No teacher, no frozen target, no checkpoint initialisation; one shared model, both branches
backpropagate. This is the step from "random subset augmentation" (D) to an explicit statement that
*different available unit sets must yield the same task estimate*, which is precisely the deployment
condition.

Two properties of this formulation are load-bearing and were wrong in the first draft of this
document, which proposed task loss on the full population plus consistency between two **disjoint
halves**:

- **Supervise both branches, not the full population.** If only the full population carries the task
  loss, the two branch decodes are constrained *only* by the consistency term, so they can agree on
  something wrong; the collapse mode then has to be suppressed by tuning `lambda`. Supervising each
  branch against the target anchors both and removes the collapse mode structurally.
- **Independent subsets, not disjoint halves.** Disjoint halving silently replaces D's
  `p ~ U(0,1)` rate with a fixed 50%, so `C − D` would confound the rate distribution with the
  consistency objective. Drawing each mask with D's exact law makes each branch's marginal identical
  to D's training distribution, so `C − D` isolates the consistency term alone.

Six things to get right:

1. **Record the realised overlap.** Independent draws can overlap heavily (two draws at `p = 0.2`
   share about 64% of units), and when both draws are near-complete the consistency term is
   trivially satisfied. Log the per-step Jaccard overlap distribution. If C returns null, this
   diagnostic is what distinguishes "the objective does not help" from "the constraint never bit".
2. **Prior art, stated first, not defended afterwards.** This loss *is* R-Drop
   (`task(pred_1) + task(pred_2) + λ·divergence(pred_1, pred_2)` over two stochastic passes; KL
   becomes MSE in the regression case), and two-view agreement is co-training. Verify the citation
   before writing. The defensible claim is **not** the objective form; it is that R-Drop's
   elementwise perturbation is the wrong perturbation for population decoding and a whole-unit one is
   right — for which the evidence is already in hand, `R = 0.1877` against `D = 0.4179` at matched
   removal amount, 0/15 sessions favouring elementwise. Position the contribution that way, with the
   R cell as its positive evidence rather than as a defensive control.
3. **The 2× compute confound must be decided at launch, not afterwards.** Two forward and two
   backward passes per sample. D is 48 epochs at ≈8h20m; the naive version is ≈16h+. Three options,
   each with a cost: accept 2× wall-clock (C is then no longer step-matched to Arm A); halve the
   batch (changes gradient noise); or concatenate the two branches into one batch of the same total
   size (step count and memory comparable, effective distinct-sample batch halved). **Pick one,
   record it in the launch receipt, and pre-register that if `C − D` is small a compute-matched D
   control at 96 epochs is required before any claim.**
4. **Consistency site.** Penalise the **behaviour predictions**, not a latent. Latent agreement
   admits degenerate rescaling/rotation solutions and does not match the deployment statement.
   Freeze one `lambda` (suggest 0.1). Do not sweep it.
5. **Interpretation depends on T.** If T shows D's effect comes from the constant-placeholder /
   attention-sink mass rather than genuine absence (§2), then the consistency term is enforcing
   invariance over *placeholder configurations*, not over sub-populations. The method would still
   likely work, but the mechanism sentence would be wrong. Do not write the claim until T reports.
6. **Small populations.** Record per-session unit counts and set a floor below which a branch is
   skipped for that sample. Machinery precedent: `streaming_calibration_module.py:41-56`
   (`split_ssc_t4_support_views`) already builds frozen disjoint views of the M24 calibration tensor;
   the required change applies subset sampling to query units instead.

Gate: `C − D` external mean `≥ +0.03` with `≥ 10/15` positive, and within `≥ −0.03`, on the governing
convention. Do not sweep `lambda` on a failure. Report `C − T` as well once T lands.

### Step 0C — test-time unit-loss curve and subset ensembling (zero training, run before C lands)

Two measurements on **existing** checkpoints (`armA_direct_t4_48/swa_final4.pt`,
`pop_robust_v1/cellD_2heads_dynamic_dropout/swa_final4.pt`), no GPU training required. Both are
prerequisites for C being a *design* contribution rather than a better number.

**(i) Unit-loss robustness curve.** Score external R² as a function of the fraction of units removed
at test time, on a fixed grid (suggest 0, 0.1, 0.25, 0.5, 0.75), several mask seeds per point,
for Arm A and D now and for T and C when they exist. Existing control conditions (`zero`,
`wrong_pair`, `destroyed_activity`) do not cover this.

This supplies the differential prediction C currently lacks. C promises only "a higher number" over
D; what it *should* promise is a **flatter curve**, which demonstrates the invariance directly
instead of inferring it from a training loss. Pre-register it that way: C's curve flatter than D's is
the claim, and it is a figure.

**(ii) Test-time subset ensembling, as a competitor to rule out.** Average predictions over several
independently subsampled populations at inference, on the unmodified Arm A and D checkpoints. If this
recovers a large share of what C is expected to deliver **without any training change**, C's value
must be recomputed against it and the paper must report it as the cheap baseline. This is the most
dangerous cheap alternative to the whole direction and it costs no training.

### Cell W — output head and within-window structure (evidence-backed, larger build)

Now motivated by data rather than intuition. DH scores 0.2562 at full-window and **−1.1373** at
last-bin; Arm A gains +0.0993 from the same granularity change. The within-window prediction profile
is therefore a large and completely uncontrolled degree of freedom, while `fc_out` is 0.73% of
parameters emitting all 50 bins from a single 512-d vector per slot with no temporal model anywhere
(`spint.py:466`).

Build the zero-initialised temporal latent residual specified in
`HANDOFF_SPINT_DECODER_DIRECTIONS_20260817.md` §5 Priority 2 — `K = 8` latent slots over a learned
temporal basis, added to the existing output so the model is bitwise identical at initialisation.
Do not change `num_covariates`; that breaks the output interface (`spint.py:421-423, 466-467`).

Report both granularities for this cell specifically, since it is the one intended to change the
within-window profile.

### Cell E — population read-out as an unbiased estimator — **CLOSED UNSTARTED (2026-08-19)**

Closed on C's counter-evidence; see §0.2. The design below is retained only so that a future reader
can see what was rejected and why. Do not start it.

The architectural expression of the same insight. Instead of obtaining subset-invariance
stochastically, make the aggregation an estimator whose *expectation* is invariant to which units are
present and whose *precision* varies: per-unit contributions combined by learned inverse-variance
weights, with the precision optionally a function of the carrier and calibration reliability. The
codebase has a reliability-encoder precedent (`B3PreservingReliabilityEncoder`). Hold this until
G/T/C have identified the active ingredient, otherwise it is another bundle.

---

## 4. Order and prohibitions

**Historical as of 2026-08-19 — every item below has executed; see §0 for outcomes.** The
prohibitions at the end of this section remain in force and are inherited by the successor document.

The operator's goal is a design contribution, not another decomposition finding, so the allocation
is innovation-forward with the claim-killer control deferred but not dropped.

**Do first, no GPU training:** Step 0C (i) and (ii) — the unit-loss robustness curve and test-time
subset ensembling on the existing Arm A and D checkpoints. (ii) can invalidate the premise of C
before a single GPU-hour is spent, and (i) establishes the baseline curves that C's claim will be
measured against.

**Two GPUs, in parallel: C and T.** C is the design cell. T resolves what D's active ingredient
actually is (§2) and is itself a candidate headline system, since true removal may simply be better
than the placeholder form. Neither blocks the other; only C's *mechanism sentence* waits on T.

**Next batch: G.** It is the claim-killer control for "this is just random input-scale
augmentation". Its prior of firing is low — R already contains the same `1/(1-p)` rescaling and came
in *below* baseline — but it is not optional before the paper is written. Do not let it slip past the
write-up.

**Later: W**, independent and larger. ~~**Hold E** until G/T/C have identified the active ingredient,
otherwise it is another bundle.~~ **E is now CLOSED UNSTARTED, not held — see §0.2.**

A standing constraint on all of it: no superiority claim over A2 is admissible until D (or its
successor) has seeds 43/44, because A2 is a three-seed reference with an external spread of 0.066.
The operator has deferred those seeds deliberately; that defers the *claim*, not the requirement.

Explicitly do not: run seeds 43/44 yet (operator decision — but note no superiority claim over
three-seed A2 is admissible until they exist); sweep dropout strength; revive the 64-head line
(DH is −1.14 at the governing granularity); revive carrier-sector sparsification (S2 gate rejected);
change width; use pretraining or any teacher checkpoint in a training path; change
`behavior_scaling_factor` silently.

---

## 5. Receipt requirements

Carry forward every fix already earned, plus one new one:

1. `n_windows` per session in every `per_session` entry.
2. `num_heads`, dropout structure, `p` distribution and clamp, `min_keep`, rescaling policy, and
   `lambda` in the integrity block — not only in a config sub-dict.
3. Both granularities and both aggregations, with (last-bin, equal-session) labelled governing.
4. The 2014 / 2015 date-block split and `sub-M_ses-CO-20141203` reported separately.
5. Conjunctive gates with no bootstrap escape clause. Note that
   `scripts/run_tfap_stage3.py:369-374` still has no branch for mechanism-pass-without-
   engineering-pass and mislabels that state `BOTH_GATES_FAIL__STOP_ROUTE`; do not copy the pattern.
6. **New: record the realized perturbation statistics** — for G the gain distribution, for T the
   realized surviving-unit counts, for C the per-session split sizes and the two loss terms
   separately. The R-versus-D result only became interpretable because amount was matched; future
   cells must make matching auditable rather than assumed.

---

## 6. Framing constraints

- ~~Until T reports, the mechanism sentence must be written as "random unit-token ablation to a shared
  constant, with gain-compensated survivors", not "training on random sub-populations".~~
  **LIFTED 2026-08-19:** T ≈ D with neither the gain nor the constant-token placeholder, so
  "training on random sub-populations" is now the licensed sentence, and T is the cleaner system to
  describe. See §0.1.
- The carrier remains necessary: A2's Z4 external is −0.1461 / −0.1189 / −0.1108. The claim is
  sparsification **plus** carrier.
- Report S2 as a negative result that narrows the mechanism, not as a failed idea to be buried.
- Any document still citing an A2 external advantage of +0.180 is wrong; the matched figure is
  **+0.0857 (8/15, CI [+0.020, +0.165])** and D now exceeds A2 on that surface outright.

## 7. Receipts

- **`results/subpop_score_v1_r2/subpop_score_receipt.json` — AUTHORITATIVE for C/G/T/W (§0)**
- `results/subpop_score_v1/…`, `results/subpop_score_v1_r1/…` — earlier cumulative cuts of the same
  scoring run (C/T, then C/G/T); shared contrasts bitwise identical to `_r2`, no `supersedes` field
- `results/subpop_step0c_v1/step0c_receipt.json` — unit-loss curves and subset ensembling
- `results/sparsification_step0_v1/step0_receipt.json` — last-bin D/DH, governing bars, scaling audit
- `results/sparsification_score_v1/sparsification_score_receipt.json` — R and S2, date blocks, S2 gate
- `results/sparsification_theta_authority_v1/theta_authority_receipt.json` — S2 carrier-angle authority
- `results/pop_robust_v1/matched_score_receipt.json` (sha `91bf81f9…`) — D/DH full-window
- `results/a2_matched_rescore_v1_r1/a2_rescore_receipt.json` — A2 matched, 3 seeds (v1 is VOID, missing `/5.0`)
- `streaming_calibration_exp/src/models/components/spint.py:26,35,401-403,449-455,464,466` — the code facts in §2
