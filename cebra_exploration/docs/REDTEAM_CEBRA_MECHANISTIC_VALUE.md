# Red team: does CEBRA have mechanistic value for our network?

**Role:** adversarial review of a coordinator conclusion. Assume it is wrong; find the strongest case
against it. **Date:** 2026-08-14. **Compute:** CPU only, `CUDA_VISIBLE_DEVICES=` on every command.
**Nothing in the repository was modified** except this file. Probe code lives in `/tmp/redteam/`.

## The conclusion under attack

> "CEBRA offers no usable mechanistic value for improving our network. Its ideas either violate our
> no-backprop constraint or are incompatible with our closed-form estimator. What survives is only
> rhetorical value for the paper plus a methodological discipline."

# Verdict: **partially overturned**

Three separate claims are bundled in that sentence and they do not share a fate.

| Claim | Verdict |
|---|---|
| "Its ideas either violate the no-backprop constraint or are incompatible with the closed-form estimator" | **Overturned.** False disjunction, refuted by the project's own documents at zero cost. §1 |
| "No usable mechanistic value" | **Overturned.** One whole CEBRA mechanism class was never seen by anyone, it is source-training-only, and it is backed by vendored code. §3–§4 |
| "What survives is only rhetorical value for the paper" | **Overturned, but not in the direction expected.** The largest single finding here is that a CEBRA fact the project treats as settled — F2b — is **wrong as stated**, and the paper's efficiency contrast is exposed. §2 |
| The coordinator's own doubt: "the source-training surface is large and unconstrained" | **Partially upheld against the coordinator.** The surface is real but it is *much* thinner than intuition suggests, and I can say exactly why: three of the five elements the coordinator listed as unexplored are already present in our architecture or already measured to be harmful to us. §7 |

**Net:** the conclusion is wrong, but so is the framing that replaced it. The correct statement is that
CEBRA's *objective-side* elements are largely dead for us for identifiable reasons, while its
*data-construction* elements are live and were never examined.

One correction to the brief I was given, since it affects what counts as the target. The brief says the
prior work explored only one source-training idea. The brainstorms in fact proposed source-training
ideas repeatedly — S1, S4, S5, S7 and Grok's 4 and 7 all name a source stage. What is true, and
sharper, is that **only two of the eighteen numbered proposals touch the source *objective*** (S4 and
Grok 4, which are the same idea and are the one being screened), and **none touches the source *data
distribution***. That is the actual gap, and it is where §3 and §4 land.

---

# §1 — The stated conclusion is refuted by the project's own documents

Zero cost, no experiment, and it should have been caught before the conclusion was written.

Of the eighteen numbered proposals across `TRACK_C_BRAINSTORM_GROK.md` (9) and
`TRACK_C_BRAINSTORM_OPUS.md` (S1–S9), **sixteen carry an explicit line reading "Violates
no-target-session-backprop? **No**"**. Only Grok 9 and S9 are labelled violating, and both are
labelled by their own authors as belonging to Track B rather than Track C. Several are explicitly
annotated "cost is at source-training time only, i.e. free for us" (S1, S4).

So the disjunction "either violate the constraint or are incompatible with the closed-form estimator"
is false on the first branch for sixteen of eighteen proposals, by the authors' own declaration. The
second branch is also weaker than stated: S1(a), S1(c), S3, S5, S7 and Grok 6 are all *closed-form
estimator changes* — they are not incompatible with the estimator, they *are* the estimator.

This matters beyond pedantry. The conclusion's stated reason is not the real reason the ideas stalled.
The real reason, visible in `COORDINATOR_VERIFIED_FINDINGS.md` F11–F14 and in
`AUDIT_OF_COORDINATOR.md` finding 7, is that the audit chain kept pulling every idea back to
estimator-level evidence about the H1 sparse carrier and then ranking it against that evidence.
That is a different failure — under-screening, not constraint conflict — and it prescribes a different
remedy: screen what you already have rather than conclude the surface is empty.

---

# §2 — F2b is wrong as stated, and this is the strongest finding in this document

**F2b is one of the three findings the project calls "the ones that matter most."** It says:

> "Multi-session CEBRA shares **ZERO** weights across sessions … one **complete, independent encoder
> per session** … total params 22,790, i.e. **O(N_sessions × full encoder)** … parameter growth: linear
> in session count."

It is then used to sharpen the paper's central efficiency contrast (`PROJECT_VISION.md:41-42`,
`README.md:30-31`), in a table row reading *weights shared across sessions: **none** | the entire
decoder* and *parameter growth: linear in session count | **zero***.

**Vendored CEBRA 0.6.1 ships a multi-session solver in which a single model is shared across all
sessions.** It is `cebra.solver.UnifiedSolver` (`cebra/solver/unified_session.py:59-143`) with
`cebra.data.UnifiedDataset` / `UnifiedLoader`. `_get_model` returns `self.model`, not
`self.model[session_id]`; `_inference` calls `_single_model_inference(batch, self.model)`.

Measured, on the *same* two sessions F2b used (N = 24 and 31, `offset10-model`,
`output_dimension=3`) — `/tmp/redteam/unified_solver_probe.py`:

```
UnifiedDataset.input_dimension: 55 (= 24 + 31)
solver class                  : UnifiedSolver
solver.model type             : Offset10Model
is nn.ModuleList              : False
total trainable params        : 13155
SAME OBJECT for both sessions : True

MultiSession per-session params: [11171, 11619] -> total 22790     <- F2b's exact numbers
Unified total params           : 13155
```

`_get_model(0) is _get_model(1)` is `True`. F2b's 11,171 / 11,619 / 22,790 reproduce exactly on the
multi-session path, so the measurement was right; **the generalisation from it was not.** "Multi-session
CEBRA shares zero weights" is a true statement about `MultiSessionSolver` and a false statement about
multi-session CEBRA in the vendored tree.

**Why the mistake was invisible.** `UnifiedSolver` is not reachable from the sklearn estimator —
`'unified' in inspect.getsource(cebra.integrations.sklearn.cebra)` is `False`. Every measurement in
this project went through `CEBRA(...)`, which only builds the `nn.ModuleList`. It is also a
post-Nature-2023 addition, so reasoning from the paper would not surface it. The word `Unified`
appears **zero times** in `cebra_exploration/docs/` — not in the brief, either brainstorm, either
cross-review, the coordinator's findings, or the audit.

**The paper consequence, and it cuts both ways.** As written, the contrast table row is falsifiable by
a reviewer who knows CEBRA 0.6: "parameter growth linear in session count" is not a property of CEBRA,
it is a property of one of its two multi-session solvers. But the correct replacement is *stronger*
than what the table currently claims, not weaker. Measured on the fitted unified model:

```
--- can the fitted unified model accept a session it was not trained on? ---
new session alone            : RuntimeError ... expected input to have 55 channels, but got 37
swap session 1 for new N=37  : RuntimeError ... expected input to have 61 channels, but got 55
all trained sessions present : OK, embedding (600, 3)
```

The unified model's input width is hard-wired to the **sum** of all training sessions' neuron counts,
and `UnifiedSolver.transform` (`:224-237`) rebuilds a `UnifiedDataset` over *all* sessions and requires
`inputs` for every one of them. So the price of sharing weights is that **a new session cannot be
embedded at all**, and even embedding a *training* session at deployment requires streaming every
other source session's neural data alongside it.

That is a far sharper statement of our advantage than a parameter count. The honest and defensible
table row is a trilemma, and we sit on the only good corner:

| | CEBRA `multi-session` | CEBRA `unified-session` | Ours |
|---|---|---|---|
| weights shared across sessions | none | **all** | all |
| params for K sessions (24/31-unit toy) | 22,790, grows with K | **13,155**, fixed | fixed |
| accepts an unseen session | via `adapt`, which does not land in the source frame (F8) | **cannot, at any cost** | closed-form solve |
| needs other sessions' data at inference | no | **yes, all of them** | no |

**Action.** Restate F2b to scope its claim to `MultiSessionSolver`, and replace the parameter-growth row
in `PROJECT_VISION.md`/`README.md`/the paper with the trilemma. This is writing, not compute, and it
removes a live reviewer objection while strengthening the claim. Note that `AUDIT_OF_COORDINATOR.md`
finding 9 already flagged the parameter-growth row as overreaching, for a different and weaker reason
(deployment vs training-time growth); the real reason is that a shared-weight solver exists.

---

# §3 — The idea the prior work missed: pseudo-session unit-set mixing

This is the strongest *idea* in this document. It is source-training-only and it touches the target
session **not at all**.

## What CEBRA does

`cebra.data.UnifiedDataset` (`cebra/data/datasets.py:442-453`):

> "Considering the sessions as a unique session, or **pseudo-session** … To do that, we sample
> ref/pos/neg for each session and **concatenate them along the neurons axis**."

and `UnifiedSampler.sample_all_sessions` (`cebra/distributions/multisession.py:508-560`):

> "The prior for all sessions, creating a **'pseudo-animal'**, where `idx` sampled in different sessions
> correspond to points in the recordings where the auxiliary variables are similar."

So: pick a time in session A; find, by nearest-neighbour search on the behavioural index, the times in
sessions B, C, … where behaviour matches; **take the union of their units as one input**.

## Why it transplants into us and not into anyone else

Our decoder is permutation-invariant over a variable-size unit set. CEBRA needs a fixed input width, so
its own pseudo-session is a rigid concatenation to a fixed total. **We can consume a pseudo-session
natively, with no architectural change** — `key_padding_mask` is already plumbed through
`CrossAttentionLayer.forward` (`SPINT-main/src/models/components/spint.py:23-33`), and unit-set
subsampling already exists as `dynamic_dropout` (`:131-137`).

The current pipeline **forbids** it by assertion. `falcon_module.training_step:138-141`:

```
if len(set(session_name)) == 1: ...
else: raise ValueError("All samples in the batch should belong to the same session")
```

## Mechanism

Two effects, and I rank them in the order I can defend them.

1. **Combinatorial expansion of the number of training domains.** The held-in/held-out gap is a
   too-few-domains generalisation gap: the source-trained decoder sees on the order of five distinct
   unit-set compositions. Chimeric sets drawn from K sessions with per-session subsampling generate
   a combinatorial number of distinct compositions, each with a valid behaviour target. This is an
   input-distribution change, so unlike a batch-ordering change it genuinely changes the objective.
2. **It removes any session-wide common mode from the input.** Within one training example, units now
   arrive with different recording gains, different baseline distributions and independently estimated
   carriers, so a route conditioned on "which session is this" has nothing consistent to key on.
   *I tested this second mechanism and it did not hold — see the honest section, §8.1.*

## What I measured

`/tmp/redteam/pseudosession_probe.py` builds a synthetic positive control in the project's own style: a
SPINT-shaped decoder (structure copied from `spint.py` — shared `fc_in` for units and queries, pre-norm
cross-attention with a shared `norm1`, FFN residual, linear read-out, last-bin MSE) with T4-shaped
closed-form carriers `[ŵ_x, ŵ_y, m, b̂]` fitted by OLS on a calibration prefix, four source sessions
and four held-out sessions with fresh units and a fresh session gain. **8 seeds per arm**, means:

| regime | arm | held-in | held-out | **gap** | gap sd |
|---|---|---:|---:|---:|---:|
| clean carrier (t_cal 800) | single_session (current) | +0.9606 | +0.9171 | **+0.0435** | 0.0238 |
| | pseudo_matched | +0.9451 | +0.9158 | **+0.0293** | 0.0115 |
| | gauge_augment | +0.9632 | +0.9248 | **+0.0384** | 0.0233 |
| noisy carrier (t_cal 60) | single_session (current) | +0.8586 | +0.7517 | **+0.1069** | 0.0714 |
| | pseudo_matched | +0.8206 | +0.7479 | **+0.0727** | 0.0442 |
| | gauge_augment | +0.8577 | +0.7895 | **+0.0682** | 0.0519 |

Read this conservatively, because the project's own history (F8's −1.2278) is a lesson about quoting
one draw as characteristic. With 8 seeds the standard error on the baseline gap is 0.008 (clean) and
0.025 (noisy), against differences of 0.014 and 0.034. **Both interventions reduce the mean gap in both
regimes and roughly halve its seed-to-seed variance, and neither difference is established at 8 seeds.**

Two design findings that *are* robust, because they are structural rather than statistical:

- **Size-matching is mandatory.** The naive union (`pseudo_session`, k sessions each contributing all
  their units) made the gap *worse* in every regime — +0.0793 vs a +0.0562 baseline in the 3-seed
  sweep — because the model then trains on sets of ~3N units and is evaluated on sets of N. Taking
  N/k units from each of k sessions (`pseudo_matched`) is the version that helps. Anyone implementing
  this without size-matching will get a negative and attribute it to the wrong thing.
- **`pseudo_matched` costs held-in R²** (−0.016 clean, −0.038 noisy) while `gauge_augment` costs
  essentially nothing (−0.001 noisy) for the same gap reduction. In the deployment-realistic noisy
  regime **the cheap idea dominates the expensive one.**

## The feasibility result that scopes the idea — and it is the most transferable number here

The construction injects label noise equal to the nearest-neighbour behaviour residual, which scales as
`n^{-1/d}` in the behavioural dimension `d`. Measured on AR(1)-smoothed behaviour, residual as a
fraction of the behaviour scale (`/tmp/redteam/match_residual_probe.py`):

| d | n=1,000 | n=5,000 | n=20,000 | n=100,000 | n for a 5% residual |
|---:|---:|---:|---:|---:|---:|
| **2** (center-out / RT planar velocity) | 0.057 | **0.026** | 0.013 | 0.006 | **1.6e3 bins** |
| 3 | 0.158 | 0.086 | 0.055 | 0.031 | 2.7e4 |
| 4 (H1 source-frozen latent) | 0.254 | 0.163 | 0.115 | 0.077 | 5.6e5 |
| **7** (H1 raw DoF) | 0.474 | **0.368** | 0.297 | 0.237 | **2.7e9 bins** |

Measured slopes track `−1/d` to within 0.03 in every row, so this is the curse of dimensionality and
not an artefact of my generator.

**The consequence redirects the whole effort.** Pseudo-session mixing is nearly free of label noise on
the 2-D center-out and RT lanes and is **infeasible on H1's 7-DoF covariate** — 37% residual at a
realistic session length, and no achievable `n` fixes it. Matching on the source-frozen 4-D latent
helps but not enough. CEBRA's own sampler anticipates this: `search_or_mask(query, threshold=...)` sits
commented out at `multisession.py:556`, i.e. reject a bad match instead of accepting it. On H1 the
acceptance rate at a 5% threshold would be negligible.

Track C spent its entire effort on H1. **The one new mechanism CEBRA offers applies to the lanes where
H1 is not** — center-out and RT, which is where the carrier's strongest results already live (external
subject-M T4/Z4 `0.3414 / −0.1434`, a margin of `+0.4848`, i.e. the carrier is doing essentially all
the work under subject shift, which is exactly a domain-count problem).

## Cheapest decisive test

CPU, behaviour arrays only, no neural data, no checkpoint, no GPU: compute the nearest-neighbour
velocity residual **between real subject-C center-out sessions** and compare it to the residual
variance of the deployed decoder. Confirms if the injected label noise is well below the model's own
error; refutes if it is comparable. That is the only gate; everything after it is 2 GPU runs.

**Falsification of the idea itself:** train one source model with size-matched pseudo-sessions and
compare held-out subject-M R² against the sealed T4 checkpoint. Kill if held-out does not improve, or
if held-in falls by more than the held-out gain.

---

# §4 — S1 deserves reinstatement, in a form nobody proposed

**Yes, reinstate it.** The audit is right that F13b's downgrade rests on an inference that does not
distinguish "the network compensated" from "the coordinate was least useful", because the carrier
columns are zero-initialised and trained under Adam. But I am not reinstating S1 as argued. Both
brainstorms and the audit treat S1 as an *interface* question — rescale the carrier vector before it
reaches `ψ_c`. That framing is what made it litigable against the trained weights, and it is why it
stalled.

**The content of S1 is a gauge problem, and its natural fix is at source training.** Three of T4's four
coordinates — `â`, `ĉ`, `b̂`, and hence `m` — are in firing-rate units. Rate units are session-specific
(sorting, thresholds, bin occupancy, animal). Only `atan2(ĉ, â)` is gauge-free. So the carrier carries a
session-specific scale that the consumer must either learn to ignore or memorise per session.

CEBRA's answer to a gauge is to quotient it out. Ours can be to **randomise it at source training**:
multiply the raw rates of each training example by a random positive scalar. Because both the activity
summary and the OLS carrier are homogeneous of degree 1 in the rates, they scale *identically and
exactly* — no approximation, no new estimator, and the target-session path is byte-for-byte unchanged.
This is `gauge_augment` in the table above, and in the deployment-realistic regime it was the best arm:
gap 0.1069 → 0.0682 with a held-in cost of 0.001.

**The honest caveat, stated before anyone else can.** In my synthetic the nuisance *is* a scalar gauge
and my augmentation is exactly its inverse group action, so that arm is close to being told the answer.
The probe shows that gauge augmentation removes a gauge when the nuisance is a gauge. It says nothing
about whether real cross-session variation is one. And there is evidence against, on H1: C2's R7
measured carrier tuning-coefficient RMS spanning only 1.30× and intercept RMS 1.11× across 13 held-in
recordings. If that is the whole scale story, gauge augmentation buys nothing on H1.

**The one unrun check that settles S1 across both lanes.** C2's own §3 item 9 says: *"I did not measure
the carrier coordinate scales on center-out / RT (T4). My claim that H1's imbalance is unusually severe
is a prediction, not a measurement."* **Nobody ran it.** It is minutes of CPU and it discriminates every
live reading at once:

- measure per-coordinate RMS of T4 on subject-C sessions and on subject-M sessions;
- if T4 is balanced where the carrier works (center-out, `+0.2490` within-domain margin) and imbalanced
  where it fails (H1-SE5, 66–125× per the audit's corrected figure), the scale hypothesis acquires a
  real cross-arm correlate and S1 is back as an explanation, not just a fix;
- if T4 is equally imbalanced and still works, S1 dies cleanly on its own evidence;
- and the *cross-session and cross-animal spread* of those scales is the number that decides whether
  gauge augmentation is worth 2 GPU runs. 1.1× says no. 3× says yes.

This is the highest expected value item in this document per unit of compute, and it has been sitting
unexecuted with a written note pointing at it since the brainstorms were filed.

---

# §5 — An immediately actionable fix for the loss currently being screened

The cross-session consistency / InfoNCE term now in screening (S4 / Grok 4) is, in both proposals,
applied **directly to the representation the linear read-out consumes** — S4 puts it on the L2-normalised
tokens, Grok 4 on `h` or on `ŷ_c` "before readout".

The project has already measured that this is the harmful direction, and nobody connected the two
documents. F15, strengthened by `AUDIT_OF_COORDINATOR.md` finding 2 to 8/8 monotone seed×sample cells at
two penalties: **as InfoNCE trains, linear decodability of the embedding falls monotonically**
(0.8084 → 0.7125 and 0.7522 → 0.6924 at λ=1; 0.992 → 0.927 at λ=1e-2), while **kNN rises in 0/8 cells,
i.e. kNN gets better**. F15's own reading is correct: "the embedding spreads toward uniformity on the
hypersphere, which is good for the contrastive objective and bad for a linear readout."

Our read-out is `v̂_c = W_ro ŷ_c + b_ro` — linear, and the sole path to the loss. So an InfoNCE term
placed on `ŷ` imports a pressure that has been measured, in this repository, to cost 0.10–0.20 R² of
*linear* decodability in CEBRA's own embeddings.

**The fix is one line and it is standard practice the prior 16 ideas never mention: put the contrastive
term behind a projection head.** Compute `z_c = P(ŷ_c)` with a small MLP `P` used *only* by the
auxiliary loss and discarded after source training. The uniformity pressure then shapes `z`, and reaches
`ŷ` only through the shared trunk, which is the whole point of the head in SimCLR-style training. Zero
deployment cost, zero change to eq. 3–7.

**Pre-registered test for the screen already in flight:** run the auxiliary term with and without the
head, and log held-in linear R² alongside the consistency metric. If the no-head arm shows the F15
signature — consistency improves while held-in linear R² declines with auxiliary weight — the head is
required and the no-head result was a false negative. This should be added to the screen now, because
it costs one extra arm and it is the difference between "the idea failed" and "the idea was attached at
the wrong point".

---

# §6 — The spectral basis: the objection is answerable, and answering it dissolves the idea

The coordinator suspects C1's Idea 1 (Laplacian eigenvectors of a calibration similarity graph as the
task basis `φ`) was suppressed too fast because it did not specify cross-session alignment. **The
objection is answerable.** Eigenvectors of a *per-session* graph are session-specific and defined only
up to sign, up to arbitrary rotation within degenerate eigenspaces, and up to reordering when eigenvalues
are close — so `c_i` fitted against `Φ_A` and against `Φ_B` are in incommensurate bases and a shared
`ψ_c` cannot read both. The answer is the device the pipeline already uses twice: **do not use the
target session's own eigenvectors.** Fit the spectral map on pooled source sessions, freeze it, and
Nyström-extend it to the target session's graph — exactly what `P_src` and the source-frozen 7→4
projection `U` already do in eq. 15.

So reinstate it? **No — and for a better reason than the one given.** Fixing the alignment objection
collapses Idea 1 into C1's own Idea 7 (source-frozen similarity embedding, Nyström-extended), which was
already on the list. And what remains of Idea 1's distinct value does not survive contact with two
results the project already owns:

1. **The conditioning half is already refuted.** A Laplacian eigenvector basis is orthogonal and
   eigenvalue-ordered — i.e. a *whitened, well-conditioned* design. The project measured that condition
   number does not predict H-SE5 forward transfer, and that reducing the ridge's dimension by PCA
   *hurt* it. `BACKGROUND_BRIEF.md:96-97` explicitly forbids reinstating posedness as a causal claim.
   Idea 1's conditioning advantage is therefore an advantage along an axis already shown not to bind.
2. **The observations half does not need an eigendecomposition.** Idea 1's real content is "use the
   ~2,400 calibration bins instead of the ~20 event snapshots". If that is the value, take it directly:
   interpolate the label onto off-event bins and add them as extra OLS rows. Same observation count,
   pooling linearity preserved for the same reason (the design stays neural-independent), no spectral
   machinery, no sign ambiguity, no alignment problem to solve, and no `λ` or `τ` to tune. **If the
   cheap version works, Idea 1 is a more expensive route to the same place; if the cheap version fails,
   the extra bins were not the bottleneck and Idea 1 fails with it.**

That is the version worth screening, and it is a simplification of C1's idea rather than a
reinstatement of it. It is also a clean discriminator: it separates "more observations" from "better
basis geometry", which Idea 1 confounds.

---

# §7 — Why the source-training surface is thinner than it looks

The coordinator asked for the specific reason if the surface turned out barren. It is not barren, but
three of the five elements listed as unexplored are dead, and each for a different and identifiable
reason. This section is the part I expect to be most useful, because it prevents three wasted screens.

## 7.1 Unit-hypersphere normalization is **already in our architecture**, at the exact point it matters

CEBRA's `FixedCosineInfoNCE` requires L2-normalised inputs, and F8/F16 measured `emb_norm ≈ 1.0` in all
13 configurations. The transplant looks obvious: our `ŷ` has no norm constraint.

But look at where the similarity is computed. `CrossAttentionLayer.forward`
(`spint.py:24-33`) applies **the same** `norm1 = LayerNorm(d_model)` to the query *and* to the
key/value before attention. LayerNorm maps `h → γ⊙(h−μ)/σ + β`, and the normalised part has L2 norm
exactly `√d`. Measured (`/tmp/redteam`, 4096 samples, d=256) across a 300× sweep of input scale:

```
 input scale      ||h||    ||LN(h)||   cv(||LN(h)||)
         0.1      1.786      15.9919         0.00005
        30.0    536.663      16.0000         0.00000
sqrt(d) = 16.0
with a non-trivial affine (gamma~N(1,.3), beta~N(0,.3)): ||LN(h)|| cv = 0.027
```

The keys and the query already lie on an affine image of a fixed-radius sphere, invariant to a 300×
change in the scale of `h`, with a residual coefficient of variation of ~3% once the affine is
non-trivial. **CEBRA's hypersphere is already present, applied to both sides of the similarity, using a
shared norm.** The "session-dependent effective attention temperature" failure mode that this element
would fix is bounded at a few percent, not orders of magnitude.

And the one place we genuinely lack it — `ŷ_c`, which has no final LayerNorm before `fc_out` — is the
one place it would *hurt*: `v̂ = W_ro ŷ` is a linear regression onto velocity, and a norm constraint on
`ŷ` would remove the magnitude degree of freedom needed to represent speed.

## 7.2 A learnable temperature is a no-op up to reparameterization

`LearnableCosineInfoNCE` learns `1/τ` multiplying a **cosine** similarity. The learnable temperature is
necessary there precisely because cosine is bounded in [−1, 1], so the logit scale cannot be learned any
other way.

Our attention computes `⟨W_q LN(q), W_k LN(k)⟩ / √d_head` with `W_q` and `W_k` **unconstrained**. The
product of their gains *is* an inverse temperature and it is already a free trained parameter. Adding an
explicit scalar `1/τ` in front adds a redundant direction to the parameterisation. This element is not
unexplored; it is already there.

## 7.3 Negatives and uniformity pressure point the wrong way for a linear read-out

This is the real answer to "why is the surface thin", and it is the project's own measurement. See §5:
F15 plus audit finding 2 establish, across 8/8 monotone cells at two penalties, that InfoNCE training
*reduces* linear decodability while *improving* kNN. Our read-out is linear and our entire comparator
convention is linear. Importing InfoNCE's uniformity pressure onto the representation the read-out
consumes imports a measured 0.10–0.20 R² liability.

This does not kill the element — §5's projection head is the standard containment — but it explains why
the "add negatives" family cannot simply be dropped into an MSE regression decoder, and it is why the
one source-objective idea in flight needs the extra arm.

## 7.4 Two more elements are already implemented under other names

- **Auxiliary-variable-conditioned batch construction, sampling-noise half.** S4's positive pairs
  ("the same channel, carriers fitted on two different calibration windows") require carriers refit at
  many windows. `build_sparse_source_assets` already fits a fresh M4 carrier at every legal contiguous
  4-trial start of every source session (C2 §0.3, which killed its own R2 on these grounds). The source
  consumer already trains on deployment-noise-matched carriers.
- **Set augmentation.** `dynamic_dropout` (`spint.py:131-137`) already drops a random fraction of units
  per batch with `p ~ U(low, high)`. Random unit subsets are already an augmentation; what is missing is
  only that the subsets never *cross* sessions, which is §3.

## 7.5 The hybrid time-plus-behaviour objective is an idea the project already has and has not connected to CEBRA

CEBRA-Time's contribution is that temporal structure is free supervision. Our decoder emits
`behavior_pred` of shape `B×W×C` — the whole window — and then `decode_last_timestep_only: true`
(set in every relevant config, e.g. `falcon_h1_carrierid_hu.yaml:5`,
`falcon_h1_sparse_event_endpoint.yaml:5`) discards all but the last bin before the MSE. Supervising the
other `W−1` bins is a source-training-only change with zero deployment cost.

**In fairness this is not new to the project** — `HANDOFF_NEXT_ROUND_DIRECTIONS_20260812.md:602` already
proposes exactly it ("49 of 50 output bins carry no gradient"). What CEBRA adds is not the idea but a
reason to raise its priority: it is the *only* legitimate CEBRA-Time analogue available to us, since a
target-session temporal contrastive fit would need backprop, and it costs one config flag to test.

---

# §8 — What I tried that did not pan out

## 8.1 The session-common-mode mechanism, which was my headline mechanism for §3, is false

I predicted that pseudo-session mixing works by destroying session-identifiable common mode, so that a
"which session is this" shortcut has nothing to key on. I tested it directly: multinomial logistic
regression from the pooled representation `ŷ` to session identity, on held-in sessions.

**Session identity is 99.7–100.0% linearly decodable from `ŷ` in every arm, including both mixing
arms** (chance 25%). Mixing does not remove session identifiability — plausibly because the reference
session's units are still present in the chimeric set and the target still comes from that session.
Whatever benefit §3's `pseudo_matched` arm shows, it is not operating through this mechanism, and
mechanism 1 (domain-count expansion) is the only one I can still defend. I have left the claim in §3 as
mechanism 2 with this refutation attached rather than quietly deleting it.

## 8.2 The naive pseudo-session union makes things worse

Reported in §3 because it is a design lesson, but it is also a failed version of my own idea: unions
that let each of k sessions contribute all its units raised the gap from +0.0562 to +0.0793. The
set-size mismatch between training (~3N) and evaluation (N) dominates any benefit.

## 8.3 A behaviour-prior regime where nothing helped

I ran a third regime with per-session behaviour offsets and scales (`beh_mean 0.5`, `beh_scale 0.4`), on
the theory that a memorisable per-session behaviour prior is the sharpest available shortcut. Baseline
gap +0.0683; every intervention was **worse** (pseudo_session +0.1508, pseudo_matched +0.0985,
gauge_augment +0.1375), and seed sd reached 0.185 — larger than every effect in the table. At 3 seeds
this regime is uninterpretable and I am not going to interpret it. I report it because dropping the one
regime where my ideas lost would be exactly the selective reporting the audit criticised in F14.

## 8.4 Session-dependent attention temperature: measured, and too small to matter

I expected unnormalised attention logits to give a session-dependent effective temperature, since
`‖W_k h_i‖` varies with a session's activity scale. §7.1 kills it: LayerNorm pins the key norm to `√d`
exactly, with ~3% residual variation under a non-trivial affine. Normalised attention entropy in the
synthetic sat at 0.78–0.85 across all arms and did not separate held-in from held-out (e.g. 0.8430 vs
0.8476 for the baseline in the noisy regime). This was a real hypothesis and it is dead.

## 8.5 The premise I was handed is partly mis-stated, and one part of it favours us

The brief presents held-in `0.4731` against held-out `0.2749` as a weakness of *our* consumer. Both
numbers are from EvalAI submission `578689` and reproduce
(`H1_SPINT_BASELINE_REPRODUCTION.md:150-154`). But the matched SPINT reproduction shows
`0.4704 / 0.2615` — **a gap of 0.209 against our 0.198.** The gap is a property of the backbone and the
benchmark split, not of the carrier; the carrier slightly *reduces* it. Two further confounds are in the
same receipt and neither is mentioned anywhere: held-in is scored at **M=4** calibration trials and
held-out at **M=3**, so part of the 0.20 is a supervision difference rather than a session difference;
and held-out std is `0.1272` against held-in `0.0393`, so the held-out aggregate is 3.2× more dispersed
and one recording can move it. This does not remove the weakness — it relocates it, and it means a
source-training change to the *backbone* is the right target, which strengthens §3 rather than
weakening it.

## 8.6 Things I did not do

No GPU. No real-data run of any kind: every number here is either read from a repository receipt, from
vendored CEBRA source, or produced by a synthetic probe in `/tmp/redteam/`. I did not open any H1
held-out artifact, did not touch `sua_exploration/mc_maze/`, `SPINT-main/src/`, or the paper, and did
not re-derive F11–F14. The T4 coordinate-scale measurement in §4 is the check I most wanted to run and
did not, because it needs the sealed center-out loaders.

---

# §9 — Ranking by expected value

| # | Item | Touches target session? | Cost | Expected value |
|---|---|---|---|---|
| 1 | **T4 coordinate scales on center-out / subject-M** (§4) | no — a measurement | minutes CPU | **highest.** Settles S1 in both lanes, gates gauge augmentation, and was already written down as unrun |
| 2 | **Restate F2b and replace the parameter-growth row** (§2) | no | writing only | **high.** Removes a live reviewer objection, replaces it with a stronger claim |
| 3 | **Projection head on the auxiliary loss in screening** (§5) | no | one extra arm in a run already planned | **high.** Difference between a false negative and a result |
| 4 | **Gauge augmentation at source training** (§4) | no — source-only | 2 GPU runs, gated on #1 | medium-high, conditional on #1 |
| 5 | **NN velocity residual between real center-out sessions** (§3) | no — a measurement | ~1 hour CPU, behaviour arrays only | medium-high. Sole gate on #7 |
| 6 | **Supervise all W bins instead of the last** (§7.5) | no | one config flag + 2 GPU runs | medium. Cheapest GPU test on the list; idea pre-exists, CEBRA supplies the priority |
| 7 | **Size-matched pseudo-session mixing** (§3) | no — source-only | datamodule work + 2 GPU runs | medium. Largest conceptual upside; center-out/RT only; gated on #5 |
| 8 | **Off-event OLS rows instead of a spectral basis** (§6) | no | CPU screen in the existing estimator-audit shape | medium. Also discriminates C1's Idea 1 |
| 9 | Target-session anything | **yes — violates the constraint** | — | excluded by construction, as Grok 9 / S9 already recorded |

Items 1, 2, 3 and 5 consume no GPU at all and together settle whether 4, 6 and 7 are worth queueing.

---

# Reproduction

```
CUDA_VISIBLE_DEVICES= PYTHONNOUSERSITE=1 \
  PYTHONPATH=<repo>/cebra_exploration/third_party/cebra \
  /home/xinyuan/miniconda3/envs/spint/bin/python -s <script>
```

| Script | Produces |
|---|---|
| `/tmp/redteam/unified_solver_probe.py` | §2 — shared-model receipt, 13,155 vs 22,790 params, new-session rejection |
| `/tmp/redteam/pseudosession_probe.py` | §3, §4, §8.1–8.4 — arm table, session-id decodability, attention entropy |
| `/tmp/redteam/match_residual_probe.py` | §3 — nearest-neighbour residual vs behaviour dimension |

`pseudosession_probe.py` regimes, 8 seeds each:
`--t-cal 800 --noise-sd 0.6` (clean carrier) and `--t-cal 60 --noise-sd 1.5` (noisy carrier), both with
`--gauge-spread 0.5 --n-source 4 --n-heldout 4 --k-mix 3 --steps 1500`. Results in
`/tmp/redteam/R{1,2}_8seed.json` and the 3-seed sweep including the naive-union and behaviour-prior arms
in `/tmp/redteam/R{1,2,3}_*.json`.

**Everything in §3, §4 and §8.1–8.4 is synthetic.** It is a positive-control instrument for deciding
which ideas are worth real compute, in the same role as the project's own synthetic control in F8. No
number from it belongs in the paper.
