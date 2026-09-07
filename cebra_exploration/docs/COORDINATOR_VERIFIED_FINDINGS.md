# Coordinator-Verified Findings

Facts established by the coordinating agent by **running code**, not by reading it. Track outputs are
audited against this file, and **where a track document disagrees with this one, this one wins.**
Everything here is reproducible with the recipe in `scripts/cebra_env.sh`.

Findings keep their original discovery numbers so that cross-references from other documents stay
valid; they are grouped by theme below rather than by number.

## Index

| # | Finding | Consequence |
|---|---|---|
| **Environment** | | |
| F1 | Vendored CEBRA 0.6.1 runs in the existing `spint` env | `PYTHONNOUSERSITE=1` mandatory or CUDA is lost |
| F5 | pypi through the proxy is unreliable | use the Tsinghua mirror recipe |
| **CEBRA mechanics** | | |
| F2 | Multi-session fit works across mismatched neuron counts | CEBRA is a real competitor on our central axis |
| **F2b** | **`MultiSessionSolver` shares ZERO weights across sessions** | root cause of F8. **But over-generalized to "CEBRA" — see the F2b addendum: `UnifiedSolver` shares one model, and the paper's efficiency row was false as written** |
| F3 | `adapt=True` retrains only the first layer, trunk bit-identical | the structural analogue of our identity token |
| F4 | `adapt=True` is unsupported after multi-session training | forces a fairness choice; superseded in part by F2b |
| F7 | `consistency_score` takes 1-D labels only | blocks the H1 API path, not the idea |
| **Our datasets** | | |
| F6 | RT is `dandi_000688/sub-C`, not `000129/sub-Indy` | the brief was wrong; RT and subject-C are the same animal |
| F10 | RT discovery was unimplemented | must bind through the sealed loader |
| **Track B audit** | | |
| **F8** | **Primary arms could not land the target in the source latent space** | arms were void; rebuilt around a joint fit |
| F9 | `collapse_multisession_to_template` broke `transform` | deleted in round 2 |
| F15 | Linear decodability falls as CEBRA trains | right trend, **wrong term blamed** — see F16 |
| **F16** | **Fixed `NORMALIZED_LAMBDA_FIXED = 1.0` costs CEBRA ~0.19 R²** | the third bias in our favour; found by audit, not by me |
| **F17** | **A contrastive auxiliary loss must sit behind a projection head, not on `ŷ`** | binding on the consistency screen's next step; also lists three CEBRA elements confirmed dead |
| **F18** | **Behaviour-matched cross-session pairing costs only 1.01–1.13× the within-session floor, in every cohort** | the prerequisite for the whole behaviour-matched family **passes**. Refutes the red team's `−1/d` redirect; H1 is undecidable on sampling, not excluded on drift; discrete direction keying is 2.8× *worse* than continuous |
| **F19** | **Intercept dominance is an H1 normalizer choice, not a carrier property** — center-out is 9.53× raw / 0.884× standardized against H1's 32–61× | **the one actionable fix produced by this sub-project**: H1 uses a single global scalar, which cannot rebalance coordinates; center-out uses per-coordinate z-scoring, which does |
| **Track C audit / the H1 sparse failure** | | |
| F11 | The failing sparse arm carries an intercept coordinate the working dense arm lacks | confirmed in code, not inferred |
| ~~F12~~ | **SUPERSEDED** — ρ = 0.808 was V2 all along (recompute gives 0.8077; V1 gives 0.7527) | the evidence was never void, but the abstention **gate** fails for three better reasons: split-half cosine ≈ 0.06 with 4/13 negative, the predicted quantity never goes negative, and it is partition- and budget-dependent |
| F13 | H-SE5 checkpoints exist; measured weights show the network **did** compensate | S1 weakens; date difference is in signed-tuning content |
| F14 | The date-2 delta is sample-weighted; equal-weighted it is −0.0048 | aggregation sensitivity is real. **The query-horizon hypothesis built on it was refuted by audit** |

**The three that matter most:** F8 (a comparator that would have produced a false conclusion),
the **F2b addendum** (a claim of mine that reached the paper and was false as written), and F16 (a
bias in our own favour that I missed entirely).

**Two independent reviews have corrected this file.** `AUDIT_OF_COORDINATOR.md` overturned the F14
query-horizon hypothesis, the "7–12×" interval and the readiness claim, and found F16.
`REDTEAM_CEBRA_MECHANISTIC_VALUE.md` overturned the blanket F2b claim and the verdict that CEBRA
offers no mechanistic value. **My two worst errors share one cause: generalizing beyond what I
measured** — once from a hyperparameter inherited across datasets, once from a single code path.
Treat unqualified claims in this file with that failure mode in mind.

**An independent Opus audit of this file is at `AUDIT_OF_COORDINATOR.md`.** It reproduced every
quantity it checked, most to the last printed digit, but overturned three inferences: the F14
query-horizon hypothesis, the withdrawn "7–12×" interval in F13b, and the readiness claim in
`PROJECT_VISION.md`. It also found F16. Corrections are folded in below and attributed at each site.
**Where this file and the audit disagree, prefer the audit** — it verified more configurations than I
did on every claim it examined.

---

# Environment

## F1 — Vendored CEBRA runs in the existing `spint` env

**Verified 2026-08-13.** CEBRA 0.6.1 from `third_party/cebra/`, `torch 2.5.1.post303`,
`numpy 2.0.1`, CUDA available on 2× RTX 3090.

`PYTHONNOUSERSITE=1` is mandatory: a user-site `torch 2.12.0+cu130` otherwise shadows the env and
reports `cuda False`, because it is too new for driver 535.309.01.

Only dependency added to `spint` was `literate-dataclasses` 0.0.6, installed `--no-deps`. Nothing
else changed.

## F5 — Network is unreliable through the proxy

`pypi.org` via clash on `127.0.0.1:7890` gives intermittent
`SSL: UNEXPECTED_EOF_WHILE_READING`. Both a `pip install cebra` and a `pip install
literate-dataclasses` failed this way. The working recipe strips the proxy and uses the Tsinghua
mirror — see `cebra_pip_install` in `scripts/cebra_env.sh`. `github.com` cloning through the proxy
worked first time.

Relevant to Track A: a large dataset download may be unreliable, and download feasibility is part of
that track's verdict rather than an afterthought.

---

# CEBRA mechanics

## F2 — Multi-session fitting across mismatched neuron counts works

A single `CEBRA` estimator fits `[X_a (400×20), X_b (400×31)]` jointly and transforms each into a
shared 3-D space via `session_id`. This is the property that makes CEBRA a real competitor on our
central axis rather than a distant classical baseline.

## F2b — **`MultiSessionSolver` shares ZERO weights across sessions** (but see the F2b addendum: this is not all of CEBRA)

> **CORRECTED AFTER RED TEAM, and this correction reaches the paper.** Everything below is accurate
> for `MultiSessionSolver`, which is the only path the sklearn estimator can reach and therefore the
> only path any of my measurements went through. **It is not true of CEBRA as a whole.** CEBRA 0.6.1
> also ships `cebra.solver.UnifiedSolver`, where **one model is shared across all sessions**. I
> generalized from a single code path to "CEBRA" without surveying the solver space. See the addendum
> at the end of this finding before quoting the efficiency contrast anywhere.

The first version of the background brief said multi-session CEBRA "fits session-specific input
layers into one shared latent space". **That is wrong about the mechanism**, and Track C1 was right
to question it. Corrected here and in the brief.

`cebra/integrations/sklearn/cebra.py:792-800` builds
`nn.ModuleList([cebra.models.init(...) for dataset in dataset.iter_sessions()])` — one **complete,
independent encoder per session**. Not a shared trunk with per-session input layers.

Measured after a real two-session fit (24 and 31 units, `offset10-model`, `output_dimension=3`):

| | |
|---|---|
| model type | `ModuleList`, 2 submodels |
| session 0 / session 1 params | 11,171 / 11,619 |
| tensors **identical** between the two encoders | **none — zero** |
| tensors differing beyond the first layer | all of them (`net.2`–`net.5`, weights and biases) |
| total params | 22,790, i.e. **O(N_sessions × full encoder)** |

So cross-session alignment in CEBRA comes **entirely from the contrastive loss** pulling independent
encoders' outputs into a common latent space. There is no weight sharing to inherit.

**This sharpens the paper's contrast rather than weakening it.**

| | CEBRA multi-session | Ours |
|---|---|---|
| weights shared across sessions | **none** | the entire decoder |
| what varies per session | a whole encoder | one cached token `E_i` |
| cost of a new session | train a full new encoder | closed-form solve + one forward pass |
| parameter growth | linear in session count *(`MultiSessionSolver` only — **not** `UnifiedSolver`, see addendum)* | **zero** |
| what aligns sessions | the InfoNCE objective | the identity token's content |

The honest framing is not "learned versus closed-form input layer" but "a full encoder per session,
aligned by a loss, versus one shared decoder with a small cached descriptor". That is a stronger
statement of our efficiency claim than the paper currently makes.

### F2b addendum — `UnifiedSolver` shares one model, and the efficiency row must be rewritten

Found by the red team, verified independently here.

```
cebra.solver.UnifiedSolver._get_model(session_id) -> return self.model      # one model, any session
cebra.data.UnifiedDataset(ds_24, ds_31).input_dimension == 55 == 24 + 31    # width is the SUM
```

| Solver | weights across sessions | parameters (2 sessions, 24+31 units) |
|---|---|---:|
| `MultiSessionSolver` | **none shared** | 22,790 (11,171 + 11,619) |
| `UnifiedSolver` | **one shared model** | **13,155** |

It was invisible to me because it is unreachable from the sklearn estimator that every measurement
used, and the string `Unified` appears nowhere in this project's documents.

**Why this must be fixed in the paper rather than quietly dropped.** The contrast table in
`PROJECT_VISION.md` and `README.md` claims "parameter growth per session: linear (CEBRA) versus zero
(ours)". Against `UnifiedSolver` that row is **false as written** — its parameter count does not grow
with session count.

**The correct claim is different, and stronger.** `UnifiedDataset` builds a "pseudo-animal" by
concatenating the units of behaviourally matched samples from every session, so the unified model's
input width is hard-wired to the **sum of the training sessions' neuron counts**. Consequences:

- it **cannot embed an unseen session at all**, at any cost — the input width is fixed at training
- even embedding a *training* session requires streaming every other source session's data alongside it

So the honest axis is not parameter growth but **whether a new session can be served at all**.
`MultiSessionSolver` needs a new encoder trained for it; `UnifiedSolver` cannot accept it in any
form; we need a closed-form solve and one forward pass. State it that way, and the claim survives
contact with both solvers instead of being refutable by the one I did not look at.

**Methodological lesson, and it is the same one as F16.** Both of my worst errors came from
generalizing beyond what I measured — there from a hyperparameter inherited across datasets, here
from one of several code paths. Measurement discipline was fine; scope discipline on the *claim* was
not.

**Consequence for Track B, and it invalidates part of F4 below.** F4 option (b) proposed freezing a
shared trunk and growing a new session-specific input layer. **There is no trunk to freeze.** The
real options are:

- **(b1)** initialize the target encoder from a source session's encoder (or an average across them)
  and fine-tune. Reasonable, but it is a design choice *we* invent, not published CEBRA, so it must
  be declared and justified rather than presented as "CEBRA".
- **(b2)** modify the vendored source to tie trunk weights across source sessions, creating a trunk
  that a new session can attach to. Larger modification, and it deviates from published CEBRA in a
  way a reviewer could challenge.
- **(b3)** single-session source plus the supported `adapt=True` path, accepted explicitly as a
  weaker-source lower bound.

Whichever Track B chooses, the choice must be stated in the protocol with its bias direction named.

## F3 — `adapt=True` retrains **only the first layer**, and this is the direct analogue of our identity token

Source: `third_party/cebra/cebra/integrations/sklearn/cebra.py:993-1012`. The state dict is split at
index 2; `pretrained_params[:2]` (first-layer weight and bias) are re-initialized to the new input
width, everything else is loaded from the pretrained model and set `requires_grad = False`.

Measured on a toy config (24-unit source → 37-unit target, `offset10-model`, `output_dimension=3`):

| | |
|---|---|
| tensors changed | `net.0.weight`, `net.0.bias` only (2 of 10) |
| parameters updated | 2,400 / 12,003 = **20.0%** |
| target-session backward pass | **required**, `max_adapt_iterations` (default 500) |

So CEBRA absorbs a changed unit set through a **learned session-specific input layer**, where we
absorb it through a **closed-form identity token**. Same problem, same structural position in the
network, opposite optimization commitment. That is the comparison the paper wants.

## F4 — **`adapt=True` is NOT supported after multi-session training.** This is the fairness problem

```
cebra/integrations/sklearn/cebra.py:981-984
if is_multisession or isinstance(self.model_, nn.ModuleList):
    raise NotImplementedError(
        "The adapt option with a multisession training is not handled. "
        "Please use adapt=True for single-trained estimators only.")
```

Empirically confirmed: multi-session `fit([X_a, X_b], ...)` followed by `fit(X_t, adapt=True)`
raises `NotImplementedError`.

**Why this matters, and it is the single most consequential fact in this file.**

Our decoder is **multi-session source-trained** — a cross-session prior learned on source data is
part of our method, not an incidental detail. Upstream CEBRA only supports adapting from a
**single-session** source model.

That leaves Track B a fork, and the choice must be made deliberately and documented:

| Option | Consequence |
|---|---|
| **(a) single-session source + `adapt=True`** | Supported upstream, zero code risk. But CEBRA gets **no multi-session source prior** while we do. The comparison is **biased in our favour** and a reviewer who knows CEBRA will say so. Not acceptable as the only arm. |
| **(b) extend the vendored source** so a multi-session-trained model can grow a new session-specific input layer with the trunk frozen | Matched source information. Requires modifying `third_party/cebra/` — which is **exactly why the source was vendored rather than pip-installed**. Must be recorded in `third_party/CEBRA_PROVENANCE.txt` under `local_modifications`. **See F2b: option (b) as worded here is not directly available, because multi-session CEBRA has no shared trunk. Use the b1/b2/b3 breakdown in F2b instead.** |

**(b) is the scientifically correct arm.** (a) is acceptable only as an additional lower bound, and
if reported it must be labelled as a single-session-source handicap rather than presented as CEBRA's
best. Reporting (a) alone would be the kind of under-tuned reimplementation that
`HANDOFF_COMPARATORS_20260812.md` §10 explicitly warns against.

Note also that (b) is the *closer* analogue of us in a second respect: freezing the trunk and
learning only the per-session input layer is precisely the "everything frozen except the identity
path" structure of our own method, with gradient descent substituted for a closed-form solve.

---

# Our datasets

## F6 — RT is `sub-C` inside `dandi_000688`, not `data/000129/sub-Indy`

Track A flagged a discrepancy in the first version of the background brief. It was right; the brief
was wrong and has been corrected.

Verified by file count in `sua_exploration/data/dandi_000688/`:

| Subject | total NWB | `ses-CO` | `ses-RT` |
|---|---:|---:|---:|
| `sub-C` | 68 | 53 | **15** |
| `sub-M` | 22 | 22 | 0 |
| `sub-J` | 3 | 3 | 0 |

The sealed loader `sua_exploration/mc_maze/rt_classical_comparators.py:117-118` hard-rejects any
file not named `sub-C_ses-RT-*`, and `EXPECTED_FOLDS = 15` at line 42. So the paper's 15 RT
outer-LOSO folds are the 15 `sub-C` random-target sessions.

`sua_exploration/data/000129/sub-Indy` is NLB MC_RTT — a single train NWB and a single test NWB. It
is **present in the repo but not used by the paper**. Same for `000128/sub-Jenkins` (NLB MC_Maze).

**Two consequences.**

1. **Any Track B RT adapter bound to `000129/sub-Indy` is wrong** and must be re-pointed at
   `dandi_000688/sub-C/sub-C_ses-RT-*.nwb` through the sealed loader. Check this first when auditing
   Track B, since Track B was dispatched with the uncorrected brief.
2. **RT and subject-C are the same animal.** The paper's center-out development domain (subject-C,
   six sessions) and its random-target task (15 sessions) share a subject. subject-M is the genuinely
   external cohort. This matters for any claim about task-geometry transfer being independent
   evidence — it is cross-*task* within an animal, not cross-animal. Worth checking that the paper
   does not overstate it.

## F7 — CEBRA's consistency metric is 1-D-label-only, which blocks the H1 API path but not the idea

Track C1 flagged this as unverified. Confirmed by reading the source.

`cebra/integrations/sklearn/metrics.py:350-352`, inside `_consistency_datasets`. **Corrected after
audit:** I originally quoted this as `ValueError`. Executed, the call raises **`NotImplementedError`**:

```
consistency_score(embeddings=E, labels=<7-DoF>, between='datasets')
-> NotImplementedError: Invalid label dimensions, expect 1D labels only, got 2D.
```

Note the message reports `ndim` (2 for an `(n, 7)` array), not the label dimensionality.

The labels are not decorative — line 365 uses them in `align_embeddings(..., n_bins=100)` to put
samples from different sessions into correspondence by binning a shared behavioural coordinate.
Cross-session consistency is meaningless without such a correspondence.

| Mode | Labels needed | Works on H1 7-DoF? |
|---|---|---|
| `between="datasets"` | 1-D only, used for binning/alignment | **No**, not through the API |
| `between="runs"` | none, but requires embeddings of the *same* samples | Not cross-session, so not our question |

**This is an API limitation, not a conceptual one, and the distinction matters.** The metric is
"align samples by a shared behavioural coordinate, then take the R² of a linear map between the two
embeddings". Nothing about that requires a 1-D label — it requires *a* correspondence. For H1 we can
reimplement it directly over the 7-DoF covariate, or bin a derived scalar (speed, or the dominant
behavioural PC), without touching CEBRA's API at all.

So any Track C idea that uses consistency as a **drift diagnostic on H1** is still live, but it must
budget for reimplementing the metric and must declare the correspondence rule it bins on. An idea
that assumes `consistency_score` can be called directly on H1 is wrong as written.

---

# Track B audit — the CEBRA comparator

## F8 — **Track B's primary arms are structurally broken: `adapt=True` cannot land the target in the source latent space**

**This is the most important finding in this file. The comparator must not be run as designed.**

Measured on a synthetic **positive control** — three "sessions" of 24, 31 and 37 units, each a
different random linear mix of the *identical* 2-D latent, i.e. the easiest transfer problem that
can be constructed. A ridge readout was fitted on source embeddings only and applied to the target.

**Read the addendum below before quoting any number in this table.** These are single-seed values.
The audit's 8-seed mean for the `adapt` row is **+0.0098**, with two seeds above +0.5; −1.2278 is an
extreme draw from a stable but strongly configuration-dependent distribution. The *direction* of the
finding is robust and reproduced; **this specific magnitude is not representative.**

| Path | source R² | **target R²** |
|---|---:|---:|
| multi-session source → collapse → `adapt=True` (template = source session 0) | 0.9891 | **−1.2278** *(extreme draw)* |
| same, template = source session 1 | 0.9891 | **−0.9265** |
| **target included in the joint multi-session fit** | 0.9924 | **+0.9940** |

**Root cause.** CEBRA's cross-session alignment comes entirely from **cross-session positive
sampling during joint multi-session training** (consistent with F2b: there are no shared weights to
carry alignment, so the loss is the only mechanism). `adapt=True` re-fits the target's first layer
against a **single-session** contrastive objective, with no term referencing the source sessions.
InfoNCE on the hypersphere is invariant to rotation of the embedding, so the adapted session lands
in *a* good latent space of its own with **no reason to share the source's orientation**. Negative
R² is the expected outcome, not a tuning failure.

**Why this matters more than a normal bug.** This is precisely the failure that voided the FA
alignment comparator — a source readout trained across mutually inconsistent latent spaces, giving
R² ≈ 0. It is also exactly the trap `HANDOFF_COMPARATORS_20260812.md` §10 warns about: *"a badly
tuned reimplementation is worse than no comparison, because a reviewer who knows the method will see
it."* Had the arms been run as written, every dataset would have returned ~0 or negative R² and we
would have concluded "CEBRA fails on our data". That conclusion would have been an artefact of our
own harness and would not have survived review.

**Fix options, in order of preference:**

1. **Joint fit including the target session** — verified to work (+0.9940). This is how CEBRA is
   actually used for multi-session data, so it is the defensible arm. Note the cost consequence, and
   note it favours us **honestly**: adaptation now requires retraining the *entire* multi-session
   model, not 20% of one encoder. The cost-of-backprop table gets much larger, and it is real.
2. **Freeze the source encoders, train only the target encoder, but keep cross-session positive
   sampling.** The minimal-cost adaptation that preserves alignment, and the closest structural
   analogue of our own "everything frozen except the identity path". Requires modifying the vendored
   source — which is why it was vendored. This is likely the **fairest** arm and should be built.
3. Post-hoc alignment of the adapted embedding onto the source space. Rejected as primary: it needs
   a correspondence, so it either consumes target labels (no longer unsupervised) or becomes a
   distribution-alignment method, i.e. it silently turns into the FA/NoMAD comparator we already have.

Whatever is chosen, the **positive control above must become a required test**: any arm that cannot
recover a shared synthetic latent is broken and its numbers are void.

### F8 addendum — independent audit reproduces the verdict but **corrects the mechanism**

An independent Opus auditor swept 13 configurations across architectures (`offset1`, `offset10`),
source iterations (250–5000) and adapt iterations (250–5000), measuring three quantities where I had
measured one. **The sweep was run twice and replicates**: the source-readout target R² agrees to the
third decimal in most cells (+0.028/+0.033, −0.497/−0.502, −0.240/−0.241, −1.035/−1.005,
+0.591/+0.582). The spread from +0.59 to −1.04 is therefore **across configurations, not run-to-run
noise** — the configuration dependence is a stable property, which is what makes point 3 below a
correction rather than an artefact.

**The verdict holds.** `adapt` target R² under the **source** readout is negative in 9 of 13 configs
(−0.095, −0.154, −0.220, −0.240, −0.367, −0.497, −0.576, −0.624, −1.035), while the joint fit gives
target ≈ source in every config (~0.59–0.82). Arms built on `adapt=True` would indeed have produced
invalid numbers.

**But three of my characterizations were too strong, and the audit is right:**

1. **The adapted embedding is not "a good space of its own with no relation to the source" — the
   relation is often nearly linear.** `adapt_linear_map_to_source_r2` reaches **0.976, 0.982, 0.988,
   0.990** at 250–500 adapt iterations. The target embedding is a good embedding in its own right too
   (target-readout R² 0.64–0.82 throughout). What fails is *only* the direct application of the
   source readout.
2. **The failure is progressive, not immediate.** Linear-map R² decays with adapt iterations
   (0.976 → 0.781 → 0.698 at 250/1000/2000), i.e. the target frame drifts steadily away from the
   source frame as adaptation proceeds.
3. **It is not deterministic.** At 250 adapt iterations the source-readout R² is **+0.028** for
   template 0 and **+0.591** for template 1. My single-seed −1.2278 was a real observation but an
   unrepresentative magnitude, and I generalized from it too confidently.

`adapt_emb_norm_mean ≈ 1.0` in all 13 configs, confirming the embeddings sit on the unit hypersphere
and supporting rotation-invariance as the operative mechanism.

**Consequence for the fix ranking above.** I ranked post-hoc linear alignment third and dismissed it.
Given that the residual relationship to the source frame is up to R² 0.99 and approximately linear, a
single linear alignment step would in fact recover most of the loss. That does not make it the right
*primary* arm — it still needs a correspondence, so it either consumes target labels or turns into
the FA/NoMAD distribution-alignment comparator we already have — but "it needs correspondence" is the
honest objection, not "it would not work". Option 3 should be re-stated on those grounds.

## F9 — `collapse_multisession_to_template` leaves the estimator unusable for `transform`

`src/cebra_comparator.py:398-428` replaces `estimator.model_` with one session's encoder and sets
`num_sessions_ = None`, but **does not replace `solver_`**, which remains a `MultiSessionSolver`.
The object is left internally inconsistent:

```
after collapse: solver class = MultiSessionSolver | model class = Offset0Model
  transform()             -> RuntimeError: No session_id provided: multisession model requires a session_id
  transform(session_id=0) -> TypeError: '>=' not supported between instances of 'int' and 'NoneType'
```

So a collapsed estimator **cannot produce a source embedding at all**, which is what the
source-fitted readout requires. Track B's 19 tests pass but none exercise `transform` after collapse
— a coverage gap on the module's most important path.

Workaround used in the audit: take source embeddings from the **original** multi-session estimator
with an explicit `session_id`, before collapsing. Note the solver *does* become a
`SingleSessionSolver` after `adapt=True` runs, so the inconsistency is confined to the window
between collapse and adapt.

**Confirmed working in the same test, and worth keeping:** after adapt, the source trunk is
genuinely preserved — `net.3/5/7` weights and biases are bit-identical to the source encoder and only
`net.1` (the first layer) changes. The freeze-and-adapt mechanism itself behaves exactly as intended.
The defect is in alignment (F8) and in object state (here), not in the layer-freezing logic.

## F10 — RT discovery is unimplemented, and must not be wired to the path in the stale brief

Track B was dispatched with the uncorrected brief (see F6) but did **not** hardcode the wrong RT
path, because it implemented no RT file discovery at all — Part A for RT rests on structural priors
carried over from the FA alignment audit rather than on opened files. No bug, but no adapter either.

When RT discovery is implemented it must bind to
`sua_exploration/data/dandi_000688/sub-C/sub-C_ses-RT-*.nwb` through
`sua_exploration/mc_maze/rt_classical_comparators.py`, **not** `data/000129/sub-Indy`.

## F16 — **The third bias in our favour, which I missed: a fixed ridge penalty costs CEBRA ~0.19 R²**

Found by the independent audit (`AUDIT_OF_COORDINATOR.md`), not by me. I had claimed to have caught
the two biases in our favour (F8 and F15). There was a third, and it is larger than the one F15
diagnosed.

`src/cebra_comparator.py` sets `NORMALIZED_LAMBDA_FIXED = 1.0` for the downstream ridge readout. That
value is calibrated for regressing on **57–176 channels of firing rates**. It is being applied to
**three unit-norm embedding coordinates** — CEBRA's embeddings lie on the unit hypersphere, which we
confirmed independently (`adapt_emb_norm_mean ≈ 1.0` in all 13 configs of the F8 addendum sweep). A
penalty sized for raw rate features is enormous relative to unit-norm inputs.

Measured on **our own positive control**: target R² rises from **0.8091 at λ=1 to 0.9979 at λ=1e-2**,
with kNN at 0.9997.

**Why this is worse than it sounds.** The 0.19 R² penalty is **2–3× larger than the linear-versus-kNN
gap that F15 diagnosed**. F15 identified a real trend but attributed it to the decoder *family* and
prescribed changing it, while never examining the penalty term declared three lines above
`KNN_NEIGHBORS`. Reporting rule 5 is directly on point, and rule 1 explicitly permits selecting λ
inside the calibration block at no methodological cost — so this is a free fix that we simply did not
make.

**Lesson for the audit trail.** Both biases I did catch were in *arm construction*. The one I missed
was in a **hyperparameter constant**. Constants inherited from another dataset's conventions deserve
the same scrutiny as the arms themselves.

## F17 — A contrastive auxiliary loss must sit behind a **projection head**, not on `ŷ` directly

Raised by the red team; binding on the cross-session consistency work now in screening.

The proposed source-training change is `L = L_MSE + λ·L_consist`, pulling together the pooled
representations `ŷ` of behaviour-matched samples from different source sessions. **As specified, the
contrastive term is applied directly to the representation the linear read-out consumes.**

That is the wrong attachment point, and we already own the evidence:

- F15 measured that InfoNCE training **reduces linear decodability** while the embedding improves by
  other measures — monotone across 8/8 seed × sample cells at two penalty settings.
- Our read-out `v̂_c = W_ro ŷ_c + b_ro` is **linear**.

So a contrastive term placed on `ŷ` optimizes a property in tension with the read-out that consumes
it. A negative result from that arm would be uninterpretable: it could not distinguish "cross-session
consistency does not help" from "we attached the term where it does most harm". Standard practice in
contrastive learning is to apply the loss to a **projection of** the representation and discard the
projection at inference, precisely so the backbone is not forced to satisfy both objectives.

**Required design change before any consistency arm is run:**

1. Apply `L_consist` to `p(ŷ)` for a small projection head `p`, trained jointly and **discarded at
   deployment** — so the online path stays byte-identical and the no-backprop boundary is untouched.
2. Log **held-in linear R² against the auxiliary weight λ**. This is the diagnostic that separates the
   two failure modes above.
3. Keep `λ = 0` exactly recovering the current model, asserted in a test.

**This does not invalidate the screen currently running.** That screen measures whether `ŷ` is
inconsistent across sessions on existing checkpoints — a diagnostic, not an implementation. Its
verdict remains meaningful: if `ŷ` is already consistent, no loss at any attachment point has
anything to fix. But a **PROCEED** verdict must be read as "proceed to a projection-head
implementation", never as "add a contrastive term to `ŷ`".

### Three CEBRA elements confirmed **dead**, so nobody re-proposes them

Also from the red team, each measured rather than argued:

| Element | Why it is dead |
|---|---|
| unit-hypersphere normalization | **already present** at the point similarity is computed — `CrossAttentionLayer` applies the *same* `LayerNorm` to query and keys; `‖LN(h)‖ = √d` held exactly across a 300× input-scale sweep. The one place we lack it, `ŷ`, is where it would hurt a linear velocity read-out |
| learnable temperature | a **no-op up to reparameterization** — our logits are unnormalized dot products whose `W_q W_k` gain product already is a learned inverse temperature |
| negatives / uniformity pressure | **points the wrong way** — it trades the linear decodability our read-out depends on for kNN performance we do not use |

## F18 — Behaviour-matched cross-session pairing is **cheap everywhere**, and the red team's `−1/d` redirect was wrong on real data

Measured on our own cohorts (`sua_exploration/docs/BEHAVIOUR_MATCHING_FEASIBILITY_20260814.md`,
receipt verified). Behaviour arrays only — no neural data, no model. The quantity is the cross-session
nearest-neighbour residual **as a ratio to the within-session floor**, paired on the same query set
with equal-sized reference sets.

| Cohort / representation | d | participation ratio | **ratio median** | verdict |
|---|---:|---:|---:|---|
| centre-out sub-C / vel-2D | 2 | 1.85 | **1.101** | TIGHT |
| centre-out sub-C / 50-bin window | 100 | 8.46 | **1.113** | TIGHT |
| centre-out sub-M / vel-2D | 2 | 1.68 | **1.069** | TIGHT |
| centre-out sub-M / window | 100 | 7.41 | **1.067** | TIGHT |
| RT sub-C / vel-2D | 2 | 1.81 | 1.125 | marginal on tail |
| RT sub-C / window | 100 | 10.91 | **1.029** | marginal on tail |
| **H1 / 7-DoF endpoint** | 7 | 6.10 | **1.009 — the tightest of all four** | marginal on tail |

**Cross-session drift is not the obstacle.** Matching costs 1.01–1.13× the within-session floor in
every cohort. Whatever limits behaviour-matched pairing, it is not that sessions have drifted apart.

**This refutes the redirect I recorded from the red team.** Its synthetic estimate — "2.6% of
behaviour scale at 2-D but 37% at 7 DoF, so the mechanism is feasible on center-out and RT and *not*
on H1" — does not survive real data. Two errors:

1. **The exponent is set by the participation ratio, not the nominal dimension.** The 50-bin window is
   nominally 100-dimensional but measures `d_eff` 6.4–15.2. A one-second planar behaviour window is a
   6–11 dimensional object.
2. **H1 is not the bad case.** Its median ratio is the *lowest* of all four cohorts. It is
   `MARGINAL` only on the p90 tail, and only because each session yields 19 reference and 20 query
   points. **H1 is undecidable through sampling, not excluded through drift** — a different verdict
   with different remedies.

The `−1/d` scaling itself **holds** (`R²` 0.962–0.999, `d_eff` within 1.5× of the participation ratio
in all six testable cases), but it governs **single-session sampling density**, not cross-session
distance. Inverting the fits, a 0.10-sd pairing tolerance on the window representation needs about
120 (sub-M), 317 (RT) and 7,400 (sub-C) reference samples — all available. 0.05 sd is out of reach
everywhere.

**Discrete direction matching is worse than continuous — my own prompt assumed the opposite.** I
briefed the agent that center-out's ~8 discrete directions make matching "trivially easy in
principle" and that this "is the one the proposed mechanisms would actually use". Wrong: keying on
direction divides the reference pool by 8 (or 40 with a phase key), and the sampling-density penalty
exceeds what the shared label buys. sub-C 2-D residual degrades 0.0051 → 0.0144, i.e. **2.8× worse**.
Two sub-M sessions are also direction-deficient — `20140627` holds only 5 of 8 directions among
usable trials — so a direction-keyed builder would silently drop up to 60% of samples there.

**Honest caveat carried from the agent.** Both tasks' sample sets are dominated by near-stationary
moments (RT median sample speed 0.22), so the *absolute* residuals are optimistic; on the fastest
quartile they are 1.6–138× larger. The **ratio** — the quantity the verdict rests on — stays at
1.00–1.21 under that stratification, so the conclusion survives but the absolute levels should not be
quoted without the caveat.

## F19 — Intercept dominance is an **H1 normalizer choice**, not a property of the carrier. This is the one actionable fix

`sua_exploration/docs/T4_COORDINATE_SCALE_AUDIT_20260814.md`, 48 center-out sessions measured with
the sealed estimator, 41 tests including bit-identical agreement with
`compute_unit_side_features_uncached`. Mechanism verified independently here.

**Center-out does not reproduce H1's imbalance.** Intercept-versus-signed-tuning ratio: **raw median
9.53×** (range 4.48–13.38) against H1's 32–61×, and **0.884× after standardization**. No session out
of 48 reaches even half of H1's lower bound, and at the injection MLP `b̂` is the *smallest*
coordinate, not the largest.

**The mechanism is the normalizer family, and both halves are confirmed:**

| | normalizer | can it rescale coordinates *relative to each other*? |
|---|---|---|
| **H1 (fails)** | `s_src = sqrt(mean(source_cache²))` = 1.0702 — **one global scalar** | **No.** A scalar divides every coordinate equally, so a 32–61× imbalance survives intact to `ψ_c` |
| **center-out (works)** | `fit_side_feature_stats` → per-column mean `[0.046, 0.454, 1.343, 10.151]`, std `[1.126, 1.285, 1.235, 9.115]` — **per-coordinate z-score**, fit on the 27 train sessions only, never refit | **Yes.** Removes the imbalance by construction |

**The actionable consequence.** Replacing H1's global scalar with **source-fitted per-coordinate
standardization** is a like-for-like change: both are fitted on source data only, so it does not touch
the no-backprop boundary and leaks nothing from the target session. It is exactly what the S1
proposals were groping toward but could not articulate, because they framed the problem as
"rebalance the carrier vector" rather than "H1 uses a normalizer family that structurally cannot
rebalance anything".

It requires retraining the H1 consumer, so it is a GPU experiment — but a single well-motivated
change with a sharp prediction, and center-out is the existence proof that the alternative behaves.

**The gauge reading: mechanism confirmed, magnitude below the predeclared bar.** `b̂` *is* the session
rate gauge, near-definitionally — `mean(b̂) / session mean rate = 1.0025 ± 0.0154` across all 48
sessions, Spearman ρ = 0.928. But mean rate varies only 2.53× across these sessions, moving `b̂`'s RMS
by 2.0× against the 3.0× predeclared, so the rule did not fire.

**A methodological catch worth keeping.** The agent noted its own predeclared statistic was the wrong
one: it used max/min of RMS, a **dispersion** measure, while a rate gauge acts on **location**.
sub-M's mean rate is 1.59× below sub-C's, so sub-M's standardized `b̂` arrives at `ψ_c` centred
**0.443 units below** sub-C's with its spread almost unchanged — invisible to a ratio of RMS. It
flagged this as post-hoc and **did not retroactively fire the reading**, which is the correct
handling. A follow-up should predeclare a location statistic.

**Two corrections to earlier descriptions.** The coordinates are in **Hz**, not spikes-per-bin —
`_pool_trial_rate_matrix` divides by trial duration — so absolute magnitudes elsewhere in this file
are 50× off, though no ratio is affected. And the large `â`/`m` cross-session ratios (6.35×, 4.06×)
are an **estimator artifact**: the two sessions driving them, `sub-M_ses-CO-20140626` and `20140627`,
are the only two with 6 and 5 of 8 target directions in the calibration prefix. Dropping one takes
`â` from 6.35× to 4.49×. Note these are the same two direction-deficient sessions independently
flagged in F18.

---

# Track C audit — the H1 sparse failure

The three F-items below were produced while auditing the two brainstorms and their cross-reviews.
Together they rank the competing explanations of why the H1 sparse carrier helps on one date and
hurts on another. This is the most transferable output of the whole sub-project, and it is about
**our** method rather than about CEBRA.

## F11 — Track C2's load-bearing inference is **confirmed in code**: the failing sparse arm carries an intercept coordinate the working dense arm does not

C2 ranked "rebalance the carrier vector before injection" first, resting on a claim it flagged as
*inferred, not read, and load-bearing*: that the dense H-C arm (which works) is 4-wide with no
intercept coordinate, while the sparse H-SE5 arm (which fails) carries one that dominates the token.

**Verified. It is not an inference.**

| | file | value |
|---|---|---|
| H-C carrier width | `SPINT-main/configs/model/falcon_h1_carrierid.yaml:20` | `carrier_dim: 4` |
| H-SE5 carrier width | `SPINT-main/configs/model/falcon_h1_sparse_event_endpoint.yaml` | `carrier_dim: 5` |
| what the 5th coordinate is | `sua_exploration/mc_maze/h1_sparse_event_endpoint_v2.py:1` docstring | *"V2 low-rank H1 endpoint carrier: **q=4 plus intercept**, ridge lambda=3"* |
| where it is appended | same file, line 125 | `np.column_stack((coefficient[1:].T, coefficient[0]))` — 4 slopes, then the intercept last |

So the one coordinate the working dense arm does **not** have is exactly the coordinate C2 measured
arriving at the injection MLP with 32–61× the RMS of the four signed-tuning coordinates.

**The injection path makes the mechanism plausible**, `SPINT-main/src/models/components/h1_sparse_event_spint.py`:

```
:42  carrier_pre_pool  = Sequential(Linear(1024, 32), ReLU())
:65  pooled            = carrier_pre_pool(temporal).mean(dim=1)          # 32-d activity summary
:67  carrier_post_pool(cat((pooled, effective), dim=-1))                 # 32 + 5 = 37 in
:49  carrier_post_pool[0].weight[:, 32:].zero_()                         # carrier columns zero-init
```

The five carrier coordinates are concatenated **raw**. There is no per-coordinate normalization
anywhere in the consumer, and the columns that touch them start at exactly zero.

**On C2's own Adam objection — it does not obviously defeat the idea.** Adam adapts the per-parameter
step size, but the forward contribution is still `w_j · x_j`: a coordinate arriving 60× smaller needs
a 60× larger weight to contribute equally, starting from zero, at `lr: 5.0e-5`, with
`weight_decay: 0.0` (both from the SE5 model config) so nothing pushes back. Whether Adam closes that
gap within the training budget is genuinely empirical — which is precisely what C2's proposed
diagnostic measures. The objection is well-posed, not fatal.

**Assessment.** This is the best-grounded idea produced by either brainstorm: it has a verified
structural asymmetry between the working and failing arms, a measured magnitude, a plausible
mechanism, and a cheap CPU diagnostic. It should be treated as the leading hypothesis for the H1
sparse failure until tested. Two caveats to carry: C2's RMS figures were measured on its own
recomputed carriers (provenance-checked against the sealed `source_audit_v2r2.json` in 13/13
sessions, which is good practice), and the whole chain is estimator-level while the paper's claim is
decoder-level R².

## F12 — **SUPERSEDED.** The ρ = 0.808 statistic was V2 all along; the gate fails for different and better reasons

> **I got this one wrong, and so did the reviewer I endorsed.** A dedicated recompute
> (`sua_exploration/docs/CARRIER_RELIABILITY_V2_RECOMPUTE_20260814.md`, receipt `35413b8a…`) settles it
> by *computing* rather than by reading citations. Verified against its receipt:
>
> | | ρ | matches |
> |---|---:|---|
> | **V2 split-half vs outcome** | **0.8077** | the disputed **0.808**, to every digit |
> | actual V1 code path | 0.7527 | not the disputed value |
> | support-event count | 0.1747 | prior 0.175 ✓ |
>
> C2 **described** a 5-parameter V2 split-half and **cited** `coefficient_split_stability`, which lives
> in V1. C1 caught the inconsistency from the citation — a real documentation error — but inferred
> that the *computation* used V1 and was therefore void. I verified C1's reading of the code and
> endorsed the inference without recomputing. **The computation was right; only the citation was
> wrong.** Reading code establishes what a function does, not which function was run.
>
> **The estimator version was never load-bearing anyway.** Crossing latent rank against basis scope,
> all four variants land in 0.75–0.81. Spearman on 13 points cannot separate 4 free parameters from 5.
>
> **But the abstention gate is still not supported**, for three reasons nobody had found:
>
> 1. **Absolute reliability is ≈ 0.** Median V2 split-half cosine **0.0598**, and 4 of 13 sessions are
>    *negative* — the two halves disagree on the sign of the encoding direction. A gate needs an
>    absolute threshold and 0.06 on a cosine scale cannot supply one.
> 2. **The predicted quantity never goes negative.** All 13 `median_delta_intercept` values are
>    positive (0.0035–0.0236). The proposal exists to handle a *decoder* delta that went
>    +0.0285 / −0.0226. The statistic has never been related to that quantity at all.
> 3. **Partition- and budget-dependent.** Random event-level splits drop ρ to 0.522 (p = 0.070), and
>    against the M3 outcome ρ = 0.467 (p = 0.108) — while at M3 the statistic is *structurally
>    uncomputable*, since the parity split leaves 4–6 events against the estimator's 8-event floor.
>
> **Conclusion: the ranking is real, the gate is not. The paper's conclusion sentence should not be
> replaced.** Provenance passed 13/13 bit-exactly, and leave-one-out ρ stays in [0.755, 0.874], so the
> ranking is not driven by one or two sessions.
>
> Original text retained below for the record.

C1's cross-review argues that C2's provenance check does not license its second idea. Verified; C1 is
right. *(This endorsement is what F12's header now supersedes.)*

Two different H1 sparse estimators exist, and they are not interchangeable:

| | V1 `h1_sparse_event_endpoint.py` | V2 `h1_sparse_event_endpoint_v2.py` |
|---|---|---|
| `CARRIER_DIM` | **4** (line 27) | **5** (line 15) |
| `RIDGE_LAMBDA` | **0.1** (line 31) | **3.0** (line 16) |
| penalty | `diag([0,1,1,1])` (line 498) → q=3 slopes + intercept | q=4 slopes + intercept |

`coefficient_split_stability` is defined in **V1** at line 592 and calls V1's
`fit_carrier_from_arrays`, i.e. q=3, λ=0.1, 4-wide.

C2's SHA provenance check was against `source_audit_v2r2.json` — **V2** carriers. It therefore covers
C2's carrier-scale measurement (idea S1, see F11) but **not** the split-half statistic behind idea S2.
C2's own report describes "a 5-parameter V2 split-half", which cannot be what V1's
`coefficient_split_stability` computes. C1 spotted the inconsistency from the citation alone.

**Scope of the damage — narrow but real.** The *idea* (abstain when the calibration carrier is
unstable, rather than trusting it unconditionally) is untouched, and remains attractive because it
converts "the H1 sparse carrier sometimes hurts" into "it never hurts, it sometimes abstains". What
is void is the specific evidence for it: `ρ = 0.808, p = 0.0008, n = 13` against support-event count's
`ρ = 0.175`. That must be recomputed with `v2.fit_carrier_arrays` before it can be cited or used to
rank S2.

Note this does **not** touch F11. S1's carrier-scale measurement is on V2 objects and is separately
corroborated by the config and docstring evidence in F11.

**Convergence worth recording.** C1 independently moved its recommended first experiment to C2's S1
probe after reviewing it, having originally recommended its own goodness-of-fit gate. Two models
starting from different rankings arrived at the same first experiment on stated technical grounds,
which is mild independent support for running S1 first.

## F13 — H-SE5 checkpoints **do exist** for both dates, and measuring them revises S1

### F13a — the checkpoint-absence claim is wrong

C2's review states that no H-SE5 checkpoints exist on disk (it reports 48 `.ckpt` files, none under
`h1_sparse_event_endpoint/`) and uses this to argue that C1's proposed screen needs two GPU replays
first. **That is incorrect.** There are 2,716 `.ckpt` files in the tree, and all four relevant H-SE5
checkpoints are present:

| Arm | Path under `SPINT-main/logs/` |
|---|---|
| date 1 Full | `h1_sparse_event_endpoint_m4_full_s42_v1/checkpoints/fixed_epoch50/epoch_049.ckpt` |
| date 1 Zero | `h1_sparse_event_endpoint_m4_zero_s42_v1/...` |
| **date 2 Full** | `h1_hse5_lodo_full_19250108_m4_s42_v1/...` |
| **date 2 Zero5** | `h1_hse5_lodo_zero5_19250108_m4_s42_v1/...` |

All four load with `carrier_post_pool.0.weight` of shape **(32, 37)** = 32 activity + **5 carrier**,
confirming they are the 5-wide sparse arm. Both `zero` arms have carrier-column weights of exactly
`0.00000`, which is the expected behaviour of a zero-carrier control and a sanity check that the
measurement is reading the right tensor.

**Consequences.** No GPU replay is needed for either agent's screen. More importantly, C2's own S1
diagnostic can now be run **directly on the failing arm at the failing date**, removing its stated
caveat about transferring a conclusion from an H-C checkpoint across arms. The *working* date-1 and
*failing* date-2 checkpoints can also be compared directly — a sharper test than either agent proposed.

### F13b — measured trained weights: the network **did** compensate, and S1 weakens

Carrier-column weight RMS in `carrier_post_pool.0.weight`, per coordinate (5th = intercept):

| Arm | c1 | c2 | c3 | c4 | **intercept** | all-carrier RMS | activity RMS |
|---|---:|---:|---:|---:|---:|---:|---:|
| date 1 Full (**works**, +0.0285) | 0.234 | 0.073 | 0.088 | 0.329 | **0.033** | 0.188 | 0.114 |
| date 2 Full (**fails**, −0.0226) | 0.214 | 0.035 | 0.037 | 0.208 | **0.032** | 0.136 | 0.118 |

**The intercept has the smallest weight of all five coordinates, on both dates** — roughly 7–10×
below the largest tuning coordinates. So the network did *not* passively accept the input imbalance:
it down-weighted the high-magnitude intercept and up-weighted the low-magnitude signed-tuning
coordinates, which is what Adam should do given enough steps. C2's stated Adam objection to its own
idea is therefore **partly vindicated**, and its separate H-C-checkpoint measurement (carrier columns
29× *below* activity columns) does not transfer to H-SE5, where carrier RMS (0.188/0.136) is
*above* activity RMS (0.114/0.118).

**WITHDRAWN — the "7–12×" effective-imbalance interval.** I previously composed C2's input-scale
measurement with my weight measurement to claim the effective `w·x` imbalance is "about 7–12×, not
32–61×", flagged only with a caveat. The independent audit recomputed all eight paired products and
found **7.8× and 9.8× on the two strongest tuning coordinates but up to 112.6× on the weakest** — a
range that *exceeds* the original claim rather than shrinking it. A caveat was not sufficient; the
interval was wrong and is withdrawn rather than qualified. The effective contribution must be
computed end-to-end in a single pass on real carrier inputs before any number is quoted.

**Also over-read.** "The network did compensate" is one reading of a small final weight, but the
carrier columns are **zero-initialized** and trained under Adam, so a small final magnitude is
equally consistent with the coordinate simply being *least useful*. F11 makes exactly this point
about zero-initialization and F13b then reasons past it. The correct statement is that the trained
weights are inconsistent with naive input-scale dominance, **not** that compensation is demonstrated.

So S1 survives as a real imbalance but is **substantially smaller than advertised**, and it is not
obviously the cause of the date-2 failure, because date 1 works with the *same* qualitative pattern.

### F13c — the date-1 vs date-2 difference points at S2, not S1

The intercept weight is essentially identical across dates (0.033 vs 0.032). What changes is the
**signed-tuning** coordinates: c2 falls 0.073 → 0.035 and c3 falls 0.088 → 0.037, roughly halving,
with total carrier weight dropping 0.188 → 0.136 while activity weight is flat (0.114 → 0.118).

On the failing date the consumer learned to **rely on the carrier less**, and specifically on its
signed-tuning content. That is more consistent with "the carrier content was less reliable on date 2"
— C2's S2 abstention hypothesis and C1's goodness-of-fit gate — than with "the intercept swamped a
good signal", which is S1.

**Caveats, and they are not small.** This is one date against one date, `n = 1`. Weight magnitude is
not the same as functional importance, since these columns feed a ReLU MLP whose downstream layers
can rescale. And the direction of causation is unresolved: the consumer may have down-weighted the
carrier *because* the content was poor, or the content may look poor *because* the consumer
down-weighted it. It is a strong lead, not a conclusion.

**Recommended next step, replacing both agents' proposals.** Run the effective-contribution probe
end-to-end on all four H-SE5 checkpoints in a single pass — real carrier inputs through the real
`ψ_c`, measuring each coordinate's contribution to token variance, at both dates. It is CPU-only,
needs no GPU replay, uses checkpoints that already exist, and discriminates S1 from S2 directly by
asking whether the working and failing dates differ in *imbalance* or in *content reliability*.

## F14 — The date-2 receipt **does** exist, and it shows the H1 sparse failure is one long recording

C2 closed its review with: *"the handoff lists the three date-2 deltas without naming recordings, and
I assumed canonical `H1_HELDIN_SESSIONS` order. The receipt is not on disk. If that ordering is wrong
the finding collapses, so confirming it is step zero."*

**The receipt is on disk**, at
`SPINT-main/pilot_artifacts/h1_hse5_lodo_19250108/terminal_evaluations/H1_HSE5_LODO_19250108_FULL_ZERO5_TERMINAL_v1.json`,
and it names every recording. No assumption is needed, and C2's assumed ordering was correct.

| Recording | Full − Zero5 | target samples |
|---|---:|---:|
| `ses-19250108T110520` | **−0.042554** | **8,330** |
| `ses-19250108T111022` | +0.001088 | 2,508 |
| `ses-19250108T111455` | +0.027104 | 2,269 |

### The headline −0.0226 is sample-weighted and comes from one recording

| Aggregation | value |
|---|---:|
| reported pooled | −0.022590 |
| **sample-weighted mean** | **−0.022145** |
| **equal-weighted mean** | **−0.004787** |

The reported figure tracks the sample-weighted mean, not the equal-weighted one. The single negative
recording carries **8,330 of 13,107 samples — 64% of the evaluation** — while 2 of 3 recordings are
positive. Weighted equally across recordings the delta is **−0.0048, about 4.6× smaller in magnitude**.

This is not an argument that the failure is illusory. It is an argument that the paper's sentence
*"changed from +0.0285 on one development date to −0.0226 on a second"* rests on one recording, and
the sensitivity of that number to the aggregation rule should be known before it is defended in
review. Whether sample weighting is the right convention is a separate question — it is defensible,
but it must be stated.

### A query-horizon hypothesis — proposed here, then **substantially refuted** by audit

**Original claim (retained for the record, now known to be wrong as a date-level explanation).** The
failing recording is 3.3–3.7× longer than the two that succeed while all three share a fixed
four-trial calibration prefix, suggesting within-session drift over a long query block. I argued this
was the most valuable hypothesis available, because re-solving mid-session is nearly free for a
closed-form estimator and impossible for a backprop-based one.

**Refutation.** I wrote that the date-1 receipt "does not expose comparable per-recording deltas, so
the pattern could not be checked on the working date". **That was false.** The data is present under
`metrics/hse5_same_checkpoint_interventions/full/per_session/<ses>/r2` against
`metrics/separately_trained_zero5/per_session/<ses>/r2` — the keys are renamed, not absent. My search
missed it because those keys hold a dict rather than a scalar. The recovered deltas sum to date 1's
published `+0.028469`, confirming they are the comparable quantity.

All five recordings, both dates:

| Date | Recording | samples | Full − Zero5 |
|---|---|---:|---:|
| 1 (works) | `ses-19250101T112404` | 2,230 | +0.035927 |
| 2 (fails) | `ses-19250108T111455` | 2,269 | +0.027104 |
| 2 (fails) | `ses-19250108T111022` | 2,508 | +0.001088 |
| **1 (works)** | **`ses-19250101T111740`** | **6,735** | **+0.025902** |
| 2 (fails) | `ses-19250108T110520` | 8,330 | −0.042554 |

**Date 1's long recording is 6,735 samples — 81% of date 2's failing 8,330 — and its delta is clearly
positive.** Length alone does not predict sign. Worse for the hypothesis, the two dates have almost
identical *mean* query length (4,483 vs 4,369 samples) while differing by ~0.051 in pooled delta, so
an axis that barely varies between dates cannot explain the difference between them.

**What survives:** a within-date ordering in which longer recordings do worse, present on both dates
but with very different magnitude (date 2 falls ~0.070 over 6,061 samples; date 1 only ~0.010 over
4,505). That is a real but second-order effect, worth roughly a fifth of what the date-level failure
requires. It does not justify first priority.

**The process error matters more than the wrong hypothesis.** I declared the one cheap disconfirming
check impossible on a false premise, and then promoted the hypothesis to first priority on the
strength of an untested story. The correct order was to look harder for the disconfirming data
*before* ranking. Credit for the catch goes to the independent audit; see
`AUDIT_OF_COORDINATOR.md`.

### Corroboration for S1's input side

The same receipt family records `s_src = 1.070213494715288` with formula
`s_src = sqrt(mean(source_cache^2)); carrier_norm = carrier / s_src`. This confirms C2's premise that
the normalizer is a **single global scalar**, so per-coordinate magnitude ratios pass through exactly.

### What still stands

The aggregation finding is untouched by the refutation above and is the durable part of F14: the
reported date-2 delta is **sample-weighted**, one recording carries 64% of the evaluation, and equal
weighting gives **−0.0048** rather than −0.0226. The paper's sentence rests on that choice, and its
sensitivity should be known before it is defended in review.

Date 1 shows the same structure in the opposite direction: equal-weighted `+0.030915` against a
published sample-weighted `+0.028469`. So both dates are aggregation-sensitive, and on **both** the
equal-weighted number is more favourable to us than the published one — worth stating plainly rather
than only where it helps.

## F15 — The F8 fix works, but linear decodability **falls** as CEBRA trains — a fairness risk

Track B's seeded sweep of the rebuilt `cebra_joint_behavior` arm against the synthetic positive
control (terminal 482282):

| samples | source iters | source R² | target R² |
|---:|---:|---:|---:|
| 240 | 200 | 0.8084 | 0.8048 |
| 240 | 400 | 0.7323 | 0.7305 |
| 240 | 800 | 0.7125 | 0.7156 |
| 400 | 200 | 0.7522 | 0.7558 |
| 400 | 400 | 0.7315 | 0.7324 |
| 400 | 800 | 0.6924 | 0.6930 |

**F8 is fixed.** Target tracks source to within ±0.004 in every configuration, against the broken
arm's 0.9891 source / −1.2278 target. That equality is the diagnostic signature of correct
cross-session alignment.

**But absolute R² decreases monotonically with source iterations** in both sample regimes
(0.8084 → 0.7125 and 0.7522 → 0.6924). More training makes the embedding *less linearly decodable*.
This is expected behaviour for InfoNCE — the embedding spreads toward uniformity on the hypersphere,
which is good for the contrastive objective and bad for a linear readout. It is very likely why
CEBRA's own papers decode with kNN rather than a linear map.

**Why this is a fairness problem for us, not a curiosity.** Track B chose **linear ridge as the
primary** downstream decoder and kNN as secondary, on the reasonable ground that it matches FALCON's
`NoMAD + WF` convention and every internal comparator. But if longer CEBRA training reliably lowers
the linear score, then a linear-primary protocol **systematically understates CEBRA**, and it does so
more the better CEBRA is trained. That is the `HANDOFF_COMPARATORS_20260812.md` §10 trap again,
arriving from an unexpected direction, and reporting rule 5 forbids leaving a comparator
disadvantaged by a convention we chose.

**Required before any scoring run:** select the source-iteration count by a criterion fixed *inside*
the source data (per reporting rule 1), and report the **kNN arm alongside the linear arm** rather
than as an optional secondary. If the two decoders disagree about CEBRA's standing, that disagreement
is the honest result and must be printed, not resolved in our favour by protocol choice.
