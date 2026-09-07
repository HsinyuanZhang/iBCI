# PRI-T × T4 feasibility: gate measurement and threat analysis

Wilson et al., *Nat. Biomed. Eng.* 2025, `s41551-025-01536-z` (PRI-T). Scratch analysis; every
file here is new, nothing outside `sua_exploration/prit_feasibility/` was touched.

---

## 0. Provenance and constraint compliance

| Check | Status |
|---|---|
| `git clone https://github.com/guyhwilson/PRI-T` | succeeded → `prit_upstream/` |
| Paper full text | retrieved (nature.com HTML); all quotes below are verbatim |
| Existing files modified | **none**; `git status` shows zero new `M` entries vs. pre-session snapshot |
| Formal test sessions opened | **none** (`forbidden_test_sessions_opened: []` in every decode receipt) |
| DANDI `cursor_vel` / `cursor_pos` / `cursor_acc` read | **never**, in any role |

The label constraint was enforced mechanically, not by inspection. `decode_forward.py` zeroes the
behaviour tensor before it reaches the dataset object, so the decoded velocity cannot be
contaminated even by accident, and each receipt records the forbidden NWB streams as
`present_but_unread: [Velocity, Position, Acceleration]`. The HMM's position observation is the
cumulative sum of the **decoded** velocity (`pseudo_pos`), which is the offline analogue of the
closed-loop cursor that PRI-T actually observes.

One read of measured kinematics does occur, in `positive_control.py`, and it is on the upstream
repo's own bundled demo array (`exampledat.mat`), not on any DANDI file. That is their data and
carries no claim of ours.

## 1. The gate number

Six A2 development sessions of DANDI 000688 sub-C, centre-out. Decoded forward-only from frozen
source checkpoints, 30 calibration-pool trials excluded, 1,206 evaluation trials total. Metric:
fraction of trials whose Viterbi-majority inferred direction equals the true `target_dir`,
8 classes, **chance = 0.125**, per-session majority class 0.132–0.159.

Two configurations are reported per arm: **paper defaults** (their recommended `gridSize=20`
→ 400 states, `stayProb=0.999`, distance-modulated `kappa`, raw velocity, every timestep
observed) and **best swept** (the maximum over 276 configurations per arm, selected on these same
six sessions, so it is an optimistic ceiling with no held-out selection).

### Arm T4 — identity fitted from the session's own labels

| Session | n | paper default | best swept | HMM-free velocity vote |
|---|---|---|---|---|
| `20151103` | 220 | 0.327 | 0.514 | 0.455 |
| `20151104` | 289 | 0.467 | 0.761 | 0.775 |
| `20151106` | 228 | 0.404 | 0.759 | 0.759 |
| `20151109` | 143 | 0.406 | 0.741 | 0.741 |
| `20151110` | 168 | 0.357 | 0.786 | 0.756 |
| `20151112` | 158 | 0.335 | 0.778 | 0.759 |
| **mean** | | **0.3827** | **0.7232** | **0.7075** |

### Arm Z4 — zero-carrier identity, the realistic cold start

| Session | n | paper default | best swept | HMM-free velocity vote |
|---|---|---|---|---|
| `20151103` | 220 | 0.318 | 0.300 | 0.268 |
| `20151104` | 289 | 0.325 | 0.422 | 0.381 |
| `20151106` | 228 | 0.197 | 0.430 | 0.452 |
| `20151109` | 143 | 0.210 | 0.245 | 0.245 |
| `20151110` | 168 | 0.327 | 0.577 | 0.488 |
| `20151112` | 158 | 0.392 | 0.652 | 0.608 |
| **mean** | | **0.2951** | **0.4377** | **0.4068** |

### Arm T4-zeroid — negative control, T4 checkpoint deployed with its identity zeroed

Mean 0.1185 (paper default) and 0.1477 (best swept) against chance 0.125; per-session 0.109–0.167;
median angular error 90°. This is the arm that corresponds to *literally* deploying our released
T4 checkpoint with no target-session labels available.

### Three readings that matter more than the headline numbers

**The HMM is not the active ingredient offline.** A one-line baseline — take the mean of the
decoded velocity over the same speed-gated timesteps and snap it to the nearest of 8 directions,
no HMM, no states, no transitions — scores 0.7075 against PRI-T's 0.7232 on T4 and 0.4068 against
0.4377 on Z4. The entire apparatus of a 400-state grid, a von Mises emission and Viterbi buys
+0.016 and +0.031. Whatever direction information exists in our decoded velocity is recoverable
without PRI-T; PRI-T is not adding inference power here, it is reading off a signal that is
either present or absent.

**The good T4 number needed a modification that is not in PRI-T.** PRI-T's emission model uses
only the *angle* between decoded velocity and the candidate target, discarding speed
(`_compute_dists_and_angles`); `adjustKappa` modulates precision by cursor-to-target *distance*,
never by speed. In closed-loop cursor control the cursor is essentially always in motion toward
something, so this is harmless. Our centre-out trials contain long hold and inter-trial periods
where decoded velocity is near-zero noise, and PRI-T weights those directions equally with the
reach. Adding a speed gate that keeps the top 25% of timesteps by decoded speed moves T4 from
0.3827 to 0.7232. That is a real and reportable finding, but it means the 0.72 figure is *our*
repair of their method plus hyperparameters tuned on the evaluation sessions, and the number you
get from dropping PRI-T in as published is 0.38.

**Label-noise geometry, which is what T4 actually cares about.** T4 consumes one direction scalar
per trial, so PRI-T's errors are shared across all units within a trial. To first order they
therefore act as a *common* attenuation of every unit's fitted `[a, c]` by
λ = E[cos(inferred − true)], plus added variance. Measured:

| Arm | accuracy | λ = E[cos δ] | δ = 0° | 45° | 90° | 135° | 180° |
|---|---|---|---|---|---|---|---|
| T4 | 0.719 | **0.896** | 0.719 | 0.258 | 0.017 | 0.004 | 0.002 |
| Z4 | 0.432 | **0.656** | 0.432 | 0.386 | 0.121 | 0.044 | 0.017 |
| T4-zeroid | 0.149 | **0.082** | 0.149 | 0.275 | 0.248 | 0.223 | 0.104 |

A common scale factor on `[a, c]` is largely absorbed by the pipeline's per-column z-scoring, so
λ = 0.896 would be close to harmless — but that is the arm that already had the labels. At the
cold start that the composition requires, λ = 0.656 with 18% of trials wrong by 90° or more, and
those are not a common scale, they are structured error in the direction of the carrier itself.

### Positive control: the implementation is correct

The obvious objection to a weak result is that we mis-transplanted the method. Ruled out on their
own data. `exampledat.mat` ships 10,000 timesteps of their closed-loop simulation described as
"after a nonstationarity occurs in the neural tuning matrix. The decoder is fixed, meaning there
is now a mismatch between it and the neural tuning" — i.e. exactly the misaligned-decoder
recalibration regime. Our inference code, unmodified, recovers the direction to the true target
with a **median absolute error of 7.97°** (λ = 0.744, 84 distinct target locations, continuous
rather than 8-class). The transplant works where PRI-T is supposed to work.

Independently, `fast_prit.py` (an O(T·S) Viterbi and forward-backward exploiting the
ε / (1−ε)/(N²−1) transition structure the paper specifies) was verified numerically equivalent to
the upstream O(T·S²) implementation, and the T4-zeroid arm returns chance, confirming the
measurement is not inflated by a degenerate path.

## 2. Verdict: **dead** — for the composition as proposed

The composition is PRI-T → T4: infer per-trial directions, feed them to the OLS carrier fit,
obtain a label-free and still BP-free pipeline. It is dead, and the reason is structural rather
than a matter of tuning.

The labels T4 needs are for the target session's **calibration pool**, which must be decoded
*before* any adaptation has happened. So the operative number is the cold-start one, and the
cold-start bracket is:

- deploy our actual T4 checkpoint with no labels, identity zeroed → **0.12–0.15, i.e. chance**;
- deploy a separately Z4-trained checkpoint that was built to run without identity → **0.30
  (paper defaults) to 0.44 (oracle-tuned)**.

Either way you are handing T4 a label set that is mostly wrong, in order to spare it 50 scalars
per session. And the 0.72 that looks encouraging is only available *after* the true labels have
been used to build the identity, which is circular in exactly the direction that kills the
proposal: inference quality offline is bounded by decoder quality, and the regime where you need
the labels is the regime where inference is worst. This is the paper's own stated limitation,
reproduced quantitatively on our data with one implementation across both settings — median 8°
error on their closed-loop-generated data, median 45° on our open-loop data at cold start.

I am deliberately not calling this "viable-within-subject-only". Our A2 setting already *is*
within-subject (sub-C source → later sub-C sessions) and it fails at cold start, so within-subject
scope does not rescue it.

The variant that *is* alive, and should be named precisely so it is not oversold: T4 → PRI-T,
where T4 is fitted from real labels first and PRI-T then reads directions off the improved decoder
at 0.72. That is a refinement loop. It does not remove the label requirement, so it contributes
nothing to the supervision-efficiency claim, and it is worth at most a sentence.

Whether the composition would work in a genuine closed loop with a human is untested and
untestable here. The paper's mechanism says it probably would. We should concede that openly
rather than simulate it weakly.

## 3. The threat paragraph

> PRI-T (Wilson et al., 2025) recalibrates a cursor decoder with zero ground-truth labels by
> inferring, with an HMM over a discretized screen, which target the user is moving toward, and
> refitting on the resulting pseudo-labels. It is the right comparison to raise against a
> label-efficiency claim, and it does not apply to our setting. PRI-T's observations are the
> decoder's *own* outputs, so the information it exploits is created by the user closing the loop:
> the paper's own resolution of why PRI-T succeeds in closed loop but fails offline with a badly
> misaligned decoder is that "the user is driving the cursor towards the target with the poorly
> aligned decoder while using visual feedback to correct for any decoding errors… These user
> corrections make the decoder output more informative of the true target location", whereas
> offline "these corrections cannot be applied… leading to worse target inferences". Our setting
> has no closed loop and no corrective behaviour by construction. We measured this directly: our
> transplant of PRI-T, validated on the authors' own released closed-loop example data (median
> direction error 7.97° with a deliberately misaligned decoder), recovers only 0.30 of eight-way
> reach directions at the authors' recommended hyperparameters, and 0.44 under hyperparameters
> tuned on the evaluation sessions themselves, when run on our frozen decoder's outputs in the
> cold-start condition that a label-free pipeline actually requires (chance 0.125, n = 1,206
> trials, six sessions). At that error rate the per-trial direction labels PRI-T supplies would
> attenuate the cosine carrier T4 estimates by λ = E[cos δ] = 0.66 with 18% of trials wrong by
> 90° or more. Two further scope differences are worth stating. PRI-T requires 2D cursor task
> structure — a discretized screen of candidate goal locations and a point-at-target intention —
> and is demonstrated only on linear (ridge) decoders, for which its weighted-least-squares refit
> is closed-form; our adaptation is of a nonlinear attention decoder, where no such closed form
> exists and where the alternative to T4 is gradient training. And PRI-T updates decoder weights
> every session, in a chain: single-day-pair recalibration fails for all of PRI-T, ADAN and FA
> stabilization above roughly 90° of encoding-subspace rotation, and it is iteration over chained
> days that recovers performance. Our decoder's weights never change.

### Verification of the four legs, and which is weakest

| Leg | Verdict | Evidence |
|---|---|---|
| 1. Closed-loop human interaction | **Verified, strongest** | Verbatim, quoted above. It is the paper's own explanation for its own offline failure, in the Results. Nothing to attack. |
| 2. Discrete-target 2D cursor task structure | **Verified but weak as framed** | The HMM state space is an N×N screen discretization, and they say they chose it "to maintain generality across cursor tasks". They validate on radial-8, grids, *random* target locations, and freeform personal use (email, web browsing, e-books) with "a uniform prior across all possible locations". The restriction is real but far broader than "discrete targets known a priori". |
| 3. ~90° failure, chain is the mechanism | **Verified** | "PRI-T, ADAN and FA stabilization all perform poorly in the face of large rotations… and seem to fail completely for rotations larger than roughly 90°"; then "when performing unsupervised recalibration using only a single pair of days all approaches appear unsuitable… We therefore reasoned that iterative recalibration approaches that use chains of days might work better". |
| 4. Per-session weight updates vs. our frozen decoder | **Verified mechanically, contestable rhetorically** | `recalibrate()` calls `decoder.fit(...)`, replacing the weights each update. True, but a reviewer can fairly say updating weights is a feature, not a cost. |

**Weakest leg: #2, the task-structure argument.** It is the one a reviewer will attack, because
the paper pre-empted it. If we write "PRI-T needs a stereotyped discrete-target task", the
freeform T11 result is a one-line rebuttal. State it as a *dimensionality and intention* constraint
instead — 2D cursor, point-at-target, a screen that can be discretized — and note that our
estimand is per-unit directional tuning rather than a cursor goal.

Leg #4 is second weakest and needs the reframing above: not "we don't update weights" but "no
closed form exists for our decoder".

### On BP-free: your suspicion is correct, and here is the precise remaining claim

PRI-T's refit **is** closed-form. The upstream code is
`decoder.fit(neural_flattened, inferredPosErr, maxProb**2)` — a scikit-learn `LinearRegression`
with `sample_weight`, i.e. weighted least squares in closed form, no optimizer, no gradients. The
regression target is `inferredTargLoc − cursorPos`, the point-at-target vector. So "BP-free" does
not separate us from PRI-T at all, and any sentence implying PRI-T needs backpropagation is
wrong and will be caught.

What survives is narrower and still worth saying: PRI-T's closed form exists *because the decoder
is linear*. The paper uses linear decoders throughout (11 mentions of "linear decoder"; the sole
"nonlinear" refers to ADAN, the competitor; the RNN mention concerns how T11's data was
*collected*, not what was recalibrated). SPINT is a nonlinear attention decoder, and T4 achieves
BP-free target-session adaptation of one, which has no PRI-T analogue. Claim that, not BP-freeness
in general.

## 4. Reproduction judgment

Your position is right on all three counts, with one addition and one correction of emphasis.

**Clone the compact repo — correct, and it was necessary.** The README's recommended defaults
(`gridSize ≥ 20`, `stayProb` 0.99–0.9999, `vmKappa` 2–8, logistic inflection ≈10% of screen width,
exponent 20–40) are not in the paper, and `adjustKappa`'s exact functional form and the fact that
speed is discarded are only visible in the source. Guessing from the abstract would have produced
a strawman.

**Skip the full `nonstationarities` repo — correct.** FA stabilization and ADAN are already closed
here for an independent and stronger reason: `label_free_alignment_baseline/` records
`generic_cca_or_procrustes` as `FAIL_CLOSED_NO_PAIRED_OBJECT_OR_ANCHOR`, and cross-unit Gram
whitening measured `+0.0017046`. A reference implementation cannot reopen a family that fails for
lack of a paired object. Take their published `0.89`–`0.91` medians as the comparator numbers.

**Skip the Dryad download and do not reproduce the closed-loop simulation or the T5 month —
correct.** Their simulator is a variant of the Willett et al. 2019 piecewise-linear-model
simulator with an added tuning model, SNR-matched to T5 and drift-matched by an exponential decay
fit (0.91–0.93 d⁻¹). It generates *synthetic* neural activity from a low-dimensional tuning
matrix. Driving SPINT with it would replace the real per-unit drift in DANDI 000688 — the thing
our claim is about — with hand-set synthetic drift, and the reviewer would discount it
accordingly. The one thing it could legitimately supply is the corrective feedback our offline
gate structurally lacks, which is precisely the mechanism at issue; but a positive result from a
simulated user is not evidence about a human, and the honest move is to concede the closed-loop
case as out of scope rather than to simulate it weakly.

**The addition.** You did not need Dryad for the one thing I thought it was needed for. The
compact repo ships `examples/exampledat.mat`: 10,000 timesteps of their closed-loop simulation
with ground-truth `targetPos`, `neuralActivity` (192 ch), the decoder `D`, and `neuralTuning` as
192×3 `[b₀, b₁, b₂]` — their own cosine parameterization. That is a free positive control for our
implementation, and it is what lets us assert that 0.30 on our data is a property of the setting
rather than a bug. If the gate had come out weak without that control, the result would have been
unusable. Worth knowing for any future transplant: check for bundled demo data before deciding a
data download is required.

## 5. Fig. 1b and Fig. 1f: confirm 1b (more strongly than you claimed), correct the label on 1f

**Fig. 1b — confirmed, and it is a stronger citation than you realized.** The caption is
per-channel: "Example neural nonstationarities from two distinct sessions (two days apart). The
dotted lines correspond to cosine tuning models fit within each session and the solid lines are
empirical firing rates (FRs) for different angles with respect to the target position", with
bootstrapped 95% CIs. The Methods then give the model and estimator verbatim:

> "The standard cosine tuning model x_t = b₀ + a·cos(θ_p − θ_t) … **is equivalent to a linear model
> of activity, x_t = b₀ + b₁cos(θ_t) + b₂sin(θ_t), where b₁ = a·cos(θ_p) and b₂ = a·sin(θ_p). We
> hence used standard least squares to fit models** …"

and, separately, "Cosine tuning is a standard model in the field that captures directional tuning,
which has been shown to be a large signal in the motor cortex". That is T4's parameterization and
T4's estimator, identically — a 2025 *Nat. Biomed. Eng.* paper using OLS-fitted `b₀, b₁, b₂` cosine
coefficients to *define* the nonstationarity it then corrects. Cite this for "field-standard
estimand", and note that their `neuralTuning` array is literally stored as 192×3 `[b₀, b₁, b₂]`.

One difference to state rather than paper over, because it favours us. They regress against the
**instantaneous** cursor angle / "ground-truth (unit) displacement vector" — a per-timestep
regressor requiring dense measured kinematics. T4 uses **one direction scalar per trial**. Same
model, same estimator family, far cheaper regressor. Do not write "we use the same procedure";
write that we estimate the same field-standard quantity from a per-trial label instead of a
per-bin kinematic stream.

**Fig. 1f — numbers right, label wrong; fix this before it reaches a reviewer.** Fig. 1f is not
per-unit tuning similarity. The Methods define it as a *population readout* quantity:

> "To measure tuning similarity, we examined the cosine angle between the weights of ridge
> regression models after collapsing across the X and Y dimension coefficients. For instance, for
> decoder weight matrices D₁, D₂ ∈ R^{192×2}, we compute cos(θ) = D₁[⋮]ᵀD₂[⋮] / (‖D₁‖_F‖D₂‖_F)"

— one scalar per session pair, from flattened decoder weight matrices, which the Results text
calls "the subspace drift" and "a global measure of all PD changes occurring across the neural
population". So it is decoder-weight subspace alignment, not tuning-curve similarity per unit.

Your `0.75` → `0.33` is numerically defensible but is partly your own arithmetic. The paper prints:
"The median tuning similarity within the session was around 0.75 (a 41° difference), but dropped
with 1–2 weeks of separation between sessions (71° difference)", and "The median coefficient of
determination (R²) correspondingly moved from 0.39 to 0.21". The `0.33` is cos 71°, which the paper
does not print. Safest citation: quote the angles (41° within session → 71° at 1–2 weeks), give
cos 71° ≈ 0.33 as your own conversion if you want the similarity scale, and attribute it to
decoder-weight subspace drift. The R² 0.39 → 0.21 is an independently quotable and useful number.

**Net effect on our framing.** Cite Fig. 1b plus the Methods for "per-unit cosine tuning fit by
OLS is the field-standard characterization, and this is the quantity T4 estimates". Cite Fig. 1f
for "the readout subspace itself rotates substantially at the 1–2 week scale, with R² falling
0.39 → 0.21" — evidence that the nonstationarity is large and matters, at the population level.
Two different levels of description, both useful, and they should not be merged into one sentence.

**One more thing in the paper that bears on us, unprompted.** Their unsupervised hyperparameter
selection — choosing the setting that maximizes Viterbi probability — costs only
Δr = −0.016 ± 0.036 versus full oracle optimization (Supplementary Fig. 1d). So PRI-T has a
legitimate label-free route to hyperparameter selection, and we cannot dismiss its tuned numbers
as oracle-dependent. Our own best-swept figures, by contrast, *were* selected on the evaluation
sessions, which is why the paper-defaults column above is the one to quote in the paper. Relatedly,
their confidence weighting helped on T5 but was "strictly worse in T11's personal use data,
implying that this setting may be participant or task dependent" — a fair, citable note on
per-deployment fragility.

---

## Files

| File | Purpose |
|---|---|
| `decode_forward.py` | Forward-only decode of the 6 A2 dev sessions under T4 / Z4 / T4-zeroid; zeroes behaviour before it reaches the dataset; writes receipts |
| `prit_gate.py` | PRI-T target inference over decoded velocity + integrated pseudo-position; hyperparameter sweep |
| `fast_prit.py` | O(T·S) Viterbi / forward-backward for the paper's transition structure, with equivalence check against upstream |
| `positive_control.py` | Validates the transplant on the upstream repo's own closed-loop demo data |
| `summarize_gate.py` | Builds the deliverable table and the label-noise geometry |
| `gate_summary.json` | Per-session gate numbers, both configurations, all three arms |
| `gate_results_task.json`, `gate_results_grid.json` | Full sweep, 276 configurations × 3 arms = 828 rows |
| `positive_control.json` | Positive-control result |
| `decoded/` | Decoded velocities, true directions, receipts |
| `prit_upstream/` | Clone of `github.com/guyhwilson/PRI-T` |
