# Design — Anchored Output Fusion V1

Date: 2026-09-02  
Status: **revised source-screen candidate; launch blocked until authority-parity audit passes**  
Working name: **AOF** (Anchored Output Fusion)  
Resource scope: physical GPU0 only; physical GPU1 is out of scope

## 1. Starting point

Three completed experiments now constrain the PF problem:

| Result | Evidence |
|---|---:|
| Pure corrected PF-R1 minus PF-MEAN, external | +0.051466, 4/6 |
| Pure corrected PF-R1 minus native POOLED, external | -0.060687, 1/6 |
| Identity-space scalar APFG minus native, external | +0.004612, 4/6 |
| APFC A-DC2 minus scalar, source validation | -0.000653, 0/2 |
| APFC A-TB4 minus scalar, source validation | -0.001493, 0/2 |

Pure Post-Fusion retains complementary information but destroys the strong
native solution. Zero anchoring repairs the catastrophic loss. Adding gate
capacity does not improve transfer. The remaining concrete hypothesis is that
the *location* of interpolation is wrong.

The current identity-space APFG computes

```text
y_id = D(x, h_native + alpha * (h_post - h_native)).
```

The frozen decoder `D` contains `fc_in`, a transformer, and `fc_out`; it is
nonlinear in the identity. Consequently identity interpolation need not retain
the useful component of the decoded Post-Fusion prediction.

## 2. Proposed operator

AOF decodes both frozen views and fuses only their output:

```text
y_native = D(x, h_native)
y_post   = D(x, h_post)
y_AOF    = y_native + beta * (y_post - y_native).
```

There is one global scalar `beta`. There is no learned backbone, vector gate,
session parameter, target update, pseudo-label, or online optimizer.

At IEEE `beta=+0.0`, the public path must return `y_native` directly rather
than perform multiply-add arithmetic. The exact-zero prediction bytes, R2,
window starts, targets, and frozen model state must equal the native POOLED
comparator.

This is not output smoothing: AOF mixes two simultaneous decoder predictions
at the same governed coordinate. It has no temporal filter and introduces no
past or future behavior output.

## 3. Divergence and convergence

The following alternatives were considered before source execution:

| Candidate | Decision | Reason |
|---|---|---|
| More APFG scalar epochs | reject | scalar already plateaued at epoch 11/12 |
| Larger temporal gate | reject | APFC-TB4 is worse than scalar on both validation sessions |
| Disagreement-conditioned gate | reject | APFC-DC2 is worse than scalar on both validation sessions |
| Per-unit identity gate | reject | unit identities do not align across sessions |
| Full 2x2 output affine | defer | six effective parameters including offset; scalar simplicity must fail first |
| Per-output-coordinate beta | defer | two parameters add little diagnostic value before scalar feasibility |
| Session-conditioned beta | reject | not identifiable on a new session without target labels |
| Joint native/PF decoder retraining | reject | loses exact nesting and reopens the failed pure-PF route |
| Feature-space fusion after `fc_in` | defer | less directly deployable and less interpretable than output fusion |
| Frozen output-space scalar | **keep** | one parameter, exact native anchor, closed-form source fit, architecturally representable in EvalAI |

The simplicity test therefore selects exactly one AOF arm. No hyperparameter
grid, seed sweep, clamp, regularizer, or alternate basis is permitted in V1.

## 4. Deployment-matched static authority

The primary source screen must match the static EvalAI contract, not the local
trial-boundary-only `UNCAPPED` law:

- frozen Selected-T4 POOLED checkpoint
  `25d7bc72b4d440004b58f1beaeadb7e15565a43e83dd1eadd160374270ec1d3e`;
- frozen student state
  `2a340745e2e1b4c7eecb4b187c9b061548bf3686f12389aae76bf53e4f3acc20`;
- the exact official `act30_dopt4` activity authority:
  `dataset.calib_trialized_neural_features[session][:30]` under the frozen
  PIT cubic interpolation law;
- first 30 public calibration trials, in their stored chronological order, as
  the static activity pool;
- D-optimal four-trial support and its selected-support4 fixed-ridge T4
  carrier;
- frozen train-only normalizer;
- no query-trial commit and no online identity update;
- governing last-bin, variance-weighted M2 R2.

The earlier G00m-linear activity reconstruction is explicitly forbidden in
AOF. It is a different activity authority from the official static payload and
would make both the zero anchor and the transfer claim ambiguous. The source
receipt must bind the frozen interpolation mode, ordered first-30 activity
digest, D-opt support digest, raw and normalized T4 digests, and the exact
official native identity/prediction witness. The native identity authority is
the existing `act30_dopt4` export receipt. The native prediction authority is
the immutable `ridge_activity30_m4 / within_post30` row stream in
`m2_t4_activity_budget_screen_v1/score.json`; its body, sidecar, topology, row
order, per-session starts/targets/windows, prediction digest, and R2 must be
descriptor-validated after attempt publication.

These are two deliberately separate numerical authorities. The official
cached identity was constructed on CPU, whereas the historical screen identity
and predictions were constructed on GPU. A bounded prelaunch seam measured on
`ses-2020-10-19-Run1` found identity max-abs `1.5497207641601562e-6`, prediction
max-abs `4.0978193283081055e-8`, and R2 absolute difference
`2.161476608808499e-8`; their SHA values are therefore not interchangeable.
AOF must preserve both honestly. A second bounded prelaunch check found that a
fresh strict-clone CPU recomputation also differs from the stored official CPU
identity by max-abs `2.980232238769531e-7`; consequently only the already
SHA-bound payload bytes are a byte authority, while every recomputation is a
numerical bridge:

- read the official CPU identity from the exact SHA-bound payload and match its
  stored identity SHA exactly;
- recompute the same identity with the fresh strict CPU clone and require
  max-abs `<=2e-6` versus the stored payload identity, without claiming SHA
  equality;
- independently recompute the GPU identity and match the sealed GPU source
  prediction SHA exactly;
- record a CPU-identity-to-GPU-identity numerical bridge for every source
  session, with identity max-abs `<=2e-6`, prediction max-abs `<=2e-6`, and R2
  absolute difference `<=2e-7`;
- never relabel CPU-recompute or cross-device numerical bridge equality as byte
  equality.

The GPU source screen uses the GPU identity/prediction branch as its governing
native comparator. A future EvalAI package must use the CPU cached identity and
repeat exporter/runtime/container validation under a separate work order.

For each session, cache both `h_native` and `h_post` offline. The proposed
runtime needs only those two identities, the frozen decoder, the global beta,
and the ordinary 50-bin neural history. It does not need trial metadata or
`on_done`. This makes AOF architecturally representable by the existing EvalAI
M2 interface with `IsTestTimeAdaptive=false`; it is **not yet packaging-proven**.
Exporter, runtime, container, latency, and exact native-parity validation remain
separate prerequisites.

## 5. Source-only closed-form fit

Use this exact, predeclared lexical source split:

```text
fit:
  ses-2020-10-19-Run1
  ses-2020-10-19-Run2
  ses-2020-10-20-Run1
  ses-2020-10-20-Run2
  ses-2020-10-27-Run1
validation:
  ses-2020-10-27-Run2
  ses-2020-10-28-Run1
```

These are source-session behavior targets. Their use must be disclosed as
`source_target_access=true` from source DataModule preparation onward. Hidden
target, external target, held-out target, and EvalAI target access all remain
false.

The inherited PIT `setup("fit")` eagerly loads all seven declared source
calibration files, including their behavior arrays. Therefore V1 does **not**
claim filesystem-level blindness to the final two source sessions. Its
predeclared code-path holdout is narrower and auditable: before beta is frozen,
the fitter must not index, digest, copy into a paired-output record, summarize,
or otherwise consume either validation session's behavior array. It must first
materialize the five fit records and freeze beta, then materialize and evaluate
the two validation records exactly once.

Materialize paired frozen predictions on the five fit sessions. Let
`d = y_post - y_native` and `e = target - y_native`, after the inherited
behavior scaling factor **5.0** and at governed last-bin coordinates. Runtime
must validate the literal factor against the frozen model/payload metadata.

To prevent a long session from dominating, fit beta to equal-session MSE:

```text
numerator   = sum_s (1 / n_s) * sum_{i in s} dot(d_i, e_i)
denominator = sum_s (1 / n_s) * sum_{i in s} dot(d_i, d_i)
beta        = numerator / denominator
```

Accumulate numerator and denominator in declared-order float64. Fail closed if
the denominator is nonpositive/nonfinite or beta is nonfinite. Do not clamp,
round, regularize, or tune beta. No optimizer and no training epoch exist.

This is the minimizer of the declared **equal-session MSE surrogate**. It is not
the direct maximizer of variance-weighted R2. R2 remains the governed reporting
and decision metric; the distinction must be explicit in every fit receipt and
paper claim.

The fit receipt must bind session order, per-session window counts, numerator,
denominator, denominator scale/conditioning evidence, beta, finite-input
checks, behavior scaling factor, native/post prediction digests, targets,
starts, identities, T4, normalizer, and model state before/after.

## 6. Source validation and kill gate

Evaluate the single fitted beta exactly once on the two held source sessions.
Publish native, post, and AOF R2 per session and equal-session means.

AOF is eligible for all-seven closed-form refit only if all conditions hold:

```text
mean(AOF - native) >= +0.005 R2
positive sessions = 2/2
worst session delta >= +0.001 R2
beta is finite and identical in every validation row
native == beta(+0.0) bitwise on every row and its prediction SHA equals the sealed GPU source anchor
```

The `+0.005` bar is deliberately above the completed scalar APFG source gain
of approximately `+0.0022`. A smaller source result would not justify another
official submission or the doubled decoder computation.

If the gate fails, close output-space PF fusion. Do not add a vector beta,
intercept, nonlinear combiner, more source sessions, or target fitting in the
same study.

If it passes, recompute the same closed-form beta once on all seven source
sessions. This is a refit, not a new selection. The validation result and the
all-seven deployment beta must both remain in the receipt graph.

## 7. Coordinated execution

One GPU0 process should:

1. publish attempt before checkpoint, data, or CUDA access;
2. prepare one source-only PIT DataModule, declare source-target access from
   that eager-load boundary, install no trainable wrapper, and strict-load one
   frozen Selected-T4 model through a loader that constructs no optimizer;
3. construct one static first-30 native/post identity pair per source session;
4. decode the official native anchor with an independent `B` call and decode
   the post identity with a second independent `B` call; on the first chunk of
   each session, additionally run one concatenated `2B` call only as a bounded
   numerical diagnostic;
5. accumulate the five-session closed-form fit;
6. evaluate the two source validation sessions once;
7. either stop with a failed source gate or perform the deterministic all-seven
   closed-form refit;
8. publish immutable source result and terminal/failure receipts.

Cache keys must include the session, ordered activity and support digests,
raw/normalized T4, normalizer, native/post identities, governed starts, neural
window digest, and frozen decoder state. Caching an identity alone is
insufficient because the paired outputs also depend on `x`.

GPU1 must not be queried, initialized, scheduled, signaled, or inspected.

Before source launch, a fixed numerical audit must compare both halves of the
diagnostic `2B` call with the corresponding independent `B` native/post calls
on the same frozen input. The predeclared tolerances are prediction maximum
absolute difference `<=2e-6` and R2 absolute difference `<=2e-7`. The `2B`
result is diagnostic only and is not substituted into the fit. Separately, the
public `beta=+0.0` sentinel must take the direct native branch: it must not
compute `h_post`, invoke the second decoder branch, or perform residual
multiply-add arithmetic. At the official batch law its full-session native
prediction SHA, window count, starts, and target evidence must equal the frozen
`act30_dopt4` GPU screen anchor; R2 absolute difference must be `<=1e-12`.

## 8. Untouched evaluation

The six local external sessions used to diagnose APFG have already been
observed and cannot confirm AOF. They may only be used later as a descriptive
mechanism audit after the source decision is frozen.

If and only if the source gate passes, a separate work order may implement and
validate an AOF package for one untouched EvalAI M2 submission using two cached
identities per dataset tag. The exact native `act30_dopt4` payload/checkpoint is
the comparator. No submission is authorized merely by this design. The future
package must declare no online adaptation and must prove:

- beta zero returns the selected native output exactly within its declared CPU
  or GPU execution authority; CPU-payload and GPU-screen bytes are not
  conflated;
- dataset-tag and batch ordering are unchanged;
- weights and cached identities never mutate;
- no trial metadata, calibration arrays, labels, optimizer, or selection law
  are present at predict time;
- prediction latency and image size remain within the evaluator envelope.

An official improvement is meaningful only at `>= +0.010` R2 over the matched
static native comparator. A smaller positive source or official delta is a
bounded diagnostic, not a headline method result.

## 9. Three validation experiments

### V0 — algebra and exact anchor

- synthetic closed-form beta recovers the known scalar optimum;
- session-balanced accumulation is invariant to within-session row partition;
- beta `+0.0` follows the direct native branch bitwise;
- model parameters and identities remain unchanged;
- synthetic V0 needs no source/hidden target or CUDA access.

### V1 — source-only STATIC30 screen

- one frozen checkpoint and one DataModule;
- five source-target fit sessions, then two source validation sessions whose
  behavior arrays were loader-resident but not consumed by the fitter;
- paired native/post batch evaluation;
- one closed-form beta, no epoch/seed/hyperparameter selection;
- exact source gate and conditional all-seven refit.

### V2 — untouched EvalAI transfer

- two cached identities per dataset tag;
- one frozen all-seven beta;
- static runtime only, no completed-trial boundary required;
- matched native zero sentinel and `>= +0.010` official effect gate.

## 10. Strongest objection

**Objection.** AOF is merely an ensemble that doubles decoder compute and may
overfit two source validation sessions; it is not a representation advance.

**Response.** The claim is deliberately narrower. AOF tests whether PF carries
complementary behavior information that survives only after nonlinear decoding,
while exactly preserving the strong native model at zero. The scalar uses only
declared source-session targets and no hidden/external target or iterative model
training; the function class and source gate are frozen before execution, and
any method claim requires untouched official transfer. If the source gate
fails, the entire output-fusion branch stops rather than expanding capacity.

## 11. Feasibility pilot

- Day 1: additive paired-output cache, closed-form fitter, and exact-zero tests.
- Day 2: no-data audit and one source-only GPU0 screen, expected minutes.
- Day 3: independent receipt audit and, only after a pass, static payload build.
- Days 4-7: local container/minival validation and one untouched EvalAI run.
- Week 2 is reserved only for audit or paper integration, not method expansion.

## 12. Two-sentence pitch

Post-Fusion contains complementary signal, but interpolating identities before
a nonlinear decoder leaves only a small transferable gain. AOF preserves the
native POOLED prediction exactly and asks whether a single source-fitted,
output-space residual coefficient can recover that signal in a static decoder
that is architecturally representable in EvalAI, pending package validation.
