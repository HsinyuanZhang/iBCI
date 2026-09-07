# Path 1 / Path 2 terminal gate audit

**Audited:** 2026-08-04 (Asia/Hong_Kong)  
**Scope:** read-only audit of (1) causal cross-budget T4/AC4 correction (Step 2 and Experiment B) and (2) fixed-K temporal-prototype memory (Step 3). This audit opened no NWB, no formal SUA path, launched no GPU, and changed no shared training source.

## Executive disposition

| Branch / claim | Current evidence status | GPU / decoder disposition | Why |
|---|---|---|---|
| Path 1: generic causal reliability-conditioned T4 correction on current sub-C development scope | **NO-GO** | **No GPU; no decoder integration** | The source-only q_unit + M scalar-error signal did not transfer distinguishably to the six target-free development sessions. It also does not identify a signed/vector correction. |
| Path 1: Experiment-B analytic estimator upgrades at M=30 | **NO-GO, terminal under its frozen protocol** | **No GPU** | EB ridge, second harmonic, and Poisson IRLS each hit the predeclared fail-fast condition; winner is no_winner_no_gpu. |
| Path 1 scientific observation that low-budget direction coefficients are noisy while baseline rate is stable | **GO as a descriptive result only** | Does not authorize an accuracy run | It is reproducible estimator characterization, not decoding R2 or evidence that a correction improves behavior. |
| Path 2: P20/fixed-K temporal prototype as a temporal-order mechanism or K/V-memory substrate | **NO-GO** | **No decoder, GPU, held-out, EvalAI, or quantization** | P20 beat a weak rate control but failed the original slot-control precision gate, then lost to the stronger order-invariant B20 control in all 4/4 source sessions. |
| Path 2: full B20 marginal carrier as a replacement route | **INDETERMINATE scientific attribution; branch stopped** | **No escalation** | B20 is a strong source-only control, but its later component/attachment audit ended b20_component_attribution_indeterminate_stop; it cannot inherit a decoder/GPU claim. |
| Any one remaining Path-1/2 experiment ready for pre-GPU-to-GPU escalation today | **NO-GO** | None | No branch meets its own frozen gate. A new independent-data protocol is required before either family is reopened. |

The important distinction is that both paths contain useful negative/mechanistic evidence, but neither has a held-out-session behaviour-decoding R2 gain. Calling either an already demonstrated BP-free calibration improvement would overstate the result.

## Evidence integrity and scope

The following immutable result artifacts were recomputed locally by SHA-256 during this audit.

| Artifact | SHA-256 | Scope / conclusion |
|---|---|---|
| results/sua_t4_cross_budget_source_audit_v1_20260802/source_audit.json | 42a48b978c4c3a27f35239adcc1c196dc8f3aaadfa7d4234955b96ee0b7c119d | Step-2A CPU-only source plus target-free development descriptor-error audit. |
| results/sua_t4_cross_budget_source_audit_v1_20260802/step2_decision.json | 7ee950cf1eb3ce6c173085e788ac45498b1417171864e37e65397cdd1d893cda | step2b_gpu_pilot=no_go; no post-hoc q expansion or formal test. |
| results/t4_estimator_b_v7_source_only_cpu_audit_v1/aggregate.json | 2909f9f7658cca4fc8d5edeb5eacbd7cb4224bc744e8d97d61d8556b1bdfbcb1 | Experiment B finished CPU-only with no_winner_no_gpu. |
| results/m1_fixed_k_temporal_prototype_gate_a_v3/source_gate_a.json | b2b1e9ff3288ef57fc26d2b446fd803155b1ec5e0ce14dc2f9cc1cbddb92f6dd | Fixed-K Gate A stopped before decoder/GPU. |
| results/m1_fixed_k_temporal_prototype_gate_a2_v1/source_gate_a2.json | 379f3c3b85fe120735802af887d8668fc4d84c3e89cb77acdaabecd485565981 | Stronger B20 control stopped P20 at Stage 0. |
| results/m1_b20_source_characterization_v1/source_characterization.json | adebc8c6619503efbd248d8825b7449b608c4667cc8860908575bd59182114ca | Downstream B20 attribution is indeterminate and terminal. |

The Experiment-B v7 prelaunch source map was checked against the current worktree: all 19 sealed entries match, including the strict 27/6/6 manifest, Step-2A source receipt, audit scripts, estimator modules, and test. The Step-2A source-audit script hash also still matches the hash recorded in its output.

Focused no-NWB contracts were rerun with third-party pytest auto-discovery disabled (the host's unrelated DANDI pytest plugin is incompatible with the installed Click version):

~~~text
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  sua_exploration/tests/test_t4_cross_budget_protocol.py \
  sua_exploration/tests/test_t4_estimator_b_v7.py \
  sua_exploration/tests/test_fixed_k_temporal_prototypes.py \
  sua_exploration/tests/test_m1_fixed_k_prototype_gate_a2.py

31 passed in 5.28s
~~~

These are implementation-contract tests, not a substitute for the already recorded biological session results.

## Path 1 — causal cross-budget T4/AC4 correction

### 1. What was actually tested

The original program had three separable layers:

1. **Step 2A:** a causal low-budget T4 estimate has measurable error, and fields computed from exactly the same first-M calibration trials predict that error across sessions.
2. **Step 2B (never launched):** an offline episodic cross-budget/dropout decoder path would require a positive Step-2A signal and a separate prelaunch.
3. **Experiment B (executed):** analytic estimator upgrades at M=30 could improve prospective source-only deviance and split-half reliability enough to justify, at most, a new GPU prelaunch.

The implemented Step-2A object is ordinary labelled cosine T4 from the chronological rewarded prefix, with descriptor [a, c, m, b], at M in {10, 15, 20, 50}. It uses the exact existing equal-per-direction-mean cosine semantics; a rank-deficient design is undefined, never zero-filled. It holds neural activity calibration fixed at 30 trials and starts any hypothetical decoder scoring at trial 50.

The seven implemented unit-level reliability fields are:

~~~text
log1p(fit residual variance), labelled spike count, labelled exposure,
total-prefix spike count, total-prefix exposure, modulation/residual, rank-valid.
~~~

The actual q_unit + M audit fits a source-only ridge predictor of **scalar squared descriptor error** from those fields plus known M. It is not the proposed vector map T_tilde = T + g(T,q): it neither predicts the signed residual T4@50 - T4@M nor supplies a learned correction vector. It also omits the originally contemplated global direction histogram, direction balance, and design-condition inputs. This is a necessary-signal audit, not evidence that an unimplemented correction MLP failed at behavior decoding.

Experiment B is a second, narrower analytic estimator selection at M=30, with source-only outer LOSO and query trials [30,50) used only for prospective count-deviance scoring. Its three candidates are:

- **EB ridge:** source-fold empirical-Bayes shrinkage of [b,a,c] toward a rotationally symmetric prior with zero directional population mean;
- **second harmonic:** fit cos(2 theta), sin(2 theta) but export/score the first-harmonic descriptor; and
- **Poisson IRLS:** equal-direction-normalised count/exposure likelihood, at most 32 analytic Newton/IRLS iterations.

No candidate is a neural network, a calibration-time backpropagation routine, or an online decoder update.

### 2. Label information and causal boundary

| Object | Labels / rates allowed | What is forbidden |
|---|---|---|
| Ordinary T4@M at deployment | Target directions and neural rates from exactly the chronological first M rewarded trials | Any query label/rate, later calibration trial, hidden T4@50 input. |
| Step-2A reliability fields | The same first-M prefix only | Direction/rate information after M. |
| Source training/LOSO teacher | First 50 trials of the source sessions, used only offline as the T4@50 descriptor-error target | A source teacher used as a held-out deployment feature. |
| Target-free development application | Causal low-M feature and fields before predicting; T4@50 is read only after prediction to score descriptor error | Behaviour decoder score or query-time adaptation. |
| Experiment-B source score | First 30 trial labels/rates fit the candidate; labels/rates in [30,50) are scorer-only count-deviance data | Development/formal sessions, decoder labels, or a deployment update. |

Thus Path 1 is **BP-free at deployment by construction**, but it never obtained a positive deployment correction. The only deployable output that exists from these experiments is an analytic four-float descriptor per unit; there is no accepted g_phi checkpoint, no target gradient state, and no query-side optimizer state.

### 3. Strongest numerical evidence

#### 3.1 The low-label problem is real, but it is not yet correctable

Across the 27 source sessions, equal-session descriptor MSE to the causal T4@50 boundary falls monotonically:

| Label budget | Mean MSE to T4@50 | Median | 80% paired MDE |
|---:|---:|---:|---:|
| 10 | 1.213260 | 0.994957 | 0.418508 |
| 15 | 0.704856 | 0.555380 | 0.269719 |
| 20 | 0.467563 | 0.360430 | 0.180178 |
| 50 | 0 | 0 | 0 |

All 33 opened source/development sessions had a rank-3 finite fit at every audited budget. The split-half finding is also directionally coherent: median flattened (a,c) Pearson rises from about 0.589 (source) / 0.624 (development) at M=10 to 0.860 / 0.874 at M=50, whereas baseline b is already about 0.991–0.993 at M=10 and 0.999 at M=50. This is good evidence that the small-budget uncertainty is mainly directional/amplitude-related rather than baseline-rate uncertainty.

It is **not** a noise-free truth comparison: T4@M and T4@50 share the first M trials. The endpoint measures approach to a causal 50-trial reference, not error to a latent ground truth.

#### 3.2 The unit-reliability increment collapsed on target-free development application

Negative delta favors q_unit + M over the comparator in descriptor-MSE units.

| Evaluation scope | Contrast | Sessions | Mean delta | Wins | 80% paired MDE | Normal 95% interval |
|---|---|---:|---:|---:|---:|---:|
| Nested source LOSO | q_unit+M - M-only | 27 | -0.7671 | 21/27 | 0.7668 | [-1.3035, -0.2307] |
| Nested source LOSO | q_unit+M - constant | 27 | -0.8566 | 22/27 | 0.8225 | [-1.4319, -0.2812] |
| Target-free development application | q_unit+M - M-only | 6 | -0.0226 | 3/6 | 0.4349 | [-0.3268, +0.2817] |
| Target-free development application | q_unit+M - constant | 6 | -0.1157 | 4/6 | 0.5104 | [-0.4726, +0.2413] |

The M-only effect itself was detectable in all 6/6 development sessions (constant - M-only = 0.0931, 95% interval [0.0355, 0.1507]). Therefore the null is not merely an underpowered pipeline that cannot detect any known effect: the incremental unit-reliability signal specifically does not transfer robustly. Removing the single most favorable development session reverses its mean direction.

**Step-2A decision: NO-GO.** This is a negative/inconclusive result for a shared, reliability-conditioned correction on this development boundary, not a proof that an SO(2)-equivariant correction can never work on independent data.

#### 3.3 Experiment B independently does not select an analytic upgrade

The frozen estimator gate required, for all 27 source outer folds, valid prospective deviance, nonnegative split-half directional reliability change, no health deterioration, mean deviance ratio <= 0.98, mean reliability delta >= 0.02, and at least 20 jointly favorable folds. Each candidate reached eight joint failures, making 20/27 mathematically impossible; fail-fast therefore stopped further folds rather than continuing to fish for a winner.

| Candidate | Completed folds before fail-fast | Mean prospective deviance ratio | Mean reliability delta | Health issue | Mean calibration ops | Persistent descriptor state |
|---|---:|---:|---:|---|---:|---:|
| EB ridge | 16 | 0.976601 | +0.000124 | none, but reliability gain is far below +0.02 | 12,294 | 999 B |
| Second harmonic | 9 | 0.994370 | -0.000415 | none | 17,706 | 880 B |
| Poisson IRLS | 8 | 1.027318 | -0.003063 on its one fully defined fold | 2 nonconvergence and 2 invalid-increase folds | 36,987 | 856 B |

These state figures are equal-session means over variable unit counts. They represent a four-FP32-number descriptor (16 N bytes) after calibration, not a decoder cache. The listed temporary workspaces are 5,994 B (EB), 13,200 B (2H), and 21,156 B (Poisson) on average. The cost is trivially compatible with a BP-free calibration flow, but compatibility is not efficacy.

**Experiment-B decision: NO-GO, terminal.** Its aggregate explicitly records winner.status = no_winner_no_gpu; no source or development decoder run was licensed.

### 4. Held-out meaning

Neither Step 2A nor Experiment B supplies a held-out-session behaviour R2 result:

- Step 2A uses 27 source sessions for nested estimator LOSO and six visible development sessions for target-free **descriptor-error** application. The six sealed formal SUA sessions were not resolved or opened.
- Experiment B is 27-source-only outer LOSO, with no development session opened and no decoder.
- A source-LOSO fold is a legitimate estimator generalization check, but it is not a final held-out-session neural-to-behaviour calibration demonstration.

### 5. Only scientifically defensible reopen condition

Do not add a wider q-MLP, a confidence-FiLM retry, new q fields, or decoder training on the same sub-C development sessions. If the family is ever reopened on an **independent subject/data boundary**, the smallest defensible new hypothesis is radial SO(2)-equivariant shrinkage:

~~~text
z = a + i c
s = f_source_only(M, residual, exposure, direction coverage, condition, |z|)
z_corrected = s z
b_corrected = b
~~~

Before any GPU/decoder request it would need a new nested source audit **and** a target-free application that beat ordinary AC4/T4 in both coefficient error and split-half reliability under a predeclared rule. That evidence does not currently exist.

## Path 2 — fixed-K temporal-prototype K/V memory

### 1. What exists versus the proposed memory

The implemented object is P20, a fixed-width static carrier built from raw neural count bins of the first 10 chronological M1 calibration trials:

~~~text
K = 4 global hard-routed slots
r = 4 causal EWMA components, alphas = (0.5, 0.25, 0.125, 0.0625)
per slot output = [count, mean(EWMA_1), ..., mean(EWMA_4)]
P20 width = K x (1+r) = 20 per unit/channel
~~~

The causal filter is reset at every trial boundary. Fold anchors are fitted from the other source sessions only, placed in a deterministic canonical order, and the accumulator keeps no raw trial x bin tensor. The source code rejects query updates after finalization.

This is **not** an implemented K/V cross-attention memory:

- no query/key/value projections were trained;
- no calibration K/V cache is passed to a decoder;
- no frozen SPINT decoder has been pretrained to consume it; and
- no decoder, GPU, behaviour R2, held-out, or EvalAI experiment was run.

A forward-only memory can be BP-free after it is designed and pretrained offline, but inserting new K/V/cross-attention inputs into a frozen decoder is not an adaptation mechanism by itself: the existing decoder has no trained consumer for those coordinates. This is why the static carrier gate was required before paying for a network-side memory path.

### 2. Labels, causal state, and resource receipt

P20 and its B20 comparison carrier use only the first-ten raw neural count bins. They do not use target direction, object identity, velocity, electrode ID, unit ID, or future neural data as a carrier input. In the original Gate A, future object/category information enters only the **offline later-neural oracle target**; it is absent from the deployed carrier. The A2 rerun removed the labelled D4 arm from the carrier/ridge gate entirely.

At N=64, the exact P20 calibration-stream state receipt is:

| Quantity | Value |
|---|---:|
| Streaming state | 1,728 FP32 scalars = 6,912 B |
| Shared anchors | 16 scalars |
| Raw support matrix retained after finalization | 0 elements |
| Per unit per bin | 25 multiplications, 37 additions, 3 comparisons |
| Output width if a trained consumer existed | 20 FP32 values per unit (5,120 B at N=64) |

The state scaling is linear in units and in the fixed K x r carrier size. These counts do **not** include cross-attention Q/K/V projections, per-bin attention MACs, or decoder activation/cache state, because no such reader was instantiated. It would be incorrect to quote a numerical latency or state cost for the proposed K/V extension as if it were measured.

### 3. Gate A: a promising proxy, but not an actionable mechanism result

Gate A used four already-seen M1 held-in-calib source sessions in an outer source-LOSO later-neural oracle proxy. That means anchors/readout exclude the left-out source session, but it is **not** a pristine held-out-session behaviour decoder test. CUDA was forced off and no minival, held-out, formal, EvalAI, or decoder path was resolved.

P20's proxy versus rate-only was large and precise:

| Contrast | Mean delta proxy R2 | Positive sessions | 95% CI | MDE80 |
|---|---:|---:|---:|---:|
| P20 - rate-only | +0.100526 | 4/4 | [+0.090798, +0.110253] | 0.012719 |

The carrier was technically repeatable: across 256 complementary spike-thinning measurements, prototype-value cosine lower 2.5% quantiles were 0.99588–0.99667 and defined-row fraction was at least 0.984375. These thinnings are measurement repeats, not extra biological sessions.

However, the required session-keyed whole-slot shuffle comparison had catastrophic extrapolation in two sessions. P20 minus shuffled carrier was positive by sign in 4/4 sessions but had:

~~~text
mean = +26.836507
95% CI = [-56.427143, +110.100156]
MDE80 = 108.863822
~~~

The predeclared distinguishability criterion failed. Replacing it after seeing the result with a sign test, a robust interval, new shuffle, outlier removal, K/r sweep, or decoder pilot is not a valid rescue.

### 4. Gate A2: the decisive simplicity control rejects temporal order

Gate A2 supplied a matched-width B20 carrier containing only:

~~~text
ten sorted support-trial log-rates
+ ten pooled valid-bin log1p(count) quantiles
~~~

B20 contains no anchor, slot, timestamp, causal filter, or temporal ordering. It beat P20 in all four source sessions:

| Carrier | Per-session later-neural proxy R2 | Mean |
|---|---|---:|
| P20 | 0.880273 / 0.891359 / 0.920905 / 0.758640 | 0.862794 |
| B20 | 0.892449 / 0.942571 / 0.963411 / 0.820599 | 0.904758 |

~~~text
P20 - B20 = -0.041963
95% CI     = [-0.076004, -0.007923]
MDE80      = 0.044507
sign       = negative in 4/4 sessions
~~~

The Stage-0 frozen decision is therefore marginal_baseline_not_beaten_stop. Its exact relative-slot null (13,824 configurations) and time-order null (4,095 schedules) were correctly not run: the predeclared early stop made further computation non-informative. The corresponding M2 P20 source-gate protocol is explicitly unexecuted and non-executable because its M1 Gate-A2 entry condition failed.

This changes the interpretation of the original positive P20-rate-only result: it is evidence that the early neural support contains useful nonlinear marginal rate/count-distribution information, **not** evidence that temporal routing, EWMAs, or fixed slots are the reason.

### 5. B20 does not create a back door to GPU

B20 was examined as a separate lower-state/carrier hypothesis. Its later M1 source row-attachment/component audit is internally valid but stopped at b20_component_attribution_indeterminate_stop:

- breaking the trial-rate block had conditional loss 0.03895–0.08250 R2 in all four sessions;
- breaking the pooled-bin-quantile block had loss only 0.00938–0.01745 R2, below the frozen 0.03 practical threshold in all four;
- yet the Q10 attachment null remained statistically distinguishable, so neither a legal R10 simplification nor a legal full-B20 composite pass occurred.

Moreover, exact B20 pooled-bin quantiles are not free hardware state. Exact computation requires an integer count histogram; the historical hardware receipt estimates about 65.3 KiB peak state for M2 N=96, M=24 and 85.6 KiB for SUA N=137, M=10 with FP32 output. This does not alter the P20 stop, but it is another reason not to label B20 a low-state memory replacement without an independently validated compression.

### 6. Path-2 terminal decision

**NO-GO for P20 K/V memory/cross-attention.** There is no evidence that temporal order survives the stronger marginal control, and no trained decoder interface exists. An independent future temporal-order experiment would first need to beat both B20 and a within-trial time-order null on new data before a decoder/GPU protocol could even be drafted. It is not a currently runnable pre-GPU candidate.

## Final decision for scheduling

Neither Path 1 nor Path 2 should consume either local or remote GPU now. Their current terminal states are useful method conclusions:

1. **Calibration-label lower bound:** low-M T4 directional components are measurably noisy while baseline rate is stable, but the tested shared reliability signal and three analytic estimator upgrades do not produce a transferable correction.
2. **Temporal-memory falsification:** a stable compact P20 proxy exists, but its apparent advantage is better explained by a stronger static marginal distribution carrier; it provides no evidence for fixed temporal slots or K/V attention.
3. **Publication-safe wording:** both results are source/development mechanism evidence. Neither is a claim of held-out-session behaviour R2 improvement, formal SUA confirmation, or a positive BP-free calibration method beyond the already established T4/AC4 substrate.

The active paired SUA/pseudo-MUA C1 program remains the only current GPU candidate outside this scope. Reopening Path 1 or Path 2 on current development data would violate their frozen kill-rule discipline rather than strengthen the paper.

## Primary evidence paths

- [Step-2/Step-3 preparation and written decisions](SUA_STEP2_CROSS_BUDGET_AND_STEP3_PROTOTYPE_PREPARATION.md)
- [Independent Step-2A result audit](SUA_STEP2A_INDEPENDENT_RESULT_AUDIT.md)
- [Experiment-B v7 aggregate](../results/t4_estimator_b_v7_source_only_cpu_audit_v1/aggregate.json)
- [M1 P20 Gate-A result audit](M1_FIXED_K_TEMPORAL_PROTOTYPE_GATE_A_RESULT_AUDIT.md)
- [M1 P20 Gate-A2 result audit](M1_FIXED_K_TEMPORAL_PROTOTYPE_GATE_A2_RESULT_AUDIT.md)
- [B20 component result audit](M1_B20_ROW_ATTACHMENT_RESULT_AUDIT.md)
- [Current work-control board](ACTIVE_EXPERIMENT_CONTROL_BOARD.md)
