# Posterior Carrier / Distributional Identity — performance-first handoff

Date: 2026-08-22

## Immediate plan

1. Build one additive headline system on the sealed Cell-D family. Do not edit Cell D, TF-SR,
   AM/IM, shared SPINT modules, existing results, or running jobs.
2. Train `POSTERIOR_CARRIER_BUDGETMIX_D_SEED42` first on the remote RTX 5070 Ti. This is a
   performance-oriented system swap, not a component-attribution experiment.
3. Re-score the sealed Cell-D point-carrier checkpoint and the new checkpoint at labelled-carrier
   budgets M30, M10, and M4 with the same last-bin, variance-weighted, equal-session estimator.
4. Run precision/shrinkage/sampling ablations only if the headline system improves the short-prefix
   degradation curve without materially damaging M30.
5. Treat H1 as a separate second-stage route. H1 is 7-DoF and uses a different estimator, decoder,
   and data authority; it cannot be presented as another score from the sub-M model.

## Decision

GO for additive implementation, source-only audits, and a reviewed remote source-training launch.
NO GO yet for target/external/H1 access or scientific scoring.

The hypothesis is worth testing, but the original brief must not be implemented literally. Two
earlier results are binding:

- confidence-FiLM exposed two confidence scalars to a consumer and gained only about +0.0034 R2;
- the frozen H1 M4 generic empirical-Bayes shrinkage program failed its terminal stability gate.

Therefore the new claim is not "confidence as two more features" and not "generic EB denoising".
The still-open hypothesis is that a session should be represented as a distribution over functional
unit identities, and that the decoder should marginalize both unit existence and carrier estimation
uncertainty during source training.

## Frozen first system

### Substrate held from Cell D

- B3S activity identity is retained.
- The activity-only B3S prefix remains M30. M4/M10/M30 below refer only to behaviour-labelled
  carrier fitting. This isolates label cost from neural observation cost.
- The fused activity-plus-identity unit token remains the atomic element.
- Cell-D whole-unit dynamic dropout U(0,1) is retained exactly and is shared across the complete
  unit token.
- Decoder width, heads, layers, queries, loss, optimizer, learning-rate schedule, batch size,
  source roster, seed 42, 48 epochs, checkpoint rule, and final-four SWA are held. Behaviour and
  neural normalization semantics are held, but the posterior carrier uses the source-only
  posterior normalizer specified below; reusing point-carrier moments for a changed carrier
  distribution is forbidden.
- No unit table, session table, subject ID, teacher checkpoint, target gradient, or target-derived
  hyperparameter is allowed.

### Label-budget schedule

Each source session uses one carrier budget for an entire logical epoch. For session index `j` and
epoch `e`:

```text
budgets = [4, 10, 30]
M(e,j) = budgets[(e + j) mod 3]
```

Across 48 epochs every source session therefore receives exactly 16 epochs at each budget. The
budget is session-consistent, never resampled per window or batch. The schedule is identical for
all later control arms.

### Closed-form posterior

For unit `i`, construct the labelled prefix rate vector `y_i` and the shared directional design:

```text
X[t] = [cos(theta_t), sin(theta_t), 1]
beta_i = [a_i, c_i, b_i]
```

Fit a conjugate Gaussian posterior with a source-only population prior. Build the prior once from
all ordinary raw M30 T4 rows in the strict source-27 roster:

```text
source_mean_b = mean_i(b_i)
tau_ac^2      = max(mean_i(a_i^2 + c_i^2) / 2, 1e-12)
tau_b^2       = max(mean_i((b_i - source_mean_b)^2), 1e-12)
```

The likelihood uses the actual integer spike count `k_it` and exposure duration `d_t` of each
labelled prefix trial:

```text
y_it = k_it / d_t
Var(y_it) approximately max(k_it, 1) / d_t^2
W_i[t,t] = d_t^2 / max(k_it, 1)
```

The `max(k,1)` rule is literal and must not be tuned. Do not reconstruct integer counts by rounding
floating rates when the source file exposes the counts directly.

```text
prior mean       mu0 = [0, 0, source_mean_b]
prior covariance S0  = diag(tau_ac^2, tau_ac^2, tau_b^2)
observation precision W_i = diagonal Poisson-rate approximation from prefix counts/exposure

S_i  = inverse(inverse(S0) + transpose(X) W_i X)
mu_i = S_i (inverse(S0) mu0 + transpose(X) W_i y_i)
```

This uses one 3x3 inverse per unit. There is no GCV grid, marginal-likelihood optimizer, target
tuning, or repeated inverse search.

The source prior over `[a,c]` is zero-mean and isotropic. It has no preferred direction and is
SO(2)-equivariant: rotating the task coordinates rotates the posterior direction by the same amount.
It shrinks uncertain directional magnitude toward zero but never toward a source-population angle.
The baseline coordinate may shrink toward the source-only mean.

An all-zero-count unit is not allowed to inherit a fictitious active source identity: its emitted
posterior-mean T4 is the exact all-zero vector, while its covariance/credibility evidence is retained
for audit and its credibility remains at the clamped minimum. This preserves the current T4
zero-spike semantics.

The deterministic posterior carrier is:

```text
T4_post = [mu_a, mu_c, sqrt(mu_a^2 + mu_c^2), mu_b]
```

Raw posterior construction occurs before normalization. Posterior samples are also constructed in
raw coordinates and converted to T4 before normalization.

### Posterior-specific source-only normalizer

Do not reuse the ordinary point-T4 normalizer. Its magnitude coordinate has a large positive raw
mean, so a low-evidence posterior magnitude near zero becomes a large negative standardized feature
rather than a neutral low-evidence observation. That would confound posterior uncertainty with a
normalization-domain shift.

Build one immutable posterior normalizer using only the strict source-27 training roster. For every
source session, construct deterministic posterior-mean T4 rows at M4, M10, and M30. Give each
budget equal weight, concatenate only those source rows, and compute float64 population moments in
the frozen row order:

```text
posterior_normalizer_mean = mean(T4_post_source_M4_M10_M30, axis=rows)
posterior_normalizer_std  = population_std(..., ddof=0)
```

Every standard deviation must be finite and strictly positive. Freeze the exact mean, standard
deviation, row count, per-budget counts, roster/order digest, raw-row digest, and normalizer body
SHA before a GPU smoke. Training samples and evaluation posterior means both reuse this same
source-only normalizer. No target, validation, external, formal, or H1 row may contribute.

This is part of the headline system swap, not an attributed normalizer ablation. A later matched
point-budget-mix control must use its own source-only point-carrier moments and must not borrow
posterior moments.

For an all-zero-count unit, the raw deterministic posterior-mean T4 remains exact all-zero and its
credibility remains at the clamped minimum. Its standardized value is recorded honestly; it is not
secretly clamped to normalized zero. The credibility bias is the mechanism that suppresses this
low-evidence unit.

### Parameter-free credibility bias

Do not append confidence scalars to B3S and do not add a confidence MLP. Compute a normalized,
rotation-invariant variance-reduction score from the `[a,c]` block only:

```text
r_i = 1 - trace(S_i[a,c ; a,c]) / (2 tau_ac^2)
r_i = clamp(r_i, 1e-6, 1)
attention_bias_i = log(r_i)
```

Do not include baseline `b` in this credibility score. Baseline rate is already strongly represented
by the activity-derived B3S path and would make the bias mostly prefer high-rate units rather than
units with well-determined tuning. The posterior may still shrink `b`; `b` simply does not control
the attention mass.

The bias is added to the existing unit-axis cross-attention logits and repeated across the existing
heads and coordinate queries. This is equivalent to multiplying each unit's pre-softmax attention
mass by its credibility. If all units have equal credibility, the bias cancels exactly under the
softmax. It adds zero learned parameters.

Cell-D dropout semantics must remain unchanged. Dropped units remain Cell-D zero placeholders with
the existing inverse-probability gain; this experiment must not silently replace Cell D with true
token removal.

### Posterior sampling during source training

At the start of each source session's logical epoch, draw exactly one session-static sample per unit:

```text
beta_sample_i ~ Normal(mu_i, S_i)
```

All windows from that session in that epoch share the same sampled carrier. Evaluation uses the
posterior mean and never samples. Sampling uses a route-local, domain-separated generator keyed by
`cell / seed / session / epoch`; it must not mutate global Python, NumPy, Torch, or CUDA RNG state.

Whole-unit dropout marginalizes unit existence; posterior sampling marginalizes uncertainty in the
functional parameters of the surviving units. This is the intended paper-level system claim if the
performance result is positive.

## Baseline and first-round comparison

The first GPU cell is the full posterior system above. It is compared with the already sealed Cell-D
point-carrier checkpoint, re-evaluated using OLS point carriers at M30/M10/M4. This is an honest
system comparison, not a one-factor causal attribution.

Do not spend the first round on a nine-cell 3-budget x 3-arm training matrix. If the headline cell
is positive, the follow-up order is:

1. posterior mean + credibility bias, without posterior sampling;
2. posterior mean alone, without credibility bias or sampling;
3. matched point-carrier budget-mix training.

These successors isolate sampling, graded trust, and robust budget exposure. They are not authorized
until the headline result exists.

## Stage-0 gates before any remote GPU launch

The additive implementation must prove on CPU/synthetic data:

1. exact permutation equivariance over units, including carrier, covariance, credibility, and
   activity rows;
2. exact SO(2) rotation equivariance of posterior `[a,c]` and no source directional mean;
3. symmetric positive-definite posterior covariance and finite outputs at M4/M10/M30;
4. exactly one 3x3 inverse per unit and no lambda/grid optimizer;
5. credibility in `(0,1]`, equal-credibility attention cancellation, and monotonic down-weighting of
   a less precise otherwise-identical unit;
6. session-static sampling, exact replay, and no mutation of host RNGs;
7. posterior sampling disabled in eval mode;
8. posterior conversion and sampling occur before the immutable posterior-specific source-only T4
   normalizer; its M4/M10/M30 inputs are equally weighted and contain no non-source row;
9. M30 neural-only B3S activity is unchanged while only labelled carrier budget varies;
10. Cell-D parameter count is unchanged and its dropout stream/law is unchanged;
11. no target, validation, external, formal, H1, checkpoint, or CUDA path is reachable from the dry
    route;
12. source-only prior and per-session posterior receipts bind exact source roster, prefix rows,
    design rank/condition, prior moments, posterior digests, budget schedule, and code closure.

After the CPU gates, run one reviewed remote source-only smoke of 100 optimizer steps. The smoke must
show finite loss, finite full model/Adam state, nonzero gradients in B3S, T4 fusion, attention, and
output paths, exact GPU identity, peak memory, and no target surface access.

## Remote RTX 5070 Ti contract

Remote host: `xinyuan@100.103.97.12`.

The current remote Torch authority is:

```text
torch                  2.13.0+cu130
torch CUDA             13.0
cuDNN                   92000
visible devices        1
logical device         cuda:0
name                    NVIDIA GeForce RTX 5070 Ti Laptop GPU
capability              12.0
total memory bytes      12346195968
```

`nvidia-smi` is currently unusable because the host has an NVML driver/library version mismatch.
Do not fabricate or infer an nvidia-smi UUID/BDF/memory receipt. The reviewed successor may use an
explicit Torch-only device authority and must record `nvml_status=UNAVAILABLE_DRIVER_LIBRARY_MISMATCH`.
Any change in the exact Torch authority above is a fail-closed stop. Do not reboot or repair the
remote driver without separate operator authorization.

Create a new remote staging root. Do not edit the remote historical repository in place. Upload an
explicit file closure plus required immutable source authorities. Verify every uploaded SHA before
the smoke or training process starts. Run under a fresh tmux session and write progress receipts at
epoch boundaries. No target/external/H1 data may be copied into or opened by the source-training
stage.

## Scoring and decision rule

Only after the new source-training terminal and SWA artifacts pass independent audit may a separate
scorer open the existing paired-view within-6 and sub-M external-15 authorities.

Report for M30/M10/M4:

- last-bin, variance-weighted R2 per session;
- equal-session mean and median;
- paired posterior-minus-point delta per session;
- positive-session count and paired bootstrap interval;
- degradation from M30 to M10 and M4 for both systems;
- aligned, zero, and complete cyclic wrong-pair controls;
- target optimizer/backward/update counts, all exactly zero.

Performance-first gate:

```text
M30 safety: posterior - point >= -0.02 mean R2
M4 headline: posterior - point >= +0.03 mean R2
M4 breadth: at least 9/15 external sessions positive
curve: posterior M30-to-M4 degradation is smaller than point degradation
anti-triviality: posterior aligned remains above both zero and wrong-pair controls
```

If M30 safety fails, stop. If the M4 headline and curve gates pass, proceed to the ordered
ablations. Values between the stop and pass boundaries are HOLD, not a claimed breakthrough.

## H1 boundary

H1 is not part of the first sub-M GPU cell. The historical H1 M4 EB shrinkage route and sparse H1
mainline are terminal under their own protocols, so a new H1 experiment requires a fresh route and
must not reuse those old gates as if they had passed.

If the sub-M headline passes, the H1 successor may test the same distributional principle using the
H1-specific 7-DoF design and consumer. It must first pass a new CPU date-LODO posterior calibration
gate and must compare point versus posterior consumers with matched retraining. No H1 claim can be
made by simply applying the sub-M checkpoint or by improving a carrier proxy without decoder gain.

## Contribution sentence if successful

Use only after the performance gates pass:

> We represent each calibration-derived unit identity as a closed-form posterior rather than a
> point token, and train the set decoder to marginalize both missing units and uncertain functional
> parameters. This preserves zero-target-gradient deployment while degrading more gracefully as
> labelled calibration shrinks.

Do not claim that empirical Bayes, confidence inputs, or Gaussian augmentation are individually
novel. The contribution is the closed-form distributional unit-identity interface, its
parameter-free credibility-weighted set read-in, and the matched short-prefix transfer result.
