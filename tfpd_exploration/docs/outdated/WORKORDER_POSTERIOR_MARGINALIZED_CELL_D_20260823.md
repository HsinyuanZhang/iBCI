# Work Order: Posterior-Marginalized Cell-D

Date: 2026-08-23 HKT

## 0. Immediate plan

Build and run exactly one performance-oriented source-training cell:

```text
POSTERIOR_MARGINALIZED_CELL_D_SEED42
```

Use the first genuinely idle compatible GPU selected by the root-reviewed
compatible-device profile at launch time.  The selected profile may be GPU0
or GPU1, but must bind the exact visible-device mapping, UUID, BDF, memory,
and Torch runtime in the immutable launch receipt.  Do not stop, migrate, or
modify another process.  Do not launch a sweep or a second posterior arm.

The sequence is:

1. additive implementation and CPU/synthetic tests;
2. source-only real-authority audit and posterior-cache calibration report;
3. one-batch GPU smoke on the idle GPU;
4. root audit;
5. one 48-epoch full source training run;
6. final-four SWA;
7. matched M30/M10/M4 within/external evaluation with deterministic point carriers.

Performance is primary. Ablations are deferred unless the single cell produces a material positive result.

## 1. Question

Can source-side marginalization over closed-form carrier uncertainty improve the transfer robustness of the successful Cell-D decoder without exposing posterior precision to the inference graph?

The previous full Posterior Carrier bundle failed badly, and PIRG showed that a static posterior-credibility gate is a learned no-op. This cell tests the one materially different remaining hypothesis: posterior uncertainty may be useful as a **training distribution**, even though it is not useful as an inference token or gate.

## 2. Two-sentence contribution pitch

Whole-unit dropout marginalizes whether a unit is present, but it treats every surviving analytic carrier as exact. We extend source-only session marginalization to carrier estimation error by sampling closed-form posterior carriers during training, while deployment remains the original deterministic, gradient-free Cell-D decoder with no additional parameters or uncertainty input.

## 3. Baseline and sole intervention

### 3.1 Held Cell-D system

Hold all of the following exactly:

- Cell-D model graph and all parameter shapes;
- initialized trainable parameter count `3,510,842` plus the exact two known uninitialized lazy keys;
- B3S M30 calibration-activity encoder;
- fused activity plus identity token topology;
- two attention heads;
- dynamic whole-unit dropout with per-step `p ~ U(0,1)` over the complete fused token;
- strict-27 source roster;
- equal-session B32 schedule, `33,925` steps per epoch;
- seed 42 canonical initial state;
- 48 epochs and `1,628,400` total optimizer steps;
- Adam, `lr=1e-4`, no weight decay, and the frozen Cell-D schedule;
- dense valid-bin supervised MSE;
- final-four checkpoints 44, 45, 46, 47 and ordered SWA;
- behavior normalizer;
- ordinary sealed OLS T4 normalizer;
- no target-session optimizer, backward, update, or label expansion.

The cell must strict-load the same canonical initial state used by the accepted equal-session Cell-D route. It must not initialize from the final Cell-D SWA because that would add extra source training budget and break attribution.

### 3.2 Only changed factor

Replace the source-training side tensor for each session and logical epoch with a sampled carrier drawn from the already implemented closed-form posterior:

```text
budget(epoch, session_index) = (4, 10, 30)[(epoch + session_index) mod 3]
raw_sample = sample from closed-form posterior(session, budget, epoch, seed=42)
training_side = sealed_OLS_normalizer(raw_sample)
```

The posterior fit and sample must be computed at a session-by-logical-epoch boundary and cached. No inverse, refit, posterior sampling, or normalization-statistic fitting may occur inside the optimizer batch loop.

The batch's neural window, target, valid mask, B3S calibration activity, and session schedule remain unchanged. Only the normalized T4 side tensor supplied to B3S changes from the ordinary deterministic point carrier to the cached posterior sample.

## 4. Critical exclusions

This cell must not include:

- posterior mean as the deployment carrier;
- a posterior-derived normalizer;
- empirical-Bayes shrinkage as a separate inference treatment;
- credibility token, PIRG gate, attention-logit bias, temperature, or per-head bias;
- a new decoder, recurrent head, SSM, RoPE, new query slots, or width/head change;
- a new activity calibration prefix;
- M4/M10 B3S calibration activity;
- target-session training or adaptation;
- teacher weights, distillation, unit tables, session tables, or subject metadata;
- AMP, TF32, `torch.compile`, or seed43's accelerated TF-SR forward path;
- a hyperparameter sweep.

The full Posterior bundle's posterior normalizer must not silently leak into this cell. Samples are normalized only with the sealed ordinary OLS T4 normalizer used by Cell D.

## 5. Training-time sampling semantics

Use the existing audited posterior implementation wherever its semantics match:

- closed-form 3x3 posterior fit;
- local domain-separated generator;
- host Python/NumPy/Torch RNG invariance;
- exact M4/M10/M30 prefix identities;
- raw `beta -> [a,c,m,b]` conversion;
- deterministic session x epoch sample identity;
- source-only cache construction.

Sampling must preserve rotation equivariance of the `[a,c]` pair. The magnitude channel is recomputed from the sampled pair. Do not independently sample or shrink direction and magnitude in inconsistent coordinate systems.

For every session and epoch, receipt evidence must bind:

- selected budget;
- posterior fit digest;
- local sampling seed digest;
- raw sampled carrier digest;
- sealed-normalized sampled carrier digest;
- finite/min/max/quantile statistics;
- source roster and unit-order digest;
- cache-build counter;
- proof that the batch loop performed zero inverse/refit/sample operations.

## 6. Source-only calibration and safety audit

Before CUDA launch, produce a CPU/source-only audit. It is not a model result and must not open within, external, target, formal, or H1 surfaces.

Required checks:

1. Exact M4/M10/M30 prefixes and unit ordering match the sealed posterior authority.
2. Posterior covariance is finite and positive semidefinite within the existing numeric tolerance.
3. Aggregate directional uncertainty contracts from M4 to M10 to M30; report violations by session and unit rather than hiding them in a mean.
4. Compare M4/M10 posterior precision with error to the same-session ordinary M30 OLS carrier as a source-side proxy. Report per-session Spearman correlation, median correlation, sign count, and raw rows. This is descriptive because M30 OLS is not ground truth.
5. Compare sampled raw/normalized carrier quantiles with ordinary Cell-D source T4. Fail if non-finite or if the predeclared raw absolute-value safety bound **1000.0** is exceeded; do not choose clipping thresholds post hoc or clip carriers.
6. Repeated cache construction is bitwise deterministic and leaves host RNG unchanged.

Only structural invalidity, non-finite values, failed covariance contraction at the aggregate level, broken unit alignment, or extreme unbounded samples should block the GPU smoke. Weak error-ranking correlation must be disclosed but does not silently change the design after the operator explicitly authorized this one performance cell.

## 7. Stage-0 and GPU smoke gates

### 7.1 Synthetic/CPU gates

- At zero posterior covariance, the sampled-side forward is bitwise identical to ordinary Cell D.
- The model graph, state keys, initialized parameter count, lazy keys, optimizer, loss, dropout law, and inference signature are unchanged.
- A nonzero covariance fixture changes only `side_features`, not neural, calibration activity, target, valid mask, or model state before optimization.
- Joint permutation of neural units, calibration units, posterior rows, and carrier rows preserves model equivariance within the existing FP32 tolerance.
- The posterior cache performs all fits/samples before the DataLoader iterator and no batch-loop inverse/refit/sample call is possible.
- M4/M10/M30 exposure counts are exact over the 48 x 27 epoch-session grid: each budget is used equally often.

### 7.2 One-batch source-only GPU smoke

Run on the selected idle GPU only after root review. Require:

- exact physical GPU identity and one visible device;
- B32 forward/loss/backward/update finite;
- all Cell-D critical gradient groups present;
- exact dynamic whole-unit-dropout call topology;
- only source data opened;
- zero target/formal/within/external resolution;
- no posterior operation inside the timed optimizer step;
- peak allocated/reserved memory and step time recorded;
- attempt, source authority, smoke, terminal or honest failure receipts immutable.

The smoke result root and full result root must be separate and fresh.

## 8. Full training

After smoke acceptance, launch one full seed-42 run on the first idle compatible GPU.

The full route must reuse the accepted equal-session lifecycle semantics:

- exactly one explicit epoch ledger claim;
- exactly one DataLoader iterator per epoch;
- complete 33,925-step epoch or honest terminal failure;
- epoch-boundary full model/optimizer/gradient proof, not per-step full scans;
- immutable epoch receipts;
- final-four checkpoints and ordered inherited Cell-D SWA (FP64 accumulation, cast to model dtype);
- launch closure equals final closure;
- no target-facing code in the source-training closure.

Report per epoch:

- mean/min/max source loss;
- steps/s and wall time;
- selected budget counts;
- posterior-cache evidence;
- sample distribution statistics;
- dropout statistics;
- model/optimizer finite proof;
- exact critical gradients;
- learning-rate endpoints;
- memory peaks;
- state digests.

Do not early-select a checkpoint from within/external behavior. The deployable artifact is the predeclared final-four SWA.

## 9. Matched evaluation

Inference must use deterministic ordinary OLS point carriers. Posterior samples, covariance, credibility, PIRG alpha, and posterior normalizer are forbidden at inference.

Evaluate:

- within and external;
- M30, M10, and M4 ordinary point-carrier budgets;
- the same last-bin, variance-weighted, equal-session metric;
- zero target optimizer/backward/update;
- sealed Cell D and the successor on identical materialized inputs;
- paired per-session deltas, sign counts, median, and deterministic bootstrap interval.

Primary performance questions:

1. M30 safety: does the successor avoid harming the governing M30 external score?
2. Short-prefix benefit: does M4/M10 degrade more gracefully than sealed Cell D?
3. Breadth: are gains distributed across sessions rather than driven by one date?

Predeclared practical interpretation:

- **Strong positive:** external M30 delta at least `0`, pooled M4 delta at least `+0.03`, and at least `9/15` external sessions positive at M4.
- **Short-prefix-only positive:** external M30 delta no worse than `-0.02`, pooled M4 delta at least `+0.03`, and breadth passes. This supports a limited-label robustness claim, not an M30 headline.
- **Null:** all key deltas remain in approximately `[-0.01,+0.01]`; stop the posterior route.
- **Negative:** M30 external delta below `-0.02` or M4 fails to improve; stop the posterior route.

Diagnostics cannot rescue a failed primary score.

## 10. Expected value and strongest objection

Expected effect is concentrated at M4/M10. M30 improvement is expected to be small or null because the point carrier is already relatively stable there.

Strongest objection: the full Posterior model already used sampling and failed, so this repeats a dead idea.

Response: the full bundle simultaneously changed estimator, normalizer, representation, and consumer. This cell keeps the successful Cell-D inference system and sealed OLS normalizer, changing only the source carrier distribution. It is the minimal experiment that can attribute a result to parameter-uncertainty marginalization rather than to a new posterior consumer.

## 11. Ownership and launch authority

Terra owns only new additive implementation, CLI, and focused tests for this cell. Terra is not alone in the codebase and must not revert or modify other agents' changes. Existing Cell-D, Posterior, PIRG, TF-SR, result, checkpoint, and authority files are immutable dependencies.

Terra must stop at a no-data/no-CUDA review boundary. Root independently audits code, closures, source-only authority, GPU availability, and the smoke/full result roots. The user's instruction in this turn authorizes using an idle GPU for this experiment after those gates pass; it does not authorize interrupting another job, overwriting a root, opening target/formal data during source training, or launching an unreviewed implementation.
