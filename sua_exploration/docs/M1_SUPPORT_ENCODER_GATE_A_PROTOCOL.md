# M1 support-set functional encoder Gate-A protocol

**Frozen:** 2026-08-02 (Asia/Hong_Kong)  
**Status:** CPU-only source-session audit authorized by the root reviewer. This protocol does not
authorize decoder training, a new EvalAI submission, a support-budget sweep, or use of hidden
held-out query data.

## 1. Question

The closed-form D4 carrier estimated four categorical calibration rates from the first ten M1
trials but did not improve the sealed local decoder endpoint. This audit asks the narrower question:

> Can a source-session-trained, four-dimensional support encoder denoise those ten-trial estimates
> well enough to predict later neural category rates in a new source session?

This is a carrier-information screen, not a behavior-decoding result. A negative or statistically
unresolvable result stops this M1 representation branch before GPU training.

## 2. Data boundary and estimand

- Dataset: the four FALCON M1 `held-in-calib` source sessions in dandiset `000941`.
- Neural inputs: the same valid-prefix per-trial spike sums and lengths used by production D4.
- Support: chronological trials `[0,10)` only, including their exposed `obj_id` labels.
- Scoring target: per-channel, per-`obj_id` mean neural rates on trials `[210,end)`.
- Outer evaluation: leave one entire source session out; no row from that session may fit or select
  a model.
- Model selection: leave one session out inside the remaining three outer-training sessions.
- Future `obj_id` labels are used only to construct and score the offline future-rate target. They
  are a **later-label oracle diagnostic** and never enter the support encoder or a deployed decoder.

All four outputs retain the fixed M1 category order `(1,2,3,4)`. Trial order is never randomized
for the primary support/query boundary.

## 3. Frozen arms

1. **D4:** the four exposure-corrected first-ten category-rate means.
2. **E4:** a shared four-output ridge map trained only on outer source sessions. Inner LOSO selects
   one of:
   - `means_direct`: `log1p(D4)` input, direct `log1p` future-rate target;
   - `stats_direct`: category means and within-category uncertainty plus support-only global
     rate/trend/exposure statistics, direct target;
   - `stats_residual`: the same statistics, predicting a residual around `log1p(D4)`.
3. **ES4:** a deterministic, session-keyed, complete non-identity row permutation of E4. A fixed
   256-permutation distribution is diagnostic only.
4. **Label-shuffle E4:** one deterministic non-identity permutation of the ten support labels,
   preserving the category counts, passed through the already selected E4 map.
5. **Rate-only:** a separately fitted ridge map whose inputs contain support rate/exposure/trend
   summaries but no condition-to-response attachment.

Ridge values are frozen to
`[0, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]`. Feature standardization and all fitted parameters use
training sessions only. Predictions are transformed back with `expm1` and clipped at zero.

## 4. Metrics, repeatability, and identifiability

Primary per-session error is raw-rate MSE over the `64 x 4` future-rate matrix. `log1p` MSE is a
diagnostic. Report paired session-wise log error ratios and their geometric means for E4 versus
D4, ES4, label-shuffle E4, and rate-only.

For each contrast, report:

- four session ratios and the number below one;
- a two-sided 95% paired Student-t interval on the mean log ratio;
- a fixed-seed session bootstrap interval;
- the multiplicative 95% precision factor
  `exp(t_0.975,3 * sd(log_ratio) / sqrt(4))`;
- the corresponding centered minimum ratio whose upper t bound would reach one. This is an
  identifiability sensitivity quantity, not a prospective power guarantee.

Estimator repeatability uses deterministic binomial thinning of each support spike count into two
half-exposure observations. Across 256 fixed seeds, report flattened Pearson and cosine agreement
for D4 and the outer-fitted E4. No chronological 5/5 split is primary because it can remove category
coverage.

## 5. Precommitted decision

`gpu_gate_pass=true` only if all conditions hold:

1. E4/D4 geometric raw-MSE ratio is at most `0.95`, with E4 better in at least `3/4` sessions;
2. E4/ES4, E4/label-shuffle, and E4/rate-only geometric ratios are each at most `0.90`, with E4
   better in at least `3/4` sessions for each contrast;
3. the upper 95% paired-t ratio bound is below `1.0` for every contrast;
4. every fold is finite, source-disjoint, and uses all four categories in support and target.

If any condition fails, the result is either `ineffective` (point estimates fail) or
`indeterminate_unresolvable` (point estimates pass but uncertainty does not). Neither result
authorizes GPU decoder fusion, a wider encoder, autoencoder/contrastive replacement, latent-width
sweep, or extra use of the same four sessions.

## 6. Artifact contract

The script writes once to `sua_exploration/results/m1_support_encoder_gate_a_v1/` and records:

- input NWB hashes plus hashes of this protocol and the audit script;
- exact folds, selected feature family/lambda, fitted-row counts, and support/target coverage;
- all arm predictions, errors, ratios, controls, repeatability, uncertainty, and gate booleans;
- a strict-JSON `audit.json`, concise `report.md`, and artifact hashes.

The implementation must refuse overwrite and pass `py_compile`, focused deterministic tests, and
`git diff --check`. No GPU device or trainer is started.
