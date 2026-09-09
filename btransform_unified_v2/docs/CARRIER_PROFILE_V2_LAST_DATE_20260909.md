# Carrier profile v2: M1 and H1, last-date transfer

Status: the complete seed-42 outer round is audited and reported. Both inner final audits passed and both profile selections are `PROMOTED_AND_LOCKED`. Outer M1 fixed epoch 24 is negative for muscle D; outer H1 fixed epoch 32 improves the equal-session mean for state D. Historical `chronological_last2_v1` results remain immutable and are not relabelled as this experiment.

## Question and endpoint

The primary question is whether a more reproducible, behavior-related per-unit profile increases fixed-endpoint `D - B`. The secondary comparison is new D versus the original carrier D under the same new source roster. A favorable change caused only by adding a source date is not evidence for the new profile.

The outer M1 split trains on `2012-09-24/26/27` and scores `2012-09-28`; the outer H1 split trains on all eleven sessions through `1925-01-19` and scores both sessions on `1925-01-20`. M1 retains the first ten native trials for support, source train `[10,310)`, source validation `[310,end)`, and target query `[10,210)`. H1 retains its first three available eval-valid native trials for support, source train `[3:-2]` at stride 4, last two source trials for validation, and all post-support target trials at stride 1.

The primary endpoints remain fixed EMA epoch 24 (M1) and 32 (H1), seed 42. Source-validation-selected checkpoints are secondary diagnostics. H1 averages session R2 within the final date. No target optimizer update is allowed. This is public-data development transfer: these final dates have appeared in earlier experiments, so this iteration does not create an independent unseen test set.

Candidate selection used earlier dates only. The bounded decoder pilots used M1 `09/24,09/26 -> 09/27` and H1 `<=01/15 -> 01/19`. Original B/D checkpoints were reused for the inner comparison only after their exact source roster, data arrays, initialization, training recipe, and endpoints were checked. The completed inner audits promoted the profiles below. The outer B, original D, and selected new D fits for both datasets have completed their source gates, target scoring, and final audits.

## Observations that motivate the redesign

M1 rSyn3 fits unit firing rate from three source-defined NNMF EMG synergy activations plus an intercept. It has no explicit reliability, gain normalization, or temporal tuning. At the old two-date holdout, its fixed-24 contribution was approximately `-0.03741`; the per-date contributions were approximately `+0.00057` on 09/27 and `-0.07539` on 09/28. The already-tested aligned, multiscale rSyn3 projection plus a frozen-B prediction-conditioned residual head produced tiny mixed deltas on three completed historical folds. That residual design is not a new candidate here.

H1 H-C fits a velocity decoder using twelve source neural principal components, projects the decoder coefficients back to unit rows, compresses seven behavioral coefficients to four axes, and applies empirical-Bayes shrinkage. A unit row therefore depends on the population covariance and the chosen neural subspace. The new question is whether direct behavioral encoding or conditional responses provide a better unit profile.

FALCON describes M1 as 16-channel EMG during reach and grasp; H1 labels contain three translation, one rotation, and three grasp-shape velocities, collected during attempted, cued open-loop movements. H1 movements have translation, orientation, and grasp phases. These facts motivate task-specific behavioral coordinates; they do not establish that either new estimator works. Sources: [official dataset descriptions](https://snel-repo.github.io/falcon/datasets.html) and [FALCON benchmark paper, dataset sections](https://proceedings.neurips.cc/paper_files/paper/2024/file/8c2e6bb15be1894b8fb4e0f9bcad1739-Paper-Datasets_and_Benchmarks_Track.pdf).

The root-executed initial H1 diagnostic used only the eleven outer source sessions. Their first three support trials contain 434–543 usable 100 ms blocks. All seven standardized velocity axes have numerical rank seven; the smallest-to-largest singular-value ratio ranges from about 0.273 to 0.497. Thus missing axes or outright rank deficiency is not supported as the primary explanation. Translation and grasp velocity RMS values differ substantially. Detailed evidence is in `results/carrier_profile_v2/h1_support_coverage.json`; future profile diagnostics must check reproducibility and effective behavioral coverage rather than assume that three trials are necessarily inadequate.

## Candidate inventory and screening

These are hypotheses, not positive decoder results. The first M1 families have completed source-only diagnostics; the global-RMS muscle profile and the H1 signed-state profile are the bounded inner-pilot choices.

| Dataset | Candidate | Testable reason | Main objection / disposition |
|---|---|---|---|
| M1 | Standardized, reliability-shrunk synergy encoding | Separate modulation shape from firing gain and uncertain slopes | Weaker source-only reproducibility than old rSyn3; negative diagnostic evidence |
| M1 | Soft behavioral prototype response | Represent nonlinear EMG-state selectivity | Ten trials poorly cover prototypes and stability was weaker than old rSyn3; negative diagnostic evidence |
| M1 | Muscle-response16, source-SVD4, global RMS | Preserve observed 16-muscle recruitment response before four-axis compression; avoid whitening weak SVD axes | Source-only stability improvement is mechanistic evidence, not a decoder result; selected for bounded inner fixed-24 pilot |
| M1 | Lagged synergy kernel | Synchronous regression may miss lead/lag | More parameters and alignment choices; deferred |
| M1 | Within-session population-relative ranks | Reduce common gain drift | Loses absolute modulation and may be a session shortcut; deferred |
| M1 | Direction, strength, and reliability decomposition | Separate weak tuning from unstable direction | Requires extra dimensions or discarding other information; deferred |
| M1 | Behavior by population-state interaction | Distinguish state-dependent gain | Weak support and overlaps the B activity pathway; deferred |
| H1 | Standardized per-unit seven-axis velocity encoding, compressed to four | Remove dependence on a neural population decoder basis | Linear tuning and four-axis compression may lose information; shortlisted |
| H1 | Signed movement-state conditional response, compressed to four | Estimate positive/negative movement preferences directly | Occupancy and phase correlations can bias means; selected for bounded inner fixed-32 pilot |
| H1 | Translation/orientation/grasp phase means plus baseline | Use known task structure | Loses within-phase direction; rejected for first round |
| H1 | Lagged axis tuning | Attempted movement may lag the cue | Cue-alignment choices and three-trial noise; deferred |
| H1 | Local nonlinear axis tuning curves | Permit saturation and asymmetric responses | Too many bins for support; signed-state version is the bounded simplification |
| H1 | Unitwise behavior canonical correlation | Emphasize behavior-predictive components | Population dependence recreates part of the old concern; deferred |
| Both | Shared behavior basis with day-specific neural alignment | Exploit stable latent relationships | Changes the model and training objective together with the carrier; deferred |
| Both | Reliability-based learned admission or residual fusion | Downweight unreliable support | Confounds profile and fusion; historical residual failures weaken its priority |

The first round holds the carrier shape at `[units,4]` and retains the original two D admission paths (identity-encoder side information and direct frontend carrier). B always ignores the carrier. New profile dimensions or a new residual head are not part of this round.

## Source-only diagnostic decisions

For M1, fit the EMG basis and behavioral prototypes only on the earlier source prefix. Compare first-ten-trial split-half profiles and first-ten versus next-ten profiles on inner dates. For H1, fit behavioral scaling and the four-axis projection only on the earlier source prefix, then compare first-three versus next-three trial profiles and individual-trial fits on inner dates. Additional source trials are used to evaluate profile reproducibility, not to increase deployment support.

Each candidate must report finite values, per-column amplitude, held-date normalized excursions, behavioral occupancy, reproducibility, and sensitivity to within-trial circular shifts of labels. M1 behavioral slopes are inspected separately from the baseline column. A stable constant or an all-zero shrinkage output does not qualify as a reliable behavioral profile. Compression axes have no intrinsic physical labels and their source-only provenance must be recorded.

The initial diagnostics decide which candidates deserve decoder pilots. A candidate with degenerate support, absent behavioral signal, or large unexplained excursions is rejected before a full fit. Improvements in profile reliability are mechanistic evidence only; the later inner-date fixed-endpoint decoder comparison selects the final candidate. Selection cannot use the outer final-date score. If neither candidate improves inner `D - B`, report that failure explicitly and choose any subsequent iteration using source-only evidence, without treating a positive final score as the selection rule.

## Execution and evidence

All numerical diagnostics, training, scoring, and statistical aggregation are executed by root. Subagents implement and statically inspect isolated modules. New scripts, results, checkpoints, and provenance live under new version directories. The prior paper and figure work remains paused.

The training comparison must bind source arrays, profile packs and source fitting authority, common initialization, sample order, dropout masks, optimizer recipe, checkpoint hashes, and target endpoint arrays. It must verify exact B carrier insensitivity and nonzero D sensitivity. Runs use `PYTHONNOUSERSITE=1` with the existing `spint` environment to avoid the incompatible user-site Torch package. Long-running jobs follow the user's ten-minute monitoring interval.

The bounded inner audits, profile locks, outer source gates, target scoring, and final audits are complete. Consolidated fixed and selected summaries are available in the linked report. No plots are requested in this iteration.


## First source-only diagnostic results

Root executed the M1 rolling diagnostic and H1 same-fold old/new diagnostic before any new decoder training. All values below concern calibration-profile reproducibility, not decoding R2.

| M1 profile | Split-half behavioral Pearson, hold 09/26 | Split-half behavioral Pearson, hold 09/27 | First10-next10 behavioral Pearson, hold 09/26 | First10-next10 behavioral Pearson, hold 09/27 |
|---|---:|---:|---:|---:|
| Original rSyn3 | 0.8035 | 0.7574 | 0.8840 | 0.9111 |
| Standardized rSyn3 | 0.6306 | 0.6305 | 0.7790 | 0.7712 |
| Soft prototype4 | 0.5108 | 0.5572 | 0.7676 | 0.7497 |

The proposed stability improvement is not supported for either initial M1 candidate. Prototype profiles reduce the largest normalized excursions (about 2.81 versus 11.08/4.56 for the original), but that alone does not establish a better behavioral identity. The 09/27 prototype has a minimum effective occupancy of about 10 bins, comparable to its 10-bin pseudocount. These candidates are retained as negative diagnostic evidence.

The preregistered second M1 construction was then evaluated without opening 09/28. It estimates a descriptive `muscle_response16` profile: each of the 16 observed EMG channels is rectified and divided by its source `[0,310)` RMS; for each unit, its M10 rate is centered and divided by the Poisson floor `sqrt(max(mean count, 1)) / 0.02 s`; and the conditional response for each muscle is the weighted standardized-rate sum divided by that muscle's weight occupancy plus 10 pseudobins. These 16-dimensional source unit rows are pooled and compressed by source SVD to four axes. This is a muscle-recruitment association profile, not a causal muscle-effect or independent-muscle-effect estimator.

The original per-column normalization whitened the four SVD axes independently. The source variance fractions of the retained four axes were 91.65%, 6.96%, 0.83%, and 0.56% in the `09/24 -> 09/26` fold, and 90.73%, 7.06%, 1.21%, and 1.00% in the `09/24,09/26 -> 09/27` fold. A preregistered global-RMS control therefore retained each source column mean but used one shared scale,

\[
s_{\mathrm{global}}=\sqrt{\operatorname{mean}_{k=1}^{4}(s_k^2)},
\]

repeated across all four columns. This deliberately does not whiten weak axes.

| M1 global-RMS muscle profile | Hold 09/26 | Hold 09/27 |
|---|---:|---:|
| Split5-5 pooled Pearson | 0.93577253 | 0.92437406 |
| First10-next10 pooled Pearson | 0.93217331 | 0.96044831 |
| Split5-5 unit median cosine | 0.9342 | 0.9432 |
| First10-next10 unit median cosine | 0.9475 | 0.9755 |
| Half-trial-null / split-noise | 5.0128 | 4.6650 |

These values exceed the original rSyn3 split Pearson values `0.8035/0.7574` and temporal values `0.8840/0.9111` on the same two rolling folds. They show that the global-RMS muscle profile is more reproducible under these source-only profile diagnostics. They do not imply a positive decoder `D - B`; the decoder comparison remains the necessary inner fixed-endpoint test. Root selected this profile for that bounded M1 fixed-24 pilot rather than inferring decoder benefit from pooled stability.

| H1 hold session | Old H-C first3-next3 pooled cosine | Velocity encoding7 | Signed state14 |
|---|---:|---:|---:|
| 01/15 T110633 | 0.3225 | 0.2052 | 0.6206 |
| 01/15 T111328 | 0.4504 | 0.3256 | 0.5451 |
| 01/19 T113543 | 0.3491 | 0.2391 | 0.3778 |
| 01/19 T114045 | 0.5986 | 0.5408 | 0.6459 |

The H1 velocity encoding candidate is weaker than old H-C on all four pooled comparisons. Signed-state responses improve pooled cosine in all four, but median per-unit cosine is lower than old H-C in three of four. This mixed result supports one signed-state decoder pilot, not a claim of uniformly improved unit tuning. The source-normalized largest absolute entry also reaches 10.15 in one held session, so numerical sensitivity and the later decoder result remain essential.

The H1 signed-state inner pilot passed actual numerical preflight: B/D each have 3,600,195 trainable parameters, shared decoder and encoder initialization, finite gradients, exact B carrier insensitivity, and D sensitivity. The nine source sessions match the old inner baseline exactly in training/validation neural arrays, labels, activity support, endpoints, segment boundaries, support/query trial lists, and stride. Root launched the 32-epoch H1 candidate training on GPU 1 at 2026-09-08 17:14:32 UTC. Its baselines were reused after passing the historical-reference audit.

The M1 global-RMS muscle fixed-e24 and H1 signed-state fixed-e32 inner runs completed their source-referenced final audits. These inner results selected profiles for the outer source-only queues; they are not last-date results.

## Locked inner promotion and outer preparation status

The locked inner promotion rule is strict: a candidate's fixed-endpoint new D must exceed both matched B and matched literal old D on its inner date; equality does not qualify. The source-validation-selected checkpoint is a reported secondary diagnostic only and cannot override this fixed-endpoint rule. If the rule fails, the result is negative inner decoder evidence; any next step is a bounded source-only revision and cannot inspect or select from an outer query.

The signed-state H1 inner source profile has retained-four-axis centered variance fractions of 33.19%, 26.31%, 20.46%, and 20.05%. Its largest-to-smallest column-scale ratio is 1.2867. This differs from the M1 weak-axis whitening mechanism: the promoted H1 profile retains per-column state scaling rather than applying the M1 global-RMS control.

Outer H1 source preparation is complete for the eleven actual source sessions: each prepared arm has 22 source NPZ files (train and validation for every session), and the corresponding non-carrier arrays are exactly equal across B, old D, and state D, with manifest and receipt linkage recorded. The prose-only outer authority clarification is also complete.

## Inner audit metrics, promotion, and current outer status

The two inner final audits passed. The promotion rule required positive fixed-endpoint candidate-D deltas against both B and literal old D in both reported metric definitions. M1 therefore locked `muscle_response16_svd4/global_rms`, and H1 locked `signed_state14/per_column`; both outer selection records are `PROMOTED_AND_LOCKED`.

The historical function named `variance_weighted_r2` actually flattens every output element and centers with one global mean. The historical-compatible values below are therefore explicitly named **legacy flattened R2**. The shared implementation was not changed, existing score receipts were not rewritten, and source checkpoint selection and training recipes were not changed. The metric erratum corrects the label and binds the corrected audit replay to the original receipt artifacts. The separately named **channel variance-weighted R2** centers each output channel before weighting; it is reported alongside the legacy metric and is not substituted for it.

| M1 fixed epoch 24, legacy flattened R2 | B | Old D | New D | New D − B | New D − old D |
|---|---:|---:|---:|---:|---:|
| `09/27` inner target | 0.730935 | 0.731502 | 0.774870 | +0.043935 | +0.043368 |

| M1 fixed epoch 24, channel variance-weighted R2 | B | Old D | New D | New D − B | New D − old D |
|---|---:|---:|---:|---:|---:|
| `09/27` inner target | 0.643317 | 0.644068 | 0.701558 | +0.058241 | +0.057490 |

| H1 fixed epoch 32, legacy flattened R2 | B | Old D | New D | New D − B | New D − old D |
|---|---:|---:|---:|---:|---:|
| `01/19` equal-session mean | 0.090765 | 0.121784 | 0.214911 | +0.124146 | +0.093127 |

| H1 fixed epoch 32, channel variance-weighted R2 | B | Old D | New D | New D − B | New D − old D |
|---|---:|---:|---:|---:|---:|
| `01/19` equal-session mean | 0.090741 | 0.121761 | 0.214890 | +0.124149 | +0.093129 |

For H1, new D improved both individual `01/19` sessions in the channelwise replay as well as their equal-session mean. Both metric definitions satisfy the promotion condition; the legacy values remain the historical-compatible primary comparison and the channelwise values are the required parallel report.

## Outer M1 fixed-endpoint result

The outer M1 source seals and paired source gate passed before scoring. The final audit also passed and verified that all three arms used the same `09/28` query labels and coordinate indices, while the source gate verified common source sessions, initialization, optimization recipe, and batch chain. These checks support a valid paired comparison; they do not turn the result into evidence that the profile should be reselected after observing the outer date.

| Outer M1 fixed epoch 24, legacy flattened R2 | B | Old D | Muscle D | Muscle D − B | Muscle D − old D |
|---|---:|---:|---:|---:|---:|
| `09/28` final target | 0.656179 | 0.656318 | 0.647832 | −0.008347 | −0.008486 |

| Outer M1 fixed epoch 24, channel variance-weighted R2 | B | Old D | Muscle D | Muscle D − B | Muscle D − old D |
|---|---:|---:|---:|---:|---:|
| `09/28` final target | 0.551133 | 0.551315 | 0.540236 | −0.010897 | −0.011078 |

The fixed primary endpoint is therefore negative for the promoted muscle D in both metric definitions. The inner M1 result was positive, but it was an earlier-date selection result and does not override this outer fixed-endpoint finding. The lock remains historical: it is not revised from this observed outer score.

For completeness, the source-validation-selected checkpoint is a separate secondary diagnostic. Its outer M1 legacy flattened R2 is B `0.673352`, old D `0.599465`, and muscle D `0.683737`; muscle D − B is `+0.010385`. The corresponding channel variance-weighted values are B `0.573553`, old D `0.477092`, and muscle D `0.587111`; muscle D − B is `+0.013558`. These values do not replace fixed epoch 24. The recorded source-selected epochs were B epoch 8, old D epoch 6, and muscle D epoch 6, whereas the primary comparison fixes every arm at epoch 24. This endpoint difference is consistent with a distinction between source-selected and fixed-epoch evaluation, but the present evidence cannot identify a causal explanation for the reversal.

No additional protocol deviation is indicated by the passed M1 source gate and final audit: target selection/optimizer use remained prohibited, and target geometry was identical across arms. The already documented metric-label erratum remains applicable; both outer M1 tables retain the same explicit legacy-flattened and channelwise labels used for the inner audit.

## Outer H1 fixed-endpoint result

Outer H1 source seals, target scoring, and the final audit passed. All three arms' `target_data_equality.json` checks passed for non-carrier X/y/masks/activity/support/query/endpoints. The H1 target pack begins as float64 but is loaded through frozen `h1_prepare` as float32; equality after that prescribed standard dtype was exact. An initial pre-cast comparison failure was resolved by this dtype-consistent verification without changing data or code, so it is not a protocol change.

| Outer H1 fixed epoch 32, legacy flattened R2 | B | Old D | State D | State D − B | State D − old D |
|---|---:|---:|---:|---:|---:|
| `01/20` equal-session mean | 0.296210 | 0.538206 | 0.563645 | +0.267435 | +0.025439 |

| Outer H1 fixed epoch 32, channel variance-weighted R2 | B | Old D | State D | State D − B | State D − old D |
|---|---:|---:|---:|---:|---:|
| `01/20` equal-session mean | 0.296189 | 0.538188 | 0.563629 | +0.267441 | +0.025441 |

State D exceeds B on both `01/20` sessions. Its comparison with old D is heterogeneous: the state D − old D difference in legacy flattened R2 is `−0.010138` on `T115044` and `+0.061016` on `T115537`, while the fixed equal-session mean improves by `+0.025439`. The mean improvement is the fixed H1 endpoint; the session split should remain visible rather than being summarized as a uniform state-D gain over old D.

The selected checkpoint is auxiliary and does not replace fixed epoch 32. Its legacy flattened equal-session means are B `0.296210`, old D `0.545660`, and state D `0.573963`, with state D − B `+0.277754` and state D − old D `+0.028303`. The recorded selected epochs are B 32, old D 29, and state D 26.

## Completed-round interpretation and artifacts

The fixed primary results point in different directions by dataset: M1 muscle D is negative against B and old D at epoch 24 under both metric definitions, while H1 state D improves the `01/20` equal-session fixed mean by about `+0.26744` against B and `+0.02544` against old D. The positive inner results, the negative outer M1 fixed result, and both metric definitions remain part of the record. Selected checkpoints do not substitute for the predeclared fixed endpoints, and the metric-label erratum remains in force.

This is one seed-42 public-development last-date evaluation. It is neither a multi-seed estimate nor evidence from a wholly new blind test set. The results therefore support the reported paired outcomes and their limits, rather than a general claim that either carrier profile transfers reliably across datasets.

The consolidated [outer results report](../results/carrier_last1_v2/report/RESULTS.md), [fixed-endpoint CSV](../results/carrier_last1_v2/report/fixed_endpoint_summary.csv), [selected-endpoint CSV](../results/carrier_last1_v2/report/selected_endpoint_summary.csv), and [summary JSON](../results/carrier_last1_v2/report/summary.json) contain the complete report rows. The immutable audit records are [M1 outer final audit](../results/carrier_last1_v2/outer/m1/final_audit.json), [H1 outer final audit](../results/carrier_last1_v2/outer/h1/final_audit.json), [M1 inner final audit](../results/carrier_last1_v2/inner/m1/final_audit.json), and [H1 inner final audit](../results/carrier_last1_v2/inner/h1/final_audit.json).
