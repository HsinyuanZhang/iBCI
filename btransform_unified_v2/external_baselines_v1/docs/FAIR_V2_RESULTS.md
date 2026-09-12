# Fair gradient-free controls v2: completed local results

Final method comparisons use EvalAI official held-out results only. The numbers in this document are local public-calibration diagnostics; they neither establish nor refute an official held-out advantage for RIFT. The six v1 packages are frozen and ineligible for submission, and their old recommendation has been withdrawn. The corrected all-source, source-selected linear controls no longer exhibit the severe negative-score failure on this local face. CORAL adds essentially zero over diagonal calibration here. Static + diag-z also gives a strong M1 local control; this finding is an implementation/protocol diagnostic, not a final model ranking.

These are local public-calibration query scores. All table entries below use the same task-specific query rows. Standard R² centers each behavioral output separately before variance weighting; legacy R² uses one global mean of flattened behavior. M1 averages three recordings, M2 six recordings, and H1 seven groups after concatenating each group's two recordings. The JSON also preserves the H1 14-recording diagnostic; it must not replace the seven-group main result.

| Method | M1 standard | M1 legacy | M2 standard | M2 legacy | H1 standard | H1 legacy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| WF-ZS, one smoothed bin | 0.389780 | 0.528665 | 0.131238 | 0.131243 | 0.080404 | 0.081534 |
| diag-z + WF, ten smoothed bins | 0.461263 | 0.583432 | 0.136042 | 0.136046 | 0.114324 | 0.115412 |
| CORAL + WF, source-selected shrinkage | 0.461263 | 0.583432 | 0.136037 | 0.136041 | 0.114322 | 0.115410 |
| AlignedFA + WF, all-electrode posterior | 0.359257 | 0.503203 | 0.081052 | 0.081057 | 0.057677 | 0.058838 |
| AlignedFA + WF, stable-only posterior variant | 0.368145 | 0.510457 | 0.081052 | 0.081057 | 0.057677 | 0.058838 |
| Static RIFT, fixed-final EMA | -0.500945 | -0.148998 | 0.268094 | 0.268098 | 0.209015 | 0.209987 |
| Static RIFT + diag-z, same frozen EMA | 0.586175 | 0.681561 | 0.271371 | 0.271375 | 0.256093 | 0.257010 |
| Static RIFT + CORAL, same frozen EMA | 0.536408 | 0.643304 | 0.248885 | 0.248889 | 0.276458 | 0.277351 |
| Full RIFT, fixed-final EMA | 0.549882 | 0.652592 | 0.274543 | 0.274547 | 0.459527 | 0.460198 |
| Full RIFT, historical local-HO epoch selection | 0.620586 | 0.707514 | 0.363374 | 0.363378 | 0.462350 | 0.463018 |

The final row has a different selection budget: historical full-RIFT epochs were selected using local held-out query labels. It is retained for continuity in these local diagnostics; neither local row substitutes for an official held-out result. The fixed-final full-RIFT row uses E24/E24/E32 and the historical row E3/E10/E16 (M1/M2/H1). Fixed-final is a retrospective target-independent reporting rule applied to existing curves; it is not evidence that the historical curves were unseen. Full-RIFT standard/legacy pairs are algebraically recovered from sealed per-recording or per-group scores and matching target SST; no new full-RIFT inference was performed.

## Paired increments

| Difference, standard R² | M1 | M2 | H1 |
| --- | ---: | ---: | ---: |
| CORAL + WF minus diag-z + WF | +0.00000049 | -0.00000496 | -0.00000139 |
| Static + diag-z minus static identity | +1.08711996 | +0.00327686 | +0.04707888 |
| Static + CORAL minus static + diag-z | -0.04976707 | -0.02248547 | +0.02036434 |
| Full RIFT fixed-final minus diag-z + WF | +0.08861929 | +0.13850140 | +0.34520330 |
| Full RIFT fixed-final minus static + diag-z | -0.03629299 | +0.00317232 | +0.20343343 |

Source CV selected CORAL shrinkage 1.0 for all three tasks. In the linear implementation this replaces the covariance with an isotropic matrix, removing off-diagonal alignment. Residual score differences below 5e-6 are not evidence of cross-channel CORAL benefit. All prescribed static arms are reported; none was picked using target scores. Their CORAL setting was fixed in advance at diagonal covariance shrinkage 0.1 and ridge 0.001, so this experiment does not establish an optimum over static CORAL settings.

## Protocol and source selection

All five linear arms pool every canonical source recording for final supervised fitting. Each source is split chronologically 80:20 by eligible endpoint, with a 21-native-bin endpoint separation to prevent causal-feature overlap. Selection maximizes equal-source-recording standard R², then refits on all eligible source rows. The saved source selection is sealed before target evaluation data are opened. Each full, unpadded raw recording is filtered continuously from zero state using the normalized FALCON 12-tap exponential kernel (20-ms bins, tau 240 ms, extent 1), without trial resets. Session mean/std are fitted only on that session's selected native support bins after the same filter. The ridge intercept is unpenalized; behavior is not standardized. CORAL and AlignedFA align every non-reference source to the latest source reference and then pool all source labels.

| Task | Source recordings / supervised windows | Target recordings / scored windows | diag-z alpha | AFA alpha / K / stable fraction |
| --- | ---: | ---: | ---: | --- |
| M1 | 4 / 213,336 | 3 / 3,881 | 100 | 100 / 40 / 0.75 |
| M2 | 7 / 101,171 | 6 / 15,403 | 100 | 100 / 40 / 1 |
| H1 | 13 / 23,212 | 14 / 33,613 | 10000 | 1000 / 40 / 1 |

The alpha grid is {1e2, 1e3, 1e4, 1e5}; CORAL shrinkage is {0, 0.1, 0.5, 1}; FA dimension is {10, 20, 40}; stable fraction is {0.5, 0.75, 1}. The one-bin WF-ZS arm selected alpha 1e3 in all three tasks. FA uses sufficient-statistics covariance EM with three deterministic starts, per-sample tolerance 1e-6, private-variance floor 1e-6 and a 10,000-iteration budget (one 50,000-iteration retry at the same tolerance). All stored source fits and selected target fits passed convergence checks. Both posterior variants selected K40 and the same fraction within each task; M1 selected 0.75, M2/H1 1.0. The all-electrode posterior follows the author release's inference form. Stable-only posterior is the separately labeled requested robustness variant. The Degenhart author code uses stable rows to estimate rotation, then all electrodes for posterior inference. This corrects the premise that all-electrode inference itself departed from the published method.

Native support uses the same trial budgets: M1 first 10 eval-valid trials, M2 first 33 trials, H1 first 3 valid TrialNum trials. All three tasks read native NWB neural bins; none uses interpolated calibration matrices. Every target recording's exact bin count, trial IDs, NWB hash and support indices are in its receipt. The static-network arms use the same raw support but map it to pooled source raw-support mean/covariance before local_conv; they do not add WF smoothing to a network trained on raw inputs. All network weights remain identical within the identity/diag/CORAL triplet. H1 continues to assume corresponding positional unit rows across dates; physical channel correspondence is unverified.

The one-bin arm is a FALCON-style local reproduction: it matches the demo's default history and exact filter, while using the consistent calibration-statistics and source-selection contract stated here. The official demo has different normalization surfaces, alpha grid and CV scoring. Published private-test WF-ZS scores are not directly comparable to this public local face. The user's provisional M2 value near 0.16 was not a retained receipt; this exact declared protocol produced 0.136042. The result is reported as measured, without target-based retuning.

## Evidence and reproduction

- [Protocol and commands](../fair_v2/README.md), [primary-source implementation audit](OFFICIAL_WF_AFA_AUDIT.md).
- [Machine-readable comparison](../fair_v2/results/comparison_v2.json) binds every input score receipt by SHA-256.
- [Independent linear verification](../fair_v2/results/verification_v2/receipt.json): reload saved numeric models without fitting, independently stream all held-out recordings, maximum prediction difference 0.0; scores, source selections, native support and code hashes verified.
- Linear receipts: [M1](../fair_v2/results/m1_v2/receipt.json), [M2](../fair_v2/results/m2_v2/receipt.json), [H1](../fair_v2/results/h1_v2/receipt.json).
- Static receipts: [M1](../fair_v2/results/static_m1_v2/report_rescored.json), [M2](../fair_v2/results/static_m2_v2/report_rescored.json), [H1](../fair_v2/results/static_h1_v2/report.json); [static implementation and validation](FAIR_V2_STATIC_RESULTS.md).
- [Full-RIFT metric restatement](../fair_v2/results/rift_reference_v2/receipt.json) binds original score receipts, checkpoint hashes, target hashes and both SST denominators.
- [V1 package freeze](../submissions/READINESS.json). No v2 image was pushed, no method registered, and no new official submission made for this work.

Regenerate this table after all receipts exist: `python fair_v2/compare.py` from `external_baselines_v1`. The joiner reads JSON only; it performs no fitting, prediction or model selection.
