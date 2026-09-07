# Frozen protocol: H1 M=4 empirical-Bayes confidence-shrinkage carrier

**Status:** New CPU-only M=4 analysis. It is not an M=2 rescue, does not alter the sealed M=4 raw-carrier receipt, and has no GPU/SPINT authorization in advance.

## Frozen source plan and target budget

For every six-date LODO fold, reconstruct and bind the source-only neural mean/scale, PCA, selected q/lambda, and 7-by-4 row basis `U` from the immutable M=4 population-decoder receipt. The existing receipt's source sessions, input hashes, q, lambda, raw R2, and raw split-pair cosine are the comparator contract. No grid, source/target/query tuning, transform choice, or prior-strength sweep is run here. Any reconstructed raw metric that differs from the bound M=4 receipt fails closed.

Only the 13 public H1 held-in-calib recordings may be opened. The target uses trials 1--4 as paired support, independent halves A=trials 1--2 and B=trials 3--4, and trials 5+ as strict chronological query. Every recording needs at least three legal later trials. Blocks are finite, trial-bounded 100-ms blocks; no threshold, minival/query/formal held-out file, target optimizer, backward pass, GPU, Docker, or EvalAI is permitted.

## Analytic empirical-Bayes carrier

For each source fold, collect source pooled M=4 decoder rows `R_s` and form `E_s=R_sU`. Pool all channels and source recordings without an electrode index. The global 4-D prior is coordinate mean `mu`; its sole scale is isotropic maximum-likelihood variance `tau^2=sum_rows ||E_s-mu||^2/(4*number_of_rows)`.

For a target ridge fit, calculate raw rows `R` and carrier `E=RU`. From its target support design only, calculate frequentist ridge covariance `G=(D'D+Lambda)^-1 D'D (D'D+Lambda)^-1`, output residual variance `sigma_j^2=RSS_j/(n-tr[D(D'D+Lambda)^-1D'])`, channel scale `h_i=p_i'G p_i/source_scale_i^2`, projected isotropic row variance `v_i=tr[h_i U' diag(sigma^2) U]/4`, and deterministic shrinkage `w_i=tau^2/(tau^2+v_i)`, `eC_i=mu+w_i(E_i-mu)`. `D` has an unpenalized intercept and `Lambda` penalizes ridge features only. Every denominator must be finite and positive. The only deployable carrier is `eC` with shape `[N,4]`; uncertainty is not appended.

The fitted pooled intercept `beta0` is retained. Query scoring must lift the carrier through the fixed source row basis: `R_C=eC U^T`, `yhat=(x-source_mean)R_C+beta0`. The raw comparator uses this same equation with pooled `R_raw`, not a separately fit model. A deterministic complete nonidentity row permutation of `R_C` is scored with the identical query/intercept equation.

## Null, attachment, and gate

For each of 31 deterministic nonzero replicates, labels are independently circularly rotated within each of the four target support trials; ridge, residual covariance, weights, and shrinkage are recomputed. Query rates and labels remain correctly paired. Attachment is the median per-channel cosine of independently shrunk carriers from A and B.

The frozen conjunction requires all six date rows defined and at least four of six dates satisfying every clause: (1) shrunk correct later-query R2 > 0; (2) shrunk correct R2 > fresh rotation-null q95; (3) shrunk split-half carrier cosine >= 0.5; (4) shrunk cosine > bound raw M=4 cosine; (5) shrunk correct R2 >= bound raw M=4 correct R2; and (6) shrunk correct R2 > deterministic row-shuffle control. Failure is permanent for this program: do not change prior, variance formula, threshold, source plan, null, or integration.
