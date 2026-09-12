# Official WF / AlignedFA primary-source audit

## Scope and sources

This is a read-only comparison of the completed `external_baselines_v1`
controls with (1) the FALCON project’s installed `decoder_demos` source and
(2) the authors’ released code accompanying Degenhart et al. (2020).  It does
not claim that the FALCON demo is a paper-wide normative baseline.  It records
the concrete implementation needed when the name “official demo WF” or
“Degenhart-style AlignedFA” is used.

Primary sources:

- Local FALCON demo installed in the prescribed environment:
  `/home/xinyuan/miniconda3/envs/spint/lib/python3.10/site-packages/decoder_demos/sklearn_decoder.py`
  and `decoding_utils.py`.
- FALCON upstream copy: <https://github.com/snel-repo/falcon-challenge/blob/main/decoder_demos/sklearn_decoder.py>.
- Degenhart, Bishop et al., *Stabilization of a brain-computer interface via
  the alignment of low-dimensional spaces of neural activity*, Nat. Biomed.
  Eng. 2020, DOI <https://doi.org/10.1038/s41551-020-0542-9>; author release:
  <https://github.com/alandegenhart/stabilizedbci>.
- Author-release source used below:
  [`fitBaseStabilizer.m`](https://raw.githubusercontent.com/alandegenhart/stabilizedbci/master/code/stabilization/fitBaseStabilizer.m),
  [`updateStabilizer.m`](https://raw.githubusercontent.com/alandegenhart/stabilizedbci/master/code/stabilization/updateStabilizer.m),
  [`identifyStableLoadingRows.m`](https://raw.githubusercontent.com/alandegenhart/stabilizedbci/master/code/stabilization/identifyStableLoadingRows.m),
  [`alignLoadingMatrices.m`](https://raw.githubusercontent.com/alandegenhart/stabilizedbci/master/code/stabilization/alignLoadingMatrices.m),
  [`learnOptimalOrthonormalTransformation.m`](https://raw.githubusercontent.com/alandegenhart/stabilizedbci/master/code/stabilization/learnOptimalOrthonormalTransformation.m),
  [`getStabilizatonMatrices.m`](https://raw.githubusercontent.com/alandegenhart/stabilizedbci/master/code/stabilization/getStabilizatonMatrices.m),
  [`fitFA.m`](https://raw.githubusercontent.com/alandegenhart/stabilizedbci/master/code/FA/fitFA.m),
  [`fastFAEStep.m`](https://raw.githubusercontent.com/alandegenhart/stabilizedbci/master/code/FA/fastFAEStep.m),
  and [`fastFAMStep.m`](https://raw.githubusercontent.com/alandegenhart/stabilizedbci/master/code/FA/fastFAMStep.m).

## Finding 1: the completed WF is not the FALCON sklearn-demo WF

The completed external control uses raw query bins, ten causal bins, fixed
`ridge=1`, and a source-only feature mean/standard deviation:

- `external_baselines_v1/data.py:26-34` creates literal-zero-left-padded raw
  endpoint windows;
- `external_baselines_v1/run.py:20-30` fits and scores that feature form;
- `external_baselines_v1/wiener.py:8-12` uses a closed-form ridge solve with
  `ridge=1` and an unpenalized intercept;
- `external_baselines_v1/run.py:84` establishes the completed defaults:
  history 10 and ridge 1.

The installed FALCON demo instead does all of the following.

1. It causally smooths the *entire raw binned-count sequence before split or
   normalization*, in `sklearn_decoder.py:27-35`.
2. It performs the chronological 80/20 split (`TRAIN_TEST=(0.8,0.2)`,
   `decoding_utils.py:5`; split at `sklearn_decoder.py:47-49`).
3. It fits `x_mean` and `x_std` on the smoothed **training prefix only**, with
   `nanmean` / `nanstd` and zero standard deviations replaced by one
   (`sklearn_decoder.py:50-55`), and applies exactly those source-prefix
   statistics to both train and test.  It deliberately does not z-score
   behavior (`sklearn_decoder.py:52-57`).
4. It uses a `GridSearchCV(Ridge(), {"alpha": np.logspace(-5,5,20)})`
   (`decoding_utils.py:106-110`).  There is no explicit `cv`, `scoring`, or
   shuffle argument.  Therefore the installed scikit-learn default applies:
   five-fold `KFold` for ordinary regression, `shuffle=False`, and the
   estimator’s default score, multi-output uniform-average R2.  The following
   held-out `decoder.score` is also uniform-average R2
   (`decoding_utils.py:110-114`), not the challenge's variance-weighted R2.
5. It applies H1-specific still-time removal before the lag matrix; it also
   removes `~eval_mask` and NaN targets after the train/test split
   (`sklearn_decoder.py:36-79, 190-191`).

Thus the criticism that raw fixed-alpha `wf_raw` is not a fair comparison to
the named FALCON sklearn decoder is supported.  It differs in filter, feature
normalization surface, history, alpha selection, and training/sample masking.

### Exact 240-ms exponential filter

`decoder_demos/filtering.py:9-34` defines `tau=240` ms, `bin_size=20` ms,
and default `extent=1`.  Its exact kernel is

\[
t_j=20j\;\mathrm{ms},\quad j=0,\ldots,11;\qquad
k_j=\frac{\exp(-t_j/240)}{\sum_{r=0}^{11}\exp(-t_r/240)}.
\]

For each channel it returns the first `T` samples of the full convolution,

\[
\tilde x_t=\sum_{j=0}^{\min(11,t)} k_j x_{t-j},
\]

so pre-recording bins are literal zero.  This is causal.  The source comment
at lines 22-24 explicitly says that a three-time-constant extent was intended
for reporting parity, while the reference hard-codes the equivalent of
`extent=1`.  `filtering.py` in the local SPINT third-party copy has the same
definition at lines 9-33.

### History and lag are not ambiguous

The CLI default is `history=0` (`sklearn_decoder.py:337-344`), so the default
demo is one contemporaneous smoothed bin.  If a caller supplies `history=h>0`,
`generate_lagged_matrix` creates, for endpoint time `t`,

\[
[\tilde x_t,\tilde x_{t-1},\ldots,\tilde x_{t-h}],
\]

in newest-to-oldest block order (`decoding_utils.py:22-40`).  The target and
blacklist are then both sliced `[h:]` (`sklearn_decoder.py:66-73`): the target
is (y_t), so this procedure introduces **no neural-to-target temporal lag**.
`apply_neural_behavioral_lag` exists in `decoding_utils.py:43-64`, but the
demo training path does not call it.

At deployment the demo sets `self.history=payload['history']+1`, appends the
latest standardized smoothed bin, and reverses the buffer before flattening
(`sklearn_decoder.py:95-100, 134-144`).  That agrees with the stated training
feature order.

### Important demo inconsistency, which fair_v2 must not silently inherit

The demo trains with *smoothed source training-prefix* statistics
(`sklearn_decoder.py:50-55`).  Yet, after fitting, it serializes per-file
means and standard deviations of **raw, unsmoothed calibration/data neural
counts** (`sklearn_decoder.py:193-205`), then at runtime smooths the stream and
subtracts those per-session raw values (`sklearn_decoder.py:154-159`).  This
is a literal reading of the primary code, but it is not a single coherent
train/inference normalization contract.  A reproducer may preserve it under
the explicit label `falcon_demo_literal`; it should not be called a clean
reference WF.

## Finding 2: current AlignedFA is only partially Degenhart-style

The current implementation honestly says it is a practical implementation,
not a byte-identical reproduction (`external_baselines_v1/core.py:1-5`).  It
fits separate source and target sklearn `FactorAnalysis` models, then uses
source/target loading matrices and iterative Procrustes pruning
(`core.py:56-86`).  Its target query representation is

\[
z_t^{\mathrm{current}}=\operatorname{FA}_{\mathrm{target}}(x_t)R,
\]

where `FactorAnalysis.transform` supplies the target posterior mean and
`R` is the Procrustes factor (`core.py:49-54, 76-85`).

The author release establishes the following more precise contract.

1. Base and update FA are fit separately to binned-count trial matrices, with
   multiple random restarts and the maximum-final-log-likelihood model chosen.
   `fitBaseStabilizer.m:37-61` uses five restarts by default; the update code
   repeats that policy (`updateStabilizer.m:33-66`).  Its default iteration
   budget is 100,000, likelihood threshold `1e-5`, and private-variance floor
   `0.1` (`fitBaseStabilizer.m:37-43`).  These differ materially from the
   current sklearn configuration (`core.py:57-70`: 1,000 iterations, tol
   `1e-3`, randomized variance initialization, and three runs in completed
   artifacts).
2. Let (C_b,C_u\in\mathbb R^{p\times q}) be base and update loadings.
   Stable rows are **not** initially every channel.  The release first removes
   all rows where either loading norm is below `ALIGN_TH`; default
   `ALIGN_TH=0.01` (`updateStabilizer.m:41-48`,
   `identifyStableLoadingRows.m:27-39`).
3. Among remaining rows it repeatedly finds the orthogonal transform,
   calculates rowwise Euclidean residuals, and retains the smallest residuals
   until exactly `ALIGN_N` rows remain
   (`identifyStableLoadingRows.m:41-58`).  The default `ALIGN_N` is all base
   rows, not a fixed fraction (`updateStabilizer.m:41-48`).  The current
   implementation's one-at-a-time max-squared-residual removal has the same
   residual ordering conditional on a fixed candidate set, but it omits the
   low-loading screen and changes the selection parameter to
   `max(q, ceil(p*stable_fraction))` (`core.py:81-85`).
4. On the stable rows (S), the author code solves
   \(\min_{W^TW=I}\lVert C_b[S]-C_u[S]W^T\rVert_F\).  It obtains
   (W=UV^T) from `svd(C_b[S]^T C_u[S])`
   (`learnOptimalOrthonormalTransformation.m:23-25`) and replaces the update
   loading matrix with (C_uW^T) (`updateStabilizer.m:68-70`).  The current
   code's `R=UV^T` from `svd(C_u[S]^T C_b[S])` and `zR` is algebraically the
   same orientation when its (R) is the factor minimizing
   \(\lVert C_uR-C_b\rVert_F\).  This portion of the critique needs a
   correction: the current code is not directionally reversed merely because
   it multiplies the latent by `R`; that is the consistent latent-coordinate
   form of the released loading rotation.
5. The released linear latent inference is explicit.  For an aligned FA with
   mean (d), loading (C), and diagonal/private covariance \(\Psi\),
   `getStabilizatonMatrices.m:1-22` uses
   \[
   \beta=C^T(CC^T+\Psi)^{-1},\quad o=-\beta d,\quad
   \ell_t=\beta x_t+o.
   \]
   With the update rotation this is equivalently
   \(\ell_t=W\,C_u^T(C_uC_u^T+\Psi_u)^{-1}(x_t-d_u)\).
   Crucially, the stable set (S) is used only to estimate (W).  After that,
   `updateStabilizer.m:68-70` rotates the complete (p\times q) update loading
   matrix, and `getStabilizatonMatrices.m:1-22` receives that complete (C),
   full (\Psi), and full mean (d).  Thus the released posterior uses every
   observed electrode; `alignChs` records the alignment rows and is not passed
   as an inference mask.  A fair source-reproduction implementation must
   export precisely this all-electrode posterior-linear map (or prove numerical
   equivalence to it), rather than describe a generic loading-only Procrustes
   operation as AlignedFA.
6. The released FA solver is covariance-based EM, not sklearn's randomized-SVD
   implementation.  `fitFA.m:5-15, 167-182` fits
   \(x_t=C\ell_t+d+\epsilon_t\), with unit latent covariance and diagonal
   (\Psi), by alternating `fastFAEStep.m` and `fastFAMStep.m`.  For an
   observed-channel block, its E-step evaluates
   \(C^T(CC^T+\Psi)^{-1}\) (`fastFAEStep.m:25-33`), then the M-step updates
   (C), (d), and diagonal (\Psi) rowwise (`fastFAMStep.m:27-107`).  An
   optimized implementation may use an algebraically equivalent latent-space
   solve, but a claim of release parity requires numerical verification; it
   cannot call a separate sklearn `FactorAnalysis` fit an exact reproduction.

The stable-electrode concern is therefore supported: current `core.py` treats
all (p) rows as candidates and never applies the author-code criterion
\(\lVert C_{b,i}\rVert_2\ge0.01\) and
\(\lVert C_{u,i}\rVert_2\ge0.01\).  It also changes `ALIGN_N` into a
postulated half-channel fraction.  Those are method changes, not harmless
implementation choices.  A separate caveat remains: FALCON H1 positional unit
rows have unverified physical correspondence, as already documented in this
project's `external_baselines_v1/README.md:76-81`; no stable-row algorithm can
create correspondence absent in the underlying data.

## fair_v2 mathematical contract

The following is the recommended contract for a genuinely paired comparison.
It has two named, non-interchangeable variants rather than silently combining
incompatible pieces.

### `WF-FALCON-literal-v2`

Use this only to reproduce the released FALCON demo exactly:

1. Causally smooth raw 20-ms count bins with the normalized 12-tap kernel
   above, from a zero initial condition.
2. On each source training recording, chronologically split at 80%; fit
   `nanmean/nanstd` on the smoothed source prefix, replace zero scales by one,
   and z-score both source train and source validation with those source-prefix
   values.
3. Apply H1 still-time and eval-mask/NaN deletion in the demo's stated order.
4. Use either (h=0) (actual CLI default) or a preregistered (h), with
   features `[x_t,...,x_{t-h}]` and target (y_t); do not insert an unstated
   target lag.  In the fair_v2 API, `history_bins` instead counts the **total**
   feature bins, so `history_bins=1` is the CLI `(h=0)` case and
   `history_bins=h+1` represents the demo's `(h)` setting.
5. Choose \(\alpha\) by `GridSearchCV(Ridge(),
   {'alpha': logspace(-5,5,20)})` with the installed sklearn defaults and
   record sklearn version, selected alpha, fold splitter, and score.
6. For literal inference, retain the documented per-session raw-statistic
   substitution.  It is reproducible but is not source-only and must be
   reported as target calibration.

### `WF-AFA-paired-clean-v2` (recommended fairness arm)

This is the arm to compare against a target-calibrated AFA without the demo's
normalization mismatch:

1. Apply the same causal 12-tap 240-ms smoothing to **both** WF and AFA query
   inputs, and use the same history (h), endpoint target (y_t), zero
   initialization, valid-bin filtering, source recordings, and source
   train/validation chronology.
2. Fit one source-side feature normalization map after smoothing and use that
   immutable map for source WF, CORAL+WF, and AFA+WF.  Do not use target
   labels.  If any target-session normalization is allowed, give it to every
   method and name it explicitly as target neural-only adaptation.
3. Select the common ridge alpha using the same source-only chronological CV
   protocol.  The only difference between paired methods is the exported
   neural-only transform applied to each query bin before shared
   normalization/history/readout.
4. Fit source and target FA on the same explicitly declared count-feature
   surface used for query alignment: if query bins are 240-ms smoothed, apply
   that causal filter to both source and target FA support before fitting. Do
   not fit FA on cubic/trialized support while applying it to a differently
   preprocessed raw query stream.  Record likelihood/restart/tolerance/private
   variance settings; for a paper-faithful AFA, use the author defaults or
   fully record a deliberate alternative.
5. Define candidate stable rows by the two-sided loading-norm threshold
   \(\|C_{b,i}\|_2\ge\theta\land\|C_{u,i}\|_2\ge\theta\), then iterative
   Procrustes pruning to a preregistered absolute `ALIGN_N`; fail closed when
   fewer than `max(q, ALIGN_N)` usable rows remain.  Record `S`, \(\theta\),
   `ALIGN_N`, residuals, and all FA parameters in the payload manifest.
6. For the author-release reproduction, export the all-electrode target map
   \(x\mapsto W C_u^T(C_uC_u^T+\Psi_u)^{-1}(x-d_u)\), calculate it in
   float64 for parity, then pass its float32 query result through the shared
   feature normalizer/history/ridge.  The transform is label-free at target
   time but must be labeled as precomputed target neural calibration when it
   uses a target calibration recording.  Here (S) determines (W), while the
   posterior still consumes all channels.

### `AFA-stable-posterior-user-variant-v2` (optional, not author reproduction)

If the experimental question specifically requires the final posterior to use
only the selected stable rows, define and report a separate variant:

\[
x\mapsto W C_u[S]^T
\bigl(C_u[S]C_u[S]^T+\Psi_u[S,S]\bigr)^{-1}(x[S]-d_u[S]).
\]

This can be a useful robustness arm, but it is a user-specified modification.
The Degenhart author release uses (S) for loading alignment and retains all
observed electrodes for posterior inference.  It must therefore not be named
or evaluated as an exact Degenhart reproduction.

This contract resolves the valid fairness criticism without claiming that
either a raw ten-bin ridge or a loading-only half-row heuristic is the official
FALCON WF or the released Degenhart AFA procedure.
