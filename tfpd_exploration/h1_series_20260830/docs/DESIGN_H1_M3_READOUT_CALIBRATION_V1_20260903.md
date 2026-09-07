# H1-M3RC: Exactly-M3 Closed-Form Readout Calibration V1

Date: 2026-09-03  
Status: `READY_FOR_SOURCE_ONLY_DATE_LODO_SCREEN`

## 1. Question

The official H1 held-out calibration surface contains exactly three legal
trials per session, but those trials contain thousands of labelled 20 ms bins
for seven output covariates.  This experiment asks whether those labels can
correct the frozen C1 output coordinates without changing its neural model,
carrier, identity, or decoder weights.

This is a Tier-2 few-shot method.  It must never be reported as label-free
adaptation.  It is nevertheless legal under the H1 calibration contract: the
same three labelled calibration trials already used by the H-C carrier are the
only target labels read.

## 2. Frozen anchor and surface

- Anchor: sealed five-fold C1 epoch-49 checkpoints.
- Support: chronological first three calibration trials, exactly.
- Identity and H-C carrier: unchanged C1 M3 construction.
- Readout-fit rows: every eval-valid bin belonging to those same three trials.
- Query: the separate held-in-minival recording for the same session.
- Window law: official H1 `W=700`, zero left padding, last-bin prediction,
  output divided by 20.
- Metric: variance-weighted R2, first equal over recordings within an outer
  date and then equal over the five outer dates.

The outer date's calibration and minival files may be opened only after the
candidate family and ridge value have been selected from source dates and an
immutable selection receipt has been published.

## 3. Candidates

The frozen C1 prediction is `p(t) in R^7` and the labelled calibration target
is `y(t) in R^7`.  Calibration means and standard deviations are frozen with
a floor of `1e-6`:

`z_p=(p-mu_p)/sigma_p`, `z_y=(y-mu_y)/sigma_y`.

Two closed-form families are considered:

- `DIA7`: seven independent affine maps from `z_p,j` to `z_y,j`;
- `MAT7`: one full affine map from `[z_p,1]` to `z_y`.

Only slope/matrix coefficients are ridge-penalized; the intercept is not.
The fixed ridge grid is `{0, 1e-6, 1e-4, 1e-2, 1, 100}`.  For each outer
fold, every candidate is independently fit on each source session's M3 and
scored on that session's separate minival recording.  Selection maximizes
equal-source-date R2.  Exact ties prefer `DIA7`, then the larger ridge value.

`TPL-M3` (the seven-dimensional calibration target mean, with no neural
prediction) is always reported as a descriptive control.  It is never a
selectable model.

## 4. Gates

The nested source-selected candidate may advance only if its outer-date gains
over frozen C1 satisfy all of:

- equal-date mean gain at least `+0.005`;
- at least `4/5` outer dates nonnegative;
- worst outer date at least `-0.010`.

To support a neural-readout statement rather than template recall, it must
also beat `TPL-M3` on mean outer-date R2 and on at least `3/5` dates.

Failure stops this route.  There is no epoch, seed, target-side hyperparameter
choice, or gradient rescue.

## 5. If the gate passes

Select family and ridge once using all held-in source dates, then fit one
closed-form map per official session from that session's M3.  Package the map
with the existing all-source C1 M3 payload.  The map is fixed before the eval
recording begins; eval activity and labels never update it.

Local packaging must prove the IEEE/no-op identity control, exact C1 state
immutability, all 27 payloads, official reset/tag behavior, and zero target
optimizer/backward/model updates.  EvalAI submission remains a separate user
authorization.

## 6. Interpretation boundary

This cell tests dense-label output-coordinate correction.  It does not test
continual activity memory or post-pooling.  The completed J3/N3 matched
training result remains a separate null: full-model source retraining was
unstable even though source loss decreased.

