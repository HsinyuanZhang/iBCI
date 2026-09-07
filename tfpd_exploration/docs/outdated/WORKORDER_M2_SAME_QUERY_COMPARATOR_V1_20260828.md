# M2 Same-Query Comparator V1

## Purpose

Measure the important M2 baselines on exactly the query rows already used by
`m2_t4_activity_budget_screen_v1`.  This is a comparison experiment, not a new
training run.  It must not update any model parameter or use query targets for
model selection.

## Surfaces

- `within_post30`: the seven source sessions, only W50 windows whose first bin
  is at or after the start of trial 30.
- `external_official_query`: the six held-out sessions and the exact official
  query windows materialized by the frozen data module.

Every cell for a session must bind the same ordered window-start digest and the
same target digest.  Scores are last-bin, two-coordinate variance-weighted R2,
then equal-session mean/median.

## Budgets and support selection

- M30 and M10 use chronological trials `0:M`.
- The T4/classical M4 cells use the frozen D-optimal four-trial subset selected
  only from finite target directions in trials `0:30`.
- Original SPINT is reported both with honest chronological M4 activity and a
  separately named D-opt matched-support M4 diagnostic.  The latter is not a
  label-free baseline.
- Dense supervision rows are W50 endpoints whose complete history stays inside
  one selected support trial.  No history bridges an omitted trial.
- A selected trial shorter than W50 contributes zero dense-label rows.  It is
  never padded into a velocity-labelled window and never joined to an adjacent
  trial.  Such a trial still contributes its real trial-direction label to the
  population-vector tuning fit.  Every row discloses selected trial lengths,
  dense-eligible indices, and short selected indices.

## Cells

1. Frozen original SPINT/F0 with chronological activity M30/M10/M4.
2. Frozen original SPINT with the matched D-optimal M4 activity subset
   (selection-control diagnostic).
3. Frozen T4 decoder with normalized side value exactly zero, i.e. the frozen
   train-population mean-carrier intervention, at M30/M10/M4.
4. Frozen T4 decoder with production OLS T4 at M30/M10/M4.
5. Reuse, without rerunning, the accepted fixed-ridge T4 static M30/M10/M4 and
   activity30 M10/M4 rows from parent score SHA
   `6bdad93328ba26c12b8aa3afffcbc3490c312dfe22939b3d3bb3c5ec5b9005ce`.
6. Direct W50x96 ridge at M30/M10/M4, with support-only standardization,
   unpenalized intercept, and fixed normalized lambda 1.0.
7. Population vector at M30/M10/M4, with trial-direction tuning and a dense
   support-only affine two-coordinate gain.

The direct Ridge and PV cells use many more velocity labels than T4 and must be
reported as dense-label classical comparators, not deployment-equivalent arms.

## Runtime and integrity

- GPU1 only; GPU0 and unrelated processes are out of scope.
- Frozen SPINT SHA:
  `fbcb9914561c4664fa0f8d0b1791e67505841d3ac470ea7ad68d54e408ca13ec`.
- Frozen T4 SHA:
  `25d7bc72b4d440004b58f1beaeadb7e15565a43e83dd1eadd160374270ec1d3e`.
- No gradients, backward call, optimizer, target adaptation, or refitting on
  query targets.
- Publish only after every row and cross-row identity check succeeds.  The
  result consists of immutable `score.json` and canonical SHA sidecar under an
  immutable result directory.
