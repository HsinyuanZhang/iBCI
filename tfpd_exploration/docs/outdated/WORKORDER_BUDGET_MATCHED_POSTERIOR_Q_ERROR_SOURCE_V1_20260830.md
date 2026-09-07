# Source-Only Posterior-q / Held-Carrier Error Audit V1

## Purpose

Test the remaining E03 mechanism claim without opening within-development,
external sub-M, formal, model, checkpoint, or GPU surfaces.

For each of the frozen 27 source sessions, use chronological rewarded trials
`[0,M)` for `M in {4,10,30}` to construct the already frozen posterior mean and
angular reliability `q`.  Use the disjoint source-only trials `[30,50)` to fit
an ordinary OLS direction reference.  The reference is scorer-only: it never
enters the posterior, q, normalizer, training, checkpoint selection, or target
inference.

## Primary readout

For every session and budget, compute the Spearman correlation across units
between q and absolute circular direction error to the disjoint OLS reference.
Higher q is expected to mean lower error, so the expected sign is negative.

Report all per-session rows, defined-unit counts/fractions, equal-session mean
and median rho, the number of negative sessions, and a deterministic 10,000
draw session bootstrap 95% interval (seed 42).

A budget has mechanism evidence only when:

- every session has at least eight defined units and at least 80% of units are
  direction-defined;
- median session rho is at most `-0.10`;
- at least 18 of 27 session rhos are negative; and
- the bootstrap upper endpoint is below zero.

M4 and M10 are the claim budgets.  M30 is descriptive and cannot rescue a
failed low-budget mechanism gate.  This audit does not select a checkpoint or
change the already running C2/C3 matched score.

## Causal and data boundaries

- Exact strict manifest: 27 source sessions only.
- Support: first M chronological rewarded trials.
- Reference: chronological rewarded trials 30 through 49, disjoint from every
  support including M30.
- No within-development, sub-M, formal, query, or target path may be resolved.
- No Torch, CUDA, model, checkpoint, optimizer, backward, or update.
- Publish attempt before importing the NWB reader or opening a source file.

## Drift policy

The numerical execution closure is strict.  Work-order and focused-test drift
is review-only and records `ACCEPTED_NON_NUMERIC_DRIFT` with
`numerical_acceptance_affected=false`; it never triggers a rerun.

## Result

Canonical result root:

`tfpd_exploration/results/budget_matched_posterior_cal_aug_v1/c3_q_error_source_v1`

Publish immutable attempt plus either audit+terminal or failure pairs.  This
work order authorizes one CPU-only source audit after independent static tests
and a fresh-root check.  It does not authorize target scoring or GPU work.
