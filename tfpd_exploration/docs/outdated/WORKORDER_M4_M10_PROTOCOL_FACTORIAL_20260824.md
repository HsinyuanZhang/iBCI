# Work Order: M4/M10 Low-Cost Protocol Factorial

Date: 2026-08-24

## Objective

Complete the missing causal, target-free short-calibration comparisons on the
sealed Cell-D SWA.  No training or target update is allowed.

## Factors

- support: chronological first M versus cue-only greedy D-optimal selection
  from the first 30 trials;
- estimator: ordinary OLS versus fixed normalized ridge lambda 0.1;
- B3S view: M30 activity versus exactly the same selected M trials used by
  the carrier.

Budgets are M4 and M10.  The D-optimal candidate pool is capped at the first
30 trials so every selected calibration trial precedes the fixed query
surface.  This is deliberately different from the earlier first-50 C1
diagnostic, which cannot be paired with query-after-30 B3S activity without
introducing future information.

Report within-6 and external-15 last-bin variance-weighted R2, equal session
means/medians, paired sign counts, and deterministic bootstrap intervals.
