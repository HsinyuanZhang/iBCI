# Workorder: matched M4/M10/M30 comparator matrix

Date: 2026-08-23

Status: authorized Phase-1 performance comparison. This workorder does not
authorize target-session optimization or access to the formal surface.

## Objective

Measure the short-calibration performance frontier on exactly the sealed
within-6 and external-15 query inputs. The matrix must distinguish a small
T4-label budget from a small *total* calibration budget. All session scores use
the valid last bin of each 50-bin query window, variance-weighted two-coordinate
R2, followed by equal session weighting.

## Budgets and regimes

- Budgets: M4, M10, and M30 chronological rewarded trials.
- `label_limited`: T4 sees M labels while B3S retains its sealed M30 activity.
- `total_calibration_limited`: both T4 and B3S see only the first M trials.
- Query windows are unchanged across budgets and never enter a fit.

## Live neural rows

1. Sealed Cell-D + ordinary OLS T4, label-limited.
2. Sealed Cell-D + ordinary OLS T4, total-calibration-limited.
3. Arm A + ordinary OLS T4, label-limited.
4. Sealed Cell-D + fixed normalized ridge-T4 (`lambda=0.1`), label-limited.
5. Sealed Cell-D + prefix-only GCV-selected ridge-T4, label-limited.

The ridge carrier penalizes only the equivariant cosine coefficients `a,c`;
the intercept is never penalized and `m=sqrt(a^2+c^2)` is derived afterward.
GCV may see only the same M prefix trials and uses a preregistered lambda grid.
Every neural row must strict-load its immutable checkpoint, stay in eval mode,
emit finite repeated outputs, and preserve the model-state digest.

## Classical rows

6. Trial-rate ridge: one neural-rate row and one mean-velocity label per prefix
   trial. This is the information-density-matched linear comparator.
7. Dense W50 ridge: all valid causal 50-bin windows inside the first M trials.
   This is a label-dense classical upper comparator and is not information
   matched to T4.
8. Population vector: prefix OLS preferred directions plus an affine 2-D gain
   fit on valid prefix windows. It is also label-dense and must be disclosed as
   such.

All ridge fits use a closed-form solve, no iterative optimizer, no target
backward pass, and a fixed normalized penalty of 1.0 unless explicitly named.

## Historical anchors

- Original-SPINT B0: the immutable three-seed within/external authority. It has
  M30 activity calibration and no T4 labels; it is budget-invariant and is not
  relabeled as an M4/M10 total-calibration result.
- A2 T4: immutable three-seed matched-rescore M30 authority only. No fabricated
  M4/M10 A2 row is allowed.

## Required gates and output

- Exact C0 Cell-D parity against the existing low-cost V3 receipt at M4/M30.
- No overlap between any classical support window and the fixed query starts.
- Exact query target digest shared by every live row for a session.
- Per-session R2, prediction/input/model/fit digests, mean/median, sign counts,
  and deterministic session-bootstrap 95% intervals.
- Target optimizer/backward/update counts all zero; formal unopened.
- One immutable attempt/receipt/terminal graph under a fresh result root.

Phase 2 may start only after this receipt identifies whether the dominant M4
loss comes from carrier estimation, activity calibration, or low-budget model
training. Phase 2 is one performance-oriented cell, not an ablation sweep.
