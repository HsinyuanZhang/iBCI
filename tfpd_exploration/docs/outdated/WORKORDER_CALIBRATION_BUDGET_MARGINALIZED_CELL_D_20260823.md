# Work Order: Calibration-Budget-Marginalized Cell D (CBM-D)

Date: 2026-08-23  
Status: performance-first Phase-2 cell; v3 successor after two fail-closed source smokes.

## V1 failure and V2 correction

The first source smoke stopped before backward or an optimizer update because
its short-budget carrier builder reused Posterior Carrier's same-row recovery
for one non-finite `target_dir`.  That made its M30 carrier differ from the
sealed ordinary-T4 datamodule input, so the parity assertion correctly failed.
The immutable v1 attempt/failure pair is a required predecessor for v2.

V2 removed that second estimator treatment and reproduced the sealed
direction-index mapping, but still passed the mapped indices through the
generic D-optimal carrier-fitting helper.  M30 remained non-identical and the
parity gate stopped the run before an iterator was created.

The v2 smoke also stopped before backward/update on the same parity gate.  A
deeper audit found that the generic D-optimal fitting helper treats a missing
direction sentinel `-1` as Python's last canonical-direction index, whereas
the sealed datamodule excludes missing directions before fitting.  V3 calls
the sealed shared `_unit_tuning_features` producer directly for every unit
and every M.  It binds both immutable pre-update failures.  No posterior
recovery, D-optimal helper, or new estimator is present in v3.

## Objective

Improve M4/M10 zero-shot transfer without enlarging the decoder or adding a
target optimizer.  The sealed Cell-D graph, standard initialization,
equal-session source schedule, Adam recipe, 48 epochs, whole-unit U(0,1)
dropout, dense all-bin behavior loss, and final-four SWA remain unchanged.

## Sole training-method treatment

The predecessor always trained with a 30-trial calibration view.  CBM-D
marginalizes the calibration budget during source training.  For every
equal-session optimizer batch it selects an integer M from 4 through 30 using
a deterministic, host-RNG-free schedule.  It jointly supplies:

- the first M calibration-activity trials to B3S; and
- the ordinary closed-form OLS T4 carrier fitted from those same first M
  labelled trials, normalized with the sealed ordinary-OLS source moments.

The schedule covers every integer M=4..30 for every source session before
repeating.  This is a distributional training intervention, not a posterior,
ridge, teacher, new identity token, or decoder-architecture change.  At
deployment, the graph is ordinary Cell D and the chosen M is simply the
available prefix length.

## Held factors

- no teacher checkpoint or teacher forward;
- no target/within/external/formal gradients or parameter updates;
- no per-unit/session table and no additional trainable parameter;
- exact Cell-D B3S+T4 additive identity path and two-head decoder;
- exact dynamic whole-unit dropout law U(0,1), one call per optimizer step;
- B32, seed 42, 48 x 33,925 source steps, frozen warmup/cosine LR and Adam;
- equal-session source sampler and one iterator per epoch;
- checkpoints 44..47 and final-four SWA.

## Performance evaluation

Use the Phase-1 comparator authority and its identical fixed inputs.  Report
M4/M10/M30 on within-6 and external-15 for both:

1. total-calibration-limited: B3S and T4 both use M; and
2. label-limited: T4 uses M while B3S retains M30.

Primary contrasts are CBM-D minus sealed Cell D at M4 and M10.  Also report
Arm A, fixed/GCV ridge-T4, dense/classical ridge, PV, original SPINT B0 and A2
where their contracts are actually comparable.  All neural decisions use
last-bin, variance-weighted per-session R2, equal session weighting, paired
sign counts and the fixed seed-42 10,000-draw bootstrap.

## Decision

- Continue if external total-calibration M4 improves by at least +0.05 with
  at least 10/15 positive sessions and M10 is non-negative.
- A smaller positive result remains descriptive, not a breakthrough.
- Stop this exact recipe if external M4 is negative or M30 regresses by more
  than 0.03.  Do not spend a new round on loss/schedule ablations unless the
  main cell is first positive.
