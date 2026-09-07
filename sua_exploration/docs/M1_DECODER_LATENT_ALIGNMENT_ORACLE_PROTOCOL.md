# M1 decoder-latent alignment oracle: CPU/inference feasibility protocol

**Frozen:** 2026-08-02 (Asia/Hong_Kong)  
**Status:** CPU/inference feasibility only. This document does **not** authorize an oracle fit,
decoder training, GPU launch, checkpoint selection, report evaluation, or EvalAI action.

## Question and distinction from stopped E4 ridge

The stopped M1 E4 Gate-A ridge asked whether a support carrier predicts a **future neural-rate
target**. Its target was category means from trials `[210,end)`; it was an explicitly disclosed
later-label oracle and did not pass its content-specific utility gate. It therefore cannot be
recast as a decoder mechanism result.

This proposed oracle has a different, narrower question: can a calibration-only, condition-aware
mapping align to the *already frozen decoder's own identity coordinate*? No future rate, behavior,
query activity, or query label forms its target.

For a frozen M1 teacher with identity modules `f_in=fc_id_in` and `f_out=fc_id_out`, and only the
chronological support tensor `C in R^[1,10,T,N]`, define the canonical target

```text
Phi_mi = f_in(C_mi)                 # T -> H, per trial/unit
E*_i    = f_out( mean_m Phi_mi )    # H -> W, per unit
```

`E* in R^[N,W]` is exact teacher identity at the additive input of the same frozen decoder:
`fc_in(x_i + E*_i)`. Thus it is in a fixed decoder-compatible coordinate, unlike PCA/loadings or
an arbitrary encoder hidden basis. The feasibility audit must prove this exact source path and
shape against the frozen checkpoint. The target has no label argument.

## Deployment boundary

The only semantic labels allowed to a future mapping are `obj_id[0:10]` from the calibration NWB.
They may form a static support condition feature. It must not receive any query-trial `obj_id`,
`tgt_loc`, `tgt_obj`, `condition_id`, behavior, future rates, or hidden EvalAI data.

This feasibility audit uses the four local M1 **held-in-calib** source files only. It does not
open held-out calibration/query files. It uses no report-window metric. The frozen teacher target
is an offline supervisory/oracle quantity, not a deployment metric and not decoder R².

## Proposed oracle family (not yet authorized to fit)

Let `E0` be an identity/no-alignment reference and let `F(S)` be a support-only per-unit feature
map. A rank-`r` decoder-coordinate residual is

```text
Delta E = A(S) B^T,  A in R^[N,r], B in R^[W,r], r in {1,2,3}
E_hat = E0 + Delta E.
```

The rank applies directly to `[N,W]` in the canonical decoder identity coordinate; it is not a
rank in an unanchored encoder hidden space. The only permitted rank grid is `{1,2,3}`. The only
permitted regularization candidates are a finite predeclared positive lambda grid. Rank and lambda
must be selected by nested leave-one-source-session-out **using only outer-train sessions**. The
outer left-out source session cannot affect rank, lambda, normalization, orthogonalization, or
feature statistics. A future formal oracle needs a separate reviewed protocol before any fitting.

## Required controls

All controls preserve the same target `E*`, rank/lambda grid and source-session split:

1. **identity/no-alignment** — `E_hat=E0`; no condition feature.
2. **rate-only** — features from support rates/exposure only; no `obj_id` assignment.
3. **condition-label shuffle** — deterministic non-identity permutation of the ten support
   labels, preserving the label multiset; complete feature construction follows the shuffled labels.
4. **orthogonal-only** — the condition feature residual after projection on rate-only features
   fitted on outer-train rows only. It tests condition information not linearly recoverable from
   rate/exposure summaries.

Condition-label shuffle is not a channel-row shuffle. It disrupts support condition assignment;
any later decoder mechanism test would need its own attachment control.

## Feasibility gates and fail-closed rules

The present audit may report `canonical_target_defined=true` only if all hold:

- CPU-loaded frozen checkpoint has the expected `net.fc_id_in.*` and `net.fc_id_out.*` tensors;
- the reconstructed path produces finite `[1,N,W]` targets from `[1,10,T,N]` held-in calibration
  tensors, with `N=64` and the teacher's identity width `W`;
- only calibration support `[0,10)` reaches this path; changing trials `>=10` leaves `E*` bitwise
  unchanged;
- all four support `obj_id` levels `{1,2,3,4}` are present for every source session;
- input manifest contains only the frozen checkpoint, static source files, D4 helper, and four
  held-in-calib NWBs.

Otherwise emit an immutable fail-closed receipt and stop. Even a pass means only that the target is
well defined. It is not evidence that a learned mapping improves behavior, decoder R², or
cross-session generalization; it does not authorize a formal oracle or GPU work.

## Frozen exclusions

No future-rate target; no report/query data; no label budget above M=10; no rank above three; no
post-hoc rank/lambda sweep; no FiLM/attention/value residual; no decoder/core-model modification;
no reuse of this audit for D4 rescue or official held-out selection.
