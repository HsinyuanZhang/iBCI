# Work Order — M2 Anchored Post-Fusion Gate V1

Date: 2026-09-02.  Status: **no-data / no-CUDA implementation preparation
only**.  This work order does not authorize a source training run, target
materialization, checkpoint access, result-root reservation, or GPU query.

## 1. Objective

Implement the smallest source-only successor to the post-fusion screen that
keeps the selected-T4 POOLED checkpoint as an exact zero-gate anchor.  The
only trainable value is one scalar `alpha`; all inherited model, decoder,
normalizer, and T4 parameters are frozen.

The authority design is
`DESIGN_ANCHORED_POSTFUSION_GATE_V1_20260902.md`, SHA-256
`2b78e5be4de814d5f0bf00048df781c80d5bed695cad8c550ddd3ea3021d4e4f`.

## 2. Additive ownership

Only these new paths belong to this route:

- `tfpd_exploration/src/m2_anchored_postfusion_gate_v1/`
- `tfpd_exploration/scripts/run_m2_anchored_postfusion_gate_v1.py`
- `tfpd_exploration/tests/test_m2_anchored_postfusion_gate_v1.py`
- this work order.

No existing PIT, B3S, POOLED, post-fusion, data-loader, scorer, checkpoint,
or result-root file may be changed.  The separate operator-corrected scorer
is out of scope.

## 3. Frozen science contract

1. Start from the selected-T4 POOLED checkpoint only.  Strict-load it before
   installing the APFG adapter; then freeze every inherited parameter.
2. `alpha` is one scalar float32 initialized to positive zero.  Adam sees
   exactly that parameter at LR `1e-4`, with no weight decay.
3. At evaluation `alpha == +0.0` uses the native B3S branch directly.  Under
   the historical `FIXED30` law, the deployed zero sentinel must be bitwise
   equal to the sealed native POOLED row.  `ZERO/UNCAPPED` is instead the
   explicit memory-only counterfactual and is not falsely required to equal
   the bounded POOLED row.  Training must retain the differentiable residual
   expression at zero.
4. Source windows must have an endpoint strictly after completion of the
   first thirty calibration trials.  Each pool is causal for that endpoint:
   M4 is D-opt support four; M10 adds the most recent six completed
   non-support trials; M30 adds the most recent twenty-six.  No unavailable
   cardinality may be clamped, padded, repeated, or filled from the future.
5. A batch must be split by exact pool state, or use per-example calibration
   tensors.  A member list for one source coordinate cannot be reused for a
   different coordinate.
6. Selection is lexical source 5/2: first five names fit, final two validate;
   select the earliest maximum equal-session validation epoch over exactly 12
   epochs.  Refit from alpha zero on all seven source sessions for that fixed
   epoch count.  No target surface participates.

## 4. Stage-0 acceptance

The no-data candidate must provide:

- a Torch-only APFG adapter that proves zero-gate native equality, alpha-only
  gradient topology, and inherited-weight immutability on synthetic tensors;
- a pure causal scheduler proving exact M4/M10/M30 member laws and rejecting
  insufficient, future, current, duplicate, padded, or reordered members;
- pure lexical 5/2 selection/refit laws;
- an inert public CLI that cannot mint or execute a live capability;
- a static explicit closure candidate, compiled and exercised with no CUDA.

Production PIT construction, selected-checkpoint descriptor admission,
source-window materialization, GPU0 preflight, immutable receipt lifecycle,
and target scoring remain a follow-on post-audit phase.  They must reuse the
existing selected-T4 POOLED/PIT stack rather than copy it.

## 5. Explicit non-goals

This phase must not access source or target data, checkpoint tensors, CUDA,
GPU0/GPU1, or canonical result roots.  It must not run a train loop or create
a result receipt.
