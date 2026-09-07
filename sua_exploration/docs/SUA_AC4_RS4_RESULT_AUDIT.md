# SUA AC4-RS4 development result audit

**Status:** complete; row attachment effective  
**Date:** 2026-08-04 HKT  
**Evidence scope:** 27 training sessions + 6 reused-development validation sessions; 0 formal SUA sessions

## Result

The dedicated AC4 row-shuffle control completed all three frozen seeds with zero failures. Each
cell trained for 12 epochs and used the fixed mean of epochs 5--12 across the same six development
sessions. Result, run-metadata, and cost hashes close for all three cells. The focused test suite
passes 18/18 after completion.

| Arm | Mean R2 | Seed means |
|---|---:|---|
| aligned AC4 `[a,c,0,0]` | 0.562753 | 0.583871 / 0.550903 / 0.553486 |
| AC4-RS4 | 0.268590 | 0.298866 / 0.262352 / 0.244553 |
| Z4 | 0.326008 | 0.338131 / 0.317025 / 0.322868 |

The frozen primary contrast is:

```text
AC4 - AC4-RS4 = +0.294163 R2
positive seed means = 3/3
positive session means = 6/6
paired two-SE lower bound = +0.279252
hierarchical-bootstrap 95% interval = [+0.204932, +0.402322]
exact paired session sign-flip p = 0.03125
decision = effective
```

The secondary contrast is also informative:

```text
AC4-RS4 - Z4 = -0.057418 R2
positive seed means = 0/3
positive session means = 0/6
hierarchical-bootstrap 95% interval = [-0.097968, -0.016899]
decision = ineffective_for_practical_0.03
```

The final aggregate is
`results/sua_t4_m30_ac4_rs4_v1/aggregate_v1.json`, SHA-256
`97f369cf67f9d88da5d343fc4bc4b72f23284e96e0db70c0a5591af5580cd0a1`.

## What the experiment establishes

AC4 does not work merely because the decoder sees a population-level distribution of coefficient
values. The large aligned-minus-shuffled gap shows that the amplitude-weighted first-harmonic
coefficients must be attached to the correct unit rows. The fact that AC4-RS4 is below Z4 further
shows that a wrong functional identity is actively harmful, not equivalent to omitting identity.

Together with Experiment A, the supported development claim is:

> A compact `[a,c]` carrier retains nearly all of full T4, and its benefit depends on the correct
> unit-to-functional-coefficient correspondence.

This does not prove that pure preferred-direction phase is sufficient: PH4 failed the full-T4
non-inferiority requirement. It also does not establish formal generalization, because the six
formal SUA sessions remain unopened.

## Execution and provenance audit

The immutable prelaunch receipt is
`results/t4_m30_ac4_rs4_prelaunch_v1_20260804/receipt.json`, SHA-256
`425d2b9818db82346ce21d31d0a00766636a247fefa07c81e392155823b4b451`.
It binds the 27/6 strict manifest, M30 boundary, three seeds, train-only normalizer, and a
session-and-seed-salted deterministic nonidentity row permutation for all 33 loaded sessions.

The sealed scheduler consumed the authorization and then stopped before any cell because of an
unbound Bash local-variable expansion. No checkpoint, score, or status existed at that point. A
hash-recorded recovery wrapper validated the exact existing claim and absence of all cell
artifacts, then called the unchanged sealed cell runner. This incident changes orchestration only;
the model, data, seeds, scorer, and decision rule remained frozen.

Measured fit wall-clock times were 6726.55, 6671.79, and 5611.14 seconds. Peak allocated CUDA
memory was approximately 415 MB per cell; peak reserved memory was approximately 1.18--1.21 GB.

## Frozen next action

The effective primary result opens Experiment B's **source-only CPU estimator selection**. It does
not authorize a GPU estimator run or formal test. The three predefined candidate families must be
evaluated by nested 27-source-session LOSO with no development-session input. At most one candidate
may survive the frozen predictive-deviance, split-half-reliability, convergence, leakage, and cost
gates. If none passes, the estimator branch stops. If exactly one passes, it requires a new
independent prelaunch receipt before any GPU work.
