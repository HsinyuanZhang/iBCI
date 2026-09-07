# Result — M2 Post-Fusion Operator-Corrected Score V1

Date: 2026-09-02. Status: **completed immutable CPU score; no nomination**.

## 1. Question

The first Post-Fusion score evaluated `PF-R1` and `PF-R50` by applying the
trained residual adapter to each trial independently and then averaging the
resulting identities.  Training had instead applied the residual adapter once
to the whole ordered activity stack.  Because the native B3S branch contains
nonlinear pre/post-pooling operations, those two graphs are not algebraically
equivalent.

This successor changed only that scoring operator:

- `PF-MEAN` retained the literal V2 singleton/incremental-pool computation as
  a byte-exact control;
- `PF-R1` and `PF-R50` used the trained adapter on the whole ordered causal
  activity stack at every decode-before-commit state;
- checkpoints, inputs, D-opt-k4 support, M30 ridge T4, targets, window starts,
  memory laws, metrics, and bootstrap law were unchanged;
- no parameter or target update occurred.

This is a fixed-checkpoint diagnostic correction, not a retraining result and
not a matched-training comparison with the historical POOLED checkpoint.

## 2. Immutable result authority

Result root:

```text
tfpd_exploration/results/m2_postfusion_operator_corrected_score_v1
```

The root contains exactly five immutable JSON bodies and five SHA-256
sidecars; all leaves are regular files with mode `0444` and `nlink=1`.

| Body | SHA-256 |
|---|---|
| `attempt.json` | `0f4e441c77f6c48c4358f0face4048ecf668bf8a6936818df3c279f436e4f57d` |
| `launch.json` | `10f4ed88a31f2cf09016c854dbbf82d51e1e4f5e670a05fbbb1f18fb6ccfa9aa` |
| `input_authority.json` | `f0bd485ec578b208c1dddb2d2b8f9c5c90d9677262b7debcfef4df9697d3ec89` |
| `score.json` | `156d7fab27c70bb01a7804bfdbb3434164c71dcab81ce3b1ad8a371eb169175d` |
| `terminal.json` | `f0f137c14c99da6e843b4887b4f3102610844fc0a3ee73fc0dfae4c547c41b01` |

Execution closure:

```text
95b8e9e07e39700089e8364f67b15a020d17eb723a56cb18c2222305a1b25225
```

Bound V2 predecessor closure:

```text
68d489aacd23611c625f98fee919430451fad251d3b681e0d01771bf546f982d
```

The terminal is terminal-only (`failure.json` absent), records 78 unique rows,
`parameter_updates=0`, `target_updates=0`, and `cuda_initialized=false`.

## 3. Primary results

Equal-session mean R2 under the `UNCAPPED` law:

| Arm | Old external score | Corrected external score | Correction gain | Corrected within score |
|---|---:|---:|---:|---:|
| PF-MEAN | 0.1869 | 0.1869 | +0.0000 | 0.6797 |
| PF-R1 | 0.1448 | **0.2384** | **+0.0936** | **0.6920** |
| PF-R50 | 0.1555 | **0.2335** | **+0.0781** | **0.6893** |

The corresponding old-to-corrected within changes were `+0.1904` for PF-R1
and `+0.1695` for PF-R50.  Therefore the prior residual-arm score materially
underestimated both learned residual variants.

### 3.1 Corrected residual versus PF-MEAN

| Contrast, external/UNCAPPED | Mean paired delta | Positive sessions | 95% session-bootstrap CI |
|---|---:|---:|---:|
| PF-R1 − PF-MEAN | **+0.0515** | **4/6** | **[+0.0024, +0.1031]** |
| PF-R50 − PF-MEAN | **+0.0466** | **5/6** | **[+0.0059, +0.0868]** |

The corrected residual operator is therefore not valueless: both residual
arms substantially and consistently improve over pure post-fusion averaging.
This is the main positive mechanistic result of the correction.

### 3.2 Corrected residual versus the strong POOLED comparator

The sealed POOLED equal-session external mean is approximately `0.2991`.

| Contrast, external/UNCAPPED | Mean paired delta | Positive sessions | 95% session-bootstrap CI |
|---|---:|---:|---:|
| PF-MEAN − POOLED | −0.1122 | 1/6 | [−0.2284, −0.0324] |
| PF-R1 − POOLED | **−0.0607** | **1/6** | **[−0.1403, −0.0037]** |
| PF-R50 − POOLED | **−0.0655** | **1/6** | **[−0.1506, −0.0063]** |

Neither corrected arm passes the frozen promotion gate of mean delta at least
`+0.010` and at least `4/6` positive sessions.  The receipt decision is:

```text
NO_NOMINATION_STOP_POSTFUSION_ARCHITECTURE_AXIS
```

This does **not** mean that the residual gate has no signal.  It means that
the jointly trained Post-Fusion checkpoints still fail to preserve the strong
cross-session POOLED solution, even after their deployment operator is made
consistent with training.

## 4. Independent integrity checks

Two independent audits reproduced the following facts:

1. The 26 PF-MEAN rows are exactly equal to the completed V2 rows in full-row
   JSON, including prediction SHA, R2, targets, starts, and window counts.
2. Recomputing all summaries, paired contrasts, 10,000-resample session
   bootstraps, and the nomination from the 78 rows reproduces the receipt
   exactly.
3. All 78 `(arm, law, surface, session)` keys are unique and all R2 values are
   finite.
4. Every row has identical model-state SHA before and after scoring, with zero
   parameter and target updates.
5. All three arms within each `(surface, law, session)` group share the exact
   input-authority key, target SHA, comparator key, and window count.
6. A source-only algebra audit showed whole-stack and singleton-pool identity
   equality for PF-MEAN at M4/M10/M30, and inequality for PF-R1/PF-R50 at all
   three cardinalities.  Thus the corrected branch is both necessary and
   selective.

Arm-specific identity and causal-trace digests are expected to differ because
the adapters produce different identities.  They are not input-authority
digests and must not be required to match across arms.

## 5. Interpretation and next experiment

The result separates two previously entangled conclusions:

1. **Implementation conclusion:** the original PF-R1/PF-R50 evaluation graph
   was wrong for the trained residual operator and depressed external R2 by
   roughly `0.08–0.09` and within R2 by `0.17–0.19`.
2. **Scientific conclusion:** correcting that graph reveals useful residual
   Post-Fusion information, but the jointly trained checkpoint remains about
   `0.06–0.07` R2 below the frozen POOLED champion on external sessions.

The warranted final test is APFG (Anchored Post-Fusion Gate), not another
full-network Post-Fusion retrain.  APFG starts from the exact selected-T4
POOLED checkpoint, freezes every inherited weight in evaluation mode, and
learns only one scalar gate from source-only causal pools.  Its `alpha=+0.0`
state must reproduce POOLED exactly.  This directly tests whether the positive
residual signal above can be retained without sacrificing the native
cross-session backbone.

If APFG fails its locked external gate, the Post-Fusion architecture axis is
closed.  If it passes, the defensible claim is narrow and strong: a
source-learned scalar gate adds complementary Post-Fusion information while
exactly nesting the established POOLED decoder as its zero state.

APFG has now completed under a same-process CPU control:
[RESULT_M2_APFG_SAME_SURFACE_TARGET_SCORE_V2_20260902.md](./RESULT_M2_APFG_SAME_SURFACE_TARGET_SCORE_V2_20260902.md).
It recovered the POOLED baseline and produced a positive external point
estimate (`+0.004612`, `4/6`), but missed the frozen `+0.010` promotion gate.
Accordingly, the scalar Post-Fusion architecture axis is closed as specified.
