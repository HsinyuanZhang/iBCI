# C2 source-only disposition audit — 2026-08-05

**Decision:** `CLOSED_CURRENT_SCOPE_NO_UNIQUE_SOURCE_LAMBDA`  
**GPU authorization:** none; zero C2 runs  
**Data boundary:** read-only code/receipt audit; no development score, formal raw/test/score, or
GPU was opened by this audit

## 1. Question

Fresh C1 trained one shared weight set on paired SUA and deterministic pseudo-MUA batches with

```text
L_C1 = 0.5 L_task(SUA) + 0.5 L_task(pseudo-MUA)
lambda_consistency = 0.
```

C1 passed its frozen non-inferiority and T4-row-attachment gates. The conditional C2 idea would
add the asymmetric source-training regularizer

```text
lambda * || y_hat_pseudo - stopgrad(y_hat_SUA) ||^2,
```

while leaving target-session calibration and inference unchanged. This audit asks whether a
pre-existing, unique, development-blind source-only rule already selects one positive `lambda`
and therefore licenses C2.

## 2. Evidence inventory

The protocol does not freeze a selector. It offers two mutually conditional future possibilities:

1. use `lambda=0.05` only after the task/consistency loss normalization is explicitly frozen; or
2. otherwise select one lambda through nested analysis on the 27 source sessions.

Neither condition was completed. There is no recorded definition of the consistency reduction and
normalization, candidate grid, nested folds, selection statistic, practical gate, tie-break, or
source-only selector receipt. Consequently `0.05` is a proposal whose prerequisite is absent, not
an already selected hyperparameter.

The immutable fresh C1 receipt binds `lambda_consistency=0` and only these twelve cells:

```text
{separate_sua_t4, separate_pseudo_mua_t4,
 shared_t4, shared_ts4} x {42,43,44}.
```

No C2 cell, selector, prelaunch, aggregate, or result root exists. PTQ and QAT are downstream
deployment studies of the frozen C1 `shared_t4` checkpoints; they do not select or authorize a C2
training objective.

The executable C1 path also rejects nonzero lambda by construction:

- `streaming_calibration_exp/src/models/paired_view_c1_module.py` rejects it in the constructor
  and again at training execution;
- `sua_exploration/mc_maze/paired_view_c1.py` rejects it in the paired datamodule;
- `sua_exploration/scripts/train_paired_view_c1_dandi688.py` instantiates only `0.0`;
- the C1 runner, verifier, and finalizer require exact `0.0` metadata;
- current tests verify two sequential half-weight task-loss backwards, not a prediction-level
  stop-gradient consistency term.

This is intentional fail-closed behavior: C2 is a new offline source-training treatment, not a
runtime flag hidden inside the completed C1 matrix.

There is also no unique C2 comparator topology. The original protocol specifies
`MVlambda-T4 x 3` plus `MVlambda-TS4 x 3`, whereas the later post-run audit proposes only three
`C2_shared_t4` runs and explicitly omits TS4. No newer signed C2 receipt resolves the conflict.

## 3. Why C1 cannot select C2 retrospectively

C1 never computed the proposed prediction discrepancy, so it contains no source distribution or
normalization scale for that loss. Its positive result proves shared-weight dual-view
non-inferiority and strong correct-row T4 content at `lambda=0`; it does not estimate the effect of
positive consistency regularization.

After seeing the C1 development endpoint, creating a new candidate grid, normalization, selector,
or comparator matrix would be a new hypothesis. Letting the six reused development sessions choose
any of those quantities would be an explicitly forbidden endpoint rescue. The choice cannot be
made identifiable by calling `0.05` a default after its stated normalization prerequisite failed to
materialize.

## 4. Frozen disposition

```text
C2 status                 = CLOSED_CURRENT_SCOPE_NO_UNIQUE_SOURCE_LAMBDA
C2 executable path        = absent
C2 selector receipt       = absent
C2 prelaunch/result root  = absent
C2 GPU runs               = 0
C2 external candidate     = false
```

This closes C2 for the current sub-C/C1 scope. It is not a negative C2 R2 result: no positive-lambda
model was trained. It is a pre-execution identifiability and governance NO-GO.

Reopening on independent data would require a genuinely new, pre-endpoint protocol that fixes:

- the exact prediction-level loss and stop-gradient direction;
- task/consistency normalization;
- source-only candidate family, nested split, statistic, threshold, and tie-break;
- one unambiguous T4/TS4 comparator matrix;
- implementation tests, cost receipt, source map, fresh prelaunch, and new result namespace.

Those requirements do not authorize such a program now.

## 5. Publication consequence

Closing C2 preserves the clean C1 claim:

> One shared offline weight set supports sorted SUA and its deterministic electrode-pooled
> pseudo-MUA view without material loss, while supervised target-session T4 calibration remains
> backprop-free.

The paper must not attribute C1 to consistency regularization, teacher-student alignment, or a C2
mechanism. `lambda_consistency=0` should be stated explicitly. The real native-MUA claim remains the
responsibility of the matched native-M2 Phase-C experiment, not a pseudo-MUA consistency rescue.

## 6. Evidence snapshot

```text
T4_NEXT_EXPERIMENT_PROTOCOL_V2.md
  ddd4b1ae31754bf5480148c24f9a514e64db1b4e8e04cf515f433aab62071f1c

C1_POSTRUN_MECHANISM_CLAIM_AUDIT_20260804.md
  ce959af02bf59e4ddbbe3d14b0fc59d2b1266f47b0e20a6abb7763e480f5c4df

streaming_calibration_exp/src/models/paired_view_c1_module.py
  16d3c318a51c8b34e1a44f725e03460446f9dee4b3130ba39d4439613211efa9

sua_exploration/mc_maze/paired_view_c1.py
  0978643d1ce90bb66733610ca0132548bfe89a5ee0a98828885a9c3a489e9c0a

fresh C1 prelaunch receipt
  8b17c19515fa0e6cc122233fd20cf7e87a616287fa247e5eacb136d6a56c9d85

fresh C1 aggregate
  32ebde0b145c63c09c1bdbc67e48582db3e2ad588e70cb5a1d52914352607e31
```
