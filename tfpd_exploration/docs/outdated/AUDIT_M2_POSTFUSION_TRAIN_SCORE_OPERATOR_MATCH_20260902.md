# Audit — M2 Post-Fusion Train/Score Operator Match (2026-09-02)

> **Resolution.** The prescribed whole-stack rescore has completed.  It
> recovered `+0.0936` external R2 for PF-R1 and `+0.0781` for PF-R50 relative
> to their mismatched V2 rows.  Corrected R1/R50 beat PF-MEAN by `+0.0515`
> (4/6) and `+0.0466` (5/6), respectively, proving that the mismatch was
> material and that the learned residual contains signal.  They nevertheless
> remain `-0.0607` and `-0.0655` below sealed POOLED, each with only 1/6
> positive external sessions.  See
> `RESULT_M2_POSTFUSION_OPERATOR_CORRECTED_SCORE_V1_20260902.md` for the
> immutable terminal and final interpretation.

Status: **RESOLVED; PF-MEAN V2 SCORE VALID; PF-R1/PF-R50 V2 ROWS SUPERSEDED BY
THE COMPLETED WHOLE-STACK OPERATOR-CORRECTED SCORE**.

## 1. Why this audit was required

The completed V2 score reported a large external loss versus POOLED:

```text
PF-MEAN  -0.1121529
PF-R1    -0.1542841
PF-R50   -0.1435619
```

Because this loss was much larger than the expected identity-axis headroom,
the result was audited for checkpoint reconstruction, T4 units, same-input
alignment, training exposure, and exact train-versus-score algebra before
interpreting it as a general failure of Post-Fusion learning.

## 2. What is not broken

The following checks passed:

- every adapter was installed before strict checkpoint loading;
- every loaded student-state digest equals the immutable screen descriptor;
- the scoring model remained frozen before and after every row;
- T4 is converted from Hz to expected counts per 20 ms bin and normalized by
  the same source-frozen statistics as the sealed POOLED route;
- PF and POOLED rows share exact query-start, target, and window-count
  authorities;
- all 78 R2 values, summaries, paired contrasts, bootstrap intervals, and the
  nomination decision recompute exactly;
- no CUDA, optimizer, backward pass, parameter update, or target update
  occurred.

There is no evidence for a checkpoint omission, a factor-of-50 T4 unit bug, or
a query/target alignment bug.

## 3. Training and deployment distributions are not matched

The source screen trained on static chronological prefixes with cardinality
cycle:

```text
M = 30, 10, 4
indices = range(M)
```

It never trained on D-opt-selected support, causal query-memory growth, or
arbitrary post-calibration trial identities. Its source selector also used a
full-block, in-sample monitor whose seven sessions and 1,011 coordinates were
contained in training.

The score called `B30 / D-opt-k4` is more precisely:

```text
four D-opt-selected activity identities
+ an M30 ridge T4 carrier
+ causal completed-query identities
+ FIXED30 or UNCAPPED identity memory
```

Training slices only the activity tensor when cycling `30/10/4`; the T4 side
tensor stays fixed. Scoring instead pairs D-opt-M4 activity with the M30 ridge
carrier. These are real support/cardinality/carrier distribution differences,
not merely different labels for the same input.

The absence of the deliberately omitted matched pre-fusion T0 means that the
absolute PF-versus-POOLED delta also combines placement, cyclic-prefix
training, source selection, and checkpoint-lineage effects. It is a valid
same-input deployment comparison, but not a clean causal estimate of placement
alone.

## 4. Exact algebra audit

For one real source session and each immutable trained checkpoint, the audit
compared:

```text
A = adapter.forward_batch(the complete M-trial stack)

B = arrival_order_mean(
      adapter.forward_batch(each individual trial with M=1)
    )
```

The test used the actual source activity geometry `[M,100,96]`, the actual
M30-derived normalized T4 side tensor, `eval()`/inference mode, CPU only, and
no behavior target or R2.

### PF-MEAN

| M | bitwise equal | max abs | relative L2 |
|---:|---:|---:|---:|
| 4 | true | 0 | 0 |
| 10 | true | 0 | 0 |
| 30 | true | 0 | 0 |

PF-MEAN therefore uses the exact same trained placement arithmetic in the
scorer. Its large external loss is not caused by decomposing a pool into
per-trial calls.

### PF-R1

| M | max abs | RMS difference | relative L2 | cosine |
|---:|---:|---:|---:|---:|
| 4 | 2.0481 | 0.1584 | 0.2885 | 0.9576 |
| 10 | 2.8058 | 0.1918 | 0.3673 | 0.9320 |
| 30 | 2.1008 | 0.1814 | 0.3670 | 0.9312 |

### PF-R50

| M | max abs | RMS difference | relative L2 | cosine |
|---:|---:|---:|---:|---:|
| 4 | 1.7815 | 0.1388 | 0.2522 | 0.9677 |
| 10 | 1.8210 | 0.1504 | 0.2889 | 0.9575 |
| 30 | 1.6300 | 0.1626 | 0.3302 | 0.9439 |

These are material differences, not floating-point tolerance effects.

## 5. Source of the residual mismatch

Training evaluates the residual arms on an M-trial stack as:

```text
native_pool(M)
+ g * (postfusion_mean(M) - native_pool(M))
```

where `g = tanh(alpha)`. The completed scorer instead evaluated each trial at
M=1 and then averaged:

```text
mean_i(
  native_single(i)
  + g * (postfusion_single(i) - native_single(i))
)
```

The post-fusion mean commutes with the final averaging, but the native B3S
branch does not:

```text
mean_i(native_single(i)) != native_pool(M)
```

because native B3S pools pre-features before a nonlinear post-pool network.
Consequently PF-R1/PF-R50 were scored under a graph that they were not trained
to implement. Their V2 rows remain immutable evidence of that literal scorer,
but they are non-governing for the claim that a learned residual gate failed.

## 6. Correct interpretation of the completed score

The evidence supports the following narrow conclusions:

1. The implemented PF-MEAN training/deployment combination fails external
   transfer. Its `-0.1121529` versus POOLED is valid for that route.
2. PF-MEAN improves within under UNCAPPED memory (`+0.0255688` versus POOLED,
   `7/7`) while losing strongly external (`1/6` positive). This is evidence of
   source/within adaptation without cross-session generalization.
3. Training did not repair the frozen Post-Fusion penalty. The frozen probe was
   `-0.0826453` versus POOLED; the trained PF-MEAN result is approximately a
   further `-0.0295076` below it on the same headline scale.
4. PF-R1/PF-R50 cannot yet decide whether a learned interpolation/extrapolation
   toward the native branch is useful, because their train and score graphs
   differ materially.
5. Nothing here invalidates pre-fusion causal activity memory/CDM. POOLED
   remains the stronger external construction.

The result does **not** prove that every possible Post-Fusion network is
incapable. It proves that the current PF-MEAN route, trained with static
chronological prefix cycling and selected by an in-sample full-block monitor,
does not transfer under D-opt-M4 plus causal query-memory deployment.

## 7. Minimal next action

Do not retrain yet. First run one fixed, CPU-only operator-correction score for
the already frozen PF-R1/PF-R50 checkpoints:

```text
for each causal current pool:
  evaluate adapter.forward_batch(the whole current activity stack)
```

This exactly matches the residual graph used in training. It must retain the
same 13 input authorities, T4, query starts, targets, FIXED30/UNCAPPED
membership, checkpoints, metric, and POOLED rows. PF-MEAN is an exact control
and must reproduce every existing PF-MEAN prediction/R2 row.

Decision:

- if PF-MEAN fails exact reproduction, the correction scorer is invalid;
- if both corrected residual arms remain materially below POOLED, close the
  Post-Fusion architecture axis without another GPU run;
- if a corrected residual arm recovers external performance, only then
  consider the matched 12-epoch pre-fusion T0 needed for causal attribution.

This is an audit correction, not a new hyperparameter search. It should take
minutes on CPU and consumes no new training budget.
