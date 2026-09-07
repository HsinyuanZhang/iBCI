# Result — M2 Post-Fusion Checkpoint Score V2 (2026-09-02)

> **2026-09-02 operator-correction addendum.** The required fixed-checkpoint
> rescore is now complete.  Whole-stack evaluation changed PF-R1 external
> UNCAPPED R2 from `0.1448` to `0.2384` (`+0.0936`) and PF-R50 from `0.1555`
> to `0.2335` (`+0.0781`).  Relative to PF-MEAN, the corrected effects are
> respectively `+0.0515` (4/6, 95% CI `[+0.0024,+0.1031]`) and `+0.0466`
> (5/6, `[+0.0059,+0.0868]`).  This confirms that the old residual rows
> materially underestimated the learned operator.  Both corrected arms still
> fail against the strong sealed POOLED comparator: PF-R1 `-0.0607` (1/6)
> and PF-R50 `-0.0655` (1/6), with both bootstrap intervals below zero.
> Therefore the historical V2 rows below remain provenance, not the final
> learned-residual result.  See
> `RESULT_M2_POSTFUSION_OPERATOR_CORRECTED_SCORE_V1_20260902.md`.

Status: **HISTORICAL V2 TERMINAL; PF-MEAN IS VALID; PF-R1/PF-R50 HAVE NOW BEEN
SUPERSEDED BY THE COMPLETED OPERATOR-CORRECTED SCORE**.

The fixed 78-row checkpoint score completed successfully. All three literal
rows failed the pre-registered primary external nomination gate against the
sealed POOLED k4 comparator. A post-result operator audit subsequently proved
that PF-MEAN was evaluated under the same placement algebra used in training,
but PF-R1/PF-R50 were not: their scorer averaged M=1 residual identities even
though training formed each residual around the native whole-pool identity.
The only coherent positive effect was confined to the seven source/within
sessions and did not transfer to the six local external sessions.

This result does not evaluate the official M2 submission surface and does not
contain a newly trained matched pre-fusion control. It is decisive negative
evidence for the implemented PF-MEAN route. PF-R1/PF-R50 remain immutable
descriptive rows but are non-governing for the learned-residual claim until a
whole-pool, training-matched rescore is completed. See
`AUDIT_M2_POSTFUSION_TRAIN_SCORE_OPERATOR_MATCH_20260902.md`.

## 1. Direct conclusion

The simplest arm, PF-MEAN, remained the best of the three Post-Fusion models,
but its external score was substantially below the historical POOLED
activity-memory comparator:

```text
PF-MEAN / UNCAPPED external mean R2       0.1869045
sealed POOLED k4 external mean R2          0.2990573
paired delta                              -0.1121529
positive sessions                         1 / 6
session-bootstrap 95% interval            [-0.2283604, -0.0324055]
```

PF-R1 and PF-R50 literal V2 rows were worse. No literal row approached the
required `+0.010` and `4/6` gate. The immutable V2 decision is:

```text
NO_NOMINATION_STOP_POSTFUSION_ARCHITECTURE_AXIS
```

Therefore this result does not authorize more epochs, a wider residual
adapter, or a target-guided Post-Fusion search. It does authorize only a fixed
operator-correction audit of the already trained residual checkpoints; that
audit adds no training and cannot change PF-MEAN.

## 2. Full equal-session means

| arm | external FIXED30 | external UNCAPPED | within FIXED30 | within UNCAPPED |
|---|---:|---:|---:|---:|
| PF-MEAN | 0.1869045 | 0.1869045 | 0.6627158 | **0.6796550** |
| PF-R1 | 0.1447733 | 0.1447733 | 0.4833119 | 0.5015923 |
| PF-R50 | 0.1554955 | 0.1554955 | 0.4985678 | 0.5198261 |

The source-only training monitor had ranked PF-MEAN, PF-R1, and PF-R50 within
approximately `0.0034` R2 of one another. The literal V2 external score places
PF-R50 `-0.03141` and PF-R1 `-0.04213` below PF-MEAN under UNCAPPED. These two
residual deltas must not be interpreted as a clean residual-gate failure: the
post-result audit measured material train/score graph differences (relative
L2 approximately `0.25--0.37`). PF-MEAN has exact whole-stack versus
per-trial-mean parity at M4/M10/M30 and remains the clean result.

## 3. Primary external contrasts versus POOLED

All values below are `PostFusion UNCAPPED - sealed POOLED k4` on the exact
same six local-external session surfaces.

| arm | mean delta | median delta | positive sessions | session-bootstrap 95% interval |
|---|---:|---:|---:|---:|
| PF-MEAN | **-0.1121529** | -0.0846008 | 1/6 | [-0.2283604, -0.0324055] |
| PF-R1 | -0.1542841 | -0.1202284 | 1/6 | [-0.2450645, -0.0687352] |
| PF-R50 | -0.1435619 | -0.0888863 | 0/6 | [-0.2445329, -0.0535543] |

PF-MEAN per-session deltas were:

| external session | delta versus POOLED |
|---|---:|
| ses-2020-10-30-Run1 | -0.0302359 |
| ses-2020-10-30-Run2 | -0.0836844 |
| ses-2020-11-18-Run1 | -0.0855171 |
| ses-2020-11-19-Run1 | +0.0078149 |
| ses-2020-11-24-Run1 | -0.3874738 |
| ses-2020-11-24-Run2 | -0.0938210 |

This is not a marginal miss caused by the `+0.010` threshold: the mean effect
is large and negative, five of six sessions regress, and the descriptive
session-bootstrap interval excludes zero.

## 4. What the positive within result does and does not mean

UNCAPPED continual identity memory improved every within session relative to
FIXED30:

| arm | UNCAPPED - FIXED30 within mean | positive sessions | bootstrap 95% interval |
|---|---:|---:|---:|
| PF-MEAN | +0.0169392 | 7/7 | [+0.0128403, +0.0213211] |
| PF-R1 | +0.0182804 | 7/7 | [+0.0129382, +0.0238043] |
| PF-R50 | +0.0212583 | 7/7 | [+0.0158587, +0.0263792] |

PF-MEAN UNCAPPED was also `+0.0255688` over POOLED within, with `7/7`
positive sessions and bootstrap interval `[+0.0144320, +0.0384555]`.

This establishes a narrow mechanism fact: once long source-session histories
cause the fixed pool to evict completed-query identities, retaining all
Post-Fusion identities can improve the within-session score. It does **not**
establish cross-session transfer. On the external surface, FIXED30 and
UNCAPPED were exactly equal for every arm because those sessions never reached
the first informative eviction; therefore the external result tests the
trained Post-Fusion identity itself, not a capacity difference.

The combined pattern is the pre-registered PIT-M2 overfit signature:

```text
within gain + external loss
```

## 5. Scientific interpretation

The negative result distinguishes two ideas that should no longer be merged:

1. **Continual activity memory remains viable.** Historical POOLED activity
   memory is still the stronger external reference. This experiment does not
   invalidate causal activity FIFO/CDM.
2. **The trained PF-MEAN readout is not supported.** It loses substantially on
   local external sessions under an algebraically matched scorer.
3. **The residual variants are unresolved, not successful.** Their current
   rows are negative, but the scorer used a different residual composition
   than training. No claim about residual capacity is valid until the fixed
   whole-pool rescore.

The likely interpretation is that the trainable Post-Fusion readout adapts to
source-session identity geometry. It can exploit longer within histories but
does not preserve the cross-session invariances carried by the sealed POOLED
construction. The all-negative residual directions observed after training
were therefore not a transferable correction.

The proper paper use is as an ablation/stopping result, not a main positive
method claim:

> Pooling learned identities after a nonlinear readout improved long-history
> within-session decoding but failed to transfer across sessions; the robust
> design remains causal pooling of activity evidence before the frozen
> identity readout.

Because no matched pre-fusion T0 was trained, absolute Post-Fusion-versus-
POOLED deltas remain an unmatched-training comparison. That limitation makes
positive attribution harder, but it does not rescue a route whose primary
external effects are uniformly far below the nomination threshold.

## 6. Runtime and immutable evidence

The first V1 scoring attempt failed before held-out construction because a
legacy dependency used an absolute `src...` import under the qualified
`tfpd_exploration.src...` package context. It produced zero rows and no R2.
That exact failure graph remains immutable. V2 changed only the import/lifecycle
seam and reused the same checkpoints, physical 78-row scorer, inputs, metrics,
memory laws, and POOLED comparator.

The successful V2 run was CPU-only and completed without retry:

```text
canonical root
  tfpd_exploration/results/m2_postfusion_checkpoint_score_v2

CPU decode batch size                         1024
first external session / six-row smoke        6.4704 s
projected full score                           84.1151 s
hard cap                                        7200 s
total rows                                      78
CUDA initialized                                false
parameter updates                               0
target updates                                  0
terminal / failure                              1 / 0
```

The canonical root contains exactly ten regular immutable leaves: five JSON
bodies and their basename-bound SHA-256 sidecars. Every leaf is `0444`, has
link count one, and independently rehashes to its sidecar:

```text
attempt.json
6bada93b3f6273a04866c0878f5367b9a9fcae4e5040e2a4ab2fcee7bb059bc5

launch.json
4e88d2b2df57430e36fcb15a5ab948ddb267534599f7507c2b53f65773a44e70

input_authority.json
92e85b8d0c1b27b4b40c857f4459969c76907e4a630057c5e697792f49cb657e

score.json
9dce9042360a4d835859532bd3ad97c0d586d74e25499a11f2898e2fce6adfef

terminal.json
75b5719e9cb46fa2c50e45131049ba1903a70a8923849f354a78b2f39eefd2f1
```

The 47-file V2 execution closure is:

```text
68d489aacd23611c625f98fee919430451fad251d3b681e0d01771bf546f982d
```

The terminal exactly links the successful screen, all three checkpoint bodies,
the historical POOLED score
`455485bd854a36392f17ec5ed029b1a4044a01a8dd7dfeae95c7763b8327f5f6`,
and the immutable V1 import-failure graph. Independent recomputation of all 78
rows' summaries, paired contrasts, bootstrap intervals, and nomination object
matched `score.json` exactly.

## 7. Decision and next work

Do not spend a second GPU run on more Post-Fusion epochs, residual widths, or
M10/M30 expansion. First perform the CPU-only whole-pool residual rescore
defined in `AUDIT_M2_POSTFUSION_TRAIN_SCORE_OPERATOR_MATCH_20260902.md`, with
PF-MEAN exact reproduction as a mandatory control. If both corrected residual
arms remain below POOLED, close the architecture axis. A matched pre-fusion T0
becomes worth considering only if that corrected rescore first recovers
external performance.

For the M2 paper path, retain the positive, externally supported components:

- the sealed T4/POOLED activity-memory result;
- causal activity-memory/CDM and its capacity/exposure analysis;
- PACD if its independent M2 score supports it;
- the explicit boundary that nonlinear learned Post-Fusion pooling overfits
  source-session geometry.

The next GPU budget should go to a different M1/M2 mechanism with a real
external comparator, not to Post-Fusion recovery.
