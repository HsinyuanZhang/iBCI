# M2 Anchored Post-Fusion Gate V1: Source Result and Target-Control Failure

Date: 2026-09-02  
Status: immutable failed attempt; source result valid, target effect resolved by completed V2 successor  
Method: APFG, one source-trained scalar gate anchored at the selected-T4 native POOLED operator

## Executive result

APFG V1 produced a small but internally consistent positive source-validation result, selected epoch 11, and completed an all-seven-session refit. It did **not** produce a target score. The attempt failed at the first target hard control because CPU-produced `APFG-ZERO` predictions were required to have the same SHA-256 as a historical POOLED prediction generated on CUDA. That cross-device bitwise equality is not a valid numerical requirement.

Therefore V1 alone establishes only that the scalar residual direction is learnable on the pre-registered source split. Target transfer is resolved by the completed same-process V2 successor: external `+0.004612` R2 with `4/6` positive sessions, below the frozen `+0.010` promotion threshold.

## Immutable result graph

Result root:

`tfpd_exploration/results/m2_anchored_postfusion_gate_v1`

The root contains six immutable body/sidecar pairs and no terminal:

| Body | SHA-256 |
|---|---|
| `attempt.json` | `e546cb34f3efc32b8d6056e5eb6dc74877a9a548543d51a1040d149f1e5407a7` |
| `launch.json` | `9747223a8a5afed21e9ff59df3d0c44bea85c9a244c6095bf5afa5ace7c9dfef` |
| `source_authority.json` | `ae1f02c97d507c28fb35165ab616100156e279fa855cb3976d61ab9ff55c0ccf` |
| `alpha_selection.json` | `c60a9e29928a8f559881745a727d76716b2b458c1aa3d892772f6b8b65eef908` |
| `input_authority.json` | `41185bceb1307cebc1afb2d7672c55bc71d601ba028a4c0d694310d8af366e24` |
| `failure.json` | `6e979a4b8abc9761cf6dd9b521bfc7c4a08422ba9d775530bb0cecfc72a23f1d` |

Execution closure recorded by the attempt:

`879ef63ce91da2c084285e72a34a7ba0d3a443246720f6bb0afb9e67e6d70934`

Failure evidence:

```text
exception_class = tfpd_exploration.src.m2_anchored_postfusion_gate_v1.scoring.ScoringError
diagnostic_message = APFG zero POOLED prediction mismatch
error_sha256 = 59297c1eebb9e534872f780b192d5e45bf3f24a2fb7f84cee38533b894bc559b
```

The failure receipt records `source_complete=true`, `target_opened=true`, and `target_complete=true`, but no `score.json` was published. Here `target_complete` means the 13 target input records were materialized; it does not mean target scoring completed.

Two non-causal bookkeeping defects are retained rather than repaired in place: `failure.progress.stage` remains the literal `launch`, and its progress-list `published_prefix` omits the already present `input_authority.json` even though the failure's digest mapping and actual root include it. The V2 predecessor codec must validate this historical split exactly; it must not normalize old bytes.

## Source-only result

The lexical source split was five fit sessions and two validation sessions. Alpha alone was trainable; the inherited selected-T4 model remained frozen and in evaluation mode. There were no teacher forwards and no target updates.

| Quantity | Result |
|---|---:|
| Selected epoch | 11 of 12 |
| Zero UNCAPPED validation R2 | 0.6767482379 |
| Learned UNCAPPED validation R2 | 0.6789508383 |
| Mean paired delta | **+0.0022026004** |
| Positive validation sessions | **2/2** |
| Refit alpha | **-0.2075902969** |
| Fit optimizer updates | 29,832 |
| All-seven refit updates | 38,005 |

Per-session UNCAPPED validation deltas:

| Session | Learned minus zero R2 |
|---|---:|
| `ses-2020-10-27-Run2` | +0.0026953304 |
| `ses-2020-10-28-Run1` | +0.0017098704 |

The selected-epoch FIXED30 readout was descriptive and not used for selection:

| Session | Zero R2 | Learned R2 | Delta |
|---|---:|---:|---:|
| `ses-2020-10-27-Run2` | 0.6483764854 | 0.6483957291 | +0.0000192437 |
| `ses-2020-10-28-Run1` | 0.6711238199 | 0.6731752829 | +0.0020514630 |

The validation curve rose monotonically through epoch 11 and changed only slightly at epoch 12. This supports the adequacy of the fixed 12-epoch screen for the one-scalar fit; it does not establish external transfer.

The negative alpha is scientifically interpretable. With `tanh(alpha) approximately -0.205`, the learned identity extrapolates slightly away from the post-fusion identity and beyond the native pre-fusion anchor. Source data therefore does not prefer moving toward pure post-fusion averaging; it prefers a small correction in the opposite direction. Whether that direction transfers is exactly the unanswered target question.

## Why target scoring failed

The historical POOLED comparator was generated on an RTX 3090 through CUDA with batch size 1024 and TF32 disabled. APFG V1 deliberately moved the target models to CPU, also using batch size 1024, and then required prediction SHA-256 and R2 to equal the historical CUDA result exactly.

The model, checkpoint, support selection, activity stack, carrier fit, side-feature normalization, query order, targets, last-bin selection, output scale, and memory law were statically aligned. The first failing predicate was prediction SHA. CPU and CUDA may use different floating-point reduction orders, so identical scientific operators need not emit byte-identical FP32 arrays across those backends. V1 also reached the two identities through different wrapper paths (`rollout_g00m` versus `rollout_apfg_raw_pool`), so only the V2 same-process sentinel can finally distinguish harmless numerical-surface drift from a remaining operator-path mismatch.

This is a control-authority defect, not evidence that:

- APFG target R2 is lower than POOLED;
- the learned alpha failed to transfer;
- the activity-memory law is wrong;
- the selected checkpoint was loaded incorrectly.

No target R2 was accepted or published, so no target performance claim is permitted from V1.

## Completed successor

The authorized successor is:

[WORKORDER_M2_APFG_SAME_SURFACE_TARGET_SCORE_V2_20260902.md](./WORKORDER_M2_APFG_SAME_SURFACE_TARGET_SCORE_V2_20260902.md)

Its completed result is:

[RESULT_M2_APFG_SAME_SURFACE_TARGET_SCORE_V2_20260902.md](./RESULT_M2_APFG_SAME_SURFACE_TARGET_SCORE_V2_20260902.md)

It binds the full immutable V1 failure graph, reuses the sealed refit alpha without retraining, and evaluates native POOLED, APFG-ZERO, and APFG-LEARNED on one CPU numerical surface. The hard bitwise sentinel becomes same-process native POOLED versus APFG-ZERO. Historical CUDA POOLED remains provenance and descriptive context only.

The governing target promotion criterion remains unchanged in meaning: on the six external sessions, `APFG-LEARNED|UNCAPPED - same-process NATIVE-POOLED|FIXED30` must have mean delta at least +0.010 and at least 4/6 positive sessions.

V2 completed all 65 rows. The same-process native/zero controls passed exactly. The external total contrast was `+0.004612` with `4/6` positive sessions and a 95% session-bootstrap interval of `[-0.003402, +0.013194]`; it therefore did not pass promotion. The within total contrast was `+0.014002`, `7/7`, but `+0.013933` of it was the uncapped activity-memory effect and only `+0.000069` was the learned-gate increment over uncapped zero.

## Paper-safe statement

> A one-parameter anchored post-fusion gate produced a consistent but small source-validation improvement (+0.0022 R2, 2/2 sessions) and selected a negative residual coefficient. The first target attempt did not yield a performance estimate because a fail-closed control incorrectly required bitwise equality between CPU and historical CUDA predictions. We therefore treat target transfer as unevaluated and use a same-process native control in the immutable successor.

Do not describe APFG V1 as a successful target method, a target null, or a target negative result.
