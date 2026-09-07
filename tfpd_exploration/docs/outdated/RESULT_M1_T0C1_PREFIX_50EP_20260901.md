# Result: M1 T0/C1 prefix, 50-epoch warmup+cosine lane

Date: 2026-09-01 (terminals 2026-09-02 00:36 HKT)

Status: all five stages terminal-OK under implementation closure
`ae1f47fbb9f07a886a3c96676144793b0abe646b83336a98e030b269be84f1e7`
and pair-spec `f75734ee7e961eb0e2dc4ab7a6aa391298f48aae3dbfdc2e9cf18d332d6588ca`.
This is a **cross-recipe** comparison against the sealed 20-epoch pair
(constant `lr=1e-5`, no scheduler), not a nested-horizon extension.

Root: `tfpd_exploration/results/m1_t0c1_prefix_v1_50ep/`

---

## Scope correction (2026-09-02): this is carrier-absent

This route never supplied a per-unit T3-like/T4-like carrier. Its ordinary
`FalconLitModule` forward consumed only live neural activity and
`calib_trialized_neural_features`; C1 varied only that neural calibration
prefix. Phase 3 compared `static_m10` with an **activity-only** causal FIFO.

Accordingly, every receipt and number below remains valid, but the estimand is
strictly the carrier-absent activity path. These results do **not** test or
reject an M1 EMG functional carrier, an independent carrier injection, carrier
quality cycling, or a carrier-by-CDM interaction. The corrected successor
design is
`tfpd_exploration/docs/DESIGN_M1_FUNCTIONAL_CARRIER_MEMORY_20260902.md`.

Do not summarize this result as “M1 carrier plus C1/CDM was null.” The accurate
summary is “activity-prefix cycling was null/negative in a carrier-absent M1
decoder; activity-only causal memory produced a small local cross-session
increment with a substantial held-in cost.”

---

## Terminals (all `0444`, sidecar-consistent)

| stage | status | terminal sha256 | wall |
| --- | --- | --- | --- |
| smoke | `COMPLETE_MATCHED_SMOKE_EQUALITY` | `7d09b3d1571e21bddb0fcf54deec5f2e6c5ebc439b194fc583bc7d57f7165ab8` | 11.4 min |
| t0 | `COMPLETE_MATCHED_ARM_TRAINING` | `a8270a5eb541b968ee547e8b097a68a74528fc6c9278430d733e7564a3cb2a47` | 25260 s (7.02 h) |
| c1 | `COMPLETE_MATCHED_ARM_TRAINING` | `431df977c9dcc64651395dc4fd949cfe417f1a7b9567c8c574fbfa2c47a7367b` | 23956 s (6.65 h) |
| probe | `COMPLETE_OVERFITTING_PROBE` | `638dec0aa636e03a43741f57b81d692fbc628a78ba7afcd81b91baebd7b4339d` | 25.8 min |
| phase3 | `COMPLETE_PHASE3_TABLE` | `d0b71443d54a5cd2ab8e7d8f7981c732d62b2f66dfbf314519ebf6fb6309437f` | 5.4 min |

A prior smoke+incomplete T0 under closure `76ed5cef…` was operator-retired
(not deleted) to `retired_closure_76ed5cef/` when the lane switched from
serial arms to concurrent arms. That smoke remains scientifically valid and
is stale only because the launcher gate entered `OWNED_PATHS`.

---

## 1. Smoke equality

All frozen clauses true, including the new LR channel. Both arms started from
initial model state `7a561ce0…`. Batch, RNG, and LR stream digests matched
across arms. Eval-mode dropout was bit-identical. T0 prefixes were all full;
C1 recorded the `10,5,2` cycle. GPU 1 UUID
`GPU-2220ed5d-25ea-1839-28d7-ad4dfa5f6c86`, PCI `00000000:03:00.0`.

---

## 2. Arm receipts

Five sealed checkpoints per arm at 0-based indices `{9,19,29,39,49}`. Epoch-1
abort estimates: t0 `21432 s`, c1 `22121 s`, both below `40000 s`.

Concurrency (amended 2026-09-01 on measurement): t0 started alone
(`concurrent_sibling_stage: null`); c1 admitted t0 as the sibling. Wall times
are contention-inflated and **not** comparable to the 20-epoch reference.

| arm | serial projection | concurrent wall | inflation |
| --- | ---: | ---: | ---: |
| t0 | 6.29 h | 7.02 h | 1.12× |
| c1 | 5.43 h | 6.65 h | 1.22× |

Pair wall is `max(t0,c1) = 7.02 h` versus serial `11.72 h` (saved ~4.7 h).

---

## 3. Probe curve (`static_m10` only)

Surfaces: `train_fit` = 20120926/27/28 (reporting); `val_heldout` =
20121004/17/24 (**selection**); `test_fold` = 20120924 (reporting).

| arm | ep | train_fit | val_heldout | test_fold |
| --- | ---: | ---: | ---: | ---: |
| t0 | 10 | 0.700552 | 0.382571 | 0.530749 |
| t0 | 20 | 0.768467 | 0.401330 | 0.535690 |
| t0 | 30 | 0.802245 | 0.458460 | 0.536376 |
| t0 | 40 | 0.814296 | 0.462556 | 0.531131 |
| t0 | 50 | 0.820038 | 0.476886 | 0.533764 |
| c1 | 10 | 0.707137 | 0.375427 | 0.522534 |
| c1 | 20 | 0.770956 | 0.421686 | 0.531294 |
| c1 | 30 | 0.807085 | 0.477518 | 0.539443 |
| c1 | 40 | 0.817920 | 0.473998 | 0.533259 |
| c1 | 50 | 0.821572 | 0.480992 | 0.527857 |

---

## 4. Selection and overfitting labels

`selected_epoch_1based_by_arm = {t0: 50, c1: 50}`. Selection never read
20120924 or `train_fit`. Both receipts set
`train_fit_read_for_selection: false` and `test_fold_read_for_selection: false`.

| arm | surface | verdict | argmax | drop to ep50 |
| --- | --- | --- | ---: | ---: |
| t0 | **val_heldout** (selection) | `OVERFITTING_ABSENT_WITHIN_50EP` | 50 | 0 |
| t0 | test_fold (diagnostic) | `OVERFITTING_INCONCLUSIVE` | 30 | 0.0026 |
| c1 | **val_heldout** (selection) | `OVERFITTING_ABSENT_WITHIN_50EP` | 50 | 0 |
| c1 | test_fold (diagnostic) | `OVERFITTING_PRESENT` | 30 | 0.0116 |

Both test-fold drops sit inside the `val_heldout` across-session SD at the
judged argmax (`drop_within_session_spread: true`). The verdict word is a
label on a curve shape, not an inferential test. Disagreement between
selection surface and fold is reported, not reconciled: the fold is n=1.

---

## 5. Phase-3 2×2×2 (epoch 50)

`table_selected.json` has `alias_of_epoch50: true` because both arms selected
epoch 50. Formal-benchmark flag is false.

| arm | deployment | surface | 50ep mean | 20ep mean | Δ (50−20), cross-recipe |
| --- | --- | --- | ---: | ---: | ---: |
| t0 | static_m10 | held-in | 0.820038 | 0.727024 | **+0.093014** |
| t0 | static_m10 | held-out | 0.533764 | 0.570744 | **−0.036980** |
| t0 | cdm_activity_fifo_m10 | held-in | 0.710800 | 0.659438 | +0.051361 |
| t0 | cdm_activity_fifo_m10 | held-out | 0.549378 | 0.584279 | −0.034901 |
| c1 | static_m10 | held-in | 0.821572 | 0.730644 | **+0.090928** |
| c1 | static_m10 | held-out | 0.527857 | 0.584876 | **−0.057019** |
| c1 | cdm_activity_fifo_m10 | held-in | 0.705873 | 0.662362 | +0.043511 |
| c1 | cdm_activity_fifo_m10 | held-out | 0.538965 | 0.594735 | −0.055770 |

Within this recipe, C1−T0:

| deployment | held-in | held-out |
| --- | ---: | ---: |
| static_m10 | +0.001534 | −0.005906 |
| cdm_activity_fifo_m10 | −0.004926 | −0.010413 |

CDM-FIFO minus static (this recipe): t0 held-in −0.109238 / held-out
**+0.015614**; c1 held-in −0.115698 / held-out **+0.011107**. The FIFO still
helps the fold and still costs the train-fit surface, as in the 20-epoch
pair, with a larger held-in cost under the longer cosine run.

Read in words: the 50-epoch warmup+cosine recipe **fits held-in much harder**
than the 20-epoch constant-`1e-5` pair and **loses held-out**. C1 does not
beat T0 on the fold in this recipe (it did, modestly, at 20 epochs). Probe
selection on `val_heldout` still picked epoch 50, so this is not an
interior-checkpoint save; the selection surface kept rising while the fold
did not.

---

## 6. Resource-interaction log (peer lane)

- This lane's every `nvidia-smi` invocation was `--id 1`. Physical GPU 0 was
  never queried by the 50-epoch launcher or trainer profiler.
- GPU 1 UUID pin and GPU-0 UUID refusal both fired on every stage.
- After the concurrency amendment, t0 and c1 were the only compute apps on
  GPU 1; smoke/probe/phase3 required an exclusive card and saw an empty one.
- The peer PACD route on GPU 0 independently recorded the SIGTERM retirement
  of closure `76ed5cef…` and the later successor lineage, with host PSI zero
  and no cross-root descriptors (`AUDIT_M1_PACD_CONCURRENT_ISOLATION_20260831.md`
  §§6.20–6.21). Target for this lane: **zero owner crossover**. Observed:
  zero.

---

## What this result does and does not say

It does say the matched T0/C1 pair can be trained to 50 epochs on GPU 1 with
the frozen cosine law, that `val_heldout` selects epoch 50 for both arms, and
that activity-FIFO CDM still gives a small fold gain over static at M10.

It does **not** say 50 epochs is a better M1 recipe than 20: held-out static
dropped 0.037 (T0) / 0.057 (C1) relative to the sealed 20-epoch line, and
that line used a different LR. It does not promote C1. It does not claim
inferential overfitting on the fold.
