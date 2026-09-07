# Root review — three proposed SUA/T4 continuations

**Date:** 2026-08-04  
**Scope:** analysis only. This review does not authorize GPU training, development/formal-test
access, EvalAI submission, or revival of a stopped historical branch.

## Executive decision

The three paths do not have equal evidence.

| Rank | Path | Decision | Reason |
|---:|---|---|---|
| **1** | Paired SUA/pseudo-MUA co-training | **Best next GPU candidate after a fresh provenance preflight** | It directly addresses the shared-encoder objective, has strong separate-view full-T4 substrate evidence, adds no deployment state, and has not received a scientific negative result. The current blocker is historical source provenance, not accuracy. |
| **2** | Causal cross-budget T4 correction | **Defer; only revive on an independent subject with a more constrained equivariant hypothesis** | Low-budget `(a,c)` noise is real, but the existing `q_unit+M` signal collapsed on target-free development and the subsequent analytic EB/2H/Poisson estimator selection produced no winner. A generic correction MLP on the same sub-C development split would be post-hoc continuation of a failed family. |
| **3** | Fixed-K temporal prototype K/V memory | **Stop as stated** | The fixed-K carrier beat rate-only, but a simpler order-invariant B20 marginal carrier beat it in 4/4 M1 sessions. Cross-attention would add state and a new network-side path without evidence that temporal order is the useful content. It also cannot be attached to a frozen decoder “without retraining” unless that decoder was pretrained to consume the memory. |

The recommended next program is therefore a clean Phase-C1 paired-view experiment, not another
T4 estimator and not prototype cross-attention.

## 1. Path 1 — causal cross-budget T4 correction

### What is correct in the hypothesis

The measured low-budget problem is real. Across the existing cross-budget audit, `(a,c)`
split-half reliability rises from approximately `0.59--0.62` at `M=10` to `0.86--0.87` at
`M=50`, while `b` is already approximately `0.99`. A forward-only source-trained correction is
also deployment-compatible: it would add no calibration backpropagation and could keep descriptor
state at four floats per unit.

### What has already failed

This path substantially overlaps two completed negative gates.

1. The Step-2A `q_unit+M` audit predicted `T4@M -> T4@50` scalar descriptor error. It improved
   nested source LOSO by `-0.7671` MSE units in 21/27 sessions, but the incremental advantage on
   six target-free development sessions fell to `-0.0226`, only 3/6 favorable, with interval
   `[-0.3268,+0.2817]`. The known M-only budget effect remained detectable in all 6/6 sessions.
2. Experiment-B v7 tested three source-only estimator upgrades at `M=30`. EB-ridge was the closest,
   with partial mean prospective-deviance ratio `0.976601`, but reliability delta was only
   `+0.000124`; it accumulated eight joint failures by fold 16. Second harmonic and Poisson IRLS
   were worse. The frozen winner was `no_winner_no_gpu`.

The earlier proxy was narrower than the newly stated idea: it predicted scalar error magnitude,
not signed `(a,c)` residual, and omitted global direction balance/design condition. Therefore the
entire mathematical possibility is not disproved. However, those omissions do not justify fitting
a wider MLP after observing the same six development sessions.

### Only defensible future version

If this family is revisited on an independent subject/data boundary, the correction should not be
an unconstrained four-coordinate MLP. Preferred-direction coordinates rotate, so an unconstrained
source map can learn coordinate-frame artifacts. The smallest defensible hypothesis is an
SO(2)-equivariant complex shrinkage:

```text
z = a + i c
s = f_source_only(M, residual, exposure, direction histogram, design condition, |z|)
z_corrected = s * z
b_corrected = b
```

An optional phase correction must itself be equivariant and separately controlled. This version
asks whether causal fit geometry predicts the amount of radial shrinkage; it does not pretend to
infer a universal signed preferred-direction residual. It remains low priority because EB already
tested a closely related shrinkage principle and did not improve reliability at M30.

### Kill rule

Do not run a decoder unless a nested source audit and one target-free application show a
predeclared improvement against ordinary AC4 in both coefficient error and split-half reliability.
Do not use the six sealed formal SUA sessions to select the correction. On current sub-C, the
generic correction-map branch remains closed.

## 2. Path 2 — fixed-K temporal prototype memory

### What is attractive

The proposal is label-free at deployment and could carry activity dynamics absent from static
AC4. The original M1 source proxy was large: fixed-K prototype minus rate-only was `+0.100526`,
positive in 4/4 sessions, with strong split-half repeatability. This is why the idea initially
looked promising.

### Decisive simplicity result

The subsequent A2 control changed the interpretation. A B20 carrier containing only sorted
support-trial log-rates and pooled count quantiles achieved mean proxy R2 `0.904758`, versus
`0.862794` for the temporal prototype. `P20-B20=-0.041963`, negative in 4/4 sessions, with CI
`[-0.076004,-0.007923]`. The evidence supports nonlinear marginal rate/count distribution, not
chronological EWMA routing or coherent temporal slots.

Consequently, turning P20 into a K/V memory and adding cross-attention would combine two unsupported
steps: temporal order as the useful carrier and attention as the correct consumer. It also changes
the deployment contract:

- existing P20 state at `N=64` is 1,728 scalars / 6,912 bytes before attention projections;
- four-float AC4 state is 256 scalars / 1,024 bytes at `N=64`;
- cross-attention adds per-bin query/key/value MACs and temporary state;
- a newly inserted attention path cannot influence a frozen decoder unless it was trained offline
  to consume that path. “Forward-only deployment” is possible; “no decoder retraining after adding
  the path” is not a credible mechanism.

### Defensible remnant

The surviving idea is not fixed-K memory but the simpler B20 marginal carrier. B20 still has only
source neural-proxy evidence and may encode persistent gain, recording quality, or burstiness
rather than behaviorally useful identity. It would need independent M2/SUA CPU replication before
any decoder or GPU work. It should not be renamed temporal memory.

### Kill rule

Do not implement K/V cross-attention from the existing P20 artifact. Reopen only if a new,
independent temporal-order experiment beats both B20 and a within-trial time-order null. Current
evidence does not meet that condition.

## 3. Path 3 — paired SUA/pseudo-MUA co-training

### Why this is the strongest path

This path composes two already-supported facts rather than trying to rescue a failed component:

1. T4 is effective in separately trained SUA and pseudo-MUA views; the pseudo-MUA bridge reported
   `T4-F0=+0.3177`, and the residual SUA advantage after T4 was only about `+0.0406`.
2. Experiment A identified AC4 `[a,c,0,0]` as sufficient relative to full T4 on **SUA**, while
   AC4 row shuffle caused `-0.294163` R2. This sharpens the SUA mechanism, but AC4 sufficiency has
   not yet been shown on pseudo-MUA and must not be silently transferred across views.

Its primary contribution is not necessarily higher R2. It is one encoder weight set that remains
non-inferior under deterministic unit-to-electrode pooling, with unchanged inference architecture
and no added calibration state. Offline training roughly doubles view exposure, but deployment
cost remains that of one encoder.

### Current blocker and clean resolution

Q3/C1 is `blocked_fail_closed` because July historical references did not pin the exact scorer
source bytes or train/validation manifest. This is a provenance failure, not a negative shared-
weight result. Do not bridge around it. Rebuild all references under one new sealed source tree:

```text
fresh separate SUA-T4 control
fresh separate pseudo-MUA-T4 control
fresh paired-view shared-T4 model
fresh paired-view shared-TS4 attachment control
```

Use paired microbatches from the same source session, separate train-only normalizers and correct
unit-level/channel-level T4 construction, and equal task-loss weighting. Full T4 is the clean C1
substrate because it is already effective in both views. AC4 may replace it only after a separate
pseudo-MUA component-attribution gate or as a later compression experiment. Phase C1 must use
`lambda_consistency=0`; otherwise capacity sharing and consistency regularization are confounded.

Only if C1 is non-inferior in both views may C2 add one-way consistency from pseudo-MUA prediction
to stop-gradient SUA prediction. The asymmetry matters because pseudo-MUA is a lossy deterministic
pooling of SUA.

### Limits on the claim

Pseudo-MUA is not native threshold-crossing MUA and shares the same recordings with SUA. A positive
result establishes robustness to a controlled split/merge granularity transform, not cross-array
or real-MUA generalization. Native-MUA or external-subject confirmation remains necessary.

## 4. Divergent candidates considered

The review considered the following variants before convergence:

1. unconstrained signed four-coordinate correction MLP;
2. scalar confidence/error predictor;
3. SO(2)-equivariant radial `(a,c)` shrinkage;
4. design-conditioned affine whitening;
5. jackknife/analytic-standard-error weighting;
6. second-harmonic nuisance correction;
7. per-unit ordered fixed-K K/V prototypes;
8. population-shared temporal prototypes;
9. B20 order-invariant marginal carrier;
10. time-order-null-gated prototypes;
11. frozen-decoder nonparametric memory retrieval;
12. paired-view shared weights without consistency;
13. one-way pseudo-MUA-to-SUA distillation;
14. whole-view dropout during source training;
15. random split/merge granularity augmentation beyond electrode pooling;
16. shared backbone with a minimal view-specific input read-in.

After applying evidence, simplicity, causal-attribution, deployment, and held-out-value filters,
the surviving order is: C1 paired shared weights; conditional C2 one-way consistency; later
split/merge augmentation; independent-subject equivariant correction. B20 is a CPU-only side
hypothesis. All fixed-K attention variants are rejected under current evidence.

## 5. Winner pitch

> A decoder trained separately for SUA and pseudo-MUA wastes the known deterministic relation
> between the two views and gives no guarantee that functional identity survives unit pooling.
> Train one T4-conditioned encoder on paired SUA/electrode-pooled views so that the same weights
> are robust to neural granularity while deployment remains forward-only and state-neutral.

Strongest objection: pseudo-MUA is derived from SUA and may make the task artificially easy.
Response: frame C1 strictly as a controlled granularity-invariance test, require fresh separate-
view controls, and reserve any MUA/generalization claim for native-MUA or external data.

## 6. Minimal validation sequence

1. **CPU/provenance Gate 0:** fresh source hashes, strict 27/6/6 manifest, paired bin-sum
   equality, unit-to-electrode mapping, separate normalizers/caches, permutation invariance,
   matched training exposure, parameter/MAC/state receipt, and zero formal paths.
2. **C1 GPU test:** fresh separate SUA-T4, separate pseudo-MUA-T4, shared paired-view T4, and
   shared paired-view TS4 under seeds 42/43/44: 12 fresh runs total. No one-seed arm selection.
   Shared weights must have paired lower bound
   at least `-0.03` versus each corresponding separate model, preserve correct-content/attachment
   evidence in both views, and not increase the absolute cross-view gap beyond the frozen margin.
3. **Conditional C2/external test:** only after C1 passes, add one source-selected asymmetric
   consistency objective. Freeze one candidate before any formal/external endpoint; real MUA is a
   separate confirmation, not a tuning set.

## 7. Two-week feasibility pilot

- **Days 1--2:** rebuild the two separate references and paired loader under a single immutable
  source/manifest receipt; run CPU invariants and cost audit.
- **Days 3--6:** execute the complete four-group, three-seed C1 matrix on two GPUs and generate one
  paired aggregate. Do not screen on seed 42.
- **Day 7:** apply frozen non-inferiority/content/gap gates. Stop if any primary gate fails.
- **Week 2 only after a C1 pass:** run one predeclared C2 consistency value or prepare a native-MUA
  compatibility receipt. Keep formal SUA sealed until a single final candidate is frozen.

## Evidence anchors

- `docs/SUA_STEP2A_INDEPENDENT_RESULT_AUDIT.md`
- `results/t4_estimator_b_v7_source_only_cpu_audit_v1/aggregate.json`
- `docs/M1_FIXED_K_TEMPORAL_PROTOTYPE_GATE_A_RESULT_AUDIT.md`
- `docs/M1_FIXED_K_TEMPORAL_PROTOTYPE_GATE_A2_RESULT_AUDIT.md`
- `docs/T4_NEXT_EXPERIMENT_PROTOCOL_V2.md`
- `results/t4_paired_view_c1_preflight_v1/receipt.json`
- `results/sua_t4_m30_component_attribution_v10/aggregate_r11.json`
- `results/sua_t4_m30_ac4_rs4_v1/aggregate_v1.json`
