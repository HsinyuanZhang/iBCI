# Design — Post-Fusion Identity Memory (Training-Side Cell, Pending GPU)

Date: 2026-09-02. Status: design reviewed; additive implementation may begin,
but GPU execution remains unauthorized until the matched smoke and receipt
audit pass. Owner: operator queue after the completed M2 A0 external gate.

## 1. Goal (operator's bar)

A modest gain is enough: external_post30_local equal-session mean from the sealed
**0.2991** (POOLED champion, k=4 D-opt + CDM) to **≥ 0.309** (+0.01). Primary
gate = +0.01 external with ≥4/6 breadth; below +0.005 = null; any within-gain-
external-loss pattern = the PIT-M2 overfit signature (report, do not promote).

## 2. The idea

Move the online memory across the fusion MLP:

```text
current (sealed):   identity = MLP( mean_pool(pre_pool(trials)), T4 )   # mean before MLP
proposed:           identity = mean_i( MLP( pre_pool(trial_i), T4 ) )  # per-trial readout, mean after
```

Frozen-weights probe (results/m2_postfusion_probe_v1/, TERMINAL
POSTFUSION_HARMFUL, −0.083 external): the harm is dominated by the OOD penalty —
per-trial MLP outputs sit far off the trained pooled-mean manifold (first-row
RMS deviation 0.08–0.17 vs identity RMS 0.41–0.55). Training-side exposure is
the only remaining route; on frozen weights the question is closed.

## 3. Why it might work (the honest best case)

- DeepSets placement hypothesis: the current route applies the nonlinear
  readout after pooling, whereas the proposal exposes each trial to the
  nonlinear readout before averaging. These are different function classes;
  neither is a strict superset of the other. The benefit, if any, must come
  from learning a less noisy additive per-trial representation, not from an
  unsupported claim of universal extra expressiveness.
- The OOD penalty vanishes by construction once training sees per-trial
  readouts (PIT-style exposure of the forward graph, not just the inputs).

## 4. Why expectations stay low (the two ceilings)

1. **Saturation ceiling**: the activity axis is nearly exhausted by the shallow
   pooled readout — k-curve (results/m2_kcurve_v1/): M4-Dopt ≈ all-15-usable
   within 0.012 (static) / 0.022 (CDM); external already SATURATES_AT_4 with
   k4 ABOVE the all-usable endpoint. A more expressive averager has ≤ +0.01–0.02
   of identity-axis headroom to chase.
2. **Program capacity law**: every expressiveness expansion (3-layer identity
   SUA lineage, learned filters, contrastive heads) failed to buy transfer;
   per-trial readouts also push T4's nonlinear interaction through single-trial
   noise — the exact mechanism that flipped PIT-M2 (within +0.113, external
   −0.029).

Equivalence note: after training, the proposed scheme = the same activity FIFO
running in a deeper feature space; ALL memory-law findings transfer unchanged
(uniform uncapped accumulation optimal; EMA/recency loses; FIFO capacity
competition with support rows). No new memory dynamics — only the feature space
moves.

## 5. Cell definition (when GPU frees)

Matched pair on the M2 sealed trainer (src/pit_m2_v1 trainer skeleton, the
PIT-M2 discipline verbatim: smoke equality, dropout-p stream digests, sealed
anchors, 2h/arm bound, seeds/protocol unchanged):

- **T0-pf**: control = the current forward law (identity from pooled mean) —
  byte-reproduce the sealed lineage (same no-op proof as pit_m2's t0m).
- **C1-pf**: intervention = per-trial identity readout, running mean over pool
  members (support trials + completed trials), fed to the decoder. T4 concat
  inside each per-trial readout (constant across trials; its interaction now
  passes through single-trial noise — disclosed risk).
- Optional third arm (only if C1-pf passes the gate): C1-pf + PIT cycle
  (10,5,2) — but see §6 ordering.

Evaluation: the k=4 D-opt support law, the sealed external_post30_local +
within_post30 partition, per-session paired deltas vs the sealed 0.2991/0.6541
rows (tolerance anchors); memory law within C1-pf = uniform uncapped (the
optimal law from m2_memory_law_scan_v1).

## 6. Prior odds (operator briefing)

~50% null (OOD gone, Jensen residual ≈ 0 → "training absorbs the placement"),
~30% small win (+0.005–0.015; passes the bar only above +0.01), ~20% PIT-M2
overfit signature. Cost ≈ 30 min/arm on the 3090 + scoring. This is the LAST
untested architecture-axis variant on M2; any outcome closes the axis.

## 7. What is already settled and must not be re-litigated

- Placement on frozen weights: HARMFUL (sealed probe).
- Memory law: uniform uncapped (sealed scan).
- Carrier/T4 updates: structurally dead on M2 (oracle-null; saturation).
- k/B selection for this V1 mechanism test is frozen now as
  `B=30 -> D-opt k=4`. The continual column uses uniform `UNCAPPED` memory.
  The incomplete `m2_kcurve_ext_v1` attempt may later be reported as
  descriptive evidence, but it cannot retrospectively change this experiment's
  configuration or select a more favorable post-fusion result.
- Trial-free fixed chunks are now closed on the governing local external
  surface. The completed A0 V2 row gives phase-0 chunk minus true-trial
  `−0.03216` mean, `−0.06776` worst, and `1/6` positive sessions; phase-50 is
  still `−0.01409` mean. In the same rows, true-trial continual activity minus
  static is `+0.00896` mean with `5/6` positives. Therefore Post-Fusion may use
  known true-trial boundaries for this local mechanistic test, but it must not
  be presented as a trial-free or official-deployment solution.

## 8. Root review: required matched decomposition

The GPU cell is admitted to the queue, but the following items are mandatory
before its work order can authorize execution.

### 8.1 The causal estimate is C1-pf minus a matched T0-pf

`0.2991` is a sealed context anchor, not the causal control for a newly trained
checkpoint. T0-pf and C1-pf must use the same initialization, seed, source
exposure, optimizer steps, learning-rate law, dropout/RNG stream, checkpoint
selection rule, and scoring rows. The primary training-effect estimate is:

```text
delta_train = score(C1-pf, matched seed) - score(T0-pf, matched seed)
```

The sealed champion is reported separately as the practical reference. A T0
reproduction miss must be disclosed and must not be charged to C1-pf.

The T0 no-op claim is conditional and exact: for an identical selected
`calib_trials` tensor and side tensor, the T0 adapter must be bitwise equal to
native B3S. Both arms then consume the same pool-exposure controller. Random
pool exposure therefore does not claim to byte-reproduce the entire historical
sealed training trajectory. A separate all-members CPU test must reproduce the
native identity, loss, and post-update state exactly.

### 8.2 Score a 2 x 2 decomposition from the same two checkpoints

Both checkpoints must be evaluated under both deployment laws:

| training placement | fixed initial support | uniform continual memory |
|---|---:|---:|
| T0-pf: pre-fusion | required | required, UNCAPPED |
| C1-pf: post-fusion | required | required, UNCAPPED |

This produces three identifiable quantities:

1. placement effect under fixed evidence;
2. continual-memory effect within each placement; and
3. placement x memory interaction.

Without the fixed-evidence C1-pf row, a positive C1-pf continual result cannot
be attributed to the training-side placement. These four scores reuse the same
two checkpoints and therefore do not require extra training.

### 8.3 Training exposure and model selection

C1-pf training must expose the actual post-fusion graph to a pre-registered
source-only distribution of pool sizes. It must not train only a single-trial
readout and then assume arbitrary-size averaging is in distribution. Pool-size
sampling, T4 reuse, padding/masks, and the running-mean arithmetic must be
receipt-bound. Checkpoint selection is source-only/grouped-OOF; neither local
external labels nor the sealed external score may select an epoch or tune a
hyperparameter.

The source-only selector must additionally prove that validation session groups
are disjoint from training groups, aggregate the validation metric as an equal
mean over session groups, and record the per-epoch pool-controller digest. If
the inherited `val_heldin/r2_mean` split cannot prove those facts, the route
must supply its own grouped-OOF evaluator before GPU admission.

### 8.4 Three-stage validation

1. **CPU/no-data constructibility:** exact T0 no-op path, permutation
   invariance, running-mean equivalence, causal update order, RNG/dropout
   pairing, and parameter-count disclosure.
2. **One short matched source smoke:** T0-pf and C1-pf must both be finite,
   connected, and free of a source grouped-OOF regression beyond the frozen
   safety tolerance. This is a feasibility/no-harm gate, not the headline.
3. **One matched full cell plus the 2 x 2 scorer:** external promotion still
   requires `delta_train >= +0.01` and at least `4/6` positive sessions. A gain
   below `+0.005` is null; within gain with external loss is the PIT-M2 overfit
   signature.

The optional PIT cycle remains closed unless C1-pf beats matched T0-pf on the
primary external gate. It cannot be opened by a within-only result.

## 9. Queue and resource isolation

The execution order is frozen as follows:

1. finish and audit the current M2 A0 external cell on physical GPU0;
2. use the pre-registered `B30/D-opt-k4/UNCAPPED` mechanism configuration;
3. write and audit a separate Post-Fusion work order and no-CUDA implementation;
4. run T0-pf, then C1-pf, on one physical GPU with matched resources;
5. score the mandatory 2 x 2 table; and
6. stop the architecture axis regardless of success or failure.

This design must not query, initialize, schedule, or interfere with the
physical GPU assigned to the concurrent M1 experiment. No Post-Fusion GPU job
may overlap the current A0 job.

## 10. Concise research claim and strongest objection

**Pitch.** Current calibration memory pools noisy trial evidence before a
nonlinear identity readout. We test whether training the readout on individual
trial identities and averaging in the learned identity space makes continual
calibration more transferable, while keeping the memory law causal and
parameter-free at deployment.

**Strongest objection.** The frozen-weight intervention loses `0.0826` external
R2 and the remaining identity-axis headroom is small, so training may merely
repair its own OOD damage and recover T0. The matched T0/C1 pair and 2 x 2
placement-by-memory decomposition answer this directly: recovery without a
positive interaction is null, not a new memory result.
