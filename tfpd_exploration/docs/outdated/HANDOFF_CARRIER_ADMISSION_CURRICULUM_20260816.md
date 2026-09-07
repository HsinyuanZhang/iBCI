# Handoff: Performance-First Calibration Pretraining and T4 Admission

Date: 2026-08-16

Status: **EXECUTED AND CLOSED (2026-08-17). The curriculum hypothesis failed. Do not schedule
further staged-admission arms from this document.** Retained for its receipts, for the confirmed
zero-gradient analysis in §0, and because Arm A — which it specified as a control — is now the
lane's baseline system. Its forward-looking authority claim below is void; ranked next steps live in
`HANDOFF_SPINT_DECODER_DIRECTIONS_20260817.md` §5.

### Verdict

Matched scorer, final-four SWA, 6 within-development sessions
(`results/gate3_within_screen_v1/within_screen_receipt.json`):

| arm | within mean R² | vs Arm A | vs Arm C |
|---|---:|---:|---:|
| A `direct_t4_48` (compute-matched control) | **0.5163** | — | — |
| C `direct_t4_33` (exposure-matched control) | 0.5135 | −0.0028 | — |
| **B `z4_pretrain_then_t4` (treatment)** | **0.3392** | **−0.1771 (0/6)** | **−0.1743 (0/6)** |
| spintshape 12-epoch prior baseline | 0.4734 | −0.0428 | — |

**The treatment loses to both controls on every session.** Arm C is what makes this interpretable:
it matches Arm B's 33 T4 epochs, so the loss cannot be attributed to reduced T4 exposure. **The
ordering itself is harmful.** Mechanism: Arm B ends with `‖W_side‖ = 0.7606` against A's `1.3992`
and C's `1.3475`, i.e. Z4-first pretraining left the T4 pathway underdeveloped at matched exposure.

Two results worth carrying forward:

1. **Arm A is the lane's new baseline.** The 48-epoch warmup+cosine + final-four-SWA recipe beats
   the 12-epoch baseline by `+0.043` within (5/6) and `+0.0708` external (13/15, bootstrap 95%
   [+0.046, +0.094], p = 0.0074), reaching external 0.1610 vs 0.0902. Any successor must clear Arm
   A, not 12-epoch spintshape.
2. **§0's zero-gradient prediction was confirmed exactly.** Throughout Arm B's 15 Z4 epochs,
   `W_side` and both Adam moments stayed elementwise zero, with the side tensor bitwise equal to
   canonical Z4. So a gain schedule on that block is a no-op until the phase boundary, and the
   argument that a continuous ramp is largely self-cancelling under weight-decay-free Adam stands.

Original status line, retained: revised next-round experiment handoff. This document defines the
scientific comparison and execution order. It does not authorize formal/organizer-held evaluation,
post-hoc target selection, or an unreviewed GPU launch.

Authority: this file supersedes the training-order and key-successor recommendations in
`HANDOFF_LARGE_ROUTE_OPTIMIZATION_20260816.md`. It retains that document's performance-first goal,
metric-parity requirement, and rule that ablations consume a small minority of compute. It also
makes the PV baseline closure and existing-checkpoint transfer audit explicit prerequisites here;
where another same-date draft implies a different GPU order, this file governs these arms.

## 0. Revised decision

The continuous T4 ramp proposed in the first version is rejected as the primary treatment.

With the current zero-initialized B3S side columns and no-weight-decay Adam, the loss sees
`alpha * W_side`. Adam is approximately invariant to gradient scale once the gradient dominates
`eps`; `W_side` can compensate for a small alpha much faster than an epoch. A multi-epoch alpha ramp
therefore does not reliably mean gradual learning of T4. It is an ambiguous branch-gain schedule.

The main training method is now a hard, baseline-preserving transition:

```text
phase 1: normalized model-visible side = canonical Z4, alpha = 0
phase 2: normalized model-visible side = canonical T4, alpha = 1
```

The transition occurs only after a source-only pilot freezes the phase-1 length. There is no
fractional alpha in the main experiment.

Run three seed-42 source-training arms:

1. `direct_t4_48`: T4 from random initialization for 48 epochs.
2. `z4_pretrain_then_t4`: Z4 pretraining for `T_pre` epochs, followed by T4 fine-tuning for
   `E_t4 = 48 - T_pre` epochs.
3. `direct_t4_exposure_matched`: T4 from the same random initialization for exactly `E_t4` epochs.

The comparisons answer different questions:

- arm 2 versus arm 1: which recipe gives the best result under the same 48-epoch training budget?
- arm 2 versus arm 3: what is the net value of Z4/calibration pretraining when T4 exposure and the
  T4-phase optimizer schedule are matched?
- arm 1 versus the current 12-epoch model: was the original training budget/schedule limiting?

This third arm is not optional mechanism ornamentation. It is the minimum control needed to make a
negative curriculum result interpretable.

## 1. Current evidence and why the original contract changed

Current development results under the historical TFPD scorer are:

| System | T4 R2 | Z4 R2 | T4-Z4 |
|---|---:|---:|---:|
| teacher-free spintshape, 12 epochs | 0.4574 | 0.1985 | +0.2589 |
| Large-v1 | 0.2562 | 0.0059 | +0.2503 |
| small bilinear | 0.0767 | 0.0216 | +0.0551 |
| A2 teacher-initialized historical reference | 0.5750 | 0.3260 | +0.2490 |

The TFPD and A2 scores are not yet exact metric/checkpoint peers. The flattened TFPD R2 is
optimistic relative to A2's per-output variance-weighted R2: the flattened denominator adds the
between-output-mean term while leaving the residual numerator unchanged. Matched rescoring can move
TFPD downward or leave it effectively unchanged; it cannot create hidden headroom toward `0.5750`.

The observed checkpoint variation also invalidates a single-checkpoint-only gate. Existing
spintshape-T4 means for epochs 4--11 are:

```text
0.4417  0.4448  0.4746  0.4745  0.4485  0.4721  0.4556  0.4471
```

Their sample standard deviation is `0.01413` and range is `0.03288`. The empirical scale of a
difference between two independent single checkpoints is approximately `0.01998`. A `+0.03` gate
on two endpoint checkpoints alone is therefore too close to checkpoint noise.

The existing spintshape-Z4 curve also does not justify a 12-epoch pretraining phase:

```text
epoch:  4       5       6       7       8       9       10      11
R2:     0.2738  0.2393  0.2011  0.1994  0.1655  0.1428  0.1931  0.1727
```

The best observed point is epoch 4, possibly earlier, and epoch 11 is `0.1010` below epoch 4. This
development curve must not be used to choose `T_pre`, but it removes the justification for
hard-coding `T_pre=12`.

## 2. Scientific claim is intentionally narrow

This is staged pretraining/fine-tuning. Do not claim otherwise.

The potential contribution is the domain-specific empirical finding that withholding task-frame
identity while first learning a calibration-conditioned set decoder improves the eventual T4
solution under controlled compute and exposure budgets.

Two-sentence claim if successful:

> We find that a calibration-conditioned set decoder benefits from learning its neural and
> calibration identity pathway before task-frame features are admitted. A source-selected Z4
> pretraining stage followed by T4 fine-tuning improves cross-session and cross-subject decoding
> without changing the deployed architecture or adding inference state.

If the matched experiments do not show a practical multi-session gain, report the recipe as
unsuccessful under this protocol. Do not claim that all information-order curricula are false.

## 3. Fixed final architecture

All three arms use the current teacher-free spintshape graph:

```text
M30 calibration neural activity -- B3S calibration encoder -- per-unit identity
T4 or Z4 side feature -----------/

query neural activity + per-unit identity
    -> shared per-unit read-in
    -> static-query cross-attention over the unit set
    -> behavior prediction
```

No teacher checkpoint, teacher logits, distillation loss, persistent NeuronID table, GRU/SSM,
extra attention layer, extra slot, FiLM path, or direct key residual is allowed in this comparison.

`SideFeatureEarlyPoolEncoder` constructs `post_pool[0]` as `Linear(68,64)` for hidden dimension 64
and T4 dimension 4. Its T4 columns `weight[:,64:68]` are already initialized to exact zero. Preserve
and receipt that behavior.

At the start of phase 1:

- every element of `W_side` is exactly numerically zero;
- Adam `exp_avg` and `exp_avg_sq` for `W_side` are absent or bitwise zero;
- with canonical Z4 input, the gradient of `W_side` is bitwise zero;
- after every phase-1 epoch, `W_side`, `exp_avg`, and `exp_avg_sq` remain elementwise exactly zero;
- Adam's scalar step counter may advance and is reported separately.

For byte receipts, normalize signed zero to positive zero before hashing these invariant tensors;
`-0.0` versus `+0.0` is not a scientific difference.

The final inference graph and canonical full-T4 input are identical for all three deployed models.

## 4. T4/Z4 construction boundary

The admission operation must occur after T4 normalization and unit/session alignment.

Correct order:

```text
raw support and labels
    -> canonical T4 builder
    -> source-only normalizer
    -> aligned normalized T4 [B,N,4]
    -> choose model-visible canonical Z4 or exact normalized T4
    -> StreamingSpintModel.forward(..., side_features=visible_side)
```

Forbidden order:

```text
alpha * raw_T4 -> normalizer
```

At phase 1, use the exact tensor returned by the canonical `z4` feature group or a
`zeros_like(normalized_T4)` tensor proven bitwise equal to it. Never compute `0 * raw_T4`; after
normalization that need not be zero, and floating multiplication can preserve negative-zero bits.

Every phase-1 epoch must assert:

- visible side is bitwise equal to `load_unit_side_features(..., group="z4")` for the same session
  and unit order;
- visible side is finite, positive zero, shape `[B,N,4]`;
- canonical normalized T4 authority bytes have not changed;
- no T4 tensor reaches `forward` or `compute_identity` through another path.

Every phase-2 epoch must assert visible side is bitwise equal to canonical aligned normalized T4.

## 5. Freeze `T_pre` with a source-only pilot

Do not choose phase length from the six within-development sessions or the observed Z4 curve.

Before official three-arm training, run one source-only Z4 boundary pilot using the same model,
initialization family, optimizer family, and phase-1 schedule intended for the official arm.

Create a deterministic source-audit split:

- source sessions only; no within-development, external-development, formal, or organizer-held rows;
- select audit query windows by a frozen hash of `(session_id, window_start, namespace)`;
- use 5% of eligible source query windows per source session, with deterministic tie-breaking;
- exclude those audit windows from pilot gradients;
- bind ordered indices, receptive-field coverage, labels, and byte hashes in the pilot preflight.

Pilot candidate endpoints are epochs `0..15`. At each endpoint, score the fixed source-audit set
with the matched per-output variance-weighted R2. Let `S_max` be the maximum source-audit score over
the 16 candidates. Freeze:

```text
selected_epoch = earliest e such that S_e >= S_max - 0.005
T_pre = selected_epoch + 1
E_t4 = 48 - T_pre
```

This is an earliest-near-best source rule, not target-guided early stopping. The `0.005` tolerance,
candidate range, tie rule, and audit split must be sealed before the pilot opens source arrays.

After the pilot, mint an immutable boundary receipt and run all three official arms fresh from the
shared canonical initial state. Do not reuse the pilot model as the official curriculum checkpoint.
This keeps `T_pre` frozen before the official comparison and prevents pilot-specific optimizer state
from entering only one arm.

## 6. Exact three-arm training contract

### 6.1 Common invariants

- strict-27 source roster and chronological M30 calibration;
- SUA center-out task;
- seed 42 for the first gate;
- one immutable initial state loaded strictly by all arms;
- identical source training examples and sampler order wherever epoch counts overlap;
- Adam with `lr=1e-4`, `betas=(0.9,0.999)`, `eps=1e-8`, `weight_decay=0`, and
  `amsgrad=false` before the applicable scheduler modifies LR;
- no arm-specific clipping, regularization, loss weight, batch size, or data augmentation;
- no target/dev-session validation, early stopping, or checkpoint selection during fitting;
- no formal or organizer-held data;
- final checkpoint and final-four SWA artifacts are immutable.

Optimizer choice is load-bearing and may not be replaced by a generic "common alternative" in the
preflight. Any optimizer change requires a new reviewed contract because it changes the meaning of
hard admission and the comparison with the historical model.

### 6.2 Arm A: `direct_t4_48`

- load canonical initial state;
- canonical full T4 for epochs `0..47`;
- one fresh Adam optimizer;
- linear LR warmup from `1e-5` to `1e-4` over epochs `0..1`;
- cosine decay to `1e-6` through epoch 47;
- 48 total T4 epochs.

This is the equal-total-compute performance baseline.

### 6.3 Arm B: `z4_pretrain_then_t4`

Phase 1:

- load canonical initial state;
- canonical Z4 for exactly `T_pre` epochs;
- fresh Adam optimizer with constant `lr=1e-4`, exactly matching the boundary pilot;
- enforce the zero-column and zero-moment invariants every epoch.

Transition:

- seal the phase-1 checkpoint and state hash;
- switch visible side atomically from canonical Z4 to canonical normalized T4;
- create a fresh Adam optimizer for all parameters;
- do not reset any model weight;
- verify the first full-T4 forward is finite and the T4 branch begins from `W_side=0`.

Phase 2:

- canonical full T4 for exactly `E_t4` epochs;
- use the exact T4-phase optimizer and LR schedule defined for arm C: linear warmup from `1e-5`
  to `1e-4` over its first two epochs, then cosine decay to `1e-6` at its final epoch;
- save the final four checkpoints for SWA.

Total model-training epochs are exactly `T_pre + E_t4 = 48`.

### 6.4 Arm C: `direct_t4_exposure_matched`

- load the same canonical initial state;
- canonical full T4 from its first batch;
- train for exactly `E_t4` epochs;
- use the same fresh Adam constructor, two-epoch warmup, cosine endpoint, LR value at every
  T4-phase step, batch order, and number of T4 optimizer steps as arm B phase 2;
- save the final four checkpoints for SWA.

Arm B versus C isolates the practical net value of Z4 pretraining before an otherwise matched T4
fine-tuning phase. Arm C is shorter in total compute by design. This comparison does not claim that
ordering alone was isolated from the extra Z4 compute; it measures whether that extra pretraining is
worthwhile at fixed T4 exposure.

## 7. SWA and deployment estimands

The primary scored artifact for each new arm is a predeclared equal-weight SWA of its final four
checkpoints. SWA is both a lower-variance estimate and a deployable single model.

SWA contract:

- exact final four consecutive checkpoints only;
- arithmetic mean of every floating model-state tensor in FP64, cast back to its declared dtype;
- non-floating buffers copied from the final checkpoint and exact-compared across the window;
- identical state keys, shapes, dtypes, architecture, and config required;
- no optimizer state in the deployable SWA;
- immutable SWA body/checkpoint pair with component checkpoint hashes;
- strict reload and finite forward smoke before scoring.

The exact final checkpoint is reported as a secondary deployable endpoint. It cannot replace SWA as
the gate estimator. The final-four mean of four separate scores may be reported as a stability
diagnostic, but it is not the primary model because it is not a single deployable checkpoint.

For the current 12-epoch baseline, construct the analogous predeclared SWA from epochs `8..11` and
report its fixed epoch-11 endpoint. Do not select the best historical epoch. A2 remains a separately
matched sealed reference under its own fixed checkpoint authority.

## 8. Zero-training work before the official GPU arms

The following precede official source training and may run in parallel where safe:

1. repair or verify the complete PV T4/Z4 baseline receipts and rescore them with the matched metric;
2. implement and validate the matched per-output variance-weighted scorer;
3. build the fixed current-spintshape epochs-8--11 SWA and score within development;
4. score the same current spintshape T4/Z4 fixed artifacts on the existing external-development
   protocol to quantify transfer degradation before optimizing the route;
5. rescore the sealed A2 references without modifying them;
6. complete the source-only boundary pilot and seal `T_pre`.

The transfer audit is an upstream descriptive baseline. It may decide whether the route is worth
continuing, but it may not tune `T_pre`, phase schedules, thresholds, or model structure.

This document is authoritative for the new training sequence. Any same-date handoff that proposes a
different GPU order is superseded for these arms; its non-training baseline and transfer tasks remain
valid unless an immutable receipt proves them already complete.

## 9. Required training diagnostics

Record at each epoch for all applicable arms:

- training loss and exact number of examples;
- learning rate and optimizer step count;
- global and per-branch gradient norms;
- clipping frequency, if clipping is later authorized by a revised contract;
- `norm(W_side)`;
- `norm(alpha * W_side)` with alpha restricted to zero or one;
- `norm(W_side * T4)` on a fixed source-only diagnostic batch;
- ratio of T4 contribution norm to calibration contribution norm at `post_pool[0]`;
- Adam `exp_avg` and `exp_avg_sq` norms for `W_side`;
- attention entropy/concentration summary;
- state-dict SHA and optimizer-state SHA;
- parameter, activation, gradient, and optimizer-state finiteness.

For arm B phase 1, any nonzero `W_side`, side moment, effective T4 contribution, or non-Z4 visible
side is a contract failure, not an interesting result.

Diagnostics explain optimization; they do not rescue an absolute performance failure.

## 10. Metric and paired statistical contract

Use one scorer for every new comparison:

- preserve the two velocity dimensions;
- variance-weighted R2;
- identical padding and query semantics to the matched A2 scorer;
- equal weight per session;
- no target updates, gradients, optimizer, checkpoint selection, or state mutation.

For every pair, compute session-paired deltas and report:

- equal-session mean;
- median;
- number positive;
- minimum and maximum;
- all individual session deltas;
- fixed-seed paired-session bootstrap 95% interval;
- exact sign pattern.

Do not treat six sessions as a high-powered significance test. The bootstrap interval is uncertainty
evidence, not a substitute for external sessions and multiple seeds.

## 11. Execution order and decisions

### Gate 0: CPU/no-target implementation gate

Require:

- scorer parity and metric-direction disclosure;
- source-audit split and `T_pre` selector tests;
- canonical T4/Z4 equality and post-normalization admission tests;
- zero-column, zero-gradient, and zero-Adam-moment phase-1 tests;
- strict initial-state equality across all arms;
- exact T4-step/LR equality for arm B phase 2 and arm C;
- SWA construction, strict reload, and adversarial state-key tests;
- fresh output topology and immutable closure;
- formal/organizer-held paths absent.

Replace the vacuous test "same model plus same zero input gives same output" with the meaningful
assertion that the phase-1 visible side is bitwise identical to the production Z4 loader for the
same session and unit order, after normalization and immediately before `forward`.

### Gate 1: source-only boundary pilot

Run the 16-endpoint Z4 pilot, publish all source-audit scores, freeze `T_pre`, and stop if the source
audit, checkpoint, or selector lineage is incomplete. Do not open development data.

### Gate 2: official seed-42 training

Train arms A and B under the frozen 48-epoch budget. Train arm C under the frozen `E_t4` exposure
budget. No scorer opens development data until all three source terminal receipts and SWA artifacts
are complete.

### Gate 3: within-development screen

Primary comparisons use SWA:

1. A versus current matched 12-epoch SWA: long-schedule effect;
2. B versus A: equal-total-budget practical curriculum effect;
3. B versus C: Z4-pretraining value at matched T4 exposure;
4. best new arm versus matched A2 T4.

Practical screen for a training-method claim:

- B minus A SWA mean `>= +0.03` and median `> 0`;
- B minus C SWA mean `>= +0.03` and median `> 0`;
- at least `4/6` paired session deltas positive in each comparison;
- no missing or non-finite session.

Classify rather than overclaim:

| Outcome | Decision |
|---|---|
| B clears both comparisons | curriculum passes seed-42 within screen |
| B beats A but not C | useful equal-budget recipe; pretraining value not isolated |
| B beats C but not A | Z4 pretraining helps at matched T4 exposure but is not the best compute use |
| B gains `+0.01..+0.03` | retain if best, but no main training-method claim |
| B gains `<+0.01` in both A and C comparisons | close this concrete recipe as impractical; do not falsify all ordering hypotheses |
| A improves while B does not | adopt long direct-T4 recipe for performance |
| no new arm improves by `+0.01` | close training-budget route and consider key-only successor |

### Gate 4: external development

Performance and novelty are separate routes:

- any new arm with within gain `>= +0.03` over the matched current baseline may proceed externally;
- a curriculum claim requires paired external B versus A and B versus C mean `>= +0.03`, median
  `>0`, and at least `9/15` positive sessions;
- report paired bootstrap intervals and all session deltas;
- interaction, attention, robustness, or wrong-pair results cannot rescue absolute failure.

### Gate 5: seeds 43 and 44

Only after an external pass, repeat the minimum required frozen arm set:

- performance-only direct-long claim: repeat arm A;
- curriculum claim: repeat A, B, and C because both causal comparisons are required;
- require positive three-seed mean effects and report seed-by-session hierarchical bootstrap.

Formal/organizer-held evaluation remains sealed until all choices and gates are frozen.

## 12. Performance-first rule

Keep the strongest honest system even if the novelty hypothesis fails.

- If A is best, adopt long direct-T4 and report training budget/schedule as material.
- If B is best but misses the method gate, use it as an engineering recipe without a major claim.
- If B clears matched exposure, matched compute, external, and multi-seed gates, promote the narrow
  calibration-pretraining/T4-fine-tuning contribution.
- If A2 initialization remains best after matched scoring and schedules, retain teacher/source
  pretraining as a performance component and call it initialization, not distillation.

Do not preserve a weaker model for a cleaner novelty story.

## 13. Conditional architecture successor

Only if the best trained teacher-free model leaves practical headroom, implement one teacher-free
key-only T4 residual:

```text
base_i  = encode_query_activity_i + encode_calibration_identity_i
key_i   = W_key(base_i) + beta * W_t4(T4_i)
value_i = W_value(base_i)
```

Requirements:

- use the best frozen teacher-free parent;
- direct-zero residual and exact parent output at initialization;
- T4 changes keys only, never values;
- one seed-42 T4 cell first;
- no FiLM, GRU/SSM, extra attention, extra slot, or width increase;
- absolute within lift over its matched parent `>= +0.03` before any sibling expansion.

The older key-residual code is a teacher/A2 warm-start pilot with CPU tests and no GPU result. It may
provide primitives but is not the required teacher-free experiment.

## 14. Minimal ablation policy

The three arms above are the primary experiment, not a broad ablation grid. No additional training
arm is authorized before their verdict.

Forward-only diagnostics after a positive result may include:

- wrong-pair T4;
- zero T4;
- destroyed or temporally scrambled calibration activity.

Deferred unless a positive result needs mechanism clarification:

- fractional alpha ramp;
- abrupt-switch timing alternatives;
- branch-specific learning rate;
- identity-source 2x2;
- value-side FiLM;
- state-conditioned attention or SSM;
- capacity and compression sweeps.

## 15. Required receipts

The successor implementation should produce:

1. matched-scorer and current-baseline rescore receipts;
2. current-spintshape external transfer receipt;
3. source-audit split and phase-boundary pilot preflight/terminal;
4. immutable boundary receipt containing `T_pre` and `E_t4`;
5. one canonical shared initial-state artifact;
6. launch, milestone, terminal, optimizer, and diagnostic receipts for A/B/C;
7. exact final checkpoint and final-four SWA artifact for each arm;
8. within aggregate and conditional external/multi-seed aggregates.

Every runtime receipt must disclose:

- teacher checkpoint/logits/loss used: false;
- Adam exact hyperparameters and weight decay zero;
- source roster, manifest, T4/Z4 authority, normalizer, and unit order;
- phase and visible-side mode per epoch;
- `W_side` and optimizer-moment invariants;
- target updates/backward/optimizer steps during scoring: zero;
- formal/organizer-held data opened: false;
- launch and final implementation closure equality.

Existing sealed A2 and TFPD artifacts remain read-only and must not be overwritten.

## 16. Stop conditions

Fail closed if:

- `T_pre` is influenced by within/external development data;
- admission is applied before normalization or unit alignment;
- phase-1 visible side differs from canonical Z4;
- `W_side`, `exp_avg`, or `exp_avg_sq` becomes nonzero during phase 1;
- the hard switch changes any model weight before the first T4 optimization step;
- B phase 2 and C differ in T4 optimizer steps or phase-local LR schedule;
- SWA membership is changed after scoring;
- development scores select a checkpoint;
- output roots are nonfresh or implementation closure drifts;
- formal/organizer-held data is opened;
- a mechanism diagnostic rescues an absolute performance failure.

The round reaches a valid terminal boundary after the A/B/C seed-42 SWA within aggregate. A negative
result closes this concrete recipe without requiring a larger ablation matrix.
