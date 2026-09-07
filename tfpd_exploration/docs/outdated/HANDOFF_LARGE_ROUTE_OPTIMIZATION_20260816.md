# Handoff: Performance-First TFPD / SPINT-Like Decoder Program

Date: 2026-08-16

Status: research-direction handoff. This document authorizes no GPU launch, target access, or
formal evaluation. All reported scores are development results.

## 0A. Status as of 2026-08-17 — items 3 and 4 have been executed

This section was added after the document was written. **§0 items 3–4 are done, and §1's table is
superseded.** Three things changed:

**1. The long-horizon convergence experiment ran and passed.** Arm A (`direct_t4_48`: 48 epochs,
warmup + cosine, final-four SWA, seed 42) under the matched scorer reaches **within 0.5163** vs the
12-epoch spintshape baseline's **0.4734** (`+0.043`, 5/6 sessions) and **external 0.1610** vs
**0.0902** (`+0.0708`, 13/15, bootstrap 95% [+0.046, +0.094], p = 0.0074). This clears §0 item 4's
`+0.03` bar, so **the training-horizon hypothesis is confirmed, not closed**, and Arm A — not
12-epoch spintshape — is now the baseline any successor must beat. Receipts:
`results/gate3_within_screen_v1/within_screen_receipt.json`,
`results/gate4_arm_external_v1/final_external_matrix_v2_receipt.json`.

**2. A carrier-admission curriculum was tested and failed decisively.** Z4-first pretraining then
T4 scores **within 0.3392**, i.e. `−0.177` against compute-matched Arm A (0/6 sessions) and
`−0.174` against exposure-matched Arm C `direct_t4_33` (0/6). Because C isolates exposure, the loss
is attributable to the *ordering*, not to reduced T4 epochs. Do not revive staged carrier
admission.

**3. External scores are now available for every arm, and they reorder the priorities.** External
native R² on 15 sub-M sessions: A2 teacher-init 0.341 (cited), Arm A 0.1610, spintshape-12ep-T4
0.0902, Arm C 0.0775, `large_t4` −0.0848, spintshape-12ep-**Z4** −0.2936, `large_z4` −0.7367,
`bl_t4` −0.9597. Two readings matter for §0 and §6:

- The A2 gap in §0's framing question is **wider externally than within** — `0.0587` within vs
  `0.180` external. Closing the within gap is not the same as closing the deployment gap.
- Activity-derived identity is a **negative** asset under subject shift (Z4 at −0.2936), while
  carrier-derived identity degrades far less: `large_t4 − spintshape_z4` is `+0.058` within but
  `+0.209` external, a ~3.6× widening. Topology-confounded, but the direction is large and is the
  most useful mechanism result the lane owns. Any proposal to add capacity to the calibration
  activity encoder now carries an explicit fingerprint risk.

Ranked next steps, including what to do while the TFAP pretraining round is in flight, are in
`HANDOFF_SPINT_DECODER_DIRECTIONS_20260817.md` §5. That document also lists the closed directions
with receipts.

## 0. Recommended next plan

The objective is the best honest system performance. We do **not** require the final decoder to
avoid SPINT-like structure, teacher initialization, source pretraining, or calibration activity.
Those are engineering choices, not ideological constraints. Their use must only be disclosed and
compared fairly.

Ablations must consume a small minority of the compute budget. Reuse sealed checkpoints, prefer
forward-only diagnostics, and add a control cell only when it resolves a decision that affects the
performance path.

Proceed in this order:

1. **Rescore existing checkpoints with one matched metric and checkpoint policy.** Preserve the
   two velocity dimensions, use variance-weighted R², aggregate equally over sessions, and report
   the fixed historical epoch-window result plus a separately defined deployable-checkpoint result.
   This is a no-training task.

2. **Use teacher-free spintshape as the main trainable backbone.** It is currently the strongest
   teacher-free development model. Large-v1 becomes a completed T4-only compression/architecture
   baseline, not the main performance system.

3. **Run one predeclared long-horizon spintshape-T4 convergence experiment.** Test whether the
   current 12-epoch limit is material using a decayed learning-rate schedule and fixed milestones.
   Do not run a broad hyperparameter sweep. Do not select an epoch after seeing target scores.

4. **Expand the training recipe only after a practical gain.** If the long-horizon T4 run improves
   the matched score by at least `+0.03`, apply the same frozen recipe to Z4 and then confirm the
   T4 system with seeds 43/44. If the gain is below `+0.01`, close the training-horizon hypothesis.

5. **Only if training alone is insufficient, test one minimal fusion successor.** Keep the
   teacher-free spintshape backbone and calibration identity. Add T4 as a direct-zero residual in
   the attention key while leaving neural/calibration values unchanged. Do not initially add FiLM,
   a GRU/SSM, dual aggregation, more slots, or more width.

6. **Use absolute performance as the gate.** A successor must improve absolute T4 performance;
   T4-Z4 interaction, wrong-pair degradation, attention entropy, or robustness cannot rescue a
   failed absolute score. Within-development performance is the screen; paired external
   development scoring is the primary adoption test.

7. **Reserve detailed ablations for a positive model.** Identity-source 2×2, value-side FiLM,
   dual attention/sum aggregation, reliability gates, and compression sweeps follow only if the
   long-training or key-only model produces a practical positive signal.

8. **Keep state-conditioned attention on hold.** The completed time-varying pre-gate is negative
   overall, and static-query spintshape already strongly outperforms Large-v1. Reopen this only
   after a positive real-data information-state diagnostic.

The immediate performance question is therefore:

> Can a properly trained teacher-free SPINT-like backbone close the gap to A2 before any new
> decoder architecture is required?

## 1. Current development evidence

The current evaluator reports the mean of six equal-session R² values, then averages over the
eight epoch-5–12 checkpoints. All TFPD rows use the same evaluator and query batches. The A2 row is
a historical reference under a different metric/checkpoint authority and is not yet a matched peer.

| Arm | Live parameters | T4 | Z4 | T4−Z4 | Wrong pair | Zero | Destroyed activity |
|---|---:|---:|---:|---:|---:|---:|---:|
| spintshape, no teacher | 3,510,842 | **0.4574** | **0.1985** | +0.2589 | −0.0822 | +0.0006 | +0.0505 |
| Large-v1 | 5,016,194 | 0.2562 | 0.0059 | +0.2503 | −0.2189 | −0.7657 | −0.0224 |
| small bilinear | 22,434 | 0.0767 | 0.0216 | +0.0551 | −0.3028 | −0.3293 | −0.1667 |
| ~~population vector~~ **VOID** | 17,908 | ~~0.0012~~ | ~~0.0037~~ | — | — | — | — |
| A2 sealed teacher-initialized reference | — | 0.5750 | 0.3260 | +0.2490 | — | — | — |

**The population-vector row is void — never cite `pv = 0.0012` as evidence about population
vectors.** `tfpd_exploration/src/tfpd/population_vector.py` computes its preferred direction as
`[a, c] / max(m, eps)` assuming a **raw** carrier, but `multisession_datamodule.py` z-scores T4
before the model sees it. With `m` z-scored, 65.7% of units get `m_z` clamped to `1e-6`, producing
direction vectors of median magnitude 3.24×10⁵ — the arm was numerically saturated, not
informative. It passed the Stage-0 synthetic gates only because `tfpd/synth.py` generates carriers
in raw units without z-scoring, so G5 does not transfer to real data. The fix is for PV to invert
the z-scoring internally from stored mean/std buffers. **Consequence: the lane currently has no
valid transparent baseline**, so no claim of the form "attention beats a simple population vector"
can be made from this table.

**The estimator used for this table is superseded.** These are means over eight epoch-5–12
checkpoints, which correspond to no deployable model and are biased upward relative to A2's metric.
For matched-scorer final-four-SWA numbers, and for external scores, use §0A.

Authoritative development receipts are under:

```text
tfpd_exploration/results/stage1_source_cells_v1/{bl,pv,large,spintshape}_*/
tfpd_exploration/results/stage1_source_cells_v1_r1/pv_z4/
```

The strongest direct conclusion is that teacher-free spintshape is substantially better than
Large-v1 under the same TFPD evaluator. The observed gap is `0.4574 − 0.2562 = 0.2012`, but this is
a **system gap**, not a causal estimate of calibration identity. The systems also differ in
per-unit read-in, fusion, temporal topology, and parameter allocation.

The similar T4−Z4 contrasts across spintshape, Large-v1, and A2 are descriptive. R² values from
different models are not additive information quantities and must not be summed to predict a
successor or define a kill threshold.

## 2. Why SPINT-like structure is now the performance starting point

SPINT is already a calibration-conditioned, permutation-invariant set-attention decoder:

```text
calibration trials -> shared per-unit identity encoder
query neural window + unit identity -> shared per-unit read-in
learned static query slots -> cross-attention over a variable-size unit set
attention output -> behavior prediction
```

It has no persistent per-unit parameter table and supports variable unit count. Therefore the
following are not useful differentiators from SPINT:

- permutation invariance;
- variable-N support;
- learned query-slot attention;
- absence of a persistent NeuronID lookup table.

This does not make SPINT-like work scientifically invalid. It changes where improvement and
novelty must come from:

- better optimization and source pretraining;
- better use of calibration information;
- task-frame carrier construction;
- better separation of routing keys and neural values;
- lower deployment representation cost;
- better cross-subject performance;
- better performance/compute Pareto trade-offs.

The main objective remains absolute decoding performance. Structural novelty is valuable only
when it contributes to that objective or produces a meaningful efficiency frontier.

## 3. Twelve epochs are not a proven performance ceiling

A2's 12-epoch task-only stage starts from a mature teacher checkpoint trained to approximately
epoch 83. Teacher-free spintshape and Large-v1 start from random initialization and receive only
12 epochs. Their total optimization budgets are not comparable.

At the same time, neither current teacher-free T4 curve is still rising at epoch 12.

Large-v1 native R² over epochs 4–11:

```text
0.2626  0.2704  0.2760  0.2325  0.2461  0.2623  0.2546  0.2447
```

Spintshape native R² over epochs 4–11:

```text
0.4417  0.4448  0.4746  0.4745  0.4485  0.4721  0.4556  0.4471
```

These curves show plateauing under the current constant-`1e-4` recipe. They do **not** prove an
architecture ceiling. A longer run with the same constant learning rate has a low prior, while a
longer schedule with decay and proper diagnostics remains justified.

The convergence experiment should use:

- the sealed epoch-11 spintshape-T4 checkpoint as an explicitly bound parent, or a fresh run with
  an exactly frozen equivalent long schedule;
- fixed milestones such as epochs 12, 24, and 48, with any later milestone frozen before launch;
- a predeclared learning-rate decay rather than blind constant-LR continuation;
- no formal data and no target-guided early stopping;
- immutable per-epoch records of train loss, source-held-out performance, learning rate,
  unclipped gradient norm, clipping frequency, branch gradient norms, and attention statistics.

Interpret the outcome as follows:

| Long-horizon T4 gain over matched 12-epoch baseline | Decision |
|---:|---|
| `>= +0.03` | Training budget/schedule is material; adopt and confirm. |
| `< +0.01` | Close the horizon hypothesis; move to fusion/identity design. |
| `+0.01` to `+0.03` | Keep the better recipe but do not treat training as the main explanation. |

If train loss continues down while held-session performance is flat or falling, the problem is
generalization, not insufficient epochs. If both plateau, the bottleneck is optimizer, model, or
information path. If both improve, the 12-epoch limit was material.

## 4. Metric and checkpoint policy must be fixed first

A2 preserves the two output dimensions and uses variance-weighted R². The TFPD evaluators expand
the validity mask to the full prediction tensor and index it, flattening both covariates into one
vector. The flattened denominator uses the grand output mean and is at least as large as the
per-output denominator. The current TFPD score is therefore biased upward relative to A2's metric;
metric parity will not create hidden headroom.

The six within-development sessions are otherwise the same A2 within-subject sessions, and their
query windows exclude the M30 calibration trials.

Two checkpoint estimands should be reported separately:

1. **Historical stability:** the existing fixed epoch-5–12 mean. Preserve it because it was the
   original development summary.
2. **Deployable model:** one exact checkpoint chosen by a predeclared source-only rule or a fixed
   milestone. Do not select a checkpoint on the six development sessions and then report those
   same sessions as an unbiased evaluation.

All existing models should be rescored before comparing exact gaps or freezing successor gates.

## 5. What Large-v1 established

Large-v1 should remain terminal. It is useful as a T4-only compression and architecture baseline.

Its central token is:

```text
token_i,t = activity_projection_i,t * carrier_projection_i
```

This has three important consequences:

1. There is no carrier-independent activity residual.
2. The same multiplied token supplies attention keys and values.
3. Zero carrier removes unit-specific identity and moves token geometry off the trained manifold.

Therefore `wrong-pair > zero` is not evidence that Large-v1 attenuates misleading identity. The
wrong-pair carrier preserves the trained carrier distribution while misassigning rows; zero erases
identity and changes the gate distribution. The honest correspondence diagnostic is the native to
wrong-pair drop:

```text
Large-v1:   0.2562 -> -0.2189
spintshape: 0.4574 -> -0.0822
```

Both systems use unit correspondence. Large-v1 does not convert its 5M parameters into competitive
absolute performance because it discards calibration identity, uses a destructive fusion operator,
and allocates much less capacity to per-unit activity encoding.

Do not spend primary compute on a wider Large-v1, more slots, more heads, or a larger GRU. A single
long-horizon Large-v1 continuation may be run only as a low-priority ceiling diagnostic after the
spintshape experiment, not as the main performance path.

## 6. Minimal architecture successor if training is insufficient

Use the trained teacher-free spintshape family as the backbone. Preserve calibration identity,
static query slots, attention capacity, neural value path, and output head. Change only how T4
enters attention:

```text
base_i = encode_query_activity_i + encode_calibration_identity_i

key_i   = W_key(base_i) + alpha * W_t4(T4_i)
value_i = W_value(base_i)
```

Requirements:

- `alpha` or the T4 key branch is direct-zero initialized;
- zero T4 exactly recovers the calibration-only baseline at initialization;
- T4 affects routing keys but does not overwrite neural values;
- T4/Z4 siblings share exact initial state and training schedule;
- no teacher parameters are required for the teacher-free comparison;
- no FiLM value path, GRU/SSM, dual sum, reliability gate, extra slots, or extra width in the first
  test.

This tests one question:

> Is T4 more useful as a routing key than as additive identity content?

**Prior evidence bracket — read before implementing (added 2026-08-17).** This design is untested,
but it sits between two terminal negatives, and the nearby variant that someone will reach for
first is already refuted.

- **Do not implement it as key *replacement*.** Sourcing keys from identity/T4 and values from
  activity was measured: `kv_e_t4 = 0.1392` against `coupled_t4 = 0.5838`, i.e. **`−0.4447`, 0/6
  sessions positive** (`sua_exploration/results/sua_t4_decoupled_kv_v1/aggregate_seed42.json`;
  `kv_e_only = 0.1740`). The sketch above survives that receipt *only* because it keeps `base_i`
  — activity plus calibration identity — inside the key and adds T4 as a zero-initialized
  residual. Removing activity from the key costs roughly three quarters of performance.
- **The adjacent residual has already returned near-zero.** A1 tested a hidden-space T4 residual on
  the token, `h = fc_in(x + activity_identity) + P(T4)`, and returned `+0.0128` against a `+0.03`
  pre-registration: `PILOT_ROUTING_STOP`
  (`sua_exploration/docs/HANDOFF_NEXT_ROUND_DIRECTIONS_20260812.md`). The key-only residual differs
  only in restricting that residual to the key projection.
- **Note what the question is actually asking.** `W_key(base)` and `W_value(base)` are separate
  projections of one source, which is what `nn.MultiheadAttention` already does
  (`CrossAttentionLayer:30-34` passes `key=value=...`; `in_proj_weight` supplies distinct
  W_Q/W_K/W_V). And T4 **already reaches the key today**, via B3S inside `id_i`. So this cell does
  not test "T4 in keys vs T4 in content" — it tests whether an *additional, key-restricted* T4
  pathway helps on top of T4 already being in the token. Phrase it that way in the receipt.

Net: keep it as the fusion-successor candidate, but its prior is poor and it should stay behind the
directions ranked in `HANDOFF_SPINT_DECODER_DIRECTIONS_20260817.md` §5, whose Priority 2 instead
attacks the decoder's lack of any temporal model. Its gates below must also be restated against
Arm A (within 0.5163 / external 0.1610), not against 12-epoch spintshape.

Run one seed-42 T4 cell first. Only if its matched within-development improvement is practical
should the Z4 sibling and external pair be completed.

Suggested sequential gate:

- within T4 mean lift over matched spintshape `>= +0.03`;
- within session-median delta `> 0`;
- after the T4 screen passes, Z4 non-inferiority `>= -0.03`;
- external T4 mean lift `>= +0.03`, median delta `> 0`, and at least 9/15 sessions positive;
- interaction and wrong-pair robustness are diagnostic only.

If the minimal key-only model fails the absolute T4 gate, close the fusion route. Do not rescue it
with FiLM, dynamic attention, dual aggregation, or additional capacity.

## 7. Ablations: small, staged, and decision-linked

Ablations should follow a positive performance result rather than precede it.

Use the smallest useful set:

1. **T4/Z4 sibling:** mandatory only after a T4 candidate passes its performance screen.
2. **Wrong-pair evaluation:** forward-only correspondence diagnostic; no retraining.
3. **Destroyed/zero activity:** forward-only shortcut diagnostic; distinguish zero activity,
   cross-trial activity swap, temporal scramble, and calibration destruction if a shortcut appears.
4. **Identity-source 2×2:** run only if needed to support the final mechanism claim. Use one
   identical backbone with calibration `{off,on}` and T4 `{off,on}`, exact shared initialization,
   and branch masking rather than changing model shape.
5. **FiLM, dual aggregation, reliability, or compression sweeps:** run only after the minimal
   key-only successor is positive.

Do not use the numerical coincidence

```text
spintshape Z4 + Large-v1 T4 ~= spintshape T4
```

as an R² decomposition, a prediction, or a kill rule. The models differ and R² is not an additive
information measure. A matched identity-source 2×2 is the proper test if that decomposition becomes
scientifically necessary.

## 8. Performance lane and compression lane

These are distinct claims and should not share a gate.

### Performance lane

Allowed inputs and methods:

- raw M30 calibration neural activity;
- T4 and other lawful calibration-derived features;
- SPINT-like set attention;
- source pretraining or teacher initialization;
- longer optimization;
- residual/keyed fusion.

Goal: maximize matched within and external R² and ultimately compare with the best A2 system.

Teacher initialization is allowed if it improves performance, but it should be described as
initialization or source pretraining, not presented as a meaningful distillation contribution when
no compression/equivalence claim is being tested.

### Compression lane

Restricted representation:

- T4-only or another compact per-unit descriptor;
- no raw calibration tensor consumed by the deployed decoder.

Goal: establish the best performance/representation-state/compute Pareto point. Large-v1's `0.2562`
is a valid baseline here even though it is not the best absolute system.

T4 uses calibration neural activity and task labels to construct its descriptor, so its advantage is
primarily downstream representation bandwidth, storage, and decoder state—not necessarily reduced
calibration acquisition.

A successor that consumes both raw calibration activity and T4 belongs to the performance lane and
cannot simultaneously claim a pure 4N deployment representation.

## 9. Source pretraining and teacher initialization

The currently observed A2-versus-spintshape gap is not a causal teacher-initialization effect.
The metrics and checkpoint policies differ, and the TFPD metric is upward biased. The unmatched
numerical difference `0.5750 − 0.4574 = 0.1176` cannot be labeled an upper bound on teacher value.

After the long-horizon audit, compare training regimes honestly:

| Regime | Interpretation |
|---|---|
| random initialization, 12 epochs | current teacher-free baseline |
| random initialization, long schedule | training-budget ceiling test |
| independent source pretraining + fine-tuning | teacher-free pretrained system |
| teacher initialization + fine-tuning | A2-like performance system |

If long teacher-free training closes most of the gap, the main benefit was optimization budget and
representation maturity. If it does not, teacher/source pretraining remains a legitimate performance
component, but not a standalone architectural innovation.

## 10. Adoption rule

The final system is judged by absolute performance under a matched metric and protocol.

- Use absolute external T4 improvement as the primary adoption gate.
- Use within-development performance as a screen and non-inferiority guard.
- Report equal-session mean, median, per-session deltas, positive-session count, and multi-seed
  stability.
- Do not let a large carrier interaction rescue an absolute gain below the practical threshold.
- Compare the promoted system with the strongest available system, including A2, not only with
  Large-v1.
- Keep formal/organizer-held data sealed until architecture, training schedule, checkpoint policy,
  and gates are frozen.

Mixing and swap-v2 already establish the precedent: a mechanism can work while the system-level
absolute gain remains too small for adoption.

## 11. Do not do

- Do not avoid SPINT-like structure merely to claim architectural independence.
- Do not expand Large-v1 width, heads, slots, or GRU as the primary route.
- Do not blindly extend the same constant learning rate.
- Do not choose the best epoch post hoc on the evaluation sessions.
- Do not spend several GPU cells on ablations before a T4 candidate improves absolute performance.
- Do not interpret native-minus-zero as pure semantic carrier value when carrier is the only identity
  channel.
- Do not use the cross-model R² sum as a decomposition or threshold.
- Do not add state-conditioned SSM/attention before a positive real-data pre-gate.
- Do not compare exact A2 and TFPD numbers before scorer and checkpoint parity.

## 12. Short research claim if the route succeeds

The performance claim should be simple:

> A properly trained calibration-conditioned set decoder achieves stronger cross-session and
> cross-subject decoding when a compact task-frame carrier controls unit routing without corrupting
> neural values.

The efficiency claim, if separately supported, is:

> A compact per-unit task-frame descriptor retains a useful fraction of full calibration-identity
> performance at substantially lower deployed representation cost.

## 13. Questions for independent review

1. Is one long-horizon spintshape-T4 run the correct first use of GPU time?
2. What fixed long-horizon learning-rate schedule and milestones best test the 12-epoch ceiling
   without becoming a broad sweep?
3. Should the first long-horizon run resume the sealed epoch-11 optimizer state or retrain from the
   shared random initialization under a fresh long schedule?
4. Is T4-as-key-only the cleanest minimal fusion successor after the training audit?
5. What source-only checkpoint selector permits one deployable checkpoint without using the six
   within-development sessions for both selection and evaluation?
6. What exact matched threshold should be required for promotion against spintshape and then A2?
