# Dual-calibrated temporal decoder exploration V1

Date: 2026-09-05 (Asia/Hong_Kong)
Status: DESIGN_FOR_IMPLEMENTATION_PLANNING
Scope: independent learned decoder exploration; no new training or decoder R2 scoring performed by this design note.

Execution successor: [two-track workorder](WORKORDER_CALIBRATION_MEMORY_AND_TEMPORAL_DECODER_PARALLEL_V1_20260905.md). The user's two parallel tracks are per-trial query-conditioned calibration memory and independent temporal decoder exploration, not two decoder arms. That workorder supersedes this note's single-GPU/two-week scheduling and defers D2/D3 and full ablations; it also requires a newly replayed M33-disjoint ext-4 reference rather than reusing historical six-session scores.

## 1. User direction and objective

The user explicitly wants an independent decoder research track in addition to calibration-interface research. Small model size is not an absolute requirement. Modern state-space models are in scope. Population-vector decoders, PV initialization, and PV performance gates are excluded from this track.

Primary objective: establish a strong causal decoder that retains calibration activity signature and tuning profile, and test whether its treatment of time and population aggregation improves cross-session decoding. Parameter count and runtime are measured outcomes, not admission gates requiring 0.2M parameters or a 100x speedup.

This track can produce either a better decoder or evidence that the dual-calibration interface transfers to a second decoder family. An ordinary SSM/attention combination is not itself a new architectural invention; stronger novelty requires a demonstrated calibration-specific mechanism.

## 2. Evidence that constrains the new track

### 2.1 TKD does not exhaust this question

`src/fable_tkd_m2_v1/model.py:489` accepts query activity, T4, and fitted reliability. It does not accept the original calibration activity signature. Its per-unit temporal encoder is `Conv1d -> Linear` without an intervening activation (`:476`). It also changes model size, initialization, temporal backbone, and pretraining history.

The M2 result is a negative result for that implemented system. The H1/688 closed-form preflights do not bound what a trained nonlinear temporal decoder can achieve. Neither a weak PV result nor a weak ridge result is a prerequisite failure for the current track.

### 2.2 A stronger negative ancestor must also be disclosed

TF-SR did preserve fused B3S/T4, use nonlinear read-in and whole-unit dropout, and train a state-conditioned two-slot population read-in followed by a GRU. On the reported seed-42 matched DANDI surface, external R2 was 0.254173 versus Cell D 0.417936, despite a small within-session improvement.

Source: `HANDOFF_CURRENT_TFSR_POSTERIOR_PIRG_RESULTS_20260823.md`, sections 2-3; architecture in `HANDOFF_TASK_FRAME_STATEFUL_READIN_20260819.md`, section 4.

Consequently, simply restoring activity signature, or appending a recurrent layer, is not a new untested solution. New candidates must distinguish aggregation order, capacity, numerical implementation, and train/test context; external transfer remains an empirical risk.

### 2.3 Read-only SSM counterexample discovered during this review

The historical `DiagSSM.step()` clamps the recurrent state at every update. Its `forward()` first computes the unclipped linear convolution and then clamps each output state. These are different functions whenever intermediate clipping affects a later state.

For decay 0.9, clipping limit 10, initial state zero, and scalar inputs `[20, -10]`:

| Quantity | Step recurrence | Convolution then clipping |
|---|---:|---:|
| First state | 10 | 10 |
| Second state | -1 | 8 |

A CPU-only check of the actual class with `d=1`, `B=C=1`, residual gain 0.5, and a constant sigmoid gate 0.5 returned:

```json
{"forward_output": [22.5, -8.0], "step_output": [22.5, -10.25], "max_abs_diff": 2.25}
```

Relevant code: `src/fable_tkd_m2_v1/model.py:119` and `:133`. Historical training and offline evaluation call `forward()`; therefore this counterexample does NOT establish the cause of the reported low offline R2 and does not invalidate the independently trained GRU result. It refutes unconditional streaming/parallel parity of this clipped implementation. Historical code and result artifacts were not changed.

The successor uses a pinned reference SSM implementation, not this implementation with another numerical patch. No internal-state clamp may be inserted into a linear convolution model while claiming the same linear recurrence.

## 3. Related work and attribution

POSSM directly motivates population tokenization followed by temporal recurrence. Its NHP table lists pretrained variants around 4.56M (S4D), 7.96M (GRU), and 8.96M (Mamba) parameters. Thus the relevant precedent is not limited to 0.2M models. It evaluates S4D, GRU, and Mamba; no result licenses assuming Mamba wins every task.

Its unit-identification and full-finetuning transfer routes use target-session optimization, unlike the intended frozen target decoder here. It also uses spike-event tokenization, whereas our first comparison preserves the existing binned input. We will call our model POSSM-inspired, not a POSSM reproduction.

Sources:

- [POSSM, NeurIPS 2025, methods and Appendix B.3](https://arxiv.org/html/2506.05320v2).
- [Mamba-2 / structured state-space duality](https://arxiv.org/abs/2405.21060).
- [Official Mamba implementation](https://github.com/state-spaces/mamba).
- [SPINT](https://arxiv.org/abs/2507.08402): activity-derived identity, permutation-invariant population decoding, and dynamic channel dropout are inherited prior art.

Mamba-2 is an implementation candidate, not a claim that POSSM used Mamba-2. Repository commit, library versions, initialization and step/scan behavior must be fixed before training.

## 4. Candidate inventory and convergence

| Candidate | Decision | Reason |
|---|---|---|
| Pure T4/PV-keyed tiny decoder | Excluded | User excludes PV; removes activity calibration and conflates compression with adaptation |
| Flatten units into a temporal Mamba feature vector | Excluded | Hard-codes roster dimension/ordering |
| Scan arbitrary unit order with Mamba | Excluded | Treats a neuron permutation as a meaningful sequence |
| Bidirectional temporal Mamba | Excluded from online comparison | Future neural samples change the information contract |
| Set encoder -> temporal Transformer | D0, required | Learned temporal reference with the same new frontend |
| Set encoder -> Mamba-2 | D1, first SSM candidate | Separate population fusion from temporal memory |
| Set encoder -> reference S4D | Reserve component comparison | Direct POSSM-adjacent SSM baseline, after frontend is viable |
| Set encoder -> GRU | Reserve component comparison | Checks whether recurrence, rather than selectivity, explains a gain |
| Shared per-unit Mamba-2 -> set readout | D2, second structural candidate | Preserve individual temporal patterns before cross-unit compression |
| Per-unit SSM -> set -> population SSM | D3, conditional extension | Tests distinct unit and population timescales |
| Query-conditioned calibration memory | Separate calibration track | Avoid confounding memory access with decoder replacement |
| Carrier-conditioned SSM dynamics | Deferred mechanism experiment | Conditional state equations need evidence; a renamed FiLM is insufficient |

Only D0/D1/D2 form the first architecture comparison. D3 and alternative temporal components are not an automatic full sweep.

## 5. Shared information contract

For each unit i, retain:

```text
X                     [B,T,N]        live binned neural observations
A_i = mean_j phi(S_ij) [B,N,H_A]      calibration activity signature
T_i                   [B,N,4]        tuning profile, using the selected dataset estimator
E_i = psi([A_i,T_i])   [B,N,H_E]      existing joint calibration representation
```

The first decoder screen uses the same checkpoint-bound calibration encoder, legal support indices, carrier estimator, normalizer and unit roster across D0/D1/D2. Calibration modules are frozen for this screen so decoder training cannot covertly change the information interface. This does not imply they must remain frozen in a later product model.

A common nonlinear local frontend forms `u_i(t) = F([local_neural_i(t), E_i])`. Keys and values may both use this fused token. There is no mandatory cosine-R2 multiplier, identity-only key constraint, or carrier-derived hard exclusion of units.

The temporal local frontend is causal. Its state/padding is included in the total history budget. All candidates consume identical binned observations; no candidate obtains precise spike times while a comparator receives only bins.

No session/unit lookup table is fitted on target data. Source optimization remains allowed. Activity signature and tuning profile are calibration data products, and dynamic model hidden state is recorded separately from either calibration memory.

## 6. D1: population-first temporal decoder

```text
causal local unit features + joint calibration E_i
    -> nonlinear shared unit token [B,T,N,d_set]
    -> learned set cross-attention over N units
    -> L population tokens per bin [B,T,L,d_set]
    -> within-bin slot mixing / concatenate-project
    -> population sequence [B,T,d_time]
    -> temporal Mamba-2 blocks
    -> task readout at the scored timestamps
```

D0 is identical except for a causal temporal Transformer in place of Mamba-2. D0/D1 use the same frontend topology and seed-matched common initialization; temporal modules have their own initialization domains. Layer/head/FFN settings are selected before decoder development scoring.

Provisional capacity point for D1: `d_set=256`, `L=8`, concatenate-project to `d_time=512`, four Mamba-2 blocks, reference expansion 2, state size 64. Count actual parameters before freezing the implementation. A wider/deeper model is allowed if source learning curves demonstrate a capacity problem; there is no small-model performance gate.

Do not call L=8 sufficient by definition. It is an initial compression choice to measure. A later L=16 control is justified only if the current frontend is the diagnosed bottleneck, not by an automatic grid.

Mechanism hypothesis: temporally integrated population features can improve decoding while activity/tuning calibration makes the per-time population features portable across sessions. This is a hypothesis, not a theorem about SSMs.

## 7. D2: unit-time-first temporal decoder

```text
causal local unit features + joint calibration E_i
    -> shared Mamba-2 applied independently along each unit's TIME axis
    -> contextual unit states at scored time [B,N,d_unit]
    -> set cross-attention over units
    -> learned population readout
```

All units share temporal-model weights. Their dynamic states are separate and follow the unit roster; no learned identity table is introduced. Permuting query units and their calibration rows together permutes the intermediate states and leaves the final prediction invariant.

D2 moves temporal modeling ahead of population compression. A unit's burst, silence, lag and short-term dynamics can be represented before units are aggregated. It does not assume activity signature uniquely measures these properties or that T4 directly inverts spikes into kinematics.

D2 has more per-bin computation/state because recurrence is replicated over N units. This is an explicit accuracy/compute tradeoff, not grounds to exclude it under the user's revised objective. Use a concrete, counted configuration reasonably near D1's trainable parameter scale; runtime need not match. A smaller unit-state width may be used to fit memory, with the difference disclosed.

If D2 exceeds D1, follow-up must include an approximately parameter-matched unit-time Transformer/TCN or a population-width control before attributing the result solely to aggregation order.

## 8. D3: hierarchical unit and population time

Only after D1 or D2 shows usable external transfer:

```text
shallow per-unit temporal model
    -> population set aggregation
    -> population temporal model
    -> task readout
```

Proposed division of labor: local unit dynamics before aggregation; shared behavioral/population dynamics after aggregation. Both terms must earn their role through paired deletion controls. An equal-budget flat temporal model is necessary to distinguish hierarchy from added depth.

Do not add query-conditioned calibration memory or new CP-FiLM profiles in this comparison.

## 9. Fair training and scoring

### Two distinct reference levels

1. Matched architecture reference: D0, with the same source data, frozen calibration inputs, training exposure and selection access as D1/D2.
2. Operational reference: current best SPINT-family checkpoint on the identical scoring surface. Its different pretraining and selection history remain explicit.

A new model below the operational reference but above D0 demonstrates promise under the tested training budget; it does not yet replace the product. Matching a pretrained champion after randomly initializing only the competitor is not a fair causal test of backbone quality.

### Exposure and selection

- Begin with the familiar 12-epoch matched screen, recording unique scored targets, optimizer steps and source-session exposure. Twelve epochs is a checkpoint in exploration, not assumed convergence for a fresh decoder.
- If models are still improving, decide a shared extension to 24/48 epochs from source/development learning curves under a written rule; retain the original 12-epoch endpoint. Do not extend only an attractive external-score arm.
- Use identical labeled query timestamps and objective normalization. Last-bin supervision and sequence supervision must not silently create different target multiplicities.
- For the first fixed-context comparison, shuffle windows normally but reset all temporal state at the start of each input window.
- Match the whole-unit dropout distribution and hold its mask constant over the window. Drop complete fused unit tokens; do not introduce flickering per-bin unit membership as an accidental third experiment.
- Choose candidates using declared visible development data. Once used for selection, that surface is development evidence, not an independent confirmation set.
- Use paired seeds 42/43 for screening where feasible; complete 42/43/44 for a superiority claim. Report paired session effects, seed effects and selection policy. No retrospective conversion of a null into non-inferiority without a defined margin and interval.
- Distillation is optional later, uses source inputs only, and is compared at matched teacher access. Output distillation alone does not guarantee out-of-source behavior.

### Readout and context

First compare all models on the exact history available to the existing dataset-specific baseline. This isolates architecture from access to extra history. For M2 the inherited reference window is 50 bins; other tasks inherit their actual window, not an assumed universal 1 s.

Only then run a separately reported persistent-state experiment. Process each new bin once, carry state through chronologically valid chunks, detach gradients at TBPTT boundaries without resetting numerical state, and reset only at the dataset-authorized boundaries. Never feed overlapping full windows repeatedly into an already persistent state.

Extra historical input is an additional resource even when legal. Persistent-state gains must be distinguished from fixed-context gains. Offline chunking, online callbacks, warm-up scoring and mask semantics must be made consistent before any deployment claim.

## 10. Three validation experiments

1. **Backbone and aggregation order:** D0/D1/D2 on identical frozen dual-calibration inputs, fixed context, and training exposure. Report accuracy before discussing speed. Compare within/source learning to external development, preserving the TF-SR transfer-failure warning.
2. **Calibration content:** on a viable decoder, measure activity-only, tuning-only and dual-calibration models with matched source training, plus meaningful row-pairing controls. If a fused parent E already contains T4, removing a separate T4 port is not a tuning ablation. Removing A/T only at scoring measures reliance/OOD sensitivity, not retrained marginal value. Do not quietly retain the supposedly removed information inside an inherited representation.
3. **History versus mechanism:** fixed-context versus persistent-state scoring, alongside a matched recurrent/Transformer reference where practical. A history benefit is reported as such. SUA/pseudo-MUA comparisons use matched electrode-derived views, support labels and scoring windows; do not presuppose the effect direction.

For a larger-is-better effect, explicitly report the matched new-decoder contrast and operational comparison. For a dual-calibration claim, report `Delta_T_given_A` and `Delta_A_given_T` under the same model family. Positive interaction is a stronger optional claim, not a prerequisite for ordinary complementarity.

## 11. Numerical and data checks before training

- Real nondegenerate tensor shapes, padding masks and a joint unit-permutation test.
- Prefix causality: perturb future neural samples and verify earlier predictions are unchanged.
- Sequence versus step, chunk versus unchunked, and cache-reset agreement on random and high-amplitude inputs; nonzero residual branches must be exercised.
- All SSM and local-convolution states covered by restart tests. A zero-initialized inactive branch alone is not a valid numerical test.
- No unreviewed state clamp; use stable reference parameterization, suitable initialization, residual normalization and training gradient clipping with explicit settings.
- Accurate input units: counts/bin versus rates/second, source-frozen normalization, actual available calibration trials.
- Count parameters, activation memory and measured examples/s on the actual N/T distributions. Arithmetic operation counts alone do not establish speed.
- Pin reference implementation/version. Mamba CUDA throughput and CPU deployment support are separate engineering questions; an unavailable fused kernel is not a scientific NO-GO for the architecture.

## 12. Initial scope and timebox

Use M2 as the first implementation surface because its binned input, selected parents and scoring harness are present. Freeze the exact current carrier/support authority before coding; do not accidentally inherit the older TKD whole-trial carrier merely because its loader is convenient. DANDI 000688 SUA/pseudo-MUA is a second mechanism surface after a working candidate; H1 follows for M3 calibration transfer. This is a priority order, not authority to run every surface immediately.

Indicative two-week feasibility box, revised after throughput measurement:

- Days 1-2: common frontend, D0/D1 implementation and numerical contracts; exact parameter/resource accounting.
- Days 3-6: paired D0/D1 learning curves and D2 implementation. Respect the current single-GPU constraint; concurrent arms only when measured memory and utilization support it.
- Days 7-10: complete the fixed-context screen and classify optimization/transfer failures. Apply a common schedule extension only if warranted.
- Days 11-14: calibration-content controls and/or the second paired seed on a viable candidate. Do not promise a completed four-dataset, three-seed matrix in this box.

The strongest objection is that TF-SR already showed a population/recurrent model overfitting source sessions. The answer is not to assume Mamba fixes it: D0/D1/D2 explicitly measure temporal-backbone effects and aggregation order with the complete calibration information, adequate capacity and a common training budget. If source fit improves but target transfer again declines, that is a negative decoder result, not evidence of a universal SSM limitation.

## 13. What can become a paper contribution

- An effective second decoder can strengthen the portability of activity+tuning calibration even if its SSM component is standard.
- A reproducible ordering effect, or a calibration-specific gain beyond a matched temporal baseline, can support a distinct architectural design contribution.
- Merely replacing a Transformer with Mamba, reaching PV initialization parity, or showing a small model is fast cannot establish that contribution.
- Target-session no-backprop adaptation, sparse/dense calibration information and task-specific carrier estimation must be reported explicitly beside related work.

No PV branch or PV gate remains in the proposed execution path. No historical result root or checkpoint was changed by this note.
