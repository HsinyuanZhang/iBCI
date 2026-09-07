# Handoff: One-Factor Causal Temporal Decoder after Cell D

Date: 2026-08-19

Status: design correction only. This revision supersedes the earlier CTF-SD proposal in this file.
It authorizes no code implementation, no source training, no target access, and no GPU launch.

## 0. Executive decision

Performance remains the primary objective. Stop spending new rounds on SPINT-neighborhood additions
such as another dropout shape, head count, consistency loss, fusion point, or output residual.

The previously proposed first cell, `CTF_SD_T4_D_SEED42`, is revoked. It was not one decoder change
relative to Cell D. It combined an encoder-information cut, a new fusion rule, a new set reader, a
new temporal model, a width change, a slot change, and a third sparsification law. A result from that
cell could only support a system-level comparison; it could not establish a causal task-frame
factorization or identify why performance changed.

The remaining research question must be split into two bets:

1. **Bet A -- untested temporal structure on the sealed Cell D graph.** Keep Cell D's fused
   B3S+T4 identity, identity application, two coordinate-tied query slots, width, attention block,
   whole-unit sparsification law, data, and training recipe fixed. Change one temporal-structure
   factor only. This is the only architecture bet that may eventually be compared to D as an
   attributable experiment.
2. **Bet B -- T4-only set reader plus recurrence.** This is not new. Large-v1 already tested this
   family and failed badly relative to D. It is not the next performance route. At most,
   `Large-v1 + D dropout` could be retained as a narrow diagnostic against sealed `large_t4`, using
   the original 12-epoch Large recipe. It must never be presented as a route that isolates a new
   factorization against D.

No first GPU cell is frozen by this document. A future Bet A cell remains forbidden until its
factor table contains exactly one changed scientific factor relative to D.

## 1. Evidence that constrains the redesign

### 1.1 Cell D is a fused system, not an activity-only decoder

The live Cell D path is:

```text
M30 calibration activity -> B3S pre_pool -> mean over calibration trials
normalized T4 ----------------------------------------------/
                                  -> post_pool(concat(mean, T4)) -> id_i[50]

query activity_i[50] + id_i[50]
    -> Cell D whole-unit dropout on the fused unit token
    -> shared 50 -> 512 -> 512 read-in
    -> two coordinate-tied query slots
    -> one cross-attention block
    -> direct 50-bin output
```

Code anchors:

```text
streaming_calibration_exp/src/models/components/streaming_encoders.py:414-444
streaming_calibration_exp/src/models/components/spint.py:445-455
```

The important fact is that Cell D identity is produced from **both** calibration activity and T4.
The query unit token then contains query activity plus that fused identity. Therefore:

- A T4-only model removes information that D uses.
- Removing B3S is an encoder-information intervention, not a decoder-only intervention.
- The Z4 external result of `-0.2936` means that activity-derived identity is harmful when used
  without T4. It does not show that calibration activity should be removed from the fused encoder.
- Large-v1 already showed that a T4-only recurrent decoder can lose to fused SPINT.

Current parameter accounting for the D graph is:

| component/accounting item | parameters |
|---|---:|
| total registered model | 3,510,842 |
| attention block | 3,152,384 |
| FFN inside attention block | 2,099,712 |
| B3S identity encoder | 18,290 |
| registered but dead `fc_id_out` | 25,650 |

This parameter budget is correct, but it does not make a T4-only replacement a matched decoder
experiment.

### 1.2 Whole-unit sparsification is D's isolated active ingredient

At the governing convention -- last-bin, variance-weighted R2, equal session weight, seed 42 -- the
completed decomposition found:

```text
Cell D external                         0.4179
Cell D within                           0.5697
Cell D - Arm A external                +0.1576
Cell T - Cell D external               -0.0099
```

The R/G/T/S2/C controls isolated the useful treatment as stochastic whole-unit deletion during
training. Carrier-sector structure, elementwise corruption, gain alone, and an explicit consistency
loss did not explain or improve the effect.

Grafting this treatment onto a new T4-only decoder is scientifically fair only if the claim is
"this complete system performs well." It is not fair if the claim is "a new task-frame decoder
factorization beat D," because the architecture and the known positive training treatment change
together.

### 1.3 D's mechanism sentence is still being resolved by AM/IM

AM and IM are not proposed future experiments. They have already been launched under:

```text
tfpd_exploration/results/aimask_v1/cellAM_activity_mask
tfpd_exploration/results/aimask_v1/cellIM_identity_mask
```

They ask whether D's gain comes from masking activity, masking the B3S-derived identity/fingerprint,
or jointly removing the complete fused token. Their terminal results must be allowed to finish.

If `IM ~= D`, D's gain is primarily consistent with fingerprint ablation. A T4-only model has no
B3S fingerprint to ablate, so importing D dropout into Large-v1 would no longer have the same
mechanistic meaning. If the operator proceeds before AM/IM terminate, the mechanism sentence must
remain explicitly unresolved.

`behavior_scaling_factor` remains teacher-free `None`. It must not change in any successor.

### 1.4 The recurrent T4-only family has already been tested

The nearest live ancestor of the revoked CTF-SD proposal is Large-v1, not POSSM.

`tfpd_exploration/src/tfpd_large.py` implements:

```text
activity token * carrier token
    -> four learned query slots
    -> cross-attention over units
    -> causal GRU(256)
    -> velocity head
```

Large-v1 is T4-only, has no B3S path, and uses no learned unit table. Its relevant results are:

```text
large_t4 within                         0.2562
large_t4 external native              -0.0848
bilinear + GRU external               -0.9597
```

Large-v1 was declared terminal in
`HANDOFF_LARGE_ROUTE_OPTIMIZATION_20260816.md` section 5. Its 12-epoch run peaked around epoch 6 and
then plateaued; its gap is not plausibly explained by stopping at epoch 12. Its multiplicative
`activity * carrier` token also destructively suppresses activity when the carrier branch is weak.

The revoked CTF-SD proposal was therefore best described as:

```text
Large-v1
+ concat instead of Hadamard fusion
+ Cell D-style sparsification
+ causal summaries
+ local convolution
+ S4D instead of GRU
+ different width and query slots
```

That is a system bundle, not a new isolated principle.

### 1.5 Cell W closes the previous temporal-residual line but does not test recurrence

Cell W was a non-causal `K=8` temporal residual on the unchanged SPINT output. It had no persistent
state. Its governing result was:

```text
Cell W - Arm A external                -0.0902
```

Cell W does not prove that a recurrent decoder cannot work. However, the decoder-directions contract
precommitted that an SSM refinement would be considered only if this residual paid. It did not pay.
Together with the negative bilinear+GRU and Large+GRU results, W does not license an S4D/Mamba/GRU
backbone sweep.

### 1.6 There is no S4D/S4/Mamba implementation to inherit

The current repository contains recurrent GRU implementations but no live S4D, S4, or Mamba model
implementation for this route. Freezing `S4D-256` as the first backbone would introduce an
unreviewed backend, new state semantics, and new numerical behavior at the same time as the science
change. S4D, trainable FIR timescales, four untied slots, RPNT-style 2D attention-score convolution,
and width 256 are removed from the first-cell design.

## 2. Why the original CTF-SD cell is forbidden

The table below counts every substantive simultaneous change from sealed Cell D. Rows marked
`changed` are independently capable of changing performance or interpretation.

| factor | sealed Cell D | revoked `CTF_SD_T4_D_SEED42` | status |
|---|---|---|---|
| identity source | B3S calibration activity fused with normalized T4 | normalized T4 only | **changed** |
| identity application | B3S emits a 50-bin identity added to query activity | T4 concatenated into a per-bin token | **changed** |
| activity information | full query activity plus fused identity | causal summaries/current count with no B3S identity | **changed** |
| temporal topology | whole 50-bin MLP read-in; no recurrent state | causal summaries + local causal convolution + recurrent SSM | **changed** |
| query slots | two coordinate-tied learned representatives | four untied global slots | **changed** |
| model width | 512 | 256 | **changed** |
| fusion law | additive activity-plus-identity | concatenation before projection | **changed** |
| sparsification law | D's placeholder-and-gain whole-unit dropout | true unit removal with `min_keep >= 1` | **changed** |
| local operator | none | left-causal convolution | **changed** |
| recurrent backend | none | S4D | **changed** |
| fixed causal summaries | none | fast/slow FIR or exponential summaries | **changed** |
| output topology | direct `[B,50,2]` trajectory | per-bin recurrent output | **changed** |
| training-loss semantics | MSE over all 50 emitted bins | previously unspecified for the new per-bin path | **unfrozen/changed** |
| recurrent reset semantics | not applicable | previously unspecified | **unfrozen/changed** |
| implementation ancestry | fused SPINT/D | Large/POSSM-like set-recurrent system | **changed** |

The count is fifteen changed or unfrozen scientific factors, not one. The cell is permanently
forbidden under its old name and contract. It may not be silently repaired and relaunched.

## 3. Split the two bets

### 3.1 Bet A: one temporal factor on the sealed D graph

Bet A asks one question:

> Can a strictly causal temporal operator improve cross-subject decoding when the complete fused
> B3S+T4 Cell D representation and its successful training treatment are otherwise held fixed?

Everything below is frozen to D:

- exact strict-27 source roster;
- exact M30 normalized T4 authority and label budget;
- B3S calibration-activity encoder and `post_pool(concat(mean, T4))` identity;
- `query_activity + identity` fusion in 50-bin window space;
- D's exact whole-unit dropout law, including placeholder/gain behavior;
- two coordinate-tied query slots;
- model width 512, one attention layer, two heads, and the current FFN;
- no teacher checkpoint, teacher tensor, unit table, or session table;
- no decoupled identity-keys/activity-values route;
- `behavior_scaling_factor = None`;
- seed 42;
- the 48-epoch warmup/cosine/final-four-SWA recipe, because this remains the D graph;
- last-bin, variance-weighted R2, equal session weight as governing scoring.

Only the temporal information-flow factor may change. Two conceptual realizations are admissible for
design review, but they are alternatives, not siblings and not a sweep:

1. **A-CausalMask:** impose a strict left-causal within-window information-flow rule while retaining
   the D modules and parameterization.
2. **A-RecurrentHead:** replace only the temporal readout semantics on the sealed D representation
   with a repository-native recurrent head, while preserving every encoder, fusion, slot, width,
   attention, dropout, and data factor above.

Neither realization is frozen here. A-CausalMask is the lower-confound option. A-RecurrentHead is
not admissible until a shape/interface audit proves that constructing a recurrent sequence does not
also change the read-in, slots, attention, loss, or identity path. If that proof cannot be written as
one changed row, A-RecurrentHead is forbidden.

### 3.2 Mandatory factor table for any proposed first Bet A cell

The next design review must fill this table with exact code-level semantics. The cell may advance
only if the final column contains exactly one `changed` row.

| factor | required reference | candidate | allowed status |
|---|---|---|---|
| source roster and support/query lineage | exact Cell D | TBD | held |
| T4 authority and normalizer | exact Cell D M30 | TBD | held |
| B3S calibration encoder | exact Cell D | TBD | held |
| identity application | `query activity + fused id` | TBD | held |
| whole-unit dropout law | exact D placeholder/gain law | TBD | held |
| query slots | two coordinate-tied slots | TBD | held |
| width / heads / layers / FFN | 512 / 2 / 1 / current FFN | TBD | held |
| temporal information-flow operator | Cell D whole-window mapping | TBD | **the only changed row** |
| training loss | predeclared all-bin or last-bin rule, matched to the chosen reference | TBD | held after freeze |
| output shape and scorer input | `[B,50,2]`, reuse `score_last_bin` | TBD | held |
| state-reset boundary | explicit window-local, trial, or session rule | TBD | part of the one temporal contract, not a second tuned factor |
| optimizer / 48-epoch schedule / SWA | exact D | TBD | held |
| seed | 42 | 42 | held |
| parameter order of magnitude | at most Cell D order | TBD | held constraint |
| unit/session tables | none | none | held |
| encoder backpropagation | current teacher-free D behavior; no new target encoder BP | TBD | held |

If a candidate requires a width change, new slots, a new dropout law, B3S removal, T4-only tokens,
new fixed summaries, a new convolution, and a new recurrent backend, it is a system swap and cannot
be called Bet A.

### 3.3 Bet B: Large-v1 plus one diagnostic treatment

Bet B is already negative as a performance family. The only clean remaining diagnostic is:

```text
Large-v1 + the exact D whole-unit dropout law
```

If retained at all, it must obey all of the following:

- compare only against sealed `large_t4`, not against D as an architecture-isolation claim;
- use Large-v1's exact 12-epoch recipe, because Large peaked early and plateaued;
- change only the dropout treatment;
- retain Large's multiplicative activity-carrier token, four slots, width, GRU, and all other code;
- wait for AM/IM, because importing D dropout may be mechanistically meaningless if D works through
  B3S fingerprint ablation;
- remain diagnostic and non-headline.

Even optimistic arithmetic does not make this a route to beat D:

```text
large_t4 external                         -0.0848
+ D's isolated gain over Arm A            +0.1576
= optimistic additive value                0.0728
Cell D external                             0.4179
remaining gap                               0.3451
```

The actual Large-to-D gap is about `0.5027`. A positive Large+dropout result would show that
whole-unit sparsification generalizes to another architecture; it would not establish a new decoder
factorization or a competitive system.

## 4. Causality and scoring contract

### 4.1 What is already causal and what is not

Last-bin scoring on a trailing 50-bin neural window is wall-clock causal for the query: the final
prediction uses only the current and previous 49 neural bins.

Cell D's training loss over all 50 outputs is not prefix-causal, because `fc_in` mixes the complete
50-bin window before every output is produced. Therefore a future Bet A implementation must not use
the word "causal" without freezing all three items below:

1. **Loss:** all-bin versus last-bin training loss.
2. **State reset:** window-local versus trial-local versus session-local state.
3. **Scoring path:** emit `[B,50,2]` and reuse the established `score_last_bin` implementation.

Changing loss while changing temporal topology is a second treatment unless the reference is
defined and matched accordingly. The factor table must expose this rather than hiding it inside the
new module.

### 4.2 Prefix test

For any model claiming prefix causality:

```text
Take two inputs with identical prefixes and arbitrary different future suffixes.
For every prefix timestep t:
  yhat_a[t] == yhat_b[t]
  state_a[t] == state_b[t]
```

Use bitwise equality where the backend permits it; otherwise freeze one numerical tolerance before
real data. Testing only the final index is insufficient. Inspecting a mask is not a substitute for
the output-and-state test.

### 4.3 Carrier normalization

Synthetic gates must consume carriers standardized exactly as the datamodule standardizes the live
T4 tensors. Raw carrier fixtures are invalid for this purpose; a previous PV route was voided for
this exact mismatch. The synthetic receipt must bind the normalizer bytes and prove equality to the
live preprocessing semantics.

## 5. Gate honesty and baseline uncertainty

Cell D currently has only seed 42. The matched A2 external seeds span approximately `0.066`:

```text
A2 seed 42 external                       0.3178
A2 seed 43 external                       0.3367
A2 seed 44 external                       0.3837
```

Therefore a single seed-42 result `+0.03` above D cannot be called a breakthrough or superiority
result. It is only a screening signal. D seeds 43 and 44 remain a blocker for any superiority claim.

For a future one-factor Bet A screen, use this interpretation:

| seed-42 paired outcome versus D | interpretation |
|---|---|
| external `>= +0.03`, within `>= -0.03`, broad session support | promising candidate; freeze it, then replicate both baseline and candidate before any superiority claim |
| external from `0` to `+0.03` | expected/no-decision band; not failure, not breakthrough, and not a license for a silent rebuild |
| external `< 0` | negative one-factor result; stop that exact factor unless a named, pre-registered follow-up already exists |
| within `< -0.03` | safety failure; stop even if an external mean is positive |

The within gate is consequential. D within is `0.5697`; Large-v1 within was `0.2562`. A T4-only
causal system would likely fail this gate for the same reason Large did. That is further reason not
to use a T4-only system swap as the first cell.

Paired per-session values, sign counts, confidence intervals, the 2014/2015 date blocks, and the
separate `20141203` session must be reported. A mean alone is not an adequate decision statistic.

## 6. Contribution sentence

The old sentence is revoked:

> "A closed-form T4 encoder instantiates a compact causal set decoder without unit IDs."

Large-v1 already implemented the substantive content of that sentence and lost to fused SPINT.

If, and only if, a one-factor Bet A cell eventually passes and replicates, the contribution sentence
must match the actual factor:

> **Within an otherwise frozen fused B3S+T4 whole-unit-sparsified decoder, a predeclared causal
> temporal information-flow operator improves cross-subject neural decoding.**

This is narrower than a new task-frame factorization, but it is honest and attributable. If the
effect is only engineering parity, the claim must be reduced accordingly.

RPNT and POSSM remain inspirations for asking about local context and state. They are not evidence
that S4D, static slots, or T4-only unit embeddings will improve this system.

## 7. Stage-0 requirements before any future implementation or launch

This document does not authorize Stage 0. A later implementation handoff must first establish:

1. the completed one-factor table with exactly one changed row;
2. exact D initial-state and module parity for every held factor;
3. exact D B3S+T4 identity and application parity;
4. exact D dropout draw/application/value-law parity;
5. exact 512/2-head/2-slot attention parity;
6. exact source schedule, optimizer, 48-epoch schedule, and SWA parity;
7. explicit loss, output, and state-reset semantics;
8. every-timestep prefix prediction/state tests;
9. unit permutation invariance;
10. zero learned unit/session tables;
11. no teacher parameter or checkpoint;
12. z-scored synthetic T4 parity with the datamodule;
13. active parameter, persistent-state, MAC, and latency accounting;
14. no target/formal data access and zero target optimizer/backward calls.

Any failure that is repaired by adding another architecture factor invalidates the one-factor cell.
The repair must return to design review rather than silently creating a new route.

## 8. Scheduling and stop rules

1. Let the already launched AM/IM cells finish. Do not duplicate, restart, or reinterpret them from
   this document.
2. Keep the D mechanism sentence unresolved until AM/IM terminate, unless the operator explicitly
   freezes that uncertainty.
3. Do not start `CTF_SD_T4_D_SEED42`; it is revoked.
4. Do not implement or launch S4D, FIR-timescale, four-slot, width-256, or RPNT-attention-convolution
   successors from this handoff.
5. Do not open a temporal-backbone sweep.
6. Before a performance superiority claim, obtain the missing D seed-43/44 evidence and replicate
   any winning candidate under the same estimator.
7. If Bet B is retained, treat it only as a 12-epoch Large-v1 diagnostic and wait for AM/IM.

The higher-level strategy remains performance-first: minimal ablations, no SPINT-neighborhood sweep,
and no prolonged defense of a complex negative bundle.

## 9. Later hypothesis, not a sweep menu

If Bet A is cleanly negative, the only substantially different factorization still worth a design
discussion is the previously proposed post-bilinear Route C:

```text
carrier defines the task-frame observation
learned dynamics predicts the next state
the model filters the observation innovation
```

This has a distinct isolating prediction that Large-v1 did not test. It is not "static slots plus a
different RNN letter." It remains a hypothesis only. This document does not authorize its code,
data access, or execution.

## 10. Fixed evidence

Sealed Cell D:

```text
tfpd_exploration/results/pop_robust_v1/cellD_2heads_dynamic_dropout/swa_final4.pt
SHA256 626f65d80fd9f4305605132175c7ea43bc0c40d6ef6203ef1830b4b2e77f33bd

tfpd_exploration/results/subpop_score_v1_r2/subpop_score_receipt.json
SHA256 b4624fa4daf8a76f0e94447b2ba84412ec964a3c34b0a41600bc0789e9c8c629
```

Matched A2 reference:

```text
tfpd_exploration/results/a2_matched_rescore_v1_r1/a2_rescore_receipt.json
SHA256 0ae0f74c6b5d606599d1108378836c8cbb7d1e5df05a453aa83f682255734d4e
```

Architecture and result records:

```text
tfpd_exploration/src/tfpd_large.py
tfpd_exploration/docs/HANDOFF_LARGE_ROUTE_OPTIMIZATION_20260816.md
tfpd_exploration/docs/HANDOFF_POST_BILINEAR_CONSUMER_EXTENSION_20260815.md
tfpd_exploration/docs/HANDOFF_SPINT_DECODER_DIRECTIONS_20260817.md
tfpd_exploration/docs/HANDOFF_SUBPOPULATION_INVARIANCE_DECOMPOSITION_20260818.md
tfpd_exploration/docs/HANDOFF_ACTIVITY_VS_IDENTITY_MASK_20260819.md
```

Reference papers:

```text
sua_exploration/papers/Fang_et_al_2026_RPNT_Robust_Pretrained_Neural_Transformer.pdf
SHA256 fc681ab2f459d812152194706e762006f773e88bfcb4df6bd12b01c720d1cc94

sua_exploration/papers/Ryoo_et_al_2025_POSSM_Generalizable_Real_Time_Neural_Decoding.pdf
SHA256 5edea2f830d3148fe88e7b920066321005381f9fd2796fc36ba585c11d7dfc92
```

## 11. Final handoff instruction

Do not build or launch the revoked CTF-SD bundle. First wait for AM/IM or explicitly preserve the
mechanism uncertainty. Then return a **design-only** Bet A proposal with a completed simultaneous-
change table. If more than one scientific factor differs from Cell D, reject the proposal before
code is written. Any eventual first architecture experiment must change one temporal factor on the
sealed D graph and nothing else.
