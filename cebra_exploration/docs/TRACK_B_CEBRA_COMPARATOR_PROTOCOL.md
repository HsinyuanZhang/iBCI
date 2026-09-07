# Frozen Protocol: CEBRA Adaptation Comparator (Track B)

> **SUPERSEDED FOR CURRENT EXECUTION — 2026-08-14.** This historical Track-B
> protocol is retained only as design lineage. The executable non-H1 route is
> governed by `TRACK_B_V2_H1_EXCLUDED_PROTOCOL.md`, which excludes H1/M2,
> freezes `offset10-model` with its exact alignment/cropping authority, selects
> geometry on source-only pseudo-target folds, and uses versioned immutable
> metric/source authorities. In particular, the `offset1-model` convention
> below must not be used by any current selector, control, or target scorer.

> ## BLOCKED LIFTED — 2026-08-13, Round 2 positive-control gate
>
> **Evidence that lifted the banner.** Required test
> `test_positive_control_gate_every_arm_including_negative_control` PASSED
> (pytest: 22 passed in 9.22s, CPU, `PYTHONNOUSERSITE=1`, seeded). Construction:
> three sessions of 24/31/37 units, different linear mixes of one shared 2-D
> `(cos t, sin t)` latent, source-fitted ridge, threshold target R² ≥ 0.70;
> negative control must stay below 0.20.
>
> | Arm | status | source R² | **target R²** |
> |---|---|---:|---:|
> | `cebra_joint_behavior` (primary) | CEBRA_DEFINABLE | 0.7528 | **0.7502** |
> | `cebra_frozen_source_adapt` | CEBRA_DEFINABLE | 0.7392 | **0.7205** |
> | `cebra_adapt_unaligned` (negative control) | CEBRA_DEFINABLE_NEGATIVE_CONTROL | 0.7415 | **−0.4670** |
> | `cebra_no_adapt` | CEBRA_UNDEFINED_UNIT_COUNT_MISMATCH | — | — |
> | `cebra_joint_time` | CEBRA_UNDEFINED_MULTISESSION_REQUIRES_AUXILIARY | — | — |
> | `cebra_joint_time_query_unlabelled` | CEBRA_UNDEFINED_MULTISESSION_REQUIRES_AUXILIARY | — | — |
>
> The gate has teeth: the unaligned arm is negative, as F8 predicted. Joint and
> frozen recover the shared latent. Official `adapt=True` without cross-session
> sampling remains structurally incapable of producing a valid number — that path
> is kept only as `cebra_adapt_unaligned` and is **never** CEBRA's best.
>
> Original coordinator diagnosis (kept for the record): on the same construction,
> multi-session source → collapse → `adapt=True` scored 0.9891 source / **−1.2278** target *(single
> seed and an extreme draw — an independent 8-seed mean is ≈ +0.01; the direction reproduces across
> 13 configurations and two runs, the magnitude does not)*
> target; joint including the target scored **+0.9940**. Cause: F2b, zero shared
> weights, alignment only from the joint contrastive loader.
>
> `collapse_multisession_to_template` was **deleted** (F9), not fixed.
>
> **Scoring on real NWBs is still refused.** Lifting BLOCKED means the arms are no
> longer structurally void. It does not authorise a scoring run.

**Date predeclared:** 2026-08-13, before any scoring arm was executed
**Status:** CPU-only. Round 2 skeleton: joint primary arm, frozen-source adapt, declared negative control. Positive-control gate passed. Scoring still refused.
**Coordinator audit status:** F8 structural void **cleared** by the table above. F2b correction remains in force.
**Scope:** Primary — subject-M (DANDI 000688, SUA and pseudo-MUA) and RT (`dandi_000688/sub-C/sub-C_ses-RT-*.nwb`, **not** `data/000129/sub-Indy`). Bindings for FALCON H1 and M2 are in the skeleton because CEBRA is **not** in FALCON Table 1; those datasets are not scored in this session.


---

## 1. Why this comparator exists

Two independent reasons, both in force for the rest of this document.

**Reason 1 — closest learned analogue of our identity token.** Our method solves "the unit set changes across sessions" in closed form. CEBRA solves the same problem by learning, and it genuinely trains across sessions with different neuron counts (verified in this environment: 20 vs 31 units, shared 3-D embedding). No other comparator we have occupies that position.

*Corrected by coordinator audit (see `COORDINATOR_VERIFIED_FINDINGS.md` F2b): multi-session CEBRA does **not** learn session-specific input maps over a shared trunk. It builds an `nn.ModuleList` of **completely independent full encoders**, one per session, sharing **zero** weights — measured, not one tensor is identical between two session encoders after a joint fit. Alignment comes entirely from the contrastive loss, and parameter cost grows as `O(N_sessions × full encoder)`. It is also the direct cause of the F8 failure. **Scope correction (red team):** this holds for `MultiSessionSolver`, the only solver the sklearn estimator reaches, **not for CEBRA as a whole** — `cebra.solver.UnifiedSolver` shares one model across sessions (13,155 parameters against 22,790 on the same two sessions). Do not state "parameter growth is linear in session count" as a property of CEBRA; the defensible axis is that `UnifiedSolver`'s input width is hard-wired to the sum of the training sessions' unit counts and so **cannot serve an unseen session at all**. See `COORDINATOR_VERIFIED_FINDINGS.md`, F2b addendum.*

**Reason 2 — the paper's missing cost-of-no-backprop baseline.** Item 5 of `sua_exploration/docs/HANDOFF_COMPARATORS_20260812.md` §7: the premise is that a backward pass is unaffordable, and the paper never says what that costs. Adapting CEBRA to a new session **requires training** (joint: every session encoder; frozen-source: the target encoder only). Measuring what that pass buys (accuracy) and what it costs (parameters updated, wall-clock, iterations, FLOPs) is a result either way. If CEBRA beats the carrier, we quantify the price of our constraint. If it does not, the constraint is nearly free. Both outcomes are publishable. The cost table is first-class output, not an appendix.

---

## 2. What CEBRA is, for the purpose of this comparator

CEBRA (Schneider, Lee & Mathis, *Nature* 2023) learns an embedding `f(x) ∈ R^d` with InfoNCE. The positive-pair distribution is defined by an auxiliary variable:

| Mode | Positives | Target labels consumed |
|---|---|---|
| CEBRA-Behavior | similar behaviour | yes, as a sampling distribution, never as a regression target |
| CEBRA-Time | temporal neighbours | none |
| CEBRA-Hybrid | both | yes |

Decoding is a separate downstream step. This comparator does not treat the embedding as a prediction.

**Official sklearn `adapt=True` (CEBRA 0.6.1).** After a *single-session* fit, `fit(X_new, adapt=True)` reinitialises the first layer to the new feature dimension, freezes every later layer, and retrains the first layer for `max_adapt_iterations` (default 500). **This is not implemented for multi-session estimators.** F8 showed it cannot land a new session in a source multi-session latent: alignment exists only while sessions share a contrastive loader.

**Adaptation protocol used here (Round 2).** How CEBRA is actually used on multi-session data: the target session **participates in the joint multi-session fit**, so cross-session positives tie it to the source. A second arm freezes the source encoders and trains only the target encoder **while keeping that loader**. Official `adapt=True` is retained only as a declared negative control (`cebra_adapt_unaligned`). The Round-1 collapse helper is **deleted** (F9): joint arms do not need it, and it left `solver_` inconsistent so `transform` could not produce a source embedding.

Vendored CEBRA was modified for the frozen-source arm only (`freeze_sessions`, `init_from`). Recorded in `third_party/CEBRA_PROVENANCE.txt` under `local_modifications`.

---

## 3. Part A — constructibility audit, run FIRST

For each dataset and view, measure and report per session **without fitting CEBRA and without scoring**:

1. **Calibration sample count vs CEBRA's batch/iteration contract.** Record `T` on the matched calibration prefix (subject-M M50, also M15/M30 if the adapter already has them; RT M24; H1 M4; M2 M24). Compare to CEBRA's recommended `batch_size ≥ 512` and to `offset1-model`'s receptive field of 1. If `T < 512`, the resolved batch size is `T` (full-batch InfoNCE), never a longer prefix. If `T` has not been measured on real NWBs, emit `CEBRA_AUDIT_UNMEASURED_CALIBRATION_LENGTH` — that is not a TOO_SHORT finding. If `T` is measured and `T < MIN_ADAPT_SAMPLES` (16), emit `CEBRA_UNDEFINED_CALIBRATION_TOO_SHORT`.
2. **Input-layer identifiability on the prefix alone.** First-layer parameter count for `offset1-model` is `N · H + H` with `H = 32`. Report `T / (N H + H)`. A ratio below 1 is **not** automatically undefined — CEBRA is iterative InfoNCE, not a closed-form solve — but it must be printed next to any later number as `underdetermined_input_layer`. If the prefix cannot even run one InfoNCE step, that is `CEBRA_UNDEFINED_CALIBRATION_TOO_SHORT`.
3. **Latent dimensionality `d`.** CEBRA's `output_dimension` is a training hyperparameter, not a variance-explained quantity. A single `d` across sessions is required because it is the shared embedding size. Predeclared grid `{3, 8, 16}`, default **8** (CEBRA's own default). Selected on source sessions only. Part A reports that a shared `d` is always *constructible* and is **not** the FA-style "is one `d80` defensible" question.
4. **Auxiliary behaviour variable.** CEBRA-Behavior needs a non-degenerate continuous or discrete label aligned to neural time. Per dataset:
   - subject-M: 2-D cursor velocity is available; trial direction is also available. Primary auxiliary is **velocity**, matching the decode target.
   - RT: discovered through the sealed loader on `sua_exploration/data/dandi_000688/sub-C/sub-C_ses-RT-*.nwb` (`find_rt_sessions` + `rt_classical_comparators.session_name_from_nwb_path`; `EXPECTED_FOLDS = 15`). Paths containing `000129` or `sub-Indy` are refused. The recorded trial-table direction field is **degenerate** (one unique value). Discrete-direction CEBRA-Behavior is `CEBRA_UNDEFINED_DEGENERATE_DIRECTION`. Continuous velocity remains the **only** legal auxiliary.
   - FALCON M2: 2-D velocity.
   - FALCON H1: 7-DoF velocity. No native discrete center-out direction field.
5. **Unit-count match for `cebra_no_adapt`.** Applying a source encoder to a target session without a new encoder requires `N_target == N_source_encoder`. If counts differ, emit `CEBRA_UNDEFINED_UNIT_COUNT_MISMATCH` and do not manufacture a mapping.

A method being structurally inapplicable is a **finding**, not a failure. Do not invent a unit map, a direction field, or extra calibration data to make an arm runnable.

---

## 4. Part B — the arms, only where Part A says definable

### Design, following FALCON's own convention

FALCON evaluates alignment as `NoMAD + WF` / `CycleGAN + WF`: an embedding or alignment front-end plus a **linear readout**. Mirror that, so this comparator sits in the same class as the published numbers:

```
source:  fit multi-session CEBRA on source sessions          -> shared embedding dim d
source:  fit a linear ridge readout on source embeddings using SOURCE labels
target:  (joint arms) the calib prefix participates in the multi-session fit
         (frozen-source) source encoders frozen; only the target encoder trains,
                         still inside the joint contrastive loader
         (unaligned)     official adapt=True, no cross-session sampling — NEGATIVE CONTROL
         (no_adapt)      refuse if N differs; else apply a source encoder as-is
deploy:  y_hat = readout( embedding_target(x) )
```

CEBRA's own papers decode with kNN. kNN is implemented as a **secondary check** that the embedding is informative, not as the primary number. Justification is in §5.

### Arms

**Primary arm: `cebra_joint_behavior`.** This is how CEBRA is used on multi-session data. Never present a handicapped arm as CEBRA's best (`HANDOFF_COMPARATORS_20260812.md` §10; reporting rule 5).

| Arm | What it does | Target labels | Query neural | Bias direction |
|---|---|---|---|---|
| **`cebra_joint_behavior` (primary)** | Target calib prefix joins the multi-session Behavior fit. Cross-session positives align it. | yes, dense velocity on the **matched calib prefix only** | **no** | Favours CEBRA on accuracy (source encoders may move; transductive on the prefix). Favours us honestly on cost (retrain the entire multi-session model). |
| `cebra_joint_time` | **Not built.** sklearn multi-session CEBRA raises `RuntimeError` without a shared auxiliary (`cebra.py:670-673`). Time is not a cross-session coordinate. Manufacturing `np.arange` would assume clocks correspond and would cheat this protocol's time-parameterised positive control. | n/a | n/a | `CEBRA_UNDEFINED_MULTISESSION_REQUIRES_AUXILIARY`. A finding, not a missing implementation. |
| `cebra_frozen_source_adapt` | Freeze source encoders, train **only** the target encoder, **keep cross-session sampling**. Vendored `freeze_sessions` + `init_from`. Closest analogue of us. | yes, calib prefix only | **no** | Slightly against CEBRA vs joint (source cannot move). Still a target backward pass, unlike a closed-form solve. |
| `cebra_adapt_unaligned` | Official `adapt=True` after copying source session 0 into a single-session estimator. **Declared negative control.** | yes, calib prefix | **no** | Structurally unaligned (F8). **Never CEBRA's best.** Must fail the positive-control gate. |
| `cebra_no_adapt` | Apply a source encoder as-is. | none | no | Zero-shot floor. Undefined when `N` differs. |
| `cebra_joint_time_query_unlabelled` | **Not built.** Unlabelled query activity cannot join a sklearn multi-session fit without an auxiliary on those samples. Assigning query behavior is a leak; assigning time indices invents correspondence. **Decision: query activity does not participate.** | n/a | declared out | Same structural block as joint Time. |

**No CEBRA-Hybrid.** It would sit between the two joint arms without a FALCON analogue.

**Query-label leak surface (new in Round 2).** A joint fit puts the target inside training. Enforce:

- Only the **matched calibration prefix** of the target may carry labels (subject-M M50, RT M24, the same budget our carrier gets).
- **No target query label may enter the fit, ever.** The estimator API raises `LabelLeakError` if `target_query_labels is not None`.
- **Decision on unlabelled query activity:** it does **not** participate. sklearn multi-session CEBRA requires an auxiliary on every sample. Inventing one is either a leak or a fake correspondence. The named variant `cebra_joint_time_query_unlabelled` exists so this choice is testable and cannot be folded silently into a primary arm.

### Matched budget

Every arm consumes the **same calibration prefix** the carrier uses on that dataset:

| Dataset | Budget | Also report if cheap |
|---|---|---|
| subject-M | M50 | M15, M30 |
| RT | M24 | — |
| FALCON M2 | M24 | — |
| FALCON H1 | M4 (4 trials) | — |

Never give CEBRA more calibration data than the carrier. Never starve it by shrinking the prefix below the carrier's. If CEBRA cannot train on that budget, that is a Part A verdict, not a longer prefix.

**Label-density mismatch, stated in advance.** CEBRA-Behavior uses **dense bin-level velocity** on that prefix as its auxiliary. The carrier uses **sparse trial-level** labels. This arm is therefore not information-matched on supervision density. Do not sparsify CEBRA's auxiliary to "help" us; report density next to the score.

### Fixed conventions

- Architecture: `offset1-model` (receptive field 1, CEBRA sklearn default). `offset10-model` would demand a longer temporal context than a short calibration prefix can guarantee.
- `output_dimension = 8`, `num_hidden_units = 32`, `learning_rate = 3e-4`, `max_iterations = 10000` (joint / source), `max_adapt_iterations = 500` (frozen-source and unaligned), `batch_size = min(512, T)` with `T` the prefix length.
- Device: **CPU**. CUDA may exist in the process; this comparator must not launch GPU work. `device="cpu"`. The runner sets `CUDA_VISIBLE_DEVICES=""`.
- Hyperparameters selected on source sessions only. No target query window may influence architecture, `d`, batch size, or iteration count. Query activity, when used, is only the declared Time variant.
- Per-session target adaptation; no transfer of a target-fitted encoder across sessions.
- Same query boundaries and sealed query identities as existing comparator arms. Subject-M binds the V9 query; RT binds the Stage-2 triple; H1 binds window manifest `665fe535…e4da`; M2 binds `native_m2_m24_ridge_w50.EXPECTED_HELDOUT_LAYOUT`.
- Same metric as every other arm on that dataset: variance-weighted / pooled R² (they are the same formula: `1 - SSE / TSS` with per-dimension centering). H1 is reported under the name `pooled_r2` to match the sealed ridge receipt.

### Sealed references, not re-run in this session

subject-M carrier 0.3568 / 0.3061, dense ridge 0.4179 / 0.4102; RT T4d 0.448176, ridge 0.200202, Zero4 0.179272; H1 ridge 0.25823473332303337, carrier 0.5000; M2 ridge 0.1139, carrier 0.2268.

---

## 5. Downstream decoder

**Primary: linear ridge readout**, fitted on **source** embeddings and **source** labels, applied to target embeddings. House solver: `sua_exploration/mc_maze/subm_v9_f0_pv_ridge.fit_ridge` / `predict_ridge`, `normalized_lambda=1`, unpenalised intercept, CPU.

**Why primary is linear, not kNN.**

1. FALCON Table 1 reports `NoMAD + WF` and `CycleGAN + WF`. A linear readout puts this comparator in the same class as those citable numbers.
2. Every internal comparator (ridge, FA alignment, Kalman) reports the same linear-readout metric. Changing the head would make CEBRA incomparable to our own table.
3. The cost axis is about the *embedding adaptation*, not about a nonparametric decoder that can hide a weak latent space.

**Secondary: kNN** (`sklearn.neighbors.KNeighborsRegressor`, `n_neighbors=3`, cosine metric — CEBRA's published default). Fitted on source embeddings. Reported in the receipt, never as the headline number. If linear is near chance and kNN is not, the embedding is nonlinearly informative and the paper must say so rather than claiming CEBRA failed.

**Where the readout is fitted.** Source only. Fitting a linear head on the target prefix would give CEBRA a second target-session estimator (the ridge) on top of the adapted encoder, which is a different method family. That is not an arm.

**Source-side fit quality is mandatory.** Every arm records source-readout training R². A target score may not be interpreted unless the source fit is sound. This is the diagnostic whose absence voided the FA alignment scoring run of 2026-08-13.

---

## 6. The cost axis (first-class)

Cost is **measured per arm**, not assumed. Joint fitting retrains the entire multi-session model. Frozen-source should be cheaper. Unaligned first-layer adapt is cheaper still and is not CEBRA's best. Carrier reference: closed-form OLS plus one forward pass, zero target gradients.

For each arm, each target session:

| Field | What |
|---|---|
| `target_parameters_updated` | count of tensors with `requires_grad=True` during the target-side update (`0` for `cebra_no_adapt`) |
| `target_parameter_count` | number of encoder scalars updated (all session encoders for joint; target encoder only for frozen-source; first layer for unaligned) |
| `target_adapt_wall_clock_s` | `time.monotonic()` around the target-side fit; labelled **CPU** |
| `target_adapt_iterations` | iterations actually run (joint: `max_iterations`; frozen-source / unaligned: `max_adapt_iterations`; `0` for no_adapt) |
| `target_adapt_flops_estimate` | documented arithmetic estimate for `offset1-model`, not a profiler measurement |

Estimate, per iteration, batch `B`, hidden `H=32`, output `d`:

```
forward(N) = 2 B N H + 2 B H H + 2 B H (H/2) + 2 B (H/2) d

joint:           iterations * sum_sessions  3 * forward(N_s)     # every encoder trainable
frozen-source:   iterations * (sum_frozen forward(N_s) + 3 * forward(N_target))
unaligned:       iterations * (2 * forward(N_target) + 2 B N_target H)
no_adapt:        0
carrier:         closed-form OLS + one forward pass
```

Frozen encoders still **forward** because they participate in cross-session InfoNCE; they do not backprop. The `3×` on trainable encoders is activation forward + backward + parameter gradient at encoder scale. This is an order-of-magnitude figure, labelled `estimate`. If the measured table comes out strongly in our favour, that is a real result — the number must be the measured one, not a projection.

**Measured on the Round 2 positive-control gate** (CPU, `offset1-model`, `d=3`, 250 iterations, three sessions 24/31/37 × 240 samples). Adaptation cost only (frozen-source wall-clock is the target-encoder joint fit, not the source pretrain). Labelled **estimate / CPU**.

| Arm | params updated | wall-clock (CPU s) | iterations | FLOPs estimate |
|---|---:|---:|---:|---:|
| `cebra_joint_behavior` | 7,945 | 3.926 | 250 | 2.771e9 |
| `cebra_frozen_source_adapt` | 2,851 | 1.252 | 250 | 1.588e9 |
| `cebra_adapt_unaligned` (negative control) | 1,216 | 0.455 | 250 | 8.064e8 |
| `cebra_no_adapt` / Time arms | 0 | 0 | 0 | 0 |
| our carrier (reference, not re-run) | 0 target gradients | closed-form OLS + 1 forward | 0 | not a CEBRA training loop |

Frozen-source updates ~36% as many encoder scalars as joint and ~32% of the wall-clock on this toy. That ratio will change on real 15-session LOSO (joint trains 15 encoders; frozen trains 1). Real-data cost is unrun.

---

## 7. Integrity gates (both required before any arm is interpreted)

1. **Sealed-reference integrity.** Reproduce that dataset's sealed reference and report the maximum deviation. Standards already met elsewhere: H1 ridge to `0.0`, subject-M ridge to `2.24e-6`. If the gate fails, **stop**. This session **wires the hook and does not execute it** (`HOOK_WIRED_NOT_EXECUTED`). Expected values are the sealed references in §4.
2. **Positive-control gate.** Three sessions, unit counts `(24, 31, 37)`, each a different random linear mix of one shared 2-D latent (`cos t`, `sin t`) plus light noise. Fit a ridge readout on **source embeddings only**. Require target R² ≥ `0.70` for every arm that claims alignment. **`cebra_adapt_unaligned` must fall below `0.20`**, proving the gate has teeth. **`cebra_no_adapt` is undefined** on this construction (mismatched N). An arm that cannot recover the latent is void and must fail loudly. Runner: `--positive-control`. Tests: `test_positive_control_gate_every_arm_including_negative_control`.

Scoring refuses to treat a target R² as interpretable unless both gates have passed.

---

## 8. Design decisions a reviewer could dispute

1. **Primary is joint fit including the target, not official `adapt=True`.** F8 showed `adapt=True` cannot align. Joint is how CEBRA is used. Cost consequence favours us honestly. Bias on accuracy favours CEBRA (source encoders move; prefix is transductive). Named in the arm table.
2. **Frozen-source requires a vendored patch.** There is no shared trunk to freeze (F2b). Freezing source *encoders* while keeping the joint loader is the minimum-cost path that still works and the closest analogue of us. Default `fit()` behaviour is unchanged when the new kwargs are omitted.
3. **`cebra_adapt_unaligned` is kept as a negative control**, not deleted. It is the mechanistic demonstration that cross-session sampling is necessary. It is never CEBRA's best.
4. **Collapse was deleted, not fixed (F9).** Joint arms do not need it. Unaligned copies source session 0 into a freshly fitted single-session estimator, then calls official `adapt=True`. Source embeddings for the readout still come from the original multi-session estimator with `session_id`.
5. **H1 and M2 are bound, not primary.** FALCON Table 1 has no CEBRA row. H1 file discovery goes **only** through `h1_sparse_event_endpoint.index_heldin_calib`. M2 file discovery is an allowlist against `native_m2_m24_ridge_w50.EXPECTED_HELDOUT_SESSIONS` under `sub-MonkeyN-held-out-calib/` — `held-out-calib` is in scope for M2 and out of scope for H1.
6. **Primary decoder is linear, not kNN.** See §5.
7. **CEBRA-Behavior uses dense velocity, not sparse direction.** Matched *prefix*, not matched *label density*. RT discrete direction is undefined.
8. **`offset1-model`, `d=8`, `H=32`.** CEBRA defaults, not a sweep.
9. **Batch size drops to `T` when `T < 512`.** We do not pad or borrow extra time.
10. **Query activity does not participate.** sklearn multi-session requires an auxiliary per sample. The named variant exists so the choice cannot be silent. Joint Time is undefined for the same reason.
11. **Scope rule is per dataset.** Subject-M V9 lives under a path containing `formal`. That token is a false positive for DANDI 000688. Scoring may bind it. H1 still fails closed on `held-out`.

---

## 9. What each outcome licenses — fixed in advance

Primary contrast per dataset and view: `carrier − cebra_joint_behavior`, with mean, median, per-session sign count, and a paired bootstrap interval. Also report `cebra_joint_behavior − cebra_no_adapt` (what any target-session update buys), `cebra_joint_behavior − cebra_frozen_source_adapt` (what moving the source buys), and `cebra_joint_behavior − cebra_joint_time` (what target labels buy inside CEBRA). Do **not** contrast against `cebra_adapt_unaligned` as if it were CEBRA's best.

- **Carrier exceeds the primary joint-behavior arm.** The paper may state that on these datasets a closed-form identity token outperforms multi-session CEBRA-Behavior under a matched calibration prefix, at far lower target-session cost. Remains a system-level statement: architectures differ.
- **Joint-behavior exceeds the carrier.** Report it plainly. The cost table is then the contribution: this is what retraining the multi-session model buys. Do not hide the accuracy loss.
- **`cebra_joint_time` is `CEBRA_UNDEFINED_MULTISESSION_REQUIRES_AUXILIARY`.** sklearn multi-session CEBRA is not CEBRA-Time. Do not manufacture a time index. That is a finding about the published API, not a missing number.
- **Frozen-source is close to joint at much lower cost.** That is the fairest analogue of us and a result either way.
- **`cebra_no_adapt` is `CEBRA_UNDEFINED_UNIT_COUNT_MISMATCH`.** That is the mechanism argument: a permutation-invariant identity path is defined precisely where a frozen encoder is not. Do not manufacture a mapping.
- **Arm `CEBRA_UNDEFINED_CALIBRATION_TOO_SHORT`.** CEBRA cannot train on the carrier's budget. That is a finding in the carrier's favour on the calibration-data axis, not a missing number.
- **Source-side readout R² is near chance, or the positive-control gate fails.** The arm is broken, not a baseline. Same class of void as the FA alignment scoring run of 2026-08-13.

Do not tune CEBRA where tuning helps us and leave it fixed elsewhere. Never present `cebra_adapt_unaligned` as CEBRA's best.

---

## 10. Interpretation limits

- sklearn multi-session CEBRA is per-session full networks plus a joint contrastive loss, not a single trunk with session-specific stems (F2b). Alignment is the loss, not inherited weights.
- This is offline, pairwise (leave-one-session-out source pool → one target). Nothing here tests long inter-session gaps or chaining.
- A joint fit is transductive on the target **calibration prefix**. Unlabelled query activity does not enter. Query labels never enter.
- Accuracy, supervision density, and target-session training cost must be reported side by side and never collapsed.
- CPU-only runs are slower than the GPU recipe in `cebra_exploration/docs/BACKGROUND_BRIEF.md` §6. Wall-clock is a CPU figure and must be labelled as such.

---

## 11. Build status as of Round 2

Implementation: `cebra_exploration/src/cebra_comparator.py`, runner `cebra_exploration/scripts/run_cebra_comparator.py`, tests `cebra_exploration/tests/test_cebra_comparator.py`. Vendored patch: `cebra/integrations/sklearn/cebra.py` (`freeze_sessions`, `init_from`), provenance in `third_party/CEBRA_PROVENANCE.txt`.

**No scoring arm is executed in this session.** Part A on real NWBs: RT discovery is implemented through the sealed loader and is exercised when the directory exists; subject-M / H1 / M2 discovery adapters exist; calibration **sample counts** on real prefixes remain unmeasured (`CEBRA_AUDIT_UNMEASURED_CALIBRATION_LENGTH`). The positive-control gate **passed** (see the lift banner). The sealed-reference integrity hook remains `HOOK_WIRED_NOT_EXECUTED`.
