# Cross-dataset functional calibration: explanation and improvement work order

Date: 2026-09-05
Status: **STAGE0_WORKORDER_READY__STAGE1_REQUIRES_BOUND_IMPLEMENTATION_SPEC__NO_JOBS_LAUNCHED**
Scope: a successor research plan, not another FiLM rescue or a rewrite of sealed experiments.
Author/reviewer: Astra. Execution belongs to a separate agent/team.
Hardware envelope: two independent RTX3090 24GB GPUs, 64GB host RAM; existing jobs have priority.

## 0. Decision and scope

Unify **the role of calibration**, not a literal four-dimensional cosine formula:

> Activity signature summarizes observed neural activity. Tuning profile supplies task-coordinate information estimated from paired calibration observations. A source-trained consumer combines them for subsequent decoding without target backpropagation.

Two parallel work streams:

- **E / explanation:** audit actual estimator semantics, population dependence, temporal relationships, and evidence boundaries. CPU/source-development only.
- **P / improvement:** prepare one M1 calibration-aware tuning-profile pilot, then a bounded matched performance comparison after its implementation specification is frozen. Keep the strong activity+decoder path and full output space.

Do not confuse these with the active M2 A-QMEM / B-decoder tracks. Their immediate decision is in [the separate shuffled12 review](REVIEW_M2_DUAL_TRACK_SHUFFLED_12EP_AND_24EP_DECISION_20260905.md). Finish that bounded comparison first; E and P preparation can overlap CPU work without taking its GPUs.

This turn delivers documents only. On handoff, Stage0 is sufficiently specified for local implementation and CPU diagnostics. Stage1 has an explicit technical freeze gate below; this is not a repeated sandbox-permission gate. The executing team must not choose undocumented estimator/initialization/selection rules on the fly or launch all proposed successors. No result roots or jobs were created by the author.

## 1. Non-negotiable evidence corrections

| Dataset | Current carrier family | Interpretation and caveat |
|---|---|---|
| M2 | Direction-conditioned affine encoding, `[a,c,sqrt(a²+c²),b]` | It has an intercept. A reach-conditioned intercept is not an independent hold response. Bind the selected MOVE-window estimator rather than replacing it with historical whole-trial T4. |
| DANDI688 | Canonical-direction response fit, `[a,c,m,b]` | Same descriptor family does not imply the same estimator as native M2. Trial/direction weighting, fitting window, normalizer and label budget matter. |
| H1 | Source neural PCA → population neural-to-behavior ridge → channel coefficient rows → source7→4 basis → EB shrinkage | A backward decoder-weight descriptor, not automatically intrinsic forward tuning. Source channel-space PCA imposes roster/correspondence assumptions. |
| M1 | Source-frozen rectified EMG synergy basis → per-unit encoding ridge, `[w1,w2,w3,b]` | Not direction T4. Reliable coefficients need not add predictive information to the existing activity path. |

Bound negative/context evidence before starting:

1. H1 historical H-C−H-S `+0.056287`, content-only H-C−H-C0 `+0.032557` with interval crossing0; interface widening CI64−CI32 `-0.020130`. Do not propose width as an established bottleneck.
2. H1 FiLM V5: EMPTY `+0.0254`, full `+0.0239`; profile-specific benefit was not demonstrated. Budget-aware adaptation must be part of the baseline, not renamed new information.
3. M1 rSyn3 M10 coefficient split-half about0.82–0.93; fold0 `0.911`. Reliability alone is not a content gate.
4. M1 calibration-aware behavior bottleneck improved a short-calibration DirectRidge system by about0.065–0.072, but hard projection of full SPINT lost0.021957 on average. That positive result is a motivation, not a prediction for this pilot.
5. Old H1 q3-AFC4 failed. H1 already has EB shrinkage, and its prior confidence proxy had weak association with split quality. A generic forward-ridge, PCA-rank or confidence-gate rerun is not a new hypothesis.
6. Full-T4 FiLM nulls, sparse-event estimator gains, and activity-memory gains are distinct estimands. None proves global identity saturation.
7. Absolute R2 values across tasks are not a common scale for judging carrier suitability. Use within-task matched deltas and record-specific aggregation.

Named sources are listed in §13. Bind hashes of documents/code/receipts actually used; verify checkpoint bytes rather than trusting `checkpoint_every_epoch=true` or a manifest path.

## 2. Scientific question and theoretical contract

Let S be a legal labeled calibration prefix, X a neural signal, and Y behavior:

`A_i = Phi(X_S)_i`

`T_i = C_task(X_S, Y_S)_i`

`Yhat_t = D_omega(X_<=t, {A_i,T_i}_{i=1..N})`.

The official term remains **tuning profile**. `C_task` is an estimator of task-related coupling; its parameters can have different forward/backward interpretations. Shared interface/role does not imply shared physiological axes or one cross-task checkpoint.

### 2.1 A limited, defensible explanation

For finite-second-moment Y under squared error and U=(query-neural history, activity signature), Bayes-optimal risks obey:

`R*(U) - R*(U,T) = E || E[Y|U,T] - E[Y|U] ||² >= 0`.

This is the classical conditional-expectation projection identity, not a new theorem or a practical non-inferiority guarantee. It separates:

- whether T contains extra task information conditional on U;
- whether short calibration estimates it reliably;
- whether a finite-data, trained consumer can exploit it.

An additional CP profile q must be assessed conditional on **both A and the existing T**, not on A alone. A nonsignificant empirical delta is not proof of conditional independence. Stable q, residual variance after a linear probe, or T4−AC4 are not information-theoretic ceilings.

Activity-only ambiguity can be illustrated by `x1=y+noise`, `x2=-y+noise` under symmetric behavior: unlabeled marginal statistics do not in general orient the task sign. This illustrates a possibility, not non-identifiability of every real SPINT representation.

### 2.2 H1's population dependence

In a simplified centered ridge model, `W_decode = (Sigma_xx + lambda I)^(-1) Sigma_xy`. A row can change with the other channels' covariance even if that unit's individual response relationship does not. Forward encoding and backward readout coefficients are not interchangeable physiological quantities.

The H1 assay below diagnoses this dependence in its **actual PCA/ridge/EB pipeline**. It does not isolate covariance alone from all latent-coordinate effects or imply that a backward descriptor is unusable. A Haufe-style forward-pattern transform and direct behavior-to-neural regression are also different estimators; do not silently equate them.

### 2.3 Predictions and falsifiers

| Hypothesis | Measurement | What would weaken it |
|---|---|---|
| H1 carrier partly reflects population composition | Other-channel perturbation with focal channel/labels fixed | Little change beyond numerical tolerance; no relationship to source transfer stability |
| M1 needs task-useful rather than merely stable coordinates | Matched calibration-aware versus fixed-basis profile | No improvement on the strong matched consumer despite valid learning |
| Static coefficients omit repeatable temporal coupling | Block/trial-held-out dynamic encoding prediction and stability | Extra lags improve support fit only, not held-trial prediction |
| Calibration coverage matters beyond trial count | Same-budget source support episodes with differing design spectrum | Gains/reliability unrelated to coverage under controlled exposure |

These are hypotheses, not labels inferred tautologically from whether a decoder improved.

## 3. Data and selection ledger

Stage0 uses existing public/source-development scopes only. Do not open organizer-hidden files, formal sealed sub-C sessions, or DANDI external15 to decide a descriptor. Previously inspected development dates remain retrospective development, not fresh test.

### 3.1 M1

- Reuse the four public held-in sessions `20120924/26/27/28` and their existing fold-local loader. Primary candidate outer fold0 leaves20120924 out; sources are26/27/28. Existing historical query surfaces are not re-declared sealed.
- Calibration is chronological first10 legal trials. Query windows must be entirely post-support; do not check only the last target bin. Use the existing window length100 and aligned target view unless a separately frozen spec states otherwise.
- Report exact output dimensions, native signed EMG scoring space, source-only scale, rectifier and NNMF provenance. The rectifier for the carrier does not change the decoder target to rectified EMG.
- Check **all pretraining lineage**, not only current gradients. A parent trained on the outer target cannot support a clean leave-session-out claim even if frozen. Use a proven source-excluded parent or classify the proposed surface as a different, explicitly pre-exposed development diagnostic; do not silently downgrade the claim and proceed under the clean label.
- Do not substitute the M4/last6-trial later-day proxy for a matched M10 endpoint.

### 3.2 H1

- CPU explanation uses public held-in calibration records with their existing date-LODO/source-authority plans; no hidden/calibration export side effects.
- Audit M3 deployment separately from historical M4 evidence. Three long trials contain many correlated observations; they are neither three scalar labels nor thousands of independent trials.
- Bind first3 trial IDs, 100-ms carrier block law, source PCA/ridge/U/EB artifacts and raw channel order. No padding a fourth trial.

### 3.3 Source selection versus product selection

Every eventual trained arm saves all12 epochs and records endpoint12. Its source-development selection uses an identical rule and candidate budget across arms; tie within1e-10 chooses earlier epoch.

Visible-development epoch-picking is allowed as a separately labeled product view, not prohibited by calling it held-out. It must list exactly which sessions/epochs were inspected. It cannot simultaneously be presented as independent evidence of unseen-session generalization. Hidden EvalAI labels/results never enter fitting or epoch choice here.

Basis/prior fitting, lag decisions and source normalizers are fold-local. For a source-validation date used in clean checkpoint selection, decide and document whether it was also used in representation/basis fitting; if it was, it is not an unseen-date test. The Stage1 spec must provide a complete named split table rather than the ambiguous phrase "source-only".

## 4. E stream: source-only explanation program

### E0 — authority and reuse, before new measurements

Produce `inventory.json` containing actual paths/hashes, tensor shapes, estimator direction, calibration/query boundaries, label type/counts/seconds, normalization authority, training lineage, available per-epoch checkpoints, and source-selection scope.

Read and reuse existing M1 `token_probe_v1` / `budget_probe_v1` receipts where applicable. Do not repeat probes or imply that an old linear redundancy estimate proves absence of nonlinear useful information.

Output `evidence_matrix.md` with current matched content deltas, system deltas, selection status, and unresolved items. An all-source model score does not fill a missing clean fold-local cell.

### E1 — H1 focal-channel population-dependence assay

Question: does the current backward descriptor change substantially when other neural inputs change, while the focal unit and behavior are unchanged?

Frozen first assay:

1. Use first3 support trials, the existing source-frozen transform, and eight focal channel indices evenly spaced across the roster, selected without labels/performance. Record the exact indices.
2. For each focal channel, generate16 deterministic masks with a dedicated seed20260905, retaining75% of the other channels (round down). Always retain the focal channel.
3. For dropped channels replace support rates by that session's support mean for that channel. Keep the original shape and source PCA; do not refit target PCA. Preserve focal rates, labels, timestamps and source transform exactly.
4. Refit the existing backward descriptor with each intervention using an isolated wrapper/copy. Report raw-row and final EB-carrier relative change, cosine, covariance/conditioning and prior shrinkage. Numerical degeneracy is recorded, not repaired by tuning lambda.
5. A separately computed per-unit forward-regression descriptor with a fixed behavior basis should be invariant to this other-channel intervention; this is a diagnostic control, **not** an automatic candidate replacement or a claim of numerical equivalence to H-C.
6. Analyze source-date/recording aggregates, not channel×mask pairs as independent biological replications. Count finite/undefined cases. No decoder R2, new H1 model, or result-guided mask choice.

The intervention alters neural correlations and their behavior relation. It measures pipeline population dependence, not an isolated causal effect of real electrode loss, and is not a deployable data augmentation by default.

### E2 — M1 dynamic coupling precheck

Compare a static source-frozen rSyn3 encoding with one named finite-lag diagnostic basis:

`z_dyn(t) = concat(z0(t-100ms), z0(t), z0(t+100ms))`.

The lag set is fixed here for a feasibility test, not selected from an open sweep. All paired samples must remain within the same calibration/source trial; use the intersection of valid rows for the static and dynamic fits. Future-within-calibration labels are available offline; this does not grant future query labels to the decoder.

- Fit using source prefix trials1–10, evaluate neural encoding prediction on disjoint source trials11–20; record if a session cannot supply legal rows. Do not apply this diagnostic to a target query and call it legal target fitting.
- Use the existing intercept-unpenalized ridge convention, with feature scaling and lambda fixed before reading results; Stage0 inventory must spell out the exact normalized objective inherited from the bank, not just a naked `lambda=1` with a changed sample count.
- Dynamic has more coefficients. Its held-trial prediction difference is feasibility evidence, not a capacity-matched neural-decoder mechanism test.
- Report per-session prediction error, coefficient stability under trial-block splits, effective design spectrum and valid exposure. No threshold chosen from a favorable lag/session. Failure means no promotion of **this** temporal descriptor, not all temporal tuning.
- Do not add these lags to the first P pilot. Time structure and learned-basis objective remain separate axes.

### E3 — coverage and argument audit

Collect Gram eigenvalues, trial/block counts, support duration and feature variation for the existing legal M1/H1 support. Reuse prior budget receipts; do not infer effective independent sample size from bin count alone.

Deliver a short explanation that separates feature-family misspecification, estimation reliability, train/deploy mismatch, and consumer sensitivity. A +0.005 result or null is not a measured global noise floor or saturation boundary.

E completion is an auditable descriptive report, not a requirement that a preferred hypothesis be true.

## 5. P stream: calibration-aware tuning-profile pilot

### 5.1 Unique first research question

Does optimizing the source behavior basis **through the short-support carrier estimator and downstream query loss** improve a strong activity+carrier consumer, relative to a fixed basis under matched source training?

Do not repeat an output-manifold hard projection. The model continues to predict all native16 EMG outputs, with its full neural/activity path preserved.

### 5.2 Intended operator

`z_eta(t) = f_eta(y_S(t))`

`Theta_S(eta) = solve(Z_S^T W Z_S + Lambda, Z_S^T W X_S)`

`T_i = source_normalize([Theta_i, intercept_i])`

`Yhat_Q = strong_consumer(X_Q, A_S, T_S)`.

Only source training differentiates through `solve`. Target computes the frozen transform and a deterministic closed-form fit using its M10 support. No target optimizer, query-label fit, target normalizer refit, or hidden-state pseudo-label learning.

The initial output interface remains4 coordinates (three coefficients plus intercept) for a controlled comparison. This is an engineering control, not a universal rank theorem. No change of rank, lag, activity law, decoder architecture or label budget in the first cell.

### 5.3 Required Stage1 implementation freeze

Before a GPU pilot is eligible, produce `STAGE1_IMPLEMENTATION_SPEC.md` with the following resolved **without inspecting new decoder scores**:

1. Actual parent checkpoint bytes and all source-exclusion lineage; exact strong-consumer topology and injection point. Reuse a carrier-aware topology; do not attach an untrained path to an activity-only checkpoint and call it the same model.
2. An explicit `f_eta`: deterministic fixed-basis initialization, trainable layer shapes, source-only scale/basis authority, sign/nonnegative semantics, coefficient order, and any bounded residual/regularizer. A trainable basis must not gain merely by arbitrarily rescaling its coordinates relative to the ridge penalty. Fix a source-defined scale/gauge or transform the penalty consistently.
3. Exact weighted ridge objective, penalty on slopes only, numerical precision, solver, training gradient contract and degeneracy handling. No frozen NumPy cache on a supposedly differentiable path.
4. Carrier normalization law and initialization parity with the chosen anchor. Reusing an old normalizer after changing coefficient coordinates is not automatically valid. Any necessary rematerialization must apply to both arms.
5. Named source fit, source selection and outer report sessions; complete support/query window disjointness. For the proposed first outer fold, source-only learned objects exclude20120924. Parent pretraining must obey the same claim.
6. Trainable parameter allowlist in both arms. The shared consumer must have exactly the same trainability/initialization/updates; the candidate alone trains the basis. Preserve all original raw-output residual capacity.
7. Training horizon12 **complete query-window passes**, effective batch32, one paired seed42, no last-batch drop, session-pure shuffled batch manifest shared by arms. Begin with AdamW LR1e-4, clip1.0, weight_decay1e-2 with standard bias/norm exclusions; freeze these settings and precision before scoring. No FiLM subsampling epoch masquerading as a full pass.
8. One-epoch warmup then constant LR through12, no automatic24 extension for this new P pilot. Save every epoch with model/basis/optimizer/RNG/sampler/normalizer state; source-picked plus endpoint12 reporting. A visible-development pick is a separate product view with identical search rights across arms.
9. Empirical timing/memory and exact per-epoch/query counts; only disposable smoke/profiling before formal initialization.

This is a deliberate implementation-spec gate: the previous conceptual discussion did not choose a unique trainable basis or bind a proven parent. The executor must not pretend those facts are already settled. Stage0 can finish even if Stage1 is blocked by missing lineage or an unresolved operator; record the concrete issue instead of expanding the method search. A new model choice beyond this spec needs a named revision, not an undocumented fallback.

### 5.4 First performance matrix

| Arm | Basis / target fit | Consumer training | Purpose |
|---|---|---|---|
| P-FIX | Fixed source basis, same M10 closed-form estimator | Matched allowed consumer parameters | Necessary trained baseline |
| P-CA | Source calibration-aware basis, same estimator structure/budget | Same consumer parameters + declared basis parameters | Candidate |
| Parent | Immutable initial strong system, score-only | None | Context for whether either fine-tune damages the existing product |

First pass is two trained arms, not a broad ablation grid. A weak DirectRidge comparator cannot replace P-FIX. The P-CA−P-FIX delta is initially a system/objective intervention including the extra trainable basis, not isolated proof of label content.

Resource decision after12:

- Both arms pass structural checks and P-CA source-selected report delta≥+0.005 versus P-FIX, without losing≥0.01 to the same-surface strong parent: eligible for **one** confirmation stage, not an immediate paper claim.
- Candidate worse or no useful signal: close this frozen pilot; no automatic rank, lag, width, learning-rate, FiLM or epoch sweep.
- Missing/invalid comparison: engineering/inference unresolved, not scientific null.

These point thresholds are deliberately modest resource-routing values, not MDE, statistical significance or non-inferiority margins. One outer fold cannot establish cross-session efficacy. The confirmation proposal should prioritize another predeclared outer session before multiplying seeds on one date; both dimensions of replication are ultimately needed.

For visible-development product selection, publish the separately picked rows even if the source-selected routing gate fails. Do not silently replace that gate. Any product deployment remains outside this work order.

## 6. Correctness tests before P training

1. Parent inventory and forward replay; zero basis perturbation reproduces the intended fixed estimator and prediction within prebound tolerances.
2. Coefficient recovery on synthetic data; intercept unpenalized; label permutation changes pairing without changing acquired-label count.
3. `solve` gradients finite and nonzero to the learned basis; finite-difference/gradcheck on a small float64 system. Nonzero gradient must survive the actual consumer.
4. Complete query/history disjointness; modifying forbidden query labels cannot change support carrier or target adaptation.
5. Simultaneous permutation of neural units, activity signatures and carrier rows preserves set-consumer output. Do not extend that claim to H1's fixed-roster estimator without a separate check.
6. Dropout masks synchronize neural/activity/carrier paths. Identical per-sample source exposure and dedicated RNG domains across paired arms.
7. Native output units and signed target view unchanged; no hidden output projection.
8. Target fit performs zero optimizer/backward steps; source learned state hashes remain unchanged through target evaluation.
9. Full resume test on a disposable instance, including basis and optimizer state; checkpoint completeness checked in actual bytes.

Different GEMM shapes/devices are not unconditionally bitwise. Use float64 estimator synthetic comparisons and prebound FP32 forward tolerances; bind the historical M1 last-bit drift disclosure rather than falsifying immutable digests.

## 7. Deferred successors, not first-wave work

| Successor | Trigger required | Mandatory boundary |
|---|---|---|
| Dynamic tuning profile | E2 held-trial evidence plus a new small spec | Do not combine it with P-CA in the first attribution |
| H1 forward-pattern carrier | E1 motivates a concrete distinction from failed q3-AFC4 | Same M3 budget and matched retraining; no post-hoc swap into frozen H-C as decisive test |
| Segment-level functional memory | Existing M2 A-QMEM establishes a useful candidate and source coverage supports segmentation | Actual legal calibration segments, not blindly K=3 trial tokens on H1 |
| Full-width/typed carrier tokens | Evidence that a fixed4 interface loses useful identifiable structure | Matched capacity; not "16 outputs implies16 carrier coordinates" |
| SUA/pMUA companion | A confirmed estimator/consumer protocol and matched signal-view authority | Pool neural rates then refit nonlinear T4 terms; matched labels/query/selection rights |

For future functional memory, store per-segment `G_j=Z_j^T Z_j` and `v_ij=Z_j^T x_ij` plus activity features. Sums recover the whole-support linear fit statistics. Query-conditioned weighting changes the estimator and must preserve conditioning/regularization; it is not just an O(1) median or an existing implemented feature.

Current M2 A-QMEM retains activity embeddings and global T4, **not** these paired functional sufficient statistics. No future query labels, no hidden query writes, no continual-boundary inference is introduced here.

## 8. Subagent ownership and parallelism

Use one coordinator plus at most three active workers. Do not spawn one agent per idea/dataset/seed. This section is a delegation plan for the executor, not a report that those agents were launched.

| Owner | Exclusive files/responsibility | Independent work |
|---|---|---|
| Coordinator | `contracts.py`, `plan.py`, runner, inventory/selection manifests, job ledger, final report | Freeze interfaces, bind authority, manage GPU leases and scope |
| E worker | `h1_population_audit.py`, `m1_dynamic_audit.py`, E-only tests/report | CPU explanation assays; no decoder training or scoring |
| P worker | `basis.py`, `carrier_solver.py`, `model_adapter.py`, P-only synthetic tests | Differentiable estimator and implementation-spec proposal on synthetic/source support |
| Shared worker | `data.py`, `training.py`, `evaluation.py`, data/selection/resume tests | Reuse canonical loaders, split proofs, caching, sampler, full checkpointing |

New owned namespace, only when execution begins:

`tfpd_exploration/src/cross_dataset_functional_calibration_v1/`

`tfpd_exploration/tests/test_cross_dataset_functional_calibration_*.py`

`tfpd_exploration/scripts/run_cross_dataset_functional_calibration_v1.py`

New execution root: `tfpd_exploration/results/cross_dataset_functional_calibration_v1/<timestamp>/`.

Reuse/import canonical data/timebase operators rather than forking whole historical packages. Any shared API change is coordinator-owned and versioned. Do not edit active M2 dual-track implementation as part of this namespace; its resume repair remains with its existing owner.

Every worker prompt must include: **You are not alone in the codebase. Modify only your owned files; do not revert others' changes; preserve historical roots and accommodate shared contracts. Do not independently claim a GPU, open a new dataset scope, or launch an unregistered experiment.**

When E finishes, reuse its slot for read-only verification/monitoring rather than creating another independent worker pool. Permissions inherit the current full-access environment; do not build approval-prompt loops into scripts.

## 9. Hardware and time budget

- Active M2 shuffled12→24 comparison has scheduling priority. E/P synthetic preparation is CPU-only while it runs. Recheck GPU leases; historical snapshots are not current availability.
- Two3090s are two24GB devices, not48GB pooled memory; no DDP for these small paired pilots.
- Once Stage1 is technically frozen and execution is directed, P-FIX and P-CA may each occupy one idle GPU. Keep the same data manifest. If only one GPU is available, run sequentially unless a measured same-card two-arm probe improves aggregate throughput≥15% within20GiB peak VRAM.
- MemAvailable≥12GiB, total working set/PSS target≤44GiB. Loader workers0 initially, at most2 after measured I/O bottleneck; Torch/BLAS2–4 threads per job. No reliance on swap, no duplicate materialized overlapping-window tensors.
- Shared memmap arrays; cache frozen support/activity once per session. Learned-basis-dependent carriers cannot be detached and cached across updates. Multiple queries sharing a support may share one fit within a gradient-valid forward batch.
- Save full model/basis/optimizer/RNG states. Background jobs use a persistent launcher and ledger; one read-only monitor records progress/resource/ETA every60s. Only scoped, verified own PIDs may be stopped.

Timeline from executor handoff:

| Window | E | P/shared | Required output |
|---|---|---|---|
| First2h | Evidence/estimator inventory | Parent availability, synthetic API, split ledger | Concrete asset/scope blockers, not a second generic plan |
| By24h | E1/E2/E3 bounded CPU report | Implementation spec and correctness tests | Stage0 conclusion; Stage1 ready or specific technical block |
| Following24–48h after spec freeze | Review held-trial evidence | Measured-cost paired12-epoch pilot if eligible | Source-selected + endpoint + labeled visible-product rows |
| Within two weeks | Cross-dataset explanation synthesis | At most one separately specified confirmation stage | Useful direction or closed pilot; no promised positive result |

CPU diagnostics budget one workday, not an open-ended lag/rank search. If a data operator or missing parent would exceed it, stop that branch with its precise missing dependency. GPU duration is predicted only after a100-step disposable profile, not extrapolated from FiLM-head timings.

## 10. Later mechanism experiments

Only after a usable performance signal, design three focused tests:

1. **Content:** correctly paired labels versus an appropriate label-association null, with same-capacity trained controls. Complete-row shuffle alone tests attachment, not all content. Jointly permuting memory keys/values is invariance, not a null.
2. **Calibration coverage:** same labeled budget, different source support coverage/conditioning, maintaining legal deployed prefixes. Show whether the new estimator changes this dependence.
3. **Transfer:** second predeclared session/task with its proper estimator semantics and label ledger. No requirement that every dataset gain; no calling a chosen best seed universal evidence.

Do not require these expensive arms before the first paired performance pilot, but do not write a second mechanism innovation without them.

## 11. Paper-safe synthesis and prohibited claims

Proposed text:

> We treat calibration as estimating the task-related role of recorded channels rather than recovering physical neuron identities. Activity signatures summarize unlabeled neural observations, while tuning profiles provide task-coordinate information from paired calibration data. A shared-across-session decoder combines these complementary sources. Directional T4 is one estimator within this framework; applicability depends on task structure, identifiable calibration coverage, and a consumer trained for the deployed estimator and budget.

For H1, explicitly identify the backward decoder-weight descriptor; do not call it a direct forward physiological tuning fit. For cross-dataset discussion, distinguish algorithmic interface generality from identical estimators, identical coordinates, equal label costs, or a single cross-task trained model.

SUA/pMUA: deterministic merging can discard heterogeneous unit responses, including cancellation of directional coefficients. Under identical full information, Bayes prediction from SUA can reproduce a pooled view, but finite-data/model performance can favor pooling. Thus no structural guarantee that every measured SUA model beats pseudo-MUA, and no favorable-row selection to enforce the narrative.

Forbidden conclusions: reliable profile⇒useful; low linear residual variance⇒information-theoretic ceiling; no FiLM gain⇒no tuning value; zero-init⇒trained non-inferiority; improvement over weak ridge⇒improvement over strong SPINT; one selected positive⇒universal transfer; sparse estimates beating one dense implementation⇒less information is intrinsically better.

## 12. Minimal handoff and stopping report

Deliver `HANDOFF_FOR_ASTRA_REVIEW.md` with:

- one decision per stream and each completed/blocked/deferred stage;
- estimator direction, label/time budget, split/parent authority;
- actual checkpoint/normalizer/basis hashes and permitted trainable state;
- CPU assay definitions/results including negative/undefined cases;
- paired training/selection/endpoint/session metrics if run;
- source-selected versus visible-product-selected separation;
- actual GPU-hour/memory and all attempted configurations;
- strongest alternative explanation and smallest justified next step.

Report structural failures as structural failures, completed nulls as completed nulls, and unrun work as NOT_RUN. Do not ask the user to approve normal file reads or routine local commands; ask for direction only if the scientific scope or unavailable authority would materially change.

## 13. Existing sources and prior art

Local evidence:

- `sua_exploration/docs/T4G_GENERALIZED_ANALYTIC_FUNCTIONAL_CARRIER_ROUTE_20260806.md` — generalized encoding family, native-M2 equivalence, failed H1 q3 pilot.
- `SPINT-main/src/data/h1_m4_eb_pilot.py` — actual population PCA/ridge/U/EB and M3 deployment estimator.
- `SPINT-main/docs/H1_CARRIERID_QUALITY_DIAGNOSTIC_PROGRAM.md` — existing shrinkage/quality/oracle boundaries.
- `sua_exploration/docs/CURRENT_RESULTS.md` — matched H1 content/width and DANDI signal-view evidence; distinguish historical context from latest products.
- `tfpd_exploration/h1_series_20260830/docs/RESULT_H1_FILM_CONTENT_DIAGNOSTIC_V5_20260904.md` — profile-content null.
- `tfpd_exploration/docs/RESULT_M1_EMG_RSYN3_STAGE0_20260902.md` and fold-local successor — reliability and scope correction.
- `tfpd_exploration/src/m1_emg_rsyn3_fold_local_v1/` and its existing token/budget/full-query receipts — reuse without mutating.
- `sua_exploration/docs/HANDOFF_M1_CALIBRATION_AWARE_BEHAVIOR_BOTTLENECK_20260824.md` — deployment-matched objective positive on ridge, hard projection negative on full SPINT.
- `tfpd_exploration/docs/RESULT_FABLE_TKD_M1_V1_20260905.md` — direct carrier-to-decoder synthesis negative; not a reason to replace the strong consumer here.

Primary literature, consulted for the preceding analysis:

- [SPINT](https://arxiv.org/abs/2507.08402): activity-derived identity is existing work, not a new contribution of this plan.
- [Haufe et al., 2014](https://pubmed.ncbi.nlm.nih.gov/24239590/): interpretation of backward weights versus forward patterns.
- [Bertinetto et al., ICLR2019](https://arxiv.org/abs/1805.08136): differentiable closed-form adaptation is prior art; novelty must be narrower and evidenced.
- [Sani et al., PSID](https://www.nature.com/articles/s41593-020-00733-0): behaviorally relevant dynamics need not coincide with dominant unsupervised neural variance; not a direct validation of our proposed carrier.

No new novelty-priority claim is made without a focused related-work check at the Stage1 freeze. The proposed contribution is a testable calibration/consumer system, not renaming ridge, PCA, FiLM, attention or an SSM.
