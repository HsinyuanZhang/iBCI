# ROOT REVIEW — SUA / FALCON M1 下一轮实验路线

**Reviewed:** 2026-08-02（Asia/Hong_Kong）  
**Status:** evidence review and conditional route only; **does not authorize a new run, official
submission, EvalAI access, datamodule change, or GPU launch**. The currently frozen D4/DS4 pilot
continues under its own protocol and is not modified here.

**Reviewed input:**
[`SUA_M1_NEXT_EXPERIMENT_ROUTE_DRAFT.md`](SUA_M1_NEXT_EXPERIMENT_ROUTE_DRAFT.md).

## 1. Root verdict

The Terra draft gets the central scientific direction right:

1. sorted SUA, derived pseudo-MUA, and FALCON M1 native channel counts are separate evidence
   domains;
2. calibration-derived functional carriers have priority over static waveform/electrode identity;
3. D4/DS4 is the minimum non-cosine M1 falsification and must finish before another M1 arm starts;
4. generic FiLM, static electrode lookup, same-electrode REL, and the current spatial prior must
   not be relaunched under a new name;
5. official held-out few-shot performance is the final endpoint and cannot be used repeatedly for
   candidate selection.

The draft is **accepted as an evidence map but not accepted verbatim as an execution plan**. The
mandatory corrections below define the reviewed route.

## 2. Mandatory corrections to the draft

### 2.1 The honest M1 held-in endpoint is already implemented

The draft says the held-in-calib post-support endpoint is not yet supported by the production
datamodule. That is stale. The query-window path now fail-closes on the allowed M1 LOSO windows
`(10,None)`, `(10,210)`, and `(210,None)`. It was exercised by the completed disjoint replay and
by `m1_clean_selection_v1`, whose frozen boundaries are:

- support `[0,10)`;
- checkpoint selection `[10,210)`;
- sealed report `[210,end)`.

No new “endpoint readiness” project is required after a D4 failure. A new candidate still needs
its own immutable receipt and boundary/hash validation, but it must reuse the existing endpoint
semantics rather than reimplement them.

### 2.2 A positive D4 must go to candidate freeze before another M1 architecture search

If the D4 first cell passes, execute only the two already authorized extension cells. If the
three-cell gate then passes, the preferred sequence is:

```text
D4 three-cell development pass
  -> freeze D4 semantics, M=10, normalization, checkpoint rule and image/hash
  -> one official M1 few-shot held-out evaluation
```

Do **not** use the same two-session local report repeatedly to choose a D4 key residual, FiLM,
latent width, support budget, or mixture before the official D4 test. The local sample is too
small for that extra model-selection layer. A later D4 interaction study would need an independent
development scope and cannot use the official score to select or rescue it.

### 2.3 D4 failure does not automatically authorize a learned encoder

A failed D4 first cell stops D4 extension immediately. A support-set encoder may then receive a
**CPU-only feasibility audit**, not an automatic GPU slot. Because only four M1 source sessions
are available, all proposed `5%/10%` proxy improvements are provisional practical margins, not
power-calibrated significance thresholds. Before freezing them, the audit must report estimator
repeatability, paired resampling uncertainty, and the minimum distinguishable proxy effect.

The future-rate diagnostic must also disclose its oracle boundary precisely:

- support condition labels may enter the carrier;
- query labels may not enter the carrier or deployed decoder;
- if future condition labels are used only to construct/score the offline predictive proxy, that
  is a later-label oracle diagnostic and must be named as such, exactly as in the D4 Gate-G audit.

### 2.4 First-five/last-five is not a valid primary PCA split at M=10

Each M1 first-ten support contains only two or three trials per category. A chronological 5/5 split
can omit categories and confound subspace stability with scheduled condition imbalance. The primary
reliability splits must retain all ten trials and their condition coverage, for example:

- odd/even temporal bins within each support trial with matched exposure; or
- deterministic spike thinning / paired-bin bootstrap within each trial.

Chronological 5/5 may be reported only as a drift stress test and must be marked `undefined` when
rank or category coverage fails. It cannot be the deployment estimator or the main stability gate.

### 2.5 “0-GPU Gate A” may still contain CPU model fitting

A meta-trained support-set encoder is not a no-training method. Its Gate A may be CPU-only and
source-only, but any learned weights, nested source-fold selection, and use of future neural targets
must be reported. Deployment remains forward-only/no-backprop; that property must not be used to
hide offline training complexity.

### 2.6 The next dynamic-interaction screen belongs to SUA first, not to M1 D4 selection

SUA already has a strong selected T4 substrate and six validation sessions. M1 D4 is still an
unproven carrier and has only two local left-out sessions in the current three-cell design.
Therefore the only retained dynamic-interaction candidate is a **SUA T4** baseline-preserving
attention-logit/key residual. It is not inserted between an M1 D4 development pass and the official
M1 evaluation.

For hardware, prefer the factorized logit form. Cache only the per-unit rank-`r` factor, not a full
`N x 512` key residual. At `N=64,r=8`, the static factor is `512` values: `2,048 B` in FP32 or
`512 B` in INT8, before metadata. Exact MAC/state/latency receipts are mandatory; a full-width
`131,072 B` FP32 cache is not the default deployable design.

## 3. Reviewed route A — FALCON M1

### A1. Current D4/DS4 decision (only active M1 GPU route)

Use the already frozen first-cell gates without reinterpretation:

```text
D4 - F0  >= +0.015 R2
D4 - T4  >  0
D4 - DS4 >= +0.010 R2
```

- Fail any comparison: stop D4, no extra seed/fold, mixture, FiLM, PCA carrier, support sweep,
  quantization, or official submission.
- Pass all: add only `fold1_seed43` and `fold2_seed42` under the frozen protocol.
- Three-cell continuation requires mean `D4-F0 > +0.03` and positive mean `D4-DS4`; even then the
  result is development evidence over two distinct sessions.
- If the three-cell gate passes: freeze one D4 candidate and proceed to one official M1 few-shot
  held-out evaluation. Do not interpose another M1 model search on the same report endpoint.

### A2. Conditional support-set functional encoder (only after D4 first-cell failure)

This is preferable to a generic autoencoder or undefined contrastive learner because the known
support condition, exposure, and neural statistics are explicit inputs, while a new session still
requires only streaming sufficient-statistic accumulation plus one forward finalization.

The CPU feasibility audit must be source-only and nested leave-one-source-session-out. Compare:

1. closed-form D4;
2. learned four-dimensional carrier E4;
3. E4 complete-row shuffle ES4;
4. support-condition-label shuffle;
5. rate/exposure-only input.

The audit must first estimate its own paired noise/MDE. Provisional practical targets are at least
`5%` lower future-rate error than D4 and at least `10%` lower than ES4, with improvement in at
least `3/4` source sessions. These numbers do not become a GPU gate unless the identifiability
receipt shows they are distinguishable. Any future-label use is scorer-only oracle information and
must be disclosed; it can never enter E4.

Only a distinguishable, content-specific Gate-A result may justify a separately reviewed one-cell
GPU protocol with F0/D4/E4/ES4. If Gate A fails, stop M1 representation search rather than increase
latent dimension, encoder width, augmentation, or contrastive-pair freedom.

### A3. Population subspace audit (diagnostic, not a ranked GPU arm)

Run only if it is useful for deciding whether A2 has a defensible population context. On source
sessions and M=10:

- use within-trial paired splits that preserve all ten trials and exposures;
- report projector trace overlap/principal angles, eigen-gap and rank degeneracy;
- prove channel-permutation covariance and comparison-statistic invariance;
- compare with row-shuffle and matched-noise nulls;
- report incremental future-rate proxy value with the same later-label disclosure as A2.

Stable eigenvalues or a Gram spectrum alone are session-level summaries, not per-channel identity.
Even a positive audit does not authorize PCA loading concat. It may only motivate a separately
specified permutation-equivariant population-context carrier after A2's evidence gate.

## 4. Reviewed route B — sorted SUA

SUA is not waiting for another proof that static T4 content exists: aligned T4 is already strongly
positive, while ordinary T4@15 label reduction, W3 shrinkage, confidence-FiLM, static electrode
gate, same-electrode REL, and the current spatial prior did not establish the required improvement.

The only retained accuracy candidate is the selected-T4, zero-initialized, factorized attention-logit
residual:

```text
teacher path: K,V,Q and decode(x + E_T4) unchanged
new path:     bias[j,i] = q_factor[j] dot u_factor(T4_i)
step 0:       bias == 0 exactly
```

Required controls are:

1. selected-T4 continuation;
2. aligned T4 residual;
3. TS4 residual, shuffling only the new residual input rows;
4. parameter-matched additive NoFiLM residual.

Before any run, prove bitwise zero-init equality, cached/on-the-fly equality, optimizer whitelist,
unit permutation behavior, and exact incremental state/MAC. Because this candidate adds compute,
the recommended seed-42 practical gate is `>= +0.03 R2` versus all three controls with all six
validation sessions positive. Failure stops the branch; no rank sweep, confidence/waveform input,
value residual, or arbitrary backbone unfreezing.

This is **SUA validation development only**. The existing `sub-C/CO/27-6-6` formal-test scope is
already occupied; confirmation requires a separately authorized external subject/scope such as
`sub-M` or `sub-J`, not another adaptive use of the occupied test sessions. No result from this
SUA route can substitute for FALCON M1 official held-out evidence.

## 5. Waveform/SNR and electrode information

No new GPU branch is retained.

- F1/F2 remain scientifically `indeterminate`, not proven universally useless.
- Their controlled observations are nevertheless non-positive and do not justify repeating raw
  static concat.
- T4GATE and REL are `ineffective`; the current geometry prior triggered its Stage-0 NO-GO.
- Waveform/SNR may remain in CPU QC: bad-channel detection, sorting audit, estimator reliability,
  coverage/abstention analysis. It does not re-enter identity or FiLM without new source-only,
  content-specific evidence and a separately defined safety objective.

## 6. Final authorization order

```text
NOW
  M1: finish D4/DS4 first cell and apply its frozen gate
  SUA: no new GPU job while the M1 decision is unresolved

IF D4 FIRST CELL PASSES
  -> frozen two-cell extension
  -> if three-cell gate passes, freeze D4
  -> one official M1 few-shot held-out evaluation

IF D4 FIRST CELL FAILS
  -> stop D4
  -> optional source-only CPU A2 identifiability + support-set feasibility audit
  -> no GPU unless proxy improvement is both content-specific and distinguishable

INDEPENDENT LATER SUA TRACK
  -> one baseline-preserving low-rank T4 logit-residual seed-42 causal screen
  -> stop unless +0.03 against continuation, shuffled residual, and matched additive control
  -> external-subject confirmation before any formal claim
```

Population PCA/loading, raw waveform concat, electrode tables/REL/spatial priors, generic FiLM,
dynamic hypernetworks, D4+T4 mixture, support-budget sweeps, and quantization are not authorized by
this review.

## 7. Execution update — 2026-08-02

This section records outcomes obtained after the route above was frozen. It supersedes the
conditional `NOW` state in section 6 but does not change the predeclared gates.

### 7.1 M1 D4/DS4 stopped locally

The first-cell support/query-disjoint report is
`results/m1_d4_pilot_v1/heldout_report.json`. The fixed comparisons were:

| contrast | paired R2 delta | gate | pass |
|---|---:|---:|:---:|
| D4 - F0 | -0.00817764 | >= +0.015 | no |
| D4 - T4 | -0.01385933 | > 0 | no |
| D4 - DS4 | +0.00217366 | >= +0.010 | no |

All three gates failed. The two extension cells, M1 architecture variants, support sweep and D4
quantization therefore remain stopped. At the user's explicit request, one frozen D4 image was
submitted to the FALCON M1 EvalAI few-shot phase together with frozen original and T4 images. That
submission is a hidden-set comparison, not permission to tune or rescue D4 from its score.

### 7.2 M1 support-set E4 CPU Gate-A also stopped

The authoritative corrected artifact is
`results/m1_support_encoder_gate_a_v2/audit.json`; v1 is retained but superseded by the numerical
correction in `M1_SUPPORT_ENCODER_GATE_A_NUMERICAL_CORRECTION.md`.

- E4 / D4 geometric raw-MSE ratio: `1.2082`, better in `2/4` source LOSO sessions, paired-t ratio
  interval `[0.6187, 2.3593]`;
- E4 / ES4: `0.0252`, better in `4/4`, interval `[0.0055, 0.1150]`;
- E4 / label-shuffle: `0.8158`, better in `4/4`, interval `[0.7780, 0.8553]`;
- E4 / rate-only: `0.9709`, better in `2/4`, interval `[0.8055, 1.1703]`.

Thus the learned code contains condition/unit attachment information, but the learned
cross-session transformation is neither more useful than D4 nor distinguishably better than a
rate-only carrier. The frozen decision is `ineffective_stop_m1_representation_branch`; no M1 E4
GPU pilot, latent-width search, generic autoencoder, contrastive rescue or population-loading
decoder arm is authorized.

### 7.3 SUA low-rank T4 logit residual completed and stopped

The factorized implementation and production preflight passed before data training. The frozen
protocol is `SUA_T4_FACTORIZED_LOGIT_RESIDUAL_PROTOCOL.md`; the preflight receipt is
`results/sua_t4_factorized_logit_residual_v1/preflight.json`, and the immutable final aggregate is
`results/sua_t4_factorized_logit_residual_v1/aggregate.json`.

All three permitted seed-42 validation arms completed. The selected T4 encoder and coupled decoder
remained frozen; only 48 rank-8 factor parameters were optimized. Scores were re-derived as the
unweighted mean over epochs 5--12:

| arm | validation R2 |
|---|---:|
| selected-T4 continuation | **0.590273** |
| aligned T4 logit residual | 0.587131 |
| residual-row-shuffled logit | 0.580729 |
| parameter-matched additive control | 0.583375 |

The aligned residual was `-0.003142 R2` versus selected-T4 continuation (3/6 sessions positive),
`+0.006402` versus shuffled residual (5/6), and `+0.003756` versus additive control (4/6). It
therefore failed both the `>=+0.03` practical threshold and the all-six-positive requirement
against every required control. The frozen disposition is
`fail_stop_no_rank_seed_confidence_formal_or_int8_follow_up_authorized`: no rank sweep, extra seed,
confidence extension, formal SUA opening, or INT8 continuation is allowed for this branch. No
formal SUA test session was opened.

### 7.4 Official EvalAI M1 original/T4/D4 comparison

At the user's explicit request, the three already frozen images were submitted once to FALCON M1
challenge 2319, phase 4599. All three submissions finished successfully. The authoritative machine
receipt is `results/evalai_m1_threeway_v1/threeway_summary.json`, schema
`evalai_m1_threeway_result_v2`, SHA-256
`fc6be339b0db98879e897a9186380b5483b082d42aa9937e1ab347a6ca25f010`.

| arm | submission | official Held-Out R2 | official Held-In R2 | normalized latency |
|---|---:|---:|---:|---:|
| original | 578244 | **0.6485909555** | **0.7501227656** | 0.1147790631 |
| T4 | 578245 | 0.6447659719 | 0.7438711960 | 0.0447179677 |
| D4 | 578247 | 0.6439926711 | 0.7402397096 | **0.0440473690** |

Aggregate T4-original deltas are `-0.0038249836` held-out R2 and `-0.0062515696`
held-in R2; D4-original deltas are `-0.0045982844` and `-0.0098830560`. Thus neither
descriptor improves official M1 accuracy over the original image. Both descriptor images do have
about a 61% lower normalized-latency value, which is a deployment observation rather than an
accuracy rescue. EvalAI exposes submission-level aggregates here, not per-session paired values;
no paired significance or session-consistency claim is permitted.

### 7.5 Closed-form decoder-latent alignment stopped at the selection gate

The independently audited v2 runner tested a support-only shared reduced-rank map from M1
calibration features to the frozen decoder identity residual. Outer-left-out `ses-20120926`
received only `E0` and first-ten-trial support features; its teacher identity was neither computed
nor stored. Scoring was restricted to full-window-disjoint trials `[10,210)`, with no report,
held-out, EvalAI, backward, or optimizer access.

The first two arms were already decisive:

| arm | locked `(rank, lambda)` | selection R2 | delta vs F0 | 95% block-bootstrap CI | MDE |
|---|---:|---:|---:|---:|---:|
| F0 identity | -- | 0.739261 | -- | -- | -- |
| full condition carrier | `(1, 100)` | 0.739687 | **+0.000426** | `[-0.000352, +0.001220]` | 0.000749 |
| rate/exposure only | `(1, 1e-4)` | 0.739941 | **+0.000681** | `[-0.000429, +0.001815]` | 0.001074 |

The full arm missed the predeclared `+0.015` identity margin by more than an order of magnitude,
its interval included zero, and it was `-0.000254 R2` below rate-only instead of at least
`+0.010` above it. Because its MDE was only `0.000749`, this is not an unidentifiable
`+0.015` effect. The conjunction had already failed, so the label-shuffle and diagnostic
rate-residualized arms were not run under the approved early-stop rule. The branch is
`ineffective`: it does not open `[210,end)`, held-out, EvalAI, a wider rank/lambda sweep, FiLM,
or quantization.

The authoritative stop receipt is
`results/m1_decoder_latent_alignment_oracle_v2/selection_early_stop.json`; the independently
audited scorer receipt is `selection_scorer_prelaunch_v4.json` with SHA-256
`19ed4356174b232f8f51b148701f817c538e2907d33a91a63e7776ed15e357b1`.

### 7.6 Low-label confidence/cross-budget correction stopped before GPU

The Step-2A source CPU audit found incremental `q_unit+M` signal in nested source LOSO but not in
the six target-free development-validation sessions: only 3/6 favored it, its interval crossed
zero, and dropping the most favorable validation session reversed the mean. The predictor also
estimated scalar error magnitude rather than a signed descriptor correction. Step-2B
cross-budget/dropout training is therefore NO-GO. The independent result audit is
`SUA_STEP2A_INDEPENDENT_RESULT_AUDIT.md`.

### 7.7 Fixed-K temporal prototypes produced a mixed source signal and stopped

The exact-four-source CPU Gate A found a large, stable prototype-versus-rate-only proxy gain:
mean `+0.100526`, 4/4 sessions positive, paired 95% CI `[+0.090798,+0.110253]`, and MDE
`0.012719`. The split-half prototype-value cosine 2.5% lower quantiles were all above `0.995`.
However, the predeclared session-keyed slot-shuffle contrast was catastrophically heterogeneous;
its CI crossed zero and its measured MDE exceeded the mean contrast. Therefore the overall gate
is `stop_cpu_gate_not_met`, not a decoder/GPU authorization. The result is retained as
source-only carrier evidence, not a claim of behavior decoding or coherent cross-session slot
semantics. Full audit: `M1_FIXED_K_TEMPORAL_PROTOTYPE_GATE_A_RESULT_AUDIT.md`.
