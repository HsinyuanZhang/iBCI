# SUA / native-MUA held-out calibration program — current status for external AI review

**Snapshot:** 2026-08-05 13:58 HKT  
**Purpose:** independent scientific and experimental-design review  
**Primary endpoint:** held-out-session, few-shot, calibration-time backpropagation-free decoding  
**Execution state at the original snapshot:** two local RTX 3090s and one remote RTX 5070 Ti were
training; the 14:07 post-snapshot update supersedes the local-r9 portion  
**Score discipline:** this document does not open or inspect the currently running native-M2
Phase-C scores, and it does not open the external `sub-M` endpoint  
**Authority:** descriptive review document only; it does not authorize a new run, score opening,
formal-test access, EvalAI submission, or protocol change

> **Latest terminal update, 02:08 HKT on 2026-08-06.** The external sub-M M30
> true-early-start replay is complete (`270/270`, 15 sessions, two views, three arms, three seeds).
> Both activity and T4 calibration stop after trial 30 and every 50-bin query history is fully
> post-boundary. On the newly available trials31--50 alone, SUA T4 R² is `0.393664`, with
> T4−Zero4 `+0.483843` and T4−TS4 `+0.508258`; pseudo-MUA gives
> `0.339323/+0.439688/+0.475302`. All four paired contrasts are 3/3 seeds and 15/15 sessions
> positive, and all hierarchical-bootstrap lower bounds are positive. This closes the behavioral
> latency gap left by the 00:40 label-budget curve: decoding can begin immediately after the
> first-30 activity/label block. It is a post-hoc characterization on the same external cohort,
> not a second cross-animal confirmation. The strict Native-M2 Ridge24-W50 closure also completed:
> Ridge/F0/T4/K4 mean R² are `.113916/.173901/.226786/.245774`; K4−T4 is only `+.018987`,
> below the frozen `.03` gate, so K4 remains a control/mechanism result rather than a T4 upgrade.
> Independent closure audit found no missing mandatory GPU cell in the three exploratory routes:
> cross-budget correction and fixed-K remain stopped by their prospective gates; paired-view C1
> is complete as shared-weight non-inferiority, not an accuracy or consistency-loss gain.

> **Terminal post-snapshot update, 00:40 HKT on 2026-08-06.** The external label-budget
> robustness curve is now complete (`450/450` new cells, with V9 M50 reused). At fixed
> query-after-50, M15 versus M50 is `-0.018713` R² on SUA and `-0.017839` on pseudo-MUA;
> both views pass the frozen grand-mean plus all-three-seed `-0.03` adequacy rule. M30 is
> essentially identical to M50 (`+0.001326/-0.000782`), and both hierarchical-bootstrap lower
> bounds are above `-0.03`. This reduces direction labels but does not show decoding can start at
> trial 15: the activity path still uses 30 trials, and the study kept all query windows after 50.
>
> **Earlier terminal post-snapshot update, 22:25 HKT.** Two questions listed as open in the 13:58
> snapshot now have results. External subject-M V9 completed all 270 cells: SUA
> T4/zero4/TS4 R² are `0.356828/-0.057766/-0.115319`, and pseudo-MUA are
> `0.306073/-0.086878/-0.164314`; all four T4-control contrasts are 3/3 seeds and 15/15
> sessions positive. The additional 150-cell F0/PV50/Ridge50 matrix gives SUA
> `-0.196077/0.115374/0.417922` and pseudo-MUA `-0.204725/0.104219/0.410193`: T4 clearly
> exceeds PV50, but not the dense-continuous-label Ridge50. Native-M2 matched Phase-C seed42
> exact14 gives SPINT/T4 `0.293110/0.382906` (`+0.089796`, 7/7 sessions positive); seeds43/44
> remain unrun. Sections below preserve the original snapshot for audit history; statements that
> sub-M has no score or that Native Stage A is still running are superseded by this update and by
> `HELDOUT_BP_FREE_PUBLICATION_EVIDENCE_MATRIX_20260805.md`.

> **Important post-snapshot update, 14:07 HKT.** The two local r9 shards were stopped at epoch 6
> after an independent score-blind audit found a formal continuation P0: r9 destroyed its sole
> private key after signing Stage-A-only execution capabilities and contains no same-root decision
> signer, Stage-B issuer, or opening issuer. No endpoint/decision/R² was opened; this is an
> engineering-control failure, not an accuracy result. Append-only r10 with a live memory-only
> signer was initially proposed, but was cancelled at 14:15 because this local security control is
> not necessary for scientific validity. r10 is instead a static-manifest, single-supervisor
> implementation that runs exact14, invokes the frozen no-output gate once, and conditionally starts
> Stage B without any manual intermediate selection. The remote shared-zero4 job continues. A separate independent
> audit also rejected external-control V7 because caller-created `TrustedRoots` could forge pinned
> provenance in-process; V7 remained blocked and exposed no external data or score, and append-only
> V8 is under repair.

## 1. Executive conclusion

The program has produced meaningful positive evidence, but the evidence is asymmetric across
datasets and should not be compressed into the statement “T4 works everywhere.” The present
best-supported interpretation is:

1. On sorted SUA development sessions, a small, correctly unit-attached functional descriptor is
   strongly useful. Most of the useful content is the first-harmonic coefficient pair `[a,c]`, not
   waveform quality, SNR, a static electrode lookup, or a larger network-side fusion block.
2. The descriptor has an exact and experimentally checked merge law from SUA units to
   deterministic electrode-pooled pseudo-MUA. One shared source-trained weight set can serve both
   views without material accuracy loss.
3. On organizer-hidden native M2, the submitted T4 system is substantially better than the
   original SPINT submission in held-out R². That is a valid end-to-end system result, but it is not
   yet a pure T4 attribution result because the two submitted packages used different decoder
   epochs.
4. Native M1 is a negative boundary: the official private-test T4 result is slightly worse than
   original SPINT, and the earlier local M1 endpoints cannot support a held-out gain claim.
5. Two decisive scientific questions remain open: a matched native-M2 T4-versus-SPINT causal
   comparison, and a new-animal `sub-M` T4-versus-zero4/TS4 endpoint. The local M2 Stage-A matrix
   and the required remote zero4 source training are running now.

The defensible contribution is therefore not “a universally superior decoder.” It is a compact,
supervised functional calibration mechanism that is fitted analytically at deployment, requires no
target-session gradient, has a principled SUA-to-pooled-MUA transformation, and has strong evidence
on SUA plus an encouraging native-M2 system result. Its dataset geometry and label-budget boundary
must be stated explicitly.

## 2. Exact method and terminology

### 2.1 T4 is an auxiliary functional carrier; the selected system still contains an activity path

For unit or channel `i`, labelled calibration trials fit

```text
r_i(t) = b_i + a_i cos(theta_t) + c_i sin(theta_t) + epsilon_i(t)
m_i    = sqrt(a_i^2 + c_i^2)
T4_i   = [a_i, c_i, m_i, b_i].
```

The selected B3S/T4 model combines this descriptor with a chronological neural-activity identity
path. Consequently:

- “T4” is the four-value functional carrier and its fitted side path;
- “B3S/T4” is the deployed model that consumes both neural support and T4;
- T4 does not replace all B3/SPINT-style calibration information;
- saying “T4 contains B3” is acceptable only as shorthand for the selected complete model, not as
  a definition of the descriptor itself.

The target-session computation is:

```text
chronological neural support --------> activity/session identity --+
                                                                  +--> cached identity --> decoder
permitted target directions + rates -> analytic T4 fit -----------+
```

### 2.2 What is trained and what is frozen

The phrase “backprop-free” applies only to target-session calibration:

```text
source training:       ordinary backpropagation is allowed
target calibration:    analytic fit + forward identity finalization; no optimizer/backward/update
online query:           frozen weights + cached identity forward
```

Different completed experiments used different source-training controls:

- the main SUA/C1 models source-trained the encoder and decoder together;
- native-M2 Phase C freezes all 31 decoder tensors of the T4 arm bit-for-bit to the matched,
  same-fold/same-seed selected SPINT decoder, while source-training only the T4-aware identity path;
- no experiment updates decoder weights on the target session.

Thus it would be incorrect to say either “the decoder was always frozen” or “the decoder was
always co-trained.” Both variants have been tested for different causal questions. The invariant
deployment property is zero target-session backward/optimizer steps.

### 2.3 Supervision and fairness

T4 is supervised. SPINT/B3 receives neural support; T4 receives the same neural support plus one
target-direction value for each eligible rewarded calibration trial. For native M2 Phase C, the
first 33 trials contain 16 finite directional trials and 17 centre/rest trials; the latter are not
assigned synthetic angles. The 16 labels jointly fit three cosine-design coefficients for all 96
channels.

Therefore:

- T4 versus SPINT is a fair **deployment-package** comparison under matched neural exposure;
- it is not an equal-label-information ablation;
- T4 versus zero4 asks whether the labelled descriptor content adds value under a matched
  four-dimensional interface;
- T4 versus TS4 asks whether the descriptor must be attached to the correct unit/channel row;
- T4 versus TS4 alone is insufficient to prove that T4 beats no descriptor, because a wrong row
  attachment can be actively harmful.

## 3. Evidence classes used in this document

| Class | Meaning |
|---|---|
| **Established result** | Completed, auditable result that supports the stated claim within its exact data scope |
| **Development evidence** | Valid causal or descriptive result on visible/reused development sessions; not an independent formal confirmation |
| **External system evidence** | Organizer-hidden score for a complete submitted system; causal attribution may still be incomplete |
| **Withdrawn / void** | A previously quoted number with support/query overlap, zero-query sessions, or another fatal endpoint defect |
| **Engineering incident** | A run stopped before a valid score payload; it is neither a positive nor negative scientific result |
| **Running / sealed** | Training is active, but the predeclared score-opening boundary has not been reached |
| **Blocked** | Infrastructure exists but a prerequisite is absent; no scientific score may be inferred |

## 4. Current evidence ledger

### 4.1 Sorted SUA: strong development mechanism evidence

The strict SUA scope uses 27 source-training sessions and six source-held-out development sessions.
The separate formal `sub-C` scope is not available for a fresh one-shot claim, so these results must
remain labelled development evidence.

| Contrast or arm | Result | Evidence-supported interpretation |
|---|---:|---|
| ordinary T4 − F0 | about `+0.2528 R²` | labelled functional calibration adds large held-out-session development value |
| ordinary T4 − TS4 | about `+0.2522 R²` | correct unit/content attachment matters, not interface width alone |
| full T4 absolute mean | `0.574976` | reference for component attribution |
| AC4 `[a,c,0,0]` absolute mean | `0.562753` | within `0.03 R²` of full T4 |
| zero4 absolute mean | `0.326008` | information-free matched-width reference |
| AC4 − AC4-RS4 | `+0.294163 R²` | 3/3 seed and 6/6 session means positive; correct coefficient-to-unit row attachment is decisive |
| AC4-RS4 − zero4 | `-0.057418 R²` | incorrect functional identity is worse than omitting it |

The strongest current mechanism claim is:

> A compact, amplitude-weighted first-harmonic coefficient carrier `[a,c]` retains nearly all of
> full T4 on the SUA development scope, and its value depends on correct unit-row attachment.

This does not prove that pure preferred-direction phase is sufficient. The phase-only PH4 arm did
not satisfy the full-T4 non-inferiority requirement. It also does not prove new-animal or formal
generalization.

### 4.2 Deterministic pseudo-MUA: a controlled granularity bridge, not native MUA

Pseudo-MUA is formed by summing sorted SUA spike counts within each electrode. In separately
trained models:

| Contrast | Mean delta R² |
|---|---:|
| pseudo-MUA T4 − F0 | `+0.317739` |
| pseudo-MUA T4 − TS4 | `+0.365674` |

The cosine fit has a known merge law. With a common design matrix, `a`, `c`, and `b` add across
units assigned to an electrode, while `m` is recomputed from the summed `a,c`. The 33-session audit
found exact binned-count conservation and pooled-rate T4 error below `4.68e-6`.

This is a meaningful mechanistic bridge because the carrier transforms predictably when units are
merged. It is not proof of native threshold-crossing MUA: pseudo-MUA shares the original SUA spike
events, sessions, behavior, and noise source.

### 4.3 Shared SUA/pseudo-MUA C1: one weight set is enough

C1 used one shared source-trained B3S/T4 weight set and the fixed objective

```text
0.5 * L_task(SUA) + 0.5 * L_task(pseudo-MUA),  lambda_consistency = 0.
```

The complete 12-cell matrix used three seeds and passed all frozen non-inferiority, attachment, and
cross-view-gap gates:

| Comparison | Mean delta R² | Sign summary | Correct conclusion |
|---|---:|---:|---|
| shared T4 − separate T4, SUA | `+0.008351` | 3/3 seeds, 4/6 sessions positive | non-inferior; not a robust improvement claim |
| shared T4 − separate T4, pseudo-MUA | `+0.012543` | 2/3 seeds, 5/6 sessions positive | non-inferior; not a robust improvement claim |
| shared T4 − shared TS4, SUA | `+0.285474` | 3/3 seeds, 6/6 sessions positive | strong correct-row attachment evidence |
| shared T4 − shared TS4, pseudo-MUA | `+0.369468` | 3/3 seeds, 6/6 sessions positive | strong correct-row attachment evidence |

The publishable result is model consolidation across a known granularity transform, not a large
accuracy improvement from co-training. There was no consistency loss, teacher-student penalty, or
view-specific head, so C1 cannot support those mechanism claims.

### 4.4 Native M2: positive official system result, attribution still open

Official EvalAI submission `578221` returned:

| Official endpoint | T4 system | original SPINT | Delta |
|---|---:|---:|---:|
| held-out R² mean | `0.30324395` | `0.18647872` | `+0.11676523` |
| held-in R² mean | `0.58760827` | `0.56824267` | `+0.01936560` |
| normalized online latency | `0.04290260` | `0.11236103` | `-0.06945843` |

This is a real organizer-hidden, end-to-end positive result. It is also evidence against the claim
that T4 necessarily creates excessive online overhead: the submitted cached-identity system was
not slower on the official normalized-latency metric.

The comparison is nevertheless confounded: original SPINT packaged an epoch-27 decoder, whereas
T4 packaged an epoch-34 teacher decoder. It is not valid to attribute the entire `+0.11676523` to
the four T4 coordinates.

Valid local future-query development evidence is directionally consistent:

| Local native-M2 endpoint | Result | Limitation |
|---|---:|---|
| M33/q33, four eligible sessions: T4 − F0 | `+0.071980`, 4/4 positive | `n=4`; exact sign p=`0.125` |
| M33/q33, four eligible sessions: T4 − TS4 | `+0.066484`, 4/4 positive | same small eligible subset |
| M24/q24, six sessions: T4 − F0 | `+0.052885`, 5/6 positive | one audited cell; not three-seed stability |
| M24/q24, common epoch 9 | `+0.046628`, 5/6 positive | checkpoint-choice diagnostic only |
| matched M24, three-seed candidate | `+0.036604` | only 2/3 seed means and 4/6 session means positive; frozen stability gate failed |

The running native-M2 Phase C is the intended matched causal bridge. It uses seven outer-session-
left-out folds, chronological support `[0,33)`, future query `[33,end)`, identical neural exposure,
the same selected SPINT decoder tensors for the paired T4 arm, and no target-session gradients.

### 4.5 Native M1: current negative boundary

The official private-test T4 held-out R² is `0.003825` below original SPINT. Earlier local M1
results cannot rescue that conclusion:

- all three local held-out-calibration files contain exactly ten trials, leaving zero future query
  after the frozen first-ten support;
- the reported internal LOSO minival is a two-trial prefix of the same calibration file and lies
  inside T4 support;
- the internal `T4−F0=+0.007837` is only 1/3 cells positive despite conditions favorable to T4;
- the old local `T4−TS4=+0.024797` content claim is withdrawn because the scored trials overlap the
  descriptor-fit trials;
- M1 direction coverage is confined to a half-plane, and its ten-trial first-harmonic fit must
  extrapolate outside the observed arc.

A clean local post-support M1 endpoint is technically constructible from the longer held-in-calib
files, but it has not been run. Given the official negative result and repeated early-gate
failures of M1 carrier variants, it is not part of the present mainline.

## 5. Explicitly withdrawn, stopped, or non-informative branches

### 5.1 Numbers that must not be cited as evidence

| Item | Disposition | Reason |
|---|---|---|
| historical M2 M33 gain near `+0.06979` | permanently withdrawn | scoring began at trial 0, overlapped the 33-trial support, and included two sessions with no post-33 query |
| historical M1 local held-out effect sizes | permanently void | all held-out-calib sessions have zero post-first-ten query |
| M1 local `T4−TS4=+0.024797` as direction-content evidence | withdrawn | descriptor fit and scored trials overlap |
| held-in/held-out M24 sign reversal as a unique mechanism | unresolved | endpoints differ in domain, file composition, support overlap, and window count; no unique cause is identifiable |
| K4 as a reliable per-channel directional identity | unsupported | typical channel direction is poorly reproducible at M24 and decoder gain misses the practical/stability gates |

### 5.2 Scientific branches stopped by frozen gates

- **K4 / KS4:** strict M24 `K4−T4=+0.018987` is below the `+0.03` margin; `K4−KS4=+0.043521`
  is positive in only 4/6 sessions. Gate-A encoding evidence remains, but the decoding mechanism
  claim does not.
- **Low-budget correction maps / estimator upgrades:** the cross-budget predictor did not transfer
  to target-free development sessions; empirical-Bayes ridge, second harmonic, and Poisson IRLS
  produced no source-only winner.
- **Fixed-K temporal prototype memory:** a simpler order-invariant B20 rate-distribution carrier
  beat the temporal prototype (`P20−B20=-0.041963`) in all four source sessions. No K/V attention
  consumer is licensed.
- **Additional FiLM/gates/attention fusion paths:** seven network-side fusion variants failed to
  establish a better consumer. Current evidence points to descriptor content and row attachment,
  not insufficient decoder interaction capacity.
- **Waveform/SNR/static electrode metadata:** the exact tested paths did not add robust decoding
  value. This does not prove that physical metadata contains no information in every possible
  model; it means the tested fusion hypotheses should not be revived on the same scope.
- **C2 consistency regularization:** no unique source-only loss normalization or lambda selector
  existed. It is closed as an identifiability/governance no-go, not as a negative R² result.

### 5.3 Engineering incidents are not negative experiments

Several native-M2 Phase-C attempts failed before a valid endpoint payload because of direct-script
import/bootstrap, Lightning sanity-validation aggregation, evaluator mode/device handling, expired
or incomplete authorization continuation, or a pre-Hydra ABI dependency. These runs produced no
scientific T4-versus-SPINT decision. Their roots and consumed nonces were retained and retired
append-only; the current r9 root uses fresh nonces and an independently checked pre-Hydra
capability path.

No external reviewer should count r1–r8 engineering attempts as negative accuracy repetitions, or
infer selection bias by treating the absence of an R² payload as a hidden poor result.

## 6. What is running now

The launch condition for this review snapshot was satisfied. Instantaneous telemetry at 13:57 HKT
was:

| Host/device | Active workload | GPU utilization | memory used | power |
|---|---|---:|---:|---:|
| local RTX 3090 GPU0 | native-M2 Phase-C r9 fold 0 | `84%` | `1,919 MiB` | `373 W` |
| local RTX 3090 GPU1 | native-M2 Phase-C r9 fold 1 | `80%` | `1,489 MiB` | `392 W` |
| remote RTX 5070 Ti Laptop | C1-matched shared-zero4 seed 42 | `37%` | `582 MiB` | `89 W` |

These instantaneous percentages are execution-health observations only. They are not throughput,
latency, convergence, or accuracy measurements.

### 6.1 Local native-M2 Phase-C r9 — both RTX 3090s

At this snapshot:

- both signed systemd shards are active;
- GPU0 runs SPINT fold 0 / seed 42; GPU1 runs SPINT fold 1 / seed 42;
- each active cell has produced epoch-0 and epoch-1 checkpoints;
- no Stage-A completion, decision, or score opening exists;
- Stage A is exactly `7 folds × {SPINT,T4} × seed42 = 14` terminal cells;
- GPU0 owns folds `0,2,4,6`; GPU1 owns folds `1,3,5`;
- within a fold the order is `SPINT -> T4`.

Only after exact 14/14 completion, zero failures, zero extras, and hash closure may the Stage-A
opener read the seven paired deltas. The predeclared severe-negative stop is:

```text
(seed42 mean T4-SPINT <= -0.03 R²) OR (positive fold deltas <= 1/7).
```

If it triggers, the native local matrix stops. If it does not trigger, the result is named
`continue_without_positive_claim`, not “pass,” and the remaining 28 cells for seeds 43/44 run.
Only the complete 42-cell matrix may support a positive claim, requiring:

1. mean paired delta at least `+0.03 R²`;
2. all three seed means positive;
3. at least 6/7 seed-averaged session means positive;
4. paired two-SE lower bound positive;
5. seed/session hierarchical-bootstrap lower bound positive;
6. finite absolute means and positive absolute T4 mean.

The active r9 program receipt SHA-256 is
`0a1069a67d4ff4d4ac145dfdb3678e9b625eabd0c5d9c73ea862bb58c5089fb1`; the portable manifest
SHA-256 is `81d7073210c51d3f4da0a7c8605680014b0b565c8f653ac3ca743552fd2d2ab3`.

### 6.2 Remote shared-zero4 source training — RTX 5070 Ti Laptop GPU

The remote machine is training the C1-matched `shared_zero4` control. At this snapshot:

- seed 42 is active and has sealed checkpoints through epoch 3;
- each checkpoint is approximately 64.77 MB;
- no formal or external `sub-M` score has been opened;
- seeds 43 and 44 remain sequential continuations;
- the append-only V3 seed-44 renewal passed static CPU checks but correctly remains
  `predecessors_pending`; it has emitted no execution capability, nonce, CUDA initialization, or
  subprocess because real seed42/43 terminal closures do not yet exist.

The control is essential because `T4−TS4` alone conflates useful content with harm from wrong row
attachment. `T4−zero4` tests whether the calibrated descriptor beats an information-free,
same-width interface under the same shared source-training recipe.

### 6.3 External `sub-M` score-only endpoint — prepared but blocked

The external scope is immutable DANDI 000688 version `0.250122.1735`, subject `sub-M`, center-out.
The frozen manifest contains 22 assets; a score-blind audit produced one common eligible cohort of
15 sessions. This is cross-animal evidence within the same Dandiset, not an independent laboratory.

The future matrix is:

```text
15 sessions × 2 views × 3 seeds × 3 arms = 270 scored cells
views = {SUA, deterministic pseudo-MUA}
arms  = {shared_t4, shared_zero4, shared_ts4}.
```

V7 was initially `BLOCKED_MISSING_ZERO4_TERMINALS`, so it had no active policy/grant and could not
open the endpoint. Independent adversarial review then found its `TrustedRoots` production boundary
forgeable in-process: a caller could create synthetic roots and mutate self-reported provenance
fields until `_assert_roots` accepted attacker-controlled policy/checkpoint/run-auth keys. V7 is
therefore rejected, not independently passed. Append-only V8 must remove caller-supplied root
objects from every production transition and internally revalidate the source-pinned anchor on
each privileged operation. The blocked state prevented external data, real checkpoint, Torch/GPU,
policy/grant, or R² access. No external `sub-M` R² exists.

Primary external comparisons are `T4−zero4` and `T4−TS4` in SUA; pseudo-MUA is a required
secondary. Scoring starts strictly after trial 50. Activity calibration uses first 30 rewarded
trials, whereas T4 labels/rates use the first 50; those budgets must not be conflated.

## 7. Remaining experiment count

There are **three execution blocks** left in the contracted mainline, corresponding to **two final
scientific endpoints** and one prerequisite control-training block:

1. **Native-M2 Phase C:** r9 was stopped as a continuation-control failure before a score existed;
   fresh r10 must rerun the 14-cell seed42 Stage A under a live-signer chain. The 28-cell seeds43/44
   Stage B remains conditional on the same frozen futility gate.
2. **Shared-zero4 source model:** three fixed seeds; seed42 is running remotely, seeds43/44 remain.
3. **External `sub-M` three-arm endpoint:** one write-once 270-cell score matrix after all nine
   checkpoint closures and independent V7 activation audit.

Quantization is paused and is not counted in the remaining mainline. No new network-side fusion,
M1 rescue, K4 precision replication, temporal-memory, or low-budget correction GPU run is planned.

## 8. Label budget, stability, compute, and state

### 8.1 Label requirement

T4 does not require dense per-bin kinematic labels. It requires one direction for each eligible
rewarded calibration trial and per-trial neural rates. The important variable is not nominal trial
count alone; it is whether the cosine design has adequate directional coverage and whether `[a,c]`
is reproducible.

Observed split-half behavior is approximately:

- `[a,c]` reliability `0.59–0.62` at `M=10`;
- `[a,c]` reliability `0.86–0.87` at `M=50`;
- baseline rate `b` near `0.99` much earlier;
- M2 per-channel direction improves near M32/M33, while M1 M10 retains half-plane extrapolation.

The supported operating points are currently SUA M50 and native-M2 M33. M10/M15 should not inherit
the same claim. The evidence suggests a label/geometry lower bound, not a requirement for a large
general labelled dataset at deployment.

### 8.2 Computational overhead

For the audited native-M2 Phase-C dimensions:

| Quantity | matched clean SPINT | T4 arm |
|---|---:|---:|
| online MACs per query window | `84,021,248` | `84,021,248` |
| support-calibration MACs | `1,875,935,232` | `21,393,408` encoder + `10,692` analytic fit |
| peak FP32 stream-calibration live state | `235,008` bytes | `64,552` bytes |
| post-finalization cached identity | `19,200` bytes | `19,200` bytes |

Under this implementation, T4 calibration is about `87.6×` lower in MACs than the matched SPINT
support path and uses about `72.5%` less peak stream-calibration state. Raw support and explicit T4
rows are released after identity finalization; online queries retain the same cached identity
shape. These are source-bound analytical counts and state contracts. The official lower normalized
latency is corroborating system evidence, not a pure mechanism benchmark because the submitted
packages differ.

### 8.3 Quantization side branch

Encoder-only W8A8 PTQ preserved accuracy on the C1 development scope, but failed the frozen
saturation gate. QAT seed43 passed all strict gates; seed42 preserved accuracy and integer parity
but failed pseudo-MUA input saturation (`0.029291 > 0.005`). There is no uniform three-seed
quantized conclusion, and quantization is intentionally paused until the FP32 causal endpoints
close. The decoder remains FP32 in this branch.

## 9. Current innovation claims and their maturity

| Candidate contribution | Current maturity |
|---|---|
| supervised, target-session BP-free functional calibration | supported as a method definition and by completed experiments |
| `[a,c]` is the dominant SUA carrier content | strong development evidence |
| correct unit/channel row attachment is necessary | strong SUA and pseudo-MUA development evidence |
| aggregation-homomorphic SUA-to-electrode carrier | analytic proof plus numerical audit |
| one shared weight set serves SUA and deterministic pseudo-MUA | completed non-inferiority evidence |
| T4 improves organizer-hidden native M2 end to end | supported by official submission |
| the official M2 gain is a pure T4 effect | not established; Phase C is running |
| T4 is uniformly useful on native M1 and M2 | contradicted by M1 |
| T4 attachment generalizes to a new animal | not yet established; `sub-M` score is blocked |
| low-budget correction reduces label needs | not supported by current frozen gates |
| temporal prototype memory is useful | not supported |
| waveform/SNR/electrode ID is universally useless | too strong; only the exact tested paths failed |
| deployment calibration is label-free | false |
| the full training procedure is backprop-free | false; only target-session calibration is BP-free |

The most coherent paper-level novelty is the combination of functional calibration, correct row
attachment, merge-equivariant carrier semantics, and target-session BP-free deployment. The M2 and
external `sub-M` closures determine whether that mechanism can be claimed beyond the current
development scope.

## 10. Questions for the external AI reviewer

Please review the project against the following questions rather than proposing an unrestricted
new architecture search:

1. Is “supervised, held-out-session backprop-free calibration” the correct and sufficiently precise
   central claim?
2. Does the native-M2 Phase-C design adequately remove the decoder-epoch confound while preserving
   a fair deployment comparison?
3. Should `T4−SPINT` be framed as deployment utility rather than descriptor-only causality because
   target labels are asymmetric?
4. Do `T4−zero4` and `T4−TS4` together identify descriptor content and row attachment cleanly enough
   for the external `sub-M` endpoint?
5. Is `[a,c]` plus its additive pooling law a stronger and more defensible novelty than preferred-
   direction phase, confidence-FiLM, waveform metadata, or electrode lookup?
6. Are the C1 shared-model results best framed as model consolidation/non-inferiority rather than
   accuracy improvement?
7. Does the official M2 latency plus the exact MAC/state accounting adequately answer the
   computational-overhead objection?
8. Which claims should remain “development evidence,” which are already publication-ready, and
   which require the two pending endpoints?
9. Are the Stage-A futility rule and full 42-cell gates statistically and operationally reasonable,
   given only seven native-M2 outer sessions?
10. After Phase C and external `sub-M`, is any additional GPU experiment necessary before method
    closure, or would further architecture search mainly increase researcher degrees of freedom?

## 11. Authoritative evidence index

- Main method/evidence closure:
  [`HELDOUT_BP_FREE_CALIBRATION_METHOD_CLOSURE_20260804.md`](HELDOUT_BP_FREE_CALIBRATION_METHOD_CLOSURE_20260804.md)
- Mechanism and stopped-idea audit:
  [`BP_FREE_CALIBRATION_MECHANISM_AND_PRE_CLOSURE_IDEA_AUDIT_20260804.md`](BP_FREE_CALIBRATION_MECHANISM_AND_PRE_CLOSURE_IDEA_AUDIT_20260804.md)
- Native M1/M2 implementation and endpoint corrections:
  [`NATIVE_MUA_T4_M1_M2_PROGRAM.md`](NATIVE_MUA_T4_M1_M2_PROGRAM.md)
- M2 corrected M24/M33 and K4 audit:
  [`M2_CPU_CORRECTION_AND_CARRIER_RESULTS_20260801.md`](M2_CPU_CORRECTION_AND_CARRIER_RESULTS_20260801.md)
- Native-M2 Phase-C protocol:
  [`M2_NATIVE_T4_SPINT_POST33_CONFIRM_V1_DRAFT_PROTOCOL_20260804.md`](M2_NATIVE_T4_SPINT_POST33_CONFIRM_V1_DRAFT_PROTOCOL_20260804.md)
- Current Phase-C device-recovery boundary:
  [`M2_NATIVE_POST33_PHASE_C_V5_DEVICE_RECOVERY_20260805.md`](M2_NATIVE_POST33_PHASE_C_V5_DEVICE_RECOVERY_20260805.md)
- Official EvalAI evidence:
  [`E8_FALCON_EVALAI_SUBMISSION_RECEIPT.md`](E8_FALCON_EVALAI_SUBMISSION_RECEIPT.md)
- SUA AC4 row-attachment result:
  [`SUA_AC4_RS4_RESULT_AUDIT.md`](SUA_AC4_RS4_RESULT_AUDIT.md)
- Paired-view C1 interpretation:
  [`C1_POSTRUN_MECHANISM_CLAIM_AUDIT_20260804.md`](C1_POSTRUN_MECHANISM_CLAIM_AUDIT_20260804.md)
- C2 no-go:
  [`SUA_C2_SOURCE_ONLY_DISPOSITION_AUDIT_20260805.md`](SUA_C2_SOURCE_ONLY_DISPOSITION_AUDIT_20260805.md)
- Formal-scope boundary:
  [`SUA_FORMAL_ONE_SHOT_READINESS_AUDIT_20260805.md`](SUA_FORMAL_ONE_SHOT_READINESS_AUDIT_20260805.md)
- External `sub-M` V7 control plane:
  [`DANDI_000688_SUBM_CO_THREE_ARM_SCORE_ONLY_PROTOCOL_V7.md`](DANDI_000688_SUBM_CO_THREE_ARM_SCORE_ONLY_PROTOCOL_V7.md)
- Live experiment board:
  [`ACTIVE_EXPERIMENT_CONTROL_BOARD.md`](ACTIVE_EXPERIMENT_CONTROL_BOARD.md)

## 12. Reviewer warnings

1. Do not cite the withdrawn historical M2 M33 or M1 local held-out numbers.
2. Do not treat pseudo-MUA as native threshold-crossing MUA.
3. Do not describe the six SUA development sessions as a fresh formal confirmation.
4. Do not infer a Phase-C score from GPU utilization, checkpoints, engineering retries, or the
   absence of a score payload.
5. Do not call Stage-A non-futility a positive result.
6. Do not call C1 an accuracy-improvement experiment; its primary result is non-inferiority plus
   correct-row attachment.
7. Do not call T4 label-free or the full source-training pipeline backprop-free.
8. Do not activate or score external `sub-M` from this document; V7 is rejected and V8 is not yet
   independently approved.
9. The working tree contains many active research artifacts and uncommitted user/agent changes.
   This file is a review snapshot, not a clean-release manifest or an immutable receipt.
