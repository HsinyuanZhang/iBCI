# CRST finite-family experiments — progress and acceptance ledger

Current concise handoff: [one-page closeout overview](HANDOFF_EXPERIMENT_CLOSEOUT_20260906.md).
This file retains historical snapshots; use that overview for the current state.

Latest status (15:16 HKT): **H1 training stopped and local evaluation completed;
the large same-surface gap warrants keeping this line paused.** All owned
training and evaluation processes have exited; all saved states remain. FLAT
committed through21 and ROUTE through22. The common21 EMA evaluation completed
15:14 in418.51s: selection `.490809/.483968`, complete20,325 pooled
`.452043/.443402`, equal-session `.440952/.432322`, worst `.310980/.181193`.
Both arms lose all13 sessions to C2; pooled deltas vsC2 are
`−.436456/−.445097`. Versus their own parent12 they improve
`+.173958/+.158759`. Root independently verified all target/session/endpoint
arrays against Original and parent12, recomputed pooled metrics, checked all11
owned artifact hashes and unchanged RAW/AdamW/EMA/RNG. This is local evaluation
only, not a held-out verdict or completion of24 epochs. No successor work is
queued. Receipt SHA `27e53118a92d9065ba4d8dae5a351e88238ba43c2b874f2369d5c88064137d06`.

Selected M1 FLAT/ROUTE pooled R²
`.811652/.812196` versus as-shipped Original `.809289` on identical31,252
positions, with local12.9–16.9× P95 speedups; session-specific losses remain.
H1 cold-prefix/dense gains do not close its quality gap. M2 QueryAge+prefix has
completed all24 epochs and source-only EMA selection: selected FLAT epoch2 and
ROUTE epoch20. Fixed ext4 pooled R² is `.143064/.296396`, respectively
`−.086134/+.067199` versus Original; ROUTE has one severe session loss, so this
is not a uniform quality win or a non-inferiority result. Both actual selected-weight
CPU complete-stream proofs passed. Six same-image2048-call repeats completed:
single-thread average latency is somewhat higher, while two-thread latency is
somewhat lower; unification takes priority over these small speed differences.
H1 QueryAge passed both-arm source1040 eligibility and an exact-candidate paired
resource smoke. Fresh QueryAge+p=.5 formal training completed the fixed12-epoch
schedule on both GPUs without capacity warm-start. Both selected EMAs were
frozen at epoch12, with selection R² `.320129/.328431` for FLAT/ROUTE.
The original supervisor's built-in exports and complete scorers finished:
complete20,325 pooled R² is `.278085/.284643`, equal-session `.267704/.275918`;
selected and endpoint12 are the same per-arm weights, not replications.
The user-authorized extension was stopped before24; the separate common21
evaluation does not fabricate an all24 selection freeze. Source208 fit is not a
generalization claim. The existing Original local complete20,325 pooled R²
`.960784` is higher than historical C2 `.888499`, but that is not a held-out
ranking or a necessary local threshold for held-out success. C2's recorded
official HO `.375989` exceeds Original paper-LR `.2615` and released-LR `.2099`;
C2's visible-HO development epoch-selection exposure remains disclosed.
The Original archive's exactly aligned2,908-point subset gives `.963693`,
versus C2's recorded direct selection scalar `.871568`. Neither comparator is
dropped or silently relabeled; H1 currently remains far below both.
Detailed
receipts and qualifications are in the updates at the end of this ledger.
The original chronological snapshots below are retained, not current live status.

Snapshot: 2026-09-06 02:19 HKT / 2026-09-05 18:19 UTC. This is a local
development progress record, not a completed-goal claim or submission authority.

## Objective and interpretation

User priority clarification (2026-09-06): **network unification takes priority
over small latency disadvantages**. Keep the common QueryAge FLAT/ROUTE network
as the active three-task direction; latency measurements describe the tradeoff
and guide a subsequent exact implementation optimization, not a reason to split
the primary backbones again. This does not waive task-quality requirements,
change frozen selections, or turn resource-safety smoke gates into optional work.
The source audit confirms a common full network-design family: local-k=5,
16-channel spatial convolution, eight width-256 slots, slot attention/FFN,
2048-to-256 pooling, the same QueryAge16 core and norm/readout path. Necessary
task input/output shapes and independently learned weights do not negate that
template. H1's local-balanced initialization/unscaled-dot spatial preset is
the material remaining explicitly named exception; the three instantiated
networks are not claimed to have identical operator settings or weights.
CausalPE artifacts remain historical controls,
not an automatic fallback merely because of a small speed difference.

The active objective has three parts: M1/M2/H1 quality improvement or
non-inferiority, a common or small interpretable family of networks, and a
substantial latency reduction relative to the user's SPINT deployment reference.
The [family freeze](FREEZE_ASTRA_FLAT_ROUTE_OPERATOR_FAMILY_20260906.md) permits
`FW-QueryAge16` and `FW-CausalPE4` as explicitly named temporal backbones on any
task, with the same backbone for the paired FLAT/ROUTE control within each
experiment. Initial assignments were M1 QueryAge and M2/H1 CausalPE; the fresh
M2 QueryAge experiment is a declared second recipe, not a relabeling of the
causal results. ROUTE is the calibration-conditioned
**spatial slot-attention logit bonus**, not a relabeling of query-only temporal
attention. Each pair shares its frontend, backbone, training and data protocol.
These paired experiments can support a routing comparison within each task;
they do not independently identify dataset and temporal-backbone effects.

H1's named source-only initialization/logit candidates are additional declared
common-frontend hypotheses applied identically to its two arms. They are not
retroactively the recipes used by the running M1/M2 pair. The historical H1
signed frontend remains a separate baseline, not a disguised family adapter.

## Live prospective training

| Task | Fixed experiment | Resource / process at snapshot | Evidence so far |
| --- | --- | --- | --- |
| M1 | Fresh seed42 QueryAge16 FLAT/ROUTE, 24 epochs, chron80 source protocol | GPU1, CPU8–11, PID1701853 | Epoch2 persisted: EMA equal-session R² F `.77496874`, R `.77459062`; RAW F `.77925459`, R `.77736008` |
| M2 | Fresh seed42 CausalPE4 FLAT/ROUTE, 24 epochs, source-only selection | GPU0, CPU12–15, PID1702813 | Early trajectory unstable; F epoch3 EMA `.10761586`, RAW `.07461993`; no early selection |
| H1 | One local-FC1-balanced initialization test, identical fixed208 source IDs and 260→1040 gate | Temporarily shares GPU0, CPU4–7, PID1704350 | Running; g=0 parity exact and route-gate gradient nonzero; not a quality result |

Both M1/M2 primary picks are the highest **EMA** equal-session source-minival
score with earliest-epoch tie break. RAW is secondary and must not silently
replace the primary criterion. All epochs/endpoints are retained. Neither
external dev nor official hidden outcomes select this new pair. Sealed carrier
construction and older model selection create historical exposure, so these
are not claims of untouched holdout generalization.

M1 stores an accumulating [epoch metrics ledger](../results/decoder_validation_v2/20260905_190000/m1/family_v1/p1_queryage16_pair_chron80_v2/epoch_metrics.json)
and [run authority](../results/decoder_validation_v2/20260905_190000/m1/family_v1/p1_queryage16_pair_chron80_v2/run_meta.json).
M2's existing runner overwrites each arm's live epoch metric; a non-invasive
exact-PID sidecar additionally writes immutable per-epoch JSON/SHA snapshots.
M2 elapsed values are **cumulative from pair start**, not per-arm epoch times.
The original one-hour sidecar bound must be renewed if the formal job outlives
it; this changes no training or selection state.

## Same-surface baseline evidence and limitations

| Surface | Comparator | Existing candidate / result | Interpretation |
| --- | --- | --- | --- |
| M1 same31,252 chron80 scored windows | Original SPINT teacher pooled `.59785792`; equal-session `.59675312` | Historical Sfix pooled `.82788515`; equal-session `.82722393`; selected prior T equal-session `.81221350` | Source/training-overlap diagnostic; Sfix is B3+rSyn3-modified and **not Original SPINT** |
| M2 seven held-in within-post30 sessions | SPINT M30 equal-session `.69657917` | Current packaged e8 `.79602345` | Same query/units but training-overlap diagnostic |
| M2 fixed local ext4 development replay | SPINT pooled `.22919761`; equal-session `.23759323` | Current packaged e8 pooled `.38750524`; equal-session `.38441527`, ahead on each of four sessions | Both historical SPINT/e8 selection have external exposure; not pristine holdout and not selection data for the new source-only pair |
| H1 same20,325 complete-stream bins | Existing C2 pooled `.88849890`; equal-session `.88796318` | Historical signed V7 FULL+canonicalM3 pooled `.61361536` | Large quality gap remains; signed V7 is outside the common eight-slot frontend family |

Authorities: [M1 Original teacher](../results/decoder_validation_v2/20260905_190000/m1/family_v1/original_teacher_chron80_overlap.json),
[M1 Sfix](../results/decoder_validation_v2/20260905_190000/m1/family_v1/sfix_epoch011_source_dev_replay.json),
[M2 SPINT within-post30](../results/m2/family_v1/spint_m30_within_post30_replay_v1.json),
[M2 e8 within-post30](../results/m2/family_v1/e8_within_post30_replay_v1.json),
[M2 ext4 paired replay](../results/m2/family_v1/ext4_e8_spint_dev_replay_v1.json),
[M2 ext4 pooled addendum](../results/m2/family_v1/ext4_e8_spint_dev_pooled_addendum_v1.json),
[H1 C2 complete-stream](../results/decoder_validation_v2/20260905_190000/h1/c2_epoch15_same_surface_v1/c2_epoch15_complete_stream.json).
Rows are distinct surfaces and must not be compared across rows as one metric.

## H1 falsification ledger

The exact source-capacity set is 13 sessions ×16 endpoints, SHA256
`da4bf975a5c1023bbffe26da87d0d4977f437d7db1db012283511ef140d96211`.

| Common frontend candidate | Fixed260 source R² F / R | Decision |
| --- | --- | --- |
| Scale1 common-set baseline | `−.01297209 / −.01297203` | Both below `.10`: stop, no1040/minival |
| Set-v2 unscaled-dot attention | `−.00140772 / −.00140641` | Both below `.10`: stop, no1040/minival |
| Set-v3 local-block-balanced initialization | Pending | Same prospective gate; no sweep |

Read-only diagnostics found that unscaled-dot attention remained diverse after
training; its failure was not a return to uniform pooling. The trained local
dynamic FC1 projection RMS was `.01602` versus static E0 `.22364` (about14×).
The single new candidate scales only the local FC1 initialization block by
`sqrt((16+700+4)/16)=sqrt45`, applied to both arms. It introduces no bypass,
FiLM, signed mixing, runtime rescaling or extra module.

At260, continue the same checkpoint/configuration to1040 only if any arm has
R²≥`.10`, prediction-std≥`.25×` target-std, and stable loss decline. At1040,
both arms must have R²≥`.50` and std≥`.50×` target-std. Passing this gate would
authorize a review, not automatically a minival experiment or formal training.
The [prospective manifest](../results/decoder_validation_v2/20260905_190000/h1/family_v1/crst_b4_set_v2_localbalanced_prospective_v1/manifest.json)
and adjacent launch authorization bind the new candidate before updates.

## Exact runtime optimization, separate from architecture training

The new local `src/family_runtime_v1` runtime has two implementation changes:

1. Repair only the four left-boundary frontend tokens and newest token, doing
   expensive MLP/slot work on five tokens instead of all nine Conv input bins.
2. For constant learned slot queries, reassociate attention algebra as
   `(q Wk) Xᵀ + q bk` and `(A X) Wv + sum(A) bv`, avoiding dense per-unit K/V
   projections without approximation. Masks, softmax, value bias, residuals,
   finite window and temporal operator remain unchanged.

M1 lifted complete-stream validation passed: all31,252 scored bins,112,985
public gap/score calls after initial W100 priming, maximum native error
`2.145767e−6`; pooled R² delta `−3.9253e−9`. The fresh zero-padded startup,
partial batches and window transitions have separate unit tests. Authority:
[complete replay receipt](../results/family_runtime_v1/m1_lifted_complete31252_v1.json).

M2 five-token current-e8 tests cover B1/B7 and W50 boundaries. M2 lifted is a
separate candidate under lifecycle/mutation and complete-replay validation;
an initial missing lifted-cache invalidation/memory-accounting issue was found
in review and corrected before any long benchmark. The v1 receipt is retained
with the v2 corrective receipt; unit-pass alone is not latency acceptance.

Short CPU PyTorch2.5.1-post303 B4/t1 M1 probes measured V3 P95 `14.99ms`,
five-token `9.85ms`, lifted `8.41ms`; M2 B7/t1 V3→five-token was
`40.38→27.95ms`. These are same-host paired diagnostics, not official latency.
M1 long3×2048 B4/t2 runs under a user-site PyTorch2.12 installation also passed
parity and show a paired speedup, but their absolute times are **not** directly
comparable with the2.5.1 receipts. Those runs are preserved; fresh explicit
`PYTHONNOUSERSITE=1`2.5.1 long runs are in progress. Other jobs use different
CPU affinity, but shared-host effects remain possible.

The CPU source-fold Original-SPINT comparison measured a different local
baseline from the user's unspecified deployment anchor. It does not close the
reported10× gap. The exact SPINT image/checkpoint/device/batch/timing boundary
still needs to be bound before any final SPINT-relative claim. Old speedups
against a naive Transformer likewise do not satisfy that acceptance target.

## Remaining acceptance gates

- Complete fixed M1/M2 paired schedules, then strictly reload the selected EMA
  checkpoints and report both arms, RAW trajectory and same-surface baselines.
- H1 must pass learnability and subsequently demonstrate quality preservation;
  frontend geometry improvement is insufficient.
- Retain a small explicitly named family and disclose every genuine deviation.
- Complete exact runtime parity and long public-boundary measurements in the
  deployment-relevant framework/container, including reset/first call/memory.
- Bind and measure the user's SPINT reference on the same boundary; no local
  source-fold or naive-baseline substitution.
- No old payloads, images or sealed outcomes were replaced by this work; no
  registry push, EvalAI submission or hidden-query action is authorized here.

## Update — 2026-09-06 02:41 HKT

The original snapshot above is retained. Since then:

- H1 local-balanced initialization passed the frozen260 extension trigger:
  F/R source R² `.12008810 / .18026345`, std `.00192060 / .00191650`, target
  std `.00522646`, with stable loss decline. The **same** checkpoints, IDs,
  configuration and attempt are continuing to cumulative1040; no minival has
  been scored for this candidate and no formal H1 run has been authorized.
- M1 epoch4 EMA equal-session R² reached F `.80956285`, R `.80950355`.
  Fixed24-epoch paired training continues. M2's early trajectory remains
  variable and its fixed schedule likewise continues without retuning.
- M1 explicit PyTorch2.5.1 B4/t2 three-process2048-call measurements completed:
  V3 P95 `9.309 / 10.141 / 9.197ms`; lifted P95 `5.915 / 6.504 / 5.779ms`.
  All runs passed output parity. This is a `1.56–1.59×` paired P95 speedup,
  separate from the retained PyTorch2.12 runs.
- M2 B7/t2 three-process2048-call measurements completed: V3 P95
  `23.351 / 23.786 / 23.577ms`; lifted P95 `13.258 / 13.621 / 13.445ms`.
  All runs passed output parity; paired P95 speedup `1.75–1.76×`.
- M2 lifted [complete2069 replay](../results/family_runtime_v1/m2_lifted_complete_ext4_v1.json)
  passed:10,839 actual public bins after stripping each W49 startup prefix,
  all2069 sealed endpoints, every endpoint checked against the independent
  raw-window full model. Maximum all-public-bin V3/lifted error `5.96046e−8`;
  endpoint/direct error `3.72529e−8`; pooled native R² `.3875054007` versus
  sealed historical e8 `.3875052420`. This verifies implementation parity,
  not new quality selection on ext4.

The M2 SPINT reference is now much better bound locally. Three fresh runs
inside its existing frozen image used its original SPINT module, the exact
byte-matched public wrapper, seven source streams, t2, and2048 timed calls.

| Same-container P95, ms | Run1 | Run2 | Run3 |
| --- | ---: | ---: | ---: |
| SPINT image-default payload | 10.703 | 11.052 | 11.013 |
| SPINT declared seed44/e8 payload | 10.698 | 10.781 | 10.965 |
| Previous V3 Transformer | 24.062 | 24.341 | 24.456 |
| Lifted Transformer | 13.804 | 13.825 | 13.906 |

Against the image-default SPINT payload, the remaining paired P95 ratio is
`1.25–1.29×`. These are whole-batch host→native NumPy timings in the same
container process, not official-service normalized latency. Cold model/load
and reset remain materially slower for the Transformer (about177–186ms vs
16–18ms for SPINT), so the warm-call ratio is not the whole startup story.
Receipts: [run1](../results/family_runtime_v1/container_comparison/m2_frozen_spint_b7_t2_2048_r1_v1.json),
[run2](../results/family_runtime_v1/container_comparison/m2_frozen_spint_b7_t2_2048_r2_v1.json),
[run3](../results/family_runtime_v1/container_comparison/m2_frozen_spint_b7_t2_2048_r3_v1.json).

An important historical packaging issue was discovered while binding this
reference: the recorded581919 image's default command reads the old seed42
payload, while its declared selected seed44/e8 payload sits at another path.
The [read-only audit](AUDIT_M2_581919_DEFAULT_PAYLOAD_20260906.md) documents the
evidence and the conditional remote-score implication. No old artifact was
changed. Official581919 score attribution to the declared seed44/e8 is
unverified until any remote command override is checked. The581973 e8 image's
default-path payload was separately checked and matches its declared bytes.

## Update — 2026-09-06 03:25 HKT

H1's common local-balanced frontend **passed the complete source1040 gate**:
FLAT source R² `.95366392`, ROUTE `.96414522`; prediction standard deviations
`.00531422 / .00533599`, versus target `.00522646`. This is a learnability
result on the fixed208 source endpoints, not a minival or C2 quality result.
The [source1040 receipt](../results/decoder_validation_v2/20260905_190000/h1/family_v1/crst_b4_set_v3_localbalanced_source208_preflight_v1/report_1040.json)
retains the exact attempt/checkpoint continuation and operator hashes.

The resource review corrected the earlier invalid transfer of signed-frontend
timing. Sequential paired effective32 updates measured `2.8023s`, implying
`6.83h` train-only for12 epochs, already beyond the provisional6h bound.
A separate [two-GPU disposable resource smoke](../results/decoder_validation_v2/20260905_190000/h1/family_v1/crst_b4_localbalanced_splitarm_recipe_smoke_v1/report.json)
measured FLAT/GPU0 `1.4451s` and ROUTE/GPU1 `1.2472s` per update, implying
about `3.52h` parallel train-only. B8 source-forward proxies separately
estimate about53s per2908-endpoint selection surface on the slower arm.
These are scheduling estimates under shared GPUs, not performance promises.

The fresh H1 split-arm formal12 run is now actually live: supervisor
`1713513`, FLAT `1713518`, ROUTE `1713519`, beginning around03:22 HKT.
It shares physical GPUs0/1 with the untouched M2/M1 jobs, respectively.
The [launch authorization](../results/decoder_validation_v2/20260905_190000/h1/family_v1/crst_b4_localbalanced_splitarm_formal_prospective_v1/launch_authorization_v1.json)
binds the source-cache bytes, source1040/resource receipts, full execution
code list, exact new output root, and recipe before updates. Both atomic
ready receipts show the same full shared-state hash, exact g0 parity and
matching sampler/dropout identities for all12 epochs; route-gate gradient
is positive. The first background invocation ended before creating a root
or workers; the same frozen command was then explicitly invoked with a
persistent foreground handle. No trained point or epoch was restarted.

The H1 recipe remains fresh seed42, micro8/effective32, AdamW1e−4 with
epoch1 linear warmup, wd`.01`, clip1, unit-dropout`.10` intersected with
bank validity, and EMA`.9995`. Atomic end-epoch checkpoints precede scoring.
All12 EMA2908 pooled scores govern the earliest-max pick; only after both
arms finish and selection freezes do strict selected/epoch12 exports and
complete20325 reports run. The6h supervisor bound includes those stages.
No callable automatic or mid-epoch resume is claimed. Cache integrity
loading deserializes cached minival arrays; scoring now occurs only at the
predeclared formal epoch boundaries.

M1's paired24-epoch run has recorded epoch8 and continues without an early
pick. At epoch6, EMA equal-session R² was F`.81138314`, R`.81181830`;
at epoch8 F`.80995143`, R`.80894404`. The new M1 ROUTE runtime now shares
one route-aware encoder across history refresh and five-token repair.
Strict tests cover fresh g0/g`.2` B1/B4 streams, the actual P1 epoch1
checkpoint, all inactive raw/frontend/KV/count state, and numerical
model/bank/mask refresh. It has not yet received a full selected-P1 replay.

M2 gained two further algebraically equivalent implementation changes:
head-grouped value GEMMs avoid replicated weight copies, and the five
causal k5 repair convolutions are evaluated as small linear projections.
The [complete linear-candidate audit](../results/family_runtime_v1/m2_linear_complete_ext4_v1.json)
passed all10839 public bins and2069 fixed endpoints. Maximum public error
versus V3 is `7.12462e−8`; endpoint/direct-full error `3.72529e−8`; native
pooled R² `.3875054019` versus sealed `.3875052420`. This validates
implementation fidelity, not a new ext4 model-selection decision.

Three fresh same-SPINT-container B7/t2 runs each used2048 timed calls:

| Whole-batch P95, ms | Run1 | Run2 | Run3 |
| --- | ---: | ---: | ---: |
| SPINT image-default payload | 11.113 | 11.050 | 10.194 |
| SPINT declared seed44/e8 payload | 10.948 | 10.993 | 10.027 |
| Previous V3 Transformer | 24.485 | 24.475 | 23.575 |
| Linear-conv/grouped-value Transformer | 12.296 | 12.453 | 12.083 |

This is a paired `1.95–1.99×` P95 improvement over V3. Remaining ratio to
image-default SPINT is `1.11–1.19×` in these local t2 container runs.
Full receipts retain the intermediate lifted/grouped engines too:
[run1](../results/family_runtime_v1/container_comparison/m2_linear_spint_b7_t2_2048_r1_v1.json),
[run2](../results/family_runtime_v1/container_comparison/m2_linear_spint_b7_t2_2048_r2_v1.json),
[run3](../results/family_runtime_v1/container_comparison/m2_linear_spint_b7_t2_2048_r3_v1.json).
The currently submitted e8 wrapper fixes t1, so t2 results must not be
presented as measured official-service latency or a promise about that image.
A matched t1 diagnostic is being collected separately.

A separate zero-history reset candidate computes the identical pre-temporal
zero frontend token once per session, materializes50 copies, and omits the
discarded reset-time temporal prediction. It leaves nonzero-history rebuilds
and all steady-state operations unchanged. B1/B7 lifecycle tests passed;
24 paired real-source reset/first8-call checks measured mean reset
`173.443→3.743ms`, with unchanged persistent tensor bytes and maximum native
first8 error `1.21072e−8`. See the [reset receipt](../results/family_runtime_v1/container_comparison/m2_zero_history_reset_b7_t2_repeat24_v1.json).
The receipt's two constructor timings are order-confounded by the first
module import; they do **not** establish a constructor/load speedup. The
accepted measurement here is the alternating-order repeated reset boundary.
The reset-only candidate is not yet a submitted or fully integrated release.

Finally, M2's new paired-training source-minival surface is1011 windows, not
the much larger within-post30 diagnostic surface. Exact e8/SPINT source-minival
baseline replay is being prepared; the earlier `.796/.696` figures must not
be used as if they were scores on these1011 windows. Both historical models
have held-in training exposure, which will be explicit in that comparison.

## Update — 2026-09-06 03:56 HKT

The exact M2 source-minival comparison is now available, and materially changes
the quality assessment on that surface. All1011 ordered windows from the same
seven sessions used by the new paired trainer were replayed without fitting.
The actual Original-SPINT model was strictly loaded from its bound historical
checkpoint, without a Lightning/data-module setup. Its identity used only the
first30 calibration trials; the compact cache contains33 trials, and both the
full stored file and the ordered first30 slice are bound in the receipt.

| Same1011 source-minival surface | Equal-session R² (primary) | Pooled R² (secondary) |
| --- | ---: | ---: |
| Current frozen e8 payload | .08125935 | .17032600 |
| Actual Original-SPINT M30 | .64101315 | .65731178 |

Original-SPINT is ahead in each of the seven sessions. The
[native baseline receipt](../results/m2/family_v1/source_minival_e8_spint_m30_replay_v1.json)
and its NPZ bind all starts, targets, predictions, input/model/code hashes and
the exact counts173/129/117/116/141/141/194. These remain training-overlap
diagnostics, not untouched generalization. The initial attempt stopped before
model construction because a validator incorrectly required exactly30 stored
trials; the one corrected prediction run explicitly used `[0:30]` of the33
stored trials. The failed attempt produced no prediction/result artifact.

The previous within-post30 `.796/.696` and ext4 `.3875/.2292` comparisons are
different surfaces and are retained, not substituted for this negative result.
At this snapshot, the new pair has FLAT epoch20 EMA `.22279088` and ROUTE
epoch19 EMA `.17378421`; neither constitutes a SPINT quality pass. The fixed
24-epoch schedule continues without a data-dependent recipe change.

The three fresh same-SPINT-container **one-thread** long runs completed, each
with128 warmup and2048 timed whole-B7 host-to-native-NumPy calls:

| P95, ms, t1 | Run1 | Run2 | Run3 |
| --- | ---: | ---: | ---: |
| SPINT image-default seed42 | 14.432 | 15.153 | 14.538 |
| SPINT declared seed44/e8 | 14.516 | 14.978 | 14.688 |
| Previous V3 Transformer | 43.669 | 44.395 | 43.801 |
| Linear-conv/grouped-value Transformer | 21.668 | 22.025 | 21.985 |

The paired P95 speedup over V3 is `1.99–2.02×`; remaining ratio to image-default
SPINT is `1.45–1.51×`. All three candidates retain maximum native output error
`5.40167e−8`. This is distinct from the earlier t2 `1.11–1.19×` remaining ratio
and is not an official-service latency measurement. Receipts:
[t1 run1](../results/family_runtime_v1/container_comparison/m2_linear_spint_b7_t1_2048_r1_v1.json),
[t1 run2](../results/family_runtime_v1/container_comparison/m2_linear_spint_b7_t1_2048_r2_v1.json),
[t1 run3](../results/family_runtime_v1/container_comparison/m2_linear_spint_b7_t1_2048_r3_v1.json).

The [cold-start full replay](../results/family_runtime_v1/m2_cold_start_complete_ext4_v1.json)
also passed:10839 public bins,2069 endpoints, all four independent W50 direct
oracles, maximum public error `7.12462e−8`, endpoint/direct error `3.72529e−8`.
Its native prediction archive is byte-identical to the linear candidate's
archive. Thus the repeated reset improvement `173.443→3.743ms` now has a full
implementation-equivalence audit. It is not a new learned quality result.

A new repository-backed [Falcon adapter](../src/family_runtime_v1/m2_falcon.py)
wraps that backend, with real NWB/competition tag mapping, active/padded batch
handling, continual observe/on_done semantics and batch-size invalidation.
Code inspection found that Falcon bins spikes as uint8; the adapter explicitly
converts real numeric inputs to contiguous float32 as the frozen wrapper does.
Two lifecycle tests cover uint8/float64/float32 inputs, padding, observation
gaps, non-contiguous input conversion and rejected inputs without state advance.
No existing image or submission entrypoint has been replaced. Full final-adapter
container latency acceptance remains pending.

H1's formal window is **W700**, unchanged from the frozen scorer/model; a
resource proxy with96 source windows meant96 endpoints, not a W96 architecture.
Both formal epoch1 EMA selection scores are approximately `−4.91747`; ROUTE
epoch2 is now `−.60386696`. The small fixed208 capacity gate passing did not
establish full validation quality, and neither formal score yet matches C2.
The two arms continue their predeclared12-epoch schedule. M1 has completed
epoch10 (EMA equal-session F `.80626736`, R `.80716627`) and is in epoch11.
Strict selected/endpoint export-and-replay finalizers for M1/M2 are being
implemented and reviewed before their training completes; no early pick or
claim that all three quality gates are met is made.

## Update — 2026-09-06 04:13 HKT

The final local M2 Falcon adapter completed one long run at each thread
setting in the same frozen Original-SPINT container. Each run used 128 warmup
and 2,048 timed real-source, whole-B7 calls, with no labels or fitting.

| P95, ms | One thread | Two threads |
| --- | ---: | ---: |
| SPINT image-default seed42 | 13.327 | 9.756 |
| SPINT declared seed44/e8 | 13.340 | 9.887 |
| Previous V3 Transformer backend | 42.143 | 23.473 |
| Final local Falcon adapter | 21.368 | 12.148 |
| Final / image-default SPINT | 1.603× | 1.245× |
| V3 / final speedup | 1.972× | 1.932× |

Both final-adapter runs have maximum native error `5.40167e−8`, owning
contiguous NumPy outputs, and unchanged code/input hashes before and after.
Reset after construction was 6.649 ms at t1 and 4.514 ms at t2; constructor
timers explicitly exclude module imports. These are paired local measurements,
not official-service results. The final adapter's ratios must not be replaced
by the smaller ratios from earlier six-engine runs: the current SPINT reference
was faster while the candidate remained near its earlier absolute latency.
No image, frozen payload, registry entry, or submission was changed.

Receipts: [final t1](../results/family_runtime_v1/container_comparison/m2_final_falcon_b7_t1_2048_v1.json),
[final t2](../results/family_runtime_v1/container_comparison/m2_final_falcon_b7_t2_2048_v1.json).

The M2 calibration-bank byte audit found exact equality between historical e8,
source-train, and source-minival E0/T/mask banks in all seven sessions. A second
diagnostic used only the retained 1,011-query archive, with no new predictions
or fitting. e8's pooled per-output correlations are `.671/.494`, versus
Original-SPINT `.841/.801`; residual variance, not global mean bias, dominates
e8's MSE. This rules out the measured bank mismatch and simple global mean-bias
explanations, but does not identify a causal mechanism or establish untouched
generalization. Exact labelled-query training exposure is being audited
separately from same-session exposure.

Receipts: [bank bytes](../results/m2/family_v1/e8_bank_byte_diagnostic_v1.json),
[archive-only residuals](../results/m2/family_v1/e8_source_minival_residual_diagnostic_v1.json).

H1's actual W700 common-family runtime now has eight passing CPU tests,
including FLAT/ROUTE, nonzero routing, exact left-four/newest frontend repair,
bank/model mutation invalidation, and a 703-bin stream crossing the full window.
Independent native full-model oracles cover startup and both sides of the
W700 rollover. This verifies implementation behavior on initialized models;
it is not trained-checkpoint quality or accepted deployment latency.

The directly comparable historical C2 selection result also exists: on the
same 2,908 endpoints its pooled R² is `.87156790` (equal-session `.86438766`).
The same C2 checkpoint's 20,325-bin complete result remains pooled `.88849890`.
Current H1 selected/epoch12 outputs still need their final exact query/target
identity check before numerical comparison. C2 remains a separate SPINT
baseline, not a member of the common Transformer family.

At this snapshot all three formal jobs are live: M2 has completed FLAT e22 /
ROUTE e21; M1's latest completed paired epoch is e11; H1 is training FLAT e3 /
ROUTE e4. Their fixed schedules, EMA-only selection rules and source boundaries
remain unchanged. Independently tested post-completion finalizers for M1/M2
will refuse incomplete histories and strictly export/reload both selected and
fixed-endpoint EMA checkpoints. No early model pick has been made.

Exposure clarification (04:14 HKT): the current M2 pair's gradient-training
queries come only from held-in-calib after the support boundary; held-in-minival
query arrays are separate and used for epoch selection, not gradient updates.
The historical Original-SPINT checkpoint has no materialized exact training
window manifest, so exposure to the specific 1,011 labels is unknown. e8's
equal calibration banks do not establish its exact training-label split either.
Earlier broad wording about "training overlap" should therefore be read as a
caution about source/selection exposure, not proof that both models trained on
these particular labels. All numerical results stand; this is not a controlled
same-training-recipe comparison or an untouched generalization result.

## Update — 2026-09-06 04:36 HKT

The initialized H1 W700 runtime now has an executed synthetic B1/t2 profile:
native full-window mean 643.820 ms / P95 645.542 ms, versus streaming public
mean 19.894 ms / P95 20.534 ms. Reset and first public call are measured
separately (2.254 / 21.817 ms). The maximum native error is `1.49012e−8`.
This uses fresh seed42 common-family weights and synthetic counts/banks, not
a trained checkpoint, source/minival data, C2 timing, or official latency.
The steady cache was explicitly seeded outside timers; lifecycle correctness
comes from the separate 703-bin test, not from this short eight-pair profile.
The first two agent drafts were rejected before execution for incorrect API/
history handling; the executed root-reviewed script rolls an independent raw
oracle and times reset, first call and steady calls at separate boundaries.
Receipt: [initialized W700 profile](../results/decoder_validation_v2/20260905_190000/h1/family_v1/h1_causal_initialized_profile_v1.json).

For the *newly trained* M2 family, a separate
[actual-family causal runtime](../src/family_runtime_v1/m2_family_causal.py)
now handles explicit FLAT/ROUTE attention, including nonzero route bias, rather
than assuming the old e8 `frontend.mha` parameter layout. The underlying
training blocks return `(hidden, cache)`, unlike the historical serialized
deployment blocks; the runtime uses that actual API. Eleven CPU tests pass
(8.79 s), including both spatial and full public-stream equivalence, W50
rollover, bank/model/PE/gate mutation, inactive padding and final-query-only
evaluation. These remain initialized-weight checks; the selected trained
exports require their post-completion full source replay before acceptance.

An archive-only M2 diagnostic further localizes part of the old e8 discrepancy.
All seven minival streams are short: there are 343 startup queries with fewer
than 50 observed bins (`start < 49`) and 668 full-history queries.

| Descriptive pooled R² | Startup, 343 | Full history, 668 |
| --- | ---: | ---: |
| Historical e8 | −.294234 | .397407 |
| Original-SPINT M30 | .673042 | .636369 |

e8's MSE is `1.53377e−4` on startup and `6.635e−5` with full history; SPINT's
is `3.8747e−5` and `4.0039e−5`. Startup is therefore a concrete diagnostic
weakness, but it is not the entire gap: e8 also trails on full-history queries.
The partitions have different targets/variances, so this is not a causal
ablation and does not change the all-1,011 primary surface or its selection
rule. No new model forward, fitting, or query replay was used for this analysis.
Receipt: [retained-archive startup diagnostic](../results/m2/family_v1/e8_source_minival_startup_diagnostic_v1.json).

## Update — 2026-09-06 05:08 HKT

M2 completed the original fixed 24-epoch paired schedule. Independent final
selection reconstruction chose FLAT epoch2 and ROUTE epoch6 by the frozen
all-1,011 equal-session EMA rule. Both selected and epoch24 checkpoints were
strictly exported to plain EMA states, reloaded and replayed on the exact same
seven-session query/start/target surface:

| Strict EMA export | Equal-session R², primary | Pooled R², secondary |
| --- | ---: | ---: |
| FLAT selected e2 | .24442853 | .24428345 |
| FLAT fixed e24 | .24357286 | .28818763 |
| ROUTE selected e6 | .28045316 | .33532798 |
| ROUTE fixed e24 | .19154733 | .24246403 |
| Historical Original-SPINT M30 | .64101315 | .65731178 |

The maximum recorded-versus-reloaded EMA primary-score difference is
`8.825e-8`. This improves on historical e8 on this surface but **does not pass
SPINT quality non-inferiority**. The earlier training-exposure qualification
still applies. Receipt: [completed strict finalizer](../results/m2/family_v1/finalize_pair_v1/receipt.json).

Snapshot audit: the two epoch1 JSON snapshots already existed, but their
separate SHA sidecars had not been written; these were sealed after completion.
The last ROUTE epoch24 snapshot was copied once from its unchanged terminal
metric after confirming the original training PID had exited. No existing
metric JSON, checkpoint or selection was changed. The
[completion addendum](../results/m2/family_v1/paired_source_only_v1/snapshot_completion_audit_v1.json)
explicitly labels the late checksum seals: a checksum computed now is not proof
of historical immutability. The finalizer binds this addendum and verifies the
original pre-update source-training inputs, sampler and model-code hashes.

The actual selected family runtimes then passed complete B1 source replay:
1,011 public predictions and 1,011 independent full-window predictions per arm,
with independent raw-window equality at every endpoint. Maximum native errors
versus the strict finalizer archives are `5.402e-8` (FLAT) and `6.557e-8` (ROUTE).
Elapsed time was 98.38 seconds. This proves implementation equivalence for these
actual trained exports, not new generalization or a different quality score.
Receipt: [actual-family complete replay](../results/family_runtime_v1/m2_family_source_complete_v1/receipt.json).

Fresh same-frozen-SPINT-container long measurements now compare the **actual
selected FLAT/ROUTE models**, not the old e8 runtime. Both thread settings use
B7, 128 warmups and 2,048 steady calls with rotating engine order, whole public
call timing, the same source-only raw stream, CPU0–3 and no GPU. This host is
not globally exclusive, and these are not official-server latency results.

| P95 whole public call | One thread | Two threads |
| --- | ---: | ---: |
| SPINT image-default seed42 | 13.174 ms | 10.528 ms |
| SPINT declared seed44/e8 | 13.534 ms | 10.373 ms |
| Actual FLAT selected e2 | 22.116 ms | 12.841 ms |
| Actual ROUTE selected e6 | 22.024 ms | 12.822 ms |
| FLAT / image-default | 1.679× | 1.220× |
| ROUTE / image-default | 1.672× | 1.218× |

Native subset errors were at most `2.399e-8`; independent raw-history checks
and both SPINT wrapper/internal-output checks passed. Constructors, resets and
first public calls are separate: for example, t2 constructors are 63.89/53.83 ms
for FLAT/ROUTE versus 13.80 ms for image-default SPINT; reset is 3.10/4.55 ms
versus .376 ms. Family construction includes its initial reset, followed by a
separately measured reset. The favorable t2 ratio must not replace the t1
result for a one-thread deployment. Receipts:
[t1](../results/family_runtime_v1/container_comparison/m2_actual_family_spint_b7_t1_2048_v1.json),
[t2](../results/family_runtime_v1/container_comparison/m2_actual_family_spint_b7_t2_2048_v1.json).

An archive-only partition of the finalized outputs localized a potential
training-history issue. FLAT e24 pooled R² is `.60101` on the 668 full-history
queries but `-.32390` on the 343 cold-start queries; SPINT is `.63637` / `.67304`
respectively. Different target variances prevent interpreting these partitions
as a causal experiment. All-1,011 equal-session R² remains primary.
Receipt: [finalized startup partition](../results/m2/family_v1/finalized_startup_diagnostic_v1.json).

One post-hoc, fixed two-epoch 2×2 diagnostic was therefore launched at
04:57 HKT (actual PID1722357, physical GPU0, CPU12–15). Each FLAT/ROUTE arm
starts from its fixed e24 EMA and is copied byte-identically into CONTROL and
PREFIX cells. Both receive the same extra source-training batches, unit masks,
targets, fresh AdamW at constant `1e-5` and fresh `.9995` EMA. Only PREFIX
randomly zeroes missing left history in half the rows (observed length1–49);
the current bin and target are unchanged. Both epochs/all four RAW+EMA outputs
are reported; the fixed epoch2 EMA prefix-minus-control effect is primary,
with no automatic selection or promotion. The disposable eight-batch smoke
estimated 37.16 minutes train-only, peak allocation 2.00 GB; the run has a hard
one-hour bound. At 05:04 HKT it had completed 1,201 updates per cell with finite
losses. This is explicitly a source-minival-motivated diagnostic, not a pristine
confirmation study. [Frozen launch authority](../results/m2/family_v1/cold_history_2x2_prospective_v1/launch_authorization_v1.json).

H1 and M1 remain on their unchanged fixed schedules. H1 FLAT has completed e6
(2,908-window selection pooled R² `.3752616`) and ROUTE e5 (`.2225096`), with
monotonically declining epoch losses and finite scores. M1 has completed e14;
its best-to-date EMA values remain e6 FLAT `.81138314`, ROUTE `.81181830`, not
yet final picks. Neither live trajectory is a completed quality claim.

## Update — 2026-09-06 05:32 HKT

A separately authorized, once-frozen **original selected** M2 ext4 comparison
completed. Only FLAT e2 and ROUTE e6 were eligible; no cold-phase model or
alternate epoch was evaluated or reselected. Existing support33 banks, strict
post-support window starts/targets and all cache/provenance files were bound
before and after. GPU0 native B8 scoring took 9.08 seconds with peak allocation
256.98 MB; batched/singleton maximum discrepancy was `1.188e-8`.

| Fixed local ext4, 2,069 windows | FLAT e2 | ROUTE e6 | Historical SPINT | Historical e8 |
| --- | ---: | ---: | ---: | ---: |
| Pooled native R² | .23054801 | .35783687 | .22919761 | .38750524 |
| Equal-four-session R² | .24995125 | .35753978 | .23759323 | .38441527 |

ROUTE is above SPINT on all four sessions; its pooled delta is `+.12863927`.
FLAT is only slightly higher overall and is worse on the 11-18 session.
Neither exceeds historical e8 overall. The source-minival negative gap is
unchanged and remains reported. Ext4 has historical development exposure and
is not an untouched confirmation surface; no formal non-inferiority conclusion
or automatic promotion follows from these point estimates.
Receipt: [fixed ext4 native comparison](../results/m2/family_v1/frozen24_selected_ext4_native_v1/receipt.json).

The same selected models then completed chronological public-runtime replay:
each advanced 10,839 real input bins and matched the new native archive on
all2,069 endpoints, with independent W50 raw-window equality at every endpoint.
Maximum errors are FLAT `4.191e-8` / ROUTE `5.961e-8`; elapsed proof time was
63.50 seconds. Runtime pooled R² is `.23054806` / `.35783709`, matching the
native result within the fixed `1e-5` score tolerance. This is a full ext4
implementation proof, not a new selection or a latency benchmark.
[Complete ext4 proof](../results/family_runtime_v1/m2_frozen24_ext4_complete_v1/receipt.json).

The corrected instrumented T1 profile now counts exactly64 global B7 warmup
and128 timed calls, with the three temporal-block durations **summed per call**.
Mean FLAT/ROUTE costs are: full public21.280/21.712 ms, spatial repair10.095/
10.378 ms, first-three temporal blocks9.616/9.660 ms, final temporal remainder
1.119/1.151 ms, audit.338/.395 ms. Direct full-native checks at0/49/50/191 pass
with maximum error `2.399e-8`. It is instrumented source-only attribution, not
the same-container SPINT benchmark. The preceding v2 attribution is retained
but rejected for incorrect global count and temporal accounting; none of its
numbers are used here. [Corrected profile](../results/m2/family_runtime_v1/m2_actual_family_profile_t1_v3.json).

Finally, the [as-executed finite-variant ledger](AS_EXECUTED_CRST_VARIANTS_20260906.md)
explicitly records H1's additional shared logit multiplier √32 and init-only
local FC1 balance √45, alongside M1 QueryAge16 and M2/H1 CausalPE4. These are
not retroactively hidden inside the original strict shared-operator freeze.
The 2×2 cold-prefix diagnostic has completed epoch1 and is finishing its fixed
epoch2; no endpoint-2 result or selection is asserted in this snapshot.

## Update — 2026-09-06 05:39 HKT

The fixed cold-prefix 2×2 diagnostic finished normally in 2,282.04 seconds
(38.03 minutes); its actual training PID exited and GPU0 is now used only by
the unchanged H1 worker. All6330 updates per cell completed. Four endpoint2
EMA exports were strictly reloaded and reproduced their native1,011-query
archives. Every RAW/EMA archive from both epochs, both epoch metric files,
eight checkpoints and four final exported/replayed endpoints passed the
separate archive-only identity/hash/score checks.

| Fixed phase epoch2 EMA, all1,011 primary | CONTROL | PREFIX | PREFIX−CONTROL |
| --- | ---: | ---: | ---: |
| FLAT equal-session R² | .23548902 | .21678979 | −.01869922 |
| ROUTE equal-session R² | .18454399 | .25124761 | +.06670362 |

The ROUTE treatment is positive against its matched extra-training control on
five of seven sessions, but its absolute score remains below original selected
ROUTE e6 `.28045316`. FLAT treatment is negative on the primary measure. No
model is selected or promoted from this phase, no extra epochs are appended,
and its checkpoints are not evaluated on ext4.

The partition analysis helps explain the asymmetric effect: ROUTE's startup
pooled R² improves from `−.211619` to `−.074273`, whereas full-history pooled
R² changes only `.450612→.460076`. FLAT's startup remains poor
(`−.317493→−.331881`) while full-history is `.589461→.598641`. These are
descriptive partitions, not replacement primary outcomes. The hypothesis has
partial support for ROUTE but does not solve the source-minival quality gap.
Receipts: [completed phase](../results/m2/family_v1/cold_history_2x2_phase_v1/report.json),
[all-archive diagnostic](../results/m2/family_v1/cold_history_2x2_archive_diagnostic_v1.json).

The post-finalization H1 proof runner is now implemented and CPU-fixture tested,
but has not been run on an unfinished model. It validates all24 immutable
per-arm epoch/checkpoint records, independent earliest selection, selected and
epoch12 export/archive hashes, source/target/session identity and a fresh full
post-replay closure. It then advances every real bin using actual selected
FLAT/ROUTE CPU runtimes, checks independent W700 state, compares all20325
native archive endpoints, and checks direct full forwards at startup/rollover/
last-session-bin subsets. A read-only cache metadata audit found20,920 public
bins per arm, including595 unscored bins which must not be skipped. At the
initialized20ms profile, public calls alone estimate13.87 minutes for two arms;
direct oracle/setup/provenance costs are additional. No trained-H1 quality or
runtime-equivalence pass is asserted before actual finalization and replay.

## Update — 2026-09-06 06:00 HKT

All six **actual selected M2 family** same-container long comparisons finished:
three fresh T1 and three fresh T2 runs, each B7 with a separate first call,
128 warmup calls and 2,048 timed public predictions. Each run rotates the four
engines' call order and checks original-wrapper/full-model output equality and
the selected family against its native full-window oracle. They use the same
frozen historical SPINT image and selected FLAT e2 / ROUTE e6 authority.

| P95 milliseconds, median [min, max] across three runs | T1 | T2 |
| --- | ---: | ---: |
| SPINT image-default | 13.959 [13.174, 14.188] | 10.504 [10.033, 10.528] |
| SPINT declared payload | 13.897 [13.534, 14.333] | 10.373 [10.114, 10.475] |
| Actual selected FLAT | 22.360 [22.116, 22.613] | 12.919 [12.841, 13.408] |
| Actual selected ROUTE | 22.245 [22.024, 22.372] | 12.844 [12.822, 13.460] |
| FLAT / default, paired ratio median [min, max] | 1.602 [1.594, 1.679] | 1.276 [1.220, 1.288] |
| ROUTE / default, paired ratio median [min, max] | 1.594 [1.577, 1.672] | 1.280 [1.218, 1.281] |

Ratios are calculated within each repetition and then summarized; they are
not ratios of independently selected minima. The corresponding median ratios
against the declared SPINT payload are T1 FLAT/ROUTE 1.609/1.601 and T2
1.277/1.270. T1 and T2 are not pooled. The corrected archive-only v2 summary
binds all six inputs and its own source before/after, records each repetition,
checks nonnegative oracle errors, and refuses overwrite. Original per-run
receipts retain summary statistics rather than raw timings: the aggregator
cannot independently recompute their sample counts or percentiles and states
this limitation. The benchmark runners themselves assert exact counts.

These are shared-host CPU public-predict measurements, **not official
end-to-end latency or a cold-start cost comparison**. Constructor, second
reset and first-call costs remain separately reported in every input receipt.
The current local actual-family gap is approximately 1.2–1.7×, not the earlier
unbound 10× deployment estimate; the two statements have different evidence
and must not be equated. No quality conclusion changes from these timings.
The retained v1 aggregate lacked a complete pre/post hash seal; use
[strengthened long-run summary v2](../results/family_runtime_v1/m2_actual_family_long_summary_v2.json),
SHA256 `65331a561de13f078bc789d3d3110d76a8c3baa72ca4adda41c8cc57cf9953d6`.

The future selected-P1 M1 proof runner now exists and passes 15 CPU tests,
including real initialized FLAT/ROUTE models with synthetic banks and sparse
scored endpoints. It does not load the running P1 checkpoints. The real proof
is gated on completed24 training plus independent finalization; it binds all48
epoch checkpoints, four selected/endpoint exports and native archives, source
file/bank/code closure, true physical unit rosters, exact source targets and
31,252 endpoint identities. Each selected arm must make112,985 public gap/score
calls plus three explicitly primed initial predictions. Raw W100 equality is
checked at every advance, including unscored gaps.

The original M1 comparison image is independently identified as
`sha256:f5af9eb29b7f86616d898261070193b1b0777db75567848c62d7f888ce3d76cd`,
whose as-shipped `/data/decoder.pkl` SHA256 is
`052e9eab7be2af8bacd5298e348d414cce60dad767d6a0880b58dd39b27279f6`.
Read-only trusted payload inspection shows W100, native divisor1, unsmoothed
neural, and7 date-keyed M10 calibration arrays of shape10×1024×64. The wrapper
is the image's original `SpintDecoder`, not a cached teacher facade; it decodes
per lane with its original calibration computation. Its checkpoint differs
from the source-fold teacher used in the M1 quality-overlap table. A new actual
P1-vs-original public benchmark is prepared but has not been run or passed.
It will use B4 with three distinct source banks and an explicitly repeated
first bank on a separate neural segment, not pretend to have four source banks.
No image build/push, official evaluation or hidden-query access occurred.

## Update — 2026-09-06 06:12 HKT

H1 FLAT completed all12 epochs and its worker exited normally. The independent
worker rule identifies e12 as its highest2,908-endpoint EMA value `.59258104`;
this is **not yet the paired supervisor freeze or complete20,325-bin score**.
ROUTE is still training epoch11, and the supervisor correctly waits for it.
M1 has completed e18 and entered e19; its final24-epoch choice remains pending.

A C2 deployment-provenance audit corrected an important possible
misinterpretation. The generic EP-FILM wrapper supports M3-fitted MAT7 maps,
but the actual immutable C2 epoch15 package does **not** use fitted readout
correction. Root directly CPU-loaded the trusted package without constructing
or forwarding a model, verified package SHA256 before/after as
`91ef13cc94b9ab865c8f926dcbb6d33e9628bf9cbc9b757665d10e33cfda144a`, and
checked all27 session maps: every weight is the7×7 identity, both means and
intercepts are zero, and both scales are one. The package explicitly marks
`readout_selection_sha256 = IDENTITY_READOUT_NO_FIT_V1`; its builder also
zeroes the final FiLM weights/biases. Thus its `.88849890` reference score
must not be explained by a non-identity M3 affine correction.
See [C2 builder](../submissions/evalai_h1_c2_ho_epoch15_v1/build_payload.py).

The C2 historical epoch was selected using visible held-out development
recordings; an exact historical training-example overlap audit is not
materialized locally. Its fixed20,325-bin result is still a useful descriptive
same-input reference once the new models' complete20,325-bin scores exist.
It is not an interchangeable number with their2,908-endpoint live selection
scores, nor a controlled architecture-only causal comparison.

No affine readout fit or new calibration variant has been executed. A future
such diagnostic would need an exact M3 endpoint manifest: the current source
cache preserves the first3 trial IDs and boundary, but not `TrialNum` or exact
support endpoint indices. The cached prefix has30,879 valid bins across13
sessions; matching a historical count alone is not an endpoint identity proof.
The initial proposed rationale that C2 already uses fitted MAT7 was withdrawn
after inspecting the actual payload; any later affine diagnostic would be a
new hypothesis and explicitly separate variant, not baseline feature parity.

For SPINT-relative timing, the genuine original H1 image is now bound:
`sha256:f719c4228c345f9a1d6aa7c1e10d63ad7b9aa1f551dc95272b0b7a612d61fac6`,
default `/data/decoder.pkl` SHA256
`20a1d41a1d82a8037579caa2e4454f56817e02f021dec2132798c7fc57849298`.
Trusted metadata inspection confirms W700, native divisor20, unsmoothed
inputs, all112 modules in eval mode, and PyTorch2.5.1.post303. Its as-shipped
first-session calibration shape is2×1024×176, while the new family's bank
derives from exactlyM3. This support-budget difference must remain visible;
the frozen original is an engineering deployment reference, not a matched
support-budget quality ablation. The C2 image has a different `.pt` payload
and contains no original `.pkl`; these two references are not interchangeable.

## Update — 2026-09-06 06:28 HKT

The genuine as-shipped original H1 reference completed in485.21 seconds inside
the immutable `f719c422…` image. It replayed all20,920 chronological public bins
across13 sessions and archived exactly20,325 scored endpoints. Independent
W700 histories matched at every public call; all65 fixed-subset native forwards
had maximum absolute error0. Image/payload/code/cache/authority hashes matched
before and after. No parameter update, calibration fit or model selection was
performed. This run measures reference quality, **not latency**.

| Frozen original H1, same20,325 scored endpoints | R² |
| --- | ---: |
| Pooled, float64 recomputation | .9607844357 |
| Equal-session mean | .9604391953 |
| Worst session (`ses-19250113T120811`) | .9531916119 |

This is materially higher than the separately frozen C2 epoch15 reference
(`.8884989023` pooled). Neither number substitutes for the other. The original
package's as-shipped two-trial calibration and unknown exact historical
training/selection exposure differ from the fresh family’s M3 source protocol;
these are descriptive fixed deployment references, not controlled
architecture-only comparisons or a formal non-inferiority test. The new H1
pair still needs its completed20,325-endpoint exports before a same-surface
quality statement is possible. ROUTE has entered epoch12; FLAT completed12.
M1 has completed19 of its fixed24 paired epochs.

Evidence: [original H1 reference receipt](../results/family_runtime_v1/original_h1_frozen_same20325_v1/receipt.json),
SHA256 `539bfa832c650df51e22be14b5ee0dba0c1f49fbb595f9fb869328a601f30c48`;
native FP64 archive SHA256
`f1bb6739415ea66bb86bf4285035f131333e350072ae364139b81488912ba81c`.
Only the four newly generated reference files' read permissions were changed
after the container exited, so the host can verify their contents; their
contents and hashes were not rewritten.

## Update — 2026-09-06 06:52 HKT

H1 completed all12 epochs per arm and its supervisor froze both selected
epochs at12. The immutable2,908-endpoint EMA selection values are FLAT
`.5925810377264197` and ROUTE `.5997886001423203`. Strict full-stream
finalization remains in progress for ROUTE; no paired20,325-bin acceptance
or selected runtime proof is declared yet. M1 has entered its21st paired epoch.

A separate M2 static-affine runtime experiment has now been measured with the
actual fixed FLAT e2 / ROUTE e6 weights. It precomposes the frozen frontend
`E0_projection + T_projection + bias` at reset, leaving dynamic layers,
weights, temporal computation and selection unchanged. It is algebraically
equivalent over real numbers, **not bit-identical FP32**, and remains isolated
in an opt-in experimental file. The existing selected runtime is unchanged.

Each long run used2,048 steady public B7 calls after128 warmup calls and a
separate first call, on CPU12–15 inside the immutable M2 SPINT image, with
four-engine order rotated each bin. All raw timing samples are archived.
These runs compare base/static family implementations only: they do not run
Original-SPINT, score labels, or update the earlier three-repetition
SPINT-relative latency estimate.

| Single long run | FLAT base/static p95, ms | ROUTE base/static p95, ms | Static/base p95 ratio, F/R |
| --- | ---: | ---: | ---: |
| T1 | 24.1273 / 24.1338 | 24.1841 / 24.3305 | 1.00027 / 1.00605 |
| T2 | 14.5194 / 14.1772 | 14.5909 / 14.1683 | .97643 / .97104 |

T1 shows no p95 improvement. The one T2 repetition shows approximately2.4–2.9%
lower p95, whereas its earlier32-call smoke had been slower. This is not
evidence of a robust material improvement across settings. Static/base paired
per-call ratios are reported separately from ratios of the two p95 values;
they are not the same statistic. This candidate is **not promoted**, and no
new full source/ext4 equivalence proof for it is claimed.

Each long run checked independent W50 histories, public outputs and retained
frontend parity at all2,177 bins; five independent native W50 forwards per
arm included startup and rollover. Maximum native output error across both
runs was below`2.43e-8`. The frontend tolerance is `atol=rtol=1e-5` (not an
absolute-only bound); its largest observed absolute difference was`1.526e-5`.
The additional precomposed tensor costs688,128 bytes per B7 runtime and is
counted exactly once in the `derived_static` storage category.

Evidence: [T1 long receipt](../results/family_runtime_v1/m2_static_affine_t1_long_v1.json),
SHA256 `258476f7d8946d117531e1f875a2b200ccb9e0ab7565b24343c942118b9479fe`;
[T2 long receipt](../results/family_runtime_v1/m2_static_affine_t2_long_v1.json),
SHA256 `2e1e8679307bb0be30e2f615e8fb61ed4c511c6fbbafb43e799526cf354f3095`.

## Update — 2026-09-06 07:00 HKT

H1's supervisor exited normally and wrote `COMPLETE`. Root independently
audited all24 immutable epoch records/checkpoints, the common freeze, both
strict finalizer receipts and all four complete exports. Both selected models
are epoch12. Their full-stream quality is **negative against both fixed
deployment references**, not a pass for the three-task objective:

| Same20,325 endpoints | Pooled R² | Equal-session R² | Worst-session R² |
| --- | ---: | ---: | ---: |
| FLAT selected e12 | .4791493714 | .4658195133 | .1494732712 |
| ROUTE selected e12 | .4199294230 | .4054386127 | .0431994880 |
| Frozen original image | .9607844357 | .9604391953 | .9531916119 |
| Historical C2 e15 aggregate | .8884989023 | .8879631849 | .8669828176 |

The archive-only comparison recomputed both family and original metrics,
required exact target/session/endpoint equality across the five FP64 archives,
and rechecked every bound input after analysis. C2 has no compatible prediction
NPZ: its rows remain imported historical aggregates bound to the same source
authority, not independently recomputed predictions. Historical training and
selection exposure is not equalized by this comparison. No formal NI claim
is made. [Complete comparison](../results/family_runtime_v1/h1_frozen_same20325_quality_v1.json),
SHA256 `324c0db5a27afe0a26d51c636b5662f7bd2e7ddb41221da1cb546e40675ecef4`.

Two fixed diagnostics now distinguish parts of the failure:

| H1 diagnostic surface | Count | FLAT selected | ROUTE selected | Original frozen |
| --- | ---: | ---: | ---: | ---: |
| Pre-existing source-train208 | 208 | .9231691081 | .9284778146 | Not evaluated here |
| Minival, endpoint `<699` | 8,702 | .2612928279 | .0887784919 | .9562372150 |
| Minival, endpoint `>=699` | 11,623 | .5950477322 | .5997874767 | .9627096510 |

The first row reuses the exact13×16 source endpoint manifest frozen before
the formal run; it is not a newly selected or resampled subset. Both current
selected EMAs were strictly loaded without optimization. Their prediction
standard deviations `.0050365/.0051257` are close to target `.0052265`.
This rules out collapse on **that fixed source subset**, not on every23,212
training endpoint. It does not measure raw-versus-EMA lag. The historical
1040-update source208 preflight used different training conditions and is
not an acceptance threshold.

The second/third rows partition the existing full prediction archives at the
fixed W700 history boundary. Short-history degradation is especially large
for ROUTE and explains the reversal from its slightly higher full-history
score to its lower complete-stream score. Full-history minival is still far
below both source208 and the fixed original reference, so fixing startup
coverage alone cannot be assumed to fix all generalization error. R² was
recomputed using each group's own target mean/SST; group R² values are not
averaged to obtain the full-stream number. Split SSE recombination is checked.

Source208 [receipt](../results/family_runtime_v1/h1_selected_source208_diagnostic_v1/receipt.json),
SHA256 `f3ac9a999fef786ca435e0ff6eefcf679b82e3b3439a535cd66ae6e65c80e614`;
fixed cold/full [archive analysis](../results/family_runtime_v1/h1_selected_cold_segments_v1.json),
SHA256 `379094dab4623444d402a159b625382bc61c320bf399a29d3e07065cfcd7e460`.

The complete selected-family CPU public-stream proof is currently running on
CPU0–3; it must finish before an actual selected H1-vs-SPINT timing benchmark.
A separate matched continued-training/control versus prefix-history
augmentation experiment is being prepared, with no new network structure.
It has **not been authorized or launched**; the original12-epoch selections
and negative results remain unchanged.

M1 clarification for upcoming image-reference work: both the current source
bank protocol and the as-shipped image use M10 context. `rSyn3-refit-v1` names a
carrier revision, not a three-trial budget. The new original-M1 scorer has
been corrected before execution; its historical training/calibration methods
are still not assumed equal to the fresh family experiment. It remains gated
on completed P1 finalization and its full selected-runtime proof.

## Update — 2026-09-06 07:14 HKT

The actual selected H1 public-runtime proof completed in903.07 seconds on
CPU0–3/T2. Each arm consumed all20,920 chronological input bins and matched
all20,325 frozen native scoring endpoints. Independent W700 histories were
checked at every public call. Maximum archive errors were `5.885e-8` FLAT and
`4.750e-8` ROUTE; fixed-subset independent full-window forward errors were
`1.118e-8` / `1.490e-8`. Fresh artifact/source closures matched before and
after. This is implementation equivalence only: the previously reported
quality failure remains unchanged and is not explained by the streaming
implementation. [Completed proof](../results/family_runtime_v1/h1_selected_family_source_complete_v1/receipt.json),
SHA256 `974be98cc86541e75839d7e14d8cdceffbdbd412d3ac4aff2f8559f4d53e29cc`.

Two disposable source-only cold-history resource smokes also completed, each
on GPU0/CPU12–15/T1, while the original M1 training remained untouched on GPU1.
Each smoke cloned one actual selected e12 EMA into CONTROL/PREFIX, then ran
the exact first32 original epoch1 batches with real MICRO8 accumulation and
one optimizer/EMA update per condition/batch. The sampler includes one
16-row remainder among those32 batches; it was retained, not padded or dropped.
Both arms have the identical sampler/dropout/prefix identity SHA
`a89ec5ad0f3de66a4475870f2b1b6b9326152de00ff3be17ca509d714b609c29`.

| Disposable source-only resource smoke | FLAT | ROUTE |
| --- | ---: | ---: |
| Elapsed training and post-audit, seconds |45.32|47.77|
| Peak allocated GPU memory, bytes |9,486,544,896|9,495,674,368|
| Remaining28 paired-update P95, seconds |1.39374|1.47113|
| Two-GPU conservative total forecast, seconds |7,556.48|7,726.19|

Each timing sample already includes both CONTROL and PREFIX updates. The
forecast therefore assumes one arm's pair per GPU, two epochs×731 batches,
adds50% training margin,3,600 seconds for full scoring and900 seconds extra
overhead. The larger estimate is128.77 minutes versus a240-minute hard bound;
this is a resource estimate, not a measured full-run duration or quality
result. Both smokes bound the completed formal, selected exports, their own
code, the diagnostic/cold-analysis receipts, and the full source closure
before and after. They performed no minival/source208 forward, scoring,
checkpoint save, model selection or promotion. The disposable updated models
were discarded. Full phase execution is still pending reviewed evaluation
and checkpoint orchestration plus an external hash-bound authorization.

Evidence: [FLAT smoke](../results/family_runtime_v1/h1_cold_resource_smoke_flat_v1/receipt.json),
SHA256 `d3f87a19b9f24a2b6095897ed3fc08b84e692bade4b4e1c07b2a493e31e7b7da`;
[ROUTE smoke](../results/family_runtime_v1/h1_cold_resource_smoke_route_v1/receipt.json),
SHA256 `5cca9d3b0ba771491b76c6de3c40a778c3fa7ba56ac22a5b3199b4433b3758d2`.

## Update — 2026-09-06 07:30 HKT

M1 P1 completed all24 paired epochs; the original training PID exited normally.
Its independent strict finalizer also completed on GPU1 and reproduced both
selected epoch6 EMAs and the separately reported epoch24 endpoints on all
31,252 frozen source-minival windows:

| M1 strict native export | Equal-session R², primary | Pooled R² |
| --- | ---: | ---: |
| FLAT selected e6 |.8113831405|.8116523260|
| ROUTE selected e6 |.8118183024|.8121964307|
| FLAT fixed e24 |.7982858058|.7987435114|
| ROUTE fixed e24 |.7994863451|.7999847452|

The selected models are above the earlier frozen teacher-overlap reference
(`.5967531154` equal-session) by about`.215`, slightly below the earlier
formal12 current-query reference (`.8122134952`), and about`.0154–.0158`
below Sfix e11 (`.8272239340`). These historical references have source-training
overlap and do not establish a new held-out non-inferiority result. In
particular, the teacher-overlap checkpoint is **not** the as-shipped immutable
Original-M1 image payload: the latter still needs its exact same-surface
replay, after the currently running selected-runtime proof. No outer query
was opened or used for selection.

Evidence: [M1 strict finalizer](../results/decoder_validation_v2/20260905_190000/m1/family_v1/p1_queryage16_pair_chron80_v2/finalized_p1/receipt.json),
SHA256 `3807fd34daa4972d2d1132b918aa5d39efbfd0427c667a64557b396c6e98731e`.
The complete selected M1 public-runtime proof now runs on CPU4–7/T2, independently
of the GPU experiments and H1 CPU0–3 latency measurements.

H1's selected-family first long measurements are now available in the immutable
Original-H1 image. These are whole public B1 calls,128 warmups plus2,048 steady
calls, with three-engine order rotated each bin and W700 histories checked at
every call. The first public call and constructor/reset remain separate.

| H1 first long repetition | Original default P95, ms | FLAT P95, ms | ROUTE P95, ms |
| --- | ---: | ---: | ---: |
| T1 |38.98982|36.22937|36.17497|
| T2 |22.65520|19.80406|19.78384|

This first repetition is about7.1–7.2% lower P95 atT1 and12.6–12.7% lower atT2
than the same-run original. Both native-oracle checks are below`6.52e-9`, and
the original wrapper agrees exactly with its native oracle. These are local
nonexclusive-host measurements, not official-service latency or quality
results. Two additional fresh repetitions at each thread setting are running;
the complete three-repetition summary is still pending. The quality deficit
of these same selected H1 models remains unchanged.

Evidence: [H1 T1 first long](../results/family_runtime_v1/h1_actual_selected_spint_t1_long2048_v1.json),
SHA256 `92b86a1e20b13f3e3fd3f6851980ad3e5bbecdc3fa32ff703d1932c210b56fc4`;
[H1 T2 first long](../results/family_runtime_v1/h1_actual_selected_spint_t2_long2048_v1.json),
SHA256 `4665ef56d103cb0bb25936b67b76949c67497b650c4b0686ffae3b6256805570`.

The separate H1 cold-history2×2 diagnostic is now actually running: FLAT on
GPU0/CPU12–15/T1 and ROUTE on GPU1/CPU8–11/T1. Each architecture owns matched
CONTROL/PREFIX clones of its frozen selected e12 EMA. Training remains exactly
two original731-update source epochs, with identical retained remainder
batches and deterministic unit masks, fresh AdamW at`1e-5`, and fresh`.9995`
EMA. No new network structure was added. Every epoch checkpoint is atomically
written and strictly reloaded from disk, including optimizer/EMA/RNG state;
e1 scores only the pre-existing source208 RAW/EMA surface, while e2 reports all
20,325 endpoints for both RAW/EMA and both conditions. The fixed primary is
e2 EMA pooled R² PREFIX-minus-matched-CONTROL; no mid-run selection or promotion.
Both workers have an explicit4-hour timeout and22GiB peak allocation limit.

Frozen worker code SHA256
`13692ec3f880b00835b91cae359a63469ec498b25a16a8a84166248741539cac`;
protocol SHA256 `6b0590bd7c530f55b2a874bc179a1e2408a4fb2c56ee1983325f420bb025cbb2`.
External root authorizations:
[FLAT](../results/family_runtime_v1/h1_cold_phase_authorization_flat_v1.json),
SHA256 `2adce6a5b75464773962775214a9a9d3c53a853dea9da5746038e70d9efabb33`;
[ROUTE](../results/family_runtime_v1/h1_cold_phase_authorization_route_v1.json),
SHA256 `a6daaef524ef653be78e124335dd38ea965efd052033594b59935ddc81c58e05`.

## Update — 2026-09-06 07:44 HKT

The complete M1 selected-runtime proof passed in641.00 seconds. Each arm
performed112,985 real public calls plus three explicitly primed initial native
predictions, covering all31,252 scored endpoints. Maximum native/archive errors
were`1.788e-6` FLAT and`1.907e-6` ROUTE, below the fixed`1e-5` tolerance; all
raw-history and source/artifact pre/post checks passed. Recomputed equal-session
R² `.8113831354/.8118183005` matches strict finalization. This is implementation
equivalence only. [M1 completed proof](../results/family_runtime_v1/m1_p1_selected_family_source_complete_v1/receipt.json),
SHA256 `f2c614f323cd2995f20d5babdfa7dee49a29e02221af465475e93d8a913f5d2a`.

The genuine immutable Original-M1 image is now replaying the same31,252 endpoints
on CPU4–7/T2, with its as-shipped M10 calibration and all112,985 intervening
public calls. Its first observed4,096 calls took91.39 seconds; this is progress
for a quality replay, **not a latency measurement**. No reference score is yet
available. Separately, the actual selected M1 B4 public benchmark passed its
32-call contract smoke and began three fresh2,048-call repetitions at both
T1/T2 on CPU0–3; no long-run M1 latency acceptance is declared yet.

All six H1 long runs completed. Root recomputed P95 directly from each run's
2,048 stored raw samples and confirmed that each run's source/artifact/image
pre/post receipts match. The summary uses family/original P95 ratios paired
within the same run, not ratios of independently chosen minima:

| H1 three fresh repetitions | T1 median [min,max] | T2 median [min,max] |
| --- | ---: | ---: |
| FLAT / Original default P95 |.92920 [.92551,.93910]|.87306 [.85811,.87415]|
| ROUTE / Original default P95 |.92781 [.92730,.94012]|.87326 [.85950,.87377]|

Thus every measured repetition is lower than its own original reference:
approximately6.1–7.4% FLAT /6.0–7.3% ROUTE atT1, and12.6–14.2% FLAT /
12.6–14.1% ROUTE atT2. These are B1, W700, nonexclusive-host local measurements;
all native subset errors remain below`6.52e-9`, and the original native error
is0. They do not establish improved quality for the negatively scored selected
H1 models or say anything about the still-training history-treatment variants.

Additional repetition evidence:
[T1 r2](../results/family_runtime_v1/h1_actual_selected_spint_t1_long2048_r2_v1.json),
SHA256 `cb3f18536e55927358074bd2b810cdc89259831bc7abe8f0054c0c0c81cea451`;
[T1 r3](../results/family_runtime_v1/h1_actual_selected_spint_t1_long2048_r3_v1.json),
SHA256 `cfe22cb3684c2bea8169590f6b1a6be936a300f4f1a80b3b3547d1293e6d0b42`;
[T2 r2](../results/family_runtime_v1/h1_actual_selected_spint_t2_long2048_r2_v1.json),
SHA256 `b574a0abecc2a6a36472e5c3d4ed6176ee56cf9cef8237e9d47f264ef3b8c7fc`;
[T2 r3](../results/family_runtime_v1/h1_actual_selected_spint_t2_long2048_r3_v1.json),
SHA256 `587efec1b30e41d21e5879085688880a6a144feb66cf40391cb00e0be6a96126`.

A bounded read-only check of the original H1 package found metric labels but
no source checkpoint/manifests or train/validation session assignments that
can be linked to its payload hash. Separately named local epoch49 checkpoints
and the image tag do not prove that linkage. Exact historical exposure therefore
remains unknown: neither leakage nor its absence has been established.

## Update — 2026-09-06 07:59 HKT

M1's first long repetition at each thread count completed in the actual frozen
Original-M1 image, using the same selected P1 states as the completed source
proof. Each measurement is a real whole B4 public call with three-engine order
rotated per bin, 128 warmups and 2,048 stored steady samples:

| M1 first long repetition | Original default P95, ms | FLAT P95, ms | ROUTE P95, ms |
| --- | ---: | ---: | ---: |
| T1 |159.25105|9.48864|9.55463|
| T2 |88.82314|6.79765|6.88708|

These first paired ratios are F/R `.05958/.06000` at T1 and `.07653/.07754`
at T2. They are strong local latency evidence, not yet the three-repetition
summary or a quality acceptance. Both native-oracle errors are `4.768e-7`;
Original native error is zero. The remaining four fresh repetitions continue
on CPU0–3; the Original-M1 all-31,252 quality replay continues independently
on CPU4–7. Evidence: [T1 r1](../results/family_runtime_v1/m1_p1_actual_selected_spint_t1_long2048_r1_v1.json),
[T2 r1](../results/family_runtime_v1/m1_p1_actual_selected_spint_t2_long2048_r1_v1.json).

H1 both cold-phase workers passed first-epoch checkpoint save/strict disk reload,
source208 RAW/EMA scoring, and entered the second fixed epoch. Source208 EMA
R² F CONTROL/PREFIX is `.929841/.927727`, R is `.935539/.933011`; RAW is
`.942805/.930519` and `.949906/.934624`, respectively. These are calibration/source
fit diagnostics only. The full e2 primary remains sealed until completion.

A read-only H1 cache bridge audit ruled out differing stored calibration-bank
bytes across train/minival: all 13 same-session E0 `[176,700]`, T `[176,4]`, and
176-active-unit masks are exactly equal, with zero bank RMS difference. The
cache SHA remained `51ff9ebfcd10a032f9c173ec426bfb4c421b751502271577582239c51bcc91b4`.
Both splits use the one frozen C2 materialized bank mapping; separately owned
cache rows must not be described as different bank content. Raw recording
distributions and query support do differ, but their marginal statistics do
not establish causation. Target alignment is consistently `start+699`; source208
has complete W700 histories, and minival startup is already separately quantified.

The original H1 formal12 immutable ledgers show both source loss and governing
2,908-endpoint EMA minival scores improving through the fixed budget boundary.
FLAT e10/e11/e12 is `.569130/.583375/.592581`; ROUTE is
`.576453/.592865/.599789`. After the epoch1 warmup, LR stayed `1e-4`, for
8,772 updates per arm with `.9995` EMA. RAW minival was not scored per epoch,
so the trajectory cannot identify EMA lag. A fixed endpoint12 RAW diagnostic
is being prepared without changing the original EMA selection; it has not
been launched. Further same-architecture training has optimization-trajectory
support, but is not yet an authorized experiment or proof the original-quality
gap will close.

## Update — 2026-09-06 08:20 HKT

Both H1 cold-history workers completed all fixed updates and all source208 /
complete20325 RAW/EMA evaluations. Elapsed times were 2,724.73 s FLAT and
2,841.96 s ROUTE. The independent archive-only summary rehashed the full
source/artifact/code/checkpoint closures, reproduced all metrics, and required
exact query/target identities across all eight final archives and both original
selected EMA baselines. No selection or promotion occurred.

| H1 fixed e2, all20,325 pooled R² | CONTROL EMA | PREFIX EMA | Primary difference | CONTROL RAW | PREFIX RAW |
| --- | ---: | ---: | ---: | ---: | ---: |
| FLAT |.49972520|.56100285|+.06127765|.51132470|.60711178|
| ROUTE |.42822748|.53439193|+.10616445|.44376253|.57868185|

Both EMA treatments improve all13 sessions versus their matched controls.
The effect is concentrated in short histories: cold-bin pooled differences
are `+.1701971/+.2995094`, again positive in all13 sessions. Full-history
differences are `−.0001365/−.0028985` (positive in only6/4 sessions). This is
evidence for the specific short-history intervention, not a solution to the
remaining full-history gap or non-inferiority versus Original-SPINT `.9607844`.
The higher RAW secondary results do not replace the fixed EMA primary.

Evidence: [archive summary](../results/family_runtime_v1/h1_cold_phase2x2_archive_summary_v1.json),
SHA256 `432ba5c56e55ce1d3440f67bcef398382c28b1bf1e8a18289c2c3835bf42d88c`;
[FLAT receipt](../results/family_runtime_v1/h1_cold_phase2x2_v1/flat/receipt.json),
SHA256 `d42f7fa3c01a4ad530ceff0c59a45fb1e007eff995f78bcf4f6cac7d6ae003fc`;
[ROUTE receipt](../results/family_runtime_v1/h1_cold_phase2x2_v1/route/receipt.json),
SHA256 `365e37c7e13b9c2639723c91839e54c65e1bc8ed4e4b5443b5102faee1656f14`.

The separate original-formal endpoint12 RAW diagnostic now runs on GPU0 with
a40-minute hard bound, using fixed source208 plus full20325 and original
selected-EMA archive controls. Its preflight validated both actual RAW payloads
(81/99 state tensors) and all83 bound files; six diagnostic tests include the
actual B8 complete-scoring and B16 source208 helpers on synthetic full-cardinality
data. No optimizer or RNG restoration is used. This is a diagnostic of the
original training endpoint, not a continuation or switch in model-selection rule.

All six M1 long benchmarks completed and their raw2,048-sample P95 values were
independently recomputed. Same-run family/original paired ratios are:

| M1 three fresh repetitions | T1 median [min,max] | T2 median [min,max] |
| --- | ---: | ---: |
| FLAT / Original default P95 |.059583 [.059184,.059768]|.076530 [.076369,.077008]|
| ROUTE / Original default P95 |.059997 [.059904,.060266]|.077354 [.076974,.077537]|

All six source/artifact/image pre/post checks agree with the completed selected
P1 proof; native max error remains `4.768e-7` and original error zero. These
are local whole-B4 measurements, approximately16.6–16.9× faster atT1 and
12.9–13.1× faster atT2, not official-service results. The Original-M1 full quality
replay is still running; its quality comparison remains pending.

New M2 preparation is additive only: the already allowed FW-QueryAge16 operator
is instantiated at W50 with the same CRST-B4 spatial FLAT/ROUTE difference.
The planned source-only resource smoke applies the same p=.5 prefix-history
augmentation to both members. This is an explicitly compound new recipe;
comparison to prior CausalPE4 training cannot isolate the temporal effect.
No formal new M2 training or real resource smoke has been launched yet.

## Update — 2026-09-06 08:32 HKT

Original-M1 completed in 2,536.34 s inside the unchanged as-shipped f5 image:
112,985 chronological public calls, three initial native calls, all31,252 scored
positions, zero native discrepancy. The independent archive-only comparison
verified both receipt chains and exact target/session/start identity across
Original, both selected P1 EMAs, and both fixed endpoint24 EMAs. All pooled,
per-session, and equal-session metrics were recomputed in float64 from the
hash-bound FP32 prediction archives.

| M1 same31,252 source-minival positions | Pooled R² | Equal-session mean | Worst session R² | Pooled delta vs Original |
| --- | ---: | ---: | ---: | ---: |
| Original as shipped |.80928876|.80879490|.80252591|—|
| P1 FLAT selected |.81165233|.81138314|.79828287|+.00236357|
| P1 ROUTE selected |.81219643|.81181830|.80265911|+.00290767|
| P1 FLAT endpoint24 |.79874351|.79828581|.78549474|−.01054525|
| P1 ROUTE endpoint24 |.79998475|.79948635|.78808372|−.00930401|

The selected point estimates improve pooled/equal-session scores while retaining
the measured local >10× latency advantage. Session20120928 nevertheless loses
`.01089958/.00652333` for FLAT/ROUTE versus its own Original score. These are
descriptive, selection-exposed comparisons with unequal historical training
exposure, not formal non-inferiority or across-session improvement. Endpoint24
is not substituted for the prospectively selected artifacts.

Evidence: [Original completed receipt](../results/family_runtime_v1/original_m1_frozen_same31252_v1/receipt.json),
SHA256 `7db884108f60c3a7474b82629c18af64eb5c90325c78a6fa4a4ef3efbeb5939d`;
[same-position archive table](../results/family_runtime_v1/m1_p1_frozen_same31252_quality_v1.json),
SHA256 `f481cd4c41f9eb201efd2b15c2e65d764b4f227dcfb087dc009e6fca1436a45e`.

H1 endpoint12 RAW diagnosis also completed, 375.67 s, peak9,439,465,472 bytes,
zero parameter updates, exact fresh pre/post source and artifact authority.

| H1 original-formal endpoint12 | FLAT RAW / selected EMA | ROUTE RAW / selected EMA |
| --- | ---: | ---: |
| All20,325 pooled R² |.46204398 / .47914937|.39459942 / .41992942|
| Cold8,702 pooled R² |.24079611 / .26129283|.13427306 / .08877849|
| Full-history11,623 pooled R² |.57958973 / .59504773|.53338745 / .59978748|
| Source208 training-fit RAW |.93679023|.92571599|

RAW is not an overall remedy for the original-formal quality gap. Training-fit
remains much higher than cross-record minival, and cold augmentation does not
close the warm-history gap. No model-selection rule was changed.
Evidence: [fixed RAW diagnostic](../results/family_runtime_v1/h1_fixed_endpoint12_raw_diagnostic_v1/receipt.json),
SHA256 `02ac2bd041ef1223e40cc7de80577cf2b1d43847dd60cfe4fd7d4c149d676504`.

M2 QueryAge+prefix resource smoke has now completed 100 real paired AdamW/EMA
updates on GPU1 using source-train only. Paired steady-batch mean was
`.06981531` s, peak1,853,059,072 bytes, total8.68 s. The conservative24×3165
paired-update forecast with50% margin plus1,800 s evaluation/checkpoint allowance
is9,754.76 s (~2.71 h), below the unchanged6 h hard bound. All frozen code,
manifest, source-array, and prior source-recipe hashes match fresh postimage
checks. Eight model/resource CPU tests passed in the intended environment.
Evidence: [100-update smoke receipt](../results/m2/queryage_family_v1/queryage_prefix_source_smoke100_v1/receipt.json),
SHA256 `a9763669bac45c0c3cb51c552f99f6ff4d7da4cac4519bf801fc149649b0edfc`.
The fresh24-epoch trainer is in review; no formal QueryAge training has started.

Next H1 preparation is a bounded training-only dense-supervision recipe:
positions349/466/582/699 in each existing W700 source window, with correctly
indexed native×20 targets and explicit validity masks. Earlier positions have
350/467/583 bins of history, not full W700. The proposed fixed2 continuation
starts from the same selected plain e12 EMA exports as the completed cold CONTROL,
with fresh AdamW at1e-5 and fresh EMA .9995, not a RAW/optimizer-resumed epoch14.
It uses no prefix augmentation and preserves the deployed model/state topology
and last-output path. Resource smoke, causality/index tests, and exact shared
sampler/mask validation are required before any formal launch. This is a
prospective supervision-density hypothesis, not an established explanation of
the generalization gap or a new checkpoint-selection opportunity.

## Update — 2026-09-06 08:44 HKT

M2 QueryAge+prefix formal24 launched at08:37 HKT on GPU1, CPU8–11, one thread,
with the passed smoke receipt above and an external21,600 s timeout. Actual
preflight validated147 source/code files and all24 manifest epochs, each3,165
paired batches. Root tests include the actual training entry on a two-epoch,
two-batch CPU fixture with real QueryAge models, optimizer/EMA updates, scoring,
disk checkpoint reload, selection, completion hashing, and deliberate artifact
mutation rejection;10 combined model/trainer tests have passed across the final
checks. The frozen trainer SHA is
`f864f067f43947de9847a644a2dd805f816ae75aa47130af1a191bb9ec0b39d4`.
First-epoch saved/reloaded RAW/EMA scores were F `.174958/−.026457`,
R `.154966/−.027936`, elapsed238.63 s. These are early trajectory observations,
not candidate acceptance or an opportunity to shorten the fixed schedule.
Live evidence: [new run root](../results/m2/queryage_family_v1/queryage_prefix_pair24_v1/).

H1 dense smoke completed20 real paired updates in32.68 s, peak9,459,227,648
bytes, paired steady P95 `1.4178827` s. Its forecast is4,909.42 s (~81.8 min)
including50% timing margin and1,800 s scoring/checkpoint allowance, below the
unchanged6 h hard bound. There was no scoring, checkpoint save, or selection.
Receipt SHA `cbb36f1eefec2a588953e029ddd8019127df93f0196cc4310e298df69c9799fc`:
[20-update resource evidence](../results/family_runtime_v1/h1_dense_phase2_pair_smoke20_v1/receipt.json).
Fresh pre/post authority matches; the formal gate also matches the smoke's
exact source/artifact/protocol/code and recomputes its forecast from the raw
16 steady-update durations. Ten helper/runner CPU tests passed, including
causality on a real W700 pair and ragged-valid-target microbatch accumulation.

H1 fixed2 dense training launched at08:44 HKT on GPU0, CPU12–15, one thread,
with a separate21,600 s timeout. Both arms load their actual selected plain
e12 EMA exports, then create fresh AdamW at1e-5 and fresh `.9995` EMA. This is
the same initialization/optimizer recipe as the completed cold CONTROL,
not RAW/optimizer continuation and not a renamed original-formal epoch14.
The only treatment is the declared valid multi-position supervision; no
prefix augmentation is applied. The unchanged source sampler/dropout identities
must match original epochs1/2, each731 batches and23,212 windows per arm.
Formal checkpoints are saved/reloaded at each stage epoch; primary is fixed
stage-e2 EMA, RAW is secondary, scoring occurs separately after completion.

Frozen H1 runner SHA
`220daf9cd4e93008a80e3ae2379c768dffc15297f9574f7ac39800c7c936991e`;
protocol SHA `e7cb463a7990a4cf9d7187ab71107c80a3203f3f164fef6beddc3e7eccf8ceb7`;
[formal authorization](../results/family_runtime_v1/h1_dense_phase2_pair_formal2_authorization_v1.json)
SHA `badcd8ffc11ceb0a2c7e8a5fb387c26b57aee4d7e13e6a38e1c924f53d720dff`.
Live evidence: [new dense run root](../results/family_runtime_v1/h1_dense_phase2_pair_formal2_v1/).
No old checkpoint, source cache, payload/image, registry entry, or submission
has been replaced. New endpoint scorers are being prepared without changing
either live training closure.

## Update — 2026-09-06 09:04 HKT

H1 dense stage1 checkpoint has persisted and stage2 is running; no dense score
has been computed. M2 QueryAge+prefix has completed6/24 epochs and is in epoch7.
The schedule and source-only earliest-tie best-EMA selection are unchanged.
Both live training source closures remain frozen.

Post-completion preparation has passed its bounded CPU checks:

- H1 dense evaluator:4 tests passed, including real cached scorer execution on
  a synthetic13-session20325-row surface with exactly8702 cold/11623 full rows
  and a208-row source surface. It binds both dense checkpoints, the actual
  completed-run authority, source208 reference, and matched cold CONTROL EMA
  archives. It scores fixed dense stage2 RAW/EMA only, with no selection.
- M2 QueryAge finalizer:6 tests passed. It revalidates the entire completed
  trainer authority, freezes source-minival EMA picks before scoring, exports
  real QueryAge states, strictly checks the seven-session1011-row native
  archive identity, recomputes FP64 metrics, and guards each scorer forward.
- M2 QueryAge runtime:5 tests passed with actual fresh seeded FLAT/ROUTE
  models, nonzero route gate, heterogeneous banks, B1/B7, startup, W50 rollover,
  state mutation and inactive-lane checks. Its temporal path recomputes the
  independent W50 QueryAge memory; it does not claim a causal KV-cache identity.

These are code/contract checks, not actual trained-payload parity, complete
dev replay, or latency measurements for the new M2 recipe. The new scorers
must pass their actual completed-artifact preflights before GPU execution.
They use fresh output roots with2400 s/22-GiB bounds and no parameter updates.

Prepared source SHA256 values:
H1 dense scorer `761ebfe2b0f151b5bfc92c3303eb48020c0675281860a0d79fa169c4223a9419`;
M2 QueryAge finalizer `975e42d44b3189fdd47e5606138aeacf7d59a57560ce788c150e2f2073b8c9ff`
(corrected its stale preparation-only module description;6 tests passed again);
M2 QueryAge runtime `4f11c10e2e493c18eb08063b6922dc4413096c0f311bab940fb5f219b5b997b0`.

## Update — 2026-09-06 09:21 HKT

H1 dense paired training completed successfully in2086.15 s, both arms exactly
1462 updates, with both stage1/stage2 checkpoints persisted. Actual completed
receipt, source/sampler/dropout authority and stage2 checkpoint deserialization
passed the independent scorer's preflight:123 bound inputs, matched cold CONTROL
EMA baseline analysis recomputed exactly, no optimizer/RNG restoration for scoring.
The independent fixed-stage2 source208/complete20325 RAW/EMA evaluation launched
on GPU0 at09:20 HKT with a2400 s timeout. It has no parameter updates, selection,
or promotion. The dense quality result remains pending.

Evidence: [completed dense train receipt](../results/family_runtime_v1/h1_dense_phase2_pair_formal2_v1/receipt.json),
SHA `e6a1eb8682b7764a16a16661c002061b77c07738a70e4661b9073f602096555a`;
[scorer authorization](../results/family_runtime_v1/h1_dense_phase2_fixed_e2_score_authorization_v1.json),
SHA `732d28a1eaa0c2625b6ea184dbdbda5f4f80d4c8926b698a817e2f04752ae556`;
[live scoring root](../results/family_runtime_v1/h1_dense_phase2_fixed_e2_score_v1/).

M2 QueryAge selected-only ext4 evaluator is prepared, not executed:6 CPU tests
passed, including the actual gated evaluator run with synthetic caches and real
QueryAge weight loads/forwards, mutation/timeout refusals and strict production
four-session2069-row archive geometry. Its authority requires completed trainer
and finalizer artifacts before opening ext4. It reports selected exports only,
with historical-development qualifiers, and cannot change source-minival picks.
Source SHA `ee1c620e9c207b982a86d0dcbdc114255648aaa259627fdd3e6068cdee47a62d`.
The corresponding actual-selected CPU streaming proof is preparation in progress.

An additive H1 QueryAge model factory is also prepared, not trained or scored:
3 CPU tests passed in9.77 s with actual W700 models. It retains the existing
set-v2-localbalanced frontend/readout/initialization and replaces only the
temporal module in both arms with the already allowed FW-QueryAge16 definition.
Names include both temporal member and the H1 spatial recipe; no hidden backbone
switch or new family member is introduced. No trained state is transferred.
Any capacity gate for this candidate remains conditional on the current dense
result, not authorized merely by passing model tests.
Model SHA `bef468cf3e7e34f511bd2a94ceab7bcfd53e959475f5ee8f83f3272923e741b9`.

## Update — 2026-09-06 09:36 HKT

The H1 dense fixed-stage2 scorer completed successfully in767.85 s, peak
9,469,646,848 bytes, zero parameter updates, with exact fresh pre/post authority
and all eight FP64 source208/complete20325 archives verified. Primary is EMA;
RAW remains secondary. The fixed dense recipe does not jointly improve both
arms or repair the complete-history quality gap.

| Fixed H1 dense stage2 | FLAT EMA | ROUTE EMA | FLAT RAW secondary | ROUTE RAW secondary |
| --- | ---: | ---: | ---: | ---: |
| Source208 training-fit R² |.92599763|.92905080|.92459752|.92039169|
| All20325 pooled R² |.50903660|.42154474|.52997374|.47080531|
| Cold8702 pooled R² |.32457347|.08211222|.37717331|.20816496|
| Full-history11623 pooled R² |.60629953|.60618170|.60943810|.61224860|
| All delta vs matched CONTROL EMA |+.00931141|−.00668274|+.03024854|+.04257783|
| Full-history delta vs matched CONTROL EMA |−.00149531|−.00141854|+.00164326|+.00464837|

Only5/13 sessions improve overall for each dense EMA arm versus its matched
CONTROL. The much larger preceding PREFIX gains remain specifically cold-history
gains; dense supervision is not a replacement or a new selection opportunity.
No continuation budget, primary criterion, or prior artifact is changed.
The Original/C2 quality gap is still substantial, with the previously disclosed
calibration/training-exposure differences unchanged.

Evidence: [completed dense scoring receipt](../results/family_runtime_v1/h1_dense_phase2_fixed_e2_score_v1/receipt.json),
SHA `3a03ddc9ca9c93fa3a78857a7dc68fffdc421fcd18f9695a1a3e2dcd1b79d2aa`.

Next H1 step is one explicit allowed temporal substitution, FW-QueryAge16,
first on the frozen208 source-capacity gate only. The new model retains the
current set-v2-localbalanced spatial recipe for both arms; it does not inherit
trained causal temporal state. A disposable20-step paired resource smoke and
strict260→1040 source-only learning gate are in preparation; no new GPU work
or formal training has been launched for it. Passing a tiny training-fit gate
would not establish full source-minival or cross-record quality.

M2's new cached QueryAge sibling has now passed5 root CPU tests in7.12 s:
actual B1/shared FLAT and B7/heterogeneous ROUTE (`g=.2`),110 startup/rollover
steps against uncached and direct models, mutation/reset/active-lane checks,
and a projection spy showing50 memory tokens at reset then exactly5 per
ordinary update. It caches only independent memory K/V, not contextualized
causal history. Persistent cache state is460800 bytes per active lane; this
does not establish actual selected-weight equivalence or a speedup yet.
Runtime SHA `a1cdeaae07bcc78fe15dd825e469e41b027202262730a93032864fd3bfb675b9`.

## Update — 2026-09-06 09:53 HKT

The temporal-convergence target is explicit: use the same FW-QueryAge16 core
on M1/W100, M2/W50 and H1/W700 if each task clears its quality gates. The core
is width256,8 heads,4 blocks,FFN512 and16 age buckets without causal PE.
FW-CausalPE4 remains a separately named control/fallback until that evidence
exists. Cached and uncached QueryAge are implementations of the same operator,
not additional temporal architectures. H1's existing unscaled-dot/localbalanced
spatial recipe remains a disclosed deviation even if all temporal cores converge.
No cross-task unification or non-inferiority is being declared in advance.

M2 is in epoch19 of its fixed24 schedule. A new root read-only check rehashed
the live run's complete source/trainer authority and matched all149 top-level
bindings to the preconstruction protocol; it performed no bank construction,
model forward, checkpoint selection or scoring. GPU1 remains dedicated to that
run. Selection will be frozen only after all24 paired epoch receipts reconcile.

The selected-weight CPU complete-ext4 proof runner is prepared for **both**
uncached and independent-memory-K/V-cached QueryAge implementations. Its6 CPU
tests passed in8.88 s, including actual W50 QueryAge runtime fixture execution
and refusal of an unauthorized runtime kind. This is code verification, not a
completed proof for the still-unselected weights. The subsequent same-frozen-
Original-image B7 latency comparator is being implemented additively; actual
timing remains conditional on selected-export and complete-stream proof passes.

H1's [source-capacity protocol](PROTOCOL_H1_QUERYAGE_SOURCE_CAPACITY_V1_20260906.md)
permits only a disposable20-update resource smoke followed by the fixed260 and
conditional cumulative1040 source208 gates. Root review is still resolving
resume/authentication and real-model test issues before any GPU authorization.
In particular, initialization g=0 parity is required only for a fresh model;
continuation instead requires exact RAW/optimizer/RNG state restoration.
GPU0 is free; no unreviewed H1 probe has been launched to fill it.

## Update — 2026-09-06 10:33 HKT

M2 QueryAge+prefix completed the fixed24 paired epochs. Fresh completion and
all-source/code audits passed before EMA-only export. The frozen source-minival
primary picks are FLAT epoch2 and ROUTE epoch20. Native source-score replay
reproduced the selected/endpoint EMA records within `1.1e-7`; no epoch was
reselected after external evaluation. The selected-only four-session2069-row
development replay completed with fresh pre/post authority, zero updates, and
native batch/singleton errors below `1.2e-8`.

| Selected M2 QueryAge+prefix | FLAT e2 | ROUTE e20 |
| --- | ---: | ---: |
| ext4 pooled native FP64 R² | .14306357 | .29639631 |
| ext4 equal-session R² | .15068464 | .29619992 |
| Pooled delta versus Original | −.08613404 | +.06719870 |
| Pooled delta versus historical packaged e8 | −.24444167 | −.09110893 |

ROUTE improves over Original on3/4 sessions but loses badly on the fourth:
session R² `−.23816213`, delta `−.25546214`. Both new arms score below their
older CausalPE counterparts overall; because the prefix recipe also differs,
this is not a clean temporal-operator ablation. These are historically exposed
development surfaces, not formal non-inferiority or untouched heldout results.
All negative arms, source picks and epoch endpoints are retained.

Evidence: [finalized source-only EMA exports](../results/m2/queryage_family_v1/queryage_prefix_pair24_finalized_v1/receipt.json),
SHA `57c214b12c2097aa6bff74451169d092296f473c1d544dad53344288c23e9dcc`;
[fixed selected ext4 replay](../results/m2/queryage_family_v1/queryage_prefix_pair24_selected_ext4_v1/receipt.json),
SHA `0442ed59d740397b621a6c75b53041bca67205ad5e9407096406371d4b966b0e`.

The actual CPU-T1 complete-stream proofs are now running in separate processes
for cached and uncached implementations. Each must cover10,839 public raw calls
and2,069 scored endpoints per arm, including direct finite-W and native-gold
comparisons. Their state is not yet a proof result. Subsequent timing must use
the immutable Original image, actual selected weights, and both complete proofs.

H1 QueryAge source-capacity CPU tests passed11/11 in472.02 s, including actual
W700 checkpoint/optimizer/RNG continuation. Its disposable20-update GPU smoke
completed in14.8913 s, peak4,790,988,800 bytes, with exactly zero fresh g=0
parity error and no scoring or retained checkpoint. The mean16 steady paired
updates was `.67109429` s, giving the predeclared260/1040 forecasts
`561.73/1646.91` s.

The fresh260 screen then completed in184.72 s. Source208 pooled R² is
FLAT `.10173889`, ROUTE `.03976077`; prediction-std/target-std ratios are
approximately `.577/.526`. The finite paired-loss decline and ANY-arm260
eligibility were recomputed from the actual receipt, and fresh21-base-file
authority plus the full owned-artifact manifest passed. The exact cumulative
1040 extension is now running from its260 model/optimizer/RNG checkpoint.
Only a BOTH-arm1040 pass can lead to a new formal-candidate review; this screen
neither uses minival rows nor establishes generalization. The shared cache
container includes both splits, so loading its bytes is not described as an
absence of minival deserialization.

Evidence: [H1 smoke receipt](../results/family_runtime_v1/h1_queryage_source_smoke20_v1/receipt.json),
SHA `1a185b90d5c38e604ae8f0c0e515b4eab1dde418245fb6adff9dfae856abff74`;
[H1 capacity260 receipt](../results/family_runtime_v1/h1_queryage_source_capacity260_v1/receipt.json),
SHA `b3698fcdca2a7e75c1e3a9dc08169e20c743e890f718447d283d4f00b004552f`.

The [round-two unified-kernel review](REVIEW_ASTRA_UNIFIED_KERNEL_ROUND2_20260906.md)
accepts temporal convergence as a target, while separating strict finite-window
boundary repair from the different birth-frozen streaming-context contract.
It does not authorize a third trained kernel. The [H1 fresh QueryAge+prefix
protocol](PROTOCOL_H1_QUERYAGE_FORMAL_PREFIX_V1_20260906.md) is likewise conditional
and non-authorizing; a future pass would concern a compound quality candidate,
not temporal causality against the old no-prefix formal recipe.

## Update — 2026-09-06 11:19 HKT

The user's priority is now explicit: keep network unification on the critical
path; modest latency differences do not justify returning to separate task
backbones. Quality and resource-safety gates remain in force. No third trained
kernel or task-specific latency architecture was opened.

H1's cumulative1040 QueryAge source screen completed in545.97 s, peak
9,565,109,248 bytes. FLAT/ROUTE source208 R² is `.97353413/.97609921`, with
prediction standard deviations `.00506855/.00517632` versus target `.00522646`.
Both predeclared1040 gates pass. Root recomputed the complete loss history,
source/code closure, checkpoint hash, owned-artifact manifests, and all three
stages' original external-authority SHA/status/bindings. This is the admission
screen, not formal minival quality, and its learned state is not a warm-start.

Evidence: [actual1040 receipt](../results/family_runtime_v1/h1_queryage_source_extend1040_v1/receipt.json),
SHA `2501118cda30796526a88b26076b3a94c9cbb4978c46b4084abbd88c81fd2c4b`;
checkpoint SHA `ea859c1cab996dd4aed222eb7b07878ad8a17efe9ae926e968013b841fee4c1e`.

The new formal trainer, guarded scorer, source-only smoke, and supervisor passed
27 combined CPU tests in1.96 s. The tests include real reduced worker→freeze→
finalizer execution, AdamW/EMA/RAW+optimizer+RNG checkpoint restoration, native
FP64 scoring, plain-EMA/NPZ exports, the20-update disposable smoke lifecycle,
and rejection gates. Separate model-free inspection of the actual immutable
cache confirmed13 minival sessions,2,908 selection points,20,325 complete points,
and all twelve source/keep/prefix schedules. Tests are not claimed to replace
actual resource evidence.

The actual two-GPU smoke then passed in19.89 s total. Each fresh arm performed
20 source updates under the intended prefix/dropout/LR/EMA recipe, measured the
exact208 source endpoints under EMA with micro8, and serialized, reloaded,
restored and removed its known disposable checkpoint. Fresh initialization parity
was exactly0 and both actual route-gate gradient magnitudes were `.00099960505`.

| Actual H1 paired resource smoke | FLAT/GPU0 | ROUTE/GPU1 |
| --- | ---: | ---: |
| Per-arm wall, seconds | 16.9312 | 16.8739 |
| Steady update P95, seconds | .64052841 | .63573890 |
| Peak CUDA allocation, bytes | 9392267264 | 9398373376 |
| Source208 EMA seconds/endpoint, micro8 | .00749629 | .00752439 |
| Conservative full-formal forecast, seconds | 11079.23 | 11019.62 |

The forecast includes1.5× training/scoring headroom, all12 minival passes,
selected/e12 complete reporting, measured checkpoint round trips and1800 s setup
allowance. Both fit the21600 s hard wall and22-GiB allocation bounds. Full
fresh source/code/auth/owned-file validation passed after the smoke.

Evidence: [paired resource smoke](../results/family_runtime_v1/h1_queryage_formal_prefix_smoke_pair_v1/receipt.json),
SHA `daece10b9c873caacd3b802f851a272c9bd2c51679171c90bb14e64d2fd7f51c`.

Fresh H1 formal training launched at11:17 HKT on the dedicated GPU0/1 pair.
The actual synchronized ready barrier passed for seed42 shared initialization
and every source/keep/prefix identity. Fixed12 epochs,731 updates/23,212 source
windows per epoch, effective32/micro8, p=.5 prefix then p=.1 unit dropout,
AdamW1e-4 with epoch-one warmup, EMA.9995. Primary selection remains EMA pooled
native FP64 R² on the frozen2,908 minival points with earliest tie, and complete
20,325-bin reporting is post-freeze. This is a QueryAge+prefix compound candidate,
not a clean temporal ablation versus old no-prefix CausalPE.

Authority: [new formal authorization](../results/decoder_validation_v2/20260905_190000/h1/queryage_family_v1/formal_prefix_split12_authorization_v1.json),
SHA `0e09b1fb0115c3b170aa8c5f62cd29a24d2414d568454a9b3a7b4ca3885e853b`;
[formal output root](../results/decoder_validation_v2/20260905_190000/h1/queryage_family_v1/formal_prefix_split12_v1/).
No formal score or final selected epoch is available yet.

M2's cached and uncached QueryAge full-stream proofs both passed10,839 public
calls and2,069 endpoints per arm. They give identical public prediction arrays
and reproduce the fixed native quality scores. Maximum public/native-gold error
is `2.02679e-7`, and maximum public/direct-window error is `8.94070e-8`.
Uncached/cached walls were214.39/221.97 s, peak RSS717,119,488/695,934,976 bytes.

Evidence: [uncached proof](../results/family_runtime_v1/m2_queryage_selected_complete_uncached_ext4_v1/receipt.json),
SHA `93775aa2b2963386681b93c0fb2265f372062f7ae3d9e1dff445fa6335c1fff7`;
[cached proof](../results/family_runtime_v1/m2_queryage_selected_complete_cached_ext4_v1/receipt.json),
SHA `bb3116055f7cd289715b4acadd720c17c5a785d89141eac563df78fd25589701`.

The six fresh2048-call same-Original-image B7 timing runs have completed and are
being independently aggregated. Each uses the actual selected states, the same
seven continuous source-train lanes,128 warmups, rotating three-engine order,
and direct public W50/original-buffer oracles. Small T1/T2 tradeoffs do not alter
the unified-network primary direction; they are local CPU development timings,
not official service latency or a quality-selection criterion.

## Update — 2026-09-06 11:29 HKT

Both H1 formal workers are live on their assigned GPUs. Epoch one completed
all731 updates with matching source/keep/prefix identity hashes and finite
2,908-point EMA scores: FLAT `−5.82402412`, ROUTE `−5.82402341`. These early
scores are poor and do not show quality preservation. No schedule, selection
rule, model, input, protocol or bound source file was changed in response;
the predeclared twelve-epoch experiment continues. The final selected state
and20,325-point complete report are not yet available.

The [three-task architecture audit](AUDIT_QUERYAGE_THREE_TASK_NETWORK_FAMILY_20260906.md)
confirms the common spatial frontend, temporal operator and readout design,
not merely a shared temporal class name. It separates necessary shape/output
adapters and training provenance from H1's real unscaled-dot/local-balanced
spatial preset. All ROUTE arms use an eight-head/eight-slot/32-dimensional
route-key logit bonus; input-carrier width follows task shape.

The [completed M2 latency summary](RESULTS_M2_QUERYAGE_SELECTED_LATENCY_20260906.md)
includes every one of the six2048-call receipts, fresh model-free authority
revalidation, per-repeat ratios, and three-repeat medians/ranges. Selected
ROUTE mean latency versus Original is +5.828% to+9.179% at T1 (median+8.162%),
and −6.711% to−3.667% at T2 (median−6.547%). T1 p95 direction varies; T2 p95
is lower in all three repeats. These same-host B7 timings are configuration-
specific and include disclosed co-hosted work; they are not an official
service result or a reason to abandon the common network.

An additive H1 QueryAge complete-stream proof consumer is being prepared
against the existing `CurrentQueryStream` runtime. Its intended admission is
the final fixed12-epoch receipt plus selected plain-EMA/NPZ authority; current
epoch checkpoints do not qualify. No production replay, new selection, runtime
optimization or benchmark has been launched for the incomplete H1 candidate.

## Update — 2026-09-06 11:50 HKT

H1 formal training has completed three of the fixed twelve epochs per arm.
Every persisted arm pair has matching sampler/keep/prefix identities. The
finite2,908-point EMA trajectory is shown without early selection:

| Epoch | FLAT EMA pooled R² | ROUTE EMA pooled R² |
| --- | ---: | ---: |
| 1 | -5.82402412 | -5.82402341 |
| 2 | -.61229772 | -.60726327 |
| 3 | -.01650618 | -.01582852 |

This is recovery from poor early predictions, not demonstrated preservation
of H1 quality. No source/model/protocol/selection change was made in response.
The two workers remain live; post-freeze complete scoring is still unavailable.

The additive [H1 selected public-proof consumer](../src/family_runtime_v1/complete_h1_queryage_source.py)
has been reviewed and corrected at root. Its21 focused tests and27 existing
formal trainer/launcher/scorer/smoke tests pass together:48 tests in10.11 s,
CPU-only, one thread, affinity16–19. The new tests include a real reduced
`run()` lifecycle using actual QueryAge and a causal-k5 frontend; chronological
gaps and W700 rollover; a last scored endpoint before an unscored raw tail;
all native metrics/disk archives; six internally rebound status/export/
checkpoint/freeze/source/metric drift cases that fail before a model or source
loader can run; and public shape/dtype/ownership/finite rejection.

The consumer revalidates the original formal source/code/capacity/smoke
authority, all12 immutable epoch records and the earliest EMA selection, the
paired readiness/final receipts, selected plain exports and native NPZ metrics.
It binds its own runtime closure, persists and rechecks a pre-authority sidecar,
checks selected state immutability, and enforces CPU-thread/wall/peak-RSS bounds.
Root separately exercised the fresh original-authority check against the actual
live formal input authority and confirmed the missing completed artifacts reject
`collect_bindings` before model construction. No actual selected-formal public
replay or proof authorization has been issued for the incomplete candidate.

Consumer SHA `8f9b9a1825176d6568f9001f5086e788d91dedb8f1ad87b855e98986fddf8264`;
test SHA `78b21984427f41fc584313bbab5340a500f8b0a276605e605b8942ef35f2e20b`.

A separate actual-source compatibility diagnostic passed using the already
completed1040-screen RAW weights, not any selected formal state. It used only
the first chronological train session, `ses-19250101T111740`, on CPU/T1. Each arm
advanced the first five observations, checked native direct-window equality at
0 and4, then reset with the independent true700-bin history at699 and checked
699→700 rollover. This is intentionally not a complete chronological replay.
ROUTE's learned gate was nonzero (absolute maximum `.03744045`). Maximum native
stream/direct error was `4.42378223e-9`; both models' state digests stayed fixed.
Wall14.06 s; the final observed RSS was731,947,008 bytes. Actual source-capacity
admission and the source/checkpoint/runtime hashes were checked before and after.
No target was scored and no minival row was forwarded; the shared cache container
was deserialized and validated and therefore includes both splits' bytes.
This supports compatibility of the real H1 unscaled/local-balanced preset with
`CurrentQueryStream`, not formal quality or selected full-stream equivalence.

The [M2 fixed-export error decomposition](DIAGNOSTIC_M2_QUERYAGE_EXT4_ERROR_DECOMPOSITION_20260906.md)
was independently recomputed from both actual native archives at root for all
eight arm/session combinations. ROUTE's fourth-session SSE is10.800232% constant
mean-bias and89.199768% centered residual. Its two centered correlations are
`.470998/.217346`, and prediction/target standard-deviation ratios are
`.968747/.749851`. Thus a constant offset alone does not account for the poor
tracking. No offset, scale, lag or alternate model was fitted or selected.
Original's exact same-surface scalar R² is available, but its matching native
prediction vector was not found; a different1,011-row archive was deliberately
not mixed into this2,069-row decomposition. This remains exposed development
evidence and does not supply a causal diagnosis or a corrective training recipe.

11:53 HKT actual-worker follow-up: epoch4 completed on both arms, FLAT EMA
`.0900775260`, ROUTE `.0844378100`, with identical declared input identities.
Both dedicated GPU workers and the original supervisor are still live. Positive
R² now is not a claim of improvement over a baseline or grounds for early
selection; the fixed12-epoch schedule remains unchanged.

## Update — 2026-09-06 12:22 HKT

The dedicated H1 workers and the original supervisor remain live. Epochs5–7
have now persisted with matching paired input identities and finite2,908-point
EMA scores; no early model selection or schedule change occurred:

| Epoch | FLAT | ROUTE |
| --- | ---: | ---: |
| 5 | .1189669167 | .1133843782 |
| 6 | .1450572295 | .1394255354 |
| 7 | .1806356888 | .1843049558 |

The [reference-surface audit](AUDIT_H1_QUERYAGE_COMPARISON_SURFACES_20260906.md)
resolves the exact comparator names and evidence strength. Original-H1 is the
as-shipped image-owned SPINT wrapper with a hash-bound complete native archive;
C2 is the separate historical epoch15 fixed-deployment/fixed-readout package,
with aggregate receipts but no pointwise prediction archive. The stronger
Original reference is retained explicitly.

| H1 reference / surface | Pooled R² | Evidence |
| --- | ---: | --- |
| Original,20,325 complete endpoints | .9607844356539761 | Actual native prediction/target/session/end archive, independently recomputed |
| Original,2,908 selection endpoints | .9636925865316174 | Exact coordinate join from the same complete archive; all targets equal current cache FP64 casts |
| Historical C2,20,325 complete endpoints | .8884989023208618 | Existing fixed-reference aggregate receipt; not recomputed from predictions |
| Historical C2,2,908 selection endpoints | .8715679049491882 | Existing direct-selection aggregate receipt; not a sliced pointwise archive |

Original receipt SHA `539bfa832c650df51e22be14b5ee0dba0c1f49fbb595f9fb869328a601f30c48`;
Original NPZ SHA `f1bb6739415ea66bb86bf4285035f131333e350072ae364139b81488912ba81c`.
The exact2,908 coordinate-list SHA is
`eab09367df01f042441a35a1c344103131469fac86d649aa204888f590ffcb8b`,
and joined-target byte SHA is
`e1b61a0f4f15342875ae88612e26ecef75e7eee937e755d704863d36744ee3a1`.
Original outputs originated as public native FP32 values; FP64 describes the
archive casts and subsequent metric arithmetic, not FP64 model inference.

A separate source-only inspection of the actual frozen Original image verified
the wrapper and SPINT code hashes. Reset performs a session-tag lookup of
payload-resident calibration features; predict consumes neural observations
and that stored tensor. The wrapper does not read evaluation targets/query
files in constructor/reset/predict/observe. The inspection did not import model
code, load the pickle, read data, run inference, or modify the image. The observed
two-trial calibration differs from the family's immutable M3 bank, and historical
payload/training provenance remains incompletely documented. These are disclosed
comparison limitations, not a reason to explain away or remove the Original score.

The new [post-freeze quality comparator](../src/family_runtime_v1/compare_h1_queryage_quality.py)
now handles both FLAT/ROUTE selected and epoch12 native archives, exact
candidate/Original targets and coordinates, all native metrics, and deltas
against both Original and C2. It validates every terminal plain-EMA/NPZ path
and hash, frozen epoch/checkpoint identity, original input sidecar/cache authority,
all formal authority, external authorization, and pre/post input/code hashes.
It does not execute a model, deserialize a model/cache/checkpoint payload, select
a model, or issue a non-inferiority/promotion claim. It explicitly labels C2's
legacy imported aggregates and duplicate selected/epoch12 identities.

Its15 tests include actual end-to-end JSON/NPZ comparison with all four exports
and both references while model calls and `torch.load` are forbidden; internally
rebound status/source/terminal-export/NaN-metric/Original-sidecar rejection; and
late authorization/archive/owned-sidecar drift without emitting a PASS receipt.
Combined with the21 public-proof and27 formal tests, all63 passed in18.81 s.
Root also exercised the actual Original/C2 baseline adapter on all20,325 points,
and actual original formal admission rehash with `torch.load` and module calls
forbidden. Both passed. Actual incomplete-formal inputs are rejected as expected;
no completed-candidate quality comparison or authorization has been issued yet.

Comparator SHA `2a87396cf063f82c288b412616398fe09ea3d4d09afaa0f1db563fb0a97e8a11`;
test SHA `172e1fba71f03a462c61f6d281e483ac7b23d8be76ff1296794af54e40629255`.

## Update — 2026-09-06 12:37 HKT

H1's original paired supervisor remains live, with the same GPU workers and
unchanged fixed twelve-epoch recipe. New frozen-selection records are:

| Epoch | FLAT EMA pooled FP64 R² | ROUTE EMA pooled FP64 R² |
| --- | ---: | ---: |
| 8 | .21315462839959531 | .21520279477612003 |
| 9 | .2427625516314761 | .24458267418625101 |

Both arms have matching sampler/prefix/dropout identities at each epoch. These
are2,908-point selection trajectories, not complete20,325-point reports or a
final selection. Epoch9 checkpoint SHAs are FLAT
`367bfc1c6942c978ad5a51a080a60bfcdbeb74c3385e56615552f6b05f986e87`
and ROUTE `cb8cf172bef549f7dba74cb34b154bf72b0f4cb1d7a9b24fa9e6d59316054c28`.

The user requested checking computation alignment when R² is low. Three
bounded read-only internal audits now cover [H1 numeric/EMA contracts](AUDIT_H1_LOW_R2_NUMERICAL_CONTRACT_20260906.md),
[H1 cached coordinates](AUDIT_H1_LOW_R2_COORDINATE_ALIGNMENT_20260906.md), and
[H1 runtime/direct plus M2 native scale](AUDIT_QUERYAGE_LOW_R2_RUNTIME_ALIGNMENT_20260906.md).
Static paths consistently map W700's last bin to the native H1 target, train
with target×20, score with prediction/20, and use EMA and FP64 pooled scoring.
The actual immutable H1 cache inspection found all23,212 train and2,908
selection windows bounds-valid and endpoint-mask-valid, with all13 training
sessions starting exactly after their calibration first-three boundary.
M2's separate native bridge is5, not20. No demonstrated local scale/index
bug was found; raw-NWB timestamp/binning/channel semantics and actual future
selected-checkpoint whole-surface behavior remain outside this static audit.
No model forward, GPU experiment, frozen-code change, tuning, or restart was
performed for these audits. This does not diagnose the low scores as a model
capacity problem, and internal agent agreement is not external replication.

The user also requested a list for third-party review. A new
[prioritized independent-audit checklist](THIRD_PARTY_AUDIT_CHECKLIST_20260906.md)
covers eight evidence questions: H1 full computation alignment, baseline and
historical payload identity, M1 small gains, M2 adverse-session quality,
training/calibration/selection exposure, selected runtime equivalence,
SPINT-relative latency, and the finite network family. It includes the
specific581919 default-payload discrepancy and the still-unverified remote
override question. Primary receipt hashes were freshly recomputed, and every
linked evidence target exists. The list requests independent calculations and
claim-scope judgments rather than relying on our PASS labels; no external
files were sent and no third-party access/submission was performed.

## Update — 2026-09-06 12:48 HKT — user freeze and existing-run closeout

The user explicitly requested ending further network changes/new experiments
after the currently deployed work closes, while preserving the order of
already launched experiments. All internal agents are now idle. The existing
H1 supervisor is the only ongoing experimental chain under this closeout:

`both fixed12 training workers -> frozen EMA selection -> both built-in
selected/epoch12 exporters and complete scorers -> final receipt -> exit`.

This is the already launched supervision path, not a newly scheduled run.
Its source has been read to confirm that no successor architecture, training,
public-runtime benchmark, or official submission follows its final receipt.
Only non-training result/identity verification and reporting will be used to
close these existing artifacts. The still-unexecuted26-file independent raw
NWB reconstruction and prepared-but-unlaunched selected QueryAge runtime proof
remain unexecuted. The third-party checklist is advisory, not an execution queue.

Epoch10 records report FLAT `.2667007189367504`, ROUTE `.272045082443344` on
the same2,908 frozen selection points. Sampler/prefix/dropout identities match
across the two arms. Checkpoint hashes are FLAT
`c424be32e21b35be91211af71874ec94252a265870d1843fa86d05b83379c38c`
and ROUTE `7bfeeb571c48c644b3866e28ef45b60602438f2924f8e28c339e015b6dcb9fce`.
The original supervisor and both workers were confirmed live at12:46 HKT;
no freeze or completed-formal receipt was present at that observation.

Two already-started static audits completed before/at the freeze and are retained:
[installed H1 loader](AUDIT_H1_INSTALLED_LOADER_SEMANTICS_20260906.md) and
[installed evaluator metric](AUDIT_H1_INSTALLED_EVALUATOR_METRIC_20260906.md).
The local H1 loader re-bins `units.spike_times` and reads the stored velocity
array; it does not consume an H1 pre-binned acquisition. A single raw-file
metadata inspection found stored rate `.02`, used by that loader as the time
interval, but no whole-cache independent re-binning was performed.
The metric audit explicitly distinguishes across-session pooled versus equal
mean reduction. For the same nonzero-variance session rows, sklearn's
variance-weighted R² and per-session SSE/SST are algebraically equivalent;
actual Original archive comparisons agreed within `5.56e-16`. H1's installed
`S...` grouping also differs from the timestamped cache naming and must not be
confused with the evaluator's separate `Run...` branch. These observations do
not establish a cause for the low H1 score or justify changing its frozen rule.
