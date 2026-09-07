# Held-out-session backprop-free calibration: method closure and evidence boundary

**Maintained:** 2026-08-05 07:23 HKT (Asia/Hong_Kong)  
**Scope:** native MUA, sorted SUA, and deterministic pseudo-MUA evidence including the completed
fresh paired-view C1 aggregate, encoder-only PTQ/QAT program, and the score-blind external-subject
`sub-M` compatibility closure. This is a living root analysis, not a formal-test authorization.

## 1. Exact problem being solved

The deployment target is a decoder that adapts to a new recording session after a small
chronological calibration block, while satisfying all of the following:

1. source-session training may use ordinary backpropagation;
2. held-out-session calibration performs no optimizer step and no backward pass;
3. calibration is finalized by causal statistics and forward computation;
4. query windows occur after the calibration boundary and model weights remain frozen;
5. the method accepts session-local variable unit/channel count and does not require physical
   unit correspondence across sessions.

This is **not** zero-shot learning and it is not necessarily label-free. Ordinary B3 activity
calibration is neural-only, whereas T4 uses one target-direction label for each rewarded trial
that enters its cosine fit. The scientifically correct phrase is therefore **supervised,
backprop-free session calibration**, not unsupervised adaptation.

The offline/online boundary is equally important. The decoder is allowed to co-adapt to the
calibration mechanism on source sessions. At deployment, only the already-trained weights are
frozen. Conflating these two boundaries led to an overly broad earlier hypothesis that a frozen
decoder must reject every new descriptor.

## 2. Current computation graph

For channel or unit `i`, the source-trained B3 activity path consumes chronological neural support
and produces a temporal identity embedding `E_i`. T4 supplies a four-value session descriptor
fitted from labelled calibration trials:

```text
r_i(t) = b_i + a_i cos(theta_t) + c_i sin(theta_t) + epsilon_i(t)
m_i    = sqrt(a_i^2 + c_i^2)
T4_i   = [a_i, c_i, m_i, b_i]

first M_activity neural trials -> B3 temporal statistics --+
                                                        +--> E_i -> frozen online decoder
first M_T4 labelled trials     -> analytic T4 fit --------+
```

The exact fusion remains the selected ordinary B3S/T4 path. There is no per-session learned unit
table, no target-session gradient, and no target-session decoder update.

### 2.1 What is trained and what is frozen

The current source trainer constructs the student decoder from the common SPINT teacher state and
copies compatible teacher identity weights into the streaming encoder. With
`freeze_decoder=false`, every student encoder and decoder parameter is trainable during source
training. This is a teacher-initialized joint fit, not a random-from-scratch decoder and not a
permanently frozen-decoder experiment.

At held-out evaluation, the selected checkpoint is loaded, all parameters have
`requires_grad=false`, the T4 descriptor is fitted analytically, and B3/T4 identity is computed by
forward evaluation. Thus:

```text
source training:       teacher initialization + encoder/decoder backpropagation
held-out calibration:  closed-form T4 + forward B3/T4 identity, no backpropagation
online query:           cached identity + frozen decoder forward
```

Any result must state all three phases. “No backpropagation” without the held-out-calibration
qualifier is misleading.

## 3. Why T4 is unusually compatible with SUA-to-MUA pooling

Pseudo-MUA is a deterministic sum of sorted SUA spike counts within an electrode. Let `G_k` be
the set of SUA units pooled into electrode channel `k`. Because the cosine regression design is
shared across those units and least-squares coefficients are linear in the response,

```text
a_k^MUA = sum_{i in G_k} a_i^SUA
c_k^MUA = sum_{i in G_k} c_i^SUA
b_k^MUA = sum_{i in G_k} b_i^SUA
m_k^MUA = sqrt((a_k^MUA)^2 + (c_k^MUA)^2).
```

This is not merely a theoretical convenience. The fresh 33-session C1 data audit reports:

- exact online binned-count conservation, maximum error `0`;
- calibration-count maximum error `4.77e-7`;
- pooled-rate T4 maximum error `4.68e-6`;
- 693 singleton-electrode identity checks;
- bitwise-equal behavior targets and valid time/window indices in both views;
- distinct train-only normalizer hashes for SUA and pseudo-MUA.

Therefore T4 is **aggregation-homomorphic in `[a,c,b]`**, with the expected nonlinear recomputation
of `m`. This is the strongest mechanistic reason presently available for testing one weight set
across sorted SUA and electrode-pooled pseudo-MUA: the functional carrier has a known transformation
law under unit merge, rather than an assumed similarity between two unrelated modalities.

Evidence: `results/t4_paired_view_c1_data_audit_v1_20260804/receipt.json`, SHA-256
`9542a393aa58176357730cb6c565241ba6696c160ad6b336d6df748c7ac8acdb`.

## 4. Accuracy evidence and boundary results

### 4.1 Sorted SUA development sessions

Under the strict 27 source / 6 reused-development / 6 sealed-formal partition, ordinary T4 is a
large and repeatable effect:

| Contrast | Mean delta R2 | Session sign | Seed sign | Interpretation |
|---|---:|---:|---:|---|
| T4 - F0 | about `+0.2528` | 6/6 positive | 3/3 positive | labelled functional calibration adds substantial held-out-session value |
| T4 - TS4 | about `+0.2522` | 6/6 positive | 3/3 positive | correct unit attachment/content, not width alone, is necessary |

The completed component attribution sharpens the mechanism:

| Arm | Mean R2 |
|---|---:|
| full T4 | `0.574976` |
| AC4 `[a,c,0,0]` | `0.562753` |
| Z4 | `0.326008` |

`T4-AC4=+0.012222`, so AC4 stays within the frozen `0.03` sufficiency margin. Conversely,
row-shuffling AC4 gives `AC4-RS4=+0.294163`, positive for all 3 seeds and all 6 sessions. The
dominant supported content is therefore the correctly attached first-harmonic coefficient pair
`[a,c]`. Baseline rate and modulation magnitude can be useful diagnostics, but current data do
not require them to explain the main SUA gain.

These are development-held-out sessions relative to source training, not a fresh formal SUA
confirmation. C1/QAT did not resolve or open the formal raw files, but the same six-session
statistical scope had already been consumed by the earlier P3 `status="started"` receipt. The two
facts must not be conflated: current-run file isolation passed, while a new independent one-shot
claim requires another subject/dataset/scope. See
`docs/SUA_FORMAL_ONE_SHOT_READINESS_AUDIT_20260805.md`.

### 4.2 Fresh `sub-M` cross-animal endpoint readiness

Because the six reused `sub-C` development sessions cannot supply a fresh confirmation, a new
subject scope was frozen before raw-data access from immutable DANDI 000688 version
`0.250122.1735`: every one of the 22 public `sub-M` center-out assets, totalling
`2,312,360,648` bytes. Asset IDs, paths, byte counts, etags and SHA-256 values were fixed before
NWB content inspection. This is a **same-Dandiset cross-animal** scope; it is not an independent
dataset or laboratory.

The corrected append-only score-blind preflight verified all 22 files against their frozen hashes
before opening them and produced one immutable common eligible cohort of **15 sessions**. Every
eligible session satisfies `0<N<100`, valid event-level spikes, finite 2-D `cm/s` cursor velocity,
chronological rewarded trials, a complete post-trial-50 query interval, rank-3 first-50
`[cos(theta),sin(theta),1]` design, and a one-electrode-per-unit deterministic pooling map. Seven
sessions are excluded by the pre-frozen `N<100` architecture gate with unit counts
`114,148,159,118,115,129,145`; one of those also contains nonfinite direction labels on two usable
rewarded trials. These are compatibility exclusions, not negative decoder scores. Therefore any
future positive result remains limited to the `N<100` regime.

The external mechanism endpoint is already frozen to terminal `shared_t4` versus matched
`shared_ts4`, seeds 42/43/44, the first 50 rewarded calibration trials, and all valid query windows
strictly after trial 50. SUA is primary; deterministic electrode-pooled pseudo-MUA is a required
secondary rather than a rescue endpoint. In each view, success requires mean paired
`T4-TS4 >= +0.03`, all three seed means positive, at least **12/15** session means positive, a
positive hierarchical-bootstrap lower bound, and positive absolute T4 grand/seed means. The
preflight loaded zero checkpoints and performed zero forwards, predictions, scores, optimizer
steps, backwards, parameter updates or GPU work. Its receipt SHA-256 is
`1d2520188f0b5b4f6827816e380abf814c5796687b15c8e18df1352749157283`.

This closes data/schema feasibility only. The first score-only runner was rejected after
adversarial review; append-only v2 now seals exact per-asset query counts, full TS4 permutations,
one authorized CPU/CUDA device identity, full normalizer/session/path provenance, float32
prediction/target arrays, and independent CPU TorchMetrics recomputation for all 180 cells. Its
final v1+v2 regression is 21/21, but its immutable prelaunch remains explicitly
`NOT_AUTHORIZED_FOR_SCORING`: a parity receipt bound to the final v2 scorer semantics and a real
Ed25519 root are both absent. The older parity-prelaunch v1 binds an intermediate v2 source hash
and is stale/non-authorizing. A second static parity draft is also non-authorizing: it sends one
reference-prepared batch through both forward wrappers and therefore does not exercise the
scorer's distinct loader/data-adapter path. That review also exposed an important budget error in
the old scorer: activity identity must use rewarded trials `first_n30`, while T4 labels/rates use
the first 50 and scoring begins only after trial 50. The next parity must compare the existing C1
and scorer loaders on one already-consumed sub-C fixture and fail closed on any chronology,
valid-start, neural, behavior, Q30 calibration, T4-row, electrode-map, prediction, target or metric
difference. That append-only v3 static package is now implemented and root-reviewed: the combined
v1/v2/v3 static suite passes 11/11, its dry run permits zero checkpoint/NWB/forward operations,
and its current execute mode fails closed before runtime-helper import. Its immutable
draft/receipt/seal hashes are `96200f47...e6157`, `c52796b1...3d8b`, and
`e0088a63...cb1`. It still needs a separately signed one-time CPU capability and an actually
successful consumed-sub-C dual-loader run before it can serve as the scorer parity pin. No
external `sub-M` R2 exists yet.

The first signed v4 CPU execution then consumed its one-time nonce and exposed a concrete adapter
schema defect before model checkpoint load, forward or R2: the adapter owner represents integral
bin coordinates as float-typed `start/stop`, whereas the existing C1 calibration builder uses them
as NumPy slice indices. The exclusive output root is empty and no external sub-M asset was opened.
This is a fail-closed compatibility incident, not an accuracy result; its receipt SHA is
`4c676146...d0bc`, and the capability cannot be replayed. The only allowed repair is append-only:
preserve the raw chronology, prove each coordinate finite/exact-integral/int64-safe, cast only the
two builder-facing slice fields, and run under a new signed single-use CPU capability.

That repair is now complete as an append-only V5/V5R2 schema bridge. It preserves each raw
chronology coordinate and changes only builder-facing `start/stop` values after proving them finite,
exact-integral and int64-safe. A new signed single-use **CPU parity** run on the already-consumed
`sub-C` development fixture produced exact reference/adapter predictions and targets, with
`input_exact=true`, 456 forward calls, and exactly equal R2
`0.4910046458244324`. The sealed parity receipt SHA-256 is
`faace6493fcf939e87bf0f5ad219f75df739434b7941fccc7f654c23d78c63e8`. This establishes the
loader/scorer adapter equivalence required before considering an external scorer; it is **not** a
scientific T4 gain and does not open or score `sub-M`
(`external_subm_scoring_performed=false`). Consequently external-subject R2 remains unobserved.

### 4.3 Deterministic pseudo-MUA bridge

Separately trained pseudo-MUA has:

| Contrast | Mean delta R2 | Decision |
|---|---:|---|
| T4 - F0 | `+0.317739` | effective |
| T4 - TS4 | `+0.365674` | effective |

This proves that T4 survives a controlled unit-to-electrode merge. It does not prove native
threshold-crossing MUA generalization, because the recordings and spikes originate from the SUA
dataset. The residual SUA-versus-pseudo-MUA difference after T4 is small enough that improving SUA
sorting metadata alone is no longer the most plausible high-value route.

### 4.4 Native FALCON M2 MUA

The strongest external result is official EvalAI M2 submission `578221`:

| Official endpoint | T4 | original SPINT | Delta |
|---|---:|---:|---:|
| held-out R2 mean | `0.30324395` | `0.18647872` | `+0.11676523` |
| held-in R2 mean | `0.58760827` | `0.56824267` | `+0.01936560` |
| normalized online latency | `0.04290260` | `0.11236103` | `-0.06945843` |

This is an important end-to-end, organizer-hidden positive result. It also directly answers the
online-overhead concern: cached T4 identity did not make online decoding slower in the submitted
system.

The causal attribution is incomplete because the original SPINT image packages an epoch-27
decoder while T4 uses an epoch-34 teacher decoder. Local matched epoch-34, M33/q33 replay gives
`T4-B0=+0.06420` on 3/4 eligible sessions and `T4-TS4=+0.09558` on 4/4, but only four sessions
have a genuine post-M33 query. A matched epoch-34 B0 hidden submission would be the cleanest
remaining formal attribution control, but it is currently **constructible rather than
submission-ready** and requires separate explicit submission authorization after packaging.

Here B0 must not mean zero identity. The matched arm derives the original-SPINT activity identity
from the same chronological first 33 neural trials using the epoch-34 teacher path
`fc_id_in -> trial mean -> fc_id_out`, while reading no target directions and no T4 descriptor.
The packaged T4 decoder has been verified bit-exact against all 31 epoch-34 teacher decoder tensors,
so this control is technically possible. The current local `b0_baseline` is nevertheless ineligible:
it records random calibration and has no q33 cached-identity payload, isolated image, container-parity
receipt, or guarded submit helper. Finally, epoch 34 is a legacy held-out-selected teacher; a future
authorized result can remove the decoder-version confound but cannot be called independent
clean-teacher confirmation. See
`docs/M2_MATCHED_EPOCH34_B0_EVALAI_ELIGIBILITY_20260804.md`, SHA-256
`ee26160160f76dd13e3abbd294154b95e421a5242ba51294063c920d5dbaf31b`.

At M24, ordinary T4 versus matched clean SPINT has a positive mean candidate
`+0.036604`, but only 2/3 seed means and 4/6 seed-averaged sessions are positive; its frozen
stability gate failed. Joint source training of decoder plus T4 gives only `+0.018389` over a
same-width zero4 control on one screened seed, again below the `+0.03` margin. These local results
show that source co-adaptation is not sufficient by itself to guarantee a stable native-MUA gain.

### 4.5 Native FALCON M1 MUA

M1 does not support the same claim. Official private-test T4 is `-0.003825` below original SPINT
held-out R2, and the earlier local M1 endpoints were support/query-overlapped or had no future
query. Narrow direction coverage and the ten-trial budget make a first-harmonic extrapolator
particularly fragile. T4 should not be advertised as uniformly effective across M1 and M2.

## 5. Label budget and estimator stability

The extra supervision is small in representation width but real in information content. A T4
session uses `M_T4` target-direction labels plus per-trial channel/unit rates. The decoder never
sees query labels. Ordinary B3 sees only neural activity support.

Measured split-half behavior explains the low-budget limit:

- `[a,c]` reliability is roughly `0.59--0.62` at `M=10`;
- it rises to about `0.86--0.87` at `M=50`;
- `b` is already near `0.99` reliability;
- direction balance and design condition matter in addition to nominal `M`.

Thus “T4 needs many labels” is too coarse. It needs enough **directionally informative** labelled
trials to make the vector coefficient reproducible. M2 M33 and SUA M50 are supported operating
points; M10/M15 cannot inherit that claim.

Two attempts to learn away low-budget error did not earn a decoder run:

1. the `q_unit+M` cross-budget error predictor improved nested source LOSO but collapsed on six
   target-free development sessions (`q_unit+M - M-only=-0.0226`, only `3/6` favorable, 95%
   interval `[-0.3268,+0.2817]` in descriptor-MSE units);
2. empirical-Bayes ridge, second harmonic, and Poisson IRLS produced no source-only winner in
   Experiment B v7. EB ridge was closest in prospective deviance (`0.976601` ratio), but its
   directional-reliability change was only `+0.000124`, far below the frozen `+0.02` gate; all
   three estimators reached the predeclared fail-fast boundary.

The correct current conclusion is a label/geometry boundary, not that a wider correction MLP has
been left untried. Any future correction should be tested on an independent subject and respect
rotation equivariance, for example scalar shrinkage of complex `z=a+ic`; it is not licensed on
the already-viewed sub-C development sessions.

This is a terminal decision for the present protocols, not a proof that calibration-budget
correction is impossible. The complete read-only evidence audit is
`docs/PATH12_TERMINAL_GATE_AUDIT_20260804.md`, SHA-256
`cbb660fdfe1cb083f6b30c4fb86395ace6e475434b9c78dbd1eda84e2c9e061a`.

## 6. Why the fixed-K temporal-memory route is closed

The fixed-K prototype carrier initially beat rate-only on the M1 source proxy by about `+0.1005`.
The decisive control was simpler B20: order-invariant sorted log-rates and count quantiles scored
`0.904758`, versus `0.862794` for the temporal prototype. `P20-B20=-0.041963`, negative in all
four sessions with a confidence interval below zero.

The evidence therefore favors nonlinear marginal rate/count distribution, not chronological
prototype order. Adding K/V memory and cross-attention would introduce both a new carrier and a
new consumer after the carrier's temporal-content gate failed. At `N=64`, P20 requires 1,728 FP32
scalars (`6,912` bytes) of calibration-stream state and would emit 1,280 FP32 carrier values
(`5,120` bytes) if a trained consumer existed. Ordinary T4 stores only 256 FP32 descriptor values
(`1,024` bytes). These figures exclude the unimplemented attention projections and activation
cache, so the proposed K/V route has neither an accuracy license nor a complete deployment-cost
receipt.

The branch can reopen only if an independent temporal-order carrier beats both B20 and a
within-trial time-order null. No such result currently exists.

## 7. Paired-view C1: completed positive shared-granularity result

C1 trains one shared B3S/T4 encoder-decoder on exactly aligned SUA and pseudo-MUA microbatches:

```text
L_C1 = 0.5 * L_task(SUA) + 0.5 * L_task(pseudo-MUA)
lambda_consistency = 0
```

Each pair performs two sequential half-weight backward passes followed by one shared optimizer
step. There is no view-specific head and no new fusion path. Sequential backward keeps peak
activation residency close to the larger single-view pass while producing the accumulated
gradient of the fixed two-view objective.

The primary question is not whether sharing creates a new peak R2. It is whether one weight set is
non-inferior to separately trained view-specific weights under a known unit-merge transform.
Frozen gates are:

1. shared T4 minus separate T4 has a paired lower bound at least `-0.03` in SUA;
2. the same non-inferiority gate passes in pseudo-MUA;
3. shared T4 beats shared TS4 in both views with positive paired bounds and stable signs;
4. sharing does not increase the absolute cross-view gap by more than `+0.03`;
5. all 12 cells use seeds 42/43/44, the fixed epoch 5--12 estimator, six held-out development
   sessions, and no formal session.

The fresh 12-cell matrix is complete and passed every frozen C1 gate. Its absolute shared-T4 means
are `0.574378` for SUA and `0.546109` for deterministic pseudo-MUA. Shared versus separately
trained T4 is `+0.008351` on SUA and `+0.012543` on pseudo-MUA, supporting non-inferiority rather
than a robust accuracy-improvement claim. Correct functional-row attachment is decisive:
shared-T4 minus shared-TS4 is `+0.285474` on SUA and `+0.369468` on pseudo-MUA, with all seed and
session means positive. The aggregate SHA-256 is
`32ebde0b145c63c09c1bdbc67e48582db3e2ad588e70cb5a1d52914352607e31`.

C2 is now closed for the current scope. A separate read-only audit found no pre-existing unique
source-only lambda rule, no frozen loss normalization, no C2 implementation/selector/prelaunch, and
conflicting T4/TS4 matrix definitions. This is a 0-GPU identifiability NO-GO, not a negative C2 R2
result. The FP32 C1 candidate remains `lambda_consistency=0`; see
`docs/SUA_C2_SOURCE_ONLY_DISPOSITION_AUDIT_20260805.md`.

The inferential unit is the paired `seed x held-out-session` score after averaging the fixed eight
checkpoint window, not an individual epoch or query window. The aggregate requires both a paired
two-standard-error bound across the three seed means and a two-way seed/session bootstrap bound;
the T4-content gate additionally requires all three seed means and at least five of six session
means to have the correct sign. TS4 has the same four-dimensional input and model width as T4 and
only breaks row attachment, so this comparison cannot be explained by parameter count. With only
three seeds and six reused development sessions, however, these bounds are a conservative decision
rule rather than population-level confirmation; any publication-wide generalization beyond sub-C
still needs the independently scoped subject/formal evidence already identified above.

## 8. Accuracy/cost accounting

The fresh C1 cost audit gives the following reference at `N=64`:

| Quantity | Value |
|---|---:|
| student parameters | `4,613,178` |
| encoder parameters | `18,290` |
| decoder parameters | `4,594,888` |
| FP32 student weights | `18,452,712` bytes |
| B3S activity-calibration MAC/session | `4,841,472` |
| coupled decoder MAC/online window | `57,970,688` |
| T4 fitted descriptor during calibration/finalize | `1,024` bytes |
| finalized identity `E` | `12,800` bytes |
| B3 activity support state | `16,384` bytes |
| activity trial buffer | `25,600` bytes |

T4 fitting is a small `O(N*M_T4)` accumulation plus a shared three-column regression solve; it
is not included in the neural MAC counter and must be reported separately in any hardware paper.
The dominant online MAC cost is the existing coupled decoder, not T4. C1 doubles view forwards
during offline source training, but deployment keeps one weight copy and adds zero state relative
to one ordinary separate T4 model.

The fitted descriptor and the finalized identity must not be blindly added as permanent online
state. In the ordinary coupled path, `[a,c,m,b]` is consumed while finalizing `E`; after that one
forward pass, future neural windows require the cached `E` but neither the raw calibration trials
nor the descriptor. A hardware implementation may retain the descriptor for audit or later
re-finalization, but that is optional state and must be labelled separately. Native-M2 Phase C is
therefore required to report calibration peak state, descriptor bytes, finalized-identity bytes,
and post-finalize required state as distinct quantities.

Evidence: `results/t4_paired_view_c1_model_cost_audit_v1_20260804/receipt.json`, SHA-256
`95cba856e3e4930bc2f7f978d0ed9b09d7c2d0ef8a76217996c8ccc4cf7b290d`.

### 8.1 Encoder-only INT8 boundary

The first encoder-only PTQ pass quantizes the B3S/T4 identity encoder to W8A8 with INT32
accumulation and integer requantization while leaving the decoder FP32 and byte-identical. Across
the three C1 seeds, mean INT8-minus-FP32 R2 is `+0.000803` for SUA and `+0.005195` for
pseudo-MUA; overflow count is zero and the independently implemented integer `E_q` codes match
exactly. Thus PTQ did not create an accuracy problem.

It did not pass the complete pre-frozen hardware gate: maximum edge saturation is
`0.022727 > 0.005`, and a legacy exact-zero comparison of dequantized floating values observes a
maximum discrepancy of about `1.91e-6`. The latter is reported separately from integer-code
parity and must not erase the original PTQ failure. Local eight-epoch encoder-only QAT seeds42/43
are now complete. Seed43 passes the full strict gate: SUA/pseudo-MUA INT8-minus-FP32 is
`+0.014480/+0.025030`, both 6/6 sessions positive, integer mismatches and INT32 overflows are zero,
maximum saturation is `2.5641e-5`, and the decoder is byte-identical FP32. Seed42 passes accuracy
(`+0.006273/-0.004463`, both above `-0.01`), parity, overflow, decoder and legacy exact-zero gates,
but fails the frozen saturation gate because pseudo-MUA input saturation is
`0.029291 > 0.005`. Its learned input scale (`0.015520`) is much smaller than seed43
(`0.036013`), identifying a cross-view range-floor problem rather than an accuracy or integer-code
problem. This failure is not reclassified. Remote seed44 remains unstarted because the 16 newly
required transfer items are still 0 present/16 missing/zero mismatch; the uniform three-seed QAT
verdict therefore remains open. Decoder tensors remain frozen FP32,
source-session labels may enter offline QAT, and no held-out-session optimizer/backward step is
introduced. QAT changes the deployable arithmetic of the identity encoder, not the T4
calibration information budget or post-finalize state contract. A successful C1 QAT result would
establish an encoder hardware path for the controlled SUA/pseudo-MUA bridge; it would not by
itself prove native-MUA accuracy or external-subject generalization.

At 09:14 HKT the remote state was re-audited and remained 44/60 exact with the same 16 missing
items and zero mismatches. The remote 5070 Ti, dedicated `spint` Python/CUDA environment and tmux
are healthy, so this is an input-closure block rather than a hardware block. Locally, both 3090s
remain occupied by the signed native-M2 Phase-C Stage-A matrix. The matched shared-zero4 source
program is now signed and queued behind a strict whole-matrix gate: exact fixed-14 verification,
zero failures, both bound shard parents exited, and both physical GPUs compute-idle. Its queue
preflight opened no formal/sub-M files and claimed no nonce. This ordering preserves the scientific
comparison while preventing either GPU overlap or an idle-gap race between consecutive Phase-C
cells.

### 8.2 Native-M2 label and computational boundary

Phase-C makes the deployment information asymmetry explicit. Both clean-SPINT and T4 receive the
same chronological first 33 neural trials and both score only post-33 query windows with a full
50-bin history. T4 additionally reads one target-direction metadata value per directional
calibration trial; it does not read dense per-bin velocity labels. In every one of the seven M2
sessions, the first 33 trials contain exactly 16 finite directional trials and 17 centre/rest
trials. Centre/rest trials remain unlabeled rather than being assigned an arbitrary direction.
Thus the held-out fit uses 16 direction labels to estimate three cosine-design coefficients for
all 96 channels simultaneously. The audited design is full rank in all seven sessions, with
2-norm condition numbers from `1.5837` to `2.0900`. This is labelled few-shot calibration, not
label-free calibration, but it is not a densely labelled bin-level adaptation procedure.

The deployment fit is analytic and BP-free: calibration-only spike sums and valid trial lengths
form per-channel rates, the fixed design `[1, cos(theta), sin(theta)]` is solved once by least
squares, and `[a,c,sqrt(a^2+c^2),b]` is standardized with statistics fitted only on the six outer
source sessions. The resulting descriptor and the same 33 neural trials are consumed by one
frozen identity-finalization forward. No held-out optimizer, gradient, backward pass or weight
update is introduced. This qualifier is essential: the T4-aware encoder/decoder is trained
offline with backpropagation on source sessions; only held-out-session calibration is BP-free.

Static exact counting also contradicts the hypothesis that T4 necessarily adds a large deployment
cost. At native-M2 dimensions, both arms share `84,021,248` online MACs per query window. The
clean-SPINT support calibration costs `1,875,935,232` MACs, whereas T4 uses `21,393,408` support-
encoder MACs plus only `10,692` MACs for the AC4 fit (`21,404,100` total): about `87.6x` less than
the matched SPINT support path. Peak FP32 stream-calibration live state is `64,552` bytes for T4
versus `235,008` bytes for SPINT, a `72.5%` reduction. After finalization, raw support and the
explicit descriptor are released; online inference retains only the cached `1 x 96 x 50` FP32
identity (`19,200` bytes) and receives neither support nor descriptor in each query batch.

These are source-bound analytical counts and an exact state contract, not yet a full production
latency claim. The prelaunch cost gate is passed, but the cost receipt correctly blocks a measured
production-efficiency claim until all required cells carry valid source and deployment runtime
evidence. Accuracy likewise remains conditional on the sealed Phase-C paired result.

## 9. Publication claims and limits

| Claim | Current status |
|---|---|
| T4 provides strong held-out-session development gain on sorted SUA | supported |
| the main SUA content is correctly attached first-harmonic `[a,c]` | supported by AC4 and row-shuffle controls |
| T4 is algebraically compatible with deterministic SUA-to-electrode pooling | supported analytically and by 33-session audit |
| T4 is useful on organizer-hidden native M2 | supported as an end-to-end system result |
| official M2 gain is a pure T4-only effect | not yet supported; decoder epoch is confounded |
| T4 is stable on native M1 | contradicted by official M1 result |
| low-budget learned correction improves T4 | not supported; branch stopped |
| temporal fixed-K K/V memory is useful | not supported; simpler B20 wins |
| one shared weight set serves both SUA and pseudo-MUA without material accuracy loss | supported by complete C1 non-inferiority gates |
| T4 attachment generalizes from `sub-C` to a new animal `sub-M` | not yet scored; N=15 compatibility is frozen, but a matched shared-zero4 source model is required before the three-arm endpoint can be authorized |
| pseudo-MUA proves native MUA generalization | false; it is a controlled granularity bridge |
| calibration uses no backpropagation | supported only with the qualifier “held-out-session calibration” |

The carrier/estimator/consumer decomposition, aggregation proof, label-information analysis, and
evidence-filtered list of pre-closure ideas are maintained separately in
[`BP_FREE_CALIBRATION_MECHANISM_AND_PRE_CLOSURE_IDEA_AUDIT_20260804.md`](BP_FREE_CALIBRATION_MECHANISM_AND_PRE_CLOSURE_IDEA_AUDIT_20260804.md).
That audit is explanatory and does not authorize any additional arm.

## 10. Remaining decision sequence

```text
completed C1 FP32 aggregate
    -> encoder-only PTQ accuracy is preserved, but frozen saturation gate fails
       -> finish fixed 8-epoch QAT for seeds 42/43/44
          -> apply accuracy/saturation/overflow/code-parity/decoder-integrity gates once

independently:
fresh native-M2 Phase-C r5 launch chain is signed and independently verified GO
    -> wait for its bound local 3090s; do not copy local auth to the remote 5070 Ti
       -> exact seed-42 7 outer sessions x paired {clean SPINT,T4} = 14 cells
          -> open the one predeclared severe-negative decision
             |-- mean delta <= -0.03 or <=1/7 positive: stop native local matrix
             '-- otherwise: continue without a positive claim to seeds 43/44
                -> exact 42-cell aggregate and six frozen positive gates

in parallel, without reusing consumed sub-C formal scope:
    -> frozen DANDI 000688 sub-M CO scope passes score-blind compatibility at N=15
       -> scorer parity passes exactly on the already-consumed sub-C fixture
          -> train and independently audit C1-matched shared-zero4 seeds42/43/44
             -> freeze shared-T4/shared-zero4/shared-TS4 before any sub-M score open
                -> issue one fresh write-once scoring authorization
                   -> open the pre-frozen three-arm endpoint once

after those closures:
    -> report sub-M only as same-Dandiset cross-animal evidence
    -> do not revive stopped fusion/correction/memory arms without new independent carrier evidence
```

The native-M2 local confirmation is deliberately independent of the C1 verdict. It uses native
threshold-crossing MUA, seven outer-session-left-out development folds, chronological support
`[0,33)`, and query `[33,end)` with complete 50-bin history after the boundary. SPINT receives the
same 33 neural support trials without target labels; T4 receives those neural trials plus the
available rewarded-trial direction labels. Source epoch selection is exact-six-source-only, the
paired T4 decoder is the same-fold/same-seed selected SPINT decoder, and target calibration is
required to report zero optimizer steps, backward calls, and updated parameter tensors. This is
the missing local causal bridge between the clean SUA mechanism evidence and the confounded but
organizer-hidden native-M2 system result. It remains development evidence, not a new hidden-test
or formal-subject confirmation.

The external three-arm supplement separates two claims that a two-arm T4/TS4 endpoint conflates.
`T4-zero4` tests whether calibrated descriptor content helps relative to an information-free,
same-width side interface under the same shared offline training recipe. `T4-TS4` tests whether
that content must be attached to the correct unit/channel row. A positive T4/TS4 difference alone
cannot establish the first claim because row shuffling may actively harm the decoder. At target
calibration, shared-zero4 must be produced by a dedicated direct-zero path: it may use first-30
neural trials for the ordinary B3 activity identity, but its descriptor/prediction path consumes
zero target-direction values, performs zero T4 trial-rate reads/fits, zero optimizer steps, and
zero backward calls. An owner/scoring loader may retain target-direction metadata solely for a
noncausal receipt; label shuffle/drop must leave first-30 selection, calibration tensors,
predictions, and R2 invariant. The ordinary source-only T4
normalizer is retained only as a coordinate/interface provenance reference and is not numerically
applied to raw zero values.

The method is already meaningful for both SUA and MUA, but in different senses: SUA supplies the
cleanest mechanism/attachment evidence, while native M2 supplies the strongest organizer-hidden
deployment result. C1 attempts to connect them through a transformation with a proven algebraic
contract. It should not erase the dataset-specific limits or be described as native-MUA proof.
