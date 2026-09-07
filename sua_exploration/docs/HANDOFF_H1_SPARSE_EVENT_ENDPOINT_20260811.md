# Handoff — H1 Sparse Movement-Event Endpoint Carrier

**Status:** V1 CPU gate sealed negative; V2 CPU gate independently verified positive; ten
post-V2 source-only CPU design screens sealed below the original material GPU-entry threshold;
matched M4 H-SE5/Zero5 first cell and the full independent audit chain completed with
`PASS_HSE5_FIRST_CELL`. That decoder result supplied new translation evidence and re-opened exactly
one five-wide fixed `ser_context_q4` development cell; no other carrier arm is queued  
**Date:** 2026-08-11; terminal update 2026-08-12  
**Review request:** audit the endpoint/event semantics, sparse-label accounting, and the deliberately
limited interpretation of the positive first cell

## 1. What changed

The old H1 Version-B candidate represented one session by eight phase-average neural profiles and
required every phase category in each split. It failed constructibility when native phases were
missing. The new candidate does not average by phase and never requires all phase names.

Each valid native movement epoch is one calibration observation:

1. read its native 7-DoF position at event start and stop from
   `acquisition/OpenLoopKinematics`;
2. form one endpoint displacement `Delta q_e`;
3. count each channel's spikes during that event and use `log1p(rate)`;
4. project displacement into a source-frozen low-dimensional basis;
5. fit one closed-form ridge row per channel.

The deployed V2 descriptor is `[w1,w2,w3,w4,b]`. No target-session neural-network optimizer step or
backward pass is used. The native dense velocity series is not opened by the carrier parser.

This is an event-level sparse calibration method, not a trial-scalar method. H1 supplies roughly
13–18 valid events in the first three trials and 17–23 in the first four trials.

## 2. Native-data facts checked before implementation

The released H1 NWBs contain:

- `OpenLoopKinematics [T,7]`, description `tx,ty,tz,rx,g1,g2,g3` — native position/state;
- `OpenLoopKinematicsVelocity [T,7]` — the dense target used by the existing carrier and decoder;
- native `epochs` with movement tags;
- `TrialNum`, `eval_mask`, and 176 unit spike-time rows.

The sparse parser accesses the first item, epochs, trial/eval metadata, and spikes. It deliberately
does not access `OpenLoopKinematicsVelocity`. Cross-trial epochs are excluded rather than clipped;
missing phase categories are allowed. A valid epoch needs at least five eval-valid 20-ms samples
and readable start/stop positions.

The endpoint label is therefore native position differencing, not integration of a dense velocity
trace. Reviewers should still verify that treating `OpenLoopKinematics` start/stop states as the
deployment annotation is appropriate for the intended H1 setup.

## 3. Frozen V1 and why it stopped

V1 fixed `q=3`, carrier `[w1,w2,w3,b]`, and normalized ridge `lambda=0.1` before the 13-session
audit. Immutable receipt:

- `sua_exploration/results/h1_sparse_event_endpoint_v1/source_audit.json`
- SHA-256 `de50d18d6905affa73267c557c76b8fcbd52e7667eb5b1ad4a2546ad7a9f148c`
- independent verifier: `verify_h1_sparse_event_endpoint_receipt.py` → `PASS`

| Budget | 3-D retained variance, median/min | correct−label-shuffle, median; signs | correct−intercept, median; signs | Gate |
|---|---:|---:|---:|---|
| M=3 | 0.580 / 0.572 | +0.0970; 13/13 positive | −0.0605; 0/13 positive | STOP |
| M=4 | 0.575 / 0.567 | +0.0804; 13/13 positive | −0.0382; 0/13 positive | STOP |

The correct endpoint pairing clearly contained functional information relative to the endpoint
shuffle, but the estimator overfit enough to lose to a per-channel support-rate intercept. The V1
GPU stop remains in force and has not been rewritten.

## 4. V2 correction and CPU evidence

V2 is explicitly a post-V1 development candidate, not pristine confirmation. It changes only:

- source-normalized ridge `lambda: 0.1 -> 3.0`;
- displacement rank `q: 3 -> 4`;
- carrier width `4 -> 5` by retaining the intercept.

The source basis uses all valid events from non-outer-date public source recordings. Target support
still uses only the first M trials. The stronger shrinkage was motivated by V1's 13/13 absolute
overfit; the fourth component was motivated by V1's failed retained-variance gate. The compact
consumer adds 32 weights relative to the four-dimensional version.

V2 receipt:

- `sua_exploration/results/h1_sparse_event_endpoint_v2/source_audit.json`
- SHA-256 `e4c12cad1e0678dec722bd622fd34a66eef1bf8205e4aac7193e8c968428f47f`
- independent verifier: `verify_h1_sparse_event_endpoint_v2_receipt.py` → `PASS`
- label-accounting correction receipt:
  `sua_exploration/results/h1_sparse_event_endpoint_v2/source_audit_v2r2.json`, SHA-256
  `de4c23ac3fd21f96c68e54b5190538189543be6f7b19e21cb5533665872a28a4`;
  independent correction verifier → `PASS` (26/26 session-budget rows, metrics/gates unchanged)

| Budget | 4-D retained variance, median/min | correct−label-shuffle, mean / median / signs | correct−intercept, mean / median / signs | Gate |
|---|---:|---:|---:|---|
| M=3 | 0.718 / 0.716 | +0.0238 / +0.0251 / 13/13 | +0.0101 / +0.0099 / 13/13 | PASS |
| M=4 | 0.718 / 0.716 | +0.0253 / +0.0253 / 13/13 | +0.0121 / +0.0115 / 13/13 | PASS |

Both leave-largest-absolute-session-out means remain positive. These are event-rate forward-transfer
diagnostics, not decoder R² results. The effect is modest after appropriate shrinkage; the GPU test
asks whether the compact consumer can use it.

## 5. Label-information accounting

Across all 13 recordings at M=3, the receipt counts:

- 2,786 acquisition-level endpoint-position scalars (`2 endpoints × 7 dimensions × events`);
- 1,393 derived displacement scalars;
- 796 projected V2 model-input scalars (`events × 4`);
- 216,153 raw per-bin velocity scalars in the matched first-three-trial dense reference.

The acquisition-coordinate ratio is therefore about 77.6× fewer reads. If the dense path is counted
after five-bin/100-ms averaging rather than as raw coordinate access, the conservative reduction is
about 15.5×. Report both definitions if this enters the paper; do not quote one without its counting
unit.

Important boundary: offline source decoder training still uses dense behavior targets. The reduction
claim applies to **target-session calibration carrier labels**, not to all labels used to train the
decoder offline.

The original V2 receipt inherited V1's three-values/event projected-input field. The immutable
`source_audit_v2r2.json` corrects it to four values/event and independently proves that no metric,
gate, source binding, acquisition count, or other field changed. Use V2r2 for label accounting and
the original V2 receipt for its directly bound runner provenance.

## 6. GPU execution and terminal audit

The first GPU pair uses M=4 because that preserves the existing fold-0 source schedule, the strict
post-four-trial query boundary, and direct comparability to the stored matched H-S and dense H-C
results. M=3 passed the CPU gate but needs its own matched H-S/Zero5 query boundary; it will not be
compared casually with M=4 references.

| Arm | GPU / tmux | Hydra output |
|---|---|---|
| H-SE5 correct | GPU0 / `hse5_full` | `SPINT-main/logs/h1_sparse_event_endpoint_m4_full_s42_v1` |
| H-SE5-Z5 separately trained | GPU1 / `hse5_zero` | `SPINT-main/logs/h1_sparse_event_endpoint_m4_zero_s42_v1` |
| terminal watcher/evaluator | CPU watcher / `hse5_watch` | launches the strict evaluator only after both terminal checkpoints exist and both training tmux sessions exit |
| independent audit chain | CPU/GPU0 watcher / `hse5_audit` | after `hse5_watch` exits, runs the structural verifier and an independent batch-29 Full/Zero5 R² pass |

Both jobs are fold date `19250101`, seed 42, FP32, fixed 50 epochs, no validation selection, batch 32,
and the established 3,610-batch source schedule. Both reached the fixed epoch-49 checkpoint. The
strict query contains 8,965 windows with SHA-256 `665fe535...e4da`.

| Arm/reference | Pooled R² | H-SE5 Full minus arm |
|---|---:|---:|
| H-SE5 Full | `0.500037` | — |
| independently trained Zero5 | `0.471569` | `+0.028469` |
| same-checkpoint Zero5 | `0.488596` | `+0.011442` |
| same-checkpoint row shuffle | `0.487493` | `+0.012544` |
| same-checkpoint label shuffle | `0.492945` | `+0.007092` |
| sealed H-S / SPINT reference | `0.496833` | `+0.003204` |
| sealed dense H-C | `0.525511` | `−0.025474` |

The primary carrier-content comparison is Full versus the independently trained Zero5 arm. It is
positive on both target recordings: `+0.02590` on `ses-19250101T111740` and `+0.03593` on
`ses-19250101T112404`. Full also beats the same-checkpoint label shuffle on both recordings
(`+0.00712`, `+0.00702`). Thus the first cell supports usable sparse endpoint content and correct
attachment/pairing, not merely a five-value width effect. The accuracy claim must remain modest:
the gain over sealed H-S is only `+0.00320`, and H-SE5 remains `0.02547` below the dense H-C arm.
This is a positive development cell, not yet a multi-date or multi-seed generalization claim.

The independent batch-29 pass reproduced Full as `0.500037260` and independently trained Zero5 as
`0.471568626`, with immutable model state and the same 8,965 query windows. The immutable terminal,
structural-audit, recomputation, and program-completion SHAs are respectively `f53f527d...1b443`,
`f2891d0a...bb581`, `2ee6bcdb...55a7e`, and `4ed0fdc7...714b`.

Evaluation initially failed closed because `np.linalg.svd` produced last-bit-different PCA arrays
under a different process/thread state. No mismatched basis was scored. A source-only immutable
snapshot was accepted only after exact recovery of the checkpoint-bound manifest `0393a0...b9a4`,
basis `d639a4...0e83`, and normalizer `c56e0f...ca03`. Snapshot receipt SHA is
`da234143...1f716`; NPZ SHA is `9ae236f9...c4737`, both mode `0444`.

The runtime must set `PYTHONNOUSERSITE=1`: the user-site Torch is CUDA 13 and incompatible with the
installed driver; the conda `spint` environment provides Torch 2.5.1/CUDA 11.8 and passed a real
one-batch GPU forward/backward preflight.

## 7. GPU implementation map

Scientific reference and CPU implementation:

- `sua_exploration/docs/H1_SPARSE_EVENT_ENDPOINT_CARRIER_PROTOCOL_20260811.md`
- `sua_exploration/docs/H1_SPARSE_EVENT_ENDPOINT_V2_ADDENDUM_20260811.md`
- `sua_exploration/mc_maze/h1_sparse_event_endpoint.py`
- `sua_exploration/mc_maze/h1_sparse_event_endpoint_v2.py`
- the two source-audit runners and two independent receipt verifiers under
  `sua_exploration/scripts/`.

GPU path:

- `SPINT-main/src/data/h1_sparse_event_endpoint.py`
- `SPINT-main/src/models/components/h1_sparse_event_spint.py`
- `SPINT-main/src/models/h1_sparse_event_module.py`
- `SPINT-main/configs/data/falcon_h1_sparse_event_endpoint.yaml`
- `SPINT-main/configs/model/falcon_h1_sparse_event_endpoint.yaml`
- `SPINT-main/configs/experiment/h1_sparse_event_endpoint_{full,zero}.yaml`
- `SPINT-main/scripts/h1_sparse_event_endpoint_evaluate.py`
- `SPINT-main/src/data/h1_sparse_event_source_snapshot.py` and
  `SPINT-main/scripts/build_h1_sparse_event_source_snapshot.py` — write-once source-only snapshot
  and fail-closed loader for the exact checkpoint-bound PCA basis/normalizer; the snapshot is used
  by both the terminal evaluator and the independent recomputation
- `SPINT-main/scripts/watch_h1_sparse_event_endpoint_pair.sh`
- `SPINT-main/scripts/watch_h1_sparse_event_endpoint_terminal_audit.sh`
- `sua_exploration/scripts/verify_h1_sparse_event_endpoint_gpu_receipt.py` — independent checkpoint,
  reference, scope, query-accounting, state-immutability, and gate-arithmetic verifier; it does not
  import the terminal evaluator
- `sua_exploration/scripts/recompute_h1_sparse_event_endpoint_gpu_r2.py` — independent Full/Zero5
  prediction pass with batch size 29 and a separate float64 R²/state-digest implementation
- `sua_exploration/scripts/verify_h1_sparse_event_endpoint_program.py` — terminal-gated final
  program verifier combining the source estimator receipts, all four CPU screens, GPU checkpoint
  audit, and independent prediction recomputation; it cannot pass before all terminal artifacts
  exist
- `SPINT-main/scripts/watch_h1_sparse_event_endpoint_program_completion.sh` — tmux
  `hse5_program`; waits for the terminal/audit chain and automatically writes the immutable final
  program-completion artifact without polling or changing either training process
- `sua_exploration/mc_maze/h1_event_carrier_{design_screen,nested_context_v2,meta_basis_v3}.py`
  and their runners — CPU-only pre-GPU event carrier design program
- `sua_exploration/mc_maze/h1_event_carrier_semantic_v4.py` and
  `sua_exploration/scripts/run_h1_event_carrier_semantic_v4.py` — endpoint displacement versus
  endpoint mean-velocity/direction-speed semantic screen
- `sua_exploration/scripts/verify_h1_event_carrier_cpu_screens.py` — independent verifier for all
  three immutable CPU design receipts
- `sua_exploration/scripts/verify_h1_event_carrier_semantic_v4.py` — independent V4 source/hash,
  aggregate, gate, scope, and terminal-status verifier

The data module reuses the matched M4 neural windows, cubic four-trial identity tensors, and fixed
calibration schedule. All 116 legal source support blocks have defined H-SE5 carriers (minimum 17
valid events). The correct and Zero5 arms have identical model topology and source schedule; Zero5
is applied at the model boundary.

Real-data verification is stronger than a source-string assertion: the CPU test suite wraps the
HDF5 file object so any attempted access to `OpenLoopKinematicsVelocity` raises immediately, then
independently reconstructs a parsed event as
`OpenLoopKinematics(stop) - OpenLoopKinematics(start)`. The complete sparse-endpoint CPU suite is
currently `10 passed`; the fixed/nested/meta/semantic carrier-design tests bring the combined CPU
suite to `23 passed`; the SPINT source-cache/query-contract test is independently `1 passed`
(re-run on 2026-08-12 before terminal evaluation), and the immutable-snapshot suite is independently
`2 passed`. The first three immutable screens re-passed their combined independent verifier, and
semantic V4 passed its separate verifier.

## 8. Completed terminal checklist and residual scope

The terminal checklist is complete. Both fixed epoch-49 checkpoints were produced and hash-bound;
the isolated evaluator used byte-identical query windows; the correct checkpoint was scored with
zero, row-shuffled, and label-shuffled carriers; the independently trained Zero5 arm was scored on
the same query set; and the result was checked by both a structural receipt verifier and a separate
batch-29 prediction/R² implementation. Pooled and per-recording effects agree in sign, and the
program-level completion artifact is immutable.

The positive first cell does not automatically authorize an open-ended expansion. Separately
trained LS/RS arms, another seed, and another date are optional precision/generalization work, not
missing implementation. A separate matched M=3 decoder boundary is needed only if the paper keeps
an organizer-budget M=3 accuracy claim; M=3 must not reuse the M=4 H-S reference.

No minival, formal held-out, EvalAI, or organizer-hidden endpoint was opened by this development
run.

## 9. Requirement-by-requirement completion audit

| Requirement | Authoritative evidence | State |
|---|---|---|
| Replace eight-phase average profiles with one observation per valid movement event | `load_event_session()` emits `MovementEvent` rows directly from native epochs; the real-data M3 test has fewer than all eight phase tags and still constructs a valid support set | PROVED |
| Use native low-dimensional endpoint displacement rather than dense velocity integration | Real-data HDF5 proxy test fails on any velocity-series access and independently reproduces `position(stop)-position(start)` to `1e-12`; V2 source basis is rank 4 | PROVED |
| Preserve per-channel functional identity | Each event supplies 176 log-rate responses; the closed-form fit returns `[176,5] = [w1,w2,w3,w4,b]` | PROVED |
| Do not use target-session backpropagation | Estimator is a closed-form normalized ridge solve; terminal checkpoint/evaluator metadata fixes target optimizer and backward steps to zero | PROVED |
| Show content rather than carrier width or baseline rate alone | CPU correct−label-shuffle and correct−intercept are positive on 13/13 recordings at M3 and M4; decoder Full−independently-trained-Zero5 is `+0.028469`, Full−same-checkpoint-label-shuffle is `+0.007092`, and both margins are positive on both target recordings | PROVED for the first M4 cell |
| Keep source/target and query boundaries matched to the existing M4 comparison | Source schedule has 116 supports and 3,610 batches/epoch; target query has 8,965 windows with SHA `665fe535...e4da`; the terminal and independent recomputation bind the same query set | PROVED |
| Quantify sparse calibration information honestly | Immutable V2r2 receipt fixes endpoint, displacement, projected-input, raw-dense, and 100-ms-equivalent counts; offline source behavior supervision is explicitly excluded from the sparse deployment claim | PROVED |
| Screen carrier refinements before any new GPU arm | Ten CPU LODO programs tested fixed, nested, source-meta-learned, endpoint velocity/direction-speed semantics, likelihood/nuisance/prior estimators, affine/quadratic source-to-future correction, low-rank multi-output fitting, nonlinear sparse-label embedding, and support-only nuisance removal; all reproduced H-SE5 exactly and independently closed below the frozen material-effect gate | PROVED; no new GPU arm |
| Produce an interpretable GPU result | Full and independently trained Zero5 finished epoch 49; strict evaluator, independent receipt verifier, and independent batch-29 R² recomputation agree to numerical tolerance | PROVED |
| Decide whether M3 needs a matched decoder cell | M3 has a different query boundary and is not required for the completed M4 first-cell mechanism claim | CONDITIONAL future work only if an M3 organizer-budget accuracy claim is retained |

The implementation objective and the bounded M4 first-cell experiment are therefore closed. The
result establishes a positive sparse-carrier mechanism cell; it does not establish multi-date,
multi-seed, or organizer-hidden generalization.

The final program verifier's static preflight has already returned
`PASS_PREFLIGHT_TERMINAL_PENDING`. Its immutable artifact is
`sua_exploration/results/h1_sparse_event_endpoint_program/H1_SPARSE_EVENT_ENDPOINT_PROGRAM_PREFLIGHT_v1.json`,
SHA-256 `18f02403141c2196874e84a02c0267a8bdf2a5b4127857f485fd1b51141456ae`,
mode `0444`. It binds and live-verifies V1, V2, V2r2, and the four CPU screens that existed when the
GPU program was frozen. Those three formerly pending inputs are now closed by the terminal receipt
`f53f527d...1b443`, structural audit `f2891d0a...bb581`, and independent recomputation
`2ee6bcdb...55a7e`. The final completion artifact is
`sua_exploration/results/h1_sparse_event_endpoint_program/H1_SPARSE_EVENT_ENDPOINT_PROGRAM_COMPLETION_v1.json`,
SHA-256 `4ed0fdc7ca694bcdebf92ad873c91adc23efb47eb3a294cea08d17b95655714b`, mode `0444`.
Estimator V5r2 postdates the immutable preflight and is bound by the corrected supplemental closure
below; no historical artifact was rewritten.

## 10. Third-party review addendum: event-tag mixing risk

The review correctly identifies the main remaining modeling assumption: H-SE5 uses one global
four-dimensional endpoint basis and one shared linear tuning row per channel across Reach, Orient,
SnapTo, Shape, Grasp, Carry, Orient2, and Release. Active 7-DoF coordinates indirectly expose much
of the event type, but the model cannot represent a tag-dependent neural meaning for otherwise
similar displacements. Global q4 compression may also allocate its limited variance across
different phase-specific subspaces. The V1-to-V2 change is consistent with this risk: increasing
retained variance from about 0.58 to 0.718 changed correct−intercept transfer from negative to only
about +0.01.

One point in the review is not accepted: this is not one 880-parameter regression fit from 13–23
observations. It is 176 parallel regressions with a common design, each using 13–23 observations to
fit five coefficients. The ratio is still small and requires shrinkage, but it is not `n < 880`.

The current M4 run remains frozen and scientifically necessary as the tag-blind baseline. It must
not be retrospectively changed. A parameter-matched tag-conditioned endpoint basis (working name
TCE5) was the first proposed follow-up:

```text
phi_e = onehot(tag_e) tensor_product endpoint_displacement_e
z_e = B_source phi_e,       z_e in R^4
log1p(rate_i,e) = b_i + w_i^T z_e
carrier_i = [w_i1, w_i2, w_i3, w_i4, b_i]
```

`B_source` must be learned only from non-outer-date source recordings and must map every tag into
one common latent coordinate system. Eight independent PCAs are not acceptable because their
rotations/signs are not aligned while target coefficients are shared. The target fit remains five
parameters per channel, uses the same event count, needs no dense velocity and no target backward
step, and does not widen decoder state. The event tag is already native epoch metadata.

Per-tag target regressions are rejected because they expand the fit toward 40 coefficients/channel
with only three or four repeats/tag. Per-trial aggregation is rejected because M3/M4 would provide
fewer observations than the five-parameter fit and would erase event-level resolution.

No TCE5 GPU run is implied. The subsequent fixed, nested, and source-meta-learned CPU screens tested
this mechanism with within-trial tag shuffle, endpoint-label shuffle, and intercept controls. Correct
tag content was real, but the best incremental gain over global H-SE5 plateaued at approximately
`+0.012` to `+0.016`, below the predeclared `+0.02` material threshold. TCE5 is therefore closed
before GPU; it is no longer the prioritized follow-up.

### 10.1 CPU design program completed before any new carrier GPU proposal

The follow-up did not lower the material-effect gate. A matched q4 candidate had to beat global
H-SE5 by mean `>= +0.02`, median `>= +0.01`, at least 10/13 positive recordings, and a positive
leave-largest-absolute-recording-out mean at both M3 and M4. It also had to beat intercept,
endpoint-label shuffle, and tag shuffle robustly. Passing this CPU gate still would not itself have
authorized GPU use.

Four immutable CPU screens were completed:

1. fixed representation matrix: global PCA, source-encoding delta, endpoint state/context,
   tag-conditioned delta/context, duration-weighted fitting, plus q7 headroom diagnostics;
2. nested source-only selection of start/midpoint/stop state, source/target ridge, response
   normalization, and session balancing, with each outer date excluded from configuration choice;
3. source-only meta-learned q4 bases optimized through the analytic M3/M4 carrier fit. This last
   screen used offline source backward steps, but target-session fitting remained closed form with
   zero optimizer/backward steps.
4. endpoint label semantics using displacement, mean endpoint velocity, or source-normalized
   direction plus log speed. This used the same two position endpoints/timestamps, did not open a
   within-event trajectory or dense velocity, and kept every carrier five-wide.

| CPU arm | M3: mean/median vs global H-SE5 | M4: mean/median vs global H-SE5 | M3/M4 signs | Mean vs intercept, M3/M4 | Result |
|---|---:|---:|---:|---:|---|
| global H-SE5 | 0 / 0 | 0 / 0 | — | +0.0101 / +0.0121 | baseline |
| source-encoding `delta+midpoint+tag` | +0.0154 / +0.0152 | +0.0144 / +0.0135 | 13/13, 13/13 | +0.0307 / +0.0348 | below +0.02 gate |
| nested context selector | +0.0142 / +0.0127 | +0.0143 / +0.0135 | 13/13, 13/13 | +0.0307 / +0.0334 | below +0.02 gate |
| meta-learned context q4 | +0.0161 / +0.0144 | +0.0123 / +0.0119 | 13/13, 13/13 | +0.0302 / +0.0320 | below +0.02 gate |
| meta-learned tag-delta q4 | +0.0149 / +0.0137 | +0.0119 / +0.0110 | 13/13, 13/13 | +0.0290 / +0.0320 | below +0.02 gate |
| source-encoding mean-velocity q4 | +0.0052 / +0.0059 | +0.0058 / +0.0047 | 8/13, 10/13 | +0.0191 / +0.0196 | below +0.02 gate |
| source-encoding direction-speed q4 | +0.0047 / +0.0061 | +0.0048 / +0.0033 | 9/13, 10/13 | +0.0190 / +0.0208 | below +0.02 gate |

The phase/context mechanism is real in this development scope: the best candidates beat tag
shuffle on 13/13 recordings, with mean margins around +0.018 to +0.049 depending on the feature
map, and beat endpoint-label shuffle on 13/13. However, the improvement over the already
tag-correlated global endpoint carrier plateaued around +0.012 to +0.016. Increasing q4 to q7 did
not improve reliably, duration weighting did not help, nested source-only tuning did not enlarge
the effect, and meta-learning reduced every source objective without increasing outer-date gain
beyond the material gate. Recasting the same endpoint information as mean velocity or
direction-plus-speed produced only about `+0.005` relative to H-SE5, so response/label units explain
at most a small part of the observed plateau in this diagnostic.

The evidence therefore supports a narrow conclusion: event tag and endpoint state contain genuine
additional functional information, but no tested five-wide carrier converts it into a sufficiently
large outer-date forward-transfer gain to justify a new GPU arm. No tag/context carrier GPU run is
queued. The frozen H-SE5 GPU pair remains the only decoder-level translation test.

Immutable receipts and independent verification:

- V1: `sua_exploration/results/h1_event_carrier_design_screen_v1/source_screen.json`, SHA-256
  `74bbc01490432794546e7ca2fd4242fbed6f2a7ebdd65786f56035eaa49bfeb3`;
- V2: `sua_exploration/results/h1_event_carrier_nested_context_v2/source_screen.json`, SHA-256
  `2c273cd78b20b4dc1a43d197588ac5a2b58823d093514196c67069e23584ae2c`;
- V3: `sua_exploration/results/h1_event_carrier_meta_basis_v3/source_screen.json`, SHA-256
  `e52a7cdd6c18c9d39af3d3cfddddd9f340d13bb5e4184feaa8a7f3a92fd80933`;
- V4: `sua_exploration/results/h1_event_carrier_semantic_v4/source_screen.json`, SHA-256
  `5a96b20493b45f448446e44005549f4d1611a11de292299be0c6bb872475f0f3`;
- `verify_h1_event_carrier_cpu_screens.py` independently returned `PASS` for all three receipts,
  including source/code hashes, baseline reproduction, summaries, nested selections, source-loss
  decrease, gates, and terminal STOP states. Its immutable verification artifact is
  `sua_exploration/results/h1_event_carrier_cpu_screens_verification_v1.json`, SHA-256
`521ea2c09b86baba00aa83723ddaa122cf71ef46d1fdafae7fbb0b854bbb590b`.
- `verify_h1_event_carrier_semantic_v4.py` independently returned `PASS` for V4, including exact
  78-value H-SE5 reproduction, live implementation/source hashes, scope, aggregate recomputation,
  and frozen gate arithmetic.

### 10.2 Estimator and source-to-future closure

V5 held the endpoint map, carrier width, support budgets, and later-event scoring fixed and changed
only the target estimator.  Its predeclared arms were a Poisson count model with duration exposure,
a within-trial contrast ridge, and source-date-LODO channel-correspondent EB shrinkage.  The
authoritative V5r2 results are:

| Estimator | M3 delta vs H-SE5, mean / median / signs | M4 delta vs H-SE5, mean / median / signs | Decision |
|---|---:|---:|---|
| Poisson count/exposure IRLS | `-0.10280 / -0.10571 / 0/13` | `-0.07918 / -0.08191 / 0/13` | STOP |
| Within-trial contrast ridge | `-0.00112 / -0.00116 / 5/13` | `+0.00045 / +0.00140 / 7/13` | STOP |
| Channel-correspondent EB shrinkage | `+0.00969 / +0.01169 / 11/13` | `+0.00557 / +0.00518 / 10/13` | below material gate; STOP |

The immutable receipt is
`sua_exploration/results/h1_event_carrier_estimator_v5/source_screen_v2.json`, SHA-256
`6fa4405e29abc478a64f1c03619f89f1bcf31e1a77db314be014b241764d2645`, mode `0444`; the independent
verifier and seven focused tests pass.  H-SE5 was reproduced in 78/78 comparisons with maximum
absolute difference `2.22e-15`.  All Poisson correct and label-shuffle fits converged.  The original
V5r1 receipt (`a50ff370...eeb86`) is preserved but invalidated because root review found a doubled
exposure-offset subtraction and the wrong ridge coefficient covariance; none of its Poisson or EB
numbers may be cited.

V5r2 rules out three narrow explanations for the observed plateau: a log-rate least-squares versus
Poisson likelihood mismatch, a dominant trial-wide rate nuisance removable by simple centering,
and a sufficiently large gain from independent channel-correspondent shrinkage.  It does not rule
out systematic support-to-future estimator bias, cross-channel target-pairing structure, nonlinear
task semantics, or chronological covariate shift.

C2F5 tested a small channel-shared affine/ridge correction from the H-SE5 support coefficient and
seven analytic fit-quality statistics to a later-event coefficient.  Every correction operator and
hyperparameter was learned by nested source-date LODO; the outer target's later labels were used for
scoring only.  It produced a small, pairing-dependent gain but missed the frozen material gate:

| Budget | C2F5−H-SE5 mean / median / signs | leave-largest-absolute-out mean |
|---|---:|---:|
| M3 | `+0.00972 / +0.01073 / 11/13` | `+0.00826` |
| M4 | `+0.00756 / +0.00844 / 11/13` | `+0.00671` |

Correct−label-shuffle remained positive (`+0.02017` at M3; `+0.02358` at M4), so the operator was
not wholly label-blind.  Its “quality-only” control removes `B_support` from the *correction input*
but retains the raw H-SE5 carrier in `B_corrected = B_support + F(quality)`; it must not be described
as a carrier-free arm.  The immutable receipt is
`sua_exploration/results/h1_calibration_future_correction_c2f5/source_screen.json`, SHA-256
`4096221bbfaf0c7c3c39e50a59e47073f48d60d6d8fcc03ea289eea413d829e8`, mode `0444`; independent
verification and eight focused tests pass.  Its terminal state is `STOP_CPU_C2F5_NOT_MATERIAL`.

LRT5 learned an outer-date-excluded source subspace of channel tuning patterns and used correctly
paired target support events to estimate a small shared coefficient matrix.  Its correct arm also
missed the material gate:

| Budget | LRT5−H-SE5 mean / median / signs | leave-largest-absolute-out mean |
|---|---:|---:|
| M3 | `+0.01165 / +0.01272 / 12/13` | `+0.01036` |
| M4 | `+0.00683 / +0.00786 / 10/13` | `+0.00567` |

Root review found that LRT5r1's source-prior-only control used `b=mean(Y)` despite nonzero prior
slopes.  The authoritative r2 control instead uses the fair support-mean alignment
`b=mean(Y)-mean(Z)@W`; the correct LRT5 arm, candidate grid, and H-SE5 reference are unchanged.
After correction, correct−source-prior is only `+0.00027` at M3 and `+0.00198` at M4, making the
target-specific contribution of the low-rank update especially weak.  The authoritative receipt is
`sua_exploration/results/h1_event_carrier_lrt5/source_screen_v2.json`, SHA-256
`cf4a8a6256a9743a8cc0ee26eef62debae28a4a51db48e8ea485063a32ed4729`, mode `0444`; its independent
verifier passes, H-SE5 is reproduced in 78/78 comparisons with maximum error zero, and the focused
test file passes.  LRT5r1 SHA `f19e333d...6ae25bc3a` remains immutable but is superseded for the
source-prior comparison.  The terminal state is `STOP_CPU_LRT5_NOT_MATERIAL`.

NLE5 tested a fixed source-trained nonlinear map from sparse event endpoint, tag, start-time, and
duration annotations into the same four latent coordinates.  Its r1 ontology check stopped because
`Release` and `SnapTo` were zero-count columns.  Root audit showed those tags are absent from every
parsed support and later event, so v2 preserved r1 and predeclared the six actually observed tags
rather than treating globally inactive columns as an unseen-target failure.  The corrected v2
result was negative relative to H-SE5:

| Budget | NLE5−H-SE5 mean / median / signs | correct−label-shuffle | correct−tag-shuffle |
|---|---:|---:|---:|
| M3 | `−0.00899 / −0.00857 / 2/13` | `+0.01605` | `+0.01637` |
| M4 | `−0.00806 / −0.00687 / 2/13` | `+0.01594` | `+0.01281` |

Thus the nonlinear representation contains real tag/endpoint content but transfers worse than the
plain q4 endpoint basis.  The authoritative v2 receipt is
`sua_exploration/results/h1_event_carrier_nle5/source_screen_v2.json`, SHA-256
`45f346decbda51742235efd63cf2feecbdb2650c2a2a141aaa1769117abef3d9`; its verifier and the combined
11 r1/v2 tests pass.  Terminal state: `STOP_CPU_NLE5_V2_NOT_MATERIAL`.

PNO5 tested a fixed support-only trial/population offset before the ordinary H-SE5 ridge fit.  Query
population statistics were forbidden; later raw log-rate was scoring-only.  Its delta versus H-SE5
was effectively zero (`+0.00003` at M3 and `−0.00020` at M4; 7/13 positive at both budgets), and the
nuisance-row-shuffle control also failed.  The immutable receipt is
`sua_exploration/results/h1_event_carrier_pno5/source_screen_v2.json`, SHA-256
`0ad94ca988b92a23b69e5be2d0f3ce42aca41c4d2e4536d3f0ef123cf7fa4f52`; its verifier recomputed the
metrics and four focused tests pass.  Terminal state: `STOP_CPU_PNO5_NOT_MATERIAL`.

QC2F5 was the final bounded source-only screen.  It tested a fixed degree-two correction of C2F5's
support coefficients and analytic quality statistics, motivated by C2F5's small consistent affine
gain.  It reproduced both H-SE5 and linear C2F5 exactly, but the nonlinear increment was negligible:

| Budget | QC2F5−H-SE5 mean / median / signs | QC2F5−linear C2F5 mean / median / signs |
|---|---:|---:|
| M3 | `+0.01104 / +0.01439 / 10/13` | `+0.00132 / +0.00220 / 7/13` |
| M4 | `+0.00962 / +0.00892 / 12/13` | `+0.00206 / +0.00317 / 8/13` |

The mean delta versus H-SE5 remained below `+0.02` at both budgets, M4 also missed the median gate,
and the required advantage over linear C2F5 failed the sign/LOO control.  Label, source-teacher,
quality-only, and intercept controls were mostly positive, so this is not a shuffle-only positive;
the quadratic terms simply did not improve the useful affine mapping.  The immutable receipt is
`sua_exploration/results/h1_calibration_future_quadratic_c2f5/source_screen.json`, SHA-256
`ef8b216c90046ca465c4bd687ea0737eb261c18caa1d787a657feb84a24fa51f`; terminal state
`STOP_CPU_QC2F5_NOT_MATERIAL`.

Root audit found that the receipt's 12 nested tie-break description strings said “smallest ridge”
although the predeclaration and executed function both used “strongest ridge.”  All 12 stored
selected lambdas already match the executed strongest-ridge rule; metrics and gates are unchanged.
The additive immutable correction is
`QC2F5_METADATA_CORRECTION_v1.json`, SHA-256
`5d66a23179bd16adba96a2f224dd4b8d9e80ba389426ebbfed1f9d53cd22e180`; its independent verifier also
confirms that the teacher control permutes complete 5-D teacher rows.  No source data or GPU was
opened for that metadata correction.

At the original CPU-closure boundary, no tested refinement cleared the frozen material threshold,
so H-SE5/Zero5 was then the sole GPU translation experiment. Section 11 records the later
changed-evidence decision and the now-terminal Context Full cell; this historical boundary is not
the current result roster.

The corrected supplemental CPU closure is
`sua_exploration/results/h1_sparse_event_endpoint_program/H1_SPARSE_EVENT_ENDPOINT_EXTENDED_CPU_CLOSURE_v2.json`,
SHA-256 `f284d53e2f956518d367fc1487777de17186955c589a4879be9a8e79d5e93a04`, mode `0444`.
Its independent verifier and focused test pass. It binds the authoritative H-SE5 V2 source audit,
the complementary V2r2 label-accounting receipt, the corrected estimator V5r2 receipt, and every
subsequent C2F5/LRT5/NLE5/PNO5/QC2F5 terminal receipt. It authorizes zero new GPU arms. Closure v1
(SHA-256 `39c2b02d...67786a3`) remains immutable but is superseded because it mislabeled H-SE5 V2/V2r2
as V5 and omitted the true estimator V5 receipts; it is not the final CPU closure.

## 11. Changed-evidence addendum: one fixed context decoder cell

The original `+0.02` CPU mean gate was fixed before any sparse endpoint carrier had been translated
through the compact decoder. H-SE5 subsequently produced a decoder-level Full-minus-independent-
Zero5 margin of `+0.028469` despite an M4 CPU correct-minus-intercept mean of only `+0.012127`.
This does not establish a proportional CPU-to-decoder law—the CPU forward-transfer and behavioral
decoder R² contrasts are different estimands—but it shows that the old threshold was too
conservative as an absolute prerequisite for a single translation test.

One candidate is therefore reopened as an explicitly exploratory, changed-evidence cell:
`ser_context_q4`. At M4 it improves over global H-SE5 by mean/median `+0.014438/+0.013464`, is
positive on 13/13 source recordings, retains a positive leave-largest-recording-out mean
`+0.012661`, and keeps the identical five-value `[w1,w2,w3,w4,b]` decoder interface. Fixed context
is preferred over nested selection and source-meta learning because it has the largest M4 mean,
the simplest source-only operator, and no target-dependent model selection.

The context raw annotation is `[endpoint displacement(7), endpoint midpoint(7), native event-tag
one-hot]`, source-standardized and compressed by the fixed source-encoding rank-4 map. Target
deployment remains a closed-form ridge-3 fit with zero optimizer/backward steps and no dense
velocity carrier access. The immutable fold-0 map SHA is `50c0c559...45e525`; its v3 snapshot also
freezes all 116 source carrier rows so neither SVD nor ridge last-bit variation reaches training.
Snapshot receipt/NPZ SHAs are `6f96206f...ff62e` and `8570fb0e...e2964`.

Only Context Full is trained. The sealed H-SE5 Zero5 is a valid common independently trained null
because a bound preflight proves identical source neural/behavior windows, M4 trial identities,
batch order, calibration-start schedule, model topology, optimizer, seed, initial state, and
`[176,5]` carrier shape; `zero_carrier=True` replaces the carrier with a literal zero tensor before
concatenation. Context/H-SE5 carrier map, cache, and normalizer hashes are expected to differ and
are irrelevant only behind this audited zero boundary. The immutable parity preflight SHA is
`e8bef701...ff411`.

The outcome rule was fixed before training: `Context Full - sealed H-SE5 Full >= +0.010` pooled and
positive on both target recordings is material, provided Full also beats common Zero5 and the
same-checkpoint zero, row, endpoint-label-shuffle, and tag-shuffle controls. A same-sign increment
strictly between zero and `+0.010` is `SMALL_POSITIVE_NONMATERIAL` and does not expand; a nonpositive
increment, target-recording sign inversion, or failed required control stops the route. Any
proportional extrapolation is launch rationale only, never a forecast or acceptance criterion.

### 11.1 Terminal Context Full result and mainline role

Context Full completed fixed epoch 49 and strict evaluation on the exact 8,965-window H-SE5 query.
It scores pooled `0.516518`, compared with H-SE5/H-S/dense H-C
`0.500037/0.496833/0.525511`; thus Context adds `+0.016481` over endpoint-only H-SE5, is
`+0.019685` above SPINT, and remains `−0.008993` below dense H-C. It recovers `64.7%` of the
H-SE5-to-H-C gap and is the selected pooled H1 sparse representative; H-SE5 is retained as the
minimal endpoint-only ablation.

Same-checkpoint tag/endpoint-label/row/zero interventions score
`0.513960/0.499189/0.499152/0.484059`, and independently trained Zero5 scores `0.471569`.
All required pooled content/attachment controls are positive, but the H-SE5-relative recording
deltas are `−0.004610/+0.077458`. The predeclared scientific status is therefore `STOP_CONTEXT`
because the both-recordings-positive clause fails. This status prevents an expansion or a claim of
recording-uniform improvement; it does not erase the verified pooled system result or its role as
the strongest tested endpoint-sparse H1 development system.

The authoritative absolute-path terminal receipt is
`SPINT-main/pilot_artifacts/h1_context_event_carrier/H1_CONTEXT_SER_Q4_M4_FOLD0_TERMINAL_v3.json`,
SHA `0460cd5f965d222041e1bc5f12ac6507b580e3a875174a4ed9bfe6ca10bd5695`, mode `0444`.
The structural verifier and independent batch-29 recomputation pass with SHAs
`51a14cfae428b9ff45e16dab97f87fdf1408f278ee5c56d6b41b129c3987d2bd` and
`98bb86f1055f0544f938363026e9bd0afa4b13cc14f7cffcac4901861f7b3750`.

## 12. Residual third-party review questions

1. Is excluding native epochs that cross `TrialNum` preferable to clipping them, or should Release
   be assigned by a different frozen rule?
2. Is `OpenLoopKinematics` endpoint differencing an acceptable sparse deployment annotation for H1?
3. Is the V2 post-V1 rationale sufficiently separated from confirmatory evidence?
4. Does source-all-event PCA create any objection beyond ordinary offline source training?
5. Should the paper count raw endpoint coordinates versus raw dense bins, or use the conservative
   100-ms derived-label comparison as primary?
6. Is the positive M4 first cell sufficient as a bounded mechanism result, or does a paper-facing
   generalization claim require another predeclared date/seed?
7. Is C2F5's small but label-pairing-dependent `+0.008`--`+0.010` gain worth retaining as a negative
   mechanism audit, or should only its below-gate status appear in the paper-facing record?
8. Does LRT5's near-zero advantage over the mean-aligned static source prior justify treating the
   low-rank result as evidence against target-specific multi-output adaptation at M3/M4?
