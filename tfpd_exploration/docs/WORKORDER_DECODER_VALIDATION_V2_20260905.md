# M2 / H1 / M1 optimized decoder validation V2

Authority: user takeover request on 2026-09-05, including Terra parallel implementation, new inference-speed-optimized decoders, a new M1 carrier, and correct, reasonable local R². This supersedes the earlier draft's no-execution / no-new-training restrictions for this local validation project. It does not authorize new EvalAI submissions, cancellation, or official-score-driven selection.

Result root: `tfpd_exploration/results/decoder_validation_v2/20260905_190000/`.

## Required end state

1. **M2:** optimize the actual SMALL EMA e19 and LARGE RAW e20 temporal decoders, retaining their weight/data/calibration authorities. Demonstrate full local prediction/R² preservation and meaningful end-to-end inference speed improvement, including legal multi-session calls. Do not substitute the old static SPINT decoder.
2. **H1:** repair the training/output unit mismatch, establish real learning rather than near-constant predictions, and validate the speed-optimized temporal design with meaningful local R². Preserve W700, N176, seven output dimensions, canonical M3 calibration, and known-source-development disclosure.
3. **M1:** build a separately named, persisted, source-fitted rSyn3 carrier revision; repair source-only fit and frozen B3 loading; train and validate the optimized temporal decoder locally. Preserve W100, N64, sixteen EMG outputs and M10 calibration. Old P/S-Fix claims and sealed hashes remain immutable.
4. **Shared correctness:** finite causal history, session/mask/bank/cache isolation, unit-permutation symmetry, offline/streaming agreement, explicit train/evaluation units, real-data tests, and reproducible model/data/checkpoint provenance.
5. **Evidence:** report complete local R² with zero/source-mean and established-reference comparisons, both endpoint and selection rules, actual latency distributions and machine/thread context. A synthetic test, tiny overfit, container parity, or merely finite/weakly positive R² is not task completion.

## Execution ownership

| Owner | Files / responsibility | GPU |
|---|---|---|
| Terra M2 | `two_mainlines_long_v1/latency_opt_v2/`, `m2_optimized_v2/`, corresponding tests/scripts and `m2/` results | Request short scoring lease |
| Terra H1 | `h1_optimized_v2/`, corresponding tests/scripts and `h1/` results | 0 |
| Terra M1 | `m1_optimized_v2/`, corresponding tests/scripts and `m1/` results | 1 |
| Coordinator | `two_mainlines_long_v1/current_query_v2/`, shared tests, this workorder, combined acceptance/benchmark scheduling | No competing training |

Existing unrelated worktree edits belong to the user. Do not revert them. New revision modules import old immutable implementations as references; they do not modify sealed submissions or old result trees. Old GPU/slot monitor is observational and may continue; its REGISTERED receipts are not live EvalAI health.

## Routes and invariants

### E: equivalent inference

Implement static calibration projections, boundary-correct sliding frontend reuse, final-layer last-query evaluation, and independent-session vectorization where justified. Keep temporal layers 1–3 full for a four-layer old Transformer. Ordinary cross-window temporal KV reuse is forbidden: window-relative positions and contextualized history make it non-equivalent.

At k=5, after sliding by one bin, retain old frontend positions 5..W-1 as new 4..W-2 and recompute new 0..3 and W-1. Startup zero inputs must be passed through biased frontend operations, not represented as zero embeddings by assumption.

Elementwise returned-space tolerance is fixed at `1e-5 + 1e-5*abs(reference)`; full local mean R² absolute delta must be at most `1e-5`. No post-hoc tolerance relaxation.

### T: finite-window current-query reader

The shared V2 temporal core follows the earlier latency design §9: four query updates, d256, eight heads, FFN512; every layer's memory K/V is projected independently from the original frontend z, not from contextualized past hidden states. The query starts at the current z and reads all legal window positions, including explicitly defined startup padding. Sixteen relative-age buckets per head are initialized to zero. This is a new trainable operator, not an exact fast path for old checkpoints.

The frontend/history and memory K/V caches repair the window's left boundary. K/V projections are refreshed for changed positions, and any model/bank/mask/device/dtype change invalidates the affected cache. T uses matched training and shared compatible initialization with its full-window control, not an unfair trained-vs-random comparison.

### Learning and units

H1 V1 trained on raw y but divided outputs by 20. V2 retains the intended deployment `/20` with training targets `20*y` and explicit round-trip assertions. M2's existing target `5*y` / output `/5` pair remains unchanged. M1 retains native EMG units unless an explicit new matched normalization contract is documented before training.

Run small real-source learning diagnostics before full training. Inspect gradients, output variation, input sensitivity, and baseline-relative error. RAW and EMA are both diagnostic views; freeze the governing view/candidate set before terminal validation. Do not infer success from a small raw MSE in small physical units.

H1's sealed source-pick set has 2,908 stride-4 full-history windows, whereas full-stream source-development evaluation has 20,325 valid bins. Report both separately; do not select the convenient surface after observing scores.

### M1 carrier and isolation

Use a new name `rSyn3-refit-v1`. Fit a single dictionary and normalization from source sessions 20120926/27/28 under a recorded recipe; persist actual dictionary/scales, per-unit raw and normalized carriers, support row identities, unit order, and hashes. Both arms and runtime load this common artifact. It is not the unrecoverable old sealed dictionary under a replacement hash.

Training must use a strict source-only fit loader, without loading outer 20120924 query data. Target M10 calibration and any terminal query evaluation are separate, after model/selection freeze. No target-query feedback is permitted to repair or select the design. Retain honest frozen B3/teacher provenance; new source-only execution does not erase parent history.

## Resource and evidence policy

- Python: `/home/xinyuan/miniconda3/envs/spint/bin/python`; use `PYTHONNOUSERSITE=1` and bounded CPU threads.
- At takeover both 3090 GPUs were idle. H1 owns GPU0 and M1 GPU1; other GPU work needs an explicit coordinator lease.
- Lightweight code/tests may run in parallel. A formal CPU latency benchmark is single-flight and should not contend with dataset materialization or other benchmarks.
- First measure 128-call microprobes, then full-history warmup and long sustained calls (target 2,048 calls × 3 fresh process replays), including preprocessing, buffers and output conversion. Record reset/cold/startup separately. Never report batch time divided by batch size as single-user latency.
- Speed gates inherited from the design: E should reduce P95 by at least 20% without worsening P99; target single-stream P95 <=15 ms and P99 <=20 ms. If that is not achieved, state it explicitly and continue the appropriate optimization branch, not a false real-time claim.
- For T, compare against the correctly trained matched full-window control (mean R² delta >=-0.005, worst-session delta >=-0.020 as the predeclared exploratory preservation margins), and show meaningful speed gain (>=3x P95 or the real-time target). A failed/constant-output full-window control cannot be used as an easy non-inferiority target; compare all candidates to working task references.
- No EvalAI writes or new official-score reads. Local source/development validation drives this project. Old failures and unsuccessful local attempts remain recorded.

## Completion audit

Each task needs a real trained/loaded artifact, its complete local score, corresponding working baseline, streaming parity tests covering reset/startup/gaps/roster changes, actual sustained latency, and a reproducible command/receipt. Inspect those outputs and their coverage before declaring the goal complete. Missing or weak evidence keeps the goal active.

## H1 source-only revision gate (2026-09-05 20:53 +08)

V2's audited formal attempt was stopped after three complete epochs after the
208-window source-capacity failure; it is not a completed twelve-epoch result.
V3's explicit centered frontend preserves its own semantics and strict contract
buffer, but its same-208 R2 of about 0.004/0.001 remains a meaningful-learning
failure. A diagnostic briefly loaded V3 weights into V2 after stripping the
contract buffer; that diagnostic is invalid, not evidence of lost V3 code.
The corrected strict V3 loader reproduces its original capacity result.
Immutable C2 scores 0.910671 on exactly these 208 source-training endpoints,
establishing a working same-data reference, not a new validation selection.

An isolated V4 hypothesis is authorized for implementation and source-capacity
testing only: direct signed activity mixing conditioned on the existing M3
identity/carrier, followed by causal k5 processing to d256 tokens. Preserve
W700/N176/D7, unit permutation/mask semantics, 20*y training and /20 deployment,
and the matched four-layer FULL/T temporal pair. The exact new frontend and
initialization must be recorded before its probe; do not mutate V2/V3 attempts.
Use the identical frozen 208 source windows and matched 260-update, LR2e-4,
effective-batch16/micro4, no-dropout diagnostic budget. Both arms must reach
training-set R2 >=0.5 and prediction standard deviation >=0.5 of target standard
deviation to pass this capacity gate. R2 in [0.1,0.5) is partial learning that
does not authorize formal training; lower scores fail. No minival or outer
query feedback is authorized by this diagnostic. Passing capacity remains
necessary but not sufficient for full local decoder acceptance.

## H1 log-age revision gate (2026-09-05 21:38 +08)

The completed V4 matched twelve-epoch attempt is retained in
`h1/paired_v4_signed_12ep_v1/`. Its frozen selected EMA models both occur at
epoch 12: FULL pooled R2 is 0.602114 on selection and 0.536064 on complete
development; linear-age T is 0.353335 and 0.205259 respectively. Independent
native-unit auditing and actual one-bin cached replay agree on both surfaces.
This is a T preservation failure, not a cache-correctness failure, and the FULL
control remains below the established C2 complete-development reference
(0.888499 pooled). These results do not complete H1 acceptance.

An isolated temporal revision uses sixteen log-age buckets, with table
`floor(log1p(age)*16/log1p(700)).clamp(max=15)`, in place of V2's linear-age
buckets. K/V are still independently projected from the original frontend
tokens and relative age is applied only when the current query reads memory.
V4's signed frontend, dimensions, loss/output units and finite history do not
change. Persistent age tables and temporal contract version 3 prevent silent
loading of V2 reader weights as this operator. V2 core numerical code remains
unchanged while the separately running M1 experiment uses it.

The source-only 208-window, fresh 1,040-update diagnostic has passed its
predeclared capacity gate: FULL 0.969666 and log-age T 0.964827, with prediction
standard deviations 0.005370 and 0.005063 against target 0.005226. The gate
does not estimate development performance. A query-only fresh-seed42
twelve-epoch formal run is authorized, with the exact V4 source data, unit
dropout, sampler and optimization schedule. It must not warmstart from this
diagnostic. The already completed V4 FULL is the frozen matched control;
document matching initialization/data/schedule rather than retraining it.
Freeze all twelve EMA checkpoints as the governing candidate set, pooled R2
on the 2,908 selection windows with earliest-epoch tie breaking, before the
run. Retain all RAW diagnostics and the epoch-12 endpoint and report the
complete 20,325-bin surface. The original T accuracy and speed acceptance
margins remain unchanged. This is iterative known-source development, not
an untouched outer-test claim.

## H1 fixed continuation diagnostic (2026-09-05 21:53 +08)

V5 completed all twelve epochs and selected EMA epoch 12. Independent native
pooled R2 is 0.344796 on selection and 0.199731 on complete development; actual
one-bin replay agrees (complete R2 delta +7.61e-9). The log-age hypothesis did
not improve V4 query accuracy and is retained as a negative result.

A new isolated V4 FULL-only continuation tests whether the original control
was undertrained. Restore the sealed V4 epoch-12 model, optimizer, EMA and RNG
states and retain its sampler, whole-unit dropout, batch recipe and LR
history. Before execution, freeze epoch-24 EMA complete R2 as the primary
endpoint and all epochs 13–24 EMA pooled selection R2 (earliest tie) as a
secondary selection surface. Export and independently audit both selected and
endpoint24 models; do not substitute RAW after observing scores.

The initial repeat-update check under the old numerical runtime produced
GPU reduction-level differences as small as 3.725e-9 in model weights. The
preflight-only `h1/v4_full_continue24_v1/` contains only its protocol and is
retained; it is not a training result. The executed continuation instead uses
the explicitly named `h1/v4_full_continue24_deterministic_v1/`, with
`CUBLAS_WORKSPACE_CONFIG=:4096:8` and deterministic Torch algorithms. It must
record exact pre-update restoration and exact model/optimizer/EMA agreement
after the same update from independently deserialized states. This is an
explicit numerical-runtime difference from the V4 launch, not a claim of
bitwise identity with an unobserved uninterrupted old-runtime training run.
Architecture and optimization recipe are otherwise preserved. A twelve-epoch
query versus this twenty-four-epoch FULL is not a matched-budget comparison.

## Continued runtime/memory and H1 quality work (2026-09-05 23:30 +08)

The user explicitly requested further inference-speed and memory improvements
for M1/M2/H1, and further H1 predictive performance. The user subsequently
instructed us not to wait for independent-reviewer data. Treat the user-given
EvalAI two-hour limit as an end-to-end runtime constraint; low normalized
latency is itself an objective, not merely completion before the limit.
Do not change or push the user's existing packages/images/submissions. New
runtime revisions and local diagnostic/container checks must stay isolated.
The earlier no-official-read rule was superseded only for the explicitly
requested read-only official latency comparison, not model/epoch selection.

Report full public host-input-to-copied-native-output time, mean/P95/P99,
reset/cold-start separately, full replay elapsed time and memory. Distinguish
owned rolling/cache bytes, shared model/bank tensors, transient allocation and
fresh-process RSS. Deduplicate aliases; neither a tensor-internal timing nor
batch time divided by B is a public single-stream latency result. Maintain
the fixed elementwise and R2 parity gates without relaxing tolerances.

M1 V3 owns a dedicated heterogeneous-bank current-query runtime with one shared
model, correct inactive-lane behavior and explicit cache invalidation. M2 V3
retains the first three full temporal blocks and full final K/V, while testing
last-Q-only projection, static bank terms and a direct newest-bin host path.
H1 V3 runtime work is independent of training: static signed-carrier query
caching and a compact, verified fixed-M3 decorator must preserve predictions.

The next bounded H1 learning experiment changes only whole-unit dropout
probability from .10 to .30, retaining V6 architecture, recency priors, shared
initialization, data, units and formal training schedule. Use the exact V4/V6
mask RNG stream and change only its threshold. Source preflight uses the
frozen 208 V6 windows, 1040 updates, LR2e-4, micro4/effective16, with dropout
disabled for scoring; both matched FULL/T arms require R2>=.5 and prediction
standard deviation >=.5 of target. Only a complete passing gate plus a
coordinator receipt binding its hash and the protocol can launch the fixed
fresh twelve-epoch matched formal pair. No dropout sweep or checkpoint warm
start is authorized. Existing T preservation margins and working C2 reference
remain unchanged; runtime optimization cannot itself improve model R2.
