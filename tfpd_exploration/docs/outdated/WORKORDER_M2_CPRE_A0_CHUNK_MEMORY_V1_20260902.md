# Work Order: M2 C-Pre and Trial-Free Chunk Memory A0 V1

**Date:** 2026-09-02

**Status:** implementation authorization only; no data, checkpoint, CUDA,
result-root reservation, or launch is authorized until the root audit issues a
separate opaque capability

**Dataset:** FALCON M2 only

**Scientific tier:** Tier 1, label-free continual calibration

**Primary purpose:** test whether deterministic, non-overlapping causal neural
chunks can replace completed-trial events after the same M10 T4 plus first-30
cached B3S initialization.

This work order supersedes no existing route. It must be implemented as an
additive successor and must not modify the sealed memory-law, chrono4,
reblock10, CDM, T4, checkpoint, normalizer, official packaging, or result
artifacts.

---

## 1. Bound review and predecessor evidence

The implementation closure must bind the exact bytes of this work order and
the current revision of:

```text
tfpd_exploration/docs/DESIGN_REVIEW_CONTINUAL_CALIBRATION_NEXT_ROUTES_20260902.md
```

The route must descriptor-rehash and semantically validate these completed
local predecessor terminals before any A0 result-root action:

```text
tfpd_exploration/results/m2_memory_law_scan_v1/terminal.json
  body SHA256 42462644b27eea8f9186b1990393b3472cca99779c4da64c1e28bc8c749e974b

tfpd_exploration/results/m2_chrono4_strict_v1/terminal.json
  body SHA256 77533ce4917047e42b32141fbe21632cea587cc6e6781f9de580decf4d29fc0a

tfpd_exploration/results/m2_reblock10_v1/terminal.json
  body SHA256 b77d8d370537d789879d83bbfa5676655884d1e6485d00dcf1c06d59b025993c
```

Required predecessor readings:

1. `UNIFORM_UNCAPPED` is the selected true-trial memory law; the completed
   EMA alpha family is not rerun or tuned;
2. chronological-first-four and first-10 reblocking are negative and may not
   replace the first-30 initialization;
3. all three routes are local-protocol evidence and claim no official
   contract.

The local static M10/activity30 anchor is:

```text
tfpd_exploration/results/m2_t4_activity_budget_screen_v1/score.json
  body SHA256 6bdad93328ba26c12b8aa3afffcbc3490c312dfe22939b3d3bb3c5ec5b9005ce
  schema m2_t4_activity_budget_screen_v1
  status TERMINAL
  checkpoint_sha256 25d7bc72b4d440004b58f1beaeadb7e15565a43e83dd1eadd160374270ec1d3e
  normalization_sha256 d17f5f4c4d106b9f19493be6f5f06846c01e917f408f516e630f5e8f09d1539e
  parameter_updates 0
  target_gradients 0
```

The anchor cell is exactly `ridge_activity30_m10`. Its frozen local summary is:

```text
within_post30 equal-session R2  0.6662654087571492  (7 sessions)
external_official_query R2      0.27232178175902705 (6 sessions)
```

These two historical numbers use different query-surface roles.  The within
surface is already post-30.  The external `0.272321...` is the full
`external_official_query` surface and is **not** a matched post-30 baseline for
A0.  It binds the frozen checkpoint, normalizer, M10 carrier construction and
activity30 initialization, but A0 must recompute its static arm on its own
common post-30 windows.

The official submission receipt is context and packaging evidence only:

```text
sua_exploration/evalai_t4_m2_activity_budget/artifacts/
  evalai_submission_581359_terminal_receipt_v1.json
body SHA256 a747ece42f8b7b467767e4d86debc0368dcc3df0db81195f0849967b35f78c07
submission_id 581359
label_budget 10
activity_budget 30
checkpoint_sha256 25d7bc72b4d440004b58f1beaeadb7e15565a43e83dd1eadd160374270ec1d3e
held_out_r2_mean 0.26383289570479845
```

This receipt does not expose a local first-prediction witness. V1 therefore
claims only a local matched A0 comparison and packaging compatibility. It must
not claim bitwise first-prediction parity to the private official package or
an official performance improvement.

---

## 2. Owned additive paths

Terra owns only:

```text
tfpd_exploration/src/m2_cpre_a0_chunk_memory_v1/__init__.py
tfpd_exploration/src/m2_cpre_a0_chunk_memory_v1/plan.py
tfpd_exploration/src/m2_cpre_a0_chunk_memory_v1/inventory.py
tfpd_exploration/src/m2_cpre_a0_chunk_memory_v1/chunk_memory.py
tfpd_exploration/src/m2_cpre_a0_chunk_memory_v1/physical.py
tfpd_exploration/src/m2_cpre_a0_chunk_memory_v1/receipts.py
tfpd_exploration/scripts/run_m2_cpre_a0_chunk_memory_v1.py
tfpd_exploration/tests/test_m2_cpre_a0_chunk_memory_v1.py
```

No shared production file may be edited. Existing private helpers may be used
by reviewed import/composition; loader, parser, T4 fit, decoder, metric, and
trial materialization code must not be copied.

The public CLI is inert by default and must reject execution without an opaque,
one-shot, in-process capability issued after root review.

---

## 3. Two-stage immutable lifecycle

### 3.1 C-Pre root

```text
tfpd_exploration/results/m2_cpre_metadata_v1
```

Exact successful bodies:

```text
attempt.json
metadata_inventory.json
terminal.json
```

Each body has one canonical `.sha256` sidecar; every file is regular, mode
`0444`, nlink 1, and no extra leaf is allowed.  Failure is stage-exact and
must preserve any canonical immutable pair that was already published:

```text
before metadata_inventory publication       attempt + failure
after metadata_inventory publication        attempt + metadata_inventory + failure
```

The failure body binds the SHA-256 of every retained prefix body.  A published
body/sidecar pair is never deleted, overwritten, or hidden merely to recreate
the shorter failure topology.  `terminal` and `failure` remain mutually
exclusive.  Failure while publishing a body/sidecar pair itself is fail-closed
for root audit; the route must not invent a valid failure graph around a
partial pair.

C-Pre must finish before an A0 capability can be issued. It may read only
chronology and window metadata. It must not load the model or checkpoint,
initialize CUDA, compute R2, read target behavior values, or select a grid from
target performance.

### 3.2 A0 shard and aggregate roots

```text
tfpd_exploration/results/m2_a0_chunk_noninferiority_v1/external
tfpd_exploration/results/m2_a0_chunk_noninferiority_v1/within
tfpd_exploration/results/m2_a0_chunk_noninferiority_v1/aggregate
```

Each scoring shard publishes:

```text
attempt.json
launch.json
input_authority.json
replay.json
terminal.json
```

with body/sidecar pairs and terminal XOR failure. The aggregate is CPU-only,
descriptor-loads both completed shards, independently recomputes every row,
contrast, gate, and digest, and then publishes attempt/input/score/terminal.

The parent and all three named subroots must be absent before reservation. Each
shard capability is bound to its exact named root, C-Pre terminal, closure,
device profile, surface, roster, and scheduler profile and is consumed once.

---

## 4. C-Pre metadata contract

The proposed exposure grid is:

```text
n in {10, 30, 60, 120, all-past}
K in {10, 30, 60, infinity}
```

For every fixed numeric `n`, `n` means the **total** number of chronological
past activity units consumed, exactly as in the review's definition of
`R2(n,K)`.  It never means "first 30 plus another `n`".  Thus the fixed suffix
for `n=10` begins after chronological unit 10 (and is then intersected with
the already frozen post-30 query surface), while `n=30` begins after unit 30.
The special `all-past` entry is not a fixed `n=total` row: it is a causal
prequential policy that starts on the post-30 query surface and consumes each
newly completed unit before later predictions.

A0 itself is initialized with activity30 and therefore starts at total
exposure `n=30`; it does not instantiate the `n=10` curve point.  The `n=10`
metadata row is retained only for the later C0 systems whose declared initial
activity support is M10.  C-Pre must never report total exposure 40 under the
label `n=10`.

C-Pre publishes, in canonical surface/session/grid order:

1. total chronological native trials;
2. first-30 calibration boundary;
3. post-support native activity units;
4. ordered valid window count and digest;
5. for every fixed total-exposure `n`, whether the session has at least `n`
   chronological units and a nonempty remaining common query suffix, plus a
   separately typed eligibility row for the causal-prequential `all-past`
   policy;
6. the suffix start, window count, and ordered-starts digest;
7. the dataset-specific reduced grid obtained without R2;
8. explicit exclusion reasons for the exact-33-trial sessions.

The primary A0 experiment does not wait for `n=60/120`: it uses every session
with a nonempty post-30 query. The larger grid is characterization evidence for
later C0 only.

---

## 5. Frozen A0 science

### 5.1 Shared initialization

Both A0 arms use exactly:

```text
carrier label budget          M10
carrier support law           chronological first 10 positions in first30,
                              with the existing finite-direction checks
initial activity identity     chronological first 30 B3S [100,N] rows
activity retained capacity    30 rows
checkpoint / normalizer       exact local anchor literals in Section 1
query surfaces                external post30 local and within post30
target optimization           forbidden
model/checkpoint mutation     forbidden
```

The new route-local state is explicitly named
`SeededActivity30RollingMemory`. It separates M10 carrier authority from the
30-row activity seed. It must not weaken or overload the existing
`ActivityMemory` invariant, pass `budget=30` to the M10 carrier, or describe
activity30 as carrier30.

Before any update, static anchor, true-trial arm, chunk phase-0 arm, and chunk
phase-50 arm must have the same initial activity identity and the same first
prediction on the same input under the same device/batch law. This is a local
within-run exact parity requirement.

The initial seed authority is the sealed activity-budget tensor
`dataset.calib_trialized_neural_features[session][:30]` used by
`ridge_activity30_m10`, together with that route's `_ridge_side` output.  The
same full sealed tensor supplies the post-30 completed-trial rows for the
`TRUE_TRIAL_MEMORY` arm.  The frozen M2 configuration uses cubic interpolation.
The local CDM helper `_native_trial_views` instead reconstructs its historical
`activities` with linear interpolation, so those returned activity rows are
explicitly forbidden as the A0 seed or true-trial reference.

An independent reconstruction audit is still required, but it must use the
frozen cubic interpolation law.  A0 compares that cubic reconstruction of the
first 30 rows to the sealed tensor and records both digests.  Exact equality
is preferred.  If they differ only numerically, the maximum absolute
difference must be at most `1e-6`, the digest drift must be disclosed, and the
sealed activity-budget tensor remains the initial seed.  A static arm built
from the same unverified reconstruction is not an acceptable anchor proof.

### 5.2 True-trial arm

After the first-30 seed:

1. predict using only activity units completed strictly before the current
   prediction coordinate;
2. when a native trial completes, take its sealed cubic
   `dataset.calib_trialized_neural_features[session]` B3S `[100,N]` row;
3. after the prediction, evict the oldest retained activity row and append the
   completed native trial;
4. recompute/cache identity for future predictions only.

The arm reuses the existing native-trial materializer and is the governing
reference. It is not the historical support-M plus FIFO-(30-M) state.

### 5.3 Fixed-chunk arm

Architecture-derived chunk geometry is frozen without target selection:

```text
chunk length L        100 neural bins (the B3S row length)
stride                100 bins, exact non-overlap
phase origin          first raw neural bin strictly after the chronological
                      first-30 calibration boundary
primary phase         offset 0 raw bins from that phase origin
sensitivity phase     offset 50 raw bins from that same phase origin;
                      descriptive only, never replaces phase 0
padding               forbidden
partial final chunk   discarded and counted
invalid/metric mask   never used to form or filter neural chunks
behavior/angle labels never read by the chunk state
```

The chunk state consumes every chronologically observed neural bin after the
calibration boundary, including inter-trial/still periods exposed by the raw
stream.  Its phase origin is therefore fixed by raw chronology and must not be
derived from the first governed window, its endpoint, `eval_mask`, or any
other scoring-surface coordinate.  In particular, the W50 offset of the first
scored post-30 window must not shift the phase-0 chunk contents.

For every raw-bin transaction at coordinate `t`:

1. if `t` is a governed scored-prediction endpoint, decode from chunks
   completed strictly before `t`; otherwise perform no decoder call;
2. append neural bin `t` only after the optional scored prediction;
3. if the append completes exactly 100 bins, finalize one `[100,N]` activity
   unit, evict the oldest retained row, append the chunk, and update identity
   for later predictions;
4. never use overlapping bins, future bins, a global counter shared by
   sessions, or behavior-derived boundaries.

Reset/done, asynchronous streams, empty streams, invalid shapes, duplicate or
skipped bin indices, and incomplete chunks must have typed fail-closed tests.
Receipts must count raw-bin transactions and actual scored decoder calls as
different quantities; a raw-bin transaction must never be described as a
prediction.

### 5.4 Frozen surfaces and order

Primary governing surface:

```text
external_post30_local, M10, six frozen sessions
```

Secondary mechanism surface:

```text
within_post30, M10, seven frozen sessions
```

The external local window set is **not** identical to the historical anchor's
full `external_official_query` set.  A0 must prove that its ordered post-30
starts are the deterministic post-30 subset of that bound full-query set, and
must disclose both counts and digests.  The historical external R2
`0.272321...` may not be inserted as A0's static row.

Instead, A0 recomputes `STATIC_ACTIVITY30_M10`, `TRUE_TRIAL_MEMORY`,
`FIXED_CHUNK_PHASE0`, and `FIXED_CHUNK_PHASE50` on the exact same post-30
target rows, validity mask, ordered window starts, batch law, and metric.  The
static recomputation uses the sealed activity30 tensor and matched M10 carrier
specified in Section 5.1.  Within-post30 should retain exact query-surface
parity with its historical within anchor.  The phase-50 sensitivity arm may
never select or replace phase 0.

---

## 6. Preregistered A0 gates

For each session:

```text
delta = R2(FIXED_CHUNK_PHASE0) - R2(TRUE_TRIAL_MEMORY)
```

Primary external noninferiority passes only if:

```text
equal-session mean delta >= -0.005
worst-session delta       >= -0.010
all 6 external sessions present
all causal, state, anchor, input, and no-update invariants pass
```

Classification:

```text
mean delta > +0.005 and >=4/6 positive
  CHUNK_MATERIALLY_BETTER_NOT_EQUIVALENT

noninferiority passes but superiority condition does not
  CHUNK_NONINFERIOR

otherwise
  CHUNK_FAILS_NONINFERIORITY
```

Within is a secondary disclosure with the same delta table and no power to
reverse the external verdict. Phase 50 is sensitivity evidence only. No
bootstrap, capacity, phase, or hyperparameter selection is performed on target
R2 in V1.

Every terminal must report per-session R2, delta, first/final identity digest,
prediction/target/window digest, update counts, evictions, discarded partial
bins, wall time, peak host/GPU memory, and zero target-gradient/update facts.

---

## 7. Reusable implementation seams

The route should compose, not copy:

```text
src/m2_memory_law_scan_v1/physical.py
  _side_tensor
  _predict_with_identity

src/m2_t4_activity_budget_screen_v1/physical.py
  _support_indices
  _ridge_side

src/cdm_p1_m2_local_v1/replay.py
  _g_session_views
  g_support_material
  g_query_rows

src/m2_precision_cdm_v2_screen_v1/physical.py
  _native_trial_views
  _query_trial_rows
  _windows
  _trial_capabilities
```

From `_native_trial_views`, A0 may reuse raw neural data, trial starts, and
first-30 carrier-rate/angle authorities.  Its linearly interpolated
`activities` return is historical LOCAL-M2 evidence only and may not feed A0
activity memory.  `_query_trial_rows` must receive the sealed cubic activity
tensor instead.

The A0 M10 carrier must use the activity-budget/581359 support law exactly:
chronological indices `[0,1,...,9]`, followed by the existing finite-angle
mask *inside* the ridge fit.  The memory-law helper `_support_for_budget(M10)`
is explicitly forbidden here because it selects the first ten finite-angle
trials and may therefore reach chronological positions beyond 9.  That helper
is valid for its own completed predecessor but does not reproduce the matched
581359 carrier.  Every live A0 session must record exact selected indices
`[0,...,9]` and bind its raw and normalized T4 digests before first-prediction
parity is accepted.

The DANDI `p4_stream_stats` materializer and P2-prime GPU-cache materializer
are design references only and may not be used as FALCON M2 data authorities.

---

## 8. CPU/no-data implementation gates

Before any C-Pre or A0 live capability:

1. public CLI inert and Torch-free under `python -S --dry-run`;
2. pure inventory tests cover exact-33, no-suffix, reduced grids, canonical
   order, and prove target values cannot influence eligibility;
3. chunk state tests cover exact 100-bin completion, phase 0/50, non-overlap,
   causality, eviction, reset/done, two interleaved streams, and final discard;
4. activity30-seeded M10 constructor proves carrier/activity authority
   separation and does not modify shared core;
5. first-prediction parity and static-anchor reconstruction are covered on an
   actual-shaped CPU synthetic route;
6. terminal and failure topologies, descriptor rehash, sidecars, modes,
   predecessor drift, closure drift, and capability reuse are adversarially
   tested;
7. existing M2 predecessor focused tests remain green or historical closure
   failures are explicitly explained rather than masked.

Terra freezes at this no-data/no-CUDA boundary for independent root audit.

---

## 9. Shared-single-GPU execution envelope

The science is independent of scheduling. Concurrency is permitted only after
a bounded parity/throughput smoke proves that two processes sharing one GPU do
not change either shard's outputs.

### 9.1 Device and shards

Use physical GPU0 only. Physical GPU1 is reserved for the user's independent
M1 work and must not be queried, initialized, used as a fallback, or included
in a multi-GPU runtime:

```text
CUDA_DEVICE_ORDER=PCI_BUS_ID
CUDA_VISIBLE_DEVICES=0
physical UUID GPU-ac7388a5-2e98-300a-fdb3-0b67bfd494d9
logical device cuda:0
```

Concurrent pair:

| Process | Surface | CPU affinity |
|---|---|---|
| shard A | external | `0-3,16-19` |
| shard B | within | `4-7,20-23` |

Each process has `num_workers=0` and sets OMP/MKL/OpenBLAS/NUMEXPR thread
counts to 1. Roots, logs, capabilities, RNG, runtime objects, and receipt
writers are distinct. No process may signal, reprioritize, retask, or modify
the other.

### 9.2 Admission and smoke

The M2 CPU sets occupy only logical CPUs
`0-7,16-23`. Logical CPUs `8-15,24-31` are left outside the M2 profiles for
the independent M1 route. Root does not alter the M1 process, affinity, root,
or launcher; if an already-running M1 job uses a broader mask, M2 still keeps
its own fixed profiles and the host-pressure gate decides admissibility.

Before the concurrent smoke:

1. GPU0 has no foreign compute owner, while GPU1 ownership is treated as
   immutable external M1 state;
2. physical UUID/CVD mapping is exact;
3. both prospective shard roots are absent;
4. `MemAvailable >= 32 GiB`;
5. CPU/memory/IO PSI `some avg10 <= 0.1` and memory/IO `full avg10 == 0`;
6. swap occupancy is disclosed but is not alone a rejection when available
   memory and PSI pass;
7. individual one-session solo smokes establish peak VRAM <= 6 GiB each;
8. aggregate projected VRAM <= 16 GiB, preserving at least 4 GiB headroom.

Then run the same bounded one-session coordinate once solo and once as the
two-process pair. Concurrency passes only if:

```text
same process-specific prediction SHA as its solo CUDA run
same identity/state/update digests
same R2 exactly, or max_abs_prediction <= 2e-6 with explicit digest drift
no CUDA/OOM/nonfinite/closure/anchor error
aggregate pair throughput >= 1.25 * serial aggregate throughput
```

If numerical parity fails or throughput gain is below 1.25x, the full shards
run serially on GPU0. This scheduling fallback changes no scientific arm and
is not a method result.

### 9.3 Live monitoring

After launch, Luna owns read-only monitoring. It checks on any exception,
process exit, terminal/failure publication, second/foreign owner, memory-floor
breach, or nonfinite/OOM evidence, and otherwise every 30 minutes. Luna may not
edit, signal, restart, change affinity, reserve roots, or launch a retry.

Immediate fail-closed conditions are CVD/UUID mismatch, any foreign GPU0
owner, an unexpected third GPU0 owner, any M2 CUDA context on physical GPU1,
M2 affinity drift/overlap, aggregate GPU0 memory above 20 GiB, free GPU0 memory
below 4 GiB, host pressure gate failure, CUDA/OOM/nonfinite error, root or
closure drift, target update, missing/mismatched sidecar, or process exit
without the declared terminal/failure topology.

---

## 10. Explicit exclusions

V1 does not authorize:

- a new official EvalAI submission;
- any target-selected chunk length, stride, phase, capacity, or alpha;
- another EMA sweep;
- first-10 or chronological-four reblocking;
- carrier updates, learned gates, readout adaptation, training, backward, or
  optimizer steps;
- modification of shared `ActivityMemory` or any sealed result;
- use of target behavior to construct chunks;
- two unprofiled full processes on one GPU;
- use of both GPUs merely because they are idle.

The only new scientific contrast is true completed-trial segmentation versus
deterministic non-overlapping 100-bin causal segmentation after the same M10
carrier and first-30 activity seed.
