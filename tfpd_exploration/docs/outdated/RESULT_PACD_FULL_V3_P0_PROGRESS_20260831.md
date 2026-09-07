# PACD Full V3 P0 Progress — 2026-08-31

## Live run

- Cell: `PACD_MATCHED_FULL_TRAINING_V3`
- Arm: P0, M30 anchor / M30 short matched control
- Canonical root: `tfpd_exploration/results/paired_anchored_calibration_dropout_full_v3/p0_fullfull_seed42`
- GPU: physical GPU0, UUID `GPU-ac7388a5-2e98-300a-fdb3-0b67bfd494d9`
- GPU1: unused
- Execution closure: `3356fb124ff4d18a2035bfda8e7eacc28a1642ce3f36b290155a5bcdf7faea2f`
- V2 failed predecessor: `c7f1a893a7c65dbb46b8493c9f8080ec12af5b3c7e97e606cb7d8aed005ea8b0`

This is a live progress record outside the execution closure. It is not a
terminal result and must not be used as a performance claim.

## Epoch 000 verified event

`epoch000.json` published naturally at 2026-08-31 04:17:44 HKT. The body and
canonical sidecar are regular mode-`0444` files and verify to:

`664880e1af5505d622f2c698cc30619514940a7c7034c5d37f2130d244c9d10c`

The receipt establishes:

| Evidence | Value |
|---|---:|
| Optimizer steps | 33,925 |
| Cumulative optimizer steps | 33,925 |
| Epoch wall time | 3,389.481 s |
| Paired steps/s | 10.0089 |
| Combined loss mean | 0.762492 |
| Valid bins per batch | 1,600 exact |
| Positive encoder-gradient steps | 33,913 |
| Accepted zero encoder-gradient steps | 12 |
| Positive decoder-gradient steps | 33,925 |
| Zero decoder-gradient steps | 0 |
| P0 prediction mismatches | 0 |
| P0 identity mismatches | 0 |
| RNG violations | 0 |
| Prefix mutations | 0 |
| Parameter-finiteness violations | 0 |
| Adam finite | true |

All 12 zero encoder-gradient steps are explicitly typed as
`all_units_dropped_valid_zero`; accepted-zero count equals total zero-encoder
count. No broader zero-gradient waiver occurred. Encoder combined-gradient
min/mean/max are `0 / 0.548986 / 3.464877`; decoder combined-gradient
min/mean/max are `0.211056 / 1.747352 / 10.469779`.

The fixed sentinels are exactly steps `0`, `1`, `16,962`, and `33,924`. Each
sentinel retained units and had positive encoder gradient. The sampler-order
digests are:

- window indices: `94659d1a9b56cd1f972ac20724bec27d49766daded3854db6d70e7a79a56e6d5`
- batched indices: `ae6d27ffdc2d1f8b820db412f2d6845871b94fe13cfb7ddcfbd2686375f1a624`

Epoch-000 model state SHA is
`a76f85fd41cedc0d01b4ae4c208beb1f1c9b021159f8e5114457bd1fecb3b19a`;
optimizer state SHA is
`3beaf82a258fd9b8e8d1cc413cf461b698e32ebdd67b5ffd08f49b8a5e20a408`.

At the observed epoch-000 throughput, the mechanical 48-epoch duration is
approximately 45.2 hours. This projection is descriptive only and is not a
stop or selection rule.

## Current decision

P0 continues without intervention. P1 and P2 remain forbidden until P0 has a
valid 48-epoch terminal, four final checkpoints, SWA, manifest, no failure,
and an independently validated immutable artifact graph.

## Epoch 001 verified event

`epoch001.json` published naturally at 2026-08-31 05:12:54 HKT. Its immutable
body/sidecar pair verifies to:

`d6fdd384b4d689ddfd15565564ea654b560046e217e00c353e0b48ca2129634a`

The second epoch adds exactly 33,925 optimizer steps, for 67,850 cumulative
steps. Combined mean loss declines from `0.762492` to `0.649434`. Epoch wall
time is 3,307.717 seconds and throughput is 10.2563 paired steps/s.

Gradient coverage remains exact: 33,904 positive encoder steps, 21 accepted
zero encoder steps, and all 21 are typed `all_units_dropped_valid_zero`;
decoder gradient is positive on all 33,925 steps. P0 prediction/identity
mismatches, RNG violations, prefix mutations, and finite-state violations are
all zero; Adam remains finite. Epoch-001 model state SHA is
`d08313b4de69a2ef58fd2a88c70c7e3e9fcdfd1fad1436574c56084358341e49`;
optimizer state SHA is
`0cfa69a4d1f3cf3f0bd7f6a5e98debc22e4615dbb2a4b802f1a282af1fe1ac53`.

## Epoch 002 verified event

`epoch002.json` published naturally at 2026-08-31 06:07:56 HKT. Its immutable
body/sidecar pair verifies to:

`19d2e3b7d697a17eac71484571cb70d40e9cd0be535d4af357598b3ae93a7e13`

Cumulative optimizer steps reach 101,775. Combined mean loss continues down
to `0.614537`; epoch wall time is 3,300.031 seconds and throughput is 10.2802
paired steps/s. Gradient coverage contains 33,909 positive encoder steps and
16 accepted zero encoder steps, all typed `all_units_dropped_valid_zero`.
Decoder gradient is positive on all 33,925 steps. P0 prediction/identity
mismatches, RNG violations, prefix mutations, and finite-state violations
remain zero; Adam is finite.

Epoch-002 model state SHA is
`ac5e5f4e379912c4cec6597a65ea89b7058bd8234c2ba9783bd29281795a3b12`;
optimizer state SHA is
`279728ce8916a76fed0ba62ce00c88eee76e6f1c09c82b79efa3d4f5cf6c45cb`.

## Epoch 003 verified event

`epoch003.json` published at 2026-08-31 07:03:08 HKT and verifies to
`a8f7b327266100553864e185c82a181e4e92c46a4161ce25a5ee2829f52e5b16`.
Cumulative optimizer steps are 135,700; combined mean loss is `0.589217`;
throughput is 10.2525 paired steps/s. All 16 zero encoder-gradient steps are
accepted only as `all_units_dropped_valid_zero`; decoder is positive on every
step. P0 equality, RNG, prefix, finite-state, and Adam invariants remain clean.

## Epoch 004 verified event

`epoch004.json` published at 2026-08-31 07:58:21 HKT and verifies to
`3c87f251cf9b05e7d37979bf0be37a94a2ad68bd8939d7aadc083fa50ecc8193`.
Cumulative optimizer steps reach 169,625; combined mean loss is `0.572074`;
throughput is 10.2454 paired steps/s. All 18 zero encoder-gradient steps are
typed accepted all-units-dropped events, decoder is positive on all steps, and
all P0/RNG/prefix/finite/Adam invariants remain clean.

## Epoch 005 verified event

`epoch005.json` published at 2026-08-31 08:53:44 HKT and verifies to
`8606e609548a7e28a75111ad81decc0e36962ca4bba6ee5fc5df557c3a22266d`.
Cumulative optimizer steps are 203,550; combined mean loss is `0.558242`;
throughput is 10.2186 paired steps/s. All 24 zero encoder-gradient steps are
typed accepted all-units-dropped events. Decoder remains positive on all
33,925 steps, and all P0/RNG/prefix/finite/Adam invariants remain clean.

## Epoch 006 verified event

`epoch006.json` published at 2026-08-31 09:49:22 HKT and verifies to
`c784feda333db280c065373e7b2e3c857f9d331738bdbb9a96929b2e409dc424`.
The body and canonical sidecar are regular mode-`0444` files, and the sidecar
binds the canonical body basename.

Cumulative optimizer steps reach 237,475. Combined mean loss continues down
to `0.544373`; epoch wall time is 3,335.878 seconds and throughput is 10.1697
paired steps/s. Gradient coverage contains 33,908 positive encoder steps and
17 accepted zero encoder steps, all and only typed
`all_units_dropped_valid_zero`. Decoder gradient is positive on all 33,925
steps. P0 prediction/identity mismatches, RNG violations, prefix mutations,
and parameter-finiteness violations remain zero; Adam remains finite.

Epoch-006 model state SHA is
`35e2a8067106c6c26557441ca9d5366eae824cf96bdbe5f58b2110e51a537463`;
optimizer state SHA is
`0fdb42bf83fc416b3775b227dcdbca029a601762dffd32520aefb229ada053aa`.

At this boundary the run is active on physical GPU0 only. Physical GPU1 is
idle at 0% utilization with 23 MiB reported memory, and the completed Stage-P
route has already terminalized. No PACD P1/P2 launch is authorized before the
P0 terminal gate described above.

The immutable attempt's complete 48-file execution closure was also rehashed
against the live worktree at this boundary: all 48 file digests still match and
the aggregate closure remains
`3356fb124ff4d18a2035bfda8e7eacc28a1642ce3f36b290155a5bcdf7faea2f`.
The concurrently developed matched scorer is additive and is not a member of
this training closure, so its review-only construction cannot alter the
running numerical job.

## Parallel-route isolation check

At 2026-08-31 10:12 HKT, the separately assigned CDM+P1 cross-dataset route
was identifiable only through
`docs/WORKORDER_CDM_P1_CROSS_DATASET_V1_20260831.md`. That work order assigns
execution to a companion machine and requires additive packages that do not
touch running roots or frozen packages. On this host, `nvidia-smi` showed only
PACD PID 783126 as a compute process, bound to physical GPU0; physical GPU1
had no compute process and remained at the 23 MiB driver baseline. No new
attempt/launch/result receipt for the cross-dataset route existed locally.

The PACD attempt's 48-file execution closure was rehashed after the parallel
work-order and additive scorer files appeared; all 48 digests remained exact.
This establishes code/root/device isolation at the observed boundary without
intervening in either route.

## Natural local launch of the parallel route

At 2026-08-31 10:36 HKT, the CDM+P1 Part-A replay launched naturally on this
host rather than on the companion machine named in its work-order scheduling
note.  This is recorded as a scheduling deviation, not silently treated as
the originally planned placement.  The live process was:

```text
PID 799940
cwd /home/xinyuan/Work_host/SPINT/tfpd_exploration
python scripts/run_cdm_p1_cross_v1.py --stage replay
CUDA_VISIBLE_DEVICES=1
result root tfpd_exploration/results/cdm_p1_cross_v1
```

The route published its own immutable mode-`0444` attempt pair before replay.
It is inference-only and its attempt records no decoder training or model /
checkpoint update.  At this boundary physical GPU1 carried only PID 799940
(about 330 MiB compute allocation), while PACD PID 783126 remained the only
physical-GPU0 compute process (about 708 MiB compute allocation).  No process
was mapped to both devices.

Host-resource inspection also found no active contention signal: PACD used
about one CPU core and 2.08 GiB RSS; CDM+P1 used about 1.13 CPU cores and
4.36 GiB RSS; sampled host CPU idle time was 93--94% and available memory was
about 48 GiB.  Although the CDM+P1 process exposed 81 threads and did not carry
explicit OMP/MKL/BLAS thread-limit environment variables, its observed total
CPU utilization remained about 113%, rather than consuming the host thread
pool.

After this local GPU1 launch, the complete PACD 48-file execution closure was
rehashed once more: all 48 files still matched and drift remained zero.  The
two result roots and additive code packages are disjoint.  Luna retains a
read-only watch for a device-map change, material CPU/memory contention,
cross-root writes, closure drift, or either route's natural terminal/failure;
no process was stopped, restarted, or modified during this audit.

## Epoch 007 and measured coexistence

`epoch007.json` published naturally at 2026-08-31 10:45 HKT.  Its body and
canonical sidecar are regular mode-`0444` files and both bind SHA-256
`9d57394264be9843e5904dee35636dccb03219065a5d9ea90f824fd99fef884c`.
The receipt records 33,925 optimizer steps and 271,400 cumulative steps.  Mean
combined loss is `0.536548`; epoch wall time is `3370.051 s` and throughput is
`10.0666` paired steps/s.

Gradient coverage remains exact: 33,907 positive encoder steps and 18 zero
encoder steps, with all 18 and only those 18 accepted under the typed
`all_units_dropped_valid_zero` reason.  Decoder gradient is positive on all
33,925 steps.  P0 prediction and identity mismatches, RNG violations, prefix
mutations, and parameter-finiteness violations are all zero; Adam remains
finite.

This epoch overlapped the naturally launched local CDM+P1 replay.  During the
later resource sample, that process had grown to about 4.4 CPU cores, 13.8 GiB
RSS, and 1.93 GiB on physical GPU1.  Host CPU still sampled at 45--47% idle
with about 39 GiB available memory.  Most importantly, PACD's observed
`3370.051 s` / `10.0666` steps/s remained inside its pre-overlap empirical
envelope (`3300--3389 s`, `10.0089--10.2802` steps/s for epochs 000--006).
Thus there is no measured PACD slowdown at this boundary, despite the parallel
route becoming materially heavier.  The PACD execution closure was rehashed
again after epoch 007 and remained 48/48 exact with zero drift.

## Matched-scorer freeze and latest isolation audit

The additive PACD matched scorer reached its final no-data review boundary
while both GPU jobs remained live.  Its focused no-CUDA suite passed 19/19;
`py_compile`, a Torch-free `python -S --dry-run`, clean-process real T0/C1/SD
descriptor integration, double closure reconstruction, and `git diff
--check` all passed.  The scorer closure is
`e81648d0f4ac867665e57893dad421ce210cccbb80f4c1713c748b93a03bb893`.

The scorer is route-owned CPU-only.  Live execution requires exactly empty
`CUDA_VISIBLE_DEVICES`, `selected_device=cpu`, and
`cuda_initialized=false`; the adapter verifies CPU model and prediction
placement.  A successful score is published by one directory rename from a
hidden staging directory into `complete/`.  Pre-commit failures remove only
the private staging directory and publish an attempt/failure topology, so no
already-published canonical receipt is deleted.  Live scoring remains
fail-closed because exact P0/P1/P2 producer terminal literals do not yet
exist.  Consequently scorer construction and testing used neither physical
GPU and could not contend with either training route.

At the latest live check PACD PID 783126 remained the sole physical-GPU0
process and CDM+P1 PID 799940 remained the sole physical-GPU1 process.  The
CDM+P1 replay had grown to about 11.3 CPU cores, 14.3 GiB RSS, and 1.94 GiB on
GPU1; PACD remained near one CPU core, 2.11 GiB RSS, and 708 MiB on GPU0.  No
process was mapped to both GPUs, the result roots remained disjoint, and the
PACD 48-file execution closure again rehashed with zero drift.  This is an
isolation finding, not a claim that the unfinished cross-dataset replay has a
performance result.

The CDM+P1 attempt's own seven-file `owned_sha256s` map was independently
checked against the live worktree after launch: every file matched exactly.
Its isolated no-user-site/no-CUDA/thread-one suite passed 22/22.  At that
boundary its canonical root still contained only the immutable mode-`0444`
attempt body/sidecar pair; no replay, terminal, or failure receipt existed.
This proves current-byte stability for both live routes while keeping the
scientific verdict pending the natural CDM+P1 terminal or failure event.

## Epoch 008, measured slowdown, and CDM retry disclosure

`epoch008.json` published naturally at 2026-08-31 11:45 HKT.  Its immutable
body and canonical sidecar are regular mode-`0444` files and both verify SHA
`44fd689eec395d2219d5b67411770002aa13be98ede38fa4a8fba032ad9a9eb3`.
It records 33,925 optimizer steps and 305,325 cumulative steps.  Mean combined
loss declined to `0.523603`.

All numerical and causal evidence remains clean: 33,915 positive encoder
steps, ten zero encoder steps all and only accepted as
`all_units_dropped_valid_zero`, 33,925 positive decoder steps, zero P0
prediction/identity mismatches, zero RNG violations, zero prefix mutations,
zero finiteness violations, and finite Adam.

The resource conclusion did change.  Epoch 008 took `3608.574 s` at `9.40122`
paired steps/s.  The pre-overlap epochs 000--006 occupied `3300--3389 s` and
`10.0089--10.2802` steps/s.  Epoch 008 is about 6.5% slower than the previous
slowest epoch, so the earlier "no measured slowdown" statement applies only
through epoch 007 and is superseded for the heavier overlap interval.  The
parallel route did not alter PACD science or bytes, but it did impose a modest
throughput cost.

Read-only log inspection also established that CDM+P1 PID 799940 failed at
about 11:30 HKT with `KeyError: 'sealed_model'` at `replay.py:459`.  An external
agent repaired the route-owned weight-swap return seam, removed and recreated
the same canonical root, published a new attempt, and launched PID 804509 at
11:39 HKT.  The new attempt binds the revised `weights.py` and test bytes, but
it has no immutable predecessor failure receipt for the removed first attempt.
This is a successor/retry-lineage anomaly, not a natural continuation or a
completed performance result.  Full evidence and the admissibility boundary
are recorded in `docs/AUDIT_CDM_P1_CROSS_V1_RETRY_20260831.md`.

To prevent further CPU scheduling contention without stopping either route,
root applied a reversible scheduling-only partition after epoch 008.  The host
has one NUMA node and 16 physical / 32 logical CPUs:

```text
PACD PID 783126  -> logical CPUs 0-3,16-19   (four physical cores)
CDM  PID 804509  -> logical CPUs 4-15,20-31  (twelve physical cores)
```

GPU assignment remains unchanged (PACD physical GPU0; CDM physical GPU1).
No code, data, model, RNG, optimizer, checkpoint, or result artifact changed.
Epoch 009 will be the first clean timing measurement under the partition; no
claim that the throughput issue is repaired is made before that receipt.

## Post-P0 admission audit: two code-level gates still required

A read-only audit of the frozen V3 execution path found that the scientific
runner remains sound, but the intended post-P0 sequencing is not yet fully
enforced by code.  The V3 work order says that P1 and P2 are forbidden until
P0 has a valid 48-epoch terminal, four checkpoints, SWA, manifest, exact
source-only facts, and no failure.  However, the current V3
`issue_capability()` delegates to the shared full-V1 issuer with a predecessor
validator that checks only the historical V2 failed graph.  It does not
descriptor-validate the completed V3 P0 graph before issuing a P1/P2
capability.  Root is obeying the sequencing rule operationally, so no P1/P2
attempt has been or will be issued under this gap; nevertheless, an
auditable successor must turn that prose rule into a capability condition.

The same audit corrected the optimistic parallel-runtime assumption.  The
shared full lifecycle currently calls GPU0-specific admission and binding
helpers and requires `CUDA_VISIBLE_DEVICES=0` with the exact physical-GPU0
UUID.  Consequently, unmodified V3 can run P1 and P2 only sequentially on
physical GPU0.  It cannot honestly place one arm on physical GPU1 merely
because GPU1 becomes idle.

The planned repair is admission/device-only and must not touch the active P0
closure.  After P0 naturally terminalizes, an additive successor will:

1. validate the complete immutable V3 P0 graph through held no-follow
   descriptors before P1/P2 capability issuance, again before attempt, and
   again before terminal;
2. expose only P1 and P2, never a second P0;
3. preserve the exact shared model, initial state, paired operator, optimizer,
   LR law, RNG/mask law, sampler, update count, checkpoint window, and SWA
   construction;
4. bind an explicit physical-GPU profile so P1 and P2 can use disjoint idle
   GPUs only if a reviewed backward-compatible device seam can prove exact
   UUID/CVD binding at every lifecycle boundary.

No running file will be edited to obtain this repair.  Until the successor is
implemented and independently tested, the conservative completion estimate
is the sequential P1/P2 schedule rather than the earlier parallel estimate.

The successor work order is now frozen at:

```text
tfpd_exploration/docs/WORKORDER_PACD_P1_P2_ADMISSION_MULTIGPU_V4_20260831.md
SHA-256 34a67357d66c36816457252e82d5ac7dcecf34340dafff1f372c67571dc09cd3
```

Independent Terra review is GO for post-P0 implementation.  The final design
uses a held V3-P0 terminal gate, exposes only P1/P2, assigns P1→physical GPU0
and P2→physical GPU1, and gives each arm an independent own-attempt-before-own-
runtime lifecycle.  It deliberately has no cross-process attempt barrier.
The required shared device-profile seam remains forbidden until this active
P0 has published and passed its immutable terminal audit.

The superseding work-order revision also freezes the exact future JSON paths
for the composite V2→V3-P0 predecessor witness and V4 attempt/launch/terminal/
failure device evidence.  Terra's second interface audit is GO: a future
mixed-lineage scorer can validate P0=V3 and P1/P2=V4 without guessing receipt
field placement.  Actual producer body literals remain correctly deferred
until those immutable terminals exist.

## Epoch 009 and the first post-partition timing

`epoch009.json` published naturally at 2026-08-31 12:44 HKT.  Its body and
canonical sidecar are regular mode-`0444` files and verify SHA
`a15c50443222544f915a8a0d3e19c89596006752f4dce7a08457a66d5f6f20c1`.
The run has now completed 339,250 optimizer steps.  Mean combined loss fell
from `0.762492` at epoch 000 to `0.516147` at epoch 009.

The epoch remains science-clean: P0 prediction mismatches, P0 identity
mismatches, RNG violations, prefix mutations, parameter-finiteness violations,
and zero-decoder-gradient steps are all zero; Adam is finite.  There are
33,912 positive encoder-gradient steps and 13 zero-encoder-gradient steps,
with every zero step exactly accepted under
`all_units_dropped_valid_zero`.  All 33,925 decoder steps retain positive
gradient evidence.  Sampler window and batched-index digests remain the exact
source-authority values.

This is the first complete epoch under the scheduler-only CPU partition.  It
took `3544.955 s` at `9.56994` paired steps/s.  That is about 1.8% faster than
the heavily contended epoch 008 (`9.40122` steps/s), but still about 6.6%
slower than the pre-overlap peak epoch 002 (`10.2802` steps/s).  The partition
therefore partially mitigates CPU contention; it does not eliminate it.  It
continues to change only scheduling, not model/data/RNG/optimizer semantics.

At the same check, PACD PID 783126 remained the sole physical-GPU0 compute
process and CDM replay PID 804509 remained the sole physical-GPU1 compute
process.  Their CPU affinity sets remained disjoint.  The second CDM replay
root still contained only its immutable attempt pair, so no CDM performance
claim was available.

## Mixed-lineage Score V2 independent audit status

The additive mixed-lineage scorer has completed its scheduler-codec repair and
an independent no-data review.  The exact combined Score V1+V2 suite passes
`41/41`; `py_compile`, Torch-free `python -S --dry-run`, and scoped `git diff
--check` also pass.  The explicit 50-file closure was independently rebuilt
twice as

`4d788ec1937483bd59e38fc11a05b5dc00c052719302a6bf86009b74c480003c`.

The repaired validator descriptor-reads and rehashes all 48 V4 epoch pairs,
checkpoints 44--47, SWA, and manifest; validates their ordered links,
gradient/sampler/finite/Adam/RNG evidence, final-four arithmetic, device
staging, exact P0 witness, lower V2 failure graph, and source authority; and
uses the real immutable V2 smoke schema rather than a synthetic approximation.
P1/P2 prediction and identity mismatch counters are correctly treated as
nonnegative descriptive facts, while P0 equality and all stochastic,
gradient, finite-state, and sampler invariants remain strict.  The live
binding also exact-compares the complete frozen producer-literal payload.
It now additionally binds the scheduler-isolation addendum SHA
`537781b92b77239f9e3f2085c257891aea7dd4f0e201bdc37cec0fabf3a1c706`
and exact-validates the separate top-level V4 attempt/launch/terminal
`scheduler` codec: arm-specific affinity, thread limits, tracked CDM route,
empty set intersection, and immutable historical observations.  The existing
exact `device` codec and all V1 scoring science remain unchanged.

This is **GO as a frozen no-data code candidate**, not permission to score.
Live P0/P1/P2 producer literals deliberately remain `None`, so authority mint
and execution still fail closed.  P0 must first reach and pass its immutable
terminal audit; P1/P2 must then be produced by the separately reviewed V4
admission/device successor.  No score authority, capability, result root,
target, checkpoint tensor, CUDA context, or GPU action was created during this
audit.  These Score V2 files are outside the active P0 execution closure and
do not affect either live GPU process.

## Post-P0 scheduler isolation addendum

A final read-only V4 interface audit found that the work order's broad
recommended CPU split would overlap the still-live CDM replay if P1 were
started immediately after P0.  No frozen authority or live file was changed.
Instead, the scheduler-only admission rule is now frozen separately in
`docs/AUDIT_PACD_V4_PARALLEL_ROUTE_ISOLATION_20260831.md`.

The post-P0 fixed placement is P1 on physical GPU0 with CPUs
`0-3,16-19`, and P2 on physical GPU1 with CPUs `4-15,20-31`.  P1 cannot
start before the complete P0 terminal gate; P2 additionally cannot start
while the current CDM PID owns GPU1 or any CPU in its set.  Neither process
may expand or mutate its affinity after attempt publication.  The future V4
implementation must bind and test this scheduler evidence, while leaving the
paired operator, epoch runner, model/data/RNG/optimizer contract and frozen
Score V2 science unchanged.

## CDM+P1 replay natural completion

At 2026-08-31 13:28 HKT, the second CDM+P1 replay process exited naturally,
released physical GPU1, and published `replay.json` plus its canonical
sidecar.  Both leaves are regular mode `0444`; the body verifies SHA-256

`6a2d015f31b1cff011c7c8c69b386272cf45cb7ebcb69bd23adf1dd2b010f164`.

The receipt reports `REPLAY_COMPLETE`, inference only, zero target optimizer,
backward, parameter-update and normalizer-update calls, exact sealed-model
restoration, exact F00/F01 anchors, exact M30 no-op, and all causal state chains
valid.  It processed 60,636 completed query trials in 6,555.63 seconds.  The
route had no decoder training or checkpoint modification.

The equal-session matrix and independently recomputed paired deltas versus
F00 are:

| Surface | Budget | F01 P1 carrier | F10 C1 weights | F11 C1 + P1 |
|---|---|---:|---:|---:|
| external | M4 | `+0.020463` (12/15) | `-0.021584` (3/15) | `-0.024468` (6/15) |
| external | M10 | `+0.002013` (10/15) | `-0.004453` (6/15) | `-0.006791` (7/15) |
| external | M30 | `0.000000` (0/15) | `-0.014963` (5/15) | `-0.014963` (5/15) |
| within | M4 | `+0.022081` (6/6) | `-0.001605` (4/6) | `+0.004812` (4/6) |
| within | M10 | `+0.006482` (5/6) | `-0.003200` (2/6) | `+0.012158` (5/6) |
| within | M30 | `0.000000` (0/6) | `+0.009085` (6/6) | `+0.009085` (6/6) |

Under the preregistered external-M4 promotion rule (`>= +0.01` and at least
10/15 positive sessions), only F01 passes.  F10 and F11 fail promotion.  All
three satisfy the broad safety floors, but F11 also fails the additivity rule:
its external-M4 mean is `0.221437`, which is `-0.044931` below the better
single factor F01 (`0.266368`).  The C1 weight arm therefore does not improve
the deployable P1 carrier; it reverses its M4 external gain.  This is a clean
negative interaction result, not a new positive combination.

This replay is not yet a route terminal.  At this boundary the root contains
only attempt and replay pairs; no `gates.json`, `terminal.json`, or
`failure.json` exists.  The earlier delete-and-recreate retry lineage remains
an audit caveat, so no final paper-grade claim is made until the external
owner publishes and an independent reviewer validates the remaining gate and
terminal graph.  No local agent launches that next stage.  GPU1 is now idle;
P2 nevertheless remains forbidden because the PACD P0 terminal gate has not
passed.

## CDM+P1 terminal and final gate audit

At 2026-08-31 13:37 HKT, the external owner naturally published the final
CDM+P1 terminal.  The root is now mode `0555` with the exact six-leaf topology

```text
attempt.json
attempt.json.sha256
replay.json
replay.json.sha256
terminal.json
terminal.json.sha256
```

Every leaf is a regular mode-`0444` file, every canonical sidecar verifies,
and no failure or extra leaf exists.  The terminal body SHA-256 is

`7482471d7948de016a4b6c45d3eab214c2d6f3a6c326da7ab64c554a43e575bb`.

It exact-links attempt
`d8c5c32cf41e18a42d9cabdcdcbcb69f4524395b1d1fa7ca46573a95c971012b`
and replay
`6a2d015f31b1cff011c7c8c69b386272cf45cb7ebcb69bd23adf1dd2b010f164`.
The terminal gate matches the independent recomputation above:

- F01 is `CELL_IMPROVES_OVER_F00`, promoted at external M4 `+0.020463`
  with 12/15 positive sessions;
- F10 and F11 are `CELL_FAILS_THE_F00_GATE`;
- the additivity gate is `ADDITIVITY_VIOLATED`, with F11 below F01 by
  `0.044931` external-M4 R2;
- all registered safety facts pass, including exact anchors, causal chains,
  sealed hyperparameters, initial-carrier invariance, M30 no-op, restored
  sealed model, no trust-region drift and zero target updates;
- the fired stop condition is exactly `additivity_violated`.

The route is therefore terminal as a scientific negative interaction result:
the P1 carrier remains a reproducible low-budget component, while C1 weights
and the combined C1+P1 system are not promoted.  The earlier failed first
attempt and delete/recreate reuse of the canonical root remain a provenance
caveat; this result is usable as audited diagnostic evidence but should not be
described as a pristine one-attempt lineage without that disclosure.

Physical GPU1 is released and idle.  The tracked-other-route scheduler state
for any future V4 receipt can now truthfully be the exact inactive `none`
mapping.  This releases the CDM-specific P2 resource condition, but it does
not release the mandatory P0-terminal predecessor gate; no PACD P1/P2 launch
is authorized yet.

## Epoch 010 verified event

`epoch010.json` published naturally at 2026-08-31 13:43 HKT.  Its regular
mode-`0444` body and canonical sidecar verify SHA-256

`52162245de5c4509eb8b65cecba4cebfc5e1f975b3a7854d906b0f06f1d2438b`.

The run has completed 373,175 optimizer steps.  Mean combined loss continues
down from `0.516147` at epoch 009 to `0.510538`.  Epoch wall time is
`3488.335 s`, or `9.72527` paired steps/s.  This is 1.6% faster than epoch 009
and 3.4% faster than the heavily contended epoch 008, although the epoch still
overlapped the CDM replay for most of its duration.  Epoch 011 will be the
first complete epoch after CDM released GPU1 and its CPU partition.

All scientific invariants remain exact: P0 prediction and identity mismatch,
RNG violation, prefix mutation, parameter-finiteness violation and zero
decoder-gradient counts are all zero; Adam is finite.  There are 33,903
positive encoder-gradient steps and 22 zero-encoder steps, with all 22 and
only those 22 accepted under `all_units_dropped_valid_zero`.  Decoder gradient
is positive on all 33,925 steps.  The four sentinels and both sampler-order
digests remain exact source-authority values.

At the observed epoch-010 rate, the remaining 37 epochs have a mechanical
duration of about 35.9 hours.  This remains a descriptive projection, not a
stop, resume, or selection rule.  PACD continues as the sole compute process
on physical GPU0; physical GPU1 remains idle.

## V4 pre-implementation isolation audit

At 2026-08-31 14:01 HKT, a requirements-level read-only audit found one
underspecified item in the future V4 isolation addendum: it required no active
host memory pressure or swap thrashing, but supplied neither an exact
measurement nor a receipt codec.  The addendum was amended outside the active
P0 closure to define a standard-library `/proc/meminfo` plus
`/proc/pressure/memory` observation.  Admission now requires at least 16 GiB
available memory, memory PSI `some avg10 <= 0.1`, and memory PSI
`full avg10 == 0.0`; swap occupancy is recorded descriptively rather than
mistaken for active thrashing.  Attempt, launch and final observations carry
monotonic timestamps and must be ordered.  The addendum also makes explicit
that its exact CPU sets supersede only the frozen work order's earlier
*recommended scheduler-only* placement.

Because the CDM route has already terminalized, a future live V4 producer
must record the inactive `none` tracked-route mapping.  The former live-CDM
shape remains only synthetic parser/adversarial coverage and is not an
admissible future production fact.  The amended addendum SHA-256 is

`7c2c27697329e13f53fc2dacd7981cec2f24b88f550ab1af38c266c4cc2dcc52`.

Only the Score V2 producer codec is being updated to understand this future
receipt evidence.  No P0 source, runtime, result, process, GPU or closure file
was changed.  Immediately after the document amendment, root reconstructed
the active P0 closure through the standard-library path: 48 explicit files,
SHA-256
`3356fb124ff4d18a2035bfda8e7eacc28a1642ce3f36b290155a5bcdf7faea2f`,
with Torch absent from the verifier process.

Terra then updated only the Score V2 plan, producer codec and focused test.
Root independently reran the complete no-user-site/no-CUDA Score V1+V2 suite:
52 tests passed with one pre-existing TorchMetrics deprecation warning.  The
inert `python -S` dry route still reports
`producer_literals_deferred=true`, `score_authorized=false`, and no Torch
import.  The explicit 50-file Score V2 closure was reconstructed twice as

`71eaf16540e5caf65629ed69bcf9ea6b4ec3b106b28487205511668a7be55eb5`.

The codec now rejects successful V4 producer graphs with low memory, either
PSI violation, inconsistent/false `pass`, non-finite values, non-monotonic
observation times or a future live-CDM claim.  It accepts both nearly full
swap and disabled swap only when the fixed memory/PSI admission formula
passes.  Score science, the 756-row order, comparators, activity-isolation
regime and gates are unchanged.  A second root-side reconstruction again
proved the active P0 closure remains the same 48-file `3356fb...ea2f` value.

## Epoch 011 verified event

`epoch011.json` published naturally at 2026-08-31 14:38:27 HKT.  Its regular
mode-`0444` body and canonical sidecar verify SHA-256

`308d48c4eb1e34e8dbdccddbe5cb3acdd36540437fc7333e6b7b5594d56c3470`.

The run has completed 407,100 optimizer steps.  Mean combined loss declined
from `0.510538` at epoch 010 to `0.502229`, a paired change of `-0.008309`
(`-1.63%`).  Epoch wall time was `3320.542 s`, or `10.21671` paired steps/s.
This is the first complete epoch after the CDM route released GPU1 and its
CPU partition, and is 5.05% faster than epoch 010.  That throughput difference
is descriptive one-epoch resource evidence, not a scientific effect estimate.

All scientific and execution invariants remain exact.  P0 prediction and
identity mismatches, RNG violations, prefix mutations, parameter-finiteness
violations and zero decoder-gradient steps are all zero; Adam is finite.
There are 33,904 positive encoder-gradient steps and 21 zero-encoder steps,
with all 21 and only those 21 accepted under
`all_units_dropped_valid_zero`.  Decoder gradient is positive on all 33,925
steps.  All four sentinel rows pass paired dropout, identity, prediction and
RNG-transition equality, and their required gradient evidence is positive.
The two sampler digests exact-match `source_authority.json`.

Root reconstructed the active 48-file P0 closure after the event as
`3356fb124ff4d18a2035bfda8e7eacc28a1642ce3f36b290155a5bcdf7faea2f`
with Torch absent.  No failure or terminal exists.  At the epoch-011 rate,
the remaining 36 epochs have a mechanical duration of approximately 33.2
hours; this is descriptive only and creates no stop, selection or launch
rule for another arm.

## Epochs 012--013 verified events

`epoch012.json` and `epoch013.json` published naturally.  Their regular
mode-`0444` body/sidecar pairs verify respectively to:

```text
epoch012  33ec3b47b8950380ae87ea65e26852c3413b77dff12b6e8c0d3257541602d00a
epoch013  ecb9e6b245e27d40d9d2f063736b97f15b8b11e981fb83bc59b27b6857663ea4
```

The run has completed 474,950 optimizer steps.  Mean combined loss continued
down from `0.502229` at epoch 011 to `0.494557` and then `0.488380`.  Epoch 012
took `3334.771 s` at `10.17311` paired steps/s; epoch 013 took `3342.175 s` at
`10.15057` paired steps/s.  These rates are back inside the pre-contention
runtime envelope and are descriptive only.

Both receipts pass the complete no-Torch read-only audit:

- exact epoch indices, 33,925 steps per epoch, and cumulative-step arithmetic;
- zero P0 prediction and identity mismatches;
- zero RNG violations, prefix mutations, finiteness violations, and decoder
  zero-gradient steps;
- finite Adam and finite anchor/short/combined loss summaries;
- exact source-authority window and batched-index sampler digests;
- exact sentinel steps `0`, `1`, `16,962`, and `33,924`;
- every zero encoder step, and only those steps, accepted under the typed
  `all_units_dropped_valid_zero` rule.

Epoch 012 has 33,903 positive encoder steps plus 22 typed accepted-zero steps;
epoch 013 has the same exact coverage.  Decoder gradient is positive on all
33,925 steps in each epoch.  Across all receipts 000--013, body digests,
canonical sidecars, modes, step arithmetic, sampler authority, P0 equality,
RNG/prefix/finite-state evidence and gradient-coverage laws all validate.

At the 2026-08-31 16:42 HKT read-only boundary, PID 783126 remains active as
the sole physical-GPU0 compute process.  Physical GPU1 is idle.  The active
attempt's 48 execution files still match their recorded byte counts and
SHA-256 values exactly; the recorded closure remains
`3356fb124ff4d18a2035bfda8e7eacc28a1642ce3f36b290155a5bcdf7faea2f`.
No P0 failure or terminal exists, and no P1/P2 capability or result root is
authorized.  At the recent epoch duration, the remaining 34 epochs project to
about 31.6 hours; this remains monitoring information, not an intervention or
selection rule.

## Epoch 014 verified event and live SUA coexistence

`epoch014.json` published naturally at 2026-08-31 17:26:04 HKT.  Its regular
mode-`0444` body and canonical sidecar verify SHA-256

`6688d88fca7ec860a656bb6f30e6d088128da8a4fda1d54022914b8c88833493`.

The run has completed 508,875 exact optimizer steps.  Mean combined loss is
`0.482755`, down from `0.488380` at epoch 013.  Epoch wall time was
`3372.223 s`, or `10.06013` paired steps/s.  This is 0.89% slower than epoch
013 and remains within the recent runtime envelope; it is not evidence of a
scientific change.

Epoch 014 and the complete receipt prefix 000--014 pass the no-Torch audit:

- every body and canonical sidecar is regular mode `0444` and digest-valid;
- epoch indices, 33,925-step cardinality and cumulative-step arithmetic are
  exact;
- P0 prediction and identity mismatches, RNG violations, prefix mutations,
  parameter-finiteness violations and zero decoder-gradient steps are zero;
- Adam is finite;
- decoder gradient is positive on all 33,925 steps;
- encoder gradient is positive on 33,916 steps, with exactly nine zero steps
  and all nine accepted under `all_units_dropped_valid_zero`;
- source-authority window and batched-index sampler digests remain exact;
- sentinel steps are exactly `0`, `1`, `16,962`, and `33,924`; and
- all sentinels preserve dropout, identity, prediction and RNG-transition
  equality with finite materialized parameters.

The active closure anchors remain byte-identical to launch, including shared
lifecycle `62d4b712...`, shared runner `adb6c471...`, paired core
`9331288c...`, and the V3 plan/predecessor/smoke leaves.  No checkpoint,
manifest, SWA, failure or terminal exists yet.

A repaired SUA carrier-transfer successor became active on physical GPU1
during epoch 014.  The two routes remain separate by GPU and result root:

```text
PACD P0  PID 783126  GPU0/CVD0  affinity 0-3,16-19
SUA P1   PID 824234  GPU1/CVD1  affinity 0-31
```

The SUA broad CPU affinity overlaps the PACD set, so the live state must not
be described as CPU-disjoint.  At the 17:26 HKT sample each process used about
one CPU, load average was approximately 2, available memory was about
48.4 GiB, and both memory and CPU PSI averages were zero.  Thus there is no
observed material host pressure at this boundary, while the affinity overlap
remains a genuine risk.  Monitoring is read-only: no affinity, priority,
process, root, GPU or worker was changed.

This coexistence cannot admit a future V4 arm.  If SUA remains active after
the P0 terminal, P1 conflicts with its broad CPU affinity and P2 conflicts
with both that affinity and physical GPU1.  Both V4 arms must wait for natural
SUA release and their own fixed resource gates; they may not move devices or
alter the other route.

## SUA natural terminal and resource release

At 2026-08-31 17:36 HKT, the separate SUA P1 process exited naturally and
released physical GPU1 and its broad `0-31` CPU affinity.  Its immutable
terminal graph validates independently of PACD: attempt
`e28aa50c...dcc`, replay `f8d3dca1...179`, terminal
`00a3c409...a1`, with all six leaves regular mode `0444` and exact body,
sidecar and cross-receipt links.  The result is external-M4 `+0.008083`
(`10/15`) and therefore fails the registered `+0.01` promotion floor; M30 is
an exact no-op.

The release removes the live SUA scheduling conflict but changes no PACD
scientific value, byte, process or decision.  P0 PID `783126` remains the sole
GPU compute owner, its active closure is untouched, and no V4 implementation,
capability or result root is admitted before the full P0 terminal graph and a
fresh post-terminal resource audit.

## Post-SUA closure and isolation recheck

At 2026-08-31 17:48 HKT, root recomputed the active execution closure twice
through the standard-library, no-user-site, no-bytecode path.  Both passes
contained exactly 48 explicit files and reproduced

`3356fb124ff4d18a2035bfda8e7eacc28a1642ce3f36b290155a5bcdf7faea2f`.

Torch was absent from `sys.modules`.  All 18 currently published P0 JSON
bodies (attempt, launch, source authority and epochs 000--014) and their
canonical sidecars are regular mode `0444` and digest-valid.  PID `783126`
remains the only compute application on physical GPU0; physical GPU1 has no
compute application.  The additive V4 producer package and V4 result root are
both absent, proving that design/scorer preparation has not crossed the
pre-terminal implementation boundary.

## Epoch 015 verified event

`epoch015.json` published atomically at 2026-08-31 18:21:36 HKT.  Its body and
canonical sidecar are regular mode `0444`, and both bind SHA-256

`1f75e4993da29173e985317f23f2cc777a37965feba86e0ae8059863ec80ba9c`.

Root independently validated the complete epoch000--015 prefix against the
actual receipt schema.  Every epoch has exactly 33,925 optimizer steps and
the cumulative count is now 542,800.  Epoch 015 has mean combined loss
`0.475843`, wall time `3329.498 s`, and `10.189224` paired steps/s.  Adam is
finite; P0 prediction and identity mismatches, RNG violations, prefix
mutations, parameter-finiteness violations and zero decoder-gradient steps
are all zero.

Decoder gradient is positive on all 33,925 steps.  Encoder gradient is
positive on 33,915 steps; the ten zero-gradient steps are all accepted under
the exact `all_units_dropped_valid_zero` reason.  The parameter topology stays
at 29 materialized plus two inactive lazy parameters.  All four sentinels at
steps `0`, `1`, `16,962`, and `33,924` preserve identity, prediction,
dropout-mask and RNG-transition equality.  Window and batched sampler SHA-256
values remain exact against source authority.

The active 48-file closure was reconstructed twice after the epoch event and
still equals `3356fb124ff4d18a2035bfda8e7eacc28a1642ce3f36b290155a5bcdf7faea2f`
without importing Torch.  PID `783126` remains live on physical GPU0; GPU1 has
no compute application.  No failure, terminal, checkpoint, SWA or manifest
exists yet, and V4 remains unimplemented and unauthorized.

## Epoch 016 verified event and new GPU1 coexistence

`epoch016.json` published naturally at 2026-08-31 19:17:47 HKT.  Its body and
canonical sidecar are regular mode `0444` and bind SHA-256

`8e2edf79a8a28620294e15369a150905186d560f90ac6afe75529c779fc348c3`.

Root independently validated the complete epoch000--016 prefix.  Epoch 016
contains exactly 33,925 optimizer steps and brings the cumulative count to
576,725.  Mean combined loss is `0.469908`; wall time is `3368.597 s`, or
`10.070959` paired steps/s.  Relative to epoch 015 this is a `1.16%` rate
decrease, within the established runtime envelope and not evidence of a
scientific or resource-induced change.

Adam and all materialized parameters remain finite.  P0 prediction/identity
mismatches, RNG violations, prefix mutations, parameter-finiteness violations
and zero decoder-gradient steps are zero.  Decoder gradient is positive for
all 33,925 steps.  Encoder gradient is positive for 33,901 steps; all 24 zero
steps have the exact `all_units_dropped_valid_zero` reason.  Sampler authority
and all four sentinel equality/RNG/finite laws remain exact.  The active
48-file closure was reconstructed twice after publication and remains
`3356fb124ff4d18a2035bfda8e7eacc28a1642ce3f36b290155a5bcdf7faea2f`
without importing Torch.

At 19:19 HKT a separate local M2 carrier route was active on physical GPU1:

```text
PACD P0   PID 783126  GPU0/CVD0  affinity 0-3,16-19  5 threads
M2 local  PID 832229  GPU1/CVD1  affinity 0-31        81 threads
```

The GPU bindings and result ownership are distinct; no cross-root access or
GPU crossing was observed.  The M2 process has broad CPU permission and thus
is not CPU-disjoint from P0, although it used approximately one CPU at the
sample.  Load average was about 2.08, CPU/memory/IO PSI averages were all zero,
and available RAM was about 50 GiB.  No material host pressure or P0 slowdown
was observed.  Monitoring did not change either process, affinity, priority,
GPU, worker count or result root.

This coexistence is acceptable for the still-running historical P0 only while
health remains normal.  It does not admit any V4 arm: P1 is barred by the P0
terminal gate, and P2 additionally conflicts with the live GPU1 owner.  A
future V4 launch must wait for natural release and repeat the frozen resource
gate from fresh process state.

### Local-M2 terminal and GPU1 release

The local M2 process exited naturally at approximately 19:20 HKT and physical
GPU1 returned to `0%` utilization with no compute PID.  Its result root
published an exact six-leaf attempt/replay/terminal graph; terminal SHA-256 is
`414f1e825a0ac0067e7c44cb2b34ea3c024381def041de77c45fc5d7d6ec3c82`.
The scientific result is a LOCAL-protocol external-M4 delta of exactly `0`
with `0/6` positive sessions; all within M4/M10/M30 deltas and external M30
are also `0`.  It is not an official-interface M2 claim.

The short overlap caused no observed P0 integrity, GPU-placement, throughput
or pressure failure.  Its release removes the immediate GPU1/CPU-affinity
conflict, but it does not admit V4 while P0 remains active and does not replace
the mandatory fresh resource check after P0 terminal.

## Epoch 017 verified event

`epoch017.json` published naturally at 2026-08-31 20:13:31 HKT.  Its body and
canonical sidecar are mode `0444`, with exact SHA-256

`2474cf4c76b4c2e3a734995ed4be5c841c13c417ea9802a3edeaeff1887d1062`.

Root independently validated the complete epoch000--017 prefix.  Epoch 017
has exactly 33,925 optimizer steps, bringing the cumulative count to 610,650.
Mean combined loss is `0.465515`; wall time is `3341.026 s`, or `10.154066`
paired steps/s.  This is `0.83%` faster than epoch 016 and lies inside the
stable pre-existing runtime envelope.

Adam and model/parameter evidence remain finite.  P0 identity/prediction
mismatches, prefix mutations, RNG violations, parameter-finiteness violations
and zero decoder-gradient steps are zero.  Decoder gradient is positive on all
33,925 steps.  Encoder gradient is positive on 33,910 steps, with all 15 zero
steps accepted under `all_units_dropped_valid_zero`.  Sampler authority and
the four fixed sentinel equality/RNG/finite laws remain exact.

The active 48-file closure was reconstructed twice after publication and
remains `3356fb124ff4d18a2035bfda8e7eacc28a1642ce3f36b290155a5bcdf7faea2f`
without importing Torch.  PID `783126` remains the only compute process on
GPU0; GPU1 is idle.  No failure, terminal, checkpoint, SWA or manifest exists,
and no V4 implementation or capability is authorized.

## Epoch 018 verified event

`epoch018.json` published naturally at 2026-08-31 21:08:55 HKT.  Its mode-
`0444` body and canonical sidecar bind SHA-256
`8608a4eeae3675ddd9603fd877cabdbcf3eb6d3945e2edd8ed204b4462bd5d21`.
Root validated the complete epoch000--018 prefix: 33,925 steps per epoch,
644,575 cumulative steps, exact sampler authority, zero P0/RNG/prefix/finite
violations and all fixed sentinel laws.  Mean combined loss is `0.459278`;
epoch wall time is `3321.844 s`, or `10.212700` paired steps/s (`0.58%` faster
than epoch 017).

Decoder gradient is positive on all 33,925 steps.  Encoder gradient is
positive on 33,903 steps; all 22 zero steps use the exact
`all_units_dropped_valid_zero` reason.  Adam and the 29-materialized/two-lazy
parameter topology remain finite.  The active 48-file closure was reconstructed
twice after publication and remains `3356fb...ea2f` without importing Torch.
P0 remains the only GPU0 compute process, GPU1 is idle, and no failure,
terminal, checkpoint, SWA, manifest or V4 action exists.

## Epoch 019 verified event

`epoch019.json` published naturally at 2026-08-31 22:04:35 HKT. Its mode-
`0444` body and canonical sidecar bind SHA-256
`e935df4aef8c8106dec09a7f6a66ad15a60026f70067777b4a1c1aff5decb646`.
Root independently validated the complete epoch000--019 prefix: 33,925 steps
per epoch and 678,500 cumulative optimizer steps. Mean combined loss is
`0.453976`; epoch wall time is `3337.766 s`, or `10.163984` paired steps/s.

Adam remains finite. P0 identity/prediction mismatches, prefix mutations, RNG
violations, parameter-finiteness violations and zero decoder-gradient steps
are all zero. Decoder gradient is positive on all 33,925 steps. Encoder
gradient is positive on 33,912 steps; all 13 zero steps are accepted under the
exact `all_units_dropped_valid_zero` reason. The 29-materialized/two-lazy
parameter topology, sampler authority, valid-bin count and four fixed
sentinel pair/RNG/finite laws remain exact.

The active 48-file closure was reconstructed twice after publication and
remains `3356fb124ff4d18a2035bfda8e7eacc28a1642ce3f36b290155a5bcdf7faea2f`
without importing Torch. PID `783126` remains the only GPU0 compute process;
GPU1 is idle. No failure, terminal, checkpoint, SWA, manifest, P1/P2
capability or V4 implementation action exists.

## Epoch 020 verified event and M1/GPU1 isolation baseline

`epoch020.json` published naturally at 2026-08-31 23:00:06 HKT.  Its regular
mode-`0444` body and canonical basename sidecar both bind SHA-256
`1dfa4cd8561ef2846798d0fa6f7c8c96316838a24361e2a74ed1372eab4630c4`.
Root independently validated the complete epoch000--020 prefix: every epoch
has exactly 33,925 optimizer steps, giving 712,425 cumulative steps.  The
sampler authorities remain the fixed window digest
`94659d1a9b56cd1f972ac20724bec27d49766daded3854db6d70e7a79a56e6d5`
and batched-index digest
`ae6d27ffdc2d1f8b820db412f2d6845871b94fe13cfb7ddcfbd2686375f1a624`.

Mean combined loss is `0.447952`; epoch wall time is `3328.663 s`, or
`10.191779` paired steps/s.  Adam is finite.  P0 identity/prediction
mismatches, prefix mutations, RNG violations, parameter-finiteness violations
and zero decoder-gradient steps are all zero.  Decoder gradient is positive on
all 33,925 steps.  Encoder gradient is positive on 33,901 steps; all 24 zero
steps are accepted only under the exact `all_units_dropped_valid_zero` rule.
The 29-materialized/two-lazy topology, 1,600 valid bins and all four fixed
sentinel pair/RNG/finite laws remain exact.

After publication, root reconstructed the explicit 48-file execution closure
twice as
`3356fb124ff4d18a2035bfda8e7eacc28a1642ce3f36b290155a5bcdf7faea2f`
without importing Torch.  No failure, terminal, checkpoint 44--47, SWA,
manifest, P1/P2 capability or V4 implementation action exists.

At the 2026-08-31 23:04 HKT isolation baseline, PACD PID `783126` remained the
only CUDA compute process and was bound to physical GPU0 with CPU affinity
`0-3,16-19`.  Physical GPU1 had no compute owner (`0%`, 23 MiB).  The newly
announced M1 comparison was therefore not yet observable as a CUDA process;
GPU1 remains reserved for it and PACD will not claim that device while the M1
route is active.  CPU, memory and IO PSI were all zero.  This is a read-only
baseline, not a claim that the future M1 PID has already been validated.

By 23:15 HKT the M1 route had created its additive work order, package and
inert public CLI under `m1_heldin_heldout_gap_v1`, but still had no CUDA PID
or result root.  Those paths are outside the PACD closure; a fresh closure
reconstruction remained exact.  Static review found that the M1 physical
payload describes a GPU1 constraint but does not yet enforce actual CVD/UUID/
PID or CPU-affinity isolation.  Root therefore recorded the mandatory
pre-launch envelope in `AUDIT_M1_PACD_CONCURRENT_ISOLATION_20260831.md` and
will not treat the route as concurrently admissible until that gate passes.

The M1 route then terminalized naturally at 23:35:52 HKT.  Its immutable
terminal SHA is
`b948976e624a5a53fbc5ac5c2f5f7cf6e379c8408b1a749b80a3b5dbada49d51`;
post-terminal GPU1 is idle and PACD PID `783126` remains the sole GPU0 compute
owner.  The M1 launch receipt records logical `cuda:0` with a declared CVD=1
operator constraint, but does not persist the physical UUID/PID/affinity and
the watcher did not capture the short scoring window.  It must therefore not
be described as CPU-only or as physically attested GPU1 execution.  The exact
scientific result and this limitation are recorded in
`RESULT_M1_HELDIN_HELDOUT_GAP_V1_20260831.md`.  No PACD process, root or
closure byte was changed.

## Epoch 021 verified event and post-M1 health check

`epoch021.json` published naturally at 2026-08-31 23:56:30 HKT.  Its regular
mode-`0444` body and canonical sidecar both bind SHA-256
`cb5fd3c9e475f17506d31db784a6a09f21e802d9ec78a2ba90b7b30ded139fa1`.
Root independently validated the complete epoch000--021 prefix: 33,925 steps
per epoch and 746,350 cumulative optimizer steps.  Mean combined loss is
`0.444111`; epoch wall time is `3380.709 s`, or `10.034876` paired steps/s.

Adam remains finite.  P0 identity/prediction mismatches, prefix mutations,
RNG violations, parameter-finiteness violations and zero decoder-gradient
steps are all zero.  Decoder gradient is positive on all 33,925 steps.
Encoder gradient is positive on 33,906 steps; all 19 zero steps use only the
exact `all_units_dropped_valid_zero` reason.  The 29-materialized/two-lazy
topology, sampler authorities, 1,600 valid bins and all four sentinel laws
remain exact.

The M1 scoring interval (approximately 23:23--23:35 HKT) falls inside this
epoch.  Relative to epoch020, wall time rose by `1.56%` and throughput fell by
`1.54%`.  The epoch receipt has no sub-epoch timing or device telemetry, so
that ordinary-sized variation cannot be attributed to M1.  All scientific
and safety invariants remain clean; there is no evidence of an attributable
interference failure.  Post-M1, GPU1 is idle and P0 remains the sole GPU0
compute owner.

The active explicit 48-file closure was reconstructed twice after publication
as
`3356fb124ff4d18a2035bfda8e7eacc28a1642ce3f36b290155a5bcdf7faea2f`
without importing Torch.  No failure, terminal, checkpoint 44--47, SWA,
manifest, P1/P2 capability or V4 implementation exists.

## Epoch 022 verified event and live M1 T0/C1 coexistence

`epoch022.json` published naturally at 2026-09-01 00:53:44 HKT.  Its regular
mode-`0444`, single-link body and canonical basename sidecar bind SHA-256
`417d04deb9de6c6fc99aafd8334ba645099d9cc3a775d1fac6bc78eecd8dd625`.
Root independently validated the complete epoch000--022 prefix: all 23 epoch
pairs have exact body/sidecar hashes and names, 33,925 optimizer steps per
epoch and 780,275 cumulative optimizer steps.  The result root contains
exactly 26 bodies/52 leaves at this boundary: attempt, launch, source
authority and epochs 000--022.  There is no failure, terminal, checkpoint
44--47, SWA or manifest.

Mean combined loss is `0.441841`; wall time is `3431.599 s`, or `9.886061`
paired steps/s.  Adam remains finite.  Identity/prediction mismatches, prefix
mutations, RNG violations, parameter-finiteness violations and zero decoder
gradient steps are all zero.  Decoder gradient is positive on all 33,925
steps.  Encoder gradient is positive on 33,901 steps; the remaining 24 steps
are accepted only under the exact `all_units_dropped_valid_zero` rule.  Both
sampler digests remain fixed, all four sentinels pass their pair/RNG/finite
laws and every sentinel carries 1,600 valid bins.

A separate M1 T0/C1 pipeline became a live compute owner during this epoch.
This is a different route from the completed held-in/held-out metric job
described above.  The read-only resource binding at 00:58 HKT was:

```text
PACD P0  PID 783126  physical GPU0 / CVD0  affinity 0-3,16-19
M1 T0     PID 847764  physical GPU1 / CVD1  affinity 0-31
pipeline  PID 847532  /tmp/m1_t0c1_pipeline.sh
```

The M1 process command invokes
`m1_t0c1_prefix_v1.driver.execute_arm(..., device='cuda:0', arm='t0')`;
because its environment has `CUDA_VISIBLE_DEVICES=1`, logical `cuda:0` maps
to physical GPU1.  Its active result subroot is
`tfpd_exploration/results/m1_t0c1_prefix_v1/t0`.  At the sample, that subroot
contained only immutable attempt and launch pairs; C1 and phase3 had not
started.  PACD remained the sole GPU0 compute owner, so GPU placement is
disjoint.  However, M1's broad `0-31` CPU affinity includes PACD's
`0-3,16-19` set, so CPU isolation is not exact.  Host CPU, memory and IO PSI
were all zero and no OOM/nonfinite/failure event was present.

Epoch022 throughput is `1.48%` below epoch021 and about `3.00%` below
epoch020.  M1 T0 started at 00:36:52 HKT, late inside the epoch, and the P0
receipt has no sub-epoch timing series.  The slowdown therefore cannot be
attributed to M1; it is recorded as a scheduling observation, not an
interference finding.  Scientific and safety invariants remain exact.

No agent changed either process, affinity, priority, environment or result
root.  No new PACD GPU job may start.  In particular, future P2 must wait for
the complete M1 T0→C1→phase3 pipeline to exit naturally and for a fresh GPU1,
PID, CPU-affinity, host-pressure and root-freshness gate.  P1 remains blocked
by the missing P0 terminal independently of GPU1.

After this event the explicit 48-file P0 closure was reconstructed twice as
`3356fb124ff4d18a2035bfda8e7eacc28a1642ce3f36b290155a5bcdf7faea2f`
without importing Torch.  The M1 paths are outside that closure and no P0
closure byte changed.

### M1 T0 fail-close and GPU1 physical release

At 2026-09-01 01:13:33 HKT, the independent M1 T0 stage fail-closed naturally
with `TrainerError('epoch loss row coverage drift')`.  Its exact six-leaf
attempt/launch/failure graph is immutable mode `0444`; body SHA-256 values are:

```text
attempt  3c72ddea583c80b7c1488e3dbfd2541fcacff2324d96cfff6c362ac6d4214744
launch   83ff27462308bc87278eea3df49dcf93e24ed85b0db501d68f6ad0ead4257e13
failure  694c93cf63a12f470e65c5b62c8830d49456048753aa0d4abbe5135715fea8b8
```

The failure records `prepared=true`, `optimizer_steps_completed=0`,
`target_metric_only=true` and `terminal_published=false`.  The pipeline logged
`chain stops`; its shell and T0 child exited naturally, and no C1 or phase3
root was created.  GPU1 returned to its normal 23 MiB display footprint with
no compute process.  P0 remained the sole GPU0 compute owner and continued
without a failure or invariant violation.

This is a physical release observation, not authorization for PACD to claim
GPU1.  The other route's owner may still diagnose or issue a reviewed retry,
and P0 has not terminalized.  Root will not modify, retry, rename or reuse the
M1 root.  Future PACD work remains blocked exactly as before.

## Epoch 023 verified event

`epoch023.json` published naturally at 2026-09-01 01:49:43 HKT.  Its regular
mode-`0444`, single-link body and canonical sidecar bind SHA-256
`471cb788abc26fc5717f2d255ce9945e01daf872b3863086efe687603fe92a06`.
Root independently validated the complete epoch000--023 prefix: all 24 epoch
pairs have exact mode/link/basename/SHA integrity, 33,925 optimizer steps per
epoch and 814,200 cumulative steps.  The root contains exactly 27 bodies/54
leaves at this boundary: attempt, launch, source authority and epochs
000--023.  No failure, terminal, checkpoint 44--47, SWA or manifest exists.

Mean anchor/combined/short loss is `0.437443`, with min `0.026991` and max
`15.212149`.  Wall time is `3356.247 s`, or `10.108016` paired steps/s.  Adam
is finite.  Identity/prediction mismatches, prefix mutations, RNG violations,
parameter-finiteness violations and zero decoder-gradient steps are all zero.
Decoder gradient is positive on all 33,925 steps.  Encoder gradient is
positive on 33,908 steps; exactly 17 zero steps are accepted under
`all_units_dropped_valid_zero`.  The 29-materialized/two-lazy topology,
fixed sampler digests, 1,600 valid bins and all four sentinel pair/RNG/finite
laws remain exact.

At this event GPU0 had only P0 PID `783126`; GPU1 had no compute process and
the failed M1 T0 pipeline had no retry or successor.  Host CPU, memory and IO
PSI remained zero.  No agent used GPU1, changed P0 affinity or modified either
result root.

The explicit 48-file execution closure was reconstructed twice after the
epoch as
`3356fb124ff4d18a2035bfda8e7eacc28a1642ce3f36b290155a5bcdf7faea2f`
without importing Torch.  No V4 implementation or capability exists.

## Epoch 024 verified event

`epoch024.json` published naturally at 2026-09-01 02:45:31 HKT.  Its regular
mode-`0444`, single-link body and canonical sidecar bind SHA-256
`e70438ebee18c199c4f2dd2a87708faad3fa559f30ad1c65af329acc9925d9ae`.
Root independently validated epoch000--024: all 25 pairs have exact
mode/link/basename/SHA integrity, 33,925 optimizer steps per epoch and 848,125
cumulative steps.  The live root contains exactly 28 bodies/56 leaves:
attempt, launch, source authority and epochs 000--024.  No failure, terminal,
checkpoint 44--47, SWA or manifest exists.

Mean anchor/combined/short loss is `0.433082`, min `0.023574`, max
`14.932917`.  Wall time is `3346.123 s`, or `10.138599` paired steps/s.  Adam
is finite.  Identity/prediction mismatches, prefix mutations, RNG violations,
parameter-finiteness violations and zero decoder-gradient steps are zero.
Decoder gradient is positive on all 33,925 steps.  Encoder gradient is
positive on 33,910 steps; the 15 accepted zero steps carry only
`all_units_dropped_valid_zero`.  The 29-materialized/two-lazy topology,
sampler digests, 1,600 valid bins and all four sentinel laws remain exact.

GPU0 had only P0 PID `783126`; GPU1 had no compute owner and no M1 retry.
Host CPU, memory and IO PSI remained zero.  The explicit 48-file closure was
again reconstructed twice as
`3356fb124ff4d18a2035bfda8e7eacc28a1642ce3f36b290155a5bcdf7faea2f`
without Torch.  No V4 implementation, capability or root action exists.

## Epoch 032 verified event and M1 T0-to-C1 transition

`epoch032.json` published naturally at 2026-09-01 10:22:10 HKT.  Its regular
mode-`0444`, single-link body and canonical sidecar bind SHA-256
`355062a43de031be56baa0df118c1ece9e155fa1984aade69e8e3b8f266adaca`.
Root independently validated the complete epoch000--032 prefix.  The P0 root
contains exactly 36 bodies and 72 body/sidecar leaves: attempt, launch, source
authority and 33 ordered epoch receipts.  Every leaf has exact mode, link,
basename-sidecar and body digest integrity.  There is no failure, terminal,
checkpoint 44--47, SWA or manifest.

Epoch032 completes 33,925 optimizer steps and raises the cumulative count to
1,119,525.  Mean anchor/short/combined loss is `0.4057208332`; wall time is
`3505.3452 s`, or `9.6780769` paired steps/s.  Adam is finite.  Prediction and
identity mismatches, prefix mutations, RNG violations, parameter-finiteness
violations and zero decoder-gradient steps are all zero.  Decoder gradient is
positive on all 33,925 steps.  Encoder gradient is positive on 33,911 steps;
all 14 zero-encoder steps and only those steps use the typed
`all_units_dropped_valid_zero` exception.  Sampler digests, 1,600 valid bins
and sentinel coordinates `[0,1,16962,33924]` remain exact.

During this epoch the independent M1 T0 route terminalized successfully and
its unchanged pipeline shell naturally started C1.  The accepted T0 graph has
29 bodies/58 leaves, terminal SHA
`4aea40a317047beec059505231bb9996190c4eb2235392d08a3c948cf7ab9aa9`,
20 epochs, 99,020 optimizer steps, best=last epoch 19, SWA disabled and zero
target optimizer/backward/update.  C1 PID `886913` started with CVD1 and broad
affinity `0-31`; at the first post-transition sample it remained pre-CUDA and
GPU1 had no compute application.  That temporary gap is not a release event:
the M1 route remains active and both future V4 CPU profiles remain blocked.
PACD stayed the sole GPU0 owner and no cross-root descriptor, cross-GPU owner,
OOM or sustained PSI pressure was observed.

At 10:25 HKT, C1 PID `886913` naturally materialized as the sole physical
GPU1 compute owner, using about 1,150 MiB.  PACD PID `783126` remained the sole
GPU0 owner.  CVD and CPU mappings remained C1=`1`/`0-31` and
P0=`0`/`0-3,16-19`; host PSI stayed zero.  This is a clean one-owner-per-GPU
transition, not a release for either V4 arm.

## Epoch 033 verified event

`epoch033.json` published naturally at 2026-09-01 11:20 HKT.  Its body and
canonical sidecar are regular, single-link mode-`0444` files and bind
SHA-256
`0e4c2426ea5da6acb8245fc2c47d5db76b8cba9f916976401f026a0488f93c7a`.
Root independently validated the complete epoch000--033 prefix: exactly 37
bodies/74 leaves, no extra or missing topology, and exact mode, link,
basename-sidecar and SHA integrity.  No checkpoint 44--47, SWA, manifest,
failure or terminal exists.

Epoch033 records 33,925 optimizer steps and 1,153,450 cumulative steps.  Mean
anchor/short/combined loss is `0.4037890943`, with min `0.0249172077` and max
`14.9687776566`.  Wall time is `3505.3843 s`, or `9.6779687` paired steps/s.
Adam is finite.  Prediction/identity mismatches, prefix mutations, RNG and
parameter-finiteness violations and zero decoder-gradient steps are zero.
Decoder gradient is positive on all 33,925 steps; encoder gradient is positive
on 33,901 steps; all 24 and only the zero-encoder steps use
`all_units_dropped_valid_zero`.  Sampler, valid-bin and sentinel laws remain
exact.

M1 C1 remained live as the sole GPU1 owner while PACD remained the sole GPU0
owner.  No C1 terminal/failure/phase3 receipt, second GPU owner, cross-root
descriptor, OOM or sustained PSI pressure was observed.

## Epoch 034 verified event

`epoch034.json` published naturally at 2026-09-01 12:19 HKT with body
SHA-256
`65d9fc22deee46ea2df64a2278dc3549b8501c160fdf64feadfd8698ab57e450`.
Its body and canonical sidecar are regular, single-link mode-`0444` files.
Root independently validated the complete epoch000--034 prefix: exactly 38
bodies/76 leaves, exact topology/mode/link/basename/SHA integrity, and no
checkpoint 44--47, SWA, manifest, failure or terminal.

Epoch034 records 33,925 optimizer steps and 1,187,375 cumulative steps.  Mean
anchor/short/combined loss is `0.4025784268`, min `0.0200983435`, max
`15.1354284286`; wall time is `3541.5240 s`, or `9.5792093` paired steps/s.
Adam is finite.  Prediction/identity mismatches, prefix mutations, RNG and
parameter-finiteness violations and zero decoder-gradient steps are zero.
Decoder gradient is positive on all 33,925 steps; encoder gradient is positive
on 33,912; all 13 zero-encoder steps use only
`all_units_dropped_valid_zero`.  Sampler, valid-bin and sentinel laws remain
exact.

M1 C1 remained the sole GPU1 owner and PACD the sole GPU0 owner.  No C1
terminal/failure/phase3, second owner, cross-root descriptor, OOM or sustained
PSI pressure appeared during the epoch.

## Epoch 035 verified event

`epoch035.json` published naturally at 2026-09-01 13:16 HKT with body
SHA-256
`c366a1265b2bea203141b2dd25c883849ecf24581de816de8b96e11520cc1891`.
Its body and canonical sidecar are regular, single-link mode-`0444` files.
Root independently validated the complete epoch000--035 prefix: exactly 39
bodies/78 leaves, exact topology/mode/link/basename/SHA integrity, and no
checkpoint 44--47, SWA, manifest, failure or terminal.

Epoch035 records 33,925 optimizer steps and 1,221,300 cumulative steps.  Mean
anchor/short/combined loss is `0.3973192692`, min `0.0200331304`, max
`14.9474067688`; wall time is `3411.9707 s`, or `9.9429343` paired steps/s.
Adam is finite.  Prediction/identity mismatches, prefix mutations, RNG and
parameter-finiteness violations and zero decoder-gradient steps are zero.
Decoder gradient is positive on all 33,925 steps; encoder gradient is positive
on 33,910; all 15 and only the zero-encoder steps use
`all_units_dropped_valid_zero`.  Materialized/lazy counts remain exactly
29/2, valid bins remain exactly 1,600, all four sentinel coordinates and
paired equality laws pass, and both sampler digests equal source authority.

PACD PID `783126` remained the sole GPU0 compute owner with CVD0 and affinity
`0-3,16-19`.  The M1 pipeline had already released naturally; GPU1 had no
compute owner and no broad M1 CPU mask remained.  No new unknown GPU owner,
cross-root descriptor, OOM or CPU/memory/IO PSI pressure appeared.  The P0
48-file closure remained exact and no V4 implementation/capability/root was
created.

## Epoch 036 verified event

`epoch036.json` published naturally at 2026-09-01 14:12 HKT with body
SHA-256
`9062a1e79c8a616c791d6fe0d6f393be545eafd7be30a99df2df08201de16ef1`.
Root independently validated epoch000--036 as one complete prefix: exactly
40 bodies/80 leaves, all regular single-link mode-`0444`, canonical sidecars
and matching body SHA values, with no checkpoint, SWA, manifest, terminal,
failure or extra leaf.

Epoch036 records 33,925 updates and 1,255,225 cumulative updates.  Mean
anchor/short/combined loss is `0.3969830376`, min `0.0214234442`, max
`15.2888050079`.  Wall time is `3359.5588 s`, or `10.0980521` paired
steps/s.  This returns to the long-run baseline and is `1.56%` faster than
epoch035, so the earlier epoch30--34 slowdown is not continuing.

Adam and the 29-materialized/two-lazy topology are finite.  Prediction and
identity mismatches, prefix mutations, RNG and finiteness violations and zero
decoder-gradient steps are zero.  Decoder gradient is positive for all 33,925
steps; encoder gradient is positive for 33,911; all 14 and only the zero
encoder steps use `all_units_dropped_valid_zero`.  Sampler, valid-bin,
sentinel and paired-equality laws remain exact.

M1's repaired Phase 3 overlapped part of this epoch on GPU1 with a broad CPU
mask, then terminalized naturally.  P0 remained the sole GPU0 owner and host
PSI was zero at terminal audit.  No causal slowdown is observed in epoch036;
its throughput instead recovered.  The P0 closure remains exact and V4 bytes
remain absent.

## Epoch 025 verified event

`epoch025.json` published naturally at 2026-09-01 03:41:01 HKT.  Its regular
mode-`0444`, single-link body and canonical sidecar bind SHA-256
`72f2b91fe93f0620889aa67936e6606f4864374b099b385639dcd66d9dccb271`.
Root independently validated epoch000--025: all 26 pairs have exact
mode/link/basename/SHA integrity, 33,925 optimizer steps per epoch and 882,050
cumulative steps.  The live root has exactly 29 bodies/58 leaves: attempt,
launch, source authority and epochs 000--025.  No failure, terminal,
checkpoint 44--47, SWA or manifest exists.

Mean anchor/combined/short loss is `0.428623`, min `0.023441`, max
`14.947138`.  Wall time is `3327.951 s`, or `10.193960` paired steps/s.  Adam
is finite.  Identity/prediction mismatches, prefix mutations, RNG violations,
parameter-finiteness violations and zero decoder-gradient steps are zero.
Decoder gradient is positive on all 33,925 steps.  Encoder gradient is
positive on 33,908 steps; the 17 accepted zero steps use only
`all_units_dropped_valid_zero`.  The 29-materialized/two-lazy topology,
fixed sampler digests, 1,600 valid bins and four sentinel laws remain exact.

GPU0 had only P0 PID `783126`; GPU1 remained without a compute owner and the
M1 pipeline had no retry/C1/phase3.  Host PSI remained zero.  The explicit
48-file closure reconstructed twice after publication as
`3356fb124ff4d18a2035bfda8e7eacc28a1642ce3f36b290155a5bcdf7faea2f`
without Torch.  No V4 implementation, capability or root action exists.

## Epoch 026 verified event

`epoch026.json` published naturally at 2026-09-01 04:36:29 HKT.  Its regular
mode-`0444`, single-link body and canonical sidecar bind SHA-256
`822839fd6c47a8f64d05a577c3f491785ea2cd5004de54fe9afbd2a706cddca6`.
Root independently validated epoch000--026: all 27 pairs have exact
mode/link/basename/SHA integrity, 33,925 optimizer steps per epoch and 915,975
cumulative steps.  The live root has exactly 30 bodies/60 leaves: attempt,
launch, source authority and epochs 000--026.  No failure, terminal,
checkpoint 44--47, SWA or manifest exists.

Mean anchor/combined/short loss is `0.423903`, min `0.028237`, max
`15.017713`.  Wall time is `3325.701 s`, or `10.200856` paired steps/s.  Adam
is finite.  Identity/prediction mismatches, prefix mutations, RNG violations,
parameter-finiteness violations and zero decoder-gradient steps are zero.
Decoder gradient is positive on all 33,925 steps.  Encoder gradient is
positive on 33,904 steps; the 21 accepted zero steps use only
`all_units_dropped_valid_zero`.  The 29-materialized/two-lazy topology,
fixed sampler digests, 1,600 valid bins and four sentinel laws remain exact.

GPU0 had only P0 PID `783126`; GPU1 remained without a compute owner and the
M1 route had no retry/C1/phase3.  Host PSI remained zero.  The explicit
48-file closure reconstructed twice after publication as
`3356fb124ff4d18a2035bfda8e7eacc28a1642ce3f36b290155a5bcdf7faea2f`
without Torch.  No V4 implementation, capability or root action exists.

## Epoch 037 verified event

`epoch037.json` published naturally at 2026-09-01 15:08 HKT.  Its regular,
single-link, mode-`0444` body and canonical sidecar both bind SHA-256
`f6e26afe88151b7d76981a50fbeca63c466329b53d3f015b50d1352df257b982`.
Root independently validated the complete epoch000--037 prefix: exactly 41
bodies and 82 body/sidecar leaves, with exact topology, canonical sidecar
basenames, modes, link counts and body SHA values.  No checkpoint 44--47,
SWA, manifest, terminal, failure or extra leaf exists.

Epoch037 records 33,925 optimizer updates and 1,289,150 cumulative updates.
Mean anchor/short/combined loss is `0.3922466938`, min `0.0225718636`, max
`15.1446676254`.  Wall time is `3347.0182 s`, or `10.1358876` paired
steps/s.  The epoch00--29 throughput baseline is `10.079687` steps/s, so the
latest epoch is `0.56%` faster than baseline.  Epoch30--36 averaged
`9.744109` steps/s, but epochs35--37 successively recovered to `9.942934`,
`10.098052` and `10.135888`; there is no evidence of a continuing slowdown.

Adam and the 29-materialized/two-lazy parameter topology are finite.
Prediction and identity mismatches, prefix mutations, RNG and finiteness
violations and zero decoder-gradient steps are all zero.  Decoder gradient is
positive on all 33,925 steps; encoder gradient is positive on 33,908; all 17
and only the zero-encoder steps use the typed
`all_units_dropped_valid_zero` condition.  Valid bins remain exactly 1,600,
all four sentinel laws pass, and the immutable sampler digests remain
`94659d1a9b56cd1f972ac20724bec27d49766daded3854db6d70e7a79a56e6d5`
for windows and
`ae6d27ffdc2d1f8b820db412f2d6845871b94fe13cfb7ddcfbd2686375f1a624`
for batches.

PID `783126` remained the sole physical-GPU0 compute owner with CVD0 and
affinity `0-3,16-19`; GPU1 had no compute owner.  No cross-result-root file
descriptor, persistent CPU/memory/IO pressure, OOM or owner crossover was
observed.  The 48-file execution closure remains exact.  P0 is now 38/48
epochs complete (`79.17%`); V4 implementation, capability and result roots
remain absent pending the full P0 terminal.

## Epoch 038 verified event

`epoch038.json` published naturally at 2026-09-01 16:04 HKT.  Body and
canonical sidecar are regular, single-link mode-`0444` files and bind
SHA-256
`26f7e0ddd89cc373046e58f56ba2209b87f12c7665ecdd876a4f2e26e3a5cce6`.
Root independently validated epoch000--038 as exactly 42 bodies/84 leaves
with exact topology, modes, canonical sidecars and body digests.  No
checkpoint, SWA, manifest, failure, terminal or extra leaf exists.

Epoch038 records 33,925 updates and 1,323,075 cumulative updates.  Mean
anchor/short/combined loss is `0.3892404799`, min `0.0250587463`, max
`15.0370740891`.  Wall time is `3336.7795 s`, or `10.1669889` paired
steps/s: `0.87%` above the epoch00--29 baseline and the third successive
recovery epoch after epoch34.  There is no continuing slowdown signal.

Adam and the 29-materialized/two-lazy topology are finite.  All P0
prediction/identity mismatches, prefix mutations, RNG and finiteness
violations and zero decoder-gradient steps are zero.  Decoder gradient is
positive for all 33,925 steps; encoder gradient is positive for 33,905; all
20 zero-encoder steps are exactly the typed
`all_units_dropped_valid_zero` case.  Valid bins, four sentinels, sampler
digests and paired-equality laws remain exact.

P0 remains the sole GPU0 owner with CVD0 and affinity `0-3,16-19`; GPU1 has
no compute owner.  No cross-root descriptor, persistent PSI pressure or OOM
was observed.  P0 is 39/48 epochs complete (`81.25%`), while all V4 code,
capability and root actions remain gated on the full terminal.

### New independent M1 50-epoch smoke on GPU1

At 16:15 HKT an independently owned M1 route started
`launch_m1_t0c1_prefix_v1_50ep.py --stage smoke` on physical GPU1.  Its
Python PID is `1020808`, CVD is `1`, and its live result family is
`tfpd_exploration/results/m1_t0c1_prefix_v1_50ep/smoke`.  The smoke had
published only immutable attempt and launch pairs at the observation.  Root
and the P0 watcher did not launch, edit, reprioritize, signal or otherwise
intervene in this route.

GPU devices remain disjoint: P0 PID `783126` is the sole GPU0 owner, and M1
PID `1020808` is the sole GPU1 owner.  The M1 process nevertheless has CPU
affinity `0-31`, which overlaps P0's fixed `0-3,16-19` set.  Both jobs use
single-thread BLAS/OpenMP settings, and the first joint sample had zero
CPU/memory/IO PSI, about 50.7 GiB available memory and no OOM.  This is a
real scheduler overlap and will be evaluated from subsequent P0 epoch wall
time rather than assumed harmless.  No affinity or process action is
authorized; V4 admission additionally remains blocked until this independent
GPU1 owner has naturally released and a fresh resource check passes.

The M1 smoke then terminalized naturally at 16:21 HKT and PID `1020808`
exited.  Root validated its exact six-body/12-leaf immutable topology:
attempt, launch, T0 smoke, C1 smoke, equality and terminal pairs, all regular
single-link mode-`0444`, with no failure or extra leaf.  Terminal SHA-256 is
`ddb7cb9b40c204f62bc94a472f6c85978fa4b5b915c2bf4973b411e8f428fb0c`;
it reports identical batch order, dropout-probability stream and LR sequence.
GPU1 returned to no compute owner.  The overlap was short and had no PSI/OOM
signal, but its effect, if any, remains deferred to epoch039's completed wall
time.  This natural release is historical evidence only; fresh V4 admission
checks remain mandatory.

At 16:24 HKT the same independent M1 pipeline proceeded from its accepted
smoke into the 50-epoch T0 stage.  The live Python PID is `1025681` (parent
shell `1025680`), CVD1, physical GPU1, with result child `t0`.  Its immutable
attempt and launch SHA-256 values are respectively
`680a699a1c3b2bc670620608079a71b37b3d61978eeb0d5bdd49f34113d5974c`
and
`699c36a36bba141f7f92a2db593d1aaeb8d9534d3e1bb084dc662dada2ada93e`;
the route closure is
`76ed5cef64a035c940acb34d72fb6b1135287969626ecb68d60cad9e73dfbe85`.
GPU device ownership remains disjoint, but the T0 affinity `0-31` overlaps
P0's `0-3,16-19`.  Initial PSI remained zero with no OOM, owner crossover or
cross-root descriptor.  No intervention is authorized; P0 monitoring will
continue to quantify any effect by completed epoch receipts.

At 16:33 HKT that T0 PID was terminated by its external operator and GPU1
released.  The operator moved the smoke and T0 roots under
`retired_closure_76ed5cef` after editing its launcher admission gate, which
changed the matched-pair implementation closure.  Immutable abort-note
SHA-256 is
`220657ae9417b60194a1e1192543c271bf41b31d7659c6144d1c1f9a89f7be0e`.
It states that the smoke was scientifically valid but superseded by stale
closure, while T0 received SIGTERM after about 12 minutes with only attempt
and launch pairs, no terminal and no failure receipt.  Therefore the retired
T0 is incomplete and is not a result.

PACD P0 was not signalled or edited, remained the sole GPU0 owner, and kept
its exact closure and epoch038 prefix.  GPU1 returned idle and host PSI
remained zero.  Root and the P0 watcher did not request, perform or assist
the external abort, root move or launcher edit.  A possible successor M1
retry must be treated as a new closure and monitored without intervention.

## Epoch 039 verified event and new M1 concurrent lineage

`epoch039.json` published naturally at 2026-09-01 17:01 HKT.  Body and
canonical sidecar are regular, single-link mode-`0444` files and bind
SHA-256
`31f4863ce9d7b49bb9785b2cdb0624baa91b30b091a8d57bb016e6985050f5fa`.
Root independently validated epoch000--039 as exactly 43 bodies/86 leaves,
with exact topology, modes, canonical sidecars and body digests.  No
checkpoint, SWA, manifest, failure, terminal or extra leaf exists.

Epoch039 records 33,925 updates and 1,357,000 cumulative updates.  Mean
anchor/short/combined loss is `0.3902852590`, min `0.0199288949`, max
`15.0300378799`.  Wall time is `3443.0851 s`, or `9.8530821` paired
steps/s: `2.25%` below the epoch00--29 baseline and about 77.4 s slower than
baseline-predicted wall time.  This is a visible but modest throughput change.
It overlaps in time with external M1 smoke/T0/C1 preparation, but zero host
PSI and the absence of sub-epoch resource attribution prevent a causal claim.

Adam and the 29-materialized/two-lazy topology remain finite.  P0 prediction
and identity mismatches, prefix mutations, RNG and finiteness violations and
zero decoder-gradient steps are all zero.  Decoder gradient is positive on
all 33,925 steps; encoder gradient is positive on 33,906; all 19 zero-encoder
steps are exactly the typed `all_units_dropped_valid_zero` condition.  Valid
bins, sentinels, sampler digests and paired-equality laws remain exact.

The external M1 route also completed a fresh smoke under closure
`ae1f47fbb9f07a886a3c96676144793b0abe646b83336a98e030b269be84f1e7`.
Root validated its exact six-body/12-leaf immutable graph; terminal SHA-256 is
`7d09b3d1571e21bddb0fcf54deec5f2e6c5ebc439b194fc583bc7d57f7165ab8`,
status `COMPLETE_MATCHED_SMOKE_EQUALITY`, with identical batch order,
dropout-probability stream and LR sequence.  New T0 PID `1052875` and C1 PID
`1055233` then started concurrently with CVD1 and affinity `0-31`; at the
audit T0 alone had materialized as GPU1's compute owner.  P0 remains the only
GPU0 owner and has no descriptor into either M1 root.  P0 is now 40/48 epochs
complete (`83.33%`); monitoring continues without intervention.

## Epoch 040 verified event

`epoch040.json` published naturally at 2026-09-01 18:01 HKT.  Body and
canonical sidecar are regular, single-link mode-`0444` files and bind
SHA-256
`ec060eab89239c084d6bf88c3a1512881fb96e7e2935da48ea6862c4056e3683`.
Root independently validated epoch000--040 as exactly 44 bodies/88 leaves,
with exact topology, modes, canonical sidecars and body digests.  No
checkpoint, SWA, manifest, failure, terminal or extra leaf exists.

Epoch040 records 33,925 updates and 1,390,925 cumulative updates.  Mean
anchor/short/combined loss is `0.3889519096`, min `0.0206724890`, max
`14.9822158813`.  Wall time is `3601.7210 s`, or `9.4191083` paired
steps/s.  This is `6.55%` below the epoch00--29 baseline, 236.0 s slower than
baseline-predicted wall time, and 4.40% slower than epoch039.  The interval
overlapped about 51 minutes of the external M1 T0/C1 double process with
affinity `0-31`; this is a meaningful temporal association but not a causal
estimate because no sub-epoch counterfactual exists.

All P0 correctness evidence remains exact.  Adam and the 29-materialized/
two-lazy topology are finite.  Prediction/identity mismatches, prefix
mutations, RNG and finiteness violations and zero decoder-gradient steps are
zero.  Decoder gradient is positive on all 33,925 steps; encoder gradient is
positive on 33,910; all 15 zero-encoder steps use exactly the typed
`all_units_dropped_valid_zero` condition.  Valid bins, sentinels, sampler
digests and paired-equality laws remain unchanged.

P0 remains GPU0's sole owner; M1 T0/C1 remain GPU1's only two owners, with no
device crossover or cross-root descriptor.  Host PSI remains zero and no OOM
occurred.  The slowdown therefore does not justify process intervention or
scientific invalidation.  It does justify the hard disjoint-CPU requirement
in the proposed PACD-2UP cohort.  P0 is now 41/48 epochs complete (`85.42%`).

## Epoch 041 verified event and persistent throughput warning

`epoch041.json` published naturally at 2026-09-01 19:05 HKT.  Body and
canonical sidecar are regular, single-link mode-`0444` files and bind
SHA-256
`e470c040e4ee7b490ff0348f6dd9d71da1241408692f540a6bb4d35c46c6389a`.
Root independently validated epoch000--041 as exactly 45 bodies/90 leaves,
with exact topology, modes, canonical sidecars and body digests.  No
checkpoint, SWA, manifest, failure, terminal or extra leaf exists.

Epoch041 records 33,925 updates and 1,424,850 cumulative updates.  Mean
anchor/short/combined loss is `0.3899381616`, min `0.0201738589`, max
`15.3110036850`.  Wall time is `3830.8542 s`, or `8.8557272` paired
steps/s: `12.14%` below the epoch00--29 baseline and 5.98% below epoch040.
Epoch040 and epoch041 are now two consecutive completed epochs below 9.5
steps/s, so the throughput warning is persistent rather than a single-epoch
fluctuation.

The warning is operational, not scientific.  Adam and the 29-materialized/
two-lazy topology are finite.  Prediction/identity mismatches, prefix
mutations, RNG and finiteness violations and zero decoder-gradient steps are
zero.  Decoder gradient is positive on all 33,925 steps; encoder gradient is
positive on 33,913; all 12 zero-encoder steps are exactly the typed
`all_units_dropped_valid_zero` case.  Valid bins, sentinels, sampler digests
and paired-equality laws remain exact.

The epoch overlapped sustained external M1 T0/C1 same-GPU concurrency with
both processes using affinity `0-31`; GPU1 utilization reached 66% and host
load about 7, while P0 remained isolated on GPU0.  No sustained PSI, OOM,
GPU crossover or cross-root descriptor was observed, so causal attribution
remains unavailable.  No process intervention is justified or authorized.
P0 is 42/48 epochs complete (`87.50%`); the expected consequence is roughly
45--60 additional minutes over the remaining six epochs if the slowdown
persists.

## Epoch 042 verified event

`epoch042.json` published naturally at 2026-09-01 20:09 HKT.  Its regular,
single-link mode-`0444` body and canonical sidecar bind SHA-256
`80a69557cabcb2deb3b531c8ebfdebe2a2a2ce6f0afe9a6b07560ab131621d52`.
Root independently validated epoch000--042 as exactly 46 bodies/92 leaves,
with exact topology, modes, sidecars and body digests.  No checkpoint, SWA,
manifest, terminal, failure or extra leaf exists.

Epoch042 records 33,925 updates and 1,458,775 cumulative updates.  Mean
anchor/short/combined loss is `0.3871353772`, min `0.0172540452`, max
`14.9219064713`.  Wall time is `3817.8648 s`, or `8.8858568` paired
steps/s.  It remains `11.84%` below the long-run baseline, but is 0.34% faster
than epoch041.  Epoch040--042 are three consecutive epochs below 9.5 steps/s,
confirming persistent degradation without evidence of continuing acceleration
of the slowdown.

All science and execution evidence remains exact: finite Adam and parameter
topology; zero prediction/identity mismatch, prefix mutation, RNG, finiteness
violation and decoder-zero steps; 33,925 positive decoder steps; 33,909
positive encoder steps; all 16 encoder-zero steps exactly typed as
`all_units_dropped_valid_zero`; unchanged valid-bin, sampler, sentinel and
paired-equality laws.

M1 T0/C1 remained simultaneous GPU1 owners with broad CPU affinity, while P0
remained GPU0's sole owner.  No persistent PSI, OOM, GPU crossover or
cross-root descriptor appeared.  No intervention is authorized.  P0 is now
43/48 epochs complete (`89.58%`) with five epochs remaining before terminal
artifacts.

## Epoch 043 verified event

`epoch043.json` published naturally at 2026-09-01 21:10 HKT.  Body and
canonical sidecar are regular, single-link mode-`0444` files and bind
SHA-256
`452902fb61179f149e52a2b535507f5f49f6b73861fc0a5815d9ae9dcebc8485`.
Root independently validated epoch000--043 as exactly 47 bodies/94 leaves,
with exact topology, modes, sidecars and body digests.  No checkpoint, SWA,
manifest, terminal, failure or extra leaf exists.

Epoch043 records 33,925 updates and 1,492,700 cumulative updates.  Mean
anchor/short/combined loss is `0.3877569515`, min `0.0190430041`, max
`15.1022777557`.  Wall time is `3667.6287 s`, or `9.2498458` paired
steps/s.  It is 8.23% below baseline but 4.10% faster than epoch042.  Epoch
040--043 are four consecutive epochs below 9.5, while the last two show no
further deterioration.

All correctness evidence remains exact: finite Adam and parameter topology;
zero prediction/identity mismatch, prefix mutation, RNG, finiteness violation
and decoder-zero steps; 33,925 positive decoder steps; 33,905 positive encoder
steps; all 20 encoder-zero steps exactly typed; unchanged valid-bin, sampler,
sentinel and paired-equality laws.  M1 T0/C1 remained GPU1 owners with broad
affinity; P0 remained GPU0's sole owner.  There was no PSI, OOM, GPU crossover
or cross-root descriptor.

P0 is now 44/48 epochs complete (`91.67%`).  Epochs 44--47 must each add its
immutable final-window checkpoint pair in addition to the epoch receipt; the
next audits therefore switch from the 47-body prefix law to the
epoch-plus-checkpoint topology.

## Epoch 044 and checkpoint verified event

`epoch044.json` and `epoch044.pt` published naturally at 2026-09-01 22:11:58
HKT. The epoch body SHA-256 is
`16ba92799d0e0c9b54fca68256df28fa0e5f2a2673ee0aedc12f8b607364a55f`;
the checkpoint body SHA-256 is
`f01bd4bc83e72f12ab1430a01dbfdc45b2732ee0506a5239a89761336ebb0139`.
Root independently validated exactly 49 bodies/98 body-and-sidecar leaves:
attempt, launch, source authority, epochs 000--044 and the epoch044
checkpoint. Every leaf is a regular, single-link mode-`0444` file with a
canonical sidecar and verified body digest. No SWA, manifest, terminal,
failure or extra leaf exists. The checkpoint is 14,056,032 bytes; the audit
did not load or interpret any checkpoint tensor.

Epoch044 records 33,925 updates and 1,526,625 cumulative updates. Mean/min/max
loss is `0.3891818326` / `0.0232340004` / `15.0401859283`; wall time is
`3691.1608 s`, or `9.1908758` paired steps/s. This is 8.82% below the
epoch00--29 baseline and is the fifth consecutive epoch below 9.5 steps/s.
The last two epochs do not show monotone deterioration, but the operational
slowdown is persistent.

All correctness laws remain exact: finite Adam; 29 materialized and two lazy
parameters; zero prediction/identity mismatch, prefix/RNG/finiteness
violations and decoder-zero steps; 33,925 positive decoder steps; 33,909
positive encoder steps; and all 16 encoder-zero steps exactly typed
`all_units_dropped_valid_zero`. Sampler and valid-bin authorities are
unchanged. The receipt records model-state digest
`23babb8a48c5ca0a895e7484966a8d12a91295c809d404092a63e3636b8a2ef2`
and optimizer-state digest
`dc71799a9e32daad68febc2733728e4941043dbaac8d84d46f227c951e90b5b1`.

At audit, P0 remained GPU0's sole compute owner. M1 T0/C1 remained the two
GPU1 owners and raised GPU1 utilization to roughly 58--60% with about 2.35
GiB used, but retained broad overlapping CPU affinity. There was no GPU
crossover, third owner, OOM, persistent PSI or cross-root descriptor. This
supports controlled two-up GPU packing only with four disjoint CPU masks; it
does not authorize intervention in the live jobs.

P0 is now 45/48 epochs complete (`93.75%`). Epochs 45--47 must each publish
the matching checkpoint, followed by exact final-four SWA, manifest and
terminal artifacts.

## Epoch 045 and checkpoint verified event

`epoch045.json` and `epoch045.pt` published naturally at 2026-09-01
23:14:54--23:15:14 HKT. Their body SHA-256 values are respectively
`d1f5d0e903508d3834ab7305edde4ecde487e2247c4c6afec00eaa8e88a4bd1b`
and
`efaa259bb47b7c1a5503373f5b377c457ba17184059563d78bbd90b933619ed7`.
The descriptor-only audit verified exactly 51 bodies/102 leaves: three base
bodies, epochs 000--045 and checkpoints 044--045. All leaves are regular,
single-link mode-`0444` files with exact canonical sidecars and body digests;
there are no symlinks, extras, failure, SWA, manifest or terminal. The
14,056,032-byte checkpoint remained opaque and no tensor was loaded.

Epoch045 records 33,925 updates and 1,560,550 cumulative updates. Mean/min/max
loss is `0.3850463758` / `0.0152799832` / `14.9557075500`, so the mean fell
below both epoch044 (`0.3891818326`) and the previous local minimum at
epoch042 (`0.3871353772`). Wall time is `3772.1109 s`, or `8.9936381`
steps/s: the sixth consecutive epoch below 9.5 and 10.77% below the long-run
baseline. Optimization is slower operationally but not divergent.

Every correctness law remains exact: finite Adam; 29 materialized/two lazy
parameters; zero prediction/identity mismatch, prefix/RNG/finiteness
violations and decoder-zero steps; 33,925 positive decoder steps; 33,911
positive encoder steps; and 14 exactly typed accepted-zero encoder steps.
Valid-bin, sampler and sentinel laws are unchanged. The receipt records
model-state digest
`f8115a88a43e487169f7af1f021e9c56e47ffe5925ed7d2170f32be670369246`
and optimizer-state digest
`865061e1174d3bbd5d488445a710f5c9f321a37847ce039f70a27adf63f97625`.

P0 remained the sole GPU0 owner; M1 T0/C1 remained the only two GPU1 owners.
There was no third owner, cross-GPU process, cross-root descriptor, OOM or
sustained PSI. P0 is now 46/48 epochs complete (`95.83%`). Only epochs 046
and 047, their checkpoints, final-four SWA, manifest and terminal remain.

## Epoch 046 and checkpoint verified event

`epoch046.json` and `epoch046.pt` published naturally at 2026-09-02
00:19:04--00:19:09 HKT. Their body SHA-256 values are
`50bd62d0949acb4d900f84bf2f9b8e2d128b7022e205e7095b0e81f19085dea0`
and
`1c6e753bd9c92a3b0b97a73edd3b72e25c10a6344c21787db5fba385e5848998`.
The descriptor-only audit verified the exact 53 bodies/106 leaves: three
base bodies, epochs 000--046 and checkpoints 044--046. Every leaf is regular,
single-link mode `0444` with an exact canonical sidecar/body digest; no
symlink, extra, failure, SWA, manifest or terminal exists. The 14,056,032-byte
checkpoint remained opaque and no tensor was loaded.

Epoch046 records 33,925 updates and 1,594,475 cumulative updates. Mean/min/max
loss is `0.3848239100` / `0.0218992047` / `14.9747810364`, a further small
improvement over epoch045 and a new local minimum. Wall time is `3846.8722 s`,
or `8.8188528` steps/s: the seventh consecutive epoch below 9.5 and 12.51%
below the long-run baseline. Science remains healthy despite lower scheduler
throughput.

All correctness laws remain exact: finite Adam; 29 materialized/two lazy
parameters; zero prediction/identity mismatch, prefix/RNG/finiteness
violations and decoder-zero steps; 33,925 positive decoder steps; 33,909
positive encoder steps; and 16 exactly typed accepted-zero encoder steps.
The receipt binds model-state digest
`da8b7168d59ee491551314c130d420f57ba901f9c546bbab698f1d68dc1e36d8`
and optimizer-state digest
`318838d153bd5dcc3fb603445c7c6653c13b8bc2eba78f8f421155ab8546d109`.

P0 remained GPU0's sole owner. The independently owned M1 pipeline had
completed T0/C1 and entered its `probe` arm on GPU1; no probe terminal or
score existed at this audit. There was no third owner, GPU crossover,
cross-root FD, OOM or sustained PSI. P0 is now 47/48 epochs complete
(`97.92%`). Only epoch047/checkpoint047 and terminal finalization remain.

## Epoch 031 verified event

`epoch031.json` published naturally at 2026-09-01 09:23:42 HKT.  Its regular
mode-`0444`, single-link body and canonical sidecar bind SHA-256
`70e183a511d71806435bb6c52f5a62f12c9da3c6c264ad6679c57c8ada3a4f8d`.
Root independently validated the exact epoch000--031 prefix: 32 epoch pairs,
33,925 optimizer steps per epoch and 1,085,600 cumulative steps.  The live
root has exactly 35 bodies/70 leaves: attempt, launch, source authority and
epochs 000--031.  No failure, terminal, checkpoint 44--47, SWA or manifest
exists.

Mean anchor/combined/short loss is `0.407618`, min `0.021524`, max
`15.053006`.  Wall time is `3547.729 s`, or `9.562454` paired steps/s.  Adam
is finite.  Identity/prediction mismatches, prefix mutations, RNG violations,
parameter-finiteness violations and zero decoder-gradient steps are zero.
Decoder gradient is positive on all 33,925 steps.  Encoder gradient is
positive on 33,910 steps; the 15 accepted zero steps use only
`all_units_dropped_valid_zero`.  The 29-materialized/two-lazy topology,
fixed sampler digests, 1,600 valid bins and sentinel coordinates
`0,1,16962,33924` remain exact.  Every sentinel passes prediction/identity/
dropout-pair equality, RNG-transition equality and calibration-prefix
nonmutation.

At the event, P0 PID `783126` remained the sole physical-GPU0 compute owner,
and M1 PID `866228` remained the sole physical-GPU1 compute owner.  M1 still
had only canonical attempt/launch receipts and no T0 terminal/failure, C1 or
phase3.  The earlier transient unknown GPU1 PID `877688` did not recur.  CPU,
memory and IO PSI averages were zero; no OOM or owner crossing was present.
Monitoring made no process, source or result-root change.

The explicit 48-file P0 closure reconstructed twice after publication as
`3356fb124ff4d18a2035bfda8e7eacc28a1642ce3f36b290155a5bcdf7faea2f`
without importing Torch.  No V4 implementation, capability, root reservation
or launch exists.

## Epoch 030 verified event

`epoch030.json` published naturally at 2026-09-01 08:24:31 HKT.  Its regular
mode-`0444`, single-link body and canonical sidecar bind SHA-256
`2d1768f278325cd4efd1aa712738495808ba042397d79350c2178ebc246bc903`.
Root independently validated the exact epoch000--030 prefix: 31 epoch pairs,
33,925 optimizer steps per epoch and 1,051,675 cumulative steps.  The live
root has exactly 34 bodies/68 leaves: attempt, launch, source authority and
epochs 000--030.  No failure, terminal, checkpoint 44--47, SWA or manifest
exists.

Mean anchor/combined/short loss is `0.411347`, min `0.024351`, max
`15.039490`.  Wall time is `3508.247 s`, or `9.670071` paired steps/s.  Adam
is finite.  Identity/prediction mismatches, prefix mutations, RNG violations,
parameter-finiteness violations and zero decoder-gradient steps are zero.
Decoder gradient is positive on all 33,925 steps.  Encoder gradient is
positive on 33,914 steps; the 11 accepted zero steps use only
`all_units_dropped_valid_zero`.  The 29-materialized/two-lazy topology,
fixed sampler digests, 1,600 valid bins and sentinel coordinates
`0,1,16962,33924` remain exact.  Every sentinel passes prediction/identity/
dropout-pair equality, RNG-transition equality and calibration-prefix
nonmutation.

At the event, PACD PID `783126` remained the sole physical-GPU0 compute
owner, while the third M1 T0 retry PID `866228` remained the sole physical-
GPU1 compute owner.  M1 still had only canonical attempt/launch receipts and
no T0 terminal/failure, C1 or phase3.  CPU affinity remained overlapping but
CPU, memory and IO PSI averages were all zero; there was no OOM, second GPU
PID or cross-GPU owner.  Monitoring made no process, source or result-root
change.

The explicit 48-file P0 closure reconstructed twice after publication as
`3356fb124ff4d18a2035bfda8e7eacc28a1642ce3f36b290155a5bcdf7faea2f`
without importing Torch.  No V4 implementation, capability, root reservation
or launch exists.

### New M1 T0 retry after epoch027

At 2026-09-01 06:11:08 HKT, an external owner archived the earlier M1 T0
failure directory as `t0_failed_loss_coverage_drift`, created a fresh
canonical `t0`, and launched a resume-from-T0 pipeline.  New shell/Python PIDs
were `860201/860203`; the Python process had `CUDA_VISIBLE_DEVICES=1` and
affinity `0-31`.  Its initial canonical root had exact attempt/launch pairs
only, with body SHA values
`4e9d9a7ea26679d69a5ae1b0529c0025f17c077e0322df99dd1a1c5c5e7aa883`
and
`83ff27462308bc87278eea3df49dcf93e24ed85b0db501d68f6ad0ead4257e13`.

At the first sample the retry was still preparing: launch recorded
`cuda_initialized=false`, and GPU1 had no registered compute PID.  It is
nevertheless a CVD1/GPU1 route and the device remains reserved.  Its broad CPU
mask overlaps P0, but host PSI remained zero.  PACD P0 remained the sole GPU0
compute owner.  Root reconstructed the exact 48-file P0 closure twice after
the external archive/retry operation; it remained
`3356fb124ff4d18a2035bfda8e7eacc28a1642ce3f36b290155a5bcdf7faea2f`
without Torch.  No root or monitoring agent changed either live process,
affinity, priority, code or result artifact.

At 06:17 HKT the M1 retry naturally materialized on physical GPU1: PID
`860203` appeared under UUID
`GPU-2220ed5d-25ea-1839-28d7-ad4dfa5f6c86` with about 1,172 MiB process
memory.  PACD PID `783126` remained the sole GPU0 owner, so GPU placement is
exactly disjoint.  The M1 CPU mask remained `0-31`, overlapping P0, while host
CPU, memory and IO PSI remained zero.  No OOM, failure, terminal or C1 event
was present.  Monitoring made no process or root change; GPU1 is now an
observed physical conflict for future P2, not merely an administrative
reservation.

## Epoch 028 verified event under live GPU1 retry

`epoch028.json` published naturally at 2026-09-01 06:28:15 HKT.  Its regular
mode-`0444`, single-link body and canonical sidecar bind SHA-256
`70ad0191efef379d0d540f7ba872868503013c31210f437dabb6f632b4a07824`.
Root independently validated epoch000--028: all 29 pairs have exact
mode/link/basename/SHA integrity, 33,925 optimizer steps per epoch and 983,825
cumulative steps.  The live root has exactly 32 bodies/64 leaves: attempt,
launch, source authority and epochs 000--028.  No failure, terminal,
checkpoint 44--47, SWA or manifest exists.

Mean anchor/combined/short loss is `0.417631`, min `0.023431`, max
`14.923334`.  Wall time is `3383.068 s`, or `10.027881` paired steps/s.  Adam
is finite.  Identity/prediction mismatches, prefix mutations, RNG violations,
parameter-finiteness violations and zero decoder-gradient steps are zero.
Decoder gradient is positive on all 33,925 steps.  Encoder gradient is
positive on 33,910 steps; the 15 accepted zero steps use only
`all_units_dropped_valid_zero`.  The 29-materialized/two-lazy topology,
fixed sampler digests, 1,600 valid bins and four sentinel laws remain exact.

The M1 retry began late inside this epoch and had materialized on GPU1 by the
event.  Relative to epoch027, epoch028 throughput is about `1.92%` lower and
wall time about `1.96%` higher.  The receipt has no sub-epoch resource series,
so the change cannot be attributed to M1; it is recorded only as a coexistence
observation.  All PACD correctness gates remain exact, host PSI was zero and
no GPU owner crossing occurred.

GPU0 had only P0 PID `783126`; GPU1 had only M1 PID `860203`.  The explicit
48-file P0 closure reconstructed twice after publication as
`3356fb124ff4d18a2035bfda8e7eacc28a1642ce3f36b290155a5bcdf7faea2f`
without Torch.  No V4 implementation, capability or root action exists.

### M1 retry aborted without a terminal graph

At 07:16 HKT, M1 retry PIDs `860201/860203` disappeared and GPU1 returned to
no compute owner.  The external owner archived the canonical `t0` root as
`t0_attempt2_aborted_slow_full_detail_recording`.  That directory has only
the earlier attempt/launch body+sidecar pairs; it has no failure or terminal,
and the pipeline log has no completion/failure/C1 line.  It is therefore an
invalid/unexplained aborted attempt, not a result.

PACD remained the sole GPU0 compute process, host PSI remained zero and no
P0 invariant failed.  Root reconstructed the P0 48-file closure twice after
the external archive action; it remained
`3356fb124ff4d18a2035bfda8e7eacc28a1642ce3f36b290155a5bcdf7faea2f`
without Torch.  GPU1 remains administratively reserved despite being
physically idle, and no PACD job will use this gap.

### Short unattributed GPU1 owner and external M1 source edit

At 07:23 HKT, GPU1 briefly exposed Python PID `865155` with about 1,170 MiB,
but it exited before its command/environment/affinity could be bound.  It
created no M1 receipt/root change and is not attributed to any route.  PACD
remained the sole GPU0 owner and host PSI remained zero.

At 07:27 HKT, an external owner rewrote
`tfpd_exploration/src/m1_t0c1_prefix_v1/plan.py` and ran compile/tests.  Root
and monitoring did not modify or stop that work.  The M1 source path is outside
the active P0 closure; the P0 48-file closure reconstructed twice unchanged.
Any future M1 launch must use a fresh M1 source closure rather than the
aborted attempt's historical closure.

### M1 C1 terminal, metric-table failure and natural resource release

At 12:49 HKT, the independent M1 route's C1 training stage terminalized
successfully and exited naturally.  Its immutable 29-body/58-leaf source-only
graph completed 20 epochs and 99,020 optimizer updates; terminal SHA-256 is
`cbef49e8e356103c56a9ffa673a268de29e1c5564fc5957c26474b1cf065da99`.
The subsequent metric-only Phase 3 failed before any cell completed because
the route plan lacked `safe_relative`; failure SHA-256 is
`3c1a92634f6b4aa5baae138feac7f776005d01884224777e00ba30aedb26ddbe`.
No target comparison table was produced, and no retry or repair was made.

The C1 process and pipeline shell then disappeared, physical GPU1 returned to
no compute owner, and the broad M1 `0-31` CPU affinity was released.  P0
remained the sole GPU0 owner with closure and state unchanged; host PSI stayed
zero.  This removes the current coexistence conflict but does not admit V4:
P0 still requires a valid terminal, and future P1/P2 capabilities require
fresh post-P0 resource observations rather than this historical sample.

The M1 training receipts have a disclosed hardware-provenance inconsistency:
launch/live evidence uses CVD1 and physical GPU1, while the stored runtime
profile contains GPU0 UUID/PCI because its `nvidia-smi --id 0` query bypassed
CVD remapping.  That defect belongs to M1 and does not enter or change the
PACD P0 closure.

The resource release was subsequently superseded by a new external M1 Phase
3 attempt.  Its owner archived the earlier failed root, repaired its own route
and launched PID `978827` on physical GPU1 with CVD1 and affinity `0-31`.
The new canonical attempt has closure
`10e3c0872bae37ab9517466527b3f1123e26fa840be1f4933e3e39ba86c32496`
and was still nonterminal at the sample.  Root/Luna did not perform or assist
the repair or retry.  P0 remained the sole GPU0 owner with no cross-root FD or
invariant drift, but the renewed broad M1 affinity again blocks both future
V4 CPU profiles and its GPU1 ownership blocks P2.  Historical release is not
current admission evidence.

## Epoch 029 verified event

`epoch029.json` published naturally at 2026-09-01 07:27 HKT.  Its regular
mode-`0444`, single-link body and canonical sidecar bind SHA-256
`64e094e21d8c5c5b1ac0510e0abe016b47b8a1e7e966d360523b183d49c8b3c2`.
Root independently validated epoch000--029: all 30 pairs have exact
mode/link/basename/SHA integrity, 33,925 optimizer steps per epoch and
1,017,750 cumulative steps.  The live root has exactly 33 bodies/66 leaves:
attempt, launch, source authority and epochs 000--029.  No failure, terminal,
checkpoint 44--47, SWA or manifest exists.

Mean anchor/combined/short loss is `0.414212`, min `0.018549`, max
`15.000161`.  Wall time is `3462.340 s`, or `9.798286` paired steps/s.  Adam
is finite.  Identity/prediction mismatches, prefix mutations, RNG violations,
parameter-finiteness violations and zero decoder-gradient steps are zero.
Decoder gradient is positive on all 33,925 steps.  Encoder gradient is
positive on 33,906 steps; the 19 accepted zero steps use only
`all_units_dropped_valid_zero`.  The 29-materialized/two-lazy topology,
fixed sampler digests, 1,600 valid bins and four sentinel laws remain exact.

Epoch029 throughput is `2.29%` below epoch028.  Its interval included portions
of the M1 retry, its unexplained exit and the short unattributed GPU1 PID, but
the P0 receipt contains no sub-epoch resource series.  No causal attribution
is made.  All correctness gates are exact and host PSI stayed effectively
zero.  The 48-file P0 closure reconstructed twice as
`3356fb124ff4d18a2035bfda8e7eacc28a1642ce3f36b290155a5bcdf7faea2f`
without Torch.

### Third M1 T0 retry with a fresh M1 closure

At 07:27:27 HKT, the M1 owner started a third T0 attempt after the source edit.
Its new attempt SHA is
`1d7886a4043738d5a49e2d9a3d496ef46664e7f11dca34fa4c06a5eae660653a`
and its M1 closure is
`3702f99941ac39664b6c4938886d8665d0d3e056fc618ecb9152135e660a2391`,
so it did not reuse the aborted attempt's closure.  Shell/Python PIDs are
`866226/866228`, CVD1, affinity `0-31`.

At 07:33 HKT, PID `866228` materialized on physical GPU1 with about 1,172 MiB.
PACD PID `783126` remained the sole GPU0 owner; host PSI remained zero and no
GPU owner crossed.  The M1 retry had only attempt/launch receipts and no
terminal/failure/C1/phase3 at the sample.  P0's 48-file closure reconstructed
twice unchanged as
`3356fb124ff4d18a2035bfda8e7eacc28a1642ce3f36b290155a5bcdf7faea2f`
without Torch.  No PACD process, code or root was modified.

## Epoch 027 verified event

`epoch027.json` published naturally at 2026-09-01 05:31:50 HKT.  Its regular
mode-`0444`, single-link body and canonical sidecar bind SHA-256
`619f72cc396f26776d532cc50adf0194ed12628089cb1abf41573ea7a8be57bb`.
Root independently validated epoch000--027: all 28 pairs have exact
mode/link/basename/SHA integrity, 33,925 optimizer steps per epoch and 949,900
cumulative steps.  The live root has exactly 31 bodies/62 leaves: attempt,
launch, source authority and epochs 000--027.  No failure, terminal,
checkpoint 44--47, SWA or manifest exists.

Mean anchor/combined/short loss is `0.420866`, min `0.020473`, max
`14.962512`.  Wall time is `3318.118 s`, or `10.224170` paired steps/s.  Adam
is finite.  Identity/prediction mismatches, prefix mutations, RNG violations,
parameter-finiteness violations and zero decoder-gradient steps are zero.
Decoder gradient is positive on all 33,925 steps.  Encoder gradient is
positive on 33,910 steps; the 15 accepted zero steps use only
`all_units_dropped_valid_zero`.  The 29-materialized/two-lazy topology,
fixed sampler digests, 1,600 valid bins and four sentinel laws remain exact.

GPU0 had only P0 PID `783126`; GPU1 remained without a compute owner and the
M1 route had no retry/C1/phase3.  Host PSI remained zero.  The explicit
48-file closure reconstructed twice after publication as
`3356fb124ff4d18a2035bfda8e7eacc28a1642ce3f36b290155a5bcdf7faea2f`
without Torch.  No V4 implementation, capability or root action exists.
