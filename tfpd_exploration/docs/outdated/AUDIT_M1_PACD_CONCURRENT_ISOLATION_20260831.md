# Audit: M1 / PACD Concurrent Isolation

Date: 2026-08-31

Status: pre-launch isolation gate.  This document is read-only review evidence
outside both execution closures.  It authorizes no source edit, capability,
result-root reservation, data/checkpoint access, CUDA initialization or
launch.

## 1. Live baseline

At 2026-08-31 23:15 HKT:

- PACD P0 PID `783126` was the sole CUDA compute owner;
- PACD was on physical GPU0, with CPU affinity `0-3,16-19`;
- physical GPU1 was `0%` with the normal 23 MiB display footprint and no
  compute PID;
- CPU, memory and IO PSI averages were zero;
- the announced M1 comparison had created code and a work order but had not
  created an observable CUDA context or result root.

The M1 route therefore is not yet a live competitor.  GPU1 remains reserved
for it; temporary idleness is not permission for PACD P2 to use GPU1.

## 2. Static path isolation

The observed M1 route is additive:

```text
tfpd_exploration/docs/WORKORDER_M1_HELDIN_HELDOUT_GAP_V1_20260831.md
tfpd_exploration/src/m1_heldin_heldout_gap_v1/
tfpd_exploration/scripts/run_m1_heldin_heldout_gap_v1.py
tfpd_exploration/results/m1_heldin_heldout_gap_v1
```

PACD P0 uses the distinct root:

```text
tfpd_exploration/results/paired_anchored_calibration_dropout_full_v3/
  p0_fullfull_seed42
```

After the M1 paths appeared, root reconstructed the PACD 48-file execution
closure as
`3356fb124ff4d18a2035bfda8e7eacc28a1642ce3f36b290155a5bcdf7faea2f`.
No new M1 path is in that closure and no PACD closure byte drifted.

The M1 public CLI is inert and rejects `--execute`.  Its planned result root
is also distinct from every PACD producer and score root.  These facts prove
static path isolation, not live device isolation.

## 3. Current launch gap

The current M1 physical backend emits the descriptive launch field:

```text
operator_gpu_constraint = CUDA_VISIBLE_DEVICES=1 (physical GPU 1 only)
```

At this audit boundary it does not yet exact-validate any of the following:

- actual `CUDA_DEVICE_ORDER` and `CUDA_VISIBLE_DEVICES`;
- physical GPU1 UUID `GPU-2220ed5d-25ea-1839-28d7-ad4dfa5f6c86`;
- logical-device-to-physical-device mapping after CVD remapping;
- selected GPU compute PID equal to the scoring process PID;
- disjoint CPU affinity;
- thread-pool limits or host-pressure state.

Therefore the text field alone is not sufficient evidence for a concurrent
launch.  No M1 root-reviewed capability should be issued until the fixed
launch envelope below is satisfied and independently observed.

## 4. Fixed concurrent launch envelope

The M1 process must use:

```text
CUDA_DEVICE_ORDER=PCI_BUS_ID
CUDA_VISIBLE_DEVICES=1
taskset CPU list = 4-15,20-31
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1
PYTHONNOUSERSITE=1
PYTHONDONTWRITEBYTECODE=1
```

With `CUDA_VISIBLE_DEVICES=1`, the application should address the selected
device as logical `cuda:0`; physical identity remains GPU1 and must be checked
through the UUID/PID observation rather than inferred from the logical index.

The M1 CPU set is deliberately disjoint from PACD P0's
`0-3,16-19`.  No process affinity, priority or environment of the already
running PACD process may be changed.

Immediately before M1 capability issuance, the root reviewer must prove:

1. GPU1 has no compute process and GPU0 has only the expected PACD PID;
2. the M1 result root is absent and its parent identity is stable;
3. the PACD result root and PID are live and unchanged;
4. host CPU, memory and IO PSI pass, with sufficient available memory;
5. the M1 closure and frozen producer/anchor graphs are current;
6. no file in the active PACD 48-file closure has changed.

Immediately after CUDA initialization and during the run, read-only evidence
must show:

- M1 is the sole compute PID on physical GPU1;
- PACD remains the sole compute PID on physical GPU0;
- M1 affinity is exactly `4-15,20-31`;
- PACD affinity remains `0-3,16-19`;
- roots and write targets remain disjoint;
- host PSI remains nonpressure and neither route reports CUDA/OOM/nonfinite
  errors.

## 5. Scheduling consequence for PACD V4

The frozen PACD V4 mapping is unchanged: P1 uses GPU0 and P2 uses GPU1.

- P0 continues without intervention.
- V4 implementation remains forbidden until the valid P0 terminal gate.
- If M1 still owns GPU1 after P0 terminal, PACD P2 waits.
- PACD P1 may start on GPU0 only after P0 releases it and only while its fixed
  CPU profile is disjoint from M1.
- There is no GPU fallback, arm migration, preemption or M1 process mutation.

## 6. Release evidence

M1 is released only after its process exits naturally, GPU1 has no compute
owner, its immutable terminal or failure graph is audited, and the host
pressure gate is sampled again.  A historical idle snapshot is not a standing
release.

### 6.1 Observed M1 terminal and honest device limitation

The M1 route published a natural terminal at 2026-08-31 23:35:52 HKT.  Its
exact 12-body/24-leaf immutable graph passed mode, link, basename-sidecar,
SHA and terminal cross-link validation.  Terminal SHA-256 is
`b948976e624a5a53fbc5ac5c2f5f7cf6e379c8408b1a749b80a3b5dbada49d51`.
The result is metric-only, with zero target optimizer/backward/update, and is
documented separately in `RESULT_M1_HELDIN_HELDOUT_GAP_V1_20260831.md`.

The launch receipt records logical `device=cuda:0` and the intended operator
constraint `CUDA_VISIBLE_DEVICES=1`.  It does not persist physical UUID, CUDA
PID or affinity.  The live watcher did not capture the approximately
12-minute scoring process; its first watcher query also used a nonfunctional
`nvidia-smi` field, so absence of a watcher event cannot prove CPU-only
execution or physical-device identity.  The honest classification is
"declared GPU1 through CVD=1, physical UUID/PID not durably proven".

After terminal, physical GPU1 was repeatedly idle with no compute owner.
PACD PID `783126` remained the sole GPU0 compute process, its affinity remained
`0-3,16-19`, and host PSI stayed zero.  The M1 live conflict is therefore
released for future scheduling, although PACD P0 still blocks all V4 work.

PACD epoch021 subsequently terminalized normally for the epoch with SHA
`cb5fd3c9e475f17506d31db784a6a09f21e802d9ec78a2ba90b7b30ded139fa1`.
It spans the M1 interval and has zero identity/prediction, prefix, RNG,
finiteness and decoder-gradient violations.  Its throughput is `1.54%` below
epoch020.  Because the receipt has no sub-epoch timing or device series, this
cannot be attributed to M1; it is evidence of clean PACD health, not proof of
exact historical non-overlap.

### 6.2 New M1 T0/C1 training pipeline on GPU1

The completed metric-only M1 route above is not the final M1 activity in this
workspace.  At 2026-09-01 00:25:44 HKT, another agent started the independent
pipeline `/tmp/m1_t0c1_pipeline.sh` (PID `847532`).  Its smoke stage completed
at 00:36:53 HKT and the T0 stage started immediately afterward as PID
`847764`.  Read-only inspection established:

```text
M1 T0 command: m1_t0c1_prefix_v1.driver.execute_arm(..., device='cuda:0', arm='t0')
CUDA_VISIBLE_DEVICES: 1
physical device: GPU1
physical GPU UUID: GPU-2220ed5d-25ea-1839-28d7-ad4dfa5f6c86
CPU affinity: 0-31
result subroot: tfpd_exploration/results/m1_t0c1_prefix_v1/t0
```

At 00:58 HKT, GPU1 had only PID `847764` as a compute owner, using about
1.2 GiB.  PACD PID `783126` remained the only GPU0 compute owner and retained
affinity `0-3,16-19`.  Thus the physical GPU mapping is disjoint, but the M1
CPU mask covers the PACD CPU set and does not satisfy the exact disjoint mask
proposed in Section 4.  Host CPU, memory and IO PSI were nevertheless zero.
This is a live scheduling risk, not evidence that either result is corrupt.

The T0 subroot contained only immutable attempt/launch pairs at the sample;
there was no T0 terminal/failure and no C1 subroot yet.  The pipeline script
is expected to chain smoke → T0 → C1 → phase3 on GPU1.  GPU1 must therefore
remain reserved through the whole pipeline, not merely until the current T0
PID exits.  PACD P2 may not mint, reserve or launch during any of those stages.
PACD P1 is separately blocked by the still-running P0 terminal gate.

The root reviewer and monitoring agent will remain read-only.  They will not
signal, renice, retask, change affinity, restart, modify code used by the live
job, or write either result root.  Release requires the pipeline shell and all
stage processes to exit naturally, immutable terminal/failure receipts to be
audited, GPU1 to have no compute owner and a fresh host-pressure/root gate to
pass.  Temporary low GPU utilization is not a release signal.

PACD epoch022 published at 00:53:44 HKT with SHA
`417d04deb9de6c6fc99aafd8334ba645099d9cc3a775d1fac6bc78eecd8dd625`.
All scientific and safety invariants remain clean.  Its throughput is `1.48%`
below epoch021, but T0 began only late inside the epoch and no sub-epoch
telemetry exists, so this cannot be causally assigned to the overlapping CPU
mask.

At 01:07 HKT, root also inspected each live process's file-descriptor access
mode through `/proc` without opening either result artifact.  PACD's only
workspace-or-`/tmp` writable descriptors were stdout/stderr to
`/tmp/pacd_p0_full_v3_20260831.log`; M1's were stdout/stderr to
`/tmp/m1_t0c1_pipeline_stdout.log`.  Neither process held an open descriptor
under the other route's result root, and neither held an open descriptor under
its own result root at that instant because published artifacts had already
been closed.  This strengthens the write-target separation observation; it
does not replace immutable-receipt validation at either terminal.

### 6.3 T0 fail-close; no C1 or phase3 execution

The T0 process and pipeline shell exited naturally at 2026-09-01 01:13:33
HKT.  The pipeline log states `STAGE t0 FAILED ... chain stops`.  The canonical
T0 root contains exactly three immutable body/sidecar pairs and no terminal:

| Body | SHA-256 |
|---|---|
| `attempt.json` | `3c72ddea583c80b7c1488e3dbfd2541fcacff2324d96cfff6c362ac6d4214744` |
| `launch.json` | `83ff27462308bc87278eea3df49dcf93e24ed85b0db501d68f6ad0ead4257e13` |
| `failure.json` | `694c93cf63a12f470e65c5b62c8830d49456048753aa0d4abbe5135715fea8b8` |

All six leaves are regular mode `0444`, link count one, and each canonical
basename sidecar matches its body.  The failure has schema
`m1_t0c1_t0_failure_v1`, class `TrainerError`, and exact error
`epoch loss row coverage drift`.  It records `prepared=true`,
`optimizer_steps_completed=0`, `target_metric_only=true` and
`terminal_published=false`.  The failure links the exact attempt body digest.
The launch schema does not carry an attempt digest, so this audit does not
invent or claim such a cross-link.

No C1 or phase3 directory was created.  Therefore this attempt produced no
trained T0 comparator, no C1 result and no phase3 result.  It must not enter a
scientific comparison table as a null effect; it is an engineering failure
before the first optimizer update.

After exit, physical GPU1 returned to 0% utilization/23 MiB with no compute
PID.  GPU0 retained only PACD PID `783126`; P0 remained healthy and host PSI
remained zero.  This proves the observed T0 process has released GPU1.  It does
not give PACD permission to take the device: the M1 owner may still perform a
reviewed diagnosis/retry, and PACD P0 remains an independent hard gate.  Root
will not modify or retry the M1 route and will continue to treat GPU1 as
administratively reserved until the owner route's next state is explicit.

### 6.4 Owner-issued T0 retry at 06:11 HKT

At 2026-09-01 06:11:08 HKT, the external M1 owner archived the earlier failed
directory from canonical `t0` to `t0_failed_loss_coverage_drift`, recreated a
fresh canonical `t0`, and started `/tmp/m1_t0c1_pipeline_from_t0.sh`.  Root and
the monitoring agent did not perform, request or modify this operation.

The archived directory retains the earlier exact six-leaf failure graph and
body digests from Section 6.3.  The path/inode identity is no longer the
original canonical identity, so future references must name the archived path
explicitly rather than pretending the historical directory was never moved.
The new canonical `t0` initially contained only four immutable mode-`0444`
leaves:

| Body | SHA-256 |
|---|---|
| `attempt.json` | `4e9d9a7ea26679d69a5ae1b0529c0025f17c077e0322df99dd1a1c5c5e7aa883` |
| `launch.json` | `83ff27462308bc87278eea3df49dcf93e24ed85b0db501d68f6ad0ead4257e13` |

The attempt records status `ATTEMPT_RESERVED`, cell
`CROSS_SESSION_WORST_GROUP_SPINT_M1_V1`, stage `t0`, closure
`2d022119ed87eda73e4400b742dddb58116c92ba018a08ab1a5dfbc8a8818925`,
`data_or_model_accessed=false`, and zero target backward/update.  Launch
records `LAUNCHED`, `cuda_initialized=false`, logical `cuda:0`, and
`CUDA_VISIBLE_DEVICES=1`.

The new pipeline shell/Python PIDs are `860201/860203`.  The Python process
has `CUDA_VISIBLE_DEVICES=1`, cwd `/home/xinyuan/Work_host/SPINT`, and broad
CPU affinity `0-31`; it therefore again overlaps PACD P0's `0-3,16-19` CPU
set.  At the 06:13 HKT sample it had not yet appeared as a physical GPU1
compute application: GPU1 was 0%/26 MiB.  This is a preparation-stage sample,
not proof that the retry will remain CPU-only.  GPU1 remains reserved for the
retry and must be resampled after CUDA context materialization.

PACD remained the sole GPU0 compute process.  Its 48-file closure reconstructed
twice unchanged after the external archive/retry action, and host CPU, memory
and IO PSI remained zero.  No PACD path, process, affinity or root was changed.
Monitoring remains read-only; root will not repair, stop, retask or restart the
M1 retry.

At 06:17:18--06:17:33 HKT, the retry naturally materialized its CUDA context.
`nvidia-smi` bound PID `860203` to physical GPU1 UUID
`GPU-2220ed5d-25ea-1839-28d7-ad4dfa5f6c86`, with about 1,172 MiB process
memory and 1,201 MiB total device memory at the sample.  GPU1 utilization was
about 16%.  This confirms the CVD1 mapping rather than merely trusting the
launch declaration.  PACD PID `783126` remained the sole GPU0 compute owner;
there was no GPU owner crossing.  M1 affinity remained `0-31`, host CPU,
memory and IO PSI remained zero, and no failure/terminal/C1 transition was
present.  The retry now creates an actual physical P2 conflict until it exits
naturally and its final graph is audited.

### 6.5 Retry process disappeared without a terminal/failure receipt

At 2026-09-01 07:16 HKT, retry PIDs `860201/860203` disappeared and GPU1 lost
its compute application.  The external owner archived the canonical `t0`
directory as `t0_attempt2_aborted_slow_full_detail_recording`.  Root and Luna
did not stop, move or edit the process/root.

The archived directory contains exactly the four immutable attempt/launch
leaves described in Section 6.4, with the same body SHA values and valid
sidecars.  It contains neither `failure.json` nor `terminal.json`.  The
pipeline log still ends at `STAGE t0 START (resume-from-t0 chain)` and records
no completion, failure or C1 transition.  Therefore this is an unexplained
process disappearance/aborted attempt, not a valid fail-close, success,
scientific result or release receipt.

GPU1 returned to its display footprint with no compute PID; PACD remained the
sole GPU0 owner and host PSI stayed zero.  The physical owner conflict is
absent at this instant, but the device remains administratively reserved: the
M1 owner has already demonstrated it may archive and issue another retry, and
there is no immutable terminal/failure governing this attempt.  PACD will not
use the gap or mutate the M1 evidence.  The P0 48-file closure reconstructed
twice unchanged after this external archive action.

### 6.6 Unattributed GPU1 PID and post-abort source mutation

At 07:23:30 HKT, physical GPU1 briefly listed Python PID `865155` using about
1,170 MiB.  The process disappeared before root or Luna could read its
`/proc` command, environment or affinity.  It produced no new M1 directory or
receipt, and the M1 pipeline log did not change.  It is therefore recorded as
an unattributed short GPU1 owner; this audit does not guess that it was an M1
stage.  PACD remained the sole GPU0 owner and host PSI remained zero.

At 07:27:16 HKT, a separate external shell PID `866156` visibly rewrote
`tfpd_exploration/src/m1_t0c1_prefix_v1/plan.py`, then invoked compile/tests.
The current file SHA after that owner write was
`df25d71d0e4cae4c427622180027aba03b6757fb37bd59f1ad2719ad8e299d82`.
Root/Luna did not initiate, read through, stop or modify the writer/test.

This source mutation occurred after the receipt-less aborted attempt.  It is
outside the PACD execution closure, whose 48-file hash reconstructed twice
unchanged.  It does mean any future M1 attempt must bind a newly reconstructed
M1 closure and cannot reuse the aborted attempt's closure/capability as current
source authority.  There was no live canonical M1 stage root or GPU1 owner at
the audit sample.

### 6.7 Third T0 retry with refreshed closure

At 2026-09-01 07:27:27 HKT, the external owner recreated canonical `t0` and
started another resume-from-T0 pipeline.  This attempt correctly carries a new
closure after the Section 6.6 source mutation:

```text
attempt SHA  1d7886a4043738d5a49e2d9a3d496ef46664e7f11dca34fa4c06a5eae660653a
launch SHA   83ff27462308bc87278eea3df49dcf93e24ed85b0db501d68f6ad0ead4257e13
closure      3702f99941ac39664b6c4938886d8665d0d3e056fc618ecb9152135e660a2391
shell/Python 866226/866228
```

Attempt/launch body+sidecar pairs are immutable mode `0444`; the attempt has
`data_or_model_accessed=false`, zero target backward/update and status
`ATTEMPT_RESERVED`.  The Python process uses CVD1/logical `cuda:0` and
affinity `0-31`.

At 07:33 HKT, PID `866228` naturally materialized on physical GPU1 UUID
`GPU-2220ed5d-25ea-1839-28d7-ad4dfa5f6c86`, using about 1,172 MiB.  PACD
remained the sole GPU0 compute owner, so GPU placement was disjoint.  The CPU
sets overlap, but host PSI remained zero.  No T0 terminal/failure, C1 or
phase3 receipt existed.  Monitoring and root performed no source, process,
affinity or result-root mutation.  P2 has an actual live GPU1 conflict for the
duration of this retry.

### 6.8 Live isolation confirmation at 07:52--07:54 HKT

The third T0 retry remained live and physically isolated at the latest sample.
`nvidia-smi` reported exactly one training compute owner on each device:

```text
physical GPU0 / GPU-ac7388a5-2e98-300a-fdb3-0b67bfd494d9
  PACD PID 783126, CVD0, about 708 MiB process memory

physical GPU1 / GPU-2220ed5d-25ea-1839-28d7-ad4dfa5f6c86
  M1 PID 866228, CVD1, about 1,172 MiB process memory
```

The M1 process's logical `cuda:0` is therefore the CVD1 view of physical
GPU1, not physical GPU0.  There was no second compute PID on either device and
no GPU owner crossing.  Both parent/worker process pairs remained live.

PACD affinity remained `0-3,16-19`; M1 affinity remained `0-31`.  Although
those CPU sets overlap, CPU, memory and IO PSI all reported
`avg10=avg60=avg300=0.00`, so the sample provides no evidence of live host
pressure.  The audit therefore records the overlap but does not retask,
renice, signal or restart either job.

PACD still ended at the complete immutable `epoch029` receipt and had no
failure/terminal.  M1 still had only the canonical T0 attempt/launch pairs and
had not published T0 terminal/failure, C1 or phase3.  Score V2, PACD P1 and
PACD P2 remain frozen/no-launch.  Luna continues event-driven, read-only
monitoring, with a 30-minute fallback report interval.

### 6.9 Live descriptor isolation for the third retry

A read-only `/proc/<pid>/fd` audit after the Section 6.8 sample found no
cross-root descriptor.  PACD PID `783126` had only its own stdout/stderr log
among workspace or `/tmp` paths:

```text
/tmp/pacd_p0_full_v3_20260831.log
```

M1 PID `866228` had only its own stdout/stderr log plus its own parent and T0
result directories:

```text
/tmp/m1_t0c1_pipeline_stdout.log
tfpd_exploration/results/m1_t0c1_prefix_v1
tfpd_exploration/results/m1_t0c1_prefix_v1/t0
```

PACD had no descriptor into the M1 result family, and M1 had no descriptor
into the PACD P0 result root.  This strengthens the live isolation evidence
but does not replace immutable receipt-graph validation or authorize any new
job.

### 6.10 Short same-GPU co-owner during the third T0 retry

At 2026-09-01 09:12:43 HKT, read-only monitoring observed a second compute
PID `877688` on physical GPU1 while M1 PID `866228` was still present.  This
was a real transient same-device double-owner event.  It was not a PACD
process: PACD remained PID `783126` on physical GPU0, and no PACD P1/P2/V4
job existed.

PID `877688` exited before its `/proc` command, environment, affinity or cwd
could be bound.  At the 09:13--09:14 HKT rechecks it was absent, GPU1 again
contained only M1 PID `866228`, and GPU0 contained only PACD PID `783126`.
The audit therefore does not guess the transient process's route or owner.

The M1 canonical T0 topology and pipeline log did not change: attempt/launch
remain the only current pairs, with no terminal, failure, C1 or phase3.
P0 remained live through epoch030 with its historical closure unchanged.
CPU, memory and IO PSI averages remained zero, with no OOM or cross-GPU
owner.  No signal, stop, retry, affinity change or file/root mutation was
used to restore the single-owner state.

This event must be disclosed with any later M1 T0 result because it is direct
evidence of short GPU1 co-tenancy, even though no correctness failure or
measurable PSI pressure was observed.  It does not admit P2: P0 still lacks a
terminal, M1 still owns GPU1, and every future V4 capability requires a new
single-owner observation.

### 6.11 Current two-GPU isolation and the resulting two-arm hold

At 2026-09-01 09:44 HKT, root repeated the live device, environment,
affinity, file-descriptor and host-pressure checks.  The physical mapping was
again one compute owner per device:

```text
GPU0 / GPU-ac7388a5-2e98-300a-fdb3-0b67bfd494d9
  PACD P0 PID 783126, CUDA_VISIBLE_DEVICES=0,
  affinity 0-3,16-19, about 708 MiB process memory

GPU1 / GPU-2220ed5d-25ea-1839-28d7-ad4dfa5f6c86
  M1 T0 PID 866228, CUDA_VISIBLE_DEVICES=1,
  affinity 0-31, about 1,172 MiB process memory
```

No second GPU1 compute owner recurred in this sample.  The earlier PID
`877688` event remains historical evidence and is not erased by the later
single-owner observation.  PACD's workspace-or-`/tmp` descriptor set exposed
only `/tmp/pacd_p0_full_v3_20260831.log`; M1 exposed only its own result parent
and `t0` directory plus `/tmp/m1_t0c1_pipeline_stdout.log`.  No cross-root
descriptor was observed.  CPU, memory and IO PSI `avg10`, `avg60` and
`avg300` were all `0.00`.

PACD had a complete immutable prefix through `epoch031`, with cumulative
optimizer steps `1,085,600` and epoch body SHA-256
`70e183a511d71806435bb6c52f5a62f12c9da3c6c264ad6679c57c8ada3a4f8d`.
There was no PACD failure or terminal.  M1 still had only the canonical T0
attempt/launch pairs, with no T0 terminal/failure, C1 or phase3 receipt.

The scheduler consequence is stronger than a GPU-only reservation.  M1's
`0-31` affinity overlaps both frozen V4 CPU profiles: P1
`0-3,16-19` and P2 `4-15,20-31`.  Therefore, if P0 terminalizes while this M1
pipeline remains live, **neither P1 nor P2 may be admitted**.  P1 must not be
started merely because GPU0 becomes idle, and P2 must not be started merely
because GPU1 utilization momentarily falls.  Release requires the M1 pipeline
and every stage process to exit naturally, its final immutable graph to be
audited, both fixed CPU profiles to be free of the tracked route, the assigned
GPU to have no compute owner, and a fresh pressure/root/device gate to pass.
No affinity edit, signal, restart, preemption or fallback device is authorized
to manufacture that release.

### 6.12 T0 accepted terminal and natural transition to C1

At 2026-09-01 10:18 HKT, the third T0 attempt terminalized successfully and
the unchanged pipeline shell naturally started C1.  Root and Luna did not
signal, restart, move, edit or otherwise cause the transition.  The completed
T0 root contains exactly 29 bodies and their 29 canonical sidecars: 58 leaves
total.  Every leaf is a regular, non-symlink, single-link mode-`0444` file;
every body digest and canonical basename sidecar verifies; no extra leaf is
present.

The accepted T0 evidence is:

```text
terminal status             COMPLETE_MATCHED_ARM_TRAINING
terminal SHA-256            4aea40a317047beec059505231bb9996190c4eb2235392d08a3c948cf7ab9aa9
training SHA-256            c77003652c6182acf33c8879e9bcca27c582f7e4cacc2f150d9c5c5962262601
checkpoint manifest SHA-256 fe9ed350cf92225426a3fe8df5cf3ecf380bea58ce308f26479a7b3c47eec260
best/last checkpoint SHA    91bd46c0261df141b4fd601eeaa0f105fb637e75aed572d734b94ae0e8b9486e
epochs                      20
optimizer steps             99,020
best/last epoch             19 / 19
elapsed seconds             9,919.4303
SWA                         disabled and artifact forbidden
target optimizer/backward/update 0
```

All twenty epoch indices are exact and ordered; cumulative optimizer steps
increase strictly to 99,020; every objective is finite; epoch 19 is the strict
minimum source-train-loss epoch.  The terminal links the exact accepted
attempt and launch body digests.  The manifest equals the terminal's embedded
manifest; best and last checkpoint bodies are byte-identical, strict-reload
true, and their state digest equals the epoch-19 model state.  This is a valid
T0 producer, not a target R2 result: the route is source-only and contains no
target optimizer/backward/update.

The pipeline retained shell PID `866226` and started C1 Python PID `886913`.
C1 published only immutable attempt/launch pairs at the first sample:

```text
attempt SHA-256 82b80ded940969339be329e6b96918ec687bfd6356fe5419eaa6546cd711910f
launch SHA-256  576673fa01cf0d34d1187dfd17dda5f338b63e3b4a808a5b79235f62dc052863
closure         3702f99941ac39664b6c4938886d8665d0d3e056fc618ecb9152135e660a2391
```

At 10:22 HKT, C1 remained alive with CVD1 and affinity `0-31` but had not yet
materialized a GPU1 compute application.  This pre-CUDA gap is not resource
release: C1 owns the route and is expected to bind GPU1.  GPU1 therefore
remains reserved, and the broad C1 affinity continues to block both future V4
CPU profiles.  PACD remained the sole GPU0 owner and published epoch032
without an invariant failure during this natural M1 transition.

At 10:25 HKT, C1 naturally materialized its CUDA context.  `nvidia-smi`
reported PID `886913` as the sole compute owner on physical GPU1 UUID
`GPU-2220ed5d-25ea-1839-28d7-ad4dfa5f6c86`, using about 1,150 MiB process
memory.  PACD PID `783126` remained the sole GPU0 owner.  C1 retained CVD1 and
affinity `0-31`; host CPU, memory and IO PSI averages remained zero.  Thus the
transition restored the expected one-owner-per-GPU mapping without a second
owner or owner crossing, while continuing to block P2 physically and both V4
arms through CPU overlap.

### 6.13 C1 accepted training terminal, Phase 3 failure and route release

At 2026-09-01 12:49 HKT, C1 PID `886913` exited naturally after publishing a
complete source-only training graph.  The pipeline then entered its
metric-only `phase3_table` stage, which failed closed immediately and caused
the pipeline shell to exit naturally.  Root and Luna did not signal, restart,
repair, rebind or otherwise intervene in either transition.

The C1 root has exactly 29 bodies and 29 canonical sidecars, 58 leaves total:
attempt, launch, source authority, twenty ordered epoch receipts, stream head,
training summary, best and last checkpoints, checkpoint manifest and
terminal.  Every leaf is a regular, non-symlink, single-link mode-`0444`
file; every body digest and canonical basename sidecar verifies; no extra
leaf or failure exists.

```text
C1 terminal status             COMPLETE_MATCHED_ARM_TRAINING
C1 terminal SHA-256            cbef49e8e356103c56a9ffa673a268de29e1c5564fc5957c26474b1cf065da99
C1 attempt SHA-256             82b80ded940969339be329e6b96918ec687bfd6356fe5419eaa6546cd711910f
C1 launch SHA-256              576673fa01cf0d34d1187dfd17dda5f338b63e3b4a808a5b79235f62dc052863
C1 source-authority SHA-256    03c10510f351bb63f7b9e85fc5cb581b4df59d5cbc2869f19acb13bbb57cb9dd
C1 training SHA-256            9ac53c034c89d23ac2e583053f4b865d7645dffde981a6dcd4b13aa54ebb9f2c
C1 checkpoint-manifest SHA-256 03bec4e7b00b9e3743d3a85fb331242f37c94f86fea12518e98377b9a767433f
C1 best/last checkpoint SHA    f302cf9838207f162b162138536782c79d872a415cac134653a717f2d9ab55e5
C1 state SHA-256               ac701caf57617a59e38543b9c64ea35a0347c5cae2798ff2e9571f8efdf4dc5f
epochs                         20
optimizer steps                99,020 (4,951 per epoch)
best/last epoch                19 / 19
source loss epoch 0 -> 19      0.09850221 -> 0.04309420
elapsed seconds                8,685.5170
SWA                            disabled and artifact forbidden
target optimizer/backward/update 0
closure                        3702f99941ac39664b6c4938886d8665d0d3e056fc618ecb9152135e660a2391
```

All epoch indices and cumulative step counts are exact, every objective is
finite, epoch 19 is the strict minimum source-loss epoch, and the terminal
links the exact attempt, launch and embedded checkpoint manifest.  Best and
last checkpoint bodies are byte-identical; their manifest state digests equal
the final and best training state.  No checkpoint tensor was deserialized by
this audit.  This proves a valid C1 training producer, not a target R2 result.

The metric-only Phase 3 root has exactly three immutable body/sidecar pairs:

```text
attempt SHA-256  a01465207c75604d682be9416312ee5c8d5cc98995870ecec45caef28111c2d9
launch SHA-256   c620a6aff24760296db6ad2c03c19421b5b474811fd07c8de1f4362a96c8cfb6
failure SHA-256  3c1a92634f6b4aa5baae138feac7f776005d01884224777e00ba30aedb26ddbe
```

It contains no terminal and reports `cells_completed=[]`,
`target_metric_only=true` and error
`AttributeError("module 'tfpd_exploration.src.m1_t0c1_prefix_v1.plan' has no
attribute 'safe_relative'")`.  Therefore no T0/C1 target comparison number
was produced.  The Phase 3 failure does not invalidate the completed C1
source-training graph, but the route must not be described as a completed
evaluation.

The independent audit found a separate hardware-provenance defect affecting
both accepted T0 and C1 training receipts.  Launch correctly says
`CUDA_VISIBLE_DEVICES=1`, and live `nvidia-smi` monitoring directly observed
their Python processes on physical GPU1 UUID
`GPU-2220ed5d-25ea-1839-28d7-ad4dfa5f6c86`.  However each
`training.json.runtime_environment.profile` records GPU0 UUID
`GPU-ac7388a5-2e98-300a-fdb3-0b67bfd494d9` and PCI bus
`00000000:01:00.0`.  Source review explains the mismatch: the trainer queries
`nvidia-smi --id 0`, which addresses physical GPU0 and is not remapped by
CVD, while Torch logical `cuda:0` is remapped to physical GPU1.  Both cards
are the same RTX 3090 model and no numerical failure is observed, but the
receipt's UUID/PCI hardware attestation is internally inconsistent and must
not be claimed as clean provenance.  This issue is disclosed, not repaired or
used to reinterpret the training result.

After Phase 3 failed, the C1 process and pipeline shell were absent, physical
GPU1 returned to zero utilization/no compute owner, and no M1 process retained
the broad `0-31` affinity.  PACD P0 remained the only GPU0 process with its
fixed `0-3,16-19` affinity, and host CPU/memory/IO PSI remained zero.  Thus
the tracked M1 resource conflict has naturally released.  Future V4
admission still requires a fresh post-P0 release/profile/pressure check; this
historical release is not itself a capability.

### 6.14 External Phase 3 retry and renewed resource occupancy

The release above lasted only until an external M1 owner archived the failed
canonical directory as `phase3_table_failed_safe_relative_attr`, repaired its
own route, and started a new canonical `phase3_table` attempt.  Root and Luna
did not perform, request or assist that rename, edit or retry.  The historical
failure graph remains immutable under its archival name with the same three
body digests; the new canonical attempt has a distinct closure and must be
audited as a new execution rather than treated as continuation of the failed
root.

At 13:46 HKT the new route published attempt SHA-256
`964c709fd94f7cb152854fee74c8259b39d560afe7c5504b30c6ff8d2cf7fe2b`,
launch SHA-256
`c620a6aff24760296db6ad2c03c19421b5b474811fd07c8de1f4362a96c8cfb6`,
and closure
`10e3c0872bae37ab9517466527b3f1123e26fa840be1f4933e3e39ba86c32496`.
It subsequently began publishing metric-cell bodies; no terminal or failure
had appeared at this sample, so no final table claim is made here.

Live `/proc` evidence binds Python PID `978827` and shell PID `978825` to
`driver.execute_phase3(...)`, cwd `/home/xinyuan/Work_host/SPINT`, CVD1 and
affinity `0-31`.  Physical GPU1 has exactly that one compute owner; PACD P0
remains the sole GPU0 owner.  There is no cross-root file descriptor or host
PSI pressure, but the renewed broad affinity overlaps both future V4 CPU
profiles and GPU1 is occupied.  Consequently the M1 release gate is again
unsatisfied.  PACD monitoring remains read-only; it will not stop, reprioritize
or repair this active M1 stage.

### 6.15 Repaired Phase 3 terminal and second natural release

The repaired Phase 3 completed naturally at 13:57 HKT.  Its root contains
exactly 20 bodies and 20 canonical sidecars: attempt, launch, sixteen
arm/deployment/session score cells, table and terminal.  All 40 leaves are
regular, single-link mode-`0444`; every sidecar and body SHA verifies; no
failure or extra leaf exists.

```text
attempt SHA-256   964c709fd94f7cb152854fee74c8259b39d560afe7c5504b30c6ff8d2cf7fe2b
launch SHA-256    c620a6aff24760296db6ad2c03c19421b5b474811fd07c8de1f4362a96c8cfb6
table SHA-256     078aaa01e2558a9f5e6b3c53412feb4a392afc25bec8468f169f57a47259de04
terminal SHA-256  821818b1ef9068aed87b437af3a8363cf2226e7778776dd9c6c10746bf22f096
status            COMPLETE_PHASE3_TABLE
formal benchmark  false
metric            variance-weighted R2
```

The immutable terminal links the exact attempt, launch and table body
digests.  The held-out surface is one fold/session (`20120924`), so these are
diagnostic matched numbers, not a formal benchmark claim:

```text
T0 static M10 held-out        0.57074392
C1 static M10 held-out        0.58487618   delta +0.01413226
T0 CDM activity FIFO held-out 0.58427864   vs T0 static +0.01353472
C1 CDM activity FIFO held-out 0.59473455   vs C1 static +0.00985837
C1 CDM vs T0 CDM                             +0.01045591
C1 CDM vs T0 static total                    +0.02399063
```

On the three held-in training sessions, C1's incremental effect is small and
positive (`+0.00362009` static, `+0.00292353` with CDM), while CDM FIFO itself
reduces held-in R2 by about `-0.068`.  This surface asymmetry and the single
held-out fold must accompany any scientific interpretation.

PIDs `978825/978827` then exited naturally.  GPU1 again had no compute owner,
no M1 process retained affinity `0-31`, PACD remained the sole GPU0 owner,
and host CPU/memory/IO PSI was zero.  The tracked resource conflict is
historically released for a second time, but a future V4 capability still
requires a fresh post-P0 observation.

### 6.16 Post-M1 PACD epoch037 isolation sample

At 15:08 HKT, after the repaired M1 Phase 3 terminal and process release,
PACD P0 naturally published `epoch037.json`, SHA-256
`f6e26afe88151b7d76981a50fbeca63c466329b53d3f015b50d1352df257b982`.
The complete epoch000--037 P0 prefix contains exactly 41 bodies/82 leaves;
all are regular, single-link mode-`0444` files with canonical matching
sidecars.  P0 remained the only GPU0 compute owner with CVD0 and affinity
`0-3,16-19`; GPU1 had no compute owner and no M1 process retained the prior
wide CPU mask.  No cross-root descriptor, persistent PSI pressure, OOM,
failure, terminal, checkpoint, SWA or manifest appeared.

Epoch037 reached `10.1358876` paired steps/s, above the epoch00--29 baseline
of `10.079687`; thus the earlier transient PACD slowdown did not persist after
M1 release.  This is clean coexistence/release evidence only.  It does not
replace the mandatory fresh resource and root-freshness check immediately
before any future V4 capability is issued.

### 6.17 PACD epoch038 after the M1 release

At 16:04 HKT PACD P0 published `epoch038.json`, SHA-256
`26f7e0ddd89cc373046e58f56ba2209b87f12c7665ecdd876a4f2e26e3a5cce6`.
The exact P0 prefix is now 42 bodies/84 leaves through epoch038, with no
failure, terminal, checkpoint, SWA, manifest or extra leaf.  PACD remains
the sole GPU0 owner with its fixed CPU affinity; GPU1 has no compute owner,
no M1 wide-affinity process has returned, and no cross-root descriptor or
persistent host pressure exists.  Throughput is `10.1669889` steps/s, above
the long-run baseline.  This is another clean post-release sample, not a
substitute for fresh V4 admission evidence.

### 6.18 New M1 50-epoch smoke occupancy

At 16:15 HKT an external owner launched a new M1 smoke route via
`tfpd_exploration/scripts/launch_m1_t0c1_prefix_v1_50ep.py --stage smoke`
on physical GPU1.  The live Python PID is `1020808`, CVD is `1`, cwd is the
repository root, and the route has open descriptors only to
`tfpd_exploration/results/m1_t0c1_prefix_v1_50ep` and its `smoke` child.
The smoke root contained immutable attempt and launch pairs at the sample.
This route is distinct from the completed 20-epoch T0/C1/Phase-3 lineage.

Physical GPU ownership remains separated: PACD P0 PID `783126` is the only
GPU0 compute process, while PID `1020808` is the only GPU1 compute process.
There is, however, CPU-scheduler overlap because M1 uses affinity `0-31` and
P0 uses `0-3,16-19`.  Both declare OMP/MKL/OpenBLAS/NumExpr thread count one.
The first joint observation showed zero CPU/memory/IO PSI, no OOM, about
50.7 GiB available memory, no GPU owner crossover and no P0 descriptor into
the M1 root.  No process or affinity was changed.  Subsequent P0 throughput
must be observed before declaring the overlap immaterial, and the future V4
resource-release gate is currently unsatisfied by the live GPU1 owner.

### 6.19 M1 50-epoch smoke terminal and release

The M1 smoke completed naturally at 16:21 HKT and PID `1020808` exited.  Root
independently validated the exact root topology: six bodies and six canonical
sidecars for attempt, launch, T0 smoke, C1 smoke, equality and terminal; all
12 leaves are regular, single-link mode-`0444`, every sidecar/body SHA
matches, and no failure or extra leaf exists.

```text
attempt SHA-256   14c867ea2a5066c99e1f16f50751952f5a90a7ed5b1a737c83e88d8b0b00e22b
launch SHA-256    e336ef659a2827e8ca2a84e89fb4c21a37036801b69d23f601356365fb2f4044
T0 smoke SHA-256  29c8b9b442bef165ab5644adacaed4c1f2e27991a9f9ee4ce27d3aa95c172936
C1 smoke SHA-256  e911109435d83acb4d097a2ef9d080392a87a20b4ba9a77e02e60f994abb194b
equality SHA-256  a40542922bc323a5de435b6773bfbfa4f6fd604591b6a02af65fb201476334d8
terminal SHA-256  ddb7cb9b40c204f62bc94a472f6c85978fa4b5b915c2bf4973b411e8f428fb0c
status            COMPLETE_MATCHED_SMOKE_EQUALITY
```

The terminal states `batch_order_identical=true`,
`dropout_p_stream_identical=true` and `lr_sequence_identical=true`.  GPU1
returned to no compute owner; P0 remained the sole GPU0 owner, no owner
crossover or cross-root descriptor occurred, and host PSI remained zero.
The smoke's CPU overlap lasted only part of P0 epoch039.  Its resource impact
must be judged from that completed epoch, while future V4 still requires a
fresh capability-time observation rather than relying on this release.

### 6.20 M1 50-epoch T0 stage begins

At 16:24 HKT the external M1 pipeline advanced from the accepted smoke to
`launch_m1_t0c1_prefix_v1_50ep.py --stage t0`.  Python PID `1025681` and
parent shell `1025680` run with CVD1 on physical GPU1; the live descriptors
point to `tfpd_exploration/results/m1_t0c1_prefix_v1_50ep/t0`.  Its regular
mode-`0444` attempt and launch pairs verify as:

```text
attempt SHA-256  680a699a1c3b2bc670620608079a71b37b3d61978eeb0d5bdd49f34113d5974c
launch SHA-256   699c36a36bba141f7f92a2db593d1aaeb8d9534d3e1bb084dc662dada2ada93e
closure SHA-256  76ed5cef64a035c940acb34d72fb6b1135287969626ecb68d60cad9e73dfbe85
```

GPU devices remain separated: PACD PID `783126` alone owns GPU0 and M1 PID
`1025681` alone owns GPU1.  CPU scheduling overlaps because T0 uses affinity
`0-31`; P0 retains `0-3,16-19`.  Both routes declare one thread for the major
math runtimes.  At admission, CPU/memory/IO PSI was zero, available memory
was about 51.1 GiB, and there was no OOM, owner crossover or P0 descriptor
into the M1 root.  No affinity, process or priority was changed.  The live
M1 owner makes the future V4 resource-release gate false until natural
release and a fresh post-P0 audit.

### 6.21 External T0 abort and closure retirement

At 16:33 HKT the external M1 owner sent SIGTERM to T0 PID `1025681`, moved
both the accepted smoke and incomplete T0 beneath
`m1_t0c1_prefix_v1_50ep/retired_closure_76ed5cef`, and published immutable
`OPERATOR_ABORT_NOTE.json`, SHA-256
`220657ae9417b60194a1e1192543c271bf41b31d7659c6144d1c1f9a89f7be0e`.
No artifact was deleted.

The note records an operator policy change from serial arms to concurrent
arms after observing low GPU utilization.  That policy edits a launcher file
inside the M1 `OWNED_PATHS`, so it changes the route closure.  Keeping the
already-running T0 would have split T0/C1 closures and invalidated the matched
pair.  The retired smoke remains a valid historical equality check but is
superseded by stale closure.  The retired T0 contains only attempt/launch
pairs; it has neither terminal nor failure because the signal bypassed the
exception handler.  It is incomplete and must never be reported as a result.

The external operator, not PACD root or watcher, performed the SIGTERM, edit
and root move.  P0 remained live on GPU0 with unchanged closure and no
cross-root descriptor.  GPU1 returned to no compute owner; PSI remained zero.
Any successor M1 attempt is a fresh lineage requiring separate observation.
The V4 resource gate remains pending rather than permanently satisfied.

### 6.22 Fresh concurrent M1 closure and PACD epoch039

The external M1 owner rebuilt the route with closure
`ae1f47fbb9f07a886a3c96676144793b0abe646b83336a98e030b269be84f1e7`
and pair-spec SHA-256
`f75734ee7e961eb0e2dc4ab7a6aa391298f48aae3dbfdc2e9cf18d332d6588ca`.
The fresh smoke terminalized successfully.  Root independently verified its
exact six-body/12-leaf graph, all regular single-link mode-`0444`, no failure
or extra leaf.  Terminal SHA-256 is
`7d09b3d1571e21bddb0fcf54deec5f2e6c5ebc439b194fc583bc7d57f7165ab8`,
status `COMPLETE_MATCHED_SMOKE_EQUALITY`; batch order, dropout-probability
stream and LR sequence are identical between the two arms.

The pipeline then started T0 PID `1052875` and C1 PID `1055233` concurrently,
both CVD1 and affinity `0-31`.  T0 became GPU1's compute owner; C1 remained in
CPU/source preparation at the sample.  PACD P0 remained GPU0's sole owner,
with no owner crossover or cross-root descriptor.  CPU sets overlap, but
CPU/memory/IO PSI remained zero and no OOM occurred.

PACD epoch039 published during this transition with SHA-256
`31f4863ce9d7b49bb9785b2cdb0624baa91b30b091a8d57bb016e6985050f5fa`
and throughput `9.8530821` steps/s, `2.25%` below its long-run baseline.  The
timing is compatible with scheduler interference but does not prove it; no
sub-epoch attribution exists and all P0 correctness gates passed.  The new
M1 owners make V4 resource admission false until both naturally release and
fresh post-P0 device/scheduler checks pass.

### 6.23 Sustained M1 concurrency and PACD epoch040

M1 T0 PID `1052875` and C1 PID `1055233` remained simultaneous physical-GPU1
owners for at least 50 minutes, each CVD1 and affinity `0-31`.  Their combined
GPU1 allocation was about 2.35 GiB and observed utilization reached 59%.
PACD P0 remained the sole GPU0 owner with affinity `0-3,16-19`; there was no
GPU crossover, third owner or cross-root descriptor.  Host PSI remained zero
and no OOM occurred.

During that sustained overlap, PACD naturally published epoch040, SHA-256
`ec060eab89239c084d6bf88c3a1512881fb96e7e2935da48ea6862c4056e3683`,
at `9.4191083` steps/s.  This is `6.55%` below the epoch00--29 baseline and
4.40% below epoch039.  The association is operationally important but not a
causal estimate: M1 and P0 do not expose a synchronized sub-epoch
counterfactual.  Every P0 science, optimizer, gradient, RNG, sampler,
finiteness and receipt gate remained exact.

No process, affinity or priority will be changed.  The evidence instead
strengthens the design requirement that any future PACD packed cohort assign
disjoint CPU masks to every process and measure combined throughput against
an isolated reference before full launch.

### 6.24 Second consecutive PACD slowdown epoch

With M1 T0/C1 still simultaneously active on GPU1 and both retaining affinity
`0-31`, PACD naturally published epoch041, SHA-256
`e470c040e4ee7b490ff0348f6dd9d71da1241408692f540a6bb4d35c46c6389a`.
Its throughput was `8.8557272` steps/s, `12.14%` below the long-run baseline
and 5.98% below epoch040.  Epoch040 and epoch041 are now two consecutive
epochs below 9.5 steps/s, which upgrades the event from a transient sample to
a persistent throughput warning.

At the audit GPU1 utilization was 66% with the two M1 owners; host load was
about 7.3.  PACD remained GPU0's only owner.  CPU/memory/IO PSI were zero at
the terminal sample, no OOM, GPU crossover or cross-root descriptor existed,
and every P0 science and correctness gate passed.  The evidence therefore
supports a scheduling association, not a causal or scientific-failure claim.
No process will be stopped, reprioritized or rebound.  Future PACD packed
execution must instead pre-register four disjoint CPU masks and pass an
isolated-versus-packed throughput smoke.

### 6.25 PACD epoch042 under unchanged M1 concurrency

M1 T0/C1 remained simultaneous GPU1 owners with affinity `0-31` while PACD
published epoch042, SHA-256
`80a69557cabcb2deb3b531c8ebfdebe2a2a2ce6f0afe9a6b07560ab131621d52`.
PACD throughput was `8.8858568` steps/s, `11.84%` below baseline but 0.34%
above epoch041.  Epoch040--042 are three consecutive low-throughput epochs.
All P0 science and receipt gates remain exact.

GPU devices remained separated; there was no third owner, cross-root
descriptor, OOM or persistent PSI.  The evidence remains associational, but
it confirms that broad overlapping CPU affinity is unacceptable for a future
PACD packed cohort.  Existing processes remain untouched; the corrective
mechanism is the next route's exact four-way CPU partition and pre-launch
packed-throughput gate.

### 6.26 PACD epoch043 partial throughput recovery

With the same M1 T0/C1 GPU1 owners and broad CPU affinity unchanged, PACD
published epoch043, SHA-256
`452902fb61179f149e52a2b535507f5f49f6b73861fc0a5815d9ae9dcebc8485`,
at `9.2498458` steps/s.  This remains 8.23% below baseline but is 4.10%
faster than epoch042.  The run has four consecutive low epochs, yet the last
two do not show continued deterioration.

Every P0 correctness gate passed; GPU devices remained separated; there was
no third owner, persistent PSI, OOM or cross-root descriptor.  Existing jobs
remain untouched.  The observed pattern continues to support strict disjoint
CPU masks and a packed-versus-isolated throughput smoke for PACD-2UP rather
than any intervention in the live jobs.

### 6.27 PACD epoch044 and first final-window checkpoint

With the same M1 T0/C1 pair simultaneously owning GPU1, P0 published
`epoch044.json` and `epoch044.pt`. Their body SHA-256 values are respectively
`16ba92799d0e0c9b54fca68256df28fa0e5f2a2673ee0aedc12f8b607364a55f`
and
`f01bd4bc83e72f12ab1430a01dbfdc45b2732ee0506a5239a89761336ebb0139`.
Root verified the exact 49-body/98-leaf prefix and the first immutable
final-four checkpoint without loading its tensors.

Epoch044 ran at `9.1908758` steps/s, 8.82% below the long-run baseline and the
fifth consecutive epoch below 9.5. Every science, optimizer, gradient, RNG,
sampler and artifact gate nevertheless passed. GPU ownership remained
disjoint: P0 alone on GPU0, and T0/C1 alone on GPU1. GPU1 showed about
58--60% utilization with about 2.35 GiB used, confirming that two small jobs
can use a card materially better than one. Both M1 jobs still had affinity
`0-31`, overlapping P0's `0-3,16-19`; the simultaneous utilization gain and
P0 slowdown therefore motivate PACD-2UP with four exact disjoint CPU masks,
not unrestricted co-tenancy. No causal attribution is made and no live
process was changed.

### 6.28 PACD epoch045 under continued two-process GPU1 occupancy

P0 naturally published epoch045 and checkpoint045 while M1 T0/C1 continued
as GPU1's only two compute owners. The immutable body SHA-256 values are
`d1f5d0e903508d3834ab7305edde4ecde487e2247c4c6afec00eaa8e88a4bd1b`
and
`efaa259bb47b7c1a5503373f5b377c457ba17184059563d78bbd90b933619ed7`.
Root verified the exact 51-body/102-leaf prefix without loading checkpoint
tensors.

Epoch045 throughput was `8.9936381` steps/s, 10.77% below baseline and the
sixth consecutive epoch below 9.5. Mean training loss nevertheless improved
to `0.3850463758`, and every correctness gate passed. GPU ownership remained
strictly separated; GPU1 used about 2.35 GiB and showed 43% at the final
sample. Both M1 processes retained broad affinity `0-31`, while P0 retained
`0-3,16-19`. Host PSI remained zero and no OOM or cross-root FD appeared.
This remains associational evidence only; no running process was modified.

### 6.29 M1 C1 natural terminal and partial GPU1 release

At 23:48:25 HKT the C1 arm atomically published a successful terminal and
PID `1055233` exited naturally. The immutable terminal body SHA-256 is
`431df977c9dcc64651395dc4fd949cfe417f1a7b9567c8c574fbfa2c47a7367b`;
body and canonical sidecar are regular, single-link mode-`0444` files with a
matching digest declaration. Status is `COMPLETE_MATCHED_ARM_TRAINING`, arm
is `c1`, best epoch is 46, last epoch is 49, every recorded checkpoint has a
strict-reload proof, SWA is explicitly forbidden, and target optimizer/
backward/update count is zero.

This is a valid trained producer, not yet a performance result. T0 remains
live and no matched held-out score has been published, so no C1 benefit or
null result is inferred from source loss. GPU1 ownership fell naturally from
the T0/C1 pair to T0 PID `1052875` alone; sampled utilization/memory fell to
about 24%/1.20 GiB. P0 remained GPU0's sole owner. The partial release does
not admit PACD work because both the P0 terminal and complete M1 route release
remain absent. No process, affinity or artifact was changed by this audit.

### 6.30 M1 T0 natural terminal and complete GPU1 release

At 2026-09-02 00:03:59--00:04:07 HKT the matched T0 control atomically
published its successful terminal and PID `1052875` exited naturally. The
terminal body SHA-256 is
`a8270a5eb541b968ee547e8b097a68a74528fc6c9278430d733e7564a3cb2a47`;
body and canonical sidecar are regular, single-link mode-`0444` files with a
matching digest declaration. Status is `COMPLETE_MATCHED_ARM_TRAINING`, best
epoch is 44, last epoch is 49, checkpoint strict reloads pass, SWA is
explicitly forbidden, and target optimizer/backward/update count is zero.

T0 and C1 are therefore both valid completed 50-epoch source-only producers.
There is still no matched held-out score or Phase-3 receipt, so their source
losses and best epochs do not determine whether C1 improves R2. After T0's
exit, GPU1 returned to 0%/23 MiB with no compute owner and neither M1 process
retained broad affinity. P0 remained GPU0's sole owner and pre-terminal. This
is a complete M1 training-resource release, but not a PACD admission: P0
terminal and fresh post-terminal cohort checks remain required. No action was
taken on the newly idle GPU1.

### 6.31 M1 probe arm and PACD epoch046

The same M1 pipeline naturally started a `probe` arm on GPU1 after T0/C1
terminalized. PID `1098142` uses CVD1, physical GPU1, affinity `0-31`, and
holds descriptors only into the M1 canonical root and its `probe` child. At
the epoch046 audit the probe root had immutable attempt/launch pairs but no
terminal, failure or score; it is not yet a scientific result.

During this independent probe, P0 published epoch046 and checkpoint046 with
body SHA-256 values
`50bd62d0949acb4d900f84bf2f9b8e2d128b7022e205e7095b0e81f19085dea0`
and
`1c6e753bd9c92a3b0b97a73edd3b72e25c10a6344c21787db5fba385e5848998`.
Root verified the exact 53-body/106-leaf P0 prefix without loading tensors.
Mean loss improved to `0.3848239100`; throughput fell to `8.8188528`
steps/s. Every correctness gate passed. GPU ownership remained separated,
PSI was zero, and no cross-root FD or OOM appeared. No causal attribution or
live intervention was made.

### 6.32 M1 probe/Phase-3 terminal and scientific result

The probe and Phase-3 stages terminalized naturally and released GPU1. Probe
terminal SHA-256 is
`638dec0aa636e03a43741f57b81d692fbc628a78ba7afcd81b91baebd7b4339d`;
Phase-3 terminal SHA-256 is
`d0b71443d54a5cd2ab8e7d8f7981c732d62b2f66dfbf314519ebf6fb6309437f`.
The Phase-3 root has exactly 38 regular bodies plus 38 canonical sidecars,
all mode `0444`, link count one and digest-valid, with no failure. The
authoritative result document is
`RESULT_M1_T0C1_PREFIX_50EP_20260901.md`.

At epoch50, static held-out R2 is `0.5337635875` for T0 and `0.5278574228`
for C1, so C1 minus T0 is `-0.0059061646`. With CDM activity FIFO the held-out
means are `0.5493777990` and `0.5389648080`, so C1 minus T0 is
`-0.0104129910`. CDM minus static is positive on the held-out fold
(`+0.0156142116` for T0; `+0.0111073852` for C1) but substantially negative
on held-in training sessions. C1 is therefore not promoted. The 50-epoch
warmup/cosine recipe fits held-in harder yet transfers worse than the sealed
20-epoch constant-LR pair; it is a cross-recipe diagnostic, not an improved
M1 recipe or formal benchmark verdict.

The concurrent T0/C1 training pair completed in 7.02 h versus projected
serial 11.72 h, saving about 4.7 h; individual wall inflation was 1.12x and
1.22x. This supports controlled same-GPU packing as an engineering technique,
not C1 as a scientific contribution. P0 remained the sole GPU0 owner with no
cross-root descriptor or GPU crossover. After Phase-3, GPU1 returned idle.
