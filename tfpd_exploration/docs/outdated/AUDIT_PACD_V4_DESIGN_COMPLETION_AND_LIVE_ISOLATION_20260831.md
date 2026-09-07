# Audit: PACD V4 Design Completion and Live Isolation

Date: 2026-08-31

Status: design-complete implementation boundary. This document is read-only
review evidence. It is outside the active V3 P0 execution closure and does not
authorize V4 code edits, capabilities, result roots, data/checkpoint access,
CUDA initialization, or launch.

## 1. Decision

The post-P0 PACD P1/P2 route is fully specified at the design and receipt-codec
level. No unresolved model, loss, optimizer, sampler, RNG, checkpoint, SWA,
device-selection, scheduler, lineage, or scoring choice remains.

Implementation is intentionally not started. The frozen work order forbids
editing the shared lifecycle until the active P0 V3 producer naturally
publishes and independently passes its complete immutable terminal graph.

The distinction is:

```text
design complete
    yes

implementation complete
    no; forbidden before the P0 terminal gate

experiment complete
    no; P1/P2 have not been admitted or launched
```

## 2. Frozen authority stack

The successor is governed by these complementary authorities:

| Authority | Role | SHA-256 |
|---|---|---|
| `WORKORDER_PACD_P1_P2_ADMISSION_MULTIGPU_V4_20260831.md` | science invariants, P0 gate, roots, fixed GPUs, receipt topology, tests and launch order | `34a67357d66c36816457252e82d5ac7dcecf34340dafff1f372c67571dc09cd3` |
| `AUDIT_PACD_V4_PARALLEL_ROUTE_ISOLATION_20260831.md` | exact CPU profiles, tracked-route release gate, scheduler and host-pressure codecs | `7c2c27697329e13f53fc2dacd7981cec2f24b88f550ab1af38c266c4cc2dcc52` |
| `AUDIT_PACD_V4_SHARED_LIFECYCLE_SEAM_20260831.md` | backward-compatible profile seam and staged CUDA/PID/failure semantics | `10bc3e81385e43bb6f9f79ebbdf13db8f5100a37de532d471f5792ecc83c9f8c` |

The no-data Score V2 producer codec binds all three authorities. Its current
closure is:

```text
b7416d0333149708be7c4898729644c14504934fb3bd03b4dc1df76194490db4
```

The frozen codec boundary has `37/37` focused no-CUDA tests passing; the
unchanged Score V1 regression has `19/19` passing with only its existing
TorchMetrics deprecation warning.  A clean `python -S` dry run imports no
Torch and reports `producer_literals_deferred=true` and
`score_authorized=false`.  These are implementation-codec results, not a live
score result.

Root independently repeated the static audit at 2026-08-31 17:57 HKT.  Two
consecutive `source_closure` reconstructions contained exactly 51 explicit
files and reproduced the closure above.  The Score V2 work order, V4 work
order, parallel-isolation audit and shared-lifecycle audit each matched their
four frozen SHA literals.  `LIVE_MIXED_PRODUCER_LITERALS` remained `None`, the
dry profile remained unauthorized, and Torch was absent from `sys.modules`.

The scorer remains fail-closed because live P0/P1/P2 producer literals are
not all available.

The review-only pre-terminal handoff checklist is
`AUDIT_PACD_P0_TO_V4_HANDOFF_TEMPLATE_20260831.md`, SHA-256
`ca362824d68968dda88dfd93518c42e6b16ffcd0c3fc41450ecad0d5630fc0b1`.
It records the exact 58-body/116-leaf topology, current immutable prefix,
epoch/checkpoint/SWA/manifest/terminal laws, resource-release gates and the
literal payload to freeze after terminal.  It is deliberately outside both
P0 and Score V2 execution closures and authorizes no implementation or live
action.

## 3. Requirement-by-requirement completion matrix

| Requirement | Design evidence | Current state |
|---|---|---|
| PACD science remains unchanged | work order Sections 1, 4 and 7 | frozen |
| P0 must precede P1/P2 | work order Sections 2 and 5 | frozen; P0 still active |
| full P0 graph is descriptor-validated | exact 58-body/116-leaf contract in Section 5 | frozen; live literals pending terminal |
| P1/P2 roots and identities | work order Section 3 | frozen |
| P1 fixed to GPU0, P2 fixed to GPU1 | work order Section 3.2 | frozen |
| no automatic fallback or load balancing | work order Sections 3.2 and 9 | frozen |
| one shared 48-epoch loop | work order Sections 4 and 7; seam audit Sections 1 and 4 | frozen |
| legacy V1/V2/V3 behavior is preserved | seam audit Sections 3--6 | frozen and test-required |
| CUDA binding precedes stack/data/checkpoint load | seam audit Section 3.2 | frozen |
| honest pre-/post-CUDA failures | seam audit Sections 3.2--3.4 | frozen |
| final physical GPU PID is the bound process | work order Section 6; Score V2 cross-link codec | frozen and adversarially tested |
| exact disjoint CPU profiles | isolation addendum Section 2 | frozen |
| procfs host-pressure gate | isolation addendum Section 4 | frozen and adversarially tested in the scorer codec |
| opaque root/arm/device/factory capability | work order Section 6 | frozen |
| V4 attempt/launch/terminal/failure placement | work order Section 6.1 | frozen |
| P1/P2 mixed-lineage scoring | work order Section 10; Score V2 | codec implemented, live literals deferred |
| public CLI cannot inject observers/authority | work order Sections 6 and 8 | frozen and test-required |
| launch only after root freshness and resource release | work order Section 9; isolation addendum Section 3 | SUA historical release observed; P0 gate and fresh launch-time recheck remain unsatisfied |

No cell in this matrix is waiting for a scientific design choice. The pending
items require producer artifacts or post-P0 implementation, not more method
ideation.

## 4. Exact implementation clarifications

These details are easy to misread and are therefore fixed explicitly.

### 4.1 Physical index types

The V4 device **profile** stores `physical_index` as the literal string
`"0"` or `"1"`, matching the frozen profile payload. The selected-GPU
**attestation** stores `physical_index` as the observed integer `0` or `1`.
The implementation must not normalize one into the other or relax the exact
codec.

### 4.2 Logical device

Both processes use logical `cuda:0` because each process exposes exactly one
physical GPU through its own CVD:

```text
P1: CVD=0 -> physical GPU0 -> logical cuda:0
P2: CVD=1 -> physical GPU1 -> logical cuda:0
```

Logical `cuda:1` is forbidden.

### 4.3 Device observation cross-links

Terminal device evidence must not merely be independently valid. It must
copy and exact-link lifecycle observations:

```text
terminal.issuance_idle  == attempt.issuance_idle
terminal.pre_attempt    == attempt.pre_attempt
terminal.post_attempt   == launch.post_attempt
terminal.context_bound  == launch.context_bound
terminal.final.pid/list == terminal.context_bound.pid/list
```

The final current PID must also have been checked against `os.getpid()` by the
live producer. An offline scorer can verify immutable cross-links but cannot
retroactively call `os.getpid()` for a historical process; this is why the
producer and scorer checks are both required.

### 4.4 Failure stage

If context creation never succeeds, failure evidence must show
`cuda_bound=false` and an idle selected GPU. If context creation succeeds and
any later stack/data/checkpoint step fails, it must show `cuda_bound=true` and
the sole selected-GPU PID equal to the current process. A device/scheduler
recheck error is additive evidence and must never replace the original
runtime exception.

### 4.5 CPU placement

The isolation addendum supersedes only the older scheduler recommendation in
the work order. Production placement is exactly:

```text
P1: 0-3,16-19
P2: 4-15,20-31
```

The science contract remains seed 42, batch 32, workers zero and all numerical
thread limits one.

## 5. Live coexistence audit

At 2026-08-31 17:19 HKT:

```text
PACD P0 V3
  PID 783126
  GPU0 / CVD0
  affinity 0-3,16-19
  one active compute process

SUA P1 transfer
  PID 824234
  GPU1 / CVD1
  affinity 0-31
  one active compute process
```

The routes have distinct GPUs, canonical roots and source packages. No
cross-root write has been observed. The SUA process nevertheless has broad
CPU affinity and therefore overlaps the PACD P0 CPU set. At the audit sample,
each process used approximately one CPU and host memory pressure was low, so
there was no observed material slowdown or memory pressure. The overlap is a
real scheduling risk and must not be described as disjoint isolation.

No process affinity, priority, GPU binding, root or worker was changed by this
audit. The SUA route is owned by another agent. Read-only monitoring reports
terminal/failure or material resource pressure; it does not intervene.

This live overlap has a simple future consequence: if SUA remains active when
P0 finishes, neither V4 arm is admissible. P2 conflicts with SUA on GPU1 and
its exact CPU set; P1 conflicts with SUA's broad CPU set. V4 waits for SUA to
exit naturally and for both fixed release gates to pass. It does not move an
arm, shrink another route, change affinity, or reuse a root.

The SUA provenance is also separate from PACD. An initial source-root failure
did not produce a complete failure graph; the canonical attempt was recreated
for the current repaired run. That history must be disclosed in the SUA
result, but it does not enter the PACD predecessor or select a PACD arm.

### SUA terminal release at 17:36 HKT

The repaired SUA route terminalized naturally before PACD P0. PID `824234`
and its timeout parent exited; physical GPU1 returned to `0%` utilization with
only the normal `23 MiB` context footprint, and the broad `0-31` CPU owner is
gone. Its immutable terminal body SHA is
`00a3c409965de95f1ef778f44263a6527f6de74710ae3f4cd844806aec4e91a1`.
The scientific result is a non-promoted external-M4 `+0.008083` (`10/15`),
not a PACD input or selection signal.

This removes the observed SUA GPU1/CPU-affinity conflict. It does **not**
admit V4: P0 remains active, the required P0 terminal predecessor does not
exist, and shared-lifecycle editing is still forbidden. After P0 terminal,
both arm capabilities must perform fresh process, GPU, CPU-affinity, host-
pressure and root-freshness checks; this historical release is evidence, not
a standing reservation.

### New local-M2 GPU1 owner at 19:19 HKT

A later independent route started naturally on physical GPU1 after the SUA
release:

```text
PID 832229
run_cdm_p1_m2_local_v1.py --stage execute --gpu-index 1 --batch-size 1024
CUDA_VISIBLE_DEVICES=1
CPU affinity 0-31, 81 threads
```

PACD P0 remains PID `783126` on GPU0/CVD0 with all five threads constrained to
`0-3,16-19`.  Thus GPU placement is disjoint, but CPU affinity is not.  At the
19:19 sample, each route consumed about one CPU, load average was about 2.08,
CPU/memory/IO PSI averages were zero, available RAM was about 50 GiB, and P0's
epoch016 throughput remained within its established envelope.  No material
interference was observed and no intervention was taken.

This observation supersedes the earlier statement that GPU1 was idle; it does
not change the frozen device mapping.  V4-P2 must not issue a capability or
reserve a root while PID `832229` or any successor owns GPU1.  V4-P1 remains
barred independently by the P0 terminal gate.  Both arms require a fresh
post-P0 resource snapshot rather than relying on either the earlier SUA
release or this transient coexistence state.

### Local-M2 terminal release at approximately 19:20 HKT

PID `832229` and its parent exited naturally after publishing a valid LOCAL-M2
terminal.  GPU1 returned to `0%` utilization with no compute application.  The
terminal body SHA-256 is
`414f1e825a0ac0067e7c44cb2b34ea3c024381def041de77c45fc5d7d6ec3c82`;
its external-M4 delta is exactly zero and it is not promoted.  This result is
scientifically and genealogically separate from PACD.

The live conflict is therefore released again.  No intervention, process
change or root mutation was used to obtain the release.  P0 remains active,
so shared V4 implementation and every P1/P2 capability remain forbidden.  A
future launch must still use a fresh resource snapshot; this completed route
does not create a standing reservation for GPU1.

### Newly announced M1 comparison on GPU1

The exact pre-launch coexistence gate is recorded in
`AUDIT_M1_PACD_CONCURRENT_ISOLATION_20260831.md`, SHA-256
`3ac421d9c6f59a16785ed12fd061640f0d232051ab4ff845893ea8abcd24f71c`.
It is review evidence outside all execution closures.

At 2026-08-31 23:04 HKT, the user reported that another agent had started an
M1 comparison on physical GPU1.  The first read-only host snapshot did not yet
show a GPU1 compute process: GPU1 was `0%` with the normal 23 MiB footprint,
while PACD PID `783126` remained the sole compute owner on GPU0 and retained
CPU affinity `0-3,16-19`.  Host CPU, memory and IO PSI were all zero.  This
means only that the M1 route had not yet created an observable CUDA context;
it is not evidence that the route is absent or released.

GPU1 is reserved for the announced M1 comparison.  PACD will not use its
temporary idle appearance as permission to claim GPU1.  When the first M1
compute PID appears, read-only monitoring must bind its physical GPU, CVD,
CPU affinity, command/cwd and result root where the PID namespace exposes
them.  It must not change the process, priority, affinity, files or root.

The frozen V4 mapping is unchanged: P1 is GPU0 and P2 is GPU1, with no
fallback or arm migration.  Consequently:

- P2 cannot mint a capability or reserve a root while the M1 route owns GPU1;
- P1 remains independently blocked by the active P0 terminal gate;
- after P0, P1 may proceed only if its exact CPU profile is also disjoint from
  every live M1 owner and the host-pressure gate passes;
- if M1 still occupies GPU1 when P1 becomes admissible, P1 may run alone on
  GPU0 while P2 waits; this is scheduling, not a science or device-map change;
- P2 starts only after M1 exits naturally and a fresh GPU1/PID/CPU/root check
  passes.

No M1 artifact is a PACD predecessor or selection signal.  This section is a
live scheduling disclosure outside all execution closures.

### M1 terminal release at 23:35 HKT

The M1 comparison terminalized naturally with terminal body SHA
`b948976e624a5a53fbc5ac5c2f5f7cf6e379c8408b1a749b80a3b5dbada49d51`.
Its exact 12-body/24-leaf graph is immutable and contains no failure.  The
metric-only result reports a fold-defined held-out minus held-in gap of
`-0.124885` with CI `[-0.133009,-0.112670]`; it is scientifically separate
from PACD and selects no PACD arm.

Post-terminal, GPU1 repeatedly had no compute owner while GPU0 retained only
PACD PID `783126`; host PSI remained zero.  The M1 scheduling conflict is
released.  Its V1 receipt records logical `cuda:0` plus a declared CVD=1
constraint but lacks durable UUID/PID/affinity evidence, so the historical
physical assignment is not upgraded beyond that declaration.  This does not
admit V4: the valid P0 terminal predecessor is still absent.

### New M1 T0/C1 pipeline owner at 00:36 HKT

The terminal release above applies only to the completed metric-only M1
comparison.  A distinct M1 T0/C1 training pipeline started later and
supersedes the statement that GPU1 is currently released.  At the 2026-09-01
00:58 HKT read-only sample:

```text
pipeline shell  PID 847532  /tmp/m1_t0c1_pipeline.sh
M1 T0 compute   PID 847764  physical GPU1 / CVD1  affinity 0-31
PACD P0 compute PID 783126  physical GPU0 / CVD0  affinity 0-3,16-19
```

The M1 T0 process uses logical `cuda:0` under `CUDA_VISIBLE_DEVICES=1`, so the
physical assignment is GPU1.  Its active result subroot is
`tfpd_exploration/results/m1_t0c1_prefix_v1/t0`; only attempt and launch pairs
were present at the sample.  The script is expected to continue to C1 and
phase3 after T0.  Therefore release is attached to the complete pipeline,
not to the lifetime of one stage PID.

The two physical GPUs are disjoint, but the M1 `0-31` CPU affinity covers the
P0 CPU set.  Host CPU, memory and IO PSI were zero and P0 epoch022 passed all
science/safety gates, so there is no observed correctness failure.  The
overlap is nevertheless a live scheduling risk and must not be normalized
into the V4 contract.  Root and monitoring agents made no process, affinity,
priority, environment, code or result-root change.

A 01:07 HKT `/proc` descriptor audit found no open descriptor from either
compute process into the other route's result root.  Their only writable
workspace-or-`/tmp` descriptors were their separate stdout/stderr log files:
`/tmp/pacd_p0_full_v3_20260831.log` for P0 and
`/tmp/m1_t0c1_pipeline_stdout.log` for M1.  This is read-only live isolation
evidence, not a substitute for final receipt-graph validation.

Scheduling consequences are now exact:

- no PACD job other than the already-running P0 may start;
- P1 remains blocked by the missing P0 terminal;
- P2 remains blocked both by the missing P0 terminal and by the live M1
  pipeline on GPU1;
- after P0 terminal, P1 may start only if its fixed CPU envelope is disjoint
  from every remaining M1 stage and the fresh host gate passes;
- P2 waits until the pipeline shell and every stage process exit naturally,
  GPU1 has no compute owner and fresh PID/affinity/pressure/root checks pass;
- low GPU utilization or a transition between T0, C1 and phase3 is not a
  release event.

The current P0 prefix through epoch031 has exact body/sidecar integrity and
1,085,600 cumulative steps.  Epoch031 SHA is
`70e183a511d71806435bb6c52f5a62f12c9da3c6c264ad6679c57c8ada3a4f8d`.
The P0 execution closure still reconstructs twice as
`3356fb124ff4d18a2035bfda8e7eacc28a1642ce3f36b290155a5bcdf7faea2f`
over 48 explicit files without importing Torch.

### M1 T0 fail-close at 01:13 HKT

The M1 T0 stage subsequently fail-closed before its first optimizer update:

```text
error: TrainerError('epoch loss row coverage drift')
prepared: true
optimizer_steps_completed: 0
target_metric_only: true
terminal_published: false
failure SHA: 694c93cf63a12f470e65c5b62c8830d49456048753aa0d4abbe5135715fea8b8
```

Its pipeline shell logged `chain stops`; the shell and child exited naturally,
and no C1 or phase3 root appeared.  The exact attempt/launch/failure graph has
six regular mode-`0444`, single-link leaves with valid canonical sidecars.
There is no scientific T0/C1 result from this attempt.

Physical GPU1 then returned to 0%/23 MiB with no compute PID, while GPU0
retained only P0 PID `783126`.  Host PSI remained zero and P0 remained healthy.
This removes the current physical GPU1 owner but does not admit V4: P0 has not
terminalized, and the M1 owner may still issue a separately reviewed retry.
Root will not alter or retry that route and will not interpret transient GPU1
idleness as permission to reserve a V4 root.

### External M1 T0 retry at 06:11 HKT

The M1 owner later archived the failed canonical `t0` directory to
`t0_failed_loss_coverage_drift`, recreated canonical `t0`, and started a
resume-from-T0 pipeline.  This was an external owner action; PACD/root did not
move, rewrite or request it.  The archived failure leaves retain their earlier
body hashes, but their directory path/inode identity changed and must be named
honestly in any later evidence.

The new retry initially published only attempt/launch body+sidecar pairs:

```text
attempt 4e9d9a7ea26679d69a5ae1b0529c0025f17c077e0322df99dd1a1c5c5e7aa883
launch  83ff27462308bc87278eea3df49dcf93e24ed85b0db501d68f6ad0ead4257e13
shell/Python PID 860201/860203
CVD 1, logical cuda:0, CPU affinity 0-31
```

At the first sample launch still recorded `cuda_initialized=false`, and GPU1
had no registered compute process.  This does not release GPU1: the process is
the active CVD1 route and may materialize a CUDA context later.  Its CPU mask
overlaps P0's `0-3,16-19`; host PSI was zero and P0 remained the sole GPU0
owner.  The historical 48-file P0 closure reconstructed twice unchanged after
the archive/retry action.  Therefore no correctness failure is observed, but
P2 is again explicitly blocked by a live M1 retry as well as by the missing P0
terminal.  Monitoring remains read-only and will not change either route.

At 06:17 HKT, the retry's CUDA context materialized and `nvidia-smi` bound
PID `860203` to physical GPU1 UUID
`GPU-2220ed5d-25ea-1839-28d7-ad4dfa5f6c86`, using about 1,172 MiB.  PACD PID
`783126` remained the only GPU0 compute owner.  Thus the fixed devices are
physically disjoint, while P2 has an actual live GPU1 conflict and the broad
M1 CPU mask still overlaps P0.  Host PSI stayed zero and neither route
reported OOM/failure/nonfinite state.  P2 cannot mint or reserve while this
PID or any retry successor owns GPU1.

At 07:16 HKT, retry PIDs `860201/860203` disappeared without publishing a
terminal or failure.  The external owner archived its four-leaf attempt/launch
directory as `t0_attempt2_aborted_slow_full_detail_recording`; the pipeline
log still has no completion/failure/C1 line.  This is not a valid producer
terminal or failure receipt and cannot be used as science or lineage.

GPU1 became physically idle again while PACD remained the sole GPU0 owner and
host PSI stayed zero.  The P0 closure reconstructed twice unchanged.  The
physical conflict is absent at the sample, but GPU1 remains administratively
reserved because the M1 owner may issue another retry and the aborted attempt
has no governing terminal.  V4 does not use or repair this evidence and P2
remains blocked independently by the missing P0 terminal.

At 07:23 HKT, GPU1 briefly showed unattributed Python PID `865155` using
about 1,170 MiB, but it exited before command/environment/affinity evidence
could be bound and produced no M1 receipt.  This is recorded without guessing
its route identity.  At 07:27 HKT, an external owner then rewrote the M1
`plan.py` and ran compile/tests.  That path is outside the P0 closure, which
reconstructed twice unchanged.  Any subsequent M1 retry must bind its new
source closure; it cannot reuse the aborted attempt's source authority.  No
V4 capability/root action is permitted during these unresolved owner changes.

At 07:27 HKT, the M1 owner issued a third T0 attempt with a refreshed closure
`3702f99941ac39664b6c4938886d8665d0d3e056fc618ecb9152135e660a2391`
and PIDs `866226/866228`.  At 07:33 HKT, Python PID `866228` materialized on
physical GPU1 with about 1,172 MiB.  PACD remained the sole GPU0 owner; host
PSI was zero and there was no owner crossing.  The retry had no terminal,
failure or C1 receipt.  This restores an actual live GPU1 conflict for P2;
monitoring remains read-only and no V4 capability/root may be issued.

### Transient second GPU1 owner at 09:12 HKT

Read-only monitoring observed unknown PID `877688` on physical GPU1 at
09:12:43 HKT while M1 PID `866228` was still present.  This is direct evidence
of transient same-GPU co-tenancy for the M1 retry.  It was not PACD: P0 stayed
on physical GPU0 as PID `783126`, and no PACD V4/P1/P2 process or root exists.

The transient PID exited before its `/proc` identity could be captured, so
this audit makes no route or owner attribution.  By 09:13--09:14 HKT, GPU1
again had only M1 PID `866228`; GPU0 still had only P0 PID `783126`.  The M1
root remained attempt/launch-only, P0 remained at epoch030 with its closure
unchanged, and CPU/memory/IO PSI averages were zero.  No process or root was
changed to obtain the release.

This event strengthens rather than relaxes the future admission rule: an
apparently idle or single-owner historical sample is never sufficient.  Each
V4 arm must obtain a fresh fixed-device observation at capability issuance,
and P2 remains inadmissible while M1 or any additional process owns GPU1.

### Current hold after the 09:44 HKT isolation recheck

The latest read-only recheck again found PACD PID `783126` as the sole GPU0
compute owner and M1 PID `866228` as the sole GPU1 compute owner.  Their CVD
bindings remained `0` and `1`, respectively; no cross-root file descriptor was
observed; CPU, memory and IO PSI averages were all zero.  PACD had reached the
valid immutable prefix through epoch034, while M1 completed T0 and naturally
entered the C1 pre-CUDA stage.

This clean physical-GPU separation does not admit a new job.  M1's current
`0-31` CPU affinity intersects both frozen V4 profiles, not only P2's.  Hence
an early P0 terminal would release GPU0 but would still leave P1 blocked by
the CPU-overlap gate; P2 would remain blocked by both GPU1 ownership and CPU
overlap.  Neither arm may launch until the M1 pipeline naturally releases and
the complete fresh scheduler/device/pressure/root gate passes.  The current
action remains read-only monitoring plus P0 receipt audit.

The accepted M1 T0 terminal is a 29-body/58-leaf source-only producer with
terminal SHA-256
`4aea40a317047beec059505231bb9996190c4eb2235392d08a3c948cf7ab9aa9`.
It completed 20 epochs and 99,020 optimizer steps with best=last epoch 19,
SWA disabled and zero target optimizer/backward/update.  This natural stage
completion does not release the M1 route: the same pipeline shell started C1
PID `886913`, CVD1, affinity `0-31`.  C1 had not yet created a GPU1 context at
the first sample, but its temporary pre-CUDA interval is explicitly not an
admission opportunity for P2.  Both V4 arms remain held by C1's broad CPU
profile, and P2 additionally remains held by the reserved GPU1 route.

At 10:25 HKT, C1 PID `886913` naturally appeared as the sole compute owner on
physical GPU1, using about 1,150 MiB, while PACD PID `783126` remained the
sole GPU0 owner.  C1 retained CVD1/affinity `0-31` and host PSI stayed zero.
This confirms the expected physical binding after the pre-CUDA interval; it
does not relax either V4 arm's hold.

### Independent pre-implementation re-audit at 12:33 HKT

Root re-read the frozen work order, the parallel-isolation addendum, the
shared-lifecycle seam audit, the active shared executor and the V3 profile
instead of relying on their earlier summaries.  All authority and scientific
anchor hashes remained exact:

```text
training work order       34a67357d66c36816457252e82d5ac7dcecf34340dafff1f372c67571dc09cd3
parallel addendum         7c2c27697329e13f53fc2dacd7981cec2f24b88f550ab1af38c266c4cc2dcc52
shared-lifecycle audit    10bc3e81385e43bb6f9f79ebbdf13db8f5100a37de532d471f5792ecc83c9f8c
active shared executor    62d4b71291ffaa7175e8cdfd42d14128ee7bea428e8c4a887eb135d8292a7f75
paired PACD core          9331288cf42d7c7924f6135f3ec8e3659f0ea19b70c8fad84448b01773c8172d
shared epoch runner       adb6c471e0e28eb544e9c7bc3f9e4a9823e2803ef33d6de5cfa5545c917a54f0
V3 plan                   0368fce6b71f42cbe45aa70e56090cdd24c985d660726c8fbdb7ebbf8f9039c3
V3 profile wrapper        c072bf7715db61cf38e7f7c2cb9e98765daf26d977ac71e736ebdd6c56f8f7ad
```

The six future V4 package/CLI/test paths were still absent.  Thus no
post-terminal seam or route bytes had been created early and the active P0
closure was not contaminated.

This re-audit also resolved the only apparent textual ambiguity.  Section 9
of the earlier work order labels `0-7,16-23` / `8-15,24-31` as a recommended
scheduler-only partition.  The later frozen isolation addendum explicitly
supersedes only that recommendation and fixes production placement to P1
`0-3,16-19` and P2 `4-15,20-31`.  The addendum is the authoritative scheduler
contract; an implementation may not choose between the two descriptions.
This precedence changes no numerical or scientific field.

The live physical profile still matched the frozen literals exactly: both
devices were NVIDIA GeForce RTX 3090 on driver `535.309.01`; GPU0 had UUID
`GPU-ac7388a5-2e98-300a-fdb3-0b67bfd494d9` and PCI bus
`00000000:01:00.0`, while GPU1 had UUID
`GPU-2220ed5d-25ea-1839-28d7-ad4dfa5f6c86` and PCI bus
`00000000:03:00.0`.  This is descriptive pre-implementation confirmation,
not a substitute for fresh capability-time observations.

At the same sample P0's latest complete receipt remained epoch034 and C1
remained attempt/launch-only.  GPU0 and GPU1 each had exactly their expected
single owner; no cross-root descriptor or host PSI pressure was observed.
Consequently the design is statically ready, but both the P0 terminal gate
and the M1 release gate remain unsatisfied.

### M1 route release at 12:49 HKT

C1 subsequently published a valid source-only training terminal and exited
naturally.  Its 29-body/58-leaf graph completed 20 epochs and 99,020 updates;
terminal SHA-256 is
`cbef49e8e356103c56a9ffa673a268de29e1c5564fc5957c26474b1cf065da99`.
The pipeline's metric-only Phase 3 then failed before any cell completed,
with immutable failure SHA-256
`3c1a92634f6b4aa5baae138feac7f776005d01884224777e00ba30aedb26ddbe`
and no target table terminal.  This does not alter PACD lineage or science.

The C1 process and pipeline shell released naturally; GPU1 had no compute
owner and no M1 process retained the broad `0-31` affinity.  PACD continued
unchanged as the sole GPU0 owner and host PSI remained zero.  The M1
coexistence hold is therefore historically released, but neither V4 arm is
admissible yet: P0 is still nonterminal, and every capability must perform a
new release/device/scheduler/pressure/root observation after P0 validation.

M1's accepted T0/C1 receipts also contain an internally inconsistent hardware
profile: their launches and live process evidence place them on CVD1/physical
GPU1, but `training.json` records GPU0's UUID/PCI because the trainer queried
physical `nvidia-smi --id 0`.  This M1 provenance defect is fully disclosed in
the M1 isolation audit.  It neither changes the PACD fixed-device literals nor
weakens V4's requirement to bind UUID/PCI/current PID correctly.

The M1 release was later superseded by an external owner action.  The owner
archived the failed Phase 3 graph, repaired its own route and started a fresh
canonical Phase 3 attempt with closure
`10e3c0872bae37ab9517466527b3f1123e26fa840be1f4933e3e39ba86c32496`.
At 13:46 HKT its Python PID `978827` occupied physical GPU1 under CVD1 and
affinity `0-31`; no terminal or failure yet existed.  PACD did not initiate or
intervene in this retry.  The renewed broad CPU overlap blocks both V4 arms,
and GPU1 ownership additionally blocks P2.  Any earlier wording that the M1
route was released is a timestamped historical observation, not current
admission evidence.

The repaired M1 Phase 3 then terminalized naturally at 13:57 HKT and its
processes exited.  The exact 20-body/40-leaf graph has terminal SHA-256
`821818b1ef9068aed87b437af3a8363cf2226e7778776dd9c6c10746bf22f096`;
it is a metric-only, single-held-out-fold diagnostic with
`formal_benchmark_verdict=false`.  GPU1 and the broad M1 affinity were again
released without PACD intervention.  This restores a historical release
sample, but V4 remains barred by the nonterminal P0 and still requires fresh
capability-time resource evidence.

### P0 epoch035 after the M1 release

At 13:16 HKT, P0 naturally published `epoch035.json`, body SHA-256
`c366a1265b2bea203141b2dd25c883849ecf24581de816de8b96e11520cc1891`.
Root independently revalidated the complete epoch000--035 prefix: 36 epochs,
39 bodies and 78 body/sidecar leaves, with exact topology, `0444` mode,
single-link regular files, canonical sidecars and body SHA values.  No
checkpoint, SWA, manifest, terminal, failure or extra leaf exists.

Epoch035 has 33,925 updates and 1,221,300 cumulative updates.  Adam and the
29-materialized/two-lazy parameter topology are finite; P0
prediction/identity mismatch, prefix mutation, RNG and finiteness violations
are all zero.  Decoder coverage is 33,925/33,925; encoder coverage is 33,910,
and all 15 zero-encoder cases are the typed
`all_units_dropped_valid_zero` condition.  Sampler, valid-bin, sentinel and
paired-equality laws remain exact.

P0 remained the sole GPU0 compute owner; GPU1 remained idle after M1's
natural release; no cross-root FD or host PSI pressure was observed.  This
clean coexistence sample still does not satisfy the terminal gate.  The route
continues read-only monitoring with V4 bytes absent.

### P0 epoch036 and throughput recovery

At 14:12 HKT P0 published `epoch036.json`, SHA-256
`9062a1e79c8a616c791d6fe0d6f393be545eafd7be30a99df2df08201de16ef1`.
Root validated epoch000--036 as exactly 40 bodies/80 leaves with all receipt
and invariant checks passing.  Epoch036 completed 33,925 steps, reaching
1,255,225 cumulative updates, at `10.0980521` steps/s.  This is back at the
long-run throughput baseline despite partial overlap with the repaired M1
Phase 3; there is no continuing slowdown signal.

All P0 mismatch, prefix, RNG and finiteness violations remain zero; Adam,
gradient coverage, sampler, valid-bin, sentinel and paired laws remain exact.
GPU0 has only P0, GPU1 is idle after M1's terminal, host PSI is zero, and no
V4 implementation/root/capability exists.  The terminal gate remains the
only current scientific predecessor gate.

### P0 epoch037 and confirmed throughput recovery

At 15:08 HKT P0 naturally published `epoch037.json`, SHA-256
`f6e26afe88151b7d76981a50fbeca63c466329b53d3f015b50d1352df257b982`.
Root independently validated epoch000--037 as exactly 41 bodies/82 leaves,
all regular single-link mode-`0444`, with canonical sidecars and matching body
digests.  No checkpoint, SWA, manifest, terminal, failure or extra leaf is
present.

Epoch037 completed 33,925 steps, reaching 1,289,150 cumulative updates, at
`10.1358876` paired steps/s.  This exceeds the epoch00--29 baseline
(`10.079687`) by `0.56%` and follows the epoch35/36 recovery, so there is no
continuing throughput degradation.  All gradient, mismatch, RNG, prefix,
finiteness, sampler, valid-bin, sentinel and paired-equality laws remain
exact; the only zero encoder-gradient cases are the 17 typed
`all_units_dropped_valid_zero` events, while decoder coverage is complete.

P0 remains the sole GPU0 owner with its fixed CPU affinity, GPU1 remains idle
after the M1 terminal, and no persistent host pressure or cross-root file
descriptor exists.  P0 is 38/48 epochs complete (`79.17%`).  The evidence
strengthens the isolation record but does not relax the full-terminal gate:
V4 implementation, capability and roots remain forbidden until the exact
58-body/116-leaf P0 graph is complete and independently accepted.

### P0 epoch038

At 16:04 HKT P0 naturally published `epoch038.json`, SHA-256
`26f7e0ddd89cc373046e58f56ba2209b87f12c7665ecdd876a4f2e26e3a5cce6`.
Root validated the complete epoch000--038 prefix as exactly 42 bodies/84
leaves with all topology, mode, link, sidecar and body-digest laws passing.
No premature checkpoint, SWA, manifest, terminal, failure or extra leaf is
present.

Epoch038 reached 1,323,075 cumulative updates at `10.1669889` paired
steps/s, `0.87%` above the epoch00--29 baseline.  All science and execution
invariants remain exact: full decoder coverage, only 20 typed empty-mask
encoder-zero cases, zero mismatch/RNG/prefix/finiteness violations, finite
Adam, and unchanged sampler/sentinel/valid-bin laws.  P0 remains the sole
GPU0 owner; GPU1 remains free; host pressure and cross-root checks are clean.
P0 is 39/48 epochs complete (`81.25%`), but the same full-terminal V4 gate
remains in force without relaxation.

### New tracked GPU1 owner after epoch038

At 16:15 HKT an independently owned M1 50-epoch smoke started on physical
GPU1 as PID `1020808`, CVD1, with result family
`tfpd_exploration/results/m1_t0c1_prefix_v1_50ep/smoke`.  PACD P0 remains
the only GPU0 owner; the M1 smoke is the only GPU1 owner.  Thus device
ownership is separated, but M1's `0-31` CPU affinity overlaps P0's
`0-3,16-19` profile.  Initial host PSI was zero, no owner crossover or
cross-root descriptor existed, and neither root nor watcher modified either
job.

This event does not alter the V4 design.  It changes the live admission state:
the tracked-route release condition is currently false, and future P2 cannot
be admitted while GPU1 remains owned.  Root will judge any throughput effect
from completed P0 epochs, not instantaneous utilization, and will not alter
the external route.  Even if the M1 smoke releases before P0 terminal, V4
still requires a fresh, immediate post-P0 scheduler/device/root-freshness
observation before capability issuance.

The M1 smoke terminalized naturally at 16:21 HKT and PID `1020808` exited.
Root validated its exact six-body/12-leaf immutable terminal graph; terminal
SHA-256 is
`ddb7cb9b40c204f62bc94a472f6c85978fa4b5b915c2bf4973b411e8f428fb0c`,
status is `COMPLETE_MATCHED_SMOKE_EQUALITY`, and there is no failure or extra
leaf.  GPU1 returned to no compute owner, while P0 remained the sole GPU0
owner without cross-root descriptors or host PSI.  The tracked route is
historically released again, but the post-P0 fresh admission condition is
not yet evaluated and therefore remains pending.

At 16:24 HKT the independent M1 pipeline advanced to its 50-epoch T0 stage,
PID `1025681`, CVD1, physical GPU1, affinity `0-31`.  Attempt and launch
SHA-256 values are
`680a699a1c3b2bc670620608079a71b37b3d61978eeb0d5bdd49f34113d5974c`
and
`699c36a36bba141f7f92a2db593d1aaeb8d9534d3e1bb084dc662dada2ada93e`.
P0 remains isolated on GPU0, but CPU affinity overlaps.  Initial host PSI,
OOM, owner-crossing and cross-root checks were clean.  The tracked-route
release gate is therefore false again; V4 must neither claim resource
freshness nor issue P2 capability while this owner remains live.

At 16:33 HKT the external M1 owner sent SIGTERM to its T0 PID, retired the
accepted smoke and incomplete T0 under `retired_closure_76ed5cef`, and
published operator-abort note SHA-256
`220657ae9417b60194a1e1192543c271bf41b31d7659c6144d1c1f9a89f7be0e`.
The note says the launcher admission gate was edited to permit concurrent M1
arms, moving that route's implementation closure.  Retired T0 has only
attempt/launch and no terminal/failure, so it is not a result.  PACD P0 was
not edited, signalled or moved; GPU1 released and P0's closure remained exact.

This external closure transition neither authorizes nor invalidates the V4
design.  It means any new M1 owner must be treated as a fresh tracked route,
and the post-P0 capability gate must observe the final live scheduler state
rather than relying on the transient release.

### P0 epoch039 and fresh concurrent M1 lineage

At 17:01 HKT P0 published `epoch039.json`, SHA-256
`31f4863ce9d7b49bb9785b2cdb0624baa91b30b091a8d57bb016e6985050f5fa`.
Root validated epoch000--039 as exactly 43 bodies/86 leaves with all
topology, mode, sidecar and digest laws passing.  Epoch039 reached 1,357,000
cumulative updates at `9.8530821` paired steps/s, `2.25%` below the long-run
baseline.  All science and execution invariants remain exact; the slowdown
is visible but cannot be attributed from the available evidence.

The external M1 route simultaneously established a fresh closure
`ae1f47fbb9f07a886a3c96676144793b0abe646b83336a98e030b269be84f1e7`,
passed its matched smoke with terminal SHA-256
`7d09b3d1571e21bddb0fcf54deec5f2e6c5ebc439b194fc583bc7d57f7165ab8`,
and started concurrent T0/C1 processes on CVD1 with affinity `0-31`.  P0
remains isolated on GPU0, but CPU profiles overlap and GPU1 is occupied by
T0.  Therefore the tracked-route release gate is false.  P0 is 40/48 epochs
complete (`83.33%`); V4 remains pre-terminal and pre-admission.

### P0 epoch040 under sustained external CPU overlap

At 18:01 HKT P0 published `epoch040.json`, SHA-256
`ec060eab89239c084d6bf88c3a1512881fb96e7e2935da48ea6862c4056e3683`.
Root validated epoch000--040 as exactly 44 bodies/88 leaves with every
topology, mode, sidecar and digest law passing.  Epoch040 reached 1,390,925
cumulative updates at `9.4191083` paired steps/s, `6.55%` below the long-run
baseline.  All correctness gates remain exact.

The full epoch overlapped sustained external M1 T0/C1 processes on GPU1,
both with affinity `0-31`; PACD remained on GPU0 but shares part of that CPU
set.  There was no host PSI, OOM, GPU crossover or cross-root descriptor.
The evidence cannot identify causality, but it proves that overlapping CPU
profiles must not be accepted for a future packed PACD cohort.  P0 is 41/48
epochs complete (`85.42%`); V4/PACD-2UP remains gated on terminal and fresh
resource admission.

### P0 epoch041 persistent-throughput warning

At 19:05 HKT P0 published `epoch041.json`, SHA-256
`e470c040e4ee7b490ff0348f6dd9d71da1241408692f540a6bb4d35c46c6389a`.
Root validated epoch000--041 as exactly 45 bodies/90 leaves with all receipt
and science laws passing.  Epoch041 reached 1,424,850 cumulative updates at
`8.8557272` paired steps/s, `12.14%` below baseline.  Together with epoch040,
this is a persistent two-epoch throughput warning.

The external M1 pair remained on GPU1 with broad affinity `0-31`; P0 remained
GPU0's sole owner.  No sustained PSI, OOM, GPU crossover or cross-root
descriptor exists, and no P0 correctness field failed.  The warning does not
authorize intervention, but it makes CPU-disjoint cohort admission mandatory
for PACD-2UP.  P0 is 42/48 epochs complete (`87.50%`); V4 and PACD-2UP remain
pre-terminal, pre-implementation and pre-admission.

### P0 epoch042

At 20:09 HKT P0 published `epoch042.json`, SHA-256
`80a69557cabcb2deb3b531c8ebfdebe2a2a2ce6f0afe9a6b07560ab131621d52`.
Root validated epoch000--042 as exactly 46 bodies/92 leaves with all science,
topology, sidecar and digest laws passing.  Throughput was `8.8858568`
steps/s, `11.84%` below baseline but 0.34% above epoch041.  This is the third
consecutive low-throughput epoch under the unchanged external M1 overlap.

There is still no P0 correctness failure, sustained PSI, OOM, device crossover
or cross-root descriptor.  No intervention or V4 relaxation follows.  P0 is
43/48 epochs complete (`89.58%`), and the next route must enforce disjoint
per-process CPU masks before testing same-GPU PACD packing.

### P0 epoch043 and final-window boundary

At 21:10 HKT P0 published `epoch043.json`, SHA-256
`452902fb61179f149e52a2b535507f5f49f6b73861fc0a5815d9ae9dcebc8485`.
Root validated epoch000--043 as exactly 47 bodies/94 leaves with all science
and artifact laws passing.  Throughput was `9.2498458` steps/s, still 8.23%
below baseline but 4.10% above epoch042.  The external M1 pair and broad CPU
overlap persisted without PSI, OOM, device crossover or root contamination.

P0 is 44/48 epochs complete (`91.67%`).  The remaining epochs 44--47 are the
final checkpoint window; each next event must publish both the epoch receipt
and its exact immutable checkpoint pair.  V4/PACD-2UP remains gated until all
four checkpoints, final-four SWA, manifest and terminal validate.

### P0 epoch044 and first final-window checkpoint

At 22:11:58 HKT P0 published `epoch044.json` and `epoch044.pt`, with body
SHA-256 values
`16ba92799d0e0c9b54fca68256df28fa0e5f2a2673ee0aedc12f8b607364a55f`
and
`f01bd4bc83e72f12ab1430a01dbfdc45b2732ee0506a5239a89761336ebb0139`.
Root validated the exact 49-body/98-leaf topology, modes, canonical sidecars
and body digests. The checkpoint was treated as an opaque 14,056,032-byte
body; no tensor was loaded. No SWA, manifest, terminal, failure or extra
leaf exists.

Epoch044 completed at `9.1908758` steps/s, 8.82% below baseline, while every
science and correctness invariant remained exact. P0 is now 45/48 epochs
complete (`93.75%`) and has entered the immutable final-four checkpoint
window. Epoch045--047 and their checkpoint pairs, final-four SWA, manifest
and terminal remain hard prerequisites. PACD-2UP also remains gated on a
post-terminal packed-versus-isolated smoke and four disjoint CPU masks; the
concurrent external M1 jobs are evidence for packing efficiency, not an
authorization to reuse their overlapping scheduler profile.

### P0 epoch045 and second final-window checkpoint

At 23:14:54--23:15:14 HKT P0 published `epoch045.json` and `epoch045.pt`,
with body SHA-256 values
`d1f5d0e903508d3834ab7305edde4ecde487e2247c4c6afec00eaa8e88a4bd1b`
and
`efaa259bb47b7c1a5503373f5b377c457ba17184059563d78bbd90b933619ed7`.
Root verified the exact 51-body/102-leaf topology, all mode/link/sidecar/body
digest laws and absence of extras. No checkpoint tensor was loaded.

Mean loss improved to `0.3850463758` while throughput fell to `8.9936381`
steps/s; all scientific and execution invariants remained exact. P0 is now
46/48 epochs complete (`95.83%`). Checkpoints 046--047, final-four SWA,
manifest and terminal remain required before any V4 or PACD-2UP capability,
root or code implementation begins.

### External M1 C1 terminal and incomplete resource release

At 23:48:25 HKT the independently owned M1 C1 arm terminalized successfully;
terminal body SHA-256 is
`431df977c9dcc64651395dc4fd949cfe417f1a7b9567c8c574fbfa2c47a7367b`.
The terminal binds 50 epochs, best epoch 46, strict-reloaded checkpoints,
no SWA and zero target optimizer/backward/update. C1 PID `1055233` then
exited naturally. This is only a producer terminal: T0 remains live and the
matched held-out scoring phase has not produced a result.

GPU1 consequently has one remaining M1 owner rather than two, but it is not
released. P0 is also still pre-terminal on GPU0. Therefore the V4/PACD-2UP
resource and lineage gates remain unsatisfied; no capability, root, code
implementation or launch follows from the partial release.

### External M1 T0 terminal and complete GPU1 release

At 2026-09-02 00:03:59--00:04:07 HKT the external matched T0 control
terminalized successfully; terminal body SHA-256 is
`a8270a5eb541b968ee547e8b097a68a74528fc6c9278430d733e7564a3cb2a47`.
It binds 50 epochs, best epoch 44, strict-reloaded checkpoints, no SWA and
zero target optimizer/backward/update. T0 PID `1052875` exited naturally, so
both M1 training arms are complete and physical GPU1 has no compute owner.

No matched held-out score/Phase-3 receipt exists yet, hence no scientific C1
verdict is available. More importantly, an idle GPU1 does not bypass the P0
lineage gate: P0 remains active on GPU0 through epoch045. V4/PACD-2UP code,
capabilities and roots remain prohibited until P0 terminalizes and a fresh
cohort-wide resource/profile audit passes. The idle device was not claimed.

### M1 probe occupancy and P0 epoch046

GPU1's idle interval ended when the independently owned M1 pipeline naturally
started its `probe` arm (PID `1098142`, CVD1, affinity `0-31`). Its root had
only attempt/launch pairs at the audit, with no terminal/failure/score. This
restores the external GPU1 occupancy gate and supplies no scientific verdict.

P0 concurrently published epoch046 and checkpoint046, body SHA-256 values
`50bd62d0949acb4d900f84bf2f9b8e2d128b7022e205e7095b0e81f19085dea0`
and
`1c6e753bd9c92a3b0b97a73edd3b72e25c10a6344c21787db5fba385e5848998`.
Root verified exact 53-body/106-leaf integrity without loading tensors. Mean
loss improved to `0.3848239100`; all correctness laws passed. P0 is 47/48
epochs complete (`97.92%`). Epoch047/checkpoint047, SWA, manifest, terminal,
P0 process release and M1 probe release remain prerequisites for any V4 or
PACD-2UP implementation or launch.

### External M1 final result and GPU1 release

The M1 probe and Phase-3 stages terminalized naturally. Probe terminal body
SHA-256 is
`638dec0aa636e03a43741f57b81d692fbc628a78ba7afcd81b91baebd7b4339d`;
Phase-3 terminal body SHA-256 is
`d0b71443d54a5cd2ab8e7d8f7981c732d62b2f66dfbf314519ebf6fb6309437f`.
GPU1 then returned to no compute owner. The result is scientifically clear:
C1 minus T0 is `-0.005906` on static held-out R2 and `-0.010413` with CDM;
C1 is not promoted. CDM minus static remains modestly positive on the fold
(`+0.015614` for T0 and `+0.011107` for C1) while negative on held-in.

The systems result is separately useful: concurrent arm wall was 7.02 h
versus projected serial 11.72 h, about 1.67x aggregate speedup, with per-arm
wall inflation 1.12x/1.22x and zero GPU-owner crossover. This motivates the
PACD-2UP smoke but does not waive its bitwise parity, disjoint-CPU or
throughput gates. P0 remains active, so an idle GPU1 still does not authorize
V4/PACD-2UP implementation or launch.

## 6. Remaining execution gates

The next actions are deterministic:

1. allow P0 to run without code or process intervention;
2. validate all 48 epoch receipts, checkpoints 44--47, final-four SWA,
   manifest and terminal through held no-follow descriptors;
3. freeze the exact P0 attempt/launch/source/epoch/checkpoint/SWA/manifest/
   terminal body SHA literals;
4. only then implement the frozen shared profile seam and additive V4 package;
5. run every V1/V2/V3/V4 no-data regression and adversarial lifecycle test;
6. descriptor-record the completed independent-route releases, including the
   announced M1 route if it has terminalized, then freshly verify that no
   other in-scope owner occupies either fixed GPU/CPU profile;
7. issue separate one-shot P1/P2 capabilities in fixed taskset/CVD processes;
8. launch the two arms without a result-dependent barrier;
9. after both terminals, bind live producer literals and run the CPU-only
   mixed-lineage matched scorer.

Before Step 2 completes, any V4 production code edit or capability is a
contract violation. Before Step 6 completes, any V4 launch is a resource-
isolation violation.

## 7. Conclusion

The new route design is complete enough for a bounded post-P0 implementation.
The correct current action is controlled waiting plus read-only audit, not a
premature code fork or launch. Completion of the overall experiment still
requires valid P0, P1 and P2 terminals and the matched score; this document
does not convert design completion into a positive result claim.
