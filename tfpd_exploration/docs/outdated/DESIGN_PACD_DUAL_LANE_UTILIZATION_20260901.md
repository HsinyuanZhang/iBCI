# Design: PACD Two-Up Packed Lanes

Date: 2026-09-01

Status: review-only throughput successor.  This document is outside every
active execution closure.  It does not authorize code edits, capability
issuance, CUDA initialization, result-root creation, process control, or
launch.  It supersedes the earlier one-process-per-GPU recommendation in this
same file.

## 1. Corrected decision

The earlier PACD-DL proposal reduced total wall time but did not solve low
utilization inside each GPU.  Live evidence now supports testing a controlled
two-process packing design:

```text
physical GPU0
  lane 0A: local P0 matched control, seed 42
  lane 0B: P1 / M4, seed 42

physical GPU1
  lane 1A: local P0 matched control, seed 42
  lane 1B: P2 / M10, seed 42
```

The execution design is **PACD-2UP**.  Each GPU runs one contemporaneous local
control and one intervention.  This is scientifically stronger than placing
two unrelated jobs together: every intervention is compared with a control
that experienced the same device, time window, host load, software closure,
and scheduler regime.

The already completed sealed P0 remains the mandatory historical predecessor.
The two local P0 replays are new matched comparators; they do not replace or
rewrite the sealed P0 result.

## 2. Evidence and expected value

The active PACD P0 process uses about 1.17 GiB of a 24 GiB GPU and often shows
only 10--25% instantaneous utilization.  The contemporaneous M1 route provides
a useful systems measurement:

| GPU1 state | Used memory | Observed utilization |
|---|---:|---:|
| one M1 process | about 1.2 GiB | about 10--25% |
| two M1 processes | about 2.35 GiB | up to 56% |

At the same sample, PACD alone used about 1.17 GiB and GPU0 showed 13% while
the packed M1 GPU showed 56%.  This demonstrates real unused capacity and
makes two-up packing worth a formal parity/throughput smoke.

This evidence does not yet prove that PACD-2UP is safe or faster.  P0
epoch039, which overlapped external M1 preparation, was 2.25% below its
long-run throughput baseline.  There was no PSI/OOM evidence and no causal
attribution is possible.  The correct response is a controlled smoke with
disjoint CPUs and matched roots, not unrestricted process packing.

The next completed P0 interval strengthened this constraint.  Epoch040 ran
for about 51 minutes while both external M1 processes were active with broad
affinity `0-31`; P0 throughput was `9.4191` steps/s, `6.55%` below its
long-run baseline.  Host PSI was still zero and causality remains unproved,
but the magnitude is large enough that disjoint per-process CPU masks are a
hard PACD-2UP admission condition rather than a scheduler preference.

Epoch041 then completed at `8.8557` steps/s, `12.14%` below baseline and
5.98% below epoch040, while the same broad-affinity M1 pair remained active.
This creates two consecutive PACD epochs below 9.5 steps/s.  It still does not
identify causality, but it definitively rules out treating CPU affinity as an
informal recommendation: any overlap between the four PACD-2UP masks is a
fail-closed admission error.

Epoch042 remained low at `8.8859` steps/s (`11.84%` below baseline) while the
same external broad-affinity pair remained active, although it improved 0.34%
from epoch041.  Three consecutive low epochs make the resource concern
persistent.  They do not invalidate same-GPU packing; they validate the
PACD-2UP distinction between **shared GPU with disjoint CPUs** and the current
unrestricted **shared GPU plus overlapping CPUs** condition.

Epoch043 recovered to `9.2498` steps/s, 4.10% above epoch042 but still 8.23%
below baseline.  The four-epoch low-throughput sequence strengthens the need
for a packed-versus-isolated speed gate; the partial recovery also confirms
that a single instantaneous utilization sample is not an adequate scheduler
decision rule.

Epoch044 then completed at `9.1909` steps/s, 8.82% below baseline, creating a
five-epoch sequence below 9.5. During the matching audit the two external M1
processes used only about 2.35 GiB on GPU1 yet produced roughly 58--60% GPU
utilization, while the single PACD process used about 1.17 GiB and showed
about 19%. This is the strongest current evidence for two-up packing, but it
arrives together with broad M1 CPU affinity `0-31` overlapping PACD's
`0-3,16-19`. PACD-2UP therefore treats GPU sharing and CPU partitioning as
one indivisible intervention: two allowlisted processes per GPU are expected,
and any overlap among the four exact CPU masks is fail-closed.

Epoch045 added a sixth low-throughput observation: `8.9936` steps/s, 10.77%
below baseline, while the external two-process GPU1 workload continued with
the same broad overlapping CPU mask. At the same time PACD mean loss improved
to a new local minimum (`0.38505`) and every correctness gate passed. This
separates science health from scheduler efficiency: PACD-2UP should preserve
the exact science contract while changing only process packing and CPU
partitioning, and must be judged by aggregate throughput plus bitwise parity,
not by utilization alone.

The external M1 C1 process later exited naturally after its successful
50-epoch terminal, leaving only T0 on GPU1. The immediate descriptive sample
moved from two-process GPU1 occupancy (about 2.35 GiB, commonly 43--60%
utilization) to one process (about 1.20 GiB, 24%). This is not a synchronized
throughput experiment and is not used as an acceptance estimate, but it
reinforces the need for the planned isolated-versus-packed PACD smoke rather
than assuming that a single small model can saturate a card.

The matched T0 process subsequently terminalized and exited as well, returning
GPU1 to 0%/23 MiB with no compute owner. Thus the external observation has a
complete natural sequence—two processes, one process, then zero—without any
PACD intervention. It remains descriptive rather than a controlled speed
study. The empty GPU1 is not launch authority: PACD-2UP still requires the P0
terminal, additive cohort implementation, fresh four-lane resource binding
and isolated-versus-packed parity/throughput smoke.

The external M1 pipeline subsequently entered a probe arm and again occupied
GPU1 with one process, confirming that instantaneous idleness is not a stable
admission signal. PACD-2UP therefore binds a complete cohort and rechecks all
owners immediately before publication rather than opportunistically claiming
a temporarily idle card. P0 epoch046 meanwhile reached `8.8189` steps/s while
its loss improved to `0.38482`; this again separates scheduler throughput
from scientific convergence.

The completed external M1 pair supplies a useful end-to-end scheduling
measurement. Concurrent wall time was 7.02 h versus a 11.72 h serial
projection, equivalent to about `1.67x` aggregate throughput and about 4.7 h
saved. Per-arm wall inflation was only 1.12x for T0 and 1.22x for C1, implying
roughly 89% and 82% isolated-throughput retention; both exceed the proposed
PACD per-lane 60% floor. Zero GPU-owner crossover was observed. This is strong
support for two-up execution, but it is not a PACD parity trial and used broad
overlapping CPU masks. PACD must still pass its own bitwise isolated-versus-
packed smoke with the four disjoint masks before any full cohort launch.

## 3. Frozen science inside every lane

All four lanes retain the existing PACD training contract:

- seed 42;
- logical batch size 32;
- `num_workers=0`;
- 48 epochs and 33,925 optimizer steps per epoch;
- inherited LR and Adam laws;
- identical SessionBatchSampler order and recorded window/batch digests;
- exact Python and Torch RNG replay;
- unchanged dynamic whole-unit dropout;
- unchanged model, paired loss, gradient rules, checkpoints, final-four SWA,
  manifest, and terminal evidence;
- no target data and no target update.

Packing changes only the scheduler cohort.  There is no model sharing,
gradient sharing, distributed data parallelism, inter-process tensor exchange,
shared optimizer, or shared result root.

## 4. Four disjoint CPU lanes

The four processes use disjoint physical-core/sibling groups:

| Lane | Scientific arm | GPU | CVD | Exact logical CPU affinity |
|---|---|---:|---:|---|
| 0A | local P0-G0 control | 0 | `0` | `0-3,16-19` |
| 0B | P1 / M4 | 0 | `0` | `4-7,20-23` |
| 1A | local P0-G1 control | 1 | `1` | `8-11,24-27` |
| 1B | P2 / M10 | 1 | `1` | `12-15,28-31` |

OMP, MKL, OpenBLAS and NumExpr remain one thread per process.  Affinity cannot
expand, migrate, or borrow after attempt publication.

The current frozen V4 profile defines only P1/P2, so PACD-2UP requires an
additive successor profile and work order.  The frozen V4 work order must not
be edited retroactively.

## 5. Cohort-bound admission

Ordinary V4 admission rejects every existing compute PID.  PACD-2UP instead
needs one transactional cohort authority that pre-registers exactly four
roots, four lane identities, two devices, four CPU masks, and the permitted
same-device peer relation.

For GPU0, the only allowed compute owners are lanes 0A and 0B.  For GPU1, the
only allowed compute owners are lanes 1A and 1B.  A missing peer is permitted
during staggered preparation; an unknown third PID, wrong root, wrong closure,
wrong CVD, wrong affinity, or cross-device owner fails closed.

The cohort authority must bind:

- the accepted complete sealed P0 predecessor;
- one implementation closure shared by all four lanes;
- the exact four prospective roots and parent inode identities;
- the exact local-control/intervention pairing;
- fixed GPU UUID and PCI identity;
- CPU masks and one-thread environment;
- maximum allowed per-process and per-GPU memory;
- start-order independence and no result-dependent cancellation;
- terminal/failure semantics for each lane separately;
- a cohort terminal that never invents a missing lane result.

Capabilities are one-shot and lane-specific.  A capability can admit its
pre-registered peer, but no general foreign-process exception exists.

## 6. Mandatory two-up smoke

PACD-2UP cannot go directly to 48 epochs.  It first runs an isolated reference
smoke and a packed smoke with the same initial state, sampler, RNG and fixed
step coordinates.

### 6.1 Scientific parity

For each lane, packed versus isolated execution must prove:

- identical ordered batch IDs and batch tensor digests;
- identical Python/Torch RNG before/after digests;
- identical dropout probability and unit mask;
- identical P0 paired prediction and identity evidence;
- identical loss and all recorded gradient evidence;
- bitwise-identical materialized model and Adam states after each audited
  update;
- identical inactive lazy-parameter topology;
- no target access and no extra CUDA RNG call;
- no receipt, root, closure, CVD, GPU UUID or affinity drift.

Scheduler timing is allowed to differ; scientific bytes are not.

### 6.2 Resource and speed gate

The packed smoke must also satisfy all of the following:

- no OOM, CUDA error, PSI breach or third compute owner;
- combined peak memory below 8 GiB per GPU;
- combined packed throughput at least 1.50 times isolated single-process
  throughput on the same GPU;
- each lane retains at least 60% of isolated per-process throughput;
- median GPU utilization increases by at least 15 percentage points;
- no lane has zero progress for more than a pre-registered observation window;
- results are unchanged if control and intervention start order is reversed.

These are admission gates, not scientific selection rules.  If any gate fails,
PACD-2UP stops before full attempt publication and the system returns to the
ordinary one-process-per-GPU PACD-DL plan.  There is no mid-training fallback,
automatic process kill, or closure-changing restart.

## 7. Full-run interpretation

The primary comparisons become:

```text
P1 / M4  - local P0-G0
P2 / M10 - local P0-G1
```

Secondary consistency checks are:

```text
local P0-G0 - sealed historical P0
local P0-G1 - sealed historical P0
local P0-G0 - local P0-G1
```

If the local controls reproduce the sealed P0 within the pre-registered
tolerance, the intervention comparison gains stronger contemporaneous
attribution.  If local controls disagree materially, P1/P2 results remain
descriptive but cannot be attributed only to PACD arm content; scheduler or
device context is then an observed interaction.

The local controls must never be averaged away before these checks.  Their
purpose is to measure the cost of packing and device/time context.

## 8. Optional input-pipeline acceleration

After PACD-2UP parity is established, a separate successor may additionally
test:

1. an ordered pinned-memory double buffer that transfers the next selected
   batch while the current batch computes;
2. per-session caching of immutable calibration/input tensors on the assigned
   GPU, keyed by exact source-authority digests.

This layer cannot add DataLoader workers, overlap optimizer steps, execute the
anchor and short stochastic forwards concurrently, or cache any
parameter-dependent identity output.  It needs its own bitwise parity and
at-least-15% throughput gate.

## 9. What remains a new scientific experiment

The following are not pure packing and remain excluded from PACD-2UP:

- batch 64/128/256;
- fewer updates or gradient accumulation to compensate for larger batches;
- simultaneous optimizer steps inside one model;
- shared model/optimizer state across processes;
- mixed precision, TF32, `torch.compile`, or CUDA graphs;
- automatic device selection or migration;
- arbitrary third-party same-GPU processes;
- padding M4/M10 to M30 to fuse stochastic forwards.

Larger batch changes Adam update count, gradient noise, dynamic-dropout
sampling frequency and exposure.  It requires a separately re-anchored
matched matrix and cannot be presented as a runtime-only change.

## 10. Execution order

```text
active sealed P0 terminal
    -> exact 58-body/116-leaf and resource-release audit
    -> wait for all external M1/GPU work to release naturally
    -> implement/test the frozen V4 device profile seam
    -> implement additive PACD-2UP cohort profile
    -> isolated versus packed parity/throughput smoke
    -> if all gates pass, reserve four fresh full roots atomically
    -> launch the four fixed lanes without result-dependent barriers
    -> read-only monitoring; no process intervention
    -> exact four-lane and cohort terminal audit
    -> CPU-only matched scoring with local-control contrasts
```

No implementation begins before the active P0 terminal.  No packed smoke
begins while the current external M1 T0/C1 processes own GPU1 or overlapping
CPU sets.

## 11. Decision

Adopt PACD-2UP as the preferred utilization candidate.  The user's objection
is correct: one process per GPU leaves substantial capacity unused.  Use that
capacity for a scientifically useful contemporaneous P0 control next to each
intervention, with strict CPU partitioning, cohort-bound peers and bitwise
parity gates.  Retain ordinary PACD-DL as the pre-attempt fallback if packed
execution fails parity or combined-throughput gates.  Do not increase batch or
alter numerical operators under the label of utilization improvement.
