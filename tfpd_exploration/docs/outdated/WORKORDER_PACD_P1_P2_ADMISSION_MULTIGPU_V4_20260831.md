# Work Order: PACD P1/P2 Admission and Fixed Multi-GPU Successor V4

Date: 2026-08-31

Status: frozen design boundary; implementation is forbidden before the active
V3 P0 run reaches a valid immutable terminal.

## 1. Purpose

This work order closes two execution-governance gaps without changing PACD
science:

1. P1 and P2 must be impossible to admit until the complete V3 P0 training
   graph has been validated through held no-follow descriptors.
2. P1 and P2 must be able to run in parallel on two pre-registered physical
   GPUs, with a fixed arm-to-device assignment and no automatic device
   selection.

The successor changes admission, provenance, and device binding only. It must
not change the Cell-D model, initial state, PACD paired operator, loss,
optimizer, learning-rate law, source roster, sampler, RNG replay, dropout law,
update count, checkpoint window, or SWA construction.

## 2. Stop-before-edit rule

The active V3 P0 execution closure includes:

```text
tfpd_exploration/src/paired_anchored_calibration_dropout_full_v1/smoke.py
```

Its currently accepted SHA-256 is:

```text
62d4b71291ffaa7175e8cdfd42d14128ee7bea428e8c4a887eb135d8292a7f75
```

The active V3 closure is 48 explicit files with SHA-256:

```text
3356fb124ff4d18a2035bfda8e7eacc28a1642ce3f36b290155a5bcdf7faea2f
```

No implementation file in that closure may be edited before P0 naturally
publishes and root independently validates its terminal. An earlier edit
would make P0 launch/final closure equality fail and would invalidate the
entire training run.

Before that terminal, the only authorized actions under this work order are:

- read-only source review;
- work-order or audit-document edits outside the V3 closure;
- synthetic designs that do not import or mutate the active runtime;
- read-only process, GPU, receipt, mode, sidecar, and digest monitoring.

No P1/P2 capability, attempt, result root, data open, checkpoint load, CUDA
initialization, or training launch is authorized before the P0 gate in
Section 5 passes.

## 3. Fixed identities

### 3.1 Successor identity

```text
CELL   = PACD_P1_P2_ADMISSION_MULTIGPU_V4
SCHEMA = pacd_p1_p2_admission_multigpu_v4
```

The successor exposes only P1 and P2. It must reject P0 as an unknown or
forbidden arm so the historical V3 P0 cannot be rerun under a new closure.

Canonical roots:

```text
P1 = tfpd_exploration/results/paired_anchored_calibration_dropout_full_v4_admission/p1_m4_seed42
P2 = tfpd_exploration/results/paired_anchored_calibration_dropout_full_v4_admission/p2_m10_seed42
```

Historical required P0 root:

```text
tfpd_exploration/results/paired_anchored_calibration_dropout_full_v3/p0_fullfull_seed42
```

### 3.2 Fixed device assignment

Device assignment is pre-registered and is not a result-dependent choice:

| Arm | Physical index | UUID | PCI bus | Process CVD | Torch logical device |
|---|---:|---|---|---|---|
| P1 | 0 | `GPU-ac7388a5-2e98-300a-fdb3-0b67bfd494d9` | `00000000:01:00.0` | `CUDA_VISIBLE_DEVICES=0` | `cuda:0` |
| P2 | 1 | `GPU-2220ed5d-25ea-1839-28d7-ad4dfa5f6c86` | `00000000:03:00.0` | `CUDA_VISIBLE_DEVICES=1` | `cuda:0` |

Both devices are `NVIDIA GeForce RTX 3090`. The reviewed driver at design
time is `535.309.01`. A driver change is not silently accepted; it requires a
new reviewed capability/work-order amendment before launch.

The runtime must never use logical `cuda:1`. A process with one physical GPU
exposed through CVD sees that card as logical `cuda:0`.

There is no fallback or load-balancing rule. P1 may not move to GPU1, P2 may
not move to GPU0, and neither arm may select a device according to observed
intermediate or final performance.

## 4. Allowed implementation scope

After the P0 gate passes, implementation may own only:

```text
tfpd_exploration/src/paired_anchored_calibration_dropout_full_v4_admission/__init__.py
tfpd_exploration/src/paired_anchored_calibration_dropout_full_v4_admission/plan.py
tfpd_exploration/src/paired_anchored_calibration_dropout_full_v4_admission/p0_admission.py
tfpd_exploration/src/paired_anchored_calibration_dropout_full_v4_admission/smoke.py
tfpd_exploration/scripts/run_pacd_full_training_v4_admission.py
tfpd_exploration/tests/test_pacd_full_training_v4_admission.py
```

One narrow shared-file edit is authorized only after P0 terminal validation:

```text
tfpd_exploration/src/paired_anchored_calibration_dropout_full_v1/smoke.py
```

That edit may add only a frozen, backward-compatible device-profile seam and
thread it through the existing single lifecycle. It may not copy, fork, or
otherwise alter the 48-epoch loop or the PACD operator.

No edit is authorized to:

```text
tfpd_exploration/src/paired_anchored_calibration_dropout_v1/core.py
tfpd_exploration/src/paired_anchored_calibration_dropout_full_v1/runner.py
tfpd_exploration/src/paired_anchored_calibration_dropout_full_v2/**
tfpd_exploration/src/paired_anchored_calibration_dropout_full_v3/**
any V1/V2/V3 result root
any target-data or scoring root
```

The currently frozen scientific code anchors are:

```text
PACD paired core
9331288cf42d7c7924f6135f3ec8e3659f0ea19b70c8fad84448b01773c8172d

PACD shared epoch runner
adb6c471e0e28eb544e9c7bc3f9e4a9823e2803ef33d6de5cfa5545c917a54f0

V3 plan
0368fce6b71f42cbe45aa70e56090cdd24c985d660726c8fbdb7ebbf8f9039c3

V3 profile wrapper
c072bf7715db61cf38e7f7c2cb9e98765daf26d977ac71e736ebdd6c56f8f7ad
```

## 5. Mandatory P0 held-graph gate

### 5.1 Descriptor discipline

The P0 validator must be standard-library-only. It must not import Torch,
load a tensor, open source/target data, initialize CUDA, or materialize a
model.

It must:

- require `O_NOFOLLOW` support;
- open the canonical P0 directory with a held directory FD;
- read every body and sidecar relative to that held FD;
- require regular, non-symlink, mode-`0444` files;
- verify body SHA-256 and the exact canonical sidecar line;
- hold and recheck the parent/named directory device and inode identities;
- reject every extra, missing, renamed, replaced, mutable, or symlink leaf.

The exact successful P0 topology contains 58 bodies and 116 body/sidecar
leaves:

```text
attempt.json
launch.json
source_authority.json
epoch000.json ... epoch047.json
epoch044.pt ... epoch047.pt
swa_final4.pt
manifest.json
terminal.json
```

`failure.json` and its sidecar must be absent.

### 5.2 P0 attempt and source contract

The validator must require:

- schema `pacd_matched_full_training_v3_attempt`;
- status `ATTEMPT_PUBLISHED`;
- `attempt["arm"] == "p0"`;
- `attempt["arm_identity"]["short_m"] == 30` and
  `attempt["arm_identity"]["root"]` equals the exact V3 P0 root-relative
  literal;
- seed 42, batch size 32, workers 0;
- 48 epochs;
- 33,925 steps per epoch;
- 1,628,400 total optimizer steps;
- exact source-only facts and `target_access=false`;
- exact 27-session ordered source roster;
- empty validation and test lists;
- exact source manifest, normalizer, T4, initial-state and sampler authority;
- `SessionBatchSampler`, shuffle true, seed 42, and stable window/batch-order
  digests.

Launch must link the exact attempt and source-authority body digests.

### 5.3 Epoch and checkpoint contract

All 48 epoch receipt descriptors and files must be ordered exactly from 000 to
047. Each must prove:

- epoch index exact;
- 33,925 optimizer steps;
- cumulative optimizer steps `(epoch + 1) * 33,925`;
- finite model and Adam state;
- zero P0 prediction mismatch;
- zero P0 identity mismatch;
- zero RNG violation;
- zero prefix mutation;
- zero finiteness violation;
- `gradient_coverage.zero_decoder_steps == 0`;
- `gradient_coverage.positive_decoder_steps > 0` and
  `gradient_coverage.positive_encoder_steps > 0`;
- `gradient_coverage.accepted_zero_encoder_steps` equals
  `gradient_coverage.zero_encoder_steps` and
  `gradient_coverage.zero_encoder_reason ==
  "all_units_dropped_valid_zero"`;
- exact sampler-order evidence equal to source authority;
- the required sentinel topology and positive encoder/decoder coverage.

The four checkpoint bodies must be exactly epochs 44, 45, 46 and 47, in that
order, with terminal and manifest links matching their body digests. The
validator must not deserialize these checkpoint tensors.

### 5.4 SWA, terminal, and lower lineage

The terminal must require:

- schema `pacd_matched_full_training_v3_terminal`;
- status `PACD_FULL_TRAINING_COMPLETE`;
- exact P0 identity;
- 48 epoch descriptors, four checkpoint descriptors, SWA and manifest;
- progress counters 48/4/true;
- target access false;
- launch and final source-closure payloads identical;
- closure SHA exactly
  `3356fb124ff4d18a2035bfda8e7eacc28a1642ce3f36b290155a5bcdf7faea2f`;
- exact inherited V2 failure lineage, including V2 failure body SHA
  `c7f1a893a7c65dbb46b8493c9f8080ec12af5b3c7e97e606cb7d8aed005ea8b0`.

The final-four SWA proof must require:

- component epochs `[44,45,46,47]` in exact order;
- strict fresh load;
- `terminal["swa"]["strict_loaded_state_sha256"]` equals both
  `terminal["swa"]["proof"]["state_before_sha256"]` and
  `terminal["swa"]["proof"]["state_after_sha256"]`;
- eval mode and no-grad;
- repeated-forward bitwise equality;
- finite outputs and parameters;
- zero dynamic dropout calls;
- state digest unchanged before/after the proof;
- manifest checkpoint and SWA links exact.

The returned admission witness must bind at least the P0 root identity,
attempt, launch, source authority, terminal, manifest, SWA, four checkpoint
body digests, final V3 closure, and the complete lower V2 witness.

## 6. Shared frozen device-profile seam

After Section 5 passes, the shared lifecycle may gain a frozen structure with
the following exact semantics:

```text
identity
physical_index
expected_uuid
expected_name
expected_driver
expected_pci_bus_id
cuda_visible_devices
logical_device = cuda:0
```

The shared file must retain two explicit controller paths:

- a legacy GPU0 controller for V1/V2/V3 that continues calling the current
  GPU0 helpers and preserves their receipt payload and admission semantics;
- an extended fixed-device controller used only by V4, with the additional
  PCI/profile/PID evidence specified here.

Existing V1/V2/V3 callers that do not provide a device mapping must remain on
the legacy controller. They must not silently acquire V4's expanded payload
or PCI/PID schema merely because the shared source file changed.

The V4 profile maps P1 and P2 to the fixed profiles in Section 3.2.

V4 device validation is a staged state machine. Static profile fields are
exact-compared throughout, but the compute-process list is deliberately not
identical before and after CUDA context creation.

The shared issuer/executor must validate device state at all of these
boundaries:

1. before capability issuance;
2. immediately before attempt publication;
3. immediately after immutable attempt and before runtime import/CUDA init;
4. immediately after explicit CUDA context materialization;
5. before every terminal or failure publication.

Each check may query only the selected physical GPU. The static profile fields
must always require:

- exact CVD string;
- exact physical index, UUID, name, driver, and PCI bus ID;
- the other PACD arm's root, capability and device are never accepted;
- no environment, root parent, named-root, profile or runtime-factory drift.

The stage-specific process contract is:

| Stage | Required selected-GPU compute state |
|---|---|
| capability issuance and pre-attempt | no compute application; `idle=true` |
| post-attempt but before context | no compute application; `idle=true` |
| after explicit context materialization | exactly one compute PID and it equals `os.getpid()` |
| terminal or failure with `progress.cuda_bound=true` | exactly one compute PID and it equals `os.getpid()` |
| failure with `progress.cuda_bound=false` | no compute application |

After attempt publication, the device binder must import Torch, recheck CVD,
require exactly one visible logical CUDA device, set logical device zero, and
explicitly materialize a minimal CUDA context before any source data or
checkpoint is opened. Only after context materialization may it require the
selected physical GPU's sole compute PID to equal the current process. It
then resets peak-memory statistics and enters the unchanged source/model
runtime. Context creation time is recorded as hardware-admission behavior; it
is not an optimizer, model, data-order or RNG change.

The extended physical query must request exactly one row with five fields:

```text
index,uuid,name,driver_version,pci.bus_id
```

It must use `nvidia-smi -i <profile.physical_index>` and must not enumerate the
other GPU. CVD plus one visible logical device, the exact selected physical
profile, and the selected-GPU PID observation form the binding proof; the
validator does not claim to query the unselected GPU.

The failure receipt must record `device_final`, `device_recheck_stage`, the
frozen device profile, `progress.cuda_bound`, and any typed device-recheck
error. A recheck failure must not be hidden by replacing it with the original
runtime error.

Production V4 uses an internal typed device observer owned by the execution
profile. Synthetic tests may use a typed private observer with a fake process
runner/Torch witness. The public CLI must not accept a callback, UUID, device
profile, boolean authorization or observer substitution. Capability issuance
must bind the observer kind together with the runtime factory and profile.

The opaque one-shot capability must bind root, arm, device profile, complete
P0 witness, current successor closure, runtime-factory identity, and parent
directory identities. Cross-arm, cross-root, cross-device, reused, forged or
stale capabilities must fail closed.

### 6.1 Exact V4 predecessor and receipt placement

The implementation must not invent field placement after training begins.
The following shapes are frozen for both V4 arms.

The profile predecessor validator returns one exact composite mapping:

```json
{
  "schema": "pacd_p1_p2_admission_multigpu_v4_predecessor_v1",
  "v2_failure": {"...": "the exact existing V2 held-failure witness"},
  "required_v3_p0": {
    "schema": "pacd_v3_p0_admission_witness_v1",
    "root_relative": "tfpd_exploration/results/paired_anchored_calibration_dropout_full_v3/p0_fullfull_seed42",
    "root_identity": {
      "parent_device": 0,
      "parent_inode": 0,
      "directory_device": 0,
      "directory_inode": 0,
      "name": "p0_fullfull_seed42"
    },
    "attempt": {"name": "attempt.json", "sha256": "<64hex>"},
    "launch": {"name": "launch.json", "sha256": "<64hex>"},
    "source_authority": {"name": "source_authority.json", "sha256": "<64hex>"},
    "epochs": [{"name": "epoch000.json", "sha256": "<64hex>"}],
    "checkpoints": [{"name": "epoch044.pt", "sha256": "<64hex>", "epoch": 44}],
    "swa": {"name": "swa_final4.pt", "sha256": "<64hex>"},
    "manifest": {"name": "manifest.json", "sha256": "<64hex>"},
    "terminal": {"name": "terminal.json", "sha256": "<64hex>"},
    "historical_v3_closure_sha256": "3356fb124ff4d18a2035bfda8e7eacc28a1642ce3f36b290155a5bcdf7faea2f",
    "v2_failure_sha256": "c7f1a893a7c65dbb46b8493c9f8080ec12af5b3c7e97e606cb7d8aed005ea8b0"
  }
}
```

The illustrative numeric inode values above are placeholders for observed
integers, not literals. The `epochs` array must contain exactly 48 ordered
descriptors and `checkpoints` exactly four ordered descriptors for epochs
44--47. No other keys are accepted in either mapping. The nested
`v2_failure` is exact-compared to the V3 terminal's own predecessor and to a
fresh call of the reviewed V2-failure validator.

The V4 attempt keeps the shared attempt fields and adds no parallel aliases.
Its exact route-specific fields are:

```json
{
  "schema": "pacd_p1_p2_admission_multigpu_v4_attempt",
  "cell": "PACD_P1_P2_ADMISSION_MULTIGPU_V4",
  "arm": "p1-or-p2",
  "arm_identity": {"short_m": 4, "root": "<exact arm root>"},
  "predecessor": {"...": "the exact composite mapping above"},
  "device": {
    "profile": {
      "identity": "<v4-p1-gpu0 or v4-p2-gpu1>",
      "physical_index": "<0 or 1>",
      "expected_uuid": "<literal>",
      "expected_name": "NVIDIA GeForce RTX 3090",
      "expected_driver": "535.309.01",
      "expected_pci_bus_id": "<literal>",
      "cuda_visible_devices": "<0 or 1>",
      "logical_device": "cuda:0"
    },
    "observer_kind": "extended-fixed-device-production-v1",
    "issuance_idle": {"...": "idle attestation"},
    "pre_attempt_idle": {"...": "identical static profile and empty process list"}
  }
}
```

For P2, `arm_identity.short_m` is 10. The ordinary fixed budget, closure,
review-evidence, ordering and target-access fields remain exactly those of the
shared attempt codec.

The V4 launch keeps its exact attempt/source-authority links and records:

```json
{
  "schema": "pacd_p1_p2_admission_multigpu_v4_launch",
  "device": {
    "profile": {"...": "exact attempt device.profile"},
    "observer_kind": "extended-fixed-device-production-v1",
    "post_attempt_idle": {"...": "empty selected-GPU process list"},
    "context_bound": {
      "stage": "active_current_pid",
      "visible_device_count": 1,
      "current_logical_device": 0,
      "logical_device": "cuda:0",
      "cuda_initialized": true,
      "current_pid": 0,
      "selected_gpu_compute_pids": [0]
    }
  }
}
```

The numeric PID values are observed runtime integers. `current_pid` must equal
the sole item in `selected_gpu_compute_pids` and `os.getpid()`.

`source_authority.json` retains the existing shared codec. Its
`attempt_sha256` is the only route identity link; it does not duplicate device
or predecessor evidence.

The V4 terminal must use:

```json
{
  "schema": "pacd_p1_p2_admission_multigpu_v4_terminal",
  "status": "PACD_FULL_TRAINING_COMPLETE",
  "predecessor": {"...": "freshly revalidated exact composite mapping"},
  "device": {
    "profile": {"...": "exact attempt profile"},
    "observer_kind": "extended-fixed-device-production-v1",
    "issuance_idle": {"...": "attempt evidence"},
    "pre_attempt_idle": {"...": "attempt evidence"},
    "post_attempt_idle": {"...": "launch evidence"},
    "context_bound": {"...": "launch evidence"},
    "final": {
      "stage": "active_current_pid",
      "current_pid": 0,
      "selected_gpu_compute_pids": [0]
    },
    "peak_allocated": 0,
    "peak_reserved": 0,
    "tf32": {"matmul_allow_tf32": false, "cudnn_allow_tf32": true}
  }
}
```

The terminal retains the shared exact attempt, launch, source-authority, 48
epoch, four checkpoint, SWA, manifest, sampler-order, closure, progress and
target-access fields. The final PID must again be exactly `os.getpid()` and
the selected GPU's sole compute PID. Launch and final V4 closure payloads must
be identical.

The V4 failure receipt uses:

```json
{
  "schema": "pacd_p1_p2_admission_multigpu_v4_failure",
  "status": "CELL_FAILED",
  "predecessor": {"...": "best safe fresh revalidation or typed error"},
  "device": {
    "profile": {"...": "exact frozen profile"},
    "observer_kind": "extended-fixed-device-production-v1",
    "device_recheck_stage": "idle_pre_cuda_or_active_current_pid",
    "final": {"...": "stage-appropriate attestation or typed error"},
    "device_recheck_error": null
  },
  "progress": {"cuda_bound": false}
}
```

If device revalidation itself fails, `device_recheck_error` contains the typed
class and message while the original runtime failure remains separately
preserved under the shared `failure` field. A failure before context requires
the idle state; a failure after `progress.cuda_bound=true` requires the sole
current-PID state. The failure codec must not claim a P0 witness or hardware
fact that could not be safely revalidated.

The extended attestation bodies may contain only these exact keys:

```text
stage, physical_index, uuid, name, driver_version, pci_bus_id,
cuda_visible_devices, compute_processes, idle, preflight_uses_torch
```

For an active context, the logical/PID keys shown above are added under the
separate `context_bound` or `final` mapping. Process entries contain exactly
`pid`, `process_name`, and `used_memory`; `used_memory` is descriptive and is
not exact-compared between context creation and terminal.

## 7. Numerical contract: no change

P1 and P2 must reuse exactly one shared executor and one shared epoch runner.
The following values are inherited, not reimplemented:

| Field | Required value |
|---|---|
| model | sealed Cell-D |
| initial state | exact existing canonical state |
| P1 views | M30 anchor + chronological M4 short |
| P2 views | M30 anchor + chronological M10 short |
| task loss | `0.5 * anchor MSE + 0.5 * short MSE` |
| paired stochastic law | exact Python/Torch RNG replay and equal unit mask |
| zero-encoder policy | accept only when the paired common mask retains zero units |
| optimizer | inherited Adam contract |
| LR | inherited step-indexed 48-epoch law |
| seed | 42 |
| batch | 32 |
| workers | 0 |
| steps | 48 x 33,925 |
| checkpoints | epochs 44, 45, 46, 47 |
| SWA | inherited final-four arithmetic and strict proof |
| target access | false throughout training |

Cross-GPU weights are not required to be bitwise identical. The scientific
comparison is justified by the pre-registered identical algorithm,
initialization, seed, source order, optimizer, update count and matched score
contract. Device assignment is disclosed as hardware evidence and is never a
selection variable.

## 8. Tests required before any V4 launch

All focused tests are no-data/no-checkpoint/no-CUDA unless explicitly using a
mocked process runner. They must cover:

1. exact valid synthetic 58-body P0 graph admits both P1 and P2;
2. missing/extra/failure/symlink/mode/body/sidecar/root-replacement P0 graphs
   reject;
3. attempt, source authority, sampler, epoch order/count/steps/cumulative
   steps, P0 equality, RNG, gradient, finite and closure drift reject;
4. checkpoint 44--47 order/link drift and every SWA/manifest proof drift
   reject;
5. P0 validation occurs before capability issuance, before attempt, and before
   terminal/failure;
6. V4 exposes only P1/P2 and cannot create or reuse a P0 root;
7. V1/V2/V3 default profile remains on the legacy GPU0 controller and retains
   the prior receipt/device payload schema;
8. P1→GPU0 and P2→GPU1 mapping is frozen and immutable;
9. CVD/index/UUID/name/driver/PCI/logical-device drift rejects;
10. idle→context-materialized current-PID→terminal state transitions pass,
    while an unexpected PID, multiple PIDs, an occupied preflight, and a
    progress/state mismatch reject;
11. pre-CUDA and post-CUDA failure branches record the correct staged device
    evidence and preserve the original error plus any recheck error;
12. cross-arm/root/device/factory/observer capability substitution and reuse
    reject;
13. a typed private test observer can drive both profiles through the one
    shared synthetic lifecycle, while no public CLI observer injection exists;
14. both profiles can execute the one shared synthetic lifecycle to a valid
    terminal without copied operator/runner code;
15. failure lifecycle is exclusive, honest and descriptor-linked;
16. `python -S` dry import does not import Torch or probe GPU/data/results;
17. current P0 closure remains reconstructible from its immutable receipt,
    while V4 reconstructs a separate new closure;
18. exact source-tree checks prove the paired core and shared epoch runner
    hashes remain the anchors listed in Section 4.

The existing PACD V1/V2/V3 and full-training regression suites must be run.
Historical closure-reconstruction tests may correctly fail current-byte
equality after the authorized post-P0 shared seam, but immutable P0 receipt
validation must continue to pass against the historical closure it records.
Such expected historical/current distinction must be explicit, never masked.

## 9. Launch gate and resource isolation

Launch order is fixed:

1. P0 V3 natural terminal;
2. independent descriptor audit of Section 5;
3. post-P0 code implementation and no-CUDA audit;
4. root revalidates new closure, exact device profiles and both fresh roots;
5. issue P1 capability in a process with CVD=0;
6. issue P2 capability in a separate process with CVD=1;
7. each process publishes its own immutable attempt before its own
   Torch/data/checkpoint/CUDA boundary;
8. each arm may enter its authorized runtime immediately after its own
   post-attempt admission and device checks pass; there is no cross-process
   attempt barrier or peer-wait loop;
9. read-only event monitoring; no intervention, retry, resume or root reuse.

The absence of a peer barrier is intentional. Both arm/device assignments and
both scientific contracts are fixed before launch, so one arm starting a few
seconds earlier creates no selection freedom. No result claim is made until
both valid terminals exist and the mixed-lineage matched scorer completes.

Recommended scheduler-only CPU partition after both launches:

```text
P1 -> logical CPUs 0-7,16-23
P2 -> logical CPUs 8-15,24-31
```

This partition changes no training semantics. OMP/MKL/OpenBLAS/NUMEXPR remain
one thread and DataLoader workers remain zero.

If either assigned GPU is occupied at preflight, that arm waits. It does not
move to the other device. A failure creates an immutable failure graph; retry
requires a new successor identity and new root.

## 10. Scoring consequence

The frozen PACD matched-score V1 currently expects all three producers to have
V3 roots, V3 schemas and direct V2-failure predecessor payloads. It must not be
silently relaxed to accept V4.

After valid P1/P2 terminals exist, a separate additive score successor must
bind the mixed producer family explicitly:

```text
P0 -> V3 historical terminal/closure/device
P1 -> V4 admission terminal/closure/GPU0 profile/P0 witness
P2 -> V4 admission terminal/closure/GPU1 profile/P0 witness
```

It must retain the already frozen same-input 756-row matched evaluation,
historical T0/C1/sealed-Cell-D comparators, activity-isolation regime,
bootstrap, gates, CPU-only execution and atomic committed bundle. No scoring
is authorized from guessed or incomplete producer literals.

## 11. Success condition

This work order is complete only when:

- P0 V3 has a valid immutable terminal;
- V4 admission/device code passes the required audits without changing the
  paired science anchors;
- P1 and P2 each produce a valid 48-epoch terminal on their fixed physical
  devices with exact P0 witness revalidation;
- the mixed-lineage score successor produces the matched comparison; and
- the final report distinguishes scientific PACD effects from admission,
  hardware, scheduling and historical-control effects.

Until then, this is an execution design, not a positive PACD result.
