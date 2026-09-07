# Audit: PACD V4 Shared-Lifecycle and Fixed-Device Seam

Date: 2026-08-31

Status: read-only pre-implementation audit. The active PACD V3 P0 execution
closure is untouched. No V4 code, capability, result root, data access,
checkpoint load, Torch import, CUDA context, or launch is authorized by this
document.

## 1. Decision

There is no scientific or model-interface blocker to the V4 P1/P2 successor.
The existing shared 48-epoch executor, epoch runner, paired PACD operator,
loader, optimizer, checkpoint and SWA paths can be reused.

There is one material lifecycle seam that must be implemented after the P0
terminal gate: device handling cannot be made correct by replacing the
hard-coded GPU0 index with a variable. The current executor combines Torch
import, CUDA binding and execution-stack loading inside one opaque runtime
factory call. V4 requires those states to be separated so receipts can state
truthfully whether CUDA was bound when a later import or stack-load failure
occurs.

The correct post-P0 implementation is therefore:

```text
unchanged shared 48-epoch executor
    + frozen legacy GPU0 controller for V1/V2/V3
    + route-owned fixed-device controller for V4
    + staged runtime preparation for V4 only
    + route-owned scheduler observer for V4
```

It must not copy the training loop or PACD operator.

## 2. Authoritative current seams

The active shared lifecycle is:

`tfpd_exploration/src/paired_anchored_calibration_dropout_full_v1/smoke.py`

The active P0 binds that file at:

`62d4b71291ffaa7175e8cdfd42d14128ee7bea428e8c4a887eb135d8292a7f75`

The immutable scientific anchors are:

```text
paired PACD core
9331288cf42d7c7924f6135f3ec8e3659f0ea19b70c8fad84448b01773c8172d

shared epoch runner
adb6c471e0e28eb544e9c7bc3f9e4a9823e2803ef33d6de5cfa5545c917a54f0

active V3 P0 closure
3356fb124ff4d18a2035bfda8e7eacc28a1642ce3f36b290155a5bcdf7faea2f
```

The current `ExecutionProfile` carries only:

```text
identity
plan
predecessor_validator
```

The current shared executor then calls GPU0-specific functions directly at
capability issuance, before attempt, after attempt and final validation.

## 3. Exact gaps found

### 3.1 Device policy is not actually profile-owned

`_issue_root_capability_after_preflight()` and `execute()` directly call:

```text
v1.preflight_gpu0_idle()
v1.recheck_gpu0_after_attempt(...)
v1._assert_cuda_binding_after_attempt(torch)
```

The production runtime factory also directly calls the last helper. A V4
profile currently cannot replace those calls without copying the executor or
modifying globals.

Required repair: add one frozen device-controller field to the execution
profile. The default controller must invoke the exact existing GPU0 helpers
and preserve legacy V1/V2/V3 receipt payloads byte-semantically. Only the V4
profile supplies the extended fixed-device controller.

### 3.2 CUDA context creation and stack loading are conflated

The current production factory performs, inside one `after_attempt()` call:

```text
import torch/lightning
bind logical cuda:0
reset peak memory
load the complete execution stack
return runtime
```

The shared executor sets `progress.cuda_bound=true` only after this call
returns. If CUDA binding succeeds but execution-stack loading then fails, the
current progress object still says `cuda_bound=false`. That is acceptable for
the historical schema but is false under V4's staged PID/device contract.

Required V4-only staged path:

```text
1. import Torch and Lightning after immutable attempt;
2. explicitly bind/materialize the selected logical cuda:0 context;
3. synchronize and obtain the selected-GPU current-PID attestation;
4. set progress.torch_imported=true and progress.cuda_bound=true;
5. reset peak-memory counters;
6. only then load the sealed execution stack;
7. only then resolve source data and deserialize the initial checkpoint.
```

Legacy profiles may retain the historical atomic factory path. The V4 path
must use typed factory methods or an equivalent typed staged-runtime object;
it must not infer context state from whether the whole factory returned.

### 3.3 Terminal validation is only logical-device validation

The current terminal path revalidates predecessor and closure, then calls
`_assert_cuda_binding_after_attempt(torch)`. It does not re-query the selected
physical GPU or prove that its sole compute PID is the current process.

Required repair: the V4 device controller supplies a final active-context
observation and exact-compares UUID, index, name, driver, PCI bus, CVD,
logical device and current PID. Legacy profiles keep their existing final
payload and check.

### 3.4 Failure does not revalidate device or scheduler state

The current exception branch revalidates closure and predecessor but publishes
no fresh device or scheduler observation. V4 requires a stage-correct failure:

- before CUDA binding: selected GPU must still be idle;
- after CUDA binding: the selected GPU must contain exactly the current PID;
- if revalidation itself fails, preserve both the original runtime error and
  the typed revalidation error.

Required repair: add profile-owned terminal/failure evidence hooks. The hooks
must never replace the original exception. Legacy hooks must reproduce the
current V1/V2/V3 schemas exactly.

### 3.5 Receipt construction is hard-coded in the shared executor

Attempt, launch, terminal and failure `device` fields are currently assembled
inline. V4 has a different exact device codec and an additional top-level
`scheduler` codec.

Required repair: the profile-owned controllers return only their exact
route-specific evidence mappings. The shared executor remains responsible for
publication and ordering. Do not branch on arm names throughout the training
loop, and do not add V4-only aliases to legacy receipts.

### 3.6 Predecessor-before-root ordering needs an explicit V4 wrapper

The current capability issuer validates artifact freshness before calling the
profile predecessor validator. The V4 work order requires the complete P0
held graph to validate before any prospective-root action.

Required repair: the additive V4 issuer first descriptor-validates the P0
graph, then calls the shared issuer. The shared issuer validates it again
through the frozen V4 profile. This preserves legacy ordering while giving V4
the required predecessor-first proof. The V4 execute path revalidates again
before attempt and before terminal/failure.

## 4. Minimal backward-compatible interface

The exact class names are not normative, but the smallest honest structure is:

```text
ExecutionProfile
  identity
  plan
  predecessor_validator
  device_controller       # legacy GPU0 by default
  scheduler_controller    # legacy no-op by default
  runtime_mode            # historical atomic or V4 staged
```

The V4 device controller owns:

```text
preflight_idle()
recheck_idle_after_attempt()
bind_context(torch)
recheck_active_current_pid(torch, progress)
failure_recheck(progress)
```

The V4 scheduler controller owns:

```text
profile
observe_before_attempt()
observe_after_attempt()
observe_final_or_failure()
```

Every controller and runtime factory has a literal identity that is bound by
the opaque one-shot capability. No controller, observer, UUID, CPU affinity or
runtime factory may be supplied through the public CLI.

The shared executor still owns, exactly once:

- attempt-before-runtime publication;
- source authority;
- the single 48-epoch loop;
- checkpoint epochs 44--47;
- final-four SWA and strict proof;
- manifest;
- terminal xor failure.

## 5. Implementation order after P0 terminal

1. Descriptor-validate the exact completed P0 graph and freeze all 58 body
   SHA literals plus the root identity.
2. Reconstruct and record the accepted historical P0 closure.
3. Add the legacy-default profile seams to the shared lifecycle.
4. Prove existing V1/V2/V3 synthetic receipt payloads do not acquire V4 keys.
5. Add the route-owned V4 P0 validator, device controller, scheduler
   controller and staged runtime wrapper.
6. Run every test in the frozen V4 work order and isolation addendum.
7. Re-run the complete V1/V2/V3 regressions, distinguishing expected
   historical current-byte closure rejection from behavioral regression.
8. Rebuild the V4 closure and independently verify both roots are fresh.
9. Issue P1 and P2 capabilities only inside their exact `taskset`/CVD/thread
   environments and only when their assigned GPUs are idle.

## 6. Additional adversarial tests required by this audit

In addition to the frozen work-order tests, require:

1. context creation succeeds and stack loading fails: failure must say
   `cuda_bound=true` and require current-PID device evidence;
2. Torch import succeeds but context creation fails: failure must say
   `cuda_bound=false` and require the idle device state;
3. active PID appears only after explicit context materialization and CUDA
   synchronization;
4. final physical-PID drift rejects even when logical `cuda:0` still appears
   valid;
5. failure-device recheck failure preserves the original runtime exception;
6. V4 predecessor validation occurs before the first root-freshness probe;
7. V4 attempt/launch/terminal scheduler observations are monotonic and the
   immutable attempt/launch observations are copied exactly into terminal;
8. starting P1 and P2 in separate synthetic processes cannot exchange roots,
   device profiles, CPU profiles, capabilities or runtime factories;
9. the shared executor and epoch runner are each invoked from one source path,
   with no route-owned copy of the 48-epoch loop;
10. legacy V1/V2/V3 dry and synthetic success/failure receipts contain no V4
    `scheduler`, PCI or current-PID keys.

## 7. Isolation conclusion

The new route remains safely separable from the completed Stage-P carrier
route and from its cross-dataset transfer. V4 reads only the completed PACD P0
graph as its scientific predecessor, owns fresh P1/P2 roots, and assigns one
fixed GPU and disjoint CPU set per arm. It never writes another route's root or
uses another route's result to select a PACD arm.

No implementation file may be changed until P0 naturally terminalizes and
the full graph passes independent audit. At that point this seam audit removes
the main foreseeable implementation ambiguity; it is not permission to launch.
