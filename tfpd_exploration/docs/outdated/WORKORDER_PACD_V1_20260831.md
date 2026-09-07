# Work Order: PACD V1 Paired Source Smoke

Date: 2026-08-31

Status: authorized only for implementation review and one bounded source smoke
on an otherwise idle GPU. This work order does not authorize full training or
target scoring.

## 1. Objective

Implement the first executable feasibility test of Paired Anchored
Calibration Dropout (PACD). For each unique source query, the route performs
two supervised Cell-D forwards before one optimizer update:

```text
anchor = B3S(first 30 calibration-activity trials), ordinary M30 T4
short  = B3S(first M calibration-activity trials), ordinary M30 T4
loss   = 0.5 * task_loss(anchor) + 0.5 * task_loss(short)
```

The three smoke arms are fixed:

- `p0`: M30 + M30;
- `p1`: M30 + M4;
- `p2`: M30 + M10.

## 2. Scientific invariants

The query neural tensor, behavior target, ordinary M30 T4, valid-bin mask,
source session, source batch indices, model initialization, optimizer, LR,
and logical optimizer-step count are identical across arms. The only
treatment is the short branch's B3S calibration-activity prefix.

Both branches must consume the same realized Cell-D dynamic whole-unit
dropout probability and the same exact unit mask. The route must accomplish
this by replaying route-owned Python/Torch RNG state around the two forwards,
not by changing Cell-D or replacing its dropout implementation. After the
pair, the global RNG stream must equal the state after one ordinary forward.

There is no prediction-consistency loss, posterior T4, reliability feature,
learned gate, target update, D-opt selection, CDM state, or new trainable
parameter.

## 3. Bounded smoke

The smoke runs `p0`, `p1`, and `p2` serially from the same immutable initial
state. Default exposure is eight optimizer steps per arm with batch size 32.
It is a feasibility test only and cannot select a method.

Required evidence:

- attempt is immutable before source data, checkpoint tensor, or CUDA access;
- exact implementation closure and sealed predecessor digests;
- exact initial-state strict reload;
- identical source batch order across all arms;
- one unique query batch and one optimizer update per paired step;
- anchor/short dropout probability and unit-mask digest equality;
- full/full P0 prediction equality before update;
- expected B3S prefix lengths 30/30, 30/4, and 30/10;
- finite branch losses, gradients, model, and Adam state;
- non-zero encoder and decoder gradient norms;
- no target/within/external/formal path opened or resolved;
- measured elapsed time, steps/s, and peak CUDA memory;
- immutable terminal or honest failure.

The launch/final execution closure contains the work order, executable PACD
package and CLI, and every actually imported loader/model dependency. The
design document and focused test file are review evidence, not executable
dependencies: their launch hashes may be recorded for audit, but a later edit
to either one must not invalidate or restart a numerically unaffected live
smoke.

## 4. Resource isolation

The sole authorized smoke device is physical GPU0:

```text
CUDA_VISIBLE_DEVICES=0
logical device=cuda:0
expected UUID=GPU-ac7388a5-2e98-300a-fdb3-0b67bfd494d9
```

GPU1 and every process using it are outside scope. The smoke uses
`num_workers=0` and one CPU thread per numerical runtime to minimize CPU and
I/O interference. Immediately before launch the route must fail closed unless
GPU0 has no compute process. It must never signal, stop, modify, or inspect
the mutable state of the GPU1 job.

## 5. Stop conditions

Stop without retry if:

- GPU0 is no longer idle;
- the result root exists;
- any closure or sealed predecessor drifts;
- exact paired dropout replay fails;
- P0 predictions differ;
- any loss, gradient, parameter, or optimizer state is non-finite;
- source preparation or smoke execution fails;
- GPU1 is selected or any target path is resolved.

No full 48-epoch PACD job is authorized by this work order. A successful
smoke returns only an implementation and throughput boundary for independent
review.

## 6. Role separation

The implementation and execution workflow is deliberately separated:

- **Terra implementation worker:** owns only the additive PACD package, CLI,
  and focused tests; may run CPU/no-CUDA checks but may not launch or touch a
  canonical PACD result root.
- **Root reviewer:** owns this work order and the PACD design document,
  independently reviews implementation bytes/tests/closure, checks resource
  isolation, and is the only role allowed to authorize the bounded smoke.
- **Luna Max watcher:** performs read-only process/GPU/receipt monitoring after
  launch; may not edit, signal, retry, restart, score, or load checkpoint
  tensors.

The implementation worker's passing tests are necessary but not sufficient
for launch. The root reviewer must repeat the critical no-CUDA suite and
preflight checks against the settled bytes before any GPU process starts.
