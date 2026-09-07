# Decision: stop new DANDI work after PACD P0

**Date:** 2026-09-02  
**Decision:** binding project-direction record; not an execution authorization  
**Effective immediately:** no new DANDI 000688 experiment may launch

## 1. Final allowed action

The sole allowed DANDI computation is the already-running PACD P0 producer:

```text
route: paired_anchored_calibration_dropout_full_v3 / p0_fullfull_seed42
pid: 783126
device: physical GPU0, UUID GPU-ac7388a5-2e98-300a-fdb3-0b67bfd494d9
result root: tfpd_exploration/results/paired_anchored_calibration_dropout_full_v3/p0_fullfull_seed42
```

It may finish its existing epoch047, checkpoint047, final-four SWA, manifest
and terminal/failure lifecycle without intervention. This permission does not
authorize a retry, repair run, replacement process or successor.

## 2. Explicitly stopped work

After P0 reaches a natural terminal or honest failure, all of the following
remain frozen and non-launchable:

- PACD P1 and P2 training;
- PACD matched scoring and mixed-lineage scoring;
- PACD speed/scheduler successors;
- CDM-D/PACD follow-up training or scoring on DANDI 000688;
- new DANDI carrier, gate, memory, ablation or comparator cells;
- reruns with new seeds, epochs, batch sizes, checkpoints or relaxed gates.

Existing code, receipts and result roots are retained as archival evidence and
must not be deleted or rewritten.

## 3. Terminal handling

The root agent and watcher may perform read-only monitoring and, after a
natural event, verify:

1. PID/tmux exit state and device release;
2. exact body/sidecar topology;
3. regular-file, mode, basename-sidecar and SHA-256 integrity;
4. epoch count and optimizer-step count;
5. checkpoint44--47 and final-four SWA links;
6. manifest and terminal/failure cross-links;
7. source-only/no-target facts;
8. absence of extras and mutually exclusive terminal/failure topology.

They may not signal the producer, change CPU affinity/priority, alter CUDA
visibility, modify its bytes, load its checkpoint tensors while it is live, or
touch another GPU owner's process or result root.

## 4. Resource boundary

P0 owns physical GPU0 only. Any GPU1 process belongs to another route and is
immutable external state. No scheduler, utilization experiment or concurrent
job may be added to GPU0 before P0 exits. After exit, GPU0 is released for
non-DANDI work only.

## 5. Research pivot

New research and GPU allocation move to official-comparison tasks, primarily
M1 and then M2. The active successor design is:

```text
tfpd_exploration/docs/DESIGN_M1_FUNCTIONAL_CARRIER_MEMORY_20260902.md
```

The DANDI P0 result may be reported as an archival method/mechanism result, but
it cannot reopen the stopped DANDI exploration queue regardless of whether its
eventual score is positive, null or negative.
