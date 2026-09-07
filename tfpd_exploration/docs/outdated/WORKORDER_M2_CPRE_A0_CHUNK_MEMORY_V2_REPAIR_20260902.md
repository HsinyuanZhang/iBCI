# Work Order Addendum: M2 C-Pre V2 Window-Coordinate Repair

Date: 2026-09-02
Status: authorized successor repair; C-Pre V1 remains immutable FAIL-CLOSED
Scope: C-Pre metadata root/schema and its A0 predecessor binding only

This addendum inherits every science, device, scheduling, no-target, A0, and
receipt rule from:

```text
tfpd_exploration/docs/WORKORDER_M2_CPRE_A0_CHUNK_MEMORY_V1_20260902.md
SHA256 5367edd6d6f3455c59fd892d5dd7113076191a1ca584fb4dfd3c975f02c7c64b
```

It changes only the items stated below.  The V1 work order and failed result
root must not be edited, deleted, renamed, retried, or described as a runtime
or GPU failure.

## 1. Immutable failed predecessor

The first C-Pre attempt correctly failed before metadata publication because
its metadata-only route compared a raw trial-start coordinate with the
historical anchor's full-W50-disjoint query coordinate.

```text
root
tfpd_exploration/results/m2_cpre_metadata_v1

attempt.json SHA256
47dabd4b1f364afb8e83402c1f7f098603e6fc3c6decbe2e1248f07699038e67

failure.json SHA256
f91383045b4c7ee63b37272419b753d8779bbb456928d2ada081176e922f150a

failure stage
metadata_provider

failure class/message
InventoryError: within post30 anchor window count/digest drift
```

The exact topology is the four regular `0444`, nlink-1 leaves
`attempt.json`, `attempt.json.sha256`, `failure.json`, and
`failure.json.sha256`.  The failure records target/data-value/model/checkpoint
access false and CUDA initialization false.  C-Pre V2 admission must
descriptor-read and validate this graph before reserving its own root.

## 2. Successor identity

```text
C-Pre root    tfpd_exploration/results/m2_cpre_metadata_v2
attempt       m2_cpre_attempt_v2
inventory     m2_cpre_metadata_inventory_v2
terminal      m2_cpre_terminal_v2
failure       m2_cpre_failure_v2
```

A0 remains the V1 experiment and retains its already frozen shard/aggregate
roots.  Its live capability must now bind an accepted C-Pre V2 terminal and
inventory, not C-Pre V1.

## 3. Correct W50 coordinate law

The dataset stores 49 bins of left padding.  A dataset window start `s`
corresponds numerically to raw output/endpoint bin `s`; its 50 raw neural bins
are `s-49 ... s`.  Therefore a window is wholly after chronological trial
boundary `raw_trial_start[n]` only when:

```text
s >= raw_trial_start[n] + (WINDOW_BINS - 1)
  = raw_trial_start[n] + 49
```

C-Pre V2 must apply this exact full-window-disjoint law.  It must not select
post-support windows using `s >= raw_trial_start[n]`.

All reported fixed-exposure rows remain intersected with the frozen post-30
query surface from the parent work order.  Consequently the effective fixed
suffix floor is:

```text
max(raw_trial_start[n] + 49, raw_trial_start[30] + 49)
```

This matters for the descriptive `n=10` row: it must not score windows from
trials 11--30 merely because C0 may later initialize activity at M10.  A0
itself still starts at total exposure 30.  `all-past` begins at the same
full-W50-disjoint post-30 floor and advances causally thereafter.

The chunk-memory phase origin is unchanged and remains the first raw neural
bin at `raw_trial_start[30]`.  The `+49` correction changes only governed
query-window eligibility; it must not shift, filter, or relabel chunk contents.

## 4. Required repair tests

Before C-Pre V2 capability issuance:

1. an actual-shaped session proves the old raw-boundary rule has exactly the
   observed count/digest mismatch and the `+49` law exactly reproduces the
   historical within-post30 anchor row;
2. external full-query authority remains unchanged, while external local
   post30 uses the `+49` disjoint suffix and is a deterministic subset;
3. `n=10` is intersected with the post30 floor; `n=30` is identical to that
   floor; larger `n` uses its own later full-window-disjoint floor;
4. C-Pre V1's four-leaf failure graph is held-descriptor validated, including
   body SHA, sidecar, mode, nlink, status, stage, error, and all no-access
   flags;
5. C-Pre V2 remains metadata-only, attempt-first, Torch/CUDA-free, and uses
   the stage-dependent immutable-prefix failure topology from the parent work
   order.

No A0 decoder, carrier, activity-memory, chunk geometry, metric, GPU device,
or acceptance threshold may change under this repair.
