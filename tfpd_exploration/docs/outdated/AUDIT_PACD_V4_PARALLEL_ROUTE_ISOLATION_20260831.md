# PACD V4 Parallel-Route Isolation Addendum

Date: 2026-08-31

Status: frozen scheduler/admission addendum, amended after the CDM route
terminal to make the host-pressure gate reproducible. This document does not
amend the PACD paired science, the active V3 closure, or the frozen V4
training work order. It resolves the host-resource coexistence condition that
must be checked before either V4 arm is admitted.

## 1. Scope

The frozen V4 work order pre-registers P1 on physical GPU0 and P2 on physical
GPU1. Its Section 9 CPU partition is explicitly described as recommended
scheduler-only placement, not as part of the numerical contract.

At this audit boundary two unrelated live processes occupy the host:

```text
PACD P0 V3
  PID 783126
  physical GPU0
  CPU affinity 0-3,16-19
  result root paired_anchored_calibration_dropout_full_v3/p0_fullfull_seed42

CDM+P1 replay
  PID 804509
  physical GPU1
  CPU affinity 4-15,20-31
  result root cdm_p1_cross_v1
```

The sets are disjoint. No V4 implementation or launch is permitted while P0
is active, irrespective of apparent GPU or CPU headroom.

## 2. Fixed successor CPU placement

To preserve the existing disjoint partition after the predecessor processes
exit, V4 uses these exact scheduler-only CPU sets for the complete lifetime of
each process:

| V4 arm | Physical GPU | Exact logical CPU affinity |
|---|---:|---|
| P1 | 0 | `0-3,16-19` |
| P2 | 1 | `4-15,20-31` |

The affinity may not be expanded, migrated, or changed after attempt
publication. DataLoader workers remain zero and OMP, MKL, OpenBLAS and
NUMEXPR thread counts remain one, as required by the training work order.

This placement is an execution-isolation choice only. It does not change
model bytes, source roster, sampler order, RNG replay, dropout masks, loss,
optimizer, update count, checkpoint selection, SWA, or scoring.

These exact sets supersede only the *recommended scheduler-only* sets printed
in Section 9 of the frozen training work order. They do not amend any
numerical field in that work order. An implementation must use the exact sets
above and must not choose between the two scheduler descriptions.

## 3. Per-arm predecessor-release gate

An arm may be admitted only after all of the following are true:

1. the complete P0 V3 immutable terminal graph passes the held-descriptor gate
   in the frozen V4 work order;
2. the arm's assigned physical GPU has no compute application;
3. no unrelated live process has any CPU in common with the arm's exact set;
4. no unrelated live process has the arm's result root open or writable;
5. the arm's canonical result root is fresh and absent;
6. the current V4 closure, device profile, runtime-factory identity and parent
   directory identities are exact;
7. the exact host-pressure observation in Section 4 passes.

The current CDM replay therefore imposes an additional literal rule:

- P2 cannot start until PID 804509 has naturally terminalized or failed and
  released physical GPU1 and CPUs `4-15,20-31`.
- P1 cannot start before P0 terminal validation. If the CDM replay remains
  live after that point, P1 may start only on CPUs `0-3,16-19`, which are
  disjoint from the CDM set. P1 never borrows CDM CPUs.

There is no timeout, forced stop, affinity mutation, process suspension, GPU
migration, root reuse, or retry to make either gate pass.

## 4. Evidence required in V4 receipts

The post-P0 V4 implementation must add one route-specific top-level
`scheduler` mapping to attempt, launch, terminal and failure receipts without
changing the shared `device` mapping or epoch runner. Keeping this evidence
outside `device` preserves the frozen exact device codec and prevents aliases.

The exact profile identities are:

```text
v4-p1-gpu0-cpu0-3-16-19
v4-p2-gpu1-cpu4-15-20-31
```

At every recorded boundary, `os.sched_getaffinity(0)` must equal the arm's
exact CPU set and the tracked route overlap must be empty. A drift or overlap
fails before data/checkpoint access when possible, and otherwise produces an
honest exclusive failure receipt. The observer is internal and cannot be
supplied through the public CLI.

The exact scheduler profile is:

```json
{
  "identity": "<exact profile identity above>",
  "logical_cpu_affinity": [0, 1, 2, 3, 16, 17, 18, 19],
  "num_workers": 0,
  "thread_limits": {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1"
  }
}
```

P2 substitutes the ordered affinity
`[4,5,6,7,8,9,10,11,12,13,14,15,20,21,22,23,24,25,26,27,28,29,30,31]`.
No extra profile keys are accepted.

Each observation has exactly:

```json
{
  "observed_affinity": ["the exact ordered profile affinity"],
  "tracked_other_route": {
    "identity": "none",
    "pid": null,
    "live": false,
    "observed_affinity": []
  },
  "overlap": [],
  "host_pressure": {"...": "the exact mapping below"}
}
```

The CDM replay terminalized before V4 implementation was authorized. A live
V4 producer must therefore use the inactive `none` mapping above. The earlier
live-CDM shape remains useful only as synthetic parser/adversarial coverage;
it is not an admissible future production observation. The observer does not
scan unrelated system processes or infer ownership from a broad process-name
match. Root separately rechecks the known in-scope process inventory before
issuing either capability.

The exact `host_pressure` mapping is read from `/proc/meminfo` and
`/proc/pressure/memory` without importing Torch, opening data, or querying
either GPU:

```json
{
  "schema": "pacd_v4_host_pressure_v1",
  "observed_monotonic_ns": 1,
  "mem_available_bytes": 1,
  "mem_available_floor_bytes": 17179869184,
  "memory_psi_some_avg10": 0.0,
  "memory_psi_some_avg10_ceiling": 0.1,
  "memory_psi_full_avg10": 0.0,
  "memory_psi_full_avg10_ceiling": 0.0,
  "swap_total_bytes": 1,
  "swap_free_bytes": 0,
  "pass": true
}
```

The numeric values shown as `1` are runtime observations, not literals.
`observed_monotonic_ns` and `mem_available_bytes` must be positive integers;
`swap_total_bytes` must be a nonnegative integer and `swap_free_bytes` an integer in
`[0, swap_total_bytes]`. The two PSI values must be finite nonnegative
numbers. `pass` is true if and only if `mem_available_bytes` is at least the
fixed 16-GiB floor, PSI `some avg10` is at most `0.1`, and PSI `full avg10`
is exactly `0.0`. Swap occupancy is descriptive: a nonempty swap device does
not itself imply active thrashing when both PSI gates pass. Missing procfs,
parse failure, a non-finite value, or a failed inequality rejects admission
or produces the honest failure branch after attempt.

Attempt, launch, and final observations must be newly sampled in monotonic
order. The terminal copies the immutable attempt and launch observations
exactly and requires
`before_attempt.observed_monotonic_ns <= after_attempt.observed_monotonic_ns
<= final.observed_monotonic_ns`. A failure records the freshest safe sample or
the typed host-pressure recheck error without inventing a passing sample.

Receipt placement is exact:

```text
attempt.scheduler  = {profile, before_attempt}
launch.scheduler   = {profile, after_attempt}
terminal.scheduler = {profile, before_attempt, after_attempt, final}
failure.scheduler  = {profile, stage, final, recheck_error}
```

Terminal `before_attempt` and `after_attempt` must equal the immutable attempt
and launch observations. Failure `stage` is one of
`before_attempt`, `after_attempt`, or `final`; `recheck_error` is null on a
successful observation and otherwise preserves the typed scheduler-recheck
error without replacing the original runtime failure.

## 5. Test additions

The V4 no-CUDA suite must additionally prove:

1. exact P1 and P2 affinity profiles pass;
2. missing, extra, swapped or expanded CPU sets reject;
3. a simulated unrelated PID with any overlapping CPU rejects;
4. a non-overlapping unrelated PID is descriptive and does not alter science;
5. P2 rejects while the simulated CDM GPU1/CPU owner is live;
6. affinity drift between capability, attempt and terminal/failure rejects;
7. V1/V2/V3 legacy receipt schemas and the shared epoch runner remain
   unchanged;
8. exact live production observations require the inactive CDM mapping, while
   the historical live-CDM shape is retained only as synthetic parser
   coverage;
9. missing/extra/non-finite host-pressure fields, a low available-memory
   value, either PSI threshold violation, a false/inconsistent `pass`, and
   non-monotonic attempt/launch/final sample times reject;
10. a nearly occupied swap device with sufficient available memory and both
    PSI gates passing remains descriptive and passes; a host with swap
    disabled (`swap_total_bytes == swap_free_bytes == 0`) also passes;
11. dry import neither enumerates processes, reads procfs, nor imports Torch.

## 6. Relationship to frozen authorities

The frozen training work order remains byte-identical with SHA-256:

```text
34a67357d66c36816457252e82d5ac7dcecf34340dafff1f372c67571dc09cd3
```

The active V3 execution closure remains byte-identical with SHA-256:

```text
3356fb124ff4d18a2035bfda8e7eacc28a1642ce3f36b290155a5bcdf7faea2f
```

This addendum must be included in the future V4 implementation closure and
its review-evidence map. It is not added to the active V3 closure. The no-data
Score V2 candidate must gain this addendum as an explicit closure authority
and exact-validate the top-level attempt/launch/terminal scheduler mappings;
that is a producer-codec change only and must not alter scoring science.

## 7. Decision

With this addendum, the post-P0 route has an explicit, testable answer to the
parallel-work requirement: V4 never takes a GPU or CPU owned by the other
route, never changes the other route, and never trades scientific identity for
faster admission. Until all applicable release gates pass, the correct action
is to wait.
