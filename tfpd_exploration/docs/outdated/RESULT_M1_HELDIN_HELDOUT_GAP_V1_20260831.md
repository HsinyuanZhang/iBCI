# Result: M1 Held-In versus Held-Out Fold Gap V1

Date: 2026-08-31

Status: complete descriptive result; not a formal benchmark verdict and not a
new trained model.

## 1. Result

The frozen fold-20120924 CS-WG source-only decoder was evaluated with the same
native M10 calibration contract on its three training sessions and its one
fold-defined held-out session.

| Arm | Sessions | Equal-session R2 mean | SD (`ddof=0`) |
|---|---:|---:|---:|
| held-in training | 20120926, 20120927, 20120928 | `0.692802` | `0.008793` |
| fold-defined held-out | 20120924 | `0.567917` | `0.000000` |

The pre-registered gap is held-out minus held-in:

```text
gap = -0.124885
95% bootstrap CI = [-0.133009, -0.112670]
verdict = HEADROOM_PRESENT
direction = HELD_OUT_WORSE
```

Per-session R2:

| Session | Producer relationship | R2 |
|---|---|---:|
| 20120924 | fold-0 outer target | `0.567917` |
| 20120926 | producer training session | `0.680587` |
| 20120927 | producer training session | `0.700926` |
| 20120928 | producer training session | `0.696893` |

This contradicts the working assumption that this frozen M1 producer has a
negligible session-transfer gap on its locally available fold surface.  It
does not contradict a different paper-level aggregation or official held-out
surface; those are not the same evaluation contract.

## 2. What the result supports

The result is evidence for a real M1 cross-session generalization problem,
not evidence for insufficient source-session fitting.  The same frozen model
scores approximately `0.693` on its training sessions and `0.568` on the
fold-defined held-out session.

The read-only activity-headroom diagnostic on the same 20120924 surface
reports:

```text
STATIC_SUPPORT          0.570744
CAUSAL_GROWING_CAP30    0.593313
delta                   +0.022569
```

That diagnostic is not part of this gap verdict, but together the two
measurements separate two effects:

1. a large producer session-transfer gap of approximately `0.125` R2; and
2. a smaller but positive calibration-activity recovery opportunity of
   approximately `0.0226` R2 on the held-out surface.

This makes a causal M1 activity-memory cell scientifically admissible, but it
does not imply that activity memory can recover the entire transfer gap.  A
successful M1 design must distinguish calibration-side recovery from decoder
distribution shift.

## 3. Exact execution evidence

Canonical root:

```text
tfpd_exploration/results/m1_heldin_heldout_gap_v1
```

The complete immutable topology is exactly 12 bodies / 24 body-plus-sidecar
leaves:

```text
attempt
launch
input_authority_<session> x4
score_<session> x4
paired_table
terminal
```

Every leaf is regular mode `0444`, link count one, and every canonical
basename sidecar matches the body SHA.  There is no failure or extra leaf.

Important body SHA-256 values:

```text
attempt       b5ad95102165e336f2fe0c771f11418d0d1923a59c68df855dd6b9ab85998f2d
launch        d7acf6c6f59328189a2fe3836d6d5948ea5ac269b7f156c559b513349e1b479a
paired_table  6c410a7ae11441e2bcfa227a9683230143c173190b7828424bad1cfd2c084271
terminal      b948976e624a5a53fbc5ac5c2f5f7cf6e379c8408b1a749b80a3b5dbada49d51
```

The 20120924 anchor reproduces the sealed comparator exactly: R2,
prediction SHA, target SHA, calibration SHA, 54,849 windows and 429 forward
batches all match.  Across all four sessions:

- evaluation mode is true;
- `torch.no_grad()` is used;
- dynamic dropout is disabled;
- model state before and after is identical;
- target optimizer, backward and update calls are all zero;
- labels enter only the governing metric.

## 4. Limitations

The held-out arm contains one fold-defined session.  Its bootstrap is
therefore degenerate on that arm; the interval measures resampling variation
among the three training sessions, not population uncertainty over held-out
sessions.  The result is descriptive and explicitly has
`formal_benchmark_verdict=false`.

The official held-out-calibration M1 sessions are not scored because the
locally sealed files contain support but no post-support query rows.  They
were not silently replaced.

The launch receipt says logical `device=cuda:0` with the operator constraint
`CUDA_VISIBLE_DEVICES=1`, so the intended physical device is GPU1.  However,
this V1 route does not persist a physical GPU UUID, CUDA process PID or CPU
affinity.  The live watcher also failed to capture the short scoring window.
Therefore the honest device statement is:

```text
intended/declared device: physical GPU1 through CVD=1
durably proven physical UUID/PID: no
post-terminal release: GPU1 idle, no compute owner
```

The absence of durable device attestation does not alter the deterministic
scores, but it prevents a stronger resource-provenance claim.

## 5. PACD coexistence

PACD P0 remained live on physical GPU0 throughout the M1 interval and after
the M1 terminal.  After terminal, GPU0 had only PACD PID `783126`, GPU1 had no
compute owner, and CPU/memory/IO PSI were zero.  No PACD result, process,
root or closure file was changed.

Because no GPU1/PID receipt exists for the historical M1 interval, exact
runtime non-overlap cannot be reconstructed retrospectively from the M1 root
alone.  PACD epoch021, which spans the M1 interval, completed with every
scientific and safety invariant clean.  Its throughput was `1.54%` below
epoch020 (`10.034876` versus `10.191779` paired steps/s), an ordinary-sized
variation that cannot be attributed without sub-epoch telemetry.  There is
therefore no attributable interference failure, but also no basis for a
claim of exact runtime non-overlap.  Future concurrent routes must use the
fixed UUID/PID/affinity attestation specified by
`AUDIT_M1_PACD_CONCURRENT_ISOLATION_20260831.md`.
