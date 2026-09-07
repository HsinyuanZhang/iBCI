# PACD V2 Source-Smoke Result

Date: 2026-08-31

Status: engineering feasibility PASS; performance remains unmeasured

## 1. Outcome

PACD V2 completed the bounded source-only smoke on physical GPU0. It ran all
three predeclared arms from the same sealed initial state:

| Arm | Paired views | Optimizer steps | Result |
|---|---:|---:|---|
| P0 | M30 / M30 | 8 | PASS |
| P1 | M30 / M4 | 8 | PASS |
| P2 | M30 / M10 | 8 | PASS |

This is not an R2 experiment. It establishes only that the paired operator,
source loader, sealed Cell-D, optimizer, RNG replay, receipt lifecycle, and
resource isolation work together on the production GPU path.

## 2. Integrity evidence

Canonical result root:

```text
tfpd_exploration/results/paired_anchored_calibration_dropout_v2/smoke_seed42
```

The root has exactly four immutable `0444` regular leaves:

```text
attempt.json
attempt.json.sha256
terminal.json
terminal.json.sha256
```

There is no `failure.json` and no extra leaf. Body SHA-256 values:

```text
attempt.json   ef2ebde24864c8105e47b6ef1c925149b128fe9531577dbf58fba227ea11eb3a
terminal.json  a04be949665a5c57091ba2192793401f43950735142fdb8fac9e2ec6829f9dfe
```

The terminal binds the attempt, the exact immutable V1 failed predecessor,
and identical launch/final 35-file execution closures:

```text
1b7ebf98a01e197718588eb9702949304f68b527ba08660ae88dd62857d11339
```

The V1 predecessor body literals are:

```text
attempt.json  e1d6cd813b2bc49d89121b3341271a0bf94176abe3811d2f3b6f6ffa4a28b986
failure.json  82ca6850cdd50f0b5ea5073eb89809a0c6ea426ac3f8f7d9e03fc44d1bd7e273
```

The predecessor was validated through held descriptors before the V2 root
was reserved and again before terminal publication.

## 3. Operator evidence

Every arm completed eight paired steps and exactly eight optimizer updates.
For every step:

- anchor and short forwards used the same realized whole-unit dropout
  probability and exact unit-mask digest;
- the short forward reproduced the anchor RNG transition;
- the global post-pair RNG state equalled the one-forward state;
- valid-bin count was positive;
- encoder and decoder gradient norms were finite and non-zero;
- all 29 materialized model parameters were finite;
- exactly two inactive lazy parameters were recorded and intentionally
  skipped by the finite-state checker.

For P0, all eight M30/M30 pairs had bitwise-equal prediction and identity
digests. This is the key paired-compute control.

## 4. Resource and isolation evidence

The selected device was logical `cuda:0`, physical GPU0, UUID:

```text
GPU-ac7388a5-2e98-300a-fdb3-0b67bfd494d9
```

GPU0 had no compute process at both pre-attempt checks. Peak CUDA memory was:

```text
allocated  231,027,712 bytes
reserved   327,155,712 bytes
```

Measured arm throughput:

| Arm | Wall seconds | Optimizer steps/s | Forwards/s |
|---|---:|---:|---:|
| P0 | 1.362 | 5.876 | 11.752 |
| P1 | 0.794 | 10.077 | 20.154 |
| P2 | 0.875 | 9.148 | 18.296 |

Total terminal wall time, including setup and sealed loading, was 14.679 s.
The concurrent Stage-P job remained on GPU1 and continued normally. PACD used
`num_workers=0` and one numerical CPU thread.

The source datamodule was materialized once with the exact 27-session source
roster. Validation, test, within-development, external sub-M, formal, and
organizer-held paths remained unopened; no scorer was called.

## 5. V1 failure and V2 repair

The sole V1 attempt failed after its first real paired optimizer step because
the PACD route called `numel()` on an inactive Torch
`UninitializedParameter`. It was an assertion implementation bug, not a
model-finiteness failure. V2 skips only that exact parameter type and still
checks every materialized parameter. A real Cell-D CPU regression reproduces
the old exception and passes the repaired checker.

The V1 result root was not deleted, modified, or reused. V2 used a new root
and explicitly preserved the V1 failure lineage.

## 6. Decision

The PACD engineering and production-path smoke gate is **PASS**. The result
does not show that PACD improves M4, M10, M30, within-session, or held-out R2.

The next meaningful experiment is matched full source training, followed by
the already specified held-out comparison. It requires a separate full-run
work order and launch authorization. A full run must preserve P0 as the
matched doubled-compute control and report P1/P2 against both P0 and the
existing static/C1 anchors. No full training is authorized by this result
document.

