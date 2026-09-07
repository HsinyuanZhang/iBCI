# T4 M2 official EvalAI submission receipt

**Submitted:** 2026-08-01T19:09:30+08:00  
**Completed:** 2026-08-01T19:31:56+08:00  
**Status:** `FORMAL_FINISHED / OFFICIAL_POSITIVE`  
**Submission:** `578221`, private, team `HKU-ECE` (`41975`)  
**Phase:** FALCON Test Phase `few-shot-test-2319` (`4599`)  
**Task:** native M2

## Official result

| Official metric | T4 `578221` | original SPINT `578218` | T4 - SPINT |
|---|---:|---:|---:|
| Held Out R2 Mean | **0.30324395** | 0.18647872 | **+0.11676523** |
| Held Out R2 Std. | **0.10823401** | 0.16214462 | **-0.05391061** |
| Held In R2 Mean | **0.58760827** | 0.56824267 | **+0.01936560** |
| Held In R2 Std. | **0.02546236** | 0.03116488 | **-0.00570252** |
| Normalized Latency | **0.04290260** | 0.11236103 | **-0.06945843** |

Observed relative changes against the original official anchor:

- held-out mean R2 is `+62.62%` relative and `+0.11677` absolute;
- held-out dispersion is `33.25%` lower;
- held-in mean R2 is `+3.41%` relative and `+0.01937` absolute;
- observed online normalized latency is `61.82%` lower, or `2.619x` faster.

The primary endpoint is the held-out result. It is clearly positive and larger
than the strict local exact-candidate q33 effect (`T4-B0=+0.06420`).

## Frozen submitted artifact

| Item | Receipt |
|---|---|
| image tag | `spint-t4-m2:e8-seed42-epoch002-dcc449a-r3` |
| local image ID | `sha256:57b1fb2418ad2fd8f2d4e62f8fddb6d4f75b9c8a072730bf7afb6af53517d260` |
| uploaded manifest digest | `sha256:57b1fb2418ad2fd8f2d4e62f8fddb6d4f75b9c8a072730bf7afb6af53517d260` |
| uploaded UUID tag | `f98cb711-ef06-4f59-82f8-4dd262657b25` |
| T4 payload SHA-256 | `dcc449a15bc478f3380c95add964fc344522a25bc9938c0a5563bf5a75ae0c96` |
| T4 checkpoint SHA-256 | `25d7bc72b4d440004b58f1beaeadb7e15565a43e83dd1eadd160374270ec1d3e` |
| epoch-34 teacher SHA-256 | `fbcb9914561c4664fa0f8d0b1791e67505841d3ac470ea7ad68d54e408ca13ec` |

The uploaded ECR manifest digest exactly equals the frozen local Docker image
ID. The guarded helper registered one submission exactly once.

## EvalAI lifecycle and metadata

| Field | Value |
|---|---|
| submission ID | `578221` |
| submitted at | `2026-08-01T11:09:30.115378Z` |
| final API `started_at` | `2026-08-01T11:31:56.282387Z` |
| completed at | `2026-08-01T11:31:56.402382Z` |
| final API `execution_time` | `0.119995` seconds |
| status | `finished` |
| visibility | private |
| ignored / flagged | false / false |
| method name | `T4 cached identity M2 M33 seed42` |
| `IsHeldOutZeroShot` | false |
| `IsTestTimeAdaptive` | false |
| `IsPretrained` | false |

The final API execution-time field is the short scoring-record interval, not
the container wall time or decoder latency. The benchmark runtime quantity is
the official `Normalized Latency` above.

The method description submitted to EvalAI explicitly states that identities
were calibrated offline from the chronological first 33 public calibration
trials and their target directions, with no hidden query labels, optimizer, or
runtime backpropagation.

## Official-file archive

Archived locally under
`sua_exploration/results/e8_falcon_evalai_m2_t4_seed42_v1/`:

| File | Bytes | SHA-256 |
|---|---:|---|
| `registration_input.json` | 194 | `ae550c9fc690bdf7d722e2c9d5c01e841ead7bf3545d06b47b3e5ec913ebb3de` |
| `stdout.txt` | 261 | `7637d8417880d3c33456a2031cb2e358b2fd8965431b2f457a4d28693642a74d` |
| `stderr.txt` | 5,280 | `98f704d1537176fdd172b7495e23b13b28bbff5ea89bd4bb0f4f16a2cb7bdecf` |
| `submission_result.json` | 223 | `9be24760235381364dc78ad14d9b431f3d8c3d60000f951a5677da7b808dd065` |
| `submission_predictions.pkl` | 1,258,463 | `7fae9664c34dc4271f95a2cf0108e7ef79a8fb73fad6a5b1684979fb5712a3d9` |
| `submission_578221_receipt.json` | see file | machine-readable receipt |

The server's stderr file is nonempty because the evaluator writes normal INFO
messages, NWB namespace warnings, progress bars, and the expected 300-second
remote sleep message there. It contains no traceback or model error. The task
finished and produced all official metrics.

The archived prediction payload contains exactly 13 M2 session keys, float32
arrays with two behavior outputs, and no target/label arrays.

## Scientific interpretation

This result establishes that the submitted supervised T4 deployment is useful
on the most important endpoint: strict organizer-held private M2 held-out data.
It also shows that cached identity eliminates the feared online-overhead
problem in this deployment: the observed official normalized latency is lower,
not higher, than the original SPINT submission.

Two qualifications remain mandatory:

1. Submission `578218` packages the original epoch-27 decoder, while T4 freezes
   the epoch-34 teacher decoder. Their official difference is therefore an
   operational system comparison, not a pure T4-only ablation. The matched
   local epoch-34 q33 comparison is positive (`+0.06420`) but is not an official
   hidden-test ablation.
2. T4 uses one target-direction label per first-33 calibration trial, whereas
   B0 does not. `T4-B0` measures the end-to-end value of labeled calibration;
   `T4-TS4` is the channel-attached-content control.

The offline T4 identity fit/export is not included in official normalized
latency. Claims should therefore say **online decoding latency** rather than
end-to-end calibration-plus-decoding latency.

No duplicate T4 submission should be created. Submission `578221` is the
frozen official result for this candidate.

