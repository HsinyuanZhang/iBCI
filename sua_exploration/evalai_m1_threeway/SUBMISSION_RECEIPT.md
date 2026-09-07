# EvalAI M1 original / T4 / D4 private-test receipt

All three private submissions to challenge 2319 phase 4599 (`test_split_m1`) finished.
Metrics are official aggregate values, **not per-session paired statistics**.

| arm | submission | Held Out R2 | Held In R2 | Normalized Latency |
|---|---:|---:|---:|---:|
| original | 578244 | 0.6485909555 | 0.7501227656 | 0.1147790631 |
| T4 | 578245 | 0.6447659719 | 0.7438711960 | 0.0447179677 |
| D4 | 578247 | 0.6439926711 | 0.7402397096 | 0.0440473690 |

Aggregate differences: T4-original = `-0.0038249836/-0.0062515696/-0.0700610954`; D4-original = `-0.0045982844/-0.0098830560/-0.0707316941`; D4-T4 = `-0.0007733008/-0.0036314864/-0.0006705987` (Held Out R2 / Held In R2 / latency).

Remote output hashes and full structured values are in `results/evalai_m1_threeway_v1/threeway_summary.json`. T4 briefly appeared as `failed` during polling but its final API status was `finished` with valid official result/stdout/stderr; this transient is retained in the monitoring history.
