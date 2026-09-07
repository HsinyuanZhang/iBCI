# Official record: M2 SMALL ext6 epoch-pick 581973

Recorded 2026-09-06. Historical comparison only. Do not use this official
score to pick later candidates.

| Field | Value |
|---|---|
| ID | **581973** |
| Status | finished |
| Method | M2 small Transformer S1-SMALL-COS EMA e8 s42 ext6-epochpick exact-E w0 |
| Image | `spint-t4-m2:small-trf-s42-ema-e8-ext6-w0-4db109e7` |
| Payload SHA-256 | `4db109e75276d6e8f47df6540b0a4c8f6e955f81af1212127723976795f2eaa4` |
| Scoring tick | `execution_time` 0.106682 s at 2026-09-05T16:11:46Z |

## Official test-split M2

| Metric | 581973 e8 | 581971 e19 | 581919 MOVE-T4 |
|---|---:|---:|---:|
| Held Out R2 Mean | **0.390305** | 0.351251 | 0.349453 |
| Held Out R2 Std. | 0.128128 | 0.132108 | — |
| Held In R2 Mean | 0.654638 | **0.660798** | 0.597230 |
| Held In R2 Std. | 0.034183 | 0.039846 | — |
| Normalized Latency | 0.736014 | 0.452500 | 0.044037 |

e8 minus e19: HO **+0.03905**, HI **−0.00616**, latency worse (0.736 vs 0.453).
e8 minus MOVE-T4 581919: HO **+0.04085**, HI **+0.05741**.
