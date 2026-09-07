# M1 fixed-K temporal prototype Gate A2 — root result audit

**Reviewed:** 2026-08-02 (Asia/Hong_Kong)  
**Disposition:** P20/fixed-K mechanism branch stopped at Stage 0; the simpler B20 marginal carrier
is retained as a new, separately gated hypothesis.

## 1. Provenance and scope

The authoritative artifact is
`results/m1_fixed_k_temporal_prototype_gate_a2_v1/source_gate_a2.json`, SHA-256
`379f3c3b85fe120735802af887d8668fc4d84c3e89cb77acdaabecd485565981`.
Execution was bound to immutable prelaunch SHA-256
`fa6b513bd881ea4c1a03f184b88909bf6a3a9603e03d8ce903b6d18b888aa2e3`.

Only the same four hash-bound M1 `held-in-calib` source NWBs were opened. The runner opened no
minival, held-out, formal-test, or EvalAI path, used no decoder, and forced CUDA off. The endpoint
remains the source-LOSO later-neural category-rate oracle proxy; it is not behaviour-decoding R2.
Future object labels were scorer-only.

All exact raw-prefix, padding, support/future boundary, width, anchor-source, finite-value,
streaming-state, and split-half repeatability contracts passed. Root focused validation was
`27 passed`; prelaunch hashes for protocol, manifest, controls, runner, canonical representation,
and the v3 exact-source loader all matched.

## 2. Stage-0 result

| carrier | per-session source-LOSO proxy R2 | mean R2 |
|---|---|---:|
| legacy rate-only | `0.774948 / 0.794927 / 0.826736 / 0.652464` | 0.762269 |
| P20 fixed-K temporal prototype | `0.880273 / 0.891359 / 0.920905 / 0.758640` | 0.862794 |
| B20 marginal distribution | **`0.892449 / 0.942571 / 0.963411 / 0.820599`** | **0.904758** |

B20 contains no labels, anchor, slot, timestamp, or temporal filter. Its width-20 coordinates are
ten sorted support-trial log-rates and ten pooled valid-bin `log1p(count)` quantiles.

The predeclared P20-minus-B20 content result is:

- per-session deltas: `-0.012176 / -0.051213 / -0.042506 / -0.061959`;
- mean `-0.041963`, negative in 4/4 sessions;
- paired 95% CI `[-0.076004,-0.007923]`;
- MDE80 `0.044507`.

Thus P20 does not beat the high-order marginal baseline in any session. The exact Stage-0 stop is
`marginal_baseline_not_beaten_stop`. Under the frozen compute-saving rule, all 13,824 relative-slot
configurations and all 4,095 random time schedules were correctly marked
`not_run_due_to_predeclared_early_stop`. The source execution ended in 58.68 seconds.

## 3. Revised interpretation of v3

The earlier v3 finding `P20-rate_only=+0.100526` remains numerically correct, as does P20's high
split-half repeatability. Gate A2 shows that neither fact identifies a temporal-prototype
mechanism. A richer order-invariant carrier performs better:

- B20 minus rate-only: mean `+0.142489`, positive in 4/4 sessions;
- paired 95% CI `[+0.108828,+0.176150]`;
- MDE80 `0.044010`.

The defensible mechanism conclusion is therefore:

> The M1 first-ten support contains stable, nonlinear marginal rate/count-distribution information
> about later channel tuning. The current evidence does not show that chronological EWMA state,
> routed slots, or fixed-K temporal prototypes add value beyond that distribution.

This is a simplicity result, not a temporal-memory success.

## 4. Frozen consequences

The following are stopped:

- P20 decoder integration, GPU pilot, held-out evaluation, EvalAI, INT8, K/r/filter/router sweep;
- the conditional M2-P20 and SUA-P20 replication protocols;
- slot/time null computation from this failed conjunction;
- any attempt to rename B20's gain as P20 temporal evidence.

B20 itself is not automatically GPU-authorized. It was revealed on the same four already-inspected
M1 source sessions and may reflect persistent channel gain, recording quality, or burstiness that
the neural oracle rewards but a behaviour decoder ignores. It receives a new evidence chain:

1. M1 source-only component, row-attachment, and reliability characterization;
2. independent M2 source replication;
3. stronger variable-N SUA source plus target-free development replication;
4. only then a separately reviewed minimal decoder cell.

If a simpler B20 component explains the full result, the claim and deployment path must shrink to
that component rather than preserving the 20-dimensional name.
