# Carrier Perf Program — local status

**Updated:** 2026-08-05 (all meaningful CPU cells sealed)  
**GPU authorization:** none  
**Handoff:** `sua_exploration/docs/HANDOFF_CARRIER_PERFORMANCE_PROGRAM_20260805.md`

## Checklist

| ID | CPU execute | GPU gate | Verdict |
|---|---|---|---|
| P0N | `mua_noise_floor_v1/` | blocked | `fold2_seed43` missing; thresholds unfrozen |
| P1A | `t4_estimator_equivalence_v1/` | — | Estimator diffs **not** ignorable; prefer B |
| P1B | `t4_falcon_estimator_b_vs_c_v1/` | advisory only | C helps rate-MSE; no frozen P0N |
| P1 close | `p1_closeout_v1/` | — | Bootstrap CI sealed; **W3 reopen NO** (FALCON/SUA var ≈ 1.67× ≪ 10×) |
| P2 RT | `p2_rt_gocue_coverage_v1/` | RT k4 eligible | **15/15** above 50% go-cue floor; **all** single `target_dir` (T4 undefined) |
| P2 proxy | `p2_general_carrier_proxy_pointer_v1/` | blocked | Points at existing Gate-A proxy |
| P3A | `p3_stage_a_baseline_receipt_v1/` | blocked | RS4/LS4 both < Z4 confirmed |
| P4 | `p4_harmonic_dispersion_strata_v1/` | **NO** | Monotonicity **failed**; mid/high strata have *negative* harmonic gain |
| P5a | `p5a_fail_closed_patch_v1/` | n/a | Production default `raise` |
| P5b | `p5b_confidence_gate_v1/` | **NO** | Gate lowers MSE but **hurts** PV-R² (no transduction); shrink ineffective |

## Headline numbers (this pass)

### P4 strata (pooled electrodes, M=30)

| stratum | n | mean dispersion | mean harmonic gain (MSE fund−harm) |
|---|---|---|---|
| low | 155 | 0.015 | **+0.096** |
| mid | 155 | 0.125 | **−0.181** |
| high | 154 | 0.512 | **−0.167** |

→ Opposite of the pooling-cancellation hypothesis on this SUA sorting. **Do not open harmonic GPU.**

### P1 closeout

- FALCON/SUA mean residual-var ratio ≈ **1.67** (floor for W3 reopen = 10) → **W3 stays closed**.
- M=30 bootstrap: mse A−B mean 0.034, 95% CI [−0.042, 0.126]; ac L2 mean 0.167, CI [0.131, 0.208].

### P5b

- Median-gate: MSE↓ but PV-R²↓ (~−0.29 SUA / −0.42 M2) → classic proxy failure.
- Shrink: MSE slightly worse, R² unchanged → **no FiLM GPU**.

### P2 RT

- Eligible for k4 segmentation: all 15 sessions.
- T4 remains undefined (single target_dir) on all 15 — k4 is the only directional carrier path.

## Still not CPU-solvable

1. Fill `fold2_seed43` (GPU cell) → freeze P0N thresholds.  
2. Any Hydra/Lightning training (P1C, P3 Stage B, P2/P6 GPU).  
3. Formal-test / EvalAI held-out.

## Do not do from this folder

1. Launch training or set a real `CUDA_VISIBLE_DEVICES`.  
2. Open SUA formal-test or EvalAI held-out NWBs.  
3. Freeze thresholds while `thresholds_frozen=false`.  
4. Restore extrapolated AC4 numbers `0.518` / `0.275`.  
5. Treat P4 bare deltas or P5b MSE-only wins as GPU authorization.
