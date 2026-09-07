# Carrier Performance Program (P0N–P6)

**Status:** All meaningful **CPU** cells sealed. **No GPU cell is authorized.**  
**Live checklist:** [`docs/PROGRAM_STATUS.md`](docs/PROGRAM_STATUS.md)  
**Authority:** [`../docs/HANDOFF_CARRIER_PERFORMANCE_PROGRAM_20260805.md`](../docs/HANDOFF_CARRIER_PERFORMANCE_PROGRAM_20260805.md)

## Sealed CPU receipts (`results/`)

| ID | Receipt | Gate |
|---|---|---|
| P0N | `mua_noise_floor_v1/` | unfrozen (missing fold2) |
| P1A/B | `t4_estimator_equivalence_v1/`, `t4_falcon_estimator_b_vs_c_v1/` | unify estimators |
| P1 close | `p1_closeout_v1/` | W3 reopen **NO** |
| P2 RT | `p2_rt_gocue_coverage_v1/` | 15/15 eligible; T4 undefined |
| P3A | `p3_stage_a_baseline_receipt_v1/` | RS4/LS4 < Z4 |
| P4 | `p4_harmonic_dispersion_strata_v1/` | harmonic GPU **NO** |
| P5a | `p5a_fail_closed_patch_v1/` | production `raise` |
| P5b | `p5b_confidence_gate_v1/` | FiLM GPU **NO** |

## Tests

```bash
cd sua_exploration/carrier_perf_program
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src:.. \
  /home/xinyuan/miniconda3/envs/spint/bin/python -m pytest -q tests
```

Guarded executes need `CARRIER_PERF_REVIEWED_CPU=YES` and `--execute`.
