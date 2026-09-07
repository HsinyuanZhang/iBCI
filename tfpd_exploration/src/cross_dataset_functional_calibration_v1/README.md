# cross_dataset_functional_calibration_v1

CPU Stage0 namespace for the 2026-09-05 cross-dataset functional calibration
work order. This package freezes interfaces and runs explanation assays. It
does not launch GPU jobs, open hidden/EvalAI labels, or invent estimators.

## Owned files

- `plan.py` — frozen literals, hashes, ridge objectives, E1/E2 constants
- `contracts.py` — named estimators and P-operator revision status
- `inventory.py` — E0 authority inventory
- `h1_population_audit.py` — E1 H1 other-channel dependence
- `m1_dynamic_audit.py` — E2 M1 static vs ±100 ms lag encoding
- `coverage_audit.py` — E3 coverage / argument audit
- `basis.py` / `carrier_solver.py` / `parent_audit.py` — named P revision (CPU)
- `execute.py` — Stage0 writer for the timestamped result root
- `execute_p_revision.py` — writes `20260905_122000/` without touching Stage0

Do not edit `tfpd_exploration/src/m2_dual_track_v1/` from this namespace.

## Frozen estimators (do not replace)

| Dataset | Named estimator | Direction |
|---|---|---|
| H1 | `fit_deployment_carrier` on first 3 TrialNum, source-frozen PCA/ridge/U/EB | backward decoder-weight |
| M1 | source-frozen rSyn3 + `fit_unit_ridge` (`[w1,w2,w3,b]`) | forward encoding |
| M1 E2 diagnostic | same ridge on `concat(z(t-100ms), z(t), z(t+100ms))` | forward encoding, not a P basis |

Undocumented bases, lambdas, PCA ranks, or parent swaps are out of contract.

## Result root

Stage0: `tfpd_exploration/results/cross_dataset_functional_calibration_v1/20260905_113700/`
P revision: `tfpd_exploration/results/cross_dataset_functional_calibration_v1/20260905_122000/`

## Execution

```
PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES= \
  /home/xinyuan/miniconda3/envs/spint/bin/python \
  tfpd_exploration/scripts/run_cross_dataset_functional_calibration_v1.py --dry-run

PYTHONNOUSERSITE=1 CUDA_VISIBLE_DEVICES= \
  /home/xinyuan/miniconda3/envs/spint/bin/python \
  tfpd_exploration/scripts/run_cross_dataset_functional_calibration_v1.py --execute
```

`--bind-p-operator` writes the named revision root. `--execute-gpu` is rejected.
