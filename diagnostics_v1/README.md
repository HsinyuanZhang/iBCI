# Carrier construction and stability audit v1

`carrier_stability.py` is a CPU-only, score-free audit runner. It never trains a decoder and never reads a held-out test surface. For each task it records the estimator binding and either reconstructs deterministic 50%, 75%, and 100% support subsets (seeds 101–110) or reports the precise frozen-input blocker. The comparison is relative Frobenius error and flattened cosine against the full-support carrier.

The executable source authorities are deliberately task-specific:

| Task | Bound production construction | Current audit state |
| --- | --- | --- |
| M2 | B3S `native_e0_and_u`; MOVE-T4 `fit_move_t4` / `t4_from_trial_sums`; frozen dual-track cache | blocked: SHA-pinned B3S checkpoint is absent, and the cache omits raw bins/angles needed to refit MOVE-T4 |
| M1 | fold-local rSyn3: source-only rank-3 NNMF then per-unit ridge `[weights_1..3, intercept]` | runner implemented; uses the full four LOSO folds, with held-out sessions restricted to M10 support |
| 688 CO | ordinary causal T4 `compute_unit_side_features_uncached` on strict 27 train + 6 validation manifest | runner implemented; production public API only exposes chronological prefixes, not seeded selected-trial subsets |
| H1 | C2-CAL-1 B2 carrier then C2 materializer | blocked for support resampling: frozen payload contains only already-materialized M3 activity/carriers. Full M3 materialization is verified for 13 source sessions. |

The H1 B2 code presently rebuilds `q=12`, `lambda=10`; this conflicts with an older `q=16`, `lambda=100` description and is reported as a source-binding inconsistency rather than silently resolved.

Run an individual task only on CPU:

```bash
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  taskset -c 10,11 nice -n 10 /home/xinyuan/miniconda3/envs/spint/bin/python \
  diagnostics_v1/carrier_stability.py --task h1 --dest diagnostics_v1/results/carrier_stability_h1_v1
```

The completed receipts are:

- `results/carrier_stability_m2_v1/report.json`: M2 checkpoint blocker, cache and normalizer binding.
- `results/carrier_stability_h1_v1/report.json`: 13-session H1 M3 materializer construction receipt and resampling blocker.
