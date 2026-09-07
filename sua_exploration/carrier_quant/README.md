# Carrier quantization candidate stack

Isolated, self-contained numerics / quantization plumbing for the H1 M=4 EB carrier
recalibration path. **Nothing in this package is imported by any active experiment.**

## Modules

| Module | Role |
| --- | --- |
| `reference.py` | O1/O2 reference implementation: `accumulate` / `solve_carrier`, numerically equivalent to `fit_frozen_carrier` |
| `folds.py` | R1 rotation fold and S1 SmoothQuant scale re-split (exact plan-time constant folds) |
| `penalty.py` | W1 ridge parameterizations (`fixed_prior`, `constant_per_sample`, optional diagonal) |
| `audit.py` | Q0 source-only numerical audit → JSON receipt |
| `fixedpoint.py` | Quantized-input simulation (int8/int16, int32 accumulators), sqrt-free LDLᵀ fp32 solve, B1 bit allocation |

## Fixed-point simulation (`fixedpoint.py`)

Honest integer-domain stress path for handoff Q2 (default: int8 inputs, int32 product
accumulators, fp32 17×17 LDL solve).

**What stays float (documented prep):**

- Rate centering and per-channel normalization `(rates - mean) / plan.scale` before
  quantization. This is reference/audit input prep, not on-chip integer math.

**What is integer:**

- `quantize_to_codes` → signed codes in `[-2^{b-1}, 2^{b-1}-1]` with explicit scales.
- `z = normalized @ pcs.T` via int8×int8 MACs into int32 (or int16-saturated stress).
- Intercept column uses fixed code `1` with scale `1.0`.
- `DtD`, `Dty`, `yty` via integer outer-product accumulation (`dtd_int`, etc.).
  Row projection MACs honor the int32 (or int16 stress) width; outer sums use int64
  accumulators under the default policy because calibration-length sums exceed int32
  range for realistic `z` magnitudes. The int16 stress path clips those sums to 16-bit.

**Scale algebra (dequantization at solve time only):**

- Column `i` of design has float value `code_i * s_i`.
- `DtD_float[i,j] ≈ DtD_int[i,j] * s_i * s_j`
- `Dty_float[i,j] ≈ Dty_int[i,j] * s_i * s_y`
- `yty_float[i,j] ≈ yty_int[i,j] * s_y * s_y`

where `s_i` is `1.0` for the intercept and `s_z = s_norm * s_pcs` for PC columns,
and `s_y` is the label tensor scale.

**fp32 LDL and outputs:**

- Normal equations are cast to fp32; `ldl_decompose` / `ldl_solve` stay in float32.
- Returned `beta`, `carrier`, `raw_carrier`, `weight`, etc. are rebuilt from that
  LDL beta (not from fp64 Cholesky).

**`int16_accumulator_stress=True`:**

- Same int8 inputs; projection MAC sums saturate to 16-bit signed range before
  forming the design matrix. DtD/Dty/yty outer sums still use int64 accumulators
  (clipping those sums to 16-bit breaks positive-semidefiniteness on realistic sizes).

**Remaining limitations:**

- Normalization before quantize is still float prep.
- Per-tensor (not per-channel) scales for the first honest simulation.
- Carrier post-beta steps (`G`, EB shrinkage) use fp64 after the fp32 LDL solve.

## Exact folds vs modeling choices

**Exact (plan-time constant folds; carrier unchanged in fp64):**

- O1 trace identity and inverse removal
- O2 sufficient-statistic refactor
- R1 orthogonal rotation of the `z` subspace (isotropic penalty only). **`beta` is not invariant** under rotation (`beta' = J^T beta`); carrier outputs that fold `pcs` are.
- S1 SmoothQuant-style re-split of `scale` / `pcs`

**Modeling choices (not evaluated here; plumbing + tests only):**

- W1 `fixed_prior` vs `constant_per_sample` penalty scaling. The two are related by an exact identity: evaluating `constant_per_sample(lambda, n_ref)` at sample count `n` is identical to `fixed_prior(lambda * n / n_ref)`. At `n = n_ref` this reduces to equality at the nominal penalty; for `n != n_ref` it pins the effective shrinkage `lambda_eff = lambda * n / n_ref` that `fixed_prior` would need to match.
- Diagonal ridge / learned whitening
- B1 reliability-weighted bit allocation (simulation only)
- Quantized int8 inputs and int16 accumulator stress test

## Fail-closed guards

1. **Date `19250101` forbidden.** `audit.assert_source_only_date` and `run_audit` raise on fold-0 target dates.
2. **R1 + non-isotropic penalty forbidden.** `apply_rotation_fold` raises if the penalty diagonal is not uniform.
3. **Audit receipt sets `source_only: true`.** No fold-0 routing use.

## Running tests

```bash
/home/xinyuan/miniconda3/envs/spint/bin/python3.10 -m pytest \
  /home/xinyuan/Work_host/SPINT/sua_exploration/carrier_quant/tests -q
```

## Running the Q0 audit later (real source data)

Only after active CPU/GPU jobs finish. Opens NWB via existing loaders; **source recordings only**.

```bash
/home/xinyuan/miniconda3/envs/spint/bin/python3.10 -m carrier_quant \
  --data-dir /path/to/nwb \
  --raw-receipt /path/to/raw_receipt.json \
  --eb-receipt /path/to/eb_receipt.json \
  --output /path/to/carrier_quant_q0_receipt.json
```

Run from `sua_exploration/` with `PYTHONPATH=.` or install in editable mode locally.

The receipt schema is `carrier_quant_q0_source_audit_v1`. It includes:

- `quantization_viable_gate.quantization_viable` — **false** if fp32-vs-fp64 per-channel carrier cosine falls below 0.9999 (conditioning must be fixed before quantization)
- Conditioning under each penalty parameterization
- Dynamic range of intermediates
- fp32 / bf16 / quantized cosines
- Merge-order sensitivity (fp accumulation is not associative)
- B1 uniform vs reliability-weighted bit allocation under a fixed budget

## W1 penalty contract

- **`fixed_prior`:** `system = DtD + lambda*I`, RHS `Dty`. Default path; bit-identical to `fit_frozen_carrier`.
- **`constant_per_sample`:** minimizes mean squared loss plus `lambda0 = lambda/n_ref` penalty. The solver uses normalized moments `(DtD/n, Dty/n)` against `system = DtD/n + lambda0*I`. Summed RSS is recovered from the original accumulators for the residual degrees-of-freedom denominator, and the sandwich `G` term carries an explicit `1/n` factor.
- **Exact identity (tested):** `constant_per_sample(lambda, n_ref)` at `n` equals `fixed_prior(lambda * n / n_ref)` on all carrier outputs. This is the precise statistical meaning of the W1 comparison: one arm holds effective shrinkage fixed as calibration length varies; the other is ordinary ridge with a penalty that scales linearly in `n`.

Synthetic tests use a geometric PC spectrum (default 100x max/min variance ratio), structured labels (`z @ coef + noise`, default SNR 10), and a data-derived `tau2` that targets the H1 Q1 EB-weight mean (0.6554) on the realized `projected_variance`. Fixture sanity tests guard spectrum ratio, label SNR, and EB weight mean/spread/range so these properties cannot silently regress.

## Discrepancies vs handoff / production code

- `fit_frozen_carrier` does not return `hat_trace`; the candidate exposes it via O1.
- Handoff section 5 Q0 lists `Dty/n`; the audit also reports `Dty` (unnormalized) for completeness.
- `__main__.py` uses existing `load_source_records` / `reconstruct_frozen_plan`; it does not modify any producer.

## What must not be claimed

Per `HANDOFF_CARRIER_OPERATOR_OPTIMIZATION_20260808.md` section 7 and 12:

- No streaming inference speedup
- No sealed-result changes from O1/O2 refactors
- SmoothQuant / QuaRot analogies are feasibility tooling, not contributions (except documenting exact R1 invariance and B1 allocation)
- Quantization is not viable until Q0 passes the fp32 gate on real source data
