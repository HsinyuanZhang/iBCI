"""Fixed-point simulation harness and B1 reliability-weighted bit allocation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .penalty import PenaltySpec, build_system_matrix
from .reference import AccumulationState, _rss_summed, _solver_moments

EPS = 1.0e-12


@dataclass(frozen=True)
class QuantConfig:
  input_bits: int = 8
  accumulator_bits: int = 32
  per_channel_scales: np.ndarray | None = None


def _q_bounds(bits: int) -> tuple[int, int]:
  qmax = (1 << (bits - 1)) - 1
  qmin = -(1 << (bits - 1))
  return qmin, qmax


def _tensor_scale(values: np.ndarray, bits: int) -> float:
  qmax = (1 << (bits - 1)) - 1
  return float(np.max(np.abs(values))) / max(qmax, 1)


def quantize_to_codes(
  values: np.ndarray,
  bits: int,
  scale: float | np.ndarray,
) -> np.ndarray:
  """Quantize to signed integer codes in ``[-2^{b-1}, 2^{b-1}-1]``."""
  qmin, qmax = _q_bounds(bits)
  values64 = np.asarray(values, dtype=np.float64)
  scale_arr = np.asarray(scale, dtype=np.float64)
  if scale_arr.ndim == 0:
    scaled = np.round(values64 / float(scale_arr))
    return np.clip(scaled, qmin, qmax).astype(np.int32)
  codes = np.empty(values64.shape, dtype=np.int32)
  for j in range(values64.shape[-1]):
    scaled = np.round(values64[..., j] / float(scale_arr[j]))
    codes[..., j] = np.clip(scaled, qmin, qmax)
  return codes


def dequantize(codes: np.ndarray, scale: float | np.ndarray) -> np.ndarray:
  """Map integer codes back to float using an explicit per-tensor or per-channel scale."""
  scale_arr = np.asarray(scale, dtype=np.float64)
  codes64 = np.asarray(codes, dtype=np.float64)
  if scale_arr.ndim == 0:
    return codes64 * float(scale_arr)
  out = np.empty(codes64.shape, dtype=np.float64)
  for j in range(codes64.shape[-1]):
    out[..., j] = codes64[..., j] * float(scale_arr[j])
  return out


def quantize_tensor(values: np.ndarray, bits: int, scale: float | np.ndarray) -> np.ndarray:
  """Dequantized grid (legacy helper); prefer ``quantize_to_codes`` + ``dequantize``."""
  return dequantize(quantize_to_codes(values, bits, scale), scale)


def simulate_bfloat16(values: np.ndarray) -> np.ndarray:
  """Simulate bfloat16 by zeroing the low 16 mantissa bits of float32."""
  x = np.asarray(values, dtype=np.float32)
  x32 = x.view(np.uint32)
  x32 = (x32 & np.uint32(0xFFFF0000)).astype(np.uint32)
  return x32.view(np.float32).astype(np.float64)


def ldl_decompose(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  """Sqrt-free LDL^T factorization for symmetric A (A = L @ diag(D) @ L.T) in float32."""
  work = np.asarray(a, dtype=np.float32, order="C").copy()
  n = work.shape[0]
  l = np.eye(n, dtype=np.float32)
  d = np.zeros(n, dtype=np.float32)
  for j in range(n):
    d[j] = work[j, j]
    if abs(float(d[j])) < 1e-30:
      d[j] = np.float32(1e-30)
    for i in range(j + 1, n):
      work[i, j] = work[i, j] / d[j]
      l[i, j] = work[i, j]
    for k in range(j + 1, n):
      for i in range(k, n):
        work[i, k] -= work[i, j] * d[j] * work[k, j]
  return l, d


def _ldl_solve_ld(l: np.ndarray, d: np.ndarray, b: np.ndarray) -> np.ndarray:
  n = l.shape[0]
  b32 = np.asarray(b, dtype=np.float32).reshape(-1)
  y = np.zeros(n, dtype=np.float32)
  for i in range(n):
    y[i] = b32[i] - np.dot(l[i, :i], y[:i])
  z = y / d
  x = np.zeros(n, dtype=np.float32)
  for i in reversed(range(n)):
    x[i] = z[i] - np.dot(l[i + 1 :, i], x[i + 1 :])
  return x


def ldl_solve_factored(l: np.ndarray, d: np.ndarray, b: np.ndarray) -> np.ndarray:
  """Solve A x = b given sqrt-free LDL factors (float32 arithmetic)."""
  if np.asarray(b).ndim == 1:
    return _ldl_solve_ld(l, d, b)
  b32 = np.asarray(b, dtype=np.float32)
  cols = [_ldl_solve_ld(l, d, b32[:, j]) for j in range(b32.shape[1])]
  return np.column_stack(cols)


def ldl_solve(a: np.ndarray, b: np.ndarray) -> np.ndarray:
  """Solve A x = b with explicit sqrt-free LDL^T in float32 arithmetic."""
  l, d = ldl_decompose(a)
  return ldl_solve_factored(l, d, b)


def _saturate_accumulator(value: int, accumulator_bits: int) -> int:
  if accumulator_bits >= 64:
    return value
  acc_max = (1 << (accumulator_bits - 1)) - 1
  acc_min = -(1 << (accumulator_bits - 1))
  if value > acc_max:
    return acc_max
  if value < acc_min:
    return acc_min
  return value


def _int_accumulate_matmul(
  lhs_codes: np.ndarray,
  rhs_codes: np.ndarray,
  *,
  accumulator_bits: int,
) -> np.ndarray:
  """Integer MAC for ``lhs @ rhs.T`` with explicit int codes (no implicit scales)."""
  lhs = np.asarray(lhs_codes, dtype=np.int64)
  rhs = np.asarray(rhs_codes, dtype=np.int64)
  out = np.zeros((lhs.shape[0], rhs.shape[0]), dtype=np.int64)
  for b in range(lhs.shape[0]):
    for j in range(rhs.shape[0]):
      acc = 0
      for k in range(lhs.shape[1]):
        acc += int(lhs[b, k]) * int(rhs[j, k])
      out[b, j] = _saturate_accumulator(acc, accumulator_bits)
  return out


def _int_accumulate_outer(
  left_codes: np.ndarray,
  right_codes: np.ndarray,
  *,
  accumulator_bits: int,
) -> np.ndarray:
  """Integer outer-product sum: ``sum_b left[b,i] * right[b,j]``.

  Row-wise projection MACs use the requested ``accumulator_bits`` width. Sufficient
  statistics sum many large projected codes and therefore accumulate in int64 by
  default; only the int16 stress path clips the final sums.
  """
  left = np.asarray(left_codes, dtype=np.int64)
  right = np.asarray(right_codes, dtype=np.int64)
  n = left.shape[0]
  out = np.zeros((left.shape[1], right.shape[1]), dtype=np.int64)
  for b in range(n):
    for i in range(left.shape[1]):
      li = int(left[b, i])
      for j in range(right.shape[1]):
        out[i, j] += li * int(right[b, j])
  if accumulator_bits <= 16:
    acc_max = (1 << 15) - 1
    acc_min = -(1 << 15)
    return np.clip(out, acc_min, acc_max)
  return out


def _dequant_dtd(dtd_int: np.ndarray, scales: np.ndarray) -> np.ndarray:
  s = np.asarray(scales, dtype=np.float64)
  return dtd_int.astype(np.float64) * s[:, None] * s[None, :]


def _dequant_dty(dty_int: np.ndarray, design_scales: np.ndarray, label_scale: float) -> np.ndarray:
  s = np.asarray(design_scales, dtype=np.float64)
  return dty_int.astype(np.float64) * s[:, None] * float(label_scale)


def _dequant_yty(yty_int: np.ndarray, label_scale: float) -> np.ndarray:
  s = float(label_scale)
  return yty_int.astype(np.float64) * s * s


def _ldl_hat_trace(l: np.ndarray, d: np.ndarray, dtd_view: np.ndarray) -> float:
  solved = ldl_solve_factored(l, d, np.asarray(dtd_view, dtype=np.float32))
  return float(np.trace(solved))


def _ldl_compute_g(l: np.ndarray, d: np.ndarray, dtd_view: np.ndarray) -> np.ndarray:
  temp = ldl_solve_factored(l, d, np.asarray(dtd_view, dtype=np.float32))
  g_t = ldl_solve_factored(l, d, temp.T.astype(np.float32))
  g = (g_t.T + g_t) / 2.0
  return g.astype(np.float64)


def _carrier_from_beta(
  beta: np.ndarray,
  state: AccumulationState,
  plan: Any,
  *,
  penalty: PenaltySpec,
  l: np.ndarray,
  d: np.ndarray,
  dtd_view: np.ndarray,
  g_scale: float,
) -> dict[str, Any]:
  """Post-beta carrier construction shared by the quantized LDL path."""
  beta64 = np.asarray(beta, dtype=np.float64)
  rss = _rss_summed(beta64, state)
  hat_trace = _ldl_hat_trace(l, d, dtd_view.astype(np.float32))
  denominator = float(state.n - hat_trace)
  if not np.isfinite(denominator) or denominator <= EPS:
    raise ValueError("ridge residual covariance degrees of freedom undefined")

  sigma2 = rss / denominator
  g = _ldl_compute_g(l, d, dtd_view.astype(np.float32)) * g_scale

  raw_rows = (plan.pcs[: plan.q].T @ beta64[1:]) / plan.scale[:, None]
  raw_carrier = raw_rows @ plan.U
  projection = plan.pcs[: plan.q].T
  channel_factor = ((projection @ g[1:, 1:]) * projection).sum(axis=1) / np.square(plan.scale)
  projected_covariance = plan.U.T @ np.diag(sigma2) @ plan.U
  projected_variance = channel_factor * np.trace(projected_covariance) / 4.0

  if (
    np.any(~np.isfinite(projected_variance))
    or np.any(projected_variance < 0.0)
    or not np.isfinite(plan.tau2)
    or plan.tau2 <= EPS
  ):
    raise ValueError("analytic EB projected variance/prior undefined")

  weight = plan.tau2 / (plan.tau2 + projected_variance)
  carrier = plan.mu[None, :] + weight[:, None] * (raw_carrier - plan.mu[None, :])
  return {
    "carrier": np.asarray(carrier, dtype=np.float64),
    "raw_carrier": np.asarray(raw_carrier, dtype=np.float64),
    "raw_rows": np.asarray(raw_rows, dtype=np.float64),
    "beta": beta64,
    "G": np.asarray(g, dtype=np.float64),
    "sigma2": np.asarray(sigma2, dtype=np.float64),
    "projected_variance": np.asarray(projected_variance, dtype=np.float64),
    "weight": np.asarray(weight, dtype=np.float64),
    "hat_trace": np.asarray(hat_trace, dtype=np.float64),
  }


def _integer_sufficient_statistics(
  design_codes: np.ndarray,
  label_codes: np.ndarray,
  *,
  design_scales: np.ndarray,
  label_scale: float,
  accumulator_bits: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, AccumulationState]:
  """Accumulate DtD/Dty/yty with integer MACs; return int accumulators and float state."""
  dtd_int = _int_accumulate_outer(design_codes, design_codes, accumulator_bits=accumulator_bits)
  dty_int = _int_accumulate_outer(design_codes, label_codes, accumulator_bits=accumulator_bits)
  yty_int = _int_accumulate_outer(label_codes, label_codes, accumulator_bits=accumulator_bits)

  dtd = _dequant_dtd(dtd_int, design_scales)
  dty = _dequant_dty(dty_int, design_scales, label_scale)
  yty = _dequant_yty(yty_int, label_scale)

  design_dim = design_codes.shape[1]
  state = AccumulationState(design_dim=design_dim)
  state.n = int(design_codes.shape[0])
  state.dtd = dtd
  state.dty = dty
  state.yty = yty
  return dtd_int, dty_int, yty_int, state


def simulate_quantized_carrier(
  rates: np.ndarray,
  labels: np.ndarray,
  plan: Any,
  *,
  config: QuantConfig | None = None,
  penalty: PenaltySpec | None = None,
  int16_accumulator_stress: bool = False,
) -> dict[str, Any]:
  """Quantized-input carrier simulation; 17x17 solve stays float32 LDL^T.

  Integer domain:
  - Normalized rates and folded PCs are quantized to int8 codes (float prep documented).
  - ``z = normalized @ pcs.T`` uses int8 x int8 -> int32 MACs.
  - DtD/Dty/yty are accumulated as integer outer products, then dequantized via scales.
  - LDL solve and carrier rebuild use the quantized-path beta (not fp64 Cholesky).
  """
  config = config or QuantConfig()
  penalty = penalty or PenaltySpec.fixed_prior(plan.ridge_lambda)
  rates64 = np.asarray(rates, dtype=np.float64)
  labels64 = np.asarray(labels, dtype=np.float64)
  bits = config.input_bits
  acc_bits = 16 if int16_accumulator_stress else config.accumulator_bits

  plan_scale = np.maximum(np.asarray(plan.scale, dtype=np.float64), 1e-12)
  normalized = (rates64 - plan.mean[None, :]) / plan_scale[None, :]
  norm_scale = _tensor_scale(normalized, bits)
  norm_codes = quantize_to_codes(normalized, bits, norm_scale)

  pcs = np.asarray(plan.pcs[: plan.q], dtype=np.float64)
  pcs_scale = _tensor_scale(pcs, bits)
  pcs_codes = quantize_to_codes(pcs, bits, pcs_scale)

  z_codes = _int_accumulate_matmul(norm_codes, pcs_codes, accumulator_bits=acc_bits)
  z_scale = norm_scale * pcs_scale

  intercept_codes = np.ones((norm_codes.shape[0], 1), dtype=np.int32)
  design_codes = np.column_stack((intercept_codes, z_codes))
  design_scales = np.concatenate(([1.0], np.full(plan.q, z_scale, dtype=np.float64)))

  label_scale = _tensor_scale(labels64, bits)
  label_codes = quantize_to_codes(labels64, bits, label_scale)

  # Sufficient-statistic outer sums use int64 accumulators; int16 stress applies to z MACs.
  stat_acc_bits = 64
  dtd_int, dty_int, yty_int, state = _integer_sufficient_statistics(
    design_codes,
    label_codes,
    design_scales=design_scales,
    label_scale=label_scale,
    accumulator_bits=stat_acc_bits,
  )

  if state.n < 2:
    raise ValueError("accumulation state must contain at least two samples")

  dtd_view, dty_view, g_scale = _solver_moments(state, penalty)
  system32 = build_system_matrix(state.dtd, state.n, penalty).astype(np.float32)
  l32, d32 = ldl_decompose(system32)
  beta32 = ldl_solve_factored(l32, d32, dty_view.astype(np.float32))

  outputs = _carrier_from_beta(
    beta32,
    state,
    plan,
    penalty=penalty,
    l=l32,
    d=d32,
    dtd_view=dtd_view,
    g_scale=g_scale,
  )

  return {
    **outputs,
    "system": np.asarray(system32, dtype=np.float64),
    "n": state.n,
    "DtD": np.asarray(state.dtd, dtype=np.float64),
    "Dty": np.asarray(state.dty, dtype=np.float64),
    "yty": np.asarray(state.yty, dtype=np.float64),
    "dtd_int": dtd_int,
    "dty_int": dty_int,
    "yty_int": yty_int,
    "design_scales": design_scales,
    "label_scale": label_scale,
    "norm_codes": norm_codes,
    "pcs_codes": pcs_codes,
    "label_codes": label_codes,
    "z_scale": z_scale,
    "ldl_L": l32,
    "ldl_D": d32,
    "accumulator_bits": acc_bits,
    "int16_stress_test": bool(int16_accumulator_stress),
  }


def allocate_bits_uniform(num_channels: int, total_bits: int, min_bits: int = 4) -> np.ndarray:
  if total_bits < num_channels * min_bits:
    raise ValueError("total bit budget too small for minimum per-channel allocation")
  base = total_bits // num_channels
  rem = total_bits % num_channels
  alloc = np.full(num_channels, base, dtype=np.int64)
  alloc[:rem] += 1
  return np.maximum(alloc, min_bits)


def allocate_bits_reliability_weighted(
  weights: np.ndarray,
  total_bits: int,
  *,
  min_bits: int = 4,
) -> np.ndarray:
  """B1: equalize expected output error under carrier_i = mu + w_i * (raw_i - mu)."""
  w = np.asarray(weights, dtype=np.float64).reshape(-1)
  n = w.size
  if total_bits < n * min_bits:
    raise ValueError("total bit budget too small for minimum per-channel allocation")
  w = np.maximum(w, 1e-12)
  w = w / w.sum()
  remaining = total_bits - n * min_bits
  if remaining <= 0:
    return np.full(n, min_bits, dtype=np.int64)
  raw = w * remaining
  alloc = np.full(n, min_bits, dtype=np.int64) + np.floor(raw).astype(np.int64)
  deficit = total_bits - int(alloc.sum())
  frac = raw - np.floor(raw)
  order = np.argsort(-frac)
  for idx in order[:deficit]:
    alloc[idx] += 1
  return alloc


def rms_normalize_carrier(carrier: np.ndarray) -> np.ndarray:
  rms = float(np.sqrt(np.mean(np.square(carrier))))
  if rms <= 0:
    raise ValueError("carrier RMS must be positive for normalization")
  return carrier / rms
