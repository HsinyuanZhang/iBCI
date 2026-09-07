"""Q0 source-only numerical audit (synthetic validation; real data runnable later)."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .fixedpoint import QuantConfig, allocate_bits_reliability_weighted, allocate_bits_uniform, simulate_quantized_carrier
from .penalty import PenaltySpec
from .reference import AccumulationState, accumulate, empty_state, fit_carrier_batch, solve_carrier

FOLD0_DATE_FORBIDDEN = "19250101"
AUDIT_SCHEMA = "carrier_quant_q0_source_audit_v1"
FP32_COSINE_GATE = 0.9999


@dataclass(frozen=True)
class AuditRecording:
  session_name: str
  date: str
  rates: np.ndarray
  labels: np.ndarray
  input_sha256: str = ""


def _canonical_json_bytes(value: Any) -> bytes:
  return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _array_sha256(value: np.ndarray) -> str:
  arr = np.ascontiguousarray(value)
  digest = hashlib.sha256()
  digest.update(str(arr.dtype).encode("ascii"))
  digest.update(json.dumps(list(arr.shape), separators=(",", ":")).encode("ascii"))
  digest.update(arr.tobytes())
  return digest.hexdigest()


def assert_source_only_date(date: str) -> None:
  if date == FOLD0_DATE_FORBIDDEN:
    raise ValueError(
      f"fail-closed: date {FOLD0_DATE_FORBIDDEN} is fold-0 target and forbidden for source-only audit"
    )


def _dynamic_range_stats(values: np.ndarray) -> dict[str, float]:
  flat = np.asarray(values, dtype=np.float64).reshape(-1)
  finite = flat[np.isfinite(flat)]
  if finite.size == 0:
    return {"min": float("nan"), "max": float("nan"), "rms": float("nan"), "q01": float("nan"), "q50": float("nan"), "q99": float("nan")}
  return {
    "min": float(np.min(finite)),
    "max": float(np.max(finite)),
    "rms": float(np.sqrt(np.mean(np.square(finite)))),
    "q01": float(np.quantile(finite, 0.01)),
    "q50": float(np.quantile(finite, 0.50)),
    "q99": float(np.quantile(finite, 0.99)),
  }


def _per_channel_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
  a64 = np.asarray(a, dtype=np.float64)
  b64 = np.asarray(b, dtype=np.float64)
  num = np.sum(a64 * b64, axis=1)
  den = np.linalg.norm(a64, axis=1) * np.linalg.norm(b64, axis=1)
  den = np.maximum(den, 1e-30)
  return num / den


def _merge_order_spread(
  rates: np.ndarray,
  labels: np.ndarray,
  plan: Any,
  *,
  penalty: PenaltySpec,
  orders: Sequence[str],
) -> dict[str, Any]:
  carriers = []
  for order_name in orders:
    perm = _block_permutation(rates.shape[0], order_name)
    state = empty_state(plan.q + 1)
    for idx in perm:
      state = accumulate(state, rates[idx : idx + 1], labels[idx : idx + 1], plan=plan)
    carriers.append(solve_carrier(state, plan, penalty=penalty, num_neurons=rates.shape[1])["carrier"])
  stacked = np.stack(carriers, axis=0)
  ref = stacked[0]
  cosines = [_per_channel_cosine(ref, c).mean() for c in stacked[1:]]
  spread = np.max(np.abs(stacked - ref), axis=(1, 2))
  return {
    "orders": list(orders),
    "max_abs_spread_per_order": [float(x) for x in spread[1:]],
    "mean_cosine_vs_canonical": [float(x) for x in cosines],
    "worst_max_abs": float(np.max(spread[1:])) if len(spread) > 1 else 0.0,
  }


def _block_permutation(n_blocks: int, order_name: str) -> np.ndarray:
  idx = np.arange(n_blocks)
  if order_name == "canonical":
    return idx
  if order_name == "reverse":
    return idx[::-1]
  if order_name == "even_odd":
    return np.concatenate([idx[::2], idx[1::2]])
  if order_name == "odd_even":
    return np.concatenate([idx[1::2], idx[::2]])
  raise ValueError(f"unknown merge order: {order_name}")


def audit_recording(
  recording: AuditRecording,
  plan: Any,
  *,
  penalty_specs: Mapping[str, PenaltySpec] | None = None,
) -> dict[str, Any]:
  assert_source_only_date(recording.date)
  penalty_specs = penalty_specs or {
    "fixed_prior": PenaltySpec.fixed_prior(plan.ridge_lambda),
    "constant_per_sample": PenaltySpec.constant_per_sample(plan.ridge_lambda, n_ref=627.0),
  }

  rates = np.asarray(recording.rates, dtype=np.float64)
  labels = np.asarray(recording.labels, dtype=np.float64)
  base = fit_carrier_batch(rates, labels, plan, penalty=penalty_specs["fixed_prior"])
  z = ((rates - plan.mean[None, :]) / plan.scale[None, :]) @ plan.pcs[: plan.q].T

  fp32_result = fit_carrier_batch(rates.astype(np.float32), labels.astype(np.float32), plan)
  from .fixedpoint import simulate_bfloat16

  bf16_rates = simulate_bfloat16(rates)
  bf16_labels = simulate_bfloat16(labels)
  bf16_result = fit_carrier_batch(bf16_rates, bf16_labels, plan)
  quant_result = simulate_quantized_carrier(rates, labels, plan)

  carrier_fp64 = base["carrier"]
  carrier_fp32 = fp32_result["carrier"]
  cosine_fp32 = _per_channel_cosine(carrier_fp64, carrier_fp32)
  cosine_bf16 = _per_channel_cosine(carrier_fp64, bf16_result["carrier"])
  cosine_quant = _per_channel_cosine(carrier_fp64, quant_result["carrier"])

  rms = float(np.sqrt(np.mean(np.square(carrier_fp64))))
  normalized = carrier_fp64 / max(rms, 1e-30)

  bit_uniform = allocate_bits_uniform(carrier_fp64.shape[0], total_bits=8 * carrier_fp64.shape[0])
  bit_weighted = allocate_bits_reliability_weighted(base["weight"], total_bits=8 * carrier_fp64.shape[0])

  conditioning = {}
  for name, spec in penalty_specs.items():
    state = AccumulationState(
      design_dim=base["DtD"].shape[0],
      n=base["n"],
      dtd=base["DtD"],
      dty=base["Dty"],
      yty=base["yty"],
    )
    from .penalty import build_system_matrix

    system = build_system_matrix(state.dtd, state.n, spec)
    conditioning[name] = {
      "condition_number": float(np.linalg.cond(system)),
      "system_dynamic_range": _dynamic_range_stats(system),
    }

  merge_orders = ("canonical", "reverse", "even_odd", "odd_even")
  merge_sensitivity = _merge_order_spread(
    rates,
    labels,
    plan,
    penalty=penalty_specs["fixed_prior"],
    orders=merge_orders,
  )

  quantization_viable = bool(np.min(cosine_fp32) >= FP32_COSINE_GATE)

  return {
    "session_name": recording.session_name,
    "date": recording.date,
    "input_sha256": recording.input_sha256 or _array_sha256(rates),
    "conditioning": conditioning,
    "dynamic_range": {
      "z": _dynamic_range_stats(z),
      "DtD": _dynamic_range_stats(base["DtD"]),
      "DtD_over_n": _dynamic_range_stats(base["DtD"] / base["n"]),
      "Dty": _dynamic_range_stats(base["Dty"]),
      "beta": _dynamic_range_stats(base["beta"]),
      "raw_rows": _dynamic_range_stats(base["raw_rows"]),
      "raw_carrier": _dynamic_range_stats(base["raw_carrier"]),
      "projected_variance": _dynamic_range_stats(base["projected_variance"]),
      "weight": _dynamic_range_stats(base["weight"]),
      "carrier_normalized": _dynamic_range_stats(normalized),
    },
    "precision_cosine": {
      "float32": {
        "per_channel": cosine_fp32.tolist(),
        "min": float(np.min(cosine_fp32)),
        "mean": float(np.mean(cosine_fp32)),
      },
      "bfloat16": {
        "per_channel": cosine_bf16.tolist(),
        "min": float(np.min(cosine_bf16)),
        "mean": float(np.mean(cosine_bf16)),
      },
      "quantized_int8_int32_accum": {
        "per_channel": cosine_quant.tolist(),
        "min": float(np.min(cosine_quant)),
        "mean": float(np.mean(cosine_quant)),
      },
    },
    "quantization_viable_gate": {
      "fp32_cosine_min_threshold": FP32_COSINE_GATE,
      "fp32_cosine_min_observed": float(np.min(cosine_fp32)),
      "quantization_viable": quantization_viable,
      "interpretation": (
        "If fp32-vs-fp64 carrier cosine is materially degraded, fix conditioning before quantization."
      ),
    },
    "merge_order_sensitivity": merge_sensitivity,
    "bit_allocation_b1": {
      "total_bits": int(8 * carrier_fp64.shape[0]),
      "uniform": bit_uniform.tolist(),
      "reliability_weighted": bit_weighted.tolist(),
    },
  }


def run_audit(
  recordings: Sequence[AuditRecording],
  plan: Any,
  *,
  output_path: str | Path | None = None,
  penalty_specs: Mapping[str, PenaltySpec] | None = None,
) -> dict[str, Any]:
  for rec in recordings:
    assert_source_only_date(rec.date)

  per_recording = [audit_recording(rec, plan, penalty_specs=penalty_specs) for rec in recordings]
  input_hashes = [rec.input_sha256 or _array_sha256(rec.rates) for rec in recordings]

  receipt: dict[str, Any] = {
    "schema": AUDIT_SCHEMA,
    "source_only": True,
    "fold0_date_forbidden": FOLD0_DATE_FORBIDDEN,
    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    "input_sha256": input_hashes,
    "recordings": per_recording,
    "summary": {
      "num_recordings": len(recordings),
      "all_quantization_viable": all(r["quantization_viable_gate"]["quantization_viable"] for r in per_recording),
      "worst_fp32_cosine_min": float(min(r["precision_cosine"]["float32"]["min"] for r in per_recording)) if per_recording else float("nan"),
    },
  }

  if output_path is not None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")

  return receipt
