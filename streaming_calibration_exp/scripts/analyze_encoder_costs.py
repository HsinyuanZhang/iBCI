"""Hardware-cost analysis for B0-B11 calibration encoders.

Computes parameter count, MAC/session, peak state, and hardware-friendliness
flags for each variant. Pure-numpy (no torch required) — derives from the
architecture specification at the canonical working point:

  N=96, T=100, M=33, W=50, D=64 (or D=32 for B10)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import Dict, List


# Canonical working point
N = 96
T = 100
M = 33
W = 50
D = 64


@dataclass
class VariantCost:
  variant: str
  description: str
  parameter_count: int
  trainable_parameter_count: int
  weight_bytes_int8: int
  mac_per_trial: int
  mac_per_session: int
  peak_state_bytes: int
  trial_buffer_bytes: int
  requires_cubic_interp: bool
  requires_general_multiplier: bool
  requires_divider: bool
  multiplier_free_prepool: bool


def affine_params(in_dim: int, layers: List[int]) -> int:
  """Total params for an affine stack specified by layer widths."""
  total = 0
  cur = in_dim
  for w in layers:
    total += cur * w + w  # weight + bias
    cur = w
  return total


def affine_mac(in_dim: int, layers: List[int]) -> int:
  total = 0
  cur = in_dim
  for w in layers:
    total += cur * w
    cur = w
  return total


# ---------------------------------------------------------------------------
# B0 / B1 — teacher replica (H=512 in canonical M2 config)
# ---------------------------------------------------------------------------

def b0_cost() -> VariantCost:
  H = 512
  # fc_id_in: LazyLinear(T->H), H->H, H->H (num_id_layers=3)
  fc_id_in_layers = [H, H, H]  # 3 affine layers, first takes T=100
  fc_id_in_params = affine_params(T, fc_id_in_layers)
  fc_id_in_mac = affine_mac(T, fc_id_in_layers)
  # fc_id_out: H->H, H->H, H->W (num_id_layers=3)
  fc_id_out_layers = [H, H, W]
  fc_id_out_params = affine_params(H, fc_id_out_layers)
  fc_id_out_mac = affine_mac(H, fc_id_out_layers)
  params = fc_id_in_params + fc_id_out_params
  mac_trial = N * fc_id_in_mac
  mac_session = M * mac_trial + N * fc_id_out_mac
  return VariantCost(
    variant="B0",
    description="Teacher MLP replica (T->H->H->H, H->H->H->W)",
    parameter_count=params,
    trainable_parameter_count=params,
    weight_bytes_int8=params,
    mac_per_trial=mac_trial,
    mac_per_session=mac_session,
    peak_state_bytes=N * H * 4,
    trial_buffer_bytes=T * N * 4,
    requires_cubic_interp=True,
    requires_general_multiplier=True,
    requires_divider=True,
    multiplier_free_prepool=False,
  )


# ---------------------------------------------------------------------------
# B2 — LatePool
# ---------------------------------------------------------------------------

def b2_cost(D: int = 512) -> VariantCost:
  # fc_id_in: T->D->D->D (3 layers)
  fc_id_in_layers = [D, D, D]
  fc_id_in_params = affine_params(T, fc_id_in_layers)
  fc_id_in_mac = affine_mac(T, fc_id_in_layers)
  # fc_id_out: D->D->D->W (3 layers)
  fc_id_out_layers = [D, D, W]
  fc_id_out_params = affine_params(D, fc_id_out_layers)
  fc_id_out_mac = affine_mac(D, fc_id_out_layers)
  params = fc_id_in_params + fc_id_out_params
  mac_trial = N * fc_id_in_mac
  mac_session = M * mac_trial + N * fc_id_out_mac
  return VariantCost(
    variant=f"B2-D{D}",
    description="LatePool (per-trial full MLP, pool H)",
    parameter_count=params,
    trainable_parameter_count=params,
    weight_bytes_int8=params,
    mac_per_trial=mac_trial,
    mac_per_session=mac_session,
    peak_state_bytes=N * D * 4 + T * N * 4,
    trial_buffer_bytes=T * N * 4,
    requires_cubic_interp=True,
    requires_general_multiplier=True,
    requires_divider=True,
    multiplier_free_prepool=False,
  )


# ---------------------------------------------------------------------------
# B3 — EarlyPool
# ---------------------------------------------------------------------------

def b3_cost(D: int = D) -> VariantCost:
  # pre_pool: Linear(T->D) + ReLU
  pre_pool_params = T * D + D
  pre_pool_mac = T * D
  # post_pool: 3-layer D->D->D->W
  post_pool_layers = [D, D, W]
  post_pool_params = affine_params(D, post_pool_layers)
  post_pool_mac = affine_mac(D, post_pool_layers)
  params = pre_pool_params + post_pool_params
  mac_trial = N * pre_pool_mac
  mac_session = M * mac_trial + N * post_pool_mac
  return VariantCost(
    variant=f"B3-D{D}",
    description="EarlyPool (T->D-ReLU, mean, 3-layer MLP)",
    parameter_count=params,
    trainable_parameter_count=params,
    weight_bytes_int8=params,
    mac_per_trial=mac_trial,
    mac_per_session=mac_session,
    peak_state_bytes=N * D * 4 + T * N * 4,
    trial_buffer_bytes=T * N * 4,
    requires_cubic_interp=True,
    requires_general_multiplier=True,
    requires_divider=True,
    multiplier_free_prepool=False,
  )


# ---------------------------------------------------------------------------
# B4 — Stats Streaming
# ---------------------------------------------------------------------------

def b4_cost(D: int = D) -> VariantCost:
  # feature_proj: Linear(4->D) + ReLU
  feat_proj_params = 4 * D + D
  feat_proj_mac = 4 * D
  # post_pool: 2-layer D->D->W (per `_build_affine_stack(D, D, 2, W)`)
  post_pool_layers = [D, W]
  post_pool_params = affine_params(D, post_pool_layers)
  post_pool_mac = affine_mac(D, post_pool_layers)
  params = feat_proj_params + post_pool_params
  mac_trial = N * feat_proj_mac
  mac_session = M * mac_trial + N * post_pool_mac
  return VariantCost(
    variant=f"B4-D{D}",
    description="Stats (mean,var,max,last per bin, 4->D, 2-layer MLP)",
    parameter_count=params,
    trainable_parameter_count=params,
    weight_bytes_int8=params,
    mac_per_trial=mac_trial,
    mac_per_session=mac_session,
    peak_state_bytes=N * (D + 16) * 4,
    trial_buffer_bytes=0,
    requires_cubic_interp=False,
    requires_general_multiplier=True,
    requires_divider=True,
    multiplier_free_prepool=True,  # only accumulators
  )


# ---------------------------------------------------------------------------
# B5 — EMA Streaming
# ---------------------------------------------------------------------------

def b5_cost(R: int = 4, D: int = D) -> VariantCost:
  # feature_proj: Linear(2R->D) + ReLU
  feat_proj_params = 2 * R * D + D
  # EMA updates: per-bin shift-add (R taps), then sum
  ema_mac = T * R  # shift-add per bin (counted as MAC for fair comparison)
  feat_proj_mac = 2 * R * D
  # post_pool: 2-layer D->D->W
  post_pool_layers = [D, W]
  post_pool_params = affine_params(D, post_pool_layers)
  post_pool_mac = affine_mac(D, post_pool_layers)
  params = feat_proj_params + post_pool_params
  mac_trial = N * (ema_mac + feat_proj_mac)
  mac_session = M * mac_trial + N * post_pool_mac
  return VariantCost(
    variant=f"B5-R{R}-D{D}",
    description="EMA bank (R power-of-2 alphas, shift-add only)",
    parameter_count=params,
    trainable_parameter_count=params,
    weight_bytes_int8=params,
    mac_per_trial=mac_trial,
    mac_per_session=mac_session,
    peak_state_bytes=N * (D + 2 * R) * 4,
    trial_buffer_bytes=0,
    requires_cubic_interp=False,
    requires_general_multiplier=False,  # power-of-two shifts in EMA
    requires_divider=True,
    multiplier_free_prepool=True,
  )


# ---------------------------------------------------------------------------
# B6 — FIR Streaming
# ---------------------------------------------------------------------------

def b6_cost(R: int = 4, K: int = 5, D: int = D) -> VariantCost:
  fir_weights = R * K
  # feature_proj: Linear(2R->D) + ReLU
  feat_proj_params = 2 * R * D + D
  fir_mac = T * R * K
  feat_proj_mac = 2 * R * D
  # post_pool: 2-layer D->D->W
  post_pool_layers = [D, W]
  post_pool_params = affine_params(D, post_pool_layers)
  post_pool_mac = affine_mac(D, post_pool_layers)
  params = fir_weights + feat_proj_params + post_pool_params
  mac_trial = N * (fir_mac + feat_proj_mac)
  mac_session = M * mac_trial + N * post_pool_mac
  return VariantCost(
    variant=f"B6-R{R}-K{K}-D{D}",
    description="FIR bank (R shared filters, learned)",
    parameter_count=params,
    trainable_parameter_count=params,
    weight_bytes_int8=params,
    mac_per_trial=mac_trial,
    mac_per_session=mac_session,
    peak_state_bytes=N * (D + 2 * R + (K - 1)) * 4,
    trial_buffer_bytes=0,
    requires_cubic_interp=False,
    requires_general_multiplier=True,  # learned FIR weights
    requires_divider=True,
    multiplier_free_prepool=False,
  )


# ---------------------------------------------------------------------------
# B7 — Count-Conditioned EarlyPool
# ---------------------------------------------------------------------------

def b7_cost(D: int = D) -> VariantCost:
  pre_pool_params = T * D + D
  pre_pool_mac = T * D
  # post_pool: 3-layer (D+1)->D->D->W
  post_pool_layers = [D, D, W]
  post_pool_params = affine_params(D + 1, post_pool_layers)
  post_pool_mac = affine_mac(D + 1, post_pool_layers)
  params = pre_pool_params + post_pool_params
  mac_trial = N * pre_pool_mac
  mac_session = M * mac_trial + N * post_pool_mac
  return VariantCost(
    variant=f"B7-D{D}",
    description="Count-Conditioned EarlyPool (B3 + survival scalar)",
    parameter_count=params,
    trainable_parameter_count=params,
    weight_bytes_int8=params,
    mac_per_trial=mac_trial,
    mac_per_session=mac_session,
    peak_state_bytes=N * D * 4 + T * N * 4,
    trial_buffer_bytes=T * N * 4,
    requires_cubic_interp=True,
    requires_general_multiplier=True,
    requires_divider=True,
    multiplier_free_prepool=False,
  )


# ---------------------------------------------------------------------------
# B8 — Fixed Random Projection
# ---------------------------------------------------------------------------

def b8_cost(D: int = D) -> VariantCost:
  # pre_pool: FIXED projection D x T (buffer, not trainable)
  pre_pool_buffer = D * T
  pre_pool_mac = T * D
  # post_pool: 3-layer D->D->D->W (learned)
  post_pool_layers = [D, D, W]
  post_pool_params = affine_params(D, post_pool_layers)
  post_pool_mac = affine_mac(D, post_pool_layers)
  trainable = post_pool_params
  return VariantCost(
    variant=f"B8-D{D}",
    description="Fixed Random Projection (Gaussian, frozen, JL-robust)",
    parameter_count=trainable + pre_pool_buffer,
    trainable_parameter_count=trainable,
    weight_bytes_int8=trainable + pre_pool_buffer,
    mac_per_trial=N * pre_pool_mac,
    mac_per_session=M * N * pre_pool_mac + N * post_pool_mac,
    peak_state_bytes=N * D * 4 + T * N * 4,
    trial_buffer_bytes=T * N * 4,
    requires_cubic_interp=True,
    requires_general_multiplier=True,
    requires_divider=True,
    multiplier_free_prepool=False,
  )


# ---------------------------------------------------------------------------
# B9 — Sparse Binary Hash
# ---------------------------------------------------------------------------

def b9_cost(K: int = 16, D: int = D) -> VariantCost:
  # pre_pool: FIXED sparse {-1,0,+1} matrix D x T with K nonzeros/row (buffer)
  pre_pool_buffer = D * K  # store as (row, col, sign) triplets
  pre_pool_mac = D * K  # only K add/subtracts per output
  # post_pool: 3-layer D->D->D->W
  post_pool_layers = [D, D, W]
  post_pool_params = affine_params(D, post_pool_layers)
  post_pool_mac = affine_mac(D, post_pool_layers)
  trainable = post_pool_params
  return VariantCost(
    variant=f"B9-K{K}-D{D}",
    description="Sparse Binary Hash ({-1,0,+1}, shift-add only)",
    parameter_count=trainable + pre_pool_buffer,
    trainable_parameter_count=trainable,
    weight_bytes_int8=trainable + pre_pool_buffer,
    mac_per_trial=N * pre_pool_mac,
    mac_per_session=M * N * pre_pool_mac + N * post_pool_mac,
    peak_state_bytes=N * D * 4 + T * N * 4,
    trial_buffer_bytes=T * N * 4,
    requires_cubic_interp=True,
    requires_general_multiplier=True,  # post_pool still needs multipliers
    requires_divider=True,
    multiplier_free_prepool=True,  # pre_pool is add/sub only
  )


# ---------------------------------------------------------------------------
# B10 — Population Stats (global identity)
# ---------------------------------------------------------------------------

def b10_cost(D: int = 32) -> VariantCost:
  # post_pool: 2-layer 4->D->W
  post_pool_layers = [D, W]
  post_pool_params = affine_params(4, post_pool_layers)
  post_pool_mac = affine_mac(4, post_pool_layers)
  # per-bin ops: 4 accumulators over N neurons (negligible)
  bin_mac = 4
  params = post_pool_params
  return VariantCost(
    variant=f"B10-D{D}",
    description="Population Stats (global identity, broadcast)",
    parameter_count=params,
    trainable_parameter_count=params,
    weight_bytes_int8=params,
    mac_per_trial=bin_mac * T,
    mac_per_session=M * bin_mac * T + post_pool_mac,
    peak_state_bytes=D * 4 + 32,
    trial_buffer_bytes=0,
    requires_cubic_interp=False,
    requires_general_multiplier=True,
    requires_divider=True,
    multiplier_free_prepool=True,
  )


# ---------------------------------------------------------------------------
# B11 — Hybrid FIR + Count
# ---------------------------------------------------------------------------

def b11_cost(R: int = 4, K: int = 5, D: int = D) -> VariantCost:
  fir_weights = R * K
  # feature_proj: Linear(2R->D) + ReLU
  feat_proj_params = 2 * R * D + D
  fir_mac = T * R * K
  feat_proj_mac = 2 * R * D
  # post_pool: 2-layer (D+1)->D->W
  post_pool_layers = [D, W]
  post_pool_params = affine_params(D + 1, post_pool_layers)
  post_pool_mac = affine_mac(D + 1, post_pool_layers)
  params = fir_weights + feat_proj_params + post_pool_params
  mac_trial = N * (fir_mac + feat_proj_mac)
  mac_session = M * mac_trial + N * post_pool_mac
  return VariantCost(
    variant=f"B11-R{R}-K{K}-D{D}",
    description="Hybrid FIR + Count-conditioned pooling",
    parameter_count=params,
    trainable_parameter_count=params,
    weight_bytes_int8=params,
    mac_per_trial=mac_trial,
    mac_per_session=mac_session,
    peak_state_bytes=N * (D + 2 * R + (K - 1)) * 4,
    trial_buffer_bytes=0,
    requires_cubic_interp=False,
    requires_general_multiplier=True,
    requires_divider=True,
    multiplier_free_prepool=False,
  )


# ---------------------------------------------------------------------------
# B12 — Streaming Hash (threshold-based, no trial buffer)
# ---------------------------------------------------------------------------

def b12_cost(K: int = 4, D: int = D) -> VariantCost:
  # thresholds buffer: [D, K] (registered, fixed)
  thresholds_buffer = D * K
  # post_pool: 2-layer D->D->W
  post_pool_layers = [D, W]
  post_pool_params = affine_params(D, post_pool_layers)
  post_pool_mac = affine_mac(D, post_pool_layers)
  trainable = post_pool_params
  # Per bin per neuron per output: K sign comparisons (counted as 0.25 MAC each
  # because comparators are ~4x cheaper than INT8 multipliers in ASIC area).
  comparisons_per_trial = N * T * D * K
  mac_equivalent = comparisons_per_trial // 4
  mac_session = M * mac_equivalent + N * post_pool_mac
  return VariantCost(
    variant=f"B12-K{K}-D{D}",
    description="Streaming threshold hash (comparator + sign-sum, no trial buffer)",
    parameter_count=trainable + thresholds_buffer,
    trainable_parameter_count=trainable,
    weight_bytes_int8=trainable + thresholds_buffer,
    mac_per_trial=mac_equivalent,
    mac_per_session=mac_session,
    peak_state_bytes=2 * N * D * 4,
    trial_buffer_bytes=0,
    requires_cubic_interp=False,
    requires_general_multiplier=True,
    requires_divider=True,
    multiplier_free_prepool=True,
  )


# ---------------------------------------------------------------------------
# B13 — Ensemble Random + Hash
# ---------------------------------------------------------------------------

def b13_cost(K: int = 16, D: int = D) -> VariantCost:
  half = D // 2
  # Projection buffer [half, T]
  proj_buffer = half * T
  # Hash buffer [half, T] with K nonzeros/row -> stored as triplets
  hash_buffer = half * K
  # post_pool: 3-layer D->D->D->W
  post_pool_layers = [D, D, W]
  post_pool_params = affine_params(D, post_pool_layers)
  post_pool_mac = affine_mac(D, post_pool_layers)
  trainable = post_pool_params
  mac_trial = N * (half * T + half * K)
  mac_session = M * mac_trial + N * post_pool_mac
  return VariantCost(
    variant=f"B13-K{K}-D{D}",
    description="Ensemble (Gaussian proj + sparse binary hash)",
    parameter_count=trainable + proj_buffer + hash_buffer,
    trainable_parameter_count=trainable,
    weight_bytes_int8=trainable + proj_buffer + hash_buffer,
    mac_per_trial=mac_trial,
    mac_per_session=mac_session,
    peak_state_bytes=N * D * 4 + T * N * 4,
    trial_buffer_bytes=T * N * 4,
    requires_cubic_interp=True,
    requires_general_multiplier=True,
    requires_divider=True,
    multiplier_free_prepool=False,  # Gaussian projection needs multipliers
  )


# ---------------------------------------------------------------------------
# B14 — Ternarized EarlyPool (STE-trained {-1,0,+1} weights)
# ---------------------------------------------------------------------------

def b14_cost(D: int = D) -> VariantCost:
  # Same param structure as B3, but weights stored as 2-bit ternary indices
  pre_pool_params = T * D + D
  pre_pool_mac = T * D
  post_pool_layers = [D, D, W]
  post_pool_params = affine_params(D, post_pool_layers)
  post_pool_mac = affine_mac(D, post_pool_layers)
  params = pre_pool_params + post_pool_params
  # Weight memory: 2 bits per weight instead of 8 bits
  weight_bytes_ternary = (params + 3) // 4  # ceil(params/4)
  mac_trial = N * pre_pool_mac
  mac_session = M * mac_trial + N * post_pool_mac
  return VariantCost(
    variant=f"B14-D{D}",
    description="Ternarized EarlyPool (STE {-1,0,+1}, multiplier-free)",
    parameter_count=params,
    trainable_parameter_count=params,
    weight_bytes_int8=weight_bytes_ternary,  # 2-bit packed
    mac_per_trial=mac_trial,
    mac_per_session=mac_session,
    peak_state_bytes=N * D * 4 + T * N * 4,
    trial_buffer_bytes=T * N * 4,
    requires_cubic_interp=True,
    requires_general_multiplier=False,  # all weights are {-1,0,+1}
    requires_divider=True,
    multiplier_free_prepool=True,
  )


def all_variants() -> List[VariantCost]:
  return [
    b0_cost(),
    b2_cost(D=512),
    b2_cost(D=128),
    b3_cost(D=64),
    b3_cost(D=128),
    b4_cost(D=64),
    b5_cost(R=4, D=64),
    b6_cost(R=4, K=5, D=64),
    b7_cost(D=64),
    b8_cost(D=64),
    b9_cost(K=16, D=64),
    b9_cost(K=8, D=64),
    b10_cost(D=32),
    b11_cost(R=4, K=5, D=64),
    b12_cost(K=4, D=64),
    b13_cost(K=16, D=64),
    b14_cost(D=64),
  ]


def format_table(variants: List[VariantCost]) -> str:
  header = (
    f"{'variant':<14} {'params':>9} {'train':>9} {'MAC/sess':>12} "
    f"{'state_B':>9} {'buf_B':>7} {'cubic':>6} {'multr':>6} {'div':>5} {'mpfree':>7}"
  )
  lines = [header, "-" * len(header)]
  for v in variants:
    lines.append(
      f"{v.variant:<14} {v.parameter_count:>9,} {v.trainable_parameter_count:>9,} "
      f"{v.mac_per_session:>12,} {v.peak_state_bytes:>9,} {v.trial_buffer_bytes:>7,} "
      f"{str(v.requires_cubic_interp):>6} {str(v.requires_general_multiplier):>6} "
      f"{str(v.requires_divider):>5} {str(v.multiplier_free_prepool):>7}"
    )
  return "\n".join(lines)


def main():
  variants = all_variants()
  print(format_table(variants))
  print()

  # Summary stats
  b3 = next(v for v in variants if v.variant == "B3-D64")
  print(f"Reference: B3-D64 = {b3.parameter_count:,} params, {b3.mac_per_session:,} MAC/sess")
  print()
  print("Per-variant improvement vs B3-D64:")
  print(f"  {'variant':<14} {'params':>10} {'MAC/sess':>12} {'state':>10}")
  for v in variants:
    p_ratio = v.parameter_count / b3.parameter_count
    m_ratio = v.mac_per_session / b3.mac_per_session
    s_ratio = v.peak_state_bytes / b3.peak_state_bytes
    print(
      f"  {v.variant:<14} {p_ratio:>9.2f}x {m_ratio:>11.2f}x {s_ratio:>9.2f}x"
    )

  # Export JSON for documentation
  data = [asdict(v) for v in variants]
  print()
  print("JSON export:")
  print(json.dumps(data, indent=2)[:2000])


if __name__ == "__main__":
  main()
