#!/usr/bin/env python3
"""Forward-only audit of the H1 CarrierID 700-dim identity token itself.

Every prior H1 CarrierID screen studied the 4-dim carrier at its input
boundary.  This module instead studies what the decoder actually consumes:
the 700-dim per-channel identity token built by
``H1CarrierIdSpint.carrierid_identity_projection`` and added to the neural
window at ``src = src + identity`` (see
``src/models/components/h1_carrierid_spint.py:147``).

CPU only.  No GPU, no training, no optimizer, no backward pass.  All
checkpoints are loaded with ``map_location="cpu"``.  Only the 11 fold-0
SOURCE recordings of ``data/000954`` are opened (``load_source_records``
touches only the held-in-calibration directory and only the 11
non-outer-date sessions of ``H1_M4_FOLD0_SOURCE``; target, formal, minival,
and EvalAI files are never indexed or opened by this module).

The two frozen source caches consulted below
(``pilot_artifacts/h1_carrierid_seed43/gpu_runs_s43_v1/shared_source_cache/hc``
and ``pilot_artifacts/h1_carrierid/gpu_runs/h32_fold0_v1/shared_source_cache/full``)
already exist on disk as immutable (mode 0444) artifacts.  The plan/cache/
normalizer reconstruction helpers imported below only *read and verify*
existing bytes against those paths -- they take the "already exists, verify
byte-for-byte identity" branch, never the "does not exist, write it" branch --
so no file under ``pilot_artifacts/`` is written by this module.

READ RULE (declared before running, evaluated against the measured numbers
in the report; this module does not act on the outcome, only measures it):

  (a) If the carrier-driven part has a tiny RMS relative to the activity
      part or to ``src``, the carrier is a weak perturbation and every
      content lever is bounded by that, which explains why content changes
      move R2 so little.
  (b) If the carrier-driven part injects only 1 effective dimension into the
      700-dim token, then the 4-wide contract is collapsing downstream, not
      upstream.
  (c) If the two parts (carrier-driven, activity-driven) are near-orthogonal,
      the carrier adds a genuinely separate direction; if near-parallel, it
      is largely redundant with the activity path.

No numeric thresholds are invented anywhere below: the "effective dimension"
statistics are the parameter-free participation ratio PR = (sum(s^2))^2 /
sum(s^4) of the singular-value spectrum ``s``, and numpy's own default-
tolerance ``matrix_rank``.  Everything else is reported as an explicit
distribution (min/25%/median/75%/max/mean/std), not thresholded.
"""
from __future__ import annotations

import hashlib
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.h1_m4_eb_pilot import (  # noqa: E402
    EB_RECEIPT_SHA256,
    H1_M4_FOLD0_SOURCE,
    RAW_RECEIPT_SHA256,
    build_carrier_cache,
    interpolate_trial_identity,
    load_source_records,
    persist_frozen_plan,
    reconstruct_frozen_plan,
)
from src.data.h1_m4_eb_normalized_v2 import fit_source_normalizer_from_cache, persist_v2_normalized_cache  # noqa: E402
from src.h1_m4_eb_normalized_v2_contract import canonical_sha256, sha256_file, state_hash  # noqa: E402
from src.models.components.h1_carrierid_spint import H1CarrierIdSpint  # noqa: E402


READ_RULE = {
    "a_weak_carrier_bounds_content": (
        "If the carrier-driven part has a tiny RMS relative to the activity part or to src, "
        "the carrier is a weak perturbation and every content lever is bounded by that, which "
        "explains why content changes move R2 so little."
    ),
    "b_carrier_collapses_downstream": (
        "If the carrier-driven part injects only 1 effective dimension into the 700-dim token, "
        "then the 4-wide contract is collapsing downstream, not upstream."
    ),
    "c_alignment": (
        "If the two parts are near-orthogonal, the carrier adds a genuinely separate direction; "
        "if near-parallel, it is largely redundant with the activity path."
    ),
}

DATA_DIR = ROOT / "data" / "000954"
RAW_RECEIPT_PATH = ROOT.parent / "sua_exploration/results/h1_m4_population_decoder_carrier_date_lodo_v1/H1_M4_POPULATION_DECODER_CARRIER_CPU_RECEIPT.json"
EB_RECEIPT_PATH = ROOT.parent / "sua_exploration/results/h1_m4_empirical_bayes_confidence_carrier_date_lodo_v1/H1_M4_EMPIRICAL_BAYES_CONFIDENCE_CARRIER_CPU_RECEIPT.json"

NET_KWARGS = dict(
    carrier_hidden_dim=32,
    carrier_dim=4,
    carrier_trial_length=1024,
    zero_carrier=False,
    model_dim=1024,
    num_covariates=7,
    window_size=700,
    num_heads=64,
    num_layers=1,
    num_id_layers=3,
    use_learnable_id=True,
    learnable_id_type="mlp",
    learnable_rep=True,
    dropout_rate=0.0,
    dynamic_dropout=True,
    dynamic_dropout_low=0.0,
    dynamic_dropout_high=1.0,
    tf_drop_rate=0.1,
    readin_layer_type="mlp",
)

CHECKPOINTS = {
    "hc_seed43_sealed_epoch49": {
        "checkpoint_path": ROOT / "pilot_artifacts/h1_carrierid_seed43/gpu_runs_s43_v1/hc/checkpoints/fixed_epoch50/epoch_049.ckpt",
        "config_path": ROOT / "pilot_artifacts/h1_carrierid_seed43/gpu_runs_s43_v1/hc/.hydra/config.yaml",
        "cache_dir": ROOT / "pilot_artifacts/h1_carrierid_seed43/gpu_runs_s43_v1/shared_source_cache/hc",
        "note": "sealed H-C epoch-49 checkpoint named in the task; seed 43.",
    },
    "hc_seed42_canonical_fold0_epoch49": {
        "checkpoint_path": ROOT / "pilot_artifacts/h1_carrierid/gpu_runs/h32_fold0_v1/full/checkpoints/fixed_epoch50/epoch_049.ckpt",
        "config_path": ROOT / "pilot_artifacts/h1_carrierid/gpu_runs/h32_fold0_v1/full/.hydra/config.yaml",
        "cache_dir": ROOT / "pilot_artifacts/h1_carrierid/gpu_runs/h32_fold0_v1/shared_source_cache/full",
        "note": "canonical fold-0 H-C ('full' arm) checkpoint, seed 42, same architecture/contract.",
    },
}


class ForwardOnlyStateGuard:
    """Hashes model state before/after every forward call; raises on drift."""

    def __init__(self, net: torch.nn.Module):
        self.net = net
        self.initial_hash = state_hash(net.state_dict())
        self.last_hash = self.initial_hash
        self.calls = 0

    def guarded_projection(self, calib: torch.Tensor, carrier: torch.Tensor) -> torch.Tensor:
        before = state_hash(self.net.state_dict())
        if before != self.last_hash:
            raise RuntimeError("model state drifted before a forward call")
        with torch.no_grad():
            out = self.net.carrierid_identity_projection(calib, carrier)
        after = state_hash(self.net.state_dict())
        if after != before:
            raise RuntimeError("model state drifted across a forward call")
        self.last_hash = after
        self.calls += 1
        return out


def load_net(checkpoint_path: Path) -> H1CarrierIdSpint:
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    state_dict = checkpoint["state_dict"]
    net_state = {key[len("net."):]: value for key, value in state_dict.items() if key.startswith("net.")}
    if len(net_state) != len(state_dict):
        raise RuntimeError(f"unexpected non-net. keys in checkpoint state_dict: {checkpoint_path}")
    net = H1CarrierIdSpint(**NET_KWARGS)
    missing, unexpected = net.load_state_dict(net_state, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"state_dict mismatch loading {checkpoint_path}: missing={missing} unexpected={unexpected}")
    net.eval()
    return net


def build_source_context(cache_dir: Path) -> dict[str, Any]:
    records = load_source_records(DATA_DIR)
    if tuple(sorted(records)) != tuple(sorted(H1_M4_FOLD0_SOURCE)):
        raise RuntimeError("source record set drifted from H1_M4_FOLD0_SOURCE")
    plan = reconstruct_frozen_plan(records, RAW_RECEIPT_PATH, EB_RECEIPT_PATH)
    persist_frozen_plan(plan, cache_dir)  # verify-only: arrays/manifest already exist under cache_dir
    carrier_cache = build_carrier_cache(records, plan, cache_dir)  # verify-only, same reason
    normalizer, raw = fit_source_normalizer_from_cache(carrier_cache)
    persist_v2_normalized_cache(cache_dir, raw, normalizer)  # verify-only, same reason
    return {"records": records, "plan": plan, "carrier_cache": carrier_cache, "normalizer": normalizer}


def _rms(array: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(array, dtype=np.float64), dtype=np.float64)))


def _distribution(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "n": int(values.size),
        "min": float(np.min(values)),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.median(values)),
        "p75": float(np.percentile(values, 75)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
    }


def _spectrum_stats(matrix: np.ndarray) -> dict[str, Any]:
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    sq = np.square(singular_values, dtype=np.float64)
    participation_ratio = float(np.square(sq.sum()) / np.sum(np.square(sq))) if np.sum(np.square(sq)) > 0 else 0.0
    numpy_rank = int(np.linalg.matrix_rank(matrix))
    return {
        "shape": list(matrix.shape),
        "singular_values": [float(v) for v in singular_values],
        "top10_singular_values": [float(v) for v in singular_values[:10]],
        "participation_ratio_effective_dim": participation_ratio,
        "numpy_default_tol_matrix_rank": numpy_rank,
        "frobenius_norm": float(np.linalg.norm(matrix)),
    }


def per_session_measurements(guard: ForwardOnlyStateGuard, context: dict[str, Any], session_name: str) -> dict[str, Any]:
    records = context["records"]
    carrier_cache = context["carrier_cache"]
    normalizer = context["normalizer"]
    record = records[session_name]

    starts = carrier_cache.starts_by_session[session_name]
    canonical_start = starts[0]
    entry = carrier_cache.get(session_name, canonical_start)
    trial_values = entry.trial_values

    identity_calib = np.stack(
        [interpolate_trial_identity(record, value) for value in trial_values], axis=0
    ).astype(np.float32)  # [4,1024,N]
    raw_carrier = np.asarray(entry.carrier, dtype=np.float64)  # [N,4]
    normalized_carrier = normalizer.normalize(raw_carrier).astype(np.float32)  # [N,4]

    calib_t = torch.from_numpy(identity_calib).unsqueeze(0)
    carrier_t = torch.from_numpy(normalized_carrier).unsqueeze(0)
    zero_calib_t = torch.zeros_like(calib_t)
    zero_carrier_t = torch.zeros_like(carrier_t)

    full = guard.guarded_projection(calib_t, carrier_t).squeeze(0).numpy().astype(np.float64)
    activity_only = guard.guarded_projection(calib_t, zero_carrier_t).squeeze(0).numpy().astype(np.float64)
    carrier_only = guard.guarded_projection(zero_calib_t, carrier_t).squeeze(0).numpy().astype(np.float64)
    both_zero = guard.guarded_projection(zero_calib_t, zero_carrier_t).squeeze(0).numpy().astype(np.float64)

    num_neurons = full.shape[0]
    if num_neurons != record.num_neurons:
        raise RuntimeError("identity token channel count does not match record neuron count")

    # 1. Full 700-dim identity token: RMS and per-channel norm distribution.
    per_channel_norm_full = np.linalg.norm(full, axis=1)
    per_channel_rms_full = per_channel_norm_full / np.sqrt(full.shape[1])
    measurement_1 = {
        "identity_full_rms_overall": _rms(full),
        "identity_full_per_channel_l2norm_distribution": _distribution(per_channel_norm_full),
        "identity_full_per_channel_rms_distribution": _distribution(per_channel_rms_full),
    }

    # 2. Decomposition: activity-driven part (carrier zeroed) vs carrier-driven
    #    part (activity zeroed), forward passes only, compared to the full forward.
    activity_norm = float(np.linalg.norm(activity_only))
    carrier_norm = float(np.linalg.norm(carrier_only))
    activity_rms = _rms(activity_only)
    carrier_rms = _rms(carrier_only)
    reconstruction = activity_only + carrier_only - both_zero
    reconstruction_residual = full - reconstruction
    measurement_2 = {
        "activity_part_norm": activity_norm,
        "carrier_part_norm": carrier_norm,
        "activity_part_rms": activity_rms,
        "carrier_part_rms": carrier_rms,
        "both_zero_baseline_rms": _rms(both_zero),
        "rms_ratio_carrier_over_activity": (carrier_rms / activity_rms) if activity_rms > 0.0 else float("nan"),
        "norm_ratio_carrier_over_activity": (carrier_norm / activity_norm) if activity_norm > 0.0 else float("nan"),
        "full_forward_rms": _rms(full),
        "nonlinear_reconstruction_residual_rms": _rms(reconstruction_residual),
        "note": (
            "activity_part = forward(calib=real, carrier=0); carrier_part = forward(calib=0, carrier=real); "
            "both_zero = forward(calib=0, carrier=0) is the shared bias-only baseline both parts sit on top of; "
            "reconstruction = activity_part + carrier_part - both_zero; residual = full - reconstruction is the "
            "unavoidable nonlinear (ReLU) cross-term, reported for honesty, not folded into either part."
        ),
    }

    # 3. Per-channel angle between carrier-driven and activity-driven contributions.
    activity_centered = activity_only - both_zero
    carrier_centered = carrier_only - both_zero
    activity_channel_norm = np.linalg.norm(activity_centered, axis=1)
    carrier_channel_norm = np.linalg.norm(carrier_centered, axis=1)
    denom = activity_channel_norm * carrier_channel_norm
    cosine = np.full(num_neurons, np.nan, dtype=np.float64)
    valid = denom > 0.0
    cosine[valid] = np.sum(activity_centered[valid] * carrier_centered[valid], axis=1) / denom[valid]
    cosine = np.clip(cosine, -1.0, 1.0)
    angle_deg = np.degrees(np.arccos(cosine))
    measurement_3 = {
        "definition": "angle between (activity_part - both_zero_baseline) and (carrier_part - both_zero_baseline), per channel, in R^700",
        "channels_with_degenerate_zero_vector": int(num_neurons - int(valid.sum())),
        "cosine_distribution": _distribution(cosine[valid]) if valid.any() else None,
        "angle_deg_distribution": _distribution(angle_deg[valid]) if valid.any() else None,
    }

    # 4. Rank / singular value spectrum of the full [N,700] token and the carrier-driven part.
    measurement_4 = {
        "full_identity_token_spectrum": _spectrum_stats(full),
        "carrier_driven_part_spectrum": _spectrum_stats(carrier_only - both_zero),
        "activity_driven_part_spectrum": _spectrum_stats(activity_only - both_zero),
    }

    # 5. Ratio of carrier-driven identity contribution to the neural window src it is added to, in RMS.
    #    src (post-permute, pre-fc_in) is exactly record.neural reshaped to [N,T]; every 700-bin sliding
    #    window is a contiguous slice of this same array, so its bulk RMS is the faithful representative
    #    scale of "the neural window src it is added to" without needing to materialize every window.
    src_all = np.asarray(record.neural, dtype=np.float64).T  # [N,T]
    src_rms = _rms(src_all)
    carrier_only_rms = _rms(carrier_only - both_zero)
    measurement_5 = {
        "src_rms_full_recording": src_rms,
        "carrier_driven_contribution_rms": carrier_only_rms,
        "ratio_carrier_driven_over_src": (carrier_only_rms / src_rms) if src_rms > 0.0 else float("nan"),
        "full_identity_over_src_ratio": (_rms(full) / src_rms) if src_rms > 0.0 else float("nan"),
        "note": "src_rms computed over the full recorded neural array (all T bins x N channels); every 700-bin window is a contiguous slice of this same array.",
    }

    return {
        "session": session_name,
        "num_neurons": num_neurons,
        "num_legal_calibration_starts": len(starts),
        "canonical_calibration_start_index": int(canonical_start),
        "canonical_calibration_trial_values": list(trial_values),
        "measurement_1_identity_token": measurement_1,
        "measurement_2_decomposition": measurement_2,
        "measurement_3_angle": measurement_3,
        "measurement_4_spectrum": measurement_4,
        "measurement_5_ratio_to_src": measurement_5,
    }


def _median_of(values: list[float]) -> float:
    finite = [v for v in values if np.isfinite(v)]
    return float(statistics.median(finite)) if finite else float("nan")


def run_checkpoint(label: str, spec: dict[str, Any]) -> dict[str, Any]:
    checkpoint_path = Path(spec["checkpoint_path"])
    cache_dir = Path(spec["cache_dir"])
    config_path = Path(spec["config_path"])
    checkpoint_sha256 = sha256_file(checkpoint_path)
    config_sha256 = sha256_file(config_path)

    context = build_source_context(cache_dir)
    net = load_net(checkpoint_path)
    guard = ForwardOnlyStateGuard(net)

    per_session = {}
    for session_name in H1_M4_FOLD0_SOURCE:
        per_session[session_name] = per_session_measurements(guard, context, session_name)

    final_hash = state_hash(net.state_dict())
    if final_hash != guard.initial_hash:
        raise RuntimeError(f"{label}: model state hash differs before vs after the full forward-only audit")

    medians = {
        "identity_full_rms_overall": _median_of([v["measurement_1_identity_token"]["identity_full_rms_overall"] for v in per_session.values()]),
        "activity_part_rms": _median_of([v["measurement_2_decomposition"]["activity_part_rms"] for v in per_session.values()]),
        "carrier_part_rms": _median_of([v["measurement_2_decomposition"]["carrier_part_rms"] for v in per_session.values()]),
        "rms_ratio_carrier_over_activity": _median_of([v["measurement_2_decomposition"]["rms_ratio_carrier_over_activity"] for v in per_session.values()]),
        "angle_deg_median_across_sessions": _median_of(
            [v["measurement_3_angle"]["angle_deg_distribution"]["median"] for v in per_session.values() if v["measurement_3_angle"]["angle_deg_distribution"]]
        ),
        "full_identity_participation_ratio": _median_of([v["measurement_4_spectrum"]["full_identity_token_spectrum"]["participation_ratio_effective_dim"] for v in per_session.values()]),
        "carrier_driven_participation_ratio": _median_of([v["measurement_4_spectrum"]["carrier_driven_part_spectrum"]["participation_ratio_effective_dim"] for v in per_session.values()]),
        "carrier_driven_numpy_rank": _median_of([v["measurement_4_spectrum"]["carrier_driven_part_spectrum"]["numpy_default_tol_matrix_rank"] for v in per_session.values()]),
        "ratio_carrier_driven_over_src": _median_of([v["measurement_5_ratio_to_src"]["ratio_carrier_driven_over_src"] for v in per_session.values()]),
        "full_identity_over_src_ratio": _median_of([v["measurement_5_ratio_to_src"]["full_identity_over_src_ratio"] for v in per_session.values()]),
    }

    return {
        "label": label,
        "note": spec["note"],
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "config_path": str(config_path),
        "config_sha256": config_sha256,
        "cache_dir": str(cache_dir),
        "state_sha256_before": guard.initial_hash,
        "state_sha256_after": final_hash,
        "state_immutable_across_all_forward_calls": final_hash == guard.initial_hash,
        "forward_calls": guard.calls,
        "source_sessions_opened": list(H1_M4_FOLD0_SOURCE),
        "source_manifest": {
            "raw_receipt_sha256": RAW_RECEIPT_SHA256,
            "eb_receipt_sha256": EB_RECEIPT_SHA256,
            "carrier_cache_sha256": context["carrier_cache"].manifest["cache_sha256"],
            "normalizer_sha256": context["normalizer"].normalizer_sha256,
            "plan_transform_sha256": context["plan"].transform_sha256,
        },
        "per_session": per_session,
        "medians_across_11_source_recordings": medians,
    }


def build_reading(results: dict[str, Any]) -> dict[str, Any]:
    reading = {}
    for label, result in results.items():
        m = result["medians_across_11_source_recordings"]
        reading[label] = {
            "a_weak_carrier_bounds_content": {
                "median_rms_ratio_carrier_over_activity": m["rms_ratio_carrier_over_activity"],
                "median_ratio_carrier_driven_over_src": m["ratio_carrier_driven_over_src"],
            },
            "b_carrier_collapses_downstream": {
                "median_carrier_driven_participation_ratio": m["carrier_driven_participation_ratio"],
                "median_carrier_driven_numpy_rank": m["carrier_driven_numpy_rank"],
            },
            "c_alignment": {
                "median_angle_deg_activity_vs_carrier": m["angle_deg_median_across_sessions"],
            },
        }
    return reading


def main() -> dict[str, Any]:
    results = {label: run_checkpoint(label, spec) for label, spec in CHECKPOINTS.items()}
    reading = build_reading(results)
    receipt = {
        "schema": "h1_identity_token_audit_v1",
        "task": "forward-only audit of the 700-dim H1 CarrierID identity token (not the 4-dim carrier)",
        "constraints": {
            "cpu_only": True,
            "no_training_no_optimizer_no_backward": True,
            "map_location": "cpu",
            "source_recordings_only": True,
            "target_formal_minival_evalai_opened": False,
        },
        "read_rule": READ_RULE,
        "results": results,
        "reading_against_read_rule": reading,
        "module_sha256": None,  # filled in after write, see write_receipt()
    }
    return receipt


def write_receipt(receipt: dict[str, Any], module_path: Path, output_path: Path) -> None:
    module_sha256 = hashlib.sha256(module_path.read_bytes()).hexdigest()
    receipt["module_sha256"] = module_sha256
    receipt["module_path"] = str(module_path)
    encoded = json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False)
    receipt["receipt_sha256_of_body_before_this_field"] = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    final_encoded = json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n"
    output_path.write_text(final_encoded, encoding="utf-8")


if __name__ == "__main__":
    receipt = main()
    write_receipt(receipt, Path(__file__).resolve(), Path(__file__).resolve().with_name("h1_identity_token_audit_receipt.json"))
    print(json.dumps({label: r["medians_across_11_source_recordings"] for label, r in receipt["results"].items()}, indent=2))
