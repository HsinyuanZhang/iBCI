#!/usr/bin/env python3
"""P0 exact replay of the sealed M1 rSyn3 template under rate normalization.

This is a standalone CPU audit.  The old source reader loads full source EMG
and M10 neural support before its original M10 mask; held-out access is public
M10 calibration support only.  It never constructs a decoder or opens a query
surface.  ``off`` deliberately uses the old reader/arithmetic path and
must reproduce every original normalized carrier byte-for-byte before any
``on`` comparison is reported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
WS = ROOT.parent
V1 = WS / "btransform_unified_v1"
for entry in (ROOT / "src", V1 / "src", V1 / "scripts", WS):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

SOURCE = ("ses-20120924", "ses-20120926", "ses-20120927", "ses-20120928")
SOURCE3 = ("ses-20120926", "ses-20120927", "ses-20120928")
HO = ("20121004", "20121017", "20121024")
ALL = SOURCE + HO
BIN_SECONDS = 0.02


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def typed_sha(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    return hashlib.sha256(value.dtype.str.encode() + str(value.shape).encode() + value.tobytes()).hexdigest()


def raw_sha(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _basis_and_old_normalizer():
    """Load the sealed source3 fit and normalizer; never refit NMF."""
    from tfpd_exploration.src.m1_emg_syn3_fcm_v1 import syn3
    from tfpd_exploration.src.m1_optimized_v2 import bank as old_bank, plan as old_plan

    sealed = old_bank.load()
    receipt = sealed["receipt"]
    if tuple(receipt["source_sessions"]) != SOURCE3:
        raise RuntimeError("sealed rSyn3 fit is not the source3 normalizer")
    blob = np.load(old_plan.BANK_NPZ, allow_pickle=False)
    basis = syn3.SourceBasis(kind="nnmf", scale=np.asarray(blob["scale"]), dictionary=np.asarray(blob["d0"]),
        activations=np.asarray(blob["activations"]), order=tuple(int(v) for v in blob["nmf_order"]),
        reconstruction_digest=str(blob["reconstruction_digest"][0]), library={"source": "sealed-rSyn3-source3"}, extra={})
    return basis, sealed, old_plan


def _z_on(rates: np.ndarray) -> np.ndarray:
    """The current muscle z law, retaining float64 old ridge arithmetic."""
    rate = np.asarray(rates, dtype=np.float64)
    mean_rate = rate.mean(axis=0)
    mean_count = mean_rate * BIN_SECONDS
    denominator = np.sqrt(np.maximum(mean_count, 1.0)) / BIN_SECONDS
    return np.ascontiguousarray((rate - mean_rate[None, :]) / denominator[None, :], dtype=np.float64)


def _fit_raw(scores: np.ndarray, rates: np.ndarray) -> np.ndarray:
    """Old per-unit ridge order: design=[1,z], gram/n+diag(0,1,1,1), rhs/n."""
    from tfpd_exploration.src.m1_emg_syn3_fcm_v1 import plan as syn3_plan
    z = np.asarray(scores, dtype=np.float64)
    r = np.asarray(rates, dtype=np.float64)
    if z.ndim != 2 or r.ndim != 2 or z.shape[0] != r.shape[0] or z.shape[1] != 3:
        raise RuntimeError("ridge input geometry drift")
    design = np.column_stack((np.ones(z.shape[0], dtype=np.float64), z))
    n = float(z.shape[0])
    gram = (design.T @ design) / n
    penalty = np.diag([0.0, syn3_plan.RIDGE_LAMBDA, syn3_plan.RIDGE_LAMBDA, syn3_plan.RIDGE_LAMBDA])
    encoded = np.empty((r.shape[1], 4), dtype=np.float64)
    for unit in range(r.shape[1]):
        # This is intentionally a per-unit GEMV, matching old
        # ``fit_unit_ridge`` rather than a matrix GEMM followed by slicing.
        rhs = (design.T @ r[:, unit]) / n
        beta = np.linalg.solve(gram + penalty, rhs)
        encoded[unit, :3] = beta[1:]
        encoded[unit, 3] = beta[0]
    if encoded.shape != (64, 4) or not np.isfinite(encoded).all():
        raise RuntimeError("unit ridge output drift")
    return encoded


def _template_raw(emg: np.ndarray, rates: np.ndarray, basis: Any) -> np.ndarray:
    """Unified normal-equation template; its off result is the P0 assertion."""
    from tfpd_exploration.src.m1_emg_rsyn3_fcm_v1 import syn3 as rsyn3
    score = rsyn3.project_basis(np.asarray(emg, dtype=np.float64), basis)
    return _fit_raw(score, rates)


def _on_raw(emg: np.ndarray, rates: np.ndarray, basis: Any) -> np.ndarray:
    """Use the same template with the explicitly normalized rate response."""
    from tfpd_exploration.src.m1_emg_rsyn3_fcm_v1 import syn3 as rsyn3
    score = rsyn3.project_basis(np.asarray(emg, dtype=np.float64), basis)
    return _fit_raw(score, _z_on(rates))

def _source_record(session: str):
    """Use the old fold-local reader and its source/target role exactly.

    That reader opens complete source EMG before the old M10 budget mask is
    applied below; it does not make this audit a query-label reader.
    """
    from btransform_unified_v1 import m1_projadd as mp
    from tfpd_exploration.src.m1_emg_rsyn3_fold_local_v1 import data as fold_data
    from tfpd_exploration.src.m1_emg_syn3_fcm_v1 import data as parent_data, plan as parent_plan
    path = parent_data.require_source_path(WS / parent_plan.SOURCE_RELATIVE[session])
    if parent_data.file_sha256(path) != parent_plan.SOURCE_FILE_SHA256[session]:
        raise RuntimeError(f"source body SHA drift: {session}")
    record = fold_data.load_fold_session(path, role="target" if session == mp.M1_OUTER_SESSION else "source")
    return record, path


def _source_arrays(record: Any) -> tuple[np.ndarray, np.ndarray]:
    from tfpd_exploration.src.m1_emg_rsyn3_fold_local_v1 import stage0
    from tfpd_exploration.src.m1_optimized_v2 import plan as old_plan
    return stage0._mask_budget(record, old_plan.SUPPORT_TRIALS)[:2]


def _ho_arrays(session: str, basis: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, Path]:
    """Only use the sanctioned M10 public-calibration reader; no FalconDataset."""
    import m1_projadd_depth2_series as legacy
    from tfpd_exploration.src.m1_b3_allsource_v1 import rsyn3_bank
    path = legacy._heldout_calib_path(session)
    record = rsyn3_bank.load_public_calib_support(path, support_trials=10)
    # Use the exact production M10 encoder for off; on alone substitutes z.
    return record.emg, record.rates, rsyn3_bank._encode_record(record, basis), path


def metrics(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    a = np.asarray(left, dtype=np.float64); b = np.asarray(right, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("comparison shape drift")
    flat_a, flat_b = a.ravel(), b.ravel()
    def correlation(x: np.ndarray, y: np.ndarray) -> tuple[float | None, float | None]:
        cosine_denom = float(np.linalg.norm(x) * np.linalg.norm(y))
        centered_x, centered_y = x - x.mean(), y - y.mean()
        pearson_denom = float(np.linalg.norm(centered_x) * np.linalg.norm(centered_y))
        return (None if cosine_denom == 0 else float(x @ y / cosine_denom),
                None if pearson_denom == 0 else float(centered_x @ centered_y / pearson_denom))
    cosine, pearson = correlation(flat_a, flat_b)
    per_axis = []
    if a.ndim == 2:
        for axis in range(a.shape[1]):
            axis_cosine, axis_pearson = correlation(a[:, axis], b[:, axis])
            per_axis.append({"axis": axis, "max_abs": float(np.max(np.abs(a[:, axis] - b[:, axis]))),
                             "cosine": axis_cosine, "pearson": axis_pearson,
                             "all_array_equal": bool(np.array_equal(a[:, axis], b[:, axis]))})
    return {"max_abs": float(np.max(np.abs(flat_a - flat_b))), "cosine": cosine, "pearson": pearson,
            "all_array_equal": bool(np.array_equal(left, right)), "per_axis": per_axis}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out.exists():
        raise FileExistsError(f"--output-dir must be fresh: {out}")
    out.mkdir(parents=True)

    # Argument validation above does not open numerical inputs.
    import m1_projadd_depth2_series as legacy
    from btransform_unified_v1 import m1_projadd as mp
    from tfpd_exploration.src.m1_emg_syn3_fcm_v1 import syn3

    basis, sealed, old_plan = _basis_and_old_normalizer()
    baseline: dict[str, np.ndarray] = {**mp.load_source_carriers()}
    outer, outer_meta = mp.encode_outer_carrier(); baseline["ses-20120924"] = outer
    for session in HO:
        baseline[session], _ = legacy.encode_heldout_calib_carrier(session)
    if set(baseline) != set(ALL):
        raise RuntimeError("baseline seven-tag roster drift")

    raw_off: dict[str, np.ndarray] = {}; raw_on: dict[str, np.ndarray] = {}
    raw_authority: dict[str, np.ndarray] = {}; input_binding: dict[str, Any] = {}
    for session in SOURCE:
        record, path = _source_record(session)
        emg, rates = _source_arrays(record)
        # The template, not the authority call, defines P0's off assertion.
        raw_off[session] = _template_raw(emg, rates, basis)
        from tfpd_exploration.src.m1_emg_rsyn3_fold_local_v1.carrier_bank import _encode_session
        raw_authority[session] = _encode_session(record, basis)
        raw_on[session] = _on_raw(emg, rates, basis)
        input_binding[session] = {"kind": "source_old_fold_reader_full_emg_then_M10_budget_mask", "path": str(path.resolve()), "sha256": sha_file(path),
                                  "role": "target" if session == "ses-20120924" else "source", "emg_typed_sha256": typed_sha(emg), "rates_typed_sha256": typed_sha(rates)}
    for session in HO:
        emg, rates, raw_authority[session], path = _ho_arrays(session, basis)
        raw_off[session] = _template_raw(emg, rates, basis)
        raw_on[session] = _on_raw(emg, rates, basis)
        input_binding[session] = {"kind": "heldout_public_calib_M10_only", "path": str(path.resolve()), "sha256": sha_file(path),
                                  "emg_typed_sha256": typed_sha(emg), "rates_typed_sha256": typed_sha(rates)}

    # Gate 1: exact current production arithmetic.  The independent template
    # must be byte-identical to the current old encoder on all seven tags.
    raw_authority_gate = {session: metrics(raw_off[session], raw_authority[session]) for session in ALL}
    if not all(row["all_array_equal"] for row in raw_authority_gate.values()):
        raise RuntimeError("unified raw template differs from current production encoder")
    # Gate 2: sealed source3 raw uses a historical floating-point artifact.
    # It is numerically bound at absolute 1e-10, rtol=0, but intentionally is
    # not claimed byte-identical.  Model input exactness is Gate 3 below.
    sealed_raw_gate = {session: metrics(raw_off[session], sealed["raw"][session]) for session in SOURCE3}
    if not all(np.allclose(raw_off[session], sealed["raw"][session], rtol=0.0, atol=1.0e-10) for session in SOURCE3):
        raise RuntimeError("template raw differs materially from sealed source3 raw")

    old_mean, old_scale = sealed["normalizer_mean"], sealed["normalizer_scale"]
    off = {s: np.ascontiguousarray(syn3.normalize_carriers(raw_off[s], old_mean, old_scale), dtype=np.float32) for s in ALL}
    on_old = {s: np.ascontiguousarray(syn3.normalize_carriers(raw_on[s], old_mean, old_scale), dtype=np.float32) for s in ALL}
    refit_mean, refit_scale = syn3.source_normalizer([raw_on[s] for s in SOURCE3])
    on_refit = {s: np.ascontiguousarray(syn3.normalize_carriers(raw_on[s], refit_mean, refit_scale), dtype=np.float32) for s in ALL}

    rows = {}
    for session in ALL:
        exact = metrics(off[session], baseline[session])
        if not exact["all_array_equal"]:
            raise RuntimeError(f"{session}: off template failed exact baseline carrier replay")
        describe = lambda value: {"dtype": value.dtype.str, "shape": list(value.shape), "raw_sha256": raw_sha(value), "typed_sha256": typed_sha(value)}
        rows[session] = {"baseline": describe(baseline[session]), "raw_off": describe(raw_off[session]), "raw_authority": describe(raw_authority[session]), "raw_on": describe(raw_on[session]),
                         "off": describe(off[session]), "on_oldnorm": describe(on_old[session]), "on_refitnorm": describe(on_refit[session]),
                         "off_vs_baseline": exact, "raw_off_vs_raw_authority": metrics(raw_off[session], raw_authority[session]), "raw_off_vs_raw_on": metrics(raw_off[session], raw_on[session]), "on_oldnorm_vs_off": metrics(on_old[session], off[session]),
                         "on_refitnorm_vs_off": metrics(on_refit[session], off[session])}
    actual_t_gate = {session: metrics(off[session], baseline[session]) for session in ALL}
    if not all(row["all_array_equal"] for row in actual_t_gate.values()):
        raise RuntimeError("normalized off carrier differs from actual baseline model input T")
    arrays = {**{f"raw_off/{s}": raw_off[s] for s in ALL}, **{f"raw_authority/{s}": raw_authority[s] for s in ALL}, **{f"raw_on/{s}": raw_on[s] for s in ALL},
              **{f"off/{s}": off[s] for s in ALL}, **{f"on_oldnorm/{s}": on_old[s] for s in ALL}, **{f"on_refitnorm/{s}": on_refit[s] for s in ALL},
              "old_normalizer_mean": old_mean, "old_normalizer_scale": old_scale, "on_source3_refit_normalizer_mean": refit_mean, "on_source3_refit_normalizer_scale": refit_scale}
    array_path = out / "p0_exact_m1_arrays.npz"; np.savez_compressed(array_path, **arrays)
    receipt = {"schema": "carrier_v4_p0_exact_m1_v1", "status": "COMPLETED", "utc": datetime.now(timezone.utc).isoformat(),
      "scope": {"task": "m1", "source_sessions": list(SOURCE), "old_normalizer_fit_scope": list(SOURCE3), "ho_sessions": list(HO),
                "basis": "loaded sealed source3 rSyn3; no NMF refit", "ridge": "per-unit design=[1,z]; gram/n + diag(0,1,1,1); rhs/n; solve; output [beta,b]",
                "on_z": "(rate-mean_rate)/(sqrt(max(mean_rate*0.02,1))/0.02)", "source_reader": "old fold reader loads complete source EMG then stage0._mask_budget selects M10 EMG/rates", "ho_access": "M10 public calib only", "query_labels_read": False, "target_calibration_values_read": True, "decoder_constructed": False},
      "sealed_source3": {"npz": str(old_plan.BANK_NPZ), "npz_sha256": sha_file(old_plan.BANK_NPZ), "receipt": str(old_plan.BANK_RECEIPT), "receipt_sha256": sha_file(old_plan.BANK_RECEIPT), "receipt_body": sealed["receipt"]},
      "baseline_paths": {"source": "btransform_unified_v1.m1_projadd.load_source_carriers + encode_outer_carrier", "heldout": "m1_projadd_depth2_series.encode_heldout_calib_carrier"},
      "gates": {"raw_template_vs_current_production": {"requirement": "all seven arrays byte-identical", "passed": True, "per_tag": raw_authority_gate},
                "raw_template_vs_sealed_source3": {"requirement": "atol=1e-10, rtol=0", "passed": True, "per_tag": sealed_raw_gate,
                   "note": "sealed raw arrays have a historical floating-point artifact; byte equality is neither required nor claimed"},
                "normalized_off_vs_actual_model_input_T": {"requirement": "all seven float32 arrays byte-identical", "passed": True, "per_tag": actual_t_gate}},
      "outer_meta": outer_meta, "input_binding": input_binding, "normalizers": {"old": {"mean_typed_sha256": typed_sha(old_mean), "scale_typed_sha256": typed_sha(old_scale)}, "on_refit_source3": {"mean_typed_sha256": typed_sha(refit_mean), "scale_typed_sha256": typed_sha(refit_scale)}},
      "tags": rows, "outputs": {"npz": str(array_path), "npz_sha256": sha_file(array_path)},
      "code": {str(Path(__file__).resolve()): sha_file(Path(__file__).resolve()), str(Path(legacy.__file__).resolve()): sha_file(Path(legacy.__file__).resolve()), str(Path(mp.__file__).resolve()): sha_file(Path(mp.__file__).resolve())}}
    atomic_json(out / "p0_exact_m1_receipt.json", receipt)

if __name__ == "__main__":
    main()
