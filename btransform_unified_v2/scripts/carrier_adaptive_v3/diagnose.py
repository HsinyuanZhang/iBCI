#!/usr/bin/env python3
"""Source-only repeated-panel diagnostics for inner adaptive projections.

This is a held-date projection diagnostic with shared source RMS.  It is a
source-only surrogate for projection stability, not decoder evidence or fully
independent end-to-end cross-validation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Mapping

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESULTS = ROOT / "results" / "carrier_adaptive_v3"
for import_path in (ROOT, ROOT / "scripts", ROOT.parent, ROOT.parent / "btransform_unified_v1" / "src", HERE):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))
from carrier_adaptive_v3 import panels, statistics


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _date(dataset: str, session: str) -> str:
    if dataset == "m1":
        return session
    match = re.match(r"ses-(\d{8})", session)
    _need(match is not None, f"H1 session lacks ses-YYYYMMDD prefix: {session}")
    value = match.group(1)
    return f"{value[:4]}-{value[4:6]}-{value[6:]}"


def _dates(bundle: panels.PanelBundle) -> dict[str, tuple[str, ...]]:
    out: dict[str, list[str]] = {}
    for session in bundle.sessions:
        out.setdefault(_date(bundle.dataset, session), []).append(session)
    return {date: tuple(sessions) for date, sessions in out.items()}


def _require_inner_roster(bundle: panels.PanelBundle) -> None:
    """Diagnose only the exact chronological inner source rosters."""
    received = tuple(bundle.sessions)
    if bundle.dataset == "m1":
        _need(received == ("ses-20120924", "ses-20120926"), "M1 diagnose requires exact inner 0924/0926 source roster")
        return
    _need(bundle.dataset == "h1", "bundle dataset must be m1 or h1")
    from btransform_unified_v1.h1_config import H1_SESSIONS_BY_DATE
    inner_dates = ("1925-01-01", "1925-01-08", "1925-01-13", "1925-01-15")
    expected = tuple(session for date in inner_dates for session in H1_SESSIONS_BY_DATE[date])
    _need(received == expected, "H1 diagnose requires exact nine-session inner roster through 1925-01-15")


def _scale(plan: statistics.AdaptiveProjection) -> np.ndarray:
    diagonal = np.diag(plan.signal_cov + plan.noise_cov)
    floor = max(float(np.mean(diagonal)) * 1e-6, 1e-12)
    return np.maximum(diagonal, floor)


def _error(prediction: np.ndarray, target: np.ndarray, scale: np.ndarray) -> float:
    """Equal feature then equal unit MSE, normalized in the fold's raw units."""
    value = np.asarray(prediction, np.float64) - np.asarray(target, np.float64)
    return float(np.mean(np.mean(np.square(value) / scale[None, :], axis=-1)))


def _fold_errors(plan: statistics.AdaptiveProjection, held: Mapping[str, np.ndarray]) -> dict[str, dict[str, float]]:
    scale = _scale(plan)
    output: dict[str, dict[str, float]] = {}
    for session, value in held.items():
        value = np.asarray(value, np.float64)
        count = value.shape[0]
        _need(count >= 2, f"{session}: diagnostic requires P>=2")
        per_method = {name: [] for name in ("raw_panel", "constant_training_mean", "pca_wiener", "snr_wiener")}
        for panel in range(count):
            raw = value[panel]
            target = (value.sum(axis=0) - raw) / (count - 1)
            per_method["raw_panel"].append(_error(raw, target, scale))
            per_method["constant_training_mean"].append(_error(np.broadcast_to(plan.mean, raw.shape), target, scale))
            per_method[plan.metadata["method"]].append(_error(statistics.reconstruct(plan, raw), target, scale))
        # One method is fitted per plan.  The caller combines the two fitted
        # methods; raw/constant use exactly this same fold scale in both.
        output[session] = {name: float(np.mean(values)) for name, values in per_method.items() if values}
    return output


def _combine_errors(first: Mapping[str, Mapping[str, float]], second: Mapping[str, Mapping[str, float]]) -> dict[str, dict[str, float]]:
    """Merge PCA and SNR errors while requiring shared comparator agreement."""
    out: dict[str, dict[str, float]] = {}
    for session in first:
        _need(session in second, "method session mismatch")
        for key in ("raw_panel", "constant_training_mean"):
            _need(abs(first[session][key] - second[session][key]) <= 1e-12 * max(1.0, abs(first[session][key])), f"shared comparator drift {session}/{key}")
        out[session] = {
            "raw_panel": first[session]["raw_panel"],
            "constant_training_mean": first[session]["constant_training_mean"],
            "pca_wiener": first[session]["pca_wiener"],
            "snr_wiener": second[session]["snr_wiener"],
        }
    return out


def _first_to_rest_errors(plan: statistics.AdaptiveProjection, source: Mapping[str, np.ndarray]) -> dict[str, dict[str, float]]:
    """Deployment-shaped variant: panel zero predicts the mean of panels 1..P."""
    output: dict[str, dict[str, float]] = {}
    scale = _scale(plan)
    for session, value in source.items():
        value = np.asarray(value, np.float64)
        _need(value.shape[0] >= 2, f"{session}: first-to-rest needs P>=2")
        raw, target = value[0], value[1:].mean(axis=0)
        output[session] = {
            "raw_panel": _error(raw, target, scale),
            "constant_training_mean": _error(np.broadcast_to(plan.mean, raw.shape), target, scale),
            plan.metadata["method"]: _error(statistics.reconstruct(plan, raw), target, scale),
        }
    return output


def _average(rows: list[Mapping[str, float]]) -> dict[str, float]:
    _need(bool(rows), "cannot average no rows")
    keys = tuple(rows[0])
    _need(all(tuple(row) == keys for row in rows), "inconsistent diagnostic rows")
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def _ratios(row: Mapping[str, float]) -> dict[str, dict[str, float]]:
    raw = max(float(row["raw_panel"]), 1e-12)
    constant = max(float(row["constant_training_mean"]), 1e-12)
    return {method: {"to_raw_panel": float(row[method] / raw), "to_constant_training_mean": float(row[method] / constant)} for method in ("pca_wiener", "snr_wiener")}


def _repeatability(plan: statistics.AdaptiveProjection, source: Mapping[str, np.ndarray]) -> dict:
    """Compare first panel and rest mean after centering each axis over units."""
    session_rows = {}
    for session, value in source.items():
        value = np.asarray(value, np.float64)
        first = statistics.project(plan, value[0]).astype(np.float64)
        rest = statistics.project(plan, value[1:].mean(axis=0)).astype(np.float64)
        left, right = first - first.mean(axis=0, keepdims=True), rest - rest.mean(axis=0, keepdims=True)
        denominator = np.sqrt(np.sum(left * left, axis=0) * np.sum(right * right, axis=0))
        pearson = np.divide(np.sum(left * right, axis=0), denominator, out=np.zeros(4), where=denominator > 0.0)
        row_denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
        row_cosine = np.divide(np.sum(left * right, axis=1), row_denominator, out=np.zeros(value.shape[1]), where=row_denominator > 0.0)
        session_rows[session] = {
            "per_axis_pearson": pearson.tolist(),
            "per_axis_rms_difference": np.sqrt(np.mean(np.square(left - right), axis=0)).tolist(),
            "row_cosine_mean": float(np.mean(row_cosine)),
            "row_cosine_min": float(np.min(row_cosine)),
            "row_cosine_max": float(np.max(row_cosine)),
        }
    pearson_rows = np.asarray([row["per_axis_pearson"] for row in session_rows.values()])
    rms_rows = np.asarray([row["per_axis_rms_difference"] for row in session_rows.values()])
    mean_pearson = np.mean(pearson_rows, axis=0)
    mean_rms = np.mean(rms_rows, axis=0)
    total_eigenvalues = np.linalg.eigvalsh(plan.signal_cov + plan.noise_cov)
    positive = total_eigenvalues[total_eigenvalues > max(float(total_eigenvalues.max()) * 1e-12, 1e-12)]
    condition = float(total_eigenvalues.max() / positive.min()) if positive.size else None
    return {
        "sessions": session_rows,
        "equal_session_mean_per_axis_pearson": mean_pearson.tolist(),
        "per_axis_pearson_min": np.min(pearson_rows, axis=0).tolist(),
        "per_axis_pearson_max": np.max(pearson_rows, axis=0).tolist(),
        "equal_session_mean_per_axis_rms_difference": mean_rms.tolist(),
        "alpha": list(plan.metadata.get("reliability", [])),
        "rho": plan.metadata.get("rho"),
        "conditioning_total_signal_plus_raw_noise": condition,
    }


def _leave_date_out(bundle: panels.PanelBundle) -> dict:
    source = {session: np.asarray(bundle.panels[session], np.float64) for session in bundle.sessions}
    dates = _dates(bundle)
    _need(len(dates) >= 2, "leave-date-out diagnostic requires at least two source dates")
    folds, date_rows = {}, []
    for date, held_sessions in dates.items():
        train = {session: value for session, value in source.items() if session not in held_sessions}
        held = {session: source[session] for session in held_sessions}
        pca, snr = statistics.fit_projection(train, "pca_wiener"), statistics.fit_projection(train, "snr_wiener")
        _need(np.allclose(_scale(pca), _scale(snr), rtol=1e-12, atol=1e-12), "method normalization diagonal drift")
        session_errors = _combine_errors(_fold_errors(pca, held), _fold_errors(snr, held))
        date_error = _average(list(session_errors.values()))
        date_rows.append(date_error)
        folds[date] = {"held_sessions": list(held_sessions), "sessions": session_errors, "equal_session_error": date_error, "ratios": _ratios(date_error), "normalization_scale": _scale(pca).tolist(), "normalization": "per-feature diag(train signal_cov + train noise_cov), floored at max(mean diagonal * 1e-6, 1e-12)", "pca_metadata": dict(pca.metadata), "snr_metadata": dict(snr.metadata)}
    aggregate = _average(date_rows)
    method_rank = min(("pca_wiener", "snr_wiener"), key=lambda method: (aggregate[method], 0 if method == "pca_wiener" else 1))
    if abs(aggregate["pca_wiener"] - aggregate["snr_wiener"]) <= 1e-12:
        method_rank = "pca_wiener"
    return {"folds": folds, "equal_date_error": aggregate, "ratios": _ratios(aggregate), "method_rank": method_rank, "method_rank_rule": "minimum equal-date normalized reconstruction MSE; absolute tie within 1e-12 selects pca_wiener"}


def _deployment_style(bundle: panels.PanelBundle, pca: statistics.AdaptiveProjection, snr: statistics.AdaptiveProjection) -> dict:
    source = {session: np.asarray(bundle.panels[session], np.float64) for session in bundle.sessions}
    _need(np.allclose(_scale(pca), _scale(snr), rtol=1e-12, atol=1e-12), "full method normalization diagonal drift")
    rows = _combine_errors(_first_to_rest_errors(pca, source), _first_to_rest_errors(snr, source))
    return {"first_panel_to_rest_mean": rows, "equal_session_error": _average(list(rows.values())), "normalization_scale": _scale(pca).tolist(), "note": "deployment-style source diagnostic: full inner source fit is used, then panel 0 predicts the mean of remaining panels; it is not held-date independent"}


def _require_v3_output(path: Path) -> Path:
    value = path.resolve()
    _need(RESULTS in (value, *value.parents), f"out must be under {RESULTS}")
    _need(value.suffix == ".json", "out must be JSON")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True, help="inner source panel bundle JSON receipt")
    parser.add_argument("--out", type=Path, required=True, help="new v3 JSON diagnostic")
    args = parser.parse_args()
    bundle_path, output = args.bundle.resolve(), _require_v3_output(args.out)
    _need(bundle_path.is_file() and bundle_path.suffix == ".json", "bundle must be an existing JSON receipt")
    _need(not output.exists(), "diagnostic output must be new")
    bundle, bundle_metadata = panels.load_bundle(bundle_path)
    _need(bundle_metadata.get("stage") in (None, "inner"), "diagnose accepts inner bundle only")
    _require_inner_roster(bundle)
    full_source = {session: np.asarray(bundle.panels[session], np.float64) for session in bundle.sessions}
    pca, snr = statistics.fit_projection(full_source, "pca_wiener"), statistics.fit_projection(full_source, "snr_wiener")
    report = {
        "schema": "carrier_adaptive_v3_source_projection_diagnostic_v1",
        "status": "SOURCE_ONLY_COMPLETE",
        "dataset": bundle.dataset,
        "sessions": list(bundle.sessions),
        "bundle": str(bundle_path),
        "bundle_sha256": _sha(bundle_path),
        "bundle_arrays_sha256": bundle_metadata.get("arrays_sha256"),
        "diagnose_code_sha256": _sha(Path(__file__).resolve()),
        "target_io": False,
        "target_probes": False,
        "diagnostic_scope": "held-date projection diagnostic with shared source RMS; source-only surrogate, not decoder proof or fully independent end-to-end CV",
        "leave_source_date_out": _leave_date_out(bundle),
        "actual_deployment_style": _deployment_style(bundle, pca, snr),
        "projected_repeatability": {"pca_wiener": _repeatability(pca, full_source), "snr_wiener": _repeatability(snr, full_source)},
        "full_inner_fit_metadata": {"pca_wiener": dict(pca.metadata), "snr_wiener": dict(snr.metadata)},
        "plan_persisted": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
