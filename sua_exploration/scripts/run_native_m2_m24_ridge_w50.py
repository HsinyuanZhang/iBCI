#!/usr/bin/env python3
"""CPU preflight and forward-only runner for ``native_m2_m24_ridge_w50_v1``.

This is an independent conventional Wiener/ridge reference for the existing
Native-M2 *M24 chronological held-out* layout.  It is deliberately separate
from SPINT/T4/K4/P2/P3 code and never writes into their result roots.

``preflight``
    Audits source shapes, first-24/raw-W50 support/query boundaries, and an
    existing strict M24 split manifest.  It does not materialize velocity
    targets, predictions, or any metric.

``forward``
    Requires a matching preflight receipt and an explicit review token.  It
    fits each selected session from its support rows only and writes immutable
    prediction/target artifacts.  It intentionally contains no R² calculation
    or selection rule.  Scoring is a separate, later action.

Use ``--sessions`` to split the exact six-session workload, for example three
sessions per GPU process.  The source/reference boundary is nevertheless
checked against the complete six-session manifest on every shard.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import sys
from typing import Any, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sua_exploration.mc_maze import native_m2_m24_ridge_w50 as ridge


FORWARD_REVIEW_TOKEN = "ROOT_REVIEWED_FORWARD_ONLY_NO_METRIC"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ridge.NativeM2RidgeError(message)


def canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(dict(value), sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ridge.NativeM2RidgeError(f"cannot read JSON object {path}: {exc}") from exc
    require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def write_exclusive(path: Path, raw: bytes) -> str:
    """Write one immutable local artifact without overwriting a prior run."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(path, 0o444)
    require(stat.S_IMODE(path.stat().st_mode) == 0o444 and path.is_file(), f"immutable write failed: {path}")
    return hashlib.sha256(raw).hexdigest()


def write_json_exclusive(path: Path, value: Mapping[str, Any]) -> str:
    return write_exclusive(path, canonical_bytes(value))


def write_npz_exclusive(path: Path, **arrays: np.ndarray) -> str:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    raw = buffer.getvalue()
    return write_exclusive(path, raw)


def parse_session_selection(raw: str | None) -> tuple[str, ...]:
    if raw is None or not raw.strip():
        return ridge.EXPECTED_HELDOUT_SESSIONS
    values = tuple(item.strip() for item in raw.split(",") if item.strip())
    require(values, "--sessions must not be empty")
    require(len(set(values)) == len(values), "--sessions contains duplicate session")
    unexpected = sorted(set(values) - set(ridge.EXPECTED_HELDOUT_SESSIONS))
    require(not unexpected, f"--sessions contains non-Native-M2-M24 session(s): {unexpected}")
    return tuple(sorted(values))


def discover_sources(data_dir: Path, sessions: tuple[str, ...]) -> dict[str, Path]:
    candidates: dict[str, list[Path]] = {session: [] for session in sessions}
    for path in sorted(data_dir.rglob("*held-out-calib*.nwb")):
        pieces = path.name.split("_")
        if len(pieces) < 2:
            continue
        session = pieces[1].split(".")[0]
        if session in candidates:
            candidates[session].append(path.resolve())
    result: dict[str, Path] = {}
    for session, paths in candidates.items():
        require(len(paths) == 1, f"{session}: expected exactly one held-out-calib NWB, found {paths}")
        result[session] = paths[0]
    return result


def load_raw_m2(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load raw M2 arrays without smoothing, filtering, or padding."""
    # Importing inside the runner keeps synthetic unit tests independent of the
    # FALCON package and makes the source-data dependency explicit.
    try:
        from falcon_challenge.config import FalconTask
        from falcon_challenge.dataloaders import load_nwb
    except ImportError as exc:
        raise ridge.NativeM2RidgeError(
            "FALCON loader unavailable; run under the pinned SPINT environment"
        ) from exc
    neural, covariates, trial_change, eval_mask = load_nwb(path, FalconTask.m2)
    return (
        np.asarray(neural, dtype=np.float32),
        np.asarray(covariates, dtype=np.float32),
        np.asarray(trial_change, dtype=bool),
        np.asarray(eval_mask, dtype=bool),
    )


def audit_sources(
    data_dir: Path,
    reference_manifest_path: Path,
    sessions: tuple[str, ...],
) -> tuple[dict[str, Any], dict[str, tuple[np.ndarray, np.ndarray, ridge.ChronologicalWindowLayout]]]:
    """Read source layout only; this function deliberately computes no R²."""
    manifest_path = reference_manifest_path.resolve()
    require(manifest_path.is_file(), f"reference split manifest not found: {manifest_path}")
    manifest = read_json_object(manifest_path)
    ridge.validate_reference_split_manifest(manifest)
    sources = discover_sources(data_dir.resolve(), sessions)
    layouts: list[ridge.ChronologicalWindowLayout] = []
    loaded: dict[str, tuple[np.ndarray, np.ndarray, ridge.ChronologicalWindowLayout]] = {}
    receipt_rows: dict[str, Any] = {}
    for session in sessions:
        path = sources[session]
        neural, covariates, trial_change, eval_mask = load_raw_m2(path)
        layout = ridge.chronological_m24_layout(session, neural, covariates, trial_change, eval_mask)
        layouts.append(layout)
        # Keeping the raw target array private in ``loaded`` is necessary for
        # forward mode, but this audit uses only its shape in the layout.
        loaded[session] = (neural, covariates, layout)
        receipt_rows[session] = {
            "source_path": str(path),
            "source_bytes": int(path.stat().st_size),
            "source_sha256": sha256_file(path),
            "layout": layout.as_audit_dict(),
        }
    ridge.validate_selected_layouts_against_reference_manifest(layouts, manifest)
    payload = {
        "schema_version": 1,
        "program_id": ridge.PROGRAM_ID,
        "mode": "preflight",
        "source_data_only": True,
        "no_velocity_values_used_in_preflight": True,
        "predictions_generated": False,
        "metric_computed": False,
        "final_scoring_implemented_in_this_runner": False,
        "reference_split_manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
        },
        "implementation": {
            "numerical_core_path": str(Path(ridge.__file__).resolve()),
            "numerical_core_sha256": sha256_file(Path(ridge.__file__).resolve()),
            "runner_path": str(Path(__file__).resolve()),
            "runner_sha256": sha256_file(Path(__file__).resolve()),
        },
        "contract": {
            "calibration_trials": ridge.CALIBRATION_TRIALS,
            "query_start_trial": ridge.CALIBRATION_TRIALS,
            "window_bins": ridge.WINDOW_BINS,
            "raw_feature_shape": [ridge.WINDOW_BINS, ridge.CHANNELS],
            "feature_dim": ridge.FEATURE_DIM,
            "output_dim": ridge.OUTPUT_DIM,
            "normalized_lambda": ridge.RIDGE_NORMALIZED_LAMBDA,
            "solver": "dual_ridge_support_only",
        },
        "selected_sessions": list(sessions),
        "session_source_receipts": receipt_rows,
        "compiled_streaming_profile": ridge.compiled_streaming_profile(),
    }
    return payload, loaded


def validate_preflight_receipt(
    path: Path,
    *,
    reference_manifest_path: Path,
    sessions: tuple[str, ...],
    expected_payload: Mapping[str, Any],
) -> dict[str, Any]:
    receipt = read_json_object(path.resolve())
    require(receipt.get("program_id") == ridge.PROGRAM_ID and receipt.get("mode") == "preflight", "preflight receipt program/mode mismatch")
    require(receipt.get("metric_computed") is False and receipt.get("predictions_generated") is False, "preflight receipt is not metrics-free")
    require(receipt.get("selected_sessions") == list(sessions), "preflight receipt session shard differs from requested forward shard")
    reference = receipt.get("reference_split_manifest", {})
    require(isinstance(reference, Mapping), "preflight receipt lacks reference manifest binding")
    expected_path = reference_manifest_path.resolve()
    require(reference.get("path") == str(expected_path), "preflight receipt manifest path drift")
    require(reference.get("sha256") == sha256_file(expected_path), "preflight receipt manifest content drift")
    # Checking individual fields is not enough: a source NWB, code file, or
    # raw query-layout change could otherwise hide behind the same session IDs.
    # Recompute the complete preflight receipt immediately before forward and
    # require byte-identical canonical content, including all NWB/core/runner
    # content hashes.  The receipt is a binding, not merely a checklist.
    require(
        canonical_bytes(receipt) == canonical_bytes(expected_payload),
        "stored preflight receipt differs from the immediately recomputed source/code audit",
    )
    return receipt


def forward_only(
    *,
    out_root: Path,
    preflight_path: Path,
    preflight_payload: Mapping[str, Any],
    loaded: Mapping[str, tuple[np.ndarray, np.ndarray, ridge.ChronologicalWindowLayout]],
    device: str,
) -> Path:
    """Fit/predict one session shard and write artifacts; deliberately no score."""
    out_root = out_root.resolve()
    require(not out_root.exists(), f"forward output root already exists; refusing to append/overwrite: {out_root}")
    out_root.mkdir(parents=True, exist_ok=False)
    per_session: dict[str, Any] = {}
    try:
        for session, (neural, covariates, layout) in loaded.items():
            support_x = ridge.materialize_raw_w50_features(neural, layout.support_target_bins)
            support_y = ridge.velocity_targets_at_bins(covariates, layout.support_target_bins)
            # The fit function accepts only support arrays: no query labels,
            # query metric, validation score, or lambda selection can enter.
            fitted = ridge.fit_dual_ridge_w50(support_x, support_y, device=device)
            compiled = ridge.compile_raw_ridge(fitted)
            query_x = ridge.materialize_raw_w50_features(neural, layout.query_target_bins)
            prediction = ridge.predict_compiled_raw_ridge(query_x, compiled)
            # Query target values are stored only as a future scorer input.  No
            # metric is calculated in this process.
            target = ridge.velocity_targets_at_bins(covariates, layout.query_target_bins)
            session_dir = out_root / "sessions" / session
            pred_path = session_dir / "predictions_targets.npz"
            state_path = session_dir / "compiled_readout.npz"
            pred_sha = write_npz_exclusive(
                pred_path,
                predictions=prediction,
                targets=target,
                query_target_bins=layout.query_target_bins,
            )
            state_sha = write_npz_exclusive(
                state_path,
                raw_weights=compiled.raw_weights,
                intercept=compiled.intercept,
                feature_mean=fitted.feature_mean,
                feature_scale=fitted.feature_scale,
                target_mean=fitted.target_mean,
                standardized_weights=fitted.standardized_weights,
            )
            per_session[session] = {
                "layout": layout.as_audit_dict(),
                "support_only_fit": {
                    "support_rows": fitted.support_rows,
                    "feature_dim": fitted.feature_dim,
                    "normalized_lambda": fitted.normalized_lambda,
                    "solver_device": device,
                },
                "prediction_target_artifact": {"path": str(pred_path), "sha256": pred_sha},
                "compiled_readout_artifact": {"path": str(state_path), "sha256": state_sha},
                "metric_computed": False,
            }
        commit = {
            "schema_version": 1,
            "program_id": ridge.PROGRAM_ID,
            "mode": "forward_only_unscored",
            "metric_computed": False,
            "selection_performed": False,
            "preflight_receipt": {"path": str(preflight_path.resolve()), "sha256": sha256_file(preflight_path.resolve())},
            "preflight_contract_sha256": hashlib.sha256(canonical_bytes(preflight_payload)).hexdigest(),
            "sessions": per_session,
            "compiled_streaming_profile": ridge.compiled_streaming_profile(),
            "scoring_notice": "No R2 is computed here. Any later metric must be a separate reviewed scorer using TorchMetrics 1.5.1.",
        }
        write_json_exclusive(out_root / "forward_commit.json", commit)
    except BaseException:
        # Do not delete partial output: it is an immutable incident record,
        # and a retry must use a fresh root rather than overwrite it.
        raise
    return out_root / "forward_commit.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "forward"), default="preflight")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "SPINT-main/data/000953")
    parser.add_argument("--reference-split-manifest", type=Path, required=True)
    parser.add_argument("--sessions", type=str, default=None, help="comma-separated exact held-out session IDs; defaults to all six")
    parser.add_argument("--out", type=Path, required=True, help="new immutable preflight receipt (preflight) or new forward output root (forward)")
    parser.add_argument("--preflight-receipt", type=Path, default=None, help="required matching receipt for forward mode")
    parser.add_argument("--device", type=str, default="cpu", help="dual solve device: cpu or cuda:<index>")
    parser.add_argument("--forward-review-token", default=None, help="required exact token after root review; forward still produces no metric")
    args = parser.parse_args()

    sessions = parse_session_selection(args.sessions)
    payload, loaded = audit_sources(args.data_dir, args.reference_split_manifest, sessions)
    if args.mode == "preflight":
        out = args.out.resolve()
        sha = write_json_exclusive(out, payload)
        print(json.dumps({"receipt": str(out), "sha256": sha, "sessions": list(sessions), "metric_computed": False}, indent=2))
        return

    require(args.preflight_receipt is not None, "forward mode requires --preflight-receipt")
    require(args.forward_review_token == FORWARD_REVIEW_TOKEN, "forward mode requires explicit root-review token")
    validate_preflight_receipt(
        args.preflight_receipt,
        reference_manifest_path=args.reference_split_manifest,
        sessions=sessions,
        expected_payload=payload,
    )
    commit = forward_only(
        out_root=args.out,
        preflight_path=args.preflight_receipt,
        preflight_payload=payload,
        loaded=loaded,
        device=args.device,
    )
    print(json.dumps({"forward_commit": str(commit), "sessions": list(sessions), "metric_computed": False}, indent=2))


if __name__ == "__main__":
    main()
