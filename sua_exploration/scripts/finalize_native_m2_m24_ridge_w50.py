#!/usr/bin/env python3
"""One-shot local finalizer for two unscored Native-M2 M24 Ridge-W50 shards.

This file is intentionally separate from the fit/forward runner.  It opens
only immutable forward artifacts, checks that their two session shards form the
exact six-session M24 grid, and computes per-session variance-weighted R² with
TorchMetrics 1.5.1.  It then reports the descriptive paired ``K4 - Ridge``
delta against a supplied completed M24 K4 metrics CSV.

It is a *closure comparator*, not a model-selection path: there is no sweep,
retraining, lambda change, or K4 revival gate here.  Invocation requires an
explicit root-review token and writes one new immutable JSON only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
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


FINALIZE_REVIEW_TOKEN = "ROOT_REVIEWED_FINALIZE_ONCE_TORCHMETRICS151"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ridge.NativeM2RidgeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(dict(value), sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ridge.NativeM2RidgeError(f"cannot read JSON object {path}: {exc}") from exc
    require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def write_exclusive(path: Path, value: Mapping[str, Any]) -> str:
    raw = canonical_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(path, 0o444)
    require(path.is_file() and stat.S_IMODE(path.stat().st_mode) == 0o444, f"immutable finalizer output failed: {path}")
    return hashlib.sha256(raw).hexdigest()


def read_completed_arm_scores(path: Path, *, arm: str) -> dict[str, float]:
    """Read exactly the six existing strict M24 arm test-heldout values."""
    scores: dict[str, float] = {}
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("split") != "test_heldout":
                    continue
                session = str(row.get("session", ""))
                require(session in ridge.EXPECTED_HELDOUT_SESSIONS, f"{arm} CSV has unexpected held-out session: {session}")
                require(int(float(row.get("M", "nan"))) == ridge.CALIBRATION_TRIALS, f"{session}: {arm} CSV is not M24")
                value = float(row["R2_variance_weighted"])
                require(math.isfinite(value), f"{session}: {arm} R² is non-finite")
                require(session not in scores, f"{arm} CSV duplicates session: {session}")
                scores[session] = value
    except OSError as exc:
        raise ridge.NativeM2RidgeError(f"cannot read {arm} metrics CSV {path}: {exc}") from exc
    require(set(scores) == set(ridge.EXPECTED_HELDOUT_SESSIONS), f"{arm} CSV must have exact six strict M24 test-heldout rows")
    return scores


def _require_close(observed: float, expected: float, what: str) -> None:
    require(math.isclose(float(observed), float(expected), rel_tol=0.0, abs_tol=1.0e-8), f"{what}: expected {expected}, found {observed}")


def validate_existing_m24_reference(
    aggregate_path: Path,
    *,
    k4_metrics: Path,
    t4_metrics: Path,
    k4_scores: Mapping[str, float],
    t4_scores: Mapping[str, float],
) -> dict[str, Any]:
    """Bind the classical comparator to the completed strict M24 K4/T4 replay.

    The old aggregate is not treated as a score oracle: its per-arm mean and
    K4−T4 paired delta must be re-derived from the pinned complete CSVs.
    """
    aggregate_path = aggregate_path.resolve()
    aggregate = read_json_object(aggregate_path)
    require(aggregate.get("formal_heldout_evaluated") is True, "M24 aggregate is not a held-out replay")
    cell = aggregate.get("cell", {})
    require(isinstance(cell, Mapping), "M24 aggregate lacks cell metadata")
    require(cell.get("task") == "m2" and cell.get("calibration_trials") == ridge.CALIBRATION_TRIALS and cell.get("window_size") == ridge.WINDOW_BINS, "M24 aggregate cell contract drift")
    arms = aggregate.get("arms")
    require(isinstance(arms, Mapping) and {"k4", "t4"}.issubset(arms), "M24 aggregate lacks K4/T4 arms")
    k4_mean = float(np.mean(list(k4_scores.values())))
    t4_mean = float(np.mean(list(t4_scores.values())))
    _require_close(arms["k4"].get("mean_r2"), k4_mean, "aggregate K4 mean")
    _require_close(arms["t4"].get("mean_r2"), t4_mean, "aggregate T4 mean")
    deltas = {session: k4_scores[session] - t4_scores[session] for session in ridge.EXPECTED_HELDOUT_SESSIONS}
    reported = aggregate.get("paired_deltas_r2", {}).get("K4_minus_T4", {})
    require(isinstance(reported, Mapping), "M24 aggregate lacks K4_minus_T4 delta")
    _require_close(reported.get("mean"), float(np.mean(list(deltas.values()))), "aggregate K4−T4 mean")
    reported_per_session = reported.get("per_session")
    require(isinstance(reported_per_session, Mapping), "M24 aggregate lacks K4−T4 per-session delta")
    for session, value in deltas.items():
        _require_close(reported_per_session.get(session), value, f"aggregate K4−T4 {session}")
    return {
        "aggregate_json": str(aggregate_path),
        "aggregate_sha256": sha256_file(aggregate_path),
        "k4_metrics_csv": str(k4_metrics.resolve()),
        "k4_metrics_sha256": sha256_file(k4_metrics.resolve()),
        "t4_metrics_csv": str(t4_metrics.resolve()),
        "t4_metrics_sha256": sha256_file(t4_metrics.resolve()),
        "aggregate_k4_minus_t4_verified": True,
    }


def _load_forward_commit(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    commit_path = path.resolve()
    commit = read_json_object(commit_path)
    require(commit.get("program_id") == ridge.PROGRAM_ID, f"{commit_path}: wrong forward program")
    require(commit.get("mode") == "forward_only_unscored", f"{commit_path}: forward commit was not unscored")
    require(commit.get("metric_computed") is False and commit.get("selection_performed") is False, f"{commit_path}: forward commit has an invalid score/selection state")
    receipt_info = commit.get("preflight_receipt")
    require(isinstance(receipt_info, Mapping), f"{commit_path}: missing preflight receipt")
    receipt_path = Path(str(receipt_info.get("path", ""))).resolve()
    require(receipt_path.is_file() and sha256_file(receipt_path) == receipt_info.get("sha256"), f"{commit_path}: preflight receipt hash drift")
    receipt = read_json_object(receipt_path)
    require(receipt.get("program_id") == ridge.PROGRAM_ID and receipt.get("mode") == "preflight", f"{commit_path}: invalid preflight receipt")
    require(receipt.get("metric_computed") is False and receipt.get("predictions_generated") is False, f"{commit_path}: preflight was not metrics-free")
    expected_contract_sha = hashlib.sha256(canonical_bytes(receipt)).hexdigest()
    require(commit.get("preflight_contract_sha256") == expected_contract_sha, f"{commit_path}: forward/preflight binding drift")
    return commit, receipt


def load_two_shard_predictions(commit_paths: list[Path]) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, Any]]:
    """Verify and collect the exact six session artifacts from two shards."""
    require(len(commit_paths) == 2, "finalizer requires exactly two independently completed forward shards")
    observed: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    common_receipt_binding: dict[str, Any] | None = None
    commit_receipts: list[dict[str, str]] = []
    for commit_path in commit_paths:
        commit, receipt = _load_forward_commit(commit_path)
        binding = {
            "reference_split_manifest": receipt.get("reference_split_manifest"),
            "implementation": receipt.get("implementation"),
            "contract": receipt.get("contract"),
        }
        if common_receipt_binding is None:
            common_receipt_binding = binding
        else:
            require(binding == common_receipt_binding, "forward shards do not bind the same source/code/contract receipt")
        sessions = commit.get("sessions")
        require(isinstance(sessions, Mapping) and sessions, f"{commit_path}: missing forward session artifacts")
        for session, row in sessions.items():
            require(session in ridge.EXPECTED_HELDOUT_SESSIONS, f"{commit_path}: unexpected session {session}")
            require(session not in observed, f"duplicate session across shards: {session}")
            require(isinstance(row, Mapping), f"{commit_path}/{session}: malformed forward row")
            layout = row.get("layout")
            require(isinstance(layout, Mapping), f"{commit_path}/{session}: missing layout")
            expected = ridge.EXPECTED_HELDOUT_LAYOUT[session]
            require(layout.get("query_rows") == expected["query_rows"], f"{session}: query-row count drift")
            artifact = row.get("prediction_target_artifact")
            require(isinstance(artifact, Mapping), f"{commit_path}/{session}: missing prediction artifact")
            path = Path(str(artifact.get("path", ""))).resolve()
            require(path.is_file() and sha256_file(path) == artifact.get("sha256"), f"{session}: prediction artifact hash drift")
            with np.load(path, allow_pickle=False) as arrays:
                prediction = np.ascontiguousarray(arrays["predictions"], dtype=np.float32)
                target = np.ascontiguousarray(arrays["targets"], dtype=np.float32)
                bins = np.ascontiguousarray(arrays["query_target_bins"], dtype=np.int64)
            require(prediction.shape == target.shape == (expected["query_rows"], ridge.OUTPUT_DIM), f"{session}: prediction/target shape drift")
            require(np.isfinite(prediction).all() and np.isfinite(target).all(), f"{session}: non-finite prediction/target artifact")
            require(ridge.sha256_array_int64(bins) == layout.get("query_target_bins_sha256"), f"{session}: query bin hash drift")
            observed[session] = (prediction, target)
        commit_receipts.append({"path": str(commit_path.resolve()), "sha256": sha256_file(commit_path.resolve())})
    require(set(observed) == set(ridge.EXPECTED_HELDOUT_SESSIONS), "two forward shards do not form the exact six-session M24 grid")
    assert common_receipt_binding is not None
    return observed, {"forward_commits": commit_receipts, "common_preflight_binding": common_receipt_binding}


def finalize(commit_paths: list[Path], aggregate_path: Path, k4_metrics: Path, t4_metrics: Path) -> dict[str, Any]:
    predictions, binding = load_two_shard_predictions(commit_paths)
    ridge_scores = {
        session: ridge.score_r2_variance_weighted_torchmetrics151(prediction, target)
        for session, (prediction, target) in sorted(predictions.items())
    }
    k4_metrics = k4_metrics.resolve()
    t4_metrics = t4_metrics.resolve()
    k4_scores = read_completed_arm_scores(k4_metrics, arm="K4")
    t4_scores = read_completed_arm_scores(t4_metrics, arm="T4")
    reference_binding = validate_existing_m24_reference(
        aggregate_path,
        k4_metrics=k4_metrics,
        t4_metrics=t4_metrics,
        k4_scores=k4_scores,
        t4_scores=t4_scores,
    )
    k4_minus_ridge = {session: k4_scores[session] - ridge_scores[session] for session in ridge.EXPECTED_HELDOUT_SESSIONS}
    t4_minus_ridge = {session: t4_scores[session] - ridge_scores[session] for session in ridge.EXPECTED_HELDOUT_SESSIONS}
    k4_minus_t4 = {session: k4_scores[session] - t4_scores[session] for session in ridge.EXPECTED_HELDOUT_SESSIONS}
    return {
        "schema_version": 1,
        "program_id": ridge.PROGRAM_ID,
        "mode": "one_shot_local_finalization",
        "torchmetrics_version": "1.5.1",
        "closure_comparator_only": True,
        "not_a_k4_revival_gate": True,
        "no_lambda_window_or_architecture_sweep": True,
        "forward_binding": binding,
        "existing_strict_m24_reference_binding": reference_binding,
        "ridge_w50_r2_variance_weighted_per_session": ridge_scores,
        "ridge_w50_r2_variance_weighted_mean_unweighted_sessions": float(np.mean(list(ridge_scores.values()))),
        "completed_k4_m24_reference": {
            "r2_variance_weighted_per_session": k4_scores,
            "r2_variance_weighted_mean_unweighted_sessions": float(np.mean(list(k4_scores.values()))),
        },
        "completed_t4_m24_reference": {
            "r2_variance_weighted_per_session": t4_scores,
            "r2_variance_weighted_mean_unweighted_sessions": float(np.mean(list(t4_scores.values()))),
        },
        "k4_minus_ridge_w50_r2_per_session": k4_minus_ridge,
        "k4_minus_ridge_w50_r2_mean_unweighted_sessions": float(np.mean(list(k4_minus_ridge.values()))),
        "t4_minus_ridge_w50_r2_per_session": t4_minus_ridge,
        "t4_minus_ridge_w50_r2_mean_unweighted_sessions": float(np.mean(list(t4_minus_ridge.values()))),
        "k4_minus_t4_r2_per_session": k4_minus_t4,
        "k4_minus_t4_r2_mean_unweighted_sessions": float(np.mean(list(k4_minus_t4.values()))),
        "interpretation_scope": "local six-session M24 chronological held-out replay; a descriptive one-shot comparison of a source-pretrained shared-decoder system (K4/T4) against a per-session dense-label direct ridge. It does not attribute the K4-versus-ridge gap to K4 descriptor content, establish a K4 mechanism, or reopen the K4 mainline.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forward-commit", type=Path, action="append", required=True, help="repeat exactly twice, once per 3-session forward shard")
    parser.add_argument("--m24-aggregate", type=Path, required=True, help="completed strict M24 aggregate_heldout.json")
    parser.add_argument("--k4-metrics", type=Path, required=True, help="completed strict M24 K4 metrics_per_session.csv")
    parser.add_argument("--t4-metrics", type=Path, required=True, help="completed strict M24 T4 metrics_per_session.csv")
    parser.add_argument("--out", type=Path, required=True, help="new immutable final JSON")
    parser.add_argument("--finalize-review-token", default=None, help="required exact root-review token")
    args = parser.parse_args()
    require(args.finalize_review_token == FINALIZE_REVIEW_TOKEN, "finalization requires explicit root-review token")
    out = args.out.resolve()
    require(not out.exists(), f"refusing to overwrite finalizer output: {out}")
    payload = finalize(list(args.forward_commit), args.m24_aggregate, args.k4_metrics, args.t4_metrics)
    sha = write_exclusive(out, payload)
    print(json.dumps({"out": str(out), "sha256": sha, "mode": payload["mode"]}, indent=2))


if __name__ == "__main__":
    main()
