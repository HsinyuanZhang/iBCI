#!/usr/bin/env python3
"""Audit completed M1 muscle and replay scans without training or scoring models.

The command reads only completed receipts and saved prediction artifacts, then
writes a new JSON/CSV comparison under ``--output-dir``.  It refuses incomplete
24-epoch scans, drifted artifact bindings, or nonidentical HO targets/starts.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

EPOCHS = tuple(range(1, 25))
HO = ("20121004", "20121017", "20121024")
TOLERANCE = 1e-7


def _dependencies() -> None:
    """Delay optional numeric imports so ``--help`` remains dependency-free."""
    global np, r2_score
    import numpy as np  # type: ignore[no-redef]
    from sklearn.metrics import r2_score  # type: ignore[no-redef]


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    _need(path.is_file(), f"missing receipt: {path}")
    value = json.loads(path.read_text())
    _need(isinstance(value, dict), f"JSON object required: {path}")
    return value


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _completed_keys(value: Mapping[str, Any]) -> None:
    _need(set(value) == {str(epoch) for epoch in EPOCHS}, "scan must have exactly epochs 1..24")


def _metric_row_new(row: Mapping[str, Any]) -> tuple[float, float]:
    return float(row["r2"]), float(row["channel_variance_weighted_r2"])


def _metric_row_baseline(row: Mapping[str, Any]) -> tuple[float, float]:
    return float(row["legacy_variance_weighted_r2"]), float(row["sklearn_channel_centered_variance_weighted_r2"])


def _artifact_new(row: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    artifact = row.get("prediction_artifact")
    _need(isinstance(artifact, Mapping), "new scored epoch missing prediction artifact")
    path = Path(artifact.get("path", ""))
    _need(path.is_file() and artifact.get("sha256") == _sha(path), f"new prediction artifact SHA drift: {path}")
    return path, dict(row["ema_ho_calib"])


def _load_new_epoch(row: Mapping[str, Any]) -> tuple[dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]], dict[str, Any], Path]:
    path, report = _artifact_new(row)
    per = report.get("per_session")
    _need(isinstance(per, Mapping) and set(per) == set(HO), "new scored epoch session roster drift")
    arrays: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    with np.load(path, allow_pickle=False) as archive:
        expected = {f"{prefix}/{session}" for prefix in ("prediction", "target", "starts") for session in HO}
        _need(set(archive.files) == expected, f"new artifact keyset drift: {path}")
        for session in HO:
            prediction = np.ascontiguousarray(archive[f"prediction/{session}"])
            target = np.ascontiguousarray(archive[f"target/{session}"])
            starts = np.ascontiguousarray(archive[f"starts/{session}"])
            item = per[session]
            _need(prediction.shape == target.shape and prediction.ndim == 2 and prediction.shape[1] == 16, f"new geometry drift {session}")
            _need(starts.dtype == np.int64 and starts.ndim == 1 and len(starts) == len(target), f"new starts drift {session}")
            _need(_array_sha(prediction) == item.get("prediction_sha256"), f"new prediction row binding {session}")
            _need(_array_sha(target) == item.get("target_sha256"), f"new target row binding {session}")
            _need(_array_sha(starts) == item.get("starts_sha256"), f"new starts row binding {session}")
            arrays[session] = (prediction, target, starts)
    return arrays, report, path


def _load_baseline_epoch(root: Path, row: Mapping[str, Any]) -> tuple[dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]], dict[str, Any]]:
    metrics = row.get("metrics")
    _need(isinstance(metrics, Mapping), "baseline epoch lacks metrics")
    per = metrics.get("per_session")
    _need(isinstance(per, Mapping) and set(per) == set(HO), "baseline epoch session roster drift")
    epoch_dir = root / str(row.get("artifact_dir", ""))
    _need(epoch_dir.is_dir(), f"baseline artifact directory missing: {epoch_dir}")
    arrays: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for session in HO:
        item = per[session]
        path = epoch_dir / str(item.get("artifact", ""))
        _need(path.is_file() and item.get("artifact_sha256") == _sha(path), f"baseline artifact SHA drift: {path}")
        with np.load(path, allow_pickle=False) as archive:
            _need(set(archive.files) == {"pred", "y", "starts"}, f"baseline artifact keyset drift: {path}")
            prediction, target, starts = (np.ascontiguousarray(archive[key]) for key in ("pred", "y", "starts"))
        _need(prediction.shape == target.shape and prediction.ndim == 2 and prediction.shape[1] == 16, f"baseline geometry drift {session}")
        _need(starts.dtype == np.int64 and starts.ndim == 1 and len(starts) == len(target), f"baseline starts drift {session}")
        _need(_array_sha(prediction) == item.get("prediction_sha256"), f"baseline prediction row binding {session}")
        _need(_array_sha(target) == item.get("target_sha256"), f"baseline target row binding {session}")
        _need(_array_sha(starts) == item.get("starts_sha256"), f"baseline starts row binding {session}")
        arrays[session] = (prediction, target, starts)
    return arrays, dict(metrics)


def _recompute_channel(arrays: Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray]], report: Mapping[str, Any], kind: str) -> None:
    per = report["per_session"]
    values = []
    for session in HO:
        prediction, target, _starts = arrays[session]
        observed = float(r2_score(target, prediction, multioutput="variance_weighted"))
        recorded = _metric_row_new(per[session])[1] if kind == "new" else _metric_row_baseline(per[session])[1]
        _need(abs(observed - recorded) <= TOLERANCE, f"{kind} channel R2 replay drift {session}: {observed} vs {recorded}")
        values.append(observed)
    aggregate = float(np.mean(values))
    key = "equal_session_mean_channel_variance_weighted_r2" if kind == "new" else "equal_session_mean_sklearn_channel_centered"
    _need(abs(aggregate - float(report[key])) <= TOLERANCE, f"{kind} channel aggregate replay drift")


def _selection(curve: Mapping[str, Mapping[str, float]], metric: str) -> int:
    return min(EPOCHS, key=lambda epoch: (-float(curve[str(epoch)][metric]), epoch))


def _validate_targets(reference: Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray]], observed: Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray]], label: str) -> None:
    for session in HO:
        _pred_a, target_a, starts_a = reference[session]
        _pred_b, target_b, starts_b = observed[session]
        _need(target_a.dtype == target_b.dtype == np.float32 and target_a.shape == target_b.shape, f"target dtype/shape mismatch {label}/{session}")
        _need(starts_a.dtype == starts_b.dtype == np.int64 and starts_a.shape == starts_b.shape, f"starts dtype/shape mismatch {label}/{session}")
        _need(_array_sha(target_a) == _array_sha(target_b), f"target SHA mismatch {label}/{session}")
        _need(_array_sha(starts_a) == _array_sha(starts_b), f"starts SHA mismatch {label}/{session}")
        _need(np.array_equal(target_a, target_b), f"target mismatch {label}/{session}")
        _need(np.array_equal(starts_a, starts_b), f"starts mismatch {label}/{session}")


def _selected_detail(new_curve: Mapping[str, Mapping[str, Any]], baseline_curve: Mapping[str, Mapping[str, Any]],
                     new_epoch: int, baseline_epoch: int, metric: str) -> dict[str, Any]:
    new_report, old_report = new_curve[str(new_epoch)], baseline_curve[str(baseline_epoch)]
    rows = {}
    for session in HO:
        new_legacy, new_channel = _metric_row_new(new_report["per_session"][session])
        old_legacy, old_channel = _metric_row_baseline(old_report["per_session"][session])
        rows[session] = {"new": {"legacy_flattened_r2": new_legacy, "channel_variance_weighted_r2": new_channel},
                         "baseline": {"legacy_flattened_r2": old_legacy, "channel_variance_weighted_r2": old_channel},
                         "delta_new_minus_baseline": {"legacy_flattened_r2": new_legacy - old_legacy,
                                                        "channel_variance_weighted_r2": new_channel - old_channel}}
    new_equal = {"legacy_flattened_r2": float(new_report["equal_session_mean"]),
                 "channel_variance_weighted_r2": float(new_report["equal_session_mean_channel_variance_weighted_r2"])}
    baseline_equal = {"legacy_flattened_r2": float(old_report["equal_session_mean_legacy"]),
                      "channel_variance_weighted_r2": float(old_report["equal_session_mean_sklearn_channel_centered"])}
    return {"metric": metric, "new_epoch": new_epoch, "baseline_epoch": baseline_epoch,
            "equal_session_mean": {"new": new_equal, "baseline": baseline_equal,
                                   "delta_new_minus_baseline": {key: new_equal[key] - baseline_equal[key] for key in new_equal}},
            "per_session": rows}


def compare(args: argparse.Namespace) -> dict[str, Any]:
    _dependencies()
    run, baseline, output = args.run_root.resolve(), args.baseline_root.resolve(), args.output_dir.resolve()
    _need(not output.exists(), f"output directory must be new: {output}")
    new_receipt = _read_json(run / "score_receipt.json")
    new_progress = _read_json(run / "score_progress.json")
    old_receipt = _read_json(baseline / "replay_receipt.json")
    _need(new_receipt.get("status") == "COMPLETED" and old_receipt.get("status") == "COMPLETED", "both completed score/replay receipts required")
    new_completed, old_completed = new_progress.get("completed"), old_receipt.get("completed")
    _need(isinstance(new_completed, Mapping) and isinstance(old_completed, Mapping), "completed epoch maps required")
    _completed_keys(new_completed); _completed_keys(old_completed)
    receipt_curve, receipt_checkpoints = new_receipt.get("ema_by_epoch"), new_receipt.get("checkpoint_sha256_by_epoch")
    _need(isinstance(receipt_curve, Mapping) and isinstance(receipt_checkpoints, Mapping), "new score receipt curve/checkpoint map required")
    _completed_keys(receipt_curve); _completed_keys(receipt_checkpoints)
    new_curve, old_curve, csv_rows = {}, {}, []
    target_reference: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None
    for epoch in EPOCHS:
        new_arrays, new_report, new_artifact = _load_new_epoch(new_completed[str(epoch)])
        old_arrays, old_report = _load_baseline_epoch(baseline, old_completed[str(epoch)])
        _need(new_report == receipt_curve[str(epoch)], f"new progress/score receipt metric drift epoch {epoch}")
        _need(new_completed[str(epoch)].get("checkpoint_sha256") == receipt_checkpoints[str(epoch)], f"new progress/score receipt checkpoint drift epoch {epoch}")
        _recompute_channel(new_arrays, new_report, "new"); _recompute_channel(old_arrays, old_report, "baseline")
        _validate_targets(new_arrays, old_arrays, f"new-vs-baseline/epoch{epoch}")
        if target_reference is None: target_reference = new_arrays
        else:
            _validate_targets(target_reference, new_arrays, f"new-within-scan/epoch{epoch}")
            _validate_targets(target_reference, old_arrays, f"baseline-within-scan/epoch{epoch}")
        new_curve[str(epoch)], old_curve[str(epoch)] = new_report, old_report
        new_legacy, new_channel = float(new_report["equal_session_mean"]), float(new_report["equal_session_mean_channel_variance_weighted_r2"])
        old_legacy, old_channel = float(old_report["equal_session_mean_legacy"]), float(old_report["equal_session_mean_sklearn_channel_centered"])
        csv_rows.extend((
            {"epoch": epoch, "arm": "new_muscle", "legacy_flattened_r2": new_legacy, "channel_variance_weighted_r2": new_channel, "prediction_artifact": str(new_artifact)},
            {"epoch": epoch, "arm": "baseline_replay", "legacy_flattened_r2": old_legacy, "channel_variance_weighted_r2": old_channel, "prediction_artifact": str(baseline / old_completed[str(epoch)]["artifact_dir"])},
        ))
    new_official, old_official = _selection(new_curve, "equal_session_mean_channel_variance_weighted_r2"), _selection(old_curve, "equal_session_mean_sklearn_channel_centered")
    new_legacy, old_legacy = _selection(new_curve, "equal_session_mean"), _selection(old_curve, "equal_session_mean_legacy")
    _need(int(new_receipt.get("selection", {}).get("epoch", -1)) == new_official, "new receipt official selection drift")
    _need(int(new_receipt.get("legacy_selection", {}).get("epoch", -1)) == new_legacy, "new receipt legacy selection drift")
    _need(int(old_receipt.get("selection", {}).get("epoch", -1)) == old_official, "baseline receipt official selection drift")
    _need(int(old_receipt.get("original_legacy_selection", {}).get("epoch", -1)) == old_legacy, "baseline receipt legacy selection drift")
    body = {
        "schema": "m1_muscle_r100_completed_comparison_v1", "status": "COMPLETED",
        "run_root": str(run), "baseline_root": str(baseline),
        "run_score_receipt_sha256": _sha(run / "score_receipt.json"), "run_score_progress_sha256": _sha(run / "score_progress.json"),
        "baseline_replay_receipt_sha256": _sha(baseline / "replay_receipt.json"),
        "epochs": list(EPOCHS), "tolerance": TOLERANCE,
        "curve": {"new_muscle": new_curve, "baseline_replay": old_curve},
        "selection": {"channel_variance_weighted_r2": {"new_epoch": new_official, "baseline_epoch": old_official,
                       "rule": "earliest maximum equal-session mean sklearn channel-centered variance-weighted R2"},
                      "legacy_flattened_r2": {"new_epoch": new_legacy, "baseline_epoch": old_legacy,
                       "rule": "earliest maximum equal-session mean legacy flattened R2"}},
        "selected_per_session": {"channel_variance_weighted_r2": _selected_detail(new_curve, old_curve, new_official, old_official, "channel_variance_weighted_r2"),
                                 "legacy_flattened_r2": _selected_detail(new_curve, old_curve, new_legacy, old_legacy, "legacy_flattened_r2")},
        "target_and_starts_exact": True,
    }
    output.mkdir(parents=True)
    _atomic_json(output / "comparison.json", body)
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("epoch", "arm", "legacy_flattened_r2", "channel_variance_weighted_r2", "prediction_artifact"))
        writer.writeheader(); writer.writerows(csv_rows)
    return {"status": "COMPLETED", "json": str(output / "comparison.json"), "csv": str(output / "comparison.csv")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--baseline-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(args), sort_keys=True))


if __name__ == "__main__":
    main()
