#!/usr/bin/env python3
"""Fail-closed paired audit for all chronological M1 Z/B/D target artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT / "src"), str(ROOT.parent / "btransform_unified_v1" / "src"), str(ROOT.parent)]

import m1_score as score
import m1_train as train
from m1_data import SOURCE_SESSIONS, TARGET_SESSIONS, materialize_sources, materialize_target
from btransform_unified_v1.r2 import variance_weighted_r2
from btransform_unified_v2.cross_session_m1_model import ARMS, B_ACTIVITY_ONLY, D_JOINT, Z_NONE


ENDPOINTS = (("selected", "target_selected_ema", "target_selected_predictions.npz"),
             ("epoch24", "target_epoch24_ema", "target_epoch24_predictions.npz"))
COORDINATES = ("window_start_padded", "output_index_padded", "output_index_query_relative", "prefix_bins")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"required JSON missing: {path}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def current_code(binding: dict[str, Any], label: str) -> None:
    require(isinstance(binding, dict) and binding, f"{label}: source code binding absent")
    for raw, digest in binding.items():
        path = Path(raw)
        require(path.is_file() and isinstance(digest, str) and sha(path) == digest, f"{label}: source-code binding drift {path}")


def date_of(session: str) -> str:
    require(session.startswith("ses-") and len(session) == len("ses-20120927"), f"invalid chronological M1 session: {session}")
    compact = session.removeprefix("ses-")
    return f"{compact[:4]}-{compact[4:6]}-{compact[6:]}"


def expected_target_arrays(target: str, source: dict[str, Any]) -> dict[str, np.ndarray]:
    """Rebuild QueryOnly physical labels/coordinates; never instantiate a model."""
    target_data = materialize_target(target, source, Z_NONE)["target"]
    labels: list[np.ndarray] = []
    starts: list[np.ndarray] = []
    for _x, y, sessions, _valid, batch_starts in train.loader(target_data, 42, shuffle=False):
        require(set(sessions) == {target}, f"{target}: QueryOnly loader session drift")
        labels.append(np.ascontiguousarray(y[:, -1].numpy()))
        starts.append(np.ascontiguousarray(batch_starts.numpy().astype(np.int64)))
    window_start = np.ascontiguousarray(np.concatenate(starts), dtype=np.int64)
    prefix = int(np.asarray(target_data.trial_start_indices[target])[0])
    return {
        "target": np.ascontiguousarray(np.concatenate(labels)),
        "window_start_padded": window_start,
        "output_index_padded": np.ascontiguousarray(window_start + target_data.window_size - 1, dtype=np.int64),
        "output_index_query_relative": np.ascontiguousarray(window_start + target_data.window_size - 1 - prefix, dtype=np.int64),
        "prefix_bins": np.asarray([prefix], dtype=np.int64),
    }


def check_artifact(path: Path, recorded_sha: Any, score_row: dict[str, Any], physical: dict[str, np.ndarray], label: str) -> dict[str, Any]:
    require(path.is_file() and isinstance(recorded_sha, str) and sha(path) == recorded_sha, f"{label}: artifact SHA/path mismatch")
    with np.load(path, allow_pickle=False) as arrays:
        require(set(arrays.files) == {"target", "prediction", *COORDINATES}, f"{label}: artifact schema mismatch")
        target, prediction = arrays["target"], arrays["prediction"]
        require(target.dtype == physical["target"].dtype and prediction.dtype == target.dtype, f"{label}: target/prediction dtype mismatch")
        require(target.shape == prediction.shape == physical["target"].shape and np.isfinite(target).all() and np.isfinite(prediction).all(), f"{label}: target/prediction shape or finiteness mismatch")
        require(np.array_equal(target, physical["target"]), f"{label}: target labels do not equal physical QueryOnly labels")
        coordinates: dict[str, np.ndarray] = {}
        for name in COORDINATES:
            value = arrays[name]
            require(value.dtype == np.int64 and value.shape == physical[name].shape and np.array_equal(value, physical[name]), f"{label}: physical coordinate mismatch {name}")
            coordinates[name] = np.ascontiguousarray(value)
        r2 = float(variance_weighted_r2(target, prediction))
        report = score_row.get("per_session")
        require(isinstance(report, dict) and set(report) == {label.split("/")[1]}, f"{label}: exact target-session score report required")
        reported = report[label.split("/")[1]]
        require(math.isfinite(float(reported)) and abs(r2 - float(reported)) <= 1e-12 and abs(float(score_row.get("equal_session_mean", float("nan"))) - r2) <= 1e-12, f"{label}: canonical R2/report mismatch")
    return {"path": str(path.resolve()), "sha256": recorded_sha, "r2": r2,
            "target_sha256": hashlib.sha256(np.ascontiguousarray(physical["target"]).tobytes()).hexdigest(), "coordinates": coordinates}


def audit_arm(run: Path, arm: str, physical: dict[str, dict[str, np.ndarray]], receipt: dict[str, Any]) -> dict[str, Any]:
    args = SimpleNamespace(dest=run, arm=arm, seed=42, device="cpu")
    preflight = read(run / "preflight.json")
    require(receipt.get("actual_source_arrays") == train.source_evidence(SOURCE), f"{arm}: receipt source evidence differs from shared materialization")
    require(preflight.get("source_evidence") == receipt["actual_source_arrays"], f"{arm}: preflight/train source evidence mismatch")
    checks = preflight.get("checks", {})
    init = checks.get("init", {}) if isinstance(checks, dict) else {}
    require(init.get("B3S_bytes_equal") is True and isinstance(init.get("shared_ZBD_hash"), str), f"{arm}: full-three-arm preflight initialization proof absent")
    require(all(isinstance(checks.get(name), dict) and checks[name].get("finite") is True and checks[name].get("has_grad") is True for name in ARMS), f"{arm}: full-three-arm preflight gradient proof absent")
    require(checks.get("Z_calibration_forbidden") is True and checks.get("valid_mask_mapping_checked") is True, f"{arm}: preflight calibration/mask proof absent")
    artifacts: dict[str, dict[str, Any]] = {}
    scores: dict[str, dict[str, Any]] = {}
    for target in TARGET_SESSIONS:
        score_dir = run / f"score_{target}"
        score_receipt = read(score_dir / "score_receipt.json")
        expected = train.metadata(args)
        for key in ("split_id", "sources", "targets", "arm", "seed", "data_contract"):
            require(score_receipt.get(key) == expected[key], f"{arm}/{target}: score metadata drift {key}")
        require(score_receipt.get("status") == "COMPLETED" and score_receipt.get("target") == target, f"{arm}/{target}: score identity/status mismatch")
        require(score_receipt.get("source_selected") == receipt["selected"] and score_receipt.get("source_curve_sha256") == receipt["source_curve_sha256"], f"{arm}/{target}: source selection/curve binding mismatch")
        require(score_receipt.get("checkpoint_sha256") == receipt["checkpoint_sha256"], f"{arm}/{target}: checkpoint binding mismatch")
        require(score_receipt.get("target_labels_used_for_selection") is False and score_receipt.get("target_optimizer_steps") == 0, f"{arm}/{target}: target-use scope mismatch")
        current_code(score_receipt.get("source_code_sha256"), f"{arm}/{target} score")
        array_specs = score_receipt.get("target_audit_arrays")
        require(isinstance(array_specs, dict), f"{arm}/{target}: array specification absent")
        artifacts[target] = {}
        scores[target] = score_receipt
        for endpoint, report_key, filename in ENDPOINTS:
            spec = array_specs.get("selected" if endpoint == "selected" else "epoch24")
            require(isinstance(spec, dict) and spec.get("file") == filename, f"{arm}/{target}/{endpoint}: artifact filename binding mismatch")
            artifacts[target][endpoint] = check_artifact(score_dir / filename, spec.get("sha256"), score_receipt.get(report_key, {}), physical[target], f"{arm}/{target}/{endpoint}")
    return {"receipt": receipt, "preflight": preflight, "artifacts": artifacts, "scores": scores}


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    require(not path.exists(), f"refusing to overwrite audit output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def add_declared_code(bound: dict[str, str], binding: dict[str, Any], label: str) -> None:
    current_code(binding, label)
    for raw, digest in binding.items():
        path = Path(raw).resolve()
        require(bound.get(str(path), digest) == digest, f"{label}: conflicting declared source-code SHA {path}")
        bound[str(path)] = digest


def all_bound_inputs(run_root: Path, arms: dict[str, dict[str, Any]]) -> dict[str, str]:
    paths: set[Path] = {Path(__file__), HERE / "m1_data.py", HERE / "m1_train.py", HERE / "m1_score.py"}
    bound: dict[str, str] = {}
    for arm in ARMS:
        run, item = run_root / arm / "s42", arms[arm]
        paths.update((run / "train_receipt.json", run / "preflight.json", run / "resume_latest.pt", run / "selected_ema.pt"))
        for epoch in range(1, train.EPOCHS + 1):
            paths.update((run / f"ema_epoch_{epoch:03d}.json", run / f"ema_epoch_{epoch:03d}.pt"))
        add_declared_code(bound, item["receipt"].get("source_code_sha256", {}), f"{arm} train")
        add_declared_code(bound, item["preflight"].get("source_code_sha256", {}), f"{arm} preflight")
        for target in TARGET_SESSIONS:
            score_dir = run / f"score_{target}"
            paths.add(score_dir / "score_receipt.json")
            for endpoint, _report_key, _filename in ENDPOINTS:
                paths.add(Path(item["artifacts"][target][endpoint]["path"]))
            add_declared_code(bound, item["scores"][target].get("source_code_sha256", {}), f"{arm}/{target} score")
    cache = run_root / "source_prepare_cache"
    require(cache.is_dir(), f"source cache directory missing: {cache}")
    paths.update(path for path in cache.iterdir() if path.is_file())
    for path in paths:
        require(path.is_file(), f"bound input missing: {path}")
        resolved, digest = str(path.resolve()), sha(path)
        require(resolved not in bound or bound[resolved] == digest, f"bound input/code SHA disagreement: {path}")
        bound[resolved] = digest
    return dict(sorted(bound.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=ROOT / "results/chronological_last2_v1/m1")
    parser.add_argument("--dest", type=Path, required=True)
    args = parser.parse_args()
    run_root, destination = args.run_root.resolve(), args.dest.resolve()
    require(run_root.is_dir(), f"M1 run root missing: {run_root}")
    # This gate deliberately precedes every target materialization.  It opens
    # only completed source-run receipts/checkpoints and current code bindings.
    source_gate: dict[str, tuple[dict[str, Any], Any, Any]] = {}
    for arm in ARMS:
        run = run_root / arm / "s42"
        source_gate[arm] = score.require_train(SimpleNamespace(dest=run, arm=arm, seed=42, device="cpu"))
    global SOURCE
    SOURCE = materialize_sources(cache=run_root / "source_prepare_cache")
    require(tuple(SOURCE["sources"]) == SOURCE_SESSIONS, "shared source materialization roster mismatch")
    physical = {target: expected_target_arrays(target, SOURCE) for target in TARGET_SESSIONS}
    arms = {arm: audit_arm(run_root / arm / "s42", arm, physical, source_gate[arm][0]) for arm in ARMS}
    source_receipts = {arm: arms[arm]["receipt"] for arm in ARMS}
    for key in ("actual_source_arrays", "batch_order_sha256", "steps"):
        require(len({json.dumps(source_receipts[arm].get(key), sort_keys=True, default=str) for arm in ARMS}) == 1, f"paired source field differs across Z/B/D: {key}")
    require(source_receipts[B_ACTIVITY_ONLY].get("initialization_hash") == source_receipts[D_JOINT].get("initialization_hash"), "B/D initialization SHA differs")
    z_preflight = arms[Z_NONE]["preflight"]
    z_path = (run_root / Z_NONE / "s42" / "preflight.json").resolve()
    require("shared_numerical_preflight" not in z_preflight, "Z preflight must be the un-derived numerical authority")
    def normalized_preflight(value: dict[str, Any]) -> dict[str, Any]:
        return {key: item for key, item in value.items() if key not in ("arm", "shared_numerical_preflight")}
    for arm in ARMS:
        preflight = arms[arm]["preflight"]
        require(preflight["checks"]["init"]["shared_ZBD_hash"] == z_preflight["checks"]["init"]["shared_ZBD_hash"], "preflight common initialization hash differs across arms")
        if arm in (B_ACTIVITY_ONLY, D_JOINT):
            shared = preflight.get("shared_numerical_preflight")
            require(isinstance(shared, dict) and set(shared) == {"path", "sha256", "scope"}, f"{arm}: derived preflight reference schema mismatch")
            require(Path(shared["path"]).resolve() == z_path and shared["sha256"] == sha(z_path) and isinstance(shared["scope"], str) and bool(shared["scope"]), f"{arm}: derived preflight reference does not bind Z numerical authority")
        require(normalized_preflight(preflight) == normalized_preflight(z_preflight), "preflight has an unexplained arm-specific change")
    rows: list[dict[str, Any]] = []
    for target in TARGET_SESSIONS:
        for endpoint, _report_key, _filename in ENDPOINTS:
            coordinate_rows = [arms[arm]["artifacts"][target][endpoint]["coordinates"] for arm in ARMS]
            require(len({arms[arm]["artifacts"][target][endpoint]["target_sha256"] for arm in ARMS}) == 1, f"{target}/{endpoint}: Z/B/D target-label drift")
            for name in COORDINATES:
                require(all(np.array_equal(coordinate_rows[0][name], row[name]) for row in coordinate_rows[1:]), f"{target}/{endpoint}: Z/B/D coordinate drift {name}")
        for arm in ARMS:
            rows.append({"dataset": "m1", "date": date_of(target), "arm": arm,
                         "selected_r2": arms[arm]["artifacts"][target]["selected"]["r2"],
                         "fixed_r2": arms[arm]["artifacts"][target]["epoch24"]["r2"],
                         "selected_epoch": source_receipts[arm]["selected"]["epoch"], "session_count": 1})
    require(len(rows) == 6, "exactly six M1 date/arm rows required")
    bound = all_bound_inputs(run_root, arms)
    output = {"schema": "m1_chronological_last2_paired_audit_v1", "status": "PASSED", "scope": "all three chronological M1 arms; two fixed target dates; source-only selection; no model forward in physical QueryOnly label/coordinate audit", "run_root": str(run_root), "bound_input_sha256": bound,
              "shared_source_evidence": source_receipts[Z_NONE]["actual_source_arrays"], "source_train_binding": {arm: {key: source_receipts[arm][key] for key in ("initialization_hash", "batch_order_sha256", "steps", "source_curve_sha256", "checkpoint_sha256")} for arm in ARMS},
              "rows": rows,
              "artifacts": {arm: {target: {endpoint: {key: value for key, value in arms[arm]["artifacts"][target][endpoint].items() if key != "coordinates"} for endpoint, _, _ in ENDPOINTS} for target in TARGET_SESSIONS} for arm in ARMS}}
    atomic_json(destination, output)
    print(json.dumps({"status": "PASSED", "dest": str(destination)}, sort_keys=True))


if __name__ == "__main__":
    main()
