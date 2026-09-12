#!/usr/bin/env python3
"""Independent read-only replay audit for completed FAIR V2 research runs.

This program never calls a fit method.  It reloads saved pickle maps/readouts,
filters every target raw stream causally from a zero state, and compares the
result with the immutable .npy predictions and stored metric reports.
"""
from __future__ import annotations
import argparse, hashlib, json, os, pickle, sys, warnings
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
warnings.filterwarnings("ignore", message="CUDA initialization:.*", category=UserWarning)
HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))
import numpy as np
from fair_v2 import data
from fair_v2.metrics import array_sha, prediction_report

EXPECTED = {"m2": (7, 6), "m1": (4, 3), "h1": (13, 14)}
CODE = ("run.py", "data.py", "numerics.py", "metrics.py")


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _kernel(config: dict) -> np.ndarray:
    taps = int(round(config["tau_ms"] * config["filter_extent"] / config["bin_ms"]))
    k = np.exp(-np.arange(taps, dtype=np.float64) * config["bin_ms"] / config["tau_ms"])
    return k / k.sum()


def stream_filter(raw: np.ndarray, config: dict) -> np.ndarray:
    """Standalone zero-state 12-tap causal filter; deliberately no numerics call."""
    raw = np.asarray(raw, np.float64)
    k = _kernel(config)
    state = np.zeros((len(k), raw.shape[1]), dtype=np.float64)
    out = np.empty_like(raw)
    for t, row in enumerate(raw):
        state[1:] = state[:-1]
        state[0] = row
        out[t] = k @ state
    return out


def stream_features(filtered: np.ndarray, endpoints: np.ndarray, normalizer, adapter,
                    history_bins: int) -> np.ndarray:
    """Run saved affine/nonlinear adapter and newest-first lag state one raw bin at a time."""
    _assert(history_bins >= 1, "history_bins must count at least current bin")
    endpoints = np.asarray(endpoints, np.int64)
    _assert(np.all(np.diff(endpoints) > 0) and endpoints[0] >= 0 and endpoints[-1] < len(filtered),
            "invalid target endpoint chronology")
    # Determine transformed width without fitting or touching any target labels.
    example = (filtered[:1] - normalizer.mean_) / normalizer.scale_
    z0 = example if adapter is None else adapter.transform(example)
    state = np.zeros((history_bins, z0.shape[1]), dtype=np.float64)
    features = np.empty((len(endpoints), history_bins * z0.shape[1]), dtype=np.float64)
    output_row = 0
    for t, row in enumerate(filtered):
        z = (row - normalizer.mean_) / normalizer.scale_
        z = z[None, :] if adapter is None else adapter.transform(z[None, :])
        state[1:] = state[:-1]
        state[0] = z[0]
        if output_row < len(endpoints) and t == endpoints[output_row]:
            features[output_row] = state.reshape(-1)
            output_row += 1
    _assert(output_row == len(endpoints), "stream ended before all query endpoints")
    return features


def hashes(item: dict) -> dict:
    return {name: array_sha(np.asarray(item[name])) for name in ("X", "Y", "starts", "support", "support_indices")}


def verify_item(item: dict, expected: dict, filtered: np.ndarray, normalizer, label: str) -> dict:
    actual = hashes(item)
    _assert(actual == expected, f"{label}: raw/support input hash drift")
    raw = np.asarray(item["X"], np.float32)[int(item["pad"]):]
    support_ix = np.asarray(item["support_indices"], np.int64)
    _assert(np.array_equal(raw[support_ix], item["support"]), f"{label}: support no longer indexes raw X")
    # Recompute only scalar normalizer evidence from raw support; no model is fitted.
    support = filtered[support_ix]
    expected_mean = support.mean(0)
    expected_scale = support.std(0); expected_scale[expected_scale == 0] = 1.
    mean_delta = float(np.max(np.abs(normalizer.mean_ - expected_mean)))
    scale_delta = float(np.max(np.abs(normalizer.scale_ - expected_scale)))
    # The independent streaming dot-product has a different IEEE-754 summation
    # order than np.convolve; permit only machine-precision evidence drift.
    _assert(mean_delta <= 1e-12, f"{label}: saved normalizer mean drift={mean_delta}")
    _assert(scale_delta <= 1e-12, f"{label}: saved normalizer scale drift={scale_delta}")
    return {"raw_bins": int(len(raw)), "support_bins": int(len(support_ix)), "support_budget": item["support_provenance"]["budget"],
            "normalizer_mean_max_abs": mean_delta, "normalizer_scale_max_abs": scale_delta}


def assert_selection(selection: dict) -> dict:
    checks = {}
    for method, selected in selection["selected"].items():
        best = None
        for candidate in selection["candidate_grids"][method]:
            if not candidate["eligible"]:
                continue
            if best is None or candidate["source_validation_standard_mean"] > best["source_validation_standard_mean"] + 1e-12:
                best = candidate
        _assert(best is not None and best == selected, f"{method}: selected source-CV candidate is not fixed-grid optimum")
        checks[method] = {"alpha": selected["alpha"], "score": selected["source_validation_standard_mean"],
                          "configuration": selected["configuration"]}
    return checks


def assert_fa_convergence(selection: dict, fitted: dict) -> dict:
    source = selection["source_fa_diagnostics"]
    for dim, sessions in source.items():
        for session, diag in sessions.items():
            _assert(diag["converged"], f"source FA dim={dim} session={session} did not converge")
    target_count = 0
    for session, adapter_row in fitted["target_adapters"].items():
        for method, adapter in adapter_row["methods"].items():
            if adapter is not None and hasattr(adapter, "target_fa"):
                _assert(adapter.target_fa.diagnostics["converged"], f"target FA {session}/{method} did not converge")
                target_count += 1
    return {"source_dimensions": sorted(source), "source_models": sum(len(x) for x in source.values()),
            "selected_target_adapter_references": target_count}


def verify_task(task: str, results_root: Path) -> dict:
    result = results_root / f"{task}_v2"
    receipt = json.loads((result / "receipt.json").read_text())
    protocol = json.loads((result / "protocol.json").read_text())
    selection = json.loads((result / "selection.json").read_text())
    _assert(receipt["status"] == "COMPLETED" and protocol["target_evaluation_loaded"] is False,
            f"{task}: sealed-run status/protocol drift")
    _assert(sha_file(result / "selection.json") == receipt["selection_sha256"], f"{task}: selection hash drift")
    _assert(sha_file(result / "models.pkl") == receipt["models_sha256"], f"{task}: pickle hash drift")
    actual_code = {name: sha_file(HERE / name) for name in CODE}
    _assert(actual_code == receipt["code_sha256"] == protocol["code_sha256"], f"{task}: code hash binding drift")
    _assert(protocol["config"] == receipt["config"] == selection["config"], f"{task}: protocol configuration drift")
    with (result / "models.pkl").open("rb") as handle:
        fitted = pickle.load(handle)
    _assert(fitted["config"] == receipt["config"], f"{task}: pickle configuration drift")
    selected = assert_selection(selection)
    fa = assert_fa_convergence(selection, fitted)
    # A shrinkage of 1 removes the empirical off-diagonal covariance structure.
    coral = selected["coral_wf"]["configuration"]
    _assert(coral["shrinkage"] == 1., f"{task}: CORAL shrinkage=1 was not source-CV selected")

    loaded = data.load_task(task, include_evaluation=True)
    source, evaluation = loaded["train"], loaded["evaluation"]
    exp_source, exp_target = EXPECTED[task]
    _assert(len(source) == exp_source and len(evaluation) == exp_target, f"{task}: session roster count drift")
    _assert(sum(len(v["Y"]) for v in source.values()) == selection["source_windows"], f"{task}: source window count drift")
    _assert(sum(len(v["Y"]) for v in evaluation.values()) == receipt["reports"]["wf_zs_h0"]["n_windows"], f"{task}: target window count drift")

    source_evidence = {}
    for session, item in source.items():
        raw = np.asarray(item["X"], np.float32)[int(item["pad"]):]
        source_evidence[session] = verify_item(item, selection["source_input_hashes"][session],
                                                stream_filter(raw, receipt["config"]),
                                                fitted["source_normalizers"][session], f"{task}/source/{session}")
    predictions = {method: {} for method in fitted["methods"]}
    target_evidence, replay = {}, {}
    context = int(loaded["metadata"]["context"])
    for session, item in evaluation.items():
        raw = np.asarray(item["X"], np.float32)[int(item["pad"]):]
        filtered = stream_filter(raw, receipt["config"])
        adapter_row = fitted["target_adapters"][session]
        target_evidence[session] = verify_item(item, receipt["target_calibration"][session]["input_hashes"],
                                                filtered, adapter_row["normalizer"], f"{task}/target/{session}")
        endpoints = np.asarray(item["starts"], np.int64) + context - 1 - int(item["pad"])
        for method, bank in fitted["methods"].items():
            features = stream_features(filtered, endpoints, adapter_row["normalizer"],
                                       adapter_row["methods"][method], bank["configuration"]["history_bins"])
            got = np.asarray(bank["readout"].predict(features), np.float32)
            saved = np.load(result / f"pred_{method}_{session}.npy")
            delta = float(np.max(np.abs(got - saved)))
            _assert(delta <= 2e-6, f"{task}/{method}/{session}: replay maxabs={delta} exceeds 2e-6")
            predictions[method][session] = got
            replay.setdefault(method, {})[session] = {"max_abs": delta, "saved_sha256": array_sha(saved),
                                                       "replay_sha256": array_sha(got)}
    reports = {method: prediction_report(task, evaluation, by_session) for method, by_session in predictions.items()}
    for method, report in reports.items():
        _assert(report == receipt["reports"][method], f"{task}/{method}: independently recomputed report differs")
    return {"status": "PASS", "source_sessions": len(source), "target_sessions": len(evaluation),
            "source_normalizer_evidence": source_evidence, "target_support_evidence": target_evidence,
            "source_cv_selection": selected, "fa_convergence": fa,
            "coral": {"selected_shrinkage": 1., "interpretation": "source-CV chose fully isotropic covariance shrinkage; no off-diagonal CORAL alignment is retained"},
            "replay": replay, "reports": reports, "code_sha256": actual_code}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-root", type=Path, default=HERE / "results")
    p.add_argument("--receipt", type=Path, default=HERE / "results" / "verification_v2" / "receipt.json")
    p.add_argument("--tasks", nargs="+", choices=("m2", "m1", "h1"), default=("m2", "m1", "h1"))
    args = p.parse_args()
    tasks = {task: verify_task(task, args.results_root) for task in args.tasks}
    value = {"schema": "fair_v2_independent_replay_verification_v1", "status": "PASS", "tasks": tasks}
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"status": value["status"], "receipt": str(args.receipt),
                      "tasks": {k: {"source_sessions": v["source_sessions"], "target_sessions": v["target_sessions"]} for k,v in tasks.items()}}, indent=2))

if __name__ == "__main__":
    main()
