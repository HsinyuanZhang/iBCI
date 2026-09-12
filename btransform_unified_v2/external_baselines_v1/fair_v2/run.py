#!/usr/bin/env python3
"""Pooled-source WF controls with source-only hyperparameter selection."""
from __future__ import annotations

import argparse
import gc
import hashlib
import itertools
import json
import os
import pickle
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_key, "2")

HERE = Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

import numpy as np
from fair_v2 import data
from fair_v2.metrics import array_sha, prediction_report, score_arrays
from fair_v2.numerics import (AlignedFAMap, CoralMap, FactorModel, Standardizer,
                             causal_lags, fit_ridge_grid, smooth_raw)


METHODS = ("wf_zs_h0", "diag_z_wf", "coral_wf", "aligned_fa_wf", "aligned_fa_stable_wf")
DEFAULT_CONFIG = {
    "schema": "fair_gf_source_selected_v2",
    "bin_ms": 20, "tau_ms": 240, "filter_extent": 1,
    "history_bins": 10, "source_train_fraction": .8,
    "alphas": [100000., 10000., 1000., 100.],
    "coral_shrinkages": [1., .5, .1, 0.], "coral_ridge": .001,
    "fa_dims": [10, 20, 40], "stable_fractions": [1., .75, .5],
    "fa_max_iter": 10000, "fa_retry_max_iter": 50000,
    "fa_tol": 1e-6, "fa_n_init": 3, "fa_noise_floor": 1e-6,
    "loading_threshold": .01, "seed": 42,
    "methods": list(METHODS),
    "selection_metric": "equal-source-recording mean sklearn variance-weighted R2",
    "tie_break": "first in fixed grid: greater alpha, greater shrinkage, smaller latent dimension, greater stable fraction",
    "target_labels_used_for_selection": False,
    "target_labels_used_for_fit": False,
}


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temp.replace(path)


def prepared_session(item: dict, context: int, config: dict) -> dict:
    raw = np.asarray(item["X"], np.float32)[int(item["pad"]):]
    endpoints = np.asarray(item["starts"], np.int64) + context - 1 - int(item["pad"])
    if np.any(endpoints < 0) or np.any(endpoints >= len(raw)) or np.any(np.diff(endpoints) <= 0):
        raise ValueError("query endpoints must be increasing native-bin coordinates")
    filtered = smooth_raw(raw, bin_ms=config["bin_ms"], tau_ms=config["tau_ms"], extent=config["filter_extent"])
    support_indices = np.asarray(item["support_indices"], np.int64)
    normalizer = Standardizer.fit(filtered[support_indices])
    z = normalizer.transform(filtered)
    return {
        "z": z, "support_z": z[support_indices], "normalizer": normalizer,
        "endpoints": endpoints, "Y": np.asarray(item["Y"], np.float32),
        "provenance": item["support_provenance"],
        "hashes": {name: array_sha(np.asarray(item[name])) for name in ("X", "Y", "starts", "support", "support_indices")},
    }


def source_split(row: dict, config: dict) -> tuple[np.ndarray, np.ndarray]:
    n = len(row["Y"])
    cutoff = int(np.floor(n * config["source_train_fraction"]))
    if cutoff < 2 or n - cutoff < 2:
        raise ValueError("source session is too short for chronological validation")
    # A 10-bin history over the 12-tap filter has a 21-bin raw receptive field.
    # Keep a common purge for all arms, including the h=0 diagnostic.
    taps = int(round(config["tau_ms"] * config["filter_extent"] / config["bin_ms"]))
    receptive_field = config["history_bins"] + taps - 1
    train_ids = np.arange(cutoff, dtype=np.int64)
    first_validation_endpoint = int(row["endpoints"][cutoff - 1]) + receptive_field
    validation_ids = np.flatnonzero((np.arange(n) >= cutoff) & (row["endpoints"] >= first_validation_endpoint))
    if len(validation_ids) < 2:
        raise ValueError("source validation is empty after the causal-history purge")
    return train_ids, validation_ids


def fit_fa(support_z: np.ndarray, dimension: int, config: dict) -> FactorModel:
    kwargs = dict(latent_dim=dimension, seed=config["seed"], tol=config["fa_tol"],
                  n_init=config["fa_n_init"], noise_floor=config["fa_noise_floor"])
    fitted = FactorModel.fit(support_z, max_iter=config["fa_max_iter"], **kwargs)
    if not fitted.diagnostics["converged"]:
        first_attempt = fitted.diagnostics
        fitted = FactorModel.fit(support_z, max_iter=config["fa_retry_max_iter"], **kwargs)
        fitted.diagnostics["initial_attempt"] = first_attempt
    if not fitted.diagnostics["converged"]:
        raise RuntimeError(f"FA dimension {dimension} failed to converge at the sealed retry budget")
    return fitted


def configurations(method: str, config: dict):
    if method in ("wf_zs_h0", "diag_z_wf"):
        yield {"history_bins": 1 if method == "wf_zs_h0" else config["history_bins"]}
    elif method == "coral_wf":
        for shrinkage in config["coral_shrinkages"]:
            yield {"history_bins": config["history_bins"], "shrinkage": shrinkage, "coral_ridge": config["coral_ridge"]}
    elif method in ("aligned_fa_wf", "aligned_fa_stable_wf"):
        for dimension, fraction in itertools.product(config["fa_dims"], config["stable_fractions"]):
            yield {"history_bins": config["history_bins"], "latent_dim": dimension,
                   "stable_fraction": fraction, "posterior": "stable" if method == "aligned_fa_stable_wf" else "all"}
    else:
        raise ValueError(method)


def source_maps(method: str, setting: dict, source: dict, reference: str, factors: dict, config: dict) -> dict:
    maps = {}
    for session, row in source.items():
        if method in ("wf_zs_h0", "diag_z_wf"):
            maps[session] = None
        elif method == "coral_wf":
            maps[session] = (None if session == reference else
                             CoralMap.fit(source[reference]["support_z"], row["support_z"],
                                          ridge=setting["coral_ridge"], shrinkage=setting["shrinkage"]))
        else:
            dim = setting["latent_dim"]
            src_fa, session_fa = factors[dim][reference], factors[dim][session]
            maps[session] = (src_fa if session == reference else
                             AlignedFAMap.fit(src_fa, session_fa, stable_fraction=setting["stable_fraction"],
                                              posterior=setting["posterior"], loading_threshold=config["loading_threshold"]))
    return maps


def feature_rows(source: dict, maps: dict, history_bins: int) -> dict:
    result = {}
    for session, row in source.items():
        z = row["z"] if maps[session] is None else maps[session].transform(row["z"])
        result[session] = causal_lags(z, row["endpoints"], history_bins=history_bins)
    return result


def source_fit(train: dict, metadata: dict, config: dict, *, emit=None) -> tuple[dict, dict, dict]:
    """This entry point has no evaluation argument and never receives target Y."""
    emit = emit or (lambda event, **values: None)
    source = {session: prepared_session(item, int(metadata["context"]), config) for session, item in train.items()}
    reference = max(source)
    splits = {session: source_split(row, config) for session, row in source.items()}
    factors, factor_diagnostics = {}, {}
    selected, grids, banks = {}, {}, {}
    for method in config["methods"]:
        rows, best, best_maps = [], None, None
        for setting in configurations(method, config):
            if "latent_dim" in setting and setting["latent_dim"] not in factors:
                dim = setting["latent_dim"]
                factors[dim] = {}
                for session, row in source.items():
                    emit("source_fa", method=method, dimension=dim, session=session)
                    factors[dim][session] = fit_fa(row["support_z"], dim, config)
                factor_diagnostics[str(dim)] = {s: f.diagnostics for s, f in factors[dim].items()}
            try:
                maps = source_maps(method, setting, source, reference, factors, config)
            except ValueError as exc:
                rows.append({"configuration": setting, "eligible": False, "reason": str(exc)})
                continue
            xs = feature_rows(source, maps, setting["history_bins"])
            tx = np.concatenate([xs[s][splits[s][0]] for s in source])
            ty = np.concatenate([source[s]["Y"][splits[s][0]] for s in source])
            readouts = fit_ridge_grid(tx, ty, config["alphas"])
            del tx, ty
            for alpha, readout in readouts.items():
                metrics = {s: score_arrays(source[s]["Y"][splits[s][1]], readout.predict(xs[s][splits[s][1]])) for s in source}
                score = float(np.mean([m["standard_variance_weighted_r2"] for m in metrics.values()]))
                candidate = {"configuration": setting, "alpha": alpha, "eligible": True,
                             "source_validation_standard_mean": score, "source_validation_per_session": metrics}
                rows.append(candidate)
                if best is None or score > best["source_validation_standard_mean"] + 1e-12:
                    best, best_maps = candidate, maps
            emit("candidate_complete", method=method, configuration=setting,
                 best_source_validation=best["source_validation_standard_mean"])
            del xs, readouts
            gc.collect()
        if best is None:
            raise RuntimeError("no eligible source-selected configuration for " + method)
        setting = best["configuration"]
        xs = feature_rows(source, best_maps, setting["history_bins"])
        full_x = np.concatenate(list(xs.values()))
        full_y = np.concatenate([source[s]["Y"] for s in source])
        readout = fit_ridge_grid(full_x, full_y, [best["alpha"]])[best["alpha"]]
        selected[method] = best
        banks[method] = {"readout": readout, "source_maps": best_maps, "configuration": setting}
        grids[method] = rows
        emit("source_fit_complete", method=method, selected=setting, alpha=best["alpha"])
        del xs, full_x, full_y
        gc.collect()
    selection = {
        "schema": "fair_gf_source_selection_v2", "config": config,
        "reference_source_session": reference, "source_session_count": len(source),
        "source_windows": sum(len(v["Y"]) for v in source.values()),
        "source_splits": {s: {"fit_rows": len(ids[0]), "validation_rows": len(ids[1]),
                               "fit_indices_sha256": array_sha(ids[0]), "validation_indices_sha256": array_sha(ids[1]),
                               "fit_last_raw_endpoint": int(source[s]["endpoints"][ids[0][-1]]),
                               "validation_first_raw_endpoint": int(source[s]["endpoints"][ids[1][0]])} for s, ids in splits.items()},
        "source_input_hashes": {s: row["hashes"] for s, row in source.items()},
        "source_support_provenance": {s: row["provenance"] for s, row in source.items()},
        "selected": selected, "candidate_grids": grids, "source_fa_diagnostics": factor_diagnostics,
        "target_labels_used_for_fit": False, "target_labels_used_for_selection": False,
    }
    fitted = {"methods": banks, "source_normalizers": {s: row["normalizer"] for s, row in source.items()},
              "reference_source_session": reference, "reference_support_z": source[reference]["support_z"],
              "source_factors": factors, "config": config}
    return fitted, selection, source


def target_fit_and_predict(fitted: dict, evaluation: dict, metadata: dict, *, emit=None) -> tuple[dict, dict, dict]:
    emit = emit or (lambda event, **values: None)
    config = fitted["config"]
    reference = fitted["reference_source_session"]
    predictions = {method: {} for method in fitted["methods"]}
    adapters, diagnostics = {}, {}
    for session, item in evaluation.items():
        # Y is copied for later metrics by the common preparation record.  It is
        # never an argument to a normalizer, aligner, FA, or readout fit here.
        row = prepared_session(item, int(metadata["context"]), config)
        factors = {}
        adapters[session] = {"normalizer": row["normalizer"], "methods": {}}
        diagnostics[session] = {"input_hashes": row["hashes"], "support_provenance": row["provenance"], "methods": {}}
        for method, bank in fitted["methods"].items():
            setting = bank["configuration"]
            if method in ("wf_zs_h0", "diag_z_wf"):
                transform = None
            elif method == "coral_wf":
                transform = CoralMap.fit(fitted["reference_support_z"], row["support_z"],
                                          ridge=setting["coral_ridge"], shrinkage=setting["shrinkage"])
            else:
                dimension = setting["latent_dim"]
                if dimension not in factors:
                    emit("target_fa", session=session, dimension=dimension)
                    factors[dimension] = fit_fa(row["support_z"], dimension, config)
                transform = AlignedFAMap.fit(fitted["source_factors"][dimension][reference], factors[dimension],
                                             stable_fraction=setting["stable_fraction"], posterior=setting["posterior"],
                                             loading_threshold=config["loading_threshold"])
            z = row["z"] if transform is None else transform.transform(row["z"])
            features = causal_lags(z, row["endpoints"], history_bins=setting["history_bins"])
            predictions[method][session] = np.asarray(bank["readout"].predict(features), np.float32)
            adapters[session]["methods"][method] = transform
            diagnostics[session]["methods"][method] = {} if transform is None else transform.diagnostics
        diagnostics[session]["fa_fit"] = {str(k): fa.diagnostics for k, fa in factors.items()}
        emit("target_session_complete", session=session)
    return predictions, adapters, diagnostics


def run(task: str, destination: Path, config: dict | None = None) -> dict:
    config = dict(DEFAULT_CONFIG if config is None else config)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("fresh result directory required: " + str(destination))
    destination.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    code_hashes = {name: sha_file(HERE / name) for name in ("run.py", "data.py", "numerics.py", "metrics.py")}
    atomic_json(destination / "protocol.json", {"task": task, "config": config, "code_sha256": code_hashes,
                                               "target_evaluation_loaded": False})
    def emit(event, **values):
        row = {"task": task, "event": event, "elapsed_seconds": time.monotonic() - start, **values}
        atomic_json(destination / "progress.json", row)
        print(json.dumps(row, allow_nan=False), flush=True)
    emit("load_source")
    loaded = data.load_task(task, include_evaluation=False)
    fitted, selection, source = source_fit(loaded["train"], loaded["metadata"], config, emit=emit)
    atomic_json(destination / "selection.json", selection)
    selection_sha = sha_file(destination / "selection.json")
    # The selection record is sealed on disk before opening the evaluation data.
    emit("source_selection_sealed", selection_sha256=selection_sha)
    del source, loaded
    gc.collect()
    evaluation_data = data.load_task(task, include_evaluation=True)
    predictions, adapters, diagnostics = target_fit_and_predict(fitted, evaluation_data["evaluation"], evaluation_data["metadata"], emit=emit)
    reports = {method: prediction_report(task, evaluation_data["evaluation"], by_session) for method, by_session in predictions.items()}
    if sha_file(destination / "selection.json") != selection_sha:
        raise RuntimeError("source selection changed after evaluation was opened")
    for method, by_session in predictions.items():
        for session, prediction in by_session.items():
            np.save(destination / f"pred_{method}_{session}.npy", prediction)
    # Reference support is no longer needed after all numeric maps have been fit.
    fitted.pop("reference_support_z")
    fitted["target_adapters"] = adapters
    with (destination / "models.pkl").open("wb") as handle:
        pickle.dump(fitted, handle, protocol=pickle.HIGHEST_PROTOCOL)
    receipt = {
        "schema": "fair_gf_baselines_receipt_v2", "status": "COMPLETED", "task": task,
        "config": config, "code_sha256": code_hashes, "selection_sha256": selection_sha,
        "models_sha256": sha_file(destination / "models.pkl"), "reports": reports,
        "source_session_count": selection["source_session_count"], "source_windows": selection["source_windows"],
        "reference_source_session": fitted["reference_source_session"], "selected": selection["selected"],
        "target_calibration": diagnostics, "target_labels_used_for_fit": False,
        "target_labels_used_for_selection": False, "target_backpropagation": False,
        "official_test_used": False, "scope": "public calibration local development",
        "support_representation": "native raw 20ms bins; common causal filtering on full recording before support selection",
        "normalization": "per-session statistics fitted only on that session's filtered calibration support",
        "fa_variants": {"aligned_fa_wf": "author-form all-electrode posterior with source-selected dimension/pruning",
                        "aligned_fa_stable_wf": "user-requested stable-only posterior variant"},
        "h1_channel_correspondence": "positional rows; physical cross-date correspondence unverified" if task == "h1" else "fixed delivered unit rows",
        "runtime_seconds": time.monotonic() - start,
    }
    atomic_json(destination / "receipt.json", receipt)
    emit("complete", scores={m: r["standard_equal_session_mean"] for m, r in reports.items()})
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("m2", "m1", "h1"), required=True)
    parser.add_argument("--dest", type=Path, required=True)
    args = parser.parse_args()
    run(args.task, args.dest)


if __name__ == "__main__":
    main()
