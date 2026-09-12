"""Source/dev-only fitted CPU baseline selection for the DANDI-688 v2 matrix."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np

from .baselines import (fit_baseline, fit_wf_fss, predict_baseline, predict_wf_fss,
                        prepare_baseline_features, refit_baseline_alpha, refit_wf_fss_alpha)
from .common import (aggregate_scores, atomic_json, digest, fresh_directory, load_records,
                     record_binding, require_full_source, score_predictions, sha256, source_hashes)
from . import protocol

PMUA_METHODS = ("wf_zs_h0", "diag_z_wf", "coral_wf", "aligned_fa_wf", "aligned_fa_stable_wf")
ALPHAS = (1e2, 1e3, 1e4, 1e5)


def _grid(method: str, smoke: bool):
    if smoke:
        base = {"alpha": 1e3, "history_bins": 10, "smooth": True, "fa_dim": 2,
                "stable_fraction": 1., "fa_n_init": 1, "fa_tol": 1e-5,
                "fa_max_iter": 1000, "fa_retry_max_iter": 10000}
        return [base]
    if method == "wf_fss":
        return [{"alpha": a, "history_bins": h, "smooth": s} for a in ALPHAS for h in (1, 10, 50) for s in (False, True)]
    if method == "wf_zs_h0": return [{"alpha": a, "history_bins": 1, "smooth": True} for a in ALPHAS]
    if method == "diag_z_wf": return [{"alpha": a, "history_bins": 10, "smooth": True} for a in ALPHAS]
    if method == "coral_wf": return [{"alpha": a, "history_bins": 10, "smooth": True, "shrinkage": sh} for a in ALPHAS for sh in (0., .1, .5, 1.)]
    if method in {"aligned_fa_wf", "aligned_fa_stable_wf"}:
        return [{"alpha": a, "history_bins": 10, "smooth": True, "fa_dim": k, "stable_fraction": f}
                for k in (5, 10, 20) for f in (.5, .75, 1.) for a in ALPHAS]
    raise ValueError(method)


def _evaluate(method: str, config: dict, source: list[Any], dev: list[Any]):
    if method == "wf_fss":
        models = {r.session_id: fit_wf_fss(r, config) for r in dev}
        predictions = {r.session_id: predict_wf_fss(models[r.session_id], r) for r in dev}
        return models, predictions
    model = fit_baseline(method, source, config)
    predictions = {r.session_id: predict_baseline(model, r) for r in dev}
    return model, predictions


def run_baselines(cache: Path, dest: Path, *, methods=None, smoke: bool = False,
                  source_records=None, dev_records=None) -> dict:
    """Select CPU baselines on dev only; this function cannot load final data."""
    destination = fresh_directory(dest)
    selected_methods = tuple(methods or (*PMUA_METHODS, "wf_fss_sua", "wf_fss_pmua"))
    if any(m not in {*PMUA_METHODS, "wf_fss_sua", "wf_fss_pmua"} for m in selected_methods): raise ValueError("unknown baseline method")
    if smoke and (source_records is None or dev_records is None):
        raise ValueError("smoke requires explicit source_records and dev_records")
    if source_records is None:
        source_pmua = load_records(cache, "pmua", "train")
        source_sua = load_records(cache, "sua", "train")
    else:
        source_pmua = list(source_records["pmua"]); source_sua = list(source_records["sua"])
    if dev_records is None:
        dev_pmua = load_records(cache, "pmua", "dev")
        dev_sua = load_records(cache, "sua", "dev")
    else:
        dev_pmua = list(dev_records["pmua"]); dev_sua = list(dev_records["sua"])
    if not smoke:
        if source_records is not None or dev_records is not None: raise ValueError("formal runner does not accept injected records")
        require_full_source(source_pmua); require_full_source(source_sua)
        if {r.session_id for r in dev_pmua} != set(protocol.DEV_SESSIONS) or {r.session_id for r in dev_sua} != set(protocol.DEV_SESSIONS) or any(r.split != "dev" for r in (*dev_pmua,*dev_sua)): raise ValueError("formal selection requires exact dev roster")
    rows, winners, artifacts = {}, {}, {}
    for method in selected_methods:
        source, dev = (source_sua, dev_sua) if method == "wf_fss_sua" else ((source_pmua, dev_pmua))
        internal = "wf_fss" if method.startswith("wf_fss_") else method
        candidates = []
        alignment_cache = {}
        for config in _grid(internal, smoke):
            try:
                if internal == "wf_fss":
                    key = tuple(sorted((name, value) for name, value in config.items() if name != "alpha"))
                    if key not in alignment_cache:
                        alignment_cache[key] = {r.session_id: fit_wf_fss(r, config) for r in dev}
                    model = {name: refit_wf_fss_alpha(base, config["alpha"]) for name, base in alignment_cache[key].items()}
                    predictions = {r.session_id: predict_wf_fss(model[r.session_id], r) for r in dev}
                else:
                    key = tuple(sorted((name, value) for name, value in config.items() if name != "alpha"))
                    if key not in alignment_cache:
                        reference = protocol.TRAIN_SESSIONS[-1] if not smoke else max(r.session_id for r in source)
                        base = fit_baseline(internal, source, config, reference_session=reference)
                        alignment_cache[key] = (base, {r.session_id: prepare_baseline_features(base, r, return_diagnostics=True) for r in dev})
                    base, prepared = alignment_cache[key]
                    model = refit_baseline_alpha(base, config["alpha"])
                    predictions = {r.session_id: np.asarray(model.readout.predict(prepared[r.session_id][0]), np.float32) for r in dev}
                scores = [score_predictions(record, predictions[record.session_id]) for record in dev]
                aggregate = aggregate_scores(scores)
                candidate = {"config": config, "eligible": True, "score": aggregate["mean_r2"], "scores": aggregate,
                             "target_alignment": {} if internal == "wf_fss" else {k:v[1] for k,v in prepared.items()}}
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
                candidate = {"config": config, "eligible": False, "reason": str(error)}
            candidates.append(candidate)
            atomic_json(destination / "progress.json", {"method": method, "candidate_count": len(candidates), "last": candidate, "final_loaded": False})
            if candidate["eligible"] and (method not in winners or candidate["score"] > winners[method]["candidate"]["score"]):
                winners[method] = {"candidate": candidate, "model": model, "predictions": predictions}
        if method not in winners:
            rows[method] = candidates
            artifacts[method] = {"status": "UNAVAILABLE" if internal.startswith("aligned_fa") else "FAILED",
                                 "reason": "no eligible candidate", "candidate_count": len(candidates),
                                 "reasons": [row.get("reason") for row in candidates if not row.get("eligible")]}
            continue
        rows[method] = candidates
        winner = winners[method]
        with (destination / f"{method}.pkl").open("wb") as handle: pickle.dump(winner["model"] if internal != "wf_fss" else {"selected_config": winner["candidate"]["config"], "label_information": "dense_velocity_on_allowed_M33_window"}, handle)
        np.savez_compressed(destination / f"{method}.dev_predictions.npz", **winner["predictions"])
        artifacts[method] = {"model": f"{method}.pkl", "model_sha256": sha256(destination / f"{method}.pkl"),
                             "predictions": f"{method}.dev_predictions.npz", "predictions_sha256": sha256(destination / f"{method}.dev_predictions.npz"),
                             "selected": winner["candidate"],
                             "information_contract": (
                                 "source: pooled 18-session Q50 physical velocity labels; target: M33 neural support only; no target velocity/direction labels"
                                 if internal != "wf_fss" else
                                 "target: same M33/MOVE time-window neural rows plus dense physical velocity labels; Full's MOVE-T4 uses per-trial direction angles, so label information is not equivalent"
                             )}
    failed_non_fa = [method for method in selected_methods if method not in winners and not method.startswith("aligned_fa")]
    status = "FAILED" if failed_non_fa else ("SMOKE" if smoke else "DEV_SELECTED")
    selection = {"schema": "dandi688_v2_cpu_baseline_selection", "status": status,
                 "final_loaded": False, "methods": list(selected_methods), "source": record_binding(source_pmua),
                 "dev": record_binding(dev_pmua), "candidate_grid": rows, "artifacts": artifacts, "code": source_hashes()}
    selection["sha256"] = digest(selection)
    atomic_json(destination / "selection.json", selection)
    atomic_json(destination / "receipt.json", {"status": selection["status"], "selection_sha256": sha256(destination / "selection.json"), "final_loaded": False, "failed_non_fa": failed_non_fa})
    if failed_non_fa:
        raise RuntimeError(f"non-FA baseline cells have no eligible candidate: {failed_non_fa}")
    return selection
