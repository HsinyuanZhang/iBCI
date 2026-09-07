#!/usr/bin/env python3
"""CPU-only source feasibility audit for a source-frozen M1 EMG AFC4 carrier.

This is not a decoder experiment.  It accepts exactly the four held-in-calib
M1 sessions named in the existing source manifest; it rejects minival,
held-out, formal, EvalAI, test, and arbitrary discovery paths.  In each
source-LOSO fold it fits EMG mean/scale/PCA from the other three source
sessions, then uses only the left-out session's first ten trials to fit the
per-unit analytic carrier.  Later trials are not read as targets or scores;
their existence is reported solely to establish that a future-query window is
structurally available for a separately approved implementation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
from falcon_challenge.config import FalconTask
from falcon_challenge.dataloaders import load_nwb
from pynwb import NWBHDF5IO

ROOT = Path(__file__).resolve().parents[2]
SUA = ROOT / "sua_exploration"
sys.path.insert(0, str(SUA))
from mc_maze.m1_emg_afc4 import (  # noqa: E402
    RIDGE_ALPHA,
    SEMANTICS_VERSION,
    SUPPORT_TRIALS,
    affine_design_report,
    afc4_from_encoding,
    deterministic_split_half_reliability,
    fit_source_frozen_emg_basis,
    fit_unit_encoding,
)


MANIFEST = SUA / "manifests" / "m1_fixed_k_temporal_prototype_gate_a_v2_source_manifest.json"
DEFAULT_OUT = SUA / "results" / "m1_emg_afc4_feasibility_v1"
SCHEMA = "m1_emg_afc4_feasibility_v1"
EXPECTED_SESSIONS = (
    "ses-20120924", "ses-20120926", "ses-20120927", "ses-20120928",
)
FORBIDDEN_PATH_TOKENS = ("held-out", "minival", "evalai", "formal", "test")
Q_VALUES = (2, 3)

# Frozen pre-decoder feasibility gates.  These are deliberately modest: their
# purpose is to reject an unidentifiable/unstable low-rank task basis, not to
# manufacture a decoder-performance claim from a source proxy.
MIN_PER_SESSION_SOURCE_FROZEN_EXPLAINED_ENERGY = 0.25
MIN_MEAN_SOURCE_FROZEN_EXPLAINED_ENERGY = 0.30
MAX_M10_AFFINE_CONDITION_NUMBER = 1.0e4
MIN_POSITIVE_SPLIT_HALF_SESSIONS = 3
Q3_INCREMENTAL_ENERGY_TO_PREFER = 0.05


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strict_json(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return [strict_json(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): strict_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [strict_json(item) for item in value]
    raise TypeError(f"cannot encode {type(value)!r} as strict JSON")


def session_of(path: Path) -> str:
    return path.name.split("_ses-")[1].split("_behavior")[0]


def load_exact_source_manifest() -> dict[str, Any]:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "m1_fixed_k_temporal_prototype_gate_a_source_manifest_v2":
        raise ValueError("unexpected M1 source-manifest schema")
    sessions = payload.get("sessions")
    if not isinstance(sessions, list) or tuple(item.get("session") for item in sessions) != EXPECTED_SESSIONS:
        raise ValueError("manifest is not the frozen ordered four-session M1 source list")
    for entry in sessions:
        relative = str(entry.get("relative_path", ""))
        if not relative.startswith("SPINT-main/data/000941/sub-MonkeyL-held-in-calib/"):
            raise ValueError(f"M1 AFC4 source leaves held-in-calib: {relative}")
        lower = relative.lower()
        if any(token in lower for token in FORBIDDEN_PATH_TOKENS):
            raise ValueError(f"forbidden source path token in {relative}")
    return payload


def source_paths(manifest: dict[str, Any]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for entry in manifest["sessions"]:
        path = ROOT / str(entry["relative_path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.name.count("held-in-calib") != 1:
            raise ValueError(f"noncanonical held-in source path: {path}")
        result[str(entry["session"])] = path
    return result


def movement_window_arrays(path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    """Return all-session EMG trial means and *support-only* neural rates.

    EMG rows are permitted for source-basis characterization.  Neural spike
    counts are computed for trials [0,10) only; no future neural query value is
    used as a target, score, fit input, or selection input by this audit.
    """
    neural, emg, trial_change, eval_mask = load_nwb(path, FalconTask.m1)
    starts = np.flatnonzero(np.asarray(trial_change, dtype=bool))
    if len(starts) < SUPPORT_TRIALS + 1:
        raise ValueError(f"{path.name} lacks a future trial after M10")
    with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
        nwb = io.read()
        trials = nwb.trials.to_dataframe()
        units = nwb.units.to_dataframe()
        raw = nwb.acquisition["preprocessed_emg"]
        first = raw.get_timeseries(list(raw.time_series.keys())[0])
        timestamps = np.asarray(first.timestamps[:], dtype=np.float64)
        spike_times = [np.asarray(unit.spike_times, dtype=np.float64) for _, unit in units.iterrows()]
    if len(trials) != len(starts) or len(units) != 64:
        raise ValueError(f"{path.name} violates expected trial/64-unit M1 contract")
    emg_rows: list[np.ndarray] = []
    support_rates = np.zeros((SUPPORT_TRIALS, len(units)), dtype=np.float64)
    for trial_index in range(len(trials)):
        onset = float(trials.iloc[trial_index]["move_onset_time"])
        stop = float(trials.iloc[trial_index]["stop_time"])
        if not math.isfinite(onset):
            onset = float(trials.iloc[trial_index]["start_time"])
        left = int(np.searchsorted(timestamps, onset, side="left"))
        right = int(np.searchsorted(timestamps, stop, side="right"))
        segment = np.asarray(emg[left:right], dtype=np.float64)
        mask = np.asarray(eval_mask[left:right], dtype=bool)
        if mask.any():
            segment = segment[mask]
        if segment.size == 0:
            start = int(starts[trial_index])
            segment = np.asarray(emg[start : start + max(right - left, 1)], dtype=np.float64)
        if segment.ndim != 2 or segment.shape[0] == 0 or not np.isfinite(segment).all():
            raise ValueError(f"{path.name} has invalid movement-window EMG at trial {trial_index}")
        emg_rows.append(segment.mean(axis=0))
        if trial_index < SUPPORT_TRIALS:
            duration = max(stop - onset, 1.0e-12)
            for unit_index, times in enumerate(spike_times):
                support_rates[trial_index, unit_index] = int(np.sum((times >= onset) & (times < stop))) / duration
    return np.vstack(emg_rows), support_rates, int(len(trials))


def pca_receipt(basis) -> dict[str, Any]:
    return {
        "q": basis.q,
        "source_mean": basis.mean,
        "source_scale": basis.scale,
        "components_rows": basis.components,
        "component_order": "descending source standardized-EMG singular value",
        "component_sign": "largest-absolute loading is nonnegative; ties use lowest loading index",
        "sign_anchor_indices": basis.sign_anchor_indices,
        "singular_values": basis.singular_values,
        "source_energy_explained_by_component": basis.source_energy_explained,
    }


def q_gate(per_session: list[dict[str, Any]], q: int) -> dict[str, Any]:
    energy = [float(item["source_frozen_explained_energy"]) for item in per_session]
    ranks = [item["m10_design"]["full_rank"] for item in per_session]
    conditions = [float(item["m10_design"]["condition_number"]) for item in per_session]
    reliability = [item["split_half"]["weight_flattened_pearson"] for item in per_session]
    positive = sum(value is not None and float(value) > 0.0 for value in reliability)
    gate = {
        "q": q,
        "all_m10_designs_full_rank": bool(all(ranks)),
        "max_m10_condition_number": max(conditions),
        "condition_gate_pass": bool(max(conditions) <= MAX_M10_AFFINE_CONDITION_NUMBER),
        "mean_source_frozen_explained_energy": float(np.mean(energy)),
        "minimum_source_frozen_explained_energy": min(energy),
        "energy_gate_pass": bool(
            min(energy) >= MIN_PER_SESSION_SOURCE_FROZEN_EXPLAINED_ENERGY
            and float(np.mean(energy)) >= MIN_MEAN_SOURCE_FROZEN_EXPLAINED_ENERGY
        ),
        "positive_split_half_weight_sessions": int(positive),
        "reliability_gate_pass": bool(positive >= MIN_POSITIVE_SPLIT_HALF_SESSIONS),
    }
    gate["pass"] = bool(
        gate["all_m10_designs_full_rank"] and gate["condition_gate_pass"]
        and gate["energy_gate_pass"] and gate["reliability_gate_pass"]
    )
    return gate


def build_audit() -> dict[str, Any]:
    manifest = load_exact_source_manifest()
    paths = source_paths(manifest)
    data: dict[str, dict[str, Any]] = {}
    for session, path in paths.items():
        emg, support_rates, n_trials = movement_window_arrays(path)
        if emg.shape[0] != n_trials or support_rates.shape != (SUPPORT_TRIALS, 64):
            raise RuntimeError(f"shape contract failed for {session}")
        data[session] = {"emg": emg, "support_rates": support_rates, "n_trials": n_trials}

    per_q: dict[str, Any] = {}
    for q in Q_VALUES:
        sessions: list[dict[str, Any]] = []
        for target in EXPECTED_SESSIONS:
            source = tuple(name for name in EXPECTED_SESSIONS if name != target)
            basis = fit_source_frozen_emg_basis((data[name]["emg"] for name in source), q=q)
            target_scores_all = basis.transform(data[target]["emg"])
            support_scores = target_scores_all[:SUPPORT_TRIALS]
            weights, intercept = fit_unit_encoding(support_scores, data[target]["support_rates"], alpha=RIDGE_ALPHA)
            descriptor = afc4_from_encoding(weights, intercept)
            sessions.append({
                "target_source_session": target,
                "basis_source_sessions": list(source),
                "source_frozen_explained_energy": basis.reconstruction_energy_explained(data[target]["emg"]),
                "m10_design": affine_design_report(support_scores),
                "split_half": deterministic_split_half_reliability(support_scores, data[target]["support_rates"], alpha=RIDGE_ALPHA),
                "descriptor_shape": list(descriptor.shape),
                "descriptor_interface": "[w1,w2,||W||,b]" if q == 2 else "[w1,w2,w3,b]",
                "source_pca": pca_receipt(basis),
                "legal_future_query": {
                    "support_trial_range": [0, SUPPORT_TRIALS],
                    "future_query_trial_range": [SUPPORT_TRIALS, data[target]["n_trials"]],
                    "future_query_trial_count": int(data[target]["n_trials"] - SUPPORT_TRIALS),
                    "nonempty": bool(data[target]["n_trials"] > SUPPORT_TRIALS),
                    "used_by_this_cpu_audit": False,
                },
            })
        per_q[str(q)] = {"sessions": sessions, "gate": q_gate(sessions, q)}

    q2, q3 = per_q["2"]["gate"], per_q["3"]["gate"]
    incremental = float(q3["mean_source_frozen_explained_energy"] - q2["mean_source_frozen_explained_energy"])
    if q3["pass"] and (not q2["pass"] or incremental >= Q3_INCREMENTAL_ENERGY_TO_PREFER):
        disposition = {"status": "pass", "selected_interface": "q3", "reason": "q3 passes and has predeclared energy advantage or q2 failed"}
    elif q2["pass"]:
        disposition = {"status": "pass", "selected_interface": "q2", "reason": "q2 passes; q3 lacks the predeclared >=0.05 mean energy advantage"}
    else:
        disposition = {"status": "stop", "selected_interface": None, "reason": "neither four-coordinate EMG AFC4 interface passed the frozen source-only feasibility gate"}

    return {
        "schema_version": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "semantics_version": SEMANTICS_VERSION,
        "scope": {
            "cpu_only": True, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "formal_held_out": False, "evalai": False, "decoder_training": False,
            "source_split": "exact four M1 held-in-calib sessions only",
            "future_neural_query_target_or_score_used": False,
        },
        "input_manifest": {"path": str(MANIFEST.relative_to(ROOT)), "sha256": sha256(MANIFEST)},
        "input_sources": {
            session: {"path": str(path.relative_to(ROOT)), "sha256": sha256(path), "size_bytes": path.stat().st_size}
            for session, path in paths.items()
        },
        "frozen_estimator": {
            "support_trials": SUPPORT_TRIALS,
            "emg_estimator": "movement-window trial mean, source-frozen standardization and PCA",
            "neural_estimator": "raw-spike movement-window Hz in calibration trials [0,10) only",
            "ridge_alpha": RIDGE_ALPHA,
            "q2_interface": "[w1,w2,||W||,b]",
            "q3_interface": "[w1,w2,w3,b]",
            "normalization_for_later_decoder": "fit side-feature mean/std on source-session descriptors only; target descriptor uses frozen source statistics",
        },
        "frozen_gate": {
            "minimum_per_session_source_frozen_explained_energy": MIN_PER_SESSION_SOURCE_FROZEN_EXPLAINED_ENERGY,
            "minimum_mean_source_frozen_explained_energy": MIN_MEAN_SOURCE_FROZEN_EXPLAINED_ENERGY,
            "maximum_m10_affine_condition_number": MAX_M10_AFFINE_CONDITION_NUMBER,
            "minimum_positive_split_half_weight_sessions": MIN_POSITIVE_SPLIT_HALF_SESSIONS,
            "q3_incremental_energy_to_prefer": Q3_INCREMENTAL_ENERGY_TO_PREFER,
        },
        "q_comparison": per_q,
        "q3_minus_q2_mean_source_frozen_explained_energy": incremental,
        "disposition": disposition,
        "limitations": [
            "Explained energy and split-half coefficient reliability are source-only representation diagnostics, not decoder R2.",
            "The source-LOSO left-out session is still a held-in source session; this audit opens no formal or EvalAI endpoint.",
            "A structural post-M10 held-in query window is reported but not evaluated here; historical minival/overlap endpoints are not reused.",
            "A pass authorizes only a separately reviewed joint-source implementation blueprint, never GPU training or an online-performance claim.",
        ],
    }


def run(out: Path) -> Path:
    if out.exists():
        raise FileExistsError(f"refusing to overwrite output path: {out}")
    out.mkdir(parents=True, exist_ok=False)
    try:
        audit = strict_json(build_audit())
        written = out / "audit.json"
        written.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (out / "audit.sha256").write_text(f"{sha256(written)}  audit.json\n", encoding="utf-8")
        return written
    except Exception:
        for child in out.iterdir():
            child.unlink()
        out.rmdir()
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    print(run(args.out))


if __name__ == "__main__":
    main()
