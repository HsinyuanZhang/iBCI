"""Materialize strict source and target H1 date-LODO arrays separately.

`--surface source` is the only mode allowed to open source records and fit H-C.
It persists all plan arrays.  `--surface target` reads that frozen authority and
opens target records only; it never refits or reads source NWBs.
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SPINT = ROOT.parent / "SPINT-main"
for path in (ROOT / "src", SPINT, ROOT.parent / "btransform_unified_v1/src"):
    sys.path.insert(0, str(path))
from src.data.h1_m4_eb_pilot import (FrozenEBPlan, _fit_without_eb, fit_deployment_carrier,
                                     index_heldin_calib, interpolate_trial_identity, load_record)
from btransform_unified_v1.h1_config import H1_SESSION_DATES, H1_SESSIONS_BY_DATE

DATA = SPINT / "data/000954"
Q, RIDGE = 12, 10.0


def sha(array: np.ndarray) -> str:
    array = np.ascontiguousarray(array)
    return hashlib.sha256(array.dtype.str.encode() + str(array.shape).encode() + array.tobytes()).hexdigest()


def endpoint_windows(record, trials, *, stride: int) -> tuple[np.ndarray, ...]:
    """Reset context at each native trial segment; support cannot enter query."""
    if stride < 1 or not trials: raise ValueError("nonempty trials and stride >= 1 required")
    trials = tuple(map(float, trials)); trial_indices = {value: index for index, value in enumerate(record.trial_values)}
    eligible = np.flatnonzero(record.eval_mask & np.isfinite(record.trial_num) & np.isin(record.trial_num, trials)).astype(np.int64)[::stride]
    X = np.zeros((len(eligible), 300, 176), dtype=np.float32); valid = np.zeros((len(eligible), 300), dtype=np.bool_)
    starts = np.empty(len(eligible), dtype=np.int64); segment_starts = np.empty(len(eligible), dtype=np.int64); native_index = np.empty(len(eligible), dtype=np.int64)
    for row, endpoint in enumerate(eligible):
        value = float(record.trial_num[endpoint]); segment = int(np.flatnonzero(record.trial_num == value)[0])
        start = max(segment, int(endpoint) - 299); width = int(endpoint) - start + 1
        X[row, 300-width:] = record.neural[start:int(endpoint)+1]; valid[row, 300-width:] = True
        starts[row], segment_starts[row], native_index[row] = start, segment, trial_indices[value]
    return X, valid, np.asarray(record.velocity[eligible], np.float32), eligible, starts, segment_starts, native_index


def fresh_plan(date: str, records: dict) -> tuple[FrozenEBPlan, float, dict]:
    names = tuple(sorted(records))
    rates = np.concatenate([np.concatenate([records[name].blocks_for(v).rates for v in records[name].trial_values[:3]]) for name in names])
    mean = rates.mean(0); scale = np.maximum(rates.std(0), 1e-6)
    _, _, right = np.linalg.svd((rates - mean) / scale, full_matrices=False)
    pcs = np.asarray(right[:Q], np.float64)
    inputs = tuple(records[name].input_sha256 for name in names)
    provisional = FrozenEBPlan(date, names, inputs, mean, scale, pcs, Q, RIDGE, np.empty((7, 4)), np.zeros(4), 1., "fresh", "fresh", "fresh", "")
    rows = np.concatenate([_fit_without_eb(records[name], provisional, records[name].trial_values[:3])["raw_rows"] for name in names])
    _, _, right = np.linalg.svd(rows, full_matrices=False)
    U = np.asarray(right[:4].T, np.float64); projected = rows @ U; mu = projected.mean(0)
    tau2 = float(np.square(projected - mu).sum() / (len(projected) * 4))
    if not np.isfinite(tau2) or tau2 <= 0:
        raise RuntimeError("invalid source-only EB prior")
    raw_hash = sha(rows)
    plan = FrozenEBPlan(date, names, inputs, mean, scale, pcs, Q, RIDGE, U, mu, tau2, raw_hash, "fresh", "fresh", raw_hash)
    deployed = np.concatenate([fit_deployment_carrier(records[name], plan, records[name].trial_values[:3])["carrier"] for name in names])
    rms = float(np.sqrt(np.mean(np.square(deployed))))
    if not np.isfinite(rms) or rms <= 0:
        raise RuntimeError("invalid source-only H-C RMS")
    body = {"schema": "fresh_h1_lodo_source_hc_v2", "fold": date, "q": Q, "lambda": RIDGE,
            "source_sessions": list(names), "source_input_sha256": list(inputs), "source_m3_only": True,
            "target_records_opened": 0, "rms": rms,
            "arrays": {"mean": sha(mean), "scale": sha(scale), "pcs": sha(pcs), "U": sha(U), "mu": sha(mu)},
            "tau2": tau2, "raw_rows_sha256": raw_hash}
    return plan, rms, body


def load_frozen_authority(directory: Path) -> tuple[FrozenEBPlan, float, dict]:
    body = json.loads((directory / "source_hc_plan.json").read_text())
    with np.load(directory / "source_hc_plan_arrays.npz", allow_pickle=False) as z:
        arrays = {key: np.asarray(z[key], np.float64) for key in ("mean", "scale", "pcs", "U", "mu")}
        tau2 = float(z["tau2"]); rms = float(z["rms"])
    if body["q"] != Q or body["lambda"] != RIDGE or body["target_records_opened"] != 0:
        raise RuntimeError("invalid frozen source authority")
    for key, value in arrays.items():
        if sha(value) != body["arrays"][key]:
            raise RuntimeError(f"frozen H-C array hash mismatch: {key}")
    plan = FrozenEBPlan(body["fold"], tuple(body["source_sessions"]), tuple(body["source_input_sha256"]),
                        arrays["mean"], arrays["scale"], arrays["pcs"], Q, RIDGE, arrays["U"], arrays["mu"], tau2,
                        body["raw_rows_sha256"], "fresh", "fresh", body["raw_rows_sha256"])
    return plan, rms, body


def write_session(path: Path, record, plan, rms: float, trials, *, stride: int) -> dict:
    support = tuple(map(float, record.trial_values[:3]))
    activity = np.stack([interpolate_trial_identity(record, value) for value in support]).astype(np.float32)
    carrier = (fit_deployment_carrier(record, plan, support)["carrier"] / rms).astype(np.float32)
    X, valid, y, ends, starts, segment_starts, native_index = endpoint_windows(record, trials, stride=stride)
    if np.any(starts < segment_starts) or np.any(np.isin(native_index, np.arange(3))): raise RuntimeError("support/query raw interval leak")
    np.savez_compressed(path, X=X, valid=valid, y=y, ends=ends, starts=starts, segment_starts=segment_starts, activity=activity, carrier=carrier,
                        support=np.asarray(support), query=np.asarray(trials), native_query_indices=np.asarray([record.trial_values.index(float(v)) for v in trials]), stride=np.int64(stride))
    return {"samples": int(len(X)), "available_trial_ids": list(map(float, record.trial_values)), "support_trials": list(support), "support_indices": [0, 1, 2],
            "query_trials": list(map(float, trials)), "query_indices": [record.trial_values.index(float(v)) for v in trials], "stride": stride,
            "X_sha256": sha(X), "y_sha256": sha(y), "activity_sha256": sha(activity), "carrier_sha256": sha(carrier),
            "endpoint_sha256": sha(ends), "starts_sha256": sha(starts), "segment_starts_sha256": sha(segment_starts), "raw_interval_disjoint": True}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--fold", required=True, choices=H1_SESSION_DATES)
    parser.add_argument("--surface", required=True, choices=("source", "target"))
    args = parser.parse_args()
    source = tuple(s for date in H1_SESSION_DATES if date != args.fold for s in H1_SESSIONS_BY_DATE[date])
    chosen = source if args.surface == "source" else tuple(H1_SESSIONS_BY_DATE[args.fold])
    directory = args.dest / args.fold / args.surface
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"refusing to overwrite {directory}")
    directory.mkdir(parents=True)
    paths = index_heldin_calib(DATA)
    if args.surface == "source":
        records = {name: load_record(paths[name]) for name in source}
        plan, rms, authority = fresh_plan(args.fold, records)
        (directory / "source_hc_plan.json").write_text(json.dumps(authority, indent=2, sort_keys=True) + "\n")
        np.savez_compressed(directory / "source_hc_plan_arrays.npz", mean=plan.mean, scale=plan.scale, pcs=plan.pcs, U=plan.U, mu=plan.mu, tau2=np.float64(plan.tau2), rms=np.float64(rms))
    else:
        plan, rms, authority = load_frozen_authority(args.dest / args.fold / "source")
        records = {name: load_record(paths[name]) for name in chosen}
    manifest = {"schema": "h1_lodo_prepare_v2", "fold": args.fold, "surface": args.surface,
                "source_authority": authority, "source_sessions": list(source), "records": {}}
    for name in chosen:
        record = records[name]
        available = tuple(record.trial_values)
        if len(available) < 6: raise RuntimeError(f"{name}: need >=6 available eval-valid trials")
        primary = available[3:-2] if args.surface == "source" else available[3:]
        manifest["records"][name] = write_session(directory / f"{name}.npz", record, plan, rms, primary, stride=4 if args.surface == "source" else 1)
        if args.surface == "source":
            manifest["records"][name]["validation"] = write_session(directory / f"{name}.val.npz", record, plan, rms, available[-2:], stride=4)
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
