#!/usr/bin/env python3
"""Export label-free CORAL and AlignedFA Falcon decoder payloads.

This is deliberately an export/replay tool, not an experiment runner.  It
loads the completed source models, reuses every saved held-out adapter, and
fits adapters only for the missing held-in sessions from neural activity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
import time
import types
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parents[1]
WORKSPACE = HERE.parents[1]
# Saved pickle classes carry this module name, while this repository's
# external_baselines_v1 is intentionally outside the installed v2 package.
# Register only that leaf package and keep the installed v2 parent available.
sys.path.insert(0, str(WORKSPACE / "btransform_unified_v1" / "src"))
_parent = "btransform_unified_v2"
if _parent not in sys.modules:
    _v2 = types.ModuleType(_parent)
    _v2.__path__ = [str(HERE.parent)]
    sys.modules[_parent] = _v2
else:
    _v2 = sys.modules[_parent]
_leaf = "btransform_unified_v2.external_baselines_v1"
if _leaf not in sys.modules:
    _module = types.ModuleType(_leaf)
    _module.__path__ = [str(HERE)]
    sys.modules[_leaf] = _module
    setattr(_v2, "external_baselines_v1", _module)
from btransform_unified_v2.external_baselines_v1 import data


HISTORY = 10
SCHEMA = "external_gf_falcon_payload_v1"
EXPECTED_ROSTER = {"m2": 13, "m1": 7, "h1": 27}
MAX_BATCH = {"m2": 7, "m1": 4, "h1": 8}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def activity(item: dict[str, Any]) -> np.ndarray:
    if "activity" not in item:
        raise KeyError("activity unavailable")
    x = np.asarray(item["activity"], np.float32)
    return np.ascontiguousarray(x.reshape(-1, x.shape[-1]))


def raw_basename(item: dict[str, Any]) -> str:
    raw = item.get("support_provenance", {}).get("raw_nwb")
    if not raw:
        raise ValueError("missing support_provenance.raw_nwb")
    return Path(raw).stem


def session_basename(task: str, session: str, role: str, item: dict[str, Any]) -> str:
    # M2 held-in cache provenance intentionally records only its activity file;
    # compose the canonical Falcon basename from the actual session identifier.
    if task == "m2" and not item.get("support_provenance", {}).get("raw_nwb"):
        return f"sub-MonkeyN-held-{'in' if role == 'heldin' else 'out'}-calib_{session}_behavior+ecephys"
    return raw_basename(item)


def falcon_tag(task: str, item: dict[str, Any]) -> str:
    # Hash the actual source filename, exactly as Falcon decoder.reset does.
    sys.path.insert(0, str(WORKSPACE / "SPINT-main"))
    from falcon_challenge.config import FalconConfig, FalconTask
    return str(FalconConfig(getattr(FalconTask, task)).hash_dataset(raw_basename(item)))


def prefix_for(tag: str) -> str:
    return "s_" + "".join(c if c.isalnum() else "_" for c in tag)


def np_numeric(x: Any, *, dtype=np.float64) -> np.ndarray:
    a = np.ascontiguousarray(np.asarray(x, dtype=dtype))
    if a.dtype == object or not np.isfinite(a).all():
        raise ValueError("payload array is non-numeric or non-finite")
    return a


def coral_arrays(adapter: Any | None, source: Any, *, identity: bool) -> dict[str, np.ndarray]:
    c = int(source.source_mean.size)
    if identity:
        return {
            "target_mean": np_numeric(source.source_mean),
            "source_mean": np_numeric(source.source_mean),
            "target_invroot": np.eye(c, dtype=np.float64),
            "source_root": np.eye(c, dtype=np.float64),
        }
    if adapter is None:
        raise ValueError("missing CORAL adapter")
    return {k: np_numeric(getattr(adapter, k)) for k in
            ("target_mean", "source_mean", "target_invroot", "source_root")}


def fa_arrays(adapter: Any | None, source_fa: Any, *, identity: bool) -> dict[str, np.ndarray]:
    fa = source_fa if identity else adapter.target_fa
    components = np_numeric(fa.components_)
    noise = np_numeric(fa.noise_variance_)
    wpsi = components / noise[None, :]
    cov_z = np.linalg.inv(np.eye(components.shape[0]) + wpsi @ components.T)
    rotation = np.eye(components.shape[0], dtype=np.float64) if identity else np_numeric(adapter.rotation)
    return {"target_fa_mean": np_numeric(fa.mean_), "Wpsi": np_numeric(wpsi),
            "cov_z": np_numeric(cov_z), "rotation": rotation}


def transform_coral(x: np.ndarray, values: dict[str, np.ndarray]) -> np.ndarray:
    return np.asarray((np.asarray(x, np.float64) - values["target_mean"]) @ values["target_invroot"]
                      @ values["source_root"] + values["source_mean"], np.float32)


def transform_fa(x: np.ndarray, values: dict[str, np.ndarray]) -> np.ndarray:
    # This is sklearn FactorAnalysis.transform written with the exported
    # Woodbury factors, followed by the stored Procrustes rotation.
    return np.asarray((np.asarray(x, np.float64) - values["target_fa_mean"]) @ values["Wpsi"].T
                      @ values["cov_z"] @ values["rotation"], np.float32)


def predict_from_arrays(item: dict[str, Any], values: dict[str, np.ndarray], *, method: str,
                        wf_mean: np.ndarray, wf_scale: np.ndarray, wf_coef: np.ndarray,
                        context: int) -> np.ndarray:
    ids = np.arange(len(item["starts"]), dtype=np.int64)
    feat = data.causal_features(item, ids, context=context, history=HISTORY)
    input_channels = np.asarray(item["X"]).shape[1]
    bins = feat.reshape(len(feat), HISTORY, input_channels)
    valid = (np.asarray(item["starts"], np.int64)[:, None] + context - HISTORY
             + np.arange(HISTORY)[None, :] >= int(item.get("pad", 0)))
    flat = bins.reshape(-1, input_channels)
    aligned = transform_coral(flat, values) if method == "coral_wf" else transform_fa(flat, values)
    aligned = aligned.reshape(len(feat), HISTORY, -1)
    aligned[~valid] = 0.0
    z = aligned.reshape(len(feat), -1).astype(np.float64)
    pred = np.c_[((z - wf_mean) / wf_scale), np.ones(len(z))] @ wf_coef
    return np.asarray(pred, np.float32)


def adapter_diagnostic(method: str, adapter: Any | None, *, identity: bool,
                       calibration_samples: int, fit_seconds: float) -> dict[str, Any]:
    if identity:
        return {"kind": "source_reference_identity", "target_labels_used": False,
                "fit_seconds": 0.0, "calibration_samples": calibration_samples}
    d = dict(adapter.diagnostics)
    d["kind"] = "reused_heldout_adapter" if fit_seconds == 0.0 else "new_heldin_neural_only_adapter"
    d["fit_seconds_export"] = fit_seconds
    d["calibration_samples"] = calibration_samples
    d["target_labels_used"] = False
    return d


def load_export_data(task: str) -> dict[str, Any]:
    """Load calibration/query planes, with a receipt-backed M2 fallback.

    The completed M2 run's six target activity files were intentionally
    removed from this workspace after fitting.  Their saved adapters remain
    authoritative; the fallback opens only the query X plane to replay them.
    """
    try:
        return data.load(task, evaluation=True)
    except FileNotFoundError as exc:
        # Do not mask an unrelated source/query dependency failure.  The only
        # accepted absence is the deliberately retired M2 held-out activity
        # surface used by the completed run before its adapters were saved.
        retired_root = (HERE.parent / "results" / "rift_v1" / "m2_joint_ext6_raw_m33_v1").resolve()
        missing = Path(exc.filename).resolve() if exc.filename else None
        try:
            is_retired_target = missing is not None and missing.is_relative_to(retired_root)
        except ValueError:
            is_retired_target = False
        if task != "m2" or not is_retired_target:
            raise
        base = data.load("m2", evaluation=False)
        receipt = json.loads((HERE / "results/m2_full_v1/receipt.json").read_text())
        _train_reader, score_reader = data.activity_data._load_m2_query_modules()
        query = score_reader.query_surface(data.activity_data.M2_QUERY)
        evaluation = {}
        for session, q in query.items():
            q = dict(q)
            q["support_provenance"] = dict(receipt["target_support_provenance"][session])
            q["hashes"] = dict(receipt["target_input_hashes"][session])
            q["_support_activity_sha256"] = receipt["target_support_hashes"][session]
            evaluation[session] = q
        base["evaluation"] = evaluation
        return base


def build_payload(task: str, method: str, destination: Path) -> dict[str, Any]:
    started = time.monotonic()
    model_path = HERE / "results" / f"{task}_full_v1" / "models.pkl"
    with model_path.open("rb") as f:
        saved = pickle.load(f)
    loaded = load_export_data(task)
    train, heldout = loaded["train"], loaded["evaluation"]
    reference = data.reference_source_day(train)
    model = saved["wf_raw"] if method == "coral_wf" else saved["fa_wf"]
    wf_mean, wf_scale, wf_coef = (np_numeric(model.mean_), np_numeric(model.scale_), np_numeric(model.coef_))
    source = saved["coral"] if method == "coral_wf" else saved["fa"]
    records: list[tuple[str, str, dict[str, Any], str]] = []
    records += [(s, "heldin", v, session_basename(task, s, "heldin", v)) for s, v in train.items()]
    records += [(s, "heldout", v, session_basename(task, s, "heldout", v)) for s, v in heldout.items()]
    if len(records) != EXPECTED_ROSTER[task]:
        raise RuntimeError(f"{task}: roster {len(records)} != {EXPECTED_ROSTER[task]}")
    tags = [falcon_tag(task, {**item, "support_provenance": {**item.get("support_provenance", {}), "raw_nwb": basename}})
            for _, _, item, basename in records]
    if len(set(tags)) != len(tags):
        raise RuntimeError(f"{task}: Falcon tag collision: {tags}")

    arrays: dict[str, np.ndarray] = {"wf_mean": wf_mean, "wf_scale": wf_scale, "wf_coef": wf_coef}
    sessions: list[dict[str, Any]] = []
    reused = saved["target_adapters"]
    for session, role, item, basename in records:
        tag = falcon_tag(task, {**item, "support_provenance": {**item.get("support_provenance", {}), "raw_nwb": basename}})
        pfx = prefix_for(tag)
        is_reference = role == "heldin" and session == reference
        adapter = None
        fit_seconds = 0.0
        if not is_reference:
            adapter = reused.get((method, session))
            if adapter is None:
                if role == "heldout":
                    raise RuntimeError(f"{task}/{method}/{session}: completed held-out adapter missing; refusing refit")
                t0 = time.monotonic()
                adapter = source.calibrate(activity(item))
                fit_seconds = time.monotonic() - t0
        values = (coral_arrays(adapter, source, identity=is_reference) if method == "coral_wf"
                  else fa_arrays(adapter, source.source_fa, identity=is_reference))
        for suffix, value in values.items():
            arrays[f"{pfx}_{suffix}"] = value
        samples = (len(activity(item)) if "activity" in item else int(adapter.diagnostics.get("target_samples", 0)))
        diag = adapter_diagnostic(method, adapter, identity=is_reference,
                                  calibration_samples=samples, fit_seconds=fit_seconds)
        support_hash = (item.get("_support_activity_sha256") or item.get("hashes", {}).get("activity")
                        or data.sha_array(activity(item)))
        sessions.append({"tag": tag, "source_session": session, "raw_basename": basename,
                         "role": role, "array_prefix": pfx, "support_activity_sha256": support_hash,
                         "calibration": diag})

    # Existing held-out adapters and stored local predictions are immutable
    # reference values.  Their replay proves the export algebra without using Y.
    checks: dict[str, Any] = {}
    by_session = {row[0]: row for row in records}
    for session, role, item, _basename in records:
        tag = falcon_tag(task, {**item, "support_provenance": {**item.get("support_provenance", {}), "raw_nwb": _basename}})
        pfx = prefix_for(tag)
        vals = {suffix: arrays[f"{pfx}_{suffix}"] for suffix in
                (("target_mean", "source_mean", "target_invroot", "source_root") if method == "coral_wf"
                 else ("target_fa_mean", "Wpsi", "cov_z", "rotation"))}
        replay = predict_from_arrays(item, vals, method=method, wf_mean=wf_mean, wf_scale=wf_scale,
                                     wf_coef=wf_coef, context=int(loaded["metadata"]["context"]))
        # All held-out sessions have a prior query prediction.  The reference
        # source's first 32 rows exercise the unaligned source pipeline.
        stored_path = HERE / "results" / f"{task}_full_v1" / f"pred_{method}_{session}.npy"
        if stored_path.is_file():
            stored = np.load(stored_path)
            if replay.shape != stored.shape:
                raise RuntimeError(f"{task}/{method}/{session}: replay shape {replay.shape} != {stored.shape}")
            err = float(np.max(np.abs(replay.astype(np.float64) - stored.astype(np.float64))))
            checks[session] = {"kind": "stored_heldout_query", "n": int(len(replay)), "max_abs_error": err,
                               "tolerance": 2e-6, "pass": err <= 2e-6}
        elif session == reference:
            # Saved source model direct replay is evaluated on a first batch only,
            # retaining the export's no-target-label property.
            direct_feature = data.causal_features(item, np.arange(32), context=int(loaded["metadata"]["context"]), history=HISTORY)
            if method == "coral_wf":
                expected = model.predict(direct_feature)
            else:
                c = int(loaded["metadata"]["units"])
                bins = direct_feature.reshape(32, HISTORY, c)
                valid = (np.asarray(item["starts"], np.int64)[:32, None] + int(loaded["metadata"]["context"]) - HISTORY
                         + np.arange(HISTORY)[None, :] >= int(item.get("pad", 0)))
                transformed = saved["fa"].source_transform(bins.reshape(-1, c)).reshape(32, HISTORY, -1)
                transformed[~valid] = 0.0
                expected = model.predict(transformed.reshape(32, -1))
            err = float(np.max(np.abs(replay[:32].astype(np.float64) - expected.astype(np.float64))))
            checks[session] = {"kind": "source_pipeline_first32", "n": 32,
                               "max_abs_error": err, "tolerance": 2e-6, "pass": err <= 2e-6}
    failed = {s: x for s, x in checks.items() if not x["pass"]}
    if failed:
        raise RuntimeError(f"{task}/{method}: replay mismatch {failed}")

    if destination.exists():
        raise FileExistsError(f"refuse to overwrite {destination}")
    destination.mkdir(parents=True)
    np.savez_compressed(destination / "payload.npz", **arrays)
    with np.load(destination / "payload.npz", allow_pickle=False) as z:
        if any(z[k].dtype == object for k in z.files):
            raise RuntimeError("object dtype escaped payload")
        payload_keys = sorted(z.files)
    feature_channels = int(wf_mean.size // HISTORY)
    manifest = {
        "schema": SCHEMA, "task": task, "method": method.removesuffix("_wf"), "training_method": method,
        "channels": int(loaded["metadata"]["units"]), "model_feature_channels": feature_channels,
        "outputs": int(wf_coef.shape[1]), "history": HISTORY, "max_batch": MAX_BATCH[task],
        "sessions": {x["tag"]: {k: v for k, v in x.items() if k != "tag"} for x in sessions},
        "expected_prediction_fields": {"task": task, "channels": int(loaded["metadata"]["units"]),
                                        "outputs": int(wf_coef.shape[1]), "history": HISTORY},
        "source": {"reference_session": reference, "models_pkl_sha256": sha256_file(model_path),
                   "source_activity_sha256": data.sha_array(activity(train[reference])),
                   "wf": "saved source-only WienerRidge; coef final row is intercept",
                   "source_fa": "saved completed sklearn FactorAnalysis; reference rotation is identity"},
        "calibration": {"target_labels_used": False, "target_backprop_used": False,
                        "heldout_adapters": "reused from completed models.pkl without refit",
                        "missing_heldin_adapters": "fit from each session activity.reshape(-1,C) only",
                        "fa_hyperparameters": {"max_iter": int(saved["fa"].max_iter), "n_init": int(saved["fa"].n_init),
                                               "latent_dim": int(saved["fa"].latent_dim), "stable_fraction": float(saved["fa"].stable_fraction)}},
        "payload": {"file": "payload.npz", "sha256": sha256_file(destination / "payload.npz"),
                    "keys": payload_keys, "numeric_only": True, "contains_raw_data": False, "contains_target_y": False},
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    audit = {"schema": "external_gf_falcon_payload_export_audit_v1", "task": task, "method": method,
             "payload_sha256": manifest["payload"]["sha256"], "payload_bytes": (destination / "payload.npz").stat().st_size,
             "tag_count": len(sessions), "tags": [x["tag"] for x in sessions], "checks": checks,
             "fit_seconds_total": time.monotonic() - started,
             "new_heldin_fa_diagnostics": {x["tag"]: x["calibration"].get("target_fit") for x in sessions
                                            if x["calibration"].get("kind") == "new_heldin_neural_only_adapter" and method == "aligned_fa_wf"}}
    (destination / "export_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    return audit


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=("m2", "m1", "h1"), required=True)
    p.add_argument("--method", choices=("coral_wf", "aligned_fa_wf"), required=True)
    p.add_argument("--dest", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(build_payload(a.task, a.method, a.dest), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
