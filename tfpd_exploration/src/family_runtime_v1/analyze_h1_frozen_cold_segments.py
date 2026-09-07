"""Archive-only descriptive cold-versus-full-W700 H1 split analysis.

No model, cache, Torch, decoder, label fitting, selection, or calibration path
is imported here.  The fixed split is endpoint ``end < 699`` (cold history)
versus ``end >= 699`` (a complete W700 history), computed from existing frozen
FP64 archives only.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from . import compare_h1_frozen_quality as compare
from .complete_h1_family_source import artifact_audit, atomic_json, require_same_files, sha, validate_archive


COUNT, SESSIONS, COLD_END_EXCLUSIVE = 20325, 13, 699
ROOT = Path(__file__).resolve().parents[2]
ORIGINAL_DIR = ROOT / "results/family_runtime_v1/original_h1_frozen_same20325_v1"
ORIGINAL_RECEIPT_SHA = compare.ORIGINAL_RECEIPT_SHA
ORIGINAL_NPZ_SHA = compare.ORIGINAL_NPZ_SHA
GO = "H1_FROZEN_COLD_SEGMENTS_GO"
STATUS = "PASS_ARCHIVE_ONLY_COLD_VS_FULL_W700_DESCRIPTIVE"


def _finite_positive(value: np.ndarray, name: str) -> None:
    if not np.isfinite(value).all():
        raise RuntimeError(name + " must be finite")
    if value.ndim != 2 or value.shape[1] != 7 or not len(value):
        raise RuntimeError(name + " must be nonempty [n,7]")


def segment_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, object]:
    prediction, target = np.asarray(prediction, dtype=np.float64), np.asarray(target, dtype=np.float64)
    _finite_positive(prediction, "prediction"); _finite_positive(target, "target")
    if prediction.shape != target.shape:
        raise RuntimeError("prediction/target shape mismatch")
    residual = prediction - target
    sse = float(np.square(residual).sum(dtype=np.float64))
    centered = target - target.mean(axis=0, keepdims=True)
    sst = float(np.square(centered).sum(dtype=np.float64))
    if not np.isfinite(sse) or not np.isfinite(sst) or sst <= 0:
        raise RuntimeError("finite positive target variance required")
    return {"count": int(len(target)), "r2_concat_float64": float(1 - sse / sst),
            "mse_float64": float(sse / target.size), "sse_float64": sse, "sst_float64": sst,
            "prediction_mean": prediction.mean(axis=0, dtype=np.float64).tolist(),
            "prediction_std": prediction.std(axis=0, dtype=np.float64).tolist(),
            "target_mean": target.mean(axis=0, dtype=np.float64).tolist(),
            "target_std": target.std(axis=0, dtype=np.float64).tolist()}


def analyze_arrays(arrays: dict[str, np.ndarray]) -> dict[str, object]:
    required = {"prediction", "target", "end", "session_id"}
    if set(arrays) != required:
        raise RuntimeError("exact frozen archive fields required")
    prediction, target, end, session_id = (arrays[name] for name in ("prediction", "target", "end", "session_id"))
    if (prediction.shape != (COUNT, 7) or target.shape != (COUNT, 7) or prediction.dtype != np.float64
            or target.dtype != np.float64 or end.shape != (COUNT,) or end.dtype != np.int64
            or session_id.shape != (COUNT,) or session_id.dtype.kind != "U"):
        raise RuntimeError("fixed FP64 archive geometry/dtype drift")
    sessions = sorted(set(session_id.tolist()))
    if len(sessions) != SESSIONS:
        raise RuntimeError("exact 13-session frozen archive required")
    if np.any(end < 0) or any(np.any(np.diff(end[session_id == session]) <= 0) for session in sessions):
        raise RuntimeError("each session requires nonnegative strictly chronological endpoints")
    masks = {"all": np.ones(COUNT, dtype=bool), "cold_history_lt_699": end < COLD_END_EXCLUSIVE,
             "full_w700_ge_699": end >= COLD_END_EXCLUSIVE}
    if np.any(masks["cold_history_lt_699"] & masks["full_w700_ge_699"]):
        raise RuntimeError("cold/full split overlap")
    if int(masks["cold_history_lt_699"].sum() + masks["full_w700_ge_699"].sum()) != COUNT:
        raise RuntimeError("cold/full split does not partition all endpoints")
    groups = {}
    for name, mask in masks.items():
        pooled = segment_metrics(prediction[mask], target[mask])
        per_session = {}
        for session in sessions:
            selected = mask & (session_id == session)
            per_session[session] = segment_metrics(prediction[selected], target[selected])
        if sum(row["count"] for row in per_session.values()) != pooled["count"]:
            raise RuntimeError("per-session/pooled group count mismatch")
        groups[name] = {"pooled": pooled, "per_session": per_session}
    if groups["all"]["pooled"]["count"] != COUNT:
        raise RuntimeError("all group endpoint count drift")
    split_sse = sum(groups[name]["pooled"]["sse_float64"] for name in ("cold_history_lt_699", "full_w700_ge_699"))
    if not np.isclose(split_sse, groups["all"]["pooled"]["sse_float64"], rtol=1e-12, atol=1e-15):
        raise RuntimeError("cold/full SSE does not recombine into all-bin SSE")
    return {"sessions": sessions, "groups": groups}


def paired_deltas(candidate: dict[str, object], original: dict[str, object]) -> dict[str, object]:
    if candidate["sessions"] != original["sessions"]:
        raise RuntimeError("candidate/original session roster drift")
    result = {}
    for group in candidate["groups"]:
        left, right = candidate["groups"][group], original["groups"].get(group)
        if right is None or left["pooled"]["count"] != right["pooled"]["count"]:
            raise RuntimeError("candidate/original group count drift")
        if any(left["per_session"][s]["count"] != right["per_session"][s]["count"] for s in candidate["sessions"]):
            raise RuntimeError("candidate/original per-session group count drift")
        result[group] = {"pooled_r2_delta": left["pooled"]["r2_concat_float64"] - right["pooled"]["r2_concat_float64"],
                         "per_session_r2_delta": {session: left["per_session"][session]["r2_concat_float64"] - right["per_session"][session]["r2_concat_float64"] for session in candidate["sessions"]}}
    return result


def _gate(args) -> None:
    if os.environ.get(GO) != "1":
        raise RuntimeError("explicit archive-only cold-segment GO required")
    if args.output.exists():
        raise FileExistsError(args.output)


def _original_receipt(path: Path, npz: Path) -> tuple[dict, Path]:
    if sha(path) != ORIGINAL_RECEIPT_SHA or sha(npz) != ORIGINAL_NPZ_SHA:
        raise RuntimeError("fixed original H1 receipt/archive SHA drift")
    receipt = json.loads(path.read_text())
    if (receipt.get("status") != "PASS_AS_SHIPPED_ORIGINAL_H1_REFERENCE_ONLY" or receipt.get("scored_count") != COUNT
            or receipt.get("public_calls") != 20920 or receipt.get("pre") != receipt.get("post")
            or receipt.get("archive", {}).get("sha256") != ORIGINAL_NPZ_SHA):
        raise RuntimeError("original H1 frozen receipt semantic drift")
    authority = Path(receipt["pre"]["authority"]["path"])
    if sha(authority) != receipt["pre"]["authority"]["sha256"]:
        raise RuntimeError("original H1 source authority SHA drift")
    return receipt, authority


def run(args) -> dict:
    _gate(args)
    audit = artifact_audit(args.formal)
    receipt, authority_path = _original_receipt(args.original_receipt, args.original_npz)
    formal_input, original_input = args.formal / "input_authority.json", args.original_receipt.parent / "input_authority.json"
    if not formal_input.is_file() or not original_input.is_file():
        raise RuntimeError("formal/original input authority missing")
    if json.loads(original_input.read_text()).get("pre") != receipt["pre"]:
        raise RuntimeError("original input authority/pre-image drift")
    bindings = json.loads(formal_input.read_text()).get("bindings", {})
    if (bindings.get("source_cache_sha256") != receipt["pre"]["cache"]["sha256"]
            or bindings.get("source_authority_sha256") != receipt["pre"]["authority"]["sha256"]):
        raise RuntimeError("formal/original frozen source authority mismatch")
    formal_npz = {arm: args.formal / "exports" / f"{arm}_selected_complete_native_float64.npz" for arm in ("flat", "route")}
    immutable = {str(path): sha(path) for path in (Path(__file__), Path(__file__).with_name("compare_h1_frozen_quality.py"),
                 Path(__file__).with_name("complete_h1_family_source.py"), Path(__file__).with_name("h1_replay_contract.py"), args.original_receipt, args.original_npz,
                 authority_path, formal_input, original_input, *formal_npz.values())}
    # Everything above binds JSON/NPZ bytes before any archive is deserialized.
    original_arrays = validate_archive(args.original_npz)
    compare._same_metrics(compare.metric(original_arrays), receipt["metrics"], prefix="original ")
    analyses = {"ORIGINAL_frozen": analyze_arrays(original_arrays)}
    for arm, path in formal_npz.items():
        arrays = validate_archive(path)
        compare.same_surface(arrays, original_arrays)
        compare._same_metrics(compare.metric(arrays), audit["final"]["complete"][arm]["reports"]["selected"]["complete"], prefix=arm + " ")
        analyses[arm.upper()] = analyze_arrays(arrays)
    for arm in ("FLAT", "ROUTE"):
        if analyses[arm]["groups"]["all"]["pooled"]["count"] != COUNT:
            raise RuntimeError("family all-endpoint count drift")
        for group in ("cold_history_lt_699", "full_w700_ge_699"):
            if len(analyses[arm]["groups"][group]["per_session"]) != SESSIONS:
                raise RuntimeError("family group must retain all 13 session rows")
    post = artifact_audit(args.formal)
    if post != audit:
        raise RuntimeError("formal artifact authority changed during archive analysis")
    immutable_post = require_same_files(immutable)
    result = {"schema": "h1_frozen_cold_vs_full_w700_archive_analysis_v1", "status": STATUS,
              "scope": "fixed existing FP64 archives only; descriptive cold/full history split; no model selection, NI, calibration, or fitting",
              "split": {"cold_history_lt_699": "endpoint end < 699", "full_w700_ge_699": "endpoint end >= 699", "all": "all 20325 endpoints"},
              "cut_is_fixed_w700_history_boundary": True, "c2_prediction_npz_available": False, "c2_excluded": True,
              "pre_artifact": audit, "post_artifact": post, "immutable_pre": immutable, "immutable_post": immutable_post,
              "analyses": analyses, "flat_minus_original": paired_deltas(analyses["FLAT"], analyses["ORIGINAL_frozen"]),
              "route_minus_original": paired_deltas(analyses["ROUTE"], analyses["ORIGINAL_frozen"]),
              "r2_note": "group R2 is recomputed from group SSE/SST; it is not a count-weighted average of subgroup R2 values"}
    atomic_json(result, args.output)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--original-receipt", type=Path, default=ORIGINAL_DIR / "receipt.json")
    parser.add_argument("--original-npz", type=Path, default=ORIGINAL_DIR / "original_h1_minival_native_float64.npz")
    result = run(parser.parse_args())
    print(json.dumps({"status": result["status"], "schema": result["schema"]}, sort_keys=True))
