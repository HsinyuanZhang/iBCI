"""Post-freeze H1 quality comparison from immutable JSON/NPZ archives only.

No model, checkpoint, or cache payload is deserialized and no forward, epoch
selection or calibration is performed. Original and C2 are separate references.
"""
from __future__ import annotations

import argparse
import os
import resource
import time
from pathlib import Path

import numpy as np

from . import complete_h1_queryage_source as proof
from .compare_h1_frozen_quality import (
    C2_PATH, C2_SHA, ORIGINAL_DIR, ORIGINAL_NPZ_SHA, ORIGINAL_RECEIPT_SHA,
    _c2, _deltas, _original_metric, same_surface,
)

ARMS, LABELS = ("flat", "route"), ("selected", "epoch12")
GO, WALL_SECONDS, RSS_LIMIT = "H1_QUERYAGE_QUALITY_COMPARISON_GO", 900, 4 << 30
sha, read, atomic = proof.sha, proof.read, proof.atomic_json
_canonical, _inside = proof._canonical, proof._inside


def closure():
    paths = [Path(__file__), *(Path(__file__).with_name(name) for name in (
        "complete_h1_queryage_source.py", "compare_h1_frozen_quality.py",
        "complete_h1_family_source.py", "h1_replay_contract.py"))]
    return {"comparison": {str(p): sha(p) for p in paths}, "proof_runtime": proof.code_closure()}


def _metrics(arrays):
    return proof._metric(arrays)


def _same(actual, reported):
    try:
        proof._compare_metrics(actual, reported, tolerance=1e-10)
    except RuntimeError as exc:
        raise RuntimeError("reported complete/per-session metric drift") from exc
    if actual["worst_session"] != reported.get("worst_session"):
        raise RuntimeError("reported worst-session identity drift")


def _candidate(formal, audit, original, *, guard=lambda: None):
    rows = {}
    for arm in ARMS:
        for label in LABELS:
            guard()
            path = formal / "exports" / f"{arm}_{label}_complete_native_float64.npz"
            plain = formal / "exports" / f"{arm}_{label}_plain_ema.pt"
            report = audit["finals"][arm]["reports"][label]
            selected = audit["selection_freeze"][label][arm]
            if (report.get("complete_archive") != str(path) or report.get("complete_archive_sha256") != sha(path)
                    or report.get("plain_ema_path") != str(plain) or report.get("plain_ema_sha256") != sha(plain)
                    or report.get("epoch") != selected["epoch"]
                    or report.get("checkpoint_sha256") != selected["checkpoint_sha256"]):
                raise RuntimeError("selected/epoch12 export identity drift")
            arrays = proof._archive(path)
            same_surface(arrays, original)
            actual = _metrics(arrays)
            _same(actual, report["complete"])
            rows[f"{arm.upper()}_{label}"] = actual
            guard()
    return rows


def collect_bindings(formal: Path, output: Path):
    formal, output = _canonical(formal), _canonical(output)
    bound = proof.collect_bindings(formal, output)
    original_receipt = ORIGINAL_DIR / "receipt.json"
    original_npz = ORIGINAL_DIR / "original_h1_minival_native_float64.npz"
    paths = [original_receipt, original_npz, ORIGINAL_DIR / "input_authority.json", C2_PATH]
    if any(not p.is_file() for p in paths):
        raise FileNotFoundError("comparison baseline input missing")
    original = read(original_receipt)
    authority = _canonical(Path(original["pre"]["authority"]["path"]))
    cache = _canonical(Path(original["pre"]["cache"]["path"]))
    paths += [authority, cache]
    for arm in ARMS:
        paths += [formal / "exports" / f"{arm}_epoch12_complete_native_float64.npz",
                  formal / "exports" / f"{arm}_epoch12_plain_ema.pt"]
    if any(not p.is_file() for p in paths):
        raise FileNotFoundError("comparison source or terminal export missing")
    return {"schema": "h1_queryage_quality_comparison_v1", "formal": str(formal), "output": str(output),
            "inputs": {**bound["inputs"], **{str(_canonical(p)): sha(p) for p in paths}},
            "code_closure": closure(), "limits": {"wall_seconds": WALL_SECONDS, "peak_rss_bytes": RSS_LIMIT}}


def _baseline(binding, audit, *, guard=lambda: None):
    guard()
    receipt_path, npz_path = ORIGINAL_DIR / "receipt.json", ORIGINAL_DIR / "original_h1_minival_native_float64.npz"
    if sha(receipt_path) != ORIGINAL_RECEIPT_SHA or sha(npz_path) != ORIGINAL_NPZ_SHA or sha(C2_PATH) != C2_SHA:
        raise RuntimeError("pinned Original/C2 baseline bytes drifted")
    original = read(receipt_path)
    sidecar = read(ORIGINAL_DIR / "input_authority.json")
    if sidecar.get("pre") != original.get("pre"):
        raise RuntimeError("Original input sidecar/pre-authority drift")
    recorded = original["pre"]
    for key in ("cache", "authority"):
        path = _canonical(Path(recorded[key]["path"]))
        if sha(path) != recorded[key]["sha256"] or binding["inputs"].get(str(path)) != recorded[key]["sha256"]:
            raise RuntimeError("Original current source authority bytes drifted")
        if audit["authority"].get("bindings", {}).get("inputs", {}).get(str(path)) != recorded[key]["sha256"]:
            raise RuntimeError("formal/Original immutable source identity differs")
    original.update({"_path": str(receipt_path), "_npz_path": str(npz_path)})
    arrays = proof._archive(npz_path)
    observed = _original_metric(original, arrays)
    c2 = _c2(read(C2_PATH), read(Path(recorded["authority"]["path"])), observed["per_session_r2_float64"])
    guard()
    return arrays, observed, c2


def run(*, formal: Path, output: Path, authorization: Path, authorization_sha256: str, allow_cpu_fixture=False):
    started = time.monotonic()
    def guard():
        if time.monotonic() - started > WALL_SECONDS or resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 > RSS_LIMIT:
            raise RuntimeError("archive-only wall/peak-RSS limit exceeded")
    formal, output, authorization = map(_canonical, (formal, output, authorization))
    roots = (formal, ORIGINAL_DIR, C2_PATH.parent, proof.CACHE.parent)
    if (output.exists() or _inside(output, authorization)
            or any(_inside(root, output) or _inside(root, authorization) for root in roots)):
        raise RuntimeError("fresh comparison output and external authorization must be disjoint from all inputs")
    if not allow_cpu_fixture and (os.environ.get(GO) != "1" or os.environ.get("CUDA_VISIBLE_DEVICES") not in ("", "-1")):
        raise RuntimeError("explicit CPU-only archive comparison GO required")
    guard(); binding = collect_bindings(formal, output); guard()
    authorized = {"schema": "h1_queryage_quality_comparison_authorization_v1", "status": "ROOT_REVIEW_GO", "bindings": binding}
    if sha(authorization) != authorization_sha256 or read(authorization) != authorized:
        raise RuntimeError("external comparison authorization drift")
    audit, _ = proof._audit_formal(formal, binding, allow_cpu_fixture=allow_cpu_fixture, guard=guard)
    original_roots = {Path(p).parent for p in audit["authority"].get("bindings", {}).get("inputs", {})}
    if any(_inside(root, output) or _inside(root, authorization) for root in original_roots):
        raise RuntimeError("comparison output/authorization overlaps original admission inputs")
    arrays, original, c2 = _baseline(binding, audit, guard=guard)
    guard(); output.mkdir()
    sidecar = output / "input_authority.json"
    sidecar_body = {"bindings": binding, "authorization_path": str(authorization), "authorization_sha256": authorization_sha256}
    atomic(sidecar_body, sidecar); sidecar_sha = sha(sidecar)
    candidate = _candidate(formal, audit, arrays, guard=guard)
    tables = {"ORIGINAL_frozen": original,
              "C2_historical_aggregate_imported": {**c2, "independently_recomputed": False,
                  "numeric_provenance": "legacy aggregate arithmetic; not recomputed from native FP64 predictions"},
              **candidate}
    post, _ = proof._audit_formal(formal, binding, allow_cpu_fixture=allow_cpu_fixture, guard=guard)
    if post != audit or collect_bindings(formal, output) != binding:
        raise RuntimeError("post-comparison formal/input/code closure drift")
    _baseline(binding, post, guard=guard)
    if sha(authorization) != authorization_sha256 or read(authorization) != authorized:
        raise RuntimeError("post-comparison external authorization drift")
    if sha(sidecar) != sidecar_sha or read(sidecar) != sidecar_body:
        raise RuntimeError("comparison pre-authority sidecar drift")
    if {p for p in output.rglob("*") if p.is_file()} != {sidecar}:
        raise RuntimeError("unexpected comparison output artifact")
    result = {"schema": "h1_queryage_same20325_descriptive_quality_v1",
              "status": "PASS_DESCRIPTIVE_ARCHIVE_COMPARISON_ONLY", "authority": binding,
              "authorization_path": str(authorization), "authorization_sha256": authorization_sha256,
              "tables": tables, "epochs": audit["selection_freeze"],
              "deltas_vs_original": {k: _deltas(k, v, original) for k, v in candidate.items()},
              "deltas_vs_c2_historical": {k: _deltas(k, v, c2) for k, v in candidate.items()},
              "worst_delta_note": "differences between each model's minimum; minima may come from different sessions",
              "selected_epoch12_note": "selected and epoch12 can be the same epoch; not independent replicates",
              "c2_prediction_npz_available": False, "c2_independently_recomputed": False,
              "noninferiority": "no prospective NI threshold; no formal or uniform-NI declaration",
              "controlled_training_claim": False, "selection_or_promotion": False,
              "comparability": "same endpoint targets; not a matched training, initialization, calibration, support-budget or exposure ablation",
              "calibration": "Original shipped two-trial calibration; QueryAge immutable M3 banks; C2 historical fixed readout",
              "cache_disclosure": "only cache file bytes are hashed; no cache/model/checkpoint deserialization or forward",
              "pre_audit": audit, "post_audit": post,
              "owned_artifact_sha256": {str(sidecar): sidecar_sha},
              "elapsed_seconds": time.monotonic() - started,
              "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)}
    guard(); atomic(result, output / "receipt.json")
    try: guard()
    except BaseException:
        atomic({"schema": result["schema"], "status": "FAILED_POST_WRITE_RESOURCE_GUARD"}, output / "receipt.json")
        raise
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    for name in ("formal", "output", "authorization"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--authorization-sha256", required=True)
    args = parser.parse_args(argv)
    return run(formal=args.formal, output=args.output, authorization=args.authorization, authorization_sha256=args.authorization_sha256)


if __name__ == "__main__":
    main()
