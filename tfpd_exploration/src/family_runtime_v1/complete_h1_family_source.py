"""Post-finalization selected H1 proof: all chronological public bins, CPU only.

The authoritative native archives supply the full 20,325-endpoint oracle.
Direct native W700 forwards check a small fixed subset including startup and
rollover. Neither endpoint/model selection nor the formal run is changed.
"""
from __future__ import annotations
import argparse
import dataclasses
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
import numpy as np
from .h1_replay_contract import replay_session

ARMS = ("flat", "route")
COUNT = 20325


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def atomic_json(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, mode="w", delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def earliest(rows):
    if len(rows) != 12 or [r["epoch"] for r in rows] != list(range(1, 13)):
        raise RuntimeError("requires immutable metrics 1..12")
    values = [r["selection"]["r2_concat_float64"] for r in rows]
    if not np.isfinite(values).all():
        raise RuntimeError("nonfinite selection")
    return rows[int(np.argmax(values))]


def artifact_audit(formal):
    # Deliberately before any torch, model, or source-cache import.
    final = read(formal / "final.json")
    if final.get("status") != "COMPLETE":
        raise RuntimeError("formal finalization incomplete")
    freeze = read(formal / "selection_freeze.json")
    if final["selection_freeze"] != freeze["selected"]:
        raise RuntimeError("root final/freeze disagreement")
    files = {}
    def bind(path, expected=None):
        actual = sha(path)
        if expected is not None and actual != expected:
            raise RuntimeError("artifact SHA drift: " + path.name)
        files[str(path)] = actual
    for name in ("final.json", "selection_freeze.json", "input_authority.json"):
        bind(formal / name)
    ready = {}
    for arm in ARMS:
        p = formal / "barrier" / (arm + ".ready.json")
        ready[arm] = read(p)
        bind(p)
        complete_path = formal / "workers" / (arm + "_complete.json")
        complete = read(complete_path)
        bind(complete_path)
        final_path = formal / "workers" / (arm + "_final.json")
        worker = read(final_path)
        bind(final_path)
        if (worker.get("status") != "COMPLETE_POST_FREEZE" or worker.get("arm") != arm
                or final["complete"][arm] != worker
                or complete.get("status") != "EPOCHS_COMPLETE_AWAITING_SUPERVISOR_FREEZE"
                or complete.get("arm") != arm):
            raise RuntimeError("worker completion/final identity drift")
        if complete["identities"] != ready[arm]["identities"] or complete["shared_sha256"] != ready[arm]["shared_sha256"]:
            raise RuntimeError("preupdate/complete identities drift")
        rows = []
        for epoch in range(1, 13):
            p = formal / "workers" / f"{arm}_epoch_{epoch:03d}_metrics.json"
            row = read(p)
            bind(p)
            checkpoint = formal / "checkpoints" / f"{arm}_epoch_{epoch:03d}.pt"
            if (row["checkpoint"] != str(checkpoint) or row["selection"]["n_bins"] != 2908
                    or row["selection"].get("finite") is not True):
                raise RuntimeError("immutable metric path/count/finite drift")
            bind(checkpoint, row["checkpoint_sha256"])
            rows.append(row)
        winner = earliest(rows)
        if complete["epochs"] != rows or complete["selected_epoch"] != winner["epoch"] or complete["selected_ema_r2_float64"] != winner["selection"]["r2_concat_float64"]:
            raise RuntimeError("all-epoch independent selection disagreement")
        for label, slot, row in (("selected", "selected", winner), ("epoch12", "endpoints", rows[-1])):
            selected = freeze[slot][arm]
            if any(selected[k] != row[k] for k in ("epoch", "checkpoint", "checkpoint_sha256")) or selected["ema_r2_float64"] != row["selection"]["r2_concat_float64"]:
                raise RuntimeError("frozen selected/endpoint mismatch")
            report = worker["reports"][label]
            if (report["epoch"] != row["epoch"] or report["checkpoint_sha256"] != row["checkpoint_sha256"]
                    or report["complete"]["n_bins"] != COUNT
                    or report["selection_reproduced"]["n_bins"] != 2908
                    or abs(report["selection_reproduced"]["r2_concat_float64"] - selected["ema_r2_float64"]) > 1e-5):
                raise RuntimeError("strict finalizer report identity drift")
            bind(formal / "exports" / f"{arm}_{label}_plain_ema.pt", report["plain_ema_sha256"])
            bind(formal / "exports" / f"{arm}_{label}_complete_native_float64.npz", report["complete_archive_sha256"])
    if ready["flat"]["identities"] != ready["route"]["identities"] or ready["flat"]["shared_sha256"] != ready["route"]["shared_sha256"]:
        raise RuntimeError("paired preupdate identities differ")
    return {"files": files, "selected": freeze["selected"], "final": final}


def require_same_files(files):
    post = {name: sha(Path(name)) for name in files}
    if post != files:
        raise RuntimeError("post-replay artifact/code/source drift")
    return post


def code_source_audit(formal):
    from tfpd_exploration.src.h1_family_v1 import familyformal_split_train as trainer
    from tfpd_exploration.src.h1_optimized_v2.cache import CACHE, ROOT as H1_ROOT
    frozen = trainer.require_frozen_inputs(formal)
    if not CACHE.is_file() or not (H1_ROOT / "source_cache_authority.json").is_file():
        raise RuntimeError("existing frozen cache only; never rebuild")
    names = ("h1_causal.py", "h1_lifted_frontend.py", "h1_replay_contract.py", "linear_conv.py", "grouped_value.py")
    paths = [Path(__file__), CACHE, H1_ROOT / "source_cache_authority.json"]
    paths += [Path(__file__).with_name(name) for name in names]
    return {"frozen": frozen, "files": {str(path): sha(path) for path in paths}}


def metric(arrays):
    p, y, sid = arrays["prediction"].astype(np.float64), arrays["target"].astype(np.float64), arrays["session_id"]
    def r2(pred, target):
        denominator = np.square(target - target.mean(0)).sum()
        if denominator <= 0:
            raise RuntimeError("zero-variance score")
        return float(1 - np.square(pred - target).sum() / denominator)
    per = {name: r2(p[sid == name], y[sid == name]) for name in sorted(set(sid.tolist()))}
    return {"n_bins": len(p), "r2_concat_float64": r2(p, y),
            "equal_session_mean_r2_float64": float(np.mean(list(per.values()))),
            "worst_session_r2_float64": min(per.values()), "per_session_r2_float64": per}


def validate_archive(path):
    with np.load(path, allow_pickle=False) as z:
        arrays = {name: z[name] for name in z.files}
    if set(arrays) != {"prediction", "target", "session_id", "end"}:
        raise RuntimeError("archive fields drift")
    if (arrays["prediction"].shape != (COUNT, 7) or arrays["target"].shape != (COUNT, 7)
            or arrays["end"].shape != (COUNT,) or arrays["session_id"].shape != (COUNT,)
            or arrays["prediction"].dtype != np.float64 or arrays["target"].dtype != np.float64
            or arrays["end"].dtype != np.int64 or arrays["session_id"].dtype.kind != "U"
            or not np.isfinite(arrays["prediction"]).all() or not np.isfinite(arrays["target"]).all()):
        raise RuntimeError("native complete archive shape/dtype/finite drift")
    return arrays


def check_metadata(archive, cache):
    expected = {"target": [], "session_id": [], "end": []}
    if len(cache["minival"]) != 13:
        raise RuntimeError("exact 13 minival sessions required")
    for session, row in sorted(cache["minival"].items()):
        ends = np.flatnonzero(row["eval_mask"])
        expected["target"].append(row["velocity"][ends].astype(np.float64))
        expected["session_id"].append(np.asarray([session] * len(ends)))
        expected["end"].append(ends.astype(np.int64))
    for key in expected:
        if not np.array_equal(np.concatenate(expected[key]), archive[key]):
            raise RuntimeError("source/archive exact metadata mismatch: " + key)


def run(formal, out, threads=2):
    if out.exists():
        raise FileExistsError(out)
    audit = artifact_audit(formal)
    if os.environ.get("H1_COMPLETE_PROOF_GO") != "1" or os.environ.get("CUDA_VISIBLE_DEVICES") not in ("", "-1"):
        raise RuntimeError("explicit GO and CPU-only environment required")
    import torch
    if torch.cuda.is_available() or threads not in (1, 2):
        raise RuntimeError("CPU-only one/two-thread proof required")
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    source = code_source_audit(formal)
    from tfpd_exploration.src.h1_optimized_v2.cache import CACHE, ROOT as H1_ROOT, validate_authority
    from tfpd_exploration.src.h1_family_v1.model import make_v2_unscaled_dot_localbalanced_pair
    from tfpd_exploration.src.two_mainlines_long_v1.decoder.h1_temporal import H1Bank
    from .h1_causal import H1CausalRuntime
    # Direct load: unlike build_or_load, this cannot rebuild any source artifact.
    cache = torch.load(CACHE, map_location="cpu", weights_only=False)
    validate_authority(cache, read(H1_ROOT / "source_cache_authority.json"))
    archives = {}
    for arm in ARMS:
        for label in ("selected", "epoch12"):
            a = validate_archive(formal / "exports" / f"{arm}_{label}_complete_native_float64.npz")
            check_metadata(a, cache)
            actual, reported = metric(a), audit["final"]["complete"][arm]["reports"][label]["complete"]
            for name in ("r2_concat_float64", "equal_session_mean_r2_float64", "worst_session_r2_float64"):
                if not np.isfinite(reported[name]) or abs(actual[name] - reported[name]) > 1e-10:
                    raise RuntimeError("native archive metric reproduction drift")
            if label == "selected":
                archives[arm] = a
    out.mkdir(parents=True)
    atomic_json({"status": "RUNNING_SELECTED_ONLY", "audit": audit, "source": source}, out / "input_authority.json")
    started, results, predictions = time.monotonic(), {}, {}
    for arm in ARMS:
        pair = make_v2_unscaled_dot_localbalanced_pair(seed=42)
        model = pair[0 if arm == "flat" else 1]
        del pair
        state = torch.load(formal / "exports" / f"{arm}_selected_plain_ema.pt", map_location="cpu", weights_only=True)
        if any(t.dtype != torch.float32 or not torch.isfinite(t).all() for t in state.values()):
            raise RuntimeError("plain EMA finite FP32 contract")
        model.load_state_dict(state, strict=True)
        model.eval()
        offset, rows, pieces = 0, {}, []
        for session, row in sorted(cache["minival"].items()):
            bank = H1Bank(*[row["bank"][name] for name in ("E0", "T", "unit_mask")])
            runtime = H1CausalRuntime(model, bank, batch_size=1)
            count = int(row["eval_mask"].sum())
            a = {key: value[offset:offset + count] for key, value in archives[arm].items()}
            def direct(raw):
                with torch.no_grad():
                    return (model.forward_last(torch.from_numpy(raw[None]), bank) / 20).numpy().copy()
            result = replay_session(runtime, row["neural"], row["velocity"], row["eval_mask"], session, a, direct)
            pieces.append(result.pop("prediction"))
            result["state_bytes"] = dataclasses.asdict(runtime.state_bytes())
            rows[session] = result
            offset += count
            atomic_json({"status": "RUNNING", "arm": arm, "session": session, "completed_scored_bins": offset,
                         "elapsed_seconds": time.monotonic() - started}, out / "live.json")
        if offset != COUNT or sum(r["public_calls"] for r in rows.values()) != 20920:
            raise RuntimeError("complete 20325 endpoint / 20920 public-bin cardinality drift")
        arrays = {**archives[arm], "prediction": np.concatenate(pieces).astype(np.float64)}
        observed = metric(arrays)
        reported = metric(archives[arm])
        for name in ("r2_concat_float64", "equal_session_mean_r2_float64", "worst_session_r2_float64"):
            if abs(observed[name] - reported[name]) > 1e-5:
                raise RuntimeError("public/native score drift")
        predictions[arm] = arrays
        results[arm] = {"selected": audit["selected"][arm], "sessions": rows, "native_metric": observed,
                        "public_calls": sum(r["public_calls"] for r in rows.values()), "scored_count": offset,
                        "max_archive_abs_error": max(r["max_abs_error"] for r in rows.values()),
                        "max_native_subset_abs_error": max(r["max_native_abs_error"] for r in rows.values())}
    post_artifact = artifact_audit(formal)
    post_source = code_source_audit(formal)
    if post_artifact != audit or post_source != source:
        raise RuntimeError("fresh post-replay full authority drift")
    require_same_files(audit["files"])
    require_same_files(source["files"])
    for arm, arrays in predictions.items():
        path = out / f"{arm}_selected_public_native_float64.npz"
        with tempfile.NamedTemporaryFile(dir=out, suffix=".npz", delete=False) as handle:
            temporary = Path(handle.name)
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
        results[arm]["archive_sha256"] = sha(path)
    result = {"status": "PASS_IMPLEMENTATION_EQUIVALENCE_ONLY", "schema": "h1_selected_family_complete_stream_proof_v1",
              "scope": "known-source minival, selected FLAT/ROUTE, all chronological bins and 20325 endpoints each; no selection",
              "threads": threads, "batch": 1, "pre_artifact": audit, "post_artifact": post_artifact,
              "pre_source": source, "post_source": post_source, "arms": results,
              "elapsed_seconds": time.monotonic() - started, "parameter_updates": 0}
    atomic_json(result, out / "receipt.json")
    print(json.dumps({"status": result["status"], "elapsed_seconds": result["elapsed_seconds"],
                      "arms": {a: {k: v for k, v in r.items() if k != "sessions"} for a, r in results.items()}}))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    run(args.formal, args.out, args.threads)
