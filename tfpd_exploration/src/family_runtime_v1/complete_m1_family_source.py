"""Post-finalization P1 FLAT/ROUTE proof on the entire frozen source tail.

This is an implementation-equivalence check, not a new selection or a timing
benchmark. Every chronological bin between the first and last scored endpoint
is consumed, including gaps. The first W100 history is explicitly primed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np

ARMS = ("flat", "route")
COUNT = 31252
SESSIONS = ("ses-20120926", "ses-20120927", "ses-20120928")
COUNTS = (10567, 9705, 10980)
PUBLIC_CALLS = (34142, 44462, 34381)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def typed_array_sha(value):
    array = np.ascontiguousarray(value)
    header = json.dumps({"dtype": str(array.dtype), "shape": list(array.shape)},
                        sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(header + array.tobytes()).hexdigest()


def split_sha(pairs):
    digest = hashlib.sha256()
    for name, start in pairs:
        digest.update(name.encode())
        digest.update(np.asarray([start], dtype=np.int64).tobytes())
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def atomic_json(body, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, mode="w", delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(body, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def earliest(history, arm):
    if len(history) != 24 or [r["epoch"] for r in history] != list(range(1, 25)):
        raise RuntimeError("all contiguous 24 epochs required")
    values = [r[arm]["ema"]["equal_session_mean_r2"] for r in history]
    if not np.isfinite(values).all():
        raise RuntimeError("nonfinite EMA selection")
    return history[int(np.argmax(values))][arm]


def artifact_audit(run_root):
    # No torch/model/source imports before the completed-run guard.
    meta = read(run_root / "run_meta.json")
    if meta.get("status") != "COMPLETE" or meta.get("completed_epochs") != 24:
        raise RuntimeError("requires completed fixed-24 P1 training")
    final = read(run_root / "finalized_p1" / "receipt.json")
    if final.get("status") != "SOURCE_MINIVAL_ONLY" or final.get("outer_query_opened") is not False:
        raise RuntimeError("requires completed source-only finalizer")
    report, history = read(run_root / "report.json"), read(run_root / "epoch_metrics.json")
    freeze = read(run_root / "selection_freeze.json")
    frozen = final["freeze_pre_inference"]
    if (freeze.get("status") != "FROZEN_PRE_INFERENCE" or freeze.get("outer_query_opened") is not False
            or freeze["freeze_pre_inference"] != frozen or meta.get("outer_query_opened") is not False
            or report.get("outer_query_opened") is not False or report.get("status") != "SOURCE_MINIVAL_ONLY"
            or meta["recipe"]["epochs"] != 24 or report["history"] != history
            or any(report[key] != meta[key] for key in ("recipe", "provenance", "split"))
            or frozen["source_closure"] != meta["provenance"]):
        raise RuntimeError("completed run/finalizer identity disagreement")
    files = {}
    def bind(path, expected=None):
        value = sha(path)
        if expected is not None and value != expected:
            raise RuntimeError("artifact SHA drift: " + str(path))
        files[str(path)] = value
    for name, key in (("run_meta.json", "meta_sha256"), ("report.json", "report_sha256"),
                      ("epoch_metrics.json", "history_sha256")):
        bind(run_root / name, frozen[key])
    bind(run_root / "selection_freeze.json")
    bind(run_root / "finalized_p1" / "receipt.json")
    selected = {}
    for arm in ARMS:
        selected[arm] = earliest(history, arm)
        for row in history:
            record = row[arm]
            path = run_root / arm / f"epoch_{row['epoch']:03d}.pt"
            if (record["checkpoint"] != str(path) or record["epoch"] != row["epoch"]
                    or record["ema"].get("n") != COUNT
                    or record["ema"].get("complete_all_sessions") is not True):
                raise RuntimeError("all-epoch checkpoint/count identity drift")
            bind(path, record["checkpoint_sha256"])
        for label, record in (("selected", selected[arm]), ("endpoint24", history[-1][arm])):
            slot = "selected_primary_ema" if label == "selected" else "endpoint24"
            item = final["exports"][arm + "_" + label]
            if (report[slot][arm] != record or frozen[label][arm] != record or item["record"] != record
                    or frozen["checkpoint_sha256"][arm + "_" + label] != record["checkpoint_sha256"]):
                raise RuntimeError("independent selected/endpoint disagreement")
            bind(run_root / "finalized_p1" / f"{arm}_{label}_ema_state.pt", item["state_sha256"])
            bind(run_root / "finalized_p1" / f"{arm}_{label}_native_source_dev.npz", item["prediction_sha256"])
    return {"files": files, "meta": meta, "selected": selected, "final": final}


def require_same_files(files):
    actual = {name: sha(name) for name in files}
    if actual != files:
        raise RuntimeError("artifact/source/code changed during proof")
    return actual


def source_audit(audit):
    from tfpd_exploration.src.m1_optimized_v2 import plan
    from tfpd_exploration.src.m1_family_v1.finalize_p1 import _live_closure
    final, meta = audit["final"], audit["meta"]
    live = _live_closure()
    if live != final["freeze_pre_inference"]["live_code_closure"]:
        raise RuntimeError("finalizer live code closure drift")
    files = {}
    def bind(path, expected=None):
        value = sha(path)
        if expected is not None and value != expected:
            raise RuntimeError("source/code SHA drift: " + str(path))
        files[str(path)] = value
    for relative, expected in meta["provenance"]["modelcode_sha256"].items():
        bind(plan.REPO_ROOT / relative, expected)
    for row in meta["provenance"]["source_files"].values():
        bind(Path(row["path"]), row["sha256"])
    bind(plan.BANK_NPZ, meta["provenance"]["bank_npz_sha256"])
    bind(plan.BANK_RECEIPT, meta["provenance"]["bank_receipt_sha256"])
    bind(plan.S_FIX_PATH, plan.S_FIX_SHA256)
    cache = plan.RESULT_ROOT / "m1_optimized_v2_source_runtime_cache.npz"
    provenance = plan.RESULT_ROOT / "rSyn3-refit-v1.source-only.provenance-supplement.npz"
    cache_receipt, provenance_receipt = read(cache.with_suffix(".receipt.json")), read(provenance.with_suffix(".receipt.json"))
    for receipt in (cache_receipt, provenance_receipt):
        if (receipt.get("outer_query_opened") is not False or receipt.get("outer_path_resolved") is not False
                or receipt["source_sessions"] != list(SESSIONS)
                or receipt["carrier_npz_sha256"] != meta["provenance"]["bank_npz_sha256"]):
            raise RuntimeError("runtime bank source-only authority drift")
    bind(cache, cache_receipt["npz_sha256"])
    bind(provenance, provenance_receipt["supplement_npz_sha256"])
    bind(cache.with_suffix(".receipt.json"))
    bind(provenance.with_suffix(".receipt.json"))
    names = ("m1_replay_contract.py", "m1.py", "m1_lifted.py", "m1_route_lifted.py", "repair.py", "lifted_attention.py")
    for path in [Path(__file__), *[Path(__file__).with_name(n) for n in names],
                 Path(__file__).parents[1] / "m1_runtime_v3" / "runtime.py"]:
        bind(path)
    return {"files": files, "live_closure": live, "cache": str(cache), "provenance": str(provenance),
            "cache_receipt": cache_receipt, "provenance_receipt": provenance_receipt}


def metric(arrays):
    p, y, s = arrays["prediction"].astype(np.float64), arrays["target"].astype(np.float64), arrays["session"]
    def r2(pred, target):
        denominator = np.square(target - target.mean(0)).sum()
        if not np.isfinite(denominator) or denominator <= 0:
            raise RuntimeError("invalid source target variance")
        return float(1 - np.square(pred - target).sum() / denominator)
    per = {name: r2(p[s == name], y[s == name]) for name in SESSIONS}
    return {"pooled_r2": r2(p, y), "equal_session_mean_r2": float(np.mean(list(per.values()))), "per_session_r2": per}


def archive_audit(path, item):
    with np.load(path, allow_pickle=False) as z:
        arrays = {key: z[key] for key in z.files}
    if (set(arrays) != {"prediction", "target", "session", "start"}
            or arrays["prediction"].shape != (COUNT, 16) or arrays["target"].shape != (COUNT, 16)
            or arrays["session"].shape != (COUNT,) or arrays["start"].shape != (COUNT,)
            or arrays["prediction"].dtype != np.float32 or arrays["target"].dtype != np.float32
            or arrays["session"].dtype.kind != "U" or arrays["start"].dtype != np.int64
            or not np.isfinite(arrays["prediction"]).all() or not np.isfinite(arrays["target"]).all()
            or {key: typed_array_sha(value) for key, value in arrays.items()} != item["array_sha256"]):
        raise RuntimeError("native source archive shape/dtype/hash drift")
    observed = metric(arrays)
    require_metrics(observed, item["metrics"], tolerance=1e-10)
    return arrays


def require_metrics(actual, expected, tolerance=1e-5):
    for key in ("pooled_r2", "equal_session_mean_r2"):
        if not np.isfinite(expected[key]) or abs(actual[key] - expected[key]) > tolerance:
            raise RuntimeError("native/public R2 drift: " + key)
    for name in SESSIONS:
        if (not np.isfinite(expected["per_session_r2"][name])
                or abs(actual["per_session_r2"][name] - expected["per_session_r2"][name]) > tolerance):
            raise RuntimeError("native/public session R2 drift")


def metadata_audit(archives, dev, banks, source, audit):
    """Bridge old runtime-cache inputs to the actual P1 native source inputs."""
    rows = {}
    with np.load(source["cache"], allow_pickle=False) as cache, np.load(source["provenance"], allow_pickle=False) as provenance:
        offset = 0
        for name, count, calls in zip(SESSIONS, COUNTS, PUBLIC_CALLS, strict=True):
            def normal_name(value):
                return value.decode() if isinstance(value, bytes) else str(value)
            pairs = [(normal_name(n), int(start)) for n, start in dev.base.window_indices if normal_name(n) == name]
            starts = np.asarray([start for _, start in pairs], dtype=np.int64)
            if (len(starts) != count or not np.all(np.diff(starts) > 0)
                    or int(starts[-1] - starts[0]) != calls
                    or split_sha(pairs) != audit["meta"]["split"][name]["dev_window_starts_sha256"]):
                raise RuntimeError("frozen source endpoint topology drift")
            raw = np.ascontiguousarray(cache[f"raw_neural/{name}"], dtype=np.float32)
            original = np.ascontiguousarray(dev.base.neural_data[name], dtype=np.float32)
            if (not np.array_equal(raw, original) or raw.shape[1:] != (64,) or not np.all(raw[:99] == 0)
                    or not np.isfinite(raw).all()
                    or array_sha(raw) != source["cache_receipt"]["arrays"][name]["raw_neural_sha256"]):
                raise RuntimeError("raw cache/native source mismatch")
            target = np.ascontiguousarray(dev.base.covariate_data[name][starts + 99], dtype=np.float32)
            for arrays in archives.values():
                if (not np.array_equal(arrays["start"][offset:offset+count], starts)
                        or not np.all(arrays["session"][offset:offset+count] == name)
                        or not np.array_equal(arrays["target"][offset:offset+count], target)):
                    raise RuntimeError("native source exact target/session/start mismatch")
            values = {}
            for key, attr in (("e0", "E0"), ("t", "T"), ("unit_mask", "unit_mask")):
                value = cache[f"bank_{key}/{name}"]
                expected = getattr(banks[name], attr).detach().cpu().numpy()
                if not np.array_equal(value, expected):
                    raise RuntimeError("runtime/native P1 bank mismatch: " + key)
                values[key] = value.copy()
            roster = provenance[f"nwb_unit_ids_in_rate_column_order/{name}"].copy()
            if (roster.shape != (64,) or len(set(roster.tolist())) != 64
                    or array_sha(roster) != source["provenance_receipt"]["rows"][name]["nwb_unit_ids_sha256"]
                    or source["provenance_receipt"]["rows"][name]["source_file_sha256"] != audit["meta"]["provenance"]["source_files"][name]["sha256"]):
                raise RuntimeError("physical unit roster mismatch")
            rows[name] = {"raw": raw, "target": target, "bank": values, "roster": roster,
                          "slice": (offset, offset + count)}
            offset += count
    if offset != COUNT:
        raise RuntimeError("complete source cardinality drift")
    return rows


def run(run_root, out, threads=2):
    if out.exists():
        raise FileExistsError(out)
    audit = artifact_audit(run_root)
    if os.environ.get("M1_COMPLETE_PROOF_GO") != "1" or os.environ.get("CUDA_VISIBLE_DEVICES") not in ("", "-1"):
        raise RuntimeError("explicit GO and CPU-only environment required")
    import torch
    if torch.cuda.is_available() or threads not in (1, 2):
        raise RuntimeError("CPU one/two-thread proof only")
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    source = source_audit(audit)
    from tfpd_exploration.src.m1_family_v1.finalize_p1 import _verify_source_closure
    from tfpd_exploration.src.m1_family_v1.family_train_v2 import _p1_models
    from tfpd_exploration.src.m1_optimized_v2.data import materialize_source_banks
    from tfpd_exploration.src.m1_runtime_v3.runtime import BankBatch
    from .m1_lifted import LiftedFiveTokenCurrentQueryStream
    from .m1_route_lifted import RouteLiftedFiveTokenCurrentQueryStream
    from .m1_replay_contract import replay_session
    dev, banks = _verify_source_closure(audit["meta"]), materialize_source_banks()
    all_archives = {f"{arm}_{label}": archive_audit(run_root / "finalized_p1" / f"{arm}_{label}_native_source_dev.npz",
                    audit["final"]["exports"][f"{arm}_{label}"]) for arm in ARMS for label in ("selected", "endpoint24")}
    rows = metadata_audit(all_archives, dev, banks, source, audit)
    del dev, banks
    require_same_files(source["files"])
    out.mkdir(parents=True)
    atomic_json({"status": "RUNNING_SELECTED_ONLY", "audit": audit, "source": source}, out / "input_authority.json")
    started, results, predictions = time.monotonic(), {}, {}
    for arm, stream_type in zip(ARMS, (LiftedFiveTokenCurrentQueryStream, RouteLiftedFiveTokenCurrentQueryStream), strict=True):
        pair = _p1_models(torch.device("cpu"))
        model = pair[0 if arm == "flat" else 1]
        del pair
        state = torch.load(run_root / "finalized_p1" / f"{arm}_selected_ema_state.pt", map_location="cpu", weights_only=True)
        if any(value.is_floating_point() and (value.dtype != torch.float32 or not torch.isfinite(value).all()) for value in state.values()):
            raise RuntimeError("finite FP32 selected EMA required")
        model.load_state_dict(state, strict=True)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        checks, pieces = {}, []
        archive = all_archives[arm + "_selected"]
        for name, row in rows.items():
            e0, t, mask = [torch.from_numpy(row["bank"][key])[None] for key in ("e0", "t", "unit_mask")]
            bank = BankBatch(e0, t, mask, (name,), (tuple(row["roster"].tolist()),))
            runtime = stream_type(model, bank)
            low, high = row["slice"]
            section = {key: value[low:high] for key, value in archive.items()}
            def initialize(history):
                runtime.refresh_state(history=torch.from_numpy(history))
            def current():
                return runtime.current_prediction().detach().cpu().numpy().copy()
            def direct(history):
                with torch.no_grad():
                    return model.forward_last(torch.from_numpy(history), bank).numpy().copy()
            def progress(body):
                atomic_json({"status": "RUNNING", "arm": arm, "session": name,
                             "elapsed_seconds": time.monotonic()-started, **body}, out / "live.json")
            result = replay_session(runtime, row["raw"], name, section, direct,
                                    native_targets=row["target"], initialize_history=initialize,
                                    current_prediction=current, progress=progress)
            pieces.append(result.pop("prediction"))
            result["state_bytes"] = runtime.state_bytes
            result["physical_unit_roster_sha256"] = array_sha(row["roster"])
            checks[name] = result
        arrays = {**archive, "prediction": np.concatenate(pieces).astype(np.float32)}
        actual, expected = metric(arrays), metric(archive)
        require_metrics(actual, expected)
        if (sum(r["public_calls"] for r in checks.values()) != sum(PUBLIC_CALLS)
                or sum(r["scored_count"] for r in checks.values()) != COUNT):
            raise RuntimeError("complete chronological replay cardinality drift")
        predictions[arm] = arrays
        results[arm] = {"selected": audit["selected"][arm], "sessions": checks, "metrics": actual,
                        "public_calls": sum(PUBLIC_CALLS), "initial_current_predictions": 3, "scored_count": COUNT,
                        "max_abs_error": float(np.abs(arrays["prediction"] - archive["prediction"]).max())}
    post_artifact, post_source = artifact_audit(run_root), source_audit(audit)
    if post_artifact != audit or post_source != source:
        raise RuntimeError("fresh full post-replay authority drift")
    require_same_files(audit["files"])
    require_same_files(source["files"])
    for arm, arrays in predictions.items():
        path = out / f"{arm}_selected_public_native.npz"
        with tempfile.NamedTemporaryFile(dir=out, suffix=".npz", delete=False) as handle:
            temporary = Path(handle.name)
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
        results[arm]["archive_sha256"] = sha(path)
    body = {"schema": "m1_p1_family_selected_complete_source_stream_v1", "status": "PASS_IMPLEMENTATION_EQUIVALENCE_ONLY",
            "scope": "known-source chron80 selected FLAT/ROUTE; not new quality/latency acceptance",
            "startup_scope": "one true W100 history primed at each session's first dev endpoint; all subsequent bins including gaps",
            "threads": threads, "batch": 1, "parameter_updates": 0, "outer_query_opened": False,
            "pre_artifact": audit, "post_artifact": post_artifact, "pre_source": source, "post_source": post_source,
            "arms": results, "elapsed_seconds": time.monotonic()-started}
    atomic_json(body, out / "receipt.json")
    print(json.dumps({"status": body["status"], "elapsed_seconds": body["elapsed_seconds"],
                      "arms": {arm: {key: value for key, value in result.items() if key != "sessions"} for arm, result in results.items()}}), flush=True)
    return body


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    run(args.run_root, args.out, args.threads)
