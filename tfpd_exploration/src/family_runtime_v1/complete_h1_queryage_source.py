"""CPU-only selected-QueryAge public-stream equivalence proof.

This is an additive post-freeze consumer.  It never trains, chooses a model,
or rebuilds the immutable H1 source cache.  The command is deliberately
separate from the formal launcher: an external authorization binds both the
completed formal receipt and this proof's own code/source closure.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

ARMS = ("flat", "route")
COUNT, SESSIONS, W, UNITS, OUT = 20325, 13, 700, 176, 7
EPOCHS, SELECTION_COUNT = 12, 2908
GO, WALL_SECONDS, RSS_LIMIT = "H1_QUERYAGE_COMPLETE_PROOF_GO", 21600, 22 << 30
ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "results/decoder_validation_v2/20260905_190000/h1/source_cache.pt"
CACHE_AUTHORITY = CACHE.with_name("source_cache_authority.json")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict): raise RuntimeError("JSON object required")
    return value


def atomic_json(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name); json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False); handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
    try: os.replace(temporary, path)
    finally: temporary.unlink(missing_ok=True)


def _canonical(path: Path) -> Path:
    path = Path(path)
    if not path.is_absolute() or path.resolve() != path: raise RuntimeError("canonical absolute path required")
    return path


def _inside(parent: Path, child: Path) -> bool:
    return os.path.commonpath((str(parent), str(child))) == str(parent)


def code_closure() -> dict[str, str]:
    paths = {"consumer": Path(__file__), "stream": ROOT / "src/two_mainlines_long_v1/current_query_v2/streaming.py",
             "query_core": ROOT / "src/two_mainlines_long_v1/current_query_v2/core.py", "frontend_cache": ROOT / "src/two_mainlines_long_v1/latency_opt_v2/exact_window.py", "frontend_cache_init": ROOT / "src/two_mainlines_long_v1/latency_opt_v2/__init__.py",
             "query_model": ROOT / "src/h1_queryage_family_v1/model.py",
             "h1_model": ROOT / "src/h1_family_v1/model.py", "temporal": ROOT / "src/two_mainlines_long_v1/decoder/h1_temporal.py",
             "h1_config": ROOT / "src/two_mainlines_long_v1/decoder/h1_config.py", "cache_validator": ROOT / "src/h1_optimized_v2/cache.py",
             "query_package": ROOT / "src/two_mainlines_long_v1/current_query_v2/__init__.py",
             "formal_launcher": ROOT / "src/h1_queryage_family_v1/formal_prefix_launcher.py",
             "formal_train": ROOT / "src/h1_queryage_family_v1/formal_prefix_train.py"}
    return {name: sha(path) for name, path in paths.items()}


def _archive(path: Path, *, count: int | None = None) -> dict[str, np.ndarray]:
    count = COUNT if count is None else count
    if count <= 0: raise RuntimeError("complete archive must have nonempty endpoint topology")
    with np.load(path, allow_pickle=False) as z: value = {name: z[name] for name in z.files}
    if set(value) != {"prediction", "target", "session_id", "end"}: raise RuntimeError("complete archive fields drift")
    p, y, sid, end = value["prediction"], value["target"], value["session_id"], value["end"]
    if (p.shape != (count, OUT) or y.shape != (count, OUT) or end.shape != (count,) or sid.shape != (count,)
            or p.dtype != np.float64 or y.dtype != np.float64 or end.dtype != np.int64 or sid.dtype.kind != "U"
            or not np.isfinite(p).all() or not np.isfinite(y).all()
            or np.any(end < 0) or any(not name for name in sid.tolist())
            or len(set(zip(sid.tolist(), end.tolist()))) != count): raise RuntimeError("complete archive geometry/dtype/finite drift")
    return value


def _metric(value: dict[str, np.ndarray]) -> dict[str, Any]:
    p, y, sid = value["prediction"], value["target"], value["session_id"]
    if not len(p): raise RuntimeError("native metric requires nonempty endpoint topology")
    def r2(a, b):
        denom = float(np.square(b - b.mean(0, keepdims=True)).sum())
        if not np.isfinite(denom) or denom <= 0: raise RuntimeError("nonpositive native target variance")
        return float(1 - np.square(a - b).sum() / denom)
    per = {name: r2(p[sid == name], y[sid == name]) for name in sorted(set(sid.tolist()))}
    return {"n_bins": int(len(p)), "r2_concat_float64": r2(p, y), "equal_session_mean_r2_float64": float(np.mean(list(per.values()))), "worst_session": min(per, key=per.get), "worst_session_r2_float64": float(min(per.values())), "per_session_r2_float64": per}


def _compare_metrics(actual, expected, *, tolerance: float) -> None:
    if actual["n_bins"] != expected.get("n_bins") or set(actual["per_session_r2_float64"]) != set(expected.get("per_session_r2_float64", {})):
        raise RuntimeError("native metric endpoint/session topology drift")
    pairs = [(actual[key], expected.get(key)) for key in ("r2_concat_float64", "equal_session_mean_r2_float64", "worst_session_r2_float64")]
    pairs += [(value, expected["per_session_r2_float64"][name]) for name, value in actual["per_session_r2_float64"].items()]
    if any(not isinstance(b, (int, float)) or not np.isfinite(b) or abs(a - b) > tolerance for a, b in pairs):
        raise RuntimeError("native FP64 aggregate/per-session metric drift")
    if tolerance == 0 and actual["worst_session"] != expected.get("worst_session"):
        raise RuntimeError("native FP64 worst-session identity drift")


def _public(value: np.ndarray) -> np.ndarray:
    if (not isinstance(value, np.ndarray) or value.shape != (1, OUT) or value.dtype != np.float32
            or not value.flags.owndata or not value.flags.c_contiguous or not np.isfinite(value).all()):
        raise RuntimeError("public output must be owned contiguous finite FP32 [1,7]")
    return value


def _fresh_formal_authority(formal: Path, recorded: dict[str, Any], *, allow_cpu_fixture=False) -> None:
    # A reduced fixture may replace the original admission inputs, never the CLI.
    if allow_cpu_fixture:
        for name, expected in recorded.get("bindings", {}).get("inputs", {}).items():
            if sha(_canonical(Path(name))) != expected: raise RuntimeError("fixture source closure drift")
        return
    if (recorded.get("status") != "ROOT_REVIEW_GO" or recorded.get("mode") != "formal"
            or recorded.get("bindings", {}).get("output") != str(formal)):
        raise RuntimeError("original formal admission identity required")
    from tfpd_exploration.src.h1_queryage_family_v1 import formal_prefix_launcher as launcher
    capacity, smoke = recorded.get("capacity_audit", {}), recorded.get("smoke_audit", {})
    current = launcher.collect_formal_authority(
        formal, _canonical(Path(capacity.get("receipt", ""))),
        _canonical(Path(capacity.get("checkpoint", ""))), _canonical(Path(smoke.get("receipt", ""))))
    if current != recorded: raise RuntimeError("fresh original formal source/code/capacity/smoke authority drift")


def _state_digest(model) -> str:
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        a = value.detach().cpu().contiguous().numpy(); h.update(name.encode()); h.update(str(a.dtype).encode()); h.update(np.asarray(a.shape, dtype=np.int64).tobytes()); h.update(a.tobytes())
    return h.hexdigest()


def collect_bindings(formal: Path, output: Path) -> dict[str, Any]:
    formal, output = _canonical(formal), _canonical(output)
    receipt, freeze, input_authority = formal / "receipt.json", formal / "selection_freeze.json", formal / "input_authority.json"
    files = [receipt, freeze, input_authority, CACHE, CACHE_AUTHORITY]
    for arm in ARMS:
        files += [formal / "workers" / f"{arm}_final.json", formal / "exports" / f"{arm}_selected_plain_ema.pt", formal / "exports" / f"{arm}_selected_complete_native_float64.npz"]
    if any(not path.is_file() for path in files): raise FileNotFoundError("required completed-formal/source artifact missing")
    return {"schema": "h1_queryage_selected_public_proof_v1", "formal": str(formal), "output": str(output),
            "inputs": {str(path): sha(path) for path in files}, "code_closure": code_closure(),
            "cpu": {"threads": 1, "interop_threads": 1, "cuda_visible_devices": ""},
            "surface": {"sessions": SESSIONS, "complete_endpoints_per_arm": COUNT, "window": W, "units": UNITS, "native_outputs": OUT},
            "limits": {"wall_seconds": WALL_SECONDS, "peak_rss_bytes": RSS_LIMIT}}


def _audit_formal(formal: Path, bindings: dict[str, Any], *, allow_cpu_fixture=False, guard=lambda: None) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]]]:
    guard()
    receipt, freeze = read(formal / "receipt.json"), read(formal / "selection_freeze.json")
    if (receipt.get("schema") != "h1_queryage_formal_prefix_complete_pair_v1" or receipt.get("status") != "COMPLETE_FIXED_FORMAL_NO_PROMOTION"
            or receipt.get("selection_freeze") != freeze or receipt.get("selection_freeze_sha256") != sha(formal / "selection_freeze.json")
            or receipt.get("authority") != receipt.get("authority_post") or receipt.get("owned_artifact_sha256") != {str(p): sha(_canonical(p)) for p in sorted(formal.rglob("*")) if p.is_file() and p != formal / "receipt.json"}): raise RuntimeError("formal completion/authority/manifest is incomplete or drifted")
    guard(); _fresh_formal_authority(formal, receipt["authority"], allow_cpu_fixture=allow_cpu_fixture); guard()
    external = _canonical(Path(receipt.get("authorization_path", "")))
    if _inside(formal, external) or sha(external) != receipt.get("authorization_sha256") or read(external) != receipt.get("authority"):
        raise RuntimeError("formal external root authorization drift")
    input_authority = read(formal / "input_authority.json")
    if input_authority.get("authority") != receipt["authority"] or input_authority.get("bindings") != receipt["authority"].get("bindings") or sha(formal / "input_authority.json") != receipt.get("input_authority_sha256"): raise RuntimeError("formal input authority identity drift")
    if not allow_cpu_fixture and (input_authority.get("authorization_path") != str(external)
            or input_authority.get("authorization_sha256") != receipt["authorization_sha256"]
            or input_authority.get("capacity_audit") != receipt["authority"].get("capacity_audit")):
        raise RuntimeError("original formal input external-authority link drift")
    if (freeze.get("schema") != "h1_queryage_formal_prefix_selection_freeze_v1"
            or set(freeze.get("selected", {})) != set(ARMS) or set(freeze.get("epoch12", {})) != set(ARMS)):
        raise RuntimeError("selected freeze topology drift")
    archives = {}
    readiness = {arm: read(formal / "barrier" / f"{arm}.ready.json") for arm in ARMS}
    if not allow_cpu_fixture:
        from tfpd_exploration.src.h1_queryage_family_v1.formal_prefix_launcher import validate_ready
        validate_ready(readiness)
    for arm in ARMS:
        final = read(formal / "workers" / f"{arm}_final.json"); selected = freeze["selected"][arm]; report = final.get("reports", {}).get("selected", {})
        plain, archive = formal / "exports" / f"{arm}_selected_plain_ema.pt", formal / "exports" / f"{arm}_selected_complete_native_float64.npz"
        complete = read(formal / "workers" / f"{arm}_complete.json"); rows = complete.get("epochs", [])
        if (len(rows) != EPOCHS or [row.get("epoch") for row in rows] != list(range(1, EPOCHS + 1)) or complete.get("arm") != arm
                or complete.get("identities") != readiness[arm].get("identities")
                or final != receipt.get("finals", {}).get(arm) or final.get("status") != "COMPLETE_POST_FREEZE" or final.get("arm") != arm or report.get("epoch") != selected.get("epoch")
                or report.get("checkpoint_sha256") != selected.get("checkpoint_sha256") or report.get("plain_ema_path") != str(plain)
                or report.get("plain_ema_sha256") != sha(plain) or report.get("complete_archive") != str(archive)
                or report.get("complete_archive_sha256") != sha(archive) or report.get("complete", {}).get("n_bins") != COUNT): raise RuntimeError("selected finalizer artifact identity drift")
        pick = max(rows, key=lambda row: row["selection"]["r2_concat_float64"])
        if (pick["epoch"] != complete.get("selected_epoch") or pick["selection"]["r2_concat_float64"] != complete.get("selected_ema_r2_float64")
                or selected != {"epoch": pick["epoch"], "ema_r2_float64": pick["selection"]["r2_concat_float64"], "checkpoint": pick["checkpoint"], "checkpoint_sha256": pick["checkpoint_sha256"]}): raise RuntimeError("all-epoch earliest selected freeze drift")
        last = rows[-1]
        if freeze["epoch12"][arm] != {"epoch": EPOCHS, "ema_r2_float64": last["selection"]["r2_concat_float64"], "checkpoint": last["checkpoint"], "checkpoint_sha256": last["checkpoint_sha256"]}:
            raise RuntimeError("fixed terminal-epoch freeze drift")
        for row in rows:
            guard()
            checkpoint = formal / "checkpoints" / f"{arm}_epoch_{row['epoch']:03d}.pt"
            if row.get("checkpoint") != str(checkpoint) or row.get("checkpoint_sha256") != sha(checkpoint) or read(formal / "workers" / f"{arm}_epoch_{row['epoch']:03d}.json") != row: raise RuntimeError("all-epoch checkpoint record drift")
            score = row.get("selection", {})
            if not np.isfinite(score.get("r2_concat_float64", np.nan)): raise RuntimeError("nonfinite epoch selection score")
            if not allow_cpu_fixture and (score.get("n_bins") != SELECTION_COUNT or score.get("finite") is not True
                    or row.get("identities") != readiness[arm]["identities"][str(row["epoch"])]) :
                raise RuntimeError("all-epoch actual selection/recipe identity drift")
        if not allow_cpu_fixture and complete.get("shared_init_sha256") != readiness[arm].get("shared_init_sha256"):
            raise RuntimeError("shared initialization completion drift")
        archives[arm] = _archive(archive)
        _compare_metrics(_metric(archives[arm]), report.get("complete", {}), tolerance=0)
        guard()
    return receipt, archives


def _load_cache(*, source_loader=None, allow_cpu_fixture=False):
    import torch
    if source_loader is not None:
        if not allow_cpu_fixture: raise RuntimeError("source injection is test-only")
        cache, authority = source_loader()
    else:
        from tfpd_exploration.src.h1_optimized_v2.cache import validate_authority
        cache, authority = torch.load(CACHE, map_location="cpu", weights_only=False), read(CACHE_AUTHORITY); validate_authority(cache, authority)
    if not isinstance(cache, dict) or not isinstance(cache.get("minival"), dict) or len(cache["minival"]) != SESSIONS: raise RuntimeError("exact thirteen-session immutable minival required")
    return cache


def _validate_archive_source(archive, cache):
    expected = {"target": [], "session_id": [], "end": []}
    for name, row in sorted(cache["minival"].items()):
        ends = _source_ends(row)
        expected["target"].append(row["velocity"][ends].astype(np.float64)); expected["session_id"].append(np.full(len(ends), name, dtype="U128")); expected["end"].append(ends)
    for key in expected:
        if not np.array_equal(archive[key], np.concatenate(expected[key])): raise RuntimeError("archive/source endpoint identity drift: " + key)


def _source_ends(row):
    neural, velocity, mask = row.get("neural"), row.get("velocity"), row.get("eval_mask")
    if (not isinstance(neural, np.ndarray) or neural.dtype != np.float32 or neural.ndim != 2 or neural.shape[1] != UNITS
            or not isinstance(velocity, np.ndarray) or velocity.dtype != np.float32 or velocity.shape != (len(neural), OUT)
            or not isinstance(mask, np.ndarray) or mask.shape != (len(neural),) or mask.dtype != np.bool_):
        raise RuntimeError("immutable H1 source geometry drift")
    ends = np.flatnonzero(mask).astype(np.int64)
    if not len(ends) or not np.isfinite(neural).all() or not np.isfinite(velocity[ends]).all():
        raise RuntimeError("nonempty finite source endpoint topology required")
    return ends


def _raw_window(neural, end):
    value = np.zeros((W, UNITS), dtype=np.float32); n = min(end + 1, W); value[-n:] = neural[end + 1 - n:end + 1]; return value


def run(*, formal: Path, output: Path, authorization: Path, authorization_sha256: str, allow_cpu_fixture=False, source_loader=None, model_factory=None) -> dict[str, Any]:
    formal, output, authorization = _canonical(formal), _canonical(output), _canonical(authorization)
    if (source_loader is not None or model_factory is not None) and not allow_cpu_fixture: raise RuntimeError("injected source/model dependencies are test-only")
    if output.exists() or any(_inside(root, output) or _inside(root, authorization) for root in (formal, CACHE.parent)) or _inside(output, authorization): raise RuntimeError("fresh output and disjoint external proof authorization required")
    started = time.monotonic()
    def early_guard():
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024  # Linux KiB
        if time.monotonic() - started > WALL_SECONDS or rss > RSS_LIMIT: raise RuntimeError("pre-admission CPU/RSS/wall guard")
    early_guard()
    bindings = collect_bindings(formal, output); authority = read(authorization)
    if sha(authorization) != authorization_sha256 or authority != {"schema": "h1_queryage_selected_public_proof_authorization_v1", "status": "ROOT_REVIEW_GO", "bindings": bindings}: raise RuntimeError("external proof authorization mismatch")
    # Original admission may import framework helpers but constructs no model.
    receipt, archives = _audit_formal(formal, bindings, allow_cpu_fixture=allow_cpu_fixture, guard=early_guard); early_guard()
    source_roots = {Path(name).parent for name in receipt["authority"].get("bindings", {}).get("inputs", {})}
    if any(_inside(root, output) or _inside(root, authorization) for root in source_roots):
        raise RuntimeError("proof output/authorization overlaps an original source root")
    if not allow_cpu_fixture and (os.environ.get(GO) != "1" or os.environ.get("CUDA_VISIBLE_DEVICES") not in ("", "-1")): raise RuntimeError("explicit CPU GO and hidden CUDA required")
    import torch
    if torch.cuda.is_available(): raise RuntimeError("proof is CPU-only")
    torch.set_num_threads(1)
    try: torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads() != 1: raise
    def guard():
        early_guard()
        if torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1:
            raise RuntimeError("CPU thread policy drift")
    guard(); output.mkdir()
    sidecar = output / "input_authority.json"
    sidecar_body = {"bindings": bindings, "authorization_path": str(authorization), "authorization_sha256": authorization_sha256}
    atomic_json(sidecar_body, sidecar); sidecar_sha = sha(sidecar); guard()
    guard(); cache = _load_cache(source_loader=source_loader, allow_cpu_fixture=allow_cpu_fixture); guard()
    for archive in archives.values(): _validate_archive_source(archive, cache)
    if model_factory is None:
        from tfpd_exploration.src.h1_queryage_family_v1.model import make_queryage_localbalanced_pair as model_factory
    from tfpd_exploration.src.two_mainlines_long_v1.current_query_v2.streaming import CurrentQueryStream
    from tfpd_exploration.src.two_mainlines_long_v1.decoder.h1_temporal import H1Bank
    results, public = {}, {}
    for arm_index, arm in enumerate(ARMS):
        guard(); pair = model_factory(seed=42); model = pair[arm_index]; del pair; guard()
        guard(); state = torch.load(formal / "exports" / f"{arm}_selected_plain_ema.pt", map_location="cpu", weights_only=True); guard()
        if not isinstance(state, dict) or not state or any(not isinstance(v, torch.Tensor) or v.dtype != torch.float32 or not torch.isfinite(v).all() for v in state.values()): raise RuntimeError("plain EMA must be finite FP32 tensors")
        guard(); model.load_state_dict(state, strict=True); model.eval(); before_state = _state_digest(model); guard(); archive, offset, pieces, sessions = archives[arm], 0, [], {}
        for name, row in sorted(cache["minival"].items()):
            guard()
            neural, velocity, mask = row["neural"], row["velocity"], row["eval_mask"]
            ends = _source_ends(row)
            if not isinstance(row.get("bank"), dict) or set(row["bank"]) != {"E0", "T", "unit_mask"}:
                raise RuntimeError("immutable H1 calibration bank topology drift")
            bank = H1Bank(*[row["bank"][key] for key in ("E0", "T", "unit_mask")])
            guard(); stream = CurrentQueryStream(model, bank, task="h1", session_id=name, unit_ids=range(UNITS), batch_size=1); guard()
            predicted, checks = [], {i for i in (0, 4, W - 1, W, int(ends[-1])) if 0 <= i <= ends[-1]}
            direct_errors = {}
            row_archive = {key: value[offset:offset + len(ends)] for key, value in archive.items()}
            index = 0
            for step in range(int(ends[-1]) + 1):
                guard()
                observation = np.ascontiguousarray(neural[step:step + 1])
                if step == ends[index]: value = _public(stream.predict(observation))[0]
                else: stream.observe(observation); value = None
                guard()
                if not np.array_equal(stream.front.raw[0].detach().cpu().numpy(), _raw_window(neural, step)): raise RuntimeError("W700 left-zero stream history drift")
                if step in checks:
                    guard()
                    with torch.no_grad(): direct = (model.forward_last(torch.from_numpy(_raw_window(neural, step)[None]), bank) / 20.0).detach().cpu().numpy()[0]
                    guard()
                    observed = stream.current_prediction().detach().cpu().numpy()[0] if value is None else value
                    guard()
                    if not np.allclose(observed, direct, atol=1e-5, rtol=1e-5): raise RuntimeError("independent QueryAge cached/direct mismatch")
                    direct_errors[str(step)] = float(np.max(np.abs(observed - direct)))
                    checks.remove(step)
                if value is not None:
                    if not np.allclose(value, row_archive["prediction"][index], atol=1e-5, rtol=1e-5): raise RuntimeError("public/archive prediction mismatch")
                    predicted.append(value.copy()); index += 1
                guard()
            if index != len(ends): raise RuntimeError("not all session endpoints replayed")
            if checks: raise RuntimeError("not all required startup/rollover/final direct checks executed")
            pieces.append(np.asarray(predicted, dtype=np.float64)); sessions[name] = {"n_bins": index, "observed_bins": int(ends[-1]) + 1, "direct_native_max_abs_error_by_end": direct_errors}
            offset += index
        if offset != COUNT: raise RuntimeError("complete endpoint cardinality drift")
        observed = {**archive, "prediction": np.concatenate(pieces)}; measured, oracle = _metric(observed), _metric(archive)
        _compare_metrics(measured, oracle, tolerance=1e-5)
        if _state_digest(model) != before_state: raise RuntimeError("proof mutated selected model parameters")
        public[arm], results[arm] = observed, {"sessions": sessions, "metric": measured, "max_abs_error": float(np.max(np.abs(observed["prediction"] - archive["prediction"])))}
    for arm, value in public.items():
        guard()
        path = output / f"{arm}_selected_public_native_float64.npz"
        with tempfile.NamedTemporaryFile(dir=output, suffix=".npz", delete=False) as handle: temporary = Path(handle.name)
        try:
            with temporary.open("wb") as handle: np.savez_compressed(handle, **value)
            os.replace(temporary, path)
        finally: temporary.unlink(missing_ok=True)
        disk = _archive(path)
        if any(not np.array_equal(disk[key], value[key]) for key in value): raise RuntimeError("public archive disk equality drift")
        results[arm]["public_archive"] = str(path); results[arm]["public_archive_sha256"] = sha(path)
        guard()
    if (collect_bindings(formal, output) != bindings
            or _audit_formal(formal, bindings, allow_cpu_fixture=allow_cpu_fixture, guard=guard)[0] != receipt):
        raise RuntimeError("post-proof formal/artifact closure drift")
    if sha(sidecar) != sidecar_sha or read(sidecar) != sidecar_body:
        raise RuntimeError("owned pre-authority sidecar drift")
    expected_owned = {sidecar, *(Path(results[arm]["public_archive"]) for arm in ARMS)}
    actual_owned = {p for p in output.rglob("*") if p.is_file()}
    if actual_owned != expected_owned: raise RuntimeError("unexpected public-proof output artifact")
    manifest = {str(path): sha(_canonical(path)) for path in sorted(actual_owned)}
    result = {"schema": "h1_queryage_selected_public_complete_proof_v1", "status": "PASS_IMPLEMENTATION_EQUIVALENCE_ONLY", "formal": str(formal), "formal_receipt_sha256": sha(formal / "receipt.json"), "bindings": bindings, "authorization_path": str(authorization), "authorization_sha256": authorization_sha256, "arms": results, "elapsed_seconds": time.monotonic() - started, "parameter_updates": 0, "trial_boundary_resets": False, "owned_artifact_sha256": manifest}
    result["runtime"] = {"torch_version": torch.__version__, "numpy_version": np.__version__,
                         "python_version": sys.version.split()[0], "cpu_affinity": sorted(os.sched_getaffinity(0)),
                         "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
                         "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)}
    guard()
    if sha(authorization) != authorization_sha256 or read(authorization) != authority: raise RuntimeError("post-proof external authorization drift")
    atomic_json(result, output / "receipt.json")
    try: guard()
    except BaseException:
        # A late disk/resource failure must not leave a successful receipt.
        atomic_json({"schema": result["schema"], "status": "FAILED_POST_WRITE_RESOURCE_GUARD"}, output / "receipt.json")
        raise
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(); parser.add_argument("--formal", type=Path, required=True); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--authorization", type=Path, required=True); parser.add_argument("--authorization-sha256", required=True)
    args = parser.parse_args(argv); return run(formal=args.formal, output=args.output, authorization=args.authorization, authorization_sha256=args.authorization_sha256)


if __name__ == "__main__": main()
