"""Post-selection-only native ext4 development replay for QueryAge FLAT/ROUTE.

This evaluator consumes exactly the two selected plain-EMA exports recorded by
the separate QueryAge finalizer.  Ext4 is never a selection surface here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch

from tfpd_exploration.src.m2_dual_track_v1 import data, plan
from tfpd_exploration.src.m2_family_v1 import evaluate_frozen_ext4 as ext4_contract
from tfpd_exploration.src.m2_same_query_comparator_v1 import core

from . import finalize_pair as finalizer
from . import model as queryage_model
from .model import make_paired_queryage_decoders

from tfpd_exploration.src.m2_b_small_stability_v1 import config as small_config
from tfpd_exploration.src.m2_b_small_stability_v1 import decoder as small_decoder
from tfpd_exploration.src.m2_dual_track_v1 import contracts, decoders
from tfpd_exploration.src.m2_family_v1 import config as family_config, decoder as family_decoder, routing
from tfpd_exploration.src.two_mainlines_long_v1.current_query_v2 import core as query_core

GO_ENV = "M2_QUERYAGE_SELECTED_EXT4_GO"
SCHEMA = "m2_queryage_selected_ext4_native_v1"
HARD_SECONDS, HARD_MEMORY_BYTES = 600, 22 << 30
BASELINE = Path(__file__).resolve().parents[1] / "m2_family_v1"  # replaced below for clarity
ROOT = Path(__file__).resolve().parents[3]
BASELINE = ROOT / "tfpd_exploration/results/m2/family_v1/ext4_e8_spint_dev_pooled_addendum_v1.json"
BASELINE_SHA256 = "a10963c3b87081fa7f401eeed1ab9c6829c613e479abe835acc0ff43405649e0"
REQUIRED_CACHE = ("X_store.npy", "target_store.npy", "eligible_starts.npy", "e0_u.pt", "T.npy",
                  "calib_activity.npy", "mapping.json", "provenance.json", "extra.json")
EXT4_TOTAL = 2069


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> str:
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
    np.savez_compressed(temporary, **arrays); os.replace(temporary, path)
    return sha(path)


def _inside(parent: Path, candidate: Path) -> bool:
    return os.path.commonpath((str(parent), str(candidate))) == str(parent)


def _guard(started: float, device: torch.device) -> None:
    if device.type == "cuda": torch.cuda.synchronize()
    if time.monotonic() - started > HARD_SECONDS:
        raise TimeoutError("selected ext4 evaluator exceeded fixed 600-second budget")
    if device.type == "cuda" and torch.cuda.max_memory_allocated() > HARD_MEMORY_BYTES:
        raise RuntimeError("selected ext4 evaluator exceeded 22-GiB allocated-memory budget")


def _execution_gate(*, physical_gpu: int, threads: int, allow_cpu_for_test: bool) -> torch.device:
    if threads != 1:
        raise RuntimeError("selected ext4 evaluator requires exactly one CPU thread")
    if allow_cpu_for_test:
        return torch.device("cpu")
    if (physical_gpu not in (0, 1) or os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical_gpu)
            or not torch.cuda.is_available()):
        raise RuntimeError("selected ext4 evaluator requires one leased physical GPU 0 or 1")
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.cuda.reset_peak_memory_stats()
    return torch.device("cuda:0")


def _selected_exports(run_root: Path, finalizer_output: Path, receipt_sha256: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Validate completion/finalizer closure before cache or model access."""
    _require_sha(receipt_sha256, "finalizer receipt SHA-256")
    receipt_path, freeze_path = finalizer_output / "receipt.json", finalizer_output / "selection_freeze.json"
    if not receipt_path.is_file() or not freeze_path.is_file() or sha(receipt_path) != receipt_sha256:
        raise RuntimeError("hash-bound completed finalizer receipt/freeze required")
    # Completion builds epoch-evidence maps with integer keys, while both
    # sealed JSON receipts necessarily reload those keys as strings. Compare
    # the same JSON representation, without altering any sealed artifact or
    # relaxing the underlying completion/source/hash checks.
    audit = json.loads(json.dumps(finalizer.validate_completion(run_root), allow_nan=False))
    receipt, freeze = _read_json(receipt_path), _read_json(freeze_path)
    fresh_source = finalizer._source_minival_authority()
    if (receipt.get("schema") != finalizer.FINALIZER_SCHEMA
            or receipt.get("status") != "SOURCE_MINIVAL_SELECTION_DIAGNOSTIC_NOT_UNTOUCHED_GENERALIZATION"
            or freeze.get("schema") != finalizer.FINALIZER_SCHEMA
            or freeze.get("status") != "SELECTION_FROZEN_PRE_EXPORT_PRE_SCORE"
            or receipt.get("audit") != audit or freeze.get("audit") != audit
            or receipt.get("authority_pre_model") != fresh_source or receipt.get("authority_post") != fresh_source
            or receipt.get("selection_freeze_sha256") != sha(freeze_path)):
        raise RuntimeError("completed finalizer schema/audit/freeze binding drift")
    exports, scores, selected = receipt.get("exports"), receipt.get("scores"), audit.get("selected_primary_ema_epoch")
    if not isinstance(exports, Mapping) or not isinstance(scores, Mapping) or not isinstance(selected, Mapping) or set(selected) != {"FLAT", "ROUTE"}:
        raise RuntimeError("exact two selected FLAT/ROUTE finalizer picks required")
    bound: dict[str, Any] = {}
    for arm in ("FLAT", "ROUTE"):
        epoch = selected[arm]
        if not isinstance(epoch, int) or not 1 <= epoch <= finalizer.EPOCHS:
            raise RuntimeError("selected finalizer epoch drift")
        key = f"{arm}_selected_epoch_{epoch:03d}"
        matches = [name for name in exports if name.startswith(f"{arm}_selected_epoch_")]
        if matches != [key] or any("endpoint24" in name for name in matches):
            raise RuntimeError("exactly one selected export and no endpoint alternative required")
        item = exports[key]
        if not isinstance(item, Mapping) or not isinstance(item.get("path"), str) or not isinstance(item.get("sha256"), str):
            raise RuntimeError("selected export path/hash contract drift")
        path = Path(item["path"])
        if not path.is_absolute() or not path.is_file() or sha(path) != item["sha256"]:
            raise RuntimeError("selected plain-EMA export changed or absent")
        score = scores.get(key)
        if not isinstance(score, Mapping) or not isinstance(score.get("archive_path"), str) or not isinstance(score.get("archive_sha256"), str):
            raise RuntimeError("selected source archive path/hash contract drift")
        source_archive = Path(score["archive_path"])
        if not source_archive.is_absolute() or not source_archive.is_file() or sha(source_archive) != score["archive_sha256"]:
            raise RuntimeError("selected source archive changed or absent")
        bound[arm] = {"key": key, "epoch": epoch, "path": str(path), "sha256": item["sha256"],
                      "source_archive_path": str(source_archive), "source_archive_sha256": score["archive_sha256"]}
    return audit, receipt, bound


def _require_sha(value: Any, label: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise RuntimeError(f"{label} must be a lowercase SHA-256")


def authority(run_root: Path, finalizer_output: Path, receipt_sha256: str, *, baseline: Path = BASELINE,
              expected_baseline_sha256: str = BASELINE_SHA256) -> dict[str, Any]:
    """Hash exact selected exports, evaluator closure, and sealed ext4 inputs."""
    audit, receipt, exports = _selected_exports(run_root, finalizer_output, receipt_sha256)
    if sha(baseline) != expected_baseline_sha256:
        raise RuntimeError("immutable e8/SPINT ext4 baseline hash drift")
    base = _read_json(baseline)
    if (base.get("schema") != "m2_family_v1_ext4_e8_spint_dev_replay_v1"
            or base.get("status") != "DEVELOPMENT_COMPARISON_NOT_UNBIASED_HELDOUT"
            or base.get("sessions") != list(plan.EXT4_SESSIONS) or not isinstance(base.get("rows"), list)
            or len(base["rows"]) != len(plan.EXT4_SESSIONS)):
        raise RuntimeError("immutable four-session ext4 development baseline required")
    code = (Path(__file__), Path(finalizer.__file__), Path(data.__file__), Path(plan.__file__), Path(core.__file__),
            Path(ext4_contract.__file__), Path(queryage_model.__file__), Path(query_core.__file__),
            Path(family_decoder.__file__), Path(family_config.__file__), Path(routing.__file__),
            Path(small_config.__file__), Path(small_decoder.__file__), Path(contracts.__file__), Path(decoders.__file__),
            Path(queryage_model.__file__).with_name("__init__.py"))
    if any(not path.is_file() for path in code):
        raise RuntimeError("evaluator code closure missing")
    rows, paths = [], [baseline, *code]
    ext4_root = data.cache_root() / "ext4"
    if not ext4_root.is_dir():
        raise RuntimeError("existing ext4 cache root required; never build")
    for session, sealed in zip(plan.EXT4_SESSIONS, base["rows"], strict=True):
        if sealed.get("session") != session or sealed.get("window_count") != plan.EXT4_EXPECTED_WINDOWS[session]:
            raise RuntimeError("sealed ext4 session roster/count drift")
        folder = ext4_root / session
        if not folder.is_dir() or any(not (folder / name).is_file() for name in REQUIRED_CACHE):
            raise RuntimeError("existing ext4 cache files required; never build")
        paths.extend(folder / name for name in REQUIRED_CACHE)
        # Hash bytes before reading the arrays and before a bank may be loaded.
        file_hashes = {name: sha(folder / name) for name in REQUIRED_CACHE}
        raw = np.load(folder / "X_store.npy", mmap_mode="r")
        starts = np.load(folder / "eligible_starts.npy", mmap_mode="r")
        target = np.load(folder / "target_store.npy", mmap_mode="r")
        mapping = _read_json(folder / "mapping.json")
        ext4_contract.verify_geometry(raw, starts, target, mapping, sealed, plan.EXT4_EXPECTED_WINDOWS[session])
        rows.append({"session": session, "window_count": len(starts), "files": file_hashes,
                     "ordered_window_starts_sha256": core.array_sha256(np.asarray(starts)),
                     "target_sha256": core.array_sha256(np.asarray(target))})
    if tuple(row["session"] for row in rows) != plan.EXT4_SESSIONS or sum(row["window_count"] for row in rows) != EXT4_TOTAL:
        raise RuntimeError("exact 2069-window ext4 surface required")
    return {"schema": SCHEMA, "run_root": str(run_root), "finalizer_root": str(finalizer_output),
            "output": None, "run_audit": audit, "finalizer_receipt_sha256": receipt_sha256,
            "finalizer_freeze_sha256": sha(finalizer_output / "selection_freeze.json"), "selected": exports,
            "baseline_sha256": expected_baseline_sha256, "baseline": base,
            "code_sha256": {str(path): sha(path) for path in code},
            "ext4_files_sha256": {str(path): sha(path) for path in paths if path.is_file()}, "rows": rows}


def collect_bindings(run_root: Path, finalizer_root: Path, finalizer_receipt_sha256: str, output: Path) -> dict[str, Any]:
    """Read-only pre-model binding API for an independently authorized replay."""
    run_root, finalizer_root, output = Path(run_root), Path(finalizer_root), Path(output)
    if (not run_root.is_absolute() or not finalizer_root.is_absolute() or not output.is_absolute() or output.exists()
            or _inside(run_root, output) or _inside(finalizer_root, output)):
        raise RuntimeError("absolute fresh output separate from trainer/finalizer required")
    bound = authority(run_root, finalizer_root, finalizer_receipt_sha256)
    bound["output"] = str(output)
    return bound


def _load_model(arm: str, export: Mapping[str, Any], device: torch.device):
    if sha(Path(export["path"])) != export["sha256"]:
        raise RuntimeError("selected export changed immediately before deserialize")
    model = make_paired_queryage_decoders(seed=42)[0 if arm == "FLAT" else 1]
    state = torch.load(export["path"], map_location="cpu", weights_only=True)
    if (not isinstance(state, Mapping) or not state
            or any(not torch.is_tensor(value) or value.dtype != torch.float32 or not bool(torch.isfinite(value).all())
                   for value in state.values())):
        raise RuntimeError("selected plain-EMA export must be finite FP32 tensors")
    model.load_state_dict(state, strict=True); model.to(device).eval()
    return model


def _state_digest(model: torch.nn.Module) -> dict[str, str]:
    return {name: hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            for name, value in model.state_dict().items()}


def score_selected(authority_value: Mapping[str, Any], device: torch.device, started: float,
                   heartbeat: Callable[[str, str, int, int], None] | None = None) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]]]:
    """Score exactly the sealed two exports on the immutable four-session ext4 surface."""
    results, all_archives = {}, {}
    rows_authority = authority_value["rows"]
    for arm in ("FLAT", "ROUTE"):
        _guard(started, device)
        model = _load_model(arm, authority_value["selected"][arm], device)
        state_pre = _state_digest(model)
        parts = {key: [] for key in ("prediction", "target", "start", "session")}; rows = []; singleton_error = 0.0
        for row in rows_authority:
            session, count = row["session"], row["window_count"]
            folder = data.cache_root() / "ext4" / session
            if not folder.is_dir() or any(not (folder / name).is_file() for name in REQUIRED_CACHE):
                raise RuntimeError("ext4 cache disappeared before bank load")
            if {name: sha(folder / name) for name in REQUIRED_CACHE} != row["files"]:
                raise RuntimeError("ext4 cache bytes changed before bank load")
            raw = np.load(folder / "X_store.npy", mmap_mode="r")
            starts = np.load(folder / "eligible_starts.npy", mmap_mode="r")
            target = np.load(folder / "target_store.npy", mmap_mode="r")
            if raw.dtype != np.float32 or starts.dtype != np.int64 or target.dtype != np.float32:
                raise RuntimeError("ext4 source native dtype drift")
            if (core.array_sha256(np.asarray(starts)) != row["ordered_window_starts_sha256"]
                    or core.array_sha256(np.asarray(target)) != row["target_sha256"]):
                raise RuntimeError("ext4 arrays changed after pre-model authority")
            _guard(started, device)
            bank = data.load_session_bank("ext4", session, device=device)
            _guard(started, device)
            predicted = []
            with torch.inference_mode():
                for off in range(0, count, 8):
                    _guard(started, device)
                    windows = np.stack([raw[int(item):int(item) + 50] for item in starts[off:off + 8]]).astype(np.float32, copy=False)
                    value = (model.forward_last(torch.from_numpy(windows).to(device), bank, bank.unit_mask) / 5).cpu().numpy()
                    _guard(started, device)
                    if off == 0:
                        _guard(started, device)
                        single = (model.forward_last(torch.from_numpy(windows[:1]).to(device), bank, bank.unit_mask) / 5).cpu().numpy()
                        _guard(started, device)
                        np.testing.assert_allclose(value[:1], single, atol=1e-5, rtol=1e-5)
                        singleton_error = max(singleton_error, float(np.abs(value[:1] - single).max()))
                    predicted.append(value.astype(np.float64))
                    if heartbeat is not None: heartbeat(arm, session, off + len(windows), count)
            prediction = np.concatenate(predicted)
            if prediction.shape != (count, 2) or not np.isfinite(prediction).all():
                raise RuntimeError("finite native ext4 prediction geometry required")
            r2 = float(core.variance_weighted_r2(np.asarray(target, dtype=np.float64), prediction))
            if not math.isfinite(r2): raise RuntimeError("nonfinite ext4 native R2")
            rows.append({"session": session, "windows": count, "native_r2": r2,
                         "ordered_window_starts_sha256": row["ordered_window_starts_sha256"], "target_sha256": row["target_sha256"]})
            parts["prediction"].append(prediction); parts["target"].append(np.asarray(target, dtype=np.float64))
            parts["start"].append(np.asarray(starts, dtype=np.int64)); parts["session"].append(np.asarray([session] * count))
        archive = {key: np.concatenate(values) for key, values in parts.items()}
        equal, pooled = _recompute_ext4(archive)
        baseline_rows = {row["session"]: row for row in authority_value["baseline"]["rows"]}
        scored_rows = []
        for row in rows:
            baseline_row = baseline_rows[row["session"]]
            scored_rows.append({**row, "delta_vs_spint": row["native_r2"] - baseline_row["spint_r2"],
                                "delta_vs_e8": row["native_r2"] - baseline_row["e8_r2"]})
        results[arm] = {"selected": authority_value["selected"][arm], "equal_session_r2": equal,
                        "pooled_r2": pooled, "max_batched_singleton_abs_error": singleton_error,
                        "rows": scored_rows,
                        "equal_delta_vs_spint": equal - authority_value["baseline"]["spint_equal_session_r2"],
                        "pooled_delta_vs_spint": pooled - authority_value["baseline"]["spint_pooled_r2"],
                        "equal_delta_vs_e8": equal - authority_value["baseline"]["e8_equal_session_r2"],
                        "pooled_delta_vs_e8": pooled - authority_value["baseline"]["e8_pooled_r2"]}
        all_archives[arm] = archive
        if _state_digest(model) != state_pre:
            raise RuntimeError("selected strict plain-EMA state changed during inference")
        model.cpu(); del model
    _paired_archive_identity(all_archives)
    return results, all_archives


def _recompute_ext4(archive: Mapping[str, np.ndarray]) -> tuple[float, float]:
    total = EXT4_TOTAL
    if (set(archive) != {"prediction", "target", "start", "session"} or archive["prediction"].shape != (total, 2)
            or archive["target"].shape != (total, 2) or archive["prediction"].dtype != np.float64
            or archive["target"].dtype != np.float64 or archive["start"].dtype != np.int64
            or archive["start"].shape != (total,) or archive["session"].shape != (total,)
            or archive["session"].dtype.kind != "U" or not np.isfinite(archive["prediction"]).all()
            or not np.isfinite(archive["target"]).all()):
        raise RuntimeError("strict 2069 native FP64 archive geometry required")
    scores = []
    cursor = 0
    for session, count in plan.EXT4_EXPECTED_WINDOWS.items():
        mask = archive["session"] == session
        if (not np.array_equal(np.flatnonzero(mask), np.arange(cursor, cursor + count))
                or np.any(archive["start"][mask] < 0) or np.any(np.diff(archive["start"][mask]) <= 0)):
            raise RuntimeError("four-session native archive ordered roster/start drift")
        score = float(core.variance_weighted_r2(archive["target"][mask], archive["prediction"][mask]))
        if not math.isfinite(score): raise RuntimeError("nonfinite session native R2")
        scores.append(score); cursor += count
    pooled = float(core.variance_weighted_r2(archive["target"], archive["prediction"]))
    if cursor != total or not math.isfinite(pooled): raise RuntimeError("nonfinite/cardinality pooled native R2")
    return float(np.mean(scores)), pooled


def recompute_metrics(arrays: Mapping[str, np.ndarray]) -> dict[str, float]:
    """Public FP64 archive metric recomputation helper."""
    equal, pooled = _recompute_ext4(arrays)
    return {"equal_session_r2": equal, "pooled_r2": pooled}


def validate_archive(arrays: Mapping[str, np.ndarray], authority_value: Mapping[str, Any]) -> dict[str, float]:
    """Require the archive to retain the sealed four-session external identity."""
    _recompute_ext4(arrays)
    cursor = 0
    for row in authority_value.get("rows", ()):
        count = row["window_count"]; piece = slice(cursor, cursor + count)
        target32 = arrays["target"][piece].astype(np.float32)
        if (not np.array_equal(arrays["target"][piece], target32.astype(np.float64))
                or not np.all(arrays["session"][piece] == row["session"])
                or core.array_sha256(arrays["start"][piece]) != row["ordered_window_starts_sha256"]
                or core.array_sha256(target32) != row["target_sha256"]):
            raise RuntimeError("native ext4 archive differs from sealed source identity")
        cursor += count
    if cursor != EXT4_TOTAL: raise RuntimeError("native ext4 archive cardinality drift")
    return recompute_metrics(arrays)


def _paired_archive_identity(archives: Mapping[str, Mapping[str, np.ndarray]]) -> None:
    if set(archives) != {"FLAT", "ROUTE"}: raise RuntimeError("exact paired archives required")
    first = archives["FLAT"]
    for field in ("target", "start", "session"):
        if not np.array_equal(first[field], archives["ROUTE"][field]):
            raise RuntimeError("FLAT/ROUTE ext4 native archive identity drift")


def prepare(run_root: Path, finalizer_output: Path, output: Path, receipt_sha256: str) -> dict[str, Any]:
    bound = collect_bindings(run_root, finalizer_output, receipt_sha256, output)
    result = {"schema": SCHEMA, "status": "PREPARED_NO_MODEL_OR_EXT4_FORWARD", "authority": bound,
              "selection_disclosure": "ext4 opened only after completed source-minival selection/finalization; no ext4-based selection",
              "parameter_updates": 0, "new_calibration_fits": 0, "selection": None, "promotion": None}
    _atomic_json(output / "preflight.json", result)
    return result


def run(run_root: Path, finalizer_output: Path, output: Path, receipt_sha256: str, *, authorization: Path,
        authorization_sha256: str, physical_gpu: int, threads: int = 1, allow_cpu_for_test: bool = False) -> dict[str, Any]:
    started = time.monotonic(); device = _execution_gate(physical_gpu=physical_gpu, threads=threads, allow_cpu_for_test=allow_cpu_for_test)
    _guard(started, device)
    if os.environ.get(GO_ENV) != "1": raise RuntimeError("explicit reviewed GO required")
    # Authorization is necessarily made *after* a durable preflight exists.
    # Hence execution accepts precisely a preflight-only output root, never an
    # arbitrary existing result directory and never a just-created unsigned one.
    if not output.is_absolute() or not (output / "preflight.json").is_file() or any(path.name != "preflight.json" for path in output.iterdir()):
        raise RuntimeError("run requires a separately prepared preflight-only output root")
    prepared = _read_json(output / "preflight.json")
    if (not run_root.is_absolute() or not finalizer_output.is_absolute() or not output.is_absolute()
            or _inside(run_root, output) or _inside(finalizer_output, output)):
        raise RuntimeError("absolute output separate from trainer/finalizer required")
    current = authority(run_root, finalizer_output, receipt_sha256); current["output"] = str(output)
    if (prepared.get("schema") != SCHEMA or prepared.get("status") != "PREPARED_NO_MODEL_OR_EXT4_FORWARD"
            or prepared.get("authority") != current):
        raise RuntimeError("prepared current trainer/finalizer/ext4 authority drift")
    _guard(started, device)
    _require_sha(authorization_sha256, "authorization SHA-256")
    if not authorization.is_absolute() or not authorization.is_file() or sha(authorization) != authorization_sha256:
        raise RuntimeError("hash-bound external authorization required")
    auth = _read_json(authorization)
    if (auth.get("status") != "ROOT_REVIEW_GO" or auth.get("authority") != prepared["authority"]
            or auth.get("preflight_sha256") != sha(output / "preflight.json")):
        raise RuntimeError("external authorization/preflight authority drift")
    _atomic_json(output / "input_authority.json", {"authority": prepared["authority"], "authorization_sha256": authorization_sha256})
    def heartbeat(arm: str, session: str, done: int, total: int) -> None:
        _guard(started, device)
        _atomic_json(output / "heartbeat.json", {"status": "SCORING", "arm": arm, "session": session,
                                                   "windows_done": done, "windows_total": total,
                                                   "elapsed_seconds": time.monotonic() - started})
    results, archives = score_selected(prepared["authority"], device, started, heartbeat)
    _guard(started, device)
    post = authority(run_root, finalizer_output, receipt_sha256); post["output"] = str(output)
    if post != prepared["authority"]: raise RuntimeError("post-score authority drift")
    for arm, archive in archives.items():
        validate_archive(archive, prepared["authority"])
        path = output / f"{arm}_selected_ext4_native_float64.npz"
        results[arm]["native_archive_path"], results[arm]["native_archive_sha256"] = str(path), _atomic_npz(path, archive)
    if any(sha(Path(value["native_archive_path"])) != value["native_archive_sha256"] for value in results.values()):
        raise RuntimeError("written native ext4 archive hash drift")
    _guard(started, device)
    result = {"schema": SCHEMA, "status": "COMPLETE_FIXED_SELECTED_EXT4_DEVELOPMENT_ONLY", "pre": prepared["authority"], "post": post,
              "arms": results, "elapsed_seconds": time.monotonic() - started, "execution_device": str(device),
              "physical_gpu": physical_gpu, "threads": threads, "peak_allocated_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else 0,
              "hard_limits": {"seconds": HARD_SECONDS, "cuda_peak_allocated_bytes": HARD_MEMORY_BYTES},
              "parameter_updates": 0, "new_calibration_fits": 0, "epoch_or_model_selection": None, "promotion": None,
              "qualification": "development comparison with historical external exposure, not untouched heldout or a selection/promotion criterion"}
    _atomic_json(output / "receipt.json", result); return result


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path); parser.add_argument("--finalizer-output", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path); parser.add_argument("--finalizer-receipt-sha256", required=True)
    parser.add_argument("--prepare", action="store_true"); parser.add_argument("--authorization", type=Path)
    parser.add_argument("--authorization-sha256"); parser.add_argument("--physical-gpu", choices=(0, 1), type=int)
    args = parser.parse_args(argv)
    if args.prepare == (args.authorization is not None): parser.error("choose --prepare or --authorization PATH")
    if args.prepare: return prepare(args.run_root, args.finalizer_output, args.output, args.finalizer_receipt_sha256)
    if args.physical_gpu is None or args.authorization_sha256 is None: parser.error("run needs --physical-gpu and --authorization-sha256")
    return run(args.run_root, args.finalizer_output, args.output, args.finalizer_receipt_sha256, authorization=args.authorization,
               authorization_sha256=args.authorization_sha256, physical_gpu=args.physical_gpu)


if __name__ == "__main__": main()
