"""Post-completion source finalizer for the M2 QueryAge+prefix paired run.

``prepare`` freezes the reconciled source-minival EMA selection before any
checkpoint deserialization, model construction, or scoring. The gated ``run``
then exports the selected and fixed endpoint EMAs and replays source-minival
only, without parameter updates. All outputs are outside the sealed trainer
root; no external development surface is opened and no selection rule changes.
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
from typing import Any, Mapping

import numpy as np
import torch

from tfpd_exploration.src.m2_b_small_stability_v1 import ema as ema_module
from tfpd_exploration.src.m2_b_small_stability_v1 import training
from tfpd_exploration.src.m2_b_small_stability_v1.ema import DecoderEMA
from tfpd_exploration.src.m2_dual_track_v1 import data, plan
from tfpd_exploration.src.m2_family_v1 import finalize_pair as prior_finalizer
from tfpd_exploration.src.m2_same_query_comparator_v1 import core

from . import train_pair
from .model import M2QueryAgeFamilyDecoder, make_paired_queryage_decoders

ARMS = train_pair.ARMS
EPOCHS = train_pair.EPOCHS
GO_ENV = "M2_QUERYAGE_FINALIZE_GO"
FINALIZER_SCHEMA = "m2_queryage_prefix_pair_completion_finalize_v1"
FINALIZE_HARD_SECONDS = 2_400
HARD_MEMORY_BYTES = 22 << 30


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


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _tree_hashes_before_completion(root: Path) -> dict[str, str]:
    """Rebuild precisely the tree the trainer sealed before completion."""
    excluded = {"artifact_hashes_pre_completion.json", "completion.json"}
    return {str(path.relative_to(root)): sha(path) for path in sorted(root.rglob("*"))
            if path.is_file() and str(path.relative_to(root)) not in excluded}


def _require_hash(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise RuntimeError(f"{label} must be a lowercase SHA-256")
    return value


def _selection_pick(history: Mapping[str, Any], arm: str) -> int:
    if not isinstance(history, Mapping) or set(history) != {"RAW", "EMA", "receipts"}:
        raise RuntimeError(f"{arm} history topology drift")
    ledgers: dict[str, dict[int, float]] = {}
    for kind in ("RAW", "EMA"):
        values = history[kind]
        if not isinstance(values, Mapping):
            raise RuntimeError(f"{arm} {kind} ledger is not a mapping")
        try:
            parsed = {int(epoch): float(value) for epoch, value in values.items()}
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"{arm} {kind} ledger key/value drift") from exc
        if tuple(sorted(parsed)) != tuple(range(1, EPOCHS + 1)) or not all(_finite(value) for value in parsed.values()):
            raise RuntimeError(f"{arm} requires {EPOCHS} contiguous finite {kind} scores")
        ledgers[kind] = parsed
    return min(ledgers["EMA"], key=lambda epoch: (-ledgers["EMA"][epoch], epoch))


def _ledger_value(ledger: Mapping[str, Any], epoch: int) -> float:
    value = ledger[str(epoch)] if str(epoch) in ledger else ledger[epoch]
    return float(value)


def _validate_epoch_artifacts(root: Path, summary: Mapping[str, Any], protocol_sha256: str) -> tuple[dict[str, int], dict[str, Any]]:
    history = summary.get("history")
    if not isinstance(history, Mapping) or set(history) != set(ARMS):
        raise RuntimeError("exact FLAT/ROUTE history required")
    picks: dict[str, int] = {}
    evidence: dict[str, Any] = {}
    batch_digests: dict[int, dict[str, str]] = {}
    for arm in ARMS:
        arm_history = history[arm]
        pick = _selection_pick(arm_history, arm)
        picks[arm] = pick
        receipts = arm_history["receipts"]
        if not isinstance(receipts, Mapping):
            raise RuntimeError(f"{arm} receipt ledger is not a mapping")
        try:
            normalized = {int(epoch): receipt for epoch, receipt in receipts.items()}
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"{arm} receipt ledger key drift") from exc
        if tuple(sorted(normalized)) != tuple(range(1, EPOCHS + 1)):
            raise RuntimeError(f"{arm} requires {EPOCHS} contiguous receipts")
        hashes: dict[int, dict[str, str]] = {}
        for epoch in range(1, EPOCHS + 1):
            recorded = normalized[epoch]
            if not isinstance(recorded, Mapping):
                raise RuntimeError(f"{arm} epoch {epoch} in-memory receipt is invalid")
            receipt_path = root / arm / f"epoch_{epoch:03d}_receipt.json"
            receipt = _read_json(receipt_path)
            receipt_sha = sha(receipt_path)
            if receipt_sha != recorded.get("receipt_sha256") or receipt != {key: value for key, value in recorded.items() if key != "receipt_sha256"}:
                raise RuntimeError(f"{arm} epoch {epoch} sealed receipt drift")
            if (receipt.get("schema") != "m2_queryage_prefix_pair_epoch_receipt_v1" or receipt.get("arm") != arm
                    or receipt.get("epoch") != epoch or receipt.get("protocol_sha256") != protocol_sha256
                    or receipt.get("actual_epoch_batches") != train_pair.UPDATES_PER_EPOCH
                    or receipt.get("endpoint_checkpoint") is not (epoch == EPOCHS)
                    or receipt.get("checkpoint_has_raw_ema_optimizer_rng") is not True):
                raise RuntimeError(f"{arm} epoch {epoch} receipt contract drift")
            actual = receipt.get("actual_batch_sha256")
            if (not isinstance(actual, Mapping) or set(actual) != {"order", "target", "prefix", "keep"}
                    or any(not isinstance(value, str) or len(value) != 64
                           or any(char not in "0123456789abcdef" for char in value) for value in actual.values())):
                raise RuntimeError(f"{arm} epoch {epoch} actual batch digest contract drift")
            normalized_digest = dict(actual)
            if epoch in batch_digests and batch_digests[epoch] != normalized_digest:
                raise RuntimeError(f"FLAT/ROUTE epoch {epoch} actual batch route drift")
            batch_digests[epoch] = normalized_digest
            for key, kind in (("raw_equal_session_r2", "RAW"), ("ema_equal_session_r2", "EMA")):
                expected = _ledger_value(arm_history[kind], epoch)
                if not _finite(receipt.get(key)) or abs(float(receipt[key]) - expected) > 1e-12:
                    raise RuntimeError(f"{arm} epoch {epoch} receipt/score ledger drift")
            checkpoint = root / arm / f"epoch_{epoch:03d}.pt"
            if not checkpoint.is_file() or receipt.get("checkpoint") != str(checkpoint) or receipt.get("checkpoint_sha256") != sha(checkpoint):
                raise RuntimeError(f"{arm} epoch {epoch} checkpoint binding drift")
            hashes[epoch] = {"receipt_sha256": receipt_sha, "checkpoint_sha256": sha(checkpoint)}
        evidence[arm] = {"selected_primary_ema_epoch": pick, "epoch_artifact_hashes": hashes}
    return picks, evidence


def _current_trainer_authority(protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Re-run the trainer's no-GPU authority gate before any model is made."""
    resource = protocol.get("resource_smoke")
    if not isinstance(resource, Mapping):
        raise RuntimeError("protocol resource-smoke authority missing")
    path, digest = resource.get("path"), resource.get("sha256")
    if not isinstance(path, str) or not isinstance(digest, str):
        raise RuntimeError("protocol resource-smoke binding malformed")
    preflight = train_pair.preflight_only(smoke_receipt=Path(path), smoke_sha256=digest)
    closure, smoke_closure = preflight.get("closure"), preflight.get("smoke_closure")
    if not isinstance(closure, Mapping) or not isinstance(smoke_closure, Mapping):
        raise RuntimeError("current trainer preflight authority malformed")
    return {**dict(closure), "hash_bound_resource_smoke_receipt": digest,
            "resource_smoke_closure_current": dict(smoke_closure)}


def validate_completion(root: Path) -> dict[str, Any]:
    """Validate the trainer's actual final schema without loading checkpoints."""
    root = Path(root)
    required = ("protocol_preconstruction.json", "selection_summary.json", "artifact_hashes_pre_completion.json", "completion.json")
    if not root.is_absolute() or not root.is_dir() or any(not (root / name).is_file() for name in required):
        raise RuntimeError("absolute completed run root with all final artifacts required")
    protocol_path, summary_path = root / "protocol_preconstruction.json", root / "selection_summary.json"
    manifest_path, completion_path = root / "artifact_hashes_pre_completion.json", root / "completion.json"
    protocol, summary, manifest, completion = (_read_json(path) for path in (protocol_path, summary_path, manifest_path, completion_path))
    protocol_sha = sha(protocol_path)
    if protocol.get("schema") != train_pair.SCHEMA or protocol.get("status") != "PROTOCOL_SAVED_PRE_CONSTRUCTION":
        raise RuntimeError("preconstruction protocol schema/status drift")
    if (summary.get("schema") != train_pair.SCHEMA or summary.get("status") != "SOURCE_MINIVAL_SELECTION_ONLY_COMPLETE"
            or summary.get("outer_or_ext4_or_public_predictions_opened") is not False or summary.get("endpoint_epoch") != EPOCHS):
        raise RuntimeError("selection summary completion/scope drift")
    if summary.get("protocol") != protocol or summary.get("authority_pre") != protocol.get("authority_pre"):
        raise RuntimeError("selection summary protocol/authority drift")
    if summary.get("authority_pre") != summary.get("authority_post"):
        raise RuntimeError("selection summary authority changed during run")
    current_authority = _current_trainer_authority(protocol)
    if summary.get("authority_pre") != current_authority:
        raise RuntimeError("completed trainer authority differs from current preflight authority")
    if (completion.get("schema") != train_pair.SCHEMA or completion.get("status") != summary["status"]
            or completion.get("authority_pre") != summary["authority_pre"] or completion.get("authority_post") != summary["authority_post"]):
        raise RuntimeError("completion schema/status/authority drift")
    if completion.get("selection_summary_sha256") != sha(summary_path):
        raise RuntimeError("completion selection summary hash drift")
    if completion.get("artifact_hash_manifest_sha256") != sha(manifest_path):
        raise RuntimeError("completion artifact manifest hash drift")
    if manifest != _tree_hashes_before_completion(root):
        raise RuntimeError("pre-completion artifact hash manifest drift")
    picks, receipt_evidence = _validate_epoch_artifacts(root, summary, protocol_sha)
    if summary.get("selected_primary_ema_epoch") != picks:
        raise RuntimeError("selection summary does not reconcile earliest EMA picks")
    return {"run_root": str(root), "protocol_sha256": protocol_sha, "selection_summary_sha256": sha(summary_path),
            "artifact_hash_manifest_sha256": sha(manifest_path), "completion_sha256": sha(completion_path),
            "selected_primary_ema_epoch": picks, "epoch_receipt_evidence": receipt_evidence}


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def prepare(run_root: Path, output: Path, completion_sha256: str) -> dict[str, Any]:
    """Persist a post-completion selection freeze in a new separate directory."""
    run_root, output = Path(run_root), Path(output)
    if (os.environ.get(GO_ENV) != "1" or not run_root.is_absolute() or not output.is_absolute() or output.exists()
            or os.path.commonpath((str(run_root), str(output))) == str(run_root)):
        raise RuntimeError("explicit GO, absolute paths, and a new output directory required")
    _require_hash(completion_sha256, label="completion SHA-256")
    completion_path = run_root / "completion.json"
    if not completion_path.is_file() or sha(completion_path) != completion_sha256:
        raise RuntimeError("hash-bound completion artifact is absent or changed")
    audit = validate_completion(run_root)
    if sha(completion_path) != completion_sha256:
        raise RuntimeError("completion artifact changed during final audit")
    code_paths = (Path(__file__), Path(train_pair.__file__))
    code_pre = {str(path): sha(path) for path in code_paths}
    result = {"schema": FINALIZER_SCHEMA, "status": "SELECTION_FROZEN_PRE_EXPORT_PRE_SCORE",
              "scope": "fresh M2 QueryAge+prefix pair only; completed source-minival selection is exposed and is not untouched generalization",
              "completion_sha256": completion_sha256, "audit": audit, "code_sha256_pre": code_pre,
              "no_checkpoint_deserialization_yet": True, "no_model_construction_yet": True, "no_scoring_yet": True}
    if {str(path): sha(path) for path in code_paths} != code_pre:
        raise RuntimeError("finalizer code changed during selection freeze")
    _atomic_json(output / "selection_freeze.json", result)
    return result


def _source_minival_authority() -> dict[str, Any]:
    """Hash the exact code and source-minival files before model construction."""
    from tfpd_exploration.src.m2_b_small_stability_v1 import config as small_config, decoder as small_decoder
    from tfpd_exploration.src.m2_dual_track_v1 import contracts, decoders
    from tfpd_exploration.src.m2_family_v1 import decoder as m2_decoder
    from tfpd_exploration.src.two_mainlines_long_v1.current_query_v2 import core as query_core

    from . import model
    modules = (Path(__file__), Path(train_pair.__file__), Path(model.__file__), Path(training.__file__),
               Path(ema_module.__file__), Path(data.__file__),
               Path(plan.__file__), Path(core.__file__), Path(prior_finalizer.__file__), Path(small_config.__file__),
               Path(small_decoder.__file__), Path(contracts.__file__), Path(decoders.__file__), Path(m2_decoder.__file__),
               Path(query_core.__file__))
    paths = list(modules)
    if any(not path.is_file() for path in paths):
        raise RuntimeError("finalizer code closure file missing")
    names = ("X_store.npy", "target_store.npy", "eligible_starts.npy", "T.npy", "e0_u.pt",
             "mapping.json", "provenance.json", "extra.json", "calib_activity.npy")
    rows = []
    source_root = plan.active_run_root() / "cache" / "source_minival"
    if not source_root.is_dir():
        raise RuntimeError("existing source-minival cache root required; no creation permitted")
    expected_counts = tuple(prior_finalizer.COUNTS)
    rows_by_session: list[dict[str, Any]] = []
    for session, expected_count in zip(plan.HELDIN_SESSIONS, expected_counts):
        folder = source_root / session
        if not folder.is_dir() or any(not (folder / name).is_file() for name in names):
            raise RuntimeError("source-minival closure file missing")
        # Seal bytes before deserializing them; do not silently coerce source dtypes.
        file_hashes = {name: sha(folder / name) for name in names}
        starts_native = np.load(folder / "eligible_starts.npy", mmap_mode="r")
        target_native = np.load(folder / "target_store.npy", mmap_mode="r")
        x_native, t_native = np.load(folder / "X_store.npy", mmap_mode="r"), np.load(folder / "T.npy", mmap_mode="r")
        if (starts_native.dtype != np.int64 or target_native.dtype != np.float32 or x_native.dtype != np.float32
                or t_native.dtype != np.float32 or starts_native.ndim != 1
                or target_native.shape != (expected_count, 2) or len(starts_native) != expected_count):
            raise RuntimeError("source-minival native dtype/count geometry drift")
        starts, target = np.asarray(starts_native), np.asarray(target_native)
        if starts.size == 0 or int(starts.min()) < 0 or int(starts.max()) + plan.WINDOW > len(x_native):
            raise RuntimeError("source-minival native window geometry drift")
        rows_by_session.append({"session": session, "files": file_hashes,
                     "ordered_window_starts_sha256": core.array_sha256(starts),
                     "target_sha256": core.array_sha256(target)})
    if len(rows_by_session) != len(expected_counts) or sum(expected_counts) != 1011:
        raise RuntimeError("source-minival seven-session roster drift")
    return {"code": {str(path): sha(path) for path in paths}, "source_minival": rows_by_session}


def _factory(arm: str) -> M2QueryAgeFamilyDecoder:
    if arm not in ARMS:
        raise RuntimeError("unknown paired arm")
    return make_paired_queryage_decoders(seed=42)[0 if arm == "FLAT" else 1]


def checkpoint_contract(payload: Mapping[str, Any], model: M2QueryAgeFamilyDecoder, *, arm: str, epoch: int,
                        audit: Mapping[str, Any]) -> dict[str, Any]:
    """Verify the real QueryAge checkpoint topology before using its EMA."""
    summary = _read_json(Path(audit["run_root"]) / "selection_summary.json")
    ema, raw = payload.get("ema"), payload.get("raw_state_dict")
    expected_step = epoch * train_pair.UPDATES_PER_EPOCH
    if (payload.get("schema") != "m2_queryage_prefix_pair_checkpoint_v1" or payload.get("cell") != f"QUERYAGE_PREFIX_{arm}"
            or payload.get("epoch") != epoch or payload.get("global_step") != expected_step
            or payload.get("batch_id") != train_pair.UPDATES_PER_EPOCH or payload.get("seed") != 42
            or payload.get("actual_epoch_batches") != train_pair.UPDATES_PER_EPOCH
            or payload.get("manifest_digest") != summary.get("manifest_digest")
            or payload.get("protocol_sha256") != audit["protocol_sha256"]
            or payload.get("authority") != summary.get("authority_pre")
            or payload.get("recipe") != summary.get("protocol")):
        raise RuntimeError("QueryAge checkpoint metadata contract drift")
    if not isinstance(ema, Mapping) or not isinstance(raw, Mapping) or ema.get("n_updates") != expected_step or ema.get("decay") != train_pair.EMA_DECAY:
        raise RuntimeError("QueryAge checkpoint EMA update/decay contract drift")
    params, model_state, shadow = dict(model.trainable_parameters()), model.state_dict(), ema.get("shadow")
    if (set(raw) != set(model_state) or not isinstance(shadow, Mapping) or set(shadow) != set(params)
            or not raw or not all(torch.is_tensor(value) and bool(torch.isfinite(value).all()) for value in raw.values())
            or not all(torch.is_tensor(value) and tuple(value.shape) == tuple(params[name].shape) and bool(torch.isfinite(value).all())
                       for name, value in shadow.items())):
        raise RuntimeError("QueryAge checkpoint RAW/EMA tensor topology drift")
    if (any(value.is_floating_point() and value.dtype != torch.float32 for value in raw.values())
            or any(value.dtype != torch.float32 for value in shadow.values())):
        raise RuntimeError("QueryAge checkpoint requires exact FP32 RAW/EMA tensors")
    return {"epoch": epoch, "global_step": expected_step, "ema_n_updates": expected_step,
            "ema_decay": train_pair.EMA_DECAY, "raw_state_keys": sorted(raw), "ema_parameter_keys": sorted(shadow)}


def _atomic_torch(path: Path, state: Mapping[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".pt", delete=False) as handle:
        temporary = Path(handle.name)
    torch.save(dict(state), temporary)
    os.replace(temporary, path)


def export_plain_ema_strict(payload: Mapping[str, Any], model: M2QueryAgeFamilyDecoder, path: Path, *, arm: str) -> dict[str, Any]:
    """Export a plain state dict: RAW buffers plus EMA trainable parameters."""
    raw, shadow = payload["raw_state_dict"], payload["ema"]["shadow"]
    model.load_state_dict(raw, strict=True)
    plain = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    for name, parameter in model.trainable_parameters().items():
        plain[name] = shadow[name].detach().cpu().clone().to(dtype=parameter.dtype)
    _atomic_torch(path, plain)
    reloaded = _factory(arm)
    exported = torch.load(path, map_location="cpu", weights_only=True)
    missing, unexpected = reloaded.load_state_dict(exported, strict=True)
    if missing or unexpected or set(exported) != set(plain) or any(not torch.equal(plain[name], reloaded.state_dict()[name]) for name in plain):
        raise RuntimeError("strict QueryAge plain-EMA export reload drift")
    return {"path": str(path), "sha256": sha(path), "state_keys": sorted(plain), "strict_plain_ema_reload": True,
            "raw_buffers_preserved": True, "ema_trainable_parameters_substituted": True}


def _write_archive(path: Path, archive: Mapping[str, np.ndarray]) -> str:
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
    np.savez_compressed(temporary, **archive)
    os.replace(temporary, path)
    return sha(path)


def _archive_identity(archives: Mapping[str, Mapping[str, np.ndarray]], authority: Mapping[str, Any] | None = None) -> dict[str, str]:
    if len(archives) != 4:
        raise RuntimeError("exactly four selected/endpoint archives required")
    first = next(iter(archives.values()))
    for key in ("target", "start", "session"):
        if key not in first:
            raise RuntimeError("native archive identity field missing")
        for archive in archives.values():
            if key not in archive or not np.array_equal(first[key], archive[key]):
                raise RuntimeError("selected/endpoint native archive identity drift")
    target = np.asarray(first["target"])
    starts, sessions = np.asarray(first["start"]), np.asarray(first["session"])
    if target.dtype != np.float32 or target.shape != (1011, 2) or starts.dtype != np.int64 or starts.shape != (1011,) or sessions.shape != (1011,):
        raise RuntimeError("exact 1011x2 native archive identity geometry required")
    expected_rows = authority.get("source_minival") if isinstance(authority, Mapping) else None
    if (not isinstance(expected_rows, list) or len(expected_rows) != 7
            or [row.get("session") for row in expected_rows] != list(plan.HELDIN_SESSIONS)):
        raise RuntimeError("exact seven-session source authority required")
    for archive in archives.values():
        _recompute_archive_score(archive)
    cursor = 0
    for row, count in zip(expected_rows, prior_finalizer.COUNTS, strict=True):
        piece_target, piece_start = target[cursor:cursor + count], starts[cursor:cursor + count]
        if (row.get("target_sha256") != core.array_sha256(piece_target)
                or row.get("ordered_window_starts_sha256") != core.array_sha256(piece_start)
                or not np.all(sessions[cursor:cursor + count] == row.get("session"))):
            raise RuntimeError("native archive differs from sealed source-minival identity")
        cursor += count
    return {key: core.array_sha256(np.asarray(first[key])) for key in ("target", "start", "session")}


def _recompute_archive_score(archive: Mapping[str, np.ndarray]) -> tuple[float, float]:
    """Compute both reported R2s from the persisted native arrays in FP64."""
    prediction = np.asarray(archive.get("prediction"))
    target = np.asarray(archive.get("target"))
    session = np.asarray(archive.get("session"))
    starts = np.asarray(archive.get("start"))
    if (set(archive) != {"prediction", "target", "session", "start"} or prediction.dtype != np.float32 or target.dtype != np.float32
            or prediction.shape != (1011,2) or target.shape != (1011,2)
            or session.shape != (1011,) or session.dtype.kind != "U" or starts.shape != (1011,)
            or starts.dtype != np.int64 or not np.isfinite(prediction).all() or not np.isfinite(target).all()):
        raise RuntimeError("native archive FP32 prediction/target drift")
    values = []
    cursor = 0
    for label, count in zip(plan.HELDIN_SESSIONS, prior_finalizer.COUNTS, strict=True):
        mask = session == label
        if (not np.array_equal(np.flatnonzero(mask), np.arange(cursor,cursor+count))
                or np.any(starts[mask] < 0) or np.any(np.diff(starts[mask]) <= 0)):
            raise RuntimeError("native archive seven-session order/count/start drift")
        values.append(float(core.variance_weighted_r2(target[mask].astype(np.float64), prediction[mask].astype(np.float64))))
        cursor += count
    pooled = float(core.variance_weighted_r2(target.astype(np.float64), prediction.astype(np.float64)))
    equal = float(np.mean(values))
    if not math.isfinite(equal) or not math.isfinite(pooled):
        raise RuntimeError("native archive FP64 score is non-finite")
    return equal, pooled


def _execution_gate(device: str, *, physical_gpu: int, threads: int, allow_cpu_for_test: bool) -> torch.device:
    if threads != 1:
        raise RuntimeError("finalizer requires exactly one CPU thread")
    if device == "cpu":
        if not allow_cpu_for_test:
            raise RuntimeError("CPU finalizer execution requires explicit test authorization")
        return torch.device("cpu")
    if (device != "cuda:0" or physical_gpu not in (0, 1)
            or os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical_gpu) or not torch.cuda.is_available()):
        raise RuntimeError("finalizer requires available cuda:0 on one leased physical GPU (0 or 1)")
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.cuda.reset_peak_memory_stats()
    return torch.device("cuda:0")


def _check_execution_budget(started: float, device: torch.device) -> None:
    if time.monotonic() - started > FINALIZE_HARD_SECONDS:
        raise RuntimeError("finalizer exceeded 2400-second hard limit")
    if device.type == "cuda" and torch.cuda.max_memory_allocated() > HARD_MEMORY_BYTES:
        raise RuntimeError("finalizer exceeded 22-GiB allocated-memory limit")


class GuardedReplay:
    """Forward-only proxy; the trained model's methods and state stay intact."""
    def __init__(self, model, callback):
        self.model, self.callback, self.calls, self.rows = model, callback, 0, 0
    @property
    def name(self): return self.model.name
    def to(self, device): self.model.to(device); return self
    def eval(self): self.model.eval(); return self
    def forward_last(self, x, bank, mask=None):
        self.callback(self.calls,self.rows)
        value = self.model.forward_last(x,bank,mask)
        self.calls += 1; self.rows += len(x)
        self.callback(self.calls,self.rows)
        return value


def run(run_root: Path, output: Path, completion_sha256: str, *, device: str = "cuda:0",
        physical_gpu: int = 0, threads: int = 1, allow_cpu_for_test: bool = False) -> dict[str, Any]:
    """Explicitly gated selected+endpoint strict-EMA export and native scoring."""
    started = time.monotonic()
    execution_device = _execution_gate(device, physical_gpu=physical_gpu, threads=threads,
                                       allow_cpu_for_test=allow_cpu_for_test)
    _check_execution_budget(started, execution_device)
    frozen = prepare(run_root, output, completion_sha256)
    _check_execution_budget(started, execution_device)
    authority_pre = _source_minival_authority()  # deliberately before any model construction
    freeze_path = Path(output) / "selection_freeze.json"
    freeze_sha = sha(freeze_path)
    audit = frozen["audit"]
    exports, scores, archives = {}, {}, {}
    for arm in ARMS:
        selected = audit["selected_primary_ema_epoch"][arm]
        for label, epoch in (("selected", selected), ("endpoint24", EPOCHS)):
            key = f"{arm}_{label}_epoch_{epoch:03d}"
            model = _factory(arm)
            checkpoint = Path(run_root) / arm / f"epoch_{epoch:03d}.pt"
            expected_checkpoint_sha = audit["epoch_receipt_evidence"][arm]["epoch_artifact_hashes"][epoch]["checkpoint_sha256"]
            if sha(checkpoint) != expected_checkpoint_sha:
                raise RuntimeError("checkpoint changed after completion selection freeze")
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            contract = checkpoint_contract(payload, model, arm=arm, epoch=epoch, audit=audit)
            export = export_plain_ema_strict(payload, model, Path(output) / f"{key}_plain_ema_state.pt", arm=arm)
            replay = _factory(arm)
            replay.load_state_dict(torch.load(export["path"], map_location="cpu", weights_only=True), strict=True)
            # The legacy native scorer has no callback parameter.  Bind a tiny
            # proxy so the finalizer's wall-clock/memory guard runs before
            # every actual model forward, rather than merely between archives.
            last_live = [-1]
            def guard(calls, rows):
                if execution_device.type == "cuda": torch.cuda.synchronize()
                _check_execution_budget(started, execution_device)
                if (calls == 0 or calls % 16 == 0) and last_live[0] != calls:
                    _atomic_json(Path(output)/"live.json", {"status":"SCORING","artifact":key,"forward_calls":calls,
                        "rows":rows,"elapsed_seconds":time.monotonic()-started})
                    last_live[0] = calls
            guarded = GuardedReplay(replay,guard)
            before_state = {name:value.detach().cpu().clone() for name,value in replay.state_dict().items()}
            score, archive = prior_finalizer.score_source_minival(guarded, execution_device)
            if any(not torch.equal(value, replay.state_dict()[name].detach().cpu()) for name,value in before_state.items()):
                raise RuntimeError("inference changed the strict plain-EMA state")
            _check_execution_budget(started, execution_device)
            score["metric"] = "variance_weighted_r2_fp64"
            archive_equal, archive_pooled = _recompute_archive_score(archive)
            if abs(float(score["equal_session_r2"]) - archive_equal) > 1e-10 or abs(float(score["pooled_r2"]) - archive_pooled) > 1e-10:
                raise RuntimeError("reported scorer values disagree with FP64 native archive recomputation")
            score["equal_session_r2"], score["pooled_r2"] = archive_equal, archive_pooled
            score["recorded_ema_equal_session_r2"] = _ledger_value(_read_json(Path(run_root) / "selection_summary.json")["history"][arm]["EMA"], epoch)
            score["reproduction_delta"] = float(score["equal_session_r2"] - score["recorded_ema_equal_session_r2"])
            if not math.isfinite(score["reproduction_delta"]) or abs(score["reproduction_delta"]) > 1e-5:
                raise RuntimeError("QueryAge recorded EMA score reproduction drift")
            archive_path = Path(output) / f"{key}_native_source_minival.npz"
            score["archive_path"], score["archive_sha256"] = str(archive_path), _write_archive(archive_path, archive)
            exports[key] = {"checkpoint_sha256": sha(checkpoint), "checkpoint_contract": contract, **export}
            scores[key], archives[key] = score, archive
    identity = _archive_identity(archives, authority_pre)
    audit_post = validate_completion(Path(run_root))
    authority_post = _source_minival_authority()
    if audit_post != audit or authority_post != authority_pre or sha(freeze_path) != freeze_sha:
        raise RuntimeError("completion/source/freeze authority changed during finalization")
    if any(sha(Path(item["path"])) != item["sha256"] for item in exports.values()):
        raise RuntimeError("plain-EMA export changed during scoring")
    if any(sha(Path(score["archive_path"])) != score["archive_sha256"] for score in scores.values()):
        raise RuntimeError("native source-minival archive changed during scoring")
    _check_execution_budget(started,execution_device)
    result = {**frozen, "status": "SOURCE_MINIVAL_SELECTION_DIAGNOSTIC_NOT_UNTOUCHED_GENERALIZATION",
              "execution_device": device, "physical_gpu": physical_gpu, "threads": threads,
              "hard_limits": {"seconds": FINALIZE_HARD_SECONDS, "cuda_peak_allocated_bytes": HARD_MEMORY_BYTES},
              "authority_pre_model": authority_pre, "authority_post": authority_post,
              "exports": exports, "scores": scores, "native_archive_identity": identity,
              "selection_freeze_sha256": freeze_sha,
              "parameter_updates":0, "no_checkpoint_deserialization_yet":False,
              "no_model_construction_yet":False, "no_scoring_yet":False,
              "elapsed_seconds":time.monotonic()-started,
              "exposure_qualification": "all 24 epoch EMA values and these native 1011 source-minival queries were selection-exposed"}
    _atomic_json(Path(output) / "receipt.json", result)
    return result


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--completion-sha256", required=True)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--device", choices=("cuda:0", "cpu"), default="cuda:0")
    parser.add_argument("--physical-gpu", choices=(0, 1), default=0, type=int)
    parser.add_argument("--threads", choices=(1,), default=1, type=int)
    parser.add_argument("--allow-cpu-for-test", action="store_true")
    args = parser.parse_args(argv)
    if args.run:
        return run(args.run_root, args.output, args.completion_sha256, device=args.device,
                   physical_gpu=args.physical_gpu, threads=args.threads, allow_cpu_for_test=args.allow_cpu_for_test)
    return prepare(args.run_root, args.output, args.completion_sha256)


if __name__ == "__main__":
    main()
