"""Freeze development selections, then permit exactly one audited final-score pass.

This module deliberately has no training, hyperparameter-search, or development-data
entry point.  The only final-NWB opening path is ``score_final`` after a cryptographic
selection seal has been checked.
"""
from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from . import data, protocol
from .common import (RECIPE, SCHEMA, aggregate_scores, atomic_json, digest, fresh_directory, load_records,
                     record_binding, score_predictions, sha256, verify_stats)
from .final_access import (FA_CELLS, FINAL_SCORE_CELLS, REQUIRED_CELLS,
                           SUPPLEMENTAL_FULL_CELLS, FinalAccess, SEAL_SCHEMA,
                           SEAL_STATUS, canonical_payload_sha256)

NEURAL_CELLS = ("full_sua", "full_pmua", "activity_sua", "activity_pmua", "raw_set_sua", "raw_set_pmua")
BASELINE_METHODS = ("wf_zs_h0", "diag_z_wf", "coral_wf", "aligned_fa_wf", "aligned_fa_stable_wf", "wf_fss_sua", "wf_fss_pmua")
BASELINE_CELL = {"wf_zs_h0": "wf_zs_h0_pmua", "diag_z_wf": "diag_z_wf_pmua", "coral_wf": "coral_wf_pmua",
                 "aligned_fa_wf": "aligned_fa_wf_pmua", "aligned_fa_stable_wf": "aligned_fa_stable_wf_pmua",
                 "wf_fss_sua": "wf_fss_sua", "wf_fss_pmua": "wf_fss_pmua"}


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _formal(value: Mapping[str, Any], *, label: str) -> None:
    if value.get("status") != "FORMAL" or value.get("final_sessions_opened") != 0:
        raise ValueError(f"{label} is not a formal no-final artifact")
    if value.get("status") == "SMOKE":
        raise ValueError(f"SMOKE artifact cannot be sealed: {label}")


def _add_artifact(artifacts: dict[str, str], path: Path) -> str:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    key = str(path)
    actual = sha256(path)
    old = artifacts.setdefault(key, actual)
    if old != actual:
        raise RuntimeError(f"artifact changed while sealing: {path}")
    return key


def _prepared_binding(path: Path, artifacts: dict[str, str]) -> dict[str, Any]:
    path = Path(path).resolve()
    receipt = _read_json(path)
    if receipt.get("schema") != "dandi688_bench_v2_prepared_v1" or receipt.get("final_sessions_opened") != 0:
        raise ValueError("prepared receipt must be a no-final DANDI v2 cache")
    if receipt.get("protocol") != protocol.protocol_dict() or receipt.get("include_dev") is not True:
        raise ValueError("prepared receipt protocol/development coverage mismatch")
    sessions = receipt.get("sessions")
    expected = set(protocol.TRAIN_SESSIONS) | set(protocol.DEV_SESSIONS)
    if not isinstance(sessions, dict) or set(sessions) != expected or len(sessions) != 24:
        raise ValueError("prepared receipt does not contain exactly the 18+6 authorized cache")
    if any(item.get("split") not in {"train", "dev"} for item in sessions.values()):
        raise ValueError("prepared receipt contains an invalid split")
    if any(row.get("split") == "final" for row in receipt.get("access_log", ())):
        raise ValueError("prepared receipt records final access")
    for session_id, item in sessions.items():
        reps = item.get("representations")
        if not isinstance(reps, dict) or set(reps) != {"sua", "pmua"}:
            raise ValueError(f"prepared receipt lacks paired cache hashes: {session_id}")
        for rep, detail in reps.items():
            cache_file = path.parent / str(detail.get("file", ""))
            if not cache_file.is_file() or detail.get("sha256") != sha256(cache_file):
                raise ValueError(f"prepared cache hash mismatch: {session_id}.{rep}")
            _add_artifact(artifacts, cache_file)
    return {"path": _add_artifact(artifacts, path), "sha256": sha256(path),
            "session_roster_sha256": digest({name: sessions[name] for name in sorted(sessions)})}


def _verify_execution_hashes(receipt: Mapping[str, Any], *, kind: str) -> None:
    from .provenance import verify_execution_hashes
    verify_execution_hashes(receipt, kind=kind)


def _cell_spec(cell: str) -> tuple[str, str, str, int]:
    """Return canonical base cell, arm, representation and frozen seed."""
    if cell in SUPPLEMENTAL_FULL_CELLS:
        base, seed = cell.rsplit("_s", 1)
        seed_value = int(seed)
    else:
        base, seed_value = cell, 42
    specs = {
        "full_sua": ("full", "sua"), "full_pmua": ("full", "pmua"),
        "activity_sua": ("activity", "sua"), "activity_pmua": ("activity", "pmua"),
        "raw_set_sua": ("raw_set", "sua"), "raw_set_pmua": ("raw_set", "pmua"),
    }
    if base in specs:
        arm, representation = specs[base]
        return base, arm, representation, seed_value
    # Non-neural score cells have no arm or seed, but their representation is
    # still explicit in the frozen cell name.
    if base.endswith("_sua"):
        return base, "", "sua", seed_value
    if base.endswith("_pmua"):
        return base, "", "pmua", seed_value
    raise ValueError(f"unknown final score cell: {cell}")


def _torch_payload(path: Path, *, label: str) -> dict[str, Any]:
    import torch
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"{label} is not a checkpoint payload")
    return payload


def _same_state(left: Mapping[str, Any], right: Mapping[str, Any], *, label: str) -> None:
    import torch
    if set(left) != set(right):
        raise ValueError(f"{label} tensor key set differs")
    for name in left:
        if not isinstance(left[name], torch.Tensor) or not isinstance(right[name], torch.Tensor) or not torch.equal(left[name], right[name]):
            raise ValueError(f"{label} tensor differs: {name}")


def _neural_entry(cell: str, path: Path, artifacts: dict[str, str], *, prepared_cache: Path, expected_seed: int = 42) -> dict[str, Any]:
    base_cell, arm, representation, cell_seed = _cell_spec(cell)
    if cell_seed != expected_seed:
        raise ValueError(f"{cell} has an invalid frozen seed")
    selection_path = Path(path).resolve()
    selection = _read_json(selection_path)
    _formal(selection, label=f"{cell} selection")
    if selection.get("schema") != SCHEMA + "_selection":
        raise ValueError(f"{cell} is not a learned decoder selection")
    checkpoint = Path(selection.get("checkpoint", "")).resolve()
    if not checkpoint.is_file() or selection.get("checkpoint_sha256") != sha256(checkpoint):
        raise ValueError(f"{cell} selected checkpoint hash mismatch")
    receipt_path, protocol_path, stats_path = selection_path.parent / "receipt.json", selection_path.parent / "protocol.json", selection_path.parent / "source_stats.json"
    receipt, run_protocol = _read_json(receipt_path), _read_json(protocol_path)
    _formal(receipt, label=f"{cell} training receipt")
    _formal(run_protocol, label=f"{cell} training protocol")
    _verify_execution_hashes(receipt, kind=f"{cell} training")
    source, development = load_records(prepared_cache, representation, "train"), load_records(prepared_cache, representation, "dev")
    if receipt.get("protocol") != protocol.protocol_dict() or run_protocol.get("protocol") != protocol.protocol_dict():
        raise ValueError(f"{cell} protocol mismatch")
    if receipt.get("representation") != representation or receipt.get("arm") != arm:
        raise ValueError(f"{cell} representation/arm does not match its matrix cell")
    if receipt.get("stage") != "train" or receipt.get("seed") != expected_seed or receipt.get("completed") is not True:
        raise ValueError(f"{cell} is not completed frozen-seed decoder training")
    if (receipt.get("source_sessions") != list(protocol.TRAIN_SESSIONS) or receipt.get("source_binding") != record_binding(source)
            or receipt.get("recipe") != RECIPE):
        raise ValueError(f"{cell} source roster/recipe mismatch")
    if receipt.get("global_step") != 24 * 3165 or receipt.get("actual_budget") != {"segments": 24, "updates_per_segment": 3165, "batch": 32}:
        raise ValueError(f"{cell} does not have the full 75,960-step budget")
    stats = _read_json(stats_path)
    verify_stats(stats, source)
    if receipt.get("source_stats_sha256") != stats.get("sha256"):
        raise ValueError(f"{cell} source-statistics binding mismatch")
    curve = receipt.get("segments")
    if not isinstance(curve, list) or len(curve) != 24:
        raise ValueError(f"{cell} lacks the complete 24-segment development curve")
    scores: list[float] = []
    selected_index = None
    for index, row in enumerate(curve, start=1):
        if not isinstance(row, dict) or row.get("segment") != index or row.get("global_step") != index * 3165 or not _finite_number(row.get("mean_loss")):
            raise ValueError(f"{cell} training curve is malformed")
        score = _validate_dev_metrics(row.get("development"), development, label=f"{cell} segment {index}")
        scores.append(score)
        if row.get("checkpoint") == checkpoint.name and row.get("checkpoint_sha256") == sha256(checkpoint):
            selected_index = index - 1
    if selected_index is None:
        raise ValueError(f"{cell} selection checkpoint is absent from formal training curve")
    if selection.get("rule") != "earliest_max_equal_session_dev_r2" or selection.get("dev_sessions") != list(protocol.DEV_SESSIONS) or selected_index != scores.index(max(scores)) or selection.get("mean_dev_r2") != scores[selected_index]:
        raise ValueError(f"{cell} selection is not the earliest complete-dev maximum")
    payload = _torch_payload(checkpoint, label=f"{cell} selected checkpoint")
    if (payload.get("schema") != SCHEMA + "_checkpoint" or payload.get("status") != "FORMAL"
            or payload.get("stage") != "train" or payload.get("arm") != arm or payload.get("representation") != representation
            or payload.get("seed") != expected_seed or payload.get("global_step") != (selected_index + 1) * 3165
            or payload.get("protocol") != protocol.protocol_dict() or payload.get("recipe") != RECIPE
            or payload.get("source_stats") != stats or not isinstance(payload.get("model_state"), dict)):
        raise ValueError(f"{cell} selected checkpoint payload mismatch")
    return {"status": "SELECTED", "kind": "neural", "selection_path": _add_artifact(artifacts, selection_path),
            "artifact_paths": [_add_artifact(artifacts, checkpoint), _add_artifact(artifacts, receipt_path),
                               _add_artifact(artifacts, protocol_path), _add_artifact(artifacts, stats_path)],
            "checkpoint_sha256": sha256(checkpoint), "selection_sha256": sha256(selection_path),
            "representation": representation, "arm": arm, "seed": expected_seed}

def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _validate_dev_metrics(metrics: Any, records: list[Any], *, label: str) -> float:
    """Validate the complete equal-session development score receipt."""
    if not isinstance(metrics, dict) or metrics.get("metric") != "physical_velocity_sklearn_variance_weighted_equal_session_mean":
        raise ValueError(f"{label} does not use the preregistered equal-session metric")
    sessions = metrics.get("sessions")
    if metrics.get("n_sessions") != len(protocol.DEV_SESSIONS) or not isinstance(sessions, list) or len(sessions) != len(protocol.DEV_SESSIONS):
        raise ValueError(f"{label} lacks all six development dates")
    by_id = {getattr(record, "session_id"): record for record in records}
    if set(by_id) != set(protocol.DEV_SESSIONS):
        raise ValueError(f"{label} loaded an invalid development roster")
    if {row.get("session_id") for row in sessions if isinstance(row, dict)} != set(protocol.DEV_SESSIONS):
        raise ValueError(f"{label} has incomplete or duplicate development dates")
    values = []
    for row in sessions:
        if not isinstance(row, dict) or not _finite_number(row.get("r2")):
            raise ValueError(f"{label} contains non-finite per-date metrics")
        if not isinstance(row.get("r2_per_output"), list) or not row["r2_per_output"] or not all(_finite_number(v) for v in row["r2_per_output"]):
            raise ValueError(f"{label} contains non-finite per-output metrics")
        record = by_id[row["session_id"]]
        if row.get("n_queries") != len(record.query_indices) or row.get("query_indices_sha256") != record.metadata["array_sha256"]["query_indices"] or row.get("velocity_sha256") != record.metadata["array_sha256"]["velocity"]:
            raise ValueError(f"{label} development metrics are not bound to the prepared cache")
        values.append(float(row["r2"]))
    mean = metrics.get("mean_r2")
    if not _finite_number(mean) or float(mean) != sum(values) / len(values):
        raise ValueError(f"{label} aggregate is not the exact equal-session mean")
    return float(mean)


def _validate_candidates(candidates: Any, expected: list[dict[str, Any]], records: list[Any], *, label: str) -> tuple[int | None, list[Any]]:
    if not isinstance(candidates, list) or [row.get("config") if isinstance(row, dict) else None for row in candidates] != expected:
        raise ValueError(f"{label} candidate grid differs from the frozen grid")
    if len({digest(config) for config in expected}) != len(expected):
        raise RuntimeError(f"internal duplicate frozen configs for {label}")
    winners: list[tuple[int, float]] = []
    for index, row in enumerate(candidates):
        if not isinstance(row, dict) or not isinstance(row.get("eligible"), bool):
            raise ValueError(f"{label} candidate eligibility is malformed")
        if row["eligible"]:
            mean = _validate_dev_metrics(row.get("scores"), records, label=f"{label} candidate {index}")
            if not _finite_number(row.get("score")) or float(row["score"]) != mean:
                raise ValueError(f"{label} candidate {index} score differs from aggregate")
            winners.append((index, mean))
        elif not isinstance(row.get("reason"), str) or not row["reason"]:
            raise ValueError(f"{label} ineligible candidate lacks a reason")
    if not winners:
        return None, candidates
    return max(winners, key=lambda item: item[1])[0], candidates


def _baseline_entries(path: Path, artifacts: dict[str, str], *, prepared_cache: Path) -> dict[str, dict[str, Any]]:
    selection_path = Path(path).resolve()
    selection = _read_json(selection_path)
    if selection.get("schema") != "dandi688_v2_cpu_baseline_selection" or selection.get("status") != "DEV_SELECTED":
        raise ValueError("baseline selection is not a formal DEV_SELECTED result")
    if selection.get("final_loaded") is not False or selection.get("protocol") not in (None, protocol.protocol_dict()):
        raise ValueError("baseline selection final/protocol provenance mismatch")
    receipt_path = selection_path.parent / "receipt.json"
    receipt = _read_json(receipt_path)
    if receipt.get("status") != "DEV_SELECTED" or receipt.get("final_loaded") is not False:
        raise ValueError("baseline receipt is not formal no-final selection")
    _verify_execution_hashes(selection, kind="baseline")
    methods, artifacts_by_method = selection.get("methods"), selection.get("artifacts")
    if set(methods or ()) != set(BASELINE_METHODS) or not isinstance(artifacts_by_method, dict):
        raise ValueError("baseline selection does not cover exact required method roster")
    if selection.get("sha256") != digest({k: v for k, v in selection.items() if k != "sha256"}) or receipt.get("selection_sha256") != sha256(selection_path):
        raise ValueError("baseline selection/receipt hash binding mismatch")
    source, dev = selection.get("source"), selection.get("dev")
    pmua_source, pmua_dev = load_records(prepared_cache, "pmua", "train"), load_records(prepared_cache, "pmua", "dev")
    sua_dev = load_records(prepared_cache, "sua", "dev")
    if source != record_binding(pmua_source) or dev != record_binding(pmua_dev):
        raise ValueError("baseline selection source/development roster mismatch")
    grid = selection.get("candidate_grid")
    if not isinstance(grid, dict) or set(grid) != set(BASELINE_METHODS):
        raise ValueError("baseline selection does not contain the complete candidate grid")
    selection_key, receipt_key = _add_artifact(artifacts, selection_path), _add_artifact(artifacts, receipt_path)
    output: dict[str, dict[str, Any]] = {}
    for method in BASELINE_METHODS:
        cell = BASELINE_CELL[method]
        info = artifacts_by_method.get(method)
        if not isinstance(info, dict):
            raise ValueError(f"baseline selection has no entry for {cell}")
        if info.get("status") == "SMOKE":
            raise ValueError(f"SMOKE baseline cannot be sealed: {cell}")
        internal = "wf_fss" if method.startswith("wf_fss_") else method
        expected = __import__("dandi688_bench_v2.baseline_runner", fromlist=["_grid"])._grid(internal, False)
        dev_records = sua_dev if method == "wf_fss_sua" else pmua_dev
        winner_index, candidates = _validate_candidates(grid[method], expected, dev_records, label=method)
        if info.get("status") == "UNAVAILABLE":
            if cell not in FA_CELLS or not info.get("reason") or winner_index is not None:
                raise ValueError(f"only rank-gated FA cells may be unavailable: {cell}")
            output[cell] = {"status": "UNAVAILABLE", "kind": "baseline", "reason": str(info["reason"]),
                            "grid_artifact": selection_key, "selection_sha256": sha256(selection_path)}
            continue
        if winner_index is None or info.get("selected") != candidates[winner_index]:
            raise ValueError(f"baseline selected candidate is not the earliest development maximum: {cell}")
        model_name, predictions_name = info.get("model"), info.get("predictions")
        if not isinstance(model_name, str) or not isinstance(predictions_name, str):
            raise ValueError(f"selected baseline lacks model/dev prediction paths: {cell}")
        model, predictions = (selection_path.parent / model_name).resolve(), (selection_path.parent / predictions_name).resolve()
        if info.get("model_sha256") != sha256(model) or info.get("predictions_sha256") != sha256(predictions):
            raise ValueError(f"baseline artifact hash mismatch: {cell}")
        if method == "wf_fss_sua":
            import numpy as np
            with np.load(predictions, allow_pickle=False) as archive:
                if set(archive.files) != set(protocol.DEV_SESSIONS) or any(len(archive[name]) != len(next(r for r in sua_dev if r.session_id == name).query_indices) for name in archive.files):
                    raise ValueError("WF-FSS SUA predictions do not bind to the prepared dev roster")
        output[cell] = {"status": "SELECTED", "kind": "baseline", "selection_path": selection_key,
                        "artifact_paths": [_add_artifact(artifacts, model), _add_artifact(artifacts, predictions), receipt_key],
                        "selection_sha256": sha256(selection_path), "model_sha256": sha256(model)}
    return output


def _static_entries(path: Path, artifacts: dict[str, str], neural_entries: Mapping[str, dict[str, Any]], *, prepared_cache: Path) -> dict[str, dict[str, Any]]:
    path = Path(path).resolve()
    receipt = _read_json(path)
    if receipt.get("schema") != SCHEMA + "_development_replay" or receipt.get("status") != "FORMAL" or receipt.get("final_sessions_opened") != 0:
        raise ValueError("static-controls receipt is not formal no-final development replay")
    variants = receipt.get("variants")
    if receipt.get("protocol") != protocol.protocol_dict():
        raise ValueError("static-controls protocol mismatch")
    _verify_execution_hashes(receipt, kind="static")
    source_binding, dev_binding = receipt.get("source_binding"), receipt.get("dev_binding")
    source, development = load_records(prepared_cache, "pmua", "train"), load_records(prepared_cache, "pmua", "dev")
    if source_binding != record_binding(source) or dev_binding != record_binding(development):
        raise ValueError("static-controls source/development roster mismatch")
    if not isinstance(variants, dict):
        raise ValueError("static-controls receipt lacks variants")
    seal_key = _add_artifact(artifacts, path)
    raw = neural_entries["raw_set_pmua"]
    checkpoint_sha = raw["checkpoint_sha256"]
    result: dict[str, dict[str, Any]] = {}
    for cell, kind in (("raw_set_diag_z_pmua", "diag_z"), ("raw_set_coral_pmua", "coral")):
        detail = variants.get(kind)
        if not isinstance(detail, dict) or detail.get("checkpoint_sha256") != checkpoint_sha:
            raise ValueError(f"static-controls {kind} is not bound to raw_set_pmua checkpoint")
        readings, index = detail.get("candidates"), detail.get("selected_index")
        expected = [{"shrinkage": .1}] if kind == "diag_z" else [{"shrinkage": value} for value in (0., .1, .5, 1.)]
        if not isinstance(readings, list) or [row.get("shrinkage") if isinstance(row, dict) else None for row in readings] != [row["shrinkage"] for row in expected] or not isinstance(index, int):
            raise ValueError(f"static-controls {kind} has invalid dev selection")
        scores = [_validate_dev_metrics(row.get("metrics") if isinstance(row, dict) else None, development,
                                        label=f"static {kind} candidate {candidate_index}")
                  for candidate_index, row in enumerate(readings)]
        if not 0 <= index < len(scores) or index != max(range(len(scores)), key=scores.__getitem__):
            raise ValueError(f"static-controls {kind} selected index is not the earliest development maximum")
        config = readings[index]
        if not isinstance(config, dict) or "shrinkage" not in config:
            raise ValueError(f"static-controls {kind} lacks selected shrinkage")
        result[cell] = {"status": "SELECTED", "kind": "static_control", "selection_path": seal_key,
                        "artifact_paths": [seal_key, *raw["artifact_paths"]], "checkpoint_sha256": checkpoint_sha,
                        "adapter_kind": kind, "shrinkage": config["shrinkage"]}
    return result



def _encoder_entries(paths: Mapping[str, Path] | None, artifacts: dict[str, str], *, prepared_cache: Path) -> dict[str, dict[str, Any]]:
    if not isinstance(paths, Mapping) or set(paths) != {"sua", "pmua"}:
        raise ValueError("encoder_checkpoints must map exactly sua and pmua to exported encoder.pt files")
    result: dict[str, dict[str, Any]] = {}
    for representation, supplied in paths.items():
        encoder_path = Path(supplied).resolve()
        receipt_path = encoder_path.with_suffix(".json")
        pretrain_receipt_path = encoder_path.parent / "receipt.json"
        pretrain_protocol_path = encoder_path.parent / "protocol.json"
        stats_path = encoder_path.parent / "source_stats.json"
        receipt, pretrain, run_protocol, stats = (_read_json(receipt_path), _read_json(pretrain_receipt_path),
                                                   _read_json(pretrain_protocol_path), _read_json(stats_path))
        source = load_records(prepared_cache, representation, "train")
        verify_stats(stats, source)
        _formal(pretrain, label=f"{representation} encoder pretraining receipt")
        _formal(run_protocol, label=f"{representation} encoder pretraining protocol")
        _verify_execution_hashes(pretrain, kind=f"{representation} encoder pretraining")
        if (receipt.get("checkpoint_sha256") != sha256(encoder_path) or receipt.get("schema") != SCHEMA + "_encoder"
                or receipt.get("representation") != representation or receipt.get("status") != "FORMAL"
                or receipt.get("source_sessions") != list(protocol.TRAIN_SESSIONS) or receipt.get("source_stats_sha256") != stats.get("sha256")
                or receipt.get("global_step") != 24 * 3165 or receipt.get("encoder_seed") != 42
                or receipt.get("selection") != "fixed_final_ema_source_only" or receipt.get("protocol") != protocol.protocol_dict()
                or receipt.get("recipe") != RECIPE or receipt.get("encoder_line") != RECIPE["encoder_line"]
                or receipt.get("encoder_carrier_fusion") != RECIPE["encoder_carrier_fusion"]
                or receipt.get("encoder_film") is not RECIPE["encoder_film"] or receipt.get("final_sessions_opened") != 0):
            raise ValueError(f"{representation} exported encoder receipt mismatch")
        if (pretrain.get("stage") != "pretrain" or pretrain.get("arm") != "full" or pretrain.get("representation") != representation
                or pretrain.get("seed") != 42 or pretrain.get("completed") is not True or pretrain.get("source_sessions") != list(protocol.TRAIN_SESSIONS)
                or pretrain.get("source_binding") != record_binding(source) or pretrain.get("source_stats_sha256") != stats.get("sha256")
                or pretrain.get("protocol") != protocol.protocol_dict() or pretrain.get("recipe") != RECIPE
                or pretrain.get("encoder_line") != RECIPE["encoder_line"]
                or pretrain.get("encoder_carrier_fusion") != RECIPE["encoder_carrier_fusion"]
                or pretrain.get("encoder_film") is not RECIPE["encoder_film"]
                or pretrain.get("global_step") != 24 * 3165 or pretrain.get("actual_budget") != {"segments": 24, "updates_per_segment": 3165, "batch": 32}):
            raise ValueError(f"{representation} encoder pretraining provenance mismatch")
        curve = pretrain.get("segments")
        if not isinstance(curve, list) or len(curve) != 24 or any(not isinstance(row, dict) or row.get("segment") != index or row.get("global_step") != index * 3165 or not _finite_number(row.get("mean_loss")) or row.get("development") is not None for index, row in enumerate(curve, 1)):
            raise ValueError(f"{representation} encoder pretraining curve is incomplete or has development metrics")
        last_path = encoder_path.parent / str(curve[-1].get("checkpoint", ""))
        if not last_path.is_file() or curve[-1].get("checkpoint_sha256") != sha256(last_path):
            raise ValueError(f"{representation} encoder final pretraining checkpoint mismatch")
        encoded, final_checkpoint = _torch_payload(encoder_path, label=f"{representation} exported encoder"), _torch_payload(last_path, label=f"{representation} final pretraining checkpoint")
        if encoded.get("receipt_binding") != digest({k: v for k, v in receipt.items() if k != "checkpoint_sha256"}) or not isinstance(encoded.get("encoder_state"), dict):
            raise ValueError(f"{representation} exported encoder payload/receipt binding mismatch")
        if (final_checkpoint.get("schema") != SCHEMA + "_checkpoint" or final_checkpoint.get("stage") != "pretrain"
                or final_checkpoint.get("global_step") != 24 * 3165 or not isinstance(final_checkpoint.get("model_state"), dict)):
            raise ValueError(f"{representation} final pretraining payload mismatch")
        final_encoder = {name.removeprefix("encoder."): tensor for name, tensor in final_checkpoint["model_state"].items() if name.startswith("encoder.")}
        _same_state(encoded["encoder_state"], final_encoder, label=f"{representation} exported encoder versus final EMA")
        result[representation] = {"path": _add_artifact(artifacts, encoder_path), "sha256": sha256(encoder_path),
                                  "receipt_path": _add_artifact(artifacts, receipt_path), "pretrain_receipt_path": _add_artifact(artifacts, pretrain_receipt_path),
                                  "pretrain_protocol_path": _add_artifact(artifacts, pretrain_protocol_path), "source_stats_path": _add_artifact(artifacts, stats_path),
                                  "last_checkpoint_path": _add_artifact(artifacts, last_path), "encoder_state": encoded["encoder_state"]}
    return result


def _bind_full_encoders(entries: Mapping[str, dict[str, Any]], encoders: Mapping[str, dict[str, Any]]) -> None:
    for cell, entry in entries.items():
        if entry.get("arm") != "full":
            continue
        representation = str(entry["representation"])
        checkpoint = next(Path(path) for path in entry["artifact_paths"] if str(path).endswith(".pt"))
        payload = _torch_payload(checkpoint, label=f"{cell} full decoder")
        encoder_state = {name.removeprefix("encoder."): tensor for name, tensor in payload["model_state"].items() if name.startswith("encoder.")}
        _same_state(encoders[representation]["encoder_state"], encoder_state, label=f"{cell} frozen encoder")
        if payload.get("encoder_checkpoint_sha256") != encoders[representation]["sha256"]:
            raise ValueError(f"{cell} does not name the sealed representation encoder")
        entry["encoder_checkpoint_sha256"] = encoders[representation]["sha256"]
        entry["encoder_path"] = encoders[representation]["path"]

def seal_selection(dest: Path, *, prepared_receipt: Path, neural_selections: Mapping[str, Path],
                   baseline_selection: Path, static_controls_receipt: Path,
                   supplemental_full_selections: Mapping[str, Path] | None = None,
                   encoder_checkpoints: Mapping[str, Path] | None = None) -> dict[str, Any]:
    """Create immutable input for final scoring from completed development choices."""
    destination = fresh_directory(dest)
    if set(neural_selections) != set(NEURAL_CELLS):
        raise ValueError("neural_selections must contain exactly the six preregistered NN cells")
    artifacts: dict[str, str] = {}
    prepared = _prepared_binding(prepared_receipt, artifacts)
    prepared_cache = Path(prepared_receipt).resolve().parent
    encoders = _encoder_entries(encoder_checkpoints, artifacts, prepared_cache=prepared_cache)
    selections = {cell: _neural_entry(cell, neural_selections[cell], artifacts, prepared_cache=prepared_cache) for cell in NEURAL_CELLS}
    supplemental = dict(supplemental_full_selections or {})
    if set(supplemental) != set(SUPPLEMENTAL_FULL_CELLS):
        raise ValueError("supplemental_full_selections must contain exactly the four frozen full seed43/44 cells")
    supplemental_entries = {cell: _neural_entry(cell, supplemental[cell], artifacts, prepared_cache=prepared_cache, expected_seed=int(cell[-2:]))
                            for cell in supplemental}
    _bind_full_encoders({**selections, **supplemental_entries}, encoders)
    selections.update(_baseline_entries(baseline_selection, artifacts, prepared_cache=prepared_cache))
    selections.update(_static_entries(static_controls_receipt, artifacts, selections, prepared_cache=prepared_cache))
    if set(selections) != set(REQUIRED_CELLS):
        raise RuntimeError("seal selection coverage differs from frozen final matrix")
    protocol_path = Path(protocol.__file__).resolve()
    roster = {"train": list(protocol.TRAIN_SESSIONS), "dev": list(protocol.DEV_SESSIONS), "final": list(protocol.FINAL_SESSIONS)}
    from .provenance import bind_execution_sources
    execution_sources = bind_execution_sources(artifacts)
    payload: dict[str, Any] = {"schema": SEAL_SCHEMA, "status": SEAL_STATUS, "model_schema": SCHEMA, "protocol": protocol.protocol_dict(),
        "protocol_source": {"path": _add_artifact(artifacts, protocol_path), "sha256": sha256(protocol_path)},
        "roster": roster, "roster_sha256": digest(roster), "prepared": prepared, "encoders": {key: {k: v for k, v in value.items() if k != "encoder_state"} for key, value in encoders.items()},
        "required_cells": list(REQUIRED_CELLS), "cell_selections": selections, "artifacts": dict(sorted(artifacts.items())),
        "supplemental_full_selections": supplemental_entries, "execution_sources": execution_sources,
        "final_sessions_opened": 0}
    payload["sha256"] = canonical_payload_sha256(payload)
    atomic_json(destination / "selection_seal.json", payload)
    return payload


def _default_score_cell(cell: str, final_records: Mapping[str, Mapping[str, data.SessionData]],
                        entry: Mapping[str, Any], *, prepared_cache: Path, device: str = "cpu") -> Mapping[str, Any]:
    """Formal fixed-model scorer.  It has no candidate-grid or dev-selection path."""
    _base_cell, _arm, representation, _seed = _cell_spec(cell)
    records = [final_records[name][representation] for name in protocol.FINAL_SESSIONS]
    if entry["kind"] == "neural" or entry["kind"] == "static_control":
        from .baselines import fitstatic_adapter
        from .training import load_trained_model, predict_network
        checkpoint = next(Path(item) for item in entry["artifact_paths"] if item.endswith(".pt"))
        model, stats = load_trained_model(checkpoint, device=device)
        source = load_records(prepared_cache, representation, "train")
        verify_stats(stats, source)
        if entry["kind"] == "static_control":
            if model.arm != "raw_set" or representation != "pmua":
                raise RuntimeError("static control is not bound to raw-set PMUA model")
            scores, diagnostics, predictions = [], {}, {}
            for record in records:
                adapter = fitstatic_adapter(source, record, kind=str(entry["adapter_kind"]),
                                            shrinkage=float(entry["shrinkage"]),
                                            reference_session=protocol.TRAIN_SESSIONS[-1])
                prediction = predict_network(model, record, stats, adapter=adapter)
                scores.append(score_predictions(record, prediction))
                predictions[record.session_id] = prediction
                diagnostics[record.session_id] = {"prediction_sha256": digest(prediction), "adapter": adapter.diagnostics}
        else:
            scores, diagnostics, predictions = [], {}, {}
            for record in records:
                prediction = predict_network(model, record, stats)
                scores.append(score_predictions(record, prediction))
                predictions[record.session_id] = prediction
                diagnostics[record.session_id] = {"prediction_sha256": digest(prediction)}
        return {"selected": True, "metrics": aggregate_scores(scores), "per_session": diagnostics,
                "_predictions": predictions, "checkpoint_sha256": entry["checkpoint_sha256"]}
    from .baselines import fit_wf_fss, predict_baseline, predict_wf_fss
    import pickle
    artifact_paths = [Path(item) for item in entry["artifact_paths"]]
    model_path = next(item for item in artifact_paths if item.suffix == ".pkl")
    predictions, diagnostics, scores = {}, {}, []
    if cell.startswith("wf_fss_"):
        with model_path.open("rb") as handle:
            saved = pickle.load(handle)
        config = saved.get("selected_config") if isinstance(saved, dict) else None
        if not isinstance(config, dict):
            raise RuntimeError("sealed WF-FSS model lacks fixed selected config")
        for record in records:
            prediction = predict_wf_fss(fit_wf_fss(record, config), record)
            predictions[record.session_id] = prediction
            scores.append(score_predictions(record, prediction))
            diagnostics[record.session_id] = {"prediction_sha256": digest(prediction)}
    else:
        with model_path.open("rb") as handle:
            model = pickle.load(handle)
        for record in records:
            try:
                prediction = predict_baseline(model, record)
            except (ValueError, RuntimeError, __import__("numpy").linalg.LinAlgError) as error:
                if cell not in FA_CELLS:
                    raise
                diagnostics[record.session_id] = {"status": "UNAVAILABLE", "reason": str(error)}
                continue
            predictions[record.session_id] = prediction
            scores.append(score_predictions(record, prediction))
            diagnostics[record.session_id] = {"status": "SCORED", "prediction_sha256": digest(prediction)}
    if cell in FA_CELLS and len(predictions) != len(records):
        return {"selected": True, "status": "UNAVAILABLE", "metrics": None, "per_session": diagnostics,
                "coverage": list(protocol.FINAL_SESSIONS), "_predictions": predictions, "model_sha256": entry["model_sha256"]}
    return {"selected": True, "metrics": aggregate_scores(scores), "per_session": diagnostics,
            "_predictions": predictions, "model_sha256": entry["model_sha256"]}


def score_final(seal_path: Path, dest: Path, *, raw_root: Path = protocol.DEFAULT_RAW_ROOT, device: str = "cpu",
                score_cell: Callable[[str, Mapping[str, data.SessionData], Mapping[str, Any]], Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Open each final session once and score each sealed choice once.

    If ``score_cell`` is omitted, use the fixed built-in neural/baseline scorer.
    The optional callback exists only for resource-free synthetic verification.
    """
    destination = fresh_directory(dest)
    capability = FinalAccess.from_manifest(Path(seal_path))
    capability.validate()  # validate every sealed byte before first final raw open
    seal = _read_json(capability.manifest_path)
    if seal.get("final_sessions_opened") != 0:
        raise PermissionError("seal already records final access")
    prepared_cache = Path(seal["prepared"]["path"]).parent
    if score_cell is None:
        def score_cell(cell: str, records: Mapping[str, Mapping[str, data.SessionData]], entry: Mapping[str, Any]) -> Mapping[str, Any]:
            return _default_score_cell(cell, records, entry, prepared_cache=prepared_cache, device=device)
    claim = capability.manifest_path.parent / ".final_score_claim"
    try:
        with claim.open("x", encoding="utf-8") as handle:
            json.dump({"seal_sha256": capability.manifest_sha256, "status": "FINAL_SCORE_CLAIMED"}, handle, sort_keys=True)
    except FileExistsError as error:
        raise PermissionError("final selection seal has already been claimed for scoring") from error
    atomic_json(destination / "final_score_started.json", {"seal_sha256": capability.manifest_sha256,
                "status": "FINAL_SCORE_STARTED", "final_sessions_opened": 0})
    access_log: list[dict[str, Any]] = []
    final_records: dict[str, Mapping[str, data.SessionData]] = {}
    for session_id in protocol.FINAL_SESSIONS:
        capability.authorize(session_id)
        final_records[session_id] = data.load_pair(session_id, purpose="final", final_access=capability,
                                                    raw_root=raw_root, access_log=access_log)
    if [item["session_id"] for item in access_log] != list(protocol.FINAL_SESSIONS):
        raise RuntimeError("final roster was not opened exactly once")
    # Compare actual electrode-key tables with the sealed prepared source, rather than trusting indices alone.
    for representation in ("sua", "pmua"):
        source_reference = load_records(prepared_cache, representation, "train")[0].metadata["canonical_electrode_keys"]
        for session_id in protocol.FINAL_SESSIONS:
            if final_records[session_id][representation].metadata["canonical_electrode_keys"] != source_reference:
                raise RuntimeError(f"final canonical M1 electrode table mismatch: {session_id}.{representation}")
    rows: dict[str, Any] = {}
    all_selections = {**seal["cell_selections"], **seal["supplemental_full_selections"]}
    if set(all_selections) != set(FINAL_SCORE_CELLS) or len(all_selections) != len(FINAL_SCORE_CELLS):
        raise RuntimeError("final score matrix must contain exactly nineteen frozen cells")
    for cell, entry in all_selections.items():
        if entry["status"] == "UNAVAILABLE":
            rows[cell] = {"status": "UNAVAILABLE", "reason": entry["reason"]}
            continue
        capability.validate()
        result = dict(score_cell(cell, final_records, entry))
        if result.get("status") == "SMOKE" or result.get("selected") is False:
            raise RuntimeError("final scorer attempted non-formal/reselected output")
        predictions = result.pop("_predictions", None)
        prediction_receipt = None
        if predictions is not None:
            import numpy as np
            output = destination / f"{cell}.final_predictions.npz"
            np.savez_compressed(output, **predictions)
            prediction_receipt = {"path": str(output), "sha256": sha256(output),
                                  "sessions": {name: {"prediction_sha256": digest(value),
                                      "query_indices_sha256": final_records[name][_cell_spec(cell)[2]].metadata["array_sha256"]["query_indices"],
                                      "velocity_sha256": final_records[name][_cell_spec(cell)[2]].metadata["array_sha256"]["velocity"]}
                                      for name, value in predictions.items()}}
        rows[cell] = {"status": result.get("status", "SCORED"), "result": result, "predictions": prediction_receipt}
    if set(rows) != set(FINAL_SCORE_CELLS) or len(rows) != len(FINAL_SCORE_CELLS):
        raise RuntimeError("final score receipt does not cover exactly nineteen frozen cells")
    receipt = {"schema": "dandi688_v2_final_score", "status": "FINAL_SCORED", "seal_path": str(capability.manifest_path),
               "seal_sha256": capability.manifest_sha256, "roster": list(protocol.FINAL_SESSIONS), "results": rows,
               "access_log": access_log, "final_sessions_opened": len(access_log)}
    atomic_json(destination / "final_score_receipt.json", receipt)
    return receipt
