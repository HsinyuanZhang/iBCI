"""CPU synthetic-cache tests for the post-selection QueryAge ext4 evaluator."""
from __future__ import annotations

from dataclasses import replace
import copy
import json
from pathlib import Path
import time

import numpy as np
import pytest
import torch

from tfpd_exploration.src.m2_dual_track_v1.contracts import make_stub_bank
from tfpd_exploration.src.m2_queryage_family_v1 import evaluate_selected_ext4 as evaluator
from tfpd_exploration.src.m2_queryage_family_v1.model import make_paired_queryage_decoders


def _synthetic_authority(tmp_path, monkeypatch):
    sessions = ("ext-a", "ext-b", "ext-c", "ext-d")
    counts = {session: 2 for session in sessions}
    monkeypatch.setattr(evaluator.plan, "EXT4_SESSIONS", sessions)
    monkeypatch.setattr(evaluator.plan, "EXT4_EXPECTED_WINDOWS", counts)
    monkeypatch.setattr(evaluator, "EXT4_TOTAL", 8)
    cache = tmp_path / "cache"; (cache / "ext4").mkdir(parents=True)
    banks, rows = {}, []
    for index, session in enumerate(sessions):
        folder = cache / "ext4" / session; folder.mkdir()
        raw = np.zeros((51, 96), dtype=np.float32); raw[-1, index] = .25
        starts = np.asarray([0, 1], dtype=np.int64)
        target = np.asarray([[float(index), float(index + 1)], [float(index + 2), float(index + 3)]], dtype=np.float32)
        np.save(folder / "X_store.npy", raw); np.save(folder / "eligible_starts.npy", starts); np.save(folder / "target_store.npy", target)
        for name in ("e0_u.pt", "T.npy", "calib_activity.npy", "mapping.json", "provenance.json", "extra.json"):
            (folder / name).write_bytes(b"fixture")
        bank = replace(make_stub_bank(seed=100 + index), session_id=session, X_store=raw,
                       eligible_starts=starts, target_store=target)
        banks[session] = bank
        rows.append({"session": session, "window_count": len(starts),
                     "files": {name: evaluator.sha(folder / name) for name in evaluator.REQUIRED_CACHE},
                     "ordered_window_starts_sha256": evaluator.core.array_sha256(starts),
                     "target_sha256": evaluator.core.array_sha256(target)})
    monkeypatch.setattr(evaluator.data, "cache_root", lambda: cache)
    monkeypatch.setattr(evaluator.data, "load_session_bank", lambda surface, session, device: banks[session])
    selected = {}
    for arm, model in zip(("FLAT", "ROUTE"), make_paired_queryage_decoders(42), strict=True):
        path = tmp_path / f"{arm}.pt"; torch.save(model.state_dict(), path)
        selected[arm] = {"key": f"{arm}_selected_epoch_001", "epoch": 1, "path": str(path), "sha256": evaluator.sha(path)}
    baseline_rows = [{"session": session, "spint_r2": 0., "e8_r2": 0.} for session in sessions]
    return {"rows": rows, "selected": selected,
            "baseline": {"rows": baseline_rows, "spint_equal_session_r2": 0., "spint_pooled_r2": 0.,
                         "e8_equal_session_r2": 0., "e8_pooled_r2": 0.}}


def test_actual_queryage_plain_state_load_and_native_four_session_score(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    authority = _synthetic_authority(tmp_path, monkeypatch)
    results, archives = evaluator.score_selected(authority, torch.device("cpu"), time.monotonic())
    assert set(results) == set(archives) == {"FLAT", "ROUTE"}
    assert all(len(results[arm]["rows"]) == 4 for arm in results)
    for arrays in archives.values():
        assert arrays["prediction"].dtype == arrays["target"].dtype == np.float64
        assert arrays["prediction"].shape == arrays["target"].shape == (8, 2)
        assert arrays["start"].dtype == np.int64
        assert evaluator.validate_archive(arrays, authority) == evaluator.recompute_metrics(arrays)
    evaluator._paired_archive_identity(archives)


def test_archive_identity_rejects_target_reordering(tmp_path, monkeypatch):
    authority = _synthetic_authority(tmp_path, monkeypatch)
    _, archives = evaluator.score_selected(authority, torch.device("cpu"), time.monotonic())
    bad = {key: value.copy() for key, value in archives["FLAT"].items()}
    bad["target"] = bad["target"][::-1].copy()
    with pytest.raises(RuntimeError, match="sealed source identity"):
        evaluator.validate_archive(bad, authority)


def test_strict_real_four_session_2069_archive_geometry_and_fp64_rejection():
    """The production metric helper is fixed to the real ext4 four-session roster."""
    assert evaluator.EXT4_TOTAL == 2069
    parts = {key: [] for key in ("prediction", "target", "start", "session")}
    for offset, (session, count) in enumerate(evaluator.plan.EXT4_EXPECTED_WINDOWS.items()):
        target = np.arange(count * 2, dtype=np.float32).reshape(count, 2) + offset
        parts["target"].append(target.astype(np.float64)); parts["prediction"].append((target * .9).astype(np.float64))
        parts["start"].append(np.arange(count, dtype=np.int64)); parts["session"].append(np.asarray([session] * count))
    arrays = {key: np.concatenate(value) for key, value in parts.items()}
    metrics = evaluator.recompute_metrics(arrays)
    assert set(metrics) == {"equal_session_r2", "pooled_r2"}
    arrays["session"] = arrays["session"].astype("S32")
    with pytest.raises(RuntimeError, match="geometry"):
        evaluator.recompute_metrics(arrays)


def test_prepare_authorize_run_cpu_fixture_with_real_queryage_forwards(tmp_path, monkeypatch):
    """Run consumes a durable preflight and performs the real model path, not a mocked scorer."""
    authority = _synthetic_authority(tmp_path, monkeypatch)
    output = (tmp_path / "out").resolve(); output.mkdir()
    bound = {**copy.deepcopy(authority), "schema": evaluator.SCHEMA, "output": str(output)}
    monkeypatch.setattr(evaluator, "authority", lambda *args, **kwargs: {**copy.deepcopy(authority), "schema": evaluator.SCHEMA, "output": None})
    preflight = {"schema": evaluator.SCHEMA, "status": "PREPARED_NO_MODEL_OR_EXT4_FORWARD", "authority": bound}
    evaluator._atomic_json(output / "preflight.json", preflight)
    authorization = (tmp_path / "authorization.json").resolve()
    authorization.write_text(json.dumps({"status": "ROOT_REVIEW_GO", "authority": bound,
                                         "preflight_sha256": evaluator.sha(output / "preflight.json")}), encoding="utf-8")
    monkeypatch.setenv(evaluator.GO_ENV, "1")
    result = evaluator.run((tmp_path / "run").resolve(), (tmp_path / "finalizer").resolve(), output, "a" * 64,
                           authorization=authorization, authorization_sha256=evaluator.sha(authorization),
                           physical_gpu=0, allow_cpu_for_test=True)
    assert result["pre"] == result["post"] == bound
    assert set(result["arms"]) == {"FLAT", "ROUTE"}
    assert (output / "receipt.json").is_file()


def test_export_mutation_and_timeout_reject_before_forward(tmp_path, monkeypatch):
    authority = _synthetic_authority(tmp_path, monkeypatch)
    Path(authority["selected"]["FLAT"]["path"]).write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="immediately before"):
        evaluator.score_selected(authority, torch.device("cpu"), time.monotonic())
    authority = _synthetic_authority(tmp_path / "timeout", monkeypatch)
    monkeypatch.setattr(evaluator, "_load_model", lambda *args: pytest.fail("guard should run before model forward/load"))
    with pytest.raises(TimeoutError, match="600"):
        evaluator.score_selected(authority, torch.device("cpu"), time.monotonic() - evaluator.HARD_SECONDS - 1)


def test_selected_export_json_and_source_archive_bindings_reject_mutation(tmp_path, monkeypatch):
    """Only trainer/finalizer external calls are patched; JSON/hash bindings are real files."""
    root, out = (tmp_path / "run").resolve(), (tmp_path / "final").resolve()
    root.mkdir(); out.mkdir()
    audit = {"selected_primary_ema_epoch": {"FLAT": 1, "ROUTE": 1},
             "epoch_receipt_evidence": {
                 arm: {"epoch_artifact_hashes": {1: {"checkpoint_sha256": "d" * 64}}}
                 for arm in ("FLAT", "ROUTE")}}
    fresh = {"source": "authority"}
    monkeypatch.setattr(evaluator.finalizer, "validate_completion", lambda value: audit)
    monkeypatch.setattr(evaluator.finalizer, "_source_minival_authority", lambda: fresh)
    exports, scores = {}, {}
    for arm in ("FLAT", "ROUTE"):
        weights, archive = out / f"{arm}.pt", out / f"{arm}.npz"
        weights.write_bytes(arm.encode()); archive.write_bytes((arm + "archive").encode())
        key = f"{arm}_selected_epoch_001"
        exports[key] = {"path": str(weights), "sha256": evaluator.sha(weights)}
        scores[key] = {"archive_path": str(archive), "archive_sha256": evaluator.sha(archive)}
    freeze = {"schema": evaluator.finalizer.FINALIZER_SCHEMA, "status": "SELECTION_FROZEN_PRE_EXPORT_PRE_SCORE", "audit": audit}
    evaluator._atomic_json(out / "selection_freeze.json", freeze)
    receipt = {"schema": evaluator.finalizer.FINALIZER_SCHEMA,
               "status": "SOURCE_MINIVAL_SELECTION_DIAGNOSTIC_NOT_UNTOUCHED_GENERALIZATION", "audit": audit,
               "authority_pre_model": fresh, "authority_post": fresh, "selection_freeze_sha256": evaluator.sha(out / "selection_freeze.json"),
               "exports": exports, "scores": scores}
    evaluator._atomic_json(out / "receipt.json", receipt)
    actual_audit, _, selected = evaluator._selected_exports(root, out, evaluator.sha(out / "receipt.json"))
    assert set(selected) == {"FLAT", "ROUTE"}
    assert actual_audit == json.loads(json.dumps(audit))
    assert set(actual_audit["epoch_receipt_evidence"]["FLAT"]["epoch_artifact_hashes"]) == {"1"}
    Path(scores["ROUTE_selected_epoch_001"]["archive_path"]).write_bytes(b"mutated")
    with pytest.raises(RuntimeError, match="source archive"):
        evaluator._selected_exports(root, out, evaluator.sha(out / "receipt.json"))
