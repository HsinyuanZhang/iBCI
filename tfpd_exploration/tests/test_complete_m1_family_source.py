"""Synthetic post-completion authority tests; never load a trained model."""
import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from tfpd_exploration.src.family_runtime_v1 import complete_m1_family_source as proof


def write(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body))


def completed(root):
    history = []
    for epoch in range(1, 25):
        row = {"epoch": epoch}
        for arm in proof.ARMS:
            checkpoint = root / arm / f"epoch_{epoch:03d}.pt"
            checkpoint.parent.mkdir(exist_ok=True)
            checkpoint.write_bytes(f"{arm}:{epoch}".encode())
            row[arm] = {"epoch": epoch, "checkpoint": str(checkpoint), "checkpoint_sha256": proof.sha(checkpoint),
                        "ema": {"equal_session_mean_r2": .8 if epoch in (3, 5) else .1,
                                "n": proof.COUNT, "complete_all_sessions": True}}
        history.append(row)
    selected = {arm: proof.earliest(history, arm) for arm in proof.ARMS}
    endpoint = {arm: history[-1][arm] for arm in proof.ARMS}
    meta = {"status": "COMPLETE", "completed_epochs": 24, "outer_query_opened": False,
            "recipe": {"epochs": 24}, "provenance": {"source": "synthetic"}, "split": {"fixed": True}}
    report = {**meta, "status": "SOURCE_MINIVAL_ONLY", "history": history,
              "selected_primary_ema": selected, "endpoint24": endpoint}
    for name, body in (("run_meta.json", meta), ("report.json", report), ("epoch_metrics.json", history)):
        write(root / name, body)
    frozen = {"meta_sha256": proof.sha(root / "run_meta.json"), "report_sha256": proof.sha(root / "report.json"),
              "history_sha256": proof.sha(root / "epoch_metrics.json"), "source_closure": meta["provenance"],
              "selected": selected, "endpoint24": endpoint,
              "checkpoint_sha256": {f"{arm}_{label}": records[arm]["checkpoint_sha256"]
                                    for label, records in (("selected", selected), ("endpoint24", endpoint)) for arm in proof.ARMS}}
    exports = {}
    out = root / "finalized_p1"
    out.mkdir()
    for label, records in (("selected", selected), ("endpoint24", endpoint)):
        for arm in proof.ARMS:
            state, archive = out / f"{arm}_{label}_ema_state.pt", out / f"{arm}_{label}_native_source_dev.npz"
            state.write_bytes(b"synthetic state")
            archive.write_bytes(b"synthetic archive")
            exports[f"{arm}_{label}"] = {"record": records[arm], "state_sha256": proof.sha(state), "prediction_sha256": proof.sha(archive)}
    write(root / "selection_freeze.json", {"status": "FROZEN_PRE_INFERENCE", "outer_query_opened": False, "freeze_pre_inference": frozen})
    write(out / "receipt.json", {"status": "SOURCE_MINIVAL_ONLY", "outer_query_opened": False,
                                 "freeze_pre_inference": frozen, "exports": exports})
    return meta


def test_incomplete_refuses_before_any_model_or_source_access(tmp_path):
    write(tmp_path / "run_meta.json", {"status": "RUNNING"})
    with pytest.raises(RuntimeError, match="completed fixed-24"):
        proof.run(tmp_path, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_independent_earliest_tie_and_all_checkpoints(tmp_path):
    completed(tmp_path)
    result = proof.artifact_audit(tmp_path)
    assert result["selected"]["flat"]["epoch"] == 3
    assert len([p for p in result["files"] if p.endswith(".pt")]) == 52
    # Epoch 9 is neither selected nor the final endpoint, and is still bound.
    (tmp_path / "route" / "epoch_009.pt").write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="SHA drift"):
        proof.artifact_audit(tmp_path)


def test_export_receipt_cannot_substitute_different_selection(tmp_path):
    completed(tmp_path)
    path = tmp_path / "finalized_p1" / "receipt.json"
    body = proof.read(path)
    body["exports"]["flat_selected"]["record"]["epoch"] = 5
    write(path, body)
    with pytest.raises(RuntimeError, match="selected/endpoint"):
        proof.artifact_audit(tmp_path)


def test_pre_post_files_catch_mutation(tmp_path):
    path = tmp_path / "x"
    path.write_bytes(b"a")
    before = {str(path): proof.sha(path)}
    assert proof.require_same_files(before) == before
    path.write_bytes(b"b")
    with pytest.raises(RuntimeError, match="changed"):
        proof.require_same_files(before)


def test_typed_hash_matches_frozen_export_convention():
    from tfpd_exploration.src.m2_same_query_comparator_v1.core import array_sha256
    for value in (np.arange(4, dtype=np.int64), np.ones((2, 16), np.float32), np.asarray(proof.SESSIONS)):
        assert proof.typed_array_sha(value) == array_sha256(value)
        assert proof.typed_array_sha(value) != proof.array_sha(value)


def test_native_archive_reproduces_metrics_and_rejects_target_corruption(tmp_path, monkeypatch):
    monkeypatch.setattr(proof, "COUNT", 6)
    y = np.arange(96, dtype=np.float32).reshape(6, 16)
    arrays = {"prediction": y + .2, "target": y, "session": np.repeat(proof.SESSIONS, 2),
              "start": np.tile(np.asarray([100, 103], np.int64), 3)}
    item = {"array_sha256": {key: proof.typed_array_sha(value) for key, value in arrays.items()}, "metrics": proof.metric(arrays)}
    path = tmp_path / "native.npz"
    np.savez_compressed(path, **arrays)
    assert proof.archive_audit(path, item)["prediction"].shape == (6, 16)
    arrays["target"][0, 0] += 1
    np.savez_compressed(path, **arrays)
    with pytest.raises(RuntimeError, match="shape/dtype/hash"):
        proof.archive_audit(path, item)


class Tensor:
    def __init__(self, value): self.value = value
    def detach(self): return self
    def cpu(self): return self
    def numpy(self): return self.value


def metadata_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(proof, "COUNT", 6)
    monkeypatch.setattr(proof, "COUNTS", (2, 2, 2))
    monkeypatch.setattr(proof, "PUBLIC_CALLS", (3, 3, 3))
    starts = np.asarray([100, 103], np.int64)
    cache, provenance, banks, neural, covariates, pairs, targets = {}, {}, {}, {}, {}, [], []
    source = {"cache": str(tmp_path / "cache.npz"), "provenance": str(tmp_path / "prov.npz"),
              "cache_receipt": {"arrays": {}}, "provenance_receipt": {"rows": {}}}
    audit = {"meta": {"split": {}, "provenance": {"source_files": {}}}}
    for index, name in enumerate(proof.SESSIONS):
        raw = np.zeros((220, 64), np.float32)
        raw[99:] = np.arange(121, dtype=np.float32)[:, None] + index
        neural[name] = raw
        covariates[name] = np.arange(220*16, dtype=np.float32).reshape(220, 16)
        targets.append(covariates[name][starts+99])
        e0, t, mask = np.ones((64, 100), np.float32), np.ones((64, 4), np.float32), np.ones(64, bool)
        for key, value in (("raw_neural", raw), ("bank_e0", e0), ("bank_t", t), ("bank_unit_mask", mask)):
            cache[f"{key}/{name}"] = value
        banks[name] = SimpleNamespace(E0=Tensor(e0), T=Tensor(t), unit_mask=Tensor(mask))
        roster = np.arange(64, dtype=np.int64) + 1000
        provenance[f"nwb_unit_ids_in_rate_column_order/{name}"] = roster
        source["cache_receipt"]["arrays"][name] = {"raw_neural_sha256": proof.array_sha(raw)}
        source["provenance_receipt"]["rows"][name] = {"nwb_unit_ids_sha256": proof.array_sha(roster), "source_file_sha256": f"file{index}"}
        audit["meta"]["provenance"]["source_files"][name] = {"sha256": f"file{index}"}
        audit["meta"]["split"][name] = {"dev_window_starts_sha256": proof.split_sha([(name, int(s)) for s in starts])}
        pairs += [(name.encode() if index == 2 else name, int(s)) for s in starts]
    np.savez_compressed(source["cache"], **cache)
    np.savez_compressed(source["provenance"], **provenance)
    dev = SimpleNamespace(base=SimpleNamespace(window_indices=pairs, neural_data=neural, covariate_data=covariates))
    a = {"start": np.tile(starts, 3), "session": np.repeat(proof.SESSIONS, 2), "target": np.concatenate(targets)}
    archives = {"flat": a, "route": copy.deepcopy(a)}
    return archives, dev, banks, source, audit


def test_exact_native_inputs_bridge_and_target_drift(tmp_path, monkeypatch):
    args = metadata_fixture(tmp_path, monkeypatch)
    rows = proof.metadata_audit(*args)
    assert list(rows) == list(proof.SESSIONS)
    assert rows[proof.SESSIONS[0]]["roster"][0] == 1000
    args[0]["route"]["target"][0, 0] += 1
    with pytest.raises(RuntimeError, match="exact target"):
        proof.metadata_audit(*args)


def test_native_bank_drift_cannot_use_older_cached_bank(tmp_path, monkeypatch):
    args = metadata_fixture(tmp_path, monkeypatch)
    args[2][proof.SESSIONS[1]].T.value[0, 0] += 1
    with pytest.raises(RuntimeError, match="P1 bank mismatch"):
        proof.metadata_audit(*args)


def test_actual_initialized_pair_stream_bridge_without_trained_weights():
    import torch
    from tfpd_exploration.src.m1_family_v1.family_train_v2 import _p1_models
    from tfpd_exploration.src.m1_runtime_v3.runtime import BankBatch
    from tfpd_exploration.src.family_runtime_v1.m1_lifted import LiftedFiveTokenCurrentQueryStream
    from tfpd_exploration.src.family_runtime_v1.m1_route_lifted import RouteLiftedFiveTokenCurrentQueryStream
    from tfpd_exploration.src.family_runtime_v1.m1_replay_contract import replay_session
    torch.set_num_threads(1)
    pair = _p1_models(torch.device("cpu"))
    generator = np.random.default_rng(194)
    raw = generator.normal(size=(104, 64)).astype(np.float32)
    starts = np.asarray([0, 1, 4], np.int64)
    bank = BankBatch(torch.randn(1, 64, 100), torch.randn(1, 64, 4), torch.ones(1, 64, dtype=torch.bool),
                     ("synthetic",), (tuple(range(64)),))
    for model, stream_type in zip(pair, (LiftedFiveTokenCurrentQueryStream, RouteLiftedFiveTokenCurrentQueryStream), strict=True):
        model.eval()
        def direct(history):
            with torch.no_grad():
                return model.forward_last(torch.from_numpy(history), bank).numpy().copy()
        archive = {"prediction": np.concatenate([direct(raw[None, s:s+100]) for s in starts]),
                   "target": np.zeros((3, 16), np.float32), "session": np.asarray(["synthetic"]*3), "start": starts}
        runtime = stream_type(model, bank)
        result = replay_session(runtime, raw, "synthetic", archive, direct, native_targets=archive["target"],
                                initialize_history=lambda h: runtime.refresh_state(history=torch.from_numpy(h)),
                                current_prediction=lambda: runtime.current_prediction().detach().numpy().copy())
        assert result["prediction"].shape == (3, 16)
        assert result["public_calls"] == 4 and result["direct_count"] == 3
        assert result["max_abs_error"] < 1e-5
        assert isinstance(runtime.state_bytes, int) and runtime.state_bytes > 0
