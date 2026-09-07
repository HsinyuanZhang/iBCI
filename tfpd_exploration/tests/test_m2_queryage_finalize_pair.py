"""Schema-level tests for the separate QueryAge completion finalizer."""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from tfpd_exploration.src.m2_queryage_family_v1 import finalize_pair as f
from tfpd_exploration.src.m2_queryage_family_v1 import train_pair as q


@pytest.fixture(autouse=True)
def _isolate_current_preflight(monkeypatch):
    """Schema fixtures are intentionally synthetic, not real source caches."""
    monkeypatch.setattr(f, "_current_trainer_authority", lambda protocol: protocol["authority_pre"])


def _put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return path


def _fixture(tmp_path, *, tamper_manifest=False, tie_at_one=True):
    root = (tmp_path / "completed_run").resolve()
    authority = {"frozen": "a" * 64}
    protocol = {"schema": q.SCHEMA, "status": "PROTOCOL_SAVED_PRE_CONSTRUCTION", "authority_pre": authority}
    protocol_path = _put(root / "protocol_preconstruction.json", protocol)
    protocol_sha = q.sha(protocol_path)
    history, picks = {}, {}
    for arm in q.ARMS:
        raw = {str(epoch): float(epoch) / 100 for epoch in range(1, q.EPOCHS + 1)}
        ema = {str(epoch): .2 for epoch in range(1, q.EPOCHS + 1)}
        winner = 1 if tie_at_one else q.EPOCHS
        ema[str(winner)] = .9
        receipts = {}
        for epoch in range(1, q.EPOCHS + 1):
            checkpoint = root / arm / f"epoch_{epoch:03d}.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_bytes(f"{arm}-{epoch}".encode())
            receipt = {"schema": "m2_queryage_prefix_pair_epoch_receipt_v1", "arm": arm, "epoch": epoch,
                       "raw_equal_session_r2": raw[str(epoch)], "ema_equal_session_r2": ema[str(epoch)],
                       "checkpoint": str(checkpoint), "checkpoint_sha256": q.sha(checkpoint),
                       "checkpoint_has_raw_ema_optimizer_rng": True, "endpoint_checkpoint": epoch == q.EPOCHS,
                       "actual_epoch_batches": q.UPDATES_PER_EPOCH,
                       "actual_batch_sha256": {name: "b" * 64 for name in ("order", "target", "prefix", "keep")},
                       "protocol_sha256": protocol_sha}
            receipt_path = _put(root / arm / f"epoch_{epoch:03d}_receipt.json", receipt)
            receipts[str(epoch)] = {**receipt, "receipt_sha256": q.sha(receipt_path)}
        history[arm] = {"RAW": raw, "EMA": ema, "receipts": receipts}
        picks[arm] = winner
    summary = {"schema": q.SCHEMA, "status": "SOURCE_MINIVAL_SELECTION_ONLY_COMPLETE", "protocol": protocol,
               "authority_pre": authority, "authority_post": authority, "history": history,
               "selected_primary_ema_epoch": picks, "endpoint_epoch": q.EPOCHS,
               "outer_or_ext4_or_public_predictions_opened": False}
    summary_path = _put(root / "selection_summary.json", summary)
    manifest = f._tree_hashes_before_completion(root)
    if tamper_manifest:
        manifest["selection_summary.json"] = "0" * 64
    manifest_path = _put(root / "artifact_hashes_pre_completion.json", manifest)
    completion = {"schema": q.SCHEMA, "status": summary["status"],
                  "selection_summary_sha256": q.sha(summary_path), "artifact_hash_manifest_sha256": q.sha(manifest_path),
                  "authority_pre": authority, "authority_post": authority}
    completion_path = _put(root / "completion.json", completion)
    return root, completion_path


def test_validate_completion_reconciles_actual_trainer_schema(tmp_path):
    root, _ = _fixture(tmp_path)
    audit = f.validate_completion(root)
    assert audit["selected_primary_ema_epoch"] == {"FLAT": 1, "ROUTE": 1}
    assert len(audit["epoch_receipt_evidence"]["FLAT"]["epoch_artifact_hashes"]) == q.EPOCHS


def test_validate_completion_rejects_manifest_and_receipt_drift(tmp_path):
    root, _ = _fixture(tmp_path / "manifest", tamper_manifest=True)
    with pytest.raises(RuntimeError, match="manifest"):
        f.validate_completion(root)
    root, completion = _fixture(tmp_path / "receipt")
    (root / "ROUTE" / "epoch_004_receipt.json").write_text("{}", encoding="utf-8")
    manifest_path = root / "artifact_hashes_pre_completion.json"
    _put(manifest_path, f._tree_hashes_before_completion(root))
    completion_body = json.loads(completion.read_text(encoding="utf-8"))
    completion_body["artifact_hash_manifest_sha256"] = q.sha(manifest_path)
    _put(completion, completion_body)
    with pytest.raises(RuntimeError, match="sealed receipt"):
        f.validate_completion(root)


def test_prepare_requires_hash_bound_completion_and_writes_outside_live_root(tmp_path, monkeypatch):
    root, completion = _fixture(tmp_path)
    monkeypatch.setenv(f.GO_ENV, "1")
    output = (tmp_path / "separate_finalizer_output").resolve()
    result = f.prepare(root, output, q.sha(completion))
    assert result["status"] == "SELECTION_FROZEN_PRE_EXPORT_PRE_SCORE"
    assert (output / "selection_freeze.json").is_file()
    assert not (root / "selection_freeze.json").exists()


def test_prepare_gates_before_reading_completion(tmp_path, monkeypatch):
    monkeypatch.delenv(f.GO_ENV, raising=False)
    monkeypatch.setattr(f, "sha", lambda _: pytest.fail("read before finalizer gate"))
    with pytest.raises(RuntimeError, match="explicit GO"):
        f.prepare((tmp_path / "run").resolve(), (tmp_path / "output").resolve(), "a" * 64)


def test_run_exports_four_strict_plain_ema_states_and_identity_locked_archives(tmp_path, monkeypatch):
    """Exercise real QueryAge factory/payload/export; scorer is a source-free fixture."""
    root, completion = _fixture(tmp_path, tie_at_one=False)
    summary = json.loads((root / "selection_summary.json").read_text(encoding="utf-8"))
    audit = {"run_root": str(root), "protocol_sha256": q.sha(root / "protocol_preconstruction.json"),
             "selected_primary_ema_epoch": {arm: q.EPOCHS for arm in q.ARMS},
             "epoch_receipt_evidence": {arm: {"epoch_artifact_hashes": {}} for arm in q.ARMS}}
    output = (tmp_path / "finalizer_output").resolve()
    frozen = {"schema": f.FINALIZER_SCHEMA, "status": "SELECTION_FROZEN_PRE_EXPORT_PRE_SCORE", "audit": audit}
    def fake_prepare(*args):
        output.mkdir()
        f._atomic_json(output / "selection_freeze.json", frozen)
        return frozen
    def real_payload(arm):
        model = f._factory(arm)
        return {"schema": "m2_queryage_prefix_pair_checkpoint_v1", "cell": f"QUERYAGE_PREFIX_{arm}",
                "epoch": q.EPOCHS, "global_step": q.EPOCHS * q.UPDATES_PER_EPOCH,
                "batch_id": q.UPDATES_PER_EPOCH, "seed": 42, "actual_epoch_batches": q.UPDATES_PER_EPOCH,
                "manifest_digest": summary.get("manifest_digest"), "protocol_sha256": audit["protocol_sha256"],
                "authority": summary["authority_pre"], "recipe": summary["protocol"],
                "raw_state_dict": {name: value.detach().clone() for name, value in model.state_dict().items()},
                "ema": {"n_updates": q.EPOCHS * q.UPDATES_PER_EPOCH, "decay": q.EMA_DECAY,
                        "shadow": {name: value.detach().clone() for name, value in model.trainable_parameters().items()}}}
    for arm in q.ARMS:
        endpoint = root / arm / "epoch_024.pt"
        torch.save(real_payload(arm), endpoint)
        audit["epoch_receipt_evidence"][arm]["epoch_artifact_hashes"][q.EPOCHS] = {"checkpoint_sha256": q.sha(endpoint)}
    calls = []
    targets=[]; starts=[]; sessions=[]; source_rows=[]
    for session,count in zip(f.plan.HELDIN_SESSIONS,f.prior_finalizer.COUNTS,strict=True):
        target=np.stack((np.linspace(-1,1,count),np.cos(np.linspace(0,2*np.pi,count))),axis=1).astype(np.float32)
        target-=target.mean(0,keepdims=True)
        start=np.arange(count,dtype=np.int64)
        targets.append(target);starts.append(start);sessions.extend([session]*count)
        source_rows.append({'session':session,'target_sha256':f.core.array_sha256(target),
            'ordered_window_starts_sha256':f.core.array_sha256(start)})
    target=np.concatenate(targets)
    archive={'prediction':target*np.float32(1-np.sqrt(.1)),'target':target,
        'start':np.concatenate(starts),'session':np.asarray(sessions)}
    measured=f._recompute_archive_score(archive)
    def scorer(model, device):
        assert (output / "selection_freeze.json").is_file()
        calls.append((model.name, str(device)))
        return {"rows": [], "equal_session_r2": measured[0], "pooled_r2": measured[1]}, archive
    monkeypatch.setattr(f, "prepare", fake_prepare)
    monkeypatch.setattr(f, "validate_completion", lambda _: audit)
    monkeypatch.setattr(f, "_source_minival_authority", lambda: {"source_minival": source_rows})
    monkeypatch.setattr(f.prior_finalizer, "score_source_minival", scorer)
    result = f.run(root, output, q.sha(completion), device="cpu", allow_cpu_for_test=True)
    assert len(result["exports"]) == len(result["scores"]) == len(calls) == 4
    assert all(score["metric"] == "variance_weighted_r2_fp64" for score in result["scores"].values())
    assert result["native_archive_identity"]["target"]
    for export in result["exports"].values():
        state = torch.load(export["path"], map_location="cpu", weights_only=True)
        assert isinstance(state, dict) and export["strict_plain_ema_reload"] is True
    assert (output / "receipt.json").is_file()
    assert result['no_model_construction_yet'] is False and result['parameter_updates']==0
    assert measured[0]==pytest.approx(.9,abs=1e-6)
    broken=dict(archive);broken['session']=np.full(1011,'fixture')
    with pytest.raises(RuntimeError,match='seven-session'):
        f._recompute_archive_score(broken)
    broken=dict(archive);broken['prediction']=archive['prediction'].astype(np.float64)
    with pytest.raises(RuntimeError,match='FP32'):
        f._recompute_archive_score(broken)


def test_guarded_replay_calls_guard_without_mutating_model_method():
    class Model(torch.nn.Module):
        name='fixture'
        def forward_last(self,x,bank,mask=None):return x[:,-1,:2]
    model=Model(); original=model.forward_last.__func__; checks=[]
    proxy=f.GuardedReplay(model,lambda calls,rows:checks.append((calls,rows)))
    out=proxy.to(torch.device('cpu')).eval().forward_last(torch.ones(3,50,96),None,None)
    assert checks==[(0,0),(1,3)] and out.shape==(3,2) and model.forward_last.__func__ is original
