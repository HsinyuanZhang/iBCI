"""Small CPU contracts for the additive selected-QueryAge public proof."""
from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from tfpd_exploration.src.family_runtime_v1 import complete_h1_queryage_source as proof
from tfpd_exploration.src.two_mainlines_long_v1.current_query_v2.core import QueryTemporalStack
from tfpd_exploration.src.two_mainlines_long_v1.current_query_v2.streaming import CurrentQueryStream
from tfpd_exploration.src.two_mainlines_long_v1.decoder.h1_temporal import H1Bank


class TinyQueryAgeH1(nn.Module):
    """A reduced test-only H1 shell using the real QueryAge cache/stream."""
    def __init__(self):
        super().__init__()
        self.cfg = SimpleNamespace(window=700, conv_kernel=5)
        self.encoder = nn.Linear(176, 8)
        self.temporal = QueryTemporalStack(width=8, heads=2, layers=1, ffn=16, window=700, age_buckets=16, seed=9)
        self.final_norm, self.readout = nn.LayerNorm(8), nn.Linear(8, 7)

    def encode_frontend(self, raw, bank):
        return self.encoder(raw)

    def forward_last(self, raw, bank):
        return self.readout(self.final_norm(self.temporal(self.encode_frontend(raw, bank))))[:, 0]


class TinyCausalQueryAgeH1(TinyQueryAgeH1):
    """The fixture frontend has a real causal k=5 receptive field."""
    def __init__(self):
        super().__init__()
        self.causal = nn.Conv1d(176, 8, kernel_size=5)
        del self.encoder

    def encode_frontend(self, raw, bank):
        value = torch.nn.functional.pad(raw.transpose(1, 2), (4, 0))
        return self.causal(value).transpose(1, 2)


def test_real_current_query_stream_has_left_zero_w700_history_and_matches_direct_queryage():
    torch.manual_seed(4)
    model = TinyQueryAgeH1().eval()
    bank = H1Bank(torch.zeros(176, 700), torch.zeros(176, 4), torch.ones(176, dtype=torch.bool))
    stream = CurrentQueryStream(model, bank, task="h1", session_id="fixture", unit_ids=range(176), batch_size=1)
    neural = np.arange(701 * 176, dtype=np.float32).reshape(701, 176) / 1e6
    for step in range(len(neural)):
        got = stream.predict(np.ascontiguousarray(neural[step:step + 1]))[0]
        raw = proof._raw_window(neural, step)
        np.testing.assert_array_equal(stream.front.raw[0].detach().numpy(), raw)
        if step in (0, 4, 699, 700):
            with torch.no_grad():
                direct = (model.forward_last(torch.from_numpy(raw[None]), bank) / 20.0).numpy()[0]
            np.testing.assert_allclose(got, direct, rtol=1e-5, atol=1e-5)


def test_complete_archive_rejects_wrong_schema_and_preserves_native_fp64(tmp_path):
    path = tmp_path / "complete.npz"
    prediction = np.arange(28, dtype=np.float64).reshape(4, 7)
    ids, ends = np.array(["a", "a", "b", "b"]), np.array([1, 2, 3, 4], dtype=np.int64)
    np.savez_compressed(path, prediction=prediction, target=prediction + 1, session_id=ids, end=ends)
    value = proof._archive(path, count=4)
    assert value["prediction"].dtype == np.float64
    assert proof._metric(value)["n_bins"] == 4
    bad = tmp_path / "bad.npz"
    np.savez_compressed(bad, prediction=prediction.astype(np.float32), target=prediction, session_id=ids, end=ends)
    try:
        proof._archive(bad, count=4)
    except RuntimeError as exc:
        assert "geometry" in str(exc)
    else:
        raise AssertionError("FP32 native archive was accepted")


def _complete_fixture(tmp_path, monkeypatch):
    """A fixture-only two-session formal tree drives the real public proof."""
    monkeypatch.setattr(proof, "COUNT", 4); monkeypatch.setattr(proof, "SESSIONS", 2)
    source, authority_path = (tmp_path / "source.pt").resolve(), (tmp_path / "source_authority.json").resolve()
    source.write_bytes(b"source"); authority_path.write_text("{}\n")
    monkeypatch.setattr(proof, "CACHE", source); monkeypatch.setattr(proof, "CACHE_AUTHORITY", authority_path)
    formal, output, external = (tmp_path / "formal").resolve(), (tmp_path.parent / (tmp_path.name + "_proof")).resolve(), (tmp_path.parent / (tmp_path.name + "_proof_go.json")).resolve()
    for name in ("workers", "exports", "checkpoints", "barrier"): (formal / name).mkdir(parents=True, exist_ok=True)
    torch.manual_seed(17)
    base = TinyCausalQueryAgeH1().eval(); state = {k: v.detach().clone() for k, v in base.state_dict().items()}
    def factory(*, seed):
        a, b = TinyCausalQueryAgeH1().eval(), TinyCausalQueryAgeH1().eval(); a.load_state_dict(state); b.load_state_dict(state); return a, b
    bank = H1Bank(torch.zeros(176, 700), torch.zeros(176, 4), torch.ones(176, dtype=torch.bool))
    cache = {"minival": {}}
    for session in ("a", "b"):
        neural = np.arange(705 * 176, dtype=np.float32).reshape(705, 176) / (1e6 if session == "a" else 2e6)
        cache["minival"][session] = {"neural": neural, "velocity": np.zeros((705, 7), dtype=np.float32), "eval_mask": np.array([False] * 699 + [True, True] + [False] * 4, dtype=np.bool_), "bank": {"E0": bank.E0, "T": bank.T, "unit_mask": bank.unit_mask}}
    prediction, target, sid, ends = [], [], [], []
    for session, row in sorted(cache["minival"].items()):
        stream = CurrentQueryStream(factory(seed=42)[0], bank, task="h1", session_id=session, unit_ids=range(176))
        for step in range(701):
            value = stream.predict(np.ascontiguousarray(row["neural"][step:step + 1]))[0]
            if row["eval_mask"][step]:
                row["velocity"][step] = value + np.float32((step - 699) * .01)
                prediction.append(value.astype(np.float64)); target.append(row["velocity"][step].astype(np.float64)); sid.append(session); ends.append(step)
    archive = {"prediction": np.asarray(prediction), "target": np.asarray(target), "session_id": np.asarray(sid), "end": np.asarray(ends, dtype=np.int64)}
    finals = {}
    for arm in proof.ARMS:
        plain, npz = formal / "exports" / f"{arm}_selected_plain_ema.pt", formal / "exports" / f"{arm}_selected_complete_native_float64.npz"
        torch.save(state, plain); np.savez_compressed(npz, **archive)
        rows = []
        for epoch in range(1, 13):
            checkpoint = formal / "checkpoints" / f"{arm}_epoch_{epoch:03d}.pt"; checkpoint.write_bytes(f"{arm}{epoch}".encode())
            row = {"epoch": epoch, "checkpoint": str(checkpoint), "checkpoint_sha256": proof.sha(checkpoint), "selection": {"r2_concat_float64": float(epoch), "n_bins": 1}, "identities": {"fixture": "ok"}}
            rows.append(row); (formal / "workers" / f"{arm}_epoch_{epoch:03d}.json").write_text(json.dumps(row, sort_keys=True))
        complete = {"arm": arm, "identities": {"fixture": "ok"}, "epochs": rows, "selected_epoch": 12, "selected_ema_r2_float64": 12.0}
        (formal / "workers" / f"{arm}_complete.json").write_text(json.dumps(complete, sort_keys=True)); (formal / "barrier" / f"{arm}.ready.json").write_text(json.dumps({"identities": {"fixture": "ok"}}))
        report = {"epoch": 12, "checkpoint_sha256": rows[-1]["checkpoint_sha256"], "plain_ema_path": str(plain), "plain_ema_sha256": proof.sha(plain), "complete_archive": str(npz), "complete_archive_sha256": proof.sha(npz), "complete": {"n_bins": 4, **proof._metric(archive)}}
        finals[arm] = {"status": "COMPLETE_POST_FREEZE", "arm": arm, "reports": {"selected": report, "epoch12": report}}
        (formal / "workers" / f"{arm}_final.json").write_text(json.dumps(finals[arm], sort_keys=True))
    freeze = {"schema": "h1_queryage_formal_prefix_selection_freeze_v1", "selected": {arm: {"epoch": 12, "ema_r2_float64": 12.0, "checkpoint": str(formal / "checkpoints" / f"{arm}_epoch_012.pt"), "checkpoint_sha256": proof.sha(formal / "checkpoints" / f"{arm}_epoch_012.pt")} for arm in proof.ARMS}}
    freeze["epoch12"] = freeze["selected"]
    (formal / "selection_freeze.json").write_text(json.dumps(freeze, sort_keys=True))
    original_authority = {"bindings": {"inputs": {str(source): proof.sha(source)}}}
    formal_auth = (tmp_path / "formal_go.json").resolve(); formal_auth.write_text(json.dumps(original_authority))
    receipt = {"schema": "h1_queryage_formal_prefix_complete_pair_v1", "status": "COMPLETE_FIXED_FORMAL_NO_PROMOTION", "selection_freeze": freeze, "selection_freeze_sha256": proof.sha(formal / "selection_freeze.json"), "authority": original_authority, "authority_post": original_authority, "authorization_path": str(formal_auth), "authorization_sha256": proof.sha(formal_auth), "finals": finals}
    (formal / "input_authority.json").write_text(json.dumps({"authority": original_authority, "bindings": original_authority["bindings"]}, sort_keys=True)); receipt["input_authority_sha256"] = proof.sha(formal / "input_authority.json")
    receipt["owned_artifact_sha256"] = {str(p): proof.sha(p) for p in sorted(formal.rglob("*")) if p.is_file() and p.name != "receipt.json"}; (formal / "receipt.json").write_text(json.dumps(receipt, sort_keys=True))
    bindings = proof.collect_bindings(formal, output); external.write_text(json.dumps({"schema": "h1_queryage_selected_public_proof_authorization_v1", "status": "ROOT_REVIEW_GO", "bindings": bindings}, sort_keys=True))
    return dict(formal=formal, output=output, authorization=external, authorization_sha256=proof.sha(external), allow_cpu_fixture=True, source_loader=lambda: (cache, {}), model_factory=factory)


def test_reduced_selected_proof_runs_full_admission_stream_gap_and_post_disk_audit(tmp_path, monkeypatch):
    kwargs = _complete_fixture(tmp_path, monkeypatch)
    result = proof.run(**kwargs)
    assert result["status"] == "PASS_IMPLEMENTATION_EQUIVALENCE_ONLY"
    assert (kwargs["output"] / "input_authority.json").is_file() and set(result["arms"]) == {"flat", "route"}
    for arm in result["arms"].values():
        for session in arm["sessions"].values():
            assert session["observed_bins"] == 701  # raw has four unscored tail bins
            assert set(session["direct_native_max_abs_error_by_end"]) == {"0", "4", "699", "700"}


@pytest.mark.parametrize("mutation", ["status", "selected_export", "checkpoint", "freeze", "source", "final_metrics"])
def test_admission_rejects_complete_tree_drift_before_model_or_source_load(tmp_path, monkeypatch, mutation):
    kwargs = _complete_fixture(tmp_path, monkeypatch)
    formal = kwargs["formal"]
    receipt = proof.read(formal / "receipt.json")
    if mutation == "status": receipt["status"] = "STILL_RUNNING"
    elif mutation == "selected_export": (formal / "exports/flat_selected_plain_ema.pt").write_bytes(b"drift")
    elif mutation == "checkpoint": (formal / "checkpoints/route_epoch_006.pt").write_bytes(b"drift")
    elif mutation == "source": proof.CACHE.write_bytes(b"drift")
    elif mutation == "freeze":
        freeze = proof.read(formal / "selection_freeze.json")
        freeze["selected"]["flat"]["epoch"] = 11
        (formal / "selection_freeze.json").write_text(json.dumps(freeze))
        receipt["selection_freeze"] = freeze
        receipt["selection_freeze_sha256"] = proof.sha(formal / "selection_freeze.json")
    elif mutation == "final_metrics":
        final = proof.read(formal / "workers/flat_final.json")
        final["reports"]["selected"]["complete"]["r2_concat_float64"] += .1
        (formal / "workers/flat_final.json").write_text(json.dumps(final))
        receipt["finals"]["flat"] = final
    # Rebind the outer inventory/authorization, so rejection must validate the
    # specific inner record rather than stop at the first stale outer hash.
    receipt["owned_artifact_sha256"] = {str(p): proof.sha(p) for p in sorted(formal.rglob("*")) if p.is_file() and p != formal / "receipt.json"}
    (formal / "receipt.json").write_text(json.dumps(receipt))
    auth = kwargs["authorization"]
    auth.write_text(json.dumps({"schema": "h1_queryage_selected_public_proof_authorization_v1", "status": "ROOT_REVIEW_GO", "bindings": proof.collect_bindings(formal, kwargs["output"])}))
    kwargs["authorization_sha256"] = proof.sha(auth)
    called = []
    def prohibited(*args, **kw):
        called.append(True)
        raise AssertionError("a model/source loader ran before admission passed")
    kwargs["model_factory"] = kwargs["source_loader"] = prohibited
    with pytest.raises(RuntimeError): proof.run(**kwargs)
    assert not called and not kwargs["output"].exists()


def test_original_formal_authority_is_freshly_recomputed_and_mismatch_rejected(tmp_path, monkeypatch):
    from tfpd_exploration.src.h1_queryage_family_v1 import formal_prefix_launcher as launcher
    formal = (tmp_path / "formal").resolve()
    recorded = {"status": "ROOT_REVIEW_GO", "mode": "formal", "bindings": {"output": str(formal)},
                "capacity_audit": {"receipt": str((tmp_path / "capacity.json").resolve()), "checkpoint": str((tmp_path / "capacity.pt").resolve())},
                "smoke_audit": {"receipt": str((tmp_path / "smoke.json").resolve())}}
    calls = []
    def fresh(*args):
        calls.append(args)
        return {**recorded, "status": "SOURCE_CHANGED"}
    monkeypatch.setattr(launcher, "collect_formal_authority", fresh)
    with pytest.raises(RuntimeError, match="fresh original formal"):
        proof._fresh_formal_authority(formal, recorded)
    assert len(calls) == 1 and calls[0][0] == formal


def test_admission_rejects_missing_formal_before_injected_factory_or_source_loader(tmp_path):
    called = []
    def bomb(*args, **kwargs):
        called.append(True); raise AssertionError("must not construct a model or source")
    formal, output, authorization = (tmp_path / "missing").resolve(), (tmp_path / "out").resolve(), (tmp_path / "go.json").resolve()
    authorization.write_text("{}")
    with pytest.raises(FileNotFoundError):
        proof.run(formal=formal, output=output, authorization=authorization, authorization_sha256=proof.sha(authorization),
                  allow_cpu_fixture=True, model_factory=bomb, source_loader=bomb)
    assert not called


def test_admission_rejects_injected_dependencies_without_fixture_before_factory(tmp_path):
    called = []
    def bomb(*args, **kwargs):
        called.append(True); raise AssertionError("factory must not be called")
    formal, output, authorization = (tmp_path / "formal").resolve(), (tmp_path / "out").resolve(), (tmp_path / "go.json").resolve()
    authorization.write_text("{}")
    with pytest.raises(RuntimeError, match="test-only"):
        proof.run(formal=formal, output=output, authorization=authorization, authorization_sha256=proof.sha(authorization), model_factory=bomb)
    assert not called


def test_admission_rejects_output_and_authorization_path_conflicts_before_factory(tmp_path):
    called = []
    def bomb(*args, **kwargs):
        called.append(True); raise AssertionError("factory must not be called")
    formal = (tmp_path / "formal").resolve(); formal.mkdir()
    inside_output, authorization = (formal / "out").resolve(), (formal / "go.json").resolve(); authorization.write_text("{}")
    with pytest.raises(RuntimeError, match="disjoint"):
        proof.run(formal=formal, output=inside_output, authorization=authorization, authorization_sha256=proof.sha(authorization), allow_cpu_fixture=True, model_factory=bomb)
    assert not called


def test_archive_and_source_reject_empty_endpoint_topology(tmp_path):
    path = tmp_path / "empty.npz"
    np.savez_compressed(path, prediction=np.empty((0, 7), np.float64), target=np.empty((0, 7), np.float64), session_id=np.empty((0,), "U1"), end=np.empty((0,), np.int64))
    with pytest.raises(RuntimeError):
        proof._metric(proof._archive(path, count=0))


@pytest.mark.parametrize("value", [
    np.zeros((7,), dtype=np.float32), np.zeros((1, 7), dtype=np.float64),
    np.zeros((2, 7), dtype=np.float32), np.zeros((1, 14), dtype=np.float32)[:, ::2],
    np.zeros((2, 7), dtype=np.float32)[:1],  # valid shape/contiguity, not owned
    np.full((1, 7), np.nan, dtype=np.float32),
])
def test_public_rejects_wrong_shape_dtype_ownership_contiguity_and_finiteness(value):
    with pytest.raises(RuntimeError, match="public output"):
        proof._public(value)


def test_source_ends_rejects_empty_and_last_endpoint_before_neural_tail():
    row = {"neural": np.zeros((705, 176), np.float32), "velocity": np.zeros((705, 7), np.float32),
           "eval_mask": np.array([False] * 705, dtype=np.bool_)}
    with pytest.raises(RuntimeError): proof._source_ends(row)
    row["eval_mask"][699:701] = True
    ends = proof._source_ends(row)
    np.testing.assert_array_equal(ends, np.array([699, 700]))
    assert int(ends[-1]) < len(row["neural"]) - 1
