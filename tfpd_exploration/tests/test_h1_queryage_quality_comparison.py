"""Archive-only contracts for the QueryAge quality table."""
from __future__ import annotations
import copy
import json
import shutil
from pathlib import Path
import numpy as np
import pytest
from tfpd_exploration.src.family_runtime_v1 import compare_h1_queryage_quality as q

def _arrays():
 p=np.arange(28,dtype=np.float64).reshape(4,7);return {"prediction":p,"target":p+np.array([[0],[1],[0],[1]],np.float64),"session_id":np.array(["a","a","b","b"]),"end":np.array([699,700,699,700],np.int64)}

def test_archive_metrics_and_surface_identity_support_four_descriptive_rows():
 original=_arrays();candidate={k:v.copy() for k,v in original.items()};actual=q._metrics(candidate)
 q.same_surface(candidate,original);q._same(actual,actual)
 rows={f"{arm}_{label}":actual for arm in ("FLAT","ROUTE") for label in ("selected","epoch12")}
 assert set(rows)=={"FLAT_selected","FLAT_epoch12","ROUTE_selected","ROUTE_epoch12"}
 delta=q._deltas("flat",actual,actual);assert delta["pooled_delta"]==0 and delta["win_sessions"]==0

@pytest.mark.parametrize("key,value",[("end",np.array([1,2,3,5],np.int64)),("session_id",np.array(["a","b","b","b"])),("target",np.zeros((4,7),np.float64))])
def test_same_surface_rejects_target_session_or_endpoint_drift(key,value):
 original=_arrays();candidate={k:v.copy() for k,v in original.items()};candidate[key]=value
 with pytest.raises(RuntimeError,match="same-20325"):q.same_surface(candidate,original)

def test_reported_complete_metric_drift_is_rejected_without_model_or_selection():
 actual=q._metrics(_arrays());bad={**actual,"per_session_r2_float64":dict(actual["per_session_r2_float64"])};bad["per_session_r2_float64"]["a"]+=.01
 with pytest.raises(RuntimeError,match="per-session"):q._same(actual,bad)


def _refresh_formal(formal, receipt):
    receipt["owned_artifact_sha256"] = {str(p): q.sha(p) for p in sorted(formal.rglob("*")) if p.is_file() and p != formal / "receipt.json"}
    (formal / "receipt.json").write_text(json.dumps(receipt))


def _authorize(kwargs):
    body = {"schema": "h1_queryage_quality_comparison_authorization_v1", "status": "ROOT_REVIEW_GO",
            "bindings": q.collect_bindings(kwargs["formal"], kwargs["output"])}
    kwargs["authorization"].write_text(json.dumps(body))
    kwargs["authorization_sha256"] = q.sha(kwargs["authorization"])


def _quality_fixture(tmp_path, monkeypatch):
    # This fixture creates model-derived archives once; q.run itself must never
    # load any model/checkpoint/cache or invoke a forward.
    from tfpd_exploration.tests.test_h1_queryage_complete_source import _complete_fixture
    from tfpd_exploration.src.family_runtime_v1 import compare_h1_frozen_quality as legacy
    source = _complete_fixture(tmp_path, monkeypatch)
    formal = source["formal"]
    receipt = q.read(formal / "receipt.json")
    for arm in q.ARMS:
        final = q.read(formal / "workers" / f"{arm}_final.json")
        terminal = copy.deepcopy(final["reports"]["selected"])
        for field, suffix in (("complete_archive", "complete_native_float64.npz"), ("plain_ema_path", "plain_ema.pt")):
            old = Path(terminal[field]); new = formal / "exports" / f"{arm}_epoch12_{suffix}"
            shutil.copyfile(old, new); terminal[field] = str(new)
            terminal["complete_archive_sha256" if field == "complete_archive" else "plain_ema_sha256"] = q.sha(new)
        final["reports"]["epoch12"] = terminal
        (formal / "workers" / f"{arm}_final.json").write_text(json.dumps(final))
        receipt["finals"][arm] = final
    original_authority = {"bindings": {"inputs": {str(p): q.sha(p) for p in (q.proof.CACHE, q.proof.CACHE_AUTHORITY)}}}
    Path(receipt["authorization_path"]).write_text(json.dumps(original_authority))
    receipt.update(authority=original_authority, authority_post=original_authority,
                   authorization_sha256=q.sha(Path(receipt["authorization_path"])))
    (formal / "input_authority.json").write_text(json.dumps({"authority": original_authority, "bindings": original_authority["bindings"]}))
    receipt["input_authority_sha256"] = q.sha(formal / "input_authority.json")
    _refresh_formal(formal, receipt)

    original_dir = (tmp_path / "original").resolve(); original_dir.mkdir()
    original_npz = original_dir / "original_h1_minival_native_float64.npz"
    shutil.copyfile(formal / "exports/flat_selected_complete_native_float64.npz", original_npz)
    metric = q._metrics(q.proof._archive(original_npz))
    pre = {key: {"path": str(p), "sha256": q.sha(p)} for key, p in (("cache", q.proof.CACHE), ("authority", q.proof.CACHE_AUTHORITY))}
    original = {"status": "PASS_AS_SHIPPED_ORIGINAL_H1_REFERENCE_ONLY", "scored_count": 4,
                "public_calls": 20920, "direct_native_count": 65, "max_native_abs_error": 0,
                "pre": pre, "post": pre, "archive": {"sha256": q.sha(original_npz)},
                "sessions": [{"original_calibration_shape": [2, 1024, 176]} for _ in range(2)], "metrics": metric}
    (original_dir / "receipt.json").write_text(json.dumps(original))
    (original_dir / "input_authority.json").write_text(json.dumps({"pre": pre}))
    c2path = tmp_path / "c2.json"
    c2 = {"status": "COMPLETE_FIXED_REFERENCE_SCORE", "mode": "complete", "n_bins": 4,
          "input_authority": q.read(q.proof.CACHE_AUTHORITY), "selection_disclosure": "fixed historical readout",
          "r2_concat": metric["r2_concat_float64"] + .05,
          "equal_session_mean_r2": metric["equal_session_mean_r2_float64"] + .05,
          "worst_session_r2": metric["worst_session_r2_float64"] + .05,
          "per_session_r2": {s: v + .05 for s, v in metric["per_session_r2_float64"].items()}}
    c2path.write_text(json.dumps(c2))
    for module in (q, legacy):
        for name, value in (("ORIGINAL_DIR", original_dir), ("ORIGINAL_RECEIPT_SHA", q.sha(original_dir / "receipt.json")),
                            ("ORIGINAL_NPZ_SHA", q.sha(original_npz)), ("C2_PATH", c2path), ("C2_SHA", q.sha(c2path))):
            monkeypatch.setattr(module, name, value)
    monkeypatch.setattr(legacy, "COUNT", 4); monkeypatch.setattr(legacy, "SESSIONS", 2)
    kwargs = {k: source[k] for k in ("formal", "output", "authorization")}
    kwargs["allow_cpu_fixture"] = True
    _authorize(kwargs)
    return kwargs


def test_real_archive_only_run_checks_four_exports_and_both_baselines_without_model_calls(tmp_path, monkeypatch):
    import torch
    kwargs = _quality_fixture(tmp_path, monkeypatch)
    def prohibited(*args, **kw):
        raise AssertionError("archive-only comparator attempted model/cache/checkpoint execution")
    monkeypatch.setattr(torch, "load", prohibited)
    monkeypatch.setattr(torch.nn.Module, "_call_impl", prohibited)
    result = q.run(**kwargs)
    assert set(result["deltas_vs_original"]) == {"FLAT_selected", "FLAT_epoch12", "ROUTE_selected", "ROUTE_epoch12"}
    assert all(value["pooled_delta"] == 0 for value in result["deltas_vs_original"].values())
    assert all(abs(value["pooled_delta"] + .05) < 1e-12 for value in result["deltas_vs_c2_historical"].values())
    assert result["c2_independently_recomputed"] is False and result["selection_or_promotion"] is False
    assert result["pre_audit"] == result["post_audit"]
    assert q.read(kwargs["output"] / "receipt.json") == result
    assert result["owned_artifact_sha256"] == {str(kwargs["output"] / "input_authority.json"): q.sha(kwargs["output"] / "input_authority.json")}


@pytest.mark.parametrize("mutation", ["status", "terminal_plain", "terminal_nan_metric", "source", "original_sidecar"])
def test_real_run_rejects_rebound_inner_authority_or_terminal_export_drift(tmp_path, monkeypatch, mutation):
    kwargs = _quality_fixture(tmp_path, monkeypatch)
    formal = kwargs["formal"]; receipt = q.read(formal / "receipt.json")
    if mutation == "status": receipt["status"] = "INCOMPLETE"
    elif mutation == "terminal_plain": (formal / "exports/route_epoch12_plain_ema.pt").write_bytes(b"drift")
    elif mutation == "terminal_nan_metric":
        final = q.read(formal / "workers/route_final.json")
        final["reports"]["epoch12"]["complete"]["per_session_r2_float64"]["a"] = float("nan")
        (formal / "workers/route_final.json").write_text(json.dumps(final)); receipt["finals"]["route"] = final
    elif mutation == "source": q.proof.CACHE.write_bytes(b"drift")
    elif mutation == "original_sidecar": (q.ORIGINAL_DIR / "input_authority.json").write_text("{}")
    _refresh_formal(formal, receipt); _authorize(kwargs)
    with pytest.raises(RuntimeError): q.run(**kwargs)
    assert not (kwargs["output"] / "receipt.json").exists()


@pytest.mark.parametrize("mutation", ["authorization", "original_archive", "own_sidecar"])
def test_late_input_or_authority_drift_never_emits_a_pass_receipt(tmp_path, monkeypatch, mutation):
    kwargs = _quality_fixture(tmp_path, monkeypatch)
    candidate = q._candidate
    def mutate_after_candidate(*args, **kw):
        result = candidate(*args, **kw)
        if mutation == "authorization": kwargs["authorization"].write_text("{}")
        elif mutation == "original_archive":
            path = q.ORIGINAL_DIR / "original_h1_minival_native_float64.npz"
            path.write_bytes(path.read_bytes() + b"drift")
        else: (kwargs["output"] / "input_authority.json").write_text("{}")
        return result
    monkeypatch.setattr(q, "_candidate", mutate_after_candidate)
    with pytest.raises(RuntimeError): q.run(**kwargs)
    assert not (kwargs["output"] / "receipt.json").exists()


def test_missing_incomplete_paths_and_go_reject_before_candidate_read(tmp_path, monkeypatch):
    def prohibited(*args, **kw): raise AssertionError("candidate archive read before admission")
    monkeypatch.setattr(q, "_candidate", prohibited)
    formal, output, auth = tmp_path / "absent", tmp_path / "out", tmp_path / "go.json"
    with pytest.raises(RuntimeError, match="GO"):
        q.run(formal=formal, output=output, authorization=auth, authorization_sha256="0" * 64)
    with pytest.raises(FileNotFoundError):
        q.run(formal=formal, output=output, authorization=auth, authorization_sha256="0" * 64, allow_cpu_fixture=True)
    with pytest.raises(RuntimeError, match="disjoint"):
        q.run(formal=formal, output=formal / "child", authorization=auth, authorization_sha256="0" * 64, allow_cpu_fixture=True)
    with pytest.raises(RuntimeError, match="canonical"):
        q.run(formal=Path("relative"), output=output, authorization=auth, authorization_sha256="0" * 64, allow_cpu_fixture=True)
