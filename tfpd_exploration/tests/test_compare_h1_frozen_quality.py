"""Pure synthetic tests for the H1 frozen archive comparator."""
from __future__ import annotations
from types import SimpleNamespace
import numpy as np
import pytest
from tfpd_exploration.src.family_runtime_v1 import compare_h1_frozen_quality as comparison

def _metrics(offset=0.):
    per={"s0":.1+offset,"s1":.2+offset}
    return {"r2_concat_float64":.15+offset,"equal_session_mean_r2_float64":.15+offset,"worst_session_r2_float64":.1+offset,"per_session_r2_float64":per}

def test_delta_table_has_exact_per_session_wins_losses_and_metric_deltas(monkeypatch):
    monkeypatch.setattr(comparison,"SESSIONS",2)
    actual=comparison._deltas("candidate",_metrics(.1),_metrics())
    assert actual["pooled_delta"] == pytest.approx(.1)
    assert actual["win_sessions"]==2 and actual["loss_sessions"]==0 and actual["tie_sessions"]==0

def test_metric_receipts_require_finite_exactly_all_session_values(monkeypatch):
    monkeypatch.setattr(comparison,"SESSIONS",2)
    comparison._same_metrics(_metrics(),_metrics())
    bad=_metrics(); bad["per_session_r2_float64"]["s1"]=float("nan")
    with pytest.raises(RuntimeError,match="finite"):
        comparison._same_metrics(bad,_metrics())
    bad=_metrics(); bad["r2_concat_float64"]+=1e-4
    with pytest.raises(RuntimeError,match="reported metric drift"):
        comparison._same_metrics(bad,_metrics())

def test_go_gate_precedes_formal_or_archive_reads(tmp_path,monkeypatch):
    args=SimpleNamespace(output=tmp_path/"out.json",formal=tmp_path/"formal",original_receipt=tmp_path/"receipt",original_npz=tmp_path/"archive")
    monkeypatch.delenv("H1_FROZEN_QUALITY_COMPARISON_GO",raising=False)
    monkeypatch.setattr(comparison,"artifact_audit",lambda _:pytest.fail("formal audit read before gate"))
    with pytest.raises(RuntimeError,match="GO"):
        comparison.run(args)
    assert not args.output.exists()

def test_constants_bind_actual_original_and_historical_c2_receipts():
    assert comparison.ORIGINAL_RECEIPT_SHA=="539bfa832c650df51e22be14b5ee0dba0c1f49fbb595f9fb869328a601f30c48"
    assert comparison.ORIGINAL_NPZ_SHA=="f1bb6739415ea66bb86bf4285035f131333e350072ae364139b81488912ba81c"
    assert comparison.C2_SHA=="96a9a496141e2bae533facb009903417f19a78ea970b0d3dc11c8742835d5c98"

def test_cli_parses_and_runs_exactly_once(monkeypatch,tmp_path):
    calls=[]
    monkeypatch.setattr(comparison,"run",lambda args:calls.append(args) or {"status":"PASS","schema":"x"})
    result=comparison.main(["--formal",str(tmp_path/"f"),"--output",str(tmp_path/"o")])
    assert result["status"]=="PASS" and len(calls)==1

@pytest.mark.parametrize("key",("target","end","session_id"))
def test_same_surface_rejects_each_metadata_dimension(key):
    base={"target":np.zeros((2,7)),"end":np.array([1,2]),"session_id":np.array(["a","b"])}
    changed={k:v.copy() for k,v in base.items()}
    changed[key].flat[0]=99 if key!="session_id" else "x"
    with pytest.raises(RuntimeError,match=key): comparison.same_surface(changed,base)

def test_original_native_receipt_rejects_malformed_native_proof(monkeypatch,tmp_path):
    monkeypatch.setattr(comparison,"COUNT",4); monkeypatch.setattr(comparison,"SESSIONS",2)
    p=tmp_path/"r"; n=tmp_path/"n"; p.write_text("x"); n.write_text("y")
    monkeypatch.setattr(comparison,"ORIGINAL_RECEIPT_SHA","x"); monkeypatch.setattr(comparison,"ORIGINAL_NPZ_SHA","y")
    monkeypatch.setattr(comparison,"sha",lambda path: "x" if str(path)==str(p) else "y")
    arrays={"prediction":np.array([[0.],[1.],[0.],[1.]]).repeat(7,1),"target":np.array([[0.],[2.],[0.],[2.]]).repeat(7,1),"end":np.arange(4),"session_id":np.array(["s0","s0","s1","s1"])}
    receipt={"_path":str(p),"_npz_path":str(n),"status":"PASS_AS_SHIPPED_ORIGINAL_H1_REFERENCE_ONLY","scored_count":4,"public_calls":20920,"pre":{},"post":{},"direct_native_count":65,"max_native_abs_error":1e-4,"archive":{"sha256":"y"},"sessions":[{"original_calibration_shape":[2,1024,176]}]*2,"metrics":comparison.metric(arrays)}
    with pytest.raises(RuntimeError,match="native"):
        comparison._original_metric(receipt,arrays)

def _run_fixture(tmp_path,monkeypatch,*,mutate=False):
    monkeypatch.setattr(comparison,"COUNT",4); monkeypatch.setattr(comparison,"SESSIONS",2)
    formal=tmp_path/"formal"; formal.mkdir(); authority=tmp_path/"authority.json"
    authority.write_text('{"arrays":{"x":1}}'); pre={"cache":{"sha256":"cache"},"authority":{"path":str(authority),"sha256":comparison.sha(authority)}}
    (formal/"input_authority.json").write_text(__import__("json").dumps({"bindings":{"source_cache_sha256":"cache","source_authority_sha256":pre["authority"]["sha256"]}}))
    receipt=tmp_path/"receipt.json"; original=tmp_path/"original.npz"; original.write_bytes(b"original")
    rows=[{"original_calibration_shape":[2,1024,176]}]*2
    arrays={"prediction":np.array([[0.],[1.],[0.],[1.]]).repeat(7,1),"target":np.array([[0.],[2.],[0.],[2.]]).repeat(7,1),"end":np.arange(4),"session_id":np.array(["s0","s0","s1","s1"])}
    body={"status":"PASS_AS_SHIPPED_ORIGINAL_H1_REFERENCE_ONLY","scored_count":4,"public_calls":20920,"pre":pre,"post":pre,"direct_native_count":65,"max_native_abs_error":0,"archive":{"sha256":comparison.sha(original)},"sessions":rows,"metrics":comparison.metric(arrays)}
    receipt.write_text(__import__("json").dumps(body)); (tmp_path/"input_authority.json").write_text(__import__("json").dumps({"pre":pre}))
    c2=tmp_path/"c2.json"; c2.write_text(__import__("json").dumps({"status":"COMPLETE_FIXED_REFERENCE_SCORE","mode":"complete","n_bins":4,"input_authority":{"arrays":{"x":1}},"selection_disclosure":"fixed readout","r2_concat":body["metrics"]["r2_concat_float64"],"equal_session_mean_r2":body["metrics"]["equal_session_mean_r2_float64"],"worst_session_r2":body["metrics"]["worst_session_r2_float64"],"per_session_r2":body["metrics"]["per_session_r2_float64"]}))
    monkeypatch.setattr(comparison,"C2_PATH",c2); monkeypatch.setattr(comparison,"C2_SHA",comparison.sha(c2)); monkeypatch.setattr(comparison,"ORIGINAL_RECEIPT_SHA",comparison.sha(receipt)); monkeypatch.setattr(comparison,"ORIGINAL_NPZ_SHA",comparison.sha(original))
    reports={arm:{"reports":{label:{"complete":comparison.metric(arrays)} for label in ("selected","epoch12")}} for arm in ("flat","route")}; audit={"files":{},"final":{"complete":reports}}
    calls=[]
    def audit_fn(_):
        calls.append(1)
        if mutate and len(calls)==2: original.write_bytes(b"changed")
        return audit
    monkeypatch.setattr(comparison,"artifact_audit",audit_fn); monkeypatch.setattr(comparison,"validate_archive",lambda _: {k:v.copy() for k,v in arrays.items()})
    monkeypatch.setenv("H1_FROZEN_QUALITY_COMPARISON_GO","1")
    return SimpleNamespace(output=tmp_path/"out.json",formal=formal,original_receipt=receipt,original_npz=original)

def test_synthetic_full_run_writes_one_result_and_refuses_overwrite(tmp_path,monkeypatch):
    args=_run_fixture(tmp_path,monkeypatch); result=comparison.run(args)
    assert result["status"].startswith("PASS") and args.output.is_file()
    with pytest.raises(FileExistsError): comparison.run(args)

def test_synthetic_run_rejects_post_read_immutable_mutation(tmp_path,monkeypatch):
    args=_run_fixture(tmp_path,monkeypatch,mutate=True)
    with pytest.raises(RuntimeError,match="post-replay"):
        comparison.run(args)
    assert not args.output.exists()

def test_original_input_authority_mismatch_is_rejected_before_archive_work(tmp_path,monkeypatch):
    args=_run_fixture(tmp_path,monkeypatch); (tmp_path/"input_authority.json").write_text("{}")
    with pytest.raises(RuntimeError,match="pre-image"):
        comparison.run(args)
