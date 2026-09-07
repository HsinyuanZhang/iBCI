"""CPU/stdlib lineage tests for the thin PACD V2 successor."""
from __future__ import annotations
import hashlib, json, os, stat, sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1]; REPO=ROOT.parent; sys.path.insert(0,str(ROOT))
from src.paired_anchored_calibration_dropout_v1 import smoke as v1
from src.paired_anchored_calibration_dropout_v2 import plan, predecessor, smoke

def _write_leaf(path: Path, body: bytes):
    path.write_bytes(body); os.chmod(path,0o444)
    side=path.with_name(path.name+".sha256"); side.write_text(f"{hashlib.sha256(body).hexdigest()}  {path.name}\n"); os.chmod(side,0o444)

def _graph(tmp_path, monkeypatch):
    relative="pred"; d=tmp_path/relative; d.mkdir()
    attempt={"source_closure":{"closure_sha256":"closure","files":{"tfpd_exploration/docs/WORKORDER_PACD_V1_20260831.md":{"sha256":"work"}}}}
    failure={"status":"CELL_FAILED","attempt_sha256":"","terminal_published":False,"target_access":False,"no_target_facts":{"source_roster_n":27},"failure":{"kind":"ValueError","detail":"uninitialized parameter","traceback":"core.py numel"}}
    ab=(json.dumps(attempt,sort_keys=True)+"\n").encode(); attempt_sha=hashlib.sha256(ab).hexdigest(); failure["attempt_sha256"]=attempt_sha
    fb=(json.dumps(failure,sort_keys=True)+"\n").encode(); failure_sha=hashlib.sha256(fb).hexdigest()
    _write_leaf(d/"attempt.json",ab); _write_leaf(d/"failure.json",fb)
    monkeypatch.setattr(plan,"PREDECESSOR_RELATIVE",relative); monkeypatch.setattr(plan,"V1_ATTEMPT_SHA256",attempt_sha); monkeypatch.setattr(plan,"V1_FAILURE_SHA256",failure_sha); monkeypatch.setattr(plan,"V1_CLOSURE_SHA256","closure"); monkeypatch.setattr(plan,"V1_WORKORDER_SHA256","work")
    return d,relative

def test_held_fd_predecessor_validates_exact_graph(tmp_path,monkeypatch):
    _d,relative=_graph(tmp_path,monkeypatch)
    out=predecessor.validate_v1_predecessor(tmp_path,relative)
    assert out["topology"]==list(predecessor.LEAVES)

@pytest.mark.parametrize("kind",["extra","body","sidecar","mode","symlink"])
def test_held_fd_predecessor_rejects_graph_drift(tmp_path,monkeypatch,kind):
    d,relative=_graph(tmp_path,monkeypatch)
    if kind=="extra": (d/"extra").write_text("x")
    elif kind=="body": os.chmod(d/"attempt.json",0o644)
    elif kind=="sidecar":
        os.chmod(d/"attempt.json.sha256",0o644); (d/"attempt.json.sha256").write_text("bad\n"); os.chmod(d/"attempt.json.sha256",0o444)
    elif kind=="mode": os.chmod(d/"failure.json",0o644)
    else:
        target=d/"failure.json"; target.rename(d/"real_failure"); (d/"failure.json").symlink_to(d/"real_failure")
    with pytest.raises(predecessor.PredecessorError): predecessor.validate_v1_predecessor(tmp_path,relative)

def test_v2_profile_is_thin_v1_executor_successor_and_exact_root(tmp_path):
    p=smoke.profile()
    assert p.cell==plan.CELL and p.predecessor_validator is predecessor.validate_v1_predecessor
    expected=tmp_path/plan.RESULT_ROOT_RELATIVE
    assert v1.require_canonical_fresh_out_dir(tmp_path,expected,p.result_root_relative)==expected
    with pytest.raises(v1.SmokeError): v1.require_canonical_fresh_out_dir(tmp_path,tmp_path/"other",p.result_root_relative)

def test_v2_fresh_temp_attempt_terminal_xor_failure_lifecycle(tmp_path):
    receipt=v1.load_stdlib_receipt_module(REPO); out=tmp_path/"v2"; out.mkdir()
    receipt.write_receipt_transactionally(out/"attempt.json",{"status":"ATTEMPT_PUBLISHED"})
    receipt.write_receipt_transactionally(out/"terminal.json",{"status":"COMPLETE"})
    assert (out/"attempt.json").is_file() and (out/"terminal.json").is_file() and not (out/"failure.json").exists()
    failed=tmp_path/"v2_failed"; failed.mkdir()
    receipt.write_receipt_transactionally(failed/"attempt.json",{"status":"ATTEMPT_PUBLISHED"})
    v1.publish_failure(failed,receipt,{"status":"CELL_FAILED"})
    assert (failed/"failure.json").is_file() and not (failed/"terminal.json").exists()

def test_v2_predecessor_gate_runs_before_any_result_root_action(tmp_path):
    events=[]
    def blocked(root):
        del root; events.append("predecessor"); raise predecessor.PredecessorError("blocked")
    p=v1.ExecutionProfile(cell=plan.CELL,schema=plan.SCHEMA,result_root_relative="v2root",bound_patterns=plan.BOUND_PATTERNS,review_evidence_paths=plan.REVIEW_EVIDENCE_PATHS,expected_sealed_sha256=plan.EXPECTED_SEALED_SHA256,work_order_relative=plan.WORK_ORDER_RELATIVE,predecessor_validator=blocked)
    args=type("Args",(),{"num_workers":0})()
    with pytest.raises(predecessor.PredecessorError): v1.execute(root=tmp_path,out_dir=tmp_path/"v2root",args=args,profile=p)
    assert events==["predecessor"] and not (tmp_path/"v2root").exists()

def test_v1_default_profile_is_unchanged():
    p=v1.v1_execution_profile(); assert p.cell=="PAIRED_ANCHORED_CALIBRATION_DROPOUT_V1" and p.result_root_relative.endswith("v1/smoke_seed42")
