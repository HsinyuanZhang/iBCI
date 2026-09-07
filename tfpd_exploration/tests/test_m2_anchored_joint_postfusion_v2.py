"""No-CUDA V2 successor/incident binding checks."""
from __future__ import annotations
import json, os, sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
def test_v2_static_closure_and_held_v1_pre_step_incident():
    from tfpd_exploration.src.m2_anchored_joint_postfusion_v2 import lifecycle, plan
    values=lifecycle.closure(ROOT)
    assert plan.INCIDENT_RELATIVE in values
    held=lifecycle.hold_v1_failure(ROOT)
    assert held["pre_source_pre_step"] is True and held["v1_failure_sha256"]==plan.V1_FAILURE_SHA256
def test_v2_holder_uses_scorer_student_digest_law_cpu_only():
    torch=pytest.importorskip("torch")
    from tfpd_exploration.src.m2_anchored_joint_postfusion_v2 import runner
    class Student(torch.nn.Module):
        def __init__(self): super().__init__(); self.weight=torch.nn.Parameter(torch.ones(1))
    from tfpd_exploration.src.m2_postfusion_checkpoint_score_v1 import physical
    holder=runner._StudentHolder(Student())
    assert runner.strict_student_digest.__name__ == "strict_student_digest"
    assert len(physical._student_state_sha(holder)) == 64
def test_v2_preflight_is_attempt_first_and_does_not_run_without_empty_cvd(tmp_path,monkeypatch):
    from tfpd_exploration.src.m2_anchored_joint_postfusion_v2 import production
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES","0")
    with pytest.raises(Exception,match="CPU-only CVD"):
        production.execute_preflight(repo_root=ROOT,root=tmp_path/"v2")
    assert not (tmp_path/"v2").exists()

def test_v2_preflight_lifecycle_binds_held_incident_before_digest_receipt(tmp_path,monkeypatch):
    from tfpd_exploration.src.m2_anchored_joint_postfusion_v2 import production
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES","")
    monkeypatch.setattr(production.runner,"source_only_cpu_strict_load",lambda **_:{"cpu_only":True,"cuda_initialized":False,"strict_load":True,"student_state_sha256":"2a340745e2e1b4c7eecb4b187c9b061548bf3686f12389aae76bf53e4f3acc20","selected_t4_checkpoint_sha256":"25d7bc72b4d440004b58f1beaeadb7e15565a43e83dd1eadd160374270ec1d3e","pit_prepare_calls":1})
    root=tmp_path/"v2"; result=production.execute_preflight(repo_root=ROOT,root=root)
    assert Path(result["root"]) == root
    assert {path.name for path in root.iterdir() if not path.name.endswith(".sha256")} == {"attempt.json","v1_incident.json","strict_digest_regression.json","terminal.json"}

def test_v2_composed_capability_requires_independent_combined_review_and_is_one_shot(tmp_path):
    from tfpd_exploration.src.m2_anchored_joint_postfusion_v2 import lifecycle
    reviewed=lifecycle.combined_closure(ROOT); digest=lifecycle.closure_digest(reviewed)
    cap=lifecycle.issue_after_independent_review(reviewed_map=reviewed,digest=digest,repo_root=ROOT,outer_root=tmp_path/"outer")
    root,identity=lifecycle.consume(cap)
    assert root.is_dir() and identity == (root.stat().st_dev,root.stat().st_ino)
    with pytest.raises(Exception,match="consumed"):
        lifecycle.consume(cap)

def test_v2_private_synthetic_composed_success_and_failure_prefix(tmp_path):
    from tfpd_exploration.src.m2_anchored_joint_postfusion_v2 import lifecycle, production
    reviewed=lifecycle.combined_closure(ROOT); digest=lifecycle.closure_digest(reviewed)
    success=lifecycle.issue_after_independent_review(reviewed_map=reviewed,digest=digest,repo_root=ROOT,outer_root=tmp_path/"success")
    value=production._execute_synthetic_composed_for_test(cap=success)
    root=Path(value["root"]); assert (root/"terminal.json").is_file() and (root/"training"/"terminal.json").is_file() and (root/"score"/"terminal.json").is_file()
    failed=lifecycle.issue_after_independent_review(reviewed_map=reviewed,digest=digest,repo_root=ROOT,outer_root=tmp_path/"failure")
    with pytest.raises(RuntimeError,match="synthetic inner admission"):
        production._execute_synthetic_composed_for_test(cap=failed,fail_after_incident=True)
    body=json.loads((tmp_path/"failure"/"failure.json").read_text())
    assert body["published_prefix"] == ["attempt.json","v1_incident.json"]
