"""No-data tests for V8's held V7 container-evidence reuse and host seam."""
from __future__ import annotations
import ast,json,subprocess,sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tfpd_exploration.submissions.evalai_m2_aof_scalar_static_v8 import binding,driver,plan
def _container():return {"image_id":"sha256:"+"a"*64,"image_tag":"synthetic-v8","base_image_id":"sha256:b179efcba8e36202aec688b3104397919576e30d4344c309febcf692d4d7aaf8","network_disabled":True,"pull":False,"container_removed":True,"command":["--evaluation","local","--split","m2","--phase","minival","--batch-size","7"],"stdout_sha256":"b"*64,"prediction_path":"/out/prediction.pkl","prediction_sha256":"c"*64,"target_path":"/out/ground_truth.pkl","target_sha256":"d"*64}
def test_v7_exact_five_body_held_fd_graph_and_sealed_container_evidence_are_bound():
 v7,payload=binding.validate_v7_failure_graph(ROOT),binding.validate_v2_sealed_artifact(ROOT)
 assert v7["bodies"]==plan.V7_FAILURE_BODIES and v7["closure_sha256"]==plan.V7_FAILURE_CLOSURE_SHA256 and payload["bodies"]==plan.V2_ARTIFACT_BODIES
 assert v7["container_minival"]["image_id"]=="sha256:e1451cb0db6913708fd779b78392b92ed7702f0038ce2bcd72960e4035cfc3c3"
def test_host_validator_bootstraps_streaming_before_unpickle_and_imports_numpy():
 source=ROOT/"tfpd_exploration/submissions/evalai_m2_aof_scalar_static_v1/validate_local.py";tree=ast.parse(source.read_text());imports=[n.names[0].name for n in tree.body if isinstance(n,ast.Import)]
 assert "numpy" in imports
 audit=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="audit_payload")
 calls=ast.unparse(audit);assert "ensure_streaming_paths" in calls and calls.index("ensure_streaming_paths")<calls.index("CPUUnpickler")
def test_v8_reuses_v7_evidence_without_docker_and_preserves_failure_prefix(tmp_path):
 build={"payload_sha256":"a"*64,"receipt_sha256":"b"*64,"payload":{},"v7_container_minival":_container()}
 success=driver._execute_synthetic_for_test(repo_root=ROOT,result_root=tmp_path/"success",build=lambda _:build,validate=lambda _:{"host":True},docker=lambda:{"container_minival":_container()})
 assert success["status"]=="LOCAL_BUILD_V8_VALIDATED_NOT_SUBMITTED"
 terminal=json.loads((tmp_path/"success"/"terminal.json").read_text());assert terminal["schema"]=="m2_aof_scalar_static_package_v8_terminal"
 assert not (ROOT/plan.ARTIFACT_ROOT_RELATIVE).exists()
def test_inert_cli_and_closure_include_v1_host_validator():
 m=driver.closure_map(ROOT);leaf="tfpd_exploration/submissions/evalai_m2_aof_scalar_static_v1/validate_local.py";assert leaf in m
 run=subprocess.run([sys.executable,str(ROOT/"tfpd_exploration/scripts/run_evalai_m2_aof_scalar_static_v8.py"),"--dry-run"],check=True,capture_output=True,text=True);assert "READY_REQUIRES_OPAQUE_CAPABILITY" in run.stdout
def test_route_artifact_parent_requires_predecessor_prefix(tmp_path):
 package,result=tmp_path/plan.PACKAGE_RELATIVE,tmp_path/"result";package.mkdir(parents=True);result.mkdir()
 with pytest.raises(Exception,match="only after"):driver._create_artifact(root=tmp_path,result=result)
 for n in driver._PRE:(result/n).write_text(n)
 made=driver._create_artifact(root=tmp_path,result=result);assert made.is_dir() and set(made.parent.iterdir())=={made}
