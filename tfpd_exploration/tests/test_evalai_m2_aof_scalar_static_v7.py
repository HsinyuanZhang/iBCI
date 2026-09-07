"""No-data/no-Docker proof tests for the thin AOF-S V7 PYTHONPATH recovery."""
from __future__ import annotations
import hashlib, subprocess, sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tfpd_exploration.submissions.evalai_m2_aof_scalar_static_v7 import binding,docker_argv,driver,plan,staged_context

def _container():
    return {"image_id":"sha256:"+"a"*64,"image_tag":"synthetic-v7","base_image_id":plan.LOCAL_DOCKER_BASE_ID,"network_disabled":True,"pull":False,"container_removed":True,"command":["--evaluation","local","--split","m2","--phase","minival","--batch-size","7"],"stdout_sha256":"b"*64,"prediction_path":"/out/prediction.pkl","prediction_sha256":"c"*64,"target_path":"/out/ground_truth.pkl","target_sha256":"d"*64}
def test_v6_exact_held_fd_graph_and_v2_payload_are_bound():
    failure,art=binding.validate_v6_failure_graph(ROOT),binding.validate_v2_sealed_artifact(ROOT)
    assert failure["bodies"]==plan.V6_FAILURE_BODIES and failure["closure_sha256"]==plan.V6_FAILURE_CLOSURE_SHA256 and art["bodies"]==plan.V2_ARTIFACT_BODIES
    incident=(ROOT/plan.V6_INCIDENT_RELATIVE).read_text(encoding="utf-8")
    assert "ModuleNotFoundError" in incident and "src" in incident
def test_closure_includes_transitively_imported_profile_leaf():
    mapping=driver.closure_map(ROOT)
    profile="tfpd_exploration/submissions/evalai_m2_aof_scalar_static_v7/profile.py"
    assert profile in plan.STATIC_CLOSURE_RELATIVES and mapping[profile]==plan.sha256_file(ROOT/profile)
def test_literal_root_pythonpath_is_exact_and_omission_or_reordering_is_rejected(tmp_path):
    docker=(ROOT/plan.PACKAGE_RELATIVE/"Dockerfile").read_text(); literal="ENV PYTHONPATH=/workspace/tfpd_exploration/submissions/evalai_m2_aof_scalar_static_v1:/workspace:/"
    assert literal in docker and "ENV PYTHONPATH=/workspace:/workspace/tfpd_exploration" not in docker and "ENV PYTHONPATH=/workspace/tfpd_exploration/submissions/evalai_m2_aof_scalar_static_v1:/workspace\n" not in docker
    artifact=tmp_path/"artifact";artifact.mkdir(); evidence=staged_context.stage_docker_context(repo_root=ROOT,artifact_root=artifact)
    assert evidence==staged_context.validate_staged_context(repo_root=ROOT,artifact_root=artifact)
    assert hashlib.sha256((artifact/plan.STAGED_CONTEXT_NAME/"decoder.pkl").read_bytes()).hexdigest()==plan.V2_ARTIFACT_BODIES[plan.PAYLOAD_NAME]
    assert set(evidence["bodies"])=={"Dockerfile","aofs_static_decoder.py","laws.py","decode.py","decoder.pkl","third_party/__init__.py","third_party/falcon_challenge/__init__.py","third_party/falcon_challenge/filtering.py"}
def test_fake_argv_uses_staged_context_not_repo_root(tmp_path):
    artifact,data,seen=tmp_path/"artifact",tmp_path/"SPINT-main/data",[];artifact.mkdir();data.mkdir(parents=True);staged_context.stage_docker_context(repo_root=ROOT,artifact_root=artifact)
    def fake(argv,*,shell,check,capture_output,text):
        assert shell is False and check is False and capture_output and text;seen.append(list(argv))
        if argv[1:3]==["image","inspect"] and argv[-1]==plan.LOCAL_DOCKER_BASE_TAG:return subprocess.CompletedProcess(argv,0,plan.LOCAL_DOCKER_BASE_ID+"\n","")
        if argv[1:3]==["image","inspect"]:return subprocess.CompletedProcess(argv,0,"sha256:"+"e"*64+"\n","")
        if argv[1]=="run":
            out=artifact/"container_minival";out.mkdir(exist_ok=True);(out/"prediction.pkl").write_bytes(b"p");(out/"ground_truth.pkl").write_bytes(b"t")
        return subprocess.CompletedProcess(argv,0,"offline","")
    docker_argv.build_and_validate_container(repo_root=tmp_path,artifact_root=artifact,runner=fake,context_repo_root=ROOT)
    assert seen[1][1:4]==["build","--network=none","--pull=false"] and seen[1][-1]==str(artifact/plan.STAGED_CONTEXT_NAME) and seen[1][-1]!=str(ROOT)
    assert seen[-1][1:5]==["run","--network=none","--pull=never","--rm"] and seen[-1][5:13]==["--env","EVAL_DATA_PATH=/input","--env","PREDICTION_PATH_LOCAL=/output/prediction.pkl","--env","GT_PATH=/output/ground_truth.pkl","--env","CUDA_VISIBLE_DEVICES="]
def test_route_parent_synthetic_lifecycle_and_inert_cli_are_no_data(tmp_path):
    package,result=tmp_path/plan.PACKAGE_RELATIVE,tmp_path/"result";package.mkdir(parents=True);result.mkdir()
    for n in driver._PRE_BUILD_PREFIX:(result/n).write_text(n)
    made=driver._create_route_artifact_root(repo_root=tmp_path,result_root=result);assert made.is_dir() and set(made.parent.iterdir())=={made}
    success=driver._execute_synthetic_for_test(repo_root=ROOT,result_root=tmp_path/"success",build=lambda _:{"payload_sha256":"a"*64,"receipt_sha256":"b"*64,"payload":{},"session_records":{}},validate=lambda _:{},docker=lambda:{"container_minival":_container()})
    assert success["status"]=="LOCAL_BUILD_V7_VALIDATED_NOT_SUBMITTED" and not (ROOT/plan.ARTIFACT_ROOT_RELATIVE).exists()
    run=subprocess.run([sys.executable,str(ROOT/"tfpd_exploration/scripts/run_evalai_m2_aof_scalar_static_v7.py"),"--dry-run"],check=True,capture_output=True,text=True);assert "READY_REQUIRES_OPAQUE_CAPABILITY" in run.stdout
@pytest.mark.parametrize("argv",[["docker","build"],["/usr/bin/docker","login"],["/usr/bin/docker","run","pull"]])
def test_docker_argv_rejects_nonlocal_or_registry_commands(argv):
    with pytest.raises(Exception):docker_argv._run(argv,runner=lambda *a,**k:subprocess.CompletedProcess(a[0],0,"",""))
