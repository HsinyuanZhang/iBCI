"""No-data/no-Docker proof tests for additive AOF-S V6 dependency recovery."""
from __future__ import annotations
import hashlib
import json
import subprocess
import sys
from pathlib import Path, PurePosixPath
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tfpd_exploration.submissions.evalai_m2_aof_scalar_static_v6 import binding, docker_argv, driver, plan, staged_context


def _container() -> dict[str, object]:
    return {"image_id": "sha256:" + "a" * 64, "image_tag": "synthetic-v6", "base_image_id": plan.LOCAL_DOCKER_BASE_ID,
            "network_disabled": True, "pull": False, "container_removed": True, "command": ["--evaluation", "local", "--split", "m2", "--phase", "minival", "--batch-size", "7"],
            "stdout_sha256": "b" * 64, "prediction_path": "/out/prediction.pkl", "prediction_sha256": "c" * 64, "target_path": "/out/ground_truth.pkl", "target_sha256": "d" * 64}


def test_exact_v5_failure_v2_payload_and_incident_bind_missing_dependency_cause():
    failure, artifact = binding.validate_v5_failure_graph(ROOT), binding.validate_v2_sealed_artifact(ROOT)
    assert failure["bodies"] == plan.V5_FAILURE_BODIES and artifact["bodies"] == plan.V2_ARTIFACT_BODIES
    incident = (ROOT / plan.V5_INCIDENT_RELATIVE).read_text(encoding="utf-8")
    assert "site-packages" in incident and "third_party      -> ModuleNotFoundError" in incident


def test_staged_context_dockerfile_copies_exact_frozen_third_party_and_preserves_nested_layout(tmp_path):
    lines = (ROOT / plan.PACKAGE_RELATIVE / "Dockerfile").read_text().splitlines()
    nested = PurePosixPath("/workspace/tfpd_exploration/submissions/evalai_m2_aof_scalar_static_v1")
    assert lines[2] == "COPY aofs_static_decoder.py laws.py " + str(nested) + "/"
    assert lines[3] == "COPY decode.py /workspace/"
    assert (nested / "aofs_static_decoder.py").parents[3] == PurePosixPath("/workspace")
    assert (nested / "laws.py").parents[3] == PurePosixPath("/workspace")
    assert "COPY decoder.pkl /data/decoder.pkl" in lines
    assert lines[4:7] == ["COPY third_party/__init__.py /workspace/third_party/__init__.py",
                           "COPY third_party/falcon_challenge/__init__.py /workspace/third_party/falcon_challenge/__init__.py",
                           "COPY third_party/falcon_challenge/filtering.py /workspace/third_party/falcon_challenge/filtering.py"]
    assert {relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() for relative in plan.THIRD_PARTY_BODIES} == plan.THIRD_PARTY_BODIES
    artifact = tmp_path / "artifact"; artifact.mkdir()
    evidence = staged_context.stage_docker_context(repo_root=ROOT, artifact_root=artifact)
    assert evidence == staged_context.validate_staged_context(repo_root=ROOT, artifact_root=artifact)
    assert set(evidence["bodies"]) == {"Dockerfile", "aofs_static_decoder.py", "laws.py", "decode.py", "decoder.pkl", "third_party/__init__.py", "third_party/falcon_challenge/__init__.py", "third_party/falcon_challenge/filtering.py"}
    (artifact / plan.STAGED_CONTEXT_NAME / "unexpected").write_text("x", encoding="ascii")
    with pytest.raises(Exception, match="exact topology"):
        staged_context.validate_staged_context(repo_root=ROOT, artifact_root=artifact)


def test_fake_argv_uses_exact_staged_context_and_networkless_env_mount_contract(tmp_path):
    artifact, data, seen = tmp_path / "artifact", tmp_path / "SPINT-main/data", []
    artifact.mkdir(); data.mkdir(parents=True); staged_context.stage_docker_context(repo_root=ROOT, artifact_root=artifact)
    def fake(argv, *, shell, check, capture_output, text):
        assert shell is False and check is False and capture_output and text; seen.append(list(argv))
        if argv[1:3] == ["image", "inspect"] and argv[-1] == plan.LOCAL_DOCKER_BASE_TAG: return subprocess.CompletedProcess(argv, 0, plan.LOCAL_DOCKER_BASE_ID + "\n", "")
        if argv[1:3] == ["image", "inspect"]: return subprocess.CompletedProcess(argv, 0, "sha256:" + "e" * 64 + "\n", "")
        if argv[1] == "run":
            output = artifact / "container_minival"; output.mkdir(exist_ok=True); (output / "prediction.pkl").write_bytes(b"p"); (output / "ground_truth.pkl").write_bytes(b"t")
        return subprocess.CompletedProcess(argv, 0, "offline", "")
    evidence = docker_argv.build_and_validate_container(repo_root=tmp_path, artifact_root=artifact, runner=fake, context_repo_root=ROOT)
    assert seen[1][1:4] == ["build", "--network=none", "--pull=false"] and seen[1][-1] == str(artifact / plan.STAGED_CONTEXT_NAME)
    assert seen[1][-1] != str(ROOT)
    assert seen[-1][1:5] == ["run", "--network=none", "--pull=never", "--rm"]
    assert seen[-1][5:13] == ["--env", "EVAL_DATA_PATH=/input", "--env", "PREDICTION_PATH_LOCAL=/output/prediction.pkl", "--env", "GT_PATH=/output/ground_truth.pkl", "--env", "CUDA_VISIBLE_DEVICES="]
    assert f"{data}:/input:ro" in seen[-1] and f"{artifact / 'container_minival'}:/output:rw" in seen[-1] and evidence["image_tag"] == plan.IMAGE_TAG


def test_route_parent_and_synthetic_lifecycle_are_no_data_and_cli_is_inert(tmp_path):
    package, result = tmp_path / plan.PACKAGE_RELATIVE, tmp_path / "result"; package.mkdir(parents=True); result.mkdir()
    for name in driver._PRE_BUILD_PREFIX: (result / name).write_text(name)
    artifact = driver._create_route_artifact_root(repo_root=tmp_path, result_root=result)
    assert artifact.is_dir() and set(artifact.parent.iterdir()) == {artifact}
    success = driver._execute_synthetic_for_test(repo_root=ROOT, result_root=tmp_path / "success", build=lambda _: {"payload_sha256": "a" * 64, "receipt_sha256": "b" * 64, "payload": {}, "session_records": {}}, validate=lambda _: {}, docker=lambda: {"container_minival": _container()})
    assert success["status"] == "LOCAL_BUILD_V6_VALIDATED_NOT_SUBMITTED" and not (ROOT / plan.ARTIFACT_ROOT_RELATIVE).exists()
    completed = subprocess.run([sys.executable, str(ROOT / "tfpd_exploration/scripts/run_evalai_m2_aof_scalar_static_v6.py"), "--dry-run"], check=True, capture_output=True, text=True)
    assert "READY_REQUIRES_OPAQUE_CAPABILITY" in completed.stdout


@pytest.mark.parametrize("argv", [["docker", "build"], ["/usr/bin/docker", "login"], ["/usr/bin/docker", "run", "pull"]])
def test_docker_argv_rejects_nonlocal_or_registry_commands(argv):
    with pytest.raises(Exception): docker_argv._run(argv, runner=lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, "", ""))
