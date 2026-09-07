"""No-data/no-Docker tests for additive AOF-S V5 parent-layout recovery."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tfpd_exploration.submissions.evalai_m2_aof_scalar_static_v5 import binding, docker_argv, driver, plan


def _container() -> dict[str, object]:
    return {"image_id": "sha256:" + "a" * 64, "image_tag": "synthetic-v5", "base_image_id": plan.LOCAL_DOCKER_BASE_ID,
            "network_disabled": True, "pull": False, "container_removed": True,
            "command": ["--evaluation", "local", "--split", "m2", "--phase", "minival", "--batch-size", "7"],
            "stdout_sha256": "b" * 64, "prediction_path": "/out/prediction.pkl", "prediction_sha256": "c" * 64,
            "target_path": "/out/ground_truth.pkl", "target_sha256": "d" * 64}


def _prefix(root: Path) -> None:
    root.mkdir(parents=True)
    for name in driver._PRE_BUILD_PREFIX:
        (root / name).write_text(name, encoding="ascii")


def test_exact_v4_failure_and_v2_artifact_are_held_fd_witnesses():
    failure = binding.validate_v4_failure_graph(ROOT)
    artifact = binding.validate_v2_sealed_artifact(ROOT)
    assert failure["bodies"] == plan.V4_FAILURE_BODIES
    assert failure["closure_sha256"] == plan.V4_FAILURE_CLOSURE_SHA256
    assert artifact["bodies"] == plan.V2_ARTIFACT_BODIES


def test_missing_route_owned_artifact_parent_is_created_only_after_prefix_and_has_exact_topology(tmp_path):
    package = tmp_path / plan.PACKAGE_RELATIVE
    package.mkdir(parents=True)
    result = tmp_path / "result"
    _prefix(result)
    artifact_parent = tmp_path / plan.ARTIFACT_ROOT_RELATIVE
    assert not artifact_parent.parent.exists()
    created = driver._create_route_artifact_root(repo_root=tmp_path, result_root=result)
    assert created == artifact_parent and created.is_dir()
    assert set(created.parent.iterdir()) == {created}
    with pytest.raises(Exception, match="fresh"):
        driver._create_route_artifact_root(repo_root=tmp_path, result_root=result)
    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    with pytest.raises(Exception, match="only after attempt/predecessor"):
        driver._create_route_artifact_root(repo_root=tmp_path, result_root=incomplete)
    assert not (tmp_path / "incomplete" / "artifacts").exists()


def test_v5_dockerfile_remains_exact_v4_layout_and_fake_argv_keeps_contract(tmp_path):
    v4 = ROOT / "tfpd_exploration/submissions/evalai_m2_aof_scalar_static_v4/Dockerfile"
    v5 = ROOT / plan.PACKAGE_RELATIVE / "Dockerfile"
    assert v5.read_bytes() == v4.read_bytes()
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    data = tmp_path / "SPINT-main/data"
    data.mkdir(parents=True)
    seen: list[list[str]] = []

    def fake(argv, *, shell, check, capture_output, text):
        assert shell is False and check is False and capture_output and text
        seen.append(list(argv))
        if argv[1:3] == ["image", "inspect"] and argv[-1] == plan.LOCAL_DOCKER_BASE_TAG:
            return subprocess.CompletedProcess(argv, 0, plan.LOCAL_DOCKER_BASE_ID + "\n", "")
        if argv[1:3] == ["image", "inspect"]:
            return subprocess.CompletedProcess(argv, 0, "sha256:" + "e" * 64 + "\n", "")
        if argv[1] == "run":
            output = artifact / "container_minival"
            output.mkdir(exist_ok=True)
            (output / "prediction.pkl").write_bytes(b"prediction")
            (output / "ground_truth.pkl").write_bytes(b"target")
        return subprocess.CompletedProcess(argv, 0, "local output", "")

    evidence = docker_argv.build_and_validate_container(repo_root=tmp_path, artifact_root=artifact, runner=fake)
    assert evidence["network_disabled"] is True and evidence["image_tag"] == plan.IMAGE_TAG
    assert seen[1][1:4] == ["build", "--network=none", "--pull=false"]
    assert seen[-1][1:5] == ["run", "--network=none", "--pull=never", "--rm"]
    assert seen[-1][5:13] == ["--env", "EVAL_DATA_PATH=/input", "--env", "PREDICTION_PATH_LOCAL=/output/prediction.pkl",
                              "--env", "GT_PATH=/output/ground_truth.pkl", "--env", "CUDA_VISIBLE_DEVICES="]
    assert f"{data}:/input:ro" in seen[-1] and f"{artifact / 'container_minival'}:/output:rw" in seen[-1]
    assert all(command[0] == "/usr/bin/docker" for command in seen)


def test_shared_lifecycle_and_inert_cli_do_not_create_real_artifacts(tmp_path):
    success = driver._execute_synthetic_for_test(
        repo_root=ROOT, result_root=tmp_path / "success",
        build=lambda payload: {"payload_sha256": "a" * 64, "receipt_sha256": "b" * 64, "payload": {}, "session_records": {}},
        validate=lambda payload: {"host_minival": {"network": False}}, docker=lambda: {"container_minival": _container()})
    assert success["status"] == "LOCAL_BUILD_V5_VALIDATED_NOT_SUBMITTED"
    assert json.loads((tmp_path / "success" / "terminal.json").read_text())["schema"] == "m2_aof_scalar_static_package_v5_terminal"
    assert not (ROOT / plan.ARTIFACT_ROOT_RELATIVE).exists()
    completed = subprocess.run([sys.executable, str(ROOT / "tfpd_exploration/scripts/run_evalai_m2_aof_scalar_static_v5.py"), "--dry-run"],
                               check=True, capture_output=True, text=True)
    assert "READY_REQUIRES_OPAQUE_CAPABILITY" in completed.stdout


@pytest.mark.parametrize("argv", [["docker", "build"], ["/usr/bin/docker", "login"], ["/usr/bin/docker", "run", "pull"]])
def test_docker_argv_rejects_nonlocal_or_registry_commands(argv):
    with pytest.raises(Exception):
        docker_argv._run(argv, runner=lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, "", ""))
