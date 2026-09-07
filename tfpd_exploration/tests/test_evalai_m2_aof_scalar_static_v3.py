"""No-data AOF-S V3 recovery and route-owned Docker argv tests."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tfpd_exploration.submissions.evalai_m2_aof_scalar_static_v3 import binding, docker_argv, driver, plan


def _container() -> dict[str, object]:
    return {
        "image_id": "sha256:" + "a" * 64, "image_tag": "synthetic-v3",
        "base_image_id": plan.LOCAL_DOCKER_BASE_ID, "network_disabled": True, "pull": False,
        "container_removed": True,
        "command": ["--evaluation", "local", "--split", "m2", "--phase", "minival", "--batch-size", "7"],
        "stdout_sha256": "b" * 64, "prediction_path": "/out/prediction.pkl", "prediction_sha256": "c" * 64,
        "target_path": "/out/ground_truth.pkl", "target_sha256": "d" * 64,
    }


def test_actual_v2_failure_and_artifact_are_exact_held_fd_witnesses():
    failure = binding.validate_v2_recovery_graph(ROOT)
    artifact = binding.validate_v2_sealed_artifact(ROOT)
    assert failure["bodies"] == plan.V2_FAILURE_BODIES
    assert failure["closure_sha256"] == plan.V2_FAILURE_CLOSURE_SHA256
    assert artifact["bodies"] == plan.V2_ARTIFACT_BODIES
    assert artifact["receipt"]["payload_sha256"] == plan.V2_ARTIFACT_BODIES[plan.PAYLOAD_NAME]


def test_v3_profiled_shared_lifecycle_success_and_failure_prefix(tmp_path):
    success = driver._execute_synthetic_for_test(
        repo_root=ROOT, result_root=tmp_path / "success",
        build=lambda payload: {"payload_sha256": "a" * 64, "receipt_sha256": "b" * 64, "payload": {}, "session_records": {}},
        validate=lambda payload: {"host_minival": {"network": False}},
        docker=lambda: {"container_minival": _container()},
    )
    assert success["status"] == "LOCAL_BUILD_V3_VALIDATED_NOT_SUBMITTED"
    result_root = tmp_path / "success"
    terminal = json.loads((result_root / "terminal.json").read_text())
    assert terminal["schema"] == "m2_aof_scalar_static_package_v3_terminal"
    assert not (result_root / "failure.json").exists()
    with pytest.raises(Exception, match="local build failed"):
        driver._execute_synthetic_for_test(
            repo_root=ROOT, result_root=tmp_path / "failure",
            build=lambda payload: (_ for _ in ()).throw(ValueError("synthetic reuse failure")),
            validate=lambda payload: {}, docker=lambda: {"container_minival": _container()},
        )
    failure = json.loads((tmp_path / "failure" / "failure.json").read_text())
    assert failure["published_prefix"] == ["attempt.json", "predecessor_authority.json"]
    assert not (tmp_path / "failure" / "terminal.json").exists()


def test_capability_root_drift_and_reuse_reject(tmp_path):
    mapping = driver.closure_map(ROOT)
    cap = driver._issue(repo_root=ROOT, result_root=tmp_path / "one", reviewed=mapping, digest=driver.closure_sha256(mapping))
    driver._consume(cap, ROOT)
    with pytest.raises(Exception, match="already consumed"):
        driver._consume(cap, ROOT)
    parent = tmp_path / "parent"
    parent.mkdir()
    cap2 = driver._issue(repo_root=ROOT, result_root=parent / "two", reviewed=mapping, digest=driver.closure_sha256(mapping))
    parent.rename(tmp_path / "moved")
    parent.mkdir()
    with pytest.raises(Exception, match="root/parent drift"):
        driver._consume(cap2, ROOT)


def test_docker_argv_is_local_networkless_and_never_sdk_or_shell(tmp_path):
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    data = tmp_path / "SPINT-main" / "data"
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
            output.mkdir(parents=True, exist_ok=True)
            (output / "prediction.pkl").write_bytes(b"prediction")
            (output / "ground_truth.pkl").write_bytes(b"target")
        return subprocess.CompletedProcess(argv, 0, "local output", "")
    # The routine must receive a repo root bearing the reviewed local data path.
    repo = tmp_path
    evidence = docker_argv.build_and_validate_container(repo_root=repo, artifact_root=artifact, runner=fake)
    assert evidence["network_disabled"] is True and evidence["pull"] is False
    assert seen[1][1:4] == ["build", "--network=none", "--pull=false"]
    assert seen[-1][1:5] == ["run", "--network=none", "--pull=never", "--rm"]
    assert seen[-1][5:13] == [
        "--env", "EVAL_DATA_PATH=/input", "--env", "PREDICTION_PATH_LOCAL=/output/prediction.pkl",
        "--env", "GT_PATH=/output/ground_truth.pkl", "--env", "CUDA_VISIBLE_DEVICES=",
    ]
    assert f"{data}:/input:ro" in seen[-1] and f"{artifact / 'container_minival'}:/output:rw" in seen[-1]
    assert all(command[0] == "/usr/bin/docker" for command in seen)
    assert not any({"login", "push", "pull"}.intersection(command) for command in seen)


def test_docker_argv_rejects_missing_symlinked_or_preexisting_minival_paths(tmp_path):
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    with pytest.raises(Exception, match="data root invalid"):
        docker_argv.build_and_validate_container(repo_root=tmp_path, artifact_root=artifact,
                                                 runner=lambda *a, **k: None)
    data_parent = tmp_path / "SPINT-main"
    data_parent.mkdir()
    actual = tmp_path / "actual_data"
    actual.mkdir()
    (data_parent / "data").symlink_to(actual, target_is_directory=True)
    with pytest.raises(Exception, match="data root invalid"):
        docker_argv.build_and_validate_container(repo_root=tmp_path, artifact_root=artifact,
                                                 runner=lambda *a, **k: None)
    (data_parent / "data").unlink()
    (data_parent / "data").mkdir()
    (artifact / "container_minival").mkdir()
    with pytest.raises(Exception, match="output must be fresh"):
        docker_argv.build_and_validate_container(repo_root=tmp_path, artifact_root=artifact,
                                                 runner=lambda *a, **k: None)


@pytest.mark.parametrize("argv", [
    ["docker", "build", "--network=none"],
    ["/usr/bin/docker", "login"],
    ["/usr/bin/docker", "run", "--network=none", "pull"],
])
def test_docker_argv_rejects_nonlocal_or_registry_commands(argv):
    with pytest.raises(Exception):
        docker_argv._run(argv, runner=lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, "", ""))


def test_v3_closure_is_exact_and_cli_is_inert():
    mapping = driver.closure_map(ROOT)
    assert set(mapping) == set(plan.STATIC_CLOSURE_RELATIVES)
    command = [sys.executable, str(ROOT / "tfpd_exploration/scripts/run_evalai_m2_aof_scalar_static_v3.py"), "--dry-run"]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    assert "READY_REQUIRES_OPAQUE_CAPABILITY" in completed.stdout
