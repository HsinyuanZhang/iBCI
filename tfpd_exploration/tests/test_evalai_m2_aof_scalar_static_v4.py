"""No-data/no-Docker proof tests for the additive AOF-S V4 layout repair."""
from __future__ import annotations

import json
import hashlib
import subprocess
import sys
from pathlib import Path, PurePosixPath

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tfpd_exploration.submissions.evalai_m2_aof_scalar_static_v4 import binding, docker_argv, driver, plan


def _container() -> dict[str, object]:
    return {
        "image_id": "sha256:" + "a" * 64, "image_tag": "synthetic-v4", "base_image_id": plan.LOCAL_DOCKER_BASE_ID,
        "network_disabled": True, "pull": False, "container_removed": True,
        "command": ["--evaluation", "local", "--split", "m2", "--phase", "minival", "--batch-size", "7"],
        "stdout_sha256": "b" * 64, "prediction_path": "/out/prediction.pkl", "prediction_sha256": "c" * 64,
        "target_path": "/out/ground_truth.pkl", "target_sha256": "d" * 64,
    }


def test_exact_v3_failure_and_v2_artifact_are_held_fd_witnesses():
    failure = binding.validate_v3_failure_graph(ROOT)
    artifact = binding.validate_v2_sealed_artifact(ROOT)
    assert failure["bodies"] == plan.V3_FAILURE_BODIES
    assert failure["closure_sha256"] == plan.V3_FAILURE_CLOSURE_SHA256
    assert artifact["bodies"] == plan.V2_ARTIFACT_BODIES
    assert artifact["receipt"]["payload_sha256"] == plan.V2_ARTIFACT_BODIES[plan.PAYLOAD_NAME]


def test_dockerfile_is_layout_only_and_restores_decoder_parent_depth():
    dockerfile = ROOT / plan.PACKAGE_RELATIVE / "Dockerfile"
    lines = dockerfile.read_text(encoding="utf-8").splitlines()
    nested = PurePosixPath("/workspace/tfpd_exploration/submissions/evalai_m2_aof_scalar_static_v1/aofs_static_decoder.py")
    assert lines == [
        "FROM spint-m2:e8-epoch027-76f0fb2", "WORKDIR /workspace",
        "COPY evalai_m2_aof_scalar_static_v1/aofs_static_decoder.py " + str(nested),
        "COPY evalai_m2_aof_scalar_static_v1/laws.py evalai_m2_aof_scalar_static_v1/decode.py /workspace/",
        "ENV PYTHONPATH=/workspace/tfpd_exploration/submissions/evalai_m2_aof_scalar_static_v1:/workspace",
        "COPY evalai_m2_aof_scalar_static_v2/artifacts/local_build_v2/t4_m2_seed42_dopt4_act30_aofs_identity.pkl /data/decoder.pkl",
        "ENTRYPOINT [\"python\", \"/workspace/decode.py\"]",
    ]
    assert nested.parents[3] == PurePosixPath("/workspace")
    source_hashes = {
        "aofs_static_decoder.py": plan.V1_DECODER_SHA256,
        "laws.py": plan.V1_LAWS_SHA256,
        "decode.py": plan.V1_DECODE_SHA256,
    }
    source = ROOT / "tfpd_exploration/submissions/evalai_m2_aof_scalar_static_v1"
    assert {name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in source_hashes} == source_hashes
    assert "aofs_static_decoder.py /workspace/aofs_static_decoder.py" not in dockerfile.read_text(encoding="utf-8")


def test_fake_argv_proves_networkless_build_and_exact_mount_environment_contract(tmp_path):
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

    evidence = docker_argv.build_and_validate_container(repo_root=tmp_path, artifact_root=artifact, runner=fake)
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


def test_shared_lifecycle_success_and_failure_prefix_remain_no_data(tmp_path):
    success = driver._execute_synthetic_for_test(
        repo_root=ROOT, result_root=tmp_path / "success",
        build=lambda payload: {"payload_sha256": "a" * 64, "receipt_sha256": "b" * 64, "payload": {}, "session_records": {}},
        validate=lambda payload: {"host_minival": {"network": False}}, docker=lambda: {"container_minival": _container()},
    )
    assert success["status"] == "LOCAL_BUILD_V4_VALIDATED_NOT_SUBMITTED"
    terminal = json.loads((tmp_path / "success" / "terminal.json").read_text())
    assert terminal["schema"] == "m2_aof_scalar_static_package_v4_terminal"
    with pytest.raises(Exception, match="local build failed"):
        driver._execute_synthetic_for_test(
            repo_root=ROOT, result_root=tmp_path / "failure",
            build=lambda payload: (_ for _ in ()).throw(ValueError("synthetic reuse failure")),
            validate=lambda payload: {}, docker=lambda: {"container_minival": _container()},
        )
    failure = json.loads((tmp_path / "failure" / "failure.json").read_text())
    assert failure["published_prefix"] == ["attempt.json", "predecessor_authority.json"]
    assert not (tmp_path / "failure" / "terminal.json").exists()


def test_closure_is_exact_and_public_cli_is_inert():
    mapping = driver.closure_map(ROOT)
    assert set(mapping) == set(plan.STATIC_CLOSURE_RELATIVES)
    completed = subprocess.run([sys.executable, str(ROOT / "tfpd_exploration/scripts/run_evalai_m2_aof_scalar_static_v4.py"), "--dry-run"],
                               check=True, capture_output=True, text=True)
    assert "READY_REQUIRES_OPAQUE_CAPABILITY" in completed.stdout


@pytest.mark.parametrize("argv", [
    ["docker", "build", "--network=none"], ["/usr/bin/docker", "login"], ["/usr/bin/docker", "run", "--network=none", "pull"],
])
def test_docker_argv_rejects_nonlocal_or_registry_commands(argv):
    with pytest.raises(Exception):
        docker_argv._run(argv, runner=lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, "", ""))
