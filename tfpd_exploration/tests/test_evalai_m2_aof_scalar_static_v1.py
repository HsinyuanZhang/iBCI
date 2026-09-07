"""No-data, no-network contract tests for the AOF-S static package."""
from __future__ import annotations

import shutil
import os
import subprocess
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE = REPO_ROOT / "tfpd_exploration/submissions/evalai_m2_aof_scalar_static_v1"
if str(PACKAGE) not in sys.path:
    sys.path.insert(0, str(PACKAGE))

import laws  # noqa: E402
import plan  # noqa: E402
from tfpd_exploration.submissions.evalai_m2_aof_scalar_static_v1 import driver  # noqa: E402
from tfpd_exploration.submissions.evalai_m2_aof_scalar_static_v1 import docker_local  # noqa: E402


def _container_evidence() -> dict[str, object]:
    return {
        "image_id": "sha256:" + "a" * 64,
        "image_tag": "spint-m2:aof-scalar-static-local-v1",
        "base_image_id": plan.LOCAL_DOCKER_BASE_ID,
        "network_disabled": True,
        "pull": False,
        "container_removed": True,
        "command": ["--evaluation", "local", "--split", "m2", "--phase", "minival", "--batch-size", "7"],
        "stdout_sha256": "b" * 64,
        "prediction_path": "/output/prediction.pkl",
        "prediction_sha256": "c" * 64,
        "target_path": "/output/ground_truth.pkl",
        "target_sha256": "d" * 64,
    }


def test_current_design_workorder_and_result_bytes_are_frozen_authority():
    plan.validate_static_authority(REPO_ROOT)


def test_reviewed_closure_is_an_external_exact_map_not_a_self_calculated_default():
    measured = driver.closure_map(REPO_ROOT)
    digest = driver.closure_sha256(measured)
    assert driver._validated_reviewed_closure(
        repo_root=REPO_ROOT,
        reviewed_closure=measured,
        reviewed_closure_sha256=digest,
    ) == measured
    forged = dict(measured)
    first = next(iter(forged))
    forged[first] = "0" * 64
    with pytest.raises(driver.LifecycleError, match="externally reviewed closure map mismatch"):
        driver._validated_reviewed_closure(
            repo_root=REPO_ROOT,
            reviewed_closure=forged,
            reviewed_closure_sha256=driver.closure_sha256(forged),
        )
    with pytest.raises(TypeError):
        driver.execute_production_local()  # type: ignore[call-arg]


def test_exact_581361_scored_lineage_is_bound_but_not_equated_to_local_payload():
    lineage = laws.validate_581361_lineage(REPO_ROOT)
    assert lineage["held_out_r2"] == 0.2897439880338965
    assert lineage["payload_sha256"] != plan.LOCAL_ACT30_PAYLOAD_SHA256


def test_actual_aofm_authority_reconstructs_the_only_legal_scalar_without_torch():
    witness = laws.validate_aofm_authority(REPO_ROOT)
    scalar = witness["scalar"]
    assert scalar["frozen_beta"] == laws.FROZEN_BETA
    assert scalar["numerator"] == laws.SCALAR_NUMERATOR
    assert scalar["denominator"] == laws.SCALAR_DENOMINATOR
    assert scalar["positive_sessions"] == 6
    assert scalar["worst_scalar_minus_native"] == laws.SCALAR_WORST_DELTA
    assert tuple(item["session"] for item in scalar["per_session"]) == laws.SOURCE_ROSTER


def test_clean_no_site_authority_codec_does_not_import_torch():
    site_packages = "/home/xinyuan/miniconda3/envs/spint/lib/python3.10/site-packages"
    environment = {
        **os.environ,
        "PYTHONNOUSERSITE": "1",
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONPATH": os.pathsep.join((str(PACKAGE), str(REPO_ROOT), site_packages)),
    }
    code = "import sys; from laws import validate_aofm_authority; validate_aofm_authority(); assert 'torch' not in sys.modules; print('clean-ok')"
    completed = subprocess.run([sys.executable, "-S", "-c", code], cwd=REPO_ROOT, env=environment, text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "clean-ok"


def test_aofm_held_graph_rejects_extra_or_sidecar_drift(tmp_path):
    source = REPO_ROOT / laws.AOFM_ROOT_RELATIVE
    copied = tmp_path / "source_oof"
    shutil.copytree(source, copied)
    for leaf in copied.iterdir():
        leaf.chmod(0o644)
        leaf.chmod(0o444)
    (copied / "unexpected.json").write_text("{}\n", encoding="utf-8")
    (copied / "unexpected.json").chmod(0o444)
    with pytest.raises(laws.AuthorityError, match="topology drift"):
        laws._read_aofm_graph(copied)
    (copied / "unexpected.json").unlink()
    sidecar = copied / "oof.json.sha256"
    sidecar.chmod(0o644)
    sidecar.write_text("0" * 64 + "  oof.json\n", encoding="ascii")
    sidecar.chmod(0o444)
    with pytest.raises(laws.AuthorityError, match="sidecar drift"):
        laws._read_aofm_graph(copied)


def test_literal_scalar_fusion_uses_positive_zero_direct_object_and_rejects_negative_zero():
    native = np.arange(6, dtype=np.float32).reshape(1, 2, 3)
    post = native + np.float32(0.25)
    assert laws.fuse_decoded_outputs(native, post, 0.0) is native
    with pytest.raises(laws.AuthorityError, match="negative zero"):
        laws.fuse_decoded_outputs(native, post, -0.0)
    fused = laws.fuse_decoded_outputs(native, post)
    expected = native + laws.FROZEN_BETA * (post - native)
    assert fused.dtype == np.float32 and fused.flags.c_contiguous
    assert np.array_equal(fused, expected)


def test_identity_map_requires_exact_13_float32_finite_96x50():
    identities = {f"tag-{index:02d}": np.zeros((96, 50), dtype=np.float32) for index in range(13)}
    digests = laws.validate_identity_map(identities)
    assert len(digests) == 13 and all(len(value) == 64 for value in digests.values())
    identities["tag-00"] = np.zeros((96, 49), dtype=np.float32)
    with pytest.raises(laws.AuthorityError, match="shape/dtype/finiteness"):
        laws.validate_identity_map(identities)


def test_local_lifecycle_success_is_prefix_linked_terminal_only(tmp_path):
    target = tmp_path / "local_build"
    capability = driver._issue_synthetic_capability_for_test(repo_root=REPO_ROOT, result_root=target)

    result = driver._execute_with_stages(
        capability,
        repo_root=REPO_ROOT,
        build_payload=lambda path: {
            "payload_sha256": "a" * 64,
            "session_records": {"tag": {"native_identity_sha256": "b" * 64}},
        },
        validate_payload=lambda path: {"payload": str(path), "zero_control": True, "network": False},
        docker_preflight=lambda: {
            "base_image_id": plan.LOCAL_DOCKER_BASE_ID,
            "network": "none",
            "pull": "false",
            "container_minival": _container_evidence(),
        },
    )
    assert result["status"] == "LOCAL_BUILD_VALIDATED_NOT_SUBMITTED"
    driver._validate_prefix(
        target,
        ("attempt.json", "predecessor_authority.json", "input_authority.json", "build.json", "validation.json", "terminal.json"),
    )
    terminal = (target / "terminal.json").read_text(encoding="utf-8")
    assert "LOCAL_BUILD_VALIDATED_NOT_SUBMITTED" in terminal and not (target / "failure.json").exists()
    with pytest.raises(driver.LifecycleError, match="already consumed"):
        driver._execute_with_stages(capability, repo_root=REPO_ROOT, build_payload=lambda p: {}, validate_payload=lambda p: {}, docker_preflight=lambda: {})


def test_local_lifecycle_failure_keeps_immutable_prefix_and_xor(tmp_path):
    target = tmp_path / "local_build_failure"
    capability = driver._issue_synthetic_capability_for_test(repo_root=REPO_ROOT, result_root=target)
    with pytest.raises(driver.LifecycleError, match="local build failed"):
        driver._execute_with_stages(
            capability,
            repo_root=REPO_ROOT,
            build_payload=lambda path: (_ for _ in ()).throw(ValueError("synthetic build failure")),
            validate_payload=lambda path: {},
            docker_preflight=lambda: {},
        )
    driver._validate_prefix(target, ("attempt.json", "predecessor_authority.json", "failure.json"))
    assert not (target / "terminal.json").exists()


def test_lifecycle_refuses_to_terminalize_without_offline_container_minival(tmp_path):
    target = tmp_path / "missing_container"
    capability = driver._issue_synthetic_capability_for_test(repo_root=REPO_ROOT, result_root=target)
    with pytest.raises(driver.LifecycleError, match="local build failed"):
        driver._execute_with_stages(
            capability,
            repo_root=REPO_ROOT,
            build_payload=lambda path: {"payload_sha256": "a" * 64, "session_records": {}},
            validate_payload=lambda path: {},
            docker_preflight=lambda: {"base_image_id": plan.LOCAL_DOCKER_BASE_ID, "network": "none", "pull": "false"},
        )
    driver._validate_prefix(
        target,
        ("attempt.json", "predecessor_authority.json", "input_authority.json", "failure.json"),
    )


def test_offline_container_minival_is_network_disabled_and_rehashes_written_outputs(tmp_path):
    root = tmp_path / "artifacts"
    root.mkdir()

    class _Containers:
        @staticmethod
        def run(image, **kwargs):
            assert image == "sha256:" + "e" * 64
            assert kwargs["network_disabled"] is True and kwargs["detach"] is False and kwargs["remove"] is True
            assert kwargs["environment"]["CUDA_VISIBLE_DEVICES"] == ""
            output = Path(next(host for host, mount in kwargs["volumes"].items() if mount["bind"] == "/output"))
            (output / "prediction.pkl").write_bytes(b"prediction")
            (output / "ground_truth.pkl").write_bytes(b"target")
            return b"local-minival-ok"

    client = type("Client", (), {"containers": _Containers()})()
    evidence = docker_local.run_container_minival(
        client,
        built_image={
            "base_image_id": plan.LOCAL_DOCKER_BASE_ID,
            "image_id": "sha256:" + "e" * 64,
            "image_tag": "local-aofs",
            "pull": "false",
            "network": "none",
        },
        repo_root=REPO_ROOT,
        artifact_root=root,
    )
    driver._validate_container_minival(evidence)
    assert evidence["prediction_sha256"] == docker_local._sha256_file(root / "container_minival" / "prediction.pkl")


def test_docker_admission_is_exact_local_id_and_never_relaxes_to_tag_only():
    class _Images:
        @staticmethod
        def get(tag):
            assert tag == docker_local.BASE_IMAGE_TAG
            return type("Image", (), {"attrs": {"Id": docker_local.BASE_IMAGE_ID}})()
    client = type("Client", (), {"images": _Images()})()
    evidence = docker_local.verify_present_base_image(client)
    assert evidence["base_image_id"] == docker_local.BASE_IMAGE_ID
    with pytest.raises(docker_local.DockerAdmissionError, match="frozen workorder literal"):
        docker_local.verify_present_base_image(client, "sha256:" + "0" * 64)


def test_runtime_validation_zero_is_one_direct_native_decode_and_negative_zero_is_rejected_before_decode():
    import torch
    from tfpd_exploration.submissions.evalai_m2_aof_scalar_static_v1.aofs_static_decoder import AofsStaticM2Decoder

    decoder = object.__new__(AofsStaticM2Decoder)
    decoder.batch_size = 1
    decoder.task_config = type("Task", (), {"n_channels": 2, "bin_size_ms": 20})()
    decoder.raw_history_buffer = np.zeros((5, 1, 2), dtype=np.float32)
    decoder.observation_buffer = np.zeros((50, 1, 2), dtype=np.float32)
    decoder.smooth_observations = False
    decoder.device = torch.device("cpu")
    decoder.local_native = [torch.zeros((1, 96, 50), dtype=torch.float32)]
    decoder.local_post = [torch.ones((1, 96, 50), dtype=torch.float32)]
    calls: list[str] = []

    def fake_decode(neural, identity):
        calls.append("post" if bool(identity.flatten()[0]) else "native")
        return torch.full((1, 50, 2), 5.0 if calls[-1] == "native" else 10.0)

    decoder._decode_with_identity = fake_decode
    zero = decoder.predict_with_beta_for_validation(np.zeros((1, 2), dtype=np.float32), 0.0)
    assert calls == ["native"] and np.array_equal(zero, np.ones((1, 2), dtype=np.float32))
    calls.clear()
    with pytest.raises(ValueError, match="negative zero"):
        decoder.predict_with_beta_for_validation(np.zeros((1, 2), dtype=np.float32), -0.0)
    assert calls == []
    learned = decoder.predict_with_beta_for_validation(np.zeros((1, 2), dtype=np.float32), laws.FROZEN_BETA)
    assert calls == ["native", "post"]
    expected = np.float32(1.0) + laws.FROZEN_BETA * (np.float32(2.0) - np.float32(1.0))
    assert np.array_equal(learned, np.full((1, 2), expected, dtype=np.float32))


def test_actual_581361_c51_payload_is_the_13_tag_governing_native_map_cpu_only():
    import torch
    from tfpd_exploration.src.pit_m2_v1.trainer import ensure_streaming_paths
    ensure_streaming_paths(REPO_ROOT)
    from sua_exploration.evalai_t4_m2.t4_spint_decoder import CPUUnpickler

    root = REPO_ROOT / "sua_exploration/evalai_t4_m2_activity_budget/artifacts"
    payload_path = root / "t4_m2_seed42_ridge_m4_activity30_identity.pkl"
    receipt_path = root / "t4_m2_seed42_ridge_m4_activity30_identity.receipt.json"
    assert laws.sha256_bytes(payload_path.read_bytes()) == laws.OFFICIAL_581361_PAYLOAD_SHA256
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    with payload_path.open("rb") as handle:
        payload = CPUUnpickler(handle).load()
    identities = payload["identity_by_dataset_tag"]
    assert len(identities) == 13 and not torch.cuda.is_initialized()
    for session, record in receipt["session_records"].items():
        assert laws.sha256_array(np.asarray(identities[record["dataset_tag"]], dtype=np.float32)) == record["identity_sha256"], session
