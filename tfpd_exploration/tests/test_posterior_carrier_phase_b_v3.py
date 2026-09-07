"""No-data/no-CUDA tests for the additive Phase-B smoke-v3 TF32 repair."""
from __future__ import annotations

import copy
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from typing import Any, Mapping

import pytest

from src.posterior_carrier_v1 import phase_b, phase_b_v2, phase_b_v3


ROOT = Path(__file__).resolve().parents[2]
SHA_A = "a" * 64
SHA_B = "b" * 64


def _source_root_payload() -> dict[str, object]:
    return {
        "schema": "posterior_carrier_strict27_source_data_root_v1",
        "source_data_root": phase_b.CANONICAL_SOURCE_DATA_ROOT,
        "directory_device": 1,
        "directory_inode": 2,
        "nwb_copied_into_stage": False,
        "nwb_symlink_or_bind_mount_authorized": False,
    }


def _identity_v3() -> phase_b_v3.RunIdentityV3:
    """A valid no-data identity built from only checked-in code closure bytes."""
    roster = [f"source-{index:02d}" for index in range(27)]
    source = {
        "schema": "posterior_carrier_target_free_source_identity_v1",
        "roster": roster,
        "roster_sha256": phase_b.sha256_bytes(phase_b.canonical_json_bytes(roster)),
        "strict_source_metadata_sha256": SHA_A,
        "manifest_sha256": SHA_A,
        "ordinary_raw_t4_semantic_sha256": SHA_A,
        "behavior_normalizer_semantic_sha256": SHA_A,
        "source_lineage_sha256": SHA_A,
        "source_data_root": _source_root_payload(),
        "source_only": True,
        "target_opened": False,
        "within_opened": False,
        "external_opened": False,
        "formal_opened": False,
        "h1_opened": False,
    }
    v2_closure = phase_b_v2.phase_b_v2_closure(ROOT)
    v1 = phase_b.RunIdentity(
        source_authority=source,
        closure=phase_b_v2.base_v1_closure_from_v2(v2_closure),
        remote_device=dict(phase_b.REMOTE_TORCH_AUTHORITY),
    )
    v2 = phase_b_v2.RunIdentityV2(base_identity=v1, closure=v2_closure)
    return phase_b_v3.RunIdentityV3(base_identity=v2, closure=phase_b_v3.phase_b_v3_closure(ROOT))


def _write_pair(directory: Path, name: str, payload: Mapping[str, object]) -> str:
    body = phase_b.canonical_json_bytes(payload)
    digest = phase_b.sha256_bytes(body)
    target = directory / name
    target.write_bytes(body)
    target.chmod(0o444)
    sidecar = directory / f"{name}.sha256"
    sidecar.write_bytes(f"{digest}  {name}\n".encode("ascii"))
    sidecar.chmod(0o444)
    return digest


def _valid_v1_failure_pair() -> dict[str, tuple[dict[str, object], str]]:
    identity = _identity_v3().base_identity.base_identity
    attempt = phase_b._attempt_payload(identity)
    attempt_sha = phase_b.sha256_bytes(phase_b.canonical_json_bytes(attempt))
    launch = phase_b._launch_payload(identity, attempt_sha)
    launch_sha = phase_b.sha256_bytes(phase_b.canonical_json_bytes(launch))
    failure = phase_b._failure_payload(
        identity=identity, attempt_sha256=attempt_sha, launch_sha256=launch_sha, stage="prepare",
        error=RuntimeError("synthetic v1 source failure"),
        progress=phase_b.SmokeExecutionProgress(source_opened=True, remote_initialized=False),
    )
    failure_sha = phase_b.sha256_bytes(phase_b.canonical_json_bytes(failure))
    return {
        "attempt.json": (attempt, attempt_sha), "launch.json": (launch, launch_sha),
        "failure.json": (failure, failure_sha),
    }


def _valid_v2_failure_pair() -> dict[str, tuple[dict[str, object], str]]:
    identity = _identity_v3().base_identity
    attempt = phase_b_v2._attempt_payload(identity)
    attempt_sha = phase_b.sha256_bytes(phase_b.canonical_json_bytes(attempt))
    launch = phase_b_v2._launch_payload(identity, attempt_sha)
    launch_sha = phase_b.sha256_bytes(phase_b.canonical_json_bytes(launch))
    failure = phase_b_v2._failure_payload(
        identity=identity, attempt_sha256=attempt_sha, launch_sha256=launch_sha, stage="prepare",
        error=phase_b_v2.PhaseBV2Error("Phase-B-v2 forbids AMP/TF32"),
        progress=phase_b.SmokeExecutionProgress(source_opened=True, remote_initialized=True),
    )
    failure_sha = phase_b.sha256_bytes(phase_b.canonical_json_bytes(failure))
    return {
        "attempt.json": (attempt, attempt_sha), "launch.json": (launch, launch_sha),
        "failure.json": (failure, failure_sha),
    }


class _Flag:
    def __init__(self, value: bool) -> None:
        self.allow_tf32 = value


class _FakeTorch:
    def __init__(self, *, matmul: bool, cudnn: bool, autocast: bool = False) -> None:
        self.backends = SimpleNamespace(cuda=SimpleNamespace(matmul=_Flag(matmul)), cudnn=_Flag(cudnn))
        self._autocast = autocast

    def is_autocast_enabled(self) -> bool:
        return self._autocast


def test_v3_identity_closure_binds_exact_accepted_v2_and_fixed_predecessors() -> None:
    identity = _identity_v3()
    phase_b_v3.validate_run_identity_v3(identity)
    assert identity.base_identity.closure["closure_sha256"] == phase_b_v3.ACCEPTED_PHASE_B_V2_CLOSURE_SHA256
    assert phase_b_v3.V2_ATTEMPT_BODY_SHA256 == "e93e1a59724d06abcc8831c3b0d4f49f81f7eecb451c6e429e96f1f70d6cea24"
    assert phase_b_v3.V2_LAUNCH_BODY_SHA256 == "fadd0c94a35a416f0c4a6d527be1ee9a85368d845d64ab64a20d2028c8ba3416"
    assert phase_b_v3.V2_FAILURE_BODY_SHA256 == "d8c5a4eac84d14187163991209b486433a9851496f7a336b9e5a3dcd7243a2af"
    assert phase_b_v3.V2_FAILURE_EXPECTED["error_class"] == "PhaseBV2Error"
    closure = copy.deepcopy(identity.closure)
    closure["sha256_by_path"][phase_b_v2.PHASE_B_V2_CLOSURE_PATHS[-1]] = SHA_B
    body = {"paths": closure["paths"], "sha256_by_path": closure["sha256_by_path"]}
    closure["closure_sha256"] = phase_b.sha256_bytes(phase_b.canonical_json_bytes(body))
    with pytest.raises(phase_b_v3.PhaseBV3Error, match="accepted immutable V2"):
        phase_b_v3.validate_phase_b_v3_closure(closure)


def test_tf32_enforcement_accepts_true_defaults_only_after_explicit_false_post_state_and_restores() -> None:
    fake = _FakeTorch(matmul=True, cudnn=True)
    policy = phase_b_v3.TF32Enforcement.enforce(fake)
    payload = policy.payload()
    assert payload["observed_pre_state"] == {
        "cuda_matmul_allow_tf32": True, "cudnn_allow_tf32": True, "amp_enabled": False,
    }
    assert payload["enforced_post_state"] == {
        "cuda_matmul_allow_tf32": False, "cudnn_allow_tf32": False, "amp_enabled": False,
    }
    assert phase_b_v3.validate_tf32_enforcement(payload) == payload
    forged = copy.deepcopy(payload)
    forged["enforced_post_state"]["cudnn_allow_tf32"] = True
    with pytest.raises(phase_b_v3.PhaseBV3Error, match="post-state"):
        phase_b_v3.validate_tf32_enforcement(forged)
    policy.restore(fake)
    assert fake.backends.cuda.matmul.allow_tf32 is True
    assert fake.backends.cudnn.allow_tf32 is True
    with pytest.raises(phase_b_v3.PhaseBV3Error, match="AMP"):
        phase_b_v3.TF32Enforcement.enforce(_FakeTorch(matmul=True, cudnn=True, autocast=True))


def test_physical_v3_sets_tf32_before_inherited_model_path_and_restores_route_local_state(monkeypatch: pytest.MonkeyPatch) -> None:
    identity = _identity_v3()
    fake_torch = _FakeTorch(matmul=True, cudnn=True)
    seen: list[str] = []

    def inherited_prepare(self: object, spec: object, base_identity: object) -> Any:
        assert fake_torch.backends.cuda.matmul.allow_tf32 is False
        assert fake_torch.backends.cudnn.allow_tf32 is False
        seen.append("inherited_model_optimizer_path")
        return SimpleNamespace(progress=phase_b.SmokeExecutionProgress(source_opened=True, remote_initialized=True))

    monkeypatch.setattr(phase_b_v3, "torch", fake_torch)
    monkeypatch.setattr(phase_b_v2.RemotePosteriorSourceSmokeBackendV2, "prepare", inherited_prepare)
    backend = phase_b_v3.RemotePosteriorSourceSmokeBackendV3(
        Path("/synthetic/stage"), source_data=SimpleNamespace(), num_workers=4,
    )
    runtime = backend.prepare(phase_b.SMOKE_SPEC, identity)
    assert seen == ["inherited_model_optimizer_path"]
    assert phase_b_v3.validate_tf32_enforcement(runtime.tf32_enforcement.payload())["enforced_before_model_construction"] is True
    backend.close(runtime)
    assert fake_torch.backends.cuda.matmul.allow_tf32 is True
    assert fake_torch.backends.cudnn.allow_tf32 is True


def test_held_no_follow_pair_reader_reads_all_pairs_and_rejects_symlink_root() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        directory = root / "immutable"
        directory.mkdir()
        payloads = {"attempt.json": {"v": 1}, "launch.json": {"v": 2}, "failure.json": {"v": 3}}
        assets = tuple((name, _write_pair(directory, name, payload)) for name, payload in payloads.items())
        loaded = phase_b_v3._read_immutable_pairs_under_held_root(directory, assets)
        assert [loaded[name][0] for name, _digest in assets] == [payloads[name] for name, _digest in assets]
        alias = root / "alias"
        alias.symlink_to(directory, target_is_directory=True)
        with pytest.raises(phase_b_v3.PhaseBV3Error, match="non-symlink"):
            phase_b_v3._read_immutable_pairs_under_held_root(alias, assets)
        # The same held-FD reader binds bytes, not just a pathname/sidecar.
        target = directory / "attempt.json"
        target.chmod(0o644)
        target.write_bytes(b'{"v":99}')
        target.chmod(0o444)
        with pytest.raises(phase_b_v3.PhaseBV3Error, match="body SHA"):
            phase_b_v3._read_immutable_pairs_under_held_root(directory, assets)


def test_predecessor_semantic_validators_accept_the_expected_failure_shapes_without_real_results() -> None:
    v1 = _valid_v1_failure_pair()
    v2 = _valid_v2_failure_pair()
    v1_result = phase_b_v3._validate_v1_predecessor_pairs(v1)
    v2_result = phase_b_v3._validate_v2_predecessor_pairs(v2)
    assert v1_result["failure_sha256"] == v1["failure.json"][1]
    assert v2_result["failure_sha256"] == v2["failure.json"][1]
    forged = dict(v2)
    bad_failure = copy.deepcopy(v2["failure.json"][0])
    bad_failure["remote_initialized"] = False
    forged["failure.json"] = (bad_failure, v2["failure.json"][1])
    with pytest.raises(phase_b.PhaseBError, match="failure binding"):
        phase_b_v3._validate_v2_predecessor_pairs(forged)
    forged = dict(v2)
    bad_failure = copy.deepcopy(v2["failure.json"][0])
    bad_failure["identity"]["v1_failed_predecessor"]["failure_body_sha256"] = SHA_B
    forged["failure.json"] = (bad_failure, v2["failure.json"][1])
    with pytest.raises(phase_b_v3.PhaseBV3Error, match="exact-payload"):
        phase_b_v3._validate_v2_predecessor_pairs(forged)
    assert phase_b_v3.V1_PREDECESSOR_ASSETS == (
        ("attempt.json", "4d6283534017938de44450a9cd1160496e1efaa1f91728a2c0d82fb2276454c3"),
        ("launch.json", "bce0ac19b3517dfeb4298faedcd322efd6f10248768ce93fef782388f21c2229"),
        ("failure.json", "a7dfaf466642ca92a945dc545ad0ec0228a0eac180dc3e24ec4d93471977738f"),
    )


class _FailAfterEnforcementBackend:
    def __init__(self) -> None:
        self.closed = False
        self.last_tf32_enforcement = phase_b_v3.TF32Enforcement(
            pre_matmul_allow_tf32=True, pre_cudnn_allow_tf32=True, pre_amp_enabled=False,
            post_matmul_allow_tf32=False, post_cudnn_allow_tf32=False, post_amp_enabled=False,
        )

    def prepare(self, spec: phase_b.SourceSmokeSpec, identity: phase_b_v3.RunIdentityV3) -> Any:
        raise phase_b.SourceSmokeExecutionError(
            stage="prepare", progress=phase_b.SmokeExecutionProgress(source_opened=True, remote_initialized=True),
            cause=RuntimeError("synthetic post-enforcement failure"),
        )

    def source_authority(self, runtime: Any, identity: phase_b_v3.RunIdentityV3) -> Mapping[str, object]:
        raise AssertionError("prepare failure must not enter source authority")

    def run_steps(self, runtime: Any, spec: phase_b.SourceSmokeSpec) -> phase_b.SmokeStepSummary:
        raise AssertionError("prepare failure must not enter steps")

    def close(self, runtime: Any | None) -> None:
        self.closed = True


def test_v3_failure_lifecycle_records_enforced_post_state_and_fresh_root() -> None:
    identity = _identity_v3()
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        (root / "tfpd_exploration/results").mkdir(parents=True)
        artifact = phase_b_v3.reserve_source_smoke_root_v3(root)
        backend = _FailAfterEnforcementBackend()
        with pytest.raises(phase_b.SourceSmokeExecutionError):
            phase_b_v3.run_source_smoke_lifecycle_v3(backend=backend, artifact=artifact, identity=identity)
        assert backend.closed is True
        assert (artifact.directory / "attempt.json").exists()
        assert (artifact.directory / "launch.json").exists()
        assert (artifact.directory / "failure.json").exists()
        assert not (artifact.directory / "terminal.json").exists()
        failure = artifact.reload_json("failure.json")
        assert failure["tf32_enforcement"]["observed_pre_state"]["cuda_matmul_allow_tf32"] is True
        assert failure["tf32_enforcement"]["enforced_post_state"] == {
            "cuda_matmul_allow_tf32": False, "cudnn_allow_tf32": False, "amp_enabled": False,
        }
        with pytest.raises(phase_b.PhaseBError, match="fresh"):
            phase_b_v3.reserve_source_smoke_root_v3(root)


def test_v3_stage_plan_is_explicit_and_carries_both_failed_predecessor_pairs() -> None:
    identity = _identity_v3()

    class _SourceCapability:
        def payload(self) -> Mapping[str, object]:
            return _source_root_payload()

    plan = phase_b_v3.remote_staging_plan_v3(
        closure=identity.closure, source_authority_sha256=SHA_A, source_data=_SourceCapability(),
    )
    paths = [item["relative_path"] for item in plan["stage_files"]]
    assert "tfpd_exploration/src/posterior_carrier_v1/phase_b_v3.py" in paths
    assert "tfpd_exploration/scripts/run_posterior_carrier_phase_b_source_smoke_v3.py" in paths
    assert not any("*" in path or "glob" in path for path in paths)
    for relative_root, assets in ((phase_b_v3.V1_FAILURE_ROOT_RELATIVE, phase_b_v3.V1_PREDECESSOR_ASSETS),
                                  (phase_b_v3.V2_FAILURE_ROOT_RELATIVE, phase_b_v3.V2_PREDECESSOR_ASSETS)):
        for name, digest in assets:
            assert f"{relative_root}/{name}" in paths
            assert f"{relative_root}/{name}.sha256" in paths
            assert any(item.get("sha256") == digest for item in plan["stage_files"])


def test_static_cli_stays_dry_and_public_execution_flags_fail_before_route_import() -> None:
    script = ROOT / "tfpd_exploration/scripts/run_posterior_carrier_phase_b_source_smoke_v3.py"
    assert "import torch" not in script.read_text(encoding="utf-8")
    command = [sys.executable, str(script), "--plan"]
    completed = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    assert "DRY_FAIL_CLOSED_PENDING_ROOT_REVIEW" in completed.stdout
    rejected = subprocess.run(
        [sys.executable, str(script), "--execute-remote-smoke-v3"], cwd=ROOT, capture_output=True, text=True,
    )
    assert rejected.returncode != 0
    assert "in-process root-reviewed capability" in rejected.stderr
