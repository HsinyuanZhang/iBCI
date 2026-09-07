"""No-data/no-CUDA gates for additive CDM-D Source Execution V2."""
from __future__ import annotations

import hashlib
import inspect
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "tfpd_exploration") not in sys.path:
    sys.path.insert(0, str(ROOT / "tfpd_exploration"))

from src.causal_dual_memory_cell_d_v1 import source_execute as v1  # noqa: E402
from src.causal_dual_memory_cell_d_v1 import source_execute_physical as v1_physical  # noqa: E402
from src.causal_dual_memory_cell_d_v1 import source_execute_v2 as v2  # noqa: E402
from src.causal_dual_memory_cell_d_v1 import source_execute_physical_v2 as physical_v2  # noqa: E402


def _sha(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def _write(path: Path, body: bytes, *, mode: int = 0o444) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    os.chmod(path, mode)


def _roster() -> tuple[str, ...]:
    return (
        v1.SOURCE_SMOKE_SESSION,
        *(f"sub-C_ses-CO-{index:08d}" for index in range(v1.STRICT_SOURCE_COUNT - 1)),
    )


def _profile() -> dict[str, object]:
    return dict(v1.COMPATIBLE_DEVICE_PROFILES["gpu1"])


def _v1_identity_payload(*, root_relative: str, closure_sha: str) -> dict[str, object]:
    return {
        "schema": "causal_dual_memory_cell_d_source_execution_identity_v1",
        "cell": v1.CELL,
        "run_spec": v1.SourceExecutionSpec(
            "source_smoke", root_relative, v1.SOURCE_SMOKE_SESSION, 30, (30, 31),
        ).payload(),
        "closure": {"schema": "synthetic_v1", "closure_sha256": closure_sha},
        "strict_train_roster": list(_roster()),
        "strict_train_roster_sha256": v1.roster_sha256(_roster()),
        "fixed_assets": {asset.label: asset.payload() for asset in v1.FIXED_ASSETS},
        "normalizers": v1.SEALED_NORMALIZERS.payload(),
        "selected_device": _profile(),
        "source_only": True,
        "within_external_formal_target_forbidden": True,
        "target_optimizer_backward_update": 0,
    }


def _make_v1_failed_predecessor(
    root: Path, *, closure_sha: str = v2.V1_IMPLEMENTATION_CLOSURE_SHA256,
) -> v2.V1FailedPredecessorExpectation:
    """Write a synthetic exact-six-leaf V1 failure under a temp held root."""
    relative = "results/synthetic_v1_failed_source_smoke"
    identity = _v1_identity_payload(root_relative=relative, closure_sha=closure_sha)
    attempt = {
        "schema": "causal_dual_memory_cell_d_source_execution_attempt_v1",
        "cell": v1.CELL,
        "status": "ATTEMPT_RESERVED",
        "identity": identity,
        "source_only": True,
        "source_resolved_or_opened": False,
        "checkpoint_opened": False,
        "cuda_initialized": False,
        "target_optimizer_backward_update": 0,
        "within_external_formal_target_forbidden": True,
    }
    attempt_body = json.dumps(attempt, sort_keys=True, separators=(",", ":")).encode()
    attempt_sha = _sha(attempt_body)
    launch = {
        "schema": "causal_dual_memory_cell_d_source_execution_launch_v1",
        "cell": v1.CELL,
        "status": "LAUNCHED",
        "identity": identity,
        "attempt_sha256": attempt_sha,
        "preflight": {
            "source_resolved_or_opened": False,
            "checkpoint_opened": False,
            "cuda_initialized": False,
        },
        "source_only": True,
        "target_optimizer_backward_update": 0,
    }
    launch_body = json.dumps(launch, sort_keys=True, separators=(",", ":")).encode()
    launch_sha = _sha(launch_body)
    flags = {
        "stage": "prepare",
        "source_resolved": True,
        "source_opened": False,
        "checkpoint_opened": False,
        "cuda_initialized": False,
        "model_forward_calls": 0,
        "backward_calls": 0,
        "optimizer_steps": 0,
        "parameter_updates": 0,
        "normalizer_refit": False,
        "within_opened": False,
        "external_opened": False,
        "formal_opened": False,
        "target_opened": False,
        "oom_retry_attempted": False,
    }
    failure = {
        "schema": "causal_dual_memory_cell_d_source_execution_failure_v1",
        "cell": v1.CELL,
        "status": "FAILED",
        "identity": identity,
        "attempt_sha256": attempt_sha,
        "launch_sha256": launch_sha,
        "source_authority_sha256": None,
        "stage": "prepare",
        "error_class": "SourceExecutionPhysicalError",
        "error_sha256": "e" * 64,
        "flags": flags,
        "terminal_published": False,
        "source_only": True,
    }
    failure_body = json.dumps(failure, sort_keys=True, separators=(",", ":")).encode()
    failure_sha = _sha(failure_body)
    directory = root / relative
    for name, body, digest in (
        ("attempt.json", attempt_body, attempt_sha),
        ("launch.json", launch_body, launch_sha),
        ("failure.json", failure_body, failure_sha),
    ):
        _write(directory / name, body)
        _write(directory / f"{name}.sha256", f"{digest}  {name}\n".encode())
    return v2.V1FailedPredecessorExpectation(relative, attempt_sha, launch_sha, failure_sha, closure_sha)


def _raw_fixture(*, session: str = v1.SOURCE_SMOKE_SESSION, units: int = 7) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    raw = np.asarray(
        [[0.5 + index, -0.25 - index, 0.1 + index * 0.01, 2.0 + index] for index in range(units)],
        dtype=np.float32,
    )
    theta = np.arctan2(raw[:, 1].astype(np.float64), raw[:, 0].astype(np.float64)).astype(np.float64)
    valid = (raw[:, 2].astype(np.float64) > 1.0e-6).astype(np.bool_)
    if v1.KNOWN_THETA_INVALID_UNIT_COUNTS.get(session, 0):
        raw[0, 2] = 0.0
        valid = (raw[:, 2].astype(np.float64) > 1.0e-6).astype(np.bool_)
    return raw, theta, valid, np.arange(units, dtype=np.int64)


def _theta_proof(session: str = v1.SOURCE_SMOKE_SESSION) -> dict[str, object]:
    raw, theta, valid, channels = _raw_fixture(session=session)
    _mask, proof = v2.validate_theta_raw_t4_authority(
        raw,
        sealed_raw_t4_sha256=v2.theta_raw_t4_float32_bytes_sha256(raw),
        sealed_theta_float64=theta,
        sealed_valid_mask=valid,
        n_units=raw.shape[0],
        channel_ids=channels,
        session=session,
        modulation_eps=1.0e-6,
    )
    return proof


def test_v2_static_identity_and_explicit_inherited_closure() -> None:
    assert v2.WORKORDER_SHA256 == "b71e8a7796db228f04c73efee841793bf0998413bbb436ecdc36d7c30d6e75ce"
    assert v2.V1_IMPLEMENTATION_CLOSURE_SHA256 == "77f9495780605d05be177500ff0ef61b3bd35a427ddd12af4fbab97b3486428b"
    assert v2.THETA_RAW_HASH_LAW == "theta_float32_contiguous_raw_bytes_v1"
    closure = v2.execution_closure_payload(ROOT)
    paths = [row["path"] for row in closure["paths"]]
    assert closure["v1_implementation_closure_sha256"] == v2.V1_IMPLEMENTATION_CLOSURE_SHA256
    assert paths[-5:] == [
        v2.WORKORDER_RELATIVE,
        "tfpd_exploration/src/causal_dual_memory_cell_d_v1/source_execute_v2.py",
        "tfpd_exploration/src/causal_dual_memory_cell_d_v1/source_execute_physical_v2.py",
        "tfpd_exploration/scripts/run_causal_dual_memory_cell_d_source_gate_v2.py",
        "tfpd_exploration/tests/test_causal_dual_memory_cell_d_source_execution_v2.py",
    ]
    assert len(closure["closure_sha256"]) == 64


def test_float32_raw_byte_hash_is_the_only_authority_domain_and_f64_tensor_digest_is_not_a_fallback() -> None:
    import torch
    from src.posterior_carrier_v1 import core as posterior_core

    raw, theta, valid, channels = _raw_fixture()
    authority = v2.theta_raw_t4_float32_bytes_sha256(raw)
    assert v2.theta_raw_t4_float32_bytes_sha256(raw.astype(np.float64)) == authority
    assert posterior_core.tensor_digest(torch.as_tensor(raw, dtype=torch.float64)) != authority
    mask, proof = v2.validate_theta_raw_t4_authority(
        raw.astype(np.float64), sealed_raw_t4_sha256=authority,
        sealed_theta_float64=theta, sealed_valid_mask=valid, n_units=raw.shape[0],
        channel_ids=channels, session=v1.SOURCE_SMOKE_SESSION, modulation_eps=1.0e-6,
    )
    assert np.array_equal(mask, valid)
    assert proof["raw_t4_sha256"] == authority
    assert proof["hash_law"] == v2.THETA_RAW_HASH_LAW


def test_raw_t4_ulp_and_topology_authority_drift_fail_closed() -> None:
    raw, theta, valid, channels = _raw_fixture()
    sha = v2.theta_raw_t4_float32_bytes_sha256(raw)
    common = dict(
        sealed_raw_t4_sha256=sha, sealed_theta_float64=theta, sealed_valid_mask=valid,
        n_units=raw.shape[0], channel_ids=channels, session=v1.SOURCE_SMOKE_SESSION,
        modulation_eps=1.0e-6,
    )
    ulp = raw.copy(); ulp[0, 0] = np.nextafter(ulp[0, 0], np.float32(np.inf))
    with pytest.raises(v2.SourceExecutionV2Error, match="float32"):
        v2.validate_theta_raw_t4_authority(ulp, **common)
    theta_bad = theta.copy(); theta_bad[0] += 0.1
    with pytest.raises(v2.SourceExecutionV2Error, match="atan2"):
        v2.validate_theta_raw_t4_authority(raw, **{**common, "sealed_theta_float64": theta_bad})
    valid_bad = valid.copy(); valid_bad[0] = ~valid_bad[0]
    with pytest.raises(v2.SourceExecutionV2Error, match="MODULATION_EPS"):
        v2.validate_theta_raw_t4_authority(raw, **{**common, "sealed_valid_mask": valid_bad})
    invalid_session = "sub-C_ses-CO-20131101"
    # A self-consistent all-valid authority row still fails because this
    # sealed session has exactly one declared undefined-theta channel.
    invalid_raw, invalid_theta, invalid_valid, invalid_channels = _raw_fixture()
    assert int((~invalid_valid).sum()) == 0
    with pytest.raises(v2.SourceExecutionV2Error, match="undefined-unit"):
        v2.validate_theta_raw_t4_authority(
            invalid_raw, sealed_raw_t4_sha256=v2.theta_raw_t4_float32_bytes_sha256(invalid_raw),
            sealed_theta_float64=invalid_theta, sealed_valid_mask=invalid_valid,
            n_units=invalid_raw.shape[0], channel_ids=invalid_channels, session=invalid_session,
            modulation_eps=1.0e-6,
        )
    nonfinite = raw.copy(); nonfinite[0, 0] = np.nan
    with pytest.raises(v2.SourceExecutionV2Error, match="finite"):
        v2.validate_theta_raw_t4_authority(nonfinite, **common)
    with pytest.raises(v2.SourceExecutionV2Error, match=r"\[n_units,4\]"):
        v2.validate_theta_raw_t4_authority(raw[:, :3], **common)
    reversed_channels = channels[::-1].copy()
    with pytest.raises(v2.SourceExecutionV2Error, match="unit-order"):
        v2.validate_theta_raw_t4_authority(raw, **{**common, "channel_ids": reversed_channels})


def test_known_invalid_topology_is_exact_per_session() -> None:
    session = "sub-C_ses-CO-20131101"
    raw, theta, valid, channels = _raw_fixture(session=session)
    _mask, proof = v2.validate_theta_raw_t4_authority(
        raw, sealed_raw_t4_sha256=v2.theta_raw_t4_float32_bytes_sha256(raw),
        sealed_theta_float64=theta, sealed_valid_mask=valid, n_units=raw.shape[0],
        channel_ids=channels, session=session, modulation_eps=1.0e-6,
    )
    assert proof["known_invalid_unit_count"] == 1


@pytest.mark.parametrize("tamper", ("body", "sidecar", "mode", "extra", "semantic", "symlink"))
def test_v1_predecessor_exact_six_leaf_held_fd_validation_rejects_tampering(tmp_path: Path, tamper: str) -> None:
    expectation = _make_v1_failed_predecessor(tmp_path)
    assert v2.validate_v1_failed_predecessor(tmp_path, expectation=expectation).expectation == expectation
    directory = tmp_path / expectation.root_relative
    if tamper == "body":
        os.chmod(directory / "attempt.json", 0o644)
        (directory / "attempt.json").write_bytes(b"{}")
        os.chmod(directory / "attempt.json", 0o444)
    elif tamper == "sidecar":
        os.chmod(directory / "launch.json.sha256", 0o644)
        (directory / "launch.json.sha256").write_bytes(b"forged\n")
        os.chmod(directory / "launch.json.sha256", 0o444)
    elif tamper == "mode":
        os.chmod(directory / "failure.json", 0o664)
    elif tamper == "extra":
        _write(directory / "extra.json", b"{}")
    elif tamper == "semantic":
        body = json.loads((directory / "failure.json").read_bytes())
        body["flags"]["source_opened"] = True
        encoded = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
        os.chmod(directory / "failure.json", 0o644)
        (directory / "failure.json").write_bytes(encoded)
        os.chmod(directory / "failure.json", 0o444)
        os.chmod(directory / "failure.json.sha256", 0o644)
        (directory / "failure.json.sha256").write_bytes(
            f"{_sha(encoded)}  failure.json\n".encode(),
        )
        os.chmod(directory / "failure.json.sha256", 0o444)
    else:
        target = tmp_path / "target.json"
        _write(target, (directory / "attempt.json").read_bytes())
        (directory / "attempt.json").unlink()
        os.symlink(target, directory / "attempt.json")
    with pytest.raises(v2.SourceExecutionV2Error):
        v2.validate_v1_failed_predecessor(tmp_path, expectation=expectation)


def _inherited_identity(spec: v1.SourceExecutionSpec, v1_closure: dict[str, object]) -> v1.SourceExecutionIdentity:
    return v1.SourceExecutionIdentity(
        spec=spec,
        closure=v1_closure,
        strict_train_roster=_roster(),
        fixed_assets={asset.label: asset.payload() for asset in v1.FIXED_ASSETS},
        normalizers=v1.SEALED_NORMALIZERS,
        selected_device=_profile(),
    )


def _session_topology(session: str) -> dict[str, object]:
    invalid = v1.KNOWN_THETA_INVALID_UNIT_COUNTS.get(session, 0)
    return {
        "session_id": session,
        "total_unit_count": 8,
        "valid_unit_count": 8 - invalid,
        "invalid_unit_count": invalid,
        "valid_mask_sha256": "b" * 64,
        "theta_valid_mask_sha256": "b" * 64,
        "invalid_assignment_is_minus_one": True,
        "theta_authority_binds_only_validity_not_budget_groups": True,
    }


def _runtime_attestation() -> dict[str, object]:
    return {
        **_profile(), "visible_devices": 1, "attested": True,
        "torch_cuda_matmul_allow_tf32": False, "torch_cudnn_allow_tf32": False,
    }


def _sealed_swa_proof() -> dict[str, object]:
    return {
        "schema": v1.SEALED_SWA_LOAD_PROOF_SCHEMA,
        "sealed_terminal_sha256": v1.SEALED_CELL_D_TERMINAL_SHA256,
        "sealed_swa_sha256": v1.SEALED_CELL_D_SWA_SHA256,
        "fresh_strict_load": True,
        "recomputed_state_dict_sha256": "c" * 64,
        "initialized_trainable_parameters": v1.SEALED_CELL_D_INITIALIZED_TRAINABLE_PARAMETERS,
        "uninitialized_lazy_keys": list(v1.SEALED_CELL_D_UNINITIALIZED_LAZY_KEYS),
        "model_eval": True, "no_grad": True, "finite_forward": True,
        "repeated_fixed_forward_bitwise_equal": True, "model_state_unchanged": True,
        "dynamic_dropout_calls": 0,
    }


def _budget_fields(budget: int) -> dict[str, object]:
    return {
        "budget_initial_carrier_recipe": "fixed_ridge_by_trial",
        "budget_initial_carrier_support_rows": budget,
        "raw_m30_t4_used_as_initializer": False,
        "initial_support_rate_domain": v1.INITIAL_SUPPORT_RATE_DOMAIN,
        "online_update_rate_domain": v1.ONLINE_UPDATE_RATE_DOMAIN,
        "initial_support_rates_sha256": "7" * 64,
        "initial_support_exposure_seconds_sha256": "8" * 64,
        "budget_initial_carrier_parity": {"mode": "fixed_ridge_by_trial"},
        "budget_initial_carrier_sha256": "3" * 64,
        "budget_groups_sha256": "4" * 64,
        "budget_group_assignment_sha256": "5" * 64,
        "budget_group_valid_mask_sha256": "6" * 64,
    }


class _MockBackend:
    def __init__(self) -> None:
        self.events: list[str] = []
        self.runtime = object()

    def preflight(self, *, root, identity, flags):
        self.events.append("preflight")
        return {"source_resolved_or_opened": False, "checkpoint_opened": False, "cuda_initialized": False}

    def prepare(self, *, root, identity, flags):
        self.events.append("prepare")
        flags.source_resolved = flags.source_opened = flags.checkpoint_opened = flags.cuda_initialized = True
        return self.runtime

    def source_authority(self, runtime, *, identity, flags):
        self.events.append("source_authority")
        proof = _theta_proof()
        topology = _session_topology(v1.SOURCE_SMOKE_SESSION)
        topology.update({
            "total_unit_count": proof["raw_t4_shape"][0],
            "valid_unit_count": proof["raw_t4_shape"][0] - proof["known_invalid_unit_count"],
            "invalid_unit_count": proof["known_invalid_unit_count"],
            "valid_mask_sha256": proof["valid_mask_sha256"],
            "theta_valid_mask_sha256": proof["valid_mask_sha256"],
            "raw_t4_channel_order_sha256": proof["canonical_unit_order_sha256"],
        })
        return {
            "schema": "causal_dual_memory_cell_d_source_execution_authority_v1",
            "cell": v1.CELL, "identity_sha256": identity.sha256,
            "strict_train_roster": list(identity.strict_train_roster),
            "strict_train_roster_sha256": v1.roster_sha256(identity.strict_train_roster),
            "normalizers": identity.normalizers.payload(),
            "fixed_assets": {asset.label: asset.payload() for asset in v1.FIXED_ASSETS},
            "runtime_environment": _runtime_attestation(), "sealed_swa_load_proof": _sealed_swa_proof(),
            "physical_session_count": 1, "strict_manifest_bound_without_nonphysical_resolution": True,
            "sessions": [topology],
            "v2_theta_raw_proofs": [proof],
            "source_only": True,
            "access": {
                "source_opened": True, "within_opened": False, "external_opened": False,
                "formal_opened": False, "target_opened": False, "optimizer_steps": 0,
                "backward_calls": 0, "parameter_updates": 0, "normalizer_refit": False,
            },
        }

    def run_smoke(self, runtime, *, identity, flags):
        self.events.append("smoke")
        flags.model_forward_calls += 8
        return {
            "schema": "causal_dual_memory_cell_d_source_execution_smoke_v1",
            "session": v1.SOURCE_SMOKE_SESSION, "budget": 30,
            "support_positions": list(range(30)), "audit_positions": [30, 31],
            "group_count": 4, "b8_threshold_applied": False, "source_only": True,
            **_budget_fields(30),
        }

    def run_budget(self, runtime, *, budget, identity, flags):
        raise AssertionError("smoke mock must not run gate budgets")

    def revalidate(self, *, root, identity, flags):
        self.events.append("revalidate")

    def resources(self, runtime, *, flags):
        self.events.append("resources")
        return {
            "endpoint_chunks_per_s": 1.0, "trials_per_s": 1.0, "wall_seconds": 1.0,
            "endpoint_chunks_completed": 1, "completed_trials": 1, "rss_bytes": 1,
            "current_cuda_allocated_bytes": 4, "current_cuda_reserved_bytes": 8,
            "peak_cuda_allocated_bytes": 4, "peak_cuda_reserved_bytes": 8,
            "selected_device": _profile(),
        }

    def close(self, runtime):
        self.events.append("close")


def test_authorization_rechecks_predecessor_and_freshness_before_backend_or_root_reservation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "tfpd_exploration/results").mkdir(parents=True)
    expectation = _make_v1_failed_predecessor(tmp_path)
    monkeypatch.setattr(v2, "V1_FAILED_PREDECESSOR", expectation)
    v1_closure = {"schema": "synthetic_v1", "closure_sha256": expectation.v1_closure_sha256}
    v2_closure = {
        "schema": "causal_dual_memory_cell_d_source_execution_closure_v2",
        "v1_implementation_closure_sha256": expectation.v1_closure_sha256,
        "paths": [], "closure_sha256": "d" * 64,
    }
    spec = v2.SOURCE_SMOKE_SPEC
    inherited = _inherited_identity(spec, v1_closure)
    identity = v2.SourceExecutionV2Identity(inherited, v2_closure, expectation)
    monkeypatch.setattr(v1, "execution_closure_payload", lambda root: v1_closure)
    monkeypatch.setattr(v2, "execution_closure_payload", lambda root: v2_closure)
    cap = v2._issue_root_reviewed_capability(
        tmp_path, identity, source_data_root=None, seal=v2._ROOT_REVIEW_SEAL,
    )
    backend = _MockBackend()
    result = v2.execute_authorized(
        tmp_path, identity=identity, capability=cap, backend=backend,
        environ={"CUDA_VISIBLE_DEVICES": "1", "CUDA_DEVICE_ORDER": "PCI_BUS_ID"},
    )
    assert result["status"] == "SMOKE_COMPLETED"
    assert backend.events == ["preflight", "prepare", "source_authority", "smoke", "revalidate", "resources", "close"]
    v1_root = tmp_path / expectation.root_relative
    assert {path.name for path in v1_root.iterdir()} == {
        "attempt.json", "attempt.json.sha256", "launch.json", "launch.json.sha256", "failure.json", "failure.json.sha256",
    }
    v2_root = tmp_path / spec.root_relative
    assert {path.name for path in v2_root.iterdir()} == {
        "attempt.json", "attempt.json.sha256", "launch.json", "launch.json.sha256",
        "source_authority.json", "source_authority.json.sha256", "smoke.json", "smoke.json.sha256",
        "terminal.json", "terminal.json.sha256",
    }
    terminal = json.loads((v2_root / "terminal.json").read_bytes())
    assert terminal["launch_closure_sha256"] == terminal["final_closure_sha256"] == "d" * 64
    # Tampering the predecessor after an authority exists blocks a fresh V2
    # attempt before backend preflight and before a second root can reserve.
    second = v1.SourceExecutionSpec("source_smoke", "tfpd_exploration/results/second_v2", v1.SOURCE_SMOKE_SESSION, 30, (30, 31))
    second_identity = v2.SourceExecutionV2Identity(_inherited_identity(second, v1_closure), v2_closure, expectation)
    os.chmod(v1_root / "failure.json", 0o644)
    (v1_root / "failure.json").write_bytes(b"{}")
    os.chmod(v1_root / "failure.json", 0o444)
    with pytest.raises(v2.SourceExecutionV2Error):
        v2._issue_root_reviewed_capability(tmp_path, second_identity, source_data_root=None, seal=v2._ROOT_REVIEW_SEAL)
    assert not (tmp_path / second.root_relative).exists()


def test_v2_physical_seam_subclasses_only_theta_provider_without_data_or_cuda(tmp_path: Path) -> None:
    assert issubclass(physical_v2.V2ThetaStrict27SessionProvider, v1_physical.ConcreteStrict27SessionProvider)
    assert issubclass(physical_v2.PhysicalSourceExecutionBackendV2, v1_physical.PhysicalSourceExecutionBackend)
    seam = inspect.getsource(physical_v2.V2ThetaStrict27SessionProvider._theta_valid_mask)
    assert "theta_raw_t4_float32_bytes_sha256" not in seam  # validation is delegated to the pure V2 helper
    assert "validate_theta_raw_t4_authority" in seam
    assert "posterior_core" not in seam
    source_cap = v1._issue_root_reviewed_source_data_capability(
        canonical_root=tmp_path / "strict_source", strict_train_roster=_roster(), seal=v1._ROOT_REVIEW_SEAL,
    )
    backend = physical_v2.build_reviewed_physical_backend(
        root=ROOT, source_data=source_cap, selected_device=_profile(),
    )
    assert isinstance(backend.provider, physical_v2.V2ThetaStrict27SessionProvider)
    assert isinstance(backend.executor, v1_physical.ConcreteCellDFourGroupExecutor)
    assert not (tmp_path / "strict_source").exists()


def test_static_cli_is_no_torch_no_data_no_cuda_no_write_and_public_execution_fails() -> None:
    script = ROOT / "tfpd_exploration/scripts/run_causal_dual_memory_cell_d_source_gate_v2.py"
    base = [sys.executable, "-I", str(script)]
    dry = subprocess.run(base + ["--dry-run"], cwd=ROOT, text=True, capture_output=True, check=True)
    payload = json.loads(dry.stdout)
    assert payload["workorder_sha256"] == v2.WORKORDER_SHA256
    assert payload["opens_source"] is payload["loads_checkpoint"] is payload["initializes_cuda"] is False
    blocked = subprocess.run(base + ["--execute", "--source-smoke"], cwd=ROOT, text=True, capture_output=True)
    assert blocked.returncode != 0
    assert "root-reviewed in-process capability" in blocked.stderr
