"""No-data/no-CUDA tests for the additive Phase-B smoke-v2 correction."""
from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from typing import Any, Mapping

import pytest
import torch

from src.posterior_carrier_v1 import core, phase_b, phase_b_v2, source_adapter, source_adapter_v2


ROOT = Path(__file__).resolve().parents[2]
SHA_A = "a" * 64
SHA_B = "b" * 64


def _synthetic_v1_closure() -> dict[str, object]:
    hashes = {relative: SHA_A for relative in phase_b.PHASE_B_CLOSURE_PATHS}
    hashes[phase_b.HANDOFF_RELATIVE] = phase_b.HANDOFF_SHA256
    body = {"paths": list(phase_b.PHASE_B_CLOSURE_PATHS), "sha256_by_path": hashes}
    return {**body, "closure_sha256": phase_b.sha256_bytes(phase_b.canonical_json_bytes(body))}


def _synthetic_v2_closure() -> dict[str, object]:
    hashes = {relative: SHA_A for relative in phase_b_v2.PHASE_B_V2_CLOSURE_PATHS}
    hashes[phase_b.HANDOFF_RELATIVE] = phase_b.HANDOFF_SHA256
    body = {"paths": list(phase_b_v2.PHASE_B_V2_CLOSURE_PATHS), "sha256_by_path": hashes}
    return {**body, "closure_sha256": phase_b.sha256_bytes(phase_b.canonical_json_bytes(body))}


def _source_root_payload() -> dict[str, object]:
    return {
        "schema": "posterior_carrier_strict27_source_data_root_v1",
        "source_data_root": phase_b.CANONICAL_SOURCE_DATA_ROOT,
        "directory_device": 1,
        "directory_inode": 2,
        "nwb_copied_into_stage": False,
        "nwb_symlink_or_bind_mount_authorized": False,
    }


def _synthetic_metadata(roster: list[str], *, source_root: Mapping[str, object]) -> dict[str, object]:
    rows = {
        session: {
            "feature_version": 1,
            "path": f"{phase_b.CANONICAL_SOURCE_DATA_ROOT}/{session}_behavior+ecephys.nwb",
            "session": session,
            "sha256": SHA_A,
            "size_bytes": 1,
            "unit_count": 2,
        }
        for session in roster
    }
    assets = {
        relative: {"sha256": SHA_A, "descriptor_identity": {"device": 1, "inode": index + 1, "mode": 0o444}}
        for index, relative in enumerate(phase_b.SOURCE_AUTHORITY_ASSET_PATHS)
    }
    return {
        "admission_preflight": {"path": phase_b.SOURCE_AUTHORITY_ASSET_SPECS[0][0], "body_sha256": SHA_A},
        "theta_receipt": {"path": phase_b.SOURCE_AUTHORITY_ASSET_SPECS[1][0], "body_sha256": SHA_A},
        "theta_artifact": {"path": phase_b.SOURCE_AUTHORITY_ASSET_SPECS[2][0], "body_sha256": SHA_A},
        "roster": list(roster),
        "normalized_t4_sha256": {session: SHA_A for session in roster},
        "raw_authority_sha256": SHA_A,
        "normalizer_authority_sha256": SHA_A,
        "side_semantic_sha256": SHA_A,
        "behavior_semantic_sha256": SHA_B,
        "manifest_sha256": SHA_B,
        "stage_manifest_relative": phase_b.STRICT27_MANIFEST_RELATIVE,
        "source_data_root": dict(source_root),
        "stage_authority_assets": assets,
        "source_lineage": {
            "path": phase_b.SOURCE_AUTHORITY_ASSET_SPECS[3][0],
            "body_sha256": SHA_A,
            "matching_authority_consumed_bytes_sha256": SHA_A,
            "rows_by_session": rows,
        },
    }


def _identity_v2(roster: list[str] | None = None, metadata: Mapping[str, object] | None = None) -> phase_b_v2.RunIdentityV2:
    roster = list(roster if roster is not None else [f"source-{index:02d}" for index in range(27)])
    source_root = _source_root_payload()
    metadata = dict(metadata if metadata is not None else _synthetic_metadata(roster, source_root=source_root))
    source = {
        "schema": "posterior_carrier_target_free_source_identity_v1",
        "roster": roster,
        "roster_sha256": phase_b.sha256_bytes(phase_b.canonical_json_bytes(roster)),
        "strict_source_metadata_sha256": phase_b.sha256_bytes(phase_b.canonical_json_bytes(metadata)),
        "manifest_sha256": metadata["manifest_sha256"],
        "ordinary_raw_t4_semantic_sha256": metadata["side_semantic_sha256"],
        "behavior_normalizer_semantic_sha256": metadata["behavior_semantic_sha256"],
        "source_lineage_sha256": metadata["source_lineage"]["body_sha256"],
        "source_data_root": source_root,
        "source_only": True,
        "target_opened": False,
        "within_opened": False,
        "external_opened": False,
        "formal_opened": False,
        "h1_opened": False,
    }
    return phase_b_v2.RunIdentityV2(
        base_identity=phase_b.RunIdentity(
            source_authority=source, closure=_synthetic_v1_closure(), remote_device=dict(phase_b.REMOTE_TORCH_AUTHORITY),
        ),
        closure=_synthetic_v2_closure(),
    )


def _input_and_recovery(session: str, *, fallback: bool) -> tuple[source_adapter.PosteriorSessionInput, dict[str, object]]:
    prefix = _prefix(missing_position=0 if fallback else None)
    if not fallback:
        # All ordinary prefix indices must remain distinct even when no row is
        # missing; the no-fallback route deliberately receives no corners.
        prefix = [{"trial_index": 1000 + index, "target_dir": source_adapter_v2.CANONICAL_DIRECTIONS_RAD[index % 8]}
                  for index in range(30)]
    theta, row_ids, recovery = source_adapter_v2.recover_same_prefix_theta(
        session=session, prefix=prefix,
        target_corners_by_trial_index={33: _audited_corners()} if fallback else {},
    )
    units = 2
    raw = torch.tensor([[0.2, -0.1, 0.3, 4.0], [-0.3, 0.4, 0.5, 5.0]], dtype=torch.float64)
    counts = torch.arange(1, 1 + units * 30, dtype=torch.int64).reshape(units, 30)
    unit_order_sha = core.tensor_digest(torch.arange(units, dtype=torch.int64))
    return source_adapter.PosteriorSessionInput(
        session_id=session, raw_m30_t4=raw, counts_m30=counts,
        exposure_m30=torch.ones(30, dtype=torch.float64), theta_m30=theta,
        prefix_row_ids=row_ids, source_path_sha256=SHA_A, unit_order_sha256=unit_order_sha,
        raw_t4_row_order_proof={
            "feature_group": "t4", "feature_version": 1, "pool_size": 30, "signal_view": "sua",
            "source_unit_count": units, "channel_ids_are_exact_arange": True, "raw_row_count": units,
            "unit_order_sha256": unit_order_sha,
            "row_semantics": "closure_bound_compute_unit_side_features_uncached_sua_rows_follow_nwb_units_order",
        },
    ), recovery


def _prefix(*, missing_position: int | None = None, missing_trial_index: int = 33) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for position in range(30):
        rows.append({
            "trial_index": missing_trial_index if position == (missing_position if missing_position is not None else -1) else 100 + position,
            "target_dir": None if position == missing_position else source_adapter_v2.CANONICAL_DIRECTIONS_RAD[position % 8],
        })
    return rows


def _audited_corners() -> list[float]:
    return [-6.67259884, 6.6410656, -4.67259884, 4.6410656]


def test_same_prefix_nan_direction_recovers_from_its_own_corners_and_binds_exact_evidence() -> None:
    prefix = _prefix(missing_position=7)
    theta, row_ids, evidence = source_adapter_v2.recover_same_prefix_theta(
        session="sub-C_ses-CO-20150313", prefix=prefix, target_corners_by_trial_index={33: _audited_corners()},
    )
    assert theta.dtype == torch.float64
    assert theta.shape == (30,)
    assert theta[7].item() == pytest.approx(3.0 * math.pi / 4.0, abs=0.0)
    assert evidence["fallback_count"] == 1
    fallback = evidence["fallback_rows"][0]
    assert fallback["trial_index"] == 33
    assert fallback["prefix_position"] == 7
    assert fallback["source"] == source_adapter_v2.FALLBACK_SOURCE
    assert fallback["snap_error_rad"] < source_adapter_v2.THETA_SNAP_TOLERANCE_RAD
    assert evidence["theta_sources_by_prefix_position"][7] == source_adapter_v2.SNAPPED_SOURCE
    assert source_adapter_v2.validate_theta_recovery_evidence(
        evidence, session="sub-C_ses-CO-20150313", prefix_row_ids=row_ids,
        theta_sha256=evidence["theta_m30_sha256"],
    ) == evidence


@pytest.mark.parametrize(
    ("corners", "message"),
    [
        (None, "target_corners"),
        ([float("nan"), 1.0, 2.0, 3.0], "finite"),
        # exact halfway angle (pi/8) is far from the strict pi/64 snap band.
        ([8.0 * math.cos(math.pi / 8.0) - 1.0, 8.0 * math.sin(math.pi / 8.0) - 1.0,
          8.0 * math.cos(math.pi / 8.0) + 1.0, 8.0 * math.sin(math.pi / 8.0) + 1.0], "too far"),
    ],
)
def test_same_prefix_missing_or_ambiguous_or_far_geometry_fails_closed(corners: object, message: str) -> None:
    prefix = _prefix(missing_position=0)
    with pytest.raises(source_adapter_v2.SourceAdapterV2Error, match=message):
        source_adapter_v2.recover_same_prefix_theta(
            session="sub-C_ses-CO-20150313", prefix=prefix, target_corners_by_trial_index={33: corners},
        )


def test_no_later_row_substitution_is_possible() -> None:
    prefix = _prefix(missing_position=0, missing_trial_index=33)
    # A later table row has valid geometry, but the selected row does not.  The
    # primitive must lookup exactly 33 and fail rather than scan for 99.
    with pytest.raises(source_adapter_v2.SourceAdapterV2Error, match="later-row substitution"):
        source_adapter_v2.recover_same_prefix_theta(
            session="sub-C_ses-CO-20150313", prefix=prefix,
            target_corners_by_trial_index={99: _audited_corners()},
        )


def test_same_prefix_theta_design_requires_full_rank() -> None:
    prefix = [{"trial_index": 100 + index, "target_dir": 0.0} for index in range(30)]
    with pytest.raises(source_adapter_v2.SourceAdapterV2Error, match="full rank"):
        source_adapter_v2.recover_same_prefix_theta(
            session="sub-C_ses-CO-20150313", prefix=prefix, target_corners_by_trial_index={},
        )


def test_recovery_evidence_and_expected_one_row_topology_are_tamper_evident() -> None:
    prefix = _prefix(missing_position=0)
    theta, row_ids, evidence = source_adapter_v2.recover_same_prefix_theta(
        session="sub-C_ses-CO-20150313", prefix=prefix, target_corners_by_trial_index={33: _audited_corners()},
    )
    tampered = copy.deepcopy(evidence)
    tampered["fallback_rows"][0]["trial_index"] = 34
    with pytest.raises(source_adapter_v2.SourceAdapterV2Error, match="body digest"):
        source_adapter_v2.validate_theta_recovery_evidence(
            tampered, session="sub-C_ses-CO-20150313", prefix_row_ids=row_ids,
            theta_sha256=core_tensor_digest(theta),
        )
    # Recompute the evidence body after moving the fallback source bit.  The
    # new validator must still reject because trial 33 is bound to row 0, not
    # merely to a syntactically valid fallback position.
    moved = copy.deepcopy(evidence)
    moved["fallback_rows"][0]["prefix_position"] = 1
    moved["theta_sources_by_prefix_position"][0] = source_adapter_v2.NATIVE_SOURCE
    moved["theta_sources_by_prefix_position"][1] = source_adapter_v2.SNAPPED_SOURCE
    _recompute_recovery_body(moved)
    with pytest.raises(source_adapter_v2.SourceAdapterV2Error, match="trial/prefix-row"):
        source_adapter_v2.validate_theta_recovery_evidence(
            moved, session="sub-C_ses-CO-20150313", prefix_row_ids=row_ids,
            theta_sha256=core_tensor_digest(theta),
        )
    assert source_adapter_v2.EXPECTED_FALLBACK_TOPOLOGY_SHA256 == "949431bc77bb484cd2a84e2469d2131bccfb3207655166b6921f41bad8e737c8"
    assert source_adapter_v2.validate_theta_fallback_topology({
        **source_adapter_v2.EXPECTED_FALLBACK_TOPOLOGY,
        "body_sha256": source_adapter_v2.EXPECTED_FALLBACK_TOPOLOGY_SHA256,
    })["body_sha256"] == source_adapter_v2.EXPECTED_FALLBACK_TOPOLOGY_SHA256
    wrong = {"schema": source_adapter_v2.THETA_FALLBACK_TOPOLOGY_SCHEMA,
             "fallback_rows": [{"session_id": "sub-C_ses-CO-20150313", "trial_index": 34}],
             "body_sha256": source_adapter_v2.EXPECTED_FALLBACK_TOPOLOGY_SHA256}
    with pytest.raises(source_adapter_v2.SourceAdapterV2Error, match="digest/binding"):
        source_adapter_v2.validate_theta_fallback_topology(wrong)


def core_tensor_digest(value: torch.Tensor) -> str:
    return core.tensor_digest(value)


def _recompute_recovery_body(value: dict[str, object]) -> None:
    value["body_sha256"] = phase_b.sha256_bytes(phase_b.canonical_json_bytes(
        {key: item for key, item in value.items() if key != "body_sha256"},
    ))


def _complete_v2_source_authority(
    identity: phase_b_v2.RunIdentityV2,
    *,
    launch_sha256: str = SHA_B,
) -> dict[str, object]:
    roster = tuple(identity.base_identity.source_authority["roster"])
    inputs: dict[str, source_adapter.PosteriorSessionInput] = {}
    recovery: dict[str, Mapping[str, object]] = {}
    for session in roster:
        item, evidence = _input_and_recovery(
            session, fallback=(session == "sub-C_ses-CO-20150313"),
        )
        inputs[session] = item
        recovery[session] = evidence
    prior, bank, _ = source_adapter.build_source_posterior_bank(roster=roster, inputs=inputs, seed=42)
    metadata = _synthetic_metadata(list(roster), source_root=identity.base_identity.source_authority["source_data_root"])
    adapter = source_adapter_v2.PhysicalPosteriorSourceAdapterV2(
        dataset=None, sampler=None, roster=roster, bank=bank, prior=prior, session_inputs=inputs,
        source_authority_metadata=metadata,
        behavior_normalizer_semantic_sha256=SHA_B, preparation_seconds=0.01, train_files=(),
        theta_recovery_by_session=recovery,
    )
    nested = source_adapter.PhysicalPosteriorSourceAdapter.source_authority_payload(
        adapter, closure=identity.base_identity.closure,
    )
    nested.update({
        "posterior_preparation_seconds": 0.01,
        "remote_torch_authority": dict(phase_b.REMOTE_TORCH_AUTHORITY),
        "posterior_credibility_statistics": source_adapter._posterior_statistics(bank),
        "optimizer": {
            "class": "Adam", "lr_constructor": 1e-4, "betas": [0.9, 0.999], "eps": 1e-8,
            "weight_decay": 0.0, "amsgrad": False, "schedule": "arm_common.lr_at_step(48,33925)",
        },
        "execution_policy": {"amp": False, "tf32": False, "torch_compile": False, "batch_size": 32},
        "launch_sha256": launch_sha256,
    })
    authority = source_adapter_v2.wrap_v1_authority_with_theta_recovery(
        adapter=adapter, nested_v1_authority=nested, v2_closure=identity.closure,
    )
    authority["launch_sha256"] = launch_sha256
    return authority


def test_v2_source_authority_validates_v1_semantics_and_one_row_recovery_only() -> None:
    roster = ["sub-C_ses-CO-20150313"] + [f"source-{index:02d}" for index in range(26)]
    source_root = _source_root_payload()
    metadata = _synthetic_metadata(roster, source_root=source_root)
    identity = _identity_v2(roster, metadata)
    authority = _complete_v2_source_authority(identity)
    validated = phase_b_v2.validate_source_authority_v2(authority, identity=identity, launch_sha256=SHA_B)
    assert validated == authority
    assert authority["theta_fallback_topology"]["body_sha256"] == source_adapter_v2.EXPECTED_FALLBACK_TOPOLOGY_SHA256
    forged = copy.deepcopy(authority)
    forged["theta_recovery_by_session"]["sub-C_ses-CO-20150313"]["fallback_rows"][0]["trial_index"] = 99
    with pytest.raises(source_adapter_v2.SourceAdapterV2Error, match="body digest"):
        phase_b_v2.validate_source_authority_v2(forged, identity=identity, launch_sha256=SHA_B)
    forged = copy.deepcopy(authority)
    recovery = forged["theta_recovery_by_session"]["sub-C_ses-CO-20150313"]
    recovery["fallback_rows"][0]["prefix_position"] = 1
    recovery["theta_sources_by_prefix_position"][0] = source_adapter_v2.NATIVE_SOURCE
    recovery["theta_sources_by_prefix_position"][1] = source_adapter_v2.SNAPPED_SOURCE
    _recompute_recovery_body(recovery)
    # The aggregate topology is intentionally unchanged (same trial 33), so
    # this exercises the per-row position binding rather than merely the
    # global one-fallback topology guard.
    with pytest.raises(source_adapter_v2.SourceAdapterV2Error, match="trial/prefix-row"):
        phase_b_v2.validate_source_authority_v2(forged, identity=identity, launch_sha256=SHA_B)
    forged = copy.deepcopy(authority)
    forged["theta_fallback_topology"]["fallback_rows"].append({"session_id": "source-00", "trial_index": 1})
    with pytest.raises(source_adapter_v2.SourceAdapterV2Error, match="digest/binding"):
        phase_b_v2.validate_source_authority_v2(forged, identity=identity, launch_sha256=SHA_B)


class _FailPrepareBackend:
    def __init__(self) -> None:
        self.closed = False

    def prepare(self, spec: phase_b.SourceSmokeSpec, identity: phase_b_v2.RunIdentityV2) -> Any:
        raise phase_b.SourceSmokeExecutionError(
            stage="prepare",
            progress=phase_b.SmokeExecutionProgress(source_opened=True, remote_initialized=False),
            cause=RuntimeError("synthetic same-prefix source failure"),
        )

    def source_authority(self, runtime: Any, identity: phase_b_v2.RunIdentityV2) -> Mapping[str, object]:
        raise AssertionError("prepare failure must not enter source authority")

    def run_steps(self, runtime: Any, spec: phase_b.SourceSmokeSpec) -> phase_b.SmokeStepSummary:
        raise AssertionError("prepare failure must not enter steps")

    def close(self, runtime: Any | None) -> None:
        self.closed = True


def test_v2_failure_lifecycle_binds_v1_failure_lineage_and_honest_progress() -> None:
    identity = _identity_v2()
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        (root / "tfpd_exploration/results").mkdir(parents=True)
        artifact = phase_b.reserve_source_smoke_root(root, relative=phase_b_v2.SOURCE_SMOKE_V2_ROOT_RELATIVE)
        backend = _FailPrepareBackend()
        with pytest.raises(phase_b.SourceSmokeExecutionError):
            phase_b_v2.run_source_smoke_lifecycle_v2(backend=backend, artifact=artifact, identity=identity)
        assert backend.closed is True
        assert (artifact.directory / "attempt.json").exists()
        assert (artifact.directory / "launch.json").exists()
        assert (artifact.directory / "failure.json").exists()
        assert not (artifact.directory / "terminal.json").exists()
        failure = artifact.reload_json("failure.json")
        assert failure["v1_failed_predecessor"]["failure_body_sha256"] == phase_b_v2.V1_FAILURE_BODY_SHA256
        assert failure["source_opened"] is True
        assert failure["remote_initialized"] is False
        assert failure["optimizer_steps_completed"] == 0


def test_v2_identity_rejects_v1_subset_or_theta_topology_drift() -> None:
    identity = _identity_v2()
    phase_b_v2.validate_run_identity_v2(identity)
    closure = copy.deepcopy(identity.closure)
    closure["sha256_by_path"][phase_b.PHASE_B_CLOSURE_PATHS[1]] = SHA_B
    body = {"paths": closure["paths"], "sha256_by_path": closure["sha256_by_path"]}
    closure["closure_sha256"] = phase_b.sha256_bytes(phase_b.canonical_json_bytes(body))
    bad = phase_b_v2.RunIdentityV2(base_identity=identity.base_identity, closure=closure)
    with pytest.raises(phase_b.PhaseBError, match="cross-binding"):
        phase_b_v2.validate_run_identity_v2(bad)


def test_physical_v2_prepare_marks_source_before_v2_adapter_and_never_reaches_cuda_on_adapter_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Physical seam regression without source files, CUDA, or model setup."""
    identity = _identity_v2()

    class _Capability:
        def payload(self) -> Mapping[str, object]:
            return identity.base_identity.source_authority["source_data_root"]

        def validate(self) -> Path:
            return Path("/synthetic/source-root")

    called = {"adapter": False}

    def fail_adapter(*args: object, **kwargs: object) -> object:
        called["adapter"] = True
        callback = kwargs["on_source_opened"]
        callback()
        raise source_adapter_v2.SourceAdapterV2Error("synthetic v2 source adapter stop")

    monkeypatch.setattr(phase_b_v2, "phase_b_v2_closure", lambda _root: identity.closure)
    monkeypatch.setattr(source_adapter_v2, "build_physical_source_adapter_v2", fail_adapter)
    backend = phase_b_v2.RemotePosteriorSourceSmokeBackendV2(
        Path("/synthetic/stage"), source_data=_Capability(), num_workers=4,
    )
    with pytest.raises(phase_b.SourceSmokeExecutionError) as caught:
        backend.prepare(phase_b.SMOKE_SPEC, identity)
    assert called["adapter"] is True
    assert caught.value.stage == "prepare"
    assert caught.value.progress.source_opened is True
    assert caught.value.progress.remote_initialized is False


def test_v2_remote_staging_plan_is_explicit_and_carries_all_v1_predecessor_pairs(monkeypatch: pytest.MonkeyPatch) -> None:
    closure = _synthetic_v2_closure()
    for relative, expected_sha, _mode in phase_b.SOURCE_AUTHORITY_ASSET_SPECS:
        closure["sha256_by_path"][relative] = expected_sha
    body = {"paths": closure["paths"], "sha256_by_path": closure["sha256_by_path"]}
    closure["closure_sha256"] = phase_b.sha256_bytes(phase_b.canonical_json_bytes(body))

    class _SourceCapability:
        def payload(self) -> Mapping[str, object]:
            return _source_root_payload()

    plan = phase_b_v2.remote_staging_plan_v2(
        closure=closure, source_authority_sha256=SHA_A, source_data=_SourceCapability(),
    )
    paths = [item["relative_path"] for item in plan["stage_files"]]
    assert "tfpd_exploration/src/posterior_carrier_v1/source_adapter_v2.py" in paths
    assert "tfpd_exploration/src/posterior_carrier_v1/phase_b_v2.py" in paths
    assert "tfpd_exploration/scripts/run_posterior_carrier_phase_b_source_smoke_v2.py" in paths
    assert not any("*" in path or "glob" in path for path in paths)
    assert plan["v1_failed_predecessor"]["failure_body_sha256"] == phase_b_v2.V1_FAILURE_BODY_SHA256
    predecessor_root = phase_b_v2.V1_FAILURE_ROOT_RELATIVE
    predecessor_files = [item for item in plan["stage_files"] if item["relative_path"].startswith(f"{predecessor_root}/")]
    expected_predecessor_files: list[dict[str, object]] = []
    for name, digest in phase_b_v2.V1_PREDECESSOR_STAGE_ASSETS:
        relative = f"{predecessor_root}/{name}"
        expected_predecessor_files.extend([
            {"relative_path": relative, "sha256": digest, "mode": 0o444,
             "role": "immutable_v1_failed_predecessor_receipt"},
            {"relative_path": f"{relative}.sha256", "contents": f"{digest}  {name}\n", "mode": 0o444,
             "role": "immutable_v1_failed_predecessor_sidecar"},
        ])
    assert predecessor_files == expected_predecessor_files
    assets = phase_b_v2.V1_PREDECESSOR_STAGE_ASSETS
    monkeypatch.setattr(phase_b_v2, "V1_PREDECESSOR_STAGE_ASSETS", assets[:-1])
    with pytest.raises(phase_b_v2.PhaseBV2Error, match="stage-asset literal"):
        phase_b_v2.remote_staging_plan_v2(closure=closure, source_authority_sha256=SHA_A, source_data=_SourceCapability())
    monkeypatch.setattr(phase_b_v2, "V1_PREDECESSOR_STAGE_ASSETS", ((assets[0][0], SHA_A), *assets[1:]))
    with pytest.raises(phase_b_v2.PhaseBV2Error, match="stage-asset literal"):
        phase_b_v2.remote_staging_plan_v2(closure=closure, source_authority_sha256=SHA_A, source_data=_SourceCapability())


def test_predecessor_pair_reader_holds_one_directory_fd_across_all_three_receipts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Read all six concrete files from one held root descriptor, no data/CUDA."""
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        synthetic_assets: list[tuple[str, str]] = []
        for name in ("attempt.json", "launch.json", "failure.json"):
            body = phase_b.canonical_json_bytes({"name": name})
            digest = phase_b.sha256_bytes(body)
            (directory / name).write_bytes(body)
            (directory / f"{name}.sha256").write_bytes(f"{digest}  {name}\n".encode("ascii"))
            os.chmod(directory / name, 0o444)
            os.chmod(directory / f"{name}.sha256", 0o444)
            synthetic_assets.append((name, digest))
        opened_directory_fds = 0
        real_open = os.open

        def observe_open(path: object, flags: int, *args: object, **kwargs: object) -> int:
            nonlocal opened_directory_fds
            if flags & os.O_DIRECTORY:
                opened_directory_fds += 1
            return real_open(path, flags, *args, **kwargs)

        monkeypatch.setattr(phase_b_v2.os, "open", observe_open)
        monkeypatch.setattr(phase_b_v2, "_validate_v1_predecessor_stage_assets", lambda: tuple(synthetic_assets))
        pairs = phase_b_v2._read_v1_predecessor_pairs(directory)
        assert list(pairs) == ["attempt.json", "launch.json", "failure.json"]
        assert opened_directory_fds == 1
        # A recomputed directory read cannot accept a substituted sidecar.
        os.chmod(directory / "launch.json.sha256", 0o600)
        (directory / "launch.json.sha256").write_bytes(b"wrong\n")
        os.chmod(directory / "launch.json.sha256", 0o444)
        with pytest.raises(phase_b_v2.PhaseBV2Error, match="sidecar"):
            phase_b_v2._read_v1_predecessor_pairs(directory)


def test_static_v2_cli_loads_no_torch_and_public_execution_flag_fails_closed() -> None:
    script = ROOT / "tfpd_exploration/scripts/run_posterior_carrier_phase_b_source_smoke_v2.py"
    environment = {**os.environ, "PYTHONNOUSERSITE": "1", "PYTHONPATH": "", "CUDA_VISIBLE_DEVICES": ""}
    code = (
        "import importlib.util,json,sys; "
        f"spec=importlib.util.spec_from_file_location('pc_b2', {str(script)!r}); "
        "module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);"
        "payload=module._load_plan();print(json.dumps({'torch_loaded':'torch' in sys.modules,'status':payload['status']}))"
    )
    completed = subprocess.run([sys.executable, "-c", code], env=environment, check=True, text=True, capture_output=True)
    assert json.loads(completed.stdout) == {"torch_loaded": False, "status": "DRY_FAIL_CLOSED_PENDING_ROOT_REVIEW"}
    rejected = subprocess.run([sys.executable, str(script), "--execute-remote-smoke-v2"], env=environment, text=True, capture_output=True)
    assert rejected.returncode != 0
    assert "root-reviewed capability" in rejected.stderr
