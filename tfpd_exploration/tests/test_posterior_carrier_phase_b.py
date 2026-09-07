"""No-data/no-CUDA tests for Posterior Carrier Phase-B source-smoke scaffolding."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
import types
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import pytest
import torch

from src.posterior_carrier_v1 import core, phase_b, source_adapter


ROOT = Path(__file__).resolve().parents[2]
SHA_A = "a" * 64
SHA_B = "b" * 64


def _make_input(session: str, *, units: int = 4) -> source_adapter.PosteriorSessionInput:
    theta = torch.arange(30, dtype=torch.float64) * (2.0 * torch.pi / 8.0)
    base = torch.arange(units, dtype=torch.int64).unsqueeze(1)
    trial = torch.arange(30, dtype=torch.int64).unsqueeze(0)
    counts = (base * 3 + trial.remainder(7) + 1).to(torch.int64)
    if session.endswith("zero"):
        counts[-1].zero_()
    raw = torch.stack((
        torch.linspace(-1.0, 1.0, units, dtype=torch.float64),
        torch.linspace(1.0, -1.0, units, dtype=torch.float64),
        torch.linspace(0.2, 1.2, units, dtype=torch.float64),
        torch.linspace(3.0, 5.0, units, dtype=torch.float64),
    ), dim=1)
    unit_order_sha = core.tensor_digest(torch.arange(units, dtype=torch.int64))
    return source_adapter.PosteriorSessionInput(
        session_id=session,
        raw_m30_t4=raw,
        counts_m30=counts,
        exposure_m30=torch.linspace(0.4, 1.5, 30, dtype=torch.float64),
        theta_m30=theta,
        prefix_row_ids=tuple(f"{session}:trial:{index}" for index in range(30)),
        source_path_sha256=SHA_A,
        unit_order_sha256=unit_order_sha,
        raw_t4_row_order_proof={
            "feature_group": "t4", "feature_version": 1, "pool_size": 30, "signal_view": "sua",
            "source_unit_count": units, "channel_ids_are_exact_arange": True,
            "raw_row_count": units, "unit_order_sha256": unit_order_sha,
            "row_semantics": "closure_bound_compute_unit_side_features_uncached_sua_rows_follow_nwb_units_order",
        },
    )


def _bank(*, roster: tuple[str, ...] = ("source-00", "source-zero")) -> phase_b.PosteriorEpochCarrierBank:
    inputs = {session: _make_input(session) for session in roster}
    _prior, bank, _posteriors = source_adapter.build_source_posterior_bank(roster=roster, inputs=inputs)
    return bank


def _synthetic_closure() -> dict[str, object]:
    hashes = {relative: SHA_A for relative in phase_b.PHASE_B_CLOSURE_PATHS}
    hashes[core.HANDOFF_RELATIVE] = core.HANDOFF_SHA256
    body = {"paths": list(phase_b.PHASE_B_CLOSURE_PATHS), "sha256_by_path": hashes}
    return {**body, "closure_sha256": phase_b.sha256_bytes(phase_b.canonical_json_bytes(body))}


def _synthetic_source_metadata(roster: list[str], *, source_data_root: Mapping[str, object]) -> dict[str, object]:
    rows = {
        session: {
            "feature_version": 1,
            "path": f"{phase_b.CANONICAL_SOURCE_DATA_ROOT}/{session}_behavior+ecephys.nwb",
            "session": session,
            "sha256": SHA_A,
            "size_bytes": 1,
            "unit_count": 1,
        }
        for session in roster
    }
    assets = {
        relative: {
            "sha256": SHA_A,
            "descriptor_identity": {"device": 1, "inode": index + 1, "mode": 0o444},
        }
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
        "source_data_root": dict(source_data_root),
        "stage_authority_assets": assets,
        "source_lineage": {
            "path": phase_b.SOURCE_AUTHORITY_ASSET_SPECS[3][0],
            "body_sha256": SHA_A,
            "matching_authority_consumed_bytes_sha256": SHA_A,
            "rows_by_session": rows,
        },
    }


def _identity() -> phase_b.RunIdentity:
    roster = [f"source-{index:02d}" for index in range(27)]
    source_root = {
        "schema": "posterior_carrier_strict27_source_data_root_v1",
        "source_data_root": phase_b.CANONICAL_SOURCE_DATA_ROOT,
        "directory_device": 1,
        "directory_inode": 2,
        "nwb_copied_into_stage": False,
        "nwb_symlink_or_bind_mount_authorized": False,
    }
    metadata = _synthetic_source_metadata(roster, source_data_root=source_root)
    return phase_b.RunIdentity(
        source_authority={
            "schema": "posterior_carrier_target_free_source_identity_v1",
            "roster": roster,
            "roster_sha256": phase_b.sha256_bytes(phase_b.canonical_json_bytes(roster)),
            "strict_source_metadata_sha256": phase_b.sha256_bytes(phase_b.canonical_json_bytes(metadata)),
            "manifest_sha256": SHA_B,
            "ordinary_raw_t4_semantic_sha256": SHA_A,
            "behavior_normalizer_semantic_sha256": SHA_B,
            "source_lineage_sha256": SHA_A,
            "source_data_root": source_root,
            "source_only": True,
            "target_opened": False,
            "within_opened": False,
            "external_opened": False,
            "formal_opened": False,
            "h1_opened": False,
        },
        closure=_synthetic_closure(),
        remote_device=dict(phase_b.REMOTE_TORCH_AUTHORITY),
    )


def _summary() -> phase_b.SmokeStepSummary:
    return phase_b.SmokeStepSummary(
        loss_first=1.0, loss_last=0.8, loss_min=0.7, loss_max=1.1, nonincreasing_transitions=75,
        losses_count=100,
        critical_gradients={
            "b3s_pre_pool_activity": True, "b3s_post_pool_activity": True, "b3s_post_pool_t4": True,
            "decoder_fc_in": True, "cross_attention": True, "ffn": True, "query_rep": True, "output_fc": True,
        },
        finite_model=True, finite_adam=True, model_state_sha256=SHA_A, optimizer_state_sha256=SHA_B,
        posterior_prepare_seconds=0.01, optimizer_core_seconds=2.0, optimizer_steps_per_second=50.0,
        peak_allocated_bytes=10, peak_reserved_bytes=12, current_allocated_bytes=9, current_reserved_bytes=11,
        posterior_cache={
            "posterior_fit_calls": 81,
            "posterior_inverse_calls": 81,
            "deterministic_mean_view_builds": 81,
            "epoch_sampled_view_builds": 27,
            "device_epoch_view_builds": 27,
            "normalized_view_builds": 108,
            "batch_loop_requests": 100,
            "batch_loop_inverse_calls": 0,
            "source_sessions": 27,
            "scheduled_session_epochs": 27,
        },
        dropout={
            "dynamic_dropout": True,
            "low": 0.0,
            "high": 1.0,
            "semantics": "complete_fused_unit_token_placeholder_with_inverse_probability_gain",
            "extra_dropout_draws": 0,
        },
        remote_device=dict(phase_b.REMOTE_TORCH_AUTHORITY),
    )


class _MockBackend:
    def __init__(self, *, failure: str | None = None, source_metadata: Mapping[str, object] | None = None) -> None:
        self.failure = failure
        self.source_metadata = source_metadata
        self.closed = False

    def prepare(self, spec: phase_b.SourceSmokeSpec, identity: phase_b.RunIdentity) -> Any:
        if self.failure == "prepare":
            raise RuntimeError("synthetic prepare failure")
        return SimpleNamespace(source_opened=True, remote_initialized=True)

    def source_authority(self, runtime: Any, identity: phase_b.RunIdentity) -> Mapping[str, object]:
        if self.failure == "source_authority":
            raise RuntimeError("synthetic authority failure")
        source = identity.source_authority
        metadata = (dict(self.source_metadata) if self.source_metadata is not None
                    else _synthetic_source_metadata(list(source["roster"]), source_data_root=source["source_data_root"]))
        roster = tuple(source["roster"])
        inputs = {session: _make_input(session, units=1) for session in roster}
        prior, bank, _posteriors = source_adapter.build_source_posterior_bank(roster=roster, inputs=inputs)
        statistics = source_adapter._posterior_statistics(bank)
        return {
            "schema": "posterior_carrier_source_authority_v1",
            "cell": core.CELL,
            "source_only": True,
            "target_opened": False,
            "within_opened": False,
            "external_opened": False,
            "formal_opened": False,
            "h1_opened": False,
            "roster": list(source["roster"]),
            "roster_sha256": source["roster_sha256"],
            "strict_source_metadata_sha256": source["strict_source_metadata_sha256"],
            "manifest_sha256": source["manifest_sha256"],
            "ordinary_raw_t4_semantic_sha256": source["ordinary_raw_t4_semantic_sha256"],
            "behavior_normalizer_semantic_sha256": source["behavior_normalizer_semantic_sha256"],
            "source_lineage_sha256": source["source_lineage_sha256"],
            "source_data_root": source["source_data_root"],
            "source_authority_metadata": metadata,
            "source_raw_m30_t4_sha256": prior.raw_m30_t4_sha256,
            "posterior_prior": prior.payload(),
            "posterior_normalizer": bank.normalizer.payload(),
            "posterior_inputs": {session: inputs[session].payload() for session in roster},
            "posterior_bank": bank.source_posterior_digest_payload(),
            "posterior_preparation_seconds": 0.01,
            "normalizer_phase_b_amendment": phase_b.PHASE_B_NORMALIZER_AMENDMENT,
            "m30_b3s_activity_prefix": "held; carrier label budgets only vary M4/M10/M30",
            "cache_read_or_write": False,
            "remote_torch_authority": dict(phase_b.REMOTE_TORCH_AUTHORITY),
            "posterior_credibility_statistics": statistics,
            "optimizer": {
                "class": "Adam", "lr_constructor": 1e-4, "betas": [0.9, 0.999], "eps": 1e-8,
                "weight_decay": 0.0, "amsgrad": False, "schedule": "arm_common.lr_at_step(48,33925)",
            },
            "execution_policy": {"amp": False, "tf32": False, "torch_compile": False, "batch_size": 32},
        }

    def run_steps(self, runtime: Any, spec: phase_b.SourceSmokeSpec) -> phase_b.SmokeStepSummary:
        if self.failure == "steps":
            raise RuntimeError("synthetic step failure")
        return _summary()

    def close(self, runtime: Any | None) -> None:
        self.closed = True


def test_static_phase_b_cli_imports_no_torch_and_fails_closed() -> None:
    script = ROOT / "tfpd_exploration/scripts/run_posterior_carrier_phase_b_source_smoke.py"
    environment = {**os.environ, "PYTHONNOUSERSITE": "1", "PYTHONPATH": "", "CUDA_VISIBLE_DEVICES": ""}
    code = (
        "import importlib.util,json,sys; "
        f"spec=importlib.util.spec_from_file_location('pc_b', {str(script)!r}); "
        "module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);"
        "payload=module._load_plan();print(json.dumps({'torch_loaded':'torch' in sys.modules,'status':payload['status']}))"
    )
    imported = subprocess.run([sys.executable, "-c", code], env=environment, check=True, text=True, capture_output=True)
    assert json.loads(imported.stdout) == {"torch_loaded": False, "status": "DRY_FAIL_CLOSED_PENDING_ROOT_REVIEW"}
    rendered = subprocess.run([sys.executable, str(script), "--plan"], env=environment, check=True, text=True, capture_output=True)
    assert json.loads(rendered.stdout)["handoff"]["sha256"] == core.HANDOFF_SHA256
    denied = subprocess.run([sys.executable, str(script), "--execute-remote-smoke"], env=environment, text=True, capture_output=True)
    assert denied.returncode != 0 and "in-process root-reviewed capability" in denied.stderr


def test_bank_precomputes_3_fits_per_session_and_27x48_epoch_views_without_batch_inverse() -> None:
    bank = _bank()
    # All 48 logical epochs are prewarmed before a hypothetical full-run batch
    # loop.  Fitting stays 2×3; sampled/normalized training views are 2×48.
    for epoch in range(48):
        bank.prewarm_epoch(epoch)
    before = bank.observer(expected_full_schedule=True)
    assert before.posterior_fit_calls == 6
    assert before.posterior_inverse_calls == 6
    assert before.epoch_sampled_view_builds == 96
    assert before.device_epoch_view_builds == 0
    assert before.batch_loop_inverse_calls == 0
    first = bank.optimizer_batch_view(session="source-00", epoch=0)
    second = bank.optimizer_batch_view(session="source-00", epoch=0)
    reference = bank.reference_uncached_training_view(session="source-00", epoch=0)
    assert first is second  # optimizer batches reuse the one prepared carrier object
    assert torch.equal(first.raw_beta, second.raw_beta)
    assert torch.equal(first.raw_beta, reference.raw_beta)
    assert torch.equal(first.normalized_t4, reference.normalized_t4)
    after = bank.observer(expected_full_schedule=True)
    assert after.batch_loop_requests == 2
    assert after.batch_loop_inverse_calls == 0
    assert after.epoch_sampled_view_builds == 96
    # An un-prewarmed batch is a hard failure rather than a hidden inverse/refit.
    fresh = _bank()
    cache_before = fresh.observer().epoch_sampled_view_builds
    with pytest.raises(phase_b.PhaseBError, match="attempted to construct"):
        fresh.optimizer_batch_view(session="source-00", epoch=0)
    assert fresh.observer().epoch_sampled_view_builds == cache_before
    fresh.prewarm_epoch(0)
    with pytest.raises(phase_b.PhaseBError, match="device posterior carrier"):
        fresh.optimizer_batch_view(session="source-00", epoch=0, device="cpu")
    assert fresh.observer().device_epoch_view_builds == 0


def test_epoch_cache_prewarm_preserves_host_rng_and_source_authority_cannot_relax_boundaries() -> None:
    bank = _bank()
    before = core.host_rng_fingerprint()
    for epoch in range(48):
        bank.prewarm_epoch(epoch)
    core.assert_host_rng_unchanged(before, core.host_rng_fingerprint())
    assert bank.observer(expected_full_schedule=True).epoch_sampled_view_builds == 96

    identity = _identity()
    authority = dict(_MockBackend().source_authority(SimpleNamespace(), identity))
    authority["launch_sha256"] = SHA_A
    authority["closure"] = identity.closure
    validated = phase_b.validate_source_authority_for_identity(
        authority,
        identity=identity,
        launch_sha256=SHA_A,
    )
    assert validated["source_only"] is True
    forged = dict(authority)
    forged["formal_opened"] = True
    with pytest.raises(phase_b.PhaseBError, match="target-surface boundary"):
        phase_b.validate_source_authority_for_identity(forged, identity=identity, launch_sha256=SHA_A)


def test_full_strict27_schedule_constructs_exactly_one_carrier_per_session_epoch_before_batches() -> None:
    roster = tuple(f"source-{index:02d}" for index in range(27))
    inputs = {session: _make_input(session, units=1) for session in roster}
    _prior, bank, _posteriors = source_adapter.build_source_posterior_bank(roster=roster, inputs=inputs)
    before = core.host_rng_fingerprint()
    for epoch in range(phase_b.SOURCE_EPOCHS):
        bank.prewarm_epoch(epoch)
        bank.materialize_epoch_for_device(epoch=epoch, device="cpu")
    core.assert_host_rng_unchanged(before, core.host_rng_fingerprint())
    preloop = bank.observer(expected_full_schedule=True)
    assert preloop.posterior_fit_calls == 27 * 3
    assert preloop.posterior_inverse_calls == 27 * 3
    assert preloop.epoch_sampled_view_builds == 27 * 48
    assert preloop.device_epoch_view_builds == 27 * 48
    assert preloop.normalized_view_builds == 27 * 48
    assert preloop.batch_loop_inverse_calls == 0
    for step in range(100):
        session = roster[step % len(roster)]
        assert bank.optimizer_batch_view(session=session, epoch=0, device="cpu").sampled is True
    postloop = bank.observer(expected_full_schedule=True)
    assert postloop.batch_loop_requests == 100
    assert postloop.posterior_inverse_calls == 27 * 3
    assert postloop.batch_loop_inverse_calls == 0
    assert postloop.device_epoch_view_builds == 27 * 48


def test_cache_and_uncached_reference_have_identical_cell_d_forward_loss_and_gradients() -> None:
    from src.tfpd_lane.pop_robust import build_population_robustness_model

    bank = _bank(roster=("source-00", "source-01"))
    bank.prewarm_epoch(0)
    bank.materialize_epoch_for_device(epoch=0, device="cpu")
    cached = bank.optimizer_batch_view(session="source-00", epoch=0, device="cpu")
    reference = bank.reference_uncached_training_view(session="source-00", epoch=0)
    assert torch.equal(cached.raw_beta, reference.raw_beta)
    assert cached.raw_beta.data_ptr() != reference.raw_beta.data_ptr()
    model = build_population_robustness_model(seed=42, cell="D")
    wrapper = core.CellDPosteriorWrapper(model).train()
    torch.manual_seed(19)
    neural = torch.randn(1, 50, cached.unit_count)
    calib = torch.randn(1, 30, 100, cached.unit_count)
    target = torch.randn(1, 50, 2)

    def run(view: core.PosteriorCarrierView) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        random.seed(71)
        torch.manual_seed(72)
        wrapper.zero_grad(set_to_none=True)
        prediction, _identity = wrapper(neural, calib_trials_m30=calib, carrier=view)
        loss = (prediction - target).square().mean()
        loss.backward()
        gradients = {name: parameter.grad.detach().clone() for name, parameter in wrapper.named_parameters()
                     if parameter.grad is not None}
        return prediction.detach().clone(), loss.detach().clone(), gradients

    cached_prediction, cached_loss, cached_gradients = run(cached)
    reference_prediction, reference_loss, reference_gradients = run(reference)
    assert torch.equal(cached_prediction, reference_prediction)
    assert torch.equal(cached_loss, reference_loss)
    assert cached_gradients.keys() == reference_gradients.keys()
    assert all(torch.equal(cached_gradients[name], reference_gradients[name]) for name in cached_gradients)
    proof = phase_b.critical_gradient_proof(wrapper)
    assert all(proof.values())


def test_source_only_lifecycle_publishes_immutable_success_or_honest_failure_in_temp_root() -> None:
    identity = _identity()
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        artifact = phase_b.reserve_source_smoke_root(root, relative="posterior_smoke")
        backend = _MockBackend()
        terminal = phase_b.run_source_smoke_lifecycle(backend=backend, artifact=artifact, identity=identity)
        assert terminal["status"] == "SOURCE_SMOKE_COMPLETE__NON_AUTHORITATIVE"
        assert backend.closed is True
        assert (artifact.directory / "failure.json").exists() is False
        assert all((artifact.directory / name).exists() for name in ("attempt.json", "launch.json", "source_authority.json", "step100.json", "terminal.json"))
        assert stat_mode(artifact.directory / "terminal.json") == 0o444
    with tempfile.TemporaryDirectory() as temporary:
        artifact = phase_b.reserve_source_smoke_root(Path(temporary), relative="posterior_smoke")
        backend = _MockBackend(failure="steps")
        with pytest.raises(RuntimeError, match="synthetic step failure"):
            phase_b.run_source_smoke_lifecycle(backend=backend, artifact=artifact, identity=identity)
        failure = artifact.reload_json("failure.json")
        assert failure["stage"] == "steps"
        assert failure["terminal_published"] is False
        assert not (artifact.directory / "terminal.json").exists()
        assert backend.closed is True

    class _TerminalReloadFault:
        def __init__(self, delegate: phase_b.ArtifactRoot) -> None:
            self._delegate = delegate

        def publish_json(self, name: str, value: Mapping[str, object]) -> str:
            return self._delegate.publish_json(name, value)

        def reload_json(self, name: str, expected_sha256: str | None = None) -> Mapping[str, object]:
            if name == "terminal.json":
                raise phase_b.PhaseBError("synthetic terminal reload fault")
            return self._delegate.reload_json(name, expected_sha256)

    with tempfile.TemporaryDirectory() as temporary:
        artifact = phase_b.reserve_source_smoke_root(Path(temporary), relative="posterior_smoke")
        with pytest.raises(phase_b.PhaseBError, match="terminal reload fault"):
            phase_b.run_source_smoke_lifecycle(
                backend=_MockBackend(), artifact=_TerminalReloadFault(artifact), identity=identity,
            )
        assert (artifact.directory / "terminal.json").exists()
        assert not (artifact.directory / "failure.json").exists()


def stat_mode(path: Path) -> int:
    return path.stat().st_mode & 0o777


def test_remote_torch_attestation_requires_exact_torch_only_authority() -> None:
    properties = SimpleNamespace(name=phase_b.REMOTE_TORCH_AUTHORITY["name"], major=12, minor=0,
                                 total_memory=phase_b.REMOTE_TORCH_AUTHORITY["total_memory_bytes"])
    fake_torch = SimpleNamespace(
        __version__="2.13.0+cu130",
        version=SimpleNamespace(cuda="13.0"),
        cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 1, get_device_properties=lambda index: properties),
        backends=SimpleNamespace(cudnn=SimpleNamespace(version=lambda: 92000)),
    )
    attestation = phase_b.attest_remote_torch_only(fake_torch, nvml_status="UNAVAILABLE_DRIVER_LIBRARY_MISMATCH")
    assert dict(attestation.payload) == phase_b.REMOTE_TORCH_AUTHORITY
    properties.total_memory -= 1
    with pytest.raises(phase_b.PhaseBError, match="authority drift"):
        phase_b.attest_remote_torch_only(fake_torch, nvml_status="UNAVAILABLE_DRIVER_LIBRARY_MISMATCH")


def test_source_input_rejects_rate_rounding_and_metadata_only_strict27_audit_stays_source_only() -> None:
    item = _make_input("source-00")
    with pytest.raises(source_adapter.SourceAdapterError, match="count semantics"):
        source_adapter.PosteriorSessionInput(
            session_id=item.session_id, raw_m30_t4=item.raw_m30_t4, counts_m30=item.counts_m30.to(torch.float64) + 0.2,
            exposure_m30=item.exposure_m30, theta_m30=item.theta_m30, prefix_row_ids=item.prefix_row_ids,
            source_path_sha256=item.source_path_sha256, unit_order_sha256=item.unit_order_sha256,
            raw_t4_row_order_proof=item.raw_t4_row_order_proof,
        )
    authority = source_adapter.verify_strict27_source_metadata(ROOT)
    roster = tuple(authority["roster"])
    rows = authority["source_lineage"]["rows_by_session"]
    assert len(roster) == 27 and len(set(roster)) == 27
    assert set(rows) == set(roster)
    assert authority["manifest_sha256"]
    assert authority["behavior_semantic_sha256"]


def test_external_source_root_capability_is_separate_from_stage_and_rejects_lineage_path_substitution() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        parent = Path(temporary)
        stage = parent / "fresh-stage"
        source = parent / "external-source"
        stage.mkdir()
        source.mkdir()
        capability = phase_b._issue_source_data_root_capability(
            source_data_root=source,
            expected_canonical_root=source,
        )
        session = "sub-C_ses-CO-synthetic"
        row = {
            "feature_version": 1,
            "session": session,
            "path": str(source / f"{session}_behavior+ecephys.nwb"),
            "sha256": SHA_A,
            "size_bytes": 1,
            "unit_count": 1,
        }
        assert capability.validate() == source
        phase_b.validate_stage_source_separation(stage_root=stage, source_data=capability)
        assert capability.session_path(session=session, lineage_row=row) == source / f"{session}_behavior+ecephys.nwb"
        assert stage != source
        forged = dict(row)
        forged["path"] = str(stage / f"{session}_behavior+ecephys.nwb")
        with pytest.raises(phase_b.PhaseBError, match="external source-data root"):
            capability.session_path(session=session, lineage_row=forged)
        nested_stage = source.parent
        with pytest.raises(phase_b.PhaseBError, match="must not contain"):
            phase_b.validate_stage_source_separation(stage_root=nested_stage, source_data=capability)

    # The real metadata-only loader binds staged authority bytes to the
    # external root without opening any NWB.  Stage and source roots are not
    # the same path, and the returned row cannot be rebound to the stage.
    capability = phase_b._issue_source_data_root_capability(
        source_data_root=Path(phase_b.CANONICAL_SOURCE_DATA_ROOT),
    )
    authority = source_adapter._load_strict27_stage_authority(ROOT, source_data=capability)
    payload = authority.payload()
    session = payload["roster"][0]
    row = dict(payload["source_lineage"]["rows_by_session"][session])
    assert authority.stage_root != capability.validate()
    row["path"] = str(ROOT / f"{session}_behavior+ecephys.nwb")
    with pytest.raises(phase_b.PhaseBError, match="external source-data root"):
        capability.session_path(session=session, lineage_row=row)


def test_failure_receipts_bind_launch_source_authority_and_actual_typed_progress() -> None:
    identity = _identity()

    class _ProgressBackend(_MockBackend):
        def __init__(self, mode: str) -> None:
            super().__init__()
            self.mode = mode

        def prepare(self, spec: phase_b.SourceSmokeSpec, identity: phase_b.RunIdentity) -> Any:
            if self.mode == "prepare_after_source":
                raise phase_b.SourceSmokeExecutionError(
                    stage="prepare",
                    progress=phase_b.SmokeExecutionProgress(source_opened=True),
                    cause=RuntimeError("synthetic source-open prepare failure"),
                )
            if self.mode == "prepare_after_cuda":
                raise phase_b.SourceSmokeExecutionError(
                    stage="prepare",
                    progress=phase_b.SmokeExecutionProgress(source_opened=True, remote_initialized=True),
                    cause=RuntimeError("synthetic CUDA prepare failure"),
                )
            return SimpleNamespace(progress=phase_b.SmokeExecutionProgress(source_opened=True, remote_initialized=True))

        def run_steps(self, runtime: Any, spec: phase_b.SourceSmokeSpec) -> phase_b.SmokeStepSummary:
            raise phase_b.SourceSmokeExecutionError(
                stage="steps",
                progress=phase_b.SmokeExecutionProgress(
                    source_opened=True, remote_initialized=True, optimizer_steps_completed=73,
                ),
                cause=RuntimeError("synthetic step-73 failure"),
            )

    for mode, expected_source, expected_cuda in (
        ("prepare_after_source", True, False),
        ("prepare_after_cuda", True, True),
    ):
        with tempfile.TemporaryDirectory() as temporary:
            artifact = phase_b.reserve_source_smoke_root(Path(temporary), relative="posterior_smoke")
            with pytest.raises(phase_b.SourceSmokeExecutionError):
                phase_b.run_source_smoke_lifecycle(backend=_ProgressBackend(mode), artifact=artifact, identity=identity)
            failure = artifact.reload_json("failure.json")
            assert failure["stage"] == "prepare"
            assert failure["source_opened"] is expected_source
            assert failure["remote_initialized"] is expected_cuda
            assert failure["optimizer_steps_completed"] == 0
            assert isinstance(failure["attempt_sha256"], str) and isinstance(failure["launch_sha256"], str)
            assert failure["source_authority_sha256"] is None

    with tempfile.TemporaryDirectory() as temporary:
        artifact = phase_b.reserve_source_smoke_root(Path(temporary), relative="posterior_smoke")
        with pytest.raises(phase_b.SourceSmokeExecutionError):
            phase_b.run_source_smoke_lifecycle(backend=_ProgressBackend("step73"), artifact=artifact, identity=identity)
        failure = artifact.reload_json("failure.json")
        source_authority = artifact.reload_json("source_authority.json")
        source_sha = phase_b.sha256_bytes(phase_b.canonical_json_bytes(source_authority))
        assert failure["stage"] == "steps"
        assert failure["source_opened"] is True and failure["remote_initialized"] is True
        assert failure["optimizer_steps_completed"] == 73
        assert failure["source_authority_sha256"] == source_sha
        assert isinstance(failure["launch_sha256"], str)
        assert not (artifact.directory / "terminal.json").exists()


def test_live_stage_closure_is_checked_before_attempt_and_staging_plan_excludes_nwbs() -> None:
    # A syntactically valid but stale closure is rejected before any output in
    # the injected artifact root is written.
    source_identity = source_adapter.build_target_free_source_identity(ROOT)
    stale = _synthetic_closure()
    stale_identity = phase_b.RunIdentity(
        source_authority=source_identity.source_authority,
        closure=stale,
        remote_device=source_identity.remote_device,
    )
    with tempfile.TemporaryDirectory() as temporary:
        artifact = phase_b.reserve_source_smoke_root(Path(temporary), relative="posterior_smoke")
        with pytest.raises(phase_b.PhaseBError, match="identity/live Phase-B closure drift before attempt"):
            phase_b.run_source_smoke_lifecycle(
                backend=_MockBackend(source_metadata=source_adapter.verify_strict27_source_metadata(ROOT)),
                artifact=artifact,
                identity=stale_identity,
                stage_root=ROOT,
            )
        assert not (artifact.directory / "attempt.json").exists()

    capability = phase_b._issue_source_data_root_capability(
        source_data_root=Path(phase_b.CANONICAL_SOURCE_DATA_ROOT),
    )
    plan = phase_b.remote_staging_plan(
        closure=phase_b.phase_b_closure(ROOT), source_authority_sha256=SHA_A, source_data=capability,
    )
    assets = plan["immutable_authority_assets"]
    assert plan["external_source_data_root"] == capability.payload()
    assert plan["nwb_assets_in_stage"] is False
    assert all(not str(asset["relative_path"]).endswith(".nwb") for asset in assets)
    assert phase_b.STRICT27_MANIFEST_RELATIVE in {asset["relative_path"] for asset in assets}
    stage_paths = [item["relative_path"] for item in plan["stage_files"]]
    assert stage_paths[:len(phase_b.PHASE_B_CLOSURE_PATHS)] == list(phase_b.PHASE_B_CLOSURE_PATHS)
    assert {f"{relative}.sha256" for relative, _sha, mode in phase_b.SOURCE_AUTHORITY_ASSET_SPECS if mode == 0o444}.issubset(stage_paths)
    assert all(not str(item["relative_path"]).endswith(".nwb") for item in plan["stage_files"])


def test_source_authority_semantic_reconstruction_rejects_forged_normalizer_bank_input_and_policy() -> None:
    identity = _identity()
    authority = dict(_MockBackend().source_authority(SimpleNamespace(), identity))
    authority["launch_sha256"] = SHA_A
    authority["closure"] = identity.closure
    phase_b.validate_source_authority_for_identity(authority, identity=identity, launch_sha256=SHA_A)

    forged_normalizer = copy.deepcopy(authority)
    forged_normalizer["posterior_normalizer"]["body_sha256"] = SHA_A
    with pytest.raises(phase_b.PhaseBError, match="normalizer body SHA"):
        phase_b.validate_source_authority_for_identity(forged_normalizer, identity=identity, launch_sha256=SHA_A)

    forged_bank = copy.deepcopy(authority)
    session = identity.source_authority["roster"][0]
    forged_bank["posterior_bank"]["posteriors"][session]["4"]["counts_sha256"] = SHA_B
    with pytest.raises(phase_b.PhaseBError, match="direct-evidence cross-binding"):
        phase_b.validate_source_authority_for_identity(forged_bank, identity=identity, launch_sha256=SHA_A)

    forged_prior_raw = copy.deepcopy(authority)
    forged_prior_raw["source_raw_m30_t4_sha256"] = SHA_A
    with pytest.raises(phase_b.PhaseBError, match="prior/source raw T4 cross-binding"):
        phase_b.validate_source_authority_for_identity(forged_prior_raw, identity=identity, launch_sha256=SHA_A)

    forged_schedule = copy.deepcopy(authority)
    forged_schedule["posterior_bank"]["schedule_sha256"] = SHA_A
    with pytest.raises(phase_b.PhaseBError, match="schedule"):
        phase_b.validate_source_authority_for_identity(forged_schedule, identity=identity, launch_sha256=SHA_A)

    forged_input = copy.deepcopy(authority)
    forged_input["posterior_inputs"][session]["raw_t4_row_order_proof"]["channel_ids_are_exact_arange"] = False
    with pytest.raises(phase_b.PhaseBError, match="unit-order proof"):
        phase_b.validate_source_authority_for_identity(forged_input, identity=identity, launch_sha256=SHA_A)

    forged_optimizer = copy.deepcopy(authority)
    forged_optimizer["optimizer"]["lr_constructor"] = 2e-4
    with pytest.raises(phase_b.PhaseBError, match="optimizer literal"):
        phase_b.validate_source_authority_for_identity(forged_optimizer, identity=identity, launch_sha256=SHA_A)
    forged_policy = copy.deepcopy(authority)
    forged_policy["execution_policy"]["torch_compile"] = True
    with pytest.raises(phase_b.PhaseBError, match="execution-policy literal"):
        phase_b.validate_source_authority_for_identity(forged_policy, identity=identity, launch_sha256=SHA_A)


def test_raw_t4_row_order_proof_requires_metadata_and_exact_datamodule_channel_axis(monkeypatch: pytest.MonkeyPatch) -> None:
    """Physical parser proof is structural; this synthetic seam opens no NWB."""
    fake_package = types.ModuleType("mc_maze")
    fake_package.__path__ = []  # type: ignore[attr-defined]
    fake_features = types.ModuleType("mc_maze.unit_side_features")
    raw = np.zeros((3, 4), dtype=np.float64)
    fake_features.compute_unit_side_features_uncached = lambda *args, **kwargs: (
        raw, SimpleNamespace(feature_group="t4", feature_version=1, pool_size=30),
    )
    monkeypatch.setitem(sys.modules, "mc_maze", fake_package)
    monkeypatch.setitem(sys.modules, "mc_maze.unit_side_features", fake_features)
    values, unit_digest, proof = source_adapter._raw_m30_t4_and_unit_order(
        path=Path("/synthetic/no-nwb-open"), session="source-00", expected_units=3,
        record_source_unit_count=3, record_channel_ids=np.arange(3, dtype=np.int64),
    )
    assert values.shape == (3, 4)
    assert proof["unit_order_sha256"] == unit_digest
    assert proof["channel_ids_are_exact_arange"] is True
    with pytest.raises(source_adapter.SourceAdapterError, match="source-prior substrate"):
        source_adapter._raw_m30_t4_and_unit_order(
            path=Path("/synthetic/no-nwb-open"), session="source-00", expected_units=3,
            record_source_unit_count=3, record_channel_ids=np.asarray([1, 0, 2], dtype=np.int64),
        )


def test_fresh_stage_with_only_explicit_closure_imports_and_builds_cell_d_cpu_without_workspace_fallback() -> None:
    """The remote stage needs no historical TFSR/tfpd_lane package imports.

    Copy only the descriptor-bound body files into a fresh tree, launch a new
    interpreter with no workspace PYTHONPATH, and build/forward the exact
    Cell-D graph through the route-local loader.  No data artifact is copied.
    """
    closure = phase_b.phase_b_closure(ROOT)
    source_data = phase_b._issue_source_data_root_capability(
        source_data_root=Path(phase_b.CANONICAL_SOURCE_DATA_ROOT),
    )
    plan = phase_b.remote_staging_plan(
        closure=closure, source_authority_sha256=SHA_A, source_data=source_data,
    )
    with tempfile.TemporaryDirectory() as temporary:
        stage = Path(temporary) / "fresh-stage"
        for entry in plan["stage_files"]:
            relative = entry["relative_path"]
            source = ROOT / relative
            destination = stage / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            if entry.get("mode") is not None:
                os.chmod(destination, int(entry["mode"]))
        environment = {
            **os.environ,
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "CUDA_VISIBLE_DEVICES": "",
            "PYTHONPATH": str(stage / "tfpd_exploration"),
        }
        code = """
import json
from pathlib import Path
import sys
import torch
from src.posterior_carrier_v1 import phase_b, source_adapter
root = Path.cwd()
closure = phase_b.phase_b_closure(root)
arm, pop = source_adapter.load_stage_runtime_helpers(root, closure=closure)
source_data = phase_b._issue_source_data_root_capability(source_data_root=Path(phase_b.CANONICAL_SOURCE_DATA_ROOT))
authority = source_adapter._load_strict27_stage_authority(root, source_data=source_data)
model = pop.build_population_robustness_model(seed=42, cell='D').eval()
with torch.no_grad():
    prediction, identity = model(torch.zeros(1, 50, 2), calib_trials=torch.zeros(1, 30, 100, 2), side_features=torch.zeros(1, 2, 4))
paths = [str(Path(module.__file__).resolve()) for module in (phase_b, source_adapter, arm, pop)]
print(json.dumps({'shape': list(prediction.shape), 'identity_shape': list(identity.shape), 'paths': paths, 'closure': closure['closure_sha256'], 'roster': len(authority.payload()['roster'])}))
"""
        completed = subprocess.run(
            [sys.executable, "-c", code], cwd=stage, env=environment,
            check=True, text=True, capture_output=True,
        )
        payload = json.loads(completed.stdout)
        assert payload["shape"] == [1, 50, 2]
        assert payload["identity_shape"] == [1, 2, 50]
        assert payload["roster"] == 27
        assert payload["closure"] == closure["closure_sha256"]
        assert all(path.startswith(str(stage)) for path in payload["paths"])
