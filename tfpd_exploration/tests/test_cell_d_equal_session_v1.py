"""Focused CPU/no-data tests for the additive equal-session Cell-D route."""
from __future__ import annotations

import copy
import hashlib
import io
import importlib.util
import inspect
import json
import os
import random
import stat
import subprocess
import sys
import tempfile
from unittest.mock import patch
from pathlib import Path

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "tfpd_exploration/src/cell_d_equal_session_v1.py"
CLI_PATH = ROOT / "tfpd_exploration/scripts/run_cell_d_equal_session_seed42.py"


def _load_route():
    name = "_cell_d_equal_session_test_module"
    sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location(name, MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


route = _load_route()


def _small_spec():
    return route.ScheduleSpec(
        cell="SYNTH_EQUAL_SESSION",
        seed=7,
        roster_size=3,
        batch_size=4,
        base_batches_per_session=3,
        extras_per_epoch=1,
        epochs=4,
    )


def _small_inventory():
    roster = ("s0", "s1", "s2")
    # Two full B4 batches per session; every session must therefore start a
    # second deterministic cycle under its per-epoch quota of 3 or 4.
    starts = {name: tuple(range(8)) for name in roster}
    return route.inventory_from_window_counts(roster, starts)


def _small_plan():
    spec = _small_spec()
    inventory = _small_inventory()
    return spec, inventory, route.build_plan_evidence(spec=spec, inventory=inventory)


def test_public_arithmetic_and_rotation_are_frozen() -> None:
    spec = route.PUBLIC_SPEC
    assert spec.steps_per_epoch == 33_925
    assert spec.total_steps == 1_628_400
    assert route.extra_session_indices(spec, 0) == tuple(range(13))
    assert route.extra_session_indices(spec, 1) == tuple(range(13, 26))
    assert route.extra_session_indices(spec, 2) == tuple(list(range(26, 27)) + list(range(12)))
    roster = tuple(f"r{index}" for index in range(27))
    counts = route.full_run_exposure(spec, roster)
    assert sorted(counts.values()) == [60_311] * 24 + [60_312] * 3
    assert sum(counts.values()) == 1_628_400
    assert max(counts.values()) - min(counts.values()) == 1


def test_sha256_local_seed_domains_are_exact_and_not_python_hash() -> None:
    spec = route.PUBLIC_SPEC
    expected = int(hashlib.sha256(b"CELL_D_EQUAL_SESSION_SEED42|global-order|42|0").hexdigest(), 16)
    assert route.local_rng_seed(spec, domain="global-order", epoch=0) == expected
    assert route.local_rng_seed(spec, domain="session-cycle", epoch=0, session="session-x", cycle=3) == int(
        hashlib.sha256(
            b"CELL_D_EQUAL_SESSION_SEED42|session-cycle|42|0|session-x|3"
        ).hexdigest(),
        16,
    )
    with pytest.raises(route.ContractError):
        route.local_rng_seed(spec, domain="unsupported", epoch=0)
    source = inspect.getsource(route)
    assert "hash(" not in source.replace("hashlib", "")


def test_source_inventory_has_one_contiguous_global_dataset_index_map() -> None:
    inventory = route.inventory_from_window_counts(
        ("a", "b"), {"a": (11, 13, 13), "b": (2, 4, 6, 8)}
    )
    assert inventory.sessions[0].dataset_indices == (0, 1, 2)
    assert inventory.sessions[1].dataset_indices == (3, 4, 5, 6)
    # Duplicate wall-clock starts are kept as distinct dataset rows, matching
    # the existing datamodule’s Dataset index semantics.
    assert inventory.sessions[0].eligible_windows == 3
    with pytest.raises(route.ContractError):
        route.SessionInventory("bad", (0, 2), "0" * 64)


def test_explicit_epoch_schedule_is_deterministic_single_session_and_cycles_only_after_exhaustion() -> None:
    spec = _small_spec()
    inventory = _small_inventory()
    schedule = route.EpochSchedule(spec=spec, inventory=inventory, epoch=0)
    replay = route.EpochSchedule(spec=spec, inventory=inventory, epoch=0)
    batches = list(schedule.iter_batches())
    assert [item.payload() for item in batches] == [item.payload() for item in replay.iter_batches()]
    assert len(batches) == spec.steps_per_epoch == 10
    by_session: dict[str, list[object]] = {name: [] for name in inventory.roster}
    for batch in batches:
        by_session[batch.session].append(batch)
        assert len(batch.dataset_indices) == 4
        assert len(set(batch.dataset_indices)) == 4
    for session, session_batches in by_session.items():
        assert len(session_batches) == 4 if session == "s0" else 3
        first_cycle = [item for item in session_batches if item.cycle == 0]
        assert len(first_cycle) == 2
        assert len({index for item in first_cycle for index in item.dataset_indices}) == 8
        assert all(item.cycle >= 1 for item in session_batches[2:])
    sampler = route.ExplicitEpochBatchSampler(schedule)
    assert list(sampler) == list(sampler)
    assert len(sampler) == len(schedule)


def test_plan_compact_evidence_ledger_and_tamper_rejection() -> None:
    spec, inventory, plan = _small_plan()
    assert plan["plan_retains_batch_lists"] is False
    assert len(plan["epochs"]) == 4
    assert all("batches" not in item for item in plan["epochs"])
    assert len({item["schedule_sha256"] for item in plan["epochs"]}) == 4
    ledger = route.EpochLedger(plan)
    for epoch in range(spec.epochs):
        ledger.claim(route.EpochSchedule(spec=spec, inventory=inventory, epoch=epoch))
    ledger.final_assert_exact()
    again = route.EpochLedger(plan)
    again.claim(route.EpochSchedule(spec=spec, inventory=inventory, epoch=0))
    with pytest.raises(route.ContractError):
        again.claim(route.EpochSchedule(spec=spec, inventory=inventory, epoch=0))
    with pytest.raises(route.ContractError):
        route.EpochLedger({**plan, "epochs": plan["epochs"][:-1]}).final_assert_exact()


def test_one_iterator_sampler_binds_delivery_order_not_prefetch_tail() -> None:
    """Evidence remains paired to delivered batches even after prefetch."""
    spec, inventory = _small_spec(), _small_inventory()
    schedule = route.EpochSchedule(spec=spec, inventory=inventory, epoch=0)
    expected = list(schedule.iter_batches())
    sampler = route._OneIteratorEpochBatchSampler(schedule)
    emitted = list(iter(sampler))
    assert len(emitted) == len(expected)
    # Materializing the producer simulates a worker queue that has advanced
    # beyond the first delivered batch.  The observer must consume FIFO, not
    # use the producer's last scheduled batch.
    assert sampler.consume_observed_batch().payload() == expected[0].payload()
    for want in expected[1:]:
        assert sampler.consume_observed_batch().payload() == want.payload()
    sampler.assert_complete()


def test_schedule_construction_preserves_python_numpy_and_torch_cpu_rng() -> None:
    spec, inventory = _small_spec(), _small_inventory()
    saved_python = random.getstate()
    saved_numpy = np.random.get_state()
    saved_torch = torch.get_rng_state()
    try:
        random.seed(991)
        np.random.seed(992)
        torch.manual_seed(993)
        before = route.rng_snapshot(numpy_module=np, torch_module=torch)
        plan = route.build_plan_evidence(
            spec=spec,
            inventory=inventory,
            snapshot_rng=lambda: route.rng_snapshot(numpy_module=np, torch_module=torch),
        )
        after = route.rng_snapshot(numpy_module=np, torch_module=torch)
        assert plan["global_rng_unchanged"] is True
        assert before == after
    finally:
        random.setstate(saved_python)
        np.random.set_state(saved_numpy)
        torch.set_rng_state(saved_torch)


def test_original_and_equal_exposure_report_keeps_floor_remainder_facts() -> None:
    spec, inventory, plan = _small_plan()
    rows = route.original_vs_equal_exposure(spec, inventory, plan)
    assert len(rows) == 3
    assert all(row["remainder_windows_discarded_per_epoch"] == 0 for row in rows)
    assert sum(row["equal_batches_full_run"] for row in rows) == spec.total_steps
    assert max(row["equal_batches_full_run"] for row in rows) - min(
        row["equal_batches_full_run"] for row in rows
    ) <= 1


def test_valid_start_adapter_exactly_mirrors_window_arithmetic() -> None:
    trials = [
        {"start": 10.0, "stop": 60.0},
        {"start": 100, "stop": 152},
    ]
    assert route.valid_starts_from_rewarded_trials(trials) == tuple(range(10, 11)) + tuple(range(100, 103))
    with pytest.raises(route.ContractError):
        route.valid_starts_from_rewarded_trials([{"start": 0, "stop": 49}])
    with pytest.raises(route.ContractError):
        route.valid_starts_from_rewarded_trials([{"start": 0.5, "stop": 60}])


def test_real_sealed_metadata_authority_validates_and_drift_fails() -> None:
    authority = route.load_sealed_cell_d_contract(ROOT)
    assert authority["trainable_parameters"] == 3_510_842
    assert authority["swa_epochs"] == [44, 45, 46, 47]
    assert authority["predecessor_wall_clock_seconds"] == 18_512
    launch = json.loads((ROOT / route.SEALED_CELL_D_LAUNCH_RELATIVE).read_text())
    terminal = json.loads((ROOT / route.SEALED_CELL_D_TERMINAL_RELATIVE).read_text())
    broken = copy.deepcopy(terminal)
    broken["head_and_dropout_config"]["num_heads"] = 64
    with pytest.raises(route.ContractError):
        route.validate_sealed_cell_d_contract(launch, broken)
    broken = copy.deepcopy(terminal)
    broken["diagnostics_per_epoch"][47]["optimizer_steps_total"] -= 1
    with pytest.raises(route.ContractError):
        route.validate_sealed_cell_d_contract(launch, broken)


def test_failed_smoke_v1_lineage_is_immutable_and_successor_smoke_is_v2() -> None:
    """The honest v1 pre-CUDA failure is predecessor evidence, never a root to reuse."""
    lineage = route.load_failed_smoke_v1_lineage(ROOT)
    assert lineage == route._failed_smoke_v1_lineage_payload()
    assert route.FAILED_SMOKE_V1_ROOT_RELATIVE.endswith("_smoke_v1")
    assert route.SOURCE_SMOKE_SPEC.root_relative.endswith("_smoke_v2")
    assert route.SOURCE_SMOKE_SPEC.root_relative != route.FAILED_SMOKE_V1_ROOT_RELATIVE
    assert lineage["failure"]["stage"] == "backend_prepare"
    assert lineage["failure"]["cuda_initialized"] is False
    assert lineage["failure"]["optimizer_steps_completed"] == 0

    # The root-review identity binds that failed attempt before any capability
    # or fresh-root request can be considered.
    identity = route._build_run_identity(ROOT)
    route._validate_run_identity(identity, route.SOURCE_SMOKE_SPEC)
    assert identity.failed_smoke_v1_lineage == lineage
    fresh = route.assert_fresh_candidate_roots(ROOT)
    assert fresh["roots"][route.SOURCE_SMOKE_SPEC.root_relative] == "absent"

    attempt = json.loads((ROOT / route.FAILED_SMOKE_V1_ATTEMPT_RELATIVE).read_text())
    launch = json.loads((ROOT / route.FAILED_SMOKE_V1_LAUNCH_RELATIVE).read_text())
    failure = json.loads((ROOT / route.FAILED_SMOKE_V1_FAILURE_RELATIVE).read_text())
    forged_failure = copy.deepcopy(failure)
    forged_failure["access_disclosure"]["cuda_initialized"] = True
    with pytest.raises(route.ContractError, match="smoke-v1 failure"):
        route.validate_failed_smoke_v1_lineage(attempt, launch, forged_failure)
    forged_launch = copy.deepcopy(launch)
    forged_launch["attempt_sha256"] = "0" * 64
    with pytest.raises(route.ContractError, match="smoke-v1 launch"):
        route.validate_failed_smoke_v1_lineage(attempt, forged_launch, failure)


def test_failed_smoke_v1_loader_binds_all_body_sidecar_pairs() -> None:
    """A successor cannot silently replace either a v1 receipt body or sidecar."""
    relatives = (
        route.FAILED_SMOKE_V1_ATTEMPT_RELATIVE,
        route.FAILED_SMOKE_V1_LAUNCH_RELATIVE,
        route.FAILED_SMOKE_V1_FAILURE_RELATIVE,
    )
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for relative in relatives:
            for suffix in ("", ".sha256"):
                source = ROOT / f"{relative}{suffix}"
                target = root / f"{relative}{suffix}"
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source.read_bytes())
                os.chmod(target, 0o444)
        assert route.load_failed_smoke_v1_lineage(root) == route._failed_smoke_v1_lineage_payload()

        sidecar = root / f"{route.FAILED_SMOKE_V1_FAILURE_RELATIVE}.sha256"
        os.chmod(sidecar, 0o644)
        sidecar.write_text(f"{'0' * 64}  failure.json\n")
        os.chmod(sidecar, 0o444)
        with pytest.raises(route.ContractError, match="sidecar"):
            route.load_failed_smoke_v1_lineage(root)

        # Repair the sidecar, then prove a recomputed sidecar cannot conceal a
        # body change because the immutable literal body SHA is also checked.
        failure_body = root / route.FAILED_SMOKE_V1_FAILURE_RELATIVE
        os.chmod(failure_body, 0o644)
        failure_body.write_bytes(b"{}\n")
        os.chmod(failure_body, 0o444)
        changed = hashlib.sha256(b"{}\n").hexdigest()
        os.chmod(sidecar, 0o644)
        sidecar.write_text(f"{changed}  failure.json\n")
        os.chmod(sidecar, 0o444)
        with pytest.raises(route.ContractError, match="SHA drift"):
            route.load_failed_smoke_v1_lineage(root)


def test_real_canonical_state_uses_scoped_torchversion_allowlist_and_exact_schema() -> None:
    """The real sealed artifact loads weights-only without persistent globals."""
    from torch.nn.parameter import UninitializedParameter
    from torch.torch_version import TorchVersion

    before = tuple(torch.serialization.get_safe_globals())
    state = route._load_canonical_initial_state(ROOT, torch)
    after_route_load = tuple(torch.serialization.get_safe_globals())
    assert after_route_load == before
    assert "decoder.fc_id_in.0.weight" in state
    assert state["decoder.fc_id_in.0.weight"].__class__ is UninitializedParameter

    body = route._read_immutable_binary(
        ROOT,
        route.CANONICAL_INITIAL_STATE_RELATIVE,
        expected_sha256=route.CANONICAL_INITIAL_STATE_SHA256,
    )
    with torch.serialization.safe_globals([UninitializedParameter, TorchVersion]):
        payload = torch.load(io.BytesIO(body), map_location="cpu", weights_only=True)
    assert tuple(torch.serialization.get_safe_globals()) == before
    route._validate_canonical_initial_state_payload(payload, torch_version_type=TorchVersion)
    assert type(payload["torch"]) is TorchVersion
    assert str(payload["torch"]) == route.CANONICAL_INITIAL_STATE_TORCH_VERSION

    forged = dict(payload)
    forged["torch"] = "2.5.1.post303"
    with pytest.raises(route.ContractError, match="canonical initial artifact"):
        route._validate_canonical_initial_state_payload(forged, torch_version_type=TorchVersion)
    forged = dict(payload)
    forged["torch"] = TorchVersion("2.5.1.post302")
    with pytest.raises(route.ContractError, match="canonical initial artifact"):
        route._validate_canonical_initial_state_payload(forged, torch_version_type=TorchVersion)
    forged = dict(payload)
    forged["kind"] = "forged-canonical-schema"
    with pytest.raises(route.ContractError, match="canonical initial artifact"):
        route._validate_canonical_initial_state_payload(forged, torch_version_type=TorchVersion)
    forged = dict(payload)
    forged["state_sha256"] = "0" * 64
    with pytest.raises(route.ContractError, match="canonical initial artifact"):
        route._validate_canonical_initial_state_payload(forged, torch_version_type=TorchVersion)


def test_physical_prepare_checks_canonical_state_before_source_open() -> None:
    """Keep the CPU-only artifact failure path ahead of the source adapter."""
    source = inspect.getsource(route.PhysicalEqualSessionBackend.prepare)
    canonical = source.index("canonical_state = _load_canonical_initial_state")
    source_open = source.index("flags.source_train_opened = True")
    source_builder = source.index("_build_strict_train_only_datamodule(")
    assert canonical < source_open < source_builder


def test_fixed_receipt_reader_binds_body_and_sidecar() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        body = b'{"ok":true}\n'
        (root / "x.json").write_bytes(body)
        digest = hashlib.sha256(body).hexdigest()
        (root / "x.json.sha256").write_text(f"{digest}  x.json\n")
        os.chmod(root / "x.json", 0o444)
        os.chmod(root / "x.json.sha256", 0o444)
        assert route._load_immutable_json(root, "x.json", expected_sha256=digest, mode=0o444) == {"ok": True}
        os.chmod(root / "x.json.sha256", 0o644)
        (root / "x.json.sha256").write_text(f"{'0' * 64}  x.json\n")
        os.chmod(root / "x.json.sha256", 0o444)
        with pytest.raises(route.ContractError):
            route._load_immutable_json(root, "x.json", expected_sha256=digest, mode=0o444)


def test_manifest_roster_is_strict_and_source_guard_rejects_nontrain_session() -> None:
    roster = route.load_strict_source_roster(ROOT)
    assert len(roster) == 27
    assert roster[0] == "sub-C_ses-CO-20131003"
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source_directory = root / route.SOURCE_DATA_RELATIVE
        source_directory.mkdir(parents=True)
        source = source_directory / "sub-C_ses-CO-20131003_behavior+ecephys.nwb"
        source.write_bytes(b"synthetic-source")
        allowed = ("sub-C_ses-CO-20131003",)
        assert route._direct_source_file(root, allowed[0], allowed_train_roster=allowed) == source
        with pytest.raises(route.ContractError):
            route._direct_source_file(root, "sub-C_ses-CO-20151103", allowed_train_roster=allowed)


def test_fresh_roots_environment_and_execution_are_fail_closed() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        assert route.assert_fresh_candidate_roots(root)["created"] is False
        assert route.validate_future_gpu0_environment(
            {"CUDA_VISIBLE_DEVICES": "0", "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
        )["logical_device"] == "cuda:0"
        with pytest.raises(route.ContractError):
            route.validate_future_gpu0_environment(
                {"CUDA_VISIBLE_DEVICES": "1", "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
            )
        with pytest.raises(route.ContractError, match="not issued"):
            route.RootReviewedExecutionCapability("a" * 64, "b" * 64, route.CELL, "full_train")
        with pytest.raises(route.ContractError, match="lacks an in-process root capability"):
            route.execute_reviewed_training(
                root=ROOT,
                execute_flag=True,
                acknowledgement_flag=True,
                capability=None,
            )
        assert not (root / route.RESULT_ROOT_RELATIVE).exists()
        assert not (root / route.SMOKE_RESULT_ROOT_RELATIVE).exists()


def test_dry_cli_never_creates_output_or_imports_route_dependencies() -> None:
    before = [(ROOT / relative).exists() for relative in (route.RESULT_ROOT_RELATIVE, route.SMOKE_RESULT_ROOT_RELATIVE)]
    environment = dict(os.environ)
    environment["PYTHONNOUSERSITE"] = "1"
    completed = subprocess.run(
        [sys.executable, str(CLI_PATH)],
        cwd=ROOT,
        env=environment,
        check=True,
        text=True,
        capture_output=True,
    )
    plan = json.loads(completed.stdout)
    assert plan["status"] == "DRY_NO_DATA_NO_WRITE"
    assert [(ROOT / relative).exists() for relative in (
        route.RESULT_ROOT_RELATIVE,
        route.SMOKE_RESULT_ROOT_RELATIVE,
    )] == before
    denied = subprocess.run(
        [sys.executable, str(CLI_PATH), "--execute", "--i-have-root-reviewed-equal-session-authorization"],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
    )
    assert denied.returncode != 0
    assert "lacks an in-process root capability" in (denied.stderr + denied.stdout)


def _tiny_lifecycle_spec():
    return route.LifecycleSpec(
        kind="synthetic",
        root_relative="synthetic_equal_session",
        epochs=2,
        steps_per_epoch=2,
        checkpoint_epochs=(0, 1),
        throughput_steps=2,
    )


def test_mock_lifecycle_publishes_reloadable_full_terminal_topology() -> None:
    spec = _tiny_lifecycle_spec()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        artifact = route.reserve_artifact_root(root, spec.root_relative, spec.topology)
        backend = route.DeterministicMockBackend()
        terminal = route.run_lifecycle(
            spec=spec,
            identity_factory=route.synthetic_identity,
            backend=backend,
            artifact=artifact,
        )
        assert terminal["status"] == "CELL_TRAINING_TERMINAL__UNSCORED"
        assert backend.closed is True
        assert backend.iterator_count == 2
        assert backend.measurement_sync_count == 1
        assert not artifact.has_name("failure.json")
        expected = {
            "attempt.json", "launch.json", "source_authority.json", "epoch000.json", "epoch001.json",
            "epoch000.pt", "epoch001.pt", "throughput2.json", "swa_final4.pt", "terminal.json",
        }
        for name in expected:
            assert artifact.has_name(name)
            body = artifact.reload_pair(name)
            assert body
            assert stat.S_IMODE((artifact.directory / name).stat().st_mode) == 0o444
            assert stat.S_IMODE((artifact.directory / f"{name}.sha256").stat().st_mode) == 0o444
        assert terminal["access_disclosure"]["target_resolved_or_opened"] is False
        assert terminal["access_disclosure"]["optimizer_steps_completed"] == 4
        assert terminal["launch_final_closure_equal"] is True
        expected_hashes = {
            name: hashlib.sha256(artifact.reload_pair(name)).hexdigest()
            for name in spec.topology
            if name not in {"terminal.json", "failure.json"} and artifact.has_name(name)
        }
        route._validate_terminal_payload(
            artifact.reload_json("terminal.json"), spec, route.synthetic_identity(),
            expected_artifact_sha256s=expected_hashes,
        )
        forged_epoch = copy.deepcopy(artifact.reload_json("epoch001.json"))
        forged_epoch["critical_gradients"]["decoder_ffn"] = False
        with pytest.raises(route.ContractError, match="critical gradient"):
            route._validate_epoch_payload(forged_epoch, spec, 1)
        tampered = copy.deepcopy(terminal)
        tampered["artifact_sha256s"]["launch.json"] = "0" * 64
        with pytest.raises(route.ContractError):
            route._validate_terminal_payload(
                tampered, spec, route.synthetic_identity(), expected_artifact_sha256s=expected_hashes,
            )


def test_public_source_authority_payload_is_bound_and_tampering_fails() -> None:
    backend = route.DeterministicMockBackend()
    identity = route.synthetic_identity()
    flags = route.LifecycleFlags(source_train_opened=True)
    runtime = backend.prepare(route.SOURCE_SMOKE_SPEC, identity, flags)
    payload = backend.source_authority(runtime, route.SOURCE_SMOKE_SPEC, identity, flags)
    # The default validator is the production boundary: its literal runtime
    # contract must reject the visibly separate no-device mock schema.
    with pytest.raises(route.ContractError, match="runtime Torch/CUDA/cuDNN"):
        route._validate_source_authority_payload(payload, route.SOURCE_SMOKE_SPEC, identity)
    route._validate_source_authority_payload(
        payload, route.SOURCE_SMOKE_SPEC, identity, allow_synthetic_runtime=True,
    )
    assert payload["schedule_plan"]["full_plan_sha256"] == route.AUDITED_FULL_PLAN_SHA256
    assert payload["schedule_plan"]["plan_retains_batch_lists"] is False
    assert len(payload["per_session_exposure"]) == 27
    assert payload["runtime_environment"]["schema"] == route.SYNTHETIC_RUNTIME_ENVIRONMENT_SCHEMA
    assert payload["runtime_environment"]["frozen_gpu_authorities"] == dict(route.FROZEN_GPU0)
    broken = copy.deepcopy(payload)
    broken["train_source_relative_paths"][0] = "val/forbidden.nwb"
    with pytest.raises(route.ContractError):
        route._validate_source_authority_payload(
            broken, route.SOURCE_SMOKE_SPEC, identity, allow_synthetic_runtime=True,
        )
    broken = copy.deepcopy(payload)
    broken["t4_authority_sha256"][broken["strict_train_roster"][0]] = "0" * 64
    with pytest.raises(route.ContractError):
        route._validate_source_authority_payload(
            broken, route.SOURCE_SMOKE_SPEC, identity, allow_synthetic_runtime=True,
        )
    broken = copy.deepcopy(payload)
    broken["runtime_environment"]["fixture_runtime"]["torch_cuda_version"] = "tampered"
    with pytest.raises(route.ContractError, match="synthetic runtime fixture"):
        route._validate_source_authority_payload(
            broken, route.SOURCE_SMOKE_SPEC, identity, allow_synthetic_runtime=True,
        )


def test_mock_source_smoke_is_exactly_one_step_and_non_authorizing() -> None:
    spec = route.SOURCE_SMOKE_SPEC
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "tfpd_exploration/results").mkdir(parents=True)
        artifact = route.reserve_artifact_root(root, spec.root_relative, spec.topology)
        backend = route.DeterministicMockBackend()
        terminal = route.run_lifecycle(
            spec=spec,
            identity_factory=route.synthetic_identity,
            backend=backend,
            artifact=artifact,
        )
        assert terminal["status"] == "SOURCE_SMOKE_COMPLETE__NON_AUTHORITATIVE"
        smoke = artifact.reload_json("smoke.json")
        assert smoke["epoch"] == 0
        assert smoke["run_spec"]["total_steps"] == 1
        assert smoke["non_authorizing"] is True
        assert set(smoke["resources"]) == {
            "cuda_memory_allocated", "cuda_memory_reserved",
            "cuda_peak_memory_allocated", "cuda_peak_memory_reserved",
            "peak_stats_reset_before_training", "rss_bytes", "throughput_windows_per_s",
        }
        assert smoke["resources"]["peak_stats_reset_before_training"] is True
        assert smoke["resources"]["cuda_peak_memory_allocated"] >= smoke["resources"]["cuda_memory_allocated"]
        assert smoke["resources"]["cuda_peak_memory_reserved"] >= smoke["resources"]["cuda_memory_reserved"]
        assert smoke["resources"]["throughput_windows_per_s"] >= 0
        authority = artifact.reload_json("source_authority.json")
        assert authority["schema"] == "cell_d_equal_session_source_authority_v1"
        assert terminal["access_disclosure"]["optimizer_steps_completed"] == 1
        assert terminal["access_disclosure"]["fixed_diagnostic_forwards"] == 0
        assert backend.measurement_sync_count == 1
        assert not (artifact.directory / "swa_final4.pt").exists()
        broken = copy.deepcopy(smoke)
        broken["schedule"]["consumed_batch_count"] = 2
        with pytest.raises(route.ContractError):
            route._validate_source_smoke_payload(broken, spec)
        broken = copy.deepcopy(smoke)
        broken["outcome"]["critical_gradients"]["decoder_ffn"] = False
        with pytest.raises(route.ContractError, match="critical gradient"):
            route._validate_source_smoke_payload(broken, spec)


@pytest.mark.parametrize(
    "fail_stage",
    ["prepare", "source_authority", "begin_epoch", "step", "end_epoch", "checkpoint", "swa"],
)
def test_mock_lifecycle_failure_is_honest_and_never_publishes_terminal(fail_stage: str) -> None:
    spec = _tiny_lifecycle_spec()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        artifact = route.reserve_artifact_root(root, spec.root_relative, spec.topology)
        backend = route.DeterministicMockBackend(fail_stage=fail_stage)
        with pytest.raises(route.ContractError, match="synthetic requested failure"):
            route.run_lifecycle(
                spec=spec,
                identity_factory=route.synthetic_identity,
                backend=backend,
                artifact=artifact,
            )
        assert backend.closed is True
        assert artifact.has_name("failure.json")
        assert not artifact.has_name("terminal.json")
        failure = artifact.reload_json("failure.json")
        assert failure["status"] == "FAILED"
        assert failure["terminal_published"] is False
        assert failure["access_disclosure"]["target_resolved_or_opened"] is False


def test_artifact_pair_corruption_and_parent_name_swap_fail_closed() -> None:
    spec = _tiny_lifecycle_spec()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        artifact = route.reserve_artifact_root(root, spec.root_relative, spec.topology)
        digest = artifact.publish_json("attempt.json", {"synthetic": True})
        assert artifact.reload_pair("attempt.json", digest)
        sidecar = artifact.directory / "attempt.json.sha256"
        os.chmod(sidecar, 0o644)
        sidecar.write_text("0" * 64 + "  attempt.json\n")
        os.chmod(sidecar, 0o444)
        with pytest.raises(route.ContractError):
            artifact.reload_pair("attempt.json", digest)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        artifact = route.reserve_artifact_root(root, spec.root_relative, spec.topology)
        moved = root / "moved"
        artifact.directory.rename(moved)
        artifact.directory.mkdir()
        with pytest.raises(route.ContractError, match="identity drift"):
            artifact.has_name("attempt.json")


def test_strict_train_only_builder_sentinel_skips_empty_val_load_split_and_nontrain_access() -> None:
    roster = tuple(f"sub-C_ses-CO-2014{index:04d}" for index in range(27))
    seen: list[str] = []
    constructed: list[dict[str, object]] = []

    class FakeDataModule:
        def __init__(self) -> None:
            self.session_files = {}
            self.session_splits = {}
            self.session_unit_counts = {"bad": 1}
            self._splits_initialized = False
            self.cache_dir = None
            self.train_dataset = None
            self.val_dataset = None
            self.test_dataset = None
            self.initialize_calls = 0
            self.load_split_calls: list[str] = []

        def _initialize_splits(self):
            # Faithful shared-control-flow fixture: setup calls the method,
            # but prebinding has already made it a no-op.  Any attempt to run
            # real split construction is an immediate failure.
            self.initialize_calls += 1
            assert self._splits_initialized is True

        def nwb_unit_count(self, _path):  # pragma: no cover - must never run
            raise AssertionError("non-train unit-count metadata read was called")

        def setup(self, stage):
            assert stage == "fit"
            self._initialize_splits()
            assert self._splits_initialized is True
            assert self.session_files["val"] == []
            assert self.session_files["test"] == []
            if self.train_dataset is None:
                self.load_split_calls.append("train")
                assert self.session_files["train"]
                self.train_dataset = object()
            # This is the real shared setup's branch shape: if val_dataset is
            # None, it would call load_split([]) and build an empty dataset.
            # The route-owned sentinel must suppress this branch completely.
            if self.val_dataset is None:  # pragma: no cover - sentinel must skip
                self.load_split_calls.append("val")
                raise AssertionError("empty val load_split was called")

    class FakeA2:
        EXPECTED_MANIFEST_SHA256 = route.MANIFEST_SHA256

    def factory(**kwargs):
        constructed.append(dict(kwargs))
        return FakeDataModule()

    def train_only_resolver(_root: Path, session: str) -> Path:
        assert session in roster
        seen.append(session)
        return Path("/synthetic") / f"{session}_behavior+ecephys.nwb"

    fake, a2, observed_roster, paths = route._construct_strict_train_only_datamodule(
        Path("/synthetic-root"),
        num_workers=4,
        datamodule_factory=factory,
        a2_module=FakeA2,
        roster=roster,
        direct_source_file=train_only_resolver,
    )
    assert seen == list(roster)
    assert observed_roster == roster
    assert a2 is FakeA2
    assert [path.name for path in paths] == [f"{session}_behavior+ecephys.nwb" for session in roster]
    assert fake.session_unit_counts == {}
    assert fake.session_splits == {"train": list(roster), "val": [], "test": []}
    assert fake.initialize_calls == 1
    assert fake.load_split_calls == ["train"]
    assert fake.val_dataset is None and fake.test_dataset is None
    assert constructed == [{
        "data_dir": str(Path("/synthetic-root").absolute() / route.SOURCE_DATA_RELATIVE),
        "task": "CO", "split_counts": (27, 6, 6), "batch_size": 32,
        "window_size": 50, "calibration_n_trials": 30, "max_trial_length": 100,
        "bin_size_ms": 20, "num_workers": 4, "pin_memory": True,
        "random_calibration": False, "seed": 42, "max_units_exclusive": 100,
        "cache_dir": None, "signal_view": "sua", "side_feature_group": "t4",
        "side_feature_pool_size": 30,
        "train_val_manifest_path": str(Path("/synthetic-root").absolute() / route.MANIFEST_RELATIVE),
    }]


def test_sealed_arm_a_source_authority_is_exact_metadata_only() -> None:
    authority = route.load_sealed_arm_a_source_authority(ROOT)
    assert authority["preflight_receipt_sha256"] == route.SEALED_ARM_A_PREFLIGHT_SHA256
    assert authority["behavior_normalizer_semantic_sha256"] == route.SEALED_BEHAVIOR_NORMALIZER_SHA256
    assert authority["t4_normalizer_semantic_sha256"] == route.SEALED_T4_NORMALIZER_SHA256
    assert len(authority["roster"]) == len(authority["t4_authority_sha256"]) == 27
    assert authority["metadata_only"] is True


def test_explicit_runtime_closure_binds_new_lifecycle_and_no_glob() -> None:
    closure = route.implementation_closure(ROOT)
    assert closure["paths"] == list(route.IMPLEMENTATION_CLOSURE)
    assert len(closure["paths"]) == len(set(closure["paths"]))
    assert all("*" not in path and "?" not in path for path in closure["paths"])
    identity = route._build_run_identity(ROOT)
    route._validate_run_identity(identity, route.FULL_TRAIN_SPEC)
    assert identity.source_schedule["plan_sha256"] == route.AUDITED_FULL_PLAN_SHA256


def test_real_cpu_cell_d_builder_matches_frozen_graph_and_preserves_lazy_topology() -> None:
    saved_python = random.getstate()
    saved_numpy = np.random.get_state()
    saved_torch = torch.get_rng_state()
    old_paths = list(sys.path)
    try:
        route._prepend_runtime_packages(ROOT)
        population = route._load_runtime_module(
            "_test_equal_session_pop_robust", ROOT / "tfpd_exploration/src/tfpd_lane/pop_robust.py"
        )
        model = population.build_population_robustness_model(seed=42, cell="D")
        count, lazy = route._lazy_safe_parameter_count(model, torch)
        assert count == 3_510_842
        assert lazy == ["decoder.fc_id_in.0.bias", "decoder.fc_id_in.0.weight"]
        assert model._pop_robust_config == {"num_heads": 2, "dynamic_dropout": True,
                                            "note": "2 heads + dynamic dropout; head count identical to arm A"}
        assert model.decoder.dynamic_dropout is True
    finally:
        random.setstate(saved_python)
        np.random.set_state(saved_numpy)
        torch.set_rng_state(saved_torch)
        sys.path[:] = old_paths


def _real_cpu_cell_d_state_payload() -> tuple[dict[str, object], str]:
    """Return a serialized-cell fixture without data, CUDA, or a checkpoint file."""
    saved_python = random.getstate()
    saved_numpy = np.random.get_state()
    saved_torch = torch.get_rng_state()
    old_paths = list(sys.path)
    try:
        route._prepend_runtime_packages(ROOT)
        population = route._load_runtime_module(
            "_test_equal_session_state_pop", ROOT / "tfpd_exploration/src/tfpd_lane/pop_robust.py"
        )
        arm_common = route._load_runtime_module(
            "_test_equal_session_state_arm", ROOT / "tfpd_exploration/src/tfpd_lane/arm_common.py"
        )
        model = population.build_population_robustness_model(seed=42, cell="D")
        return dict(model.state_dict()), arm_common.state_sha256(model)
    finally:
        random.setstate(saved_python)
        np.random.set_state(saved_numpy)
        torch.set_rng_state(saved_torch)
        sys.path[:] = old_paths


def _torch_save_payload(payload: object) -> bytes:
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    return buffer.getvalue()


def _torch_load_payload(body: bytes) -> dict[str, object]:
    from torch.nn.parameter import UninitializedParameter

    with torch.serialization.safe_globals([UninitializedParameter]):
        loaded = torch.load(io.BytesIO(body), map_location="cpu", weights_only=True)
    assert isinstance(loaded, dict)
    return loaded


def test_physical_checkpoint_and_swa_reload_recompute_tensor_state_digest() -> None:
    """A recomputed outer pair cannot conceal a mutated stored tensor."""
    state_dict, digest = _real_cpu_cell_d_state_payload()
    backend = route.PhysicalEqualSessionBackend(ROOT)
    binding = {"synthetic-binding": "state-recompute"}
    checkpoint = {
        "schema": "cell_d_equal_session_checkpoint_v1",
        "cell": route.CELL,
        "epoch": 44,
        "global_step": 45 * route.PUBLIC_SPEC.steps_per_epoch,
        "state_dict": state_dict,
        "state_dict_sha256": digest,
        "binding": binding,
    }
    checkpoint_body = _torch_save_payload(checkpoint)
    before = route.rng_snapshot(numpy_module=np, torch_module=torch)
    backend.validate_checkpoint(
        checkpoint_body,
        epoch=44,
        global_step=45 * route.PUBLIC_SPEC.steps_per_epoch,
        spec=route.FULL_TRAIN_SPEC,
        binding=binding,
    )
    assert route.rng_snapshot(numpy_module=np, torch_module=torch) == before

    proof = {
        "window_epochs": [44, 45, 46, 47],
        "component_state_sha256": {str(epoch): digest for epoch in (44, 45, 46, 47)},
        "fp64_arithmetic": True,
        "fresh_strict_load": True,
        "eval_mode": True,
        "eval_no_mask": True,
        "repeat_bitwise_equal": True,
        "state_unchanged": True,
        "prediction_shape": [4, 50, 2],
        "prediction_sha256": "a" * 64,
        "state_sha256_before_eval": digest,
        "state_sha256_after_eval": digest,
        "uninitialized_lazy_keys": list(route.CELL_D_UNINITIALIZED_LAZY_KEYS),
    }
    swa_body = _torch_save_payload({
        "schema": "cell_d_equal_session_swa_v1",
        "cell": route.CELL,
        "state_dict": state_dict,
        "state_dict_sha256": digest,
        "binding": binding,
        "proof": proof,
    })
    backend.validate_swa(swa_body, spec=route.FULL_TRAIN_SPEC, binding=binding)
    assert route.rng_snapshot(numpy_module=np, torch_module=torch) == before

    # Re-open, mutate one initialized tensor, and publish a new body+sidecar.
    # The claimed inner digest is deliberately retained, so only a strict
    # fresh-model recomputation can reject the forgery.
    tampered = _torch_load_payload(checkpoint_body)
    tampered_state = tampered["state_dict"]
    assert isinstance(tampered_state, dict)
    with torch.no_grad():
        tampered_state["decoder.fc_out.bias"][0].add_(1.0)
    tampered_checkpoint_body = _torch_save_payload(tampered)
    tampered_swa = _torch_load_payload(swa_body)
    tampered_swa_state = tampered_swa["state_dict"]
    assert isinstance(tampered_swa_state, dict)
    with torch.no_grad():
        tampered_swa_state["decoder.fc_out.bias"][0].add_(1.0)
    tampered_swa_body = _torch_save_payload(tampered_swa)
    lazy_tampered = _torch_load_payload(checkpoint_body)
    lazy_state = lazy_tampered["state_dict"]
    assert isinstance(lazy_state, dict)
    lazy_state["decoder.fc_id_in.0.weight"] = torch.zeros(1, 1)
    lazy_tampered_body = _torch_save_payload(lazy_tampered)
    with tempfile.TemporaryDirectory() as directory:
        artifact = route.reserve_artifact_root(
            Path(directory), "recomputed_pairs", ("checkpoint.pt", "swa.pt", "lazy.pt")
        )
        checkpoint_outer = artifact.publish_bytes("checkpoint.pt", tampered_checkpoint_body)
        swa_outer = artifact.publish_bytes("swa.pt", tampered_swa_body)
        lazy_outer = artifact.publish_bytes("lazy.pt", lazy_tampered_body)
        with pytest.raises(route.ContractError, match="state SHA"):
            backend.validate_checkpoint(
                artifact.reload_pair("checkpoint.pt", checkpoint_outer),
                epoch=44,
                global_step=45 * route.PUBLIC_SPEC.steps_per_epoch,
                spec=route.FULL_TRAIN_SPEC,
                binding=binding,
            )
        with pytest.raises(route.ContractError, match="state SHA"):
            backend.validate_swa(
                artifact.reload_pair("swa.pt", swa_outer),
                spec=route.FULL_TRAIN_SPEC,
                binding=binding,
            )
        with pytest.raises(route.ContractError, match="serialized lazy topology"):
            backend.validate_checkpoint(
                artifact.reload_pair("lazy.pt", lazy_outer),
                epoch=44,
                global_step=45 * route.PUBLIC_SPEC.steps_per_epoch,
                spec=route.FULL_TRAIN_SPEC,
                binding=binding,
            )


def test_exact_critical_gradient_map_covers_real_cell_d_paths_and_rejects_one_key_forgery() -> None:
    saved_python = random.getstate()
    saved_numpy = np.random.get_state()
    saved_torch = torch.get_rng_state()
    old_paths = list(sys.path)
    try:
        route._prepend_runtime_packages(ROOT)
        population = route._load_runtime_module(
            "_test_equal_session_gradient_pop", ROOT / "tfpd_exploration/src/tfpd_lane/pop_robust.py"
        )
        model = population.build_population_robustness_model(seed=42, cell="D").train()
        torch.manual_seed(9)
        neural = torch.randn(1, 50, 4)
        calib = torch.randn(1, 1, 100, 4)
        side = torch.randn(1, 4, 4)
        # No artificial dynamic-unit loss in this test: the live paths are
        # checked with an aligned synthetic B3S+T4 forward only.
        with patch("random.uniform", lambda _low, _high: 0.0):
            prediction, _ = model(neural, calib_trials=calib, side_features=side)
        prediction.square().mean().backward()
        proof = route._critical_gradient_proof(model, torch)
        assert set(proof) == set(route.CRITICAL_GRADIENT_PATHS)
        assert all(proof.values())
        route._validate_critical_gradient_proof(proof)
        contract = route._critical_gradient_contract_payload()
        assert contract["required_paths"] == {key: dict(value) for key, value in route.CRITICAL_GRADIENT_PATHS.items()}
        assert contract["excluded_uninitialized_lazy_keys"] == list(route.CELL_D_UNINITIALIZED_LAZY_KEYS)
        forged = dict(proof)
        forged["decoder_ffn"] = False
        with pytest.raises(route.ContractError, match="critical gradient"):
            route._validate_critical_gradient_proof(forged)
    finally:
        random.setstate(saved_python)
        np.random.set_state(saved_numpy)
        torch.set_rng_state(saved_torch)
        sys.path[:] = old_paths


def test_runtime_gpu_identity_and_peak_memory_schema_fail_closed_without_cuda() -> None:
    runtime = route._frozen_runtime_environment_fixture()
    route._validate_runtime_environment_payload(runtime)
    for path, replacement in (
        (("physical_nvidia_smi", "uuid"), "GPU-not-frozen"),
        (("physical_nvidia_smi", "bdf"), "00000000:02:00.0"),
        (("physical_nvidia_smi", "name"), "NVIDIA not-the-frozen-device"),
        (("gpu_authorities", "torch_total_memory_bytes"), 24_576),
        (("physical_nvidia_smi", "nvidia_smi_memory_total_mib"), 24_256),
        (("torch_properties", "torch_total_memory_bytes"), 25_435_111_423),
    ):
        forged = copy.deepcopy(runtime)
        forged[path[0]][path[1]] = replacement
        with pytest.raises(route.ContractError):
            route._validate_runtime_environment_payload(forged)
    for key, replacement in (
        ("torch_version", "2.5.1.post302"),
        ("torch_cuda_version", "12.1"),
        ("cudnn_version", 90_301),
    ):
        forged = copy.deepcopy(runtime)
        forged[key] = replacement
        with pytest.raises(route.ContractError, match="runtime Torch/CUDA/cuDNN version drift"):
            route._validate_runtime_environment_payload(forged)
    synthetic = route._synthetic_runtime_environment_payload()
    route._validate_synthetic_runtime_environment_payload(synthetic)
    with pytest.raises(route.ContractError):
        route._validate_runtime_environment_payload(synthetic)
    with pytest.raises(route.ContractError):
        route._validate_synthetic_runtime_environment_payload(runtime)
    resources = {
        "cuda_memory_allocated": 11,
        "cuda_memory_reserved": 13,
        "cuda_peak_memory_allocated": 17,
        "cuda_peak_memory_reserved": 19,
        "peak_stats_reset_before_training": True,
        "rss_bytes": 23,
        "throughput_windows_per_s": 29,
    }
    assert route._resource_payload(resources) == resources
    forged_resources = dict(resources)
    forged_resources["cuda_peak_memory_allocated"] = 10
    with pytest.raises(route.ContractError, match="peak memory"):
        route._resource_payload(forged_resources)
    forged_resources = dict(resources)
    forged_resources["peak_stats_reset_before_training"] = False
    with pytest.raises(route.ContractError, match="peak-memory reset"):
        route._resource_payload(forged_resources)


def test_source_smoke_cli_dual_flags_still_fail_before_capability_or_outputs() -> None:
    environment = dict(os.environ)
    environment["PYTHONNOUSERSITE"] = "1"
    before = [(ROOT / relative).exists() for relative in (route.RESULT_ROOT_RELATIVE, route.SMOKE_RESULT_ROOT_RELATIVE)]
    completed = subprocess.run(
        [sys.executable, str(CLI_PATH), "--source-smoke", "--i-have-root-reviewed-equal-session-smoke-authorization"],
        cwd=ROOT, env=environment, text=True, capture_output=True,
    )
    assert completed.returncode != 0
    assert "lacks an in-process root capability" in (completed.stdout + completed.stderr)
    assert [(ROOT / relative).exists() for relative in (route.RESULT_ROOT_RELATIVE, route.SMOKE_RESULT_ROOT_RELATIVE)] == before
