"""No-data/no-CUDA gates for CDM-D Source Execution V1."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import stat
import subprocess
import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "tfpd_exploration") not in sys.path:
    sys.path.insert(0, str(ROOT / "tfpd_exploration"))

from src.causal_dual_memory_cell_d_v1 import source_execute as execute  # noqa: E402
from src.causal_dual_memory_cell_d_v1 import source_execute_physical as physical  # noqa: E402


def _sha(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def _roster() -> tuple[str, ...]:
    return (
        execute.SOURCE_SMOKE_SESSION,
        *tuple(
            f"sub-C_ses-CO-{index:08d}"
            for index in range(execute.STRICT_SOURCE_COUNT - 1)
        ),
    )


def _manifest(roster: tuple[str, ...] | None = None) -> bytes:
    train = list(_roster() if roster is None else roster)
    value = {
        "schema_version": 1,
        "task": "CO",
        "split_counts": [27, 6, 6],
        "session_splits": {
            "train": train,
            "val": [f"sub-C_ses-CO-val-{index}" for index in range(6)],
            "test": [f"sub-C_ses-CO-test-{index}" for index in range(6)],
        },
    }
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _write(path: Path, body: bytes, *, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    os.chmod(path, mode)


def _synthetic_asset(label: str, relative: str, body: bytes, *, mode: int = 0o444,
                     sidecar: bool = True) -> execute.FixedAsset:
    return execute.FixedAsset(label, relative, _sha(body), mode, sidecar)


def _profile() -> dict[str, object]:
    return dict(execute.COMPATIBLE_DEVICE_PROFILES["gpu1"])


def _runtime_attestation() -> dict[str, object]:
    return {
        **_profile(),
        "visible_devices": 1,
        "attested": True,
        "torch_cuda_matmul_allow_tf32": False,
        "torch_cudnn_allow_tf32": False,
    }


def _identity(spec: execute.SourceExecutionSpec = execute.SOURCE_SMOKE_SPEC,
              closure_sha: str = "a" * 64) -> execute.SourceExecutionIdentity:
    return execute.SourceExecutionIdentity(
        spec=spec,
        closure={"schema": "synthetic", "closure_sha256": closure_sha},
        strict_train_roster=_roster(),
        fixed_assets={asset.label: asset.payload() for asset in execute.FIXED_ASSETS},
        normalizers=execute.SEALED_NORMALIZERS,
        selected_device=_profile(),
    )


def _session_topology(session: str | None = None) -> dict[str, object]:
    # A generic (no-session) topology is used only by explicit tamper cases.
    # Every durable session row must instead use the exact sealed count for
    # that source session, including the 1+2 known undefined-theta rows.
    invalid = 0 if session is None else execute.KNOWN_THETA_INVALID_UNIT_COUNTS.get(session, 0)
    value: dict[str, object] = {
        "total_unit_count": 8,
        "valid_unit_count": 8 - invalid,
        "invalid_unit_count": invalid,
        "valid_mask_sha256": "b" * 64,
        "theta_valid_mask_sha256": "b" * 64,
        "invalid_assignment_is_minus_one": True,
        "theta_authority_binds_only_validity_not_budget_groups": True,
    }
    if session is not None:
        value["session_id"] = session
    return value


def _sealed_swa_load_proof() -> dict[str, object]:
    return {
        "schema": execute.SEALED_SWA_LOAD_PROOF_SCHEMA,
        "sealed_terminal_sha256": execute.SEALED_CELL_D_TERMINAL_SHA256,
        "sealed_swa_sha256": execute.SEALED_CELL_D_SWA_SHA256,
        "fresh_strict_load": True,
        "recomputed_state_dict_sha256": "c" * 64,
        "initialized_trainable_parameters": execute.SEALED_CELL_D_INITIALIZED_TRAINABLE_PARAMETERS,
        "uninitialized_lazy_keys": list(execute.SEALED_CELL_D_UNINITIALIZED_LAZY_KEYS),
        "model_eval": True,
        "no_grad": True,
        "finite_forward": True,
        "repeated_fixed_forward_bitwise_equal": True,
        "model_state_unchanged": True,
        "dynamic_dropout_calls": 0,
    }


def _memory_transition_evidence(*, budget: int = 30, accepted: bool = True) -> dict[str, object]:
    activity = "a" * 64
    carrier_before = "b" * 64
    # The M30 rows are the separate post-support B8 safety pool: pseudo
    # outcomes are produced but can never mutate deployment state.
    carrier_after = carrier_before if budget == 30 else ("c" * 64 if accepted else carrier_before)
    value: dict[str, object] = {
        "all_four_groups_finalized": True,
        "group_label_broadcast": False,
        "all_group_inputs_physically_sliced": True,
        "held_group_count": 4,
        "label_join_before_outcomes": False,
        "deployment_activity_memory_before_sha256": activity,
        "deployment_activity_memory_after_sha256": activity,
        "carrier_state_before_sha256": carrier_before,
        "carrier_state_after_sha256": carrier_after,
        "accepted_group_update_count": 4 if accepted else 0,
    }
    if budget == 30:
        value.update({
            "activity_fifo_capacity": 0,
            "activity_query_count_before": 0,
            "activity_query_count_after": 0,
            "m30_audit_enters_deployment_activity_memory": False,
        })
    return value


def _budget_initial_receipt_fields(budget: int) -> dict[str, object]:
    return {
        "budget_initial_carrier_recipe": "fixed_ridge_by_trial",
        "budget_initial_carrier_support_rows": budget,
        "raw_m30_t4_used_as_initializer": False,
        "initial_support_rate_domain": execute.INITIAL_SUPPORT_RATE_DOMAIN,
        "online_update_rate_domain": execute.ONLINE_UPDATE_RATE_DOMAIN,
        "initial_support_rates_sha256": "7" * 64,
        "initial_support_exposure_seconds_sha256": "8" * 64,
        "budget_initial_carrier_parity": {"mode": "fixed_ridge_by_trial"},
        "budget_initial_carrier_sha256": "3" * 64,
        "budget_groups_sha256": "4" * 64,
        "budget_group_assignment_sha256": "5" * 64,
        "budget_group_valid_mask_sha256": "6" * 64,
    }


class _MockBackend:
    def __init__(self, *, stop_m30: bool = False, fail_prepare: bool = False) -> None:
        self.stop_m30 = stop_m30
        self.fail_prepare = fail_prepare
        self.events: list[str] = []
        self.runtime = object()

    def preflight(self, *, root, identity, flags):
        self.events.append("preflight")
        return {
            "source_resolved_or_opened": False,
            "checkpoint_opened": False,
            "cuda_initialized": False,
        }

    def prepare(self, *, root, identity, flags):
        self.events.append("prepare")
        if self.fail_prepare:
            raise RuntimeError("synthetic prepare failure")
        flags.source_resolved = True
        flags.source_opened = True
        flags.checkpoint_opened = True
        flags.cuda_initialized = True
        return self.runtime

    def source_authority(self, runtime, *, identity, flags):
        self.events.append("source_authority")
        return {
            "schema": "causal_dual_memory_cell_d_source_execution_authority_v1",
            "cell": execute.CELL,
            "identity_sha256": identity.sha256,
            "strict_train_roster": list(identity.strict_train_roster),
            "strict_train_roster_sha256": execute.roster_sha256(identity.strict_train_roster),
            "normalizers": identity.normalizers.payload(),
            "fixed_assets": {asset.label: asset.payload() for asset in execute.FIXED_ASSETS},
            "runtime_environment": _runtime_attestation(),
            "sealed_swa_load_proof": _sealed_swa_load_proof(),
            "physical_session_count": (
                1 if identity.spec.kind == "source_smoke" else len(identity.strict_train_roster)
            ),
            "strict_manifest_bound_without_nonphysical_resolution": True,
            "sessions": [
                _session_topology(session)
                for session in ((execute.SOURCE_SMOKE_SESSION,) if identity.spec.kind == "source_smoke"
                                else identity.strict_train_roster)
            ],
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
            "session": "sub-C_ses-CO-20131003", "budget": 30,
            "support_positions": list(range(30)), "audit_positions": [30, 31],
            "group_count": 4, "b8_threshold_applied": False, "source_only": True,
            **_budget_initial_receipt_fields(30),
        }

    def run_budget(self, runtime, *, budget, identity, flags):
        self.events.append(f"budget:{budget}")
        passing = 13 if self.stop_m30 and budget == 30 else 14
        return tuple({
            "schema": "causal_dual_memory_cell_d_source_execution_session_b8_v1",
            "budget": budget, "session": session, "status": "COMPLETE_FIXED_POOL",
            "pass": index < passing, "unit_topology": _session_topology(),
            "source_authority_unit_topology": _session_topology(session),
            "source_only": True, "target_optimizer_backward_update": 0,
            **_budget_initial_receipt_fields(budget),
        } for index, session in enumerate(identity.strict_train_roster))

    def resources(self, runtime, *, flags):
        self.events.append("resources")
        return {
            "endpoint_chunks_per_s": 1.0, "trials_per_s": 1.0, "wall_seconds": 1.0,
            "endpoint_chunks_completed": 1, "completed_trials": 1,
            "rss_bytes": 1, "current_cuda_allocated_bytes": 4, "current_cuda_reserved_bytes": 8,
            "peak_cuda_allocated_bytes": 4, "peak_cuda_reserved_bytes": 8,
            "selected_device": _profile(),
        }

    def revalidate(self, *, root, identity, flags):
        self.events.append("revalidate")

    def close(self, runtime):
        self.events.append("close")


def _patched_lifecycle(monkeypatch, closure: dict[str, object]) -> None:
    monkeypatch.setattr(execute, "execution_closure_payload", lambda root: closure)


def _root_for_result(tmp_path: Path) -> Path:
    (tmp_path / "tfpd_exploration/results").mkdir(parents=True)
    return tmp_path


def _capability(identity: execute.SourceExecutionIdentity) -> execute.SourceExecutionCapability:
    return execute._issue_root_reviewed_capability(identity, seal=execute._ROOT_REVIEW_SEAL)


def test_static_constants_bind_exact_accepted_predecessor_closures_and_assets() -> None:
    assert execute.WORKORDER_SHA256 == "644ad075ee1f0d7bcd9d97e80ed5b765190bc9b4de278baaef309d0017fe9981"
    assert execute.SOURCE_AUDIT_CLOSURE_SHA256 == "cfc192c45f674ccd4d56f19e3d5a58dc9146cf2fb02b02d4f561e7e0bb2db4dc"
    assert execute.STAGE0_CLOSURE_SHA256 == "3ab6d3de931e4630cb9c80b07e25e3b38af4c444f3c937d3d28560b596c29590"
    assert [asset.label for asset in execute.FIXED_ASSETS] == [
        "strict_manifest", "sealed_cell_d_terminal", "sealed_cell_d_swa", "theta_receipt", "theta_artifact",
    ]
    assert execute.SOURCE_SMOKE_SPEC.payload()["smoke"] == {
        "session": "sub-C_ses-CO-20131003", "budget": 30,
        "support_positions": list(range(30)), "audit_positions": [30, 31],
        "group_count": 4, "max_endpoints_per_forward_chunk": 128,
    }


def test_descriptor_fixed_asset_rejects_sidecar_mode_symlink_and_swap(tmp_path: Path) -> None:
    root = tmp_path
    body = b'{"immutable":true}'
    asset = _synthetic_asset("synthetic", "fixed/authority.json", body)
    leaf = root / asset.relative
    _write(leaf, body, mode=0o444)
    _write(leaf.with_name("authority.json.sha256"), f"{asset.sha256}  authority.json\n".encode(), mode=0o444)
    bound = execute.descriptor_read_fixed_asset(root, asset)
    assert bound.body == body and bound.payload()["bytes"] == len(body)
    os.chmod(leaf, 0o664)
    with pytest.raises(execute.SourceExecutionError, match="mode"):
        execute.descriptor_read_fixed_asset(root, asset)
    os.chmod(leaf, 0o444)
    os.chmod(leaf.with_name("authority.json.sha256"), 0o644)
    leaf.with_name("authority.json.sha256").write_bytes(b"forged\n")
    os.chmod(leaf.with_name("authority.json.sha256"), 0o444)
    with pytest.raises(execute.SourceExecutionError, match="sidecar"):
        execute.descriptor_read_fixed_asset(root, asset)
    leaf.with_name("authority.json.sha256").unlink()
    leaf.unlink()
    target = root / "target"
    _write(target, body, mode=0o444)
    os.symlink(target, leaf)
    with pytest.raises(execute.SourceExecutionError):
        execute.descriptor_read_fixed_asset(root, asset)


def test_manifest_keeps_val_test_inert_and_rejects_nontrain_topology() -> None:
    roster = execute.parse_strict_train_roster(_manifest())
    assert roster == _roster()
    bad = json.loads(_manifest())
    bad["session_splits"]["val"][0] = roster[0]
    with pytest.raises(execute.SourceExecutionError, match="overlap"):
        execute.parse_strict_train_roster(json.dumps(bad).encode())


def test_exact_normalizers_and_dynamic_profiles_reject_numeric_runtime_tf32_and_ordinal_drift() -> None:
    assert execute.SEALED_NORMALIZERS.payload()["raw_before_t4_normalization"] is True
    profile = _profile()
    assert execute.validate_selected_device_environment(profile, {
        "CUDA_VISIBLE_DEVICES": "1", "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
    })["logical_device"] == "cuda:0"
    wrong = dict(profile); wrong["uuid"] = "GPU-wrong"
    with pytest.raises(execute.SourceExecutionError, match="exact reviewed"):
        execute.validate_compatible_device_profile(wrong)
    with pytest.raises(execute.SourceExecutionError, match="CUDA_VISIBLE_DEVICES"):
        execute.validate_selected_device_environment(profile, {"CUDA_VISIBLE_DEVICES": "0", "CUDA_DEVICE_ORDER": "PCI_BUS_ID"})
    attestation = _runtime_attestation(); attestation["torch_cuda_matmul_allow_tf32"] = True
    with pytest.raises(execute.SourceExecutionError, match="TF32"):
        execute.validate_runtime_attestation(profile, attestation)
    with pytest.raises(execute.SourceExecutionError, match="normalizer"):
        execute.SourceNormalizers(behavior_mean=(0.0, execute.SEALED_BEHAVIOR_MEAN[1]))


def test_execution_closure_reconstructs_current_explicit_paths() -> None:
    closure = execute.execution_closure_payload(ROOT)
    assert closure["stage0_closure_sha256"] == execute.STAGE0_CLOSURE_SHA256
    assert closure["source_audit_closure_sha256"] == execute.SOURCE_AUDIT_CLOSURE_SHA256
    paths = [row["path"] for row in closure["paths"]]
    assert "tfpd_exploration/src/causal_dual_memory_cell_d_v1/source_execute.py" in paths
    assert "tfpd_exploration/src/causal_dual_memory_cell_d_v1/source_execute_physical.py" in paths
    assert "tfpd_exploration/scripts/run_causal_dual_memory_cell_d_source_gate.py" in paths
    assert "tfpd_exploration/tests/test_causal_dual_memory_cell_d_source_execution.py" in paths
    assert len(closure["closure_sha256"]) == 64


def test_reviewed_factory_is_side_effect_free_and_owns_concrete_source_model_seams(tmp_path: Path) -> None:
    """The real factory must not leave parser/model construction to root code."""
    source_capability = execute._issue_root_reviewed_source_data_capability(
        canonical_root=tmp_path / "strict_source_only",
        strict_train_roster=_roster(),
        seal=execute._ROOT_REVIEW_SEAL,
    )
    backend = physical.build_reviewed_physical_backend(
        root=ROOT, source_data=source_capability, selected_device=_profile(),
    )
    assert isinstance(backend.provider, physical.ConcreteStrict27SessionProvider)
    assert isinstance(backend.executor, physical.ConcreteCellDFourGroupExecutor)
    assert not (tmp_path / "strict_source_only").exists()


@pytest.mark.parametrize("budget,prefix_length", ((4, 4), (10, 10), (4, 15), (10, 20), (4, 30), (10, 30)))
def test_variable_prefix_held_slice_binds_exact_causal_lengths_and_rejects_padding_or_future_rows(
    budget: int, prefix_length: int,
) -> None:
    del budget  # Parameter documents both M4 and M10 causal trajectories.
    units = 8
    neural = np.arange(3 * 50 * units, dtype=np.float32).reshape(3, 50, units)
    activity = np.arange(prefix_length * 100 * units, dtype=np.float32).reshape(prefix_length, 100, units)
    side = np.arange(units * 4, dtype=np.float32).reshape(units, 4)
    held = np.asarray([True, True, False, False, False, False, False, False], dtype=np.bool_)
    exact = physical._variable_prefix_array_digest(activity)
    sliced = physical.physically_slice_variable_prefix_held_units(
        neural, activity, side, held, expected_prefix_length=prefix_length,
        full_prefix_activity_sha256=exact,
    )
    assert sliced.b3s_activity_stack.shape == (prefix_length, 100, 6)
    assert sliced.expected_prefix_length == prefix_length
    # A padded 30-row stack cannot masquerade as an early prefix.
    if prefix_length < 30:
        padded = np.concatenate((activity, np.zeros((30 - prefix_length, 100, units), dtype=np.float32)), axis=0)
        with pytest.raises(physical.SourceExecutionPhysicalError, match="exact causal"):
            physical.physically_slice_variable_prefix_held_units(
                neural, padded, side, held, expected_prefix_length=prefix_length,
                full_prefix_activity_sha256=exact,
            )
    # An extra/future row changes the held-memory authority rather than being
    # accepted merely because the first rows happen to be unchanged.
    altered = activity.copy()
    altered[-1, -1, -1] += 1.0
    with pytest.raises(physical.SourceExecutionPhysicalError, match="not the exact causal"):
        physical.physically_slice_variable_prefix_held_units(
            neural, altered, side, held, expected_prefix_length=prefix_length,
            full_prefix_activity_sha256=exact,
        )


def test_shared_cell_d_cpu_forward_accepts_real_variable_b3s_prefixes_without_cuda() -> None:
    """A real CPU graph proves the shared B3S contract, not a fake adapter."""
    import torch
    from src.tfpd_lane import pop_robust

    assert not torch.cuda.is_initialized()
    model = pop_robust.build_population_robustness_model(seed=42, cell="D").eval()
    with torch.no_grad():
        outputs = []
        for prefix_length in (4, 10, 30):
            output, _ = model(
                torch.zeros((1, 50, 8), dtype=torch.float32),
                calib_trials=torch.zeros((1, prefix_length, 100, 8), dtype=torch.float32),
                side_features=torch.zeros((1, 8, 4), dtype=torch.float32),
            )
            outputs.append(output)
    assert [tuple(item.shape) for item in outputs] == [(1, 50, 2)] * 3
    assert all(bool(torch.isfinite(item).all()) for item in outputs)
    assert not torch.cuda.is_initialized()


def test_held_b3s_rebuilder_uses_native_counts_and_interval_not_accepted_activity() -> None:
    """The provider's rebuild proof must not be a decorative accepted-view copy."""
    from src.causal_dual_memory_cell_d_v1 import core
    from src.causal_dual_memory_cell_d_v1 import source_adapter

    session = execute.SOURCE_SMOKE_SESSION
    trial_id = f"{session}:trial:17"
    channels = np.arange(4, dtype=np.int64)
    full_native = np.arange(80 * 4, dtype=np.int64).reshape(80, 4) % 5
    start, stop = 11, 71
    endpoints = np.arange(start + 49, stop, dtype=np.int64)
    calls: list[tuple[str, int, int, int, int, float, bool]] = []

    def faithful_builder(counts, trials, m_trials, t_bins, n_units, pad_value, interpolate_trials):
        calls.append((
            core.array_digest(counts), int(trials[0]["start"]), int(trials[0]["stop"]),
            int(m_trials), int(n_units), float(pad_value), bool(interpolate_trials),
        ))
        result = np.full((m_trials, t_bins, n_units), pad_value, dtype=np.float32)
        trial = np.asarray(counts)[int(trials[0]["start"]):int(trials[0]["stop"])]
        # Faithful shape/pad semantics are enough for this no-data injection;
        # production supplies the actual closure-bound _build_calib_trials.
        result[0, :min(t_bins, trial.shape[0])] = trial[:t_bins]
        return result

    accepted = faithful_builder(full_native, [{"start": start, "stop": stop}], 1, 100, 4, -1.0, True)[0]
    calls.clear()

    def make_record(accepted_activity: np.ndarray):
        channel_sha = core.channel_order_digest(channels)
        descriptor = source_adapter.SourceTrialDescriptor(
            session_id=session,
            trial_id=trial_id,
            source_record_sha256="1" * 64,
            channel_order_sha256=channel_sha,
            full_native_counts_sha256=core.array_digest(full_native),
            accepted_b3s_activity_sha256=core.array_digest(accepted_activity),
            raw_trial_binding_sha256=source_adapter.raw_trial_binding_sha256(
                session_id=session, trial_id=trial_id, source_record_sha256="1" * 64,
                channel_order_sha256=channel_sha, full_native_counts_sha256=core.array_digest(full_native),
                rewarded_interval_start_bin=start, rewarded_interval_stop_bin=stop,
                accepted_b3s_activity_sha256=core.array_digest(accepted_activity),
                window_endpoint_bins_sha256=core.array_digest(endpoints),
                neural_endpoint_available_sha256=core.array_digest(np.ones(endpoints.shape, dtype=np.bool_)),
                b3s_rebuilder_semantics=source_adapter.B3S_REBUILDER_SEMANTICS,
            ),
        )
        return source_adapter.HeldSourceTrialRecord(
            descriptor=descriptor, channel_ids=channels, full_native_binned_counts=full_native,
            rewarded_interval_start_bin=start, rewarded_interval_stop_bin=stop,
            accepted_calibration_b3s_activity=accepted_activity,
            rebuild_b3s_activity=lambda held: physical._rebuild_b3s_activity_from_held_native(
                held, builder=faithful_builder,
            ),
            window_endpoint_bins=endpoints,
            neural_endpoint_available=np.ones(endpoints.shape, dtype=np.bool_),
        )

    roster = source_adapter.StrictSourceRoster(_roster())
    views = source_adapter.materialize_source_trial_views(make_record(accepted), roster=roster)
    assert views.b3s_activity.activity.shape == (100, 4)
    assert calls == [(core.array_digest(full_native), start, stop, 1, 4, -1.0, True)]

    # A coherently re-bound but altered accepted activity must still fail: the
    # callable rebuilds from native bytes and interval, never from that field.
    altered = accepted.copy(); altered[0, 0] += 1.0
    with pytest.raises(source_adapter.SourceAdapterError, match="rebuild differs"):
        source_adapter.materialize_source_trial_views(make_record(altered), roster=roster)
    assert calls[-1] == (core.array_digest(full_native), start, stop, 1, 4, -1.0, True)


def _support_only_material(*, raw_m30_value: float) -> tuple[physical.Strict27SessionMaterial, tuple[str, ...]]:
    """Small typed source fixture for real budget-state construction only."""
    from src.causal_dual_memory_cell_d_v1 import core

    session = execute.SOURCE_SMOKE_SESSION
    channels = np.arange(8, dtype=np.int64)
    channel_sha = core.channel_order_digest(channels)
    ids = tuple(f"{session}:trial:{index}" for index in range(60))
    views: dict[str, object] = {}
    exact_duration_rates: dict[str, np.ndarray] = {}
    exact_duration_exposure: dict[str, float] = {}
    for index, trial_id in enumerate(ids):
        activity = core.B3SInterpolatedSpikeCountTrial(
            np.full((100, 8), float(index + 1), dtype=np.float32), session, trial_id, channel_sha,
        )
        counts = core.NativeRewardedTrialSpikeCounts(
            np.full((5, 8), index % 7 + 1, dtype=np.int64), session, trial_id, channel_sha, 0, 5,
        )
        views[trial_id] = SimpleNamespace(b3s_activity=activity, carrier_counts=counts)
        if index < 30:
            exposure = 0.071 + index * 0.001
            raw_count = np.full(8, 3 * (index + 1), dtype=np.float64)
            exact_duration_rates[trial_id] = raw_count / exposure
            exact_duration_exposure[trial_id] = exposure
    material = physical.Strict27SessionMaterial(
        session_id=session,
        ordered_rewarded_trial_ids=ids,
        trial_views_by_id=views,
        source_theta_topology=physical.SourceThetaTopology(channels, np.ones(8, dtype=np.bool_)),
        m4_d_optimal_indices=(0, 8, 16, 24),
        raw_t4_channel_order_sha256=channel_sha,
        theta_valid_mask_sha256="a" * 64,
        theta_invalid_unit_count=0,
        source_descriptor_sha256="b" * 64,
        raw_m30_t4=np.full((8, 4), raw_m30_value, dtype=np.float64),
        support_direction_indices_by_id={trial_id: index % 8 for index, trial_id in enumerate(ids[:30])},
        exact_duration_support_rates_by_id=exact_duration_rates,
        exact_duration_support_exposure_seconds_by_id=exact_duration_exposure,
    )
    return material, ids


@pytest.mark.parametrize("budget", (4, 10, 30))
def test_budget_specific_fixed_ridge_initial_carrier_and_groups_ignore_raw_m30_values(budget: int) -> None:
    """No M30 raw carrier/post-budget label may leak into M4/M10 starts."""
    from src.causal_dual_memory_cell_d_v1 import core

    material_a, ids = _support_only_material(raw_m30_value=0.0)
    material_b, _ = _support_only_material(raw_m30_value=12345.0)
    executor_a = physical.ConcreteCellDFourGroupExecutor(root=ROOT)
    executor_b = physical.ConcreteCellDFourGroupExecutor(root=ROOT)
    # begin_session_budget needs only its state holder; no model/source/CUDA
    # path is reachable in this support-only CPU proof.
    executor_a._runtime = physical._ConcreteExecutorState({}, None, None, None)  # type: ignore[arg-type]
    executor_b._runtime = physical._ConcreteExecutorState({}, None, None, None)  # type: ignore[arg-type]
    support_indices = (0, 8, 16, 24) if budget == 4 else tuple(range(budget))
    support = tuple(ids[index] for index in support_indices)
    proof_a = executor_a.begin_session_budget(
        material=material_a, budget=budget, support_trial_ids=support, flags=execute.RuntimeFlags(),
    )
    proof_b = executor_b.begin_session_budget(
        material=material_b, budget=budget, support_trial_ids=support, flags=execute.RuntimeFlags(),
    )
    group_a = executor_a.groups_for_session_budget(material=material_a, budget=budget)
    group_b = executor_b.groups_for_session_budget(material=material_b, budget=budget)
    state_a = executor_a._runtime.state_by_session_budget[(material_a.session_id, budget)]
    state_b = executor_b._runtime.state_by_session_budget[(material_b.session_id, budget)]
    assert proof_a["initial_carrier_recipe"] == "fixed_ridge_by_trial"
    assert proof_a["initial_carrier_parity"]["mode"] == core.CarrierFitMode.FIXED_RIDGE_BY_TRIAL.value
    assert proof_a["raw_m30_t4_used_as_initializer"] is False
    assert proof_a["initial_carrier_sha256"] == proof_b["initial_carrier_sha256"]
    assert proof_a["budget_groups_sha256"] == proof_b["budget_groups_sha256"] == group_a.digest == group_b.digest
    assert np.array_equal(group_a.assignment, group_b.assignment)
    assert state_a.state.digest == state_b.state.digest
    assert np.array_equal(state_a.read_prediction_inputs().active_t4, state_b.read_prediction_inputs().active_t4)


@pytest.mark.parametrize("budget,support_indices,mutated_index", (
    (4, (0, 8, 16, 24), 29),
    (10, tuple(range(10)), 29),
))
def test_m30_only_label_mutation_cannot_change_m4_m10_initial_consumer_state(
    budget: int, support_indices: tuple[int, ...], mutated_index: int,
) -> None:
    """Later M30-prefix labels cannot leak into short-budget initial groups."""
    from dataclasses import replace

    material, ids = _support_only_material(raw_m30_value=0.0)
    altered_labels = dict(material.support_direction_indices_by_id)
    altered_labels[ids[mutated_index]] = (altered_labels[ids[mutated_index]] + 3) % 8
    altered = replace(material, support_direction_indices_by_id=altered_labels)
    support = tuple(ids[index] for index in support_indices)
    first = physical.ConcreteCellDFourGroupExecutor(root=ROOT)
    second = physical.ConcreteCellDFourGroupExecutor(root=ROOT)
    first._runtime = physical._ConcreteExecutorState({}, None, None, None)  # type: ignore[arg-type]
    second._runtime = physical._ConcreteExecutorState({}, None, None, None)  # type: ignore[arg-type]
    evidence_a = first.begin_session_budget(
        material=material, budget=budget, support_trial_ids=support, flags=execute.RuntimeFlags(),
    )
    evidence_b = second.begin_session_budget(
        material=altered, budget=budget, support_trial_ids=support, flags=execute.RuntimeFlags(),
    )
    state_a = first._runtime.state_by_session_budget[(material.session_id, budget)]
    state_b = second._runtime.state_by_session_budget[(altered.session_id, budget)]
    assert evidence_a["initial_carrier_sha256"] == evidence_b["initial_carrier_sha256"]
    assert evidence_a["budget_groups_sha256"] == evidence_b["budget_groups_sha256"]
    assert state_a.state.digest == state_b.state.digest
    assert np.array_equal(state_a.read_prediction_inputs().active_t4, state_b.read_prediction_inputs().active_t4)


def test_initial_support_rates_use_exact_duration_not_native_20ms_online_rate() -> None:
    """Initial fixed-ridge parity is defined in the historical rate domain."""
    from src.calibration_budget_comparators_v1 import fit_ridge_t4
    from src.causal_dual_memory_cell_d_v1 import core

    material, ids = _support_only_material(raw_m30_value=0.0)
    raw_counts = np.asarray([[2, 5, 11], [7, 3, 13], [17, 19, 23], [29, 31, 37]], dtype=np.int64)
    exposure = np.asarray([0.031, 0.047, 0.089], dtype=np.float64)
    exact_from_provider = physical._exact_duration_support_rate_matrix(raw_counts, exposure)
    assert np.array_equal(exact_from_provider, raw_counts.T / exposure[:, None])
    # This deliberately differs from a 20-ms binned mean for non-bin-exact
    # durations, making a hidden estimator-domain substitution observable.
    assert not np.array_equal(exact_from_provider[0], raw_counts[:, 0] / 0.020)
    support = tuple(ids[index] for index in (0, 1, 2, 3))
    rates, labels, evidence = physical.ConcreteCellDFourGroupExecutor._support_rates_and_labels(
        material=material, support_trial_ids=support, core=core,
    )
    expected = np.stack([material.exact_duration_support_rates_by_id[item] for item in support])
    native_20ms = np.stack([
        core.scalar_rates_from_native_rewarded_counts(material.trial_views_by_id[item].carrier_counts)
        for item in support
    ])
    assert np.array_equal(rates, expected)
    assert not np.array_equal(rates, native_20ms)
    assert evidence["initial_support_rate_domain"] == execute.INITIAL_SUPPORT_RATE_DOMAIN
    assert evidence["online_update_rate_domain"] == execute.ONLINE_UPDATE_RATE_DOMAIN
    angles = np.asarray([core.CANONICAL_DIRECTIONS_RAD[int(item)] for item in labels], dtype=np.float64)
    comparator, _proof = fit_ridge_t4(rates, angles, normalized_lambda=0.1)
    candidate, parity = physical.ConcreteCellDFourGroupExecutor._budget_initial_carrier(
        rates=rates, labels=labels, budget=4, core=core,
    )
    assert np.array_equal(candidate, comparator)
    assert parity["initial_carrier_parity"]["mode"] == "fixed_ridge_by_trial"


def test_executor_resources_measure_actual_forward_counters_and_synchronize_before_reading() -> None:
    """A terminal resource receipt may not fabricate zero measured throughput."""
    import time

    class FakeCuda:
        def __init__(self) -> None:
            self.synchronize_calls: list[int] = []
        def synchronize(self, device: int) -> None:
            self.synchronize_calls.append(device)
        def memory_allocated(self, device: int) -> int: return 11
        def memory_reserved(self, device: int) -> int: return 17
        def max_memory_allocated(self, device: int) -> int: return 23
        def max_memory_reserved(self, device: int) -> int: return 29

    cuda = FakeCuda()
    executor = physical.ConcreteCellDFourGroupExecutor(root=ROOT)
    executor._runtime = physical._ConcreteExecutorState(
        modules={"torch": SimpleNamespace(cuda=cuda)}, model=None, device=None, behavior_normalizer=None,
        endpoint_chunks_completed=12, completed_trials=3,
        measurement_started_monotonic=time.monotonic() - 0.05,
    )
    receipt = dict(executor.resources())
    receipt["selected_device"] = _profile()
    execute._validate_resources(receipt)
    assert cuda.synchronize_calls == [0]
    assert receipt["endpoint_chunks_per_s"] > 0.0 and receipt["trials_per_s"] > 0.0
    assert receipt["wall_seconds"] > 0.0
    assert receipt["peak_cuda_allocated_bytes"] >= receipt["current_cuda_allocated_bytes"]
    assert receipt["peak_cuda_reserved_bytes"] >= receipt["current_cuda_reserved_bytes"]
    executor._runtime.endpoint_chunks_completed = 0
    with pytest.raises(physical.SourceExecutionPhysicalError, match="completed measured"):
        executor.resources()


@pytest.mark.parametrize("matmul_before,cudnn_before", ((False, False), (False, True), (True, False), (True, True)))
def test_executor_tf32_is_route_local_and_restored_even_without_runtime(
    matmul_before: bool, cudnn_before: bool,
) -> None:
    """The physical close path restores process-global TF32 flags on all exits."""
    fake_torch = SimpleNamespace(
        backends=SimpleNamespace(
            cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=matmul_before)),
            cudnn=SimpleNamespace(allow_tf32=cudnn_before),
        ),
    )
    success_executor = physical.ConcreteCellDFourGroupExecutor(root=ROOT)
    success_executor._enforce_route_local_tf32_false(fake_torch)
    assert fake_torch.backends.cuda.matmul.allow_tf32 is False
    assert fake_torch.backends.cudnn.allow_tf32 is False
    success_executor.close()
    assert fake_torch.backends.cuda.matmul.allow_tf32 is matmul_before
    assert fake_torch.backends.cudnn.allow_tf32 is cudnn_before

    # Repeat through the backend's no-runtime failure close path: this is the
    # path used when model construction fails after attestation.
    executor = physical.ConcreteCellDFourGroupExecutor(root=ROOT)
    executor._enforce_route_local_tf32_false(fake_torch)

    class Provider:
        def close(self) -> None: pass

    backend = physical.PhysicalSourceExecutionBackend(provider=Provider(), executor=executor)
    # This simulates an exception after attestation but before a concrete
    # runtime object exists; lifecycle finally still invokes backend.close.
    backend.close(None)
    assert fake_torch.backends.cuda.matmul.allow_tf32 is matmul_before
    assert fake_torch.backends.cudnn.allow_tf32 is cudnn_before


def test_capability_and_public_flags_fail_before_backend_or_root(monkeypatch, tmp_path: Path) -> None:
    identity = _identity()
    _patched_lifecycle(monkeypatch, dict(identity.closure))
    backend = _MockBackend()
    with pytest.raises(execute.SourceExecutionError, match="capability"):
        execute.execute_authorized(_root_for_result(tmp_path), identity=identity, capability=None, backend=backend,
                                   environ={"CUDA_VISIBLE_DEVICES": "1", "CUDA_DEVICE_ORDER": "PCI_BUS_ID"})
    assert backend.events == []
    assert not (tmp_path / execute.SOURCE_SMOKE_ROOT_RELATIVE).exists()


def test_artifact_root_pairs_reload_and_roll_back_owned_partial_publication(tmp_path: Path, monkeypatch) -> None:
    root = _root_for_result(tmp_path)
    artifact = execute.reserve_artifact_root(root, spec=execute.SOURCE_SMOKE_SPEC, roster=_roster())
    try:
        digest = artifact.publish_json_pair("attempt.json", {"x": 1})
        assert artifact.read_json_pair("attempt.json", expected_sha256=digest) == {"x": 1}
        original = artifact._write_leaf
        calls = {"n": 0}

        def fail_second(name: str, body: bytes):
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("synthetic sidecar write failure")
            return original(name, body)

        monkeypatch.setattr(artifact, "_write_leaf", fail_second)
        with pytest.raises(RuntimeError, match="sidecar"):
            artifact.publish_json_pair("launch.json", {"x": 2})
        assert not (artifact.directory / "launch.json").exists()
        assert not (artifact.directory / "launch.json.sha256").exists()
    finally:
        artifact.close()


def test_smoke_lifecycle_is_attempt_before_source_then_transactional_terminal(monkeypatch, tmp_path: Path) -> None:
    identity = _identity()
    _patched_lifecycle(monkeypatch, dict(identity.closure))
    root = _root_for_result(tmp_path)
    backend = _MockBackend()
    outcome = execute.execute_authorized(
        root, identity=identity, capability=_capability(identity), backend=backend,
        environ={"CUDA_VISIBLE_DEVICES": "1", "CUDA_DEVICE_ORDER": "PCI_BUS_ID"},
    )
    assert outcome["status"] == "SMOKE_COMPLETED"
    assert backend.events == ["preflight", "prepare", "source_authority", "smoke", "revalidate", "resources", "close"]
    artifact = root / execute.SOURCE_SMOKE_ROOT_RELATIVE
    assert {item.name for item in artifact.iterdir()} == {
        "attempt.json", "attempt.json.sha256", "launch.json", "launch.json.sha256",
        "source_authority.json", "source_authority.json.sha256", "smoke.json", "smoke.json.sha256",
        "terminal.json", "terminal.json.sha256",
    }
    assert stat.S_IMODE((artifact / "terminal.json").stat().st_mode) == 0o444


def test_full_gate_stop_is_completed_terminal_not_failure(monkeypatch, tmp_path: Path) -> None:
    identity = _identity(execute.SOURCE_GATE_SPEC)
    _patched_lifecycle(monkeypatch, dict(identity.closure))
    root = _root_for_result(tmp_path)
    backend = _MockBackend(stop_m30=True)
    outcome = execute.execute_authorized(
        root, identity=identity, capability=_capability(identity), backend=backend,
        environ={"CUDA_VISIBLE_DEVICES": "1", "CUDA_DEVICE_ORDER": "PCI_BUS_ID"},
    )
    assert outcome["status"] == "STOP_SOURCE_B8_CONSTRUCTIBILITY"
    assert backend.events.count("budget:30") == 1
    assert "budget:10" not in backend.events and "budget:4" not in backend.events
    artifact = root / execute.SOURCE_GATE_ROOT_RELATIVE
    assert (artifact / "terminal.json").exists() and not (artifact / "failure.json").exists()
    terminal = json.loads((artifact / "terminal.json").read_text())
    assert terminal["status"] == "STOP_SOURCE_B8_CONSTRUCTIBILITY"


def test_prepare_failure_publishes_honest_failure_only(monkeypatch, tmp_path: Path) -> None:
    identity = _identity()
    _patched_lifecycle(monkeypatch, dict(identity.closure))
    root = _root_for_result(tmp_path)
    backend = _MockBackend(fail_prepare=True)
    with pytest.raises(RuntimeError, match="synthetic prepare"):
        execute.execute_authorized(
            root, identity=identity, capability=_capability(identity), backend=backend,
            environ={"CUDA_VISIBLE_DEVICES": "1", "CUDA_DEVICE_ORDER": "PCI_BUS_ID"},
        )
    artifact = root / execute.SOURCE_SMOKE_ROOT_RELATIVE
    failure = json.loads((artifact / "failure.json").read_text())
    assert failure["stage"] == "prepare"
    assert failure["flags"]["source_opened"] is False
    assert failure["terminal_published"] is False
    assert not (artifact / "terminal.json").exists()


def test_session_and_budget_validators_preserve_all_27_and_14_of_27_rule() -> None:
    rows = tuple({
        "schema": "causal_dual_memory_cell_d_source_execution_session_b8_v1",
        "budget": 30, "session": session, "status": "COMPLETE_FIXED_POOL",
        "pass": index < 14, "unit_topology": _session_topology(session),
        "source_authority_unit_topology": _session_topology(session),
        "source_only": True, "target_optimizer_backward_update": 0,
        **_budget_initial_receipt_fields(30),
    } for index, session in enumerate(_roster()))
    aggregate = execute._aggregate_budget(rows, budget=30, roster=_roster())
    assert aggregate["passing_session_count"] == 14 and aggregate["breadth_pass"] is True
    bad = list(rows); bad[0] = {**bad[0], "unit_topology": {**_session_topology(_roster()[0]), "invalid_unit_count": 3}}
    with pytest.raises(execute.SourceExecutionError, match="topology"):
        execute._aggregate_budget(bad, budget=30, roster=_roster())
    with pytest.raises(execute.SourceExecutionError, match="manifest order"):
        execute._aggregate_budget(tuple(reversed(rows)), budget=30, roster=_roster())


def test_physical_backend_prebinds_only_train_then_joins_truth_after_four_outcomes(monkeypatch) -> None:
    from src.causal_dual_memory_cell_d_v1 import core
    from src.causal_dual_memory_cell_d_v1 import source_adapter

    roster = _roster()
    # Invalid rows must carry assignment -1 in the accepted Stage-0 topology.
    assignment = np.asarray([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int64)
    valid = np.ones(8, dtype=np.bool_)
    group = core.ComplementaryGroups(
        channel_ids=np.arange(8, dtype=np.int64),
        assignment=assignment,
        valid_mask=valid,
    )
    materials = tuple(physical.Strict27SessionMaterial(
        session_id=session,
        ordered_rewarded_trial_ids=tuple(f"{session}-trial-{index:02d}" for index in range(60)),
        trial_views_by_id={f"{session}-trial-{index:02d}": object() for index in range(60)},
        source_theta_topology=physical.SourceThetaTopology(group.channel_ids, group.valid_mask),
        m4_d_optimal_indices=(0, 8, 16, 24), raw_t4_channel_order_sha256="c" * 64,
        theta_valid_mask_sha256="d" * 64, theta_invalid_unit_count=0, source_descriptor_sha256="e" * 64,
    ) for session in roster)
    events: list[str] = []

    class Provider:
        def prebind_train_only(self, *, manifest_roster, open_roster):
            events.append("prebind")
            return {"strict_train_roster": list(manifest_roster), "physical_open_roster": list(open_roster),
                    "nontrain_path_or_unit_counter_called": False,
                    "val_test_paths_constructed": False, "nonphysical_source_paths_resolved": False}
        def open_train_sessions(self, *, roster):
            events.append("open")
            assert tuple(roster) == (execute.SOURCE_SMOKE_SESSION,)
            return materials[:1]
        def join_audit_true_direction(self, *, session_id, trial_id):
            events.append("truth")
            return source_adapter.AuditOnlyTrueDirection(session_id, trial_id, 0.0)
        def close(self): events.append("provider-close")

    class Executor:
        def load_strict_sealed_swa(self, *, root, swa_bytes, selected_device, flags):
            events.append("model")
            flags.checkpoint_opened = True; flags.cuda_initialized = True
            return {
                "runtime_environment": _runtime_attestation(),
                "sealed_swa_load_proof": _sealed_swa_load_proof(),
            }
        def execute_completed_trial(self, *, material, budget, trial_id, support_trial_ids, flags):
            events.append("forward")
            flags.model_forward_calls += 4
            return physical.FinalizedFourGroupPseudo(
                material.session_id, trial_id, (0, 0, 0, 0), (None, None, None, None), "f" * 64,
                _memory_transition_evidence(budget=budget),
            )
        def begin_session_budget(self, *, material, budget, support_trial_ids, flags):
            return {
                "session_id": material.session_id, "budget": budget,
                "support_trial_ids": list(support_trial_ids), "fresh_for_session_budget": True,
                "inherits_prior_session_or_budget_state": False,
                "initial_activity_memory_sha256": "1" * 64,
                "initial_carrier_state_sha256": "2" * 64,
                "activity_fifo_capacity": 0 if budget == 30 else 30 - budget,
                "initial_activity_query_count": 0,
                    "initial_carrier_recipe": "fixed_ridge_by_trial",
                    "initial_carrier_support_rows": budget,
                    "raw_m30_t4_used_as_initializer": False,
                    "initial_support_rate_domain": execute.INITIAL_SUPPORT_RATE_DOMAIN,
                    "online_update_rate_domain": execute.ONLINE_UPDATE_RATE_DOMAIN,
                    "initial_support_rates_sha256": "7" * 64,
                    "initial_support_exposure_seconds_sha256": "8" * 64,
                    "initial_carrier_parity": {"mode": "fixed_ridge_by_trial"},
                "initial_carrier_sha256": "3" * 64,
                "budget_groups_sha256": "4" * 64,
                "budget_group_assignment_sha256": "5" * 64,
                "budget_group_valid_mask_sha256": "6" * 64,
            }
        def groups_for_session_budget(self, *, material, budget):
            return group
        def resources(self): return {}
        def close(self): events.append("executor-close")

    body = _manifest(roster)
    assets = {asset.label: execute.BoundAsset(asset, body if asset.label == "strict_manifest" else b"x", 1, 2)
              for asset in execute.FIXED_ASSETS}
    monkeypatch.setattr(execute, "descriptor_read_fixed_assets", lambda root: assets)
    identity = _identity()
    backend = physical.PhysicalSourceExecutionBackend(provider=Provider(), executor=Executor())
    flags = execute.RuntimeFlags()
    preflight = backend.preflight(root=Path("/synthetic"), identity=identity, flags=flags)
    assert preflight["train_only_provider_not_called"] is True and events == []
    runtime = backend.prepare(root=Path("/synthetic"), identity=identity, flags=flags)
    authority = backend.source_authority(runtime, identity=identity, flags=flags)
    assert authority["sessions"][0]["invalid_unit_count"] == 0
    smoke = backend.run_smoke(runtime, identity=identity, flags=flags)
    assert smoke["audit_positions"] == [30, 31] and events.index("truth") > events.index("forward")
    backend.close(runtime)
    assert events[-2:] == ["executor-close", "provider-close"]


def test_physical_prebind_rejects_nontrain_and_runtime_profile_drift(monkeypatch) -> None:
    identity = _identity()
    backend = physical.PhysicalSourceExecutionBackend(provider=object(), executor=object())
    monkeypatch.setattr(execute, "descriptor_read_fixed_assets", lambda root: {})
    with pytest.raises(execute.SourceExecutionError):
        backend.preflight(root=Path("/synthetic"), identity=identity, flags=execute.RuntimeFlags())
    wrong = _runtime_attestation(); wrong["torch_total_memory_bytes"] = 1
    with pytest.raises(execute.SourceExecutionError, match="runtime"):
        execute.validate_runtime_attestation(_profile(), wrong)


def test_static_cli_imports_no_torch_writes_nothing_and_rejects_public_execution(tmp_path: Path) -> None:
    script = ROOT / "tfpd_exploration/scripts/run_causal_dual_memory_cell_d_source_gate.py"
    code = (
        "import importlib.util,sys;"
        f"s=importlib.util.spec_from_file_location('cdm_exec_cli',{str(script)!r});"
        "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);m.main(['--dry-run']);"
        "print('TORCH_IMPORTED='+str('torch' in sys.modules))"
    )
    completed = subprocess.run(
        [sys.executable, "-S", "-c", code], cwd=tmp_path,
        env={**os.environ, "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": ""},
        text=True, capture_output=True, check=True,
    )
    assert "TORCH_IMPORTED=False" in completed.stdout and list(tmp_path.iterdir()) == []
    rejected = subprocess.run([sys.executable, script, "--execute"], text=True, capture_output=True)
    assert rejected.returncode == 2 and "capability" in rejected.stderr
