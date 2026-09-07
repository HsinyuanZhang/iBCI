"""Focused CPU/synthetic gates for POSTERIOR_MARGINALIZED_CELL_D_SEED42.

This suite never resolves an NWB, target/within/external/formal surface,
checkpoint tensor, CUDA device, GPU, authority root, or result root.  It
tests the one allowed treatment mechanically: cached sampled posterior side
features during source training, with the ordinary sealed OLS carrier held at
inference.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest
import torch

from src.posterior_carrier_v1 import core as posterior
from src.posterior_marginalized_cell_d_v1 import plan
from src.posterior_marginalized_cell_d_v1.core import (
    PMCError,
    PosteriorMarginalizedSideCache,
    PosteriorPrefixInputs,
    PosteriorSideReplacingIterator,
    assert_joint_unit_permutation_equivariance,
    degenerate_sampled_side,
    validate_sealed_ordinary_ols_normalizer,
)
from src.posterior_marginalized_cell_d_v1.lifecycle import (
    PMCLifecycleError,
    PosteriorMarginalizedEqualSessionAdapter,
    deferred_execution_message,
    require_no_direct_equal_session_lifecycle_reuse,
)
from src.posterior_marginalized_cell_d_v1.physical import (
    GPU0_AUTHORITY,
    GPU1_AUTHORITY,
    validate_future_compatible_device_environment,
    validate_future_gpu1_environment,
)
from src.posterior_marginalized_cell_d_v1.runner import (
    PMC_FULL_TRAIN_SPEC,
    PMC_SOURCE_SMOKE_SPEC,
    PMCArtifactRoot,
    PMCIdentity,
    PMCRunnerError,
    _issue_root_review_capability_for_audited_route,
    _validate_source_authority,
    execute_authorized,
    validate_terminal_payload,
)
from src.posterior_marginalized_cell_d_v1.source_audit import (
    ApprovedPhaseBV2Authority,
    PMCLivePrefixMaterial,
    bind_live_inputs_to_approved_phase_b_v2,
    descriptive_source_calibration_audit,
    load_approved_phase_b_v2_authority_from_full_import,
)
from src.posterior_carrier_v1 import phase_b_v2, source_adapter_v2


ROOT = Path(__file__).resolve().parents[2]


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _roster(count: int = 27) -> tuple[str, ...]:
    return tuple(f"source-{index:02d}" for index in range(count))


def _sealed_normalizer(*, dtype: torch.dtype = torch.float32) -> posterior.FrozenSourceT4Normalizer:
    return posterior.FrozenSourceT4Normalizer(
        mean=torch.tensor(plan.SEALED_OLS_T4_MEAN_FLOAT32, dtype=dtype),
        std=torch.tensor(plan.SEALED_OLS_T4_STD_FLOAT32, dtype=dtype),
        authority_sha256=plan.SEALED_OLS_T4_NORMALIZER_SHA256,
    )


def _prefix_inputs(*, session: str, units: int = 5, dtype: torch.dtype = torch.float32) -> PosteriorPrefixInputs:
    counts: dict[int, torch.Tensor] = {}
    exposures: dict[int, torch.Tensor] = {}
    theta: dict[int, torch.Tensor] = {}
    offset = int(session.rsplit("-", 1)[-1]) if session.rsplit("-", 1)[-1].isdigit() else 0
    for budget in plan.BUDGETS:
        theta_row = torch.arange(budget, dtype=dtype) * (2.0 * torch.pi / budget)
        base = torch.arange(units * budget, dtype=torch.int64).reshape(units, budget)
        counts[budget] = (base + offset + 1).remainder(7) + 1
        exposures[budget] = torch.linspace(0.6, 1.6, budget, dtype=dtype)
        theta[budget] = theta_row
    return PosteriorPrefixInputs(
        counts_by_budget=counts,
        exposure_by_budget=exposures,
        theta_by_budget=theta,
        unit_order_sha256=_sha(f"unit-order/{session}"),
        source_file_sha256=_sha(f"source-file/{session}"),
        prefix_row_ids_sha256_by_budget={budget: _sha(f"prefix-rows/{session}/m{budget}") for budget in plan.BUDGETS},
        theta_recovery_evidence_sha256=_sha(f"theta-recovery/{session}"),
        theta_recovery_closure_sha256=_sha("approved-theta-recovery-v2-closure"),
    )


def _prior(roster: tuple[str, ...], *, units: int = 5, dtype: torch.dtype = torch.float32) -> posterior.SourcePrior:
    raw = torch.stack((
        torch.linspace(-1.0, 1.0, units, dtype=dtype),
        torch.linspace(1.2, -0.7, units, dtype=dtype),
        torch.linspace(0.3, 2.0, units, dtype=dtype),
        torch.linspace(-0.5, 0.9, units, dtype=dtype),
    ), dim=-1)
    # The magnitude column is not used to fit the prior moment, but keep raw
    # T4 structurally correct for its receipt digest.
    raw[:, 2] = torch.linalg.vector_norm(raw[:, :2], dim=-1)
    return posterior.SourcePrior.from_raw_m30_t4(raw, roster)


def _cache(*, roster: tuple[str, ...] | None = None, units: int = 5) -> PosteriorMarginalizedSideCache:
    roster = _roster() if roster is None else roster
    return PosteriorMarginalizedSideCache(
        roster=roster,
        prefix_inputs_by_session={session: _prefix_inputs(session=session, units=units) for session in roster},
        prior=_prior(roster, units=units),
        sealed_ols_normalizer=_sealed_normalizer(),
        seed=42,
    )


def test_real_sealed_ols_float32_literals_recompute_authority_and_reject_value_or_dtype_drift() -> None:
    normalizer = _sealed_normalizer()
    assert validate_sealed_ordinary_ols_normalizer(normalizer) is normalizer
    assert plan.sealed_ols_t4_normalizer_sha256() == plan.SEALED_OLS_T4_NORMALIZER_SHA256
    with pytest.raises(PMCError, match="numeric literal"):
        validate_sealed_ordinary_ols_normalizer(posterior.FrozenSourceT4Normalizer(
            mean=normalizer.mean + torch.tensor((1e-5, 0.0, 0.0, 0.0)), std=normalizer.std,
            authority_sha256=normalizer.authority_sha256,
        ))
    with pytest.raises(PMCError, match="float32"):
        validate_sealed_ordinary_ols_normalizer(posterior.FrozenSourceT4Normalizer(
            mean=normalizer.mean.to(torch.float64), std=normalizer.std.to(torch.float64),
            authority_sha256=normalizer.authority_sha256,
        ))


def _build_actual_cell_d_without_leaking_rng() -> torch.nn.Module:
    """CPU graph only; restore model construction's deliberate seed reset."""
    import random
    import numpy as np
    from src.tfpd_lane.pop_robust import build_population_robustness_model

    py_state, np_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    try:
        model = build_population_robustness_model(seed=42, cell="D")
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)
    return model


def test_static_dry_cli_is_stdlib_only_and_refuses_execution() -> None:
    script = ROOT / "tfpd_exploration/scripts/run_posterior_marginalized_cell_d_seed42.py"
    environment = {**os.environ, "PYTHONNOUSERSITE": "1", "PYTHONPATH": "", "CUDA_VISIBLE_DEVICES": ""}
    code = (
        "import importlib.util, json, sys; "
        f"spec=importlib.util.spec_from_file_location('pmc_dry', {str(script)!r}); "
        "module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); "
        "payload=module._load_plan().dry_plan(); "
        "print(json.dumps({'torch_loaded':'torch' in sys.modules,'status':payload['status']}))"
    )
    imported = subprocess.run([sys.executable, "-S", "-c", code], env=environment, capture_output=True, text=True, check=True)
    assert json.loads(imported.stdout) == {
        "torch_loaded": False,
        "status": "DRY_NO_DATA_NO_NWB_NO_CHECKPOINT_NO_CUDA_NO_GPU_NO_WRITE_NO_LAUNCH",
    }
    rendered = subprocess.run([sys.executable, "-S", str(script), "--dry-run"], env=environment, capture_output=True, text=True, check=True)
    payload = json.loads(rendered.stdout)
    assert payload["workorder"]["sha256"] == plan.WORKORDER_SHA256
    assert payload["training_spec"]["total_steps"] == 1_628_400
    rejected = subprocess.run([sys.executable, "-S", str(script), "--execute"], env=environment, capture_output=True, text=True)
    assert rejected.returncode != 0
    assert "only --dry-run" in rejected.stderr


def test_exact_48x27_rotation_is_balanced_without_rng() -> None:
    before = torch.get_rng_state().clone()
    schedule = plan.build_budget_schedule()
    assert torch.equal(before, torch.get_rng_state())
    assert len(schedule) == 48 and all(len(row) == 27 for row in schedule)
    assert schedule[0][:6] == (4, 10, 30, 4, 10, 30)
    assert schedule[1][:6] == (10, 30, 4, 10, 30, 4)
    for session_index in range(27):
        values = [row[session_index] for row in schedule]
        assert {budget: values.count(budget) for budget in plan.BUDGETS} == {4: 16, 10: 16, 30: 16}
    assert len(plan.budget_schedule_sha256(schedule)) == 64


def test_cache_fits_once_per_session_budget_then_samples_once_per_session_epoch_without_global_rng() -> None:
    cache = _cache()
    before = posterior.host_rng_fingerprint()
    cache.fit_all_source_posteriors()
    entries = cache.prewarm_epoch(0)
    posterior.assert_host_rng_unchanged(before, posterior.host_rng_fingerprint())
    assert len(entries) == 27
    assert [entry.budget for entry in entries[:6]] == [4, 10, 30, 4, 10, 30]
    observed = cache.observer()
    assert observed.posterior_fit_calls == 27 * 3
    assert observed.posterior_inverse_calls == 27 * 3
    assert observed.sampled_view_builds == 27
    assert observed.cached_session_epochs == 27
    assert observed.batch_loop_posterior_fit_calls == 0
    assert observed.batch_loop_inverse_calls == 0
    assert observed.batch_loop_sampling_calls == 0
    assert observed.batch_loop_normalizer_fit_calls == 0
    receipt = cache.epoch_receipt_payload(0)
    assert receipt["sealed_ordinary_ols_normalizer_sha256"] == plan.SEALED_OLS_T4_NORMALIZER_SHA256
    assert receipt["posterior_normalizer_used"] is False
    assert all(row["consumer_cell"] == plan.CELL for row in receipt["rows"])
    assert all(row["posterior_sampling_domain_cell"] == posterior.CELL for row in receipt["rows"])
    assert all(row["sampled"] is True for row in receipt["rows"])


def test_full_rotation_builds_1296_cached_samples_and_exact_16_per_budget_per_source() -> None:
    cache = _cache()
    cache.fit_all_source_posteriors()
    all_entries = []
    for epoch in range(48):
        all_entries.extend(cache.prewarm_epoch(epoch))
    assert len(all_entries) == 48 * 27
    observed = cache.observer()
    assert observed.posterior_fit_calls == 27 * 3
    assert observed.sampled_view_builds == 48 * 27
    for session_index, session in enumerate(cache.roster):
        values = [entry.budget for entry in all_entries if entry.session == session]
        assert {budget: values.count(budget) for budget in plan.BUDGETS} == {4: 16, 10: 16, 30: 16}
        assert values[0] == plan.budget_for(0, session_index)


def test_cache_rejects_posterior_normalizer_or_unprepared_batch() -> None:
    roster = _roster(1)
    raw = {budget: {roster[0]: torch.ones(3, 4, dtype=torch.float64) * (budget + 1)} for budget in plan.BUDGETS}
    posterior_normalizer = posterior.PosteriorSourceT4Normalizer.fit(source_roster=roster, raw_mean_t4_by_budget=raw)
    with pytest.raises(PMCError, match="ordinary OLS"):
        PosteriorMarginalizedSideCache(
            roster=roster,
            prefix_inputs_by_session={roster[0]: _prefix_inputs(session=roster[0], units=3)},
            prior=_prior(roster, units=3),
            sealed_ols_normalizer=posterior_normalizer,  # type: ignore[arg-type]
        )
    cache = _cache(roster=roster, units=3)
    with pytest.raises(PMCError, match="prewarmed"):
        cache.entry_for_optimizer_batch(session=roster[0], epoch=0)


def test_theta_recovery_binding_and_float64_sample_to_float32_sealed_ols_boundary_are_explicit() -> None:
    roster = _roster(1)
    inputs = _prefix_inputs(session=roster[0], units=3, dtype=torch.float64)
    with pytest.raises(PMCError, match="same-prefix theta recovery"):
        replace(inputs, theta_recovery_semantics="invented-later-row-recovery")
    cache = PosteriorMarginalizedSideCache(
        roster=roster,
        prefix_inputs_by_session={roster[0]: inputs},
        prior=_prior(roster, units=3, dtype=torch.float64),
        sealed_ols_normalizer=_sealed_normalizer(dtype=torch.float32),
    )
    cache.fit_all_source_posteriors()
    entry = cache.prewarm_epoch(0)[0]
    assert entry.view.raw_t4.dtype == torch.float64
    assert entry.view.normalized_t4.dtype == torch.float32
    normalizer = cache.sealed_ols_normalizer
    expected = ((entry.view.raw_t4 - normalizer.mean.to(torch.float64)) / normalizer.std.to(torch.float64)).to(torch.float32)
    assert torch.equal(entry.view.normalized_t4, expected)
    boundary = entry.payload()["normalization_numeric_boundary"]
    assert boundary == {
        "raw_sample_dtype": "torch.float64",
        "sealed_ols_moments_promoted_to_raw_dtype": "torch.float64",
        "model_side_dtype": "torch.float32",
        "single_cast_to_sealed_cell_d_side_dtype": True,
        "posterior_specific_moments_used": False,
    }
    side = cache.side_for_optimizer_batch(
        session=roster[0], epoch=0, template_side=torch.zeros(2, 3, 4, dtype=torch.float32),
        expected_device=torch.device("cpu"),
    )
    assert side.dtype == torch.float32 and side.shape == (2, 3, 4)


def test_degenerate_zero_covariance_sampled_side_is_exact_ordinary_side_and_real_cell_d_forward() -> None:
    normalizer = _sealed_normalizer()
    beta = torch.tensor(((0.5, -0.25, 0.7), (-0.8, 0.2, -0.4), (0.1, 0.3, 0.25)), dtype=torch.float32)
    zero = torch.zeros(3, dtype=torch.bool)
    raw, sampled_side = degenerate_sampled_side(posterior_mean_beta=beta, zero_spike_mask=zero, normalizer=normalizer)
    ordinary = normalizer.normalize_raw(posterior.beta_to_raw_t4(beta, zero))
    assert torch.equal(sampled_side, ordinary)
    assert torch.equal(raw, posterior.beta_to_raw_t4(beta, zero))

    model = _build_actual_cell_d_without_leaking_rng().eval()
    from torch.nn.parameter import UninitializedParameter

    live_parameters = sum(
        int(parameter.numel()) for parameter in model.parameters()
        if not isinstance(parameter, UninitializedParameter)
    )
    lazy_keys = tuple(sorted(
        name for name, parameter in model.named_parameters()
        if isinstance(parameter, UninitializedParameter)
    ))
    assert live_parameters == plan.SEALED_CELL_D_INITIALIZED_PARAMETERS
    assert lazy_keys == plan.SEALED_CELL_D_LAZY_KEYS
    assert model.decoder.transformer.layers[0].cross_attn.num_heads == 2
    assert model.decoder.dynamic_dropout is True
    assert model.decoder.dynamic_dropout_low == 0.0 and model.decoder.dynamic_dropout_high == 1.0
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8,
        weight_decay=0.0, amsgrad=False,
    )
    assert optimizer.defaults == {
        "lr": 1e-4, "betas": (0.9, 0.999), "eps": 1e-8,
        "weight_decay": 0.0, "amsgrad": False, "maximize": False,
        "foreach": None, "capturable": False, "differentiable": False, "fused": None,
    }
    neural = torch.randn(1, 50, 3)
    calibration = torch.randn(1, 30, 100, 3)
    side = ordinary.unsqueeze(0)
    from src.tfpd_lane.arm_common import state_sha256

    before = state_sha256(model)
    with torch.no_grad():
        baseline, identity_a = model(neural, calib_trials=calibration, side_features=side)
        treatment, identity_b = model(neural, calib_trials=calibration, side_features=sampled_side.unsqueeze(0))
    assert torch.equal(identity_a, identity_b)
    assert torch.equal(baseline, treatment)
    assert state_sha256(model) == before
    assert not torch.cuda.is_initialized()


def test_nonzero_posterior_changes_only_side_and_batch_loop_has_no_posterior_ops() -> None:
    roster = _roster(1)
    cache = _cache(roster=roster, units=4)
    cache.fit_all_source_posteriors()
    cache.prewarm_epoch(0)
    neural = torch.randn(2, 50, 4)
    behavior = torch.randn(2, 50, 2)
    calibration = torch.randn(2, 30, 100, 4)
    ordinary_side = torch.randn(2, 4, 4)
    from src.tfpd_lane.arm_common import state_sha256

    model = _build_actual_cell_d_without_leaking_rng()
    state_before = state_sha256(model)
    batch: tuple[Any, ...] = (neural, behavior, calibration, [roster[0], roster[0]], ordinary_side, "held-tail")
    injected = next(PosteriorSideReplacingIterator(
        iter((batch,)), cache=cache, epoch=0, expected_device=torch.device("cpu"),
    ))
    assert injected[0] is neural and injected[1] is behavior and injected[2] is calibration and injected[3] is batch[3]
    assert injected[5] == "held-tail"
    assert injected[4].shape == ordinary_side.shape
    assert not torch.equal(injected[4], ordinary_side)
    assert state_sha256(model) == state_before
    observer = cache.observer()
    assert observer.batch_loop_side_lookups == 1
    observer.assert_batch_loop_clean()


def test_pre_materialized_non_cpu_side_replaces_cpu_loader_side_without_batch_conversion() -> None:
    """Catch the real DataLoader order: ordinary side is CPU before train_step."""
    roster = _roster(1)
    cache = _cache(roster=roster, units=3)
    cache.fit_all_source_posteriors()
    cache.prewarm_epoch(0)
    # This pure contract uses device descriptors rather than allocating CUDA:
    # loader-side CPU storage is intentionally not compared with the prepared
    # cuda:0 side.  The inherited train_step's later ``.to(cuda:0)`` will be
    # a no-op for the prepared side.
    cache._validate_prepared_side_contract(
        template_shape=(2, 3, 4), template_dtype=torch.float32,
        prepared_shape=(2, 3, 4), prepared_dtype=torch.float32,
        prepared_device=torch.device("cuda:0"), expected_device=torch.device("cuda:0"),
    )
    with pytest.raises(PMCError, match="device conversion"):
        cache._validate_prepared_side_contract(
            template_shape=(2, 3, 4), template_dtype=torch.float32,
            prepared_shape=(2, 3, 4), prepared_dtype=torch.float32,
            prepared_device=torch.device("cpu"), expected_device=torch.device("cuda:0"),
        )
    batch = (
        torch.zeros(2, 50, 3), torch.zeros(2, 50, 2), torch.zeros(2, 30, 100, 3),
        [roster[0], roster[0]], torch.zeros(2, 3, 4, dtype=torch.float32),
    )
    injected = next(PosteriorSideReplacingIterator(
        iter((batch,)), cache=cache, epoch=0, expected_device=torch.device("cpu"),
    ))
    assert injected[4].device.type == "cpu"
    assert injected[4].dtype == batch[4].dtype and injected[4].shape == batch[4].shape
    assert injected[0] is batch[0] and injected[1] is batch[1] and injected[2] is batch[2]
    assert cache.observer().batch_loop_side_lookups == 1
    assert not torch.cuda.is_initialized()


def test_joint_unit_permutation_preserves_actual_cell_d_equivariance_and_cached_side_alignment() -> None:
    roster = _roster(1)
    cache = _cache(roster=roster, units=5)
    cache.fit_all_source_posteriors()
    entry = cache.prewarm_epoch(0)[0]
    model = _build_actual_cell_d_without_leaking_rng().eval()
    neural = torch.randn(1, 50, 5)
    calibration = torch.randn(1, 30, 100, 5)
    ordinary = torch.randn(1, 5, 4)
    permutation = torch.tensor((4, 1, 3, 0, 2), dtype=torch.long)
    permuted_neural, permuted_calibration, _permuted_ordinary = assert_joint_unit_permutation_equivariance(
        cache_entry=entry, permutation=permutation, neural=neural, calibration=calibration, ordinary_side=ordinary,
    )
    side = entry.view.normalized_for_batch(1)
    side_permuted = entry.view.joint_permute(permutation).normalized_for_batch(1)
    with torch.no_grad():
        output, identity = model(neural, calib_trials=calibration, side_features=side)
        permuted_output, permuted_identity = model(
            permuted_neural, calib_trials=permuted_calibration, side_features=side_permuted,
        )
    torch.testing.assert_close(permuted_identity, identity[:, permutation, :], rtol=0.0, atol=2e-6)
    torch.testing.assert_close(permuted_output, output, rtol=0.0, atol=2e-6)


class _FakeEqualSessionBackend:
    def __init__(self, batch: Any) -> None:
        self.batch = batch
        self.prewarm_was_complete = False
        self.observed_batch: Any | None = None

    def begin_epoch(self, runtime: dict[str, Any], epoch: int, flags: Any) -> dict[str, object]:
        self.prewarm_was_complete = bool(runtime["cache"].observer().cached_session_epochs == len(runtime["cache"].roster))
        runtime["epoch_iterator"] = iter((self.batch,))
        return {"epoch": epoch, "one_iterator": True}

    def train_step(self, runtime: dict[str, Any], *, global_step: int, expected_lr: float,
                   require_full_proof: bool, flags: Any) -> dict[str, object]:
        self.observed_batch = next(runtime["epoch_iterator"])
        return {"global_step": global_step, "lr": expected_lr, "full": require_full_proof}


def test_equal_session_adapter_prewarm_precedes_inherited_iterator_and_delegates_train_step() -> None:
    roster = _roster(1)
    cache = _cache(roster=roster, units=3)
    batch = (
        torch.randn(2, 50, 3), torch.randn(2, 50, 2), torch.randn(2, 30, 100, 3),
        [roster[0], roster[0]], torch.zeros(2, 3, 4),
    )
    base = _FakeEqualSessionBackend(batch)
    adapter = PosteriorMarginalizedEqualSessionAdapter(base=base, cache=cache)
    runtime: dict[str, Any] = {"epoch_iterator": None, "cache": cache, "device": torch.device("cpu")}
    evidence = adapter.begin_epoch(runtime, 0, flags=object())
    assert base.prewarm_was_complete
    assert evidence["pmc_prepared_before_inherited_iterator"] is True
    assert evidence["pmc_only_changed_batch_field"] == "side_features_index_4"
    step = adapter.train_step(runtime, global_step=0, expected_lr=1e-5, require_full_proof=False, flags=object())
    assert step["global_step"] == 0
    assert base.observed_batch is not None
    assert base.observed_batch[0] is batch[0]
    assert not torch.equal(base.observed_batch[4], batch[4])
    assert adapter.cache_evidence()["batch_loop_inverse_calls"] == 0


def test_adapter_rejects_preexisting_iterator_and_direct_old_lifecycle_identity_is_disclosed() -> None:
    cache = _cache(roster=_roster(1), units=2)
    adapter = PosteriorMarginalizedEqualSessionAdapter(base=_FakeEqualSessionBackend(tuple()), cache=cache)
    with pytest.raises(PMCLifecycleError, match="prewarm"):
        adapter.begin_epoch({"epoch_iterator": iter(())}, 0, flags=object())
    require_no_direct_equal_session_lifecycle_reuse()
    assert "requires a separate reviewed strict-27 source authority" in deferred_execution_message()


def test_plan_closure_is_explicit_and_rejects_leaf_or_aggregate_drift() -> None:
    closure = plan.implementation_closure(ROOT)
    validated = plan.validate_implementation_closure(closure)
    assert validated["closure_sha256"] == closure["closure_sha256"]
    missing = {**closure, "sha256_by_path": dict(closure["sha256_by_path"])}
    missing["sha256_by_path"].pop(plan.IMPLEMENTATION_CLOSURE[-1])
    with pytest.raises(plan.PMCPlanError, match="topology"):
        plan.validate_implementation_closure(missing)
    drift = {**closure, "sha256_by_path": dict(closure["sha256_by_path"])}
    drift["sha256_by_path"][plan.IMPLEMENTATION_CLOSURE[1]] = "0" * 64
    with pytest.raises(plan.PMCPlanError, match="aggregate"):
        plan.validate_implementation_closure(drift)


def test_prospective_roots_are_read_only_and_fresh(tmp_path: Path) -> None:
    assert all(plan.assert_fresh_prospective_roots(tmp_path).values())
    existing = tmp_path / plan.SOURCE_SMOKE_ROOT_RELATIVE
    existing.mkdir(parents=True)
    with pytest.raises(plan.PMCPlanError, match="not fresh"):
        plan.assert_fresh_prospective_roots(tmp_path)


def test_spec_scoped_freshness_allows_full_after_independent_smoke_root_exists(tmp_path: Path) -> None:
    smoke = tmp_path / plan.SOURCE_SMOKE_ROOT_RELATIVE
    smoke.mkdir(parents=True)
    assert plan.assert_fresh_prospective_roots(tmp_path, spec_kind="full_train") == {
        plan.FULL_TRAIN_ROOT_RELATIVE: True,
    }
    with pytest.raises(plan.PMCPlanError, match="not fresh"):
        plan.assert_fresh_prospective_roots(tmp_path, spec_kind="source_smoke")


def _v2_roster() -> tuple[str, ...]:
    return ("sub-C_ses-CO-20150313",) + tuple(f"strict-{index:02d}" for index in range(26))


def _live_v2_material(session: str, *, special_fallback: bool) -> PMCLivePrefixMaterial:
    prefix: list[dict[str, object]] = []
    corners: dict[int, object] = {}
    for index in range(30):
        trial = 33 if special_fallback and index == 0 else 1000 + index
        direction = -3.0 * torch.pi / 4.0 + (index % 8) * torch.pi / 4.0
        row: dict[str, object] = {"trial_index": trial, "target_dir": float(direction)}
        if special_fallback and index == 0:
            row["target_dir"] = float("nan")
            centre = float(8.0 / (2.0 ** 0.5))
            corners[trial] = [-(centre + 1.0), centre - 1.0, -(centre - 1.0), centre + 1.0]
        prefix.append(row)
    theta, row_ids, evidence = source_adapter_v2.recover_same_prefix_theta(
        session=session, prefix=prefix, target_corners_by_trial_index=corners,
    )
    units = 3
    counts = (torch.arange(units * 30, dtype=torch.int64).reshape(units, 30) % 7) + 1
    exposure = torch.linspace(0.5, 1.5, 30, dtype=torch.float64)
    raw = torch.stack((
        torch.linspace(-0.2, 0.3, units, dtype=torch.float64),
        torch.linspace(0.4, -0.1, units, dtype=torch.float64),
        torch.linspace(0.5, 0.7, units, dtype=torch.float64),
        torch.linspace(1.0, 2.0, units, dtype=torch.float64),
    ), dim=-1)
    unit_order = torch.arange(units, dtype=torch.int64)
    proof = {
        "feature_group": "t4", "feature_version": 1, "pool_size": 30, "signal_view": "sua",
        "source_unit_count": units, "channel_ids_are_exact_arange": True, "raw_row_count": units,
        "unit_order_sha256": posterior.tensor_digest(unit_order),
        "row_semantics": "closure_bound_compute_unit_side_features_uncached_sua_rows_follow_nwb_units_order",
    }
    return PMCLivePrefixMaterial(
        session=session, raw_m30_t4=raw, counts_m30=counts, exposure_m30=exposure, theta_m30=theta,
        prefix_row_ids=row_ids, source_file_sha256=_sha(f"source/{session}"),
        unit_order_sha256=proof["unit_order_sha256"], raw_t4_row_order_proof=proof,
        theta_recovery_evidence=evidence,
    )


def _synthetic_phase_b_v2_authority(live: dict[str, PMCLivePrefixMaterial]) -> dict[str, object]:
    roster = tuple(live)
    recovery = {session: material.theta_recovery_evidence for session, material in live.items()}
    inputs: dict[str, object] = {}
    for session, material in live.items():
        inputs[session] = {
            "raw_m30_t4_sha256": posterior.tensor_digest(material.raw_m30_t4),
            "counts_m30_sha256": posterior.tensor_digest(material.counts_m30),
            "exposure_m30_sha256": posterior.tensor_digest(material.exposure_m30),
            "theta_m30_sha256": posterior.tensor_digest(material.theta_m30),
            "source_path_sha256": material.source_file_sha256,
            "unit_order_sha256": material.unit_order_sha256,
            "prefix_rows_sha256": hashlib.sha256(
                posterior.canonical_json_bytes(list(material.prefix_row_ids))
            ).hexdigest(),
            "prefix_row_ids": list(material.prefix_row_ids),
            "unit_count": material.raw_m30_t4.shape[0],
            "raw_t4_row_order_proof": dict(material.raw_t4_row_order_proof),
            "prefix_evidence_by_budget": {
                str(budget): {
                    "counts_sha256": posterior.tensor_digest(material.counts_m30[:, :budget]),
                    "exposure_sha256": posterior.tensor_digest(material.exposure_m30[:budget]),
                    "theta_sha256": posterior.tensor_digest(material.theta_m30[:budget]),
                }
                for budget in plan.BUDGETS
            },
        }
    return {
        "v1_compatible_authority": {"roster": list(roster), "posterior_inputs": inputs},
        "theta_recovery_by_session": recovery,
        "theta_fallback_topology": source_adapter_v2.theta_fallback_topology(
            roster=roster, recovery_by_session=recovery,
        ),
        "closure": {"closure_sha256": _sha("phase-b-v2-closure")},
    }


def test_phase_b_v2_authority_bridge_calls_full_validator_and_revalidates_same_prefix_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    live = {
        session: _live_v2_material(session, special_fallback=session == "sub-C_ses-CO-20150313")
        for session in _v2_roster()
    }
    authority = _synthetic_phase_b_v2_authority(live)
    called: list[tuple[object, object, object]] = []

    def full_validator(value: Mapping[str, object], *, identity: object, launch_sha256: str) -> dict[str, object]:
        called.append((value, identity, launch_sha256))
        return dict(value)

    monkeypatch.setattr(phase_b_v2, "validate_source_authority_v2", full_validator)
    approved = ApprovedPhaseBV2Authority(
        identity=object(),  # exercised only through the injected full-validator seam in this no-data fixture.
        launch_sha256=_sha("phase-b-v2-launch"), source_authority=authority,
        imported_full_source_authority_sha256=plan.PHASE_B_V2_IMPORT_SOURCE_AUTHORITY_SHA256,
    )
    bound = bind_live_inputs_to_approved_phase_b_v2(approved=approved, live_by_session=live)
    assert len(called) == 1
    assert bound.recovery_topology["body_sha256"] == plan.THETA_FALLBACK_TOPOLOGY_SHA256
    assert bound.payload()["source_rows"]["sub-C_ses-CO-20150313"]["prefix_row_ids"][0].endswith(":trial:33")

    altered = dict(live)
    original = altered["strict-00"]
    altered["strict-00"] = PMCLivePrefixMaterial(
        **{**original.__dict__, "counts_m30": original.counts_m30 + 1},
    )
    with pytest.raises(PMCError, match="live source input"):
        bind_live_inputs_to_approved_phase_b_v2(approved=approved, live_by_session=altered)


def test_actual_imported_full_source_authority_rehydrates_v2_without_source_or_cuda() -> None:
    approved = load_approved_phase_b_v2_authority_from_full_import(ROOT)
    validated = approved.validate()
    assert approved.imported_full_source_authority_sha256 == plan.PHASE_B_V2_IMPORT_SOURCE_AUTHORITY_SHA256
    assert validated["schema"] == "posterior_carrier_source_authority_v2"
    assert approved.identity.payload()["closure"]["closure_sha256"] == "9a692c704f2c31c470e9ee465307c9bb1ea4ec2b73b0d156218e06b387abb853"
    assert not torch.cuda.is_initialized()


def test_imported_source_authority_pair_rejects_sidecar_or_body_substitution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = ROOT / plan.PHASE_B_V2_IMPORT_SOURCE_AUTHORITY_RELATIVE
    body, sidecar = source.read_bytes(), source.with_name("source_authority.json.sha256").read_bytes()
    relative = "mirror/source_authority.json"
    destination = tmp_path / relative
    destination.parent.mkdir(parents=True)
    destination.write_bytes(body)
    destination.chmod(0o444)
    destination.with_name("source_authority.json.sha256").write_bytes(b"0" * len(sidecar))
    destination.with_name("source_authority.json.sha256").chmod(0o444)
    monkeypatch.setattr(plan, "PHASE_B_V2_IMPORT_SOURCE_AUTHORITY_RELATIVE", relative)
    with pytest.raises(PMCError, match="sidecar"):
        load_approved_phase_b_v2_authority_from_full_import(tmp_path)
    destination.with_name("source_authority.json.sha256").chmod(0o644)
    destination.with_name("source_authority.json.sha256").write_bytes(sidecar)
    destination.with_name("source_authority.json.sha256").chmod(0o444)
    destination.chmod(0o644)
    destination.write_bytes(body[:-1] + (b" " if body[-1:] != b" " else b"\n"))
    destination.chmod(0o444)
    with pytest.raises(PMCError, match="body SHA"):
        load_approved_phase_b_v2_authority_from_full_import(tmp_path)
    assert not torch.cuda.is_initialized()


def test_descriptive_source_calibration_audit_is_in_memory_source_only_and_exposes_blockers() -> None:
    roster = _roster(1)
    cache = _cache(roster=roster, units=4)
    ordinary = {roster[0]: torch.zeros(4, 4, dtype=torch.float64)}
    audit = descriptive_source_calibration_audit(cache=cache, ordinary_raw_m30_by_session=ordinary)
    assert audit["writes_result_artifacts"] is False
    assert audit["source_only"] is True
    assert audit["cache_observer"]["batch_loop_inverse_calls"] == 0
    assert audit["hard_raw_abs_bound"] == plan.RAW_SAMPLE_ABS_SAFETY_BOUND
    assert len(audit["per_session_rows"]) == 1
    assert not torch.cuda.is_initialized()


def test_descriptive_audit_discloses_a_local_contraction_departure_without_vetoing_an_aggregate_safe_smoke() -> None:
    """A §6 session-level diagnostic is not silently promoted into a gate.

    This deliberately perturbs only one synthetic session's M30 covariance so
    its local M10→M30 trace increases by a tiny amount.  The three-session
    aggregate still contracts, which is the contract's actual smoke gate.
    The result must therefore retain the session row as a disclosure while
    leaving the source-only smoke structurally eligible.
    """
    roster = _roster(3)
    cache = _cache(roster=roster, units=4)
    cache.fit_all_source_posteriors()
    session = roster[0]
    carrier_m10 = cache._posteriors[session][10]
    carrier_m30 = cache._posteriors[session][30]
    covariance = carrier_m10.covariance * 1.0001
    credibility = posterior.directional_credibility(
        covariance,
        tau_ac2=carrier_m10.prior.tau_ac2,
        zero_spike_mask=carrier_m10.zero_spike_mask,
    )
    cache._posteriors[session][30] = replace(
        carrier_m30,
        covariance=covariance,
        credibility=credibility,
        attention_bias=torch.log(credibility),
    )

    audit = descriptive_source_calibration_audit(
        cache=cache,
        ordinary_raw_m30_by_session={item: torch.zeros(4, 4) for item in roster},
    )

    assert audit["per_session_rows"][0]["directional_uncertainty_contracts_M4_ge_M10_ge_M30"] is False
    assert audit["aggregate_directional_uncertainty_contracts_M4_ge_M10_ge_M30"] is True
    assert audit["disclosures"] == [f"{session}:directional_uncertainty_noncontracting"]
    assert audit["smoke_blockers"] == []
    assert audit["smoke_structurally_eligible"] is True
    assert audit["hard_raw_abs_bound"] == 1_000.0
    assert not torch.cuda.is_initialized()


def test_gpu1_static_contract_is_exact_and_does_not_initialize_cuda() -> None:
    payload = validate_future_gpu1_environment({"CUDA_VISIBLE_DEVICES": "1", "CUDA_DEVICE_ORDER": "PCI_BUS_ID"})
    assert payload["gpu_authority"] == GPU1_AUTHORITY
    with pytest.raises(Exception, match="CUDA_VISIBLE_DEVICES"):
        validate_future_gpu1_environment({"CUDA_VISIBLE_DEVICES": "0", "CUDA_DEVICE_ORDER": "PCI_BUS_ID"})
    assert not torch.cuda.is_initialized()


def test_compatible_device_profiles_are_selected_by_exact_profile_not_fixed_ordinal() -> None:
    gpu0 = validate_future_compatible_device_environment(
        GPU0_AUTHORITY, {"CUDA_VISIBLE_DEVICES": "0", "CUDA_DEVICE_ORDER": "PCI_BUS_ID"},
    )
    gpu1 = validate_future_compatible_device_environment(
        GPU1_AUTHORITY, {"CUDA_VISIBLE_DEVICES": "1", "CUDA_DEVICE_ORDER": "PCI_BUS_ID"},
    )
    assert gpu0["gpu_authority"] == GPU0_AUTHORITY
    assert gpu1["gpu_authority"] == GPU1_AUTHORITY
    forged = dict(GPU0_AUTHORITY)
    forged["torch_total_memory_bytes"] = GPU1_AUTHORITY["torch_total_memory_bytes"]
    with pytest.raises(Exception, match="compatible local authority"):
        plan.validate_compatible_device_profile(forged)
    assert not torch.cuda.is_initialized()


class _SmokeLifecycleBackend:
    """Dependency-injected no-data backend for PMC receipt topology tests."""

    def __init__(self, *, fail_prepare: bool = False) -> None:
        self.fail_prepare = fail_prepare
        self.closed = False

    def prepare(self, *, spec: Any, identity: Any, progress: Any) -> dict[str, object]:
        if self.fail_prepare:
            raise RuntimeError("synthetic prepare failure")
        progress.source_opened = True
        return {"runtime": True}

    def source_authority(self, runtime: Any, *, identity: PMCIdentity) -> Mapping[str, object]:
        assert runtime["runtime"] is True
        return {
            "schema": "posterior_marginalized_cell_d_source_authority_v1", "cell": plan.CELL,
            "source_only": True, "target_opened": False, "within_opened": False,
            "external_opened": False, "formal_opened": False,
            "target_optimizer_steps": 0, "target_backward_calls": 0, "target_update_calls": 0,
            "source_binding": _smoke_source_binding(identity),
            "cache_policy": {"fits_and_inverses_before_iterator": True, "samples_before_iterator": True,
                             "device_cache_before_iterator": True, "batch_loop_inverse_calls": 0,
                             "batch_loop_sampling_calls": 0, "batch_loop_normalizer_fit_calls": 0},
            "normalizer": {"semantic_sha256": plan.SEALED_OLS_T4_NORMALIZER_SHA256,
                           "mean_float32": list(plan.SEALED_OLS_T4_MEAN_FLOAT32),
                           "std_float32": list(plan.SEALED_OLS_T4_STD_FLOAT32),
                           "posterior_specific_normalizer_used": False},
            "gpu": _attested_runtime(identity.gpu), "identity": identity.payload(),
        }

    def begin_epoch(self, runtime: Any, *, epoch: int, progress: Any) -> Mapping[str, object]:
        return {"epoch": epoch, "one_iterator": True, "schedule_sha256": _sha(f"schedule/{epoch}")}

    def train_step(self, runtime: Any, *, epoch: int, global_step: int, require_full_proof: bool, progress: Any) -> Mapping[str, object]:
        return {"loss": 1.0, "lr": 1e-5, "dropout_p": 0.25, "full_proof": require_full_proof,
                "batch": {"session": "source-00", "epoch": epoch, "global_step": global_step}}

    def end_epoch(self, runtime: Any, *, epoch: int, rows: Sequence[Mapping[str, object]], progress: Any) -> Mapping[str, object]:
        return {
            "epoch": epoch, "steps": len(rows), "cumulative_optimizer_steps": len(rows) * (epoch + 1),
            "loss": {"mean": 1.0, "min": 1.0, "max": 1.0}, "lr": {"first": 1e-5, "last": 1e-5},
            "dropout": {"p_min": 0.25, "p_max": 0.25, "p_mean": 0.25},
            "cache": {"batch_loop_inverse_calls": 0, "batch_loop_sampling_calls": 0,
                      "batch_loop_normalizer_fit_calls": 0},
            "proof": {"finite_model": True, "finite_optimizer": True,
                      "critical_gradients": {"b3s": True, "decoder": True}},
            "resources": {"synthetic": True},
            "diagnostic_side": "ordinary_ols_point_side__held_auxiliary_diagnostic_not_pmc_training_sample",
        }

    def checkpoint(self, runtime: Any, *, epoch: int, global_step: int, binding: Mapping[str, object]) -> bytes:
        return b"synthetic checkpoint"

    def swa(self, runtime: Any, *, checkpoint_bodies: Mapping[int, bytes], binding: Mapping[str, object]) -> tuple[bytes, Mapping[str, object]]:
        return b"synthetic swa", {"synthetic": True}

    def final_reverify(self, runtime: Any, *, identity: PMCIdentity) -> Mapping[str, object]:
        return identity.closure

    def close(self, runtime: Any | None) -> None:
        self.closed = True


def _smoke_source_binding(identity: PMCIdentity | None = None) -> dict[str, str]:
    return {
        "approved_phase_b_v2_authority_sha256": (
            identity.phase_b_v2_authority_sha256 if identity is not None else _sha("phase-b-authority")
        ),
        "approved_phase_b_v2_closure_sha256": (
            identity.phase_b_v2_closure_sha256 if identity is not None else _sha("phase-b-closure")
        ),
    }


def _attested_runtime(profile: Mapping[str, object]) -> dict[str, object]:
    """Synthetic durable counterpart of the physical device attestation."""
    return {
        **dict(profile),
        "visible_devices": 1,
        "attested": True,
        "torch_cuda_matmul_allow_tf32": False,
        "torch_cudnn_allow_tf32": False,
    }


def _canonical_json_sha(value: Mapping[str, object]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")).hexdigest()


def _pmc_identity(*, gpu: Mapping[str, object] = GPU1_AUTHORITY) -> PMCIdentity:
    binding = _smoke_source_binding()
    return PMCIdentity(
        closure=plan.implementation_closure(ROOT), phase_b_v2_authority_sha256=_sha("phase-b-authority"),
        phase_b_v2_closure_sha256=_sha("phase-b-closure"), source_binding_sha256=_canonical_json_sha(binding),
        gpu=gpu,
    )


def test_pmc_source_smoke_lifecycle_requires_opaque_capability_and_writes_only_atomic_test_receipts(tmp_path: Path) -> None:
    identity = _pmc_identity()
    backend = _SmokeLifecycleBackend()
    capability = _issue_root_review_capability_for_audited_route(
        identity=identity, spec=PMC_SOURCE_SMOKE_SPEC,
    )
    result = execute_authorized(tmp_path, spec=PMC_SOURCE_SMOKE_SPEC, identity=identity,
                                capability=capability, backend=backend)
    root = tmp_path / plan.SOURCE_SMOKE_ROOT_RELATIVE
    assert backend.closed and result["root"] == str(root)
    assert {path.name for path in root.iterdir()} == {
        "attempt.json", "attempt.json.sha256", "launch.json", "launch.json.sha256",
        "source_authority.json", "source_authority.json.sha256", "epoch_00.json", "epoch_00.json.sha256",
        "smoke.json", "smoke.json.sha256", "terminal.json", "terminal.json.sha256",
    }
    with pytest.raises(Exception, match="not fresh"):
        execute_authorized(tmp_path, spec=PMC_SOURCE_SMOKE_SPEC, identity=identity,
                           capability=capability, backend=_SmokeLifecycleBackend())


def test_pmc_lifecycle_prepare_failure_is_honest_and_has_no_terminal(tmp_path: Path) -> None:
    identity = _pmc_identity()
    capability = _issue_root_review_capability_for_audited_route(
        identity=identity, spec=PMC_SOURCE_SMOKE_SPEC,
    )
    with pytest.raises(RuntimeError, match="synthetic prepare failure"):
        execute_authorized(tmp_path, spec=PMC_SOURCE_SMOKE_SPEC, identity=identity,
                           capability=capability, backend=_SmokeLifecycleBackend(fail_prepare=True))
    root = tmp_path / plan.SOURCE_SMOKE_ROOT_RELATIVE
    assert (root / "attempt.json").is_file() and (root / "failure.json").is_file()
    assert not (root / "terminal.json").exists()


def test_source_authority_binds_the_selected_device_and_post_enforcement_tf32_state() -> None:
    identity = _pmc_identity(gpu=GPU0_AUTHORITY)
    backend = _SmokeLifecycleBackend()
    payload = dict(backend.source_authority({"runtime": True}, identity=identity))
    assert _validate_source_authority(payload, identity)["gpu"] == _attested_runtime(GPU0_AUTHORITY)
    for key, forged_value in (
        ("uuid", GPU1_AUTHORITY["uuid"]),
        ("torch_cuda_matmul_allow_tf32", True),
        ("torch_cudnn_allow_tf32", True),
    ):
        forged = {**payload, "gpu": dict(payload["gpu"])}
        forged["gpu"][key] = forged_value
        with pytest.raises(PMCRunnerError, match="runtime evidence"):
            _validate_source_authority(forged, identity)


def test_route_owned_artifact_reload_rejects_post_publish_body_or_sidecar_drift(tmp_path: Path) -> None:
    """Every downstream provenance edge is read from the immutable pair."""
    artifact = PMCArtifactRoot(tmp_path / "pmc-artifacts")
    artifact.reserve()
    try:
        payload = {"schema": "synthetic", "value": 1}
        digest = artifact.publish_json("row.json", payload)
        assert artifact.reload_json("row.json", digest) == payload
        body = artifact.path / "row.json"
        os.chmod(body, 0o644)
        body.write_bytes(b'{"schema":"synthetic","value":2}')
        os.chmod(body, 0o444)
        with pytest.raises(PMCRunnerError, match="digest"):
            artifact.reload_json("row.json", digest)

        # Restore a digest-valid body but forge the independently verified
        # basename-only sidecar: the second half of the durable pair matters.
        os.chmod(body, 0o644)
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        body.write_bytes(encoded)
        os.chmod(body, 0o444)
        sidecar = artifact.path / "row.json.sha256"
        os.chmod(sidecar, 0o644)
        sidecar.write_bytes(f"{'0' * 64}  row.json\n".encode("ascii"))
        os.chmod(sidecar, 0o444)
        with pytest.raises(PMCRunnerError, match="sidecar"):
            artifact.reload_json("row.json", digest)
    finally:
        artifact.close()


def test_terminal_validator_binds_smoke_and_full_artifact_graphs() -> None:
    identity = _pmc_identity(gpu=GPU0_AUTHORITY)
    smoke_epochs = {"0": _sha("epoch0")}
    smoke_artifacts: dict[str, object] = {
        "epoch_sha256": smoke_epochs, "smoke_sha256": _sha("smoke"),
        "checkpoint_sha256": {}, "swa_sha256": None, "swa_manifest_sha256": None, "swa_proof": None,
    }
    smoke_terminal = {
        "schema": "posterior_marginalized_cell_d_terminal_v1", "cell": plan.CELL, "status": "TERMINAL",
        "spec": PMC_SOURCE_SMOKE_SPEC.payload(), "identity": identity.payload(),
        "attempt_sha256": _sha("attempt"), "launch_sha256": _sha("launch"),
        "source_authority_sha256": _sha("source"), "optimizer_steps_completed": 1, "epoch_count": 1,
        "launch_final_closure_equal": True, "source_only": True,
        "target_optimizer_steps": 0, "target_backward_calls": 0, "target_update_calls": 0,
        "artifacts": smoke_artifacts,
    }
    assert validate_terminal_payload(
        smoke_terminal, identity=identity, spec=PMC_SOURCE_SMOKE_SPEC,
        attempt_sha256=_sha("attempt"), launch_sha256=_sha("launch"), source_authority_sha256=_sha("source"),
        epoch_sha256=smoke_epochs, artifacts=smoke_artifacts,
    ) == smoke_terminal
    full_epochs = {str(epoch): _sha(f"epoch/{epoch}") for epoch in range(plan.EPOCHS)}
    checkpoint_sha = {str(epoch): _sha(f"checkpoint/{epoch}") for epoch in plan.CHECKPOINT_EPOCHS}
    full_artifacts: dict[str, object] = {
        "epoch_sha256": full_epochs, "smoke_sha256": None, "checkpoint_sha256": checkpoint_sha,
        "swa_sha256": _sha("swa"), "swa_manifest_sha256": _sha("manifest"), "swa_proof": {"fp64_arithmetic": True},
    }
    full_terminal = {
        **smoke_terminal, "spec": PMC_FULL_TRAIN_SPEC.payload(),
        "optimizer_steps_completed": plan.TOTAL_STEPS, "epoch_count": plan.EPOCHS, "artifacts": full_artifacts,
    }
    assert validate_terminal_payload(
        full_terminal, identity=identity, spec=PMC_FULL_TRAIN_SPEC,
        attempt_sha256=_sha("attempt"), launch_sha256=_sha("launch"), source_authority_sha256=_sha("source"),
        epoch_sha256=full_epochs, artifacts=full_artifacts,
    ) == full_terminal
    forged = dict(full_terminal)
    forged_artifacts = dict(full_artifacts)
    forged_artifacts["swa_manifest_sha256"] = _sha("forged-manifest")
    forged["artifacts"] = forged_artifacts
    with pytest.raises(PMCRunnerError, match="artifact graph"):
        validate_terminal_payload(
            forged, identity=identity, spec=PMC_FULL_TRAIN_SPEC,
            attempt_sha256=_sha("attempt"), launch_sha256=_sha("launch"), source_authority_sha256=_sha("source"),
            epoch_sha256=full_epochs, artifacts=full_artifacts,
        )


def test_actual_equal_session_checkpoint_to_swa_accepts_one_common_pmc_provenance_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the real inherited physical checkpoint/SWA methods on CPU.

    The canonical-state strict verifier is separately exercised by its frozen
    route.  This synthetic CPU contract patches only that external artifact
    lookup, while retaining the real Cell-D graph, actual checkpoint bodies,
    real FP64 SWA accumulation, strict fresh load, and exact common binding.
    """
    from src import cell_d_equal_session_v1 as equal_session
    from src.tfpd_lane import arm_common, pop_robust

    model = _build_actual_cell_d_without_leaking_rng().cpu()
    base = equal_session.PhysicalEqualSessionBackend(ROOT)
    runtime = {
        "torch": torch, "model": model, "optimizer": torch.optim.Adam(model.parameters(), lr=1e-4),
        "arm_common": arm_common, "pop_robust": pop_robust, "device": torch.device("cpu"),
        "fixed": (
            torch.randn(4, 50, 3), torch.randn(4, 50, 2), torch.randn(4, 30, 100, 3),
            ["synthetic"] * 4, torch.randn(4, 3, 4),
        ),
    }
    binding = {
        "cell": plan.CELL, "run_spec": PMC_FULL_TRAIN_SPEC.payload(), "launch_sha256": _sha("launch"),
        "launch_closure_sha256": _sha("closure"),
        "canonical_initial_state_sha256": plan.CANONICAL_INITIAL_STATE_STATE_SHA256,
        "schedule_plan_sha256": "17c1ee6d1ddc62e50aa62c5e25ff800a1612631b4aa36f7508805ce6f83fc98b",
        "source_authority_sha256": _sha("source"), "identity": _pmc_identity().payload(),
    }
    checkpoints = {
        epoch: base.make_checkpoint(runtime, epoch, (epoch + 1) * plan.STEPS_PER_EPOCH, binding).body
        for epoch in plan.CHECKPOINT_EPOCHS
    }
    # This is not a model/format stub: the base's real SWA path will still
    # strict-load the synthetic state into a fresh Cell-D graph.  Only the
    # fixed sealed-artifact hash predicate is replaced for this no-checkpoint
    # test fixture.
    monkeypatch.setattr(
        equal_session, "_strict_recompute_cell_d_state_digest",
        lambda **kwargs: {"state_dict_sha256": kwargs["claimed_sha256"]},
    )
    swa = base.build_swa(runtime, checkpoints, equal_session.FULL_TRAIN_SPEC, binding)
    validated = base.validate_swa(swa.body, spec=equal_session.FULL_TRAIN_SPEC, binding=binding)
    assert validated["binding"] == binding and swa.proof["fp64_arithmetic"] is True
    assert not torch.cuda.is_initialized()


def test_actual_inherited_train_step_preserves_schedule_loss_lr_dropout_and_only_replaces_side() -> None:
    """Exercise the production Cell-D step through the PMC side seam on CPU.

    This is intentionally stronger than the mock adapter test above.  It uses
    the real equal-session batch sampler and ``PhysicalEqualSessionBackend``
    with a real Cell-D graph, Adam optimizer, dense valid-bin MSE, and the
    existing dynamic whole-unit dropout recorder.  The only synthetic inputs
    are the in-memory batch and source inventory; no dataset/NWB/checkpoint or
    CUDA path is touched.
    """
    from src import cell_d_equal_session_v1 as equal_session
    from src.tfpd_lane import arm_common, pop_robust

    roster = _roster(27)
    inventory = equal_session.inventory_from_window_counts(
        roster, {session: tuple(range(plan.BATCH_SIZE)) for session in roster},
    )
    schedule = equal_session.EpochSchedule(spec=equal_session.PUBLIC_SPEC, inventory=inventory, epoch=0)
    sampler = equal_session._OneIteratorEpochBatchSampler(schedule, maximum_batches=1)
    scheduled_indices = next(iter(sampler))
    scheduled = sampler._pending[0]
    assert scheduled.dataset_indices == tuple(scheduled_indices)

    cache = _cache(roster=roster, units=3)
    cache.fit_all_source_posteriors()
    cache.prewarm_epoch(0)
    cache.materialize_epoch_for_device(epoch=0, device=torch.device("cpu"))
    expected_side = cache._epoch_entries[(scheduled.session, 0)].view.normalized_for_batch(plan.BATCH_SIZE)
    neural = torch.randn(plan.BATCH_SIZE, 50, 3)
    behavior = torch.randn(plan.BATCH_SIZE, 50, 2)
    behavior[0, 0] = -1.0  # Explicitly prove dense valid-bin masking, not an all-valid shortcut.
    calibration = torch.randn(plan.BATCH_SIZE, 30, 100, 3)
    ordinary_side = torch.zeros(plan.BATCH_SIZE, 3, 4)
    batch = (neural, behavior, calibration, [scheduled.session] * plan.BATCH_SIZE, ordinary_side)

    class _CaptureIterator:
        def __init__(self, source: Any) -> None:
            self.source = source
            self.batch: Any | None = None

        def __iter__(self) -> "_CaptureIterator":
            return self

        def __next__(self) -> Any:
            self.batch = next(self.source)
            return self.batch

    replacement = PosteriorSideReplacingIterator(
        iter((batch,)), cache=cache, epoch=0, expected_device=torch.device("cpu"),
    )
    captured = _CaptureIterator(replacement)
    model = _build_actual_cell_d_without_leaking_rng().cpu()
    reference = _build_actual_cell_d_without_leaking_rng().cpu()
    reference.load_state_dict(model.state_dict(), strict=True)
    model.train()
    reference.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.0)
    runtime: dict[str, Any] = {
        "torch": torch,
        "model": model,
        "optimizer": optimizer,
        "arm_common": arm_common,
        "pop_robust": pop_robust,
        "device": torch.device("cpu"),
        "epoch_iterator": captured,
        "epoch_sampler": sampler,
    }
    expected_lr = float(arm_common.lr_at_step(0, plan.EPOCHS, plan.STEPS_PER_EPOCH))
    python_rng, torch_rng = random.getstate(), torch.get_rng_state().clone()
    with pop_robust.dynamic_dropout_recorder() as expected_dropout:
        prediction, _ = reference(neural, calib_trials=calibration, side_features=expected_side)
    valid = (behavior != -1.0).all(dim=-1)
    expected_loss = (((prediction - behavior).square().sum(dim=-1) * valid).sum()
                     / (valid.sum() * behavior.shape[-1]))
    # The real train step must see the exact same dropout draws as this
    # reference forward.  The PMC injector consumes no global RNG.
    random.setstate(python_rng)
    torch.set_rng_state(torch_rng)
    before_fc_out = model.decoder.fc_out.weight.detach().clone()
    flags = equal_session.LifecycleFlags(source_train_opened=True)
    base = equal_session.PhysicalEqualSessionBackend(ROOT)
    outcome = base.train_step(
        runtime, global_step=0, expected_lr=expected_lr, require_full_proof=True, flags=flags,
    )

    assert captured.batch is not None
    assert captured.batch[0] is neural and captured.batch[1] is behavior
    assert captured.batch[2] is calibration and captured.batch[3] is batch[3]
    assert torch.equal(captured.batch[4], expected_side)
    assert not torch.equal(captured.batch[4], ordinary_side)
    assert replacement.replaced_batches == 1
    assert sampler.iter_calls == 1 and sampler.emitted_batches == 1 and sampler.observed_batches == 1
    assert outcome.batch_evidence == {
        "session": scheduled.session, "dataset_indices": list(scheduled.dataset_indices),
        "cycle": scheduled.cycle, "epoch": 0, "global_step": 0,
    }
    assert outcome.loss == pytest.approx(float(expected_loss.detach().item()), abs=1e-7)
    assert outcome.lr == expected_lr == optimizer.param_groups[0]["lr"]
    assert outcome.dropout_p == expected_dropout["sampled_p"][0]
    assert expected_dropout["uniform_calls"] == 1 and len(expected_dropout["dropout_calls"]) == 1
    assert outcome.critical_gradients == {name: True for name in equal_session.CRITICAL_GRADIENT_PATHS}
    assert outcome.finite_model is True and outcome.finite_optimizer is True
    assert flags.backward_calls == 1 and flags.update_calls == 1
    assert not torch.equal(model.decoder.fc_out.weight.detach(), before_fc_out)
    assert cache.observer().batch_loop_side_lookups == 1
    assert cache.observer().batch_loop_inverse_calls == 0 and cache.observer().batch_loop_sampling_calls == 0
    assert not torch.cuda.is_initialized()
