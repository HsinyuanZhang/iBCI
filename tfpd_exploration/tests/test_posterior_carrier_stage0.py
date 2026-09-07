"""CPU-only synthetic gates for the posterior-carrier Stage-0 implementation."""
from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import pytest
import torch

from src.posterior_carrier_v1 import core
from src.posterior_carrier_v1 import plan


ROOT = Path(__file__).resolve().parents[2]
SHA_A = "a" * 64
SHA_B = "b" * 64


def _raw_source_t4(*, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    return torch.tensor(
        [
            [1.0, 2.0, 2.2360679, 4.0],
            [-2.0, 0.5, 2.0615528, 5.0],
            [0.25, -1.0, 1.0307764, 7.0],
            [1.5, -0.5, 1.5811388, 3.0],
            [-0.75, -1.25, 1.4577379, 6.0],
            [0.5, 1.25, 1.3462912, 4.5],
        ],
        dtype=dtype,
    )


def _fixture(
    budget: int = 4,
    *,
    dtype: torch.dtype = torch.float64,
    include_zero_unit: bool = True,
) -> tuple[core.SourcePrior, core.PosteriorCarrier, core.FrozenSourceT4Normalizer, torch.Tensor, torch.Tensor, torch.Tensor]:
    roster = tuple(f"source-{index:02d}" for index in range(6))
    prior = core.SourcePrior.from_raw_m30_t4(_raw_source_t4(dtype=dtype), roster)
    theta = torch.arange(budget, dtype=dtype) * (2.0 * torch.pi / budget)
    row = torch.arange(budget, dtype=torch.int64)
    counts = torch.stack(
        (
            row.remainder(5) + 1,
            (2 * row + 1).remainder(7) + 1,
            (3 * row + 2).remainder(6) + 1,
            torch.zeros_like(row) if include_zero_unit else row.remainder(4) + 2,
        ),
        dim=0,
    )
    exposure = torch.linspace(0.5, 1.5, budget, dtype=dtype)
    carrier = core.fit_conjugate_posterior(counts=counts, exposure=exposure, theta=theta, prior=prior)
    normalizer = core.FrozenSourceT4Normalizer(
        mean=torch.tensor((10.0, -2.0, 3.0, 7.0), dtype=dtype),
        std=torch.tensor((2.0, 3.0, 4.0, 5.0), dtype=dtype),
        authority_sha256=SHA_A,
    )
    return prior, carrier, normalizer, counts, exposure, theta


def _restore_rng(py_state: object, np_state: tuple[object, ...], torch_state: torch.Tensor) -> None:
    random.setstate(py_state)
    np.random.set_state(np_state)
    torch.set_rng_state(torch_state)


def _build_actual_cell_d_without_leaking_rng() -> torch.nn.Module:
    """Build only the CPU graph and restore construction's global Torch seed effect."""
    from src.tfpd_lane.pop_robust import build_population_robustness_model

    py_state, np_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    try:
        model = build_population_robustness_model(seed=42, cell="D")
    finally:
        _restore_rng(py_state, np_state, torch_state)
    return model


def test_static_dry_cli_imports_no_torch_and_has_no_execution_path() -> None:
    script = ROOT / "tfpd_exploration/scripts/run_posterior_carrier_stage0.py"
    environment = {**os.environ, "PYTHONNOUSERSITE": "1", "PYTHONPATH": "", "CUDA_VISIBLE_DEVICES": ""}
    code = (
        "import importlib.util, json, sys; "
        f"spec=importlib.util.spec_from_file_location('pc_dry', {str(script)!r}); "
        "module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); "
        "payload=module._load_dry_plan(); "
        "print(json.dumps({'torch_loaded': 'torch' in sys.modules, 'status': payload['status']}))"
    )
    imported = subprocess.run([sys.executable, "-c", code], env=environment, capture_output=True, text=True, check=True)
    assert json.loads(imported.stdout) == {
        "torch_loaded": False,
        "status": "DRY_NO_DATA_NO_CACHE_NO_CHECKPOINT_NO_CUDA_NO_GPU_NO_WRITE_NO_LAUNCH",
    }
    rendered = subprocess.run([sys.executable, str(script), "--preflight"], env=environment, capture_output=True, text=True, check=True)
    payload = json.loads(rendered.stdout)
    assert payload["handoff"]["sha256"] == plan.HANDOFF_SHA256
    assert payload["target_or_external_or_h1_authorized"] is False
    assert "cuda" in payload["status"].lower()
    rejected = subprocess.run([sys.executable, str(script), "--execute"], env=environment, capture_output=True, text=True)
    assert rejected.returncode != 0
    assert "not implemented" in rejected.stderr


def test_exact_budget_mix_schedule_is_session_static_and_balanced() -> None:
    schedule = core.build_budget_schedule(epochs=48, session_count=27)
    core.verify_balanced_48_epoch_schedule(schedule, session_count=27)
    assert schedule[0][:3] == (4, 10, 30)
    assert schedule[1][:3] == (10, 30, 4)
    assert all(schedule[epoch][session] == core.budget_for_epoch(epoch, session)
               for epoch in range(48) for session in range(27))
    for session in range(27):
        assert [schedule[epoch][session] for epoch in range(48)].count(4) == 16
        assert [schedule[epoch][session] for epoch in range(48)].count(10) == 16
        assert [schedule[epoch][session] for epoch in range(48)].count(30) == 16


def test_source_prior_uses_exact_raw_m30_moments_and_zero_directional_mean() -> None:
    raw = _raw_source_t4()
    prior = core.SourcePrior.from_raw_m30_t4(raw, tuple(f"source-{index:02d}" for index in range(6)))
    expected_mean_b = float(raw[:, 3].mean())
    expected_tau_ac2 = max(float((raw[:, 0].square() + raw[:, 1].square()).mean() / 2), 1e-12)
    expected_tau_b2 = max(float((raw[:, 3] - expected_mean_b).square().mean()), 1e-12)
    assert prior.source_mean_b == pytest.approx(expected_mean_b)
    assert prior.tau_ac2 == pytest.approx(expected_tau_ac2)
    assert prior.tau_b2 == pytest.approx(expected_tau_b2)
    payload = prior.payload()
    assert payload["mu0"][:2] == [0.0, 0.0]
    assert payload["directional_prior_mean_exact_zero"] is True
    assert payload["directional_prior_isotropic"] is True


def test_count_exposure_precision_is_literal_and_one_inverse_per_unit() -> None:
    prior, _carrier, _normalizer, counts, exposure, theta = _fixture(4)
    calls: list[torch.Tensor] = []

    def counted_inverse(matrices: torch.Tensor) -> torch.Tensor:
        calls.append(matrices.detach().clone())
        return torch.linalg.inv(matrices)

    carrier = core.fit_conjugate_posterior(
        counts=counts,
        exposure=exposure,
        theta=theta,
        prior=prior,
        inverse_fn=counted_inverse,
    )
    assert len(calls) == 1
    assert calls[0].shape == (counts.shape[0], 3, 3)
    assert carrier.audit.inverse_call_count == 1
    assert carrier.audit.inverse_count == counts.shape[0]
    assert carrier.audit.inverse_input_shape == (counts.shape[0], 3, 3)
    assert carrier.audit.lambda_grid_or_optimizer_used is False
    design = core.directional_design(theta)
    expected_w = exposure.square().unsqueeze(0) / counts.to(torch.float64).clamp_min(1.0)
    expected_precision = prior.precision_tensor(device=theta.device, dtype=torch.float64).unsqueeze(0) + torch.matmul(
        design.transpose(0, 1).unsqueeze(0), design.unsqueeze(0) * expected_w.unsqueeze(-1)
    )
    assert torch.equal(calls[0], expected_precision)
    with pytest.raises(core.PosteriorCarrierError, match="rates may not be rounded"):
        core.fit_conjugate_posterior(
            counts=counts.to(torch.float64) + 0.25,
            exposure=exposure,
            theta=theta,
            prior=prior,
        )


def test_zero_spike_semantics_and_spd_finite_outputs_at_m4_m10_m30() -> None:
    for budget in (4, 10, 30):
        _prior, carrier, _normalizer, _counts, _exposure, _theta = _fixture(budget)
        assert carrier.audit.design_rank == 3
        assert math_isfinite(carrier.audit.design_condition_number)
        assert torch.equal(carrier.raw_t4[-1], torch.zeros(4, dtype=carrier.raw_t4.dtype))
        assert carrier.credibility[-1].item() == pytest.approx(core.CREDIBILITY_FLOOR)
        assert torch.isfinite(carrier.mean).all() and torch.isfinite(carrier.covariance).all()
        assert torch.allclose(carrier.covariance, carrier.covariance.transpose(-1, -2), atol=1e-10, rtol=0.0)
        torch.linalg.cholesky(carrier.covariance)


def math_isfinite(value: float) -> bool:
    return isinstance(value, float) and value > 0 and value < float("inf")


def test_posterior_permutation_equivariance_is_exact_for_unit_fields_and_activity_rows() -> None:
    prior, carrier, normalizer, counts, exposure, theta = _fixture(10, include_zero_unit=False)
    permutation = torch.tensor((2, 0, 3, 1), dtype=torch.long)
    permuted = core.fit_conjugate_posterior(
        counts=counts.index_select(0, permutation), exposure=exposure, theta=theta, prior=prior,
    )
    assert torch.equal(permuted.mean, carrier.mean.index_select(0, permutation))
    assert torch.equal(permuted.covariance, carrier.covariance.index_select(0, permutation))
    assert torch.equal(permuted.raw_t4, carrier.raw_t4.index_select(0, permutation))
    assert torch.equal(permuted.credibility, carrier.credibility.index_select(0, permutation))
    activity_rows = torch.arange(4 * 7, dtype=torch.float64).reshape(4, 7)
    assert torch.equal(activity_rows.index_select(0, permutation), activity_rows[permutation])
    view = core.posterior_mean_view(carrier, normalizer)
    joint = view.joint_permute(permutation)
    assert torch.equal(joint.raw_t4, view.raw_t4.index_select(0, permutation))
    assert torch.equal(joint.normalized_t4, view.normalized_t4.index_select(0, permutation))
    assert torch.equal(joint.credibility, view.credibility.index_select(0, permutation))


def test_so2_equivariance_rotates_only_directional_posterior_and_preserves_credibility() -> None:
    prior, carrier, _normalizer, counts, exposure, theta = _fixture(10, include_zero_unit=False)
    angle = torch.tensor(0.731, dtype=theta.dtype)
    rotated = core.fit_conjugate_posterior(counts=counts, exposure=exposure, theta=theta + angle, prior=prior)
    rotation = torch.stack((
        torch.stack((torch.cos(angle), -torch.sin(angle))),
        torch.stack((torch.sin(angle), torch.cos(angle))),
    ))
    expected_mean = torch.matmul(rotation, carrier.mean[:, :2].unsqueeze(-1)).squeeze(-1)
    full_rotation = torch.eye(3, dtype=theta.dtype)
    full_rotation[:2, :2] = rotation
    expected_covariance = torch.matmul(torch.matmul(full_rotation.unsqueeze(0), carrier.covariance), full_rotation.transpose(0, 1))
    assert torch.allclose(rotated.mean[:, :2], expected_mean, atol=2e-12, rtol=2e-12)
    assert torch.allclose(rotated.mean[:, 2], carrier.mean[:, 2], atol=2e-12, rtol=2e-12)
    assert torch.allclose(rotated.covariance, expected_covariance, atol=2e-12, rtol=2e-12)
    assert torch.allclose(rotated.credibility, carrier.credibility, atol=2e-12, rtol=2e-12)
    assert prior.payload()["mu0"][:2] == [0.0, 0.0]


def test_directional_credibility_ignores_baseline_variance_and_attention_bias_is_exactly_parameter_free() -> None:
    covariance_a = torch.diag(torch.tensor((0.2, 0.4, 0.01), dtype=torch.float64)).unsqueeze(0)
    covariance_b = torch.diag(torch.tensor((0.2, 0.4, 1000.0), dtype=torch.float64)).unsqueeze(0)
    credibility_a = core.directional_credibility(covariance_a, tau_ac2=1.0)
    credibility_b = core.directional_credibility(covariance_b, tau_ac2=1.0)
    assert torch.equal(credibility_a, credibility_b)
    assert credibility_a.item() == pytest.approx(0.7)
    logits = torch.tensor([[[[0.2, -0.3, 1.1], [0.1, 0.0, -0.4]]]], dtype=torch.float64)
    equal = torch.full((3,), 0.5, dtype=torch.float64)
    assert torch.equal(core.centered_attention_bias(equal, batch_size=1, num_heads=1, query_count=2), torch.zeros((1, 2, 3), dtype=torch.float64))
    assert torch.equal(core.add_credibility_to_logits(logits, equal), logits)
    masses = torch.softmax(core.add_credibility_to_logits(torch.zeros((1, 1, 1, 2), dtype=torch.float64),
                                                          torch.tensor((0.8, 0.2), dtype=torch.float64)), dim=-1)
    assert masses[0, 0, 0, 0] > masses[0, 0, 0, 1]
    assert masses[0, 0, 0, 0].item() == pytest.approx(0.8)


def test_session_static_sampling_replays_exactly_without_mutating_host_rngs_and_eval_never_samples() -> None:
    _prior, carrier, normalizer, _counts, _exposure, _theta = _fixture(10)
    sampler = core.SessionStaticPosteriorSampler(seed=42, normalizer=normalizer)
    before = core.host_rng_fingerprint()
    first = sampler.carrier_for(carrier, session_id="source-00", epoch=7, training=True)
    second = sampler.carrier_for(carrier, session_id="source-00", epoch=7, training=True)
    after = core.host_rng_fingerprint()
    core.assert_host_rng_unchanged(before, after)
    assert first.sampled is True and second.sampled is True
    assert torch.equal(first.raw_beta, second.raw_beta)
    assert torch.equal(first.raw_t4, second.raw_t4)
    assert torch.equal(first.normalized_for_batch(4)[0], first.normalized_for_batch(4)[3])
    replay = core.SessionStaticPosteriorSampler(seed=42, normalizer=normalizer).carrier_for(
        carrier, session_id="source-00", epoch=7, training=True,
    )
    assert torch.equal(first.raw_beta, replay.raw_beta)
    assert not torch.equal(first.raw_beta, sampler.carrier_for(carrier, session_id="source-00", epoch=8, training=True).raw_beta)
    cache_before_eval = sampler.cached_keys
    evaluation = sampler.carrier_for(carrier, session_id="source-00", epoch=7, training=False)
    assert evaluation.sampled is False and evaluation.session_id is None and evaluation.epoch is None
    assert sampler.cached_keys == cache_before_eval
    assert torch.equal(evaluation.raw_t4, carrier.raw_t4)


def test_sampling_builds_raw_t4_before_frozen_normalization() -> None:
    _prior, carrier, normalizer, _counts, _exposure, _theta = _fixture(10)
    sampled = core.SessionStaticPosteriorSampler(seed=43, normalizer=normalizer).carrier_for(
        carrier, session_id="source-01", epoch=4, training=True,
    )
    expected_raw = core.beta_to_raw_t4(sampled.raw_beta, sampled.zero_spike_mask)
    expected_normalized = (expected_raw - normalizer.mean) / normalizer.std
    assert torch.equal(sampled.raw_t4, expected_raw)
    assert torch.equal(sampled.normalized_t4, expected_normalized)
    assert torch.equal(sampled.raw_t4[-1], torch.zeros(4, dtype=sampled.raw_t4.dtype))
    assert not torch.equal(sampled.normalized_t4[-1], torch.zeros(4, dtype=sampled.raw_t4.dtype))


def test_posterior_specific_source_normalizer_uses_equal_m4_m10_m30_float64_rows() -> None:
    roster = ("source-00", "source-01")
    # The rows are deliberately distinct by budget, session, and unit.  The
    # source-only normalizer must use every row in the frozen budget-major,
    # strict-roster/unit order rather than borrow ordinary point-T4 moments.
    rows = {
        4: {
            "source-00": torch.tensor([[0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 1.0, 2.0]], dtype=torch.float64),
            "source-01": torch.tensor([[2.0, 0.0, 2.0, 3.0]], dtype=torch.float64),
        },
        10: {
            "source-00": torch.tensor([[3.0, 1.0, 3.2, 4.0], [4.0, 1.0, 4.1, 5.0]], dtype=torch.float64),
            "source-01": torch.tensor([[5.0, 1.0, 5.1, 6.0]], dtype=torch.float64),
        },
        30: {
            "source-00": torch.tensor([[6.0, 2.0, 6.3, 7.0], [7.0, 2.0, 7.3, 8.0]], dtype=torch.float64),
            "source-01": torch.tensor([[8.0, 2.0, 8.3, 9.0]], dtype=torch.float64),
        },
    }
    normalizer = core.PosteriorSourceT4Normalizer.fit(source_roster=roster, raw_mean_t4_by_budget=rows)
    expected_rows = torch.cat([
        rows[4]["source-00"], rows[4]["source-01"],
        rows[10]["source-00"], rows[10]["source-01"],
        rows[30]["source-00"], rows[30]["source-01"],
    ], dim=0)
    assert normalizer.mean.dtype == torch.float64 and normalizer.std.dtype == torch.float64
    assert torch.equal(normalizer.mean, expected_rows.mean(dim=0))
    assert torch.equal(normalizer.std, expected_rows.sub(expected_rows.mean(dim=0)).square().mean(dim=0).sqrt())
    assert normalizer.row_count == 9
    assert normalizer.per_budget_row_counts == {4: 3, 10: 3, 30: 3}
    assert normalizer.payload()["source_only"] is True
    assert normalizer.payload()["contains_only_deterministic_posterior_means"] is True
    assert normalizer.authority_sha256 == normalizer.payload()["body_sha256"]
    # The raw zero carrier stays raw zero; its standardized representation is
    # intentionally not silently clamped to zero by the new normalizer.
    zero_standardized = normalizer.normalize_raw(torch.zeros((1, 4), dtype=torch.float64))
    assert not torch.equal(zero_standardized, torch.zeros_like(zero_standardized))
    bad = {budget: dict(by_session) for budget, by_session in rows.items()}
    bad[4]["source-01"] = torch.cat((bad[4]["source-01"], bad[4]["source-01"]), dim=0)
    with pytest.raises(core.PosteriorCarrierError, match="equal source-row weight"):
        core.PosteriorSourceT4Normalizer.fit(source_roster=roster, raw_mean_t4_by_budget=bad)


def test_cell_d_wrapper_preserves_parameters_dropout_stream_and_held_m30_activity() -> None:
    model = _build_actual_cell_d_without_leaking_rng()
    wrapper = core.CellDPosteriorWrapper(model)
    audit = wrapper.preservation_audit()
    assert audit.base_live_parameter_count == 3_510_842
    assert audit.wrapper_live_parameter_count == 3_510_842
    assert audit.wrapper_new_parameter_count == 0
    assert audit.parameter_object_ids_identical is True
    assert audit.dynamic_dropout and audit.dropout_low == 0.0 and audit.dropout_high == 1.0
    _prior4, carrier4, normalizer4, _counts4, _exposure4, _theta4 = _fixture(4, dtype=torch.float32)
    _prior10, carrier10, _normalizer10, _counts10, _exposure10, _theta10 = _fixture(10, dtype=torch.float32)
    carrier4_view = core.posterior_mean_view(carrier4, normalizer4)
    # The B3S neural activity summary only consumes the fixed M30 neural
    # calibration tensor; M4/M10 apply solely to labelled carrier fitting.
    torch.manual_seed(123)
    neural = torch.randn(1, 50, carrier4.unit_count)
    calib_m30 = torch.randn(1, 30, 100, carrier4.unit_count)
    prefix_before = wrapper.b3s_m30_activity_prefix(calib_m30)
    prefix_after = wrapper.b3s_m30_activity_prefix(calib_m30)
    assert torch.equal(prefix_before, prefix_after)
    assert carrier4.raw_t4.shape == carrier10.raw_t4.shape

    equal_view = replace(carrier4_view, credibility=torch.full_like(carrier4_view.credibility, 0.5))
    model.eval()
    wrapper.eval()
    with torch.no_grad():
        baseline_prediction, baseline_identity = model(
            neural, calib_trials=calib_m30, side_features=equal_view.normalized_for_batch(1),
        )
        wrapped_prediction, wrapped_identity = wrapper(
            neural, calib_trials_m30=calib_m30, carrier=equal_view,
        )
    assert torch.equal(wrapped_prediction, baseline_prediction)
    assert torch.equal(wrapped_identity, baseline_identity)

    # The equal-credibility branch literally delegates to Cell-D, so the
    # dynamic U(0,1) placeholder dropout and all downstream RNG draws match.
    py_state, np_state, torch_state = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    try:
        model.train()
        wrapper.train()
        random.seed(37)
        torch.manual_seed(101)
        seeded_py, seeded_torch = random.getstate(), torch.get_rng_state().clone()
        with torch.no_grad():
            expected_prediction, expected_identity = model(
                neural, calib_trials=calib_m30, side_features=equal_view.normalized_for_batch(1),
            )
        expected_after = (random.getstate(), torch.get_rng_state().clone())
        random.setstate(seeded_py)
        torch.set_rng_state(seeded_torch)
        with torch.no_grad():
            actual_prediction, actual_identity = wrapper(
                neural, calib_trials_m30=calib_m30, carrier=equal_view,
            )
        actual_after = (random.getstate(), torch.get_rng_state().clone())
        assert torch.equal(actual_prediction, expected_prediction)
        assert torch.equal(actual_identity, expected_identity)
        assert actual_after[0] == expected_after[0]
        assert torch.equal(actual_after[1], expected_after[1])
    finally:
        _restore_rng(py_state, np_state, torch_state)


def test_cell_d_wrapper_is_permutation_equivariant_with_nonuniform_credibility() -> None:
    model = _build_actual_cell_d_without_leaking_rng().eval()
    wrapper = core.CellDPosteriorWrapper(model).eval()
    _prior, carrier, normalizer, _counts, _exposure, _theta = _fixture(4, dtype=torch.float32)
    view = core.posterior_mean_view(carrier, normalizer)
    torch.manual_seed(211)
    neural = torch.randn(1, 50, view.unit_count)
    calib = torch.randn(1, 30, 100, view.unit_count)
    permutation = torch.tensor((2, 0, 3, 1), dtype=torch.long)
    with torch.no_grad():
        prediction, identity = wrapper(neural, calib_trials_m30=calib, carrier=view)
        permuted_prediction, permuted_identity = wrapper(
            neural[:, :, permutation], calib_trials_m30=calib[:, :, :, permutation], carrier=view.joint_permute(permutation),
        )
    assert torch.equal(permuted_identity, identity.index_select(1, permutation))
    assert torch.allclose(permuted_prediction, prediction, rtol=2e-6, atol=2e-6)


def test_source_prior_and_session_posterior_receipts_bind_roster_prefix_design_schedule_and_closure() -> None:
    prior, carrier, _normalizer, _counts, _exposure, _theta = _fixture(10)
    closure = core.stage0_closure(ROOT)
    assert closure["sha256_by_path"][plan.HANDOFF_RELATIVE] == plan.HANDOFF_SHA256
    source_receipt = core.build_source_prior_receipt(
        prior=prior,
        source_roster=prior.source_roster,
        raw_m30_prefix_rows_sha256=SHA_B,
        closure=closure,
    )
    core.validate_source_prior_receipt(source_receipt, prior=prior, closure=closure)
    schedule = core.build_budget_schedule(epochs=48, session_count=3)
    session_receipt = core.build_session_posterior_receipt(
        session_id="source-01",
        session_index=1,
        epoch=0,
        schedule=schedule,
        prefix_row_ids=tuple(f"source-01-label-{index:02d}" for index in range(10)),
        carrier=carrier,
        source_prior_receipt=source_receipt,
        closure=closure,
    )
    core.validate_session_posterior_receipt(
        session_receipt,
        carrier=carrier,
        schedule=schedule,
        source_prior_receipt=source_receipt,
        closure=closure,
    )
    tampered = dict(session_receipt)
    tampered["prefix_rows_sha256"] = SHA_A
    with pytest.raises(core.PosteriorCarrierError, match="binding drift"):
        core.validate_session_posterior_receipt(
            tampered,
            carrier=carrier,
            schedule=schedule,
            source_prior_receipt=source_receipt,
            closure=closure,
        )
