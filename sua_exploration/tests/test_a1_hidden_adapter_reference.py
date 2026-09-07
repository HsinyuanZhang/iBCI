"""Focused CPU-only tests for the independent A1 hidden-adapter oracle.

These tests use synthetic tensors only.  They do not import production A1 code,
open A2 artifacts, load NWB files, write receipts, or request a GPU.
"""

from __future__ import annotations

from dataclasses import replace
from io import BytesIO
from pathlib import Path

import pytest
import torch

from sua_exploration.mc_maze import a1_hidden_adapter_reference as ref


SUA = Path(__file__).resolve().parents[1]
CONTRACT = SUA / "docs" / "A1_HIDDEN_SPACE_CARRIER_CONTRACT_20260813.md"


def _adamw(parameters):
    """A deterministic, production-like CPU optimizer configuration."""

    return torch.optim.AdamW(
        parameters,
        lr=1.0e-3,
        betas=(0.9, 0.999),
        eps=1.0e-8,
        weight_decay=1.0e-2,
        foreach=False,
    )


def _zero_carrier_training_step(
    w_model: ref.WAddReference,
    h_model: ref.HAddReference,
    w_optimizer: torch.optim.Optimizer,
    h_optimizer: torch.optim.Optimizer,
    *,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one paired task-only source step with the H/Z4 carrier port at zero."""

    batch = ref.synthetic_batch(batch_size=2, num_units=7, seed=seed, zero_t4=True)
    w_optimizer.zero_grad(set_to_none=True)
    h_optimizer.zero_grad(set_to_none=True)
    w_loss = ref.task_only_mse(w_model(batch.x, batch.z4), batch.target)
    h_loss = ref.task_only_mse(h_model(batch.x, batch.z4, batch.t4), batch.target)
    assert torch.equal(w_loss, h_loss)
    w_loss.backward()
    h_loss.backward()
    ref.assert_shared_gradients_bit_equal(w_model, h_model)
    ref.assert_projection_grad_exact_zero(h_model)
    w_optimizer.step()
    h_optimizer.step()
    ref.assert_shared_parameters_bit_equal(w_model, h_model)
    ref.assert_shared_optimizer_state_bit_equal(
        w_model, w_optimizer, h_model, h_optimizer
    )
    return w_loss.detach(), h_loss.detach()


def test_matrix_is_exact_minimal_2x2_and_h_z4_is_an_alias() -> None:
    ref.validate_minimal_a1_matrix()
    matrix = ref.matrix_cells_by_key()
    assert set(matrix) == {"W/Z4", "W/T4", "H/Z4", "H/T4"}
    assert matrix["W/Z4"].provenance == "sealed_a2_reuse"
    assert matrix["W/T4"].provenance == "sealed_a2_reuse"
    assert matrix["H/Z4"].structural_alias_of == "W/Z4"
    assert matrix["H/Z4"].is_new_family is False
    assert matrix["H/T4"].is_new_family is True


def test_matrix_rejects_a_separate_h_z4_family() -> None:
    invalid = list(ref.A1_MATRIX_CELLS)
    invalid[2] = replace(
        invalid[2], provenance="new_hidden_adapter", structural_alias_of=None, is_new_family=True
    )
    with pytest.raises(ref.A1HiddenAdapterContractError, match="H/Z4 must be an exact structural alias"):
        ref.validate_minimal_a1_matrix(invalid)


def test_projection_spec_and_constructor_are_no_bias_direct_zero_and_rng_invariant() -> None:
    torch.manual_seed(1202)
    caller_rng_before = torch.random.get_rng_state()
    w_model, h_model, audit = ref.paired_references(seed=1203)
    assert torch.equal(caller_rng_before, torch.random.get_rng_state())
    assert audit.passes
    ref.assert_shared_parameters_bit_equal(w_model, h_model)
    projection = h_model.carrier_projection
    assert tuple(projection.weight.shape) == (512, 4)
    assert projection.bias is None
    assert torch.equal(projection.weight.detach(), torch.zeros_like(projection.weight))

    zero_t4 = torch.zeros((2, 7, 4), dtype=torch.float32)
    assert torch.equal(projection(zero_t4), torch.zeros((2, 7, 512), dtype=torch.float32))
    with torch.no_grad():
        projection.weight.fill_(0.125)
    assert torch.equal(projection(zero_t4), torch.zeros((2, 7, 512), dtype=torch.float32))


def test_projection_spec_rejects_bias_or_non_direct_zero_initialization() -> None:
    with pytest.raises(ref.A1HiddenAdapterContractError, match="forbids a bias"):
        ref.validate_hidden_carrier_projection_spec(
            replace(ref.HIDDEN_T4_PROJECTION_SPEC, has_bias=True)
        )
    with pytest.raises(ref.A1HiddenAdapterContractError, match="direct_zero_no_rng"):
        ref.validate_hidden_carrier_projection_spec(
            replace(ref.HIDDEN_T4_PROJECTION_SPEC, initialization="kaiming_uniform")
        )


def test_source_training_contract_keeps_teacher_declaratively_frozen_and_joint_student_trainables() -> None:
    model = ref.HAddReference()
    ref.validate_h_source_training_parameter_contract(model)
    assert ref.SOURCE_TRAINING_SEMANTICS["teacher_frozen"] is True
    assert ref.SOURCE_TRAINING_SEMANTICS["loss_mode"] == "task_only"
    assert ref.source_trainable_parameter_names(model) == (
        "activity_identity.weight",
        "fc_in.weight",
        "fc_in.bias",
        "decoder.weight",
        "decoder.bias",
        "carrier_projection.weight",
    )
    model.decoder.bias.requires_grad_(False)
    with pytest.raises(ref.A1HiddenAdapterContractError, match="joint student decoder"):
        ref.validate_h_source_training_parameter_contract(model)


@pytest.mark.parametrize("batch_size", (1, 2))
@pytest.mark.parametrize("num_units", (1, 7, 64))
def test_zero_carrier_forward_is_torch_equal_for_required_b_and_n(
    batch_size: int, num_units: int
) -> None:
    w_model, h_model, _ = ref.paired_references(seed=1911)
    batch = ref.synthetic_batch(
        batch_size=batch_size, num_units=num_units, seed=2100 + batch_size * 100 + num_units, zero_t4=True
    )
    w_hidden = w_model.hidden(batch.x, batch.z4)
    h_hidden = h_model.hidden(batch.x, batch.z4, batch.t4)
    assert torch.equal(w_hidden, h_hidden)
    assert torch.equal(w_model(batch.x, batch.z4), h_model(batch.x, batch.z4, batch.t4))
    ref.assert_z4_alias_bit_equal(
        w_model, h_model, x=batch.x, z4=batch.z4, carrier_port=batch.t4
    )


def test_task_only_loss_shared_gradients_and_p_grad_are_bit_exact_for_z4_alias() -> None:
    w_model, h_model, _ = ref.paired_references(seed=3201)
    w_optimizer = _adamw(w_model.parameters())
    h_optimizer = _adamw(h_model.parameters())
    w_loss, h_loss = _zero_carrier_training_step(
        w_model, h_model, w_optimizer, h_optimizer, seed=3202
    )
    assert torch.equal(w_loss, h_loss)
    assert torch.equal(
        h_model.carrier_projection.weight.detach(),
        torch.zeros_like(h_model.carrier_projection.weight),
    )


def test_three_production_like_optimizer_steps_preserve_shared_params_and_state() -> None:
    w_model, h_model, _ = ref.paired_references(seed=4101)
    w_optimizer = _adamw(w_model.parameters())
    h_optimizer = _adamw(h_model.parameters())

    for offset in range(3):
        _zero_carrier_training_step(
            w_model, h_model, w_optimizer, h_optimizer, seed=4102 + offset
        )

    ref.assert_shared_parameters_bit_equal(w_model, h_model)
    ref.assert_shared_optimizer_state_bit_equal(
        w_model, w_optimizer, h_model, h_optimizer
    )
    p_state = ref.optimizer_state_by_parameter_name(
        h_model, h_optimizer, include_adapter=True
    )["carrier_projection.weight"]
    assert torch.equal(p_state["exp_avg"], torch.zeros_like(p_state["exp_avg"]))
    assert torch.equal(p_state["exp_avg_sq"], torch.zeros_like(p_state["exp_avg_sq"]))


def test_checkpoint_roundtrip_preserves_hidden_prediction_and_optimizer_state() -> None:
    model = ref.HAddReference()
    optimizer = _adamw(model.parameters())
    batch = ref.synthetic_batch(batch_size=1, num_units=7, seed=5101, zero_t4=False)
    optimizer.zero_grad(set_to_none=True)
    loss = ref.task_only_mse(model(batch.x, batch.z4, batch.t4), batch.target)
    loss.backward()
    optimizer.step()
    hidden_before = model.hidden(batch.x, batch.z4, batch.t4).detach().clone()
    prediction_before = model(batch.x, batch.z4, batch.t4).detach().clone()

    buffer = BytesIO()
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, buffer)
    buffer.seek(0)
    payload = torch.load(buffer, map_location="cpu", weights_only=True)

    restored = ref.HAddReference()
    restored_optimizer = _adamw(restored.parameters())
    restored.load_state_dict(payload["model"])
    restored_optimizer.load_state_dict(payload["optimizer"])

    assert torch.equal(hidden_before, restored.hidden(batch.x, batch.z4, batch.t4))
    assert torch.equal(prediction_before, restored(batch.x, batch.z4, batch.t4))
    for name, parameter in model.state_dict().items():
        assert torch.equal(parameter, restored.state_dict()[name])
    ref.assert_optimizer_state_bit_equal(
        model,
        optimizer,
        restored,
        restored_optimizer,
        include_adapter=True,
    )


def test_nonzero_t4_backpropagates_to_p_and_an_update_changes_hidden_and_prediction() -> None:
    model = ref.HAddReference()
    optimizer = _adamw(model.parameters())
    batch = ref.synthetic_batch(batch_size=1, num_units=7, seed=6101, zero_t4=False)
    hidden_before = model.hidden(batch.x, batch.z4, batch.t4).detach().clone()
    prediction = model(batch.x, batch.z4, batch.t4)
    prediction_before = prediction.detach().clone()

    optimizer.zero_grad(set_to_none=True)
    loss = ref.task_only_mse(prediction, batch.target)
    loss.backward()
    p_gradient = model.carrier_projection.weight.grad
    assert p_gradient is not None
    assert torch.count_nonzero(p_gradient).item() > 0
    optimizer.step()

    hidden_after = model.hidden(batch.x, batch.z4, batch.t4).detach()
    prediction_after = model(batch.x, batch.z4, batch.t4).detach()
    assert not torch.equal(model.carrier_projection.weight.detach(), torch.zeros_like(p_gradient))
    assert not torch.equal(hidden_before, hidden_after)
    assert not torch.equal(prediction_before, prediction_after)


def test_nonzero_h_z4_carrier_port_fails_the_structural_alias_control() -> None:
    w_model, h_model, _ = ref.paired_references(seed=7101)
    batch = ref.synthetic_batch(batch_size=1, num_units=7, seed=7102, zero_t4=True)
    invalid_port = batch.t4.clone()
    invalid_port[0, 0, 0] = 1.0
    with pytest.raises(ref.A1HiddenAdapterContractError, match="carrier port must be exact zero"):
        ref.assert_z4_alias_bit_equal(
            w_model, h_model, x=batch.x, z4=batch.z4, carrier_port=invalid_port
        )


def test_a2_reuse_validator_accepts_the_exact_contract_and_rejects_any_drift() -> None:
    evidence = ref.expected_a2_reuse_evidence()
    ref.validate_a2_reuse_evidence(evidence)

    drifted = ref.expected_a2_reuse_evidence()
    drifted["t4_normalizer"]["value_sha256"] = "0" * 64  # type: ignore[index]
    with pytest.raises(ref.A1HiddenAdapterContractError, match="A2 reuse denied"):
        ref.validate_a2_reuse_evidence(drifted)

    target_update = ref.expected_a2_reuse_evidence()
    target_update["target_session_updates"]["decoder_weight_updates"] = True  # type: ignore[index]
    with pytest.raises(ref.A1HiddenAdapterContractError, match="decoder_weight_updates"):
        ref.validate_a2_reuse_evidence(target_update)

    wrong_bundle = ref.expected_a2_reuse_evidence()
    wrong_bundle["sealed_w_cells"]["W/T4"]["source_checkpoint_sha256_bundle_sha256"] = "0" * 64  # type: ignore[index]
    with pytest.raises(ref.A1HiddenAdapterContractError, match="source_checkpoint_sha256_bundle_sha256"):
        ref.validate_a2_reuse_evidence(wrong_bundle)


def test_score_only_primary_interaction_keeps_the_2x2_formula_and_alias_check() -> None:
    scores = {"W/Z4": 0.20, "W/T4": 0.34, "H/Z4": 0.20, "H/T4": 0.41}
    interaction = ref.score_only_primary_interaction(scores)
    assert interaction == pytest.approx((0.41 - 0.20) - (0.34 - 0.20))
    assert interaction == pytest.approx(0.07)

    invalid_scores = dict(scores)
    invalid_scores["H/Z4"] = 0.2000001
    with pytest.raises(ref.A1HiddenAdapterContractError, match="structural alias score exactly"):
        ref.score_only_primary_interaction(invalid_scores)


def test_stage_p_gate_is_attainable_but_remains_routing_only() -> None:
    passing = ref.frozen_stage_p_routing_decision(
        w_t4_scores=[0.0] * 6,
        h_t4_scores=[0.04] * 6,
    )
    assert passing.passes_all
    assert passing.mean_delta >= 0.03
    assert passing.median_delta > 0.0
    assert passing.positive_session_count == 6
    assert passing.authorizes_gpu is False

    failing = ref.frozen_stage_p_routing_decision(
        w_t4_scores=[0.0] * 6,
        h_t4_scores=[0.04, 0.04, 0.04, -0.01, -0.01, -0.01],
    )
    assert failing.passes_all is False
    assert failing.positive_session_count == 3
    assert failing.passes_positive_session_count is False

    median_failure = ref.frozen_stage_p_routing_decision(
        w_t4_scores=[0.0] * 6,
        h_t4_scores=[0.10, 0.10, -0.01, -0.01, -0.01, -0.01],
    )
    assert median_failure.median_delta < 0.0
    assert median_failure.passes_median_positive is False


def test_schema_is_non_authorizing_and_never_an_official_receipt_writer() -> None:
    schema = ref.a1_development_receipt_schema()
    assert schema["non_authorizing_status"] == "DEVELOPMENT_ROUTING_ONLY_NO_GPU_AUTHORITY"
    assert "official_receipt_minting" in schema["prohibited"]
    assert schema["integrity_fields"]["body_write"].startswith("O_CREAT|O_EXCL")
    assert "chmod 0444" in schema["integrity_fields"]["sidecar"]


def test_independent_contract_records_the_full_reuse_and_no_go_boundary() -> None:
    text = CONTRACT.read_text(encoding="utf-8")
    for required in (
        "5b1459df7f65b8dd4cf4ebb9e29b7f82a6def6fc538af71bd822ee26fc7305fc",
        "8ecdabb8226834ed0a419a16ad4b13b43814018f1b1e34297d018539690dfbbd",
        "H/Z4` is not a separately initialized model",
        "O_CREAT|O_EXCL",
        "Stage P is routing only",
        "complete B1 lattice requires a new binding and a fresh rerun",
        "**GPU GO: NO.**",
    ):
        assert required in text
