from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

from mc_maze import misleading_identity_swap_v2_core as core


def _descriptors() -> dict[str, torch.Tensor]:
    return {
        "source_session_a": torch.tensor([
            [0.0, 0.0, 0.0, 1.0],
            [0.1, 0.0, 0.1, 1.1],
            [2.0, 2.0, 2.8, 3.0],
            [2.1, 2.0, 2.9, 3.1],
            [4.0, 4.0, 5.6, 5.0],
            [4.1, 4.0, 5.7, 5.1],
            [6.0, 6.0, 8.5, 7.0],
            [6.1, 6.0, 8.6, 7.1],
        ], dtype=torch.float64),
        "source_session_b": torch.tensor([
            [0.2, 0.0, 0.2, 0.8],
            [0.3, 0.0, 0.3, 0.9],
            [1.0, 1.0, 1.4, 2.0],
            [1.1, 1.0, 1.5, 2.1],
            [3.0, 3.0, 4.2, 4.0],
            [3.1, 3.0, 4.3, 4.1],
        ], dtype=torch.float64),
    }


def _authority() -> dict:
    return core.build_matching_authority(_descriptors())


def test_config_freezes_fraction_matrix_gates_and_non_authorizing_state() -> None:
    config = core.validate_config()
    assert config["intervention"]["fraction"] == 0.5
    assert config["stage"]["fresh_training_cells"] == list(core.CELLS)
    assert config["frozen_training"]["epoch_window"] == list(range(5, 13))
    assert config["gates"]["external_clean_input_t4_lift_min"] == 0.03
    assert config["gates"]["within_clean_input_t4_delta_min"] == -0.03
    assert config["gates"]["external_t4_robustness_contrast"] == "diagnostic_non_rescuing"
    assert config["execution"]["official_preflight_minted"] is True
    assert config["execution"]["gpu_launch_implemented"] is True
    assert config["execution"]["target_scorer_implemented"] is True


def test_sealed_a2_parent_bindings_match_t4_z4_and_bind_exact_receipts() -> None:
    for domain in core.DOMAINS:
        binding = core.load_a2_parent_domain_bindings(domain)
        assert binding["domain"] == domain
        assert set(binding["receipts"]) == {"t4", "z4"}
        assert len(binding["domain_sessions"]) == len(binding["session_query_receipts"])
        for row in binding["receipts"].values():
            assert len(row["sha256"]) == 64
            assert Path(row["path"]).is_file()


def test_config_tamper_fails_closed(tmp_path: Path) -> None:
    config = core.load_json_object(core.CONFIG_PATH)
    config["intervention"]["application_point"] = "after_side_concat"
    poisoned = tmp_path / "poisoned.json"
    poisoned.write_text(__import__("json").dumps(config), encoding="utf-8")
    with pytest.raises(core.SwapV2ContractError, match="swap point drift"):
        core.validate_config(poisoned)


def test_authority_is_deterministic_complete_and_byte_shared_by_siblings() -> None:
    left = _authority()
    right = _authority()
    assert left == right
    verified = core.VerifiedMatchingAuthority.from_payload(left)
    assert verified.sha256 == core.canonical_json_sha256(right)
    for row in left["sessions"]:
        assert set(row["mappings"]) == {str(value) for value in range(12)}
        for epoch, mapping in row["mappings"].items():
            permutation = torch.tensor(mapping["permutation"])
            assert torch.equal(permutation[permutation], torch.arange(permutation.numel()))
            assert mapping["lightning_current_epoch"] == int(epoch)
            assert mapping["epoch_number_one_based"] == int(epoch) + 1
            assert mapping["selected_count"] % 2 == 0
    core.verify_sibling_authority_bindings(
        {"t4": verified.sha256, "z4": verified.sha256},
        expected_sha256=verified.sha256,
    )


@pytest.mark.parametrize(
    "poison",
    [
        lambda payload: payload.update({"fraction": 0.25}),
        lambda payload: payload["descriptor_schema"].update({"rate_coordinate": "extra_rate"}),
        lambda payload: payload["sessions"][0].update({"descriptor_sha256": "0" * 64}),
        lambda payload: payload["sessions"][0]["mappings"]["3"]["permutation"].reverse(),
        lambda payload: payload["sessions"][0]["mappings"]["3"].update({"selected_count": 2}),
        lambda payload: payload.update({"formal_subc_test_nwb_opened": True}),
        lambda payload: payload.update({"source_only": False}),
    ],
)
def test_authority_tampering_fails_closed(poison) -> None:
    payload = copy.deepcopy(_authority())
    poison(payload)
    with pytest.raises(core.SwapV2ContractError):
        core.validate_matching_authority(payload)


def test_sibling_authority_mismatch_fails_closed() -> None:
    digest = core.VerifiedMatchingAuthority.from_payload(_authority()).sha256
    with pytest.raises(core.SwapV2ContractError, match="byte-identical"):
        core.verify_sibling_authority_bindings(
            {"t4": digest, "z4": "0" * 64}, expected_sha256=digest
        )


def test_target_diagnostic_authority_is_truthful_and_separate() -> None:
    payload = core.build_target_diagnostic_authority(
        _descriptors(), domain="external_subject_M"
    )
    assert core.validate_target_diagnostic_authority(payload) == payload
    assert payload["source_only"] is False
    assert payload["target_nwb_opened"] is True
    assert payload["target_optimizer_or_backward_steps"] == 0
    assert payload["target_query_velocity_used_for_weight_updates"] is False
    with pytest.raises(core.SwapV2ContractError):
        core.validate_matching_authority(payload)


def test_target_diagnostic_authority_tamper_fails_closed(tmp_path: Path) -> None:
    payload = core.build_target_diagnostic_authority(
        _descriptors(), domain="within_subject"
    )
    body, _side, digest = core.write_immutable_json_pair(tmp_path / "target.json", payload)
    verified = core.load_verified_runtime_authority(body, expected_kind="target_diagnostic")
    assert verified.sha256 == digest
    poisoned = copy.deepcopy(payload)
    poisoned["target_query_velocity_used_for_weight_updates"] = True
    with pytest.raises(core.SwapV2ContractError):
        core.validate_target_diagnostic_authority(poisoned)


def _score_matrix() -> dict:
    return {
        "clean_z4": {
            "within_subject": {"clean": 0.30, "swapped_diagnostic": 0.10},
            "external_subject_M": {"clean": -0.10, "swapped_diagnostic": -0.30},
        },
        "clean_t4": {
            "within_subject": {"clean": 0.50, "swapped_diagnostic": 0.20},
            "external_subject_M": {"clean": 0.30, "swapped_diagnostic": -0.10},
        },
        "swap_z4": {
            "within_subject": {"clean": 0.31, "swapped_diagnostic": 0.20},
            "external_subject_M": {"clean": -0.08, "swapped_diagnostic": -0.18},
        },
        "swap_t4": {
            "within_subject": {"clean": 0.48, "swapped_diagnostic": 0.35},
            "external_subject_M": {"clean": 0.34, "swapped_diagnostic": 0.24},
        },
    }


def test_stage_p_aggregate_uses_only_primary_gates_and_reports_diagnostics() -> None:
    result = core.aggregate_stage_p(_score_matrix())
    assert result["primary_gates"]["external_clean_input_t4_lift"] == pytest.approx(0.04)
    assert result["primary_gates"]["within_clean_input_t4_delta"] == pytest.approx(-0.02)
    assert result["external_t4_robustness_contrast_diagnostic_non_rescuing"] == pytest.approx(0.30)
    assert result["verdict"] == "STAGE_P_PASS_PRIMARY_GATES__SUCCESSOR_REQUIRED"
    assert result["robustness_can_rescue"] is False
    assert result["interaction_can_rescue"] is False


def test_positive_robustness_cannot_rescue_failed_external_gate() -> None:
    scores = _score_matrix()
    scores["swap_t4"]["external_subject_M"]["clean"] = 0.31
    scores["swap_t4"]["external_subject_M"]["swapped_diagnostic"] = 0.40
    result = core.aggregate_stage_p(scores)
    assert result["external_t4_robustness_contrast_diagnostic_non_rescuing"] > 0
    assert result["primary_gates"]["external_clean_input_t4_lift_pass"] is False
    assert result["verdict"] == "STOP_STAGE_P_PRIMARY_GATE_FAILURE"


@pytest.mark.parametrize("occupied", ["body", "sidecar", "both"])
def test_immutable_writer_rejects_any_output_conflict_without_new_file(
    tmp_path: Path, occupied: str
) -> None:
    body = tmp_path / "authority.json"
    sidecar = core.sidecar_path(body)
    if occupied in {"body", "both"}:
        body.write_text("occupied", encoding="utf-8")
    if occupied in {"sidecar", "both"}:
        sidecar.write_text("occupied", encoding="utf-8")
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    with pytest.raises(FileExistsError):
        core.write_immutable_json_pair(body, _authority())
    after = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    assert after == before


def test_immutable_development_authority_roundtrip(tmp_path: Path) -> None:
    body, sidecar, digest = core.write_immutable_json_pair(
        tmp_path / "authority.json", _authority()
    )
    assert body.stat().st_mode & 0o777 == 0o444
    assert sidecar.stat().st_mode & 0o777 == 0o444
    payload, observed = core.load_verified_immutable_json(body)
    assert observed == digest
    assert core.validate_matching_authority(payload) == _authority()
    authority = core.load_verified_authority(body)
    assert authority.sha256 == digest
    assert authority.canonical_sha256 == core.canonical_json_sha256(_authority())
    assert authority.immutable_file_path == str(body.resolve())


def test_authority_byte_binding_requires_the_exact_immutable_file(tmp_path: Path) -> None:
    payload = _authority()
    with pytest.raises(core.SwapV2ContractError, match="requires a verified immutable"):
        core.VerifiedMatchingAuthority.from_payload(payload, consumed_bytes_sha256="0" * 64)
    body, _sidecar, digest = core.write_immutable_json_pair(tmp_path / "authority.json", payload)
    with pytest.raises(core.SwapV2ContractError, match="consumed-byte SHA"):
        core.VerifiedMatchingAuthority.from_payload(
            payload, consumed_bytes_sha256="0" * 64, immutable_file_path=body
        )
    other = _authority()
    other["sessions"][0]["descriptor_values"][0][0] = 999.0
    with pytest.raises(core.SwapV2ContractError):
        core.VerifiedMatchingAuthority.from_payload(other, immutable_file_path=body)
    verified = core.VerifiedMatchingAuthority.from_payload(payload, immutable_file_path=body)
    assert verified.sha256 == digest


def test_immutable_authority_symlink_is_rejected(tmp_path: Path) -> None:
    body, _sidecar, _digest = core.write_immutable_json_pair(tmp_path / "authority.json", _authority())
    alias = tmp_path / "authority_alias.json"
    alias.symlink_to(body)
    with pytest.raises(core.SwapV2ContractError, match="may not be a symlink"):
        core.load_verified_immutable_json(alias)
