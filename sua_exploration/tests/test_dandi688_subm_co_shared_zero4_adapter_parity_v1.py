from __future__ import annotations

from dataclasses import replace
import inspect
from pathlib import Path
import sys

import numpy as np
import pytest

from sua_exploration.mc_maze import subm_co_shared_zero4_adapter_parity_v1 as parity
from sua_exploration.scripts import (
    run_dandi688_subm_co_shared_zero4_adapter_parity_v1 as runner,
)
from sua_exploration.scripts import (
    write_dandi688_subm_co_shared_zero4_adapter_parity_prelaunch_v1 as prelaunch,
)


ROOT = Path(__file__).resolve().parents[2]


def _trials() -> list[dict[str, object]]:
    return [
        {
            "trial_index": index,
            "start": index * 100,
            "stop": index * 100 + 80,
            "target_dir": float(index % 4),
        }
        for index in range(60)
    ]


def _receipt() -> dict[str, object]:
    return {
        "target_direction_label_reads_for_descriptor": 0,
        "t4_trial_rate_reads_for_descriptor": 0,
        "target_t4_rate_fit_calls": 0,
        "raw_t4_constructed": False,
        "source_t4_normalizer_arithmetic_performed": False,
        "bitwise_float32_zero": True,
    }


def _record(variant: str = "original") -> parity.PredictionInputRecord:
    neural = np.zeros((200, 3), dtype=np.float32)
    behavior = np.zeros((200, 2), dtype=np.float32)
    calibration = np.zeros((30, 100, 3), dtype=np.float32)
    valid_starts = np.arange(50, 80, dtype=np.int64)
    side = np.zeros((3, 4), dtype=np.float32)
    return parity.PredictionInputRecord(
        view="sua",
        variant=variant,
        session="synthetic",
        neural=neural,
        behavior=behavior,
        calibration=calibration,
        valid_starts=valid_starts,
        side_features=side,
        selection_indices=tuple(range(30)),
        structural_trial_sha256="1" * 64,
        label_sha256=("2" if variant == "original" else "3") * 64,
        descriptor_receipt=_receipt(),
        source_unit_count=3,
    )


def test_label_shuffle_and_drop_change_only_label_evidence() -> None:
    original = parity.label_variant(_trials(), "original")
    structure = parity._structural_trial_digest(original)
    labels = parity._label_digest(original)
    for variant in ("label_shuffle", "label_drop"):
        changed = parity.label_variant(original, variant)
        assert parity._structural_trial_digest(changed) == structure
        assert parity._label_digest(changed) != labels
    assert all("target_dir" not in row for row in parity.label_variant(original, "label_drop"))


def test_unknown_label_variant_fails_closed() -> None:
    with pytest.raises(parity.SharedZero4ParityError, match="unsupported"):
        parity.label_variant(_trials(), "invented")


def test_query_valid_starts_are_reconstructed_only_after_trial_50() -> None:
    trials = _trials()
    observed = parity._expected_query_valid_starts(trials)
    expected = []
    for trial in trials[50:]:
        expected.extend(range(int(trial["start"]), int(trial["stop"]) - 50 + 1))
    assert observed.dtype == np.int64
    assert np.array_equal(observed, np.asarray(expected, dtype=np.int64))
    assert int(observed.min()) >= int(trials[50]["start"])


def test_bitwise_zero_rejects_negative_zero_wrong_dtype_and_wrong_shape() -> None:
    assert parity._bitwise_positive_float32_zero(np.zeros((3, 4), dtype=np.float32))
    assert not parity._bitwise_positive_float32_zero(np.full((3, 4), -0.0, dtype=np.float32))
    assert not parity._bitwise_positive_float32_zero(np.zeros((3, 4), dtype=np.float64))
    assert not parity._bitwise_positive_float32_zero(np.zeros((3, 3), dtype=np.float32))


@pytest.mark.parametrize(
    "field,value",
    (
        ("target_direction_label_reads_for_descriptor", 1),
        ("t4_trial_rate_reads_for_descriptor", 1),
        ("target_t4_rate_fit_calls", 1),
        ("raw_t4_constructed", True),
        ("source_t4_normalizer_arithmetic_performed", True),
        ("bitwise_float32_zero", False),
    ),
)
def test_descriptor_access_or_zero_proof_drift_fails_closed(field: str, value: object) -> None:
    receipt = _receipt()
    receipt[field] = value
    with pytest.raises(parity.SharedZero4ParityError):
        parity._descriptor_receipt_checked(receipt)


def test_prediction_input_exactness_accepts_label_only_change_and_rejects_activity_change() -> None:
    original = _record("original")
    shuffled = _record("label_shuffle")
    parity._assert_same_prediction_input(original, shuffled)
    changed_calibration = shuffled.calibration.copy()
    changed_calibration[0, 0, 0] = 1.0
    with pytest.raises(parity.SharedZero4ParityError, match="calibration values changed"):
        parity._assert_same_prediction_input(
            original, replace(shuffled, calibration=changed_calibration)
        )


def test_channel_count_only_helper_does_not_numerically_read_label_rate_or_normalizer() -> None:
    exploration_root = str(ROOT / "sua_exploration")
    if exploration_root not in sys.path:
        sys.path.insert(0, exploration_root)
    from mc_maze.paired_view_c1_shared_zero4 import (
        attach_standardized_zero4_to_evaluation_record,
        standardized_zero4,
    )

    assert list(inspect.signature(standardized_zero4).parameters) == ["channel_count"]

    class Poison:
        def __array__(self, *args, **kwargs):
            raise AssertionError("forbidden numerical descriptor input was read")

        def __iter__(self):
            raise AssertionError("forbidden descriptor input was iterated")

    record = {
        "n_units": 3,
        "neural": np.zeros((10, 3), dtype=np.float32),
        "target_direction_values": Poison(),
        "fit_rate_values": Poison(),
        "side_mean": Poison(),
        "side_std": Poison(),
    }
    updated = attach_standardized_zero4_to_evaluation_record(record)
    assert parity._bitwise_positive_float32_zero(updated["side_features"])
    assert updated["zero4_descriptor_receipt"]["target_direction_label_reads_for_descriptor"] == 0
    assert updated["zero4_descriptor_receipt"]["t4_trial_rate_reads_for_descriptor"] == 0
    assert updated["zero4_descriptor_receipt"]["source_t4_normalizer_arithmetic_performed"] is False


def test_static_authority_is_fixed_consumed_subc_and_has_no_external_capability() -> None:
    authority = prelaunch.static_authority(ROOT)
    assert authority["fixed_fixture"]["session"] == "sub-C_ses-CO-20151103"
    assert authority["matrix"]["views"] == ["sua", "pseudo_mua"]
    assert authority["matrix"]["activity_first_n"] == 30
    assert authority["matrix"]["t4_comparator_pool_and_query_boundary"] == 50
    assert "external_subm_access" in authority["forbidden_operations"]
    assert "external_scoring_authorization_or_signature_creation" in authority["forbidden_operations"]


def test_runner_cli_exposes_no_data_checkpoint_model_or_authorization_argument() -> None:
    parser_source = Path(runner.__file__).read_text(encoding="utf-8")
    assert "--nwb" not in parser_source
    assert "--checkpoint" not in parser_source
    assert "--model" not in parser_source
    assert "--authorization" not in parser_source
    assert "--signature" not in parser_source
    args = runner.parse_args(["--mode", "dry-run"])
    assert vars(args) == {"mode": "dry-run"}


def test_dependency_source_drift_fails_before_fixture_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    relative = next(iter(parity.DEPENDENCY_SOURCE_PINS))
    monkeypatch.setitem(parity.DEPENDENCY_SOURCE_PINS, relative, "0" * 64)
    with pytest.raises(parity.SharedZero4ParityError, match="dependency source SHA-256 drift"):
        parity.verify_dependency_sources(ROOT)


def test_fixed_consumed_subc_integration_both_views_no_model_no_r2(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    result = parity.execute_consumed_subc_zero4_parity(ROOT)
    assert result["status"] == "PASS_CONSUMED_SUBC_SHARED_ZERO4_INPUT_PARITY_NO_MODEL_NO_R2"
    assert result["scope"]["views"] == ["sua", "pseudo_mua"]
    assert result["scope"]["checkpoint_files_opened"] == 0
    assert result["scope"]["model_forward_calls"] == 0
    assert result["scope"]["r2_computations"] == 0
    assert result["scope"]["external_subm_accessed"] is False
    assert result["scope"]["external_subm_scored"] is False
    assert result["zero4_descriptor_contract"]["source_t4_normalizer_value_reads"] == 0
    for view in ("sua", "pseudo_mua"):
        trace = result["views"][view]
        assert trace["activity_policy"]["indices"] == list(range(30))
        assert trace["comparator_and_query_policy"]["query_trials_start"] == 50
        assert trace["comparator_and_query_policy"]["valid_starts_exact_post50_reconstruction"]
        assert trace["channel_contract"]["side_shape"][1] == 4
        assert trace["channel_contract"]["side_matches_signal_view_channel_axis"]
        assert trace["label_invariance"]["prediction_input_exact_after_shuffle"]
        assert trace["label_invariance"]["prediction_input_exact_after_drop"]
