from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "sua_exploration/scripts/aggregate_t4_paired_view_c1_three_arm_terminal.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("c1_three_arm_terminal_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


agg = _load_module()


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _completion(path: Path) -> None:
    _write(
        path,
        {
            "status": agg.EXPECTED_COMPLETION_STATUS,
            "completed_seeds": [
                {"seed": seed, "status": "completed"} for seed in agg.SEEDS
            ],
            "scheduler_development_score_invocations": 0,
            "evaluator_development_score_invocations_before_completion": 0,
            "development_score_authorized_by_this_receipt": False,
        },
    )


def _zero4_payload(
    *, path: Path, completion: Path, seed: int, view: str, values: list[float]
) -> None:
    sessions = [f"dev_{index}" for index in range(6)]
    mean = float(np.mean(values))
    _write(
        path,
        {
            "schema_version": 1,
            "purpose": "shared_zero4_terminal_fixed_development_evaluation",
            "generated_by": "eval_paired_view_c1_shared_zero4_terminal.py",
            "variant": "B3S",
            "seed": seed,
            "task": "CO",
            "signal_view": view,
            "shared_weights": True,
            "side_feature_group": "shared_zero4_direct_standardized",
            "checkpoint_selection_rule": "fixed_terminal_epoch_011_no_selection",
            "checkpoint_epoch_index": 11,
            "protocol_epoch_number": 12,
            "uses_backward_gradients": False,
            "development_uses_backward_gradients": False,
            "no_test_files_evaluated": True,
            "formal_sua_files_opened": False,
            "subm_nwb_files_opened": False,
            "matrix_completion_status": agg.EXPECTED_COMPLETION_STATUS,
            "matrix_completion_receipt": str(completion.resolve()),
            "matrix_completion_receipt_sha256": _sha(completion),
            "protocol": {
                "source_activity_calibration_n": 10,
                "development_activity_calibration_n": 30,
                "descriptor_label_pool_n": None,
                "query_start_trial": 50,
                "trials_30_49_enter_zero4_identity_or_descriptor": False,
            },
            "descriptor_access": {
                "target_direction_label_reads_for_descriptor": 0,
                "t4_trial_rate_reads_for_descriptor": 0,
                "target_t4_rate_fit_calls": 0,
                "raw_t4_constructed": False,
                "source_t4_normalizer_arithmetic_performed": False,
                "all_development_records_bitwise_float32_zero": True,
            },
            "session_splits": {"val": sessions},
            "per_session_r2": dict(zip(sessions, values)),
            "mean_r2": mean,
            "variant_score": mean,
            "checkpoint": f"fixture/seed{seed}/epoch_011.ckpt",
            "checkpoint_sha256": f"{seed:064x}"[-64:],
            "run_metadata_sha256": "a" * 64,
        },
    )


def test_incomplete_matrix_opens_no_score_artifact_from_any_arm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    completion = tmp_path / "completion.json"
    _completion(completion)
    zero_root = tmp_path / "zero4"
    # Five deliberately invalid JSON score files: if the fail-closed gate opens
    # even one of them, the test fails before returning its missing report.
    for seed in agg.SEEDS:
        for view in agg.VIEWS:
            if (seed, view) == (44, "pseudo_mua"):
                continue
            path = agg._zero4_paths(zero_root)[(seed, view)]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("not-json\n", encoding="utf-8")

    opened: list[Path] = []
    original = agg.legacy.load_json

    def spy(path: Path):
        opened.append(Path(path).resolve())
        return original(path)

    monkeypatch.setattr(agg.legacy, "load_json", spy)
    monkeypatch.setattr(
        agg,
        "_load_t4_ts4",
        lambda **_kwargs: pytest.fail("T4/TS4 loader must not run before all zero4 slots exist"),
    )
    args = argparse.Namespace(
        out=tmp_path / "must_not_exist.json",
        zero4_result_root=zero_root,
        zero4_matrix_completion_receipt=completion,
        t4_ts4_receipt=tmp_path / "legacy_receipt.json",
        t4_ts4_result_root=tmp_path / "legacy_results",
        bootstrap_draws=1000,
        bootstrap_seed=7,
    )
    code, report = agg.run(args)
    assert code == 3
    assert report["status"] == "missing_zero4_terminal_seeds"
    assert report["complete_two_view_score_seed_count"] == 2
    assert report["missing_score_slot_count"] == 1
    assert report["score_artifacts_opened"] == 0
    assert opened == [completion.resolve()]
    assert not args.out.exists()


def test_two_terminal_seeds_with_all_score_paths_still_opens_no_score(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    completion = tmp_path / "completion.json"
    _completion(completion)
    receipt = json.loads(completion.read_text(encoding="utf-8"))
    receipt["completed_seeds"] = receipt["completed_seeds"][:2]
    _write(completion, receipt)
    zero_root = tmp_path / "zero4"
    for path in agg._zero4_paths(zero_root).values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("not-json\n", encoding="utf-8")
    original = agg.legacy.load_json
    opened: list[Path] = []

    def spy(path: Path):
        opened.append(Path(path).resolve())
        return original(path)

    monkeypatch.setattr(agg.legacy, "load_json", spy)
    args = argparse.Namespace(
        out=tmp_path / "must_not_exist.json",
        zero4_result_root=zero_root,
        zero4_matrix_completion_receipt=completion,
        t4_ts4_receipt=tmp_path / "legacy_receipt.json",
        t4_ts4_result_root=tmp_path / "legacy_results",
        bootstrap_draws=1000,
        bootstrap_seed=7,
    )
    code, report = agg.run(args)
    assert code == 3
    assert report["terminal_seed_count"] == 2
    assert report["complete_two_view_score_seed_count"] == 3
    assert report["missing_terminal_seeds"] == [44]
    assert report["score_artifacts_opened"] == 0
    assert opened == [completion.resolve()]


def test_three_arm_aggregate_reports_exact_matrices_means_two_se_and_bootstrap() -> None:
    base = np.arange(18, dtype=np.float64).reshape(3, 6) / 100.0
    matrices = {
        view: {
            "shared_t4": base + (0.50 if view == "sua" else 0.40),
            "shared_zero4": base + (0.20 if view == "sua" else 0.10),
            "shared_ts4": base + (0.30 if view == "sua" else 0.25),
        }
        for view in agg.VIEWS
    }
    result = agg.aggregate_matrices(
        matrices,
        sessions=[f"dev_{index}" for index in range(6)],
        draws=1000,
        bootstrap_seed=123,
    )
    sua = result["sua"]
    assert np.asarray(sua["absolute"]["shared_t4"]["seed_by_session_r2"]).shape == (3, 6)
    assert sua["absolute"]["shared_t4"]["seed_mean_r2"] == pytest.approx(
        matrices["sua"]["shared_t4"].mean(axis=1)
    )
    t4_z4 = sua["paired_contrasts"]["shared_t4_minus_shared_zero4"]
    assert np.asarray(t4_z4["all_18_seed_session_deltas"]).shape == (3, 6)
    assert t4_z4["mean_delta"] == pytest.approx(0.30)
    assert t4_z4["seed_mean_deltas"] == pytest.approx([0.30, 0.30, 0.30])
    assert t4_z4["session_mean_deltas"] == pytest.approx([0.30] * 6)
    assert t4_z4["seed_mean_se_paired"] == pytest.approx(0.0, abs=1e-15)
    assert t4_z4["paired_two_se_interval"] == pytest.approx([0.30, 0.30])
    assert t4_z4["two_way_seed_session_bootstrap_95"]["draws"] == 1000
    t4_ts4 = result["pseudo_mua"]["paired_contrasts"]["shared_t4_minus_shared_ts4"]
    assert t4_ts4["mean_delta"] == pytest.approx(0.15)
    for view in agg.VIEWS:
        assert result[view]["frozen_view_interpretation"] == {
            "classification": "both_effective",
            "allowed_classifications": [
                "both_effective", "absolute_only", "attachment_only", "neither"
            ],
            "shared_t4_minus_shared_zero4_effective": True,
            "shared_t4_minus_shared_ts4_effective": True,
            "zero4_contrast_role": "absolute_descriptor_content_system_contrast",
            "ts4_contrast_role": "row_attachment_content_sensitivity_system_contrast",
            "evidence_scope": "reused_development_only_not_formal_not_subm",
            "formal_claim_authorized": False,
            "no_cross_view_rescue": True,
            "cross_view_average_computed": False,
        }
        for contrast in result[view]["paired_contrasts"].values():
            decision = contrast["frozen_effectiveness_decision"]
            assert decision["overall_effective"] is True
            assert all(decision["gates"].values())
            assert decision["no_cross_view_rescue"] is True


def _decision_fixture(
    *,
    mean_delta: float = 0.03,
    seed_means: list[float] | None = None,
    session_means: list[float] | None = None,
    two_se_lower: float = 0.01,
    bootstrap_lower: float = 0.01,
    absolute_grand: float = 0.20,
    absolute_seed_means: list[float] | None = None,
) -> dict:
    paired = {
        "mean_delta": mean_delta,
        "seed_mean_deltas": seed_means or [0.03, 0.03, 0.03],
        "session_mean_deltas": session_means or [0.03] * 6,
        "paired_two_se_lower": two_se_lower,
        "two_way_seed_session_bootstrap_95": {"lower": bootstrap_lower},
    }
    absolute = {
        "grand_mean_r2": absolute_grand,
        "seed_mean_r2": absolute_seed_means or [0.20, 0.20, 0.20],
    }
    return agg._frozen_effectiveness_decision(paired, absolute_t4=absolute)


def test_frozen_decision_mean_delta_exactly_plus_0p03_passes() -> None:
    decision = _decision_fixture(mean_delta=0.03)
    assert decision["gates"]["mean_delta_at_least_plus_0p03"] is True
    assert decision["overall_effective"] is True


def test_frozen_decision_four_of_six_positive_sessions_fails() -> None:
    decision = _decision_fixture(session_means=[0.05, 0.04, 0.03, 0.02, 0.0, -0.01])
    assert decision["gates"]["at_least_five_of_six_session_means_positive"] is False
    assert decision["overall_effective"] is False


@pytest.mark.parametrize(
    ("two_se_lower", "bootstrap_lower", "failed_gate"),
    [
        (-0.001, 0.01, "paired_two_se_lower_positive"),
        (0.01, -0.001, "two_way_bootstrap_lower_positive"),
        (0.0, 0.01, "paired_two_se_lower_positive"),
        (0.01, 0.0, "two_way_bootstrap_lower_positive"),
    ],
)
def test_frozen_decision_ci_touching_or_crossing_zero_fails(
    two_se_lower: float, bootstrap_lower: float, failed_gate: str
) -> None:
    decision = _decision_fixture(
        two_se_lower=two_se_lower, bootstrap_lower=bootstrap_lower
    )
    assert decision["gates"][failed_gate] is False
    assert decision["overall_effective"] is False


def test_frozen_decision_negative_absolute_t4_fails() -> None:
    decision = _decision_fixture(
        absolute_grand=-0.10, absolute_seed_means=[-0.08, -0.10, -0.12]
    )
    assert decision["gates"]["absolute_t4_grand_mean_positive"] is False
    assert decision["gates"]["all_three_absolute_t4_seed_means_positive"] is False
    assert decision["overall_effective"] is False


@pytest.mark.parametrize(
    ("absolute_effective", "attachment_effective", "expected"),
    [
        (True, True, "both_effective"),
        (True, False, "absolute_only"),
        (False, True, "attachment_only"),
        (False, False, "neither"),
    ],
)
def test_frozen_view_interpretation_has_all_four_exclusive_states(
    absolute_effective: bool, attachment_effective: bool, expected: str
) -> None:
    assert agg._view_interpretation(
        absolute_effective=absolute_effective,
        attachment_effective=attachment_effective,
    ) == expected


def test_complete_synthetic_gate_writes_final_only_after_all_zero4_slots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    completion = tmp_path / "completion.json"
    _completion(completion)
    zero_root = tmp_path / "zero4"
    sessions = [f"dev_{index}" for index in range(6)]
    for seed_index, seed in enumerate(agg.SEEDS):
        for view_index, view in enumerate(agg.VIEWS):
            values = [0.10 + seed_index * 0.01 + view_index * 0.02 + j * 0.001 for j in range(6)]
            _zero4_payload(
                path=agg._zero4_paths(zero_root)[(seed, view)],
                completion=completion,
                seed=seed,
                view=view,
                values=values,
            )

    def fake_legacy(**_kwargs):
        return (
            {
                view: {
                    "shared_t4": np.full((3, 6), 0.50 + index * 0.02),
                    "shared_ts4": np.full((3, 6), 0.30 + index * 0.01),
                }
                for index, view in enumerate(agg.VIEWS)
            },
            sessions,
            {"fixture": {"score_values": "not persisted in evidence"}},
        )

    monkeypatch.setattr(agg, "_load_t4_ts4", fake_legacy)
    output = tmp_path / "aggregate.json"
    args = argparse.Namespace(
        out=output,
        zero4_result_root=zero_root,
        zero4_matrix_completion_receipt=completion,
        t4_ts4_receipt=tmp_path / "legacy_receipt.json",
        t4_ts4_result_root=tmp_path / "legacy_results",
        bootstrap_draws=1000,
        bootstrap_seed=9,
    )
    code, report = agg.run(args)
    assert code == 0 and report["status"].startswith("completed_reused_development")
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["evidence_scope"] == "reused_development_only_not_formal_not_subm"
    assert payload["frozen_protocol"] == {
        "activity_calibration_trials": 30,
        "calibration_time_backward_or_optimizer_steps": 0,
        "checkpoint_estimators_are_intentionally_different": True,
        "query_start_trial": 50,
        "shared_t4_checkpoint_estimator": "mean_protocol_epochs_5_through_12",
        "shared_ts4_checkpoint_estimator": "mean_protocol_epochs_5_through_12",
        "shared_zero4_checkpoint_estimator": "fixed_epoch_011_checkpoint_protocol_epoch_12_only",
        "t4_ts4_descriptor_pool_trials": 50,
        "zero4_descriptor_pool_trials": None,
    }
    assert set(payload["views"]) == set(agg.VIEWS)
    assert os.stat(output).st_mode & 0o777 == 0o444


def test_zero4_loader_rejects_checkpoint_or_protocol_drift(tmp_path: Path) -> None:
    completion = tmp_path / "completion.json"
    _completion(completion)
    path = tmp_path / "zero.json"
    _zero4_payload(
        path=path,
        completion=completion,
        seed=42,
        view="sua",
        values=[0.1] * 6,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["checkpoint_epoch_index"] = 10
    _write(path, payload)
    with pytest.raises(ValueError, match="identity drift"):
        agg._load_zero4_view(
            path=path,
            seed=42,
            view="sua",
            completion_path=completion,
            completion_sha256=_sha(completion),
        )
