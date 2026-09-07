"""Unit tests for the CEBRA adaptation comparator (synthetic data only)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "cebra_exploration/src"
VENDOR = REPO_ROOT / "cebra_exploration/third_party/cebra"
for path in (REPO_ROOT, REPO_ROOT / "sua_exploration", SRC, VENDOR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("PYTHONNOUSERSITE", "1")

import cebra_comparator as core  # noqa: E402


def _rng() -> np.random.Generator:
    return core.rng_from_material(core.SEED_MATERIAL)


def test_r2_helpers_match_hand_computed_values() -> None:
    truth = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 5.0], [4.0, 4.0]], dtype=np.float64)
    estimate = np.array([[1.1, 1.8], [2.2, 3.1], [2.7, 5.4], [3.8, 4.2]], dtype=np.float64)
    residual = np.square(truth - estimate).sum(axis=0)
    total = np.square(truth - truth.mean(axis=0, keepdims=True)).sum(axis=0)
    vw_manual = 1.0 - float(residual.sum()) / float(total.sum())
    assert core.r2_pooled(truth, estimate) == pytest.approx(vw_manual)
    assert core.r2_variance_weighted(estimate, truth) == pytest.approx(vw_manual)
    assert core.score_predictions(estimate, truth, dataset="rt") == pytest.approx(vw_manual)
    assert core.score_predictions(estimate, truth, dataset="subject_m") == pytest.approx(vw_manual)
    assert core.score_predictions(estimate, truth, dataset="falcon_h1") == pytest.approx(vw_manual)


def test_matched_budget_refuses_extra_and_starved_prefix() -> None:
    assert core.assert_matched_budget("subject_m", 50) == 50
    assert core.assert_matched_budget("rt", 24) == 24
    with pytest.raises(core.CebraComparatorError, match="more than the carrier budget"):
        core.assert_matched_budget("subject_m", 51)
    with pytest.raises(core.CebraComparatorError, match="starve CEBRA"):
        core.assert_matched_budget("rt", 23)


def test_first_layer_params_and_flop_estimate_are_deterministic() -> None:
    assert core.first_layer_parameter_count(20) == 20 * 32 + 32
    assert core.samples_per_input_layer_parameter(50, 90) == pytest.approx(50 / (90 * 32 + 32))
    first = core.estimate_offset1_adapt_flops(
        n_neurons=20, n_hidden=32, n_output=8, batch_size=16, n_iterations=5
    )
    second = core.estimate_offset1_adapt_flops(
        n_neurons=20, n_hidden=32, n_output=8, batch_size=16, n_iterations=5
    )
    assert first == second
    assert first > 0
    joint = core.estimate_offset1_training_flops(
        session_n_neurons=(24, 31, 37),
        trainable_sessions=(True, True, True),
        batch_size=16,
        n_iterations=5,
        n_output=3,
    )
    frozen = core.estimate_offset1_training_flops(
        session_n_neurons=(24, 31, 37),
        trainable_sessions=(False, False, True),
        batch_size=16,
        n_iterations=5,
        n_output=3,
    )
    assert joint > frozen > 0


def test_no_adapt_undefined_when_unit_counts_differ() -> None:
    verdict = core.no_adapt_verdict([20, 31], 40)
    assert verdict["verdict"] == "CEBRA_UNDEFINED_UNIT_COUNT_MISMATCH"
    definable = core.no_adapt_verdict([20, 31, 40], 40)
    assert definable["verdict"] == "CEBRA_DEFINABLE"


def test_rt_degenerate_direction_is_undefined_velocity_is_not() -> None:
    row = core.session_row_from_counts(
        session_name="rt_fold0",
        n_channels=64,
        n_samples=800,
        n_unique_discrete_labels=1,
        has_continuous_velocity=True,
    )
    aux = core.auxiliary_verdicts(row)
    assert aux["cebra_behavior_direction"]["verdict"] == "CEBRA_UNDEFINED_DEGENERATE_DIRECTION"
    assert aux["cebra_behavior_velocity"]["verdict"] == "CEBRA_DEFINABLE"


def test_part_a_structural_priors_match_protocol() -> None:
    subm = core.audit_from_structural_prior("subject_m", view="sua")
    assert subm["verdicts"]["cebra_no_adapt"]["verdict"] == "CEBRA_UNDEFINED_UNIT_COUNT_MISMATCH"
    assert subm["verdicts"]["cebra_joint_behavior"]["verdict"] == "CEBRA_AUDIT_UNMEASURED_CALIBRATION_LENGTH"
    assert subm["verdicts"]["cebra_joint_time"]["verdict"] == core.MULTISESSION_TIME_VERDICT
    assert subm["verdicts"]["cebra_joint_time_query_unlabelled"]["verdict"] == core.MULTISESSION_TIME_VERDICT
    assert subm["verdicts"]["cebra_frozen_source_adapt"]["verdict"] == "CEBRA_AUDIT_UNMEASURED_CALIBRATION_LENGTH"
    assert "cebra_no_adapt" not in subm["buildable_arms"]
    assert subm["structural_prior"]["sample_counts_measured"] is False
    assert core.PRIMARY_ARM == "cebra_joint_behavior"

    rt = core.audit_from_structural_prior("rt")
    assert rt["verdicts"]["cebra_no_adapt"]["verdict"] == "CEBRA_UNDEFINED_UNIT_COUNT_MISMATCH"
    assert rt["verdicts"]["cebra_behavior_direction"]["verdict"] == "CEBRA_UNDEFINED_DEGENERATE_DIRECTION"
    assert rt["verdicts"]["cebra_joint_behavior"]["verdict"] == "CEBRA_AUDIT_UNMEASURED_CALIBRATION_LENGTH"

    h1 = core.audit_from_structural_prior("falcon_h1")
    assert h1["verdicts"]["cebra_no_adapt"]["verdict"] == "CEBRA_DEFINABLE"
    assert h1["verdicts"]["cebra_joint_behavior"]["verdict"] == "CEBRA_AUDIT_UNMEASURED_CALIBRATION_LENGTH"
    assert h1["verdicts"]["cebra_joint_time"]["verdict"] == core.MULTISESSION_TIME_VERDICT
    assert h1["structural_prior"]["sample_counts_measured"] is False

    m2 = core.audit_from_structural_prior("falcon_m2")
    assert m2["verdicts"]["cebra_no_adapt"]["verdict"] == "CEBRA_DEFINABLE"
    assert m2["verdicts"]["cebra_joint_behavior"]["verdict"] == "CEBRA_DEFINABLE"
    assert m2["verdicts"]["cebra_joint_time"]["verdict"] == core.MULTISESSION_TIME_VERDICT
    assert m2["verdicts"]["cebra_adapt_unaligned"]["verdict"] == "CEBRA_DEFINABLE_NEGATIVE_CONTROL"
    assert m2["verdicts"]["min_n_samples"] >= 512


def test_calibration_too_short_is_a_finding() -> None:
    rows = [
        core.session_row_from_counts(
            session_name="a",
            n_channels=10,
            n_samples=8,
            n_unique_discrete_labels=None,
            has_continuous_velocity=True,
        ),
        core.session_row_from_counts(
            session_name="b",
            n_channels=10,
            n_samples=8,
            n_unique_discrete_labels=None,
            has_continuous_velocity=True,
        ),
    ]
    verdicts = core.emit_arm_verdicts(dataset="falcon_h1", rows=rows)
    assert verdicts["cebra_joint_behavior"]["verdict"] == "CEBRA_UNDEFINED_CALIBRATION_TOO_SHORT"
    assert verdicts["cebra_joint_time"]["verdict"] == core.MULTISESSION_TIME_VERDICT
    assert verdicts["cebra_frozen_source_adapt"]["verdict"] == "CEBRA_UNDEFINED_CALIBRATION_TOO_SHORT"


def test_integrity_gate_hook_and_failure() -> None:
    hooked = core.integrity_gate("rt")
    assert hooked["status"] == "HOOK_WIRED_NOT_EXECUTED"
    assert hooked["passed"] is None
    passed = core.integrity_gate("rt", reproduced={"t4d": 0.448176, "ridge": 0.200202, "zero4": 0.179272})
    assert passed["status"] == "PASSED"
    assert passed["max_deviation"] == pytest.approx(0.0)
    with pytest.raises(core.CebraComparatorError, match="integrity gate failed"):
        core.integrity_gate("rt", reproduced={"t4d": 0.0})


def test_cpu_device_is_required() -> None:
    assert core.require_cpu_device("cpu") == "cpu"
    with pytest.raises(core.CebraComparatorError, match="refuses GPU"):
        core.require_cpu_device("cuda")
    with pytest.raises(core.CebraComparatorError, match="refuses GPU"):
        core.require_cpu_device("cuda_if_available")


def test_h1_discovery_routes_through_index_heldin_calib() -> None:
    source = Path(core.__file__).read_text(encoding="utf-8")
    assert "index_heldin_calib" in source
    assert "def discover_h1_sessions" in source
    h1_block = source.split("def discover_h1_sessions")[1].split("def discover_m2_sessions")[0]
    assert "rglob" not in h1_block
    assert "glob(" not in h1_block
    assert "*held-out" not in h1_block
    assert "index_heldin_calib" in h1_block


def test_m2_discovery_is_allowlist_not_raw_glob() -> None:
    source = Path(core.__file__).read_text(encoding="utf-8")
    m2_block = source.split("def discover_m2_sessions")[1].split("def _source_pool")[0]
    assert "rglob" not in m2_block
    assert "EXPECTED_HELDOUT_SESSIONS" in m2_block
    assert "_behavior+ecephys.nwb" in m2_block
    assert "held-out-calib" in m2_block


def test_collapse_helper_was_deleted() -> None:
    source = Path(core.__file__).read_text(encoding="utf-8")
    assert "def collapse_multisession_to_template" not in source
    assert "def official_adapt_is_multisession_blocked" not in source
    assert "def adapt_target_input_layer" not in source
    assert "cebra_behavior_adapt" not in source
    assert "cebra_time_adapt" not in source


def test_batch_size_never_exceeds_prefix() -> None:
    assert core.resolve_batch_size(40) == 40
    assert core.resolve_batch_size(2000) == core.RECOMMENDED_BATCH_SIZE


def _synthetic_sessions(*, n_channels: tuple[int, ...], n_samples: int = 48) -> tuple[list[np.ndarray], list[np.ndarray]]:
    rng = _rng()
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    t = np.linspace(0.0, 2.0 * np.pi, n_samples, endpoint=False)
    labels = np.stack([np.cos(t), np.sin(t)], axis=1)
    for width in n_channels:
        weights = rng.normal(size=(width, 2))
        neural = labels @ weights.T + 0.05 * rng.normal(size=(n_samples, width))
        xs.append(np.asarray(neural, dtype=np.float64))
        ys.append(np.asarray(labels, dtype=np.float64))
    return xs, ys


def test_multisession_mismatched_n_fits_on_cpu_tiny() -> None:
    xs, ys = _synthetic_sessions(n_channels=(8, 11), n_samples=40)
    estimator = core.fit_source_cebra(xs, ys, max_iterations=2, output_dimension=3)
    assert estimator.num_sessions == 2
    emb0 = core.transform_embedding(estimator, xs[0], session_id=0)
    emb1 = core.transform_embedding(estimator, xs[1], session_id=1)
    assert emb0.shape == (40, 3)
    assert emb1.shape == (40, 3)
    with pytest.raises(Exception):
        core.transform_embedding(estimator, xs[1], session_id=0)


def test_frozen_source_encoders_are_bit_identical_after_joint_adapt() -> None:
    xs, ys = _synthetic_sessions(n_channels=(8, 11, 13), n_samples=40)
    source = core.fit_source_cebra(xs[:2], ys[:2], max_iterations=3, output_dimension=3)
    before_0 = core.numpy_encoder_params(source, 0)
    before_1 = core.numpy_encoder_params(source, 1)
    joint = core.fit_joint_cebra(
        xs,
        ys,
        max_iterations=4,
        output_dimension=3,
        freeze_sessions=[0, 1],
        init_from=source,
    )
    assert core.encoder_params_equal(before_0, core.numpy_encoder_params(joint, 0))
    assert core.encoder_params_equal(before_1, core.numpy_encoder_params(joint, 1))
    assert all(not parameter.requires_grad for parameter in joint.model_[0].parameters())
    assert all(not parameter.requires_grad for parameter in joint.model_[1].parameters())
    assert all(parameter.requires_grad for parameter in joint.model_[2].parameters())
    target_emb = core.transform_embedding(joint, xs[2], session_id=2)
    source_emb = core.transform_embedding(joint, xs[0], session_id=0)
    assert target_emb.shape == (40, 3)
    assert source_emb.shape == (40, 3)


def test_target_query_labels_never_reach_the_estimator() -> None:
    xs, ys = _synthetic_sessions(n_channels=(8, 11), n_samples=40)
    leaked = np.asarray(ys[1], dtype=np.float64)
    with pytest.raises(core.LabelLeakError, match="query labels must not enter"):
        core.run_arm_on_synthetic_fold(
            arm="cebra_joint_behavior",
            neural_sessions=xs,
            label_sessions=ys,
            target_index=1,
            dataset="subject_m",
            max_source_iterations=1,
            output_dimension=3,
            target_query_labels=leaked,
        )
    with pytest.raises(core.LabelLeakError):
        core.refuse_target_query_labels(leaked)
    core.refuse_target_query_labels(None)


def test_query_unlabelled_arm_rejects_missing_query_activity() -> None:
    xs, ys = _synthetic_sessions(n_channels=(8, 11), n_samples=40)
    with pytest.raises(core.CebraComparatorError, match="query-unlabelled variant requires"):
        core.run_arm_on_synthetic_fold(
            arm="cebra_joint_time_query_unlabelled",
            neural_sessions=xs,
            label_sessions=ys,
            target_index=1,
            dataset="subject_m",
            max_source_iterations=1,
            output_dimension=3,
        )


def test_no_adapt_refuses_mismatched_n_without_mapping() -> None:
    xs, ys = _synthetic_sessions(n_channels=(8, 11), n_samples=40)
    result = core.run_arm_on_synthetic_fold(
        arm="cebra_no_adapt",
        neural_sessions=xs,
        label_sessions=ys,
        target_index=1,
        dataset="subject_m",
        max_source_iterations=2,
        output_dimension=3,
    )
    assert result["status"] == "CEBRA_UNDEFINED_UNIT_COUNT_MISMATCH"
    assert result["target_score"] is None
    assert result["cost"]["target_parameter_count"] == 0


def test_rt_discovery_refuses_indy_and_uses_sealed_loader() -> None:
    source = Path(core.__file__).read_text(encoding="utf-8")
    rt_block = source.split("def discover_rt_sessions")[1].split("def discover_h1_sessions")[0]
    assert "000129" in rt_block
    assert "sub-Indy" in rt_block
    assert "find_rt_sessions" in rt_block
    assert "session_name_from_nwb_path" in rt_block
    assert "rt_classical_comparators" in rt_block
    with pytest.raises(core.CebraComparatorError, match="refusing NLB MC_RTT"):
        core.discover_rt_sessions(Path("/tmp/data/000129/sub-Indy"))
    real = REPO_ROOT / "sua_exploration/data/dandi_000688/sub-C"
    if real.is_dir():
        sessions = core.discover_rt_sessions(real)
        assert len(sessions) == core.RT_EXPECTED_FOLDS
        assert all(name.startswith("ses-RT-") for name in sessions)


def test_dry_run_does_not_score() -> None:
    payload = core.dry_run_all(
        repo_root=REPO_ROOT,
        data_dir_subject_m=None,
        data_dir_rt=None,
        data_dir_h1=None,
        data_dir_m2=None,
    )
    assert payload["scoring_arm_executed"] is False
    assert payload["device"] == "cpu"
    assert payload["primary_arm"] == "cebra_joint_behavior"
    assert payload["negative_control_arm"] == "cebra_adapt_unaligned"
    assert payload["target_query_labels_in_fit"] is False
    assert payload["query_activity_in_primary"] is False
    assert "falcon_h1" in payload["adapters"]
    assert payload["adapters"]["falcon_h1"]["discovery"].endswith("index_heldin_calib")
    assert payload["adapters"]["falcon_h1"]["held_out_in_scope"] is False
    assert payload["adapters"]["falcon_m2"]["held_out_calib_in_scope"] is True
    assert payload["adapters"]["rt"]["integrity_gate"]["status"] == "HOOK_WIRED_NOT_EXECUTED"
    assert "NOT data/000129/sub-Indy" in payload["adapters"]["rt"]["discovery"]


def test_implementation_files_exist() -> None:
    assert (REPO_ROOT / "cebra_exploration/docs/TRACK_B_CEBRA_COMPARATOR_PROTOCOL.md").is_file()
    assert (REPO_ROOT / "cebra_exploration/src/cebra_comparator.py").is_file()
    assert (REPO_ROOT / "cebra_exploration/scripts/run_cebra_comparator.py").is_file()
    assert (REPO_ROOT / "cebra_exploration/tests/test_cebra_comparator.py").is_file()
    provenance = (REPO_ROOT / "cebra_exploration/third_party/CEBRA_PROVENANCE.txt").read_text(encoding="utf-8")
    assert "freeze_sessions" in provenance
    assert "init_from" in provenance
    assert "local_modifications" in provenance


def test_positive_control_gate_every_arm_including_negative_control() -> None:
    """Required gate: recover a shared synthetic latent, or the arm is void.

    The negative-control arm must fail recovery so the gate is proven to have teeth.
    ``cebra_no_adapt`` is undefined on mismatched N and is asserted as such.
    """
    gate = core.run_positive_control_gate()
    assert gate["status"] == "PASSED"
    for arm in core.ARMS:
        assert arm in gate["arms"]
        assert gate["checks"][arm]["passed"] is True
    for arm in ("cebra_joint_behavior", "cebra_frozen_source_adapt"):
        target_r2 = gate["arms"][arm]["target_r2"]
        source_r2 = gate["arms"][arm]["source_r2"]
        assert source_r2 is not None and source_r2 >= core.POSITIVE_CONTROL_MIN_TARGET_R2
        assert target_r2 is not None and target_r2 >= core.POSITIVE_CONTROL_MIN_TARGET_R2
    unaligned = gate["arms"][core.NEGATIVE_CONTROL_ARM]["target_r2"]
    assert unaligned is not None
    assert unaligned < core.POSITIVE_CONTROL_UNALIGNED_MAX_TARGET_R2
    assert gate["arms"]["cebra_no_adapt"]["status"] == "CEBRA_UNDEFINED_UNIT_COUNT_MISMATCH"
    assert gate["arms"]["cebra_joint_time"]["status"] == core.MULTISESSION_TIME_VERDICT
    assert gate["arms"]["cebra_joint_time_query_unlabelled"]["status"] == core.MULTISESSION_TIME_VERDICT
    frozen_cost = gate["arms"]["cebra_frozen_source_adapt"]["cost"]["target_parameter_count"]
    joint_cost = gate["arms"]["cebra_joint_behavior"]["cost"]["target_parameter_count"]
    assert 0 < frozen_cost < joint_cost
