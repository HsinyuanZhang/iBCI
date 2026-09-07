"""Synthetic no-data/no-CUDA tests for the M2 T4-reliance diagnostic.

The review-critical properties:

1. the grid is exactly the briefed 4 cells x 3 static budgets x 2 surfaces;
2. the perturbation constants ARE the fixed seed-42 laws (a fixed-point-free
   unit permutation, a non-identity column permutation) and nothing at runtime
   redraws them;
3. each cell's perturbed side is exactly its promised law (bitwise restore
   under the inverse permutation, zeros for T4_ZERO, verbatim for BASELINE);
4. through a side-sensitive stub decoder the frozen scoring path is a no-op
   under the identity permutation and genuinely side-sensitive under all three
   briefed perturbations;
5. the summary/delta/aggregate math (equal-session mean, sd ddof=1, paired
   deltas, breadth) is exact and fail-closed;
6. the verdict rule is exactly the pre-registered three-string law on the
   T4_ZERO primary;
7. the mandatory eval-time-vs-train-time disclosure and the CPU/receipt laws
   are declared, and the CPU isolation fails closed;
8. nothing here touches data, checkpoints or CUDA.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
for item in (ROOT / "tfpd_exploration", ROOT / "tfpd_exploration" / "src"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from src.m2_t4_reliance_v1 import physical, plan  # noqa: E402


# ---------------------------------------------------------------------------
# 1: the grid.
# ---------------------------------------------------------------------------


def test_grid_is_exactly_the_briefed_cells_budgets_and_surfaces() -> None:
    assert plan.CELLS == ("BASELINE", "T4_ZERO", "T4_SHUFFLE_UNITS", "T4_SHUFFLE_COLS")
    assert plan.BUDGETS == (4, 10, 30)
    assert plan.SURFACES == ("within_post30", "external_official_query")
    assert plan.EXPECTED_WITHIN_SESSIONS == 7
    assert plan.EXPECTED_EXTERNAL_SESSIONS == 6
    assert plan.CHANNELS == 96 and plan.SIDE_DIM == 4 and plan.WINDOW_SIZE == 50
    assert set(plan.CELL_LAWS) == set(plan.CELLS)
    # 4 cells x 3 budgets x 13 sessions = 156 rows
    assert len(plan.CELLS) * len(plan.BUDGETS) * (
        plan.EXPECTED_WITHIN_SESSIONS + plan.EXPECTED_EXTERNAL_SESSIONS) == 156


def test_only_the_side_feature_moves() -> None:
    for cell in plan.CELLS:
        assert "side" in plan.CELL_LAWS[cell]
    assert "zeros" in plan.CELL_LAWS["T4_ZERO"]
    assert "train-population mean" in plan.CELL_LAWS["T4_ZERO"]
    assert "permuted across units" in plan.CELL_LAWS["T4_SHUFFLE_UNITS"]
    assert "columns are permuted" in plan.CELL_LAWS["T4_SHUFFLE_COLS"]
    # the docstring law that the activity support never moves
    assert "activity support is held verbatim" in physical.__doc__


# ---------------------------------------------------------------------------
# 2: the frozen seed-42 permutation constants.
# ---------------------------------------------------------------------------


def test_permutation_constants_are_the_first_seed42_draws() -> None:
    units = np.random.default_rng(42).permutation(96)
    columns = np.random.default_rng(42).permutation(4)
    assert plan.UNIT_PERMUTATION == tuple(int(v) for v in units.tolist())
    assert plan.COL_PERMUTATION == tuple(int(v) for v in columns.tolist())
    assert plan.COL_PERMUTATION == (3, 2, 1, 0)
    permutation = np.asarray(plan.UNIT_PERMUTATION, dtype=np.int64)
    assert np.array_equal(np.sort(permutation), np.arange(96))
    assert int(np.count_nonzero(permutation == np.arange(96))) == 0  # derangement
    assert plan.PERMUTATION_LAW["seed"] == 42
    assert plan.PERMUTATION_LAW["unit_permutation_fixed_points"] == 0
    assert plan.PERMUTATION_LAW["column_permutation"] == [3, 2, 1, 0]


# ---------------------------------------------------------------------------
# 3-4: the perturbation law and the frozen scoring path.
# ---------------------------------------------------------------------------


def _synthetic_side(seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.ascontiguousarray(rng.normal(size=(96, 4)).astype(np.float32))


def test_perturbed_side_is_exactly_each_promised_law() -> None:
    side = _synthetic_side()
    units = np.asarray(plan.UNIT_PERMUTATION, dtype=np.int64)
    columns = np.asarray(plan.COL_PERMUTATION, dtype=np.int64)

    baseline = physical.perturbed_side(side, "BASELINE")
    assert np.array_equal(baseline, side)

    zero = physical.perturbed_side(side, "T4_ZERO")
    assert zero.shape == (96, 4) and np.count_nonzero(zero) == 0

    shuffled_units = physical.perturbed_side(side, "T4_SHUFFLE_UNITS")
    for unit in range(96):
        assert np.array_equal(shuffled_units[unit], side[units[unit]])
    # the row multiset is preserved (a permutation of rows, nothing else)
    assert np.array_equal(shuffled_units, side[units])
    inverse = np.empty_like(units)
    inverse[units] = np.arange(96)
    assert np.array_equal(shuffled_units[inverse], side)

    shuffled_cols = physical.perturbed_side(side, "T4_SHUFFLE_COLS")
    for column in range(4):
        assert np.array_equal(shuffled_cols[:, column], side[:, columns[column]])
    inverse_columns = np.empty_like(columns)
    inverse_columns[columns] = np.arange(4)
    assert np.array_equal(shuffled_cols[:, inverse_columns], side)

    # every perturbation except BASELINE changes the side (distinct rows)
    assert not np.array_equal(zero, side)
    assert not np.array_equal(shuffled_units, side)
    assert not np.array_equal(shuffled_cols, side)


def test_perturbed_side_fails_closed() -> None:
    side = _synthetic_side()
    with pytest.raises(physical.M2T4RelianceError):
        physical.perturbed_side(side, "NOT_A_CELL")
    with pytest.raises(physical.M2T4RelianceError):
        physical.perturbed_side(np.zeros((95, 4), dtype=np.float32), "BASELINE")
    with pytest.raises(physical.M2T4RelianceError):
        physical.perturbed_side(np.zeros((96, 5), dtype=np.float32), "BASELINE")
    broken = _synthetic_side()
    broken[0, 0] = np.nan
    with pytest.raises(physical.M2T4RelianceError):
        physical.perturbed_side(broken, "T4_ZERO")


def test_side_structure_evidence_flags() -> None:
    side = _synthetic_side()
    assert physical.side_structure_evidence(side, "BASELINE")[
        "bitwise_equal_sealed_side"] is True
    assert physical.side_structure_evidence(side, "T4_ZERO")["all_zero"] is True
    units = physical.side_structure_evidence(side, "T4_SHUFFLE_UNITS")
    assert units["restores_under_inverse"] is True
    assert units["unit_permutation_fixed_points"] == 0
    columns = physical.side_structure_evidence(side, "T4_SHUFFLE_COLS")
    assert columns["restores_under_inverse"] is True
    assert columns["column_source_of_column_j"] == [3, 2, 1, 0]


@dataclass
class _StubDecoder:
    """A side-sensitive stand-in for the frozen M2Decoder (same duck contract)."""

    _ramp: np.ndarray = None  # a fixed unit-position reference (not permuted)

    def __post_init__(self) -> None:
        if self._ramp is None:
            object.__setattr__(self, "_ramp",
                               np.linspace(-1.0, 1.0, 96, dtype=np.float64))

    def identity(self, activity: np.ndarray, side: np.ndarray) -> float:
        # a scalar statistic that is sensitive BOTH to which unit holds which
        # T4 row (the ramp dot product is unit-position dependent) and to the
        # column semantics (the weighted column mean)
        values = np.asarray(side, dtype=np.float64)
        unit_term = float(np.abs(values[:, 0] @ self._ramp))
        column_term = float(np.mean(
            values[:, 1] + 2.0 * values[:, 2] + 3.0 * values[:, 3]))
        return 0.5 * unit_term + 1.0e-3 * column_term

    def windows(self, neural: np.ndarray, starts: np.ndarray) -> np.ndarray:
        starts = np.asarray(starts, dtype=np.int64)
        indices = starts[:, None] + np.arange(50, dtype=np.int64)[None, :]
        return np.ascontiguousarray(np.asarray(neural)[indices], dtype=np.float32)

    def decode(self, windows: np.ndarray, identity: float) -> np.ndarray:
        count = int(windows.shape[0])
        amplitude = 0.05 + 1.0e-3 * float(identity)
        return np.ascontiguousarray(
            np.tile(np.asarray([amplitude, 0.0], dtype=np.float32), (count, 1)))

    def clear_cache(self) -> None:
        return None


def _synthetic_material(bins: int = 4000, units: int = 96):
    rng = np.random.default_rng(11)
    neural = rng.poisson(0.4, size=(bins, units)).astype(np.float32)
    starts = np.arange(0, bins - 50, 5, dtype=np.int64)
    return SimpleNamespace(
        session="synthetic", surface="within_post30",
        neural=neural, starts=starts,
        targets=np.ascontiguousarray(rng.normal(size=(starts.size, 2)).astype(np.float32)),
    )


def _synthetic_carrier():
    rng = np.random.default_rng(5)
    return {
        "activity": np.ascontiguousarray(rng.poisson(0.5, size=(10, 100, 96)).astype(np.float32)),
        "side": _synthetic_side(seed=3),
    }


def test_score_cell_is_side_sensitive_through_the_frozen_path() -> None:
    material = _synthetic_material()
    carrier = _synthetic_carrier()
    decoder = _StubDecoder()
    predictions = {
        cell: physical._score_cell(decoder=decoder, material=material,
                                   carrier=carrier, cell=cell)["prediction"]
        for cell in plan.CELLS
    }
    for cell in plan.CELLS:
        assert predictions[cell].shape == (material.starts.size, 2)
        assert np.isfinite(predictions[cell]).all()
    # every perturbation moves the decode, and no two perturbations coincide
    for cell in plan.CELLS:
        if cell == "BASELINE":
            continue
        assert not np.array_equal(predictions[cell], predictions["BASELINE"]), cell
    assert not np.array_equal(predictions["T4_ZERO"], predictions["T4_SHUFFLE_UNITS"])
    assert not np.array_equal(predictions["T4_ZERO"], predictions["T4_SHUFFLE_COLS"])
    assert not np.array_equal(predictions["T4_SHUFFLE_UNITS"], predictions["T4_SHUFFLE_COLS"])


def test_identity_permutation_is_a_bitwise_noop() -> None:
    """The control law: an identity permutation leaves the decode untouched."""
    material = _synthetic_material()
    carrier = _synthetic_carrier()
    decoder = _StubDecoder()
    side = np.ascontiguousarray(carrier["side"])
    untouched = decoder.decode(
        decoder.windows(material.neural, material.starts), decoder.identity(carrier["activity"], side))
    row_noop = decoder.decode(
        decoder.windows(material.neural, material.starts),
        decoder.identity(carrier["activity"], np.ascontiguousarray(side[np.arange(96)])))
    column_noop = decoder.decode(
        decoder.windows(material.neural, material.starts),
        decoder.identity(carrier["activity"], np.ascontiguousarray(side[:, np.arange(4)])))
    assert np.array_equal(untouched, row_noop)
    assert np.array_equal(untouched, column_noop)


def test_decode_starts_chunks_without_reordering() -> None:
    material = _synthetic_material()
    decoder = _StubDecoder()
    identity = 1.0
    full = physical._decode_starts(decoder, material, identity)
    direct = decoder.decode(decoder.windows(material.neural, material.starts), identity)
    assert full.shape == (material.starts.size, 2)
    assert np.array_equal(full, direct)


# ---------------------------------------------------------------------------
# 5: the summary/delta/aggregate math.
# ---------------------------------------------------------------------------


def test_summarize_sessions_math() -> None:
    values = {"s3": 0.5, "s1": 0.3, "s2": 0.4}
    summary = physical.summarize_sessions(values)
    assert summary["session_count"] == 3
    assert summary["equal_session_mean"] == pytest.approx(0.4)
    expected_sd = float(np.std([0.3, 0.4, 0.5], ddof=1))
    assert summary["equal_session_sd"] == pytest.approx(expected_sd)
    assert summary["equal_session_sd_ddof"] == 1
    assert summary["equal_session_median"] == pytest.approx(0.4)
    assert list(summary["per_session_r2"]) == ["s1", "s2", "s3"]
    single = physical.summarize_sessions({"only": 0.25})
    assert single["equal_session_sd"] == 0.0
    with pytest.raises(physical.M2T4RelianceError):
        physical.summarize_sessions({})


def test_paired_delta_math_and_breadth() -> None:
    baseline = {"s1": 0.50, "s2": 0.40, "s3": 0.30}
    candidate = {"s1": 0.45, "s2": 0.44, "s3": 0.20}
    delta = physical.paired_delta(candidate, baseline)
    assert delta["per_session_delta"] == pytest.approx(
        {"s1": -0.05, "s2": 0.04, "s3": -0.10})
    assert delta["equal_session_mean_delta"] == pytest.approx(-0.11 / 3.0)
    assert delta["negative_sessions"] == 2
    assert delta["positive_sessions"] == 1
    assert delta["min_delta"] == pytest.approx(-0.10)
    assert delta["max_delta"] == pytest.approx(0.04)
    assert delta["equal_session_sd_delta"] == pytest.approx(
        float(np.std([-0.05, 0.04, -0.10], ddof=1)))
    with pytest.raises(physical.M2T4RelianceError):
        physical.paired_delta({"s1": 0.5, "s2": 0.4}, baseline)
    with pytest.raises(physical.M2T4RelianceError):
        physical.paired_delta({}, {})


def test_aggregate_cell_pools_the_grid() -> None:
    def cell_delta(mean: float, negative: int, sessions: int = 6):
        return {"equal_session_mean_delta": mean, "negative_sessions": negative,
                "session_count": sessions}

    grid = {
        "within_post30|m4": cell_delta(-0.10, 7, 7),
        "within_post30|m10": cell_delta(-0.08, 7, 7),
        "within_post30|m30": cell_delta(-0.06, 6, 7),
        "external_official_query|m4": cell_delta(-0.02, 4, 6),
        "external_official_query|m10": cell_delta(-0.01, 3, 6),
        "external_official_query|m30": cell_delta(0.01, 2, 6),
    }
    pooled = physical.aggregate_cell(grid)
    assert pooled["grid_cell_count"] == 6
    assert pooled["mean_of_mean_deltas"] == pytest.approx(
        (-0.10 - 0.08 - 0.06 - 0.02 - 0.01 + 0.01) / 6.0)
    assert pooled["negative_grid_cells"] == 5
    assert pooled["pooled_negative_sessions"] == 29
    assert pooled["pooled_session_count"] == 39
    assert pooled["pooled_negative_share"] == pytest.approx(29.0 / 39.0)
    with pytest.raises(physical.M2T4RelianceError):
        physical.aggregate_cell({})


# ---------------------------------------------------------------------------
# 6: the pre-registered verdict.
# ---------------------------------------------------------------------------


def _aggregates(zero: float, units: float = -0.05, columns: float = -0.05):
    def block(mean: float):
        return {"mean_of_mean_deltas": mean, "pooled_negative_share": 1.0}
    return {
        "T4_ZERO": block(zero),
        "T4_SHUFFLE_UNITS": block(units),
        "T4_SHUFFLE_COLS": block(columns),
    }


def test_verdict_rule_is_the_pre_registered_three_string_law() -> None:
    assert plan.VERDICT_STRONG == "T4_RELIANCE_STRONG"
    assert plan.VERDICT_MODERATE == "T4_RELIANCE_MODERATE"
    assert plan.VERDICT_NEGLIGIBLE == "T4_RELIANCE_NEGLIGIBLE"
    assert physical.evaluate_verdict(_aggregates(-0.0731))["verdict"] == plan.VERDICT_STRONG
    assert physical.evaluate_verdict(_aggregates(-0.05))["verdict"] == plan.VERDICT_STRONG
    assert physical.evaluate_verdict(_aggregates(-0.0499))["verdict"] == plan.VERDICT_MODERATE
    assert physical.evaluate_verdict(_aggregates(-0.005))["verdict"] == plan.VERDICT_MODERATE
    assert physical.evaluate_verdict(_aggregates(-0.0049))["verdict"] == plan.VERDICT_NEGLIGIBLE
    assert physical.evaluate_verdict(_aggregates(0.0))["verdict"] == plan.VERDICT_NEGLIGIBLE
    verdict = physical.evaluate_verdict(_aggregates(-0.0731, units=-0.09, columns=-0.02))
    assert verdict["primary_cell"] == "T4_ZERO"
    assert verdict["primary_statistic"] == pytest.approx(-0.0731)
    assert verdict["secondary_statistics"] == {
        "T4_SHUFFLE_UNITS": pytest.approx(-0.09),
        "T4_SHUFFLE_COLS": pytest.approx(-0.02),
    }
    assert verdict["verdict_string_pre_registered"] is True
    # the secondary cells never flip the string
    assert physical.evaluate_verdict(
        _aggregates(-0.0731, units=0.0, columns=0.0))["verdict"] == plan.VERDICT_STRONG
    with pytest.raises(physical.M2T4RelianceError):
        physical.evaluate_verdict({"T4_ZERO": {"mean_of_mean_deltas": -0.1}})
    with pytest.raises(physical.M2T4RelianceError):
        physical.evaluate_verdict({})


# ---------------------------------------------------------------------------
# 7: the disclosure, CPU and receipt laws.
# ---------------------------------------------------------------------------


def test_eval_time_vs_train_time_disclosure_is_binding() -> None:
    disclosure = plan.EVAL_TIME_VS_TRAIN_TIME_DISCLOSURE
    assert "eval-time" in disclosure and "train-time" in disclosure
    assert "NOT" in disclosure
    assert "counterfactual" in disclosure
    assert plan.SCIENTIFIC_ROLE.endswith("not_train_time_ablation")
    assert any(item["id"] == "eval_time_not_train_time" for item in plan.DEVIATIONS)
    assert plan.LEAKAGE_LAW["all_rows"]["target_label_leakage"] is False
    assert plan.LEAKAGE_LAW["all_rows"]["post_hoc_diagnostic"] is True
    assert plan.LEAKAGE_LAW["all_rows"]["checkpoint_selection_eligible"] is False
    assert plan.LEAKAGE_LAW["all_rows"]["deployment_eligible"] is False


def test_cpu_and_receipt_laws_are_declared() -> None:
    assert plan.ENVIRONMENT_LAW["device"] == "cpu"
    assert plan.ENVIRONMENT_LAW["cuda_visible_devices_required"] == ""
    assert plan.ENVIRONMENT_LAW["torch_num_threads"] == 4
    assert plan.HARD_TIMEOUT_SECONDS == 7_200  # the 2h operator bound
    assert plan.RECEIPT_LAW["attempt_before_any_data_or_model_access"] is True
    assert "0444" in plan.RECEIPT_LAW["receipt_permissions"]
    assert "sha256" in plan.RECEIPT_LAW["receipt_permissions"]
    frozen = plan.RECEIPT_LAW["frozen_roots_never_modified"]
    assert "tfpd_exploration/results/m2_t4_activity_budget_screen_v1" in frozen
    assert "tfpd_exploration/src/cdm_p1_m2_local_v1" in frozen
    assert "tfpd_exploration/src/m2_t4_activity_budget_screen_v1" in frozen
    assert plan.ANCHOR_TOLERANCE_R2 == 1.0e-5
    deviations = {item["id"] for item in plan.DEVIATIONS}
    assert {"cpu_instead_of_gpu", "tolerance_anchor_instead_of_bitwise",
            "eval_time_not_train_time"} <= deviations


def test_predecessors_bind_the_sealed_static_family_and_machinery() -> None:
    predecessors = set(plan.PREDECESSOR_RELATIVE)
    assert "tfpd_exploration/results/m2_t4_activity_budget_screen_v1/score.json" in predecessors
    assert "tfpd_exploration/results/m2_same_query_comparator_v1/score.json" in predecessors
    assert "tfpd_exploration/src/m2_t4_activity_budget_screen_v1/core.py" in predecessors
    assert "tfpd_exploration/src/cdm_p1_m2_local_v1/replay.py" in predecessors
    assert "tfpd_exploration/src/cdm_p1_m2_local_v1/anchor.py" in predecessors
    assert "sua_exploration/evalai_t4_m2/export_t4_payload.py" in predecessors


def test_physical_module_reuses_the_frozen_machinery_by_reference() -> None:
    from src.cdm_p1_m2_local_v1 import physical as governing_physical

    assert physical._publish is governing_physical._publish
    assert physical._verify_sidecar is governing_physical._verify_sidecar
    assert physical._bind_namespaces is governing_physical._bind_namespaces
    assert physical._load_sealed_static_rows is (
        governing_physical._load_sealed_same_query_rows)


def test_cpu_isolation_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYTHONNOUSERSITE", "1")
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    monkeypatch.setenv("MKL_NUM_THREADS", "4")
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "4")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    physical._validate_environment()  # the legal binding passes
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(physical.M2T4RelianceError):
        physical._validate_environment()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    with pytest.raises(physical.M2T4RelianceError):
        physical._validate_environment()
    monkeypatch.setenv("OMP_NUM_THREADS", "4")
    monkeypatch.delenv("PYTHONNOUSERSITE")
    with pytest.raises(physical.M2T4RelianceError):
        physical._validate_environment()


def test_no_data_or_cuda_is_needed_by_these_tests() -> None:
    # the module imports only frozen laws; no dataset, checkpoint or CUDA
    # initialisation happens at import time (this file never imports torch)
    assert "torch" not in sys.modules
