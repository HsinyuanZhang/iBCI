from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scripts import h1_carrierid_same_checkpoint_dose_response as runner
from src.data.h1_carrierid_dose_response import (
    CarrierOverrideTargetDataset,
    DOSE_GRID,
    REPEAT_SEEDS,
    partial_label_overrides,
    partial_row_corruption,
)
from src.data.h1_m4_eb_pilot import TrialBlocks


@dataclass
class _SyntheticRecord:
    session_name: str
    trials: tuple[TrialBlocks, ...]

    def blocks_for(self, trial_number: float) -> TrialBlocks:
        return next(trial for trial in self.trials if trial.trial_number == float(trial_number))


def _record() -> _SyntheticRecord:
    trials = []
    for trial_index in range(4):
        offset = trial_index * 4
        velocity = np.column_stack(
            [np.arange(offset, offset + 4, dtype=np.float64) + 100.0 * dim for dim in range(7)]
        )
        rates = np.arange(12, dtype=np.float64).reshape(4, 3) + trial_index * 1000.0
        trials.append(
            TrialBlocks(
                trial_number=float(trial_index + 1),
                rates=rates,
                velocity=velocity,
                block_indices=np.arange(20, dtype=np.int64).reshape(4, 5) + trial_index * 100,
            )
        )
    return _SyntheticRecord("synthetic-session", tuple(trials))


def _sorted_rows(values: np.ndarray) -> list[tuple[float, ...]]:
    return sorted(tuple(float(item) for item in row) for row in np.asarray(values))


def test_design_grid_and_repeats_are_frozen() -> None:
    assert DOSE_GRID == (0.0, 0.25, 0.5, 0.75, 1.0)
    assert REPEAT_SEEDS == (2026080801, 2026080802, 2026080803, 2026080804)


def test_partial_row_corruption_is_exact_nested_deterministic_derangement() -> None:
    carrier = np.arange(40, dtype=np.float64).reshape(10, 4)
    seed = REPEAT_SEEDS[0]
    observed: dict[float, set[int]] = {}
    for p, expected_count in ((0.0, 0), (0.25, 3), (0.5, 5), (0.75, 8), (1.0, 10)):
        corrupted, audit = partial_row_corruption(
            carrier, session_name="ses-test", p=p, repeat_seed=seed
        )
        changed = set(np.flatnonzero(np.any(corrupted != carrier, axis=1)).tolist())
        assert len(changed) == expected_count
        assert audit.selected_rows == expected_count
        assert audit.all_selected_assignments_deranged
        assert _sorted_rows(corrupted) == _sorted_rows(carrier)
        repeated, repeated_audit = partial_row_corruption(
            carrier, session_name="ses-test", p=p, repeat_seed=seed
        )
        assert np.array_equal(corrupted, repeated)
        assert audit == repeated_audit
        observed[p] = changed
    assert observed[0.0] <= observed[0.25] <= observed[0.5] <= observed[0.75] <= observed[1.0]


def test_partial_label_corruption_changes_only_exact_velocity_rows() -> None:
    record = _record()
    original_rates = [trial.rates.copy() for trial in record.trials]
    original = np.concatenate([trial.velocity for trial in record.trials])
    overrides, audit = partial_label_overrides(
        record, (1.0, 2.0, 3.0, 4.0), p=0.25, repeat_seed=REPEAT_SEEDS[1]
    )
    corrupted = np.concatenate([overrides[value] for value in (1.0, 2.0, 3.0, 4.0)])
    changed = np.flatnonzero(np.any(corrupted != original, axis=1))
    assert len(changed) == 4
    assert audit.total_rows == 16
    assert audit.selected_rows == 4
    assert audit.all_selected_assignments_deranged
    assert _sorted_rows(corrupted) == _sorted_rows(original)
    assert all(np.array_equal(before, trial.rates) for before, trial in zip(original_rates, record.trials))
    repeated, repeated_audit = partial_label_overrides(
        record, (1.0, 2.0, 3.0, 4.0), p=0.25, repeat_seed=REPEAT_SEEDS[1]
    )
    assert all(np.array_equal(overrides[key], repeated[key]) for key in overrides)
    assert audit == repeated_audit


class _BaseDataset:
    def __init__(self) -> None:
        self.records = {"a": object(), "b": object()}
        carrier = np.zeros((3, 4), dtype=np.float32)
        self.support = {
            name: SimpleNamespace(carriers={"full": carrier.copy()}) for name in self.records
        }
        self.plan = object()
        self.window_indices = [("a", 10), ("b", 20)]
        self.window_indices_sha256 = "query-hash"

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int):
        session = self.window_indices[index][0]
        return (
            np.full((2, 3), index, dtype=np.float32),
            np.full((2, 2), index, dtype=np.float32),
            np.full((4, 5, 3), index, dtype=np.float32),
            session,
            np.zeros((3, 4), dtype=np.float32),
        )


def test_overlay_changes_only_carrier_and_preserves_window_object() -> None:
    base = _BaseDataset()
    carriers = {
        "a": np.ones((3, 4), dtype=np.float32),
        "b": np.full((3, 4), 2.0, dtype=np.float32),
    }
    overlay = CarrierOverrideTargetDataset(base, carriers, audit={"frozen": True})
    assert overlay.window_indices is base.window_indices
    assert overlay.window_indices_sha256 == base.window_indices_sha256
    for index in range(2):
        base_item = base[index]
        item = overlay[index]
        assert all(np.array_equal(item[position], base_item[position]) for position in (0, 1, 2))
        assert item[3] == base_item[3]
        assert np.array_equal(item[4], carriers[item[3]])


def _metric(value: float) -> dict:
    return {
        "pooled_r2": value,
        "per_session": {name: {"r2": value, "samples": 1} for name in runner.H1_M4_FOLD0_TARGET},
    }


def test_curve_summary_reports_monotonicity_and_delta_from_p0() -> None:
    shards = {}
    for seed in REPEAT_SEEDS:
        for p in DOSE_GRID[1:]:
            shards[("row", p, seed)] = {"metrics": _metric(0.6 - 0.2 * p)}
    summary = runner._curve_summary(_metric(0.6), shards, kind="row", recording=None)
    assert summary["mean_curve_nonincreasing"] is True
    assert summary["nonincreasing_repeat_count"] == 4
    assert summary["by_p"][-1]["mean_delta_vs_p0"] == -0.2
    shards[("row", 0.75, REPEAT_SEEDS[0])] = {"metrics": _metric(0.59)}
    violated = runner._curve_summary(_metric(0.6), shards, kind="row", recording=None)
    assert violated["per_repeat"][str(REPEAT_SEEDS[0])]["nonincreasing"] is False


def test_target_loader_is_inside_explicit_execution_after_preflight_binding() -> None:
    source = Path(runner.__file__).read_text(encoding="utf-8")
    function = source[source.index("def _prepare_opened_development"):source.index("def _override_dataset")]
    assert function.index("_require_preflight()") < function.index("from src.data.h1_m4_eb_pilot import load_target_records")
    assert function.index("_load_carrierid_checkpoint") < function.index("load_target_records(cfg.data.data_dir)")
    assert 'source.setup("fit")' not in function  # setup is confined to the source-only helper
    assert "model.eval()" in function
    assert "parameter.requires_grad_(False)" in function


def test_nonzero_execution_is_blocked_by_runtime_p0_gate() -> None:
    source = Path(runner.__file__).read_text(encoding="utf-8")
    execute_all = source[source.index("def execute_all_missing"):source.index("def execute_one")]
    assert execute_all.index("_require_runtime_p0_gate(preflight)") < execute_all.index("_prepare_opened_development(device)")
    execute_one = source[source.index("def execute_one"):source.index("def _metric_scalar")]
    assert execute_one.index("_require_runtime_p0_gate(preflight)") < execute_one.index("_prepare_opened_development(device)")
