"""Pure contracts for the additive H1 exposure-matched D-S4e/D-Q4e path."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from src.data import h1_carrierid_distribution as base
from src.data import h1_carrierid_distribution_exposure as exposure
from src.data.h1_m4_eb_pilot import H1_M4_FOLD0_SOURCE
from scripts import h1_carrierid_distribution_exposure_preflight as preflight


def _schedules() -> tuple[base.EightTrialSchedule, ...]:
    # Seven schedules for the first six source sessions and six for the rest:
    # 6*7 + 5*6 = 72.  Every canonical source session remains represented.
    values = []
    for session_index, name in enumerate(H1_M4_FOLD0_SOURCE):
        for local in range(7 if session_index < 6 else 6):
            start = session_index * 100 + local * 8
            values.append(base.EightTrialSchedule(
                session_name=name,
                start_index=start,
                support_values=(float(start), float(start + 1), float(start + 2), float(start + 3)),
                query_values=(float(start + 4), float(start + 5), float(start + 6), float(start + 7)),
                query_first_bin=700 + start,
            ))
    return tuple(values)


def test_quotas_preserve_115520_samples_and_session_batch_alignment():
    schedules = _schedules()
    quotas = exposure._balanced_schedule_quotas(schedules)
    assert len(quotas) == 72
    assert sum(quotas) == 115_520
    assert all(value > 32 for value in quotas)
    for name in H1_M4_FOLD0_SOURCE:
        total = sum(value for schedule, value in zip(schedules, quotas) if schedule.session_name == name)
        assert total % 32 == 0


def test_even_cyclic_selector_is_unique_and_full_range_not_fixed32():
    schedule = _schedules()[0]
    candidates = np.arange(10_000, dtype=np.int64)
    selected = exposure._even_cyclic_selection(candidates, count=1_604, schedule=schedule)
    assert len(selected) == len(set(selected)) == 1_604
    assert min(selected) < 20 and max(selected) > 9_900


def test_exposure_source_module_has_no_target_loader_dependency():
    source = (Path(__file__).resolve().parents[1] / "src/data/h1_carrierid_distribution_exposure.py").read_text(
        encoding="utf-8"
    )
    assert "load_target_records" not in source
    assert "StrictTargetDataset" not in source
    assert "EXPOSURE_SAMPLES_PER_EPOCH = 115_520" in source
    assert "EXPOSURE_BATCHES_PER_EPOCH = 3_610" in source


def test_terminal_gates_are_fixed_before_target_evaluation():
    gates = preflight.frozen_terminal_branch_gates()
    assert gates["validity"]["d_s4e_pooled_r2_min"] == 0.4955107931
    assert gates["only_after_validity_pass"]["estimator"]["pooled_q4e_minus_s4e_min"] == 0.03
    assert gates["only_after_validity_pass"]["consumer"]["no_session_sign_reversal_definition"] == (
        "delta_session_0 * delta_session_1 >= 0"
    )
