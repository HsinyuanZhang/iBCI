from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "streaming_calibration_exp/scripts/audit_h1_m4_population_decoder_carrier_date_lodo.py"
SPEC = importlib.util.spec_from_file_location("h1_m4_population_decoder_carrier", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)


def _record(date: str, serial: int) -> audit.m2.FullRecord:
    rng = np.random.default_rng(7000 + serial)
    mixing = rng.normal(scale=0.25, size=(7, 176))
    trials = []
    for number in range(1, 8):
        phase = np.linspace(0.0, 2.0 * np.pi, 36, endpoint=False) + number * 0.13
        velocity = np.column_stack([np.sin((axis + 1) * phase) for axis in range(7)])
        trials.append(
            audit.m2.h1.TrialBlocks(
                trial_number=float(number),
                rates=30.0 + velocity @ mixing,
                velocity=velocity,
                audit={"synthetic": True},
            )
        )
    return audit.m2.FullRecord(f"ses-{date}T{serial:06d}", date, None, f"synthetic-{date}", tuple(trials))


def test_m4_plan_excludes_outer_and_only_queries_trials_five_onward() -> None:
    records = {}
    for serial, date in enumerate(audit.m2.h1.H1_DATES):
        record = _record(date, serial)
        records[record.session_name] = record
    outer = audit.m2.h1.H1_DATES[0]
    plan = audit._build_plan(records, outer)
    assert all(records[name].date != outer for name in plan.source_sessions)
    expected = np.concatenate([audit._support_rates(record) for record in records.values() if record.date != outer], axis=0)
    np.testing.assert_allclose(plan.mean, expected.mean(axis=0))

    output = audit.run_audit(records)
    date_row = next(row for row in output["date_lodo"] if row["date"] == outer)
    assert date_row["status"] == "defined"
    result = date_row["records"][0]
    assert result["support_trial_numbers"] == [1.0, 2.0, 3.0, 4.0]
    assert result["query_trial_numbers"] == [5.0, 6.0, 7.0]
    assert result["support_exposure_seconds"] == 14.4
    assert len(result["rotation_null_r2"]) == audit.NULL_REPLICATES
