from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "streaming_calibration_exp/scripts/audit_h1_population_decoder_carrier_m2_date_lodo.py"
SPEC = importlib.util.spec_from_file_location("h1_population_decoder_carrier", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
carrier = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = carrier
SPEC.loader.exec_module(carrier)


def _record(date: str, serial: int) -> carrier.FullRecord:
    """Four trial blocks with an exact population y<-X relation."""

    rng = np.random.default_rng(1000 + serial)
    trials = []
    mixing = rng.normal(scale=0.3, size=(7, 176))
    for number in range(1, 5):
        phase = np.linspace(0.0, 2.0 * np.pi, 30, endpoint=False) + number * 0.17
        velocity = np.column_stack([np.sin(phase * (dimension + 1)) for dimension in range(7)])
        rates = 20.0 + velocity @ mixing
        trials.append(
            carrier.h1.TrialBlocks(
                trial_number=float(number),
                rates=rates,
                velocity=velocity,
                audit={"synthetic": True},
            )
        )
    return carrier.FullRecord(
        session_name=f"ses-{date}T{serial:06d}",
        date=date,
        path=None,
        input_sha256=f"synthetic-{date}-{serial}",
        trials=tuple(trials),
    )


def test_population_decoder_plan_excludes_outer_date_and_scores_only_later_trials() -> None:
    records = {_record(date, index).session_name: _record(date, index) for index, date in enumerate(carrier.h1.H1_DATES)}
    outer = carrier.h1.H1_DATES[0]
    plan = carrier._build_plan(records, outer)

    assert all(records[name].date != outer for name in plan.source_sessions)
    expected_source_support = np.concatenate(
        [carrier._source_support_rates(record) for record in records.values() if record.date != outer], axis=0
    )
    np.testing.assert_allclose(plan.mean, expected_source_support.mean(axis=0))

    output = carrier.run_audit(records)
    assert output["design"]["target_query"].startswith("only all chronological trials after")
    outer_row = next(row for row in output["date_lodo"] if row["date"] == outer)
    assert outer_row["status"] == "defined"
    assert outer_row["records"][0]["support_trial_numbers"] == [1.0, 2.0]
    assert outer_row["records"][0]["query_trial_numbers"] == [3.0, 4.0]
    assert len(outer_row["records"][0]["rotation_null_replicate_r2"]) == carrier.NULL_REPLICATES
