"""Audit and fail-closed guard tests."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from carrier_quant.audit import (
  AUDIT_SCHEMA,
  AuditRecording,
  FOLD0_DATE_FORBIDDEN,
  assert_source_only_date,
  run_audit,
)

from carrier_quant.tests.synthetic_helpers import make_synthetic_calibration, make_synthetic_plan


def test_fold0_date_guard() -> None:
  with pytest.raises(ValueError, match=FOLD0_DATE_FORBIDDEN):
    assert_source_only_date(FOLD0_DATE_FORBIDDEN)


def test_run_audit_synthetic(tmp_path: Path) -> None:
  plan = make_synthetic_plan(seed=30)
  cal = make_synthetic_calibration(plan, seed=31, n_blocks=80)
  rec = AuditRecording(
    session_name="ses-19250108T110520",
    date="19250108",
    rates=cal.rates,
    labels=cal.labels,
    input_sha256="testhash",
  )
  out_path = tmp_path / "audit.json"
  receipt = run_audit([rec], cal.plan, output_path=out_path)

  assert receipt["schema"] == AUDIT_SCHEMA
  assert receipt["source_only"] is True
  assert receipt["fold0_date_forbidden"] == FOLD0_DATE_FORBIDDEN
  assert "quantization_viable_gate" in receipt["recordings"][0]
  assert "merge_order_sensitivity" in receipt["recordings"][0]
  assert out_path.exists()
  loaded = json.loads(out_path.read_text(encoding="utf-8"))
  assert loaded["schema"] == AUDIT_SCHEMA


def test_audit_rejects_fold0_recording() -> None:
  plan = make_synthetic_plan(seed=40)
  cal = make_synthetic_calibration(plan, seed=41, n_blocks=40)
  rec = AuditRecording(
    session_name="ses-19250101T111740",
    date=FOLD0_DATE_FORBIDDEN,
    rates=cal.rates,
    labels=cal.labels,
  )
  with pytest.raises(ValueError, match=FOLD0_DATE_FORBIDDEN):
    run_audit([rec], cal.plan)
