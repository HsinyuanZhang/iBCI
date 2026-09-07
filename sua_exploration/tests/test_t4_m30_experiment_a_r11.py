from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_sha_locked_historical_reference_loads_without_residual_field():
    import aggregate_t4_m30_experiment_a_r11 as r11

    root = Path(__file__).resolve().parents[2]
    receipt = json.loads(r11.RECEIPT.read_text())
    path = root / "sua_exploration/results/sua_spint_t4_mainline_fp32_v1/t4_s42.json"
    values, evidence = r11.load(
        path,
        "t4",
        42,
        new=False,
        status_dir=None,
        train_sha="unused-for-reference",
        ref_hashes=receipt["qualified_reference_artifacts"],
    )
    assert values.shape == (8, 6)
    assert "pre-logit-residual-schema" in evidence["historical_schema_compatibility"]


def test_historical_reference_still_requires_receipt_hash(tmp_path: Path):
    import aggregate_t4_m30_experiment_a_r11 as r11

    root = Path(__file__).resolve().parents[2]
    receipt = json.loads(r11.RECEIPT.read_text())
    original = root / "sua_exploration/results/sua_spint_t4_mainline_fp32_v1/t4_s42.json"
    changed = tmp_path / "t4_s42.json"
    changed.write_bytes(original.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="unqualified reference"):
        r11.load(
            changed,
            "t4",
            42,
            new=False,
            status_dir=None,
            train_sha="unused-for-reference",
            ref_hashes=receipt["qualified_reference_artifacts"],
        )
