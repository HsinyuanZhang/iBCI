"""Synthetic no-data contracts for the W3 alignment scorer (CPU-only)."""
from __future__ import annotations

from pathlib import Path
import textwrap

import numpy as np
import pytest

from src.data import h1_carrier_alignment_score as scorer


def _recording(rng: np.random.Generator, n_channels: int = 176, spread: float = 1.0) -> np.ndarray:
    return rng.normal(scale=spread, size=(n_channels, scorer.CARRIER_WIDTH)).astype(np.float64)


def test_known_separability_value_is_returned_exactly():
    # Columns with known population variances: separability = sum of column variances.
    carrier = np.array(
        [
            [0.0, 10.0, 0.0, 0.0],
            [2.0, 10.0, 0.0, 0.0],
            [4.0, 10.0, 0.0, 0.0],
            [6.0, 10.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    expected = float(np.var([0.0, 2.0, 4.0, 6.0]) + np.var([10.0, 10.0, 10.0, 10.0]))
    assert scorer.separability(carrier) == pytest.approx(expected, rel=0.0, abs=1.0e-12)
    # Column 0 is the only varying column: variance of an arithmetic sequence is 5.0.
    assert scorer.separability(carrier) == pytest.approx(5.0, rel=0.0, abs=1.0e-12)


def test_zero_drift_raises_and_does_not_divide_by_zero():
    # Every source recording shares the same carrier distribution -> zero drift.
    identical = _recording(np.random.default_rng(7), n_channels=12)
    carriers = {name: identical.copy() for name in scorer.SOURCE_RECORDINGS[:3]}
    with pytest.raises(scorer.DriftUndefinedError, match="drift is zero/undefined"):
        scorer.score(carriers)
    # drift() itself returns exactly zero here, not inf/nan.
    assert scorer.drift(carriers) == pytest.approx(0.0, abs=1.0e-12)


def test_non_source_recording_name_is_rejected():
    rng = np.random.default_rng(11)
    carriers = {name: _recording(rng) for name in scorer.SOURCE_RECORDINGS[:3]}
    carriers["ses-NOT_A_SOURCE_RECORDING"] = _recording(rng)
    with pytest.raises(scorer.CarrierAlignmentScoreError, match="non-source recording"):
        scorer.score(carriers)


def test_functions_are_deterministic_across_two_calls():
    rng = np.random.default_rng(123)
    carriers = {name: _recording(rng) for name in scorer.SOURCE_RECORDINGS}
    first = scorer.score(carriers)
    second = scorer.score(carriers)
    assert first == second
    # separability / drift are pure: same array in -> same scalar out.
    assert scorer.separability(carriers[scorer.SOURCE_RECORDINGS[0]]) == scorer.separability(
        carriers[scorer.SOURCE_RECORDINGS[0]]
    )
    assert scorer.drift(carriers) == scorer.drift(carriers)


def test_module_imports_no_live_producer_from_section_1_3():
    # The scorer module must be isolated: it imports none of the frozen live
    # producer modules.  Inspect the scorer's own import statements (not global
    # sys.modules, which other tests in the same session may pollute).
    import ast

    producer_modules = {
        "src.data.h1_m4_eb_pilot",
        "src.data.h1_carrierid_date_lodo_source",
        "src.data.h1_carrierid_date_lodo_ci",
        "src.data.h1_carrierid_date_lodo_ci_target",
        "src.data.h1_m4_cce_date_lodo",
        "streaming_calibration_exp",
    }
    tree = ast.parse(Path(scorer.__file__).read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imported.add(node.module)
    assert scorer.__file__ is not None
    leaked = producer_modules & imported
    assert leaked == set(), f"scorer imports live producer modules: {leaked}"


def test_score_value_and_receipt_roundtrip(tmp_path: Path):
    rng = np.random.default_rng(2026)
    base = {name: _recording(rng, spread=2.0) for name in scorer.SOURCE_RECORDINGS}
    # Add a small per-recording mean shift so drift is nonzero and well-defined.
    for offset, name in enumerate(scorer.SOURCE_RECORDINGS):
        base[name] = base[name] + 0.05 * offset
    result = scorer.score(base)
    # Score equals mean separability divided by drift exactly.
    assert result["score"] == pytest.approx(
        result["mean_separability"] / result["drift"], rel=0.0, abs=1.0e-12
    )
    assert result["recording_names"] == list(scorer.SOURCE_RECORDINGS)
    assert result["uses_target_data"] is False and result["uses_labels"] is False
    receipt = scorer.write_receipt(tmp_path / "w3_score.json", result)
    assert Path(receipt["receipt_path"]).is_file()
    # Refuses to overwrite.
    with pytest.raises(FileExistsError):
        scorer.write_receipt(tmp_path / "w3_score.json", result)


def test_frozen_definition_and_version_are_present_and_stable():
    assert scorer.VERSION == "h1_carrier_alignment_score_v1"
    assert scorer.MODULE_STATUS == "IMPLEMENTED_NOT_RUN_TARGET_GATE_CLOSED"
    assert "separability(C_r)" in scorer.SCORER_DEFINITION
    assert "drift({C_r})" in scorer.SCORER_DEFINITION
    assert textwrap.dedent(scorer.SCORER_DEFINITION).startswith("W3 alignment score")
    # The 11 source recordings are exactly the fold-0 source set.
    assert len(scorer.SOURCE_RECORDINGS) == 11
    assert len(set(scorer.SOURCE_RECORDINGS)) == 11
