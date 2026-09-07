"""No-query contracts and audit parity for the isolated RT strong-LS-v2 primitive."""
from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from src.data import afc4_xls_v2 as XLS


ROOT = Path(__file__).resolve().parents[1]
AUDIT_V2_PATH = ROOT.parent / "sua_exploration/results/rt_afc4_ls_null_strength_audit_v2/RT_AFC4_LS_NULL_STRENGTH_SUPPORT_AUDIT_v2.json"
AUDIT_V1_SCRIPT = ROOT / "scripts/audit_rt_afc4_ls_null_strength.py"
SPEC = importlib.util.spec_from_file_location("rt_afc4_ls_null_strength_audit_v1_for_xls_test", AUDIT_V1_SCRIPT)
assert SPEC is not None and SPEC.loader is not None
V1 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = V1
SPEC.loader.exec_module(V1)


def _diverse_reaches(*, shift: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    groups = np.repeat(np.arange(8, dtype=np.int64), 7)
    labels = []
    for reach in range(8):
        angle = shift + 2.0 * np.pi * reach / 8.0
        for block in range(7):
            labels.append((0.8 + 0.03 * block) * np.asarray([
                np.cos(angle + (block - 3) * 0.02), np.sin(angle + (block - 3) * 0.02),
            ]))
    return groups, np.asarray(labels, dtype=np.float64)


def test_synthetic_generator_is_session_namespaced_cross_reach_and_exact_multiset() -> None:
    groups, velocity = _diverse_reaches()
    permutation, selection = XLS.deterministic_random_cross_reach_derangement(
        groups, velocity, session_name="synthetic-rt", seed=42,
    )
    np.testing.assert_array_equal(np.sort(permutation), np.arange(groups.size))
    np.testing.assert_array_equal(np.sort(velocity[permutation], axis=0), np.sort(velocity, axis=0))
    assert np.all(permutation != np.arange(groups.size))
    assert np.all(groups[permutation] != groups)
    assert selection["permutation_diagnostics"]["same_reach_assigned_blocks"] == 0
    assert XLS.permutation_sha256(permutation) == hashlib.sha256(permutation.astype(np.int64).tobytes()).hexdigest()

    alternate, _ = XLS.deterministic_random_cross_reach_derangement(
        groups, velocity, session_name="other-session", seed=42,
    )
    assert not np.array_equal(permutation, alternate)
    assert tuple(inspect.signature(XLS.deterministic_random_cross_reach_derangement).parameters) == (
        "groups", "velocity", "session_name", "seed",
    )


def test_synthetic_generator_fails_closed_when_cross_reach_or_direction_gate_is_impossible() -> None:
    two_reaches = np.repeat(np.arange(2, dtype=np.int64), 4)
    labels = np.tile(np.asarray([[1.0, 0.0]], dtype=np.float64), (two_reaches.size, 1))
    with pytest.raises(XLS.Afc4XlsV2Error, match="at least three accepted reaches"):
        XLS.deterministic_random_cross_reach_derangement(two_reaches, labels, session_name="too-few", seed=42)

    groups = np.repeat(np.arange(4, dtype=np.int64), 6)
    labels = np.tile(np.asarray([[1.0, 0.0]], dtype=np.float64), (groups.size, 1))
    with pytest.raises(XLS.Afc4XlsV2Error, match="no session-namespaced random cross-reach"):
        XLS.deterministic_random_cross_reach_derangement(groups, labels, session_name="collinear", seed=42)


def test_audit_receipt_parity_helpers_reject_wrong_session_or_sha() -> None:
    receipt = json.loads(AUDIT_V2_PATH.read_text(encoding="utf-8"))
    row = receipt["fold_rows"][0]
    expected = XLS.audited_session_expected_sha256(receipt, session_name=row["session_name"])
    assert expected == row["v2_random_cross_reach_null"]["permutation_sha256"]
    with pytest.raises(XLS.Afc4XlsV2Error, match="exactly one row"):
        XLS.audited_session_expected_sha256(receipt, session_name="not-a-real-session")
    with pytest.raises(XLS.Afc4XlsV2Error, match="permutation SHA mismatch"):
        XLS.assert_audited_session_parity(np.arange(3, dtype=np.int64), audit_receipt=receipt,
                                          session_name=row["session_name"])


def test_real_rt_support_prefixes_reproduce_all_immutable_v2_permutation_shas_without_query() -> None:
    """Open each audited M24 support prefix only; never materialize a query."""

    receipt = json.loads(AUDIT_V2_PATH.read_text(encoding="utf-8"))
    rows = receipt["fold_rows"]
    assert len(rows) == 15
    unavailable = [row["nwb_path"] for row in rows if not Path(row["nwb_path"]).is_file()]
    if unavailable:
        pytest.skip(f"local RT support source is unavailable: {unavailable[0]}")
    actual = {}
    for row in rows:
        raw = V1.load_rt_m24_support(Path(row["nwb_path"]))
        assert raw["session_name"] == row["session_name"]
        assert raw["support_trial_index_range"] == [0, 24]
        assert "query" not in raw
        _rates, velocity, groups = V1.collect_m24_blocks(raw)
        permutation, _selection = XLS.deterministic_random_cross_reach_derangement(
            groups, velocity, session_name=raw["session_name"], seed=42,
        )
        actual[raw["session_name"]] = XLS.assert_audited_session_parity(
            permutation, audit_receipt=receipt, session_name=raw["session_name"],
        )
    assert actual == {
        row["session_name"]: row["v2_random_cross_reach_null"]["permutation_sha256"] for row in rows
    }


def test_primitive_is_not_an_active_loader_or_decoder_integration() -> None:
    source = (ROOT / "src/data/afc4_xls_v2.py").read_text(encoding="utf-8")
    for forbidden in ("pynwb", "torch", "falcon_k4_features", "rt_datamodule", "DataModule", "load_rt_session"):
        assert forbidden not in source
    names = XLS.deterministic_random_cross_reach_derangement.__code__.co_names
    assert "fit_descriptor" not in names and "lstsq" not in names and "neural" not in names
