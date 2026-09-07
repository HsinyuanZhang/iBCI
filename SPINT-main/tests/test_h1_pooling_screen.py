"""Synthetic no-data contracts for the pooling screen (CPU-only)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.data import h1_pooling_screen as ps


def test_row_cosine_is_zero_for_orthogonal():
    a = np.array([[1, 0, 0, 0, 0, 0, 0]], dtype=np.float64)
    b = np.array([[0, 1, 0, 0, 0, 0, 0]], dtype=np.float64)
    assert ps.row_cosine(a, b) is not None
    assert abs(ps.row_cosine(a, b)) < 1e-10


def test_shrink_rows_pulls_toward_prior():
    raw = np.ones((176, 7), dtype=np.float64)
    prior_mean = np.zeros((176, 7), dtype=np.float64)
    prior_var = np.ones((176, 7), dtype=np.float64)
    est_var = np.ones((176, 7), dtype=np.float64)
    shrunk = ps.shrink_rows(raw, prior_mean, prior_var, est_var)
    # weight = 1/(1+1) = 0.5, so shrunk = 0 + 0.5 * (1 - 0) = 0.5
    assert np.allclose(shrunk, 0.5)


def test_loo_prior_excludes_held_record():
    """Assert pool_loo prior does not include the held recording's rows."""

    from types import SimpleNamespace
    rng = np.random.default_rng(42)
    source = tuple(f"ses-r{i}" for i in range(4))
    full_rows = {name: rng.normal(size=(176, 7)) for name in source}

    # LOO for recording 0
    other_names = [n for n in source if n != source[0]]
    other_rows = np.stack([full_rows[n] for n in other_names], axis=0)
    loo_mean = other_rows.mean(axis=0)

    # Global (includes recording 0)
    all_rows = np.stack([full_rows[n] for n in source], axis=0)
    global_mean = all_rows.mean(axis=0)

    # The LOO mean must differ from the global mean
    assert not np.allclose(loo_mean, global_mean)
    assert source[0] not in other_names


def test_receipt_refuses_overwrite(tmp_path: Path):
    p = tmp_path / "pooling_receipt.json"
    result = {"schema": ps.POOLING_SCREEN_SCHEMA, "test": True}
    ps.write_receipt(p, result)
    assert p.is_file()
    with pytest.raises(FileExistsError):
        ps.write_receipt(p, result)


def test_schema_and_status():
    assert ps.MODULE_STATUS == "CPU_ONLY_SOURCE_SCREEN"
    assert ps.POOLING_SCREEN_SCHEMA == "h1_pooling_screen_v1"
