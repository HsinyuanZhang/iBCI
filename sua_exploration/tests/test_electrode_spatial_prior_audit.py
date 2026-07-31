from __future__ import annotations

import numpy as np
import pytest

from scripts.audit_electrode_spatial_prior import (
    adjacency_count_distribution,
    neighbor_target_ac,
    parse_cmp,
    permute_electrode_coordinates,
    shrink_ac,
    summarize_log_mse_contrast,
)


def test_cmp_parser_and_adjacency_reject_identity_lookup_ambiguity(tmp_path):
    path = tmp_path / "map.cmp"
    path.write_text(
        "\n".join(
            [
                "// c r b e l",
                "fixture map",
                "0 0 A 1 e1",
                "1 0 A 2 e2",
                "0 1 B 1 e3",
                "1 1 B 2 e4",
            ]
        ),
        encoding="utf-8",
    )
    observed = parse_cmp(path)
    assert observed == {
        ("A", 1): (0, 0),
        ("A", 2): (1, 0),
        ("B", 1): (0, 1),
        ("B", 2): (1, 1),
    }
    assert adjacency_count_distribution(dict(enumerate(observed.values()))) == {
        3: 4
    }


def test_cmp_parser_rejects_duplicate_coordinate(tmp_path):
    path = tmp_path / "bad.cmp"
    path.write_text(
        "fixture map\n0 0 A 1 e1\n0 0 A 2 e2\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate coordinate"):
        parse_cmp(path)


def test_neighbor_target_is_electrode_weighted_excludes_self_and_zero_falls_back():
    # Electrode 0 has two units. It must contribute its one electrode mean
    # [2,0], not receive twice the weight when electrode 1 borrows from it.
    ac = np.asarray(
        [
            [1.0, 0.0],
            [3.0, 0.0],
            [4.0, 0.0],
            [9.0, 0.0],
        ]
    )
    electrode_ids = np.asarray([0, 0, 1, 2])
    coordinates = {0: (0, 0), 1: (1, 0), 2: (3, 0)}
    target, counts = neighbor_target_ac(ac, electrode_ids, coordinates)
    assert np.allclose(
        target,
        np.asarray(
            [
                [4.0, 0.0],
                [4.0, 0.0],
                [2.0, 0.0],
                [0.0, 0.0],
            ]
        ),
    )
    assert np.array_equal(counts, np.asarray([1, 1, 1, 0]))

    factor = np.asarray([0.5, 0.5, 0.5, 0.5])
    observed = shrink_ac(ac, factor, target)
    # The no-neighbor unit is exactly the shrink-to-zero arm.
    assert observed[-1, 0] == pytest.approx(4.5)


def test_coordinate_shuffle_is_deterministic_and_preserves_coordinate_set():
    coordinates = {index: (index % 3, index // 3) for index in range(9)}
    first = permute_electrode_coordinates(coordinates, seed=42)
    second = permute_electrode_coordinates(coordinates, seed=42)
    different = permute_electrode_coordinates(coordinates, seed=43)
    assert first == second
    assert first != different
    assert set(first) == set(coordinates)
    assert set(first.values()) == set(coordinates.values())


def test_log_mse_summary_requires_material_consistent_session_gain():
    rows = {}
    for index in range(27):
        baseline = np.zeros(4)
        rows[f"s{index:02d}"] = {
            "log_mse": {
                "neighbor": np.full(4, np.log(0.95)),
                "zero": baseline,
            }
        }
    summary = summarize_log_mse_contrast(
        rows,
        treatment="neighbor",
        control="zero",
    )
    assert summary["geometric_session_mse_ratio"] == pytest.approx(0.95)
    assert summary["sessions_improved"] == 27
    assert summary["passes_material_train_only_gate"] is True

