from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "sua_exploration/scripts/audit_m2_uncertainty_identifiability.py"


def load_module():
    spec = importlib.util.spec_from_file_location("m2_uncertainty_audit", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v2_separates_strict_future_query_from_incomplete_historical_design(tmp_path: Path) -> None:
    module = load_module()
    out = tmp_path / "audit"
    module.run(out)
    aggregate = json.loads((out / "aggregate.json").read_text())
    assert aggregate["evidence_A_strict_future_query"]["design"].startswith("fold1 seeds 42/43/44")
    assert aggregate["evidence_B_historical_context"]["design"] == "f1s42/f1s43/f2s42 incomplete factorial"
    assert "unified M2 noise floor" in aggregate["explicitly_not_claimed"]
    assert "T4-F0 or K4 MDE" in aggregate["explicitly_not_claimed"]
    raw = json.loads((out / "evidence_A_raw_clustered_deltas.json").read_text())
    assert len(raw["rows"]) == 18
    assert {row["seed"] for row in raw["rows"]} == {42, 43, 44}
    assert {row["fold"] for row in raw["rows"]} == {1}
    assert len(raw["per_session_seed_averaged_means"]) == 6
    with (out / "evidence_A_mde_sensitivity.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) >= 42
    assert {int(row["effective_independent_session_clusters"]) for row in rows} == {3, 4, 6, 8, 10, 12}


def test_refuses_to_overwrite_output_directory(tmp_path: Path) -> None:
    module = load_module()
    out = tmp_path / "audit"
    module.run(out)
    try:
        module.run(out)
    except FileExistsError:
        pass
    else:
        raise AssertionError("audit must not overwrite an existing result directory")
