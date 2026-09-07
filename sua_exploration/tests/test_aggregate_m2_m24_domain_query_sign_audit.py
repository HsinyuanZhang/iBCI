from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "sua_exploration/scripts/aggregate_m2_m24_domain_query_sign_audit.py"


def module():
    spec = importlib.util.spec_from_file_location("m2_m24_domain_query_aggregate", SCRIPT)
    assert spec and spec.loader
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def test_expected_seal_grid_is_exactly_twelve_scorable_cells() -> None:
    loaded = module()
    assert len(loaded.EXPECTED) == 12
    assert all(not (domain == "heldin" and query == 24) for _, domain, query, _ in loaded.EXPECTED)
    assert {(arm, policy) for arm, domain, query, policy in loaded.EXPECTED if domain == "heldin" and query == 0} == {("f0", "best"), ("f0", "epoch9"), ("t4", "best"), ("t4", "epoch9")}


def test_current_profiles_build_and_verify_the_fail_closed_seal() -> None:
    loaded = module()
    seal = loaded.build_seal_list(loaded.RESULT_ROOT)
    rows = loaded.read_sealed(seal)
    assert len(rows) == 12
    assert all(row["cuda_available_at_preflight"] is False for row in rows.values())
    assert all(audit["full_window_disjoint"] for row in rows.values() if row["cell"]["domain"] == "heldout" and row["cell"]["query_start_trial"] == 24 for audit in row["query_window_audit"].values())


def test_aggregate_marks_q0_as_diagnostic_and_domain_query_as_nonidentifiable(tmp_path: Path) -> None:
    loaded = module()
    seal = loaded.build_seal_list(loaded.RESULT_ROOT)
    seal_path = tmp_path / "seal.json"
    seal_path.write_text(json.dumps(seal), encoding="utf-8")
    report = loaded.aggregate(loaded.read_sealed(seal), seal_path)
    contract = report["interpretation_contract"]
    assert "diagnostic/resubstitution" in contract["heldout_q0"]
    assert "not identifiable" in contract["domain_by_query_identifiability"]
    assert "not a pure session-domain effect" in contract["endpoint_composition_limit"]
    assert report["reports_by_checkpoint_policy"]["best"]["heldout_q24"]["T4_minus_F0"]["equal_session_mean_r2"] > 0
