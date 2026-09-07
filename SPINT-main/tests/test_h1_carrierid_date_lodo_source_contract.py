"""No-data contract tests for H1 CarrierID confirmatory date-LODO Phase 1."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import src.data.h1_carrierid_date_lodo_source as source
from scripts.h1_carrierid_date_lodo_source_plan import (
    FOLD0_PREFLIGHT_SCHEMA,
    PREFLIGHT_SCHEMA,
    PREFLIGHT_STATUS,
    build_plan,
    validate_phase1_preflight,
)
from src.data.h1_m4_eb_pilot import H1_HELDIN_SESSIONS, session_date
from src.h1_m4_cce_contract import CONFIRMATORY_DATES, sha256_file


def _immutable_json(path: Path, body: dict) -> str:
    path.write_text(json.dumps(body, sort_keys=True), encoding="utf-8")
    path.chmod(0o444)
    return sha256_file(path)


def _valid_source_manifest(date: str) -> dict:
    sources = list(source.source_sessions_for_date(date))
    targets = list(source.target_sessions_for_date(date))
    binding = {
        "outer_date": date,
        "source_cache_sha256": "a" * 64,
        "normalizer_sha256": "b" * 64,
        "normalized_cache_sha256": "c" * 64,
        "calibration_schedule_sha256": "d" * 64,
        "batch_order_sha256": "e" * 64,
        "source_window_indices_sha256": "f" * 64,
    }
    return {
        "schema": source.SOURCE_MANIFEST_SCHEMA,
        "status": "PASS_SOURCE_ONLY_SHARED_BUNDLE_NOT_LAUNCHED",
        "outer_date": date,
        "source_sessions": sources,
        "source_session_count": len(sources),
        "target_filename_index": {
            "outer_date": date,
            "source_recordings_opened": len(sources),
            "source_sessions_opened": sources,
            "target_recordings_opened": 0,
            "target_sessions_indexed": targets,
            "target_filenames_indexed_only": [f"fake_{name}.nwb" for name in targets],
            "target_bytes_read": 0,
        },
        "arms": {
            "H-S": {"training_wrapper_status": "PHASE2_NOT_IMPLEMENTED", "shared_source_binding": dict(binding)},
            "H-C": {"training_wrapper_status": "PHASE2_NOT_IMPLEMENTED", "shared_source_binding": dict(binding)},
        },
        "source_only_scope": {
            "target_recordings_opened": 0,
            "target_bytes_read": 0,
            "minival_opened_or_enumerated": False,
            "formal_heldout_opened_or_enumerated": False,
            "evalai_opened_or_enumerated": False,
            "cuda_constructed_or_launched": False,
            "trainer_constructed_or_launched": False,
            "checkpoint_created_or_loaded": False,
            "cce_residual_model_imported_or_constructed": False,
        },
    }


def _valid_preflight(tmp_path: Path) -> dict:
    rows = {}
    for date in CONFIRMATORY_DATES:
        manifest = _valid_source_manifest(date)
        manifest_path = tmp_path / f"{date}.json"
        manifest_sha = _immutable_json(manifest_path, manifest)
        rows[date] = {
            "outer_date": date,
            "source_session_count": manifest["source_session_count"],
            "source_sessions": manifest["source_sessions"],
            "target_filenames_indexed_only": manifest["target_filename_index"]["target_filenames_indexed_only"],
            "source_manifest_path": str(manifest_path),
            "source_manifest_sha256": manifest_sha,
            "shared_source_binding": manifest["arms"]["H-S"]["shared_source_binding"],
            "h_s_h_c_share_binding": True,
            "target_recordings_opened": 0,
            "target_bytes_read": 0,
        }
    return {
        "schema": PREFLIGHT_SCHEMA,
        "status": PREFLIGHT_STATUS,
        "confirmatory_dates": list(CONFIRMATORY_DATES),
        "date_bundles": rows,
        "phase_boundaries": {"phase2_training_wrappers_status": "NOT_IMPLEMENTED", "gpu_launch_authorized": False},
        "code_sha256": {
            "scripts/h1_carrierid_date_lodo_source_preflight.py": "0" * 64,
            "src/data/h1_carrierid_date_lodo_source.py": "1" * 64,
            "src/data/h1_m4_cce_date_lodo.py": "2" * 64,
            "src/data/h1_m4_eb_pilot.py": "3" * 64,
            "src/h1_m4_cce_contract.py": "4" * 64,
        },
        "source_only_scope": {
            "target_recordings_opened_total": 0,
            "target_bytes_read_total": 0,
            "minival_opened_or_enumerated": False,
            "formal_heldout_opened_or_enumerated": False,
            "evalai_opened_or_enumerated": False,
            "cuda_constructed_or_launched": False,
            "trainer_constructed_or_launched": False,
            "checkpoint_created_or_loaded": False,
            "cce_residual_model_imported_or_constructed": False,
        },
    }


def test_all_five_date_partitions_are_exact_disjoint_and_have_expected_source_counts() -> None:
    expected_counts = {"19250108": 10, "19250113": 11, "19250115": 11, "19250119": 11, "19250120": 11}
    all_sessions = set(H1_HELDIN_SESSIONS)
    for date in CONFIRMATORY_DATES:
        sources = source.source_sessions_for_date(date)
        targets = source.target_sessions_for_date(date)
        assert len(sources) == expected_counts[date]
        assert not set(sources).intersection(targets)
        assert set(sources).union(targets) == all_sessions
        assert all(session_date(name) != date for name in sources)
        assert all(session_date(name) == date for name in targets)
    with pytest.raises(Exception, match="confirmatory date"):
        source.load_source_records_with_target_filename_index(Path("/tmp/000954"), "19250101")


def test_loader_indexes_target_names_but_never_passes_target_paths_to_record_loader(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    date = "19250108"
    paths = {name: tmp_path / f"public_{name}.nwb" for name in H1_HELDIN_SESSIONS}
    monkeypatch.setattr(source, "index_heldin_calib", lambda _root: paths)
    opened: list[str] = []

    def fake_loader(path: Path) -> SimpleNamespace:
        session = next(name for name, candidate in paths.items() if candidate == path)
        opened.append(session)
        return SimpleNamespace(date=session_date(session))

    records, audit = source.load_source_records_with_target_filename_index(tmp_path / "000954", date, record_loader=fake_loader)
    assert tuple(records) == source.source_sessions_for_date(date)
    assert tuple(opened) == source.source_sessions_for_date(date)
    assert not set(opened).intersection(source.target_sessions_for_date(date))
    assert audit.manifest()["target_recordings_opened"] == 0
    assert audit.manifest()["target_bytes_read"] == 0
    assert audit.target_filenames == tuple(paths[name].name for name in source.target_sessions_for_date(date))


def test_schedule_draws_actual_legal_support_start_values_not_index_positions() -> None:
    cache = SimpleNamespace(source_sessions=("ses-19250108T110520",), starts_by_session={"ses-19250108T110520": (5, 9)})
    windows = source.SourceWindowIndex(
        cache=cache,
        window_indices=(("ses-19250108T110520", 0),) * 32,
        window_indices_sha256="a" * 64,
    )
    schedule = source.SourceSchedule(windows, "19250108")
    assert schedule.schedule.shape[0] == 50
    assert set(schedule.schedule.ravel()).issubset({5, 9})
    assert {5, 9}.intersection(set(schedule.schedule.ravel()))


def test_source_manifest_requires_identical_hs_hc_binding_and_no_target_access() -> None:
    manifest = _valid_source_manifest("19250113")
    source.validate_source_bundle_manifest(manifest, outer_date="19250113")
    manifest["arms"]["H-C"]["shared_source_binding"] = {"outer_date": "19250113"}
    with pytest.raises(Exception, match="do not share"):
        source.validate_source_bundle_manifest(manifest, outer_date="19250113")


def test_phase2_plan_rejects_fold0_and_declares_ten_not_launched_cells(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="fold-0"):
        validate_phase1_preflight({"schema": FOLD0_PREFLIGHT_SCHEMA})
    receipt = _valid_preflight(tmp_path)
    preflight_path = tmp_path / "preflight.json"
    _immutable_json(preflight_path, receipt)
    plan = build_plan(receipt, preflight_path=preflight_path)
    assert plan["cell_count"] == 10
    assert all(cell["launch_authorized"] is False for cell in plan["phase2_cells"])
    assert all(cell["training_wrapper_status"] == "NOT_IMPLEMENTED" for cell in plan["phase2_cells"])
    assert all(cell["outer_date"] != "19250101" for cell in plan["phase2_cells"])


def test_new_phase1_sources_do_not_import_target_loader_or_model_paths() -> None:
    root = Path(__file__).resolve().parents[1]
    data_text = (root / "src/data/h1_carrierid_date_lodo_source.py").read_text(encoding="utf-8")
    preflight_text = (root / "scripts/h1_carrierid_date_lodo_source_preflight.py").read_text(encoding="utf-8")
    plan_text = (root / "scripts/h1_carrierid_date_lodo_source_plan.py").read_text(encoding="utf-8")
    combined = "\n".join((data_text, preflight_text, plan_text))
    assert "load_target_records_for_date" not in combined
    assert "src.models." not in combined
    assert "torch.cuda" not in combined
