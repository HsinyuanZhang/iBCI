"""CPU-only synthetic end-to-end checks for the Flat final-score gate and merged report.

The fixtures contain no DANDI material.  They exercise the real capability, score
entry point, receipt writer, and report validator using six fake final sessions.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean, stdev
from types import SimpleNamespace

import numpy as np
import pytest

from dandi688_bench_v2 import protocol
from dandi688_bench_v2.common import digest, sha256
from dandi688_bench_v2.data import SessionData
from dandi688_bench_v2.final_access import FINAL_SCORE_CELLS, REQUIRED_CELLS, SEAL_SCHEMA, SEAL_STATUS
from dandi688_bench_v2.flat_extension import finalize, report
from dandi688_bench_v2.flat_extension.contract import FLAT_RECIPE, flat_training_source_hashes
from dandi688_bench_v2.flat_extension.finalize import FLAT_FINAL_SCHEMA, FLAT_SEAL_SCHEMA, FLAT_SEAL_STATUS, _canonical
from dandi688_bench_v2.flat_extension.run_campaign import SCHEMA as PLAN_SCHEMA, STATUS as PLAN_STATUS, _digest, _tasks


# This is the immutable plan order: six main tasks, then four Full supplements.
CELLS6 = ["full_flat_sua", "activity_flat_sua", "raw_set_flat_sua",
          "full_flat_pmua", "activity_flat_pmua", "raw_set_flat_pmua"]
CELLS10 = CELLS6 + ["full_flat_sua_s43", "full_flat_sua_s44", "full_flat_pmua_s43", "full_flat_pmua_s44"]


def _json(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return path


def _artifact(path: Path, contents: str = "fixture") -> Path:
    path.write_text(contents, encoding="utf-8")
    return path.resolve()


def _metric(roster: tuple[str, ...], offset: float, *, records=None, representation=None) -> dict:
    sessions = [{"session_id": sid, "n_queries": 3, "r2": offset + index / 100,
                 **({"query_indices_sha256": records[sid][representation].metadata["array_sha256"]["query_indices"],
                     "velocity_sha256": records[sid][representation].metadata["array_sha256"]["velocity"]}
                    if records is not None else {})}
                for index, sid in enumerate(roster)]
    return {"metric": "synthetic", "n_sessions": 6,
            "mean_r2": float(np.mean([row["r2"] for row in sessions])), "sessions": sessions}


def _record(sid: str, rep: str, *, canonical: tuple[str, ...] = ("e0", "e1"), query_seed: int = 0,
            velocity_seed: int = 0) -> SessionData:
    query = np.array([0, 1, 2], dtype=np.int64) + query_seed
    velocity = np.array([[1., 0.], [0., 1.], [1., 1.], [2., 2.]], dtype=np.float64) + velocity_seed
    hashes = {"query_indices": digest(query), "velocity": digest(velocity)}
    return SessionData(sid, "final", rep, np.zeros((4, 1)), velocity, np.arange(5.), query,
                       np.array([0]), np.array([0]), np.zeros((1, 1)), np.zeros((1, 1)), np.zeros(1),
                       np.array([0]), np.array([0]), np.array([True]),
                       {"canonical_electrode_keys": list(canonical), "array_sha256": hashes, "raw_nwb_sha256": "r"})


def _base_receipt(tmp: Path, roster: tuple[str, ...], records: dict[str, dict[str, SessionData]]) -> Path:
    seal = tmp / "base.seal.json"
    base_seal = {"schema": SEAL_SCHEMA, "status": SEAL_STATUS,
                 "cell_selections": {c: {} for c in REQUIRED_CELLS},
                 "supplemental_full_selections": {c: {} for c in FINAL_SCORE_CELLS if c not in REQUIRED_CELLS}}
    _json(seal, base_seal)
    rows = {}
    for index, cell in enumerate(FINAL_SCORE_CELLS):
        rep = "pmua" if "pmua" in cell else "sua"
        preds = {sid: np.full((3, 2), index + 1., dtype=np.float64) for sid in roster}
        path = tmp / f"base.{cell}.npz"; np.savez_compressed(path, **preds)
        rows[cell] = {"status": "SCORED", "result": {"metrics": _metric(roster, .1 + index / 1000)},
                      "predictions": {"path": str(path), "sha256": sha256(path), "sessions": {
                          sid: {"prediction_sha256": digest(preds[sid]),
                                "query_indices_sha256": records[sid][rep].metadata["array_sha256"]["query_indices"],
                                "velocity_sha256": records[sid][rep].metadata["array_sha256"]["velocity"]}
                          for sid in roster}}}
    return _json(tmp / "base.final.json", {"schema": "dandi688_v2_final_score", "status": "FINAL_SCORED",
                                             "seal_path": str(seal), "seal_sha256": sha256(seal), "roster": list(roster),
                                             "results": rows, "final_sessions_opened": 6})


def _campaign(tmp: Path, cells: list[str] = CELLS10):
    """Build a fully self-hashed synthetic dependency graph accepted by FlatFinalAccess."""
    tmp.mkdir(parents=True, exist_ok=True)
    roster = tuple(protocol.FINAL_SESSIONS)
    records = {sid: {rep: _record(sid, rep) for rep in ("sua", "pmua")} for sid in roster}
    base = _base_receipt(tmp, roster, records)
    prepared = _json(tmp / "prepared_receipt.json", {"schema": "dandi688_bench_v2_prepared_v1", "final_sessions_opened": 0,
        "sessions": {sid: {"split": "train"} for sid in protocol.TRAIN_SESSIONS} |
                    {sid: {"split": "dev"} for sid in protocol.DEV_SESSIONS}})
    encoders = {rep: _artifact(tmp / f"encoder.{rep}.pt", rep) for rep in ("sua", "pmua")}
    base_campaign = tmp / "base_campaign"; base_campaign.mkdir()
    main, supp, tasks = _tasks(len(cells) == 6, base_campaign)
    assert main + supp == cells
    plan = {"schema": PLAN_SCHEMA, "status": PLAN_STATUS, "base_results_already_observed": True,
            "planned_cells": cells, "flat_final_cells": cells, "main_cells": main, "supplemental_cells": supp,
            "training_tasks": tasks, "recipe": FLAT_RECIPE,
            "budget": {"segments": 24, "updates_per_segment": 3165, "batch": 32, "total_steps": 75960},
            "flat_training_source_hashes": flat_training_source_hashes(),
            "base_final_receipt": {"path": str(base), "sha256": sha256(base)},
            "prepared_receipt": {"path": str(prepared), "sha256": sha256(prepared)},
            "encoder_checkpoints": {rep: {"path": str(p), "sha256": sha256(p)} for rep, p in encoders.items()},
            "cache": str(tmp / "cache"), "base_campaign": str(base_campaign), "final_roster": list(roster)}
    plan["sha256"] = _digest(plan); plan_path = _json(tmp / "plan.json", plan)
    artifacts = {str(plan_path.resolve()): sha256(plan_path), str(base.resolve()): sha256(base), str(prepared.resolve()): sha256(prepared),
                 **{str(p): sha256(p) for p in encoders.values()}}
    entries = {}
    for cell in cells:
        checkpoint = _artifact(tmp / f"{cell}.pt", cell); artifacts[str(checkpoint)] = sha256(checkpoint)
        arm, representation, seed = finalize._spec(cell)
        entries[cell] = {"status": "SELECTED", "arm": arm, "representation": representation, "seed": seed,
                         "checkpoint_sha256": sha256(checkpoint), "artifact_paths": [str(checkpoint)]}
    seal_data = {"schema": FLAT_SEAL_SCHEMA, "status": FLAT_SEAL_STATUS, "protocol": protocol.protocol_dict(),
                 "plan_path": str(plan_path.resolve()), "plan_sha256": sha256(plan_path),
                 "base_final_receipt": str(base.resolve()), "base_final_receipt_sha256": sha256(base),
                 "prepared": {"path": str(prepared.resolve()), "sha256": sha256(prepared)},
                 "encoders": {rep: {"path": str(p), "sha256": sha256(p)} for rep, p in encoders.items()}, "required_cells": cells,
                 "cell_selections": entries, "artifacts": artifacts, "final_sessions_opened": 0}
    seal_data["sha256"] = _canonical(seal_data); seal = _json(tmp / "selection_seal.json", seal_data)
    return SimpleNamespace(roster=roster, records=records, base=base, seal=seal, cells=cells)


def _patch_final_loaders(monkeypatch: pytest.MonkeyPatch, fixture, *, canonical_mismatch=False, digest_mismatch=False):
    calls = []
    def load_pair(sid, **kwargs):
        calls.append(sid)
        kwargs["access_log"].append({"session_id": sid, "purpose": "final", "path": "synthetic"})
        result = fixture.records[sid]
        if canonical_mismatch and sid == fixture.roster[-1]:
            result = dict(result); result["sua"] = _record(sid, "sua", canonical=("wrong",))
        if digest_mismatch and sid == fixture.roster[-1]:
            result = dict(result); result["pmua"] = _record(sid, "pmua", query_seed=8)
        return result
    monkeypatch.setattr(finalize.data, "load_pair", load_pair)
    monkeypatch.setattr(finalize, "load_records", lambda _cache, rep, _split: [_record("source", rep)])
    return calls


def _scorer(roster, *, omit=None, malformed=None):
    def score(cell, records, entry):
        rep = entry["representation"]
        predictions = {sid: np.full((3, 2), 3., dtype=np.float64) for sid in roster if sid != omit}
        if malformed == "nonfinite": predictions[roster[0]][0, 0] = np.nan
        if malformed == "shape": predictions[roster[0]] = np.ones((2, 2))
        offset = {42: .20, 43: .32, 44: .44}[entry["seed"]]
        return {"selected": True, "metrics": _metric(roster, offset, records=records, representation=rep), "_predictions": predictions}
    return score


def _seal_with_synthetic_training_inspectors(monkeypatch: pytest.MonkeyPatch, fixture, tmp_path: Path) -> Path:
    """Call the production sealer while substituting only GPU/training receipt inspection."""
    prepared = tmp_path / "prepared_receipt.json"
    encoders = {rep: tmp_path / f"encoder.{rep}.pt" for rep in ("sua", "pmua")}
    plan_path = tmp_path / "plan.json"

    def add(artifacts, path):
        path = Path(path).resolve(); artifacts[str(path)] = sha256(path); return str(path)

    monkeypatch.setattr(finalize, "_prepared_binding", lambda path, artifacts: {
        "path": add(artifacts, path), "sha256": sha256(path)})
    monkeypatch.setattr(finalize, "_encoder_entries", lambda paths, artifacts, **_kwargs: {
        rep: {"path": add(artifacts, path), "sha256": sha256(path), "encoder_state": {}}
        for rep, path in paths.items()})

    def selected(cell, selection, artifacts, **_kwargs):
        arm, rep, seed = finalize._spec(cell); checkpoint = Path(selection).resolve().with_name("checkpoint.pt")
        checkpoint.write_text(cell)
        return {"status": "SELECTED", "arm": arm, "representation": rep, "seed": seed,
                "checkpoint_sha256": sha256(checkpoint), "artifact_paths": [add(artifacts, checkpoint)]}
    monkeypatch.setattr(finalize, "_entry", selected)
    plan = json.loads(plan_path.read_text())
    selections = {}
    for task in plan["training_tasks"]:
        path = plan_path.parent / task["run"] / "selection.json"
        path.parent.mkdir(); path.write_text(task["cell"]); selections[task["cell"]] = path
    finalize.seal_selection(tmp_path / "sealed", prepared_receipt=prepared, plan_path=plan_path,
                            selections=selections, base_final_receipt=fixture.base, encoder_checkpoints=encoders)
    return tmp_path / "sealed/selection_seal.json"


def test_cpu_score_final_then_merged_report_has_ten_flat_and_29_rows(monkeypatch, tmp_path):
    f = _campaign(tmp_path); calls = _patch_final_loaders(monkeypatch, f)
    receipt = finalize.score_final(f.seal, tmp_path / "flat_score", device="cpu", score_cell=_scorer(f.roster))
    assert receipt["schema"] == FLAT_FINAL_SCHEMA and calls == list(f.roster)
    outputs = report.build_report(f.base, tmp_path / "flat_score/final_score_receipt.json", tmp_path / "report")
    with outputs["combined_results"].open(newline="", encoding="utf-8") as handle:
        combined = list(csv.DictReader(handle))
    assert len(combined) == 19 + 10
    by_cell = {row["cell"]: row for row in combined}
    assert set(by_cell) == set(FINAL_SCORE_CELLS) | set(f.cells)
    for cell in f.cells:
        row = receipt["results"][cell]
        with np.load(row["predictions"]["path"], allow_pickle=False) as archive:
            assert set(archive.files) == set(f.roster)
            for day in f.roster:
                assert archive[day].shape == (3, 2)
                assert np.isfinite(archive[day]).all()
        arm, rep, seed = finalize._spec(cell)
        learned = f"{arm}_{rep}" + ("" if seed == 42 else f"_s{seed}")
        for day, metric in zip(f.roster, row["result"]["metrics"]["sessions"], strict=True):
            learned_r2 = next(x["r2"] for x in json.loads(f.base.read_text())["results"][learned]["result"]["metrics"]["sessions"] if x["session_id"] == day)
            flat_r2 = metric["r2"]
            assert float(by_cell[cell][day]) == pytest.approx(flat_r2)
            assert np.isclose(learned_r2 - flat_r2, .1 + FINAL_SCORE_CELLS.index(learned) / 1000 - {42: .20, 43: .32, 44: .44}[seed])
    with outputs["paired_differences"].open(newline="", encoding="utf-8") as handle:
        paired = list(csv.DictReader(handle))
    assert len(paired) == 10 * 7
    for cell in f.cells:
        arm, rep, seed = finalize._spec(cell); learned = f"{arm}_{rep}" + ("" if seed == 42 else f"_s{seed}")
        expected = .1 + FINAL_SCORE_CELLS.index(learned) / 1000 - {42: .20, 43: .32, 44: .44}[seed]
        rows = [r for r in paired if (r["arm"], r["representation"], int(r["seed"])) == (arm, rep, seed)]
        assert len(rows) == 7
        assert all(float(r["learned_minus_flat_r2"]) == pytest.approx(expected) for r in rows)
    summary = json.loads(outputs["full_seed_summary"].read_text())
    for rep in ("sua", "pmua"):
        for variant, cells in (("learned", [f"full_{rep}", f"full_{rep}_s43", f"full_{rep}_s44"]),
                               ("flat", [f"full_flat_{rep}", f"full_flat_{rep}_s43", f"full_flat_{rep}_s44"])):
            expected = [float(by_cell[cell]["mean_r2"]) for cell in cells]
            actual = summary["full"][rep][variant]
            assert actual["seed_mean_r2"] == pytest.approx(expected)
            assert actual["three_seed_mean_r2"] == pytest.approx(mean(expected))
            assert actual["three_seed_sample_sd_r2"] == pytest.approx(stdev(expected))
        assert len(set(summary["full"][rep]["flat"]["seed_mean_r2"])) == 3


def test_seed42_only_six_cell_chain_reports_25_rows_without_flat_seed_fabrication(monkeypatch, tmp_path):
    f = _campaign(tmp_path, CELLS6)
    seal_path = _seal_with_synthetic_training_inspectors(monkeypatch, f, tmp_path)
    _patch_final_loaders(monkeypatch, f)
    receipt = finalize.score_final(seal_path, tmp_path / "score", device="cpu", score_cell=_scorer(f.roster))
    outputs = report.build_report(f.base, tmp_path / "score/final_score_receipt.json", tmp_path / "report")

    with outputs["combined_results"].open(newline="", encoding="utf-8") as handle:
        combined = list(csv.DictReader(handle))
    assert len(combined) == 25
    assert sum(row["variant"] == "flat_neural" for row in combined) == 6
    assert sum(row["variant"] == "learned_slope_neural" for row in combined) == 10
    assert sum(row["variant"] == "existing_control" for row in combined) == 9
    assert {row["cell"] for row in combined if row["variant"] == "flat_neural"} == set(CELLS6)

    with outputs["paired_differences"].open(newline="", encoding="utf-8") as handle:
        paired = list(csv.DictReader(handle))
    assert len(paired) == 6 * 7
    base = json.loads(f.base.read_text())["results"]
    for cell in CELLS6:
        arm, rep, seed = finalize._spec(cell); assert seed == 42
        learned = f"{arm}_{rep}"
        expected = .1 + FINAL_SCORE_CELLS.index(learned) / 1000 - .20
        rows = [row for row in paired if (row["arm"], row["representation"], int(row["seed"])) == (arm, rep, 42)]
        assert len(rows) == 7
        assert all(float(row["learned_minus_flat_r2"]) == pytest.approx(expected) for row in rows)
        # The reported six-day Flat values remain exactly paired to learned seed42.
        flat_sessions = receipt["results"][cell]["result"]["metrics"]["sessions"]
        learned_sessions = base[learned]["result"]["metrics"]["sessions"]
        assert [x["r2"] - y["r2"] for x, y in zip(learned_sessions, flat_sessions, strict=True)] == pytest.approx([expected] * 6)
        with np.load(receipt["results"][cell]["predictions"]["path"], allow_pickle=False) as archive:
            assert set(archive.files) == set(f.roster)
            assert all(archive[day].shape == (3, 2) and np.isfinite(archive[day]).all() for day in f.roster)

    summary = json.loads(outputs["full_seed_summary"].read_text())
    for rep in ("sua", "pmua"):
        flat = summary["full"][rep]["flat"]
        assert flat["available_seeds"] == [42]
        assert flat["three_seed_mean_r2"] is None and flat["three_seed_sample_sd_r2"] is None
        learned = summary["full"][rep]["learned"]
        assert learned["available_seeds"] == [42, 43, 44]
        assert learned["three_seed_mean_r2"] is not None and learned["three_seed_sample_sd_r2"] is not None
    markdown = outputs["report"].read_text(encoding="utf-8")
    assert all(row["cell"] in markdown for row in combined)
    assert "full_flat_sua_s43" not in markdown and "full_flat_pmua_s44" not in markdown


def test_score_claim_is_exclusive(monkeypatch, tmp_path):
    f = _campaign(tmp_path); _patch_final_loaders(monkeypatch, f)
    finalize.score_final(f.seal, tmp_path / "one", score_cell=_scorer(f.roster))
    with pytest.raises(PermissionError, match="claim"):
        finalize.score_final(f.seal, tmp_path / "two", score_cell=_scorer(f.roster))


def test_seal_selection_then_cap_accepts_real_plan_and_bound_paths(monkeypatch, tmp_path):
    """Exercise the actual sealer/cap handshake; only training-heavy inspectors are stubbed."""
    f = _campaign(tmp_path)
    plan_path = tmp_path / "plan.json"

    with pytest.raises(ValueError, match="roster"):
        finalize.seal_selection(tmp_path / "bad-seal", prepared_receipt=tmp_path / "prepared_receipt.json", plan_path=plan_path,
                                selections={}, base_final_receipt=f.base,
                                encoder_checkpoints={rep: tmp_path / f"encoder.{rep}.pt" for rep in ("sua", "pmua")})
    seal_path = _seal_with_synthetic_training_inspectors(monkeypatch, f, tmp_path)
    seal = json.loads(seal_path.read_text())
    assert seal["plan_sha256"] == sha256(plan_path) and seal["base_final_receipt_sha256"] == sha256(f.base)
    cap = finalize.FlatFinalAccess.from_manifest(seal_path)
    cap.validate()


@pytest.mark.parametrize("kind", ["canonical", "query_digest"])
def test_score_rejects_bad_final_binding_before_score_callback(monkeypatch, tmp_path, kind):
    f = _campaign(tmp_path); _patch_final_loaders(monkeypatch, f, canonical_mismatch=kind == "canonical", digest_mismatch=kind == "query_digest")
    called = False
    def never(*_args):
        nonlocal called; called = True; return {}
    with pytest.raises(RuntimeError):
        finalize.score_final(f.seal, tmp_path / "score", score_cell=never)
    assert not called


def test_report_rejects_missing_prediction_date_npz_hash_and_receipt_roster_binding(monkeypatch, tmp_path):
    # The scorer rejects incomplete callback output before it can become a receipt.
    f = _campaign(tmp_path); _patch_final_loaders(monkeypatch, f)
    with pytest.raises(RuntimeError, match="incomplete predictions"):
        finalize.score_final(f.seal, tmp_path / "score", score_cell=_scorer(f.roster, omit=f.roster[-1]))
    # Report revalidates an otherwise sealed receipt if its NPZ loses one date.
    f = _campaign(tmp_path / "missing-date"); _patch_final_loaders(monkeypatch, f)
    finalize.score_final(f.seal, tmp_path / "missing-date/score", score_cell=_scorer(f.roster))
    receipt_path = tmp_path / "missing-date/score/final_score_receipt.json"
    receipt = json.loads(receipt_path.read_text()); row = receipt["results"][f.cells[0]]
    npz = Path(row["predictions"]["path"])
    with np.load(npz, allow_pickle=False) as archive:
        np.savez_compressed(npz, **{sid: archive[sid] for sid in f.roster[:-1]})
    row["predictions"]["sha256"] = sha256(npz); _json(receipt_path, receipt)
    with pytest.raises(ValueError, match="coverage"):
        report.build_report(f.base, receipt_path, tmp_path / "missing-date/report")
    # A valid receipt is rejected when its stored NPZ is changed after scoring.
    f = _campaign(tmp_path / "hash"); _patch_final_loaders(monkeypatch, f)
    finalize.score_final(f.seal, tmp_path / "hash/score", score_cell=_scorer(f.roster))
    npz = tmp_path / "hash/score/full_flat_sua.final_predictions.npz"; npz.write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA"):
        report.build_report(f.base, tmp_path / "hash/score/final_score_receipt.json", tmp_path / "hash/report")
    # A receipt cannot silently drop a Flat cell or bind it to different query/truth rows.
    f = _campaign(tmp_path / "binding"); _patch_final_loaders(monkeypatch, f)
    receipt = finalize.score_final(f.seal, tmp_path / "binding/score", score_cell=_scorer(f.roster))
    receipt["results"].pop(f.cells[-1]); _json(tmp_path / "binding/score/final_score_receipt.json", receipt)
    with pytest.raises(ValueError, match="exactly match"):
        report.build_report(f.base, tmp_path / "binding/score/final_score_receipt.json", tmp_path / "binding/missing-cell")
    f = _campaign(tmp_path / "digest"); _patch_final_loaders(monkeypatch, f)
    finalize.score_final(f.seal, tmp_path / "digest/score", score_cell=_scorer(f.roster))
    receipt_path = tmp_path / "digest/score/final_score_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["results"][f.cells[0]]["predictions"]["sessions"][f.roster[0]]["velocity_sha256"] = "different"
    _json(receipt_path, receipt)
    with pytest.raises(ValueError, match="binding mismatch"):
        report.build_report(f.base, receipt_path, tmp_path / "digest/report")
    # Updating the outer NPZ file SHA does not waive the sealed per-array digest.
    f = _campaign(tmp_path / "array-digest"); _patch_final_loaders(monkeypatch, f)
    finalize.score_final(f.seal, tmp_path / "array-digest/score", score_cell=_scorer(f.roster))
    receipt_path = tmp_path / "array-digest/score/final_score_receipt.json"
    receipt = json.loads(receipt_path.read_text()); row = receipt["results"][f.cells[0]]; npz = Path(row["predictions"]["path"])
    with np.load(npz, allow_pickle=False) as archive:
        altered = {day: archive[day] for day in f.roster}
    altered[f.roster[0]] = altered[f.roster[0]] + 1.
    np.savez_compressed(npz, **altered); row["predictions"]["sha256"] = sha256(npz); _json(receipt_path, receipt)
    with pytest.raises(ValueError, match="binding mismatch"):
        report.build_report(f.base, receipt_path, tmp_path / "array-digest/report")


@pytest.mark.parametrize("malformed", ["shape", "nonfinite"])
def test_report_rejects_malformed_prediction_arrays(monkeypatch, tmp_path, malformed):
    f = _campaign(tmp_path); _patch_final_loaders(monkeypatch, f)
    with pytest.raises(RuntimeError, match="shape/finite"):
        finalize.score_final(f.seal, tmp_path / "score", score_cell=_scorer(f.roster, malformed=malformed))
    f = _campaign(tmp_path / "report"); _patch_final_loaders(monkeypatch, f)
    finalize.score_final(f.seal, tmp_path / "report/score", score_cell=_scorer(f.roster))
    receipt_path = tmp_path / "report/score/final_score_receipt.json"
    receipt = json.loads(receipt_path.read_text()); row = receipt["results"][f.cells[0]]
    npz = Path(row["predictions"]["path"])
    with np.load(npz, allow_pickle=False) as archive:
        changed = {sid: archive[sid] for sid in f.roster}
    changed[f.roster[0]] = (np.ones((2, 2)) if malformed == "shape" else np.full((3, 2), np.nan))
    np.savez_compressed(npz, **changed); row["predictions"]["sha256"] = sha256(npz); _json(receipt_path, receipt)
    with pytest.raises(ValueError, match="prediction"):
        report.build_report(f.base, receipt_path, tmp_path / "report/report")


def test_flat_seal_rejects_unsealed_plan_selfhash_and_bad_checkpoint(tmp_path):
    with pytest.raises(PermissionError):
        finalize.FlatFinalAccess.from_manifest(tmp_path / "missing.json")
    f = _campaign(tmp_path)
    plan = json.loads((tmp_path / "plan.json").read_text()); plan["sha256"] = "0" * 64; _json(tmp_path / "plan.json", plan)
    with pytest.raises(PermissionError, match="plan/dependency"):
        finalize.FlatFinalAccess.from_manifest(f.seal)
    f = _campaign(tmp_path / "checkpoint")
    seal = json.loads(f.seal.read_text()); cell = f.cells[0]; seal["cell_selections"][cell]["checkpoint_sha256"] = "0" * 64; seal["sha256"] = _canonical(seal); _json(f.seal, seal)
    with pytest.raises(PermissionError, match="selected entry"):
        finalize.FlatFinalAccess.from_manifest(f.seal)
