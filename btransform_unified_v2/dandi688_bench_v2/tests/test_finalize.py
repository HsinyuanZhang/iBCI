from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from dandi688_bench_v2 import data, protocol
from dandi688_bench_v2.common import RECIPE, SCHEMA, digest, record_binding, sha256, source_hashes
from dandi688_bench_v2.final_access import FINAL_SCORE_CELLS, REQUIRED_CELLS, SUPPLEMENTAL_FULL_CELLS
from dandi688_bench_v2.finalize import NEURAL_CELLS, seal_selection, score_final


def _json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True))


def _prepared(tmp_path: Path) -> Path:
    sessions = {}
    root = tmp_path / "cache"; root.mkdir()
    for name in (*protocol.TRAIN_SESSIONS, *protocol.DEV_SESSIONS):
        reps = {}
        for rep in ("sua", "pmua"):
            file = root / f"{name}.{rep}.npz"; file.write_bytes(f"{name}-{rep}".encode())
            reps[rep] = {"file": file.name, "sha256": sha256(file)}
        sessions[name] = {"split": "train" if name in protocol.TRAIN_SESSIONS else "dev", "representations": reps}
    path = root / "prepared_receipt.json"
    _json(path, {"schema": "dandi688_bench_v2_prepared_v1", "protocol": protocol.protocol_dict(),
                 "include_dev": True, "sessions": sessions, "access_log": [], "final_sessions_opened": 0})
    return path

def _stats(representation: str) -> dict:
    state = {"schema": SCHEMA + "_source_stats", "status": "FORMAL", "representation": representation,
             "source_sessions": list(protocol.TRAIN_SESSIONS), "source_binding": record_binding(_records(representation, "train"))}
    state["sha256"] = digest(state)
    return state


def _checkpoint(path: Path, payload: dict) -> None:
    torch.save(payload, path)


def _encoders(tmp_path: Path) -> dict[str, Path]:
    output = {}
    for representation in ("sua", "pmua"):
        run = tmp_path / f"pretrain_{representation}"; run.mkdir()
        stats = _stats(representation); _json(run / "source_stats.json", stats)
        state = {"encoder.weight": torch.tensor([1.0 if representation == "sua" else 2.0])}
        last = run / "segment_24.pt"
        _checkpoint(last, {"schema": SCHEMA + "_checkpoint", "status": "FORMAL", "stage": "pretrain", "arm": "full",
                           "representation": representation, "seed": 42, "global_step": 75960, "model_state": state,
                           "source_stats": stats, "protocol": protocol.protocol_dict(), "recipe": RECIPE})
        curve = [{"segment": index, "global_step": index * 3165, "mean_loss": 1.0,
                  "checkpoint": "segment_24.pt" if index == 24 else f"segment_{index:02d}.pt",
                  "checkpoint_sha256": sha256(last) if index == 24 else "unused", "development": None}
                 for index in range(1, 25)]
        receipt = {"status": "FORMAL", "final_sessions_opened": 0, "protocol": protocol.protocol_dict(),
                   "representation": representation, "arm": "full", "stage": "pretrain", "seed": 42, "completed": True,
                   "source_sessions": list(protocol.TRAIN_SESSIONS), "source_binding": record_binding(_records(representation, "train")),
                   "source_stats_sha256": stats["sha256"], "recipe": RECIPE, "code_hashes": source_hashes(),
                   "encoder_line": RECIPE["encoder_line"], "encoder_carrier_fusion": RECIPE["encoder_carrier_fusion"],
                   "encoder_film": RECIPE["encoder_film"], "global_step": 75960, "actual_budget": {"segments": 24, "updates_per_segment": 3165, "batch": 32}, "segments": curve}
        _json(run / "receipt.json", receipt); _json(run / "protocol.json", {"status": "FORMAL", "final_sessions_opened": 0, "protocol": protocol.protocol_dict()})
        encoder = run / "encoder.pt"
        encoder_receipt = {"schema": SCHEMA + "_encoder", "protocol": protocol.protocol_dict(), "representation": representation,
                           "status": "FORMAL", "source_sessions": list(protocol.TRAIN_SESSIONS), "source_stats_sha256": stats["sha256"],
                           "global_step": 75960, "encoder_seed": 42, "selection": "fixed_final_ema_source_only", "recipe": RECIPE,
                           "encoder_line": RECIPE["encoder_line"], "encoder_carrier_fusion": RECIPE["encoder_carrier_fusion"],
                           "encoder_film": RECIPE["encoder_film"], "code_hashes": source_hashes(), "final_sessions_opened": 0}
        _checkpoint(encoder, {"encoder_state": {"weight": state["encoder.weight"]}, "receipt_binding": digest(encoder_receipt)})
        encoder_receipt["checkpoint_sha256"] = sha256(encoder); _json(run / "encoder.json", encoder_receipt)
        output[representation] = encoder
    return output


def _neural(tmp_path: Path, cell: str, encoders: dict[str, Path], *, smoke: bool = False) -> Path:
    run = tmp_path / f"run_{cell}"; run.mkdir()
    base_cell = cell.rsplit("_s", 1)[0] if cell in SUPPLEMENTAL_FULL_CELLS else cell
    arm, representation = base_cell.rsplit("_", 1)
    seed = int(cell[-2:]) if cell in SUPPLEMENTAL_FULL_CELLS else 42
    status = "SMOKE" if smoke else "FORMAL"; stats = _stats(representation); _json(run / "source_stats.json", stats)
    model_state = {"decoder.weight": torch.tensor([float(seed)])}
    encoder_hash = None
    if arm == "full":
        exported = torch.load(encoders[representation], map_location="cpu", weights_only=False)["encoder_state"]
        model_state.update({f"encoder.{name}": value for name, value in exported.items()})
        encoder_hash = sha256(encoders[representation])
    checkpoint = run / "segment_01.pt"
    _checkpoint(checkpoint, {"schema": SCHEMA + "_checkpoint", "status": status, "arm": arm, "representation": representation,
                             "seed": seed, "stage": "train", "global_step": 3165, "model_state": model_state, "source_stats": stats,
                             "protocol": protocol.protocol_dict(), "recipe": RECIPE, "encoder_checkpoint_sha256": encoder_hash})
    curve = [{"segment": index, "global_step": index * 3165, "mean_loss": 1.0,
              "checkpoint": checkpoint.name if index == 1 else f"segment_{index:02d}.pt",
              "checkpoint_sha256": sha256(checkpoint) if index == 1 else "unused", "development": _metrics()}
             for index in range(1, 25)]
    receipt = {"status": status, "final_sessions_opened": 0, "protocol": protocol.protocol_dict(),
               "representation": representation, "arm": arm, "stage": "train", "seed": seed, "completed": True,
               "source_sessions": list(protocol.TRAIN_SESSIONS), "source_binding": record_binding(_records(representation, "train")),
               "source_stats_sha256": stats["sha256"], "recipe": RECIPE, "code_hashes": source_hashes(), "global_step": 75960,
               "actual_budget": {"segments": 24, "updates_per_segment": 3165, "batch": 32}, "segments": curve}
    _json(run / "receipt.json", receipt); _json(run / "protocol.json", {"status": status, "final_sessions_opened": 0, "protocol": protocol.protocol_dict()})
    selection = {"schema": SCHEMA + "_selection", "status": status, "final_sessions_opened": 0,
                 "checkpoint": str(checkpoint.resolve()), "rule": "earliest_max_equal_session_dev_r2",
                 "dev_sessions": list(protocol.DEV_SESSIONS), "mean_dev_r2": _metrics()["mean_r2"], "checkpoint_sha256": sha256(checkpoint)}
    path = run / "selection.json"; _json(path, selection)
    return path


def _records(representation: str, split: str):
    roster = protocol.TRAIN_SESSIONS if split == "train" else protocol.DEV_SESSIONS
    return [type("R", (), {"session_id": name, "representation": representation, "split": split,
                             "query_indices": [0], "metadata": {"array_sha256": {"query_indices": f"q-{name}", "velocity": f"v-{name}"},
                                                               "raw_nwb_sha256": f"raw-{name}"}})()
            for name in roster]


@pytest.fixture(autouse=True)
def _fake_prepared_records(monkeypatch, request):
    if request.node.name == "test_formal_baseline_grid_binds_to_real_prepared_cache":
        return
    import dandi688_bench_v2.finalize as finalize
    monkeypatch.setattr(finalize, "load_records", lambda _cache, rep, split="train", **_kwargs: _records(rep, split))


def _metrics() -> dict:
    sessions = [{"session_id": record.session_id, "n_queries": 1, "r2": float(index), "r2_per_output": [float(index), float(index)],
                 "query_indices_sha256": record.metadata["array_sha256"]["query_indices"], "velocity_sha256": record.metadata["array_sha256"]["velocity"]}
                for index, record in enumerate(_records("pmua", "dev"), 1)]
    return {"metric": "physical_velocity_sklearn_variance_weighted_equal_session_mean", "n_sessions": 6,
            "sessions": sessions, "mean_r2": sum(row["r2"] for row in sessions) / 6}


def _baseline(tmp_path: Path) -> Path:
    run = tmp_path / "baselines"; run.mkdir()
    methods = ["wf_zs_h0", "diag_z_wf", "coral_wf", "aligned_fa_wf", "aligned_fa_stable_wf", "wf_fss_sua", "wf_fss_pmua"]
    from dandi688_bench_v2.baseline_runner import _grid
    artifacts, candidate_grid = {}, {}
    for method in methods:
        internal = "wf_fss" if method.startswith("wf_fss_") else method
        candidate_grid[method] = [{"config": config, "eligible": True, "score": _metrics()["mean_r2"], "scores": _metrics()} for config in _grid(internal, False)]
        if method.startswith("aligned_fa"):
            candidate_grid[method] = [{"config": row["config"], "eligible": False, "reason": "rank gate"}
                                      for row in candidate_grid[method]]
            artifacts[method] = {"status": "UNAVAILABLE", "reason": "rank gate"}
        else:
            model, pred = run / f"{method}.pkl", run / f"{method}.dev_predictions.npz"
            model.write_bytes(method.encode())
            if method == "wf_fss_sua":
                import numpy as np
                np.savez_compressed(pred, **{record.session_id: [0.] for record in _records("sua", "dev")})
            else: pred.write_bytes(b"dev")
            artifacts[method] = {"model": model.name, "model_sha256": sha256(model),
                                 "predictions": pred.name, "predictions_sha256": sha256(pred), "selected": candidate_grid[method][0]}
    binding = record_binding(_records("pmua", "train"))
    selection = {"schema": "dandi688_v2_cpu_baseline_selection", "status": "DEV_SELECTED", "final_loaded": False,
                 "methods": methods, "artifacts": artifacts, "candidate_grid": candidate_grid, "code": source_hashes(), "source": binding,
                 "dev": record_binding(_records("pmua", "dev"))}
    selection["sha256"] = digest(selection)
    path = run / "selection.json"; _json(path, selection)
    _json(run / "receipt.json", {"status": "DEV_SELECTED", "final_loaded": False, "selection_sha256": sha256(path)})
    return path


def _static(tmp_path: Path, raw_selection: Path) -> Path:
    checkpoint = Path(json.loads(raw_selection.read_text())["checkpoint"])
    variants = {"diag_z": {"checkpoint_sha256": sha256(checkpoint), "selected_index": 0,
                             "candidates": [{"shrinkage": .1, "metrics": _metrics()}]},
                "coral": {"checkpoint_sha256": sha256(checkpoint), "selected_index": 0,
                          "candidates": [{"shrinkage": shrinkage, "metrics": _metrics()} for shrinkage in (0., .1, .5, 1.)]}}
    path = tmp_path / "static_controls.json"
    _json(path, {"schema": SCHEMA + "_development_replay", "status": "FORMAL",
                 "final_sessions_opened": 0, "protocol": protocol.protocol_dict(),
                 "source_binding": record_binding(_records("pmua", "train")),
                 "dev_binding": record_binding(_records("pmua", "dev")), "variants": variants,
                 "code_hashes": source_hashes()})
    return path


def _inputs(tmp_path: Path, *, smoke: bool = False):
    tmp_path.mkdir(parents=True, exist_ok=True)
    encoders = _encoders(tmp_path)
    neural = {cell: _neural(tmp_path, cell, encoders, smoke=smoke and cell == "full_sua") for cell in NEURAL_CELLS}
    supplemental = {cell: _neural(tmp_path, cell, encoders) for cell in SUPPLEMENTAL_FULL_CELLS}
    return _prepared(tmp_path), neural, supplemental, _baseline(tmp_path), _static(tmp_path, neural["raw_set_pmua"]), encoders


def test_seal_rejects_smoke_and_records_all_provenance(tmp_path):
    prepared, neural, supplemental, baseline, static, encoders = _inputs(tmp_path, smoke=True)
    with pytest.raises(ValueError, match="formal"):
        seal_selection(tmp_path / "seal", prepared_receipt=prepared, neural_selections=neural,
                       baseline_selection=baseline, static_controls_receipt=static, encoder_checkpoints=encoders)
    prepared, neural, supplemental, baseline, static, encoders = _inputs(tmp_path / "valid")
    seal = seal_selection(tmp_path / "seal_valid", prepared_receipt=prepared, neural_selections=neural,
                          baseline_selection=baseline, static_controls_receipt=static,
                          supplemental_full_selections=supplemental, encoder_checkpoints=encoders)
    assert set(seal["cell_selections"]) == set(REQUIRED_CELLS)
    assert set(seal["supplemental_full_selections"]) == set(SUPPLEMENTAL_FULL_CELLS)
    assert seal["prepared"]["sha256"] == sha256(prepared)
    assert seal["roster_sha256"] == digest(seal["roster"])
    assert seal["final_sessions_opened"] == 0


@pytest.mark.parametrize("supplemental", [{}, {"full_sua_s43": "placeholder"}])
def test_seal_requires_exact_four_full_supplemental_selections(tmp_path, supplemental):
    prepared, neural, _full_supplemental, baseline, static, encoders = _inputs(tmp_path)
    with pytest.raises(ValueError, match="supplemental_full_selections"):
        seal_selection(tmp_path / "seal", prepared_receipt=prepared, neural_selections=neural,
                       baseline_selection=baseline, static_controls_receipt=static,
                       supplemental_full_selections=supplemental, encoder_checkpoints=encoders)


def test_score_final_validates_seal_then_opens_each_final_once_and_scores_once(tmp_path, monkeypatch):
    prepared, neural, supplemental, baseline, static, encoders = _inputs(tmp_path)
    seal = tmp_path / "seal" / "selection_seal.json"
    seal_selection(seal.parent, prepared_receipt=prepared, neural_selections=neural,
                   baseline_selection=baseline, static_controls_receipt=static,
                   supplemental_full_selections=supplemental, encoder_checkpoints=encoders)
    opened, calls = [], []
    def fake_load(session_id, *, purpose, final_access, raw_root, access_log):
        assert purpose == "final"; final_access.authorize(session_id)
        opened.append(session_id); access_log.append({"session_id": session_id, "split": "final", "purpose": purpose})
        return {"sua": type("R", (), {"metadata": {"canonical_electrode_keys": None, "array_sha256": {"query_indices": "sua-query", "velocity": "sua-velocity"}}})(),
                "pmua": type("R", (), {"metadata": {"canonical_electrode_keys": None, "array_sha256": {"query_indices": "pmua-query", "velocity": "pmua-velocity"}}})()}
    monkeypatch.setattr(data, "load_pair", fake_load)
    # Synthetic cache files are hash fixtures, so stub the source key lookup.
    import dandi688_bench_v2.finalize as finalize
    monkeypatch.setattr(finalize, "load_records", lambda *args: [type("R", (), {"metadata": {"canonical_electrode_keys": None}})()])
    # Callback tests do not need record fields used by the built-in key comparison.
    def scorer(cell, records, entry):
        calls.append(cell)
        assert set(records) == set(protocol.FINAL_SESSIONS)
        return {"selected": True, "cell": cell, "_predictions": {name: [1.0] for name in protocol.FINAL_SESSIONS}}
    receipt = score_final(seal, tmp_path / "final_score", score_cell=scorer)
    assert opened == list(protocol.FINAL_SESSIONS)
    assert len(calls) == len(FINAL_SCORE_CELLS) - 2
    assert receipt["final_sessions_opened"] == 6
    assert len(receipt["results"]) == 19
    assert set(receipt["results"]) == set(FINAL_SCORE_CELLS)
    assert receipt["results"]["full_sua_s43"]["predictions"]["sessions"][protocol.FINAL_SESSIONS[0]]["query_indices_sha256"] == "sua-query"
    assert receipt["results"]["full_pmua_s43"]["predictions"]["sessions"][protocol.FINAL_SESSIONS[0]]["query_indices_sha256"] == "pmua-query"
    with pytest.raises(FileExistsError):
        score_final(seal, tmp_path / "final_score", score_cell=scorer)
    with pytest.raises(PermissionError):
        score_final(seal, tmp_path / "second_final_score", score_cell=scorer)


def test_score_final_rejects_tampered_seal_before_final_open(tmp_path, monkeypatch):
    prepared, neural, supplemental, baseline, static, encoders = _inputs(tmp_path)
    seal_dir = tmp_path / "seal"
    seal_selection(seal_dir, prepared_receipt=prepared, neural_selections=neural,
                   baseline_selection=baseline, static_controls_receipt=static,
                   supplemental_full_selections=supplemental, encoder_checkpoints=encoders)
    payload = json.loads((seal_dir / "selection_seal.json").read_text())
    artifact = prepared
    artifact.write_bytes(b"changed")
    monkeypatch.setattr(data, "load_pair", lambda *a, **k: pytest.fail("final opened despite invalid seal"))
    with pytest.raises(PermissionError):
        score_final(seal_dir / "selection_seal.json", tmp_path / "final_score", score_cell=lambda *x: {})


def test_seal_rejects_selection_that_is_not_earliest_curve_maximum(tmp_path):
    prepared, neural, supplemental, baseline, static, encoders = _inputs(tmp_path)
    selection = neural["full_sua"]
    payload = json.loads(selection.read_text())
    payload["mean_dev_r2"] = 1.0
    payload["checkpoint"] = str((selection.parent / "segment_02.pt").resolve())
    (selection.parent / "segment_02.pt").write_bytes(b"other")
    payload["checkpoint_sha256"] = sha256(selection.parent / "segment_02.pt")
    receipt = json.loads((selection.parent / "receipt.json").read_text())
    receipt["segments"][1]["checkpoint_sha256"] = payload["checkpoint_sha256"]
    _json(selection.parent / "receipt.json", receipt)
    _json(selection, payload)
    with pytest.raises(ValueError, match="earliest"):
        seal_selection(tmp_path / "seal", prepared_receipt=prepared, neural_selections=neural,
                       baseline_selection=baseline, static_controls_receipt=static,
                       supplemental_full_selections=supplemental, encoder_checkpoints=encoders)


def _rewrite_baseline(path: Path, payload: dict) -> None:
    payload["sha256"] = digest({key: value for key, value in payload.items() if key != "sha256"})
    _json(path, payload)
    receipt_path = path.parent / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["selection_sha256"] = sha256(path)
    _json(receipt_path, receipt)


@pytest.mark.parametrize("mutation, message", [
    (lambda payload: payload["candidate_grid"]["wf_zs_h0"].pop(), "candidate grid"),
    (lambda payload: payload["candidate_grid"]["wf_zs_h0"].__setitem__(1, payload["candidate_grid"]["wf_zs_h0"][0]), "candidate grid"),
    (lambda payload: payload["artifacts"]["wf_zs_h0"].update(selected=payload["candidate_grid"]["wf_zs_h0"][1]), "earliest"),
    (lambda payload: payload["candidate_grid"]["wf_zs_h0"][0]["scores"]["sessions"].pop(), "six development"),
    (lambda payload: payload["source"][protocol.TRAIN_SESSIONS[0]].update(raw_nwb_sha256="wrong"), "roster mismatch"),
])
def test_seal_rejects_baseline_grid_and_prepared_binding_tampering(tmp_path, mutation, message):
    prepared, neural, supplemental, baseline, static, encoders = _inputs(tmp_path)
    payload = json.loads(baseline.read_text())
    mutation(payload)
    _rewrite_baseline(baseline, payload)
    with pytest.raises(ValueError, match=message):
        seal_selection(tmp_path / "seal", prepared_receipt=prepared, neural_selections=neural,
                       baseline_selection=baseline, static_controls_receipt=static,
                       supplemental_full_selections=supplemental, encoder_checkpoints=encoders)


def test_seal_rejects_static_winner_and_cache_binding_tampering(tmp_path):
    prepared, neural, supplemental, baseline, static, encoders = _inputs(tmp_path)
    payload = json.loads(static.read_text())
    payload["variants"]["coral"]["selected_index"] = 3
    with pytest.raises(ValueError, match="earliest"):
        _json(static, payload)
        seal_selection(tmp_path / "bad_winner", prepared_receipt=prepared, neural_selections=neural,
                       baseline_selection=baseline, static_controls_receipt=static,
                       supplemental_full_selections=supplemental, encoder_checkpoints=encoders)
    payload["variants"]["coral"]["selected_index"] = 0
    payload["dev_binding"][protocol.DEV_SESSIONS[0]]["array_sha256"]["velocity"] = "wrong"
    _json(static, payload)
    with pytest.raises(ValueError, match="roster mismatch"):
        seal_selection(tmp_path / "bad_binding", prepared_receipt=prepared, neural_selections=neural,
                       baseline_selection=baseline, static_controls_receipt=static,
                       supplemental_full_selections=supplemental, encoder_checkpoints=encoders)


def test_formal_baseline_grid_binds_to_real_prepared_cache():
    """Read-only regression for the frozen B3S CPU baseline evidence."""
    import dandi688_bench_v2.finalize as finalize
    root = Path(__file__).resolve().parents[1]
    entries = finalize._baseline_entries(root / "results" / "baselines_dev_formal" / "selection.json", {},
                                        prepared_cache=root / "results" / "prepared_2015_m33_v2")
    assert set(entries) == {"wf_zs_h0_pmua", "diag_z_wf_pmua", "coral_wf_pmua", "aligned_fa_wf_pmua",
                            "aligned_fa_stable_wf_pmua", "wf_fss_sua", "wf_fss_pmua"}

def test_default_fa_scorer_records_all_dates_when_one_target_rank_fails(tmp_path, monkeypatch):
    import pickle
    from types import SimpleNamespace
    from dandi688_bench_v2.final_access import FA_CELLS
    from dandi688_bench_v2.finalize import _default_score_cell
    model = tmp_path / "fa.pkl"
    with model.open("wb") as handle:
        pickle.dump({"fixed": True}, handle)
    records = {name: {"pmua": SimpleNamespace(session_id=name)} for name in protocol.FINAL_SESSIONS}
    import dandi688_bench_v2.baselines as baselines
    monkeypatch.setattr(baselines, "predict_baseline", lambda _model, record: (_ for _ in ()).throw(ValueError("rank deficient"))
                        if record.session_id == protocol.FINAL_SESSIONS[2] else [1.0, 2.0])
    import dandi688_bench_v2.finalize as finalize
    monkeypatch.setattr(finalize, "score_predictions", lambda record, prediction: {"session_id": record.session_id, "r2": 0.0})
    cell = next(iter(FA_CELLS))
    result = _default_score_cell(cell, records, {"kind": "baseline", "artifact_paths": [str(model)], "model_sha256": sha256(model)}, prepared_cache=tmp_path)
    assert result["status"] == "UNAVAILABLE" and result["metrics"] is None
    assert result["coverage"] == list(protocol.FINAL_SESSIONS)
    assert result["per_session"][protocol.FINAL_SESSIONS[2]]["status"] == "UNAVAILABLE"

def _seal_inputs(tmp_path: Path):
    prepared, neural, supplemental, baseline, static, encoders = _inputs(tmp_path)
    kwargs = dict(prepared_receipt=prepared, neural_selections=neural, baseline_selection=baseline,
                  static_controls_receipt=static, supplemental_full_selections=supplemental,
                  encoder_checkpoints=encoders)
    return kwargs, neural, encoders


def test_default_supplemental_sua_uses_sua_records_and_prediction_metadata(tmp_path, monkeypatch):
    """Regression: full_sua_s43 is SUA despite its trailing seed suffix."""
    from types import SimpleNamespace
    import dandi688_bench_v2.finalize as finalize
    records = {name: {"sua": SimpleNamespace(session_id=name, representation="sua"),
                      "pmua": SimpleNamespace(session_id=name, representation="pmua")}
               for name in protocol.FINAL_SESSIONS}
    monkeypatch.setattr(finalize, "load_records", lambda *_args: [SimpleNamespace()])
    monkeypatch.setattr("dandi688_bench_v2.training.load_trained_model", lambda *_args, **_kwargs: (SimpleNamespace(arm="full"), {}))
    seen = []
    monkeypatch.setattr("dandi688_bench_v2.training.predict_network", lambda _model, record, _stats: seen.append(record.representation) or [1., 2.])
    monkeypatch.setattr(finalize, "verify_stats", lambda *_args: None)
    monkeypatch.setattr(finalize, "score_predictions", lambda record, _prediction: {"session_id": record.session_id, "r2": 0.0})
    result = finalize._default_score_cell("full_sua_s43", records, {"kind": "neural", "artifact_paths": [str(tmp_path / "x.pt")], "checkpoint_sha256": "x"}, prepared_cache=tmp_path)
    assert seen == ["sua"] * len(protocol.FINAL_SESSIONS)
    assert set(result["_predictions"]) == set(protocol.FINAL_SESSIONS)


def test_score_final_rejects_canonical_mismatch_before_scorer(tmp_path, monkeypatch):
    kwargs, _neural, _encoders = _seal_inputs(tmp_path)
    seal_selection(tmp_path / "seal", **kwargs)
    import dandi688_bench_v2.finalize as finalize
    source = type("Source", (), {"metadata": {"canonical_electrode_keys": "source"}})()
    monkeypatch.setattr(finalize, "load_records", lambda *_args: [source])
    def fake_load(session_id, *, access_log, **_kwargs):
        access_log.append({"session_id": session_id, "split": "final", "purpose": "final"})
        return {"sua": type("R", (), {"metadata": {"canonical_electrode_keys": "source"}})(),
                "pmua": type("R", (), {"metadata": {"canonical_electrode_keys": "wrong"}})()}
    monkeypatch.setattr(data, "load_pair", fake_load)
    with pytest.raises(RuntimeError, match="canonical"):
        score_final(tmp_path / "seal" / "selection_seal.json", tmp_path / "final", score_cell=lambda *_args: pytest.fail("scorer ran"))


@pytest.mark.parametrize("mutation, message", [
    ("dev_mean", "aggregate"), ("payload_seed", "payload mismatch"), ("payload_step", "payload mismatch"),
    ("frozen_encoder", "frozen encoder"), ("encoder_source", "source data differ"),
])
def test_seal_rejects_neural_and_encoder_integrity_tampering(tmp_path, mutation, message):
    kwargs, neural, encoders = _seal_inputs(tmp_path)
    if mutation == "encoder_source":
        path = encoders["sua"].parent / "source_stats.json"
        payload = json.loads(path.read_text()); payload["source_binding"][protocol.TRAIN_SESSIONS[0]]["raw_nwb_sha256"] = "wrong"
        payload["sha256"] = digest({key: value for key, value in payload.items() if key != "sha256"}); _json(path, payload)
    else:
        selection = neural["full_sua"]
        chosen = Path(json.loads(selection.read_text())["checkpoint"])
        payload = torch.load(chosen, map_location="cpu", weights_only=False)
        if mutation == "payload_seed": payload["seed"] = 43
        elif mutation == "payload_step": payload["global_step"] = 999
        elif mutation == "frozen_encoder": payload["model_state"]["encoder.weight"] = torch.tensor([99.])
        else:
            receipt = json.loads((selection.parent / "receipt.json").read_text())
            receipt["segments"][0]["development"]["mean_r2"] += 1.0; _json(selection.parent / "receipt.json", receipt)
        if mutation != "dev_mean":
            _checkpoint(chosen, payload)
            selection_payload = json.loads(selection.read_text()); selection_payload["checkpoint_sha256"] = sha256(chosen); _json(selection, selection_payload)
            receipt = json.loads((selection.parent / "receipt.json").read_text()); receipt["segments"][0]["checkpoint_sha256"] = sha256(chosen); _json(selection.parent / "receipt.json", receipt)
    with pytest.raises(ValueError, match=message):
        seal_selection(tmp_path / "seal", **kwargs)
