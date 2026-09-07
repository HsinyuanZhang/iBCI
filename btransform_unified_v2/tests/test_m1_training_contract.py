"""Behavioral contract checks for the standalone formal M1 RIFT runner."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/rift_v1/m1_train.py"


def _runner():
    spec = importlib.util.spec_from_file_location("m1_rift_train_contract", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _formal_meta(runner):
    return {"source_hashes": {"runner": "a"}, "source_contract": {"face": "four-session", "total_windows": 213336}, "initialization_sha256": "init"}


def _checkpoint(runner, *, smoke=False, epoch=1):
    return {"schema": "m1_rift_epoch_checkpoint_v2", "cell": runner.CELL, "smoke": smoke, "epoch": epoch,
            "config": {"seed": 42, "context_bins": 100, "proj_dim": 16, "epochs": 24, "batch": 32, "lr": 1e-4, "bias_mode": "recency", "attention_backend": "local"},
            "source_hashes": {"runner": "a"}, "source_contract": {"face": "four-session", "total_windows": 213336}, "initialization_sha256": "init"}


def test_recipe_and_native_fullsession_constants() -> None:
    runner = _runner()
    assert (runner.SEED, runner.CONTEXT, runner.PROJ_DIM, runner.EPOCHS, runner.BATCH) == (42, 100, 16, 24, 32)
    assert runner.UPDATES_PER_EPOCH == 6665 and runner.EXPECTED_WINDOWS == 213336
    assert runner.SOURCE_SESSIONS == ("ses-20120924", "ses-20120926", "ses-20120927", "ses-20120928")


def test_m1_padded_timeline_mask_keeps_real_zero_boundary() -> None:
    runner = _runner()
    mask = runner.valid_mask_from_padded_starts((97, 99), device=torch.device("cpu"))
    assert mask.dtype == torch.bool and mask.shape == (2, 100)
    assert mask[0, :2].tolist() == [False, False]
    assert bool(mask[0, 2]) and bool(mask[1, 0]) and bool(mask[1].all())


def test_ema_helper_evaluates_and_restores_train_mode() -> None:
    runner = _runner()

    class EMA:
        shadow = {"weight": torch.tensor([2.0])}

    model = torch.nn.Linear(1, 1, bias=False); model.train(); original = model.weight.detach().clone()
    seen = []
    runner._with_ema_eval(model, EMA(), lambda: seen.append((model.training, model.weight.detach().clone())))
    assert seen[0][0] is False and torch.equal(seen[0][1], torch.tensor([[2.0]]))
    assert model.training and torch.equal(model.weight.detach(), original)


def test_checkpoint_rejects_smoke_and_stale_contract(tmp_path: Path) -> None:
    runner = _runner(); meta = _formal_meta(runner); path = tmp_path / "epoch_001.pt"
    torch.save(_checkpoint(runner, smoke=True), path)
    with pytest.raises(RuntimeError, match="checkpoint contract mismatch"):
        runner._validate_checkpoint(path, meta, expected_epoch=1)
    torch.save(_checkpoint(runner), path)
    stale = {**meta, "source_hashes": {"runner": "changed"}}
    with pytest.raises(RuntimeError, match="source_hashes mismatch"):
        runner._validate_checkpoint(path, stale, expected_epoch=1)


def test_completed_score_rows_require_full_exact_heldout_surface() -> None:
    runner = _runner()
    valid = {"partial": False, "n_windows": 3881, "equal_session_mean": 0.1, "per_session": {
        "20121004": {"window_count": 1305, "r2": 0.1}, "20121017": {"window_count": 1295, "r2": 0.2}, "20121024": {"window_count": 1281, "r2": 0.3}}}
    runner._validate_scored_report(valid)
    invalid = {**valid, "partial": True}
    with pytest.raises(RuntimeError, match="incomplete"):
        runner._validate_scored_report(invalid)


def test_score_resume_skips_valid_completed_epochs(tmp_path: Path, monkeypatch) -> None:
    """A fully validated progress file resumes without rescoring its epochs."""
    runner = _runner(); meta = _formal_meta(runner)
    monkeypatch.setattr(runner, "_source_hashes", lambda: meta["source_hashes"])
    model = torch.nn.Linear(1, 1, bias=False)
    from btransform_unified_v1.ema import DecoderEMA
    ema = DecoderEMA(model, decay=0.9995)
    monkeypatch.setattr(runner, "_decoder", lambda _device: torch.nn.Linear(1, 1, bias=False))
    monkeypatch.setattr(runner, "_ho_material", lambda: {})
    ho_contract = {"sessions": list(runner.HO), "per_session": {}, "total_windows": 3881, "query_pad_bins": 99}
    monkeypatch.setattr(runner, "_ho_contract", lambda _material: ho_contract)
    monkeypatch.setattr(runner, "_assert_ho_repeatable", lambda *_args: {"status": "PASSED"})
    monkeypatch.setattr(runner, "_assert_full_stream_parity", lambda *_args: {"status": "PASSED"})
    report = {"partial": False, "n_windows": 3881, "equal_session_mean": 0.1, "per_session": {
        "20121004": {"window_count": 1305, "r2": 0.1}, "20121017": {"window_count": 1295, "r2": 0.1}, "20121024": {"window_count": 1281, "r2": 0.1}}}
    completed = {}
    for epoch in range(1, 25):
        path = tmp_path / f"epoch_{epoch:03d}.pt"; state = _checkpoint(runner, epoch=epoch)
        state.update({"raw_state_dict": model.state_dict(), "ema": ema.state_dict()}); torch.save(state, path)
        completed[str(epoch)] = {"checkpoint_sha256": runner._sha_file(path), "ema_ho_calib": report}
    (tmp_path / "run_meta.json").write_text(__import__("json").dumps({"status": "FORMAL", "cell": runner.CELL, **meta}))
    (tmp_path / "train_receipt.json").write_text(__import__("json").dumps({"status": "COMPLETED", "cell": runner.CELL, "epochs": 24, "steps": 159960}))
    (tmp_path / "score_progress.json").write_text(__import__("json").dumps({"schema": "m1_rift_score_progress_v2", "cell": runner.CELL, "source_hashes": meta["source_hashes"], "ho_contract": ho_contract, "completed": completed}))
    args = type("Args", (), {"dest": tmp_path, "device": "cpu", "cpu_threads": 1})()
    assert runner.run_score(args)["status"] == "SCORE_COMPLETED"
