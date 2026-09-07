from __future__ import annotations

from pathlib import Path
import hashlib
import json
import stat

import numpy as np
import pytest


def test_actual_v1_incident_graph_and_scientific_state_are_bound():
    from tfpd_exploration.h1_series_20260830.src.h1_calibration_profile_film_v2.evaluate import (
        validate_v1_incident,
    )

    repo = Path(__file__).resolve().parents[2]
    witness = validate_v1_incident(repo)
    assert witness["validated"] is True
    assert witness["failure_error"].endswith("LP-ZERO prediction anchor drift")
    assert witness["first_fold_film_state_sha256"] == {
        "EP-FILM": "60eb1161d9621b0830d226e0eaf07aefb0ee1f56eb72d42880e95aa518bc6b58",
        "LP-FILM": "fba420f403b248ee0a6fe6c1898f704dc6ae4be0be03d3a92324771afe979736",
    }


def test_v2_anchor_accepts_historical_sha_drift_but_v1_strict_mode_rejects(monkeypatch):
    from tfpd_exploration.h1_series_20260830.src.h1_calibration_profile_film_v1 import evaluate
    from tfpd_exploration.h1_series_20260830.src.h1_cross_record_postpool_v1.core import array_sha256

    ep = np.full((2, 7), 0.25, dtype=np.float32)
    lp = np.full((2, 7), 0.50, dtype=np.float32)
    ep_film = np.full((2, 7), 0.30, dtype=np.float32)
    lp_film = np.full((2, 7), 0.60, dtype=np.float32)
    target = np.zeros((2, 7), dtype=np.float32)

    def fake_zero(net, row, *, device, late):
        del net, row, device
        return (lp, 0.40000004) if late else (ep, 0.30000004)

    def fake_film(net, film, row, *, device, late):
        del net, film, row, device
        return (lp_film, 0.45) if late else (ep_film, 0.35)

    monkeypatch.setattr(evaluate, "_predict_zero", fake_zero)
    monkeypatch.setattr(evaluate, "_predict_film", fake_film)
    row = {
        "session": "toy",
        "target_stream": target,
        "endpoints": np.array([0, 1], dtype=np.int64),
        "public": {"session": "toy"},
    }
    old = {
        "session": "toy",
        "prediction_sha256": {"FROZEN-C1": "0" * 64, "LP-R3": "1" * 64},
        "target_sha256": array_sha256(target),
        "r2": {"FROZEN-C1": 0.3, "LP-R3": 0.4},
    }
    scores, rows = evaluate._score(
        object(), object(), {"EP-FILM": object(), "LP-FILM": object()}, [row], [old],
        device="cpu", require_historical_prediction_sha=False, anchor_r2_tolerance=1.0e-7,
    )
    assert scores["EP-ZERO"] == pytest.approx(0.30000004)
    assert rows[0]["zero_anchor"]["same_process_repeat_prediction_sha_exact"] is True
    assert rows[0]["zero_anchor"]["historical_prediction_sha_exact"] == {
        "EP-ZERO": False,
        "LP-ZERO": False,
    }
    with pytest.raises(Exception, match="EP-ZERO prediction anchor drift"):
        evaluate._score(
            object(), object(), {"EP-FILM": object(), "LP-FILM": object()}, [row], [old],
            device="cpu", require_historical_prediction_sha=True, anchor_r2_tolerance=1.0e-7,
        )


def test_v2_anchor_still_rejects_same_process_nondeterminism(monkeypatch):
    from tfpd_exploration.h1_series_20260830.src.h1_calibration_profile_film_v1 import evaluate
    from tfpd_exploration.h1_series_20260830.src.h1_cross_record_postpool_v1.core import array_sha256

    first = np.zeros((1, 7), dtype=np.float32)
    second = np.ones((1, 7), dtype=np.float32)
    calls = iter(((first, 0.1), (first, 0.2), (second, 0.1), (first, 0.2)))
    monkeypatch.setattr(evaluate, "_predict_zero", lambda *args, **kwargs: next(calls))
    monkeypatch.setattr(evaluate, "_predict_film", lambda *args, **kwargs: (first, 0.1))
    row = {
        "session": "toy", "target_stream": first, "endpoints": np.array([0]),
        "public": {"session": "toy"},
    }
    old = {
        "session": "toy",
        "prediction_sha256": {"FROZEN-C1": array_sha256(first), "LP-R3": array_sha256(first)},
        "target_sha256": array_sha256(first),
        "r2": {"FROZEN-C1": 0.1, "LP-R3": 0.2},
    }
    with pytest.raises(Exception, match="same-process repeat drift"):
        evaluate._score(
            object(), object(), {"EP-FILM": object(), "LP-FILM": object()}, [row], [old],
            device="cpu", require_historical_prediction_sha=False, anchor_r2_tolerance=1.0e-7,
        )


def test_v2_dry_cli_is_inert_and_does_not_import_torch():
    import json
    import os
    import subprocess
    import sys

    repo = Path(__file__).resolve().parents[2]
    script = repo / "tfpd_exploration/h1_series_20260830/scripts/run_h1_calibration_profile_film_v2.py"
    env = dict(os.environ)
    env["PYTHONPATH"] = ":".join((
        str(repo / "tfpd_exploration/h1_series_20260830/src"),
        str(repo / "SPINT-main"),
        str(repo),
    ))
    result = subprocess.run(
        [sys.executable, "-S", str(script)], env=env, text=True, capture_output=True, check=True,
    )
    payload = json.loads(result.stdout)
    assert payload["status"] == "INERT_USE_EXECUTE"
    assert payload["result_root"].endswith("h1_calibration_profile_film_v2")


def test_actual_v2_terminal_graph_and_predeclared_decision_are_valid():
    repo = Path(__file__).resolve().parents[2]
    root = repo / "tfpd_exploration/h1_series_20260830/results/h1_calibration_profile_film_v2"
    dates = ("19250108", "19250113", "19250115", "19250119", "19250120")
    bodies = {"attempt.json", "score.json", "terminal.json"}
    bodies |= {f"training_{date}.json" for date in dates}
    bodies |= {f"fold_{date}.json" for date in dates}
    bodies |= {
        f"checkpoint_{date}_{arm}.pt"
        for date in dates
        for arm in ("ep-film", "lp-film")
    }
    assert {path.name for path in root.iterdir()} == bodies | {f"{name}.sha256" for name in bodies}
    digests = {}
    for name in bodies:
        body = root / name
        side = root / f"{name}.sha256"
        assert stat.S_IMODE(body.stat().st_mode) == stat.S_IMODE(side.stat().st_mode) == 0o444
        assert body.stat().st_nlink == side.stat().st_nlink == 1
        digest = hashlib.sha256(body.read_bytes()).hexdigest()
        assert side.read_text(encoding="ascii") == f"{digest}  {name}\n"
        digests[name] = digest
    assert digests["score.json"] == "8b56b105bf1cd965ffa9d74e323dbce7b5d2e2410a8abfb88b5eaa2151494813"
    assert digests["terminal.json"] == "f6432d0832d0173313dc9678078e6e4d348928218396b602d7320ec1b7dcb38d"

    score = json.loads((root / "score.json").read_text(encoding="utf-8"))
    terminal = json.loads((root / "terminal.json").read_text(encoding="utf-8"))
    assert score["status"] == terminal["status"] == "COMPLETE_H1_CALIBRATION_PROFILE_FILM_V2_SOURCE_OOF"
    assert terminal["score_sha256"] == digests["score.json"]
    assert score["decision"] == terminal["decision"]
    decision = score["decision"]
    assert decision["classification"] == "FILM_EARLY_REPLICATION_ONLY"
    assert decision["early_film_gate"] == {
        "contrast": "EP-FILM_MINUS_EP-ZERO",
        "values": [
            0.021928217494672775,
            0.06403301976471798,
            0.016313906806193823,
            -0.008003867716063096,
            0.025375610771390167,
        ],
        "mean": 0.02392937742418233,
        "nonnegative_dates": 4,
        "worst": -0.008003867716063096,
        "required_mean": 0.005,
        "required_nonnegative_dates": 4,
        "required_worst": -0.01,
        "pass": True,
    }
    assert decision["late_film_gate"]["mean"] == -0.00021630915713821697
    assert decision["late_film_gate"]["pass"] is False
    assert decision["pooling_at_zero_mean"] == 0.0400181831916593
    assert decision["selected_h1_product"] == "LP-R3"
    assert terminal["formal_heldout_opened"] is terminal["evalai_opened"] is False
    assert terminal["gpu1_touched"] is False
    for fold in score["folds"]:
        assert fold["state_before_sha256"] == fold["state_after_sha256"]
        assert fold["target_optimizer_steps"] == fold["target_backward_steps"] == fold["target_model_updates"] == 0
        for row in fold["target_rows"]:
            anchor = row["zero_anchor"]
            assert anchor["same_process_repeat_prediction_sha_exact"] is True
            assert max(anchor["historical_r2_abs_error"].values()) <= 1.0e-7
