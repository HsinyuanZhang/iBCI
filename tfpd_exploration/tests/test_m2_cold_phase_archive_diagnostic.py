import json
import numpy as np
import pytest

from tfpd_exploration.src.m2_family_v1 import cold_phase_archive_diagnostic as diagnostic


def test_metric_and_paired_effect_agree_with_direct_arrays():
    y = np.arange(32, dtype=np.float64).reshape(16, 2) / 10
    session = np.asarray(["a"] * 8 + ["b"] * 8)
    control = diagnostic.metrics(y, y + .2, session)
    prefix = diagnostic.metrics(y, y + .1, session)
    result = diagnostic.effect({"all": prefix}, {"all": control})["all"]
    assert result["positive_session_count"] == 2
    assert result["pooled_mse_delta"] == pytest.approx(-.03)
    assert result["equal_session_r2_delta"] > 0
    with pytest.raises(RuntimeError, match="nonfinite"):
        diagnostic.metrics(y, y * np.nan, session)


def test_archive_exact_identity_rejects_each_wrong_field():
    arrays = {"prediction": np.zeros((1011, 2)), "target": np.ones((1011, 2)),
              "start": np.arange(1011), "session": np.asarray(["s"] * 1011)}
    reference = {k: arrays[k].copy() for k in ("target", "start", "session")}
    diagnostic.check_arrays(arrays, reference)
    for key in reference:
        bad = {k: value.copy() for k, value in arrays.items()}
        bad[key][0] = "x" if key == "session" else -99
        with pytest.raises(RuntimeError, match=key):
            diagnostic.check_arrays(bad, reference)


def test_unfinished_guard_before_historical_arrays(tmp_path, monkeypatch):
    (tmp_path / "report.json").write_text(json.dumps({"status": "TRAINING"}))
    monkeypatch.setattr(diagnostic, "load_arrays", lambda *args: pytest.fail("must not load historical arrays"))
    with pytest.raises(RuntimeError, match="completed diagnostic"):
        diagnostic.run(tmp_path, tmp_path / "output.json")
