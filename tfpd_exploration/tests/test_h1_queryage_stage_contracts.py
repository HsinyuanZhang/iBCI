"""Root-owned pure predecessor/protocol tests; no real caches or model forwards."""
import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from tfpd_exploration.src.h1_queryage_family_v1 import source_capacity as s


def _base(tmp_path, output, mode):
    source = tmp_path / "source.json"
    if not source.exists():
        s.atomic({"source": "fixture"}, source)
    files = {str(source): s.sha(source)}
    return {"protocol": s.CONFIG, "protocol_sha256": s.dig(s.CONFIG),
            "base_files": files.copy(), "files": {}, "inputs": files.copy(),
            "output": str(output), "mode": mode, "physical_gpu": 0, "threads": 1,
            "minival_rows_used": False}


def _smoke(tmp_path):
    out = tmp_path / "smoke"
    pre = _base(tmp_path, out, "smoke20")
    s.atomic({"bindings": pre}, out / "input_authority.json")
    body = {"mode": "smoke20", "status": "PASS_SMOKE_NO_CAPACITY_CLAIM",
            "updates": 20, "pre": pre, "post": copy.deepcopy(pre),
            "timing": [1.] * 20, "scores": {}, "checkpoint": None,
            "losses": [{"flat": 1., "route": 1.}] * 20,
            "owned_artifact_sha256": s.manifest(out)}
    path = out / "receipt.json"
    s.atomic(body, path)
    current = _base(tmp_path, tmp_path / "capacity", "capacity260")
    current["inputs"][str(path)] = s.sha(path)
    return path, body, current


def _scores(flat=.2, route=-.2):
    return {arm: {"pooled": {"r2_concat_float64": value,
                            "prediction_std_float64": .75,
                            "target_std_float64": 1.}}
            for arm, value in (("flat", flat), ("route", route))}


def _losses():
    return [{"flat": 2. - i / 260, "route": 2. - i / 260} for i in range(260)]


def _prior(tmp_path):
    smoke, _, pre = _smoke(tmp_path)
    s.validate_smoke(smoke, pre)
    out = Path(pre["output"])
    s.atomic({"bindings": pre}, out / "input_authority.json")
    checkpoint = out / "checkpoint.json"
    s.atomic({"fixture": "hash-only predecessor; not deserialized"}, checkpoint)
    losses = _losses()
    body = {"mode": "capacity260", "status": "COMPLETE_SOURCE_CAPACITY_NO_FORMAL",
            "updates": 260, "pre": pre, "post": copy.deepcopy(pre),
            "losses": losses, "history_losses": copy.deepcopy(losses),
            "scores": _scores(), "eligible_for_next_stage": True,
            "checkpoint": {"path": str(checkpoint), "sha256": s.sha(checkpoint)},
            "owned_artifact_sha256": s.manifest(out)}
    path = out / "receipt.json"
    s.atomic(body, path)
    current = _base(tmp_path, tmp_path / "extension", "extend1040")
    current["inputs"].update({str(smoke): s.sha(smoke), str(path): s.sha(path),
                              str(checkpoint): s.sha(checkpoint)})
    return path, checkpoint, body, current


def test_smoke_full_closure_and_exact_owned_manifest(tmp_path):
    path, body, current = _smoke(tmp_path)
    assert s.validate_smoke(path, current) == body
    missing = copy.deepcopy(body)
    missing["pre"]["base_files"] = {}
    missing["post"] = copy.deepcopy(missing["pre"])
    s.atomic(missing, path)
    current["inputs"][str(path)] = s.sha(path)
    with pytest.raises(RuntimeError, match="source/code"):
        s.validate_smoke(path, current)
    s.atomic(body, path)
    current["inputs"][str(path)] = s.sha(path)
    s.atomic({"mutated": True}, path.parent / "input_authority.json")
    with pytest.raises(RuntimeError, match="owned artifact"):
        s.validate_smoke(path, current)


def test_any260_both1040_and_finite_decline_contract():
    losses = _losses()
    assert s.capacity_eligible("capacity260", _scores(), losses) is True
    assert s.capacity_eligible("capacity260", _scores(-.2, .2), losses) is True
    assert s.capacity_eligible("capacity260", _scores(-.2, -.2), losses) is False
    assert s.capacity_eligible("capacity260", _scores(.2, .2), list(reversed(losses))) is False
    assert s.capacity_eligible("extend1040", _scores(.6, .4), losses) is False
    assert s.capacity_eligible("extend1040", _scores(.6, .6), losses) is True
    losses[0]["route"] = float("nan")
    with pytest.raises(RuntimeError, match="finite paired"):
        s.capacity_eligible("capacity260", _scores(), losses)


def test_prior_validates_checkpoint_and_retained_history(tmp_path):
    path, checkpoint, body, current = _prior(tmp_path)
    assert s.validate_prior(path, checkpoint, current) == body
    body["losses"][0]["flat"] += .5
    s.atomic(body, path)
    current["inputs"][str(path)] = s.sha(path)
    with pytest.raises(RuntimeError, match="loss history"):
        s.validate_prior(path, checkpoint, current)


def test_prior_does_not_trust_eligibility_boolean(tmp_path):
    path, checkpoint, body, current = _prior(tmp_path)
    body["scores"] = _scores(-.2, -.2)
    s.atomic(body, path)
    current["inputs"][str(path)] = s.sha(path)
    with pytest.raises(RuntimeError, match="recomputed"):
        s.validate_prior(path, checkpoint, current)


def test_strict_recursive_disk_comparison_and_fixed_forecast():
    state = {"tensor": torch.arange(3), "rng": (np.arange(4), [1, None, "raw"])}
    assert s.same(state, copy.deepcopy(state))
    changed = copy.deepcopy(state)
    changed["tensor"][1] = 10
    assert not s.same(state, changed)
    assert not s.same(torch.ones(2), torch.ones(3))
    assert s.smoke_forecasts([1.] * 20) == {"capacity260": 690., "total1040": 2160.}
    with pytest.raises(RuntimeError, match="exact20"):
        s.smoke_forecasts([1.] * 19)


def test_preflight_mode_output_and_authorization_are_bound(tmp_path, monkeypatch):
    out = tmp_path / "fresh"
    auth = tmp_path / "authorization.json"
    monkeypatch.setenv("H1_QUERYAGE_CAPACITY_GO", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setattr(s, "bindings", lambda output, **kw: _base(tmp_path, output, kw["mode"]))
    s.atomic({"status": "ROOT_REVIEW_GO", "bindings": _base(tmp_path, out, "smoke20")}, auth)
    assert s.preflight(out, auth, s.sha(auth), "smoke20")["mode"] == "smoke20"
    with pytest.raises(RuntimeError, match="absolute paths"):
        s.preflight(Path("relative"), auth, s.sha(auth), "smoke20")
    s.atomic({"status": "ROOT_REVIEW_GO", "bindings": _base(tmp_path, out, "capacity260")}, auth)
    with pytest.raises(RuntimeError, match="external authorization"):
        s.preflight(out, auth, s.sha(auth), "smoke20")
