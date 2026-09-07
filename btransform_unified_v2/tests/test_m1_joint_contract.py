from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
import types

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/rift_v1/m1_joint_train.py"


def runner():
    spec = importlib.util.spec_from_file_location("m1_joint_runner", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


def test_joint_checkpoint_binds_live_b3s_provenance_and_formal_step_law(tmp_path):
    module = runner()
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    ema = module.DecoderEMA(model, decay=.9)
    meta = {"arm": "B_ACTIVITY_ONLY", "seed": 42, "source_hashes": {}, "source_contract": {},
            "initialization_sha256": "x", "b3s": {"sfix_sha256": "fixed"}}
    payload = module._checkpoint_payload(model, optimizer, ema, epoch=1, step=1, smoke=True, meta=meta, device=torch.device("cpu"))
    assert payload["schema"] == "m1_rift_joint_epoch_checkpoint_v1"
    assert payload["config"]["b3s"] == meta["b3s"]
    payload.update({"smoke": False, "global_step": module.UPDATES_PER_EPOCH})
    path = tmp_path / "epoch_001.pt"
    torch.save(payload, path)
    assert module._validate_checkpoint(path, meta, expected_epoch=1)["epoch"] == 1
    payload["global_step"] = 1
    torch.save(payload, path)
    with pytest.raises(RuntimeError, match="step/epoch"):
        module._validate_checkpoint(path, meta, expected_epoch=1)


def test_joint_cli_refuses_score_smoke_and_nonformal_epoch_count(monkeypatch, tmp_path):
    module = runner()
    for extra in (("--stage", "score", "--max-updates-smoke", "1"), ("--epochs", "2")):
        monkeypatch.setattr("sys.argv", ["m1_joint_train.py", "--dest", str(tmp_path), "--arm", "B_ACTIVITY_ONLY", *extra])
        with pytest.raises(SystemExit) as error:
            module.main()
        assert error.value.code == 2


def test_score_progress_binds_arm_seed_sampler_and_full_provenance(monkeypatch, tmp_path):
    module = runner()
    source_contract = {"total_windows": module.EXPECTED_WINDOWS, "bound": "all source array hashes"}
    b3s = {"sfix_sha256": "fixed"}
    meta = {"status": "FORMAL", "cell": module.CELL, "arm": "B_ACTIVITY_ONLY", "seed": 42,
            "source_hashes": {}, "source_contract": source_contract, "b3s": b3s}
    receipt = {"schema": "m1_rift_joint_train_receipt_v1", "status": "COMPLETED", "cell": module.CELL,
               "arm": "B_ACTIVITY_ONLY", "seed": 42, "sampler_seed": 42, "epochs": 24,
               "steps": 24 * module.UPDATES_PER_EPOCH, "source_hashes": {}, "source_contract": source_contract, "b3s": b3s}
    (tmp_path / "run_meta.json").write_text(json.dumps(meta)); (tmp_path / "train_receipt.json").write_text(json.dumps(receipt))
    completed = {str(epoch): {"checkpoint_sha256": hashlib.sha256(str(epoch).encode()).hexdigest(),
                 "ema_ho_calib": {"partial": False, "n_windows": 3881, "equal_session_mean": .1,
                 "per_session": {name: {"window_count": n, "r2": .1} for name, n in {"20121004": 1305, "20121017": 1295, "20121024": 1281}.items()}}}
                 for epoch in range(1, 25)}
    for epoch in range(1, 25): (tmp_path / f"epoch_{epoch:03d}.pt").write_bytes(str(epoch).encode())
    ho = {"sessions": ["20121004", "20121017", "20121024"]}
    progress = {"schema": "m1_rift_joint_score_progress_v1", "cell": module.CELL, "arm": "B_ACTIVITY_ONLY", "seed": 42,
                "sampler_seed": 42, "source_hashes": {}, "source_contract": source_contract, "b3s": b3s, "ho_contract": ho, "completed": completed}
    (tmp_path / "score_progress.json").write_text(json.dumps(progress))
    class FakeModel:
        def install_session_memory(self, *args): pass
        def to(self, *args): return self
        def load_state_dict(self, *args): pass
    class FakeEMA:
        def load_state_dict(self, *args): pass
    monkeypatch.setattr(module, "_source_hashes", lambda: {})
    monkeypatch.setattr(module, "_b3s_provenance", lambda: b3s)
    monkeypatch.setattr(module, "_ho_material", lambda: {name: {"bank": object(), "calib10": object()} for name in module.HO})
    monkeypatch.setattr(module, "_ho_contract", lambda _: ho)
    monkeypatch.setattr(module, "_decoder", lambda *args: FakeModel())
    monkeypatch.setattr(module, "DecoderEMA", lambda *args, **kwargs: FakeEMA())
    monkeypatch.setattr(module, "_validate_checkpoint", lambda *args, **kwargs: {"raw_state_dict": {}, "ema": {}})
    monkeypatch.setattr(module, "_assert_ho_repeatable", lambda *args: {"status": "PASSED"})
    monkeypatch.setattr(module, "_assert_full_stream_parity", lambda *args: {"status": "PASSED"})
    args = types.SimpleNamespace(dest=tmp_path, arm="B_ACTIVITY_ONLY", seed=42, device="cpu", cpu_threads=1)
    assert module.run_score(args)["status"] == "SCORE_COMPLETED"
    for key, value in (("arm", "D_JOINT"), ("seed", 43), ("source_contract", {"drift": True})):
        mutated = json.loads((tmp_path / "score_progress.json").read_text()); mutated[key] = value
        (tmp_path / "score_progress.json").write_text(json.dumps(mutated))
        with pytest.raises(RuntimeError, match="provenance"):
            module.run_score(args)
