from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


SCRIPT = Path(__file__).parents[1] / "scripts/diagnostics_v1/summarize_m1_joint_pair.py"
spec = importlib.util.spec_from_file_location("m1_joint_summary", SCRIPT)
mod = importlib.util.module_from_spec(spec)
assert spec.loader
spec.loader.exec_module(mod)


def _digest(char: str = "a") -> str:
    return char * 64


def _source_contract() -> dict:
    return {"sessions": list(mod.SOURCE_SESSIONS), "heldout_sessions": [], "total_windows": 213336,
            "updates_per_epoch": mod.UPDATES_PER_EPOCH, "sampler_batch_sha256": _digest("e"), "bank_report": {"fixed": True},
            "sampler": {"batch": 32, "seed": 42, "shuffle": True, "balance": False, "reshuffle_each_epoch": False},
            "bank_hashes": {s: {"budget": 10, "e0_sha256": _digest("b"), "carrier_sha256": _digest("c")} for s in mod.SOURCE_SESSIONS},
            "query_hashes": {s: {"query_pad_bins": 99, "window_count": i + 1, "eval_mask_sha256": _digest("f"), "window_starts_sha256": _digest("a"), "validity": "all source windows post-M10 legal; explicit all-true mask"} for i, s in enumerate(mod.SOURCE_SESSIONS)},
            "windows_by_session": {s: i + 1 for i, s in enumerate(mod.SOURCE_SESSIONS)}}


def _ho(*, full: bool = True) -> dict:
    keys = ("body_sha256", "carrier_sha256", "e0_sha256", "starts_sha256", "target_sha256")
    if full:
        keys += ("neural_sha256", "covariate_sha256", "calib10_sha256")
    return {"sessions": list(mod.HO_SESSIONS), "total_windows": 3881, "query_pad_bins": 99,
            "per_session": {s: {"window_count": mod.HO_WINDOWS[s], **{key: _digest(chr(97 + i)) for i, key in enumerate(keys)}} for s in mod.HO_SESSIONS}}


def _meta() -> dict:
    b3s = {"sfix_path": "/tmp/fixed.pt", "sfix_sha256": _digest("b")}
    return {"arm": "B_ACTIVITY_ONLY", "seed": 42, "source_hashes": {"/tmp/source.py": _digest("c")},
            "source_contract": _source_contract(), "initialization_sha256": _digest("d"), "b3s": b3s}


def _checkpoint(meta: dict, epoch: int) -> dict:
    raw = {"decoder.weight": torch.ones(2)}
    return {"schema": "m1_rift_joint_epoch_checkpoint_v1", "cell": mod.CELL, "smoke": False,
            "epoch": epoch, "global_step": epoch * mod.UPDATES_PER_EPOCH,
            "config": {"arm": meta["arm"], "seed": 42, "sampler_seed": 42, "context_bins": 100,
                       "proj_dim": None, "epochs": 24, "batch": 32, "lr": 1e-4, "bias_mode": "recency",
                       "attention_backend": "local", "b3s": meta["b3s"]}, "source_hashes": meta["source_hashes"],
            "source_contract": meta["source_contract"], "initialization_sha256": meta["initialization_sha256"],
            "raw_state_dict": raw, "ema": {"decay": .9995, "n_updates": epoch * mod.UPDATES_PER_EPOCH, "shadow": {"decoder.weight": torch.ones(2)}}}


def test_checkpoint_identity_rejects_wrong_identity_and_nonfinite_raw(tmp_path):
    meta = _meta(); path = tmp_path / "epoch_001.pt"
    torch.save(_checkpoint(meta, 1), path)
    assert mod.validate_checkpoint_identity(path, meta, 1) is None
    bad = _checkpoint(meta, 1); bad["global_step"] = 1; torch.save(bad, path)
    assert "identity mismatch" in mod.validate_checkpoint_identity(path, meta, 1)
    bad = _checkpoint(meta, 1); bad["raw_state_dict"]["decoder.weight"][0] = float("nan"); torch.save(bad, path)
    assert "non-finite" in mod.validate_checkpoint_identity(path, meta, 1)


def test_pair_input_drift_is_rejected_before_any_paired_summary():
    source = _source_contract(); joint_ho = _ho()
    b = {"meta": {"source_contract": source, "b3s": {"fixed": 1}}, "score": {"ho_contract": joint_ho}}
    d = {"meta": {"source_contract": {**source, "total_windows": 1}, "b3s": {"fixed": 1}}, "score": {"ho_contract": joint_ho}}
    concat = {"meta": {"source_contract": source}, "score": {"ho_contract": _ho(full=False)}}
    errors = mod.inputs_comparable(b, d, concat)
    assert "B/D source_contract drift" in errors
