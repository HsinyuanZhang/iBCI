"""TFAP Stage-2 handoff tests (user audit ruling item 5, required before launch).

- strict whole-state handoff: a full stage-1-shaped state loads strict=True with
  a bitwise state-SHA roundtrip; any missing key raises (no partial load, no
  silent reset of decoder or encoder);
- fresh-Adam discipline at the Stage-2 boundary: the fine-tune runner's source
  constructs exactly one Adam AFTER the strict load and never loads optimizer
  state; a freshly constructed Adam over the loaded model starts with empty
  state (no stage-1 carryover);
- source-only data surface: the fine-tune runner reuses the arm-A datamodule
  builder that empties the validation roster before setup, so the within-dev
  and formal rosters are never opened during fitting (behavioral: val dataset
  has zero windows, train has the strict-27 sessions);
- P-Z4 zero invariants on the REAL model with the SEALED 128 payload: one Adam
  step under `zeros_like(standardized T4)` leaves W_side, exp_avg, exp_avg_sq
  elementwise exactly zero, while the identical batch under T4 admission
  produces a nonzero W_side gradient (the two arms differ only in the visible
  side, by construction of the same runner).
"""

from __future__ import annotations

import importlib.util
import re
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.tfpd_lane import arm_common as ac

# the spintshape builder imports src.models.* (owned by streaming_calibration_exp);
# extend the already-imported ROOT/src package path so both trees resolve here too
import src as _src_pkg

_STREAMING_SRC = str(ROOT.parent / "streaming_calibration_exp" / "src")
if _STREAMING_SRC not in _src_pkg.__path__:
    _src_pkg.__path__.append(_STREAMING_SRC)
for _extra in (ROOT.parent / "sua_exploration",):
    if str(_extra) not in sys.path:
        sys.path.append(str(_extra))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _spintshape():
    return _load("tfap_handoff_spintshape", "src/tfpd/spintshape_module.py")


def _real_model():
    return _spintshape().build_spintshape_model(seed=42)


def test_strict_whole_state_handoff_no_partial_load():
    model = _real_model()
    payload = {"state_dict": model.state_dict()}
    fresh = _real_model()
    fresh.load_state_dict(payload["state_dict"], strict=True)
    assert ac.state_sha256(fresh) == ac.state_sha256(model)
    tampered = {k: v for k, v in payload["state_dict"].items()}
    removed = sorted(tampered)[0]
    tampered.pop(removed)
    with pytest.raises(RuntimeError):
        fresh.load_state_dict(tampered, strict=True)
    # the missing key must be a real model key (either decoder or encoder side)
    assert removed in {k for k in payload["state_dict"]}


def test_finetune_runner_fresh_adam_after_strict_load():
    source = (ROOT / "scripts/run_tfap_finetune.py").read_text()
    # exactly one Adam constructor, and it appears after the strict whole-model load
    assert len(re.findall(r"torch\.optim\.Adam\(", source)) == 1
    load_pos = source.index("model.load_state_dict(state, strict=True)")
    adam_pos = source.index("torch.optim.Adam(")
    assert load_pos < adam_pos
    # optimizer state from stage 1 is never carried over
    assert "optimizer.load_state_dict" not in source
    assert "load_optimizer" not in source
    # behavioral: a fresh Adam over the loaded model starts empty
    model = _real_model()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8,
        weight_decay=0.0, amsgrad=False,
    )
    assert len(optimizer.state) == 0


def test_finetune_data_surface_is_source_only():
    source = (ROOT / "scripts/run_admission_arm.py").read_text()
    assert 'dm.session_files["val"] = []' in source  # before setup: within-dev never opened
    arm_runner = _load("tfap_handoff_arm_runner", "scripts/run_admission_arm.py")
    args = types.SimpleNamespace(
        train_batch_size=32, num_workers=0, seed=42,
    )
    dm, _a2 = arm_runner.build_datamodule(args)
    assert len(dm.session_splits["train"]) == 27
    assert len(dm.session_splits["val"]) == 6  # roster strings only
    assert len(dm.val_dataset.window_indices) == 0  # zero within-dev windows loaded
    assert dm.train_dataset is not None and len(dm.train_dataset.window_indices) > 0


@pytest.mark.real_payload
def test_pz4_zero_invariants_with_sealed_128_payload():
    payload_path = ROOT / "results/tfap_stage0_v1/jenkins_derived_payload.npz"
    if not payload_path.is_file():
        pytest.skip("sealed stage-0 payload not present")
    with np.load(payload_path, allow_pickle=False) as data:
        t4_std = torch.from_numpy(data["t4_standardized"].copy()).float()
        calib = torch.from_numpy(data["calib_trials"][:1].copy()).float()
    n_units = t4_std.shape[0]
    torch.manual_seed(0)
    neural = torch.rand(2, 50, n_units)

    model = _real_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.0)
    z4 = ac.admit_side(t4_std.unsqueeze(0).expand(2, -1, -1), "z4")
    assert int(torch.count_nonzero(z4).item()) == 0 and not z4.signbit().any()
    model.train()
    pred, _ = model(neural, calib_trials=calib.expand(2, -1, -1, -1), side_features=z4)
    loss = (pred ** 2).mean()
    loss.backward()
    encoder = model.id_encoder
    hidden, side_dim = encoder.hidden_dim, encoder.side_dim
    grad_block = encoder.post_pool[0].weight.grad[:, hidden : hidden + side_dim]
    assert int(torch.count_nonzero(grad_block).item()) == 0
    optimizer.step()
    w = ac.w_side_block(model)
    assert int(torch.count_nonzero(w).item()) == 0 and not w.signbit().any()
    exp_avg = ac.w_side_moment(model, optimizer, "exp_avg")
    exp_avg_sq = ac.w_side_moment(model, optimizer, "exp_avg_sq")
    assert int(torch.count_nonzero(exp_avg).item()) == 0
    assert int(torch.count_nonzero(exp_avg_sq).item()) == 0

    # the T4-visible twin on the identical batch immediately drives W_side
    optimizer.zero_grad(set_to_none=True)
    pred, _ = model(neural, calib_trials=calib.expand(2, -1, -1, -1), side_features=t4_std.unsqueeze(0).expand(2, -1, -1))
    (pred ** 2).mean().backward()
    grad_t4 = encoder.post_pool[0].weight.grad[:, hidden : hidden + side_dim]
    assert int(torch.count_nonzero(grad_t4).item()) > 0


def test_stage3_dual_gates_frozen_in_source():
    source = (ROOT / "scripts/run_tfap_stage3.py").read_text()
    assert "engineering_gate" in source and "mechanism_gate" in source
    assert "BOTH_GATES_PASS__TFAP_ADOPTED_CANDIDATE" in source
    assert "ENGINEERING_ONLY__GENERIC_PRETRAINING_RECIPE" in source
    assert "BOTH_GATES_FAIL__STOP_ROUTE" in source
    assert "bootstrap_95_interval" in source  # CI-lower-bound alternative branch
    # provable discipline anchors
    assert "state_unchanged_during_scoring" in source
    assert "grads_all_none_after_scoring" in source
    assert "optimizer_instances_created" in source
    assert "stage2_provenance" in source
