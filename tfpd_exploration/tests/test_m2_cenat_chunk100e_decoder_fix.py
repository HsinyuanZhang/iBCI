"""Regression tests for the two official-eval decoder bugs in Ce-NAT TTA.

Bug 1: official test phase waves 7 then 6 files; predict must loop over
``len(slots)``, not ``batch_size``.
Bug 2: a shared observation buffer must be rolled ONCE per predict step;
rolling once per slot wraps every other slot's history.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PKG = REPO_ROOT / "tfpd_exploration/submissions/evalai_m2_cenat_chunk100e_v1"
PAYLOAD = PKG / "artifacts/t4_m2_seed42_cenat_chunk100e_tta.pkl"
for path in (REPO_ROOT, PKG, REPO_ROOT / "streaming_calibration_exp",
             REPO_ROOT / "tfpd_exploration/submissions/evalai_m2_apfg_static_v1"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


pytest.importorskip("torch")
if not PAYLOAD.exists():
    pytest.skip("Ce-NAT payload not present", allow_module_level=True)

import src.models.streaming_calibration_module  # noqa: F401
sys.path.append(str(REPO_ROOT / "SPINT-main"))

from falcon_challenge.config import FalconConfig, FalconTask
from cenat_chunk_decoder import CenatChunkTTADecoder


def _fake_tag(session: str) -> Path:
    return Path(f"sub-MonkeyN-held-in-calib_{session}_behavior+ecephys.nwb")


def _make_decoder(batch_size: int) -> CenatChunkTTADecoder:
    return CenatChunkTTADecoder(
        task_config=FalconConfig(task=FalconTask.m2),
        model_path=str(PAYLOAD),
        batch_size=batch_size,
    )


def test_wave2_six_file_batch_does_not_indexerror():
    """Official wave 2 has 6 eval NWBs while the decoder is constructed at 7."""
    from sua_exploration.evalai_t4_m2.export_t4_payload import load_frozen_model_and_data

    _model, data_module, _tc, _meta = load_frozen_model_and_data()
    held_out = sorted(data_module.val_heldout_dataset.calib_trialized_neural_features)
    held_in = sorted(data_module.train_dataset.calib_trialized_neural_features)
    assert len(held_out) == 6 and len(held_in) == 7
    decoder = _make_decoder(7)
    decoder.reset(dataset_tags=[_fake_tag(s) for s in held_in])
    seven = np.zeros((7, 96), dtype=np.float32)
    seven[0] = 1.0
    out7 = decoder.predict(seven)
    assert out7.shape == (7, 2)
    decoder.reset(dataset_tags=[_fake_tag(s) for s in held_out])
    six = np.zeros((6, 96), dtype=np.float32)
    six[0] = 1.0
    out6 = decoder.predict(six)
    assert out6.shape == (6, 2)
    with pytest.raises(ValueError, match="active wave slots"):
        decoder.predict(seven)


def test_batch_roll_matches_independent_batch1_streams():
    """Two slots in one predict step must match two independent batch-1 runs."""
    from sua_exploration.evalai_t4_m2.export_t4_payload import load_frozen_model_and_data

    _model, data_module, _tc, _meta = load_frozen_model_and_data()
    sessions = sorted(data_module.val_heldout_dataset.calib_trialized_neural_features)[:2]
    streams = []
    for session in sessions:
        neural = np.asarray(
            data_module.val_heldout_dataset.neural_data[session], dtype=np.float32)
        streams.append(np.ascontiguousarray(neural[49:], dtype=np.float32))
    n_bins = min(400, *(int(s.shape[0]) for s in streams))
    tags = [_fake_tag(s) for s in sessions]

    batched = _make_decoder(2)
    batched.reset(dataset_tags=tags)
    solo = [_make_decoder(1) for _ in sessions]
    for decoder, tag in zip(solo, tags):
        decoder.reset(dataset_tags=[tag])

    max_abs = 0.0
    for t in range(n_bins):
        batch = np.stack([streams[0][t], streams[1][t]], axis=0)
        joint = batched.predict(batch)
        left = solo[0].predict(streams[0][t : t + 1])
        right = solo[1].predict(streams[1][t : t + 1])
        max_abs = max(
            max_abs,
            float(np.abs(joint[0] - left[0]).max()),
            float(np.abs(joint[1] - right[0]).max()),
        )
    assert max_abs <= 1e-6, f"batch isolation broken: max abs {max_abs}"
