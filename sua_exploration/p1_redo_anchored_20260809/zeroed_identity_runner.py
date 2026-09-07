#!/usr/bin/env python3
"""Run streaming_calibration_exp/src/train.py test-only, with the identity
token forced to exactly zero immediately before it enters the neural window.

Used for both the M2 and M1 same-checkpoint identity-zeroing diagnostics
(P1-REDO-ANCHORED, 2026-08-09). Both tasks route through
``StreamingCalibrationLitModule.student``, an instance of
``StreamingSpintModel`` (streaming_calibration_exp/src/models/components/streaming_spint.py).

Zeroing point (verified by reading the file directly, not inferred):
``StreamingSpintModel.decode_with_identity`` combines the identity token into
the neural window at the line

    src = src + identity                      # streaming_spint.py:276

(inside ``if self.fixed_slot_router is None:``, which is the live path for
the B3S variant used by both M2/T4 and M1/T4 checkpoints here -- neither
config sets a nonzero ``fixed_slot_count``). ``identity`` itself is produced
by ``StreamingSpintModel.compute_identity``/``forward`` just before this
call (lines ~251-259 and ~632-669).

This script monkeypatches ``StreamingSpintModel.decode_with_identity`` at
runtime (no file on disk is modified) so that the ``identity`` argument is
replaced with ``torch.zeros_like(identity)`` immediately on entry, then calls
the *original* unmodified implementation. No other line of behavior changes:
same checkpoint, same weights, same dataset, forward-only, no gradient.

Usage: identical CLI overrides to ``streaming_calibration_exp/src/train.py``,
e.g.:

    python zeroed_identity_runner.py experiment=... ckpt_path=... train=false test=true ...

Must be invoked with cwd=streaming_calibration_exp (matches the convention
used by every sealed run this diagnostic anchors against), and should be run
with CUDA_VISIBLE_DEVICES="" to force CPU (the LightningModule's teacher
checkpoint loader does not accept a map_location override and otherwise
always tries the checkpoint's original CUDA device).
"""
from __future__ import annotations

import sys
from pathlib import Path

STREAMING_ROOT = Path("/home/xinyuan/Work_host/SPINT/streaming_calibration_exp").resolve()
if str(STREAMING_ROOT) not in sys.path:
    sys.path.insert(0, str(STREAMING_ROOT))

import torch  # noqa: E402

from src.models.components import streaming_spint  # noqa: E402

_ORIGINAL_DECODE_WITH_IDENTITY = streaming_spint.StreamingSpintModel.decode_with_identity
_CALL_COUNT = {"n": 0}


def _zeroed_decode_with_identity(self, neural, identity, neuron_gate=None):
    _CALL_COUNT["n"] += 1
    if not torch.is_tensor(identity):
        raise TypeError("identity must be a tensor to zero it")
    zeroed = torch.zeros_like(identity)
    if not torch.equal(zeroed, torch.zeros_like(identity)):
        raise RuntimeError("zeroing failed")
    return _ORIGINAL_DECODE_WITH_IDENTITY(self, neural, zeroed, neuron_gate=neuron_gate)


streaming_spint.StreamingSpintModel.decode_with_identity = _zeroed_decode_with_identity


def main() -> None:
    from src.train import main as hydra_main

    hydra_main()
    print(f"[zeroed_identity_runner] decode_with_identity called {_CALL_COUNT['n']} time(s) with identity forced to zero", file=sys.stderr)


if __name__ == "__main__":
    main()
