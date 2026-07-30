"""E1: post-hoc SWA (stochastic weight averaging) checkpoint construction for DANDI 000688
streaming-calibration runs trained with ``train_variant_dandi688.py --checkpoint_every_epoch``.

ROADMAP.md "当前实验计划（2026-07-26 起）" E1: this attacks sigma_seed (the dominant
variance component measured in side_feature_ablation_v2, CURRENT_RESULTS.md section I) by
averaging trained *weights* across a trailing window of per-epoch checkpoints from the same
run, producing one merged model that is then evaluated exactly once with the existing
gradient-free protocol -- unlike M3, which averages *measurements* (protocol scores) and
only reduces the epoch-noise component, never sigma_seed.

This module only builds and (optionally) evaluates checkpoints; the R2 computation itself is
never reimplemented here -- callers pass the merged checkpoint's path to
``select_gradient_free_protocol_dandi688.evaluate_fixed_protocol_over_validation_sessions``,
the same function every other DANDI 000688 eval script in this repo uses.

Implementation cautions (verified empirically against this run family, see
sua_exploration/docs/CONVERGENCE_AND_SWA.md):
  - Only floating-point tensors are arithmetic-averaged.
  - Tensors under the "teacher." prefix are never averaged -- the teacher is frozen and must
    not be modified; they are copied verbatim from one checkpoint after asserting they are
    bit-identical across the whole averaging window (i.e. actually frozen, not silently
    drifting).
  - Any non-floating-point tensor (integer/bool buffers, step counters, etc.) is copied
    verbatim, never averaged, IF one ever appears -- none currently exist in the B3/EarlyPool
    encoder + SpintModel decoder/teacher path (grep for register_buffer/BatchNorm across
    streaming_calibration_exp/src turned up no BatchNorm anywhere in the repo and no
    persistent integer buffers reachable from variant B3; the only persistent buffers
    (float32 "projection"/"hash_matrix") belong to unrelated encoder variants B9/B12/B13,
    never instantiated for B3). LayerNorm (used inside the SpintModel decoder/teacher
    transformer) carries only learnable weight/bias floats and no running statistics, so
    unlike BatchNorm it requires no post-averaging recomputation pass.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Mapping, Sequence

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_epoch_window_dandi688 import epoch_checkpoint_path  # noqa: E402  (reused, not reimplemented)

TEACHER_PREFIX = "teacher."


def average_student_state_dicts(
    state_dicts: Sequence[Mapping[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Merge >=1 per-epoch state_dicts from the SAME run into one SWA state_dict.

    - Keys under ``TEACHER_PREFIX`` are copied verbatim from the first state_dict, after
      asserting they are bit-identical across every input (never averaged, never silently
      accepted if the "frozen" assumption turns out false).
    - Floating-point tensors elsewhere are arithmetic-averaged (stacked and mean'd in
      float64 for numerical stability, then cast back to the original dtype).
    - Non-floating-point tensors elsewhere are copied verbatim from the first state_dict,
      after asserting they agree across every input (cannot be meaningfully averaged; a
      disagreement would mean these checkpoints are not from one deterministic run).

    Raises ValueError on any key-set, shape, or dtype mismatch, or on a "must be identical"
    tensor that is not -- this function never silently drops or coerces a discrepancy.
    """
    if not state_dicts:
        raise ValueError("state_dicts must be non-empty")
    reference_keys = set(state_dicts[0].keys())
    for index, sd in enumerate(state_dicts):
        if set(sd.keys()) != reference_keys:
            raise ValueError(
                f"state_dicts[{index}] key set differs from state_dicts[0]; all inputs must "
                "be checkpoints from the same run/architecture"
            )

    averaged: dict[str, torch.Tensor] = {}
    for key in reference_keys:
        tensors = [sd[key] for sd in state_dicts]
        first = tensors[0]
        for index, tensor in enumerate(tensors):
            if tensor.shape != first.shape or tensor.dtype != first.dtype:
                raise ValueError(
                    f"key {key!r}: state_dicts[{index}] has shape/dtype "
                    f"{tuple(tensor.shape)}/{tensor.dtype}, expected "
                    f"{tuple(first.shape)}/{first.dtype}"
                )

        if key.startswith(TEACHER_PREFIX):
            if not all(torch.equal(tensor, first) for tensor in tensors[1:]):
                raise ValueError(
                    f"teacher tensor {key!r} is not bit-identical across the averaging "
                    "window; the teacher must be frozen -- refusing to average or silently "
                    "pick one value"
                )
            averaged[key] = first.clone()
        elif torch.is_floating_point(first):
            stacked = torch.stack([tensor.double() for tensor in tensors], dim=0)
            averaged[key] = stacked.mean(dim=0).to(dtype=first.dtype)
        else:
            if not all(torch.equal(tensor, first) for tensor in tensors[1:]):
                raise ValueError(
                    f"non-floating-point tensor {key!r} differs across the averaging "
                    "window; integer/bool buffers cannot be meaningfully averaged and none "
                    "are expected to vary within one run"
                )
            averaged[key] = first.clone()
    return averaged


def load_epoch_state_dicts(
    run_dir: Path, epochs: Sequence[int]
) -> tuple[list[dict[str, torch.Tensor]], list[dict], list[Path]]:
    """Load the raw Lightning checkpoint dicts for ``epochs`` (1-indexed protocol epochs) of
    one run. Returns (state_dicts, full_checkpoint_dicts, checkpoint_paths)."""
    epoch_ckpt_dir = Path(run_dir) / "epoch_ckpts"
    ckpt_paths = [epoch_checkpoint_path(epoch_ckpt_dir, epoch) for epoch in epochs]
    missing = [str(path) for path in ckpt_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing checkpoint(s) for SWA window epochs={list(epochs)} in {epoch_ckpt_dir}: "
            f"{missing}"
        )
    checkpoints = [torch.load(str(path), map_location="cpu", weights_only=False) for path in ckpt_paths]
    state_dicts = [ckpt["state_dict"] for ckpt in checkpoints]
    return state_dicts, checkpoints, ckpt_paths


def build_swa_checkpoint(run_dir: Path, epochs: Sequence[int], out_path: Path) -> Path:
    """Average epoch checkpoints ``epochs`` (1-indexed protocol epochs) of ``run_dir`` into
    one SWA checkpoint saved at ``out_path``, in the same ``{"state_dict", "hyper_parameters"}``
    shape ``select_gradient_free_protocol_dandi688.load_frozen_model`` reads from any other
    checkpoint. Placed under ``out_path`` so callers control ``out_path.parent.parent`` (must
    stay == ``run_dir`` for ``evaluate_fixed_protocol_over_validation_sessions``'s side-feature
    run_metadata auto-discovery to keep resolving to this run's real run_metadata.json)."""
    if len(epochs) < 1:
        raise ValueError("epochs must be non-empty")
    state_dicts, checkpoints, ckpt_paths = load_epoch_state_dicts(run_dir, epochs)
    hyper_parameters = checkpoints[0].get("hyper_parameters", {})
    for index, ckpt in enumerate(checkpoints[1:], start=1):
        if ckpt.get("hyper_parameters", {}) != hyper_parameters:
            raise ValueError(
                f"hyper_parameters of checkpoint index {index} ({ckpt_paths[index]}) differs "
                f"from checkpoint 0 ({ckpt_paths[0]}); inputs must be one run's own epochs"
            )
    averaged_state_dict = average_student_state_dicts(state_dicts)
    merged = {
        "state_dict": averaged_state_dict,
        "hyper_parameters": hyper_parameters,
        "swa_source_run_dir": str(Path(run_dir).resolve()),
        "swa_source_checkpoints": [str(path.resolve()) for path in ckpt_paths],
        "swa_epochs": list(epochs),
        "swa_num_averaged": len(epochs),
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(merged, out_path)
    return out_path
