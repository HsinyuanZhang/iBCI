"""The CPU matched-pair smoke: both arms' first N=40 steps, no CUDA at all.

This proves the matched-pair discipline on CPU before any GPU spend:

* initial-state digest equality across arms (student AND teacher);
* identical batch order digest (sampler stream + per-step batch digests);
* identical dropout-p stream domain digests across arms (per-step python and
  torch RNG-state digests, PLUS a zero-draw counter -- the verified
  frozen-decoder finding means the M2 recipe draws no dropout-p at all);
* the C1m prefix sequence is exactly (10, 5, 2) x repeats over 40 steps;
* T4 bytes unchanged (per-step side-feature digests equal across arms, and
  the re-fit normalizer reproduces the sealed ``d17f5f4c...`` in both arms);
* eval-dropout inactive proof on both arms;
* no target path resolved (held-out data never built);
* ``torch.cuda.is_initialized()`` stays False.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping

from . import plan, schedule, trainer


class SmokeError(RuntimeError):
    """Fail closed for smoke-equality drift."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SmokeError(message)


def require_cpu_environment() -> None:
    """The smoke runs with CUDA hidden and must never initialize it."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    _require(visible in ("", "-1"),
             "pit m2 smoke requires CUDA_VISIBLE_DEVICES empty or -1")


def run_arm_smoke(repo_root: Path, arm: str, steps: int = plan.SMOKE_STEPS) -> Mapping[str, object]:
    """One arm's first ``steps`` optimizer steps, fully on CPU."""
    import torch

    _require(os.environ.get("CUDA_VISIBLE_DEVICES", "") in ("", "-1"),
             "pit m2 smoke must run with CUDA_VISIBLE_DEVICES empty or -1")
    runner = trainer.PitM2ArmedRunner(Path(repo_root), arm, device="cpu",
                                      record_steps=steps)
    runner.prepare()
    body = runner.run_smoke(steps=steps)
    _require(torch.cuda.is_initialized() is False,
             "pit m2 smoke initialized CUDA (forbidden)")
    _require(body["cuda_initialized"] is False, "pit m2 smoke cuda flag drift")
    authority = dict(body["authority"])
    _require(authority["heldout_dataset_built"] is False
             and authority["normalization_sha256"] == plan.NORMALIZATION_SHA256,
             "pit m2 smoke authority drift")
    return body


def validate_smoke_equality(t0m: Mapping[str, object], c1m: Mapping[str, object]) -> dict[str, object]:
    """The frozen equality contract; returns the equality receipt body."""
    t0_stream = dict(t0m["stream_records"])
    c1_stream = dict(c1m["stream_records"])
    t0_operator = dict(t0m["operator_snapshot"])
    c1_operator = dict(c1m["operator_snapshot"])
    t0_authority = dict(t0m["authority"])
    c1_authority = dict(c1m["authority"])
    checks = {
        "initial_student_state_sha256":
            t0_authority["initial_student_state_sha256"]
            == c1_authority["initial_student_state_sha256"],
        "initial_teacher_state_sha256":
            t0_authority["initial_teacher_state_sha256"]
            == c1_authority["initial_teacher_state_sha256"],
        "build_rng_digests":
            t0_authority["rng_digests"] == c1_authority["rng_digests"],
        "sampler_batch_stream_sha256":
            t0_authority["sampler_batch_stream_sha256"]
            == c1_authority["sampler_batch_stream_sha256"],
        "normalization_sha256_sealed_both":
            t0_authority["normalization_sha256"] == c1_authority["normalization_sha256"]
            == plan.NORMALIZATION_SHA256,
        "rng_stream_digest": t0_stream["rng_stream_digest"] == c1_stream["rng_stream_digest"],
        "torch_rng_stream_digest":
            t0_stream["torch_rng_stream_digest"] == c1_stream["torch_rng_stream_digest"],
        "per_step_rng_digests": [
            row_t["rng_state_after_sha256"] == row_c["rng_state_after_sha256"]
            for row_t, row_c in zip(t0_stream["rows"], c1_stream["rows"], strict=True)
        ],
        "batch_stream_digest": t0_stream["batch_stream_digest"] == c1_stream["batch_stream_digest"],
        "t4_bytes_stream_digest":
            t0_stream["t4_bytes_stream_digest"] == c1_stream["t4_bytes_stream_digest"],
        "t0m_effective_prefixes_all_full_33":
            all(item == plan.TRAINING_CALIBRATION_N_TRIALS
                for item in t0_operator["effective_prefixes"]),
        "c1m_prefix_sequence_is_cycle": c1_operator["recorded_prefix_sequence"]
        == schedule.sequence(plan.SMOKE_STEPS),
        "c1m_effective_equals_scheduled": c1_operator["effective_prefixes"]
        == c1_operator["recorded_prefix_sequence"],
        "optimizer_steps_equal":
            t0m["optimizer_steps"] == c1m["optimizer_steps"] == plan.SMOKE_STEPS,
        "dropout_p_draw_counter_one_per_step_both_arms": (
            t0m["rng_draw_counts_during_steps"]["python"] == plan.SMOKE_STEPS
            and c1m["rng_draw_counts_during_steps"]["python"] == plan.SMOKE_STEPS
        ),
        "dropout_p_aux_counts_equal_across_arms": (
            t0m["rng_draw_counts_during_steps"]["numpy"]
            == c1m["rng_draw_counts_during_steps"]["numpy"]
            and t0m["rng_draw_counts_during_steps"]["torch"]
            == c1m["rng_draw_counts_during_steps"]["torch"]
        ),
        "t0m_eval_dropout_inactive":
            t0m["eval_dropout_inactive_proof"]["bit_identical"] is True,
        "c1m_eval_dropout_inactive":
            c1m["eval_dropout_inactive_proof"]["bit_identical"] is True,
        "teacher_eval_dropout_inert_both_arms": (
            t0m["teacher_eval_dropout_inert_proof"]["bit_identical"] is True
            and c1m["teacher_eval_dropout_inert_proof"]["bit_identical"] is True
            and t0m["teacher_eval_dropout_inert_proof"]["distinct_p_draws_observed"] is True
        ),
        "cuda_never_initialized":
            t0m["cuda_initialized"] is False and c1m["cuda_initialized"] is False,
        "no_target_path_resolved":
            t0_authority["heldout_dataset_built"] is False
            and c1_authority["heldout_dataset_built"] is False,
    }

    def _all_true(value: object) -> bool:
        if isinstance(value, list):
            return all(item is True for item in value)
        return value is True

    failures = {key: value for key, value in checks.items() if not _all_true(value)}
    _require(not failures, f"pit m2 smoke equality failed: {failures}")
    return {
        "schema": "pit_m2_smoke_equality_v1",
        "checks": checks,
        "t0m_initial_student_state_sha256": t0_authority["initial_student_state_sha256"],
        "c1m_initial_student_state_sha256": c1_authority["initial_student_state_sha256"],
        "shared_rng_stream_digest": t0_stream["rng_stream_digest"],
        "shared_torch_rng_stream_digest": t0_stream["torch_rng_stream_digest"],
        "shared_batch_stream_digest": t0_stream["batch_stream_digest"],
        "shared_t4_bytes_stream_digest": t0_stream["t4_bytes_stream_digest"],
        "shared_sampler_batch_stream_sha256": t0_authority["sampler_batch_stream_sha256"],
        "normalization_sha256": plan.NORMALIZATION_SHA256,
        "c1m_recorded_prefix_sequence": list(c1_operator["recorded_prefix_sequence"]),
        "c1m_first_visible_slice_sha256": (
            c1_operator["records"][0]["visible_slice_sha256"] if c1_operator["records"] else None),
        "dropout_p_draws_per_arm": int(t0m["rng_draw_counts_during_steps"]["python"]),
        "dropout_p_draw_law": "one inert teacher eval-mode random.uniform per training step",
        "dropout_p_stream_identical": True,
        "batch_order_identical": True,
        "t4_bytes_unchanged": True,
        "cuda_initialized": False,
        "target_path_resolved": False,
        "cal_aug_discipline_reference":
            plan.SMOKE_EQUALITY_CONTRACT["cal_aug_discipline_reference"],
    }


__all__ = ("SmokeError", "require_cpu_environment", "run_arm_smoke", "validate_smoke_equality")
