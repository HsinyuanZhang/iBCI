"""Static contract for the bounded PACD V1 paired source smoke.

This module is intentionally stdlib-only so the public CLI can describe the
plan without importing Torch, opening data, reserving a root, or touching a
GPU.
"""

from __future__ import annotations

from dataclasses import dataclass


CELL = "PAIRED_ANCHORED_CALIBRATION_DROPOUT_V1"
SCHEMA = "paired_anchored_calibration_dropout_v1"
SEED = 42
FULL_M = 30
SMOKE_STEPS = 8
TRAIN_BATCH_SIZE = 32
EPOCHS_FOR_LR = 48
SOURCE_SESSIONS_REQUIRED = 27

RESULT_ROOT_RELATIVE = (
    "tfpd_exploration/results/paired_anchored_calibration_dropout_v1/smoke_seed42"
)
WORK_ORDER_RELATIVE = "tfpd_exploration/docs/WORKORDER_PACD_V1_20260831.md"
DESIGN_RELATIVE = (
    "tfpd_exploration/docs/DESIGN_PAIRED_ANCHORED_CALIBRATION_DROPOUT_20260830.md"
)
INITIAL_STATE_RELATIVE = (
    "tfpd_exploration/results/admission_arms_v1/canonical_initial_state.pt"
)
SEALED_TERMINAL_RELATIVE = (
    "tfpd_exploration/results/pop_robust_v1/cellD_2heads_dynamic_dropout/"
    "terminal_receipt.json"
)
SEALED_SWA_RELATIVE = (
    "tfpd_exploration/results/pop_robust_v1/cellD_2heads_dynamic_dropout/"
    "swa_final4.pt"
)

# Literal predecessor bytes are bound before their payloads are ever loaded.
# These are the sealed receipt sidecar values, copied here rather than inferred
# from a mutable file at launch.
EXPECTED_SEALED_SHA256 = {
    SEALED_TERMINAL_RELATIVE: "b3431db41efee937e83245e010ffaba50e50c517b6679d1d5285c889edbb7442",
    SEALED_SWA_RELATIVE: "626f65d80fd9f4305605132175c7ea43bc0c40d6ef6203ef1830b4b2e77f33bd",
    INITIAL_STATE_RELATIVE: "b0a340fe4d09eac1f2b498658d39a8a87e87f52304cad2539753ae9e040fcbd4",
}

EXPECTED_GPU_PHYSICAL_INDEX = "0"
EXPECTED_GPU_UUID = "GPU-ac7388a5-2e98-300a-fdb3-0b67bfd494d9"
EXPECTED_CUDA_VISIBLE_DEVICES = "0"

OWNED_IMPLEMENTATION_PATHS = (
    "tfpd_exploration/src/paired_anchored_calibration_dropout_v1/__init__.py",
    "tfpd_exploration/src/paired_anchored_calibration_dropout_v1/plan.py",
    "tfpd_exploration/src/paired_anchored_calibration_dropout_v1/core.py",
    "tfpd_exploration/src/paired_anchored_calibration_dropout_v1/smoke.py",
    "tfpd_exploration/scripts/run_pacd_smoke_v1.py",
    "tfpd_exploration/tests/test_paired_anchored_calibration_dropout_v1.py",
)

# The work order and design are reviewer-owned reference bytes.  The work
# order is part of the execution closure; the design is attempt-only review
# evidence.  This implementation package never edits either document.
BOUND_REFERENCE_PATHS = (WORK_ORDER_RELATIVE, DESIGN_RELATIVE)

REUSED_IMPLEMENTATION_PATHS = (
    "tfpd_exploration/src/cal_aug_v1/receipts.py",
    "tfpd_exploration/src/cal_aug_v1/schedule.py",
    "tfpd_exploration/src/cal_aug_v1/plan.py",
    "tfpd_exploration/src/tfpd_lane/pop_robust.py",
    "tfpd_exploration/src/tfpd_lane/arm_common.py",
    "tfpd_exploration/src/tfpd_lane/matched_scorer.py",
    "tfpd_exploration/src/tfpd_lane/receipt.py",
    "tfpd_exploration/scripts/run_pop_robust_cell.py",
    "tfpd_exploration/scripts/run_admission_arm.py",
)

IMPLEMENTATION_PATHS = OWNED_IMPLEMENTATION_PATHS + REUSED_IMPLEMENTATION_PATHS

# These are repository-root-relative *explicit* paths.  The run closure is
# re-computed immediately before immutable attempt publication and immediately
# before terminal/failure publication.  It intentionally excludes the design
# and tests: review-only edits must never kill an in-flight source smoke.
# It includes every executed dependency inherited through
# ``run_admission_arm``'s BOUND_PATTERNS and AUTHORITY_PATTERNS.
BOUND_PATTERNS = (
    "tfpd_exploration/src/__init__.py",
    "tfpd_exploration/src/paired_anchored_calibration_dropout_v1/__init__.py",
    "tfpd_exploration/src/paired_anchored_calibration_dropout_v1/plan.py",
    "tfpd_exploration/src/paired_anchored_calibration_dropout_v1/core.py",
    "tfpd_exploration/src/paired_anchored_calibration_dropout_v1/smoke.py",
    "tfpd_exploration/scripts/run_pacd_smoke_v1.py",
    "tfpd_exploration/docs/WORKORDER_PACD_V1_20260831.md",
    "tfpd_exploration/src/cal_aug_v1/receipts.py",
    "tfpd_exploration/src/cal_aug_v1/plan.py",
    "tfpd_exploration/src/cal_aug_v1/__init__.py",
    "tfpd_exploration/src/tfpd_lane/pop_robust.py",
    "tfpd_exploration/src/tfpd_lane/arm_common.py",
    "tfpd_exploration/src/tfpd_lane/matched_scorer.py",
    "tfpd_exploration/src/tfpd_lane/receipt.py",
    "tfpd_exploration/scripts/run_pop_robust_cell.py",
    "tfpd_exploration/scripts/run_admission_arm.py",
    "tfpd_exploration/scripts/make_canonical_initial_state.py",
    "tfpd_exploration/src/tfpd/spintshape_module.py",
    "sua_exploration/mc_maze/multisession_datamodule.py",
    "sua_exploration/mc_maze/unit_side_features.py",
    "sua_exploration/mc_maze/a2_matched_subject_shift_v2_core.py",
    "sua_exploration/mc_maze/datamodule.py",
    "sua_exploration/mc_maze/__init__.py",
    "streaming_calibration_exp/src/models/__init__.py",
    "streaming_calibration_exp/src/models/components/__init__.py",
    "streaming_calibration_exp/src/models/components/spint.py",
    "streaming_calibration_exp/src/models/components/streaming_spint.py",
    "streaming_calibration_exp/src/models/components/streaming_encoders.py",
    "streaming_calibration_exp/src/models/components/rt_ld_gain.py",
)

# Design/test bytes remain in the attempt as review evidence only.  They are
# intentionally absent from ``BOUND_PATTERNS`` and therefore cannot cause a
# launch-to-terminal closure mismatch.
REVIEW_EVIDENCE_PATHS = (
    DESIGN_RELATIVE,
    "tfpd_exploration/tests/test_paired_anchored_calibration_dropout_v1.py",
)


@dataclass(frozen=True)
class ArmSpec:
    name: str
    short_m: int
    role: str

    def __post_init__(self) -> None:
        if self.name not in {"p0", "p1", "p2"}:
            raise ValueError(f"unknown PACD arm: {self.name!r}")
        if self.short_m not in {4, 10, 30}:
            raise ValueError(f"invalid PACD short prefix: {self.short_m}")

    def payload(self) -> dict[str, object]:
        return {
            "name": self.name,
            "anchor_m": FULL_M,
            "short_m": self.short_m,
            "loss_weights": {"anchor": 0.5, "short": 0.5},
            "role": self.role,
        }


ARM_SPECS = (
    ArmSpec("p0", 30, "full/full matched compute and RNG control"),
    ArmSpec("p1", 4, "primary M4 paired calibration-dropout treatment"),
    ArmSpec("p2", 10, "companion M10 paired calibration-dropout treatment"),
)


def dry_payload() -> dict[str, object]:
    return {
        "cell": CELL,
        "schema": SCHEMA,
        "kind": "bounded_source_smoke_only",
        "arms": [arm.payload() for arm in ARM_SPECS],
        "seed": SEED,
        "steps_per_arm": SMOKE_STEPS,
        "batch_size": TRAIN_BATCH_SIZE,
        "device": {
            "cuda_visible_devices": EXPECTED_CUDA_VISIBLE_DEVICES,
            "logical": "cuda:0",
            "physical_index": EXPECTED_GPU_PHYSICAL_INDEX,
            "uuid": EXPECTED_GPU_UUID,
        },
        "inference_graph_changed": False,
        "new_trainable_parameters": 0,
        "target_access": False,
        "full_training_authorized": False,
        "result_root": RESULT_ROOT_RELATIVE,
        "work_order": WORK_ORDER_RELATIVE,
        "no_torch_import": True,
    }
