"""Static CAL-AUG V1 contract: every free parameter frozen in module bytes.

Everything that could bias the matched comparison or the readouts is frozen
HERE, in bytes the attempt receipt pins by SHA-256 *before* any data or model
access (work order section 7; guidance section 13):

* the pinned SHA-256 literals of the two sealed trainers this lane
  importlib-loads, the binding work order and the immutable sealed
  predecessors (guidance section 2.1 table);
* the arm names, the default prefix cycle, the readout prefix lengths;
* the frozen matched-training budget (seed 42, 27 source sessions, batch 32,
  33,925 steps/epoch x 48 epochs, Adam 1e-4 + warmup/cosine, final-four SWA);
* the device binding, throughput-probe geometry and hard wall-clock timeout;
* the mechanism (section 5) and deployment (section 6) gate margins, breadth
  denominators and disposition strings, with the predeclared boundary rule.

Boundary rule (mirrors ``ac3_utility_bridge_v1.plan.GATES``): every margin is
the EXACT ``>=`` comparison on the float64 equal-session mean, so a value
exactly at the margin passes and a value 1e-13 below it fails.  The program
epsilon ``GATE_BOUNDARY_EPSILON`` is a disclosed boundary BAND recorded as
``within_epsilon_band_of_boundary`` for operator review; it never widens a
margin and never flips a verdict.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

CELL = "CAL_AUG_MATCHED_T0_C1_B3S_PREFIX_V1"
SCHEMA = "cal_aug_v1"

WORK_ORDER_RELATIVE = "tfpd_exploration/docs/WORKORDER_CAL_AUG_V1_20260829.md"
GUIDANCE_RELATIVE = (
    "tfpd_exploration/docs/EXECUTION_GUIDANCE_AFTER_AC3_AND_CONTINUITY_AUDIT_20260829.md"
)
RESULT_ROOT_RELATIVE = "tfpd_exploration/results/cal_aug_v1"

TFPD_ROOT_RELATIVE = "tfpd_exploration"

# ---------------------------------------------------------------------------
# Sealed boundary: pinned literals (guidance section 2.1), verified fail-closed
# at import time of the successor runners, BEFORE any data or model access.
# ---------------------------------------------------------------------------

#: ``relative to the REPOSITORY root`` -> pinned SHA-256.
PINNED_SHA256 = {
    "tfpd_exploration/scripts/run_pop_robust_cell.py": (
        "761cc5306e7d1c1e4684771edeae36331f535f13d8cb07cf56db711fa6b7190b"
    ),
    "tfpd_exploration/scripts/run_admission_arm.py": (
        "a1073e6271771cb49322a4dac3ae5b6164117683b56069c6441b305c215fb5ad"
    ),
    "tfpd_exploration/docs/WORKORDER_CAL_AUG_V1_20260829.md": (
        "8ac4d550548f327675d737c4963a6fe04c97e2a2764a265734bb56e8dc0fa555"
    ),
    # the frozen lane helpers the sealed runner reuses (its own BOUND_PATTERNS)
    "tfpd_exploration/src/tfpd_lane/pop_robust.py": (
        "a211a7c982c0ed417bc78bf45509b38412715cfbf3e5d73cb5e4fe4282ec3021"
    ),
    "tfpd_exploration/src/tfpd_lane/arm_common.py": (
        "9df0af9ee238b41ce290624aad071da626e68302f1940c05d07d6481caa3c2e8"
    ),
    "tfpd_exploration/src/tfpd_lane/matched_scorer.py": (
        "797204fec20e3545f782c7fb0d62441d8ead79dc5ded0f0b347f22233dba9aed"
    ),
    "tfpd_exploration/src/tfpd_lane/receipt.py": (
        "8eac2e9214229149f2fd76567e788c166bcfb385daf17f5d90198a972d7a7e94"
    ),
}

#: Immutable sealed predecessors (guidance section 2.1), each sidecar-pinned.
SEALED_PREDECESSOR_FILES = (
    "tfpd_exploration/results/pop_robust_v1/cellD_2heads_dynamic_dropout/"
    "terminal_receipt.json",
    "tfpd_exploration/results/pop_robust_v1/cellD_2heads_dynamic_dropout/"
    "swa_final4.pt",
    "tfpd_exploration/results/admission_arms_v1/canonical_initial_state.pt",
)
SEALED_PREDECESSOR_SHA256 = {
    SEALED_PREDECESSOR_FILES[0]: (
        "b3431db41efee937e83245e010ffaba50e50c517b6679d1d5285c889edbb7442"
    ),
    SEALED_PREDECESSOR_FILES[1]: (
        "626f65d80fd9f4305605132175c7ea43bc0c40d6ef6203ef1830b4b2e77f33bd"
    ),
    SEALED_PREDECESSOR_FILES[2]: (
        "b0a340fe4d09eac1f2b498658d39a8a87e87f52304cad2539753ae9e040fcbd4"
    ),
}

#: Source-closure glob patterns for this lane's own bytes plus the sealed
#: runner surface it reuses (relative to the tfpd_exploration root).
BOUND_PATTERNS = (
    "src/cal_aug_v1/*.py",
    "scripts/run_cal_aug_cell_v1.py",
    "scripts/run_cal_aug_smoke_v1.py",
    "scripts/run_cal_aug_mechanism_v1.py",
    "scripts/run_cal_aug_deployment_v1.py",
    "src/tfpd_lane/pop_robust.py",
    "src/tfpd_lane/arm_common.py",
    "src/tfpd_lane/matched_scorer.py",
    "src/tfpd_lane/receipt.py",
    "scripts/run_pop_robust_cell.py",
    "scripts/run_admission_arm.py",
)

# ---------------------------------------------------------------------------
# Arms, operator schedule, readouts.
# ---------------------------------------------------------------------------

ARMS = ("t0", "c1")
ARM_ROLES = {
    "t0": "matched control: SAME successor runner, operator disabled (hook registered, schedule returns the input unchanged)",
    "c1": "deterministic chronological B3S prefix-length cycle on the training forward",
}
DEFAULT_CYCLE_TEXT = "30,10,4"
DEFAULT_CYCLE = (30, 10, 4)
#: Operator domain: chronological prefix of the 30-trial B3S calibration block.
MAX_CALIBRATION_TRIALS = 30
#: Mechanism readout prefixes (work order section 5): the base M30 plus the
#: shortened prefixes whose degradation is measured.
READOUT_PREFIXES = (30, 10, 4)
DEGRADED_PREFIXES = (10, 4)
#: Per-step operator recording budget (first N training forwards).
RECORD_STEPS_DEFAULT = 40  # superseded 2026-08-29: was 200; see the note at PROBE_WARMUP_STEPS

# ---------------------------------------------------------------------------
# Frozen matched-training budget (work order section 2; guidance section 6.3).
# ---------------------------------------------------------------------------

SEED = 42
SOURCE_SESSIONS_REQUIRED = 27
TRAIN_BATCH_SIZE = 32
STEPS_PER_EPOCH = 33_925
EPOCHS = 48
TOTAL_OPTIMIZER_STEPS = EPOCHS * STEPS_PER_EPOCH  # 1,628,400
EXPECTED_MANIFEST_SHA256 = "4607e979c6c2ff451c147a8d9878fe1080b9d3e9bbc7304b559616eb2a13a0c9"
BEHAVIOR_NORMALIZER_SEMANTIC_PREFIX = "f062506c"

# ---------------------------------------------------------------------------
# Device binding, throughput probe, hard timeout (work order section 4).
# ---------------------------------------------------------------------------

#: Same physical GPU for both arms, serially (guidance section 6.4).
BOUND_GPU_UUID = "GPU-ac7388a5-2e98-300a-fdb3-0b67bfd494d9"
BOUND_GPU_VISIBILITY = "CUDA_VISIBLE_DEVICES=0"
#: In-run throughput probe: measured over optimizer steps [20, 20+N).
#: Amendment (2026-08-29, after the first probe reading): the probe window must
#: sit OUTSIDE the operator's digest-recording window (record_steps), whose
#: per-step sha256 over the ~34 MB calib block costs ~50-150 ms and measured
#: 16.1 steps/s against the sealed 87.99 steps/s anchor.  Recording now covers
#: exactly the smoke's binding steps (40) and the probe starts after it.
RECORD_STEPS_DEFAULT = 40
PROBE_WARMUP_STEPS = 250
PROBE_DEFAULT_STEPS = 100
#: Sealed anchor: 18,512 s for 1,628,400 steps (87.99 steps/s, RTX 3090).
SEALED_ANCHOR_SECONDS_PER_RUN = 18_512.0
SEALED_ANCHOR_STEPS_PER_SECOND = TOTAL_OPTIMIZER_STEPS / SEALED_ANCHOR_SECONDS_PER_RUN
#: Hard wall-clock timeout per arm (work order section 4: 8 h), with an atomic
#: CELL_FAILED receipt on breach.
HARD_TIMEOUT_SECONDS = 8 * 3600
#: Planning ceiling for the two training arms; exceeding needs re-authorization.
PLANNING_CEILING_GPU_HOURS = 12.0

# ---------------------------------------------------------------------------
# Gates — mechanism registration (work order section 5; guidance section 6.6).
# ---------------------------------------------------------------------------

GATE_BOUNDARY_EPSILON = 1.0e-12
MECHANISM_RECOVERY_MARGIN_R2 = 0.01
MECHANISM_POSITIVE_SOURCE_SESSIONS_REQUIRED = 18
MECHANISM_M30_SAFETY_MARGIN_R2 = -0.01

MECHANISM_REGISTERED = "SOURCE_MECHANISM_REGISTERED"
MECHANISM_NULL = "SOURCE_MECHANISM_NOT_REGISTERED"

# ---------------------------------------------------------------------------
# Gates — deployment (work order section 6; guidance section 6.7).
# ---------------------------------------------------------------------------

BUDGETS = (4, 10, 30)
EXTERNAL_SESSIONS = 15
WITHIN_SESSIONS = 6
EXTERNAL_POSITIVE_REQUIRED = 10
LOWER_CONTINUATION_DELTA_R2 = 0.015
PRIMARY_DELTA_R2 = 0.03
EXTERNAL_M30_SAFETY_MARGIN_R2 = -0.02
WITHIN_SAFETY_MARGIN_R2 = -0.02
BOOTSTRAP_SEED = 42

LOWER_CONTINUATION_PASSED = "EXTERNAL_CONTINUATION_GATE_PASSED"
LOWER_CONTINUATION_FAILED = "EXTERNAL_CONTINUATION_GATE_FAILED"
PRIMARY_CLAIM_PASSED = "PRIMARY_PERFORMANCE_CLAIM_PASSED"
PRIMARY_CLAIM_FAILED = "PRIMARY_PERFORMANCE_CLAIM_NOT_MET"
MECHANISM_POSITIVE_DEPLOYMENT_INCONCLUSIVE = (
    "MECHANISM_POSITIVE__DEPLOYMENT_INCONCLUSIVE"
)

GATE_SPEC = {
    "mechanism_registration": {
        "expression": (
            ">=1 of M4/M10 prefix_robustness_recovery >= +0.01 R2 AND positive "
            "source sessions >= 18/27 AND C1-T0 at M30 >= -0.01"
        ),
        "scope": "source-roster mechanism evidence, never a deployment claim",
    },
    "lower_continuation": {
        "expression": (
            ">=1 of external M4/M10: mean delta >= +0.015, >= 10/15 positive, "
            "fixed-bootstrap lower bound >= 0; the other low budget: mean delta "
            ">= 0; external M30: mean delta >= -0.02; within at every budget: "
            "mean delta >= -0.02"
        ),
        "authority": "authorizes only the next predeclared single-axis study",
    },
    "primary_claim": {
        "expression": (
            ">=1 of external M4/M10: mean delta >= +0.03 and >= 10/15 positive, "
            "with the other-low-budget >= 0 clause and all M30/within safety "
            "conditions of the lower gate"
        ),
    },
    "never_average_budgets": (
        "budgets are never averaged to hide an M30 regression; every safety "
        "clause is evaluated per budget"
    ),
    "boundary_rule": (
        "exact >= on the float64 equal-session mean; a miss inside the 1e-12 "
        "program epsilon of the boundary is recorded in "
        "within_epsilon_band_of_boundary and never flips the verdict"
    ),
}

# ---------------------------------------------------------------------------
# Result roots and owned bytes.
# ---------------------------------------------------------------------------

RESULT_SUBROOTS = {
    "smoke": "smoke",
    "probe": "probe",
    "t0": "t0",
    "c1": "c1",
    "mechanism": "mechanism",
    "deployment": "deployment",
}

OWNED_PATHS = (
    WORK_ORDER_RELATIVE,
    "tfpd_exploration/src/cal_aug_v1/__init__.py",
    "tfpd_exploration/src/cal_aug_v1/plan.py",
    "tfpd_exploration/src/cal_aug_v1/schedule.py",
    "tfpd_exploration/src/cal_aug_v1/hook.py",
    "tfpd_exploration/src/cal_aug_v1/receipts.py",
    "tfpd_exploration/src/cal_aug_v1/mechanism.py",
    "tfpd_exploration/src/cal_aug_v1/deployment.py",
    "tfpd_exploration/scripts/run_cal_aug_cell_v1.py",
    "tfpd_exploration/scripts/run_cal_aug_smoke_v1.py",
    "tfpd_exploration/scripts/run_cal_aug_mechanism_v1.py",
    "tfpd_exploration/scripts/run_cal_aug_deployment_v1.py",
    "tfpd_exploration/tests/test_cal_aug_v1.py",
)


def owned_sha256s(repo_root: Path) -> dict[str, str]:
    base = Path(repo_root).absolute()
    return {
        relative: hashlib.sha256((base / relative).read_bytes()).hexdigest()
        for relative in OWNED_PATHS
    }


def gate_spec_payload() -> dict[str, object]:
    return {
        "mechanism_registration": dict(GATE_SPEC["mechanism_registration"]),
        "lower_continuation": dict(GATE_SPEC["lower_continuation"]),
        "primary_claim": dict(GATE_SPEC["primary_claim"]),
        "never_average_budgets": GATE_SPEC["never_average_budgets"],
        "boundary_rule": GATE_SPEC["boundary_rule"],
        "boundary_epsilon": GATE_BOUNDARY_EPSILON,
        "margins": {
            "mechanism_recovery_r2": MECHANISM_RECOVERY_MARGIN_R2,
            "mechanism_positive_source_sessions": MECHANISM_POSITIVE_SOURCE_SESSIONS_REQUIRED,
            "mechanism_m30_safety_r2": MECHANISM_M30_SAFETY_MARGIN_R2,
            "lower_continuation_delta_r2": LOWER_CONTINUATION_DELTA_R2,
            "primary_delta_r2": PRIMARY_DELTA_R2,
            "external_positive_required": EXTERNAL_POSITIVE_REQUIRED,
            "external_m30_safety_r2": EXTERNAL_M30_SAFETY_MARGIN_R2,
            "within_safety_r2": WITHIN_SAFETY_MARGIN_R2,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
    }
