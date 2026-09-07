"""Frozen contract for the M1 matched T0/C1 50-epoch prefix extension.

The sealed 20-epoch pair is imported, never edited.  Every literal here is
taken from ``tfpd_exploration/docs/DESIGN_M1_T0C1_PREFIX_50EP_20260901.md``.
Unchanged predecessor values are re-exported; new values are defined once.
This module does not import Torch, does not touch the filesystem at import
time, and does not initialize CUDA.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from tfpd_exploration.src.cross_session_worst_group_v1 import source_lifecycle as v1
from tfpd_exploration.src.m1_t0c1_prefix_v1 import plan as v1_plan


class M1T0C150EpPlanError(RuntimeError):
    """Fail closed for the 50-epoch pair contract, literals, or closure drift."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise M1T0C150EpPlanError(message)


# ---------------------------------------------------------------------------
# Re-exported predecessor helpers and unchanged literals (spec §2)
# ---------------------------------------------------------------------------

canonical_json_bytes = v1_plan.canonical_json_bytes
sha256_bytes = v1_plan.sha256_bytes
require_sha = v1_plan.require_sha
safe_relative = v1_plan.safe_relative

CELL = v1_plan.CELL
WORKORDER_RELATIVE = v1_plan.WORKORDER_RELATIVE
WORKORDER_SHA256 = v1_plan.WORKORDER_SHA256
HELDOUT_FOLD_SESSIONS = v1_plan.HELDOUT_FOLD_SESSIONS
HELDIN_TRAINING_SESSIONS = v1_plan.HELDIN_TRAINING_SESSIONS
SCORE_ORDER = v1_plan.SCORE_ORDER
FOLD_TARGET_OF_SESSION = v1_plan.FOLD_TARGET_OF_SESSION
BUDGET_LABEL = v1_plan.BUDGET_LABEL
METRIC_LABEL = v1_plan.METRIC_LABEL
MODEL_SHAPE = dict(v1_plan.MODEL_SHAPE)
EVAL_BATCH_SIZE = v1_plan.EVAL_BATCH_SIZE

ARMS = v1_plan.ARMS
ARM_ROLES = dict(v1_plan.ARM_ROLES)
CYCLE = v1_plan.CYCLE
CYCLE_LAW = v1_plan.CYCLE_LAW

SEED = v1_plan.SEED
STEPS_PER_EPOCH = v1_plan.STEPS_PER_EPOCH
ADAM_LR = v1_plan.ADAM_LR
ADAM_WEIGHT_DECAY = v1_plan.ADAM_WEIGHT_DECAY
TRAIN_BATCH_SIZE = v1_plan.TRAIN_BATCH_SIZE
OBJECTIVE_LAMBDA = v1_plan.OBJECTIVE_LAMBDA
OBJECTIVE_TAU = v1_plan.OBJECTIVE_TAU
NO_SWA = v1_plan.NO_SWA
RECORD_STEPS = v1_plan.RECORD_STEPS
SMOKE_STEPS = v1_plan.SMOKE_STEPS
HARD_TIMEOUT_SECONDS_PER_ARM = v1_plan.HARD_TIMEOUT_SECONDS_PER_ARM

DERIVATIVE_SCAN_OMISSION = v1_plan.DERIVATIVE_SCAN_OMISSION
STEP_COST_PROBE = v1_plan.STEP_COST_PROBE
PHASE1_MOTIVATION = v1_plan.PHASE1_MOTIVATION
PHASE3_DEPLOYMENTS = v1_plan.PHASE3_DEPLOYMENTS
PHASE3_SURFACES = v1_plan.PHASE3_SURFACES
PHASE3_TABLE_AXIS = v1_plan.PHASE3_TABLE_AXIS
CDM_FIFO_LAW = v1_plan.CDM_FIFO_LAW
ACCEPTED_V6_GRAPH_SHA256 = v1_plan.ACCEPTED_V6_GRAPH_SHA256
SOURCE_PREPARATION_LAW = v1_plan.SOURCE_PREPARATION_LAW
BOUND_GPU = v1_plan.BOUND_GPU
PHASE1_GAP_ROOT_ANCHOR = dict(v1_plan.PHASE1_GAP_ROOT_ANCHOR)

THIS_LANE_SMOKE_STATUS = v1_plan.SEALED_SMOKE_STATUS
THIS_LANE_ARM_STATUS = v1_plan.SEALED_ARM_STATUS
THIS_LANE_PROBE_STATUS = "COMPLETE_OVERFITTING_PROBE"
THIS_LANE_PHASE3_STATUS = "COMPLETE_PHASE3_TABLE"

# ---------------------------------------------------------------------------
# Identity and roots (spec §2.1)
# ---------------------------------------------------------------------------

PHASE = "m1_t0c1_prefix_v1_50ep"
RESULT_ROOT_RELATIVE = "tfpd_exploration/results/m1_t0c1_prefix_v1_50ep"
SMOKE_ROOT_RELATIVE = f"{RESULT_ROOT_RELATIVE}/smoke"
ARM_ROOT_RELATIVE = {"t0": f"{RESULT_ROOT_RELATIVE}/t0", "c1": f"{RESULT_ROOT_RELATIVE}/c1"}
PROBE_ROOT_RELATIVE = f"{RESULT_ROOT_RELATIVE}/probe"
PHASE3_ROOT_RELATIVE = f"{RESULT_ROOT_RELATIVE}/phase3_table"
STAGE_ORDER = ("smoke", "t0", "c1", "probe", "phase3")

# ---------------------------------------------------------------------------
# Budget (spec §2.2) — EPOCHS is the one horizon change besides the LR law
# ---------------------------------------------------------------------------

EPOCHS = 50
TOTAL_OPTIMIZER_STEPS = EPOCHS * STEPS_PER_EPOCH
_require(STEPS_PER_EPOCH == 4951, "50-epoch steps/epoch drifted from spec §2.2")
_require(TOTAL_OPTIMIZER_STEPS == 247_550, "50-epoch total optimizer steps drifted from spec §2.2")
_require(HARD_TIMEOUT_SECONDS_PER_ARM == 43_200, "50-epoch per-arm timeout drifted from spec §2.2")

#: Measured 20-epoch cost (spec §2.2).  Full-run means MUST NOT be called
#: post-memoization rates: epoch 0 fills the episode memo (~1851 s) against
#: ~425 s later.  True post-memo rates are the audit values below.
PRECEDENT_20EP_ELAPSED_SECONDS = {"t0": 9919.430263, "c1": 8685.517042}
PRECEDENT_20EP_FULL_RUN_SECONDS_PER_STEP = {"t0": 0.10018, "c1": 0.08771}
POST_MEMO_SECONDS_PER_STEP = {"t0": 0.08577056604618638, "c1": 0.07313887172792367}
EPOCH_0_SECONDS_APPROX = 1851
LATER_EPOCH_SECONDS_APPROX = 425
LINEAR_PROJECTION_50EP_HOURS = {"t0": 6.29, "c1": 5.43}
WORST_OBSERVED_EPOCH_PROJECTION_HOURS = {"t0": 7.11}
RUNTIME_ABORT_ESTIMATE_SECONDS = 40_000
_require(RUNTIME_ABORT_ESTIMATE_SECONDS == 40_000, "runtime abort heuristic drifted from spec §2.2")

OPERATOR_RESOLUTION = {
    "schema": "m1_t0c1_prefix_v1_50ep_operator_cost_resolution_v1",
    "decision": "12 h hard timeout PER ARM; t0 and c1 may share GPU 1 as a concurrent pair; "
                "smoke/probe/phase3 exclusive; full 50-epoch pair, no epoch reduction",
    "order": list(STAGE_ORDER),
    "gpu": "CUDA_VISIBLE_DEVICES=1 (physical GPU 1 only), device cuda:0",
    "measured_20ep_elapsed_seconds": dict(PRECEDENT_20EP_ELAPSED_SECONDS),
    "full_run_seconds_per_step_not_post_memo": dict(PRECEDENT_20EP_FULL_RUN_SECONDS_PER_STEP),
    "post_memo_seconds_per_step": dict(POST_MEMO_SECONDS_PER_STEP),
    "linear_projection_50ep_hours": dict(LINEAR_PROJECTION_50EP_HOURS),
    "worst_observed_epoch_projection_hours": dict(WORST_OBSERVED_EPOCH_PROJECTION_HOURS),
    "runtime_abort_estimate_seconds": RUNTIME_ABORT_ESTIMATE_SECONDS,
    "hard_timeout_seconds_per_arm": HARD_TIMEOUT_SECONDS_PER_ARM,
}

#: GPU-1 arm concurrency (t0 and c1 may share the card; other stages exclusive).
#: The epoch-1 abort heuristic is intentionally unrelaxed: contention that
#: pushes epoch_0 + 49 * epoch_1 above 40000 s is a genuine abort.
CONCURRENT_ARM_STAGES = ("t0", "c1")
EXCLUSIVE_CARD_STAGES = ("smoke", "probe", "phase3")
MAX_GPU1_USED_MEMORY_MIB_BEFORE_START = 8192
ARM_CONCURRENCY = "concurrent_pair_on_gpu1"
ARM_CONCURRENCY_TIMING_DISCLOSURE = (
    "per-epoch wall times under concurrency are inflated by CPU/GPU contention "
    "and are therefore NOT comparable to the sealed 20-epoch reference timings, "
    "while the numerics are unaffected because cudnn.benchmark is off and every "
    "stream is per-process deterministic"
)
_require(MAX_GPU1_USED_MEMORY_MIB_BEFORE_START == 8192,
         "GPU-1 pre-start used-memory headroom drifted from 8192 MiB")
_require(RUNTIME_ABORT_ESTIMATE_SECONDS == 40_000,
         "concurrency law must not relax the 40000 s abort heuristic")

ARM_CONCURRENCY_LAW = {
    "schema": "m1_t0c1_prefix_v1_50ep_arm_concurrency_law_v1",
    "concurrency_permitted_stages": list(CONCURRENT_ARM_STAGES),
    "exclusive_card_stages": list(EXCLUSIVE_CARD_STAGES),
    "permitted_sibling": "cmdline invokes this launcher AND --stage in {t0, c1} "
                         "AND that stage differs from the stage now starting",
    "fail_closed": [
        "unresolvable compute-app cmdline",
        "same-stage sibling (double-launch)",
        "more than one sibling",
        "foreign compute app",
        "gpu1 used_memory_mib strictly greater than 8192",
        "exclusive stage with any compute app on GPU 1",
    ],
    "max_used_memory_mib_before_start": MAX_GPU1_USED_MEMORY_MIB_BEFORE_START,
    "arm_concurrency": ARM_CONCURRENCY,
    "timing_disclosure": ARM_CONCURRENCY_TIMING_DISCLOSURE,
    "numerics_unaffected_because": (
        "cudnn.benchmark is off (PyTorch default False); "
        "per-step batch/RNG/LR streams are CPU-side and per-process deterministic"
    ),
    "runtime_abort_heuristic_unchanged": True,
    "runtime_abort_estimate_seconds": RUNTIME_ABORT_ESTIMATE_SECONDS,
    "runtime_abort_comparison": "strictly greater than 40000 s aborts; exactly 40000 does not",
}

#: Frozen launch envelope (spec §2.9).  Bound into every attempt and launch.
LAUNCH_ENVELOPE = {
    "PYTHON": "/home/xinyuan/miniconda3/envs/spint/bin/python",
    "CUDA_VISIBLE_DEVICES": "1",
    "PYTHONNOUSERSITE": "1",
    "PYTHONPATH": "/home/xinyuan/Work_host/SPINT",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "cwd": "/home/xinyuan/Work_host/SPINT",
    "expected_torch": "2.5.1.post303 / cuda 11.8 / cudnn 90300",
    "expected_device": {
        "name": "NVIDIA GeForce RTX 3090",
        "total_memory_bytes": 25438126080,
        "capability": [8, 6],
    },
}

#: Frozen device identity (spec §2.10).  The parent profiler must not be called.
DEVICE_IDENTITY_LAW = {
    "schema": "m1_t0c1_prefix_v1_50ep_device_identity_law_v1",
    "query": "nvidia-smi --id ${CUDA_VISIBLE_DEVICES}  (i.e. --id 1)",
    "required_uuid": "GPU-2220ed5d-25ea-1839-28d7-ad4dfa5f6c86",
    "required_pci_bus_id": "00000000:03:00.0",
    "refused_uuid_gpu0": "GPU-ac7388a5-2e98-300a-fdb3-0b67bfd494d9",
    "parent_live_device_profile_forbidden": True,
}

# ---------------------------------------------------------------------------
# LR schedule law (spec §2.3) — imported at runtime from arm_common, not copied
# ---------------------------------------------------------------------------

LR_SCHEDULE_KIND = "warmup_then_cosine"
WARMUP_EPOCHS = 2
WARMUP_STEPS = 9_902
LR_WARMUP_START = 1e-5
LR_WARMUP_END = 1e-4
LR_FINAL = 1e-6
PHASE_LOCAL_STEPS = True
_require(WARMUP_STEPS == WARMUP_EPOCHS * STEPS_PER_EPOCH, "warmup_steps drifted from 2 * 4951")
_require(ADAM_LR == LR_WARMUP_START, "Adam constructor LR must equal lr_at_step(0)")

LR_SCHEDULE_LAW = {
    "schema": "m1_t0c1_prefix_v1_50ep_lr_schedule_law_v1",
    "kind": LR_SCHEDULE_KIND,
    "n_epochs": EPOCHS,
    "steps_per_epoch": STEPS_PER_EPOCH,
    "total_steps": TOTAL_OPTIMIZER_STEPS,
    "warmup_epochs": WARMUP_EPOCHS,
    "warmup_steps": WARMUP_STEPS,
    "lr_warmup_start": LR_WARMUP_START,
    "lr_warmup_end": LR_WARMUP_END,
    "lr_final": LR_FINAL,
    "phase_local_steps": PHASE_LOCAL_STEPS,
    "imported_from": "tfpd_exploration/src/tfpd_lane/arm_common.py",
    "reimplemented": False,
}

#: Copy of the sealed dropout proof plus the reproducible hash convention
#: (spec §2.5).  The predecessor dict is not mutated.
DROPOUT_PROOF = {
    **dict(v1_plan.DROPOUT_PROOF),
    "hash_convention": {
        "encoding": "utf-8",
        "join": '"\\n".join(source.splitlines()[start:end])',
        "trailing_newline": False,
        "byte_count": 508,
        "m1_slice_0based": [448, 455],
        "spint_original_slice_0based": [130, 137],
        "cited_lines_1based": {"m1": "449-455", "spint_original": "131-137"},
        "note": "1-based inclusive lines 449-455 / 131-137 map to 0-based slices "
                "[448:455] / [130:137]; 7 lines joined with newline and NO trailing newline",
        "with_trailing_newline_sha256": "31ed62f79bb28005a0635cc9b75c34412edfb3b8bfd96076df736e27df36a729",
        "without_trailing_newline_sha256": "eacb0402448636cd0e757e9d1a28065209a5decacdbe68fe73722f634f858884",
    },
}
_require(DROPOUT_PROOF["block_sha256_both_trees"]
         == "eacb0402448636cd0e757e9d1a28065209a5decacdbe68fe73722f634f858884",
         "dropout block sha drifted from spec §2.5")

# ---------------------------------------------------------------------------
# SPINT resource parity (spec §1.1)
# ---------------------------------------------------------------------------

SPINT_RESOURCE_PARITY = {
    "schema": "m1_t0c1_prefix_v1_50ep_spint_resource_parity_v1",
    "spint_m1_released": {
        "citation": "SPINT-main/README.md:48",
        "lr": 1e-4,
        "max_epochs": 50,
        "scheduler": None,
        "batch_size": 32,
        "window": 100,
        "calibration_trials": 10,
        "seed": 42,
        "optimizer": "Adam, weight_decay=0.0",
        "selection": "val_heldout/r2_mean max",
        "periodic_checkpoint_every_n_epochs": 10,
        "training_sessions": 4,
        "steps_per_epoch_approx": 6667,
        "total_optimizer_steps_approx": 333350,
    },
    "this_lane": {
        "epochs": EPOCHS,
        "batch_size": TRAIN_BATCH_SIZE,
        "window": 100,
        "calibration_trials": 10,
        "seed": SEED,
        "optimizer": "Adam, weight_decay=0.0",
        "peak_lr": LR_WARMUP_END,
        "scheduler": LR_SCHEDULE_KIND,
        "checkpoint_cadence_epochs": 10,
        "selection": "val_heldout equal-session-mean static_m10 R2, mode=max",
        "training_sessions": 3,
        "steps_per_epoch": STEPS_PER_EPOCH,
        "total_optimizer_steps": TOTAL_OPTIMIZER_STEPS,
    },
    "parity": ["epochs", "batch_size", "window", "calibration_trials", "seed",
               "optimizer", "checkpoint_cadence", "selection_criterion", "peak_lr"],
    "peak_lr_parity_only": True,
    "mean_lr_below_spint_constant": True,
    "disclosed_deficits": {
        "training_sessions": "3 vs 4 (LODO-mandated; 20120924 is the fold target)",
        "steps_per_epoch": "4951 vs ~6667",
        "total_optimizer_steps": {"this_lane": 247550, "spint_approx": 333350, "fraction_below": 0.26},
        "horizon_compensation_rejected": "70 epochs would give 346570 steps; operator kept 50",
    },
}

SUPERSEDED_BY_50EP_LAW = {
    "schema": "m1_t0c1_prefix_v1_50ep_source_authority_overlay_v1",
    "note": "inherited prepare() fragment reports the 20-epoch matched-ERM route spec; "
            "those fields do not describe this lane",
    "stale_keys": {
        "epoch_budget": {"inherited_value": 20, "true_value": 50},
        "scheduler": {"inherited_value": "None", "true_value": "warmup_then_cosine"},
        "adam_lr": {"inherited_value": 1e-5,
                    "true_value": "constructor lr_at(0)=1e-5; per-step warmup_then_cosine peak 1e-4"},
    },
    "true_epochs": EPOCHS,
    "true_total_optimizer_steps": TOTAL_OPTIMIZER_STEPS,
    "true_lr_schedule_kind": LR_SCHEDULE_KIND,
}

# ---------------------------------------------------------------------------
# Checkpoint policy (spec §3)
# ---------------------------------------------------------------------------

#: 1-based epochs 10, 20, 30, 40, 50 = 0-based indices below.  Epoch 50 is
#: the terminal state.
CHECKPOINT_EPOCH_INDICES = (9, 19, 29, 39, 49)
CHECKPOINT_EPOCHS_1BASED = (10, 20, 30, 40, 50)
EPOCH_50_INDEX = 49
_require(CHECKPOINT_EPOCH_INDICES == (9, 19, 29, 39, 49), "checkpoint epoch indices drifted from spec §3")
_require(tuple(index + 1 for index in CHECKPOINT_EPOCH_INDICES) == CHECKPOINT_EPOCHS_1BASED,
         "0-based/1-based checkpoint epoch pairing drifted from spec §3")
_require(EPOCH_50_INDEX == CHECKPOINT_EPOCH_INDICES[-1] == EPOCHS - 1,
         "epoch-50 terminal index drifted")


def epoch_checkpoint_filename(epoch_index: int) -> str:
    """Leaf name for one sealed 0-based epoch checkpoint."""
    _require(type(epoch_index) is int and epoch_index in CHECKPOINT_EPOCH_INDICES,
             f"epoch checkpoint index is not one of {CHECKPOINT_EPOCH_INDICES}: {epoch_index!r}")
    return f"checkpoint_epoch_{epoch_index:02d}.pt"


BEST_CHECKPOINT_FILENAME = "checkpoint_best_source_train_loss.pt"

CHECKPOINT_POLICY = {
    "schema": "m1_t0c1_prefix_v1_50ep_checkpoint_policy_v1",
    "epoch_indices_0based": list(CHECKPOINT_EPOCH_INDICES),
    "epoch_indices_1based": list(CHECKPOINT_EPOCHS_1BASED),
    "epoch_50_is_terminal": True,
    "keep_best_source_train_loss_leaf": True,
    "probe_and_phase3_select_only_from_epoch_checkpoints": True,
    "swa_enabled": False,
    "swa_artifact_forbidden": True,
    "strict_reload": True,
}

# ---------------------------------------------------------------------------
# Smoke equality (spec §2.6) — predecessor contract plus the new LR channel
# ---------------------------------------------------------------------------

SMOKE_EQUALITY_CONTRACT = {
    "schema": "m1_t0c1_prefix_v1_50ep_smoke_equality_contract_v1",
    "steps_per_arm": SMOKE_STEPS,
    "must_be_equal_across_arms": [
        "initial_model_state_sha256",
        "per_step_batch_digest (episode row sample-id stream)",
        "per_step_python_rng_state_digest (the dropout-p stream; one random.uniform per forward)",
        "per_step optimizer-step count and one-forward-per-step",
        "per_step_lr_sequence equals [lr_at_step(i, 50, 4951) for i in range(12)]",
    ],
    "c1_only": "per-step scheduled/effective prefix follows (10, 5, 2) with visible-slice digests",
    "t0_only": "effective prefix is the full 10-trial block every step (operator disabled)",
    "eval_dropout_inactive_proof": "model.eval() forward repeated twice yields identical prediction digests",
    "lr_channel": "the schedule consumes no RNG, so the matched-pair equality contract is unaffected by it",
    "cal_aug_discipline_reference": "tfpd_exploration/src/cal_aug_v1 (hook/schedule/receipt conventions)",
}

# ---------------------------------------------------------------------------
# Predecessor 20-epoch digests (spec §2.8) — provenance, never written
# ---------------------------------------------------------------------------

PREDECESSOR_20EP_ARM_TERMINAL_SHA256 = dict(v1_plan.SEALED_ARM_TERMINAL_SHA256)
PREDECESSOR_20EP_SMOKE_TERMINAL_SHA256 = v1_plan.SEALED_SMOKE_TERMINAL_SHA256
PREDECESSOR_20EP_PHASE3_TERMINAL_SHA256 = (
    "821818b1ef9068aed87b437af3a8363cf2226e7778776dd9c6c10746bf22f096"
)
_require(
    PREDECESSOR_20EP_ARM_TERMINAL_SHA256["t0"]
    == "4aea40a317047beec059505231bb9996190c4eb2235392d08a3c948cf7ab9aa9",
    "20-epoch t0 terminal digest drifted from spec §2.8",
)
_require(
    PREDECESSOR_20EP_ARM_TERMINAL_SHA256["c1"]
    == "cbef49e8e356103c56a9ffa673a268de29e1c5564fc5957c26474b1cf065da99",
    "20-epoch c1 terminal digest drifted from spec §2.8",
)
_require(
    PREDECESSOR_20EP_SMOKE_TERMINAL_SHA256
    == "ed15417eec1a4a10aafd9e67b13514ebd43f04b0f15cea12bb615185078ceb6f",
    "20-epoch smoke terminal digest drifted from spec §2.8",
)
_require(
    PREDECESSOR_20EP_PHASE3_TERMINAL_SHA256
    == "821818b1ef9068aed87b437af3a8363cf2226e7778776dd9c6c10746bf22f096",
    "20-epoch phase3 terminal digest drifted from spec §2.8",
)

# ---------------------------------------------------------------------------
# Probe law (spec §4 / §4.0 / §4.0.1 / §4.1)
# ---------------------------------------------------------------------------

PROBE_DEPLOYMENT = "static_m10"
OVERFITTING_DROP_THRESHOLD = 0.005
OVERFITTING_VERDICT_PRESENT = "OVERFITTING_PRESENT"
OVERFITTING_VERDICT_ABSENT = "OVERFITTING_ABSENT_WITHIN_50EP"
OVERFITTING_VERDICT_INCONCLUSIVE = "OVERFITTING_INCONCLUSIVE"

TRAIN_FIT_SESSIONS = HELDIN_TRAINING_SESSIONS
TEST_FOLD_SESSIONS = HELDOUT_FOLD_SESSIONS
VAL_HELDOUT_SESSIONS = ("20121004", "20121017", "20121024")
SELECTION_SURFACE = "val_heldout"
PROBE_SESSIONS = tuple(SCORE_ORDER) + VAL_HELDOUT_SESSIONS
_require(VAL_HELDOUT_SESSIONS == ("20121004", "20121017", "20121024"),
         "val_heldout sessions drifted from spec §4.0")
_require("20120924" not in VAL_HELDOUT_SESSIONS, "test fold leaked into val_heldout")
_require(len(PROBE_SESSIONS) == 7 and len(set(PROBE_SESSIONS)) == 7,
         "probe grid is 7 distinct sessions")

VAL_HELDOUT_PATH_TEMPLATE = (
    "SPINT-main/data/000941/sub-MonkeyL-held-out-calib/"
    "sub-MonkeyL-held-out-calib_ses-{session}_behavior+ecephys.nwb"
)
VAL_HELDOUT_BODY_SHA256 = {
    "20121004": "782c1fd090facfb3c50b6a85da8209ca3aea71f66bd4edc13d61fe4a55a3429d",
    "20121017": "8dd22c67500445ec1e0c11980475badbba9e960a05db16b78aaaad5502b3c652",
    "20121024": "bbeb6c7d66e2c2e9b76506021a6cf8800bc19021b6d1fd03c4eb6de71490bafb",
}
VAL_HELDOUT_FORBIDDEN_TOKENS = ("minival", "test", "evalai", "formal", "held-out-minival")
FOLD_TARGET_EXCLUDED_FROM_SELECTION = "20120924"

TRAIN_FIT_WINDOW_RETENTION = {
    "20120926": {"retained": 54467, "total": 54476},
    "20120927": {"retained": 49189, "total": 49228},
    "20120928": {"retained": 54766, "total": 54783},
    "fraction": 0.9997,
    "note": "99.97% of every held-in window is a training row; train_fit cannot select",
}


def val_heldout_relative_path(session_id: str) -> str:
    _require(session_id in VAL_HELDOUT_SESSIONS, f"not a val_heldout session: {session_id!r}")
    return VAL_HELDOUT_PATH_TEMPLATE.format(session=session_id)


def assert_evaluation_only_path(relative: str) -> str:
    """Fail closed on a forbidden token or a non-held-out-calib path (spec §4.0.1)."""
    checked = safe_relative(relative)
    lowered = checked.replace("\\", "/")
    for token in VAL_HELDOUT_FORBIDDEN_TOKENS:
        _require(token not in lowered, f"val_heldout path contains forbidden token {token!r}: {checked}")
    _require("held-out-calib" in lowered, f"val_heldout path is not held-out-calib: {checked}")
    _require(FOLD_TARGET_EXCLUDED_FROM_SELECTION not in checked,
             "20120924 must not appear in a val_heldout path")
    return checked


def assert_selection_sessions(sessions) -> tuple[str, ...]:
    """Fail closed if the selection input is not exactly the val_heldout triple."""
    _require(isinstance(sessions, (tuple, list)), "selection sessions must be a sequence")
    ordered = tuple(str(item) for item in sessions)
    _require(FOLD_TARGET_EXCLUDED_FROM_SELECTION not in ordered,
             "20120924 reached the selection input")
    _require(ordered == VAL_HELDOUT_SESSIONS,
             f"selection sessions must be exactly {VAL_HELDOUT_SESSIONS}, saw {ordered}")
    return ordered


def assert_val_heldout_body_digest(session_id: str, body: bytes) -> str:
    """Fail closed if an opened val_heldout body does not match the frozen digest."""
    _require(session_id in VAL_HELDOUT_BODY_SHA256, f"not a val_heldout session: {session_id!r}")
    _require(isinstance(body, (bytes, bytearray)), "val_heldout body must be bytes")
    digest = sha256_bytes(bytes(body))
    expected = VAL_HELDOUT_BODY_SHA256[session_id]
    _require(digest == expected,
             f"val_heldout body digest mismatch for {session_id}: {digest} != {expected}")
    return digest


def runtime_abort_estimate_seconds(epoch_0_seconds: float, epoch_1_seconds: float) -> float:
    """Spec §2.2: ``epoch_0_seconds + 49 * epoch_1_seconds``."""
    return float(epoch_0_seconds) + 49.0 * float(epoch_1_seconds)


def assert_runtime_abort_heuristic(epoch_0_seconds: float, epoch_1_seconds: float) -> float:
    """Raise if the §2.2 estimate exceeds 40000 s; return the estimate otherwise."""
    estimate = runtime_abort_estimate_seconds(epoch_0_seconds, epoch_1_seconds)
    _require(
        estimate <= RUNTIME_ABORT_ESTIMATE_SECONDS,
        f"runtime abort heuristic: epoch_0_seconds + 49 * epoch_1_seconds = {estimate} "
        f"exceeds {RUNTIME_ABORT_ESTIMATE_SECONDS}",
    )
    return estimate


_VAL_HELDOUT_ACCESS_LAW_BODY = {
    "schema": "m1_t0c1_50ep_val_heldout_access_law_v1",
    "purpose": "checkpoint selection only, mirroring SPINT val_heldout/r2_mean",
    "training_use": False,
    "gradient_updates": 0,
    "optimizer_steps": 0,
    "labels_used_for": "metric only",
    "sessions": list(VAL_HELDOUT_SESSIONS),
    "opened_paths": [val_heldout_relative_path(session_id) for session_id in VAL_HELDOUT_SESSIONS],
    "body_sha256": dict(VAL_HELDOUT_BODY_SHA256),
    "forbidden_tokens": list(VAL_HELDOUT_FORBIDDEN_TOKENS),
    "fold_target_excluded_from_selection": FOLD_TARGET_EXCLUDED_FROM_SELECTION,
}
VAL_HELDOUT_ACCESS_LAW = {
    **_VAL_HELDOUT_ACCESS_LAW_BODY,
    "law_sha256": sha256_bytes(canonical_json_bytes(_VAL_HELDOUT_ACCESS_LAW_BODY)),
}
_require(
    VAL_HELDOUT_ACCESS_LAW["law_sha256"]
    == "24205e652628aaf428e4d5bc27cad9c6386589c719db02613e9e3870c7fd341a",
    "val_heldout access law_sha256 drifted from spec §4.0.1 (hash of body excluding law_sha256)",
)
for _session_id in VAL_HELDOUT_SESSIONS:
    assert_evaluation_only_path(val_heldout_relative_path(_session_id))
    _require(FOLD_TARGET_EXCLUDED_FROM_SELECTION not in val_heldout_relative_path(_session_id),
             "20120924 leaked into a frozen val_heldout path")

PROBE_LAW = {
    "schema": "m1_t0c1_prefix_v1_50ep_probe_law_v1",
    "train_fit_surface": "equal-session mean over (20120926, 20120927, 20120928)",
    "val_heldout_surface": "equal-session mean over (20121004, 20121017, 20121024)",
    "test_fold_surface": "20120924",
    "train_fit_sessions": list(TRAIN_FIT_SESSIONS),
    "val_heldout_sessions": list(VAL_HELDOUT_SESSIONS),
    "test_fold_sessions": list(TEST_FOLD_SESSIONS),
    "probe_sessions": list(PROBE_SESSIONS),
    "grid": "2 arms x 5 checkpoints x 7 sessions = 70 scored cells",
    "deployment": PROBE_DEPLOYMENT,
    "deployment_note": "static_m10 ONLY (operator decision; the anchors were measured under this law). "
                       "CDM-FIFO is reserved for task 3.",
    "scorer": "m1_t0c1_prefix_v1.phase3.score_static",
    "metric": METRIC_LABEL,
    "selection_surface": SELECTION_SURFACE,
    "sessions_argument_mandatory": True,
    "sessions_must_be_exactly_val_heldout": True,
    "train_fit_is_reporting_only": True,
    "test_fold_is_reporting_only": True,
    "train_fit_window_retention": TRAIN_FIT_WINDOW_RETENTION,
    "verdict": {
        "evaluated_on": SELECTION_SURFACE,
        "test_fold_is_labelled_diagnostic": True,
        "descriptive_not_inferential": True,
        "OVERFITTING_PRESENT": "val_heldout argmax is not epoch 50 AND drop from that maximum to epoch 50 exceeds 0.005 R2",
        "OVERFITTING_ABSENT_WITHIN_50EP": "val_heldout argmax is epoch 50",
        "OVERFITTING_INCONCLUSIVE": "interior max but drop <= 0.005",
        "threshold": OVERFITTING_DROP_THRESHOLD,
        "comparison": "strict > for PRESENT; drop == 0.005 is INCONCLUSIVE",
        "spread_sd_ddof": 0,
        "spread_yardstick": "val_heldout across-session SD at the argmax of the curve being judged",
    },
    "observed_body_sha256": "persisted alongside frozen body_sha256 on every opened val_heldout record",
    "access_law_sha256": VAL_HELDOUT_ACCESS_LAW["law_sha256"],
}

SELECTION_RULE = {
    "schema": "m1_t0c1_prefix_v1_50ep_selection_rule_v1",
    "selected_epoch": "argmax over CHECKPOINT_EPOCH_INDICES of the val_heldout equal-session-mean static_m10 R2",
    "val_heldout_sessions": list(VAL_HELDOUT_SESSIONS),
    "tie_break": "smallest epoch index",
    "mode": "max",
    "surface": SELECTION_SURFACE,
    "sessions_argument": "mandatory; must be exactly VAL_HELDOUT_SESSIONS; None is refused",
    "forbidden_as_selection_input": ["20120924", "train_fit"],
}

# ---------------------------------------------------------------------------
# Phase-3 surface (spec §5) and 20-epoch different-recipe reference (spec §5.1)
# ---------------------------------------------------------------------------

CROSS_RECIPE_NOTE = (
    "the 20-epoch line used constant lr=1e-5 with no scheduler; this 50-epoch "
    "lane uses full-horizon warmup_then_cosine (peak 1e-4).  Every comparison "
    "is a cross-recipe delta, not a nested-horizon delta."
)

REFERENCE_20EP_EQUAL_SESSION_MEAN = {
    "t0": {
        "static_m10": {
            "heldin_training": 0.7270238995552063,
            "heldout_fold": 0.5707439184188843,
        },
        "cdm_activity_fifo_m10": {
            "heldin_training": 0.6594383120536804,
            "heldout_fold": 0.5842786431312561,
        },
    },
    "c1": {
        "static_m10": {
            "heldin_training": 0.7306439876556396,
            "heldout_fold": 0.5848761796951294,
        },
        "cdm_activity_fifo_m10": {
            "heldin_training": 0.6623618404070536,
            "heldout_fold": 0.5947345495223999,
        },
    },
}

REFERENCE_20EP_HELDIN_PER_SESSION = {
    "t0": {
        "20120926": 0.7142195701599121,
        "20120927": 0.7335554957389832,
        "20120928": 0.7332966327667236,
    },
    "c1": {
        "20120926": 0.7211775779724121,
        "20120927": 0.7341418266296387,
        "20120928": 0.7366125583648682,
    },
}

REFERENCE_20EP_TRAIN_LOSS_CONTEXT = {
    "best_epoch_index_both_arms": 19,
    "t0_final_epoch_mean": 0.042958451470428254,
    "c1_final_epoch_mean": 0.043094195018068535,
    "note": "source train loss had not bottomed out at 20 epochs; empirical motivation for extending the horizon",
}

# ---------------------------------------------------------------------------
# This lane's 50-epoch terminals are discovered at run time, not plan-frozen
# ---------------------------------------------------------------------------

THIS_LANE_SMOKE_BINDING = {
    "binding": "verified_at_run_time",
    "role": "this_lane_50ep_smoke_terminal",
    "expected_status": THIS_LANE_SMOKE_STATUS,
    "root_relative": SMOKE_ROOT_RELATIVE,
}
THIS_LANE_ARM_BINDING = {
    "binding": "verified_at_run_time",
    "role": "this_lane_50ep_arm_terminals",
    "expected_status": THIS_LANE_ARM_STATUS,
    "root_relative_by_arm": dict(ARM_ROOT_RELATIVE),
}

# ---------------------------------------------------------------------------
# Closure paths
# ---------------------------------------------------------------------------

RUNTIME_PATHS: tuple[str, ...] = v1_plan.RUNTIME_PATHS + (
    "tfpd_exploration/src/tfpd_lane/arm_common.py",
)
OWNED_PATHS: tuple[str, ...] = (
    "tfpd_exploration/src/m1_t0c1_prefix_v1_50ep/__init__.py",
    "tfpd_exploration/src/m1_t0c1_prefix_v1_50ep/plan.py",
    "tfpd_exploration/src/m1_t0c1_prefix_v1_50ep/lr_schedule.py",
    "tfpd_exploration/src/m1_t0c1_prefix_v1_50ep/trainer.py",
    "tfpd_exploration/src/m1_t0c1_prefix_v1_50ep/receipts.py",
    "tfpd_exploration/src/m1_t0c1_prefix_v1_50ep/probe.py",
    "tfpd_exploration/src/m1_t0c1_prefix_v1_50ep/phase3.py",
    "tfpd_exploration/src/m1_t0c1_prefix_v1_50ep/driver.py",
    "tfpd_exploration/scripts/run_m1_t0c1_prefix_v1_50ep.py",
    "tfpd_exploration/scripts/launch_m1_t0c1_prefix_v1_50ep.py",
    "tfpd_exploration/tests/test_m1_t0c1_prefix_v1_50ep.py",
)
LAUNCHER_RELATIVE = "tfpd_exploration/scripts/launch_m1_t0c1_prefix_v1_50ep.py"
_require(LAUNCHER_RELATIVE in OWNED_PATHS, "launcher must be in the implementation closure")


@dataclass(frozen=True)
class PairSpec50:
    """The sole 50-epoch matched-pair surface; no training choice remains at run time."""

    root_relative: str = RESULT_ROOT_RELATIVE

    def payload(self) -> dict[str, object]:
        return {
            "schema": "m1_t0c1_prefix_v1_50ep_pair_spec_v1",
            "cell": CELL,
            "phase": PHASE,
            "root_relative": self.root_relative,
            "arms": list(ARMS),
            "arm_roles": dict(ARM_ROLES),
            "objective": {"system": "MATCHED_ERM", "lambda": OBJECTIVE_LAMBDA, "tau": OBJECTIVE_TAU},
            "budget": {
                "seed": SEED, "epochs": EPOCHS, "steps_per_epoch": STEPS_PER_EPOCH,
                "total_optimizer_steps": TOTAL_OPTIMIZER_STEPS, "batch_size": TRAIN_BATCH_SIZE,
                "optimizer": "Adam", "adam_lr": ADAM_LR, "adam_weight_decay": ADAM_WEIGHT_DECAY,
                "adam_betas": [0.9, 0.999], "adam_eps": 1e-8, "amsgrad": False,
                "scheduler": LR_SCHEDULE_KIND, "swa": False,
                "checkpoint_selection": "five_sealed_epoch_checkpoints_plus_epoch_mean_source_train_loss_first_minimum_leaf",
            },
            "lr_schedule_law": dict(LR_SCHEDULE_LAW),
            "cycle_law": CYCLE_LAW,
            "source_preparation_law": SOURCE_PREPARATION_LAW,
            "dropout_proof": DROPOUT_PROOF,
            "derivative_scan_omission": DERIVATIVE_SCAN_OMISSION,
            "step_cost_probe": STEP_COST_PROBE,
            "smoke_equality_contract": SMOKE_EQUALITY_CONTRACT,
            "checkpoint_policy": dict(CHECKPOINT_POLICY),
            "probe_law": dict(PROBE_LAW),
            "selection_rule": dict(SELECTION_RULE),
            "hard_timeout_seconds_per_arm": HARD_TIMEOUT_SECONDS_PER_ARM,
            "operator_resolution": OPERATOR_RESOLUTION,
            "arm_concurrency_law": dict(ARM_CONCURRENCY_LAW),
            "launch_envelope": dict(LAUNCH_ENVELOPE),
            "device_identity_law": dict(DEVICE_IDENTITY_LAW),
            "spint_resource_parity": dict(SPINT_RESOURCE_PARITY),
            "superseded_by_50ep_law": dict(SUPERSEDED_BY_50EP_LAW),
            "val_heldout_access_law": dict(VAL_HELDOUT_ACCESS_LAW),
            "phase3_surface": {
                "arms": list(ARMS),
                "deployments": list(PHASE3_DEPLOYMENTS),
                "surfaces": list(PHASE3_SURFACES),
                "cdm_fifo_law": CDM_FIFO_LAW,
                "tables": ("epoch50", "selected_epoch"),
                "alias_of_epoch50_when_both_arms_select_50": True,
            },
            "sessions": {
                "heldout_fold": list(HELDOUT_FOLD_SESSIONS),
                "heldin_training": list(HELDIN_TRAINING_SESSIONS),
                "train_fit": list(TRAIN_FIT_SESSIONS),
                "val_heldout": list(VAL_HELDOUT_SESSIONS),
                "test_fold": list(TEST_FOLD_SESSIONS),
                "score_order": list(SCORE_ORDER),
                "probe_sessions": list(PROBE_SESSIONS),
            },
            "predecessor_20ep_terminal_sha256": dict(PREDECESSOR_20EP_ARM_TERMINAL_SHA256),
            "predecessor_20ep_smoke_terminal_sha256": PREDECESSOR_20EP_SMOKE_TERMINAL_SHA256,
            "predecessor_20ep_phase3_terminal_sha256": PREDECESSOR_20EP_PHASE3_TERMINAL_SHA256,
            "reference_20ep_equal_session_mean": REFERENCE_20EP_EQUAL_SESSION_MEAN,
            "cross_recipe_note": CROSS_RECIPE_NOTE,
            "phase1_motivation": PHASE1_MOTIVATION,
            "formal_benchmark_verdict": False,
            "launcher_relative": LAUNCHER_RELATIVE,
        }

    @property
    def sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.payload()))


def _closure_leaf_sha(root: Path, relative: str) -> str:
    try:
        return v1._read_regular_no_follow(Path(root), relative)
    except v1.SourceLifecycleError as error:
        raise M1T0C150EpPlanError(f"m1 t0c1 50ep closure leaf drift: {relative}") from error


def implementation_closure(root: Path) -> dict[str, object]:
    """Predecessor 20-epoch closure plus this package's bytes (incl. arm_common)."""
    try:
        inherited = v1_plan.implementation_closure(Path(root))
    except v1_plan.M1T0C1PlanError as error:
        raise M1T0C150EpPlanError("m1 t0c1 50ep inherited 20-epoch closure drift") from error
    rows = [dict(row) for row in inherited.get("paths", [])]  # type: ignore[union-attr]
    _require(isinstance(rows, list) and rows, "m1 t0c1 50ep inherited closure topology drift")
    seen = {row.get("path") for row in rows}
    extra = ("tfpd_exploration/src/tfpd_lane/arm_common.py", *OWNED_PATHS)
    for relative in extra:
        if relative in seen:
            continue
        rows.append({"path": relative, "sha256": _closure_leaf_sha(Path(root), relative)})
        seen.add(relative)
    _require("tfpd_exploration/src/tfpd_lane/arm_common.py" in seen,
             "m1 t0c1 50ep closure is missing arm_common.py")
    body = {
        "schema": "m1_t0c1_prefix_v1_50ep_closure_v1",
        "predecessor_20ep_closure_sha256": inherited.get("closure_sha256"),
        "result_root_relative": RESULT_ROOT_RELATIVE,
        "paths": rows,
    }
    return {**body, "closure_sha256": sha256_bytes(canonical_json_bytes(body))}


def validate_current_closure(root: Path, value: Mapping[str, object]) -> dict[str, object]:
    _require(isinstance(value, Mapping), "m1 t0c1 50ep closure must be a mapping")
    rebuilt = implementation_closure(Path(root))
    _require(dict(value) == rebuilt, "m1 t0c1 50ep closure/current-byte drift")
    return rebuilt


def dry_plan(root: Path | None = None) -> dict[str, object]:
    del root
    spec = PairSpec50()
    payload = {
        "cell": CELL,
        "phase": PHASE,
        "prospective_roots": {
            "smoke": SMOKE_ROOT_RELATIVE, "t0": ARM_ROOT_RELATIVE["t0"],
            "c1": ARM_ROOT_RELATIVE["c1"], "probe": PROBE_ROOT_RELATIVE,
            "phase3": PHASE3_ROOT_RELATIVE,
        },
        "pair_spec": spec.payload(),
        "phase1_gap_root_anchor": dict(PHASE1_GAP_ROOT_ANCHOR),
        "public_execution_authorized": False,
        "opens_nwb_or_checkpoint": False,
        "imports_torch": False,
        "initializes_cuda": False,
        "creates_root_or_receipt": False,
    }
    _require(payload["imports_torch"] is False
             and payload["initializes_cuda"] is False
             and payload["creates_root_or_receipt"] is False,
             "50-epoch dry_plan purity flags drifted")
    return payload


__all__ = (
    "M1T0C150EpPlanError", "CELL", "PHASE", "RESULT_ROOT_RELATIVE", "SMOKE_ROOT_RELATIVE",
    "ARM_ROOT_RELATIVE", "PROBE_ROOT_RELATIVE", "PHASE3_ROOT_RELATIVE", "ARMS", "ARM_ROLES",
    "CYCLE", "CYCLE_LAW", "SEED", "EPOCHS", "STEPS_PER_EPOCH", "TOTAL_OPTIMIZER_STEPS",
    "ADAM_LR", "ADAM_WEIGHT_DECAY", "TRAIN_BATCH_SIZE", "OBJECTIVE_LAMBDA", "OBJECTIVE_TAU",
    "NO_SWA", "RECORD_STEPS", "SMOKE_STEPS", "HARD_TIMEOUT_SECONDS_PER_ARM",
    "OPERATOR_RESOLUTION", "ARM_CONCURRENCY_LAW", "ARM_CONCURRENCY",
    "ARM_CONCURRENCY_TIMING_DISCLOSURE", "CONCURRENT_ARM_STAGES",
    "EXCLUSIVE_CARD_STAGES", "MAX_GPU1_USED_MEMORY_MIB_BEFORE_START",
    "LAUNCH_ENVELOPE", "DEVICE_IDENTITY_LAW", "DROPOUT_PROOF",
    "DERIVATIVE_SCAN_OMISSION", "PHASE1_MOTIVATION", "SPINT_RESOURCE_PARITY",
    "SUPERSEDED_BY_50EP_LAW", "PHASE3_DEPLOYMENTS", "ACCEPTED_V6_GRAPH_SHA256",
    "SOURCE_PREPARATION_LAW", "STEP_COST_PROBE", "safe_relative",
    "PREDECESSOR_20EP_ARM_TERMINAL_SHA256", "PREDECESSOR_20EP_SMOKE_TERMINAL_SHA256",
    "PREDECESSOR_20EP_PHASE3_TERMINAL_SHA256", "THIS_LANE_SMOKE_STATUS", "THIS_LANE_ARM_STATUS",
    "PHASE3_SURFACES", "CDM_FIFO_LAW", "SMOKE_EQUALITY_CONTRACT", "BOUND_GPU", "LR_SCHEDULE_LAW",
    "CHECKPOINT_EPOCH_INDICES", "CHECKPOINT_EPOCHS_1BASED", "EPOCH_50_INDEX",
    "epoch_checkpoint_filename", "BEST_CHECKPOINT_FILENAME", "PROBE_LAW", "SELECTION_RULE",
    "SELECTION_SURFACE", "OVERFITTING_DROP_THRESHOLD", "TRAIN_FIT_SESSIONS",
    "VAL_HELDOUT_SESSIONS", "TEST_FOLD_SESSIONS", "PROBE_SESSIONS", "VAL_HELDOUT_BODY_SHA256",
    "VAL_HELDOUT_ACCESS_LAW", "VAL_HELDOUT_FORBIDDEN_TOKENS", "assert_evaluation_only_path",
    "assert_selection_sessions", "assert_val_heldout_body_digest",
    "assert_runtime_abort_heuristic", "runtime_abort_estimate_seconds",
    "RUNTIME_ABORT_ESTIMATE_SECONDS", "PairSpec50", "canonical_json_bytes", "sha256_bytes",
    "require_sha", "implementation_closure", "validate_current_closure", "dry_plan",
    "RUNTIME_PATHS", "OWNED_PATHS", "REFERENCE_20EP_EQUAL_SESSION_MEAN", "CROSS_RECIPE_NOTE",
)
