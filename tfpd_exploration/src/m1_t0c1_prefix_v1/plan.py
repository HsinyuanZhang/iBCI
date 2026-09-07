"""Frozen contract for the M1 matched T0/C1 calibration-prefix training pair.

Everything that could bias the matched pair or the Phase-3 readout is frozen
here before any data or model access: the pre-declared prefix cycle, the
matched 20-epoch no-SWA budget, the SPINT-original dropout proof, the device
binding, and the per-arm hard wall-clock timeout.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping

from tfpd_exploration.src.cross_session_worst_group_v1 import source_lifecycle as v1
from tfpd_exploration.src.m1_heldin_heldout_gap_v1 import plan as gap_plan


class M1T0C1PlanError(RuntimeError):
    """Fail closed for static pair contract, literals, or closure drift."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise M1T0C1PlanError(message)


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def require_sha(value: object, label: str) -> str:
    try:
        return gap_plan.require_sha(value, label)
    except gap_plan.M1GapPlanError as error:
        raise M1T0C1PlanError(str(error)) from error


def safe_relative(value: object) -> str:
    """Delegates to the sealed Phase-1 gap helper (repo-relative, no ..)."""
    try:
        return gap_plan.safe_relative(value)
    except gap_plan.M1GapPlanError as error:
        raise M1T0C1PlanError(str(error)) from error


CELL = gap_plan.CELL
PHASE = "m1_t0c1_prefix_v1"
WORKORDER_RELATIVE = gap_plan.WORKORDER_RELATIVE
WORKORDER_SHA256 = gap_plan.WORKORDER_SHA256
RESULT_ROOT_RELATIVE = "tfpd_exploration/results/m1_t0c1_prefix_v1"
SMOKE_ROOT_RELATIVE = f"{RESULT_ROOT_RELATIVE}/smoke"
ARM_ROOT_RELATIVE = {"t0": f"{RESULT_ROOT_RELATIVE}/t0", "c1": f"{RESULT_ROOT_RELATIVE}/c1"}
PHASE3_ROOT_RELATIVE = f"{RESULT_ROOT_RELATIVE}/phase3_table"

HELDOUT_FOLD_SESSIONS = gap_plan.HELDOUT_FOLD_SESSIONS
HELDIN_TRAINING_SESSIONS = gap_plan.HELDIN_TRAINING_SESSIONS
SCORE_ORDER = gap_plan.SCORE_ORDER
FOLD_TARGET_OF_SESSION = gap_plan.FOLD_TARGET_OF_SESSION
BUDGET_LABEL = gap_plan.BUDGET_LABEL
METRIC_LABEL = gap_plan.METRIC_LABEL
MODEL_SHAPE = dict(gap_plan.MODEL_SHAPE)
EVAL_BATCH_SIZE = gap_plan.EVAL_BATCH_SIZE

ARMS = ("t0", "c1")
ARM_ROLES = {
    "t0": "matched control: SAME runner, operator registered but disabled (kwargs unchanged)",
    "c1": "deterministic chronological calibration-prefix cycle on the training forward",
}

#: Pre-declared operator cycle.  M1 has exactly one sealed budget (native M10),
#: so per the {full, half, quarter} law the cycle is (10, 5, 2); the quarter
#: term floors 10/4 = 2.5 to the integer 2 (disclosed, not silently rounded).
CYCLE = (10, 5, 2)
CYCLE_LAW = {
    "schema": "m1_t0c1_prefix_cycle_law_v1",
    "basis": "M1 has a single sealed calibration budget (native M10, calibration_trials=10); "
             "no M30/M10/M4 analog exists in the worst-group/fold assets",
    "declared_cycle": list(CYCLE),
    "terms": {"full": 10, "half": 5, "quarter_floored": 2},
    "quarter_disclosure": "10/4 = 2.5 floored to the integer prefix 2; never rounded up",
    "advance": "one training forward per step; M = cycle[step % 3] (deterministic, no RNG)",
    "training_gated": True,
    "eval_untouched": True,
}

SEED = 42
EPOCHS = 20
STEPS_PER_EPOCH = 4951
TOTAL_OPTIMIZER_STEPS = EPOCHS * STEPS_PER_EPOCH
ADAM_LR = 1.0e-5
ADAM_WEIGHT_DECAY = 0.0
TRAIN_BATCH_SIZE = 32
OBJECTIVE_LAMBDA = 0.0
OBJECTIVE_TAU = 0.01
NO_SWA = True
RECORD_STEPS = 40
SMOKE_STEPS = 12
#: Per-arm hard wall-clock timeout (operator resolution 2026-08-31): the 12 h
#: bound applies PER ARM, serially.  The line's 20-epoch precedent measured
#: 38,869.9 s (10.8 h) per arm, so the full 20-epoch pair runs unshrunk.
HARD_TIMEOUT_SECONDS_PER_ARM = 12 * 3600
PRECEDENT_ELAPSED_SECONDS = 38869.855224059895
PRECEDENT_SOURCE = (
    "tfpd_exploration/results/cross_session_worst_group_m1_source_full_v1_no_swa/"
    "fold_20120924_cswg/training.json (resources.elapsed_seconds)"
)
OPERATOR_RESOLUTION = {
    "schema": "m1_t0c1_operator_cost_resolution_v1",
    "decision": "12 h hard timeout PER ARM, serially; full 20-epoch pair, no epoch reduction",
    "order": ["t0", "c1"],
    "gpu": "CUDA_VISIBLE_DEVICES=1 (physical GPU 1 only), device cuda:0",
    "precedent_elapsed_seconds_per_arm": PRECEDENT_ELAPSED_SECONDS,
    "precedent_source": PRECEDENT_SOURCE,
}

#: Training-loop deviation relative to the frozen CS-WG loop: the per-step
#: ``torch.autograd.grad`` derivative observation scan (retain_graph=True) is
#: omitted.  It is a pure diagnostic that consumes no RNG and changes no
#: gradient value; both arms share the identical loop, and the omission is
#: disclosed here rather than applied silently.
DERIVATIVE_SCAN_OMISSION = {
    "omitted": True,
    "what": "per-step torch.autograd.grad(objective.loss, session_losses, retain_graph=True) scan",
    "reason": "pure observation, numerically inert for training (no RNG consumption, no gradient change)",
    "both_arms_identical_loop": True,
}

#: Steady-state step-cost measurement (60-step GPU-1 probe after the t0
#: attempt-2 abort) and the disclosed, semantics-preserving memoization that
#: keeps one arm inside the 12 h bound.
STEP_COST_PROBE = {
    "schema": "m1_t0c1_step_cost_probe_v1",
    "measured_per_step_seconds": 0.5794594965331877,
    "marks_seconds_total_60_steps": {"episode": 18.368, "forward": 16.251,
                                      "backward": 0.088, "gradcheck": 0.055,
                                      "record": 0.007},
    "dominant_cost": "episode assembly (O(pool-rows) per-step candidate scan)",
    "fix": "memoize the deterministic epoch-local episode stream across epochs",
    "semantics": ("build_balanced_episode is documented 'without global RNG' with "
                  "stable sha-based offsets; the episode for a given epoch-local "
                  "step index is identical every epoch, so the memo replays the "
                  "exact frozen episode sequence"),
    "attempt1_effective_per_step_seconds": 0.373,
    "aborted_attempt2_root": "t0_attempt2_aborted_slow_full_detail_recording",
}

#: The SPINT-original dropout proof bound into every attempt BEFORE launch.
DROPOUT_PROOF = {
    "schema": "m1_t0c1_spint_original_dropout_proof_v1",
    "law": "whole-unit Bernoulli dropout mask F.dropout(ones(B,N), p) applied to src after the "
           "learnable-ID addition and before fc_in; with dynamic_dropout, p = random.uniform(low, "
           "high) drawn once per training forward; inactive in eval (training=False)",
    "operator_environment_description_matches": (
        "cal_aug_v1 sealed operator environment: dynamic whole-unit dropout, p~U(0,1) per "
        "training forward, one random.uniform draw, eval inactive"
    ),
    "m1_line_block": {
        "path": "streaming_calibration_exp/src/models/components/spint.py",
        "lines": "449-455",
    },
    "spint_original_block": {
        "path": "SPINT-main/src/models/components/spint.py",
        "lines": "131-137",
    },
    "block_sha256_both_trees": "eacb0402448636cd0e757e9d1a28065209a5decacdbe68fe73722f634f858884",
    "byte_identical": True,
    "m1_recipe_config": {
        "path": "streaming_calibration_exp/configs/model/falcon_m1_source_only_decoder.yaml",
        "dynamic_dropout": True, "dynamic_dropout_low": 0.0, "dynamic_dropout_high": 1.0,
        "dropout_rate": 0.0, "tf_drop_rate": 0.1,
    },
    "spint_reference_configs": [
        "SPINT-main/configs/model/falcon_h1.yaml:29-31",
        "SPINT-main/configs/model/falcon_m2_post33_confirm_v3.yaml:30-32",
    ],
    "frozen_training_receipt_records": ("dynamic_dropout_preserved", True),
    "override_applied": False,
    "verified_before_launch": True,
}

#: Phase-1 motivation, bound into the terminal receipts as directed.
PHASE1_MOTIVATION = {
    "schema": "m1_t0c1_phase1_motivation_binding_v1",
    "phase1_root_relative": gap_plan.RESULT_ROOT_RELATIVE,
    "phase1_paired_table_sha256": "6c410a7ae11441e2bcfa227a9683230143c173190b7828424bad1cfd2c084271",
    "phase1_verdict": "HEADROOM_PRESENT",
    "phase1_direction": "HELD_OUT_WORSE",
    "phase1_gap": -0.12488512198130286,
    "phase1_gap_ci95": [-0.133009135723114, -0.11267030239105225],
    "note": ("Phase 1's gap verdict MOTIVATES rather than discourages Phase 3: the CDM/C1 "
             "readout is a headroom-capture test, not a negative-control confirmation"),
}

#: Phase-3 2x2x2 readout surface.
PHASE3_DEPLOYMENTS = ("static_m10", "cdm_activity_fifo_m10")
PHASE3_SURFACES = ("heldin_training", "heldout_fold")
PHASE3_TABLE_AXIS = ("arm", "deployment", "surface")
CDM_FIFO_LAW = {
    "schema": "m1_t0c1_cdm_activity_fifo_law_v1",
    "selection": "ROLLING_FIXED_M from m1_h1_activity_headroom_v1.core.selection_for_output_trial",
    "rule": ("for output trial t >= 10 use the last 10 completed trials [t-10, t); before that "
             "the frozen support [0,10); label-free, causal, cardinality always 10"),
    "trial_boundaries": "dataset.trial_start_indices[session] via output-trial mapping "
                        "(414 trials on 20120924; boundaries exist on M1)",
    "support_trials": 10,
}

#: The sealed, terminal-OK smoke predecessor every later stage binds by digest.
SEALED_SMOKE_TERMINAL_SHA256 = "ed15417eec1a4a10aafd9e67b13514ebd43f04b0f15cea12bb615185078ceb6f"
SEALED_SMOKE_STATUS = "COMPLETE_MATCHED_SMOKE_EQUALITY"

#: The sealed, terminal-OK arm predecessors phase 3 binds by digest.
SEALED_ARM_TERMINAL_SHA256 = {
    "t0": "4aea40a317047beec059505231bb9996190c4eb2235392d08a3c948cf7ab9aa9",
    "c1": "cbef49e8e356103c56a9ffa673a268de29e1c5564fc5957c26474b1cf065da99",
}
SEALED_ARM_STATUS = "COMPLETE_MATCHED_ARM_TRAINING"

SMOKE_EQUALITY_CONTRACT = {
    "schema": "m1_t0c1_smoke_equality_contract_v1",
    "steps_per_arm": SMOKE_STEPS,
    "must_be_equal_across_arms": [
        "initial_model_state_sha256",
        "per_step_batch_digest (episode row sample-id stream)",
        "per_step_python_rng_state_digest (the dropout-p stream; one random.uniform per forward)",
        "per_step optimizer-step count and one-forward-per-step",
    ],
    "c1_only": "per-step scheduled/effective prefix follows (10, 5, 2) with visible-slice digests",
    "t0_only": "effective prefix is the full 10-trial block every step (operator disabled)",
    "eval_dropout_inactive_proof": "model.eval() forward repeated twice yields identical prediction digests",
    "cal_aug_discipline_reference": "tfpd_exploration/src/cal_aug_v1 (hook/schedule/receipt conventions)",
}

PHASE1_GAP_ROOT_ANCHOR = {
    "root_relative": gap_plan.RESULT_ROOT_RELATIVE,
    "anchor_score_sha256_20120924": "27e743be55ab57297dce88d7be487a341c55732cbde3f43d32b6892b5d53b29f",
}

BOUND_GPU = {"cuda_visible_devices": "1", "torch_device": "cuda:0", "operator": OPERATOR_RESOLUTION["gpu"]}

#: The frozen accepted-V6 smoke predecessor graph every full 20-epoch run in
#: this line binds before source preparation (terminal-pinned immutable root
#: ``cross_session_worst_group_m1_source_smoke_v6``).
ACCEPTED_V6_GRAPH_SHA256 = "32750a6a9f4a726ee63a59508a210d9f184d224f27cbc8c80354faca021618bf"
SOURCE_PREPARATION_LAW = {
    "schema": "m1_t0c1_source_preparation_law_v1",
    "path": "V6-bound full source preparation: sealed-metadata read, stratum authority fit, "
            "deterministic common-intersect-min-count-2 pruned pools, audit-to-full rebind",
    "raw_per_session_distinct_strata": {"20120926": 47, "20120927": 51, "20120928": 50},
    "eligible_common_min2_strata": 43,
    "minimum_rows_per_eligible_stratum": 2,
    "accepted_v6_graph_sha256": ACCEPTED_V6_GRAPH_SHA256,
    "identical_to_frozen_full_runs": True,
}

RUNTIME_PATHS: tuple[str, ...] = (
    "tfpd_exploration/src/m1_heldin_heldout_gap_v1/plan.py",
    "tfpd_exploration/src/m1_heldin_heldout_gap_v1/score.py",
    "tfpd_exploration/src/m1_heldin_heldout_gap_v1/physical.py",
    "tfpd_exploration/src/cross_session_worst_group_fold20120924_score_v1/plan.py",
    "tfpd_exploration/src/cross_session_worst_group_fold20120924_score_v1/score.py",
    "tfpd_exploration/src/cross_session_worst_group_fold20120924_score_v1/physical.py",
    "tfpd_exploration/src/cross_session_worst_group_v1/source_audit_v2.py",
    "tfpd_exploration/src/cross_session_worst_group_v1/source_physical.py",
    "tfpd_exploration/src/cross_session_worst_group_v1/source_reader.py",
    "tfpd_exploration/src/m1_h1_activity_headroom_v1/core.py",
    "tfpd_exploration/src/m1_h1_activity_headroom_v1/m1.py",
)
OWNED_PATHS: tuple[str, ...] = (
    WORKORDER_RELATIVE,
    "tfpd_exploration/src/m1_t0c1_prefix_v1/__init__.py",
    "tfpd_exploration/src/m1_t0c1_prefix_v1/plan.py",
    "tfpd_exploration/src/m1_t0c1_prefix_v1/schedule.py",
    "tfpd_exploration/src/m1_t0c1_prefix_v1/hook.py",
    "tfpd_exploration/src/m1_t0c1_prefix_v1/trainer.py",
    "tfpd_exploration/src/m1_t0c1_prefix_v1/receipts.py",
    "tfpd_exploration/src/m1_t0c1_prefix_v1/phase3.py",
    "tfpd_exploration/src/m1_t0c1_prefix_v1/driver.py",
    "tfpd_exploration/scripts/run_m1_t0c1_prefix_v1.py",
    "tfpd_exploration/tests/test_m1_t0c1_prefix_v1.py",
)


@dataclass(frozen=True)
class PairSpec:
    """The sole matched-pair surface; no training choice remains at run time."""

    root_relative: str = RESULT_ROOT_RELATIVE

    def payload(self) -> dict[str, object]:
        return {
            "schema": "m1_t0c1_prefix_pair_spec_v1",
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
                "scheduler": "None", "swa": False, "checkpoint_selection": "epoch_mean_source_train_loss_first_minimum",
            },
            "cycle_law": CYCLE_LAW,
            "source_preparation_law": SOURCE_PREPARATION_LAW,
            "dropout_proof": DROPOUT_PROOF,
            "derivative_scan_omission": DERIVATIVE_SCAN_OMISSION,
            "step_cost_probe": STEP_COST_PROBE,
            "smoke_equality_contract": SMOKE_EQUALITY_CONTRACT,
            "hard_timeout_seconds_per_arm": HARD_TIMEOUT_SECONDS_PER_ARM,
            "operator_resolution": OPERATOR_RESOLUTION,
            "phase3_surface": {
                "arms": list(ARMS),
                "deployments": list(PHASE3_DEPLOYMENTS),
                "surfaces": list(PHASE3_SURFACES),
                "cdm_fifo_law": CDM_FIFO_LAW,
            },
            "sessions": {
                "heldout_fold": list(HELDOUT_FOLD_SESSIONS),
                "heldin_training": list(HELDIN_TRAINING_SESSIONS),
                "score_order": list(SCORE_ORDER),
            },
            "phase1_motivation": PHASE1_MOTIVATION,
            "formal_benchmark_verdict": False,
        }

    @property
    def sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.payload()))


def _closure_leaf_sha(root: Path, relative: str) -> str:
    try:
        return v1._read_regular_no_follow(Path(root), relative)
    except v1.SourceLifecycleError as error:
        raise M1T0C1PlanError(f"m1 t0c1 closure leaf drift: {relative}") from error


def implementation_closure(root: Path) -> dict[str, object]:
    """Explicit closure over the Phase-1 gap closure plus this lane's bytes."""
    try:
        inherited = gap_plan.implementation_closure(Path(root))
    except gap_plan.M1GapPlanError as error:
        raise M1T0C1PlanError("m1 t0c1 inherited gap closure drift") from error
    rows = [dict(row) for row in inherited.get("paths", [])]  # type: ignore[union-attr]
    _require(isinstance(rows, list) and rows, "m1 t0c1 inherited closure topology drift")
    seen = {row.get("path") for row in rows}
    for relative in (*RUNTIME_PATHS, *OWNED_PATHS):
        if relative in seen:
            continue
        rows.append({"path": relative, "sha256": _closure_leaf_sha(Path(root), relative)})
        seen.add(relative)
    workorder = next((row for row in rows if row.get("path") == WORKORDER_RELATIVE), None)
    _require(isinstance(workorder, Mapping) and workorder.get("sha256") == WORKORDER_SHA256,
             "m1 t0c1 workorder literal/body drift")
    body = {
        "schema": "m1_t0c1_prefix_closure_v1",
        "current_m1_gap_closure_sha256": inherited.get("closure_sha256"),
        "result_root_relative": RESULT_ROOT_RELATIVE,
        "paths": rows,
    }
    return {**body, "closure_sha256": sha256_bytes(canonical_json_bytes(body))}


def validate_current_closure(root: Path, value: Mapping[str, object]) -> dict[str, object]:
    _require(isinstance(value, Mapping), "m1 t0c1 closure must be a mapping")
    rebuilt = implementation_closure(Path(root))
    _require(dict(value) == rebuilt, "m1 t0c1 closure/current-byte drift")
    return rebuilt


def dry_plan(root: Path | None = None) -> dict[str, object]:
    del root
    spec = PairSpec()
    return {
        "cell": CELL,
        "phase": PHASE,
        "prospective_roots": {
            "smoke": SMOKE_ROOT_RELATIVE, "t0": ARM_ROOT_RELATIVE["t0"],
            "c1": ARM_ROOT_RELATIVE["c1"], "phase3": PHASE3_ROOT_RELATIVE,
        },
        "pair_spec": spec.payload(),
        "phase1_gap_root_anchor": dict(PHASE1_GAP_ROOT_ANCHOR),
        "public_execution_authorized": False,
        "opens_nwb_or_checkpoint": False,
        "imports_torch": False,
        "initializes_cuda": False,
        "creates_root_or_receipt": False,
    }


__all__ = (
    "M1T0C1PlanError", "CELL", "PHASE", "RESULT_ROOT_RELATIVE", "SMOKE_ROOT_RELATIVE",
    "ARM_ROOT_RELATIVE", "PHASE3_ROOT_RELATIVE", "ARMS", "ARM_ROLES", "CYCLE", "CYCLE_LAW",
    "SEED", "EPOCHS", "STEPS_PER_EPOCH", "TOTAL_OPTIMIZER_STEPS", "ADAM_LR", "ADAM_WEIGHT_DECAY",
    "TRAIN_BATCH_SIZE", "OBJECTIVE_LAMBDA", "OBJECTIVE_TAU", "NO_SWA", "RECORD_STEPS",
    "SMOKE_STEPS", "HARD_TIMEOUT_SECONDS_PER_ARM", "PRECEDENT_ELAPSED_SECONDS", "OPERATOR_RESOLUTION",
    "DROPOUT_PROOF", "DERIVATIVE_SCAN_OMISSION", "PHASE1_MOTIVATION", "PHASE3_DEPLOYMENTS",
    "ACCEPTED_V6_GRAPH_SHA256", "SOURCE_PREPARATION_LAW", "STEP_COST_PROBE", "safe_relative",
    "SEALED_ARM_TERMINAL_SHA256", "SEALED_ARM_STATUS",
    "PHASE3_SURFACES", "CDM_FIFO_LAW", "SMOKE_EQUALITY_CONTRACT", "BOUND_GPU", "PairSpec",
    "canonical_json_bytes", "sha256_bytes", "require_sha", "implementation_closure",
    "validate_current_closure", "dry_plan",
)
