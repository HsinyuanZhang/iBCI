"""Frozen contract for the M1 Tier 1/2 pack successor (C1 + chunk-CDM + SWA + W32).

Preserves the sealed T0/C1 20-epoch recipe (seed 42, Adam 1e-5, batch 32,
cycle (10,5,2), 3 source sessions, left-out 20120924, MATCHED_ERM λ=0, no
50-epoch cosine). Public CLI may `--execute`. Packs two arms per GPU.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping

from tfpd_exploration.src.m1_heldin_heldout_gap_v1 import plan as gap_plan
from tfpd_exploration.src.m1_t0c1_prefix_v1 import plan as t0c1_plan


class PlanError(RuntimeError):
    """Fail closed for static pack-cell contract drift."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PlanError(message)


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


CELL = gap_plan.CELL
PHASE = "m1_tier12_pack_v1"
RESULT_ROOT_RELATIVE = "tfpd_exploration/results/m1_tier12_pack_v1"
SEALED_T0C1_RESULT_ROOT_RELATIVE = t0c1_plan.RESULT_ROOT_RELATIVE
SEALED_T0C1_PHASE3_TABLE_RELATIVE = t0c1_plan.PHASE3_ROOT_RELATIVE + "/table.json"
SEALED_T0C1_PHASE3_ROOT_RELATIVE = t0c1_plan.PHASE3_ROOT_RELATIVE

HELDOUT_FOLD_SESSIONS = t0c1_plan.HELDOUT_FOLD_SESSIONS
HELDIN_TRAINING_SESSIONS = t0c1_plan.HELDIN_TRAINING_SESSIONS
SCORE_ORDER = t0c1_plan.SCORE_ORDER
METRIC_LABEL = t0c1_plan.METRIC_LABEL
MODEL_SHAPE = dict(t0c1_plan.MODEL_SHAPE)
EVAL_BATCH_SIZE = t0c1_plan.EVAL_BATCH_SIZE

SEED = t0c1_plan.SEED
EPOCHS = t0c1_plan.EPOCHS
STEPS_PER_EPOCH = t0c1_plan.STEPS_PER_EPOCH
TOTAL_OPTIMIZER_STEPS = EPOCHS * STEPS_PER_EPOCH
ADAM_LR = t0c1_plan.ADAM_LR
ADAM_WEIGHT_DECAY = t0c1_plan.ADAM_WEIGHT_DECAY
TRAIN_BATCH_SIZE = t0c1_plan.TRAIN_BATCH_SIZE
OBJECTIVE_LAMBDA = t0c1_plan.OBJECTIVE_LAMBDA
OBJECTIVE_TAU = t0c1_plan.OBJECTIVE_TAU
RECORD_STEPS = t0c1_plan.RECORD_STEPS
HARD_TIMEOUT_SECONDS_PER_ARM = t0c1_plan.HARD_TIMEOUT_SECONDS_PER_ARM
ACCEPTED_V6_GRAPH_SHA256 = t0c1_plan.ACCEPTED_V6_GRAPH_SHA256
CYCLE = t0c1_plan.CYCLE
CYCLE_LAW = dict(t0c1_plan.CYCLE_LAW)
DROPOUT_PROOF = dict(t0c1_plan.DROPOUT_PROOF)
DERIVATIVE_SCAN_OMISSION = dict(t0c1_plan.DERIVATIVE_SCAN_OMISSION)
SOURCE_PREPARATION_LAW = dict(t0c1_plan.SOURCE_PREPARATION_LAW)
PHASE1_MOTIVATION = dict(t0c1_plan.PHASE1_MOTIVATION)
SEALED_ARM_TERMINAL_SHA256 = dict(t0c1_plan.SEALED_ARM_TERMINAL_SHA256)
SEALED_ARM_STATUS = t0c1_plan.SEALED_ARM_STATUS
CDM_FIFO_LAW = dict(t0c1_plan.CDM_FIFO_LAW)

_require(SEED == 42 and EPOCHS == 20 and STEPS_PER_EPOCH == 4951, "sealed T0/C1 budget drift")
_require(ADAM_LR == 1.0e-5 and TRAIN_BATCH_SIZE == 32, "sealed T0/C1 optimizer/batch drift")
_require(CYCLE == (10, 5, 2) and OBJECTIVE_LAMBDA == 0.0, "sealed T0/C1 cycle/ERM drift")
_require(HELDIN_TRAINING_SESSIONS == ("20120926", "20120927", "20120928"), "source session drift")
_require(HELDOUT_FOLD_SESSIONS == ("20120924",), "left-out session drift")

SCHEDULER = None
COSINE_50EP_FORBIDDEN = True
COSINE_50EP_NOTE = (
    "Do not use the failed 50-epoch cosine recipe (m1_t0c1_prefix_v1_50ep). "
    "This cell is Adam 1e-5, no scheduler, 20 epochs."
)

CHUNK_BINS = 1024
CHUNK_SUPPORT_TRIALS = 10
CHUNK_LAW = {
    "schema": "m1_tier12_chunk_cdm_law_v1",
    "chunk_bins": CHUNK_BINS,
    "basis": "M1 interpolated trial length; C-Pre analog",
    "slice": "neural_data[session][k*1024:(k+1)*1024] if complete; drop trailing remainder < 1024",
    "causal": "a query window whose last bin falls in chunk k may use only chunks strictly before k",
    "cardinality": CHUNK_SUPPORT_TRIALS,
    "bootstrap": (
        "if fewer than 10 completed chunks before k, use frozen support trials [0,10) of "
        "the trialized calib features (same as CDM-A / static M10 before trial 10); "
        "hybrid bootstrap disclosed; do not invent partial-chunk identities"
    ),
    "identity": (
        "mean-pool selected chunks through fc_id_in then fc_id_out; or if bootstrap, "
        "use support trial tensors shape [10,1024,64] exactly as static M10"
    ),
}

IDENTITY_WIDTH_STOCK = 1024
IDENTITY_WIDTH_W32 = 32
SWA_EPOCH_INDICES = (16, 17, 18, 19)
SWA_WINDOW = "final_4_epoch_mean_fp64"

CHUNK_CDM_ARMS = ("t0", "c1")
TRAIN_ARMS = ("t0_swa", "c1_swa", "t0_w32", "c1_w32")
CLI_ARMS = ("t0", "c1", "t0_swa", "c1_swa", "t0_w32", "c1_w32")
MODES = ("chunk_cdm", "train", "score")

FOUR_SESSION_ALL_SOURCE_OUT_OF_SCOPE = {
    "in_train_roster": False,
    "sessions": ["20121004", "20121017", "20121024"],
    "reason": (
        "cannot locally score official held-out query; keep 3-session LOSO so numbers "
        "are comparable to sealed T0/C1"
    ),
    "multi_seed_ensemble_out_of_scope": True,
}

ARM_SPECS = {
    "t0_swa": {
        "prefix": "t0", "prefix_kind": "t0", "identity_width": IDENTITY_WIDTH_STOCK, "swa": True,
        "stock_spint": True, "role": "disabled prefix, stock W1024, SWA final-4",
    },
    "c1_swa": {
        "prefix": "c1", "prefix_kind": "c1", "identity_width": IDENTITY_WIDTH_STOCK, "swa": True,
        "stock_spint": True, "role": "cycle (10,5,2), stock W1024, SWA final-4",
    },
    "t0_w32": {
        "prefix": "t0", "prefix_kind": "t0", "identity_width": IDENTITY_WIDTH_W32, "swa": False,
        "stock_spint": False, "role": "disabled prefix, identity_width=32, no SWA",
    },
    "c1_w32": {
        "prefix": "c1", "prefix_kind": "c1", "identity_width": IDENTITY_WIDTH_W32, "swa": False,
        "stock_spint": False, "role": "cycle (10,5,2), identity_width=32, no SWA",
    },
}

CHUNK_CDM_ROOT_RELATIVE = {
    "t0": f"{RESULT_ROOT_RELATIVE}/chunk_cdm/t0",
    "c1": f"{RESULT_ROOT_RELATIVE}/chunk_cdm/c1",
}
TRAIN_ROOT_RELATIVE = {arm: f"{RESULT_ROOT_RELATIVE}/{arm}" for arm in TRAIN_ARMS}
SCORE_ROOT_RELATIVE = {arm: f"{RESULT_ROOT_RELATIVE}/score/{arm}" for arm in TRAIN_ARMS}

PACK_LIMIT = 2
OWN_TOKEN = "run_m1_tier12_pack_v1.py"
OWN_TOKENS = (OWN_TOKEN,)
GPU0_INDEX = 0
GPU1_INDEX = 1
GPU0_UUID = "GPU-ac7388a5-2e98-300a-fdb3-0b67bfd494d9"
GPU1_UUID = "GPU-2220ed5d-25ea-1839-28d7-ad4dfa5f6c86"
GPU0_ALLOWED = False
GPU0_DISPLAY_MEMORY_MIB_ALLOWED = 453
DEVICE_NAME = "NVIDIA GeForce RTX 3090"
ELIGIBLE_GPU_INDICES = (GPU1_INDEX,)

ANCHOR_TOLERANCE = 1.0e-6
ANCHOR_T0_STATIC_20120924_LITERAL = 0.5707439184
ANCHOR_T0_TRIAL_CDM_20120924_LITERAL = 0.5842786431

# rSyn3 Z-Fix official extent. Score-time slice only; does not change training.
QUERY_Q10_END = (10, None)
QUERY_Q10_END_LABEL = "q10_end"
QUERY_Q10_END_SESSIONS = ("20120924",)

READ_RULES = {
    "schema": "m1_tier12_read_rules_v1",
    "primary_surface": "fold-0 20120924 variance-weighted last-bin R2",
    "governing_window": "full_session",
    "comparison_window_q10_end": {
        "query": [QUERY_Q10_END[0], QUERY_Q10_END[1]],
        "sessions": list(QUERY_Q10_END_SESSIONS),
        "governing": False,
        "purpose": (
            "apples-to-apples with rSyn3 Z-Fix official extent [10,end); "
            "SWA/W32 read rules stay on the sealed full-session T0/C1 surface"
        ),
    },
    "formal_benchmark_verdict": False,
    "chunk_cdm": {
        "PRESERVES": "T0 chunk-CDM - T0 static >= 0.5 * (T0 trial-CDM - T0 static)",
        "else": "DROPS",
    },
    "swa": {
        "HELPS": "arm_swa static - sealed arm static >= +0.005",
        "HURTS": "arm_swa static - sealed arm static <= -0.005",
        "else": "NEUTRAL",
        "independently_for": ["t0", "c1"],
    },
    "w32": {
        "WINS": "t0_w32 static - sealed T0 static >= +0.01",
        "NONINFERIOR": "t0_w32 static - sealed T0 static >= -0.03 (H1 gate)",
        "else": "INFERIOR",
    },
    "checkpoint_selection": "not from target labels; SWA window frozen as final-4; W32 has no SWA",
}

OWNED_PATHS: tuple[str, ...] = (
    "tfpd_exploration/src/m1_tier12_pack_v1/__init__.py",
    "tfpd_exploration/src/m1_tier12_pack_v1/plan.py",
    "tfpd_exploration/src/m1_tier12_pack_v1/gpu.py",
    "tfpd_exploration/src/m1_tier12_pack_v1/chunk.py",
    "tfpd_exploration/src/m1_tier12_pack_v1/score.py",
    "tfpd_exploration/src/m1_tier12_pack_v1/train.py",
    "tfpd_exploration/src/m1_tier12_pack_v1/execute.py",
    "tfpd_exploration/src/m1_tier12_pack_v1/receipts.py",
    "tfpd_exploration/scripts/run_m1_tier12_pack_v1.py",
    "tfpd_exploration/tests/test_m1_tier12_pack_v1.py",
)

SEALED_RESULT_ROOTS: tuple[str, ...] = (
    "tfpd_exploration/results/m1_t0c1_prefix_v1",
    "tfpd_exploration/results/m1_t0c1_prefix_v1_50ep",
    "tfpd_exploration/results/m1_emg_syn3_fcm_v1",
    "tfpd_exploration/results/m1_emg_rsyn3_fcm_v1",
    "tfpd_exploration/results/m1_heldin_heldout_gap_v1",
    "tfpd_exploration/results/cross_session_worst_group_m1_source_full_v1_no_swa",
    "tfpd_exploration/results/cross_session_worst_group_m1_source_smoke_v6",
)


def prefix_kind(arm: str) -> str:
    if arm in ARM_SPECS:
        return str(ARM_SPECS[arm]["prefix_kind"])
    if arm in CHUNK_CDM_ARMS:
        return arm
    raise PlanError(f"unknown arm {arm!r}")


def validate_mode_arm(mode: str, arm: str) -> None:
    _require(mode in MODES, f"unknown mode {mode!r}")
    if mode == "chunk_cdm":
        _require(arm in CHUNK_CDM_ARMS, "chunk_cdm arms are t0/c1 (sealed predecessors)")
    else:
        _require(arm in TRAIN_ARMS, "train/score arms are t0_swa/c1_swa/t0_w32/c1_w32")


def gpu_uuid(gpu_index: int) -> str:
    if gpu_index == GPU0_INDEX:
        return GPU0_UUID
    if gpu_index == GPU1_INDEX:
        return GPU1_UUID
    raise PlanError(f"unsupported gpu index {gpu_index}")


def stage_relative(mode: str, arm: str) -> str:
    validate_mode_arm(mode, arm)
    if mode == "chunk_cdm":
        return f"chunk_cdm/{arm}"
    if mode == "train":
        return arm
    return f"score/{arm}"


def parent_relative_for_stage(mode: str) -> str | None:
    if mode == "chunk_cdm":
        return f"{RESULT_ROOT_RELATIVE}/chunk_cdm"
    if mode == "score":
        return f"{RESULT_ROOT_RELATIVE}/score"
    return RESULT_ROOT_RELATIVE


@dataclass(frozen=True)
class PairSpec:
    root_relative: str = RESULT_ROOT_RELATIVE

    def payload(self) -> dict[str, object]:
        return {
            "schema": "m1_tier12_pack_pair_spec_v1",
            "cell": CELL,
            "phase": PHASE,
            "root_relative": self.root_relative,
            "chunk_cdm_arms": list(CHUNK_CDM_ARMS),
            "train_arms": list(TRAIN_ARMS),
            "arm_specs": {key: dict(value) for key, value in ARM_SPECS.items()},
            "objective": {"system": "MATCHED_ERM", "lambda": OBJECTIVE_LAMBDA, "tau": OBJECTIVE_TAU},
            "budget": {
                "seed": SEED, "epochs": EPOCHS, "steps_per_epoch": STEPS_PER_EPOCH,
                "total_optimizer_steps": TOTAL_OPTIMIZER_STEPS, "batch_size": TRAIN_BATCH_SIZE,
                "optimizer": "Adam", "adam_lr": ADAM_LR, "adam_weight_decay": ADAM_WEIGHT_DECAY,
                "scheduler": SCHEDULER, "cosine_50ep_forbidden": COSINE_50EP_FORBIDDEN,
            },
            "cycle_law": CYCLE_LAW,
            "chunk_law": CHUNK_LAW,
            "swa": {"epoch_indices_0based": list(SWA_EPOCH_INDICES), "window": SWA_WINDOW, "fp64": True},
            "w32": {"identity_width": IDENTITY_WIDTH_W32, "model_dim": 1024, "window_size": 100, "swa": False},
            "sessions": {
                "heldout_fold": list(HELDOUT_FOLD_SESSIONS),
                "heldin_training": list(HELDIN_TRAINING_SESSIONS),
                "score_order": list(SCORE_ORDER),
                "four_session_all_source": dict(FOUR_SESSION_ALL_SOURCE_OUT_OF_SCOPE),
            },
            "gpu": {
                "gpu0_uuid": GPU0_UUID, "gpu1_uuid": GPU1_UUID,
                "pack_limit": PACK_LIMIT, "own_token": OWN_TOKEN,
                "gpu0_allowed": GPU0_ALLOWED,
                "gpu0_display_memory_mib_allowed": GPU0_DISPLAY_MEMORY_MIB_ALLOWED,
                "device": DEVICE_NAME, "gpu_index_choices": [GPU1_INDEX],
            },
            "read_rules": dict(READ_RULES),
            "formal_benchmark_verdict": False,
            "public_execution_authorized": True,
            "predecessor_t0c1_root": SEALED_T0C1_RESULT_ROOT_RELATIVE,
            "cosine_50ep_note": COSINE_50EP_NOTE,
        }

    @property
    def sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.payload()))


def dry_plan(root: Path | None = None) -> dict[str, object]:
    del root
    spec = PairSpec()
    return {
        "cell": CELL,
        "phase": PHASE,
        "prospective_roots": {
            **CHUNK_CDM_ROOT_RELATIVE,
            **TRAIN_ROOT_RELATIVE,
            **SCORE_ROOT_RELATIVE,
        },
        "pair_spec": spec.payload(),
        "train_arms": list(TRAIN_ARMS),
        "pack_limit": PACK_LIMIT,
        "chunk_bins": CHUNK_BINS,
        "cycle": list(CYCLE),
        "cosine_50ep_forbidden": COSINE_50EP_FORBIDDEN,
        "scheduler": SCHEDULER,
        "heldin_training_sessions": list(HELDIN_TRAINING_SESSIONS),
        "four_session_all_source_in_train_roster": False,
        "query_q10_end": {
            "query": [QUERY_Q10_END[0], QUERY_Q10_END[1]],
            "sessions": list(QUERY_Q10_END_SESSIONS),
            "score_time_only": True,
            "governing": False,
        },
        "pack_optional_for_single_occupancy": True,
        "public_execution_authorized": True,
        "opens_nwb_or_checkpoint": False,
        "imports_torch": False,
        "initializes_cuda": False,
        "creates_root_or_receipt": False,
        "cli": OWN_TOKEN,
        "execute_requires": ["--gpu-authorized", "--gpu-index 1", "--mode", "--arm"],
        "pack": "optional; omit for single occupancy. Wave launches use --pack on GPU 1 only.",
        "public_gpu_capability": False,
        "gpu": {
            "gpu0_allowed": GPU0_ALLOWED,
            "pack_limit": PACK_LIMIT,
            "eligible_indices": list(ELIGIBLE_GPU_INDICES),
            "own_token": OWN_TOKEN,
        },
        "four_session_all_source": False,
    }


CHUNK_ARMS = CHUNK_CDM_ARMS
ARM_SPEC = ARM_SPECS
STOCK_IDENTITY_WIDTH = IDENTITY_WIDTH_STOCK
W32_IDENTITY_WIDTH = IDENTITY_WIDTH_W32
SUPPORT_TRIALS = CHUNK_SUPPORT_TRIALS
WINDOW = int(MODEL_SHAPE["window"])
UNITS = int(MODEL_SHAPE["units"])
OUTPUTS = int(MODEL_SHAPE["outputs"])
SCORE_SESSIONS = ("20120924",)
OUT_OF_SCOPE = {
    "multi_seed": True,
    "four_session_all_source": True,
}
T0_STATIC_20120924 = 0.5707439184188843
T0_TRIAL_CDM_20120924 = 0.5842786431312561
C1_STATIC_20120924 = 0.5848761796951294
C1_TRIAL_CDM_20120924 = 0.5947345495223999
ANCHORS = {
    "t0": {"static_m10": T0_STATIC_20120924, "cdm_trial_fifo_m10": T0_TRIAL_CDM_20120924},
    "c1": {"static_m10": C1_STATIC_20120924, "cdm_trial_fifo_m10": C1_TRIAL_CDM_20120924},
}


def dry_cli_payload() -> dict[str, object]:
    return dry_plan()


def chunk_root_relative(arm: str) -> str:
    _require(arm in CHUNK_CDM_ARMS, f"unknown chunk arm {arm}")
    return CHUNK_CDM_ROOT_RELATIVE[arm]


def train_root_relative(arm: str) -> str:
    _require(arm in TRAIN_ARMS, f"unknown train arm {arm}")
    return TRAIN_ROOT_RELATIVE[arm]


def score_root_relative(arm: str) -> str:
    _require(arm in TRAIN_ARMS, f"unknown score arm {arm}")
    return SCORE_ROOT_RELATIVE[arm]


def implementation_closure(root: Path) -> dict[str, object]:
    rows = []
    for relative in OWNED_PATHS:
        path = Path(root) / relative
        _require(path.is_file() and not path.is_symlink(), f"closure leaf missing: {relative}")
        rows.append({"path": relative, "sha256": sha256_bytes(path.read_bytes())})
    body = {
        "schema": "m1_tier12_pack_closure_v1",
        "result_root_relative": RESULT_ROOT_RELATIVE,
        "paths": rows,
    }
    return {**body, "closure_sha256": sha256_bytes(canonical_json_bytes(body))}


__all__ = (
    "PlanError", "CELL", "PHASE", "RESULT_ROOT_RELATIVE", "CHUNK_CDM_ARMS", "TRAIN_ARMS",
    "CYCLE", "CHUNK_BINS", "PACK_LIMIT", "OWN_TOKEN", "SEED", "EPOCHS", "PairSpec",
    "dry_plan", "prefix_kind", "validate_mode_arm", "canonical_json_bytes", "sha256_bytes",
)
