"""Frozen contract for all-source B3 / B3S-rSyn3 students vs official Original SPINT.

Train on the four public held-in calibration sessions with the official
epoch-19 teacher. Local 20120924 R² after this fit is in-train, not official
held-out. The comparison target is EvalAI held-out R² 0.648591 (sub 578244).
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping


class PlanError(RuntimeError):
    """Fail closed for static B3 all-source contract drift."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PlanError(message)


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


CELL = "M1_B3_ALLSOURCE_V1"
PHASE = "m1_b3_allsource_v1"
RESULT_ROOT_RELATIVE = "tfpd_exploration/results/m1_b3_allsource_v1"

SOURCE_SESSIONS = ("20120924", "20120926", "20120927", "20120928")
SOURCE_SESSION_NAMES = (
    "ses-20120924",
    "ses-20120926",
    "ses-20120927",
    "ses-20120928",
)
LATER_CALIB_SESSIONS = ("20121004", "20121017", "20121024")
PUBLIC_CALIB_SESSIONS = SOURCE_SESSIONS + LATER_CALIB_SESSIONS

TEACHER_CHECKPOINT_RELATIVE = (
    "SPINT-main/logs/train/runs/2026-07-21-19-11-01/checkpoints/best_ckpt/epoch_019.ckpt"
)
TEACHER_SHA256 = "c81a2bbd860452e6186a9ecf55c0b747da61baef4fae3212f61521be68cc5ac2"

SEED = 42
EPOCHS = 12
ADAM_LR = 1.0e-4
ADAM_WEIGHT_DECAY = 0.0
TRAIN_BATCH_SIZE = 32
CALIBRATION_N_TRIALS = 10
HIDDEN_DIM = 64
IDENTITY_DIM = 100
CHANNELS = 64
TRIAL_BINS = 1024
WINDOW_SIZE = 100
SCHEDULER = None
COSINE_50EP_FORBIDDEN = True
HARD_TIMEOUT_SECONDS_PER_ARM = 8 * 3600

M4_QUERY_SUPPORT = 4
M4_QUERY_START = 4
M4_QUERY_ROOT_RELATIVE = f"{RESULT_ROOT_RELATIVE}/m4_query_v2"
M4_QUERY_NOT_OFFICIAL = (
    "M4-support / remaining-query is a local diagnostic. Source-session query "
    "windows were inside the train files. Later-day files have 10 trials, so "
    "query is the last 6. Neither surface is EvalAI hidden held-out."
)
M4_QUERY_INCLUDE_SOURCE_IN_TRAIN_DEFAULT = False
M4_QUERY_LATER_DAY_QUERY_TRIALS = 6

CARRIER_K_PROBE_ROOT_RELATIVE = f"{RESULT_ROOT_RELATIVE}/carrier_k_probe_v1"
CARRIER_K_ARM = "b3s_rsyn3"
CARRIER_K_METHODS = ("chronological", "dopt_tgt_loc", "dopt_emg_syn3")
CARRIER_K_BUDGETS = (3, 4, 5, 6)
CARRIER_K_POOL_TRIALS = 10
CARRIER_K_QUERY_START = 10
CARRIER_K_N_VARIANTS = 13
CARRIER_K_NOTE = (
    "Frozen b3s_rsyn3 student. Carrier/unit-ridge uses selected k calib trials; "
    "encoder identity still uses all 10 calib neural trials. Ranking R² is the "
    "in-train source public-calib remaining-query after trial 10 "
    "(n_trials>10, query_start_trial=10, calibration_n_trials=10). Later-day "
    "10-trial files have no leftover query; R² is null and only identity/carrier "
    "hashes are stored. Not EvalAI hidden held-out. NMF dictionary stays "
    "source-frozen. D-opt k=10 among 10 is omitted as a duplicate of chrono-10."
)
CARRIER_K_LATER_DAY_ROOT_RELATIVE = f"{RESULT_ROOT_RELATIVE}/carrier_k_later_day_v1"
CARRIER_K_LATER_DAY_ARMS = ("b3s_rsyn3", "b3s_rsyn3_freeze")
CARRIER_K_LATER_DAY_SUPPORT = 4
CARRIER_K_LATER_DAY_QUERY_START = 4
CARRIER_K_LATER_DAY_BUDGETS = (2, 3)
CARRIER_K_LATER_DAY_BASELINE_K = 4
CARRIER_K_LATER_DAY_N_VARIANTS = 7
CARRIER_K_LATER_DAY_NOTE = (
    "Ranking R² is later-day public-calib remaining-query (identity neural = "
    "first 4, query = last 6). Carrier ridge uses k trials selected from those "
    "4 support trials only; D-opt never sees query trials. Encoder identity is "
    "M4, not M10, because later-day files have 10 trials and post-M10 leftover "
    "query does not exist. k=5/6 and D-opt k=4 are omitted (k=4 D-opt is the "
    "full support set). Not EvalAI hidden held-out. Source in-train is not scored."
)

REMOTE_PUSH_ROOT_RELATIVE = f"{RESULT_ROOT_RELATIVE}/remote_push_v1"
REMOTE_CHALLENGE_ID = 2319
REMOTE_PHASE_ID = 4599
REMOTE_PHASE_SLUG = "few-shot-test-2319"
REMOTE_TEAM_ID = 41975
REMOTE_BATCH_SIZE = 4
REMOTE_BASE_IMAGE = "spint-original-m1:e9-epoch019-052e9ea"
REMOTE_BASE_IMAGE_ID = "sha256:f5af9eb29b7f86616d898261070193b1b0777db75567848c62d7f888ce3d76cd"
PAYLOAD_SCHEMA = "m1_b3_allsource_cached_identity_v1"

OFFICIAL_ORIGINAL_HELDOUT_R2 = 0.648591
OFFICIAL_ORIGINAL_SUBMISSION = 578244
OFFICIAL_T4_HELDOUT_R2 = 0.644766
OFFICIAL_T4_SUBMISSION = 578245
OFFICIAL_T4_NOT_FAIR_NOTE = (
    "EvalAI T4 0.644766 used a frozen original decoder plus a T4 encoder fitted "
    "on fold-1 three sources only. It is not a fair all-source B3S+T4 number."
)

PACK_LIMIT = 2
OWN_TOKEN = "run_m1_b3_allsource_v1.py"
OWN_TOKENS = (OWN_TOKEN,)
GPU0_INDEX = 0
GPU1_INDEX = 1
GPU0_UUID = "GPU-ac7388a5-2e98-300a-fdb3-0b67bfd494d9"
GPU1_UUID = "GPU-2220ed5d-25ea-1839-28d7-ad4dfa5f6c86"
GPU0_ALLOWED = False
GPU0_DISPLAY_MEMORY_MIB_ALLOWED = 453
DEVICE_NAME = "NVIDIA GeForce RTX 3090"
ELIGIBLE_GPU_INDICES = (GPU1_INDEX,)

HYDRA_TRAIN_ENTRY_RELATIVE = "streaming_calibration_exp/src/train.py"
DATA_DIR_RELATIVE = "SPINT-main/data/000941"

ARM_SPECS = {
    "b3": {
        "experiment": "m1_b3_allsource_b3",
        "variant": "B3",
        "side_feature_group": "none",
        "side_dim": 0,
        "freeze_decoder": False,
        "loss_mode": "task_only",
        "activity_prefix_cycle": None,
        "role": "our encoder, no T4, joint decoder; strongest beat-original candidate",
    },
    "b3s_t4": {
        "experiment": "m1_b3_allsource_b3s_t4",
        "variant": "B3S",
        "side_feature_group": "t4",
        "side_dim": 4,
        "freeze_decoder": False,
        "loss_mode": "task_only",
        "activity_prefix_cycle": None,
        "role": "abandoned vanilla T4 control; do not relaunch as wave-1",
    },
    "b3s_t4_encoder": {
        "experiment": "m1_b3_allsource_b3s_t4_encoder",
        "variant": "B3S",
        "side_feature_group": "t4",
        "side_dim": 4,
        "freeze_decoder": True,
        "loss_mode": "task_only",
        "activity_prefix_cycle": None,
        "role": "encoder-only T4 control vs the previous fold-1 frozen-decoder T4",
    },
    "b3s_rsyn3": {
        "experiment": "m1_b3_allsource_b3s_rsyn3",
        "variant": "B3S",
        "side_feature_group": "rsyn3",
        "side_dim": 4,
        "freeze_decoder": False,
        "loss_mode": "task_only",
        "activity_prefix_cycle": None,
        "role": "our encoder plus EMG-rSyn3 carrier (relu+NMF3+ridge), joint decoder",
    },
    "b3s_rsyn3_freeze": {
        "experiment": "m1_b3_allsource_b3s_rsyn3_freeze",
        "variant": "B3S",
        "side_feature_group": "rsyn3",
        "side_dim": 4,
        "freeze_decoder": True,
        "loss_mode": "task_only",
        "activity_prefix_cycle": None,
        "role": "wave-2: freeze 578244 decoder, train rSyn3 encoder only",
    },
    "b3s_rsyn3_acyc": {
        "experiment": "m1_b3_allsource_b3s_rsyn3_acyc",
        "variant": "B3S",
        "side_feature_group": "rsyn3",
        "side_dim": 4,
        "freeze_decoder": False,
        "loss_mode": "task_only",
        "activity_prefix_cycle": (10, 5, 2),
        "role": "wave-2: joint decoder, train-time calib prefix cycle (10,5,2); eval stays M10",
    },
    "b3s_rsyn3_freeze_top4": {
        "experiment": "m1_b3_allsource_b3s_rsyn3_freeze",
        "variant": "B3S",
        "side_feature_group": "rsyn3",
        "side_dim": 4,
        "freeze_decoder": True,
        "loss_mode": "task_only",
        "activity_prefix_cycle": None,
        "source_train_arm": "b3s_rsyn3_freeze",
        "carrier_method": "dopt_tgt_loc",
        "carrier_k": 4,
        "encoder_neural_trials": 10,
        "role": "package-only: freeze student, M10 neural identity, D-opt tgt_loc k=4 carrier",
    },
    "b3s_rsyn3_acyc_top4": {
        "experiment": "m1_b3_allsource_b3s_rsyn3_acyc",
        "variant": "B3S",
        "side_feature_group": "rsyn3",
        "side_dim": 4,
        "freeze_decoder": False,
        "loss_mode": "task_only",
        "activity_prefix_cycle": (10, 5, 2),
        "source_train_arm": "b3s_rsyn3_acyc",
        "carrier_method": "dopt_tgt_loc",
        "carrier_k": 4,
        "encoder_neural_trials": 10,
        "role": "package-only: acyc student, M10 neural identity, D-opt tgt_loc k=4 carrier",
    },
}

ACTIVITY_CYCLE = (10, 5, 2)
TRAIN_ARMS = (
    "b3", "b3s_rsyn3", "b3s_t4", "b3s_t4_encoder",
    "b3s_rsyn3_freeze", "b3s_rsyn3_acyc",
)
WAVE1_ARMS = ("b3", "b3s_rsyn3")
WAVE2_ARMS = ("b3s_rsyn3_freeze", "b3s_rsyn3_acyc")
M4_QUERY_ARMS = WAVE1_ARMS + WAVE2_ARMS
TOP4_PACKAGE_ARMS = ("b3s_rsyn3_freeze_top4", "b3s_rsyn3_acyc_top4")
TOP4_CARRIER_METHOD = "dopt_tgt_loc"
TOP4_CARRIER_K = 4
PACKAGE_ARMS = TRAIN_ARMS + TOP4_PACKAGE_ARMS
REMOTE_ARMS = WAVE1_ARMS + TOP4_PACKAGE_ARMS
MODES = ("train", "package")

READ_RULES = {
    "schema": "m1_b3_allsource_read_rules_v1",
    "comparison_target": {
        "surface": "official_heldout_r2",
        "original_spint": OFFICIAL_ORIGINAL_HELDOUT_R2,
        "submission_id": OFFICIAL_ORIGINAL_SUBMISSION,
        "n_train_sessions": 4,
    },
    "not_comparable": {
        "fold_local_20120924_after_all_source_fit": (
            "20120924 is in the train roster; a post-fit number is not official held-out"
        ),
        "fold_local_loso_zfix_0.637": "three-source student, not 4-source EvalAI",
        "evalai_t4_0.644766": OFFICIAL_T4_NOT_FAIR_NOTE,
    },
    "formal_benchmark_verdict": False,
    "checkpoint_selection": "fixed 12-epoch train/loss only; no local val/test",
}

OWNED_PATHS: tuple[str, ...] = (
    "tfpd_exploration/src/m1_b3_allsource_v1/__init__.py",
    "tfpd_exploration/src/m1_b3_allsource_v1/plan.py",
    "tfpd_exploration/src/m1_b3_allsource_v1/gpu.py",
    "tfpd_exploration/src/m1_b3_allsource_v1/execute.py",
    "tfpd_exploration/src/m1_b3_allsource_v1/receipts.py",
    "tfpd_exploration/src/m1_b3_allsource_v1/package.py",
    "tfpd_exploration/src/m1_b3_allsource_v1/rsyn3_bank.py",
    "tfpd_exploration/src/m1_b3_allsource_v1/carrier_k.py",
    "tfpd_exploration/src/m1_b3_allsource_v1/carrier_k_probe.py",
    "tfpd_exploration/src/m1_b3_allsource_v1/carrier_k_later_day.py",
    "tfpd_exploration/src/m1_b3_allsource_v1/m4_query.py",
    "tfpd_exploration/src/m1_b3_allsource_v1/runtime.py",
    "tfpd_exploration/src/m1_b3_allsource_v1/decode_remote.py",
    "tfpd_exploration/src/m1_b3_allsource_v1/submit_remote.py",
    "tfpd_exploration/src/m1_b3_allsource_v1/Dockerfile",
    "tfpd_exploration/scripts/run_m1_b3_allsource_v1.py",
    "tfpd_exploration/scripts/run_m1_b3_allsource_m4_query.py",
    "tfpd_exploration/scripts/run_m1_b3_allsource_carrier_k_probe.py",
    "tfpd_exploration/scripts/run_m1_b3_allsource_carrier_k_later_day.py",
    "tfpd_exploration/scripts/run_m1_b3_allsource_remote.py",
    "tfpd_exploration/scripts/run_m1_b3_allsource_remote_timed.py",
    "tfpd_exploration/tests/test_m1_b3_allsource_v1.py",
    "streaming_calibration_exp/src/data/falcon_m1_all_source_b3_datamodule.py",
    "streaming_calibration_exp/src/data/falcon_m1_all_source_b3_rsyn3_datamodule.py",
    "streaming_calibration_exp/configs/data/falcon_m1_all_source_b3.yaml",
    "streaming_calibration_exp/configs/data/falcon_m1_all_source_b3s_t4.yaml",
    "streaming_calibration_exp/configs/data/falcon_m1_all_source_b3s_rsyn3.yaml",
    "streaming_calibration_exp/configs/model/streaming_b3s_rsyn3_m1.yaml",
    "streaming_calibration_exp/configs/experiment/m1_b3_allsource_b3.yaml",
    "streaming_calibration_exp/configs/experiment/m1_b3_allsource_b3s_t4.yaml",
    "streaming_calibration_exp/configs/experiment/m1_b3_allsource_b3s_t4_encoder.yaml",
    "streaming_calibration_exp/configs/experiment/m1_b3_allsource_b3s_rsyn3.yaml",
    "streaming_calibration_exp/configs/experiment/m1_b3_allsource_b3s_rsyn3_freeze.yaml",
    "streaming_calibration_exp/configs/experiment/m1_b3_allsource_b3s_rsyn3_acyc.yaml",
    "streaming_calibration_exp/configs/callbacks/m1_all_source_rsyn3_acyc.yaml",
    "tfpd_exploration/src/m1_b3_allsource_v1/acyc.py",
)

SEALED_RESULT_ROOTS: tuple[str, ...] = (
    "tfpd_exploration/results/m1_t0c1_prefix_v1",
    "tfpd_exploration/results/m1_t0c1_prefix_v1_50ep",
    "tfpd_exploration/results/m1_emg_syn3_fcm_v1",
    "tfpd_exploration/results/m1_emg_rsyn3_fcm_v1",
    "tfpd_exploration/results/m1_heldin_heldout_gap_v1",
    "tfpd_exploration/results/m1_tier12_pack_v1",
    "tfpd_exploration/results/cross_session_worst_group_m1_source_full_v1_no_swa",
    "tfpd_exploration/results/cross_session_worst_group_m1_source_smoke_v6",
)

_require(SEED == 42 and EPOCHS == 12, "B3 student budget drift")
_require(ADAM_LR == 1.0e-4 and TRAIN_BATCH_SIZE == 32, "B3 optimizer/batch drift")
_require(SOURCE_SESSIONS == ("20120924", "20120926", "20120927", "20120928"), "source session drift")
_require(len(TRAIN_ARMS) == 6 and WAVE1_ARMS == ("b3", "b3s_rsyn3"), "wave-1 pack roster drift")
_require(WAVE2_ARMS == ("b3s_rsyn3_freeze", "b3s_rsyn3_acyc"), "wave-2 pack roster drift")
_require(M4_QUERY_ARMS == WAVE1_ARMS + WAVE2_ARMS, "M4 query arm roster drift")
_require(TOP4_PACKAGE_ARMS == ("b3s_rsyn3_freeze_top4", "b3s_rsyn3_acyc_top4"), "top4 package roster")
_require(TOP4_CARRIER_METHOD == "dopt_tgt_loc" and TOP4_CARRIER_K == 4, "top4 carrier recipe")
_require(PACKAGE_ARMS == TRAIN_ARMS + TOP4_PACKAGE_ARMS, "package roster drift")
_require(REMOTE_ARMS == WAVE1_ARMS + TOP4_PACKAGE_ARMS, "remote roster drift")
_require(ARM_SPECS["b3s_rsyn3_freeze_top4"]["source_train_arm"] == "b3s_rsyn3_freeze", "freeze top4 source")
_require(ARM_SPECS["b3s_rsyn3_acyc_top4"]["source_train_arm"] == "b3s_rsyn3_acyc", "acyc top4 source")
_require(ARM_SPECS["b3s_rsyn3_freeze_top4"]["carrier_k"] == 4, "freeze top4 k")
_require(ARM_SPECS["b3s_rsyn3_freeze_top4"]["encoder_neural_trials"] == 10, "top4 encoder stays M10")
_require(ACTIVITY_CYCLE == (10, 5, 2), "activity prefix cycle drift")
_require(ARM_SPECS["b3s_rsyn3_freeze"]["freeze_decoder"] is True, "freeze arm drift")
_require(ARM_SPECS["b3s_rsyn3_acyc"]["activity_prefix_cycle"] == ACTIVITY_CYCLE, "acyc arm drift")
_require(M4_QUERY_SUPPORT == 4 and M4_QUERY_START == 4, "M4 query protocol drift")
_require(M4_QUERY_LATER_DAY_QUERY_TRIALS == 6, "later-day remaining-query drift")
_require(M4_QUERY_INCLUDE_SOURCE_IN_TRAIN_DEFAULT is False, "source in-train is not the default M4 surface")
_require(CARRIER_K_ARM == "b3s_rsyn3", "carrier-k probe arm drift")
_require(CARRIER_K_BUDGETS == (3, 4, 5, 6), "carrier-k budget drift")
_require(CARRIER_K_POOL_TRIALS == 10 and CARRIER_K_QUERY_START == 10, "carrier-k pool/query drift")
_require(CARRIER_K_N_VARIANTS == 13, "carrier-k grid size drift")
_require("evalai" not in CARRIER_K_PROBE_ROOT_RELATIVE, "carrier-k path token")
_require(CARRIER_K_LATER_DAY_ARMS == ("b3s_rsyn3", "b3s_rsyn3_freeze"), "later-day carrier-k arms")
_require(CARRIER_K_LATER_DAY_SUPPORT == 4 and CARRIER_K_LATER_DAY_QUERY_START == 4, "later-day M4 protocol")
_require(CARRIER_K_LATER_DAY_BUDGETS == (2, 3), "later-day k=2,3; k=4 is chrono baseline only")
_require(CARRIER_K_LATER_DAY_N_VARIANTS == 7, "later-day carrier-k grid size")
_require("held-out" not in CARRIER_K_LATER_DAY_ROOT_RELATIVE, "later-day carrier-k path token")
_require("evalai" not in CARRIER_K_LATER_DAY_ROOT_RELATIVE, "later-day carrier-k path token")
_require(REMOTE_CHALLENGE_ID == 2319 and REMOTE_PHASE_ID == 4599, "remote challenge/phase drift")
_require(REMOTE_TEAM_ID == 41975 and REMOTE_BATCH_SIZE == 4, "remote team/batch drift")
_require("evalai" not in REMOTE_PUSH_ROOT_RELATIVE, "remote push path token")
_require(GPU0_ALLOWED is False and PACK_LIMIT == 2, "GPU pack contract drift")
_require(COSINE_50EP_FORBIDDEN is True and SCHEDULER is None, "scheduler contract drift")


def validate_mode_arm(mode: str, arm: str) -> None:
    _require(mode in MODES, f"unknown mode {mode!r}")
    if mode == "train":
        _require(arm in TRAIN_ARMS, f"cannot train {arm}")
        return
    _require(arm in PACKAGE_ARMS, f"cannot package {arm}")


def source_train_arm(arm: str) -> str:
    spec = ARM_SPECS.get(arm)
    _require(isinstance(spec, dict), f"unknown arm {arm}")
    source = str(spec.get("source_train_arm", arm))
    _require(source in TRAIN_ARMS, f"source train arm {source} is not trainable")
    return source


def gpu_uuid(gpu_index: int) -> str:
    if gpu_index == GPU0_INDEX:
        return GPU0_UUID
    if gpu_index == GPU1_INDEX:
        return GPU1_UUID
    raise PlanError(f"unsupported gpu index {gpu_index}")


def stage_relative(mode: str, arm: str) -> str:
    validate_mode_arm(mode, arm)
    if mode == "train":
        return arm
    return f"export/{arm}"


def parent_relative_for_stage(mode: str) -> str | None:
    if mode == "package":
        return f"{RESULT_ROOT_RELATIVE}/export"
    return RESULT_ROOT_RELATIVE


def train_root_relative(arm: str) -> str:
    _require(arm in TRAIN_ARMS, f"unknown train arm {arm}")
    return f"{RESULT_ROOT_RELATIVE}/{arm}"


def package_root_relative(arm: str) -> str:
    _require(arm in PACKAGE_ARMS, f"unknown package arm {arm}")
    return f"{RESULT_ROOT_RELATIVE}/export/{arm}"


def m4_query_root_relative(arm: str) -> str:
    _require(arm in M4_QUERY_ARMS, f"M4 query is not defined for {arm}")
    return f"{M4_QUERY_ROOT_RELATIVE}/{arm}"


def remote_push_root_relative(arm: str) -> str:
    _require(arm in REMOTE_ARMS, f"remote push is not defined for {arm}")
    return f"{REMOTE_PUSH_ROOT_RELATIVE}/{arm}"


def carrier_k_probe_root_relative() -> str:
    return f"{CARRIER_K_PROBE_ROOT_RELATIVE}/{CARRIER_K_ARM}"


def carrier_k_later_day_root_relative(arm: str) -> str:
    _require(arm in CARRIER_K_LATER_DAY_ARMS, f"later-day carrier-k is not defined for {arm}")
    return f"{CARRIER_K_LATER_DAY_ROOT_RELATIVE}/{arm}"


@dataclass(frozen=True)
class PairSpec:
    root_relative: str = RESULT_ROOT_RELATIVE

    def payload(self) -> dict[str, object]:
        return {
            "schema": "m1_b3_allsource_pair_spec_v1",
            "cell": CELL,
            "phase": PHASE,
            "root_relative": self.root_relative,
            "train_arms": list(TRAIN_ARMS),
            "wave1_arms": list(WAVE1_ARMS),
            "wave2_arms": list(WAVE2_ARMS),
            "arm_specs": {key: dict(value) for key, value in ARM_SPECS.items()},
            "budget": {
                "seed": SEED,
                "epochs": EPOCHS,
                "batch_size": TRAIN_BATCH_SIZE,
                "optimizer": "Adam",
                "adam_lr": ADAM_LR,
                "adam_weight_decay": ADAM_WEIGHT_DECAY,
                "scheduler": SCHEDULER,
                "cosine_50ep_forbidden": COSINE_50EP_FORBIDDEN,
                "loss_mode": "task_only",
            },
            "sessions": {
                "source": list(SOURCE_SESSIONS),
                "later_calib_export_only": list(LATER_CALIB_SESSIONS),
                "public_calib_package": list(PUBLIC_CALIB_SESSIONS),
            },
            "teacher": {
                "path": TEACHER_CHECKPOINT_RELATIVE,
                "sha256": TEACHER_SHA256,
            },
            "gpu": {
                "gpu0_uuid": GPU0_UUID,
                "gpu1_uuid": GPU1_UUID,
                "pack_limit": PACK_LIMIT,
                "own_token": OWN_TOKEN,
                "gpu0_allowed": GPU0_ALLOWED,
                "gpu0_display_memory_mib_allowed": GPU0_DISPLAY_MEMORY_MIB_ALLOWED,
                "device": DEVICE_NAME,
                "gpu_index_choices": [GPU1_INDEX],
            },
            "read_rules": dict(READ_RULES),
            "formal_benchmark_verdict": False,
            "public_execution_authorized": True,
            "hydra_entry": HYDRA_TRAIN_ENTRY_RELATIVE,
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
            **{arm: train_root_relative(arm) for arm in TRAIN_ARMS},
            **{f"package_{arm}": package_root_relative(arm) for arm in PACKAGE_ARMS},
            **{f"m4_query_{arm}": m4_query_root_relative(arm) for arm in M4_QUERY_ARMS},
            **{f"remote_push_{arm}": remote_push_root_relative(arm) for arm in REMOTE_ARMS},
            "carrier_k_probe": carrier_k_probe_root_relative(),
            **{
                f"carrier_k_later_day_{arm}": carrier_k_later_day_root_relative(arm)
                for arm in CARRIER_K_LATER_DAY_ARMS
            },
        },
        "carrier_k_probe": {
            "arm": CARRIER_K_ARM,
            "methods": list(CARRIER_K_METHODS),
            "budgets": list(CARRIER_K_BUDGETS),
            "pool_trials": CARRIER_K_POOL_TRIALS,
            "query_start_trial": CARRIER_K_QUERY_START,
            "encoder_uses_all_pool_neural": True,
            "n_variants": CARRIER_K_N_VARIANTS,
            "root_relative": carrier_k_probe_root_relative(),
            "note": CARRIER_K_NOTE,
            "formal_benchmark_verdict": False,
            "local_source_post_m10_is_in_train": True,
            "later_day_r2": None,
        },
        "carrier_k_later_day": {
            "arms": list(CARRIER_K_LATER_DAY_ARMS),
            "methods": list(CARRIER_K_METHODS),
            "budgets": list(CARRIER_K_LATER_DAY_BUDGETS),
            "baseline_k": CARRIER_K_LATER_DAY_BASELINE_K,
            "support_trials": CARRIER_K_LATER_DAY_SUPPORT,
            "query_start_trial": CARRIER_K_LATER_DAY_QUERY_START,
            "encoder_neural_trials": CARRIER_K_LATER_DAY_SUPPORT,
            "selector_pool_is_support_only": True,
            "n_variants": CARRIER_K_LATER_DAY_N_VARIANTS,
            "root_relative": CARRIER_K_LATER_DAY_ROOT_RELATIVE,
            "note": CARRIER_K_LATER_DAY_NOTE,
            "formal_benchmark_verdict": False,
            "local_source_post_m10_is_in_train": True,
        },
        "m4_query": {
            "support_trials": M4_QUERY_SUPPORT,
            "query_start_trial": M4_QUERY_START,
            "later_day_query_trials": M4_QUERY_LATER_DAY_QUERY_TRIALS,
            "include_source_in_train_default": M4_QUERY_INCLUDE_SOURCE_IN_TRAIN_DEFAULT,
            "root_relative": M4_QUERY_ROOT_RELATIVE,
            "note": M4_QUERY_NOT_OFFICIAL,
            "formal_benchmark_verdict": False,
        },
        "pair_spec": spec.payload(),
        "train_arms": list(TRAIN_ARMS),
        "wave1_arms": list(WAVE1_ARMS),
        "wave2_arms": list(WAVE2_ARMS),
        "top4_package_arms": list(TOP4_PACKAGE_ARMS),
        "remote_arms": list(REMOTE_ARMS),
        "pack_limit": PACK_LIMIT,
        "cosine_50ep_forbidden": COSINE_50EP_FORBIDDEN,
        "scheduler": SCHEDULER,
        "source_sessions": list(SOURCE_SESSIONS),
        "four_session_all_source_in_train_roster": True,
        "local_20120924_is_not_official_heldout": True,
        "official_original_heldout_r2": OFFICIAL_ORIGINAL_HELDOUT_R2,
        "pack_optional_for_single_occupancy": True,
        "public_execution_authorized": True,
        "opens_nwb_or_checkpoint": False,
        "imports_torch": False,
        "initializes_cuda": False,
        "creates_root_or_receipt": False,
        "cli": OWN_TOKEN,
        "execute_requires": ["--gpu-authorized", "--gpu-index 1", "--mode train", "--arm"],
        "pack": "optional; omit for single occupancy. Wave-1/wave-2 launches use --pack on GPU 1 only.",
        "public_gpu_capability": False,
        "gpu": {
            "gpu0_allowed": GPU0_ALLOWED,
            "pack_limit": PACK_LIMIT,
            "eligible_indices": list(ELIGIBLE_GPU_INDICES),
            "own_token": OWN_TOKEN,
        },
        "formal_benchmark_verdict": False,
    }


def implementation_closure(root: Path) -> dict[str, object]:
    rows = []
    for relative in OWNED_PATHS:
        path = Path(root) / relative
        _require(path.is_file() and not path.is_symlink(), f"closure leaf missing: {relative}")
        rows.append({"path": relative, "sha256": sha256_bytes(path.read_bytes())})
    body = {
        "schema": "m1_b3_allsource_closure_v1",
        "result_root_relative": RESULT_ROOT_RELATIVE,
        "paths": rows,
    }
    return {**body, "closure_sha256": sha256_bytes(canonical_json_bytes(body))}


__all__ = (
    "PlanError",
    "CELL",
    "PHASE",
    "RESULT_ROOT_RELATIVE",
    "TRAIN_ARMS",
    "WAVE1_ARMS",
    "PACK_LIMIT",
    "OWN_TOKEN",
    "SEED",
    "EPOCHS",
    "PairSpec",
    "dry_plan",
    "validate_mode_arm",
    "canonical_json_bytes",
    "sha256_bytes",
)
