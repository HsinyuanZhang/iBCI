"""Frozen pre-registration for PIT-M2: the C1 prefix-cycle training pair on
M2's own sealed training line (prefix-invariance training, matched pair).

Everything that could bias the matched pair or the 2x2 readout is frozen here
before any data or model access:

* WHICH sealed M2 trainer this lane binds to (the only trainer that produced
  the ``m2_t4_activity_budget`` checkpoints, the ``25d7bc72`` family), pinned
  by immutable-artifact digests because the streaming_calibration_exp source
  tree is LIVE (shared with other routes; bytes recorded per attempt, never
  edited by this lane);
* the pre-declared prefix cycle ``(10, 5, 2)`` and its single-budget M10 law;
* the 2x2 factorial cells F00m/F10m/F01m/F11m, with F00m/F01m SEALED rows
  referenced (never rerun);
* the primary gate (F10m - F00m >= +0.01 on the official-contract static
  surface with predeclared breadth) and the stacking-reading secondary
  (F11m - F01m, reported with NO promotion attached);
* the verified SPINT-original dropout finding for THIS trainer (the shared
  dropout block is present but UNREACHABLE under the sealed recipe's
  ``freeze_decoder=true``);
* the source-preparation law (the sealed Hydra config, the chronological
  first-33 block, the train-only T4 normalizer that must reproduce
  ``d17f5f4c...`` bit-exactly);
* the CPU smoke equality contract (N=40 steps, no CUDA at all).
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping


class PitM2PlanError(RuntimeError):
    """Fail closed for static pair contract, literals, or closure drift."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PitM2PlanError(message)


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require_sha(value: object, label: str) -> str:
    _require(isinstance(value, str) and len(value) == 64
             and all(character in "0123456789abcdef" for character in value),
             f"pit m2 sha literal drift: {label}")
    return str(value)


def safe_relative(value: object) -> str:
    """Repo-relative path with no ``..``, no absolute, no symlink escape."""
    _require(isinstance(value, str) and value and not value.startswith("/"),
             "pit m2 relative path must be a non-empty relative string")
    parts = [part for part in value.split("/") if part not in ("", ".")]
    _require(all(part != ".." for part in parts), "pit m2 relative path escapes the repo root")
    return "/".join(parts)


SCHEMA = "pit_m2_v1"
PHASE = "pit_m2_v1"
CELL = "PIT_M2_PREFIX_INVARIANCE_TRAINING_V1"
RESULT_ROOT_RELATIVE = "tfpd_exploration/results/pit_m2_v1"
SMOKE_ROOT_RELATIVE = f"{RESULT_ROOT_RELATIVE}/smoke"
ARM_ROOT_RELATIVE = {"t0m": f"{RESULT_ROOT_RELATIVE}/t0m", "c1m": f"{RESULT_ROOT_RELATIVE}/c1m"}
PHASE3_ROOT_RELATIVE = f"{RESULT_ROOT_RELATIVE}/phase3_table"

# ---------------------------------------------------------------------------
# The sealed M2 trainer this lane binds to (the 25d7bc72 checkpoint family).
# ---------------------------------------------------------------------------

#: The single run whose ``best.ckpt`` IS the sealed ``m2_t4_activity_budget``
#: checkpoint (verified by sha256 against every sealed M2 screen plan).
SEALED_RUN_ID = "m2_spint_t4_mainline_fp32_v1_t4_m2_s42_20260730_131806"
SEALED_RUN_DIR_RELATIVE = (
    "streaming_calibration_exp/outputs/streaming_calibration/"
    "m2_spint_t4_mainline_fp32_v1_t4_m2_s42_20260730_131806"
)
SEALED_CHECKPOINT_RELATIVE = f"{SEALED_RUN_DIR_RELATIVE}/checkpoints/best.ckpt"
SEALED_CHECKPOINT_SHA256 = "25d7bc72b4d440004b58f1beaeadb7e15565a43e83dd1eadd160374270ec1d3e"
SEALED_RESOLVED_CONFIG_RELATIVE = f"{SEALED_RUN_DIR_RELATIVE}/resolved_config.yaml"
SEALED_RESOLVED_CONFIG_SHA256 = "2c331befba13e518efdfd2028bbb13e4243287ad5af6b32d425e3c19dbaf3b2d"
SEALED_RUN_METADATA_RELATIVE = f"{SEALED_RUN_DIR_RELATIVE}/run_metadata.json"
TEACHER_CHECKPOINT_RELATIVE = (
    "SPINT-main/logs/train/runs/2026-07-07-16-05-16/checkpoints/best_ckpt/epoch_034.ckpt"
)
TEACHER_CHECKPOINT_SHA256 = "fbcb9914561c4664fa0f8d0b1791e67505841d3ac470ea7ad68d54e408ca13ec"
NORMALIZATION_SHA256 = "d17f5f4c4d106b9f19493be6f5f06846c01e917f408f516e630f5e8f09d1539e"
M2_DATA_DIR_RELATIVE = "SPINT-main/data/000953"

#: file:line binding of the sealed trainer.  These source files are LIVE
#: (uncommitted shared-tree edits by other routes are normal here); the lane
#: therefore binds the RECIPE by the immutable artifacts above (resolved
#: config, checkpoint, teacher, normalizer) and records the live source bytes
#: in every attempt receipt.  This lane never edits any of them.
TRAINER_BINDING = {
    "schema": "pit_m2_sealed_trainer_binding_v1",
    "trainer_entry": {
        "path": "streaming_calibration_exp/src/train.py",
        "lines": "218-304",
        "what": "hydra train(): L.seed_everything(seed, workers=True); instantiate(cfg.data); "
                "instantiate(cfg.model); trainer.fit(model, datamodule)",
    },
    "launch_command": {
        "path": "streaming_calibration_exp/scripts/run_m2_spint_t4_mainline.sh",
        "lines": "38-56",
        "what": "experiment=b3s_t4_m2_loso_internal, run_id=m2_spint_t4_mainline_fp32_v1_t4_m2, "
                "seed=42, data.validation_protocol=minival, data.loso_fold=null, "
                "data.include_heldout_in_fit=false, data.include_heldout_in_test=true, "
                "data.random_calibration=false, data.calibration_n_trials=33, "
                "trainer.max_epochs=12, trainer.accelerator=gpu, trainer.devices=1",
    },
    "experiment_config": "streaming_calibration_exp/configs/experiment/b3s_t4_m2_loso_internal.yaml",
    "model_config": "streaming_calibration_exp/configs/model/streaming_b3s_t4.yaml",
    "lit_module": {
        "path": "streaming_calibration_exp/src/models/streaming_calibration_module.py",
        "lines": "539-718",
        "what": "StreamingCalibrationLitModule.model_step/training_step: student forward "
                "self.student(neural, calib_trials=calib, side_features=..., ...) and the "
                "no_grad eval-mode teacher targets from the FULL calib block",
    },
    "student_module": {
        "path": "streaming_calibration_exp/src/models/components/streaming_spint.py",
        "lines": "255-263,682-721",
        "what": "StreamingSpintModel.compute_identity/forward: calib_trials kwarg is the "
                "seam the PIT operator rewrites",
    },
    "datamodule": "streaming_calibration_exp/src/data/falcon_datamodule.py",
    "checkpoint_lineage": {
        "sealed_checkpoint_sha256": SEALED_CHECKPOINT_SHA256,
        "sealed_selected_by": "val_heldin/r2_mean (max)",
        "sealed_best_epoch_0based": 2,
        "sealed_selected_metric_value": 0.6472744345664978,
    },
    "live_tree_disclosure": (
        "the streaming_calibration_exp and SPINT-main trees are shared and LIVE; this lane "
        "records their bytes per attempt and never edits them; the scientific binding is the "
        "immutable artifact sha256 set above"
    ),
}

#: Live source files whose bytes every attempt receipt records (no pinning).
LIVE_SOURCE_RELATIVE = (
    "streaming_calibration_exp/src/train.py",
    "streaming_calibration_exp/scripts/run_m2_spint_t4_mainline.sh",
    "streaming_calibration_exp/configs/experiment/b3s_t4_m2_loso_internal.yaml",
    "streaming_calibration_exp/configs/model/streaming_b3s_t4.yaml",
    "streaming_calibration_exp/configs/model/_streaming_base.yaml",
    "streaming_calibration_exp/configs/data/falcon_m2.yaml",
    "streaming_calibration_exp/src/models/streaming_calibration_module.py",
    "streaming_calibration_exp/src/models/components/streaming_spint.py",
    "streaming_calibration_exp/src/models/components/streaming_encoders.py",
    "streaming_calibration_exp/src/models/components/spint.py",
    "streaming_calibration_exp/src/data/falcon_datamodule.py",
    "SPINT-main/src/models/components/spint.py",
)

#: Immutable sealed anchors verified fail-closed at every attempt.
IMMUTABLE_ANCHOR_RELATIVE = (
    SEALED_CHECKPOINT_RELATIVE,
    SEALED_RESOLVED_CONFIG_RELATIVE,
    SEALED_RUN_METADATA_RELATIVE,
    TEACHER_CHECKPOINT_RELATIVE,
    "tfpd_exploration/results/m2_t4_activity_budget_screen_v1/score.json",
    "tfpd_exploration/results/m2_precision_cdm_v2_screen_v1/score.json",
)
IMMUTABLE_ANCHOR_SHA256 = {
    SEALED_CHECKPOINT_RELATIVE: SEALED_CHECKPOINT_SHA256,
    SEALED_RESOLVED_CONFIG_RELATIVE: SEALED_RESOLVED_CONFIG_SHA256,
    TEACHER_CHECKPOINT_RELATIVE: TEACHER_CHECKPOINT_SHA256,
    "tfpd_exploration/results/m2_t4_activity_budget_screen_v1/score.json":
        "6bdad93328ba26c12b8aa3afffcbc3490c312dfe22939b3d3bb3c5ec5b9005ce",
    "tfpd_exploration/results/m2_precision_cdm_v2_screen_v1/score.json":
        "455485bd854a36392f17ec5ed029b1a4044a01a8dd7dfeae95c7763b8327f5f6",
}

# ---------------------------------------------------------------------------
# Arms and the pre-declared prefix cycle.
# ---------------------------------------------------------------------------

ARMS = ("t0m", "c1m")
ARM_ROLES = {
    "t0m": "matched control: SAME sealed trainer, operator registered but disabled "
           "(kwargs unchanged, full first-33 block every step)",
    "c1m": "deterministic chronological calibration-prefix cycle on the student "
           "training forward (the PIT arm)",
}

#: The M2 trainer's native training calibration block (sealed recipe literal).
TRAINING_CALIBRATION_N_TRIALS = 33
#: The single deployment budget of this line (the sealed static M10 family).
DEPLOYMENT_BUDGET_M = 10
MAX_CALIBRATION_TRIALS = TRAINING_CALIBRATION_N_TRIALS
#: Frozen M2 geometry (the sealed screens' literals).
WINDOW_SIZE = 50
TRIAL_LENGTH = 100
CHANNELS = 96
OUTPUT_DIM = 2
BEHAVIOR_SCALE = 5.0
RIDGE_NORMALIZED_LAMBDA = 0.1

#: Pre-declared operator cycle, mirroring the M1 precedent.  M2's own T4 line
#: deploys at a SINGLE budget -- the sealed static M10 support (first-10
#: chronological) and its M10 FIFO mirror -- so per the {full, half, quarter}
#: law the cycle is (10, 5, 2): full = the M10 deployment budget, half = 5,
#: quarter floors 10/4 = 2.5 to the integer 2 (disclosed, never rounded up).
#: The trainer's native block stays the untouched first-33 (the t0m control
#: sees all 33 every step); the c1m arm NEVER sees beyond the first 10.
CYCLE = (10, 5, 2)
CYCLE_LAW = {
    "schema": "pit_m2_prefix_cycle_law_v1",
    "basis": (
        "M2 is a single-budget M10 line: the sealed deployment surfaces of the "
        "25d7bc72 family are the static M10 support (chronological first-10) and its "
        "m10 activity FIFO; no M30 deployment exists for this lane"
    ),
    "declared_cycle": list(CYCLE),
    "terms": {"full": 10, "half": 5, "quarter_floored": 2},
    "quarter_disclosure": "10/4 = 2.5 floored to the integer prefix 2; never rounded up",
    "training_block_disclosure": (
        "the sealed trainer's native block is the chronological first-33 "
        "(calibration_n_trials=33); t0m sees the full 33 every step, c1m sees at "
        "most the first 10 (the cycle full term equals the M10 deployment budget, "
        "mirroring the M1 native-M10 precedent where full was the whole block)"
    ),
    "advance": "one student training forward per optimizer step; M = cycle[step % 3] "
               "(deterministic, no RNG)",
    "training_gated": True,
    "eval_untouched": True,
    "teacher_target_law": (
        "the no-grad eval-mode teacher targets are computed from the FULL "
        "first-33 block in BOTH arms (distillation target unchanged; the operator "
        "touches only the student's calib_trials kwarg)"
    ),
}

# ---------------------------------------------------------------------------
# The frozen sealed-recipe training budget (mirrors the sealed run exactly).
# ---------------------------------------------------------------------------

SEED = 42
EPOCHS = 12
TRAIN_BATCH_SIZE = 32
ADAM_LR = 1.0e-4
ADAM_WEIGHT_DECAY = 0.0
CHECKPOINT_MONITOR = "val_heldin/r2_mean"
CHECKPOINT_MODE = "max"
NO_EARLY_STOPPING = True
CHECK_VAL_EVERY_N_EPOCH = 1
GRADIENT_CLIP_VAL = 0.0
SCHEDULER = None
#: Deterministic per-epoch batch order (sealed: shuffle=True with
#: sampler_seed=42, reshuffle_train_sampler_each_epoch=false, balance=false).
SAMPLER_LAW = {
    "kind": "SessionBatchSampler(shuffle=True, seed=42, balance_sessions=false, "
            "reshuffle_each_epoch=false)",
    "drop_last": True,
    "batches_per_epoch_sealed": 3618,
    "train_windows_sealed": 115911,
    "note": "the batch order is a pure function of the dataset and seed 42; "
            "identical across arms by construction and proven by digest",
}
#: Deviations from the sealed GPU run, disclosed before launch:
DEVIATIONS = {
    "schema": "pit_m2_deviations_v1",
    "torch_version": (
        "VERIFIED 2026-09-01: under the mandated PYTHONNOUSERSITE=1 envelope the "
        "route environment (/home/xinyuan/miniconda3/envs/spint, python 3.10.15) "
        "provides torch 2.5.1.post303 + cu11.8 -- the SAME version family the "
        "sealed run recorded (torch=2.5.1.post303).  A user-site torch "
        "2.12.0+cu130 shadows the env when user-site is enabled and cannot even "
        "initialize CUDA on this driver; that path is exactly what the "
        "PYTHONNOUSERSITE=1 envelope forbids, so the deviation is the envelope "
        "requirement itself, not a version change"
    ),
    "display_only_callbacks": (
        "RichModelSummary/LearningRateMonitor/OverrideEpochStepCallback are "
        "display-only; the train stage keeps ModelCheckpoint semantics "
        "(monitor val_heldin/r2_mean, mode max) implemented in the route "
        "receipt rather than in a hydra output directory"
    ),
    "lightning_deterministic_flag": (
        "kept at the sealed trainer's value (false); matched-pair equality is "
        "proven by the CPU smoke digests inside one process, not assumed"
    ),
    "sanity_check_disabled_in_full_arms": (
        "the sealed trainer used Lightning's default num_sanity_val_steps=2; the "
        "route receipt's best-checkpoint law would otherwise record a sanity-check "
        "metric row before epoch 0 (Lightning's own ModelCheckpoint ignores "
        "sanity-checking by construction), so the full arms set "
        "num_sanity_val_steps=0; the sanity check never touches training state, "
        "and both arms share the identical loop"
    ),
    "parallel_single_arm_processes": (
        "operator amendment 2026-09-02: the two arms may run as two parallel "
        "processes on one card; each process re-seeds itself via "
        "L.seed_everything(42) inside prepare(), so the arms never share RNG "
        "state (the M1 50ep precedent ran the same way)"
    ),
}

HARD_TIMEOUT_SECONDS_PER_ARM = 2 * 3600
#: Target-GPU-idle thresholds (operator amendment 2026-09-02).
GPU_IDLE_MEMORY_THRESHOLD_MIB = 100
GPU_IDLE_UTILIZATION_PERCENT = 0
#: Sealed 12-epoch run measured ~19 min wall on one RTX 3090 including test;
#: the per-arm bound keeps >6x headroom.
OPERATOR_RESOLUTION = {
    "schema": "pit_m2_operator_resolution_v2",
    "decision": (
        "the train stage REFUSES to run without --gpu-authorized AND a live "
        "TARGET-GPU-idle check (the target card must have < "
        f"{GPU_IDLE_MEMORY_THRESHOLD_MIB} MiB used, 0% utilization, and NO "
        "compute apps); GPUs 0 and 1 were occupied by other routes at design "
        "time (GPU 0 mainline, GPU 1 the 50-epoch M1 pair)"
    ),
    "order": ["t0m", "c1m"],
    "gpu_selection": "--gpu-index {0,1} selects the single visible card at launch",
    "amendment_20260902": {
        "authority": "operator order 2026-09-02 (GPU 1 freed after the 50ep M1 pair)",
        "changes": [
            "the GPU guard is relaxed from BOTH-idle to TARGET-GPU-idle: only the "
            "selected card must be idle; a busy other card no longer refuses the run",
            "a --arm {t0m,c1m} single-arm mode is added so the two arms may run as "
            "two PARALLEL processes on the same card (per-process seeds; the M1 50ep "
            "precedent ran the same way); the serial default path is unchanged",
        ],
        "guard_still_refuses": (
            "if the target card has compute apps, or >= "
            f"{GPU_IDLE_MEMORY_THRESHOLD_MIB} MiB used, or >0% utilization"
        ),
        "phase3_unchanged": "needs both arms COMPLETE; runs after the pair",
    },
}

# ---------------------------------------------------------------------------
# The verified SPINT-original dropout finding for THIS trainer.
# ---------------------------------------------------------------------------

#: Verified 2026-09-01 by direct source inspection AND by the live CPU smoke
#: preflight: the SPINT-original whole-unit dynamic-dropout block exists
#: byte-identical in both trees.  On the M2 sealed recipe the two consumers
#: behave differently:
#:   * the STUDENT's copy is UNREACHABLE (``freeze_decoder=true`` pins
#:     ``_decoder_frozen=True`` and the branch is ``if not
#:     self._decoder_frozen:``) -- zero dropout-p draws from the student;
#:   * the TEACHER (eval, no_grad, one target forward per training step)
#:     draws ``p = random.uniform(low, high)`` UNCONDITIONALLY (the draw sits
#:     before the ``training=`` gate) but the mask is INERT in eval
#:     (``F.dropout(..., training=self.training)`` with training=False leaves
#:     the all-ones mask unchanged) -- exactly ONE inert dropout-p draw per
#:     training step.
#: That teacher draw IS the pair's dropout-p stream, and it is arm-independent
#: (the operator never touches the teacher path), so equality of the per-step
#: python RNG-state digests across arms IS equality of the dropout-p stream --
#: the exact cal_aug_v1/m1 law, now carried on M2.
DROPOUT_PROOF = {
    "schema": "pit_m2_spint_original_dropout_proof_v1",
    "law": "whole-unit Bernoulli dropout mask F.dropout(ones(B,N), p) applied to src "
           "after the identity addition and before fc_in; with dynamic_dropout, "
           "p = random.uniform(low, high) drawn once per forward; the MASK is "
           "inactive whenever the module is not in training mode",
    "spint_original_block": {
        "path": "SPINT-main/src/models/components/spint.py",
        "lines": "131-137",
        "sha256_at_design": "31ed62f79bb28005a0635cc9b75c34412edfb3b8bfd96076df736e27df36a729",
    },
    "streaming_teacher_block": {
        "path": "streaming_calibration_exp/src/models/components/spint.py",
        "lines": "449-455",
        "sha256_at_design": "31ed62f79bb28005a0635cc9b75c34412edfb3b8bfd96076df736e27df36a729",
        "byte_identical_with_spint_original": True,
    },
    "student_gating_block": {
        "path": "streaming_calibration_exp/src/models/components/streaming_spint.py",
        "lines": "336-345",
        "sha256_at_design": "01a195994555e40c884f237b4d60f5ebc654101ed5c0e34bfd25bbf4cb8bdbef",
        "gating": "if not self._decoder_frozen: -> dynamic/dropout_rate branches",
    },
    "m2_recipe_finding": {
        "sealed_freeze_decoder": True,
        "student_dropout_branch_reachable": False,
        "student_dropout_p_draws_per_training_step": 0,
        "student_decoder_pinned_eval": True,
        "teacher_mode": "eval() under torch.no_grad(), one target forward per training step",
        "teacher_dropout_p_draws_per_training_step": 1,
        "teacher_draw_is_inert_in_eval": True,
        "teacher_draw_is_arm_independent": True,
        "conclusion": (
            "the SPINT-original dropout law is CARRIED verbatim from the m1/cal_aug "
            "precedents: the pair's dropout-p stream is the teacher's one "
            "random.uniform per training step (inert in eval, identical across "
            "arms); the student itself draws none; equality of the per-step "
            "python RNG-state digests across arms IS equality of the dropout-p "
            "stream, and the counting probe pins the draw count at exactly one "
            "per step in both arms"
        ),
    },
    "override_applied": False,
    "verified_before_launch": True,
}

# ---------------------------------------------------------------------------
# Source preparation law (no target path resolved).
# ---------------------------------------------------------------------------

SOURCE_PREPARATION_LAW = {
    "schema": "pit_m2_source_preparation_law_v1",
    "config": (
        "instantiate the SEALED resolved_config.yaml data and model nodes "
        "(sha256 " + SEALED_RESOLVED_CONFIG_SHA256[:16] + "...) with "
        "teacher_ckpt_path and data_dir bound to the immutable anchors"
    ),
    "seed": "L.seed_everything(42, workers=True) before every arm construction",
    "setup": "datamodule.prepare_data(); datamodule.setup('fit'); litmodule.setup('fit')",
    "normalizer": (
        "the train-only T4 normalizer is re-fit and MUST reproduce sha256 "
        + NORMALIZATION_SHA256 + " bit-exactly (proves the data pipeline and "
        "config binding end to end)"
    ),
    "sessions": "the 7 M2 held-in sessions (calib + minival NWBs) only",
    "no_target_path_resolved": (
        "stage 'fit' never builds val_heldout_dataset (include_heldout_in_fit=false); "
        "the held-out calib/query NWBs of the 6 external sessions are never opened "
        "in the smoke or train stages"
    ),
    "chronological_calibration": "random_calibration=false; calib_start_trial_idx=0 (first-33)",
}

# ---------------------------------------------------------------------------
# The 2x2 factorial and the gates.
# ---------------------------------------------------------------------------

#: first digit: model/training axis (0 = the sealed 25d7bc72 lineage, referenced;
#: 1 = fresh PIT-trained c1m arm).  second digit: deployment path (0 = the
#: official-contract static surface, 1 = the local FIFO path).
FACTORIAL_AXIS_TRAIN = {
    0: "the sealed 25d7bc72 checkpoint lineage (SEALED rows, referenced not rerun)",
    1: "the fresh PIT-trained c1m arm (this lane's sole trained intervention arm)",
}
FACTORIAL_AXIS_DEPLOYMENT = {
    0: "official-contract static: external_official_query, chronological first-10 "
       "support, static selected-M activity rows, frozen ridge T4 side "
       "(the sealed m2_t4_activity_budget_screen_v1 ridge_static_m10 law verbatim)",
    1: "local FIFO: external_post30_local, the m10 activity-only linear-B3S FIFO "
       "advancing on every completed valid trial (the sealed "
       "m2_precision_cdm_v2_screen_v1 m10_activity_only law verbatim)",
}
FACTORIAL_CELLS = ("F00m", "F10m", "F01m", "F11m")
SEALED_CELLS = ("F00m", "F01m")
FRESH_CELLS = ("F10m", "F11m")

#: F00m: the sealed static M10 row (referenced).
F00M_ANCHOR = {
    "schema": "pit_m2_f00m_sealed_anchor_v1",
    "path": "tfpd_exploration/results/m2_t4_activity_budget_screen_v1/score.json",
    "sha256": "6bdad93328ba26c12b8aa3afffcbc3490c312dfe22939b3d3bb3c5ec5b9005ce",
    "cell": "ridge_static_m10",
    "surface": "external_official_query",
    "checkpoint_sha256": SEALED_CHECKPOINT_SHA256,
    "session_count": 6,
    "equal_session_mean": 0.22325208564476937,
    "per_session_r2": {
        "ses-2020-10-30-Run1": 0.36977910664989855,
        "ses-2020-10-30-Run2": 0.2952468639789825,
        "ses-2020-11-18-Run1": 0.21234468116185035,
        "ses-2020-11-19-Run1": 0.11311615689251309,
        "ses-2020-11-24-Run1": 0.1983322727669885,
        "ses-2020-11-24-Run2": 0.1506934324183834,
    },
    "rerun_by_this_lane": False,
}

#: F01m: the sealed local-FIFO M10 row (referenced).
F01M_ANCHOR = {
    "schema": "pit_m2_f01m_sealed_anchor_v1",
    "path": "tfpd_exploration/results/m2_precision_cdm_v2_screen_v1/score.json",
    "sha256": "455485bd854a36392f17ec5ed029b1a4044a01a8dd7dfeae95c7763b8327f5f6",
    "cell": "m10_activity_only",
    "surface": "external_post30_local",
    "checkpoint_sha256": SEALED_CHECKPOINT_SHA256,
    "session_count": 6,
    "equal_session_mean": 0.2096870455654696,
    "per_session_r2": {
        "ses-2020-10-30-Run1": 0.377867393743974,
        "ses-2020-10-30-Run2": 0.2425326979671275,
        "ses-2020-11-18-Run1": 0.32266335346061614,
        "ses-2020-11-19-Run1": -0.022113591502448175,
        "ses-2020-11-24-Run1": 0.23228212536592038,
        "ses-2020-11-24-Run2": 0.10489029435762799,
    },
    "rerun_by_this_lane": False,
}

EXTERNAL_SESSION_COUNT = 6
WITHIN_SESSION_COUNT = 7
#: Frozen session rosters (the sealed screens' own rosters; verified against
#: the sealed anchors and the smoke authority).  Phase-3 publishes exactly one
#: score leaf per (arm, path, surface, session) -- the attempt-1 bug published
#: one leaf per (arm, path, surface) while looping over sessions, so the
#: second session collided O_EXCL with the first and the compensating cleanup
#: removed the first body (FileExistsError(17) after "1 cell completed").
EXTERNAL_SESSION_NAMES = (
    "ses-2020-10-30-Run1", "ses-2020-10-30-Run2", "ses-2020-11-18-Run1",
    "ses-2020-11-19-Run1", "ses-2020-11-24-Run1", "ses-2020-11-24-Run2",
)
WITHIN_SESSION_NAMES = (
    "ses-2020-10-19-Run1", "ses-2020-10-19-Run2", "ses-2020-10-20-Run1",
    "ses-2020-10-20-Run2", "ses-2020-10-27-Run1", "ses-2020-10-27-Run2",
    "ses-2020-10-28-Run1",
)
SURFACE_SESSION_ROSTERS = {
    "external_official_query": EXTERNAL_SESSION_NAMES,
    "external_post30_local": EXTERNAL_SESSION_NAMES,
    "within_post30": WITHIN_SESSION_NAMES,
}
PHASE3_ATTEMPT1_DISCLOSURE = {
    "phase3_duplicate_leaf_publish": (
        "attempt 1 (2026-09-02, roots phase3_table_attempt1_race_collision and "
        "phase3_table, both preserved) failed with FileExistsError(17) after one "
        "session: the driver published one leaf per (arm,path,surface) while "
        "iterating sessions; fixed by one leaf per (arm,path,surface,session)"
    ),
    "phase3_surface_clobbered_summary": (
        "attempt 3 (2026-09-02, root preserved as "
        "phase3_table_attempt3_surface_clobbered_summary) scored ALL 52 "
        "per-session cells successfully -- the frozen rosters are loader-truth "
        "-- but the summary map was keyed by (arm,path) while iterating two "
        "surfaces per path, so the within_post30 summary clobbered the "
        "external one and build_table's primary contrast failed with "
        "'paired session set mismatch'; fixed by keying (arm,path,surface) "
        "and binding the gate cells to the external roster fail-closed"
    ),
    "arm_stale_val_tracker": (
        "the attempt-1 arms (roots preserved as *_attempt1_stale_val_tracker) "
        "recorded epoch rows from trainer.callback_metrics, which a validation "
        "Callback sees BEFORE the module hook logs val_heldin/r2_mean: epoch 0 "
        "recorded no row and epochs 1-11 recorded the previous epoch's metric, "
        "so best-checkpoint selection paired epoch-k weights with epoch-(k-1) "
        "metrics and the roots failed terminal validation (11 epoch rows, "
        "failure.json beside terminal.json); fixed by computing the metric "
        "fresh from the module's own R2 states (trainer.epoch_val_metric); "
        "the arms MUST be rerun before phase3"
    ),
}

GATE_BOUNDARY_EPSILON = 1.0e-12
GATES = {
    "schema": "pit_m2_gates_v1",
    "primary": {
        "expression": (
            "F10m - F00m >= +0.01 equal-session mean R2 on the official-contract "
            "static surface (external_official_query, static M10) AND positive "
            "external sessions >= 4/6"
        ),
        "driving_cell": "F10m",
        "reference_cell": "F00m",
        "delta_floor": 0.01,
        "breadth_min": 4,
        "breadth_denominator": 6,
        "breadth_rationale": (
            "two-thirds of the 6-session M2 external roster, mirroring the sealed "
            "cdm_p1_m2_local_v1 breadth law (4/6)"
        ),
        "disposition_passed": "PIT_M2_PRIMARY_GATE_PASSED",
        "disposition_failed": "PIT_M2_PRIMARY_GATE_NOT_MET",
    },
    "secondary_stacking": {
        "expression": (
            "F11m - F01m >= +0.01 equal-session mean R2 on the local FIFO path "
            "(external_post30_local, m10 activity-only) AND positive external "
            "sessions >= 4/6"
        ),
        "driving_cell": "F11m",
        "reference_cell": "F01m",
        "delta_floor": 0.01,
        "breadth_min": 4,
        "breadth_denominator": 6,
        "role": (
            "stacking-reading SECONDARY: reported for disclosure only, NO promotion "
            "attached -- the primary verdict never rests on it"
        ),
        "disposition": "REPORTED_NO_PROMOTION",
    },
    "reported_not_gated": [
        "within_post30 static M10 (both model axes)",
        "within_post30 local FIFO M10 (both model axes)",
        "the fresh t0m control arm on every path (matched-pair replication control)",
    ],
    "boundary_rule": (
        "exact >= on the float64 equal-session mean; a miss inside the 1e-12 program "
        "epsilon of the boundary is disclosed as within_epsilon_band_of_boundary and "
        "never flips the verdict (the cal_aug_v1/cdm_p1_m2_local_v1 convention)"
    ),
    "nulls_reported_as_nulls": True,
}

# ---------------------------------------------------------------------------
# Phase-3 scoring surfaces (frozen; the sealed screens' own laws verbatim).
# ---------------------------------------------------------------------------

PHASE3_SURFACES = {
    "static": ("external_official_query", "within_post30"),
    "fifo": ("external_post30_local", "within_post30"),
}
STATIC_LAW = {
    "scorer": "tfpd_exploration.src.m2_t4_activity_budget_screen_v1.physical._score_session (verbatim)",
    "cell": "ridge_static_m10 (carrier_budget=10, activity_budget=10, chronological first-10)",
    "side": "fit_ridge_t4 over the selected support, frozen normalizer "
            "(normalized_lambda=0.1, side=(raw-mean)/std)",
    "identity_path": "student.compute_identity(activity[B,10,100,96], side_features=[1,96,4])",
    "decode": "student(neural, identity=identity), last timestep, /5.0",
    "r2": "variance_weighted_r2 over the session's ordered query windows",
}
FIFO_LAW = {
    "scorer": "tfpd_exploration.src.m2_precision_cdm_v2_screen_v1.physical (verbatim helpers)",
    "cell": "m10_activity_only (system=activity_only, budget=10, support=first-10 "
            "finite-direction from first-30)",
    "memory": "linear-B3S activity FIFO advancing on every completed valid trial "
              "(commit_activity_only), frozen canonical fixed-ridge carrier",
    "short_trial_policy": (
        "activity_advances_carrier_rejects_velocity_shape_no_padding_no_"
        "cross_trial_history"
    ),
}

#: The sealed, terminal-OK CPU smoke predecessor (2026-09-01) every train /
#: phase3 stage binds by digest before any data or model access.
SEALED_SMOKE_ROOT_RELATIVE = SMOKE_ROOT_RELATIVE
SEALED_SMOKE_TERMINAL_SHA256 = "317c36c27f8d1ce4997a8b7f4b95a1327f1e127554ca9c621c8a43254f6a6484"
SEALED_SMOKE_STATUS = "COMPLETE_PIT_M2_MATCHED_SMOKE_EQUALITY"
#: The superseded first sealing (wrong torch-version deviation literal,
#: environment fact corrected the same day; preserved as an attempt root).
SUPERSEDED_SMOKE_ROOT_RELATIVE = f"{RESULT_ROOT_RELATIVE}/smoke_attempt1_deviation_literal_drift"

# ---------------------------------------------------------------------------
# The CPU smoke equality contract.
# ---------------------------------------------------------------------------

SMOKE_STEPS = 40
RECORD_STEPS = 40
SMOKE_EQUALITY_CONTRACT = {
    "schema": "pit_m2_smoke_equality_contract_v1",
    "steps_per_arm": SMOKE_STEPS,
    "device": "cpu only; CUDA_VISIBLE_DEVICES empty and torch.cuda.is_initialized() "
              "must stay False, asserted at terminal",
    "must_be_equal_across_arms": [
        "initial_model_state_sha256 (student+teacher state digest after setup)",
        "per_step batch order digest (sampler batched_indices and per-step batch bytes)",
        "per_step python and torch RNG state digests (the dropout-p stream domain)",
        "per_step T4 side-feature bytes digest (T4 bytes unchanged)",
        "optimizer step count (one student training forward per step)",
        "normalization sha256 == the sealed d17f5f4c... (both arms)",
    ],
    "c1m_only": "per-step scheduled/effective prefix follows (10, 5, 2) with visible-slice digests",
    "t0m_only": "effective prefix is the full 33-trial block every step (operator disabled)",
    "dropout_p_draw_counter": (
        "every random.uniform call during the 40 steps is counted: exactly ONE "
        "per step in BOTH arms (the teacher's inert eval-mode p draw, the "
        "carried cal_aug/m1 dropout-p stream); the student itself draws none "
        "(frozen decoder), and numpy/torch counts must be equal across arms"
    ),
    "eval_dropout_inactive_proof": (
        "student.eval() forward repeated twice yields identical prediction "
        "digests (student has no draw at all under the frozen recipe); the "
        "teacher's eval forward repeated twice is also bit-identical DESPITE "
        "drawing a fresh inert p each time (the mask is gated on training)"
    ),
    "no_target_path_resolved": (
        "val_heldout_dataset is None after setup('fit') and the resolved file "
        "roster contains only held-in calib/minival NWBs"
    ),
    "cal_aug_discipline_reference": "tfpd_exploration/src/cal_aug_v1 (hook/schedule/receipt conventions)",
}

# ---------------------------------------------------------------------------
# Environment law for the GPU stages (train / phase3).
# ---------------------------------------------------------------------------

ENVIRONMENT_LAW = {
    "python": "/home/xinyuan/miniconda3/envs/spint/bin/python",
    "no_user_site": True,
    "cuda_device_order": "PCI_BUS_ID",
    "single_visible_card": "CUDA_VISIBLE_DEVICES=<gpu-index> selected at launch",
    "authorization": (
        "--gpu-authorized AND the TARGET gpu idle (memory < "
        f"{GPU_IDLE_MEMORY_THRESHOLD_MIB} MiB, utilization 0%, no compute apps) "
        "required before the train and phase3 stages import torch; other cards "
        "may be busy (amendment 2026-09-02)"
    ),
    "parallel_single_arm_mode": (
        "--arm {t0m,c1m} runs one arm in its own process; the two arms may "
        "share one card as parallel processes (per-process seeds)"
    ),
    "cpu_smoke_needs_no_authorization": True,
}

OWNED_PATHS = (
    "tfpd_exploration/src/pit_m2_v1/__init__.py",
    "tfpd_exploration/src/pit_m2_v1/plan.py",
    "tfpd_exploration/src/pit_m2_v1/schedule.py",
    "tfpd_exploration/src/pit_m2_v1/hook.py",
    "tfpd_exploration/src/pit_m2_v1/trainer.py",
    "tfpd_exploration/src/pit_m2_v1/smoke.py",
    "tfpd_exploration/src/pit_m2_v1/receipts.py",
    "tfpd_exploration/src/pit_m2_v1/phase3.py",
    "tfpd_exploration/src/pit_m2_v1/driver.py",
    "tfpd_exploration/scripts/run_pit_m2_v1.py",
    "tfpd_exploration/tests/test_pit_m2_v1.py",
)

#: Route-owned runtime imports (read-only reuse; never edited).
RUNTIME_PATHS = (
    "tfpd_exploration/src/m2_t4_activity_budget_screen_v1/core.py",
    "tfpd_exploration/src/m2_t4_activity_budget_screen_v1/physical.py",
    "tfpd_exploration/src/m2_t4_activity_budget_screen_v1/plan.py",
    "tfpd_exploration/src/m2_precision_cdm_v2_screen_v1/physical.py",
    "tfpd_exploration/src/m2_precision_cdm_v2_screen_v1/plan.py",
    "tfpd_exploration/src/pseudo_mua_precision_cdm_v2_screen_v1/core.py",
    "tfpd_exploration/src/pseudo_mua_precision_cdm_v2_screen_v1/physical.py",
    "tfpd_exploration/src/causal_dual_memory_cell_d_v1/core.py",
    "tfpd_exploration/src/calibration_budget_comparators_v1.py",
    "tfpd_exploration/src/cal_aug_v1/hook.py",
    "tfpd_exploration/src/cal_aug_v1/schedule.py",
    "tfpd_exploration/src/m1_t0c1_prefix_v1/hook.py",
    "tfpd_exploration/src/m1_t0c1_prefix_v1/schedule.py",
)


@dataclass(frozen=True)
class PairSpec:
    """The sole matched-pair surface; no training choice remains at run time."""

    root_relative: str = RESULT_ROOT_RELATIVE

    def payload(self) -> dict[str, object]:
        return {
            "schema": "pit_m2_pair_spec_v1",
            "cell": CELL,
            "phase": PHASE,
            "root_relative": self.root_relative,
            "arms": list(ARMS),
            "arm_roles": dict(ARM_ROLES),
            "trainer_binding": TRAINER_BINDING,
            "cycle_law": CYCLE_LAW,
            "budget": {
                "seed": SEED, "epochs": EPOCHS, "batch_size": TRAIN_BATCH_SIZE,
                "optimizer": "Adam", "adam_lr": ADAM_LR, "adam_weight_decay": ADAM_WEIGHT_DECAY,
                "scheduler": None, "early_stopping": False,
                "checkpoint_selection": f"{CHECKPOINT_MONITOR} ({CHECKPOINT_MODE})",
                "sampler": SAMPLER_LAW,
            },
            "dropout_proof": DROPOUT_PROOF,
            "source_preparation_law": SOURCE_PREPARATION_LAW,
            "deviations": DEVIATIONS,
            "factorial": {
                "cells": list(FACTORIAL_CELLS),
                "sealed_cells": list(SEALED_CELLS),
                "fresh_cells": list(FRESH_CELLS),
                "axis_train": {str(key): value for key, value in FACTORIAL_AXIS_TRAIN.items()},
                "axis_deployment": {str(key): value for key, value in FACTORIAL_AXIS_DEPLOYMENT.items()},
                "f00m_anchor": F00M_ANCHOR,
                "f01m_anchor": F01M_ANCHOR,
            },
            "gates": GATES,
            "static_law": STATIC_LAW,
            "fifo_law": FIFO_LAW,
            "smoke_equality_contract": SMOKE_EQUALITY_CONTRACT,
            "hard_timeout_seconds_per_arm": HARD_TIMEOUT_SECONDS_PER_ARM,
            "operator_resolution": OPERATOR_RESOLUTION,
            "environment_law": ENVIRONMENT_LAW,
            "formal_benchmark_verdict": False,
        }

    @property
    def sha256(self) -> str:
        return sha256_bytes(canonical_json_bytes(self.payload()))


def implementation_closure(root: Path) -> dict[str, object]:
    """Digest the immutable anchors (fail-closed) and record the live bytes."""
    base = Path(root).absolute()
    anchors: dict[str, str] = {}
    for relative in IMMUTABLE_ANCHOR_RELATIVE:
        expected = IMMUTABLE_ANCHOR_SHA256.get(relative)
        digest = sha256_file(base / relative)
        if expected is not None:
            _require(digest == expected,
                     f"pit m2 immutable anchor drift: {relative} {digest} != {expected}")
        anchors[relative] = digest
    live: dict[str, str] = {}
    for relative in LIVE_SOURCE_RELATIVE:
        live[relative] = sha256_file(base / relative)
    owned: dict[str, str] = {}
    for relative in OWNED_PATHS:
        path = base / relative
        _require(path.is_file(), f"pit m2 owned path missing at closure time: {relative}")
        owned[relative] = sha256_file(path)
    body = {
        "schema": "pit_m2_closure_v1",
        "immutable_anchors_sha256": anchors,
        "live_source_sha256_at_attempt": live,
        "owned_sha256": owned,
    }
    return {**body, "closure_sha256": sha256_bytes(canonical_json_bytes(body))}


def validate_current_closure(root: Path, value: Mapping[str, object]) -> dict[str, object]:
    _require(isinstance(value, Mapping), "pit m2 closure must be a mapping")
    rebuilt = implementation_closure(Path(root))
    _require(dict(value) == rebuilt, "pit m2 closure/current-byte drift")
    return rebuilt


def dry_plan(root: Path | None = None) -> dict[str, object]:
    del root
    spec = PairSpec()
    return {
        "cell": CELL,
        "phase": PHASE,
        "stages": ("attempt", "smoke", "train", "phase3"),
        "prospective_roots": {
            "smoke": SMOKE_ROOT_RELATIVE,
            "t0m": ARM_ROOT_RELATIVE["t0m"],
            "c1m": ARM_ROOT_RELATIVE["c1m"],
            "phase3": PHASE3_ROOT_RELATIVE,
        },
        "pair_spec_sha256": spec.sha256,
        "sealed_trainer_binding": {
            "run_id": SEALED_RUN_ID,
            "checkpoint_sha256": SEALED_CHECKPOINT_SHA256,
            "resolved_config_sha256": SEALED_RESOLVED_CONFIG_SHA256,
            "teacher_checkpoint_sha256": TEACHER_CHECKPOINT_SHA256,
            "normalization_sha256": NORMALIZATION_SHA256,
        },
        "train_stage_refusal": (
            "refuses without --gpu-authorized AND a live TARGET-GPU-idle check "
            "(memory < 100 MiB, 0% utilization, no compute apps on the selected "
            "card; other cards may be busy -- amendment 2026-09-02)"
        ),
        "single_arm_mode": "--arm {t0m,c1m} trains one arm per process (parallel pair)",
        "opens_nwb_or_checkpoint": False,
        "imports_torch": False,
        "initializes_cuda": False,
        "creates_root_or_receipt": False,
        "public_execution_authorized": False,
    }


__all__ = (
    "PitM2PlanError", "SCHEMA", "PHASE", "CELL", "RESULT_ROOT_RELATIVE", "SMOKE_ROOT_RELATIVE",
    "ARM_ROOT_RELATIVE", "PHASE3_ROOT_RELATIVE", "ARMS", "ARM_ROLES", "CYCLE", "CYCLE_LAW",
    "TRAINING_CALIBRATION_N_TRIALS", "DEPLOYMENT_BUDGET_M", "MAX_CALIBRATION_TRIALS",
    "SEED", "EPOCHS", "TRAIN_BATCH_SIZE", "ADAM_LR", "ADAM_WEIGHT_DECAY", "CHECKPOINT_MONITOR",
    "CHECKPOINT_MODE", "NO_EARLY_STOPPING", "SAMPLER_LAW", "DEVIATIONS",
    "HARD_TIMEOUT_SECONDS_PER_ARM", "OPERATOR_RESOLUTION", "DROPOUT_PROOF",
    "SOURCE_PREPARATION_LAW", "TRAINER_BINDING", "LIVE_SOURCE_RELATIVE",
    "IMMUTABLE_ANCHOR_RELATIVE", "IMMUTABLE_ANCHOR_SHA256", "FACTORIAL_CELLS", "SEALED_CELLS",
    "FRESH_CELLS", "FACTORIAL_AXIS_TRAIN", "FACTORIAL_AXIS_DEPLOYMENT", "F00M_ANCHOR",
    "F01M_ANCHOR", "EXTERNAL_SESSION_COUNT", "WITHIN_SESSION_COUNT", "GATES",
    "GATE_BOUNDARY_EPSILON", "PHASE3_SURFACES", "STATIC_LAW", "FIFO_LAW",
    "SMOKE_STEPS", "RECORD_STEPS", "SMOKE_EQUALITY_CONTRACT", "ENVIRONMENT_LAW",
    "GPU_IDLE_MEMORY_THRESHOLD_MIB", "GPU_IDLE_UTILIZATION_PERCENT", "OWNED_PATHS",
    "RUNTIME_PATHS", "SEALED_RUN_ID", "SEALED_RUN_DIR_RELATIVE", "SEALED_CHECKPOINT_RELATIVE",
    "SEALED_CHECKPOINT_SHA256", "SEALED_RESOLVED_CONFIG_RELATIVE", "SEALED_RESOLVED_CONFIG_SHA256",
    "SEALED_RUN_METADATA_RELATIVE", "TEACHER_CHECKPOINT_RELATIVE", "TEACHER_CHECKPOINT_SHA256",
    "NORMALIZATION_SHA256", "M2_DATA_DIR_RELATIVE", "PairSpec", "canonical_json_bytes",
    "sha256_bytes", "sha256_file", "require_sha", "safe_relative", "implementation_closure",
    "validate_current_closure", "dry_plan",
)
