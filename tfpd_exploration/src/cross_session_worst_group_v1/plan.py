"""Static CS-WG Stage-0 contracts and explicit implementation closure.

There is deliberately no model construction here.  This module records the
future route's immutable M1 graph and matched-ERM boundary so later execution
can validate a real materialized model rather than treating a sampler/loss
experiment as an independent architecture.
"""
from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


CELL = "CROSS_SESSION_WORST_GROUP_SPINT_M1_V1"
WORKORDER_RELATIVE = "tfpd_exploration/docs/WORKORDER_CS_WG_STAGE0_20260825.md"
WORKORDER_SHA256 = "225fccec7588e28c25d1e4c3240066eb896fcd32c48e11236aaa8d45f8cb491d"
# The source-decoder implementation is deliberately bound from its actual
# streaming-calibration tree, not the similarly named generic SPINT-main
# examples.  The three present fold configs establish the native M1 decoder
# recipe; the future fourth LOSO fold remains a route-owned matched-run
# responsibility rather than an invitation to substitute a generic config.
M1_FALCON_FORWARD_INTERFACE_RELATIVE = "streaming_calibration_exp/src/models/falcon_module.py"
M1_SPINT_MODEL_RELATIVE = "streaming_calibration_exp/src/models/components/spint.py"
M1_MODEL_CONFIG_RELATIVE = "streaming_calibration_exp/configs/model/falcon_m1_source_only_decoder.yaml"
M1_DATA_CONFIG_RELATIVE = "streaming_calibration_exp/configs/data/falcon_m1_source_only_decoder_fold0.yaml"
M1_DATA_FOLD_CONFIG_RELATIVES = (
    M1_DATA_CONFIG_RELATIVE,
    "streaming_calibration_exp/configs/data/falcon_m1_source_only_decoder_fold1.yaml",
    "streaming_calibration_exp/configs/data/falcon_m1_source_only_decoder_fold2.yaml",
)
M1_SOURCE_DECODER_DATAMODULE_RELATIVE = "streaming_calibration_exp/src/data/falcon_m1_source_only_decoder_datamodule.py"
M1_FALCON_DATAMODULE_RELATIVE = "streaming_calibration_exp/src/data/falcon_datamodule.py"
# Direct repository imports reached when the source-only datamodule is loaded.
# These are explicit rather than globbed so a future staging/preflight route
# cannot silently borrow similarly named modules from the generic SPINT tree.
M1_SOURCE_DECODER_RUNTIME_IMPORT_RELATIVES = (
    "streaming_calibration_exp/src/data/__init__.py",
    M1_SOURCE_DECODER_DATAMODULE_RELATIVE,
    M1_FALCON_DATAMODULE_RELATIVE,
    "streaming_calibration_exp/src/data/validation_protocol.py",
    "streaming_calibration_exp/src/data/falcon_t4_features.py",
    "streaming_calibration_exp/src/data/falcon_d4_features.py",
    "streaming_calibration_exp/src/data/falcon_k4_features.py",
    "streaming_calibration_exp/src/data/afc4_xls_v2_adapter.py",
    "streaming_calibration_exp/src/data/afc4_xls_v2.py",
    "streaming_calibration_exp/src/data/falcon_n4_features.py",
    "streaming_calibration_exp/third_party/__init__.py",
    "streaming_calibration_exp/third_party/catalyst/__init__.py",
    "streaming_calibration_exp/third_party/catalyst/distributed_sampler.py",
    "streaming_calibration_exp/third_party/falcon_challenge/__init__.py",
    "streaming_calibration_exp/third_party/falcon_challenge/filtering.py",
)

M1_WINDOW_SIZE = 100
M1_RAW_BEHAVIOR_OUTPUTS = 16
M1_LIVE_PARAMETERS_AFTER_LAZY1024 = 15_007_496
M1_FINAL_BIN_ONLY_RAW_OUTPUT_MSE = True
M1_B3S_IDENTITY_PATH = "frozen_current_b3s_identity_path"
M1_BASELINE_INFERENCE_TOPOLOGY = "exact_frozen_m1_inference_topology"
M1_BASELINE_TRAINING_STEP_MIXED_SESSION_POLICY = "rejects_mixed_session_batches"

HELD_IN_SOURCE_SESSIONS = ("20120924", "20120926", "20120927", "20120928")
OFFICIAL_HELD_OUT_SESSIONS_FORBIDDEN = 3
TOTAL_BATCH_SIZE = 32
NORM_QUANTILE_COUNT = 4

# These are the existing M1 source-decoder controls, expressed as typed
# literals rather than future caller knobs.  CS-WG is a system-level training
# treatment, not permission to search a second optimizer/configuration grid.
M1_CALIBRATION_BUDGET = "M10"
M1_CALIBRATION_TRIALS = 10
M1_UNIT_COUNT = 64
# This is B3S's padded/interpolated per-trial time axis.  It is intentionally
# distinct from the decoder window W=100: reusing W here materializes the
# wrong LazyLinear graph.
M1_B3S_MAX_TRIAL_LENGTH = 1024
M1_CALIBRATION_SHAPE_PER_ROW = (
    M1_CALIBRATION_TRIALS,
    M1_B3S_MAX_TRIAL_LENGTH,
    M1_UNIT_COUNT,
)
M1_FORWARD_INPUT_KEYS = ("x", "calib_trialized_neural_features")
M1_EPOCH_BUDGET = 20
M1_OPTIMIZER_LITERAL = "Adam"
M1_ADAM_LR = 1.0e-5
M1_ADAM_WEIGHT_DECAY = 0.0
M1_LR_SCHEDULE_LITERAL = "None"
M1_HIDDEN_INNER_GRID_OR_SWEEP_FORBIDDEN = True
M1_HISTORICAL_FOLD_WALL_MINUTES = 68.8
M1_OUTER_FOLD_COUNT = 4
M1_MATCHED_DEVELOPMENT_RUN_COUNT = 2 * M1_OUTER_FOLD_COUNT
M1_HELD_IN_UNIT_ID_SHAPE = (M1_UNIT_COUNT,)
# The detailed inherited checkpoint/SWA mechanism is an immutable source
# decoder authority at physical-preflight time.  Stage 0 requires the same
# literal on the paired systems and records this explicit placeholder rather
# than inventing a different checkpoint policy.
M1_CHECKPOINT_SWA_RULE_LITERAL = "exact_inherited_m1_source_decoder_checkpoint_swa_rule"


class CSWGPlanError(RuntimeError):
    """Fail closed for Stage-0 run-spec or closure drift."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CSWGPlanError(message)


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _safe_relative(relative: str) -> Path:
    path = Path(relative)
    _require(not path.is_absolute() and ".." not in path.parts and path.name not in {"", ".", ".."},
             "CS-WG closure path is unsafe")
    return path


def _regular_sha256(root: Path, relative: str) -> str:
    path = Path(root).absolute() / _safe_relative(relative)
    try:
        info = os.lstat(path)
    except OSError as error:
        raise CSWGPlanError(f"CS-WG closure path is inaccessible: {relative}") from error
    _require(stat.S_ISREG(info.st_mode) and not stat.S_ISLNK(info.st_mode),
             f"CS-WG closure path is not a regular non-symlink: {relative}")
    with open(path, "rb") as handle:
        return sha256_bytes(handle.read())


_CLOSURE_PATHS = (
    WORKORDER_RELATIVE,
    M1_MODEL_CONFIG_RELATIVE,
    *M1_DATA_FOLD_CONFIG_RELATIVES,
    "streaming_calibration_exp/src/__init__.py",
    "streaming_calibration_exp/src/models/__init__.py",
    M1_FALCON_FORWARD_INTERFACE_RELATIVE,
    "streaming_calibration_exp/src/models/components/__init__.py",
    M1_SPINT_MODEL_RELATIVE,
    *M1_SOURCE_DECODER_RUNTIME_IMPORT_RELATIVES,
    "tfpd_exploration/src/__init__.py",
    "tfpd_exploration/src/cross_session_worst_group_v1/__init__.py",
    "tfpd_exploration/src/cross_session_worst_group_v1/plan.py",
    "tfpd_exploration/src/cross_session_worst_group_v1/core.py",
    "tfpd_exploration/scripts/run_cross_session_worst_group_stage0.py",
    "tfpd_exploration/tests/test_cross_session_worst_group_stage0.py",
)


def execution_closure_payload(root: Path) -> dict[str, object]:
    """Rebuild an explicit, no-glob Stage-0 closure from regular source leaves."""
    rows = [{"path": path, "sha256": _regular_sha256(Path(root), path)} for path in _CLOSURE_PATHS]
    _require(rows[0]["sha256"] == WORKORDER_SHA256, "CS-WG Stage-0 workorder SHA drift")
    return {
        "schema": "cross_session_worst_group_stage0_closure_v1",
        "paths": rows,
        "closure_sha256": sha256_bytes(_json_bytes(rows)),
    }


@dataclass(frozen=True)
class BaselineGraphContract:
    """The graph that CS-WG and matched ERM must both preserve exactly."""

    window_size: int = M1_WINDOW_SIZE
    raw_behavior_outputs: int = M1_RAW_BEHAVIOR_OUTPUTS
    unit_count: int = M1_UNIT_COUNT
    calibration_trials: int = M1_CALIBRATION_TRIALS
    b3s_max_trial_length: int = M1_B3S_MAX_TRIAL_LENGTH
    forward_input_keys: tuple[str, str] = M1_FORWARD_INPUT_KEYS
    final_bin_only_raw_output_mse: bool = M1_FINAL_BIN_ONLY_RAW_OUTPUT_MSE
    b3s_identity_path: str = M1_B3S_IDENTITY_PATH
    baseline_inference_topology: str = M1_BASELINE_INFERENCE_TOPOLOGY
    live_parameters_after_lazy1024: int = M1_LIVE_PARAMETERS_AFTER_LAZY1024
    target_finetuning_or_backpropagation: bool = False

    def __post_init__(self) -> None:
        _require(
            self.window_size == M1_WINDOW_SIZE
            and self.raw_behavior_outputs == M1_RAW_BEHAVIOR_OUTPUTS
            and self.unit_count == M1_UNIT_COUNT
            and self.calibration_trials == M1_CALIBRATION_TRIALS
            and self.b3s_max_trial_length == M1_B3S_MAX_TRIAL_LENGTH
            and self.forward_input_keys == M1_FORWARD_INPUT_KEYS
            and self.final_bin_only_raw_output_mse is True
            and self.b3s_identity_path == M1_B3S_IDENTITY_PATH
            and self.baseline_inference_topology == M1_BASELINE_INFERENCE_TOPOLOGY
            and self.live_parameters_after_lazy1024 == M1_LIVE_PARAMETERS_AFTER_LAZY1024
            and self.target_finetuning_or_backpropagation is False,
            "CS-WG baseline M1 graph contract drift",
        )

    def payload(self) -> dict[str, object]:
        return {
            "window_size": self.window_size,
            "raw_behavior_outputs": self.raw_behavior_outputs,
            "unit_count": self.unit_count,
            "calibration_trials": self.calibration_trials,
            "b3s_max_trial_length": self.b3s_max_trial_length,
            "calibration_shape_per_row": list(M1_CALIBRATION_SHAPE_PER_ROW),
            "forward_input_keys": list(self.forward_input_keys),
            "falcon_forward_interface_source": M1_FALCON_FORWARD_INTERFACE_RELATIVE,
            "spint_model_source": M1_SPINT_MODEL_RELATIVE,
            "model_config_source": M1_MODEL_CONFIG_RELATIVE,
            "data_config_source": M1_DATA_CONFIG_RELATIVE,
            "data_config_sources": list(M1_DATA_FOLD_CONFIG_RELATIVES),
            "source_decoder_datamodule_source": M1_SOURCE_DECODER_DATAMODULE_RELATIVE,
            "falcon_datamodule_source": M1_FALCON_DATAMODULE_RELATIVE,
            "final_bin_only_raw_output_mse": self.final_bin_only_raw_output_mse,
            "b3s_identity_path": self.b3s_identity_path,
            "baseline_inference_topology": self.baseline_inference_topology,
            "live_parameters_after_lazy1024": self.live_parameters_after_lazy1024,
            "target_finetuning_or_backpropagation": self.target_finetuning_or_backpropagation,
        }


M1_GRAPH_CONTRACT = BaselineGraphContract()


@dataclass(frozen=True)
class M1SourceDecoderRecipe:
    """Exact non-search training recipe shared by every outer-fold pair.

    The checkpoint/SWA authority is deliberately named rather than recreated
    here: this additive Stage-0 package does not import the baseline trainer.
    A future physical route must resolve that exact inherited authority once
    and bind its value identically into CS-WG and matched ERM receipts.
    """

    calibration_budget: str = M1_CALIBRATION_BUDGET
    calibration_trials: int = M1_CALIBRATION_TRIALS
    b3s_max_trial_length: int = M1_B3S_MAX_TRIAL_LENGTH
    epoch_budget: int = M1_EPOCH_BUDGET
    optimizer_literal: str = M1_OPTIMIZER_LITERAL
    adam_lr: float = M1_ADAM_LR
    adam_weight_decay: float = M1_ADAM_WEIGHT_DECAY
    lr_schedule_literal: str = M1_LR_SCHEDULE_LITERAL
    final_bin_only_raw_output_mse: bool = True
    total_batch_size: int = TOTAL_BATCH_SIZE
    hidden_inner_grid_or_sweep_forbidden: bool = M1_HIDDEN_INNER_GRID_OR_SWEEP_FORBIDDEN

    def __post_init__(self) -> None:
        _require(
            self.calibration_budget == M1_CALIBRATION_BUDGET
            and self.calibration_trials == M1_CALIBRATION_TRIALS
            and self.b3s_max_trial_length == M1_B3S_MAX_TRIAL_LENGTH
            and self.epoch_budget == M1_EPOCH_BUDGET
            and self.optimizer_literal == M1_OPTIMIZER_LITERAL
            and self.adam_lr == M1_ADAM_LR
            and self.adam_weight_decay == M1_ADAM_WEIGHT_DECAY
            and self.lr_schedule_literal == M1_LR_SCHEDULE_LITERAL
            and self.final_bin_only_raw_output_mse is True
            and self.total_batch_size == TOTAL_BATCH_SIZE
            and self.hidden_inner_grid_or_sweep_forbidden is True,
            "CS-WG fixed M1 source-decoder recipe drift",
        )

    def payload(self) -> dict[str, object]:
        return {
            "calibration_budget": self.calibration_budget,
            "calibration_trials": self.calibration_trials,
            "b3s_max_trial_length": self.b3s_max_trial_length,
            "calibration_shape_per_row": list(M1_CALIBRATION_SHAPE_PER_ROW),
            "epoch_budget": self.epoch_budget,
            "optimizer": self.optimizer_literal,
            "adam_lr": self.adam_lr,
            "adam_weight_decay": self.adam_weight_decay,
            "lr_schedule": self.lr_schedule_literal,
            "final_bin_only_raw_output_mse": self.final_bin_only_raw_output_mse,
            "total_batch_size": self.total_batch_size,
            "hidden_inner_grid_or_sweep_forbidden": self.hidden_inner_grid_or_sweep_forbidden,
        }


M1_SOURCE_DECODER_RECIPE = M1SourceDecoderRecipe()


@dataclass(frozen=True)
class RouteOwnedMixedSessionTrainingStepContract:
    """The only permitted training-step seam; M1 inference itself is unchanged."""

    baseline_training_step_policy: str = M1_BASELINE_TRAINING_STEP_MIXED_SESSION_POLICY
    route_owned_override_required: bool = True
    single_concatenated_m1_forward_required: bool = True
    physical_preflight_exact_concat_compatibility_required: bool = True
    per_row_calibration_and_session_ownership_required: bool = True
    unit_padding_masks_or_shape_coercion_forbidden: bool = True

    def __post_init__(self) -> None:
        _require(
            self.baseline_training_step_policy == M1_BASELINE_TRAINING_STEP_MIXED_SESSION_POLICY
            and self.route_owned_override_required is True
            and self.single_concatenated_m1_forward_required is True
            and self.physical_preflight_exact_concat_compatibility_required is True
            and self.per_row_calibration_and_session_ownership_required is True
            and self.unit_padding_masks_or_shape_coercion_forbidden is True,
            "CS-WG route-owned mixed-session training-step contract drift",
        )

    def payload(self) -> dict[str, object]:
        return {
            "baseline_training_step_policy": self.baseline_training_step_policy,
            "route_owned_override_required": self.route_owned_override_required,
            "single_concatenated_m1_forward_required": self.single_concatenated_m1_forward_required,
            "physical_preflight_exact_concat_compatibility_required": self.physical_preflight_exact_concat_compatibility_required,
            "per_row_calibration_and_session_ownership_required": self.per_row_calibration_and_session_ownership_required,
            "exact_baseline_forward_input_keys": list(M1_FORWARD_INPUT_KEYS),
            "unit_padding_masks_or_shape_coercion_forbidden": self.unit_padding_masks_or_shape_coercion_forbidden,
        }


MIXED_SESSION_TRAINING_STEP_CONTRACT = RouteOwnedMixedSessionTrainingStepContract()


def validate_materialized_baseline_model(*, live_parameter_count: int, topology: str) -> None:
    """Future physical route must call this after LazyLinear materialization."""
    _require(type(live_parameter_count) is int
             and live_parameter_count == M1_LIVE_PARAMETERS_AFTER_LAZY1024,
             "CS-WG materialized M1 live-parameter count drift")
    _require(topology == M1_BASELINE_INFERENCE_TOPOLOGY,
             "CS-WG materialized M1 inference topology drift")


@dataclass(frozen=True)
class EpisodeQuota:
    """One total-B32 source-only episode quota, never three B32 forwards."""

    step_index: int
    sessions: tuple[str, ...]
    counts: tuple[int, ...]
    system_phase: str

    def __post_init__(self) -> None:
        _require(type(self.step_index) is int and self.step_index >= 0, "episode quota step drift")
        _require(self.system_phase in {"outer_fold", "all_four_source"}, "episode quota phase drift")
        _require(len(self.sessions) == len(self.counts) and len(set(self.sessions)) == len(self.sessions),
                 "episode quota session topology drift")
        _require(all(session in HELD_IN_SOURCE_SESSIONS for session in self.sessions),
                 "episode quota may contain only held-in M1 source sessions")
        _require(sum(self.counts) == TOTAL_BATCH_SIZE and all(type(item) is int and item > 0 for item in self.counts),
                 "episode quota must be exact total B32")
        if self.system_phase == "outer_fold":
            _require(len(self.sessions) == 3 and sorted(self.counts) == [10, 11, 11],
                     "outer-fold episode must rotate exact 11/11/10 quotas")
        else:
            _require(len(self.sessions) == 4 and self.counts == (8, 8, 8, 8),
                     "all-four-source episode must use exact 8/8/8/8 quotas")

    @property
    def by_session(self) -> dict[str, int]:
        return dict(zip(self.sessions, self.counts, strict=True))

    def payload(self) -> dict[str, object]:
        return {
            "step_index": self.step_index,
            "sessions": list(self.sessions),
            "counts": list(self.counts),
            "total_windows": TOTAL_BATCH_SIZE,
            "single_concatenated_forward_required": True,
            "system_phase": self.system_phase,
        }


def outer_fold_episode_quota(source_sessions: Sequence[str], *, step_index: int) -> EpisodeQuota:
    sessions = tuple(source_sessions)
    _require(len(sessions) == 3 and len(set(sessions)) == 3,
             "outer-fold quota requires exactly three distinct source sessions")
    _require(all(session in HELD_IN_SOURCE_SESSIONS for session in sessions),
             "outer-fold quota source session is not held-in M1")
    _require(type(step_index) is int and step_index >= 0, "outer-fold quota step drift")
    low_index = step_index % 3
    counts = tuple(10 if index == low_index else 11 for index in range(3))
    return EpisodeQuota(step_index, sessions, counts, "outer_fold")


def all_four_source_episode_quota(source_sessions: Sequence[str], *, step_index: int) -> EpisodeQuota:
    sessions = tuple(source_sessions)
    _require(sessions == HELD_IN_SOURCE_SESSIONS,
             "all-four-source quota requires exact ordered held-in M1 source roster")
    return EpisodeQuota(step_index, sessions, (8, 8, 8, 8), "all_four_source")


@dataclass(frozen=True)
class CSWGRunSpec:
    """Typed source-only system spec for either CS-WG or matched ordinary ERM."""

    system: str
    outer_target_session: str | None
    source_sessions: tuple[str, ...]
    initialization_seed: int
    optimizer_literal: str = M1_OPTIMIZER_LITERAL
    lr_schedule_literal: str = M1_LR_SCHEDULE_LITERAL
    epoch_budget: int = M1_EPOCH_BUDGET
    checkpoint_swa_rule: str = M1_CHECKPOINT_SWA_RULE_LITERAL
    graph: BaselineGraphContract = M1_GRAPH_CONTRACT
    total_batch_size: int = TOTAL_BATCH_SIZE
    lambda_: float = 0.0
    tau: float = 1.0
    norm_quantile_count: int = NORM_QUANTILE_COUNT
    checkpoint_training_sessions: tuple[str, ...] = ()
    calibration_budget: str = M1_CALIBRATION_BUDGET
    calibration_trials: int = M1_CALIBRATION_TRIALS
    adam_lr: float = M1_ADAM_LR
    adam_weight_decay: float = M1_ADAM_WEIGHT_DECAY
    hidden_inner_grid_or_sweep_forbidden: bool = M1_HIDDEN_INNER_GRID_OR_SWEEP_FORBIDDEN

    def __post_init__(self) -> None:
        _require(self.system in {"CS_WG", "MATCHED_ERM"}, "CS-WG system kind drift")
        sources = tuple(self.source_sessions)
        _require(len(sources) in {3, 4} and len(set(sources)) == len(sources)
                 and all(item in HELD_IN_SOURCE_SESSIONS for item in sources),
                 "CS-WG source-session topology drift")
        if self.outer_target_session is None:
            _require(len(sources) == 4 and set(sources) == set(HELD_IN_SOURCE_SESSIONS),
                     "all-four-source run spec drift")
        else:
            _require(self.outer_target_session in HELD_IN_SOURCE_SESSIONS
                     and self.outer_target_session not in sources
                     and len(sources) == 3
                     and set(sources) | {self.outer_target_session} == set(HELD_IN_SOURCE_SESSIONS),
                     "outer held-in LOSO run spec drift")
        _require(type(self.initialization_seed) is int
                 and self.epoch_budget == M1_EPOCH_BUDGET
                 and self.total_batch_size == TOTAL_BATCH_SIZE
                 and self.optimizer_literal == M1_OPTIMIZER_LITERAL
                 and self.lr_schedule_literal == M1_LR_SCHEDULE_LITERAL
                 and self.calibration_budget == M1_CALIBRATION_BUDGET
                 and self.calibration_trials == M1_CALIBRATION_TRIALS
                 and self.adam_lr == M1_ADAM_LR
                 and self.adam_weight_decay == M1_ADAM_WEIGHT_DECAY
                 and self.hidden_inner_grid_or_sweep_forbidden is True
                 and isinstance(self.checkpoint_swa_rule, str) and self.checkpoint_swa_rule
                 and isinstance(self.graph, BaselineGraphContract)
                 and type(self.norm_quantile_count) is int and self.norm_quantile_count >= 2,
                 "CS-WG immutable run-spec/source-decoder recipe drift")
        _require(0.0 <= float(self.lambda_) <= 1.0 and float(self.tau) > 0.0,
                 "CS-WG robust objective lambda/tau range drift")
        training = tuple(self.checkpoint_training_sessions)
        _require(training == sources, "CS-WG checkpoint must train only on exact run-spec source sessions")
        _require(self.outer_target_session is None or self.outer_target_session not in training,
                 "CS-WG matched checkpoint trained on held-in outer target")
        if self.system == "MATCHED_ERM":
            _require(float(self.lambda_) == 0.0,
                     "matched ERM must carry lambda=0 rather than a CS-WG objective")

    def payload(self) -> dict[str, object]:
        return {
            "system": self.system,
            "outer_target_session": self.outer_target_session,
            "source_sessions": list(self.source_sessions),
            "initialization_seed": self.initialization_seed,
            "optimizer_literal": self.optimizer_literal,
            "lr_schedule_literal": self.lr_schedule_literal,
            "epoch_budget": self.epoch_budget,
            "checkpoint_swa_rule": self.checkpoint_swa_rule,
            "graph": self.graph.payload(),
            "total_batch_size": self.total_batch_size,
            "lambda": self.lambda_,
            "tau": self.tau,
            "norm_quantile_count": self.norm_quantile_count,
            "checkpoint_training_sessions": list(self.checkpoint_training_sessions),
            "source_decoder_recipe": M1_SOURCE_DECODER_RECIPE.payload(),
            "calibration_budget": self.calibration_budget,
            "calibration_trials": self.calibration_trials,
            "adam_lr": self.adam_lr,
            "adam_weight_decay": self.adam_weight_decay,
            "hidden_inner_grid_or_sweep_forbidden": self.hidden_inner_grid_or_sweep_forbidden,
            "route_owned_mixed_session_training_step": MIXED_SESSION_TRAINING_STEP_CONTRACT.payload(),
            "source_only": True,
            "official_held_out_sessions_forbidden": OFFICIAL_HELD_OUT_SESSIONS_FORBIDDEN,
            "single_concatenated_forward_required": True,
        }


def build_outer_fold_specs(
    *,
    outer_target_session: str,
    initialization_seed: int,
    checkpoint_swa_rule: str = M1_CHECKPOINT_SWA_RULE_LITERAL,
    lambda_: float,
    tau: float,
    norm_quantile_count: int = NORM_QUANTILE_COUNT,
) -> tuple[CSWGRunSpec, CSWGRunSpec]:
    """Build the inseparable CS-WG / matched-ERM same-fold performance pair."""
    _require(outer_target_session in HELD_IN_SOURCE_SESSIONS, "outer target is not a held-in M1 source session")
    sources = tuple(session for session in HELD_IN_SOURCE_SESSIONS if session != outer_target_session)
    common = dict(
        outer_target_session=outer_target_session,
        source_sessions=sources,
        initialization_seed=initialization_seed,
        checkpoint_swa_rule=checkpoint_swa_rule,
        graph=M1_GRAPH_CONTRACT,
        tau=tau,
        norm_quantile_count=norm_quantile_count,
        checkpoint_training_sessions=sources,
    )
    cswg = CSWGRunSpec(system="CS_WG", lambda_=lambda_, **common)
    erm = CSWGRunSpec(system="MATCHED_ERM", lambda_=0.0, **common)
    validate_matched_same_fold_pair(cswg, erm)
    return cswg, erm


def build_all_outer_fold_spec_pairs(
    *, initialization_seed: int, lambda_: float, tau: float,
    checkpoint_swa_rule: str = M1_CHECKPOINT_SWA_RULE_LITERAL,
) -> tuple[tuple[CSWGRunSpec, CSWGRunSpec], ...]:
    """Build exactly the four held-in LOSO CS-WG / matched-ERM pairs.

    This deliberately constructs fold 3 (target ``20120928``), even though an
    older source-only decoder exposed only folds 0--2.  It is a fixed
    development matrix, not an inner hyperparameter sweep.
    """
    pairs = tuple(
        build_outer_fold_specs(
            outer_target_session=target,
            initialization_seed=initialization_seed,
            checkpoint_swa_rule=checkpoint_swa_rule,
            lambda_=lambda_,
            tau=tau,
        )
        for target in HELD_IN_SOURCE_SESSIONS
    )
    _require(len(pairs) == M1_OUTER_FOLD_COUNT
             and tuple(pair[0].outer_target_session for pair in pairs) == HELD_IN_SOURCE_SESSIONS,
             "CS-WG exact four-fold held-in LOSO topology drift")
    return pairs


def validate_matched_same_fold_pair(cswg: CSWGRunSpec, erm: CSWGRunSpec) -> None:
    _require(isinstance(cswg, CSWGRunSpec) and isinstance(erm, CSWGRunSpec)
             and cswg.system == "CS_WG" and erm.system == "MATCHED_ERM",
             "CS-WG matched pair system identity drift")
    for field_name in (
        "outer_target_session", "source_sessions", "initialization_seed", "optimizer_literal",
        "lr_schedule_literal", "epoch_budget", "checkpoint_swa_rule", "graph",
        "total_batch_size", "tau", "norm_quantile_count", "checkpoint_training_sessions",
        "calibration_budget", "adam_lr", "adam_weight_decay", "hidden_inner_grid_or_sweep_forbidden",
    ):
        _require(getattr(cswg, field_name) == getattr(erm, field_name),
                 f"CS-WG matched ERM differs on frozen same-fold field: {field_name}")
    _require(erm.lambda_ == 0.0 and 0.0 <= cswg.lambda_ <= 1.0,
             "CS-WG versus matched ERM objective boundary drift")
    _require(cswg.outer_target_session not in erm.checkpoint_training_sessions,
             "historical outer-target-trained checkpoint is not a valid matched ERM reference")


def dry_plan(root: Path | None = None) -> dict[str, object]:
    """Static public plan; it imports no Torch and touches no data or roots."""
    value: dict[str, object] = {
        "cell": CELL,
        "phase": "stage0_cpu_synthetic_only",
        "workorder_sha256": WORKORDER_SHA256,
        "baseline_graph": M1_GRAPH_CONTRACT.payload(),
        "route_owned_mixed_session_training_step": MIXED_SESSION_TRAINING_STEP_CONTRACT.payload(),
        "held_in_source_sessions": list(HELD_IN_SOURCE_SESSIONS),
        "outer_held_in_loso_targets": list(HELD_IN_SOURCE_SESSIONS),
        "held_in_m1_unit_id_shape_metadata_only": list(M1_HELD_IN_UNIT_ID_SHAPE),
        "m1_forward_input_keys": list(M1_FORWARD_INPUT_KEYS),
        "m1_b3s_max_trial_length": M1_B3S_MAX_TRIAL_LENGTH,
        "m1_calibration_shape_per_row": list(M1_CALIBRATION_SHAPE_PER_ROW),
        "physical_preflight_requires_exact_three_session_concat_compatibility": True,
        "physical_preflight_requires_per_row_calibration_and_session_ownership": True,
        "heterogeneous_unit_padding_or_masks_forbidden": True,
        "source_decoder_recipe": M1_SOURCE_DECODER_RECIPE.payload(),
        "official_held_out_sessions_forbidden": OFFICIAL_HELD_OUT_SESSIONS_FORBIDDEN,
        "outer_fold_quota": "11/11/10 rotating one low-quota session per step",
        "all_four_source_quota": "8/8/8/8",
        "future_four_fold_matched_erm_plus_cswg_training_runs": M1_MATCHED_DEVELOPMENT_RUN_COUNT,
        "historical_baseline_fold_wall_minutes": M1_HISTORICAL_FOLD_WALL_MINUTES,
        "historical_reference_minutes_for_eight_runs_if_each_matches_baseline": (
            M1_HISTORICAL_FOLD_WALL_MINUTES * M1_MATCHED_DEVELOPMENT_RUN_COUNT
        ),
        "actual_cswg_runtime_requires_measured_physical_receipts": True,
        "hidden_inner_grid_or_sweep_forbidden": True,
        "execution_authorized": False,
        "opens_source_or_target": False,
        "loads_checkpoint": False,
        "initializes_cuda": False,
        "creates_root_or_receipt": False,
        "trains": False,
        "scores": False,
        "launches": False,
    }
    if root is not None:
        value["closure"] = execution_closure_payload(Path(root))
    return value
