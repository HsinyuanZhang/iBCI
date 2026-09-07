"""Dry/fail-closed matched scorer for the Posterior Carrier full-training arm.

This is intentionally an additive scorer contract, not a shortcut around the
running training route.  Importing the module uses only the standard library:
it never imports Torch, opens a result/checkpoint/NWB/cache path, resolves a
within/external/H1 asset, initializes CUDA, creates a canonical authority or
score root, or launches work.  A later physical backend must be separately
audited against this contract after the immutable full terminal and SWA exist.

The score matrix keeps three concepts separate:

* the sealed Cell-D checkpoint is re-evaluated with the closed-form OLS point
  carrier at M30/M10/M4, forming the true matched point-system baseline;
* the posterior-trained system is evaluated at the same three fixed budgets;
* zero and cyclic wrong-pair are M30 posterior-system diagnostics only.  They
  cannot rescue the externally governed point-system comparison.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Protocol, Sequence


CELL = "POSTERIOR_CARRIER_BUDGETMIX_D_SEED42"
PHASE = "POSTERIOR_CARRIER_MATCHED_SCORE_V1"
SCHEMA = "posterior_carrier_matched_score_v1"

WORKORDER_RELATIVE = "tfpd_exploration/docs/WORKORDER_POSTERIOR_CARRIER_MATCHED_SCORE_20260822.md"
WORKORDER_SHA256 = "ecb6e6630ae9bbb0e214334868037e500341f093fa8ffa0e31ef91593513b79a"
FULL_TRAIN_ROOT_RELATIVE = "tfpd_exploration/results/posterior_carrier_budgetmix_d_seed42_full_train_v1"
FULL_TRAIN_TERMINAL_RELATIVE = f"{FULL_TRAIN_ROOT_RELATIVE}/terminal.json"
FULL_TRAIN_SWA_RELATIVE = f"{FULL_TRAIN_ROOT_RELATIVE}/swa_final4.pt"
FULL_TRAIN_SOURCE_AUTHORITY_RELATIVE = f"{FULL_TRAIN_ROOT_RELATIVE}/source_authority.json"
FULL_TRAIN_CHECKPOINT_RELATIVES = tuple(
    f"{FULL_TRAIN_ROOT_RELATIVE}/checkpoint-{epoch:02d}.pt" for epoch in (44, 45, 46, 47)
)
# A remote full-training result is never treated as if it were produced by the
# local score stage.  A root-reviewed import copies its immutable bytes into
# this separate, fresh local mirror and records the remote descriptor identity
# in the preflight.  The physical backend reads only this exact local mirror.
FULL_IMPORT_MIRROR_ROOT_RELATIVE = (
    "tfpd_exploration/results/posterior_carrier_budgetmix_d_seed42_full_train_import_mirror_v1"
)

AUTHORITY_ROOT_RELATIVE = "tfpd_exploration/results/posterior_carrier_budgetmix_d_seed42_score_authority_v1"
SCORE_ROOT_RELATIVE = "tfpd_exploration/results/posterior_carrier_budgetmix_d_seed42_score_v1"
AUTHORITY_TOPOLOGY = ("official_preflight.json", "root_authorization.json")
SCORE_TOPOLOGY = ("attempt.json", "input_authority.json", "score.json", "terminal.json", "failure.json")

WITHIN = "within"
EXTERNAL = "external"
H1 = "h1"
PUBLIC_SURFACES = (WITHIN, EXTERNAL)
EXPECTED_SESSION_COUNTS = {WITHIN: 6, EXTERNAL: 15}
BUDGETS = (30, 10, 4)
SAFETY_BUDGET = 30
HEADLINE_BUDGET = 4
PAIRED_BOOTSTRAP_DRAWS = 10_000
PAIRED_BOOTSTRAP_SEED = 42
PAIRED_BOOTSTRAP_CI_LEVEL = 0.95

SEALED_POINT_MODE = "sealed_cell_d_ols_point"
POSTERIOR_MODE = "posterior_mean_precision"
ZERO_MODE = "posterior_zero_carrier_m30_diagnostic"
WRONG_PAIR_MODE = "posterior_cyclic_wrong_pair_m30_diagnostic"

METRIC_CONTRACT = {
    "estimator": "tfpd_lane.matched_scorer.session_r2",
    "torchmetrics": "torchmetrics.regression.R2Score(multioutput='variance_weighted')",
    "query": "last_bin_only_of_each_valid_50_bin_window",
    "equal_weight_per_session": True,
    "window_bins": 50,
    "behavior_coordinates": 2,
    "paired_bootstrap": {
        "draws": PAIRED_BOOTSTRAP_DRAWS,
        "seed": PAIRED_BOOTSTRAP_SEED,
        "ci_level": PAIRED_BOOTSTRAP_CI_LEVEL,
        "sampler": "sha256_domain_separated_with_replacement",
    },
}

EXECUTION_POLICY = {
    "target_optimizer_steps": 0,
    "target_backward_calls": 0,
    "target_update_calls": 0,
    "normalizer_refit": False,
    "eval_mode": True,
    "dropout_disabled": True,
    "no_grad": True,
    "formal_opened": False,
    "h1_opened_without_separate_authority": False,
}

IMPLEMENTATION_CLOSURE = (
    WORKORDER_RELATIVE,
    "tfpd_exploration/src/__init__.py",
    "tfpd_exploration/src/posterior_carrier_v1/__init__.py",
    "tfpd_exploration/src/posterior_carrier_v1/plan.py",
    "tfpd_exploration/src/posterior_carrier_v1/matched_score.py",
    "tfpd_exploration/src/posterior_carrier_v1/matched_score_physical.py",
    "tfpd_exploration/src/posterior_carrier_v1/full_result_import.py",
    "tfpd_exploration/scripts/run_posterior_carrier_matched_score.py",
    "tfpd_exploration/scripts/run_posterior_carrier_full_result_import.py",
    "tfpd_exploration/tests/test_posterior_carrier_matched_score.py",
    "tfpd_exploration/tests/test_posterior_carrier_full_result_import.py",
    "tfpd_exploration/src/posterior_carrier_v1/full_train.py",
    "tfpd_exploration/src/posterior_carrier_v1/core.py",
    "tfpd_exploration/src/posterior_carrier_v1/source_adapter.py",
    "tfpd_exploration/src/posterior_carrier_v1/source_adapter_v2.py",
    "tfpd_exploration/src/posterior_carrier_v1/phase_b.py",
    "tfpd_exploration/src/posterior_carrier_v1/phase_b_v2.py",
    "tfpd_exploration/src/posterior_carrier_v1/phase_b_v3.py",
    "tfpd_exploration/src/tfpd_lane/arm_common.py",
    "tfpd_exploration/src/tfpd_lane/pop_robust.py",
    "tfpd_exploration/src/tfpd_lane/matched_scorer.py",
    "sua_exploration/mc_maze/__init__.py",
    "sua_exploration/mc_maze/datamodule.py",
    "sua_exploration/mc_maze/multisession_datamodule.py",
    "sua_exploration/mc_maze/unit_side_features.py",
    "streaming_calibration_exp/src/models/components/spint.py",
    "streaming_calibration_exp/src/models/components/streaming_encoders.py",
    "streaming_calibration_exp/src/models/components/streaming_spint.py",
    "streaming_calibration_exp/src/models/components/rt_ld_gain.py",
)

PHYSICAL_PREFLIGHT_SCHEMA = "posterior_carrier_matched_score_target_free_preflight_v1"
PHYSICAL_ROOT_AUTH_SCHEMA = "posterior_carrier_matched_score_root_authorization_v1"
PHYSICAL_INPUT_ASSET_SCHEMA = "posterior_carrier_matched_score_input_asset_v1"

# These are the sealed Cell-D training-path ordinary-T4 and behavior
# normalizer values.  They are closure-bound literals rather than values
# inferred from any within/external input.  The point replay is intentionally
# a sealed-system comparison; it must not silently acquire posterior
# point-budget-mix moments.
SEALED_CELL_D_OLS_T4_NORMALIZER_SEMANTIC_SHA256 = (
    "293b8a55417b7acbe5215404003b7d019f3b56b1199c7887dbf91e7dcd2ad5b0"
)
SEALED_CELL_D_OLS_T4_MEAN_FLOAT32 = (
    0.04627712443470955, 0.4544036388397217, 1.3432163000106812, 10.150517463684082,
)
SEALED_CELL_D_OLS_T4_STD_FLOAT32 = (
    1.126278281211853, 1.284820556640625, 1.2352101802825928, 9.115250587463379,
)
SEALED_BEHAVIOR_NORMALIZER_SEMANTIC_SHA256 = (
    "f062506cb1db65e2a0872c55af2b542a9e3638fc5588e86735cd293dc890a391"
)
SEALED_BEHAVIOR_MEAN_FLOAT32 = (-0.001148765324614942, 0.002653369214385748)
SEALED_BEHAVIOR_STD_FLOAT32 = (8.63547420501709, 8.086690902709961)


class ScoreError(RuntimeError):
    """Fail-closed error for posterior-carrier scoring provenance or semantics."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ScoreError(message)


def _json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha(value: object, label: str = "SHA-256") -> str:
    if not isinstance(value, str) or len(value) != 64 or any(item not in "0123456789abcdef" for item in value):
        raise ScoreError(f"{label} must be an exact lowercase SHA-256")
    return value


def _finite(value: object, label: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ScoreError(f"{label} must be finite")
    result = float(value)
    if nonnegative and result < 0:
        raise ScoreError(f"{label} must be nonnegative")
    return result


def _safe_name(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or "/" in value or "\\" in value or value in {".", ".."}:
        raise ScoreError(f"{label} must be one safe nonempty path component")
    return value


def _strict_unique_roster(value: Sequence[str], *, count: int, surface: str) -> tuple[str, ...]:
    if not isinstance(value, (tuple, list)):
        raise ScoreError(f"{surface} roster must be an ordered sequence")
    roster = tuple(value)
    if (len(roster) != count or len(set(roster)) != count
            or any(not isinstance(name, str) or not name for name in roster)):
        raise ScoreError(f"{surface} roster must contain exactly {count} unique nonempty sessions")
    return roster


@dataclass(frozen=True)
class ImplementationClosure:
    """Explicit source closure; no glob/path discovery is accepted."""

    sha256_by_path: Mapping[str, str]

    def payload(self) -> dict[str, object]:
        if not isinstance(self.sha256_by_path, Mapping) or set(self.sha256_by_path) != set(IMPLEMENTATION_CLOSURE):
            raise ScoreError("implementation closure path topology drift")
        hashes = {path: _sha(self.sha256_by_path[path], f"closure SHA {path}") for path in IMPLEMENTATION_CLOSURE}
        if hashes[WORKORDER_RELATIVE] != WORKORDER_SHA256:
            raise ScoreError("matched-score workorder SHA drift")
        body = {"paths": list(IMPLEMENTATION_CLOSURE), "sha256_by_path": hashes}
        return {**body, "closure_sha256": _digest(_json(body))}


def implementation_closure(root: Path) -> ImplementationClosure:
    """Read only explicitly named code/doc bytes; never a result or data path."""
    base = Path(root).absolute()
    hashes: dict[str, str] = {}
    for relative in IMPLEMENTATION_CLOSURE:
        path = base / relative
        info = os.lstat(path)
        if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
            raise ScoreError(f"implementation closure path is missing/aliased: {relative}")
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            opened = os.fstat(descriptor)
            if (not stat.S_ISREG(opened.st_mode) or stat.S_ISLNK(opened.st_mode)
                    or (opened.st_dev, opened.st_ino, opened.st_size) != (info.st_dev, info.st_ino, info.st_size)):
                raise ScoreError(f"implementation closure path identity drift: {relative}")
            blocks: list[bytes] = []
            while True:
                block = os.read(descriptor, 1 << 20)
                if not block:
                    break
                blocks.append(block)
            hashes[relative] = _digest(b"".join(blocks))
        finally:
            os.close(descriptor)
        after = os.lstat(path)
        if (after.st_dev, after.st_ino, after.st_size) != (info.st_dev, info.st_ino, info.st_size):
            raise ScoreError(f"implementation closure path changed during read: {relative}")
    return ImplementationClosure(sha256_by_path=hashes)


@dataclass(frozen=True)
class FullTrainingEvidence:
    """Terminal/full-SWA facts that a later descriptor-safe loader must prove."""

    terminal_sha256: str
    swa_sha256: str
    swa_state_sha256: str
    source_authority_sha256: str
    checkpoint_sha256s: Mapping[str, str]
    terminal_status: str
    full_closure_sha256: str

    def payload(self) -> dict[str, object]:
        expected_checkpoints = {"44", "45", "46", "47"}
        if not isinstance(self.checkpoint_sha256s, Mapping) or set(self.checkpoint_sha256s) != expected_checkpoints:
            raise ScoreError("full predecessor final-four checkpoint topology drift")
        checkpoints = {key: _sha(self.checkpoint_sha256s[key], f"checkpoint {key} SHA") for key in sorted(expected_checkpoints)}
        if self.terminal_status != "FULL_TRAINING_COMPLETE__SOURCE_ONLY__AWAITING_SEPARATE_SCORER":
            raise ScoreError("posterior full predecessor is not a completed source-only terminal")
        return {
            "terminal_relative": FULL_TRAIN_TERMINAL_RELATIVE,
            "terminal_sha256": _sha(self.terminal_sha256, "full terminal SHA"),
            "swa_relative": FULL_TRAIN_SWA_RELATIVE,
            "swa_sha256": _sha(self.swa_sha256, "full SWA SHA"),
            "swa_state_sha256": _sha(self.swa_state_sha256, "full SWA state SHA"),
            "source_authority_relative": FULL_TRAIN_SOURCE_AUTHORITY_RELATIVE,
            "source_authority_sha256": _sha(self.source_authority_sha256, "full source-authority SHA"),
            "checkpoint_sha256s": checkpoints,
            "terminal_status": self.terminal_status,
            "full_closure_sha256": _sha(self.full_closure_sha256, "full closure SHA"),
        }


@dataclass(frozen=True)
class SealedCellDEvidence:
    """Sealed point-system evidence; M30 parity is a prerequisite to M4/M10."""

    terminal_sha256: str
    swa_sha256: str
    m30_ols_point_last_bin_table_sha256: str
    sealed_point_replay_authority_sha256: str

    def payload(self) -> dict[str, object]:
        return {
            "terminal_sha256": _sha(self.terminal_sha256, "sealed Cell-D terminal SHA"),
            "swa_sha256": _sha(self.swa_sha256, "sealed Cell-D SWA SHA"),
            "m30_ols_point_last_bin_table_sha256": _sha(
                self.m30_ols_point_last_bin_table_sha256, "sealed Cell-D M30 OLS-point governing table SHA",
            ),
            "sealed_point_replay_authority_sha256": _sha(
                self.sealed_point_replay_authority_sha256, "sealed Cell-D point-replay authority SHA",
            ),
            "required_ols_point_budgets": list(BUDGETS),
            "native_m30_parity_required_before_successor_system_scoring": True,
        }


@dataclass(frozen=True)
class ScoreIdentity:
    """All future authority is explicit and target-free before input resolution."""

    full_training: FullTrainingEvidence
    sealed_cell_d: SealedCellDEvidence
    closure: ImplementationClosure
    within_roster: tuple[str, ...]
    external_roster: tuple[str, ...]

    def payload(self) -> dict[str, object]:
        closure = self.closure.payload()
        within = _strict_unique_roster(self.within_roster, count=EXPECTED_SESSION_COUNTS[WITHIN], surface=WITHIN)
        external = _strict_unique_roster(self.external_roster, count=EXPECTED_SESSION_COUNTS[EXTERNAL], surface=EXTERNAL)
        return {
            "cell": CELL,
            "phase": PHASE,
            "workorder": {"relative_path": WORKORDER_RELATIVE, "sha256": WORKORDER_SHA256},
            "full_training": self.full_training.payload(),
            "sealed_cell_d": self.sealed_cell_d.payload(),
            "closure": closure,
            "rosters": {WITHIN: list(within), EXTERNAL: list(external)},
            "metric": dict(METRIC_CONTRACT),
            "execution_policy": dict(EXECUTION_POLICY),
            "h1": {"included": False, "reason": "requires_separate_future_authority"},
        }


@dataclass(frozen=True)
class ScoreCell:
    surface: str
    mode: str
    budget: int
    role: str
    diagnostic: bool

    def __post_init__(self) -> None:
        if self.surface not in PUBLIC_SURFACES or self.budget not in BUDGETS:
            raise ScoreError("score-cell surface/budget drift")
        allowed = {
            # The baseline is a distinct, sealed Cell-D *system* replayed
            # with OLS point carriers.  It is not the posterior model fed a
            # point-like input, which would answer a different question.
            SEALED_POINT_MODE: (self.budget, "sealed_point_system", False),
            POSTERIOR_MODE: (self.budget, "posterior_system", False),
            ZERO_MODE: (SAFETY_BUDGET, "posterior_m30_diagnostic", True),
            WRONG_PAIR_MODE: (SAFETY_BUDGET, "posterior_m30_diagnostic", True),
        }
        if self.mode not in allowed:
            raise ScoreError("unknown score mode")
        expected_budget, expected_role, expected_diagnostic = allowed[self.mode]
        if self.budget != expected_budget or self.role != expected_role or self.diagnostic is not expected_diagnostic:
            raise ScoreError("score-cell attribution/role drift")

    def payload(self) -> dict[str, object]:
        return {
            "surface": self.surface,
            "mode": self.mode,
            "budget": self.budget,
            "role": self.role,
            "diagnostic": self.diagnostic,
        }


def score_matrix() -> tuple[ScoreCell, ...]:
    """One frozen system-by-budget order; no observed row can be selected."""
    cells: list[ScoreCell] = []
    for surface in PUBLIC_SURFACES:
        for budget in BUDGETS:
            cells.append(ScoreCell(surface, SEALED_POINT_MODE, budget, "sealed_point_system", False))
        for budget in BUDGETS:
            cells.append(ScoreCell(surface, POSTERIOR_MODE, budget, "posterior_system", False))
        # Diagnostics intentionally remain M30-only.  The receipt and gate
        # label their anti-triviality condition accordingly rather than using
        # them to rescue an M4 headline result.
        cells.append(ScoreCell(surface, ZERO_MODE, SAFETY_BUDGET, "posterior_m30_diagnostic", True))
        cells.append(ScoreCell(surface, WRONG_PAIR_MODE, SAFETY_BUDGET, "posterior_m30_diagnostic", True))
    return tuple(cells)


@dataclass(frozen=True)
class SessionInput:
    """Input facts shared by every forward mode for one session."""

    surface: str
    session: str
    n_windows: int
    neural_sha256: str
    calibration_m30_sha256: str
    last_bin_target_sha256: str
    last_bin_valid_mask_sha256: str
    last_bin_valid_count: int
    posterior_prefix_input_sha256s: Mapping[str, str]
    ols_point_prefix_input_sha256s: Mapping[str, str]
    matched_prefix_row_ids_sha256s: Mapping[str, str]
    normalized_posterior_carrier_sha256s: Mapping[str, str]
    normalized_ols_point_carrier_sha256s: Mapping[str, str]
    target_theta_recovery_evidence: Mapping[str, object]

    def payload(self) -> dict[str, object]:
        if self.surface not in PUBLIC_SURFACES or not isinstance(self.session, str) or not self.session:
            raise ScoreError("session-input surface/session drift")
        if type(self.n_windows) is not int or self.n_windows <= 0:
            raise ScoreError("session-input n_windows drift")
        if type(self.last_bin_valid_count) is not int or self.last_bin_valid_count != self.n_windows:
            raise ScoreError("session-input governing valid-count drift")
        prefix_keys = {str(item) for item in BUDGETS}
        for label, mapping in (
            ("posterior prefix", self.posterior_prefix_input_sha256s),
            ("OLS point prefix", self.ols_point_prefix_input_sha256s),
            ("matched posterior/OLS prefix rows", self.matched_prefix_row_ids_sha256s),
            ("normalized posterior carrier", self.normalized_posterior_carrier_sha256s),
            ("normalized OLS point carrier", self.normalized_ols_point_carrier_sha256s),
        ):
            if not isinstance(mapping, Mapping) or set(mapping) != prefix_keys:
                raise ScoreError(f"session-input {label} budget topology drift")
            for budget in prefix_keys:
                _sha(mapping[budget], f"session-input {label} M{budget} SHA")
        theta_recovery = _validate_target_theta_recovery_evidence(
            self.target_theta_recovery_evidence, session=self.session,
        )
        # The shared per-budget row IDs are a substantive cross-proof, not a
        # second caller hash.  The posterior recovery receipt exposes the M30
        # ordered prefix; M4/M10 must be its exact leading slices.
        for budget_text in sorted(prefix_keys):
            budget = int(budget_text)
            expected_prefix_digest = _digest(_json(theta_recovery["prefix_row_ids"][:budget]))
            if self.matched_prefix_row_ids_sha256s[budget_text] != expected_prefix_digest:
                raise ScoreError("session-input posterior/OLS shared prefix-row binding drift")
        return {
            "surface": self.surface,
            "session": self.session,
            "n_windows": self.n_windows,
            "neural_sha256": _sha(self.neural_sha256, "session neural SHA"),
            "calibration_m30_sha256": _sha(self.calibration_m30_sha256, "session calibration M30 SHA"),
            "last_bin_target_sha256": _sha(self.last_bin_target_sha256, "session last-bin target SHA"),
            "last_bin_valid_mask_sha256": _sha(self.last_bin_valid_mask_sha256, "session last-bin valid-mask SHA"),
            "last_bin_valid_count": self.last_bin_valid_count,
            "posterior_prefix_input_sha256s": {key: self.posterior_prefix_input_sha256s[key] for key in sorted(prefix_keys)},
            "ols_point_prefix_input_sha256s": {key: self.ols_point_prefix_input_sha256s[key] for key in sorted(prefix_keys)},
            "matched_prefix_row_ids_sha256s": {
                key: self.matched_prefix_row_ids_sha256s[key] for key in sorted(prefix_keys)
            },
            "point_and_posterior_prefix_rows_identical": True,
            "normalized_posterior_carrier_sha256s": {
                key: self.normalized_posterior_carrier_sha256s[key] for key in sorted(prefix_keys)
            },
            "normalized_ols_point_carrier_sha256s": {
                key: self.normalized_ols_point_carrier_sha256s[key] for key in sorted(prefix_keys)
            },
            "target_theta_recovery_evidence": theta_recovery,
            "target_theta_recovery_body_sha256": theta_recovery["body_sha256"],
            "target_theta_fallback_count": theta_recovery["fallback_count"],
        }


def _validate_target_theta_recovery_evidence(value: object, *, session: str) -> dict[str, object]:
    """Validate a target-local same-prefix fallback without a source aggregate.

    A target session may legitimately have a missing ``target_dir`` in an
    already selected M30 calibration row.  This scorer is allowed to preserve
    the exact same-row geometry fallback, but is *not* allowed to apply the
    strict-27 source aggregate fallback-topology gate to a target roster.  The
    durable input row therefore carries the full local evidence and binds each
    fallback's original trial index to its selected prefix position.
    """
    required = {
        "schema", "session_id", "prefix_row_ids", "prefix_rows_sha256", "theta_m30_sha256",
        "theta_sources_by_prefix_position", "fallback_rows", "fallback_count", "no_later_row_substitution",
        "design_matrix_rank", "design_matrix_condition", "body_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ScoreError("target theta-recovery evidence schema drift")
    copied = _copy_json_mapping(value, "target theta-recovery evidence")
    body = {key: copied[key] for key in required - {"body_sha256"}}
    if (
        copied["schema"] != "posterior_carrier_same_prefix_theta_recovery_v2"
        or copied["session_id"] != session
        or copied["body_sha256"] != _digest(_json(body))
        or copied["no_later_row_substitution"] is not True
        or copied["design_matrix_rank"] != 3
    ):
        raise ScoreError("target theta-recovery evidence binding drift")
    prefix_rows = copied["prefix_row_ids"]
    sources = copied["theta_sources_by_prefix_position"]
    fallbacks = copied["fallback_rows"]
    if (
        not isinstance(prefix_rows, list) or len(prefix_rows) != 30
        or len(set(prefix_rows)) != 30
        or any(not isinstance(row, str) or not row.startswith(f"{session}:trial:") for row in prefix_rows)
        or copied["prefix_rows_sha256"] != _digest(_json(prefix_rows))
        or not isinstance(sources, list) or len(sources) != 30
        or any(source not in {"native_target_dir", "same_prefix_target_corners_canonical_snap"} for source in sources)
        or not isinstance(fallbacks, list) or type(copied["fallback_count"]) is not int
        or copied["fallback_count"] != len(fallbacks)
        or sum(source == "same_prefix_target_corners_canonical_snap" for source in sources) != len(fallbacks)
    ):
        raise ScoreError("target theta-recovery local prefix topology drift")
    _sha(copied["theta_m30_sha256"], "target theta M30 SHA")
    _sha(copied["body_sha256"], "target theta-recovery body SHA")
    _finite(copied["design_matrix_condition"], "target theta design condition", nonnegative=True)
    seen: set[int] = set()
    for fallback in fallbacks:
        if not isinstance(fallback, Mapping):
            raise ScoreError("target theta fallback row schema drift")
        position, trial_index = fallback.get("prefix_position"), fallback.get("trial_index")
        if (
            type(position) is not int or not 0 <= position < 30 or position in seen
            or type(trial_index) is not int or fallback.get("source") != "same_prefix_trial_target_corners_xyxy_center"
            or sources[position] != "same_prefix_target_corners_canonical_snap"
            or prefix_rows[position] != f"{session}:trial:{trial_index}"
            or not isinstance(fallback.get("geometry"), Mapping)
        ):
            raise ScoreError("target theta fallback must bind the selected prefix row")
        seen.add(position)
    return copied


@dataclass(frozen=True)
class InputAuthority:
    """No-cache, one-pass input materialization authority for the matrix."""

    records: tuple[SessionInput, ...]
    source_posterior_normalizer_sha256: str
    behavior_normalizer_sha256: str
    shared_input_pass: bool
    cache_read_or_write: bool

    def payload(self, *, identity: ScoreIdentity) -> dict[str, object]:
        _require(self.shared_input_pass is True, "all score modes must share one physical input pass")
        _require(self.cache_read_or_write is False, "matched scorer forbids cache reads/writes")
        expected = {
            WITHIN: _strict_unique_roster(identity.within_roster, count=6, surface=WITHIN),
            EXTERNAL: _strict_unique_roster(identity.external_roster, count=15, surface=EXTERNAL),
        }
        if not isinstance(self.records, tuple) or len(self.records) != 21:
            raise ScoreError("input authority must contain exact within-6 plus external-15 records")
        rows = [record.payload() for record in self.records]
        # The roster authority, rather than an incidental lexical ordering of
        # session names, defines the one legal physical-pass order.  Some
        # sealed rosters are date ordered; sorting them here would silently
        # change the paired estimand.
        expected_order = [
            (surface, session)
            for surface in PUBLIC_SURFACES
            for session in expected[surface]
        ]
        actual_order = [(row["surface"], row["session"]) for row in rows]
        if actual_order != expected_order:
            raise ScoreError("input authority records must use exact roster-defined order")
        return {
            "schema": "posterior_carrier_matched_score_input_authority_v1",
            "identity": identity.payload(),
            "records": rows,
            "source_posterior_normalizer_sha256": _sha(
                self.source_posterior_normalizer_sha256, "posterior source normalizer SHA",
            ),
            "behavior_normalizer_sha256": _sha(self.behavior_normalizer_sha256, "behavior normalizer SHA"),
            "shared_input_pass": True,
            "cache_read_or_write": False,
            "target_optimizer_steps": 0,
            "target_backward_calls": 0,
            "target_update_calls": 0,
            "normalizer_refit": False,
            "formal_opened": False,
        }


@dataclass(frozen=True)
class SessionScore:
    session: str
    n_windows: int
    r2: float
    prediction_sha256: str
    input_record_sha256: str

    def payload(self) -> dict[str, object]:
        if not isinstance(self.session, str) or not self.session or type(self.n_windows) is not int or self.n_windows <= 0:
            raise ScoreError("session score session/window schema drift")
        return {
            "session": self.session,
            "n_windows": self.n_windows,
            "r2": _finite(self.r2, "session last-bin R2"),
            "prediction_sha256": _sha(self.prediction_sha256, "prediction SHA"),
            "input_record_sha256": _sha(self.input_record_sha256, "input-record SHA"),
        }


@dataclass(frozen=True)
class ModeEvidence:
    """Forward-only result for exactly one frozen score cell."""

    cell: ScoreCell
    model_system: str
    model_swa_sha256: str
    sessions: tuple[SessionScore, ...]
    input_authority_sha256: str
    model_state_before_sha256: str
    model_state_after_sha256: str
    eval_mode: bool
    dropout_disabled: bool
    gradients_none: bool
    finite_outputs: bool
    repeated_fixed_batch_bitwise_equal: bool
    b3s_m30_recomputed: bool
    posterior_mean_eval: bool
    credibility_mode: str
    target_optimizer_steps: int = 0
    target_backward_calls: int = 0
    target_update_calls: int = 0
    normalizer_refit: bool = False

    def payload(self, *, input_payload: Mapping[str, object], identity: ScoreIdentity) -> dict[str, object]:
        cell = self.cell
        expected_roster = (
            identity.within_roster if cell.surface == WITHIN else identity.external_roster
        )
        if not isinstance(self.sessions, tuple) or tuple(item.session for item in self.sessions) != tuple(expected_roster):
            raise ScoreError("mode evidence session roster/order drift")
        inputs_by_session = {
            row["session"]: row for row in input_payload["records"] if row["surface"] == cell.surface
        }
        values = [item.payload() for item in self.sessions]
        for value in values:
            expected_input = inputs_by_session.get(value["session"])
            if expected_input is None or value["n_windows"] != expected_input["n_windows"]:
                raise ScoreError("mode evidence session window count/input drift")
            if value["input_record_sha256"] != _digest(_json(expected_input)):
                raise ScoreError("mode evidence did not bind exact shared input record")
        expected_credibility = {
            SEALED_POINT_MODE: "sealed_cell_d_ols_point_no_posterior_bias",
            POSTERIOR_MODE: "posterior_precision_logit_bias",
            ZERO_MODE: "posterior_zero_carrier_m30_diagnostic",
            WRONG_PAIR_MODE: "posterior_cyclic_wrong_pair_m30_diagnostic",
        }[cell.mode]
        expected_posterior_mean_eval = cell.mode != SEALED_POINT_MODE
        expected_model_system = "sealed_cell_d_checkpoint" if cell.mode == SEALED_POINT_MODE else "posterior_carrier_full_swa"
        expected_model_swa_sha256 = (
            identity.sealed_cell_d.swa_sha256 if cell.mode == SEALED_POINT_MODE else identity.full_training.swa_sha256
        )
        if (
            self.input_authority_sha256 != _digest(_json(input_payload))
            or self.model_state_before_sha256 != self.model_state_after_sha256
            or self.eval_mode is not True or self.dropout_disabled is not True
            or self.gradients_none is not True or self.finite_outputs is not True
            or self.repeated_fixed_batch_bitwise_equal is not True or self.b3s_m30_recomputed is not True
            or self.posterior_mean_eval is not expected_posterior_mean_eval or self.credibility_mode != expected_credibility
            or self.model_system != expected_model_system or self.model_swa_sha256 != expected_model_swa_sha256
            or self.target_optimizer_steps != 0 or self.target_backward_calls != 0
            or self.target_update_calls != 0 or self.normalizer_refit is not False
        ):
            raise ScoreError("mode evidence forward-only/model-state/carrier semantics drift")
        return {
            "cell": cell.payload(),
            "model_system": expected_model_system,
            "model_swa_sha256": _sha(expected_model_swa_sha256, "mode model SWA SHA"),
            "sessions": values,
            "input_authority_sha256": _sha(self.input_authority_sha256, "mode input-authority SHA"),
            "model_state_before_sha256": _sha(self.model_state_before_sha256, "model-state-before SHA"),
            "model_state_after_sha256": _sha(self.model_state_after_sha256, "model-state-after SHA"),
            "eval_mode": True,
            "dropout_disabled": True,
            "gradients_none": True,
            "finite_outputs": True,
            "repeated_fixed_batch_bitwise_equal": True,
            "b3s_m30_recomputed": True,
            "posterior_mean_eval": expected_posterior_mean_eval,
            "credibility_mode": expected_credibility,
            "target_optimizer_steps": 0,
            "target_backward_calls": 0,
            "target_update_calls": 0,
            "normalizer_refit": False,
        }


@dataclass
class ScoreFlags:
    """Runtime facts that cannot be inferred after a forward pass."""

    within_opened: bool = False
    external_opened: bool = False
    h1_opened: bool = False
    formal_opened: bool = False
    target_optimizer_steps: int = 0
    target_backward_calls: int = 0
    target_update_calls: int = 0
    forward_cells: list[tuple[str, str, int]] = field(default_factory=list)

    def record_forward(self, cell: ScoreCell) -> None:
        self.forward_cells.append((cell.surface, cell.mode, cell.budget))

    def payload(self) -> dict[str, object]:
        return {
            "within_opened": self.within_opened,
            "external_opened": self.external_opened,
            "h1_opened": self.h1_opened,
            "formal_opened": self.formal_opened,
            "target_optimizer_steps": self.target_optimizer_steps,
            "target_backward_calls": self.target_backward_calls,
            "target_update_calls": self.target_update_calls,
            "forward_cells": [list(item) for item in self.forward_cells],
        }


def _validate_flags(flags: ScoreFlags, *, expected_cells: Sequence[ScoreCell]) -> None:
    expected = [(cell.surface, cell.mode, cell.budget) for cell in expected_cells]
    if (
        flags.within_opened is not True or flags.external_opened is not True
        or flags.h1_opened is not False or flags.formal_opened is not False
        or flags.target_optimizer_steps != 0 or flags.target_backward_calls != 0
        or flags.target_update_calls != 0 or flags.forward_cells != expected
    ):
        raise ScoreError("matched-score read/update/forward ordering boundary drift")


def _median(values: Sequence[float], *, label: str) -> float:
    if not values:
        raise ScoreError(f"{label} requires at least one value")
    ordered = sorted(_finite(value, label) for value in values)
    middle = len(ordered) // 2
    return ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2.0


def _deterministic_paired_bootstrap_ci(deltas: Sequence[float], *, label: str) -> dict[str, object]:
    """A domain-separated bootstrap that never consumes host/global RNG state."""
    if not deltas:
        raise ScoreError("paired bootstrap requires nonempty deltas")
    values = tuple(_finite(value, "paired bootstrap delta") for value in deltas)
    draws: list[float] = []
    domain = f"{CELL}|{PHASE}|paired-bootstrap-v1|{PAIRED_BOOTSTRAP_SEED}|{label}".encode("utf-8")
    for draw in range(PAIRED_BOOTSTRAP_DRAWS):
        total = 0.0
        for position in range(len(values)):
            token = hashlib.sha256(domain + b"|" + str(draw).encode("ascii") + b"|" + str(position).encode("ascii")).digest()
            index = int.from_bytes(token[:8], "big", signed=False) % len(values)
            total += values[index]
        draws.append(total / len(values))
    ordered = sorted(draws)
    tail = (1.0 - PAIRED_BOOTSTRAP_CI_LEVEL) / 2.0
    lower_index = int(math.floor(tail * (PAIRED_BOOTSTRAP_DRAWS - 1)))
    upper_index = int(math.ceil((1.0 - tail) * (PAIRED_BOOTSTRAP_DRAWS - 1)))
    return {
        "method": "sha256_domain_separated_paired_bootstrap_with_replacement",
        "seed": PAIRED_BOOTSTRAP_SEED,
        "draws": PAIRED_BOOTSTRAP_DRAWS,
        "ci_level": PAIRED_BOOTSTRAP_CI_LEVEL,
        "lower_index": lower_index,
        "upper_index": upper_index,
        "lower": ordered[lower_index],
        "upper": ordered[upper_index],
    }


def _paired_summary(left: Mapping[str, float], right: Mapping[str, float], *, label: str) -> dict[str, object]:
    if tuple(left) != tuple(right) or not left:
        raise ScoreError(f"paired comparison {label} session topology drift")
    deltas = [left[session] - right[session] for session in left]
    return {
        "label": label,
        "sessions": list(left),
        "deltas": deltas,
        "mean": sum(deltas) / len(deltas),
        "median": _median(deltas, label=f"{label} paired median"),
        "n_positive": sum(item > 0.0 for item in deltas),
        "n_total": len(deltas),
        "min": min(deltas),
        "max": max(deltas),
        "paired_bootstrap_ci": _deterministic_paired_bootstrap_ci(deltas, label=label),
    }


def _mode_lookup(evidence: Sequence[Mapping[str, object]], *, surface: str, mode: str, budget: int) -> Mapping[str, object]:
    expected = [
        cell for cell in score_matrix()
        if cell.surface == surface and cell.mode == mode and cell.budget == budget
    ]
    if len(expected) != 1:
        raise ScoreError("frozen score-matrix lookup drift")
    hits = [item for item in evidence if item["cell"] == expected[0].payload()]
    if len(hits) != 1:
        raise ScoreError("score evidence cell topology drift")
    return hits[0]


def _session_value_map(payload: Mapping[str, object]) -> dict[str, float]:
    sessions = payload.get("sessions")
    if not isinstance(sessions, list):
        raise ScoreError("score evidence sessions schema drift")
    return {str(row["session"]): float(row["r2"]) for row in sessions}


def _system_budget_summary(payload: Mapping[str, object], *, system: str, surface: str, budget: int) -> dict[str, object]:
    values = _session_value_map(payload)
    if len(values) != EXPECTED_SESSION_COUNTS[surface]:
        raise ScoreError("system/budget session count drift")
    ordered = list(values)
    r2 = [values[session] for session in ordered]
    return {
        "system": system,
        "surface": surface,
        "budget": budget,
        "sessions": ordered,
        "r2": r2,
        "equal_session_mean": sum(r2) / len(r2),
        "equal_session_median": _median(r2, label=f"{system} M{budget} {surface} R2"),
        "n_sessions": len(r2),
    }


def _degradation(m30: Mapping[str, float], lower_budget: Mapping[str, float], *, label: str) -> dict[str, object]:
    if tuple(m30) != tuple(lower_budget):
        raise ScoreError(f"{label} degradation session topology drift")
    values = [m30[session] - lower_budget[session] for session in m30]
    return {
        "label": label,
        "sessions": list(m30),
        "per_session_m30_minus_lower": values,
        "equal_session_mean": sum(values) / len(values),
        "equal_session_median": _median(values, label=label),
    }


def _verdict(*, paired: Mapping[str, Mapping[str, object]],
             degradation: Mapping[str, Mapping[str, object]],
             diagnostics: Mapping[str, object]) -> dict[str, object]:
    """Exact external-only STOP/HOLD/PASS decision from the frozen handoff."""
    m30 = paired["m30"]
    m4 = paired["m4"]
    posterior_curve = degradation["posterior_m30_to_m4"]["equal_session_mean"]
    point_curve = degradation["sealed_point_m30_to_m4"]["equal_session_mean"]
    posterior_aligned = diagnostics["posterior_aligned_m30"]["equal_session_mean"]
    zero = diagnostics["posterior_zero_m30"]["equal_session_mean"]
    wrong = diagnostics["posterior_cyclic_wrong_pair_m30"]["equal_session_mean"]
    gates = {
        "m30_safety_mean_delta_gte_minus_0_02": m30["mean"] >= -0.02,
        "m4_headline_mean_delta_gte_plus_0_03": m4["mean"] >= 0.03,
        "m4_external_positive_sessions_gte_9_of_15": m4["n_positive"] >= 9 and m4["n_total"] == 15,
        "curve_posterior_m30_to_m4_less_than_sealed_point": posterior_curve < point_curve,
        # Diagnostics are explicitly M30-only; their truth value cannot alter
        # a failed safety/headline/breadth/curve condition.
        "m30_anti_triviality_posterior_above_zero_and_wrong_pair": (
            posterior_aligned > zero and posterior_aligned > wrong
        ),
    }
    if not gates["m30_safety_mean_delta_gte_minus_0_02"]:
        status = "STOP"
    elif all(gates.values()):
        status = "PASS"
    else:
        status = "HOLD"
    return {
        "surface": EXTERNAL,
        "status": status,
        "rule": (
            "STOP_if_M30_safety_fails_else_PASS_if_M4_headline_breadth_curve_and_M30_anti_triviality_all_pass_else_HOLD"
        ),
        "m30_safety_delta_mean": m30["mean"],
        "m4_headline_delta_mean": m4["mean"],
        "m4_positive_sessions": m4["n_positive"],
        "m4_total_sessions": m4["n_total"],
        "posterior_m30_to_m4_degradation": posterior_curve,
        "sealed_point_m30_to_m4_degradation": point_curve,
        "posterior_m30_mean": posterior_aligned,
        "zero_m30_mean": zero,
        "wrong_pair_m30_mean": wrong,
        "gates": gates,
        "diagnostic_scope": "posterior_system_M30_only",
    }


def build_score_payload(*, identity: ScoreIdentity, input_payload: Mapping[str, object],
                        evidence: Sequence[Mapping[str, object]], flags: ScoreFlags) -> dict[str, object]:
    """Construct the immutable result without selecting a system or budget."""
    cells = score_matrix()
    if len(evidence) != len(cells):
        raise ScoreError("score evidence matrix count drift")
    for actual, expected in zip(evidence, cells):
        if actual.get("cell") != expected.payload():
            raise ScoreError("score evidence matrix ordering drift")
    _validate_flags(flags, expected_cells=cells)
    systems: dict[str, object] = {}
    paired: dict[str, object] = {}
    degradation: dict[str, object] = {}
    diagnostics: dict[str, object] = {}
    for surface in PUBLIC_SURFACES:
        point_by_budget = {
            budget: _session_value_map(_mode_lookup(
                evidence, surface=surface, mode=SEALED_POINT_MODE, budget=budget,
            ))
            for budget in BUDGETS
        }
        posterior_by_budget = {
            budget: _session_value_map(_mode_lookup(
                evidence, surface=surface, mode=POSTERIOR_MODE, budget=budget,
            ))
            for budget in BUDGETS
        }
        systems[surface] = {
            "sealed_cell_d_ols_point": {
                f"m{budget}": _system_budget_summary(
                    _mode_lookup(evidence, surface=surface, mode=SEALED_POINT_MODE, budget=budget),
                    system="sealed_cell_d_ols_point", surface=surface, budget=budget,
                )
                for budget in BUDGETS
            },
            "posterior_carrier": {
                f"m{budget}": _system_budget_summary(
                    _mode_lookup(evidence, surface=surface, mode=POSTERIOR_MODE, budget=budget),
                    system="posterior_carrier", surface=surface, budget=budget,
                )
                for budget in BUDGETS
            },
        }
        paired[surface] = {
            f"m{budget}": _paired_summary(
                posterior_by_budget[budget], point_by_budget[budget],
                label=f"{surface}_posterior_minus_sealed_point_m{budget}",
            )
            for budget in BUDGETS
        }
        degradation[surface] = {
            "posterior_m30_to_m10": _degradation(
                posterior_by_budget[30], posterior_by_budget[10], label=f"{surface}_posterior_m30_to_m10",
            ),
            "posterior_m30_to_m4": _degradation(
                posterior_by_budget[30], posterior_by_budget[4], label=f"{surface}_posterior_m30_to_m4",
            ),
            "sealed_point_m30_to_m10": _degradation(
                point_by_budget[30], point_by_budget[10], label=f"{surface}_sealed_point_m30_to_m10",
            ),
            "sealed_point_m30_to_m4": _degradation(
                point_by_budget[30], point_by_budget[4], label=f"{surface}_sealed_point_m30_to_m4",
            ),
        }
        zero_m30 = _session_value_map(_mode_lookup(
            evidence, surface=surface, mode=ZERO_MODE, budget=SAFETY_BUDGET,
        ))
        wrong_m30 = _session_value_map(_mode_lookup(
            evidence, surface=surface, mode=WRONG_PAIR_MODE, budget=SAFETY_BUDGET,
        ))
        diagnostics[surface] = {
            "scope": "posterior_system_M30_only__non_governing",
            "posterior_aligned_m30": _system_budget_summary(
                _mode_lookup(evidence, surface=surface, mode=POSTERIOR_MODE, budget=SAFETY_BUDGET),
                system="posterior_carrier", surface=surface, budget=SAFETY_BUDGET,
            ),
            "posterior_zero_m30": _system_budget_summary(
                _mode_lookup(evidence, surface=surface, mode=ZERO_MODE, budget=SAFETY_BUDGET),
                system="posterior_carrier", surface=surface, budget=SAFETY_BUDGET,
            ),
            "posterior_cyclic_wrong_pair_m30": _system_budget_summary(
                _mode_lookup(evidence, surface=surface, mode=WRONG_PAIR_MODE, budget=SAFETY_BUDGET),
                system="posterior_carrier", surface=surface, budget=SAFETY_BUDGET,
            ),
            "zero_minus_aligned": _paired_summary(
                zero_m30, posterior_by_budget[30], label=f"{surface}_posterior_zero_minus_aligned_m30",
            ),
            "wrong_pair_minus_aligned": _paired_summary(
                wrong_m30, posterior_by_budget[30], label=f"{surface}_posterior_wrong_pair_minus_aligned_m30",
            ),
        }
    verdict = _verdict(
        paired=paired[EXTERNAL], degradation=degradation[EXTERNAL], diagnostics=diagnostics[EXTERNAL],
    )
    return {
        "schema": SCHEMA,
        "cell": CELL,
        "phase": PHASE,
        "identity": identity.payload(),
        "input_authority_sha256": _digest(_json(input_payload)),
        "metric": dict(METRIC_CONTRACT),
        "matrix": list(evidence),
        "system_budget_r2": systems,
        "paired_posterior_minus_sealed_point": paired,
        "degradation_predeclared": degradation,
        "diagnostics_m30_only_non_governing": diagnostics,
        "external_verdict": verdict,
        "no_posthoc_budget_or_diagnostic_selection": True,
        "h1_not_used_as_external_substitute": True,
        "flags": flags.payload(),
        "status": "MATCHED_SCORE_COMPLETE__M4_HEADLINE_PREDECLARED__DIAGNOSTICS_NON_GOVERNING",
    }


def validate_score_payload(value: Mapping[str, object], *, identity: ScoreIdentity,
                           input_payload: Mapping[str, object]) -> dict[str, object]:
    expected = {
        "schema", "cell", "phase", "identity", "input_authority_sha256", "metric", "matrix",
        "system_budget_r2", "paired_posterior_minus_sealed_point", "degradation_predeclared",
        "diagnostics_m30_only_non_governing", "external_verdict",
        "no_posthoc_budget_or_diagnostic_selection", "h1_not_used_as_external_substitute", "flags", "status",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ScoreError("score payload schema drift")
    if (
        value["schema"] != SCHEMA or value["cell"] != CELL or value["phase"] != PHASE
        or value["identity"] != identity.payload() or value["input_authority_sha256"] != _digest(_json(input_payload))
        or value["metric"] != METRIC_CONTRACT or value["no_posthoc_budget_or_diagnostic_selection"] is not True
        or value["h1_not_used_as_external_substitute"] is not True
        or value["status"] != "MATCHED_SCORE_COMPLETE__M4_HEADLINE_PREDECLARED__DIAGNOSTICS_NON_GOVERNING"
        or not isinstance(value["matrix"], list) or not isinstance(value["flags"], Mapping)
    ):
        raise ScoreError("score payload immutable binding drift")
    # Reconstructing from evidence prevents a forged aggregate, a favorable
    # budget substitution, or diagnostics masquerading as a primary outcome.
    flags = ScoreFlags(
        within_opened=value["flags"].get("within_opened", False),
        external_opened=value["flags"].get("external_opened", False),
        h1_opened=value["flags"].get("h1_opened", False),
        formal_opened=value["flags"].get("formal_opened", False),
        target_optimizer_steps=value["flags"].get("target_optimizer_steps", -1),
        target_backward_calls=value["flags"].get("target_backward_calls", -1),
        target_update_calls=value["flags"].get("target_update_calls", -1),
        forward_cells=[tuple(item) for item in value["flags"].get("forward_cells", [])],
    )
    rebuilt = build_score_payload(identity=identity, input_payload=input_payload, evidence=value["matrix"], flags=flags)
    if rebuilt != dict(value):
        raise ScoreError("score payload aggregate/selection drift")
    return rebuilt


def _directory_identity(path: Path) -> tuple[int, int]:
    info = os.lstat(path)
    if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise ScoreError("expected one real directory")
    return (info.st_dev, info.st_ino)


def _read_fd_all(descriptor: int) -> bytes:
    blocks: list[bytes] = []
    while True:
        block = os.read(descriptor, 1 << 20)
        if not block:
            return b"".join(blocks)
        blocks.append(block)


def _write_full(descriptor: int, body: bytes) -> None:
    cursor = 0
    while cursor < len(body):
        written = os.write(descriptor, body[cursor:])
        if written <= 0:
            raise ScoreError("short artifact write")
        cursor += written


@dataclass(frozen=True)
class ArtifactRoot:
    """A directory capability for one fresh immutable receipt topology.

    The implementation deliberately keeps writes descriptor-relative and
    verifies both the named child and its parent identity.  This is useful in
    synthetic lifecycle tests today and is the same fail-closed publication
    boundary a future reviewed physical scorer would use.
    """

    directory: Path
    topology: tuple[str, ...]
    identity: tuple[int, int]
    parent: Path
    parent_identity: tuple[int, int]

    def _check_name(self, name: str) -> None:
        if not isinstance(name, str) or name not in self.topology or "/" in name or name in {"", ".", ".."}:
            raise ScoreError("artifact leaf is outside exact topology")

    def _assert_named_identity(self) -> None:
        if _directory_identity(self.directory) != self.identity:
            raise ScoreError("artifact root path identity drift")
        try:
            parent_fd = os.open(self.parent, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0))
        except OSError as error:
            raise ScoreError("cannot open artifact parent") from error
        try:
            parent_info = os.fstat(parent_fd)
            if (parent_info.st_dev, parent_info.st_ino) != self.parent_identity:
                raise ScoreError("artifact parent identity drift")
            named = os.stat(self.directory.name, dir_fd=parent_fd, follow_symlinks=False)
            if (
                not stat.S_ISDIR(named.st_mode)
                or stat.S_ISLNK(named.st_mode)
                or (named.st_dev, named.st_ino) != self.identity
            ):
                raise ScoreError("artifact named-root identity drift")
        finally:
            os.close(parent_fd)

    @staticmethod
    def _leaf_exists(directory_fd: int, name: str) -> bool:
        try:
            os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        except FileNotFoundError:
            return False
        return True

    def has_name(self, name: str) -> bool:
        self._check_name(name)
        self._assert_named_identity()
        directory_fd = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0))
        try:
            return self._leaf_exists(directory_fd, name)
        finally:
            os.close(directory_fd)

    def publish_bytes(self, name: str, body: bytes) -> str:
        """Publish one body/sidecar pair or remove only newly owned leaves."""
        self._check_name(name)
        if not isinstance(body, bytes):
            raise ScoreError("artifact body must be bytes")
        self._assert_named_identity()
        directory_fd = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0))
        made: list[tuple[str, int, int]] = []
        try:
            info = os.fstat(directory_fd)
            if (info.st_dev, info.st_ino) != self.identity:
                raise ScoreError("artifact root changed before publication")
            if self._leaf_exists(directory_fd, name) or self._leaf_exists(directory_fd, name + ".sha256"):
                raise ScoreError("artifact body/sidecar collision")
            digest = _digest(body)
            leaves = (
                (name, body),
                (name + ".sha256", f"{digest}  {name}\n".encode("ascii")),
            )
            for leaf, payload in leaves:
                fd = os.open(
                    leaf,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                    0o600,
                    dir_fd=directory_fd,
                )
                try:
                    opened = os.fstat(fd)
                    if not stat.S_ISREG(opened.st_mode):
                        raise ScoreError("artifact O_EXCL did not create regular leaf")
                    made.append((leaf, opened.st_dev, opened.st_ino))
                    _write_full(fd, payload)
                    os.fchmod(fd, 0o444)
                    os.fsync(fd)
                finally:
                    os.close(fd)
            os.fsync(directory_fd)
            if self.reload_pair(name, digest) != body:
                raise ScoreError("artifact post-publication body reload drift")
            self._assert_named_identity()
            return digest
        except BaseException:
            for leaf, device, inode in reversed(made):
                try:
                    current = os.stat(leaf, dir_fd=directory_fd, follow_symlinks=False)
                    if (current.st_dev, current.st_ino) == (device, inode):
                        os.unlink(leaf, dir_fd=directory_fd)
                except OSError:
                    pass
            try:
                os.fsync(directory_fd)
            except OSError:
                pass
            raise
        finally:
            os.close(directory_fd)

    def publish_json(self, name: str, value: Mapping[str, object]) -> str:
        return self.publish_bytes(name, _json(value))

    def reload_pair(self, name: str, expected_sha256: str | None = None) -> bytes:
        self._check_name(name)
        self._assert_named_identity()
        directory_fd = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0))
        try:
            def read(leaf: str) -> bytes:
                try:
                    fd = os.open(leaf, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=directory_fd)
                except OSError as error:
                    raise ScoreError(f"artifact leaf missing/corrupt: {leaf}") from error
                try:
                    info = os.fstat(fd)
                    if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o444:
                        raise ScoreError("artifact leaf type/mode drift")
                    return _read_fd_all(fd)
                finally:
                    os.close(fd)

            body = read(name)
            digest = _digest(body)
            if expected_sha256 is not None and digest != _sha(expected_sha256, "expected artifact SHA"):
                raise ScoreError("artifact body SHA drift")
            if read(name + ".sha256") != f"{digest}  {name}\n".encode("ascii"):
                raise ScoreError("artifact sidecar drift")
            self._assert_named_identity()
            return body
        finally:
            os.close(directory_fd)

    def reload_json(self, name: str, expected_sha256: str | None = None) -> dict[str, object]:
        try:
            value = json.loads(self.reload_pair(name, expected_sha256))
        except (TypeError, json.JSONDecodeError) as error:
            raise ScoreError("artifact JSON decode drift") from error
        if not isinstance(value, dict):
            raise ScoreError("artifact JSON root must be object")
        return value

    def publish_group(
        self,
        bodies: Mapping[str, bytes],
        *,
        post_publish: Callable[[Mapping[str, bytes], Mapping[str, str]], None] | None = None,
    ) -> dict[str, str]:
        """Atomically publish a scientific receipt group or roll it back.

        ``score.json`` has no independent scientific meaning.  The lifecycle
        therefore uses this only for score+terminal together.
        """
        if not isinstance(bodies, Mapping) or not bodies:
            raise ScoreError("artifact publication group must be nonempty")
        names = tuple(sorted(bodies))
        if len(names) != len(set(names)):
            raise ScoreError("duplicate publication group leaf")
        for name in names:
            self._check_name(name)
            if not isinstance(bodies[name], bytes):
                raise ScoreError("artifact group body must be bytes")
        self._assert_named_identity()
        directory_fd = os.open(self.directory, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0))
        made: list[tuple[str, int, int]] = []
        try:
            if (os.fstat(directory_fd).st_dev, os.fstat(directory_fd).st_ino) != self.identity:
                raise ScoreError("artifact root changed before group publication")
            for name in names:
                if self._leaf_exists(directory_fd, name) or self._leaf_exists(directory_fd, name + ".sha256"):
                    raise ScoreError("artifact group body/sidecar collision")
            digests = {name: _digest(bodies[name]) for name in names}
            leaves: list[tuple[str, bytes]] = []
            for name in names:
                leaves.extend(((name, bodies[name]), (name + ".sha256", f"{digests[name]}  {name}\n".encode("ascii"))))
            for leaf, payload in leaves:
                fd = os.open(
                    leaf,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                    0o600,
                    dir_fd=directory_fd,
                )
                try:
                    opened = os.fstat(fd)
                    if not stat.S_ISREG(opened.st_mode):
                        raise ScoreError("artifact group O_EXCL did not create regular leaf")
                    made.append((leaf, opened.st_dev, opened.st_ino))
                    _write_full(fd, payload)
                    os.fchmod(fd, 0o444)
                    os.fsync(fd)
                finally:
                    os.close(fd)
            os.fsync(directory_fd)
            for name in names:
                self.reload_pair(name, digests[name])
            if post_publish is not None:
                post_publish(dict(bodies), dict(digests))
            self._assert_named_identity()
            return digests
        except BaseException:
            for leaf, device, inode in reversed(made):
                try:
                    current = os.stat(leaf, dir_fd=directory_fd, follow_symlinks=False)
                    if (current.st_dev, current.st_ino) == (device, inode):
                        os.unlink(leaf, dir_fd=directory_fd)
                except OSError:
                    pass
            try:
                os.fsync(directory_fd)
            except OSError:
                pass
            raise
        finally:
            os.close(directory_fd)


def reserve_artifact_root(parent: Path, name: str, *, topology: tuple[str, ...] = SCORE_TOPOLOGY) -> ArtifactRoot:
    """Reserve a fresh synthetic or later-reviewed canonical artifact root."""
    _safe_name(name, "artifact root name")
    if (
        not isinstance(topology, tuple)
        or not topology
        or len(topology) != len(set(topology))
        or any(not isinstance(item, str) or not item or "/" in item for item in topology)
    ):
        raise ScoreError("artifact root topology drift")
    parent = Path(parent).absolute()
    parent_identity = _directory_identity(parent)
    parent_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0))
    try:
        parent_info = os.fstat(parent_fd)
        if (parent_info.st_dev, parent_info.st_ino) != parent_identity:
            raise ScoreError("artifact parent changed before reservation")
        try:
            os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise ScoreError("fresh artifact root required")
        os.mkdir(name, 0o755, dir_fd=parent_fd)
        os.fsync(parent_fd)
        made = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        if not stat.S_ISDIR(made.st_mode) or stat.S_ISLNK(made.st_mode):
            raise ScoreError("reserved artifact root is not one real directory")
        identity = (made.st_dev, made.st_ino)
    finally:
        os.close(parent_fd)
    return ArtifactRoot(parent / name, topology, identity, parent, parent_identity)


def _session_input_from_payload(value: Mapping[str, object]) -> SessionInput:
    expected = {
        "surface", "session", "n_windows", "neural_sha256", "calibration_m30_sha256",
        "last_bin_target_sha256", "last_bin_valid_mask_sha256", "last_bin_valid_count",
        "posterior_prefix_input_sha256s", "ols_point_prefix_input_sha256s",
        "matched_prefix_row_ids_sha256s", "point_and_posterior_prefix_rows_identical",
        "normalized_posterior_carrier_sha256s", "normalized_ols_point_carrier_sha256s",
        "target_theta_recovery_evidence", "target_theta_recovery_body_sha256", "target_theta_fallback_count",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ScoreError("session-input payload schema drift")
    return SessionInput(
        surface=value["surface"], session=value["session"], n_windows=value["n_windows"],
        neural_sha256=value["neural_sha256"], calibration_m30_sha256=value["calibration_m30_sha256"],
        last_bin_target_sha256=value["last_bin_target_sha256"],
        last_bin_valid_mask_sha256=value["last_bin_valid_mask_sha256"],
        last_bin_valid_count=value["last_bin_valid_count"],
        posterior_prefix_input_sha256s=value["posterior_prefix_input_sha256s"],
        ols_point_prefix_input_sha256s=value["ols_point_prefix_input_sha256s"],
        matched_prefix_row_ids_sha256s=value["matched_prefix_row_ids_sha256s"],
        normalized_posterior_carrier_sha256s=value["normalized_posterior_carrier_sha256s"],
        normalized_ols_point_carrier_sha256s=value["normalized_ols_point_carrier_sha256s"],
        target_theta_recovery_evidence=value["target_theta_recovery_evidence"],
    )


def validate_input_authority_payload(value: Mapping[str, object], *, identity: ScoreIdentity) -> dict[str, object]:
    expected = {
        "schema", "identity", "records", "source_posterior_normalizer_sha256", "behavior_normalizer_sha256",
        "shared_input_pass", "cache_read_or_write", "target_optimizer_steps", "target_backward_calls",
        "target_update_calls", "normalizer_refit", "formal_opened",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ScoreError("input authority schema drift")
    if value["identity"] != identity.payload() or not isinstance(value["records"], list):
        raise ScoreError("input authority identity/records drift")
    rebuilt = InputAuthority(
        records=tuple(_session_input_from_payload(item) for item in value["records"]),
        source_posterior_normalizer_sha256=value["source_posterior_normalizer_sha256"],
        behavior_normalizer_sha256=value["behavior_normalizer_sha256"],
        shared_input_pass=value["shared_input_pass"],
        cache_read_or_write=value["cache_read_or_write"],
    ).payload(identity=identity)
    if rebuilt != dict(value):
        raise ScoreError("input authority exact reconstruction drift")
    return rebuilt


def _score_cell_from_payload(value: Mapping[str, object]) -> ScoreCell:
    expected = {"surface", "mode", "budget", "role", "diagnostic"}
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ScoreError("score-cell payload schema drift")
    return ScoreCell(
        surface=value["surface"], mode=value["mode"], budget=value["budget"],
        role=value["role"], diagnostic=value["diagnostic"],
    )


def _session_score_from_payload(value: Mapping[str, object]) -> SessionScore:
    expected = {"session", "n_windows", "r2", "prediction_sha256", "input_record_sha256"}
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ScoreError("session-score payload schema drift")
    return SessionScore(
        session=value["session"], n_windows=value["n_windows"], r2=value["r2"],
        prediction_sha256=value["prediction_sha256"], input_record_sha256=value["input_record_sha256"],
    )


def validate_mode_payload(value: Mapping[str, object], *, input_payload: Mapping[str, object],
                          identity: ScoreIdentity) -> dict[str, object]:
    expected = {
        "cell", "model_system", "model_swa_sha256", "sessions", "input_authority_sha256",
        "model_state_before_sha256", "model_state_after_sha256",
        "eval_mode", "dropout_disabled", "gradients_none", "finite_outputs",
        "repeated_fixed_batch_bitwise_equal", "b3s_m30_recomputed", "posterior_mean_eval", "credibility_mode",
        "target_optimizer_steps", "target_backward_calls", "target_update_calls", "normalizer_refit",
    }
    if not isinstance(value, Mapping) or set(value) != expected or not isinstance(value["sessions"], list):
        raise ScoreError("mode evidence payload schema drift")
    rebuilt = ModeEvidence(
        cell=_score_cell_from_payload(value["cell"]),
        model_system=value["model_system"], model_swa_sha256=value["model_swa_sha256"],
        sessions=tuple(_session_score_from_payload(item) for item in value["sessions"]),
        input_authority_sha256=value["input_authority_sha256"],
        model_state_before_sha256=value["model_state_before_sha256"],
        model_state_after_sha256=value["model_state_after_sha256"],
        eval_mode=value["eval_mode"], dropout_disabled=value["dropout_disabled"],
        gradients_none=value["gradients_none"], finite_outputs=value["finite_outputs"],
        repeated_fixed_batch_bitwise_equal=value["repeated_fixed_batch_bitwise_equal"],
        b3s_m30_recomputed=value["b3s_m30_recomputed"], posterior_mean_eval=value["posterior_mean_eval"],
        credibility_mode=value["credibility_mode"], target_optimizer_steps=value["target_optimizer_steps"],
        target_backward_calls=value["target_backward_calls"], target_update_calls=value["target_update_calls"],
        normalizer_refit=value["normalizer_refit"],
    ).payload(input_payload=input_payload, identity=identity)
    if rebuilt != dict(value):
        raise ScoreError("mode evidence exact reconstruction drift")
    return rebuilt


def validate_score_identity(identity: ScoreIdentity) -> dict[str, object]:
    """Validate every static binding before a public path can resolve assets."""
    payload = identity.payload()
    if payload["h1"] != {"included": False, "reason": "requires_separate_future_authority"}:
        raise ScoreError("H1 boundary drift")
    if tuple(cell.payload() for cell in score_matrix()) != tuple(
        cell.payload() for cell in score_matrix()
    ):
        raise ScoreError("score matrix non-determinism")
    return payload


def _physical_module() -> object:
    """Load the standard-library physical substrate only on an explicit path.

    The helper itself has no Torch/data imports, but keeping it out of the dry
    import path makes the public CLI's boundary self-evident and keeps future
    physical dependency loading behind the same reviewed capability gate.
    """
    import importlib.util
    import sys

    relative = "tfpd_exploration/src/posterior_carrier_v1/matched_score_physical.py"
    path = Path(__file__).resolve().parents[3] / relative
    spec = importlib.util.spec_from_file_location("_posterior_carrier_matched_score_physical", path)
    if spec is None or spec.loader is None:
        raise ScoreError("cannot load closure-bound physical scorer substrate")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _copy_json_mapping(value: Mapping[str, object], label: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ScoreError(f"{label} must be a mapping")
    try:
        copied = json.loads(_json(value))
    except (TypeError, ValueError) as error:
        raise ScoreError(f"{label} must be JSON-serializable") from error
    if not isinstance(copied, dict):
        raise ScoreError(f"{label} JSON root drift")
    return copied


def _validate_input_assets(
    value: object,
    *,
    identity: ScoreIdentity,
) -> dict[str, list[dict[str, object]]]:
    """Validate durable rows that were derived by the physical authority path.

    This function intentionally has no caller-to-authority conversion.  The
    only construction route is ``derive_fixed_input_assets`` in the physical
    substrate, and final authorization descriptor-rederives it before any
    evaluator asset factory can run.
    """
    physical = _physical_module()
    try:
        return physical.validate_fixed_input_assets(
            value, within_roster=identity.within_roster, external_roster=identity.external_roster,
        )
    except physical.PhysicalScoreError as error:
        raise ScoreError(f"fixed input-asset authority drift: {error}") from error


def _validate_normalizer_bindings(
    value: object,
    *,
    full_source_authority_sha256: str,
    sealed_point_replay_authority_sha256: str,
    expected_posterior_payload: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Bind the distinct sealed-point and posterior normalizer families.

    The original Cell-D point system is deliberately *not* recast as a
    point-budget-mix-trained control.  Its M10/M4 replay reuses the ordinary
    OLS normalizer from its own sealed training lineage.  The posterior arm,
    in contrast, uses its new distributional source authority.  Keeping these
    provenance roots separate makes the comparison honest rather than
    accidentally claiming a one-factor budget-training control.
    """
    expected = {"ordinary_point", "posterior_distribution", "behavior", "no_target_refit"}
    if not isinstance(value, Mapping) or set(value) != expected or value.get("no_target_refit") is not True:
        raise ScoreError("physical normalizer binding schema/no-refit drift")
    ordinary = value.get("ordinary_point")
    posterior = value.get("posterior_distribution")
    behavior = value.get("behavior")
    ordinary_keys = {
        "semantic_sha256", "sealed_cell_d_point_replay_authority_sha256", "mean", "std",
        "raw_before_normalization", "system_comparison_role", "not_point_budget_mix_trained_control",
    }
    posterior_keys = {"body_sha256", "source_authority_sha256", "payload", "raw_before_normalization"}
    behavior_keys = {"semantic_sha256", "source_authority_sha256", "mean", "std"}
    if (
        not isinstance(ordinary, Mapping) or set(ordinary) != ordinary_keys
        or not isinstance(posterior, Mapping) or set(posterior) != posterior_keys
        or not isinstance(behavior, Mapping) or set(behavior) != behavior_keys
        or ordinary.get("raw_before_normalization") is not True
        or posterior.get("raw_before_normalization") is not True
        or ordinary.get("sealed_cell_d_point_replay_authority_sha256")
        != sealed_point_replay_authority_sha256
        or ordinary.get("system_comparison_role") != "sealed_cell_d_training_time_ols_normalizer_reused_for_m30_m10_m4_replay"
        or ordinary.get("not_point_budget_mix_trained_control") is not True
        or ordinary.get("semantic_sha256") != SEALED_CELL_D_OLS_T4_NORMALIZER_SEMANTIC_SHA256
        or ordinary.get("mean") != list(SEALED_CELL_D_OLS_T4_MEAN_FLOAT32)
        or ordinary.get("std") != list(SEALED_CELL_D_OLS_T4_STD_FLOAT32)
        or posterior.get("source_authority_sha256") != full_source_authority_sha256
        or behavior.get("source_authority_sha256") != full_source_authority_sha256
        or behavior.get("semantic_sha256") != SEALED_BEHAVIOR_NORMALIZER_SEMANTIC_SHA256
        or behavior.get("mean") != list(SEALED_BEHAVIOR_MEAN_FLOAT32)
        or behavior.get("std") != list(SEALED_BEHAVIOR_STD_FLOAT32)
    ):
        raise ScoreError("physical source-only normalizer cross-binding drift")
    for label, block in (("ordinary point", ordinary), ("behavior", behavior)):
        mean, std = block.get("mean"), block.get("std")
        expected_len = 4 if label == "ordinary point" else 2
        if (
            not isinstance(mean, list) or not isinstance(std, list) or len(mean) != expected_len or len(std) != expected_len
            or any(isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item)) for item in mean)
            or any(isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item)) or float(item) <= 0 for item in std)
        ):
            raise ScoreError(f"physical {label} source normalizer numeric drift")
        _sha(block.get("semantic_sha256"), f"physical {label} normalizer semantic SHA")
    payload = _copy_json_mapping(posterior.get("payload"), "posterior source normalizer payload")
    if (
        payload.get("body_sha256") != posterior.get("body_sha256")
        or payload.get("body_sha256") != _digest(_json({key: item for key, item in payload.items() if key != "body_sha256"}))
    ):
        raise ScoreError("posterior normalizer body/payload SHA drift")
    _sha(posterior.get("body_sha256"), "posterior source normalizer body SHA")
    if expected_posterior_payload is not None:
        expected_payload = _copy_json_mapping(expected_posterior_payload, "derived full posterior normalizer payload")
        if payload != expected_payload:
            raise ScoreError("posterior normalizer does not equal the unique completed-full source authority")
    return {
        "ordinary_point": _copy_json_mapping(ordinary, "ordinary point normalizer"),
        "posterior_distribution": {
            "body_sha256": posterior["body_sha256"],
            "source_authority_sha256": posterior["source_authority_sha256"],
            "payload": payload,
            "raw_before_normalization": True,
        },
        "behavior": _copy_json_mapping(behavior, "behavior normalizer"),
        "no_target_refit": True,
    }


def _normalizer_bindings_from_frozen_authorities(
    *,
    root: Path,
    physical: object,
    provenance: object,
    identity: ScoreIdentity,
) -> dict[str, object]:
    """Build the two normalizer families without accepting caller numerics.

    The sealed Cell-D OLS and behavior moments are closure-bound literal
    values.  The posterior normalizer is descriptor-derived from the one
    completed full source authority, including its exact body payload.  This
    operation validates only receipt bytes; it does not deserialize a model
    checkpoint or resolve an evaluation input.
    """
    try:
        posterior_payload = physical.derive_posterior_normalizer_from_full_mirror(
            Path(root), provenance=provenance,
            terminal_sha256=identity.full_training.terminal_sha256,
            swa_sha256=identity.full_training.swa_sha256,
            swa_state_sha256=identity.full_training.swa_state_sha256,
            source_authority_sha256=identity.full_training.source_authority_sha256,
            checkpoint_sha256s=identity.full_training.checkpoint_sha256s,
            full_closure_sha256=identity.full_training.full_closure_sha256,
        )
    except physical.PhysicalScoreError as error:
        raise ScoreError(f"cannot derive frozen posterior normalizer from completed full authority: {error}") from error
    candidate = {
        "ordinary_point": {
            "semantic_sha256": SEALED_CELL_D_OLS_T4_NORMALIZER_SEMANTIC_SHA256,
            "sealed_cell_d_point_replay_authority_sha256": identity.sealed_cell_d.sealed_point_replay_authority_sha256,
            "mean": list(SEALED_CELL_D_OLS_T4_MEAN_FLOAT32),
            "std": list(SEALED_CELL_D_OLS_T4_STD_FLOAT32),
            "raw_before_normalization": True,
            "system_comparison_role": "sealed_cell_d_training_time_ols_normalizer_reused_for_m30_m10_m4_replay",
            "not_point_budget_mix_trained_control": True,
        },
        "posterior_distribution": {
            "body_sha256": posterior_payload.get("body_sha256"),
            "source_authority_sha256": identity.full_training.source_authority_sha256,
            "payload": posterior_payload,
            "raw_before_normalization": True,
        },
        "behavior": {
            "semantic_sha256": SEALED_BEHAVIOR_NORMALIZER_SEMANTIC_SHA256,
            "source_authority_sha256": identity.full_training.source_authority_sha256,
            "mean": list(SEALED_BEHAVIOR_MEAN_FLOAT32),
            "std": list(SEALED_BEHAVIOR_STD_FLOAT32),
        },
        "no_target_refit": True,
    }
    return _validate_normalizer_bindings(
        candidate, full_source_authority_sha256=identity.full_training.source_authority_sha256,
        sealed_point_replay_authority_sha256=identity.sealed_cell_d.sealed_point_replay_authority_sha256,
        expected_posterior_payload=posterior_payload,
    )


def _validate_device_contract(value: object) -> dict[str, object]:
    """Keep the physical device exact without guessing a future scorer host."""
    expected = {
        "cuda_visible_devices", "cuda_device_order", "logical_device", "uuid", "bdf", "name",
        "nvidia_smi_memory_total_mib", "torch_total_memory_bytes", "torch_version", "torch_cuda_version", "cudnn_version",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ScoreError("physical device contract schema drift")
    strings = ("cuda_visible_devices", "cuda_device_order", "logical_device", "uuid", "bdf", "name", "torch_version", "torch_cuda_version")
    if any(not isinstance(value.get(key), str) or not value[key] for key in strings):
        raise ScoreError("physical device contract string field drift")
    if (
        value["cuda_visible_devices"] != "0" or value["cuda_device_order"] != "PCI_BUS_ID"
        or value["logical_device"] != "cuda:0" or not value["uuid"].startswith("GPU-")
        or len(value["bdf"].split(":")) != 3
        or type(value["nvidia_smi_memory_total_mib"]) is not int or value["nvidia_smi_memory_total_mib"] <= 0
        or type(value["torch_total_memory_bytes"]) is not int or value["torch_total_memory_bytes"] <= 0
        or type(value["cudnn_version"]) is not int or value["cudnn_version"] <= 0
    ):
        raise ScoreError("physical device contract numeric/topology drift")
    return _copy_json_mapping(value, "physical device contract")


def build_target_free_preflight(
    *,
    root: Path,
    identity: ScoreIdentity,
    full_import_provenance: Mapping[str, object],
    sealed_cell_d_artifacts: Mapping[str, object],
    device_contract: Mapping[str, object],
) -> dict[str, object]:
    """Build, but never publish, the root-reviewable physical score preflight."""
    identity_payload = validate_score_identity(identity)
    physical = _physical_module()
    provenance = physical.provenance_from_payload(full_import_provenance)
    if provenance.local_mirror_relative != FULL_IMPORT_MIRROR_ROOT_RELATIVE:
        raise ScoreError("full import provenance local mirror root drift")
    if provenance.remote_terminal_sha256 != identity.full_training.terminal_sha256:
        raise ScoreError("full import provenance/score identity terminal drift")
    sealed_expected = {
        "terminal_relative": physical.SEALED_CELL_D_TERMINAL_RELATIVE,
        "terminal_sha256": identity.sealed_cell_d.terminal_sha256,
        "swa_relative": physical.SEALED_CELL_D_SWA_RELATIVE,
        "swa_sha256": identity.sealed_cell_d.swa_sha256,
        "baseline_relative": physical.SEALED_CELL_D_BASELINE_RELATIVE,
        "baseline_receipt_sha256": physical.SEALED_CELL_D_BASELINE_SHA256,
        "m30_ols_point_last_bin_table_sha256": identity.sealed_cell_d.m30_ols_point_last_bin_table_sha256,
        "sealed_point_replay_authority_sha256": identity.sealed_cell_d.sealed_point_replay_authority_sha256,
    }
    if _copy_json_mapping(sealed_cell_d_artifacts, "sealed Cell-D artifact binding") != sealed_expected:
        raise ScoreError("sealed Cell-D physical artifact binding drift")
    # The caller can name the reviewed score roster but cannot inject rows.
    # This descriptor-read operation is metadata-only: it does not resolve an
    # NWB target pathname or open a model tensor.
    try:
        derived_assets = physical.derive_fixed_input_assets(
            Path(root), within_roster=identity.within_roster, external_roster=identity.external_roster,
        )
    except physical.PhysicalScoreError as error:
        raise ScoreError(f"fixed input-asset derivation failed before preflight publication: {error}") from error
    checked_assets = _validate_input_assets(derived_assets, identity=identity)
    checked_normalizers = _normalizer_bindings_from_frozen_authorities(
        root=Path(root), physical=physical, provenance=provenance, identity=identity,
    )
    closure = identity.closure.payload()
    payload = {
        "schema": PHYSICAL_PREFLIGHT_SCHEMA,
        "status": "PREFLIGHT_ACCEPTED__NO_TARGET_PATH_RESOLVED",
        "cell": CELL,
        "phase": PHASE,
        "identity": identity_payload,
        "implementation_closure": closure,
        "full_import_provenance": provenance.payload(),
        "sealed_cell_d_artifacts": sealed_expected,
        "input_assets": checked_assets,
        "normalizers": checked_normalizers,
        "device_contract": _validate_device_contract(device_contract),
        "authority_root_relative": AUTHORITY_ROOT_RELATIVE,
        "score_root_relative": SCORE_ROOT_RELATIVE,
        "score_topology": list(SCORE_TOPOLOGY),
        "full_predecessor_required_before_execution": True,
        "remote_stage_identity_rebuilt_locally": False,
        "boundaries": dict(EXECUTION_POLICY),
    }
    return validate_target_free_preflight(payload, identity=identity)


def validate_target_free_preflight(value: Mapping[str, object], *, identity: ScoreIdentity) -> dict[str, object]:
    expected = {
        "schema", "status", "cell", "phase", "identity", "implementation_closure", "full_import_provenance",
        "sealed_cell_d_artifacts", "input_assets", "normalizers", "device_contract", "authority_root_relative",
        "score_root_relative", "score_topology", "full_predecessor_required_before_execution",
        "remote_stage_identity_rebuilt_locally", "boundaries",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ScoreError("physical target-free preflight schema drift")
    if (
        value.get("schema") != PHYSICAL_PREFLIGHT_SCHEMA
        or value.get("status") != "PREFLIGHT_ACCEPTED__NO_TARGET_PATH_RESOLVED"
        or value.get("cell") != CELL or value.get("phase") != PHASE
        or value.get("identity") != validate_score_identity(identity)
        or value.get("implementation_closure") != identity.closure.payload()
        or value.get("authority_root_relative") != AUTHORITY_ROOT_RELATIVE
        or value.get("score_root_relative") != SCORE_ROOT_RELATIVE
        or value.get("score_topology") != list(SCORE_TOPOLOGY)
        or value.get("full_predecessor_required_before_execution") is not True
        or value.get("remote_stage_identity_rebuilt_locally") is not False
        or value.get("boundaries") != EXECUTION_POLICY
    ):
        raise ScoreError("physical target-free preflight immutable binding drift")
    physical = _physical_module()
    provenance = physical.provenance_from_payload(value.get("full_import_provenance"))
    if provenance.local_mirror_relative != FULL_IMPORT_MIRROR_ROOT_RELATIVE:
        raise ScoreError("physical target-free preflight local mirror root drift")
    if provenance.remote_terminal_sha256 != identity.full_training.terminal_sha256:
        raise ScoreError("physical target-free preflight full import terminal drift")
    expected_sealed = {
        "terminal_relative": physical.SEALED_CELL_D_TERMINAL_RELATIVE,
        "terminal_sha256": identity.sealed_cell_d.terminal_sha256,
        "swa_relative": physical.SEALED_CELL_D_SWA_RELATIVE,
        "swa_sha256": identity.sealed_cell_d.swa_sha256,
        "baseline_relative": physical.SEALED_CELL_D_BASELINE_RELATIVE,
        "baseline_receipt_sha256": physical.SEALED_CELL_D_BASELINE_SHA256,
        "m30_ols_point_last_bin_table_sha256": identity.sealed_cell_d.m30_ols_point_last_bin_table_sha256,
        "sealed_point_replay_authority_sha256": identity.sealed_cell_d.sealed_point_replay_authority_sha256,
    }
    if value.get("sealed_cell_d_artifacts") != expected_sealed:
        raise ScoreError("physical target-free preflight sealed Cell-D binding drift")
    _validate_input_assets(value.get("input_assets"), identity=identity)
    _validate_normalizer_bindings(
        value.get("normalizers"),
        full_source_authority_sha256=identity.full_training.source_authority_sha256,
        sealed_point_replay_authority_sha256=identity.sealed_cell_d.sealed_point_replay_authority_sha256,
    )
    _validate_device_contract(value.get("device_contract"))
    return _copy_json_mapping(value, "physical target-free preflight")


def _rederive_preflight_authorities(
    root: Path,
    *,
    preflight: Mapping[str, object],
    identity: ScoreIdentity,
) -> None:
    """Require durable rows/moments to equal fresh fixed-authority derivation.

    The schema validator above is intentionally path-free so it can reload an
    already immutable receipt.  Any transition that may publish or consume a
    preflight must additionally call this routine.  It performs metadata and
    completed-receipt reads only; no evaluation asset pathname, model tensor,
    Torch, or CUDA object is touched.
    """
    physical = _physical_module()
    try:
        assets = physical.derive_fixed_input_assets(
            Path(root), within_roster=identity.within_roster, external_roster=identity.external_roster,
        )
    except physical.PhysicalScoreError as error:
        raise ScoreError(f"fixed input-asset re-derivation failed: {error}") from error
    if assets != preflight.get("input_assets"):
        raise ScoreError("preflight input assets differ from fixed manifest/ledger re-derivation")
    provenance = physical.provenance_from_payload(preflight.get("full_import_provenance"))
    normalizers = _normalizer_bindings_from_frozen_authorities(
        root=Path(root), physical=physical, provenance=provenance, identity=identity,
    )
    if normalizers != preflight.get("normalizers"):
        raise ScoreError("preflight normalizers differ from completed-full authority re-derivation")


def build_root_authorization(*, official_preflight_sha256: str, preflight: Mapping[str, object]) -> dict[str, object]:
    _sha(official_preflight_sha256, "official physical preflight SHA")
    if not isinstance(preflight, Mapping):
        raise ScoreError("physical root authorization requires preflight mapping")
    return {
        "schema": PHYSICAL_ROOT_AUTH_SCHEMA,
        "status": "ROOT_AUTHORIZED",
        "cell": CELL,
        "phase": PHASE,
        "official_preflight_sha256": official_preflight_sha256,
        "identity_sha256": _digest(_json(preflight.get("identity"))),
        "implementation_closure": _copy_json_mapping(preflight.get("implementation_closure"), "authorization closure"),
        "full_import_provenance_sha256": preflight.get("full_import_provenance", {}).get("provenance_sha256")
        if isinstance(preflight.get("full_import_provenance"), Mapping) else None,
        "device_contract": _copy_json_mapping(preflight.get("device_contract"), "authorization device contract"),
        "authority_root_relative": AUTHORITY_ROOT_RELATIVE,
        "score_root_relative": SCORE_ROOT_RELATIVE,
        "target_free_preflight_required": True,
        "explicit_execution_capability_required": True,
    }


def validate_root_authorization(
    value: Mapping[str, object], *, official_preflight_sha256: str, preflight: Mapping[str, object], identity: ScoreIdentity,
) -> dict[str, object]:
    preflight_checked = validate_target_free_preflight(preflight, identity=identity)
    expected = {
        "schema", "status", "cell", "phase", "official_preflight_sha256", "identity_sha256", "implementation_closure",
        "full_import_provenance_sha256", "device_contract", "authority_root_relative", "score_root_relative",
        "target_free_preflight_required", "explicit_execution_capability_required",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ScoreError("physical root authorization schema drift")
    rebuilt = build_root_authorization(official_preflight_sha256=official_preflight_sha256, preflight=preflight_checked)
    if rebuilt != dict(value):
        raise ScoreError("physical root authorization exact binding drift")
    return rebuilt


_EXECUTION_SEAL = object()
_ROOT_PUBLICATION_SEAL = object()
_PHYSICAL_EXECUTION_SEAL = object()


@dataclass(frozen=True)
class ExecutionCapability:
    """Opaque in-process review capability; public CLI flags cannot forge it."""

    identity_sha256: str
    _seal: object = field(repr=False, compare=False)


@dataclass(frozen=True)
class RootPublicationCapability:
    """Opaque capability for the two durable target-free authority receipts."""

    _seal: object = field(repr=False, compare=False)


@dataclass(frozen=True)
class PhysicalExecutionCapability:
    """Opaque second-stage capability bound to the reviewed preflight body."""

    identity_sha256: str
    official_preflight_sha256: str
    _seal: object = field(repr=False, compare=False)


def _issue_execution_capability(identity: ScoreIdentity) -> ExecutionCapability:
    return ExecutionCapability(_digest(_json(validate_score_identity(identity))), _EXECUTION_SEAL)


def _require_execution_capability(value: object, *, identity: ScoreIdentity) -> ExecutionCapability:
    if not isinstance(value, ExecutionCapability) or value._seal is not _EXECUTION_SEAL:
        raise ScoreError("in-process root-reviewed execution capability required")
    if value.identity_sha256 != _digest(_json(validate_score_identity(identity))):
        raise ScoreError("execution capability identity drift")
    return value


def _issue_root_publication_capability() -> RootPublicationCapability:
    return RootPublicationCapability(_ROOT_PUBLICATION_SEAL)


def _issue_physical_execution_capability(
    identity: ScoreIdentity, *, official_preflight_sha256: str,
) -> PhysicalExecutionCapability:
    return PhysicalExecutionCapability(
        identity_sha256=_digest(_json(validate_score_identity(identity))),
        official_preflight_sha256=_sha(official_preflight_sha256, "official physical preflight SHA"),
        _seal=_PHYSICAL_EXECUTION_SEAL,
    )


def _require_physical_execution_capability(
    value: object, *, identity: ScoreIdentity, official_preflight_sha256: str,
) -> PhysicalExecutionCapability:
    if (
        not isinstance(value, PhysicalExecutionCapability) or value._seal is not _PHYSICAL_EXECUTION_SEAL
        or value.identity_sha256 != _digest(_json(validate_score_identity(identity)))
        or value.official_preflight_sha256 != _sha(official_preflight_sha256, "official physical preflight SHA")
    ):
        raise ScoreError("root-reviewed physical execution capability required before target resolution")
    return value


def _canonical_results_parent(root: Path) -> Path:
    parent = Path(root).absolute() / "tfpd_exploration" / "results"
    _directory_identity(parent)
    return parent


def reserve_authority_artifact(root: Path, capability: object) -> ArtifactRoot:
    if not isinstance(capability, RootPublicationCapability) or capability._seal is not _ROOT_PUBLICATION_SEAL:
        raise ScoreError("only root may reserve posterior score authority artifact")
    return reserve_artifact_root(_canonical_results_parent(root), Path(AUTHORITY_ROOT_RELATIVE).name, topology=AUTHORITY_TOPOLOGY)


def reserve_score_artifact(root: Path, capability: object) -> ArtifactRoot:
    if not isinstance(capability, PhysicalExecutionCapability) or capability._seal is not _PHYSICAL_EXECUTION_SEAL:
        raise ScoreError("only reviewed physical capability may reserve posterior score artifact")
    return reserve_artifact_root(_canonical_results_parent(root), Path(SCORE_ROOT_RELATIVE).name, topology=SCORE_TOPOLOGY)


def publish_target_free_preflight(
    artifact: ArtifactRoot,
    *,
    root: Path,
    capability: object,
    payload: Mapping[str, object],
    identity: ScoreIdentity,
) -> str:
    if not isinstance(capability, RootPublicationCapability) or capability._seal is not _ROOT_PUBLICATION_SEAL:
        raise ScoreError("only root may publish posterior score preflight")
    if artifact.topology != AUTHORITY_TOPOLOGY or artifact.has_name("root_authorization.json"):
        raise ScoreError("posterior score authority publication topology/order drift")
    checked = validate_target_free_preflight(payload, identity=identity)
    _rederive_preflight_authorities(Path(root), preflight=checked, identity=identity)
    return artifact.publish_json("official_preflight.json", checked)


def publish_root_authorization(
    artifact: ArtifactRoot,
    *,
    root: Path,
    capability: object,
    payload: Mapping[str, object],
    identity: ScoreIdentity,
) -> str:
    if not isinstance(capability, RootPublicationCapability) or capability._seal is not _ROOT_PUBLICATION_SEAL:
        raise ScoreError("only root may publish posterior score authorization")
    if artifact.topology != AUTHORITY_TOPOLOGY or not artifact.has_name("official_preflight.json"):
        raise ScoreError("posterior score root authorization requires durable preflight")
    preflight_body = artifact.reload_pair("official_preflight.json")
    try:
        preflight = json.loads(preflight_body)
    except (TypeError, json.JSONDecodeError) as error:
        raise ScoreError("durable posterior score preflight JSON drift") from error
    if not isinstance(preflight, Mapping):
        raise ScoreError("durable posterior score preflight root drift")
    checked = validate_target_free_preflight(preflight, identity=identity)
    _rederive_preflight_authorities(Path(root), preflight=checked, identity=identity)
    authorization = validate_root_authorization(
        payload, official_preflight_sha256=_digest(preflight_body), preflight=checked, identity=identity,
    )
    return artifact.publish_json("root_authorization.json", authorization)


def verify_physical_authorization(root: Path, *, identity: ScoreIdentity) -> tuple[dict[str, object], str, dict[str, object], str]:
    """Reload reviewed authority through a held FD before a score root exists."""
    physical = _physical_module()
    directory = Path(root).absolute() / AUTHORITY_ROOT_RELATIVE
    with physical.ImmutableDirectory.open(directory) as held:
        if set(held.names()) != {
            "official_preflight.json", "official_preflight.json.sha256",
            "root_authorization.json", "root_authorization.json.sha256",
        }:
            raise ScoreError("posterior score authority immutable topology drift")
        preflight_file = held.read_pair("official_preflight.json")
        authorization_file = held.read_pair("root_authorization.json")
        held.reverify()
    preflight = preflight_file.json_object()
    authorization = authorization_file.json_object()
    checked = validate_target_free_preflight(preflight, identity=identity)
    checked_auth = validate_root_authorization(
        authorization, official_preflight_sha256=preflight_file.sha256, preflight=checked, identity=identity,
    )
    if implementation_closure(Path(root)).payload() != checked["implementation_closure"]:
        raise ScoreError("current physical scorer closure differs from reviewed preflight")
    _rederive_preflight_authorities(Path(root), preflight=checked, identity=identity)
    return checked, preflight_file.sha256, checked_auth, authorization_file.sha256


class ScoreBackend(Protocol):
    """Dependency-injected backend contract; production stays unlicensed here."""

    def prepare(self, *, identity: ScoreIdentity, flags: ScoreFlags) -> None:
        ...

    def resolve_inputs(self, *, identity: ScoreIdentity, flags: ScoreFlags) -> InputAuthority:
        ...

    def score_cell(self, *, cell: ScoreCell, input_payload: Mapping[str, object],
                   flags: ScoreFlags) -> ModeEvidence:
        ...

    def reverify_after_forwards(self, *, identity: ScoreIdentity, flags: ScoreFlags) -> ImplementationClosure:
        ...

    def close(self) -> None:
        ...


class NoLiveBackend:
    """Public dry route sentinel: it performs no path, Torch, or device action."""

    def prepare(self, *, identity: ScoreIdentity, flags: ScoreFlags) -> None:
        raise ScoreError("physical posterior-carrier scorer is not authorized in this scaffold")

    def resolve_inputs(self, *, identity: ScoreIdentity, flags: ScoreFlags) -> InputAuthority:
        raise AssertionError("NoLiveBackend.prepare must fail before input resolution")

    def score_cell(self, *, cell: ScoreCell, input_payload: Mapping[str, object],
                   flags: ScoreFlags) -> ModeEvidence:
        raise AssertionError("NoLiveBackend cannot score")

    def reverify_after_forwards(self, *, identity: ScoreIdentity, flags: ScoreFlags) -> ImplementationClosure:
        raise AssertionError("NoLiveBackend cannot reverify")

    def close(self) -> None:
        return None


def _attempt_payload(identity: ScoreIdentity) -> dict[str, object]:
    return {
        "schema": "posterior_carrier_matched_score_attempt_v1",
        "cell": CELL,
        "phase": PHASE,
        "identity": validate_score_identity(identity),
        "matrix": [cell.payload() for cell in score_matrix()],
        "target_resolved": False,
        "within_or_external_resolved": False,
        "formal_resolved": False,
        "execution_status": "ATTEMPT_DURABLE__FULL_TERMINAL_AND_SWA_REQUIRED",
    }


def validate_attempt_payload(value: Mapping[str, object], *, identity: ScoreIdentity) -> dict[str, object]:
    expected = {
        "schema", "cell", "phase", "identity", "matrix", "target_resolved", "within_or_external_resolved",
        "formal_resolved", "execution_status",
    }
    if not isinstance(value, Mapping) or set(value) != expected or dict(value) != _attempt_payload(identity):
        raise ScoreError("attempt receipt schema/binding drift")
    return dict(value)


def _failure_payload(*, identity: ScoreIdentity, stage: str, flags: ScoreFlags,
                     attempt_sha256: str | None, input_authority_sha256: str | None) -> dict[str, object]:
    if not isinstance(stage, str) or not stage:
        raise ScoreError("failure stage drift")
    if attempt_sha256 is not None:
        _sha(attempt_sha256, "failure attempt SHA")
    if input_authority_sha256 is not None:
        _sha(input_authority_sha256, "failure input authority SHA")
    return {
        "schema": "posterior_carrier_matched_score_failure_v1",
        "cell": CELL,
        "phase": PHASE,
        "identity": validate_score_identity(identity),
        "stage": stage,
        "attempt_sha256": attempt_sha256,
        "input_authority_sha256": input_authority_sha256,
        "flags": flags.payload(),
        "target_resolved": False,
        "formal_opened": flags.formal_opened,
        "terminal_published": False,
        "traceback_sha256": _digest(traceback.format_exc().encode("utf-8")),
    }


def validate_failure_payload(value: Mapping[str, object], *, identity: ScoreIdentity) -> dict[str, object]:
    expected = {
        "schema", "cell", "phase", "identity", "stage", "attempt_sha256", "input_authority_sha256", "flags",
        "target_resolved", "formal_opened", "terminal_published", "traceback_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ScoreError("failure receipt schema drift")
    if (
        value["schema"] != "posterior_carrier_matched_score_failure_v1" or value["cell"] != CELL
        or value["phase"] != PHASE or value["identity"] != validate_score_identity(identity)
        or not isinstance(value["stage"], str) or not value["stage"]
        or value["target_resolved"] is not False or value["terminal_published"] is not False
        or value["formal_opened"] is not False or not isinstance(value["flags"], Mapping)
    ):
        raise ScoreError("failure receipt immutable-boundary drift")
    if value["attempt_sha256"] is not None:
        _sha(value["attempt_sha256"], "failure attempt SHA")
    if value["input_authority_sha256"] is not None:
        _sha(value["input_authority_sha256"], "failure input-authority SHA")
    _sha(value["traceback_sha256"], "failure traceback SHA")
    flags = value["flags"]
    if (
        flags.get("h1_opened") is not False or flags.get("formal_opened") is not False
        or flags.get("target_optimizer_steps") != 0 or flags.get("target_backward_calls") != 0
        or flags.get("target_update_calls") != 0
    ):
        raise ScoreError("failure receipt update/formal boundary drift")
    return dict(value)


def _terminal_payload(*, identity: ScoreIdentity, attempt_sha256: str, input_authority_sha256: str,
                      score_sha256: str, final_closure: Mapping[str, object]) -> dict[str, object]:
    closure = identity.closure.payload()
    if dict(final_closure) != closure:
        raise ScoreError("launch/final implementation closure drift")
    return {
        "schema": "posterior_carrier_matched_score_terminal_v1",
        "cell": CELL,
        "phase": PHASE,
        "status": "MATCHED_SCORE_COMPLETE__M4_HEADLINE_PREDECLARED__DIAGNOSTICS_NON_GOVERNING",
        "identity": validate_score_identity(identity),
        "attempt_sha256": _sha(attempt_sha256, "terminal attempt SHA"),
        "input_authority_sha256": _sha(input_authority_sha256, "terminal input-authority SHA"),
        "score_sha256": _sha(score_sha256, "terminal score SHA"),
        "launch_closure": closure,
        "final_closure": closure,
        "target_optimizer_steps": 0,
        "target_backward_calls": 0,
        "target_update_calls": 0,
        "formal_opened": False,
        "h1_not_scored": True,
        "score_terminal_transactional_group": True,
    }


def validate_terminal_payload(value: Mapping[str, object], *, identity: ScoreIdentity,
                              score_sha256: str | None = None) -> dict[str, object]:
    expected = {
        "schema", "cell", "phase", "status", "identity", "attempt_sha256", "input_authority_sha256",
        "score_sha256", "launch_closure", "final_closure", "target_optimizer_steps", "target_backward_calls",
        "target_update_calls", "formal_opened", "h1_not_scored", "score_terminal_transactional_group",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ScoreError("terminal receipt schema drift")
    closure = identity.closure.payload()
    if (
        value["schema"] != "posterior_carrier_matched_score_terminal_v1" or value["cell"] != CELL
        or value["phase"] != PHASE
        or value["status"] != "MATCHED_SCORE_COMPLETE__M4_HEADLINE_PREDECLARED__DIAGNOSTICS_NON_GOVERNING"
        or value["identity"] != validate_score_identity(identity) or value["launch_closure"] != closure
        or value["final_closure"] != closure or value["target_optimizer_steps"] != 0
        or value["target_backward_calls"] != 0 or value["target_update_calls"] != 0
        or value["formal_opened"] is not False or value["h1_not_scored"] is not True
        or value["score_terminal_transactional_group"] is not True
    ):
        raise ScoreError("terminal receipt binding drift")
    for key in ("attempt_sha256", "input_authority_sha256", "score_sha256"):
        _sha(value[key], f"terminal {key}")
    if score_sha256 is not None and value["score_sha256"] != _sha(score_sha256, "expected terminal score SHA"):
        raise ScoreError("terminal does not bind exact score SHA")
    return dict(value)


def _publish_failure(artifact: ArtifactRoot, *, identity: ScoreIdentity, stage: str, flags: ScoreFlags,
                     attempt_sha256: str | None, input_authority_sha256: str | None) -> None:
    if artifact.has_name("terminal.json"):
        raise ScoreError("failure terminal cannot coexist with successful terminal")
    if artifact.has_name("failure.json"):
        return
    payload = _failure_payload(
        identity=identity, stage=stage, flags=flags, attempt_sha256=attempt_sha256,
        input_authority_sha256=input_authority_sha256,
    )
    validate_failure_payload(payload, identity=identity)
    digest = artifact.publish_json("failure.json", payload)
    validate_failure_payload(artifact.reload_json("failure.json", digest), identity=identity)


def run_score_lifecycle(
    *,
    artifact: ArtifactRoot,
    identity: ScoreIdentity,
    execution_capability: ExecutionCapability,
    backend: ScoreBackend,
    final_authorization_reverify: Callable[[], None] | None = None,
) -> dict[str, object]:
    """Mockable, no-partial-result lifecycle for a later reviewed backend.

    This function intentionally never resolves a live input itself.  The
    attempt is immutable before the injected backend can do so, and score plus
    terminal have one transactional publication point.
    """
    _require_execution_capability(execution_capability, identity=identity)
    flags = ScoreFlags()
    attempt_sha256: str | None = None
    input_authority_sha256: str | None = None
    terminal_published = False
    try:
        attempt = _attempt_payload(identity)
        validate_attempt_payload(attempt, identity=identity)
        attempt_sha256 = artifact.publish_json("attempt.json", attempt)
        validate_attempt_payload(artifact.reload_json("attempt.json", attempt_sha256), identity=identity)

        backend.prepare(identity=identity, flags=flags)
        if flags.h1_opened or flags.formal_opened:
            raise ScoreError("backend crossed H1/formal boundary during prepare")
        authority = backend.resolve_inputs(identity=identity, flags=flags)
        input_payload = authority.payload(identity=identity)
        validate_input_authority_payload(input_payload, identity=identity)
        if flags.h1_opened or flags.formal_opened:
            raise ScoreError("backend crossed H1/formal boundary during input resolution")
        input_authority_sha256 = artifact.publish_json("input_authority.json", input_payload)
        validate_input_authority_payload(
            artifact.reload_json("input_authority.json", input_authority_sha256), identity=identity,
        )

        evidence: list[dict[str, object]] = []
        for cell in score_matrix():
            item = backend.score_cell(cell=cell, input_payload=input_payload, flags=flags)
            if item.cell != cell:
                raise ScoreError("backend returned score evidence for wrong matrix cell")
            payload = item.payload(input_payload=input_payload, identity=identity)
            validate_mode_payload(payload, input_payload=input_payload, identity=identity)
            flags.record_forward(cell)
            evidence.append(payload)
        _validate_flags(flags, expected_cells=score_matrix())
        final_closure = backend.reverify_after_forwards(identity=identity, flags=flags).payload()
        if final_closure != identity.closure.payload():
            raise ScoreError("post-forward implementation closure drift")
        if final_authorization_reverify is not None:
            # The physical route reloads the durable authority pair through a
            # held descriptor boundary after every forward, before either
            # scientific result leaf can become immutable.
            final_authorization_reverify()
        _validate_flags(flags, expected_cells=score_matrix())

        score_payload = build_score_payload(
            identity=identity, input_payload=input_payload, evidence=evidence, flags=flags,
        )
        validate_score_payload(score_payload, identity=identity, input_payload=input_payload)
        score_body = _json(score_payload)
        score_sha256 = _digest(score_body)
        terminal_payload = _terminal_payload(
            identity=identity, attempt_sha256=attempt_sha256,
            input_authority_sha256=input_authority_sha256, score_sha256=score_sha256,
            final_closure=final_closure,
        )
        validate_terminal_payload(terminal_payload, identity=identity, score_sha256=score_sha256)
        terminal_body = _json(terminal_payload)

        def validate_group(bodies: Mapping[str, bytes], digests: Mapping[str, str]) -> None:
            if (
                bodies.get("score.json") != score_body or bodies.get("terminal.json") != terminal_body
                or digests.get("score.json") != score_sha256
            ):
                raise ScoreError("score/terminal group body binding drift")
            reloaded_score = artifact.reload_json("score.json", score_sha256)
            reloaded_terminal = artifact.reload_json("terminal.json", _digest(terminal_body))
            validate_score_payload(reloaded_score, identity=identity, input_payload=input_payload)
            validate_terminal_payload(reloaded_terminal, identity=identity, score_sha256=score_sha256)

        hashes = artifact.publish_group(
            {"score.json": score_body, "terminal.json": terminal_body}, post_publish=validate_group,
        )
        terminal_published = True
        reloaded = artifact.reload_json("terminal.json", hashes["terminal.json"])
        validate_terminal_payload(reloaded, identity=identity, score_sha256=hashes["score.json"])
        if artifact.has_name("failure.json"):
            raise ScoreError("successful terminal cannot coexist with failure receipt")
        return reloaded
    except BaseException:
        if attempt_sha256 is not None and not terminal_published:
            try:
                _publish_failure(
                    artifact, identity=identity, stage="lifecycle", flags=flags,
                    attempt_sha256=attempt_sha256, input_authority_sha256=input_authority_sha256,
                )
            except BaseException:
                # A failure publication must never hide the original causal
                # error, and its own publisher rolls back incomplete leaves.
                pass
        raise
    finally:
        backend.close()


def dry_plan() -> dict[str, object]:
    """Static plan for the zero-argument CLI; imports no ML/runtime package."""
    return {
        "cell": CELL,
        "phase": PHASE,
        "status": "DRY_ONLY__NO_FULL_TERMINAL_OR_SWA_READ__NO_TORCH_NO_DATA_NO_CUDA_NO_WRITE",
        "workorder": {"relative_path": WORKORDER_RELATIVE, "sha256": WORKORDER_SHA256},
        "preconditions": {
            "full_terminal": FULL_TRAIN_TERMINAL_RELATIVE,
            "full_swa": FULL_TRAIN_SWA_RELATIVE,
            "full_source_authority": FULL_TRAIN_SOURCE_AUTHORITY_RELATIVE,
            "final_four_checkpoints": list(FULL_TRAIN_CHECKPOINT_RELATIVES),
            "import_mirror": FULL_IMPORT_MIRROR_ROOT_RELATIVE,
            "separate_root_review_required": True,
        },
        "matrix": [cell.payload() for cell in score_matrix()],
        "metric": dict(METRIC_CONTRACT),
        "primary": {
            "safety": "posterior_m30_minus_sealed_cell_d_ols_point",
            "headline": "posterior_m4_minus_sealed_cell_d_ols_point",
            "breadth": "external_positive_sessions_posterior_m4_minus_sealed_point_m4",
            "curve": "posterior_m30_to_m4_degradation_less_than_sealed_point_m30_to_m4",
            "no_budget_selection": True,
            "diagnostics_non_governing": [ZERO_MODE, WRONG_PAIR_MODE],
        },
        "boundaries": dict(EXECUTION_POLICY),
        "h1": "not_included_without_separate_authority",
        "canonical_roots": {
            "authority": AUTHORITY_ROOT_RELATIVE,
            "score": SCORE_ROOT_RELATIVE,
            "creation": "forbidden_until_separate_root_review",
        },
    }


def execute_authorized(
    root: Path | None = None,
    *,
    identity: ScoreIdentity | None = None,
    execute: bool = False,
    root_reviewed: bool = False,
    execution_capability: object | None = None,
    backend: ScoreBackend | None = None,
) -> dict[str, object]:
    """Root-only physical entry point; the public CLI cannot satisfy it.

    The two visible flags document operator intent, but deliberately have no
    authority by themselves.  Only an opaque capability issued after durable
    preflight/root-authorization reload may pass the first guard.  That guard
    is intentionally before a canonical result root, a mirror result body, a
    target pathname, Torch, or CUDA is touched.
    """
    if execution_capability is None:
        raise ScoreError(
            "dry until a separate physical-route review: "
            "root-reviewed in-process physical execution capability required before all access"
        )
    if execute is not True or root_reviewed is not True:
        raise ScoreError("physical score requires both explicit reviewed execution flags")
    if root is None:
        raise ScoreError("physical score root is required only after opaque capability verification")
    if not isinstance(identity, ScoreIdentity):
        raise ScoreError("root-only physical entry requires an explicit reviewed score identity")
    base = Path(root).absolute()
    # This descriptor-safe authority reload is intentionally before score-root
    # reservation and before the physical helper can open the imported mirror.
    preflight, preflight_sha, authorization, authorization_sha = verify_physical_authorization(base, identity=identity)
    _require_physical_execution_capability(
        execution_capability, identity=identity, official_preflight_sha256=preflight_sha,
    )
    # The caller cannot make the backend choose another model/result family:
    # this default is the explicit closure-bound physical implementation.
    if backend is None:
        physical = _physical_module()
        import sys

        contract = sys.modules.get(__name__)
        if contract is None:
            raise ScoreError("physical scorer contract module is not registered")
        backend = physical.PhysicalPosteriorMatchedBackend(root=base, preflight=preflight, contract=contract)
    artifact = reserve_score_artifact(base, execution_capability)

    def final_authorization_reverify() -> None:
        final_preflight, final_preflight_sha, final_authorization, final_authorization_sha = verify_physical_authorization(
            base, identity=identity,
        )
        if (
            final_preflight_sha != preflight_sha or final_authorization_sha != authorization_sha
            or final_preflight != preflight or final_authorization != authorization
        ):
            raise ScoreError("physical score authority pair changed during forwards")

    # The legacy lifecycle capability is minted only *inside* this verified
    # root-only route.  It exists to keep the dependency-injected no-data
    # tests useful; it is not an alternate public authorization channel.
    return run_score_lifecycle(
        artifact=artifact,
        identity=identity,
        execution_capability=_issue_execution_capability(identity),
        backend=backend,
        final_authorization_reverify=final_authorization_reverify,
    )
