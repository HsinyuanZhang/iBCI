"""No-data matched-score contract for the PMC-D source-training cell.

This module is deliberately an audit/scaffold boundary.  It validates the
eventual PMC-D full-training terminal, final-four checkpoint/SWA provenance,
the sealed Cell-D baseline authority, and future per-session score evidence,
but it does not discover result paths, open NWB files, load checkpoint tensors,
create result roots, initialize CUDA, or run a live scorer.

The only live metric seam is a lazy call to the reviewed
``tfpd_lane.matched_scorer`` implementation.  A future root-reviewed backend
must provide the already materialized fixed-bin-49 arrays and deterministic
ordinary OLS point-T4 inference evidence; posterior samples, posterior means,
posterior normalizers, covariance, and credibility are never accepted as an
inference contract here.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import plan


CELL = plan.CELL
PHASE = "POSTERIOR_MARGINALIZED_CELL_D_MATCHED_SCORE_V1"
SCHEMA = "posterior_marginalized_cell_d_matched_score_v1"

WORKORDER_RELATIVE = "tfpd_exploration/docs/WORKORDER_POSTERIOR_MARGINALIZED_CELL_D_MATCHED_SCORE_20260823.md"
WORKORDER_SHA256 = "e210134538b266e436f6eb26e19f27fec2abd93c868caeb832254a0c6e09399c"

FULL_TRAIN_ROOT_RELATIVE = plan.FULL_TRAIN_ROOT_RELATIVE
FULL_TRAIN_TERMINAL_RELATIVE = f"{FULL_TRAIN_ROOT_RELATIVE}/terminal.json"
FULL_TRAIN_SWA_RELATIVE = f"{FULL_TRAIN_ROOT_RELATIVE}/swa_final4.pt"
FULL_TRAIN_SWA_MANIFEST_RELATIVE = f"{FULL_TRAIN_ROOT_RELATIVE}/swa_manifest.json"
FULL_TRAIN_CHECKPOINT_RELATIVES = tuple(
    f"{FULL_TRAIN_ROOT_RELATIVE}/checkpoint_epoch_{epoch:02d}.pt"
    for epoch in plan.CHECKPOINT_EPOCHS
)

# These are provenance labels only.  They are never opened by this scaffold.
SEALED_CELL_D_TERMINAL_RELATIVE = (
    "tfpd_exploration/results/pop_robust_v1/cellD_2heads_dynamic_dropout/terminal_receipt.json"
)
SEALED_CELL_D_SWA_RELATIVE = (
    "tfpd_exploration/results/pop_robust_v1/cellD_2heads_dynamic_dropout/swa_final4.pt"
)
SEALED_CELL_D_BASELINE_RELATIVE = (
    "tfpd_exploration/results/sparsification_score_v1/sparsification_score_receipt.json"
)
SEALED_CELL_D_TERMINAL_SHA256 = "b3431db41efee937e83245e010ffaba50e50c517b6679d1d5285c889edbb7442"
SEALED_CELL_D_SWA_SHA256 = "626f65d80fd9f4305605132175c7ea43bc0c40d6ef6203ef1830b4b2e77f33bd"
SEALED_CELL_D_BASELINE_SHA256 = "583b899bb9e6b132a50b23552d9734b0ccb9b12ecc5c43c27c96ba49bd71980f"
SEALED_CELL_D_WITHIN_MEAN_R2 = 0.5696851710478464
SEALED_CELL_D_EXTERNAL_MEAN_R2 = 0.4179362749059995

AUTHORITY_ROOT_RELATIVE = "tfpd_exploration/results/posterior_marginalized_cell_d_score_authority_v1"
SCORE_ROOT_RELATIVE = "tfpd_exploration/results/posterior_marginalized_cell_d_score_v1"

WITHIN = "within"
EXTERNAL = "external"
SURFACES = (WITHIN, EXTERNAL)
EXPECTED_SESSION_COUNTS = {WITHIN: 6, EXTERNAL: 15}
# Keep the canonical pre-registered rotation order from the PMC plan.  Gate
# logic addresses M30/M4 by name, so changing display/order cannot change the
# scientific decision surface.
BUDGETS = tuple(plan.BUDGETS)
HEADLINE_BUDGET = 4
SAFETY_BUDGET = 30
PAIRED_BOOTSTRAP_SEED = 42
PAIRED_BOOTSTRAP_DRAWS = 10_000
M30_SAFETY_MIN = -0.02
M4_HEADLINE_MIN = 0.03
M4_EXTERNAL_POSITIVE_MIN = 9

SYSTEM_PMC = "pmc_d_final_four_swa"
SYSTEM_SEALED = "sealed_cell_d_swa"
SYSTEMS = (SYSTEM_PMC, SYSTEM_SEALED)
INFERENCE_SEMANTICS = "deterministic_ordinary_ols_point_t4_only"

METRIC_CONTRACT = {
    "estimator": "tfpd_lane.matched_scorer.session_r2",
    "torchmetrics": "torchmetrics.regression.R2Score(multioutput='variance_weighted')",
    "query": "fixed_bin_49_only_of_each_50_bin_window",
    "governing_bin": 49,
    "window_bins": 50,
    "behavior_coordinates": 2,
    "equal_weight_per_session": True,
    "paired_bootstrap": {
        "seed": PAIRED_BOOTSTRAP_SEED,
        "draws": PAIRED_BOOTSTRAP_DRAWS,
        "ci_level": 0.95,
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
    "posterior_sample_at_inference": False,
    "posterior_mean_at_inference": False,
    "posterior_normalizer_at_inference": False,
    "posterior_credibility_at_inference": False,
    "formal_opened": False,
}

# The training closure is composed rather than copied.  The three scorer
# leaves are intentionally separate from the source-training closure so a
# future scorer can never silently change the training identity.
SCORER_LOCAL_CLOSURE = (
    WORKORDER_RELATIVE,
    "tfpd_exploration/src/posterior_marginalized_cell_d_v1/matched_score.py",
    "tfpd_exploration/src/posterior_marginalized_cell_d_v1/matched_score_physical.py",
    "tfpd_exploration/scripts/run_posterior_marginalized_cell_d_matched_score.py",
    "tfpd_exploration/tests/test_posterior_marginalized_cell_d_matched_score.py",
    "tfpd_exploration/src/cell_d_equal_session_score_v1.py",
    "tfpd_exploration/src/posterior_carrier_v1/matched_score_physical.py",
    "tfpd_exploration/src/tfpd_lane/arm_common.py",
    "tfpd_exploration/src/tfpd_lane/pop_robust.py",
    "tfpd_exploration/src/tfpd_lane/matched_scorer.py",
    "sua_exploration/mc_maze/multisession_datamodule.py",
    "sua_exploration/mc_maze/unit_side_features.py",
)


class ScoreError(RuntimeError):
    """Raised whenever score provenance or evaluation semantics drift."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ScoreError(message)


def _json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha(value: object, label: str = "SHA-256") -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ScoreError(f"{label} must be an exact lowercase SHA-256")
    return value


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ScoreError(f"{label} must be finite")
    return float(value)


def _safe_component(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or value in {".", ".."} or "/" in value or "\\" in value:
        raise ScoreError(f"{label} must be one safe path component")
    return value


def _ordered_roster(value: Sequence[str], *, surface: str) -> tuple[str, ...]:
    expected = EXPECTED_SESSION_COUNTS.get(surface)
    if expected is None or not isinstance(value, (tuple, list)):
        raise ScoreError(f"{surface} roster surface drift")
    roster = tuple(value)
    if len(roster) != expected or len(set(roster)) != expected or any(not isinstance(item, str) or not item for item in roster):
        raise ScoreError(f"{surface} roster must contain exactly {expected} unique sessions")
    return roster


def _mapping_copy(value: Mapping[str, object], label: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ScoreError(f"{label} must be a mapping")
    return dict(value)


def _read_regular_no_follow(path: Path) -> tuple[bytes, str]:
    """Read one explicit source file with a held inode-safe descriptor."""
    try:
        before = os.lstat(path)
    except OSError as error:
        raise ScoreError(f"scorer closure path is inaccessible: {path}") from error
    if not stat.S_ISREG(before.st_mode) or stat.S_ISLNK(before.st_mode):
        raise ScoreError(f"scorer closure path is missing/aliased: {path}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ScoreError(f"scorer closure path is inaccessible: {path}") from error
    try:
        opened = os.fstat(descriptor)
        identity = (before.st_dev, before.st_ino, before.st_size)
        if (opened.st_dev, opened.st_ino, opened.st_size) != identity or stat.S_ISLNK(opened.st_mode):
            raise ScoreError(f"scorer closure path identity drift: {path}")
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 1 << 20)
            if not block:
                break
            chunks.append(block)
        body = b"".join(chunks)
    finally:
        os.close(descriptor)
    after = os.lstat(path)
    if (after.st_dev, after.st_ino, after.st_size) != identity:
        raise ScoreError(f"scorer closure path changed during read: {path}")
    return body, _digest(body)


def scorer_closure_paths() -> tuple[str, ...]:
    """Compose the frozen PMC training closure with explicit scorer leaves."""
    return tuple(dict.fromkeys((*plan.IMPLEMENTATION_CLOSURE, *SCORER_LOCAL_CLOSURE)))


@dataclass(frozen=True)
class ImplementationClosure:
    sha256_by_path: Mapping[str, str]

    def payload(self) -> dict[str, object]:
        paths = scorer_closure_paths()
        if not isinstance(self.sha256_by_path, Mapping) or set(self.sha256_by_path) != set(paths):
            raise ScoreError("matched-score implementation closure topology drift")
        hashes = {path: _sha(self.sha256_by_path[path], f"closure SHA {path}") for path in paths}
        if hashes[WORKORDER_RELATIVE] != WORKORDER_SHA256:
            raise ScoreError("matched-score workorder SHA drift")
        body = {
            "paths": list(paths),
            "sha256_by_path": hashes,
            "training_closure_predecessor_sha256": plan.PHASE1_ACCEPTED_CLOSURE_SHA256,
        }
        return {**body, "closure_sha256": _digest(_json(body))}


def implementation_closure(root: Path) -> ImplementationClosure:
    """Hash only explicit source/doc/test leaves; never results or data."""
    base = Path(root).absolute()
    hashes: dict[str, str] = {}
    for relative in scorer_closure_paths():
        _, digest = _read_regular_no_follow(base / relative)
        hashes[relative] = digest
    return ImplementationClosure(hashes)


@dataclass(frozen=True)
class FinalFourSWAProvenance:
    """Validated, source-only PMC full terminal plus ordered final-four SWA."""

    terminal_sha256: str
    swa_sha256: str
    swa_manifest_sha256: str
    source_authority_sha256: str
    checkpoint_sha256s: Mapping[str, str]
    training_closure_sha256: str
    terminal_status: str = "TERMINAL"

    def payload(self) -> dict[str, object]:
        _sha(self.terminal_sha256, "PMC terminal SHA")
        _sha(self.swa_sha256, "PMC SWA SHA")
        _sha(self.swa_manifest_sha256, "PMC SWA manifest SHA")
        _sha(self.source_authority_sha256, "PMC source-authority SHA")
        _sha(self.training_closure_sha256, "PMC training closure SHA")
        if self.terminal_status != "TERMINAL":
            raise ScoreError("PMC full terminal status is not TERMINAL")
        if not isinstance(self.checkpoint_sha256s, Mapping) or set(self.checkpoint_sha256s) != {
            str(epoch) for epoch in plan.CHECKPOINT_EPOCHS
        }:
            raise ScoreError("PMC final-four checkpoint topology drift")
        checkpoints = {
            key: _sha(self.checkpoint_sha256s[key], f"PMC checkpoint {key} SHA")
            for key in sorted(self.checkpoint_sha256s)
        }
        return {
            "schema": "posterior_marginalized_cell_d_final_four_swa_provenance_v1",
            "cell": CELL,
            "terminal_relative": FULL_TRAIN_TERMINAL_RELATIVE,
            "terminal_sha256": self.terminal_sha256,
            "swa_relative": FULL_TRAIN_SWA_RELATIVE,
            "swa_sha256": self.swa_sha256,
            "swa_manifest_relative": FULL_TRAIN_SWA_MANIFEST_RELATIVE,
            "swa_manifest_sha256": self.swa_manifest_sha256,
            "checkpoint_sha256s": checkpoints,
            "source_authority_relative": f"{FULL_TRAIN_ROOT_RELATIVE}/source_authority.json",
            "source_authority_sha256": self.source_authority_sha256,
            "training_closure_sha256": self.training_closure_sha256,
            "checkpoint_epochs": list(plan.CHECKPOINT_EPOCHS),
            "terminal_status": self.terminal_status,
            "optimizer_steps_completed": plan.TOTAL_STEPS,
            "target_optimizer_steps": 0,
            "target_backward_calls": 0,
            "target_update_calls": 0,
            "inference": INFERENCE_SEMANTICS,
            "posterior_sample_at_inference": False,
            "posterior_mean_at_inference": False,
            "posterior_normalizer_at_inference": False,
            "posterior_credibility_at_inference": False,
        }


def _expected_full_spec() -> dict[str, object]:
    return {
        "schema": "posterior_marginalized_cell_d_execution_spec_v1",
        "kind": "full_train",
        "epochs": plan.EPOCHS,
        "steps_per_epoch": plan.STEPS_PER_EPOCH,
        "total_steps": plan.TOTAL_STEPS,
        "checkpoint_epochs": list(plan.CHECKPOINT_EPOCHS),
        "batch_size": plan.BATCH_SIZE,
        "seed": plan.SEED,
        "source_only": True,
    }


def _validate_terminal_identity(value: object) -> dict[str, object]:
    """Require the exact runner identity before accepting a PMC terminal.

    The scorer does not mint or discover this identity.  It nevertheless
    rechecks the complete typed boundary so a partial mapping cannot relabel a
    posterior-specific normalizer, a target-capable run, or an unbound source
    closure as the frozen PMC-D experiment.
    """
    identity = _mapping_copy(value, "PMC terminal identity")
    required = {
        "schema", "cell", "phase", "closure", "predecessor_phase1_closure_sha256",
        "approved_phase_b_v2_authority_sha256", "approved_phase_b_v2_closure_sha256",
        "pmc_source_binding_sha256", "sealed_cell_d", "sole_intervention", "inference", "gpu", "boundaries",
    }
    _require(set(identity) == required, "PMC terminal identity schema/topology drift")
    _require(identity.get("schema") == "posterior_marginalized_cell_d_identity_v1"
             and identity.get("cell") == CELL
             and identity.get("phase") == "SOURCE_TRAINING_POSTERIOR_MARGINALIZATION"
             and identity.get("predecessor_phase1_closure_sha256") == plan.PHASE1_ACCEPTED_CLOSURE_SHA256,
             "PMC terminal identity route/predecessor drift")
    for key in (
        "approved_phase_b_v2_authority_sha256", "approved_phase_b_v2_closure_sha256",
        "pmc_source_binding_sha256",
    ):
        _sha(identity.get(key), f"PMC terminal identity {key}")
    try:
        closure = plan.validate_implementation_closure(identity.get("closure"))
        gpu = plan.validate_compatible_device_profile(identity.get("gpu"))
    except Exception as error:  # pragma: no cover - exact plan error is route-specific
        raise ScoreError("PMC terminal identity closure/device authority drift") from error
    _require(identity.get("closure") == closure, "PMC terminal identity closure canonicalization drift")
    _require(identity.get("gpu") == gpu, "PMC terminal identity GPU profile drift")
    _require(identity.get("sealed_cell_d") == {
        "canonical_initial_state_artifact_sha256": plan.CANONICAL_INITIAL_STATE_SHA256,
        "canonical_initial_state_state_sha256": plan.CANONICAL_INITIAL_STATE_STATE_SHA256,
        "ordinary_ols_t4_normalizer_sha256": plan.SEALED_OLS_T4_NORMALIZER_SHA256,
        "ordinary_ols_t4_mean_float32": list(plan.SEALED_OLS_T4_MEAN_FLOAT32),
        "ordinary_ols_t4_std_float32": list(plan.SEALED_OLS_T4_STD_FLOAT32),
        "initialized_trainable_parameters": plan.SEALED_CELL_D_INITIALIZED_PARAMETERS,
        "uninitialized_lazy_keys": list(plan.SEALED_CELL_D_LAZY_KEYS),
    }, "PMC terminal held Cell-D identity drift")
    _require(identity.get("sole_intervention") == "source_training_side_cached_posterior_sample_only"
             and identity.get("inference") == INFERENCE_SEMANTICS,
             "PMC terminal intervention/inference identity drift")
    _require(identity.get("boundaries") == {
        "source_only": True,
        "within_opened": False,
        "external_opened": False,
        "formal_opened": False,
        "target_opened": False,
        "target_optimizer_steps": 0,
        "target_backward_calls": 0,
        "target_update_calls": 0,
        "posterior_normalizer_used": False,
        "posterior_credibility_or_attention_bias_used": False,
    }, "PMC terminal target/posterior identity boundary drift")
    return identity


def _validate_swa_proof(value: object, checkpoints: Mapping[str, object]) -> dict[str, object]:
    """Validate the physical Cell-D SWA proof carried by the PMC terminal."""
    proof = _mapping_copy(value, "PMC SWA proof")
    required = {
        "window_epochs", "component_state_sha256", "fp64_arithmetic", "fresh_strict_load",
        "eval_mode", "eval_no_mask", "repeat_bitwise_equal", "state_unchanged",
        "prediction_shape", "prediction_sha256", "state_sha256_before_eval",
        "state_sha256_after_eval", "state_sha256", "uninitialized_lazy_keys",
    }
    _require(set(proof) == required, "PMC SWA proof schema drift")
    _require(proof.get("window_epochs") == list(plan.CHECKPOINT_EPOCHS)
             and all(proof.get(flag) is True for flag in (
                 "fp64_arithmetic", "fresh_strict_load", "eval_mode", "eval_no_mask",
                 "repeat_bitwise_equal", "state_unchanged",
             )), "PMC SWA final-four/reload proof drift")
    components = proof.get("component_state_sha256")
    _require(isinstance(components, Mapping)
             and set(components) == {str(epoch) for epoch in plan.CHECKPOINT_EPOCHS}
             and all(_sha(item, f"PMC SWA component {key} SHA") for key, item in components.items()),
             "PMC SWA component digest topology drift")
    _require(proof.get("prediction_shape") == [4, 50, 2]
             and proof.get("uninitialized_lazy_keys") == list(plan.SEALED_CELL_D_LAZY_KEYS),
             "PMC SWA strict-reload prediction/lazy proof drift")
    for key in ("prediction_sha256", "state_sha256", "state_sha256_before_eval", "state_sha256_after_eval"):
        _sha(proof.get(key), f"PMC SWA proof {key}")
    _require(proof["state_sha256"] == proof["state_sha256_before_eval"]
             == proof["state_sha256_after_eval"],
             "PMC SWA state mutated during strict-reload proof")
    return proof


def validate_pmc_terminal_and_swa(
    terminal: Mapping[str, object],
    swa_manifest: Mapping[str, object],
    *,
    terminal_sha256: str,
    swa_sha256: str,
    swa_manifest_sha256: str,
) -> FinalFourSWAProvenance:
    """Validate future terminal/manifest mappings without opening any path."""
    terminal = _mapping_copy(terminal, "PMC terminal")
    manifest = _mapping_copy(swa_manifest, "PMC SWA manifest")
    _require(terminal.get("schema") == "posterior_marginalized_cell_d_terminal_v1"
             and terminal.get("cell") == CELL and terminal.get("status") == "TERMINAL",
             "PMC terminal schema/status/cell drift")
    _require(terminal.get("spec") == _expected_full_spec(), "PMC terminal full-training spec drift")
    _require(terminal.get("source_only") is True and terminal.get("launch_final_closure_equal") is True,
             "PMC terminal source-only/closure boundary drift")
    _require(terminal.get("optimizer_steps_completed") == plan.TOTAL_STEPS
             and terminal.get("epoch_count") == plan.EPOCHS,
             "PMC terminal optimizer/epoch count drift")
    for key in ("target_optimizer_steps", "target_backward_calls", "target_update_calls"):
        _require(terminal.get(key) == 0, f"PMC terminal {key} drift")
    identity = _validate_terminal_identity(terminal.get("identity"))
    identity_closure = identity["closure"]
    training_closure_sha256 = identity_closure.get("closure_sha256") if isinstance(identity_closure, Mapping) else None
    _sha(training_closure_sha256, "PMC terminal training closure SHA")

    artifacts = terminal.get("artifacts")
    _require(isinstance(artifacts, Mapping), "PMC terminal artifact graph missing")
    _require(set(artifacts) == {
        "epoch_sha256", "smoke_sha256", "checkpoint_sha256", "swa_sha256", "swa_manifest_sha256", "swa_proof",
    }, "PMC terminal artifact graph topology drift")
    epoch_sha256 = artifacts.get("epoch_sha256")
    _require(isinstance(epoch_sha256, Mapping)
             and set(epoch_sha256) == {str(epoch) for epoch in range(plan.EPOCHS)}
             and all(_sha(item, f"PMC epoch {key} SHA") for key, item in epoch_sha256.items()),
             "PMC terminal epoch artifact map drift")
    checkpoints = artifacts.get("checkpoint_sha256")
    _require(isinstance(checkpoints, Mapping), "PMC terminal checkpoint map missing")
    expected_keys = {str(epoch) for epoch in plan.CHECKPOINT_EPOCHS}
    _require(set(checkpoints) == expected_keys, "PMC terminal final-four checkpoint map drift")
    _require(all(_sha(checkpoints[key], f"PMC checkpoint {key} SHA") for key in expected_keys),
             "PMC terminal checkpoint digest drift")
    _require(artifacts.get("swa_sha256") == swa_sha256
             and artifacts.get("swa_manifest_sha256") == swa_manifest_sha256
             and artifacts.get("smoke_sha256") is None,
             "PMC terminal SWA/manifest binding drift")
    swa_proof = _validate_swa_proof(artifacts.get("swa_proof"), checkpoints)

    _require(set(manifest) == {"schema", "cell", "swa_sha256", "proof", "binding", "checkpoint_sha256"},
             "PMC SWA manifest topology drift")
    _require(manifest.get("schema") == "posterior_marginalized_cell_d_swa_manifest_v1"
             and manifest.get("cell") == CELL
             and manifest.get("swa_sha256") == swa_sha256,
             "PMC SWA manifest schema/cell/SHA drift")
    _require(manifest.get("checkpoint_sha256") == dict(checkpoints)
             and manifest.get("proof") == swa_proof,
             "PMC SWA manifest checkpoint/proof binding drift")
    binding = manifest.get("binding")
    _require(isinstance(binding, Mapping), "PMC SWA manifest binding missing")
    _require(binding.get("run_spec") == _expected_full_spec()
             and binding.get("identity") == dict(identity),
             "PMC SWA manifest training binding drift")
    _sha(terminal_sha256, "PMC terminal SHA")
    _sha(swa_sha256, "PMC SWA body SHA")
    _sha(swa_manifest_sha256, "PMC SWA manifest SHA")
    source_authority_sha256 = terminal.get("source_authority_sha256")
    _sha(source_authority_sha256, "PMC terminal source-authority SHA")
    return FinalFourSWAProvenance(
        terminal_sha256=terminal_sha256,
        swa_sha256=swa_sha256,
        swa_manifest_sha256=swa_manifest_sha256,
        source_authority_sha256=source_authority_sha256,
        checkpoint_sha256s=dict(checkpoints),
        training_closure_sha256=training_closure_sha256,
    )


@dataclass(frozen=True)
class BaselineSessionScore:
    session: str
    n_windows: int
    r2: float

    def payload(self) -> dict[str, object]:
        _safe_component(self.session, "baseline session")
        if type(self.n_windows) is not int or self.n_windows <= 0:
            raise ScoreError("baseline window count drift")
        return {"session": self.session, "n_windows": self.n_windows, "r2": _finite(self.r2, "baseline R2")}


@dataclass(frozen=True)
class SealedCellDEvidence:
    """Sealed Cell-D terminal/SWA plus the governing last-bin baseline table."""

    baseline_receipt_sha256: str
    terminal_sha256: str = SEALED_CELL_D_TERMINAL_SHA256
    swa_sha256: str = SEALED_CELL_D_SWA_SHA256
    within: tuple[BaselineSessionScore, ...] = ()
    external: tuple[BaselineSessionScore, ...] = ()

    def rows(self, surface: str) -> tuple[BaselineSessionScore, ...]:
        if surface == WITHIN:
            return self.within
        if surface == EXTERNAL:
            return self.external
        raise ScoreError("unknown sealed Cell-D surface")

    def payload(self) -> dict[str, object]:
        _sha(self.baseline_receipt_sha256, "sealed Cell-D baseline receipt SHA")
        _sha(self.terminal_sha256, "sealed Cell-D terminal SHA")
        _sha(self.swa_sha256, "sealed Cell-D SWA SHA")
        _require(self.baseline_receipt_sha256 == SEALED_CELL_D_BASELINE_SHA256
                 and self.terminal_sha256 == SEALED_CELL_D_TERMINAL_SHA256
                 and self.swa_sha256 == SEALED_CELL_D_SWA_SHA256,
                 "sealed Cell-D artifact authority SHA drift")
        rows_payload: dict[str, list[dict[str, object]]] = {}
        for surface in SURFACES:
            rows = self.rows(surface)
            expected = EXPECTED_SESSION_COUNTS[surface]
            if len(rows) != expected or len({row.session for row in rows}) != expected:
                raise ScoreError(f"sealed Cell-D {surface} baseline count drift")
            if tuple(row.session for row in rows) != tuple(sorted(row.session for row in rows)):
                raise ScoreError(f"sealed Cell-D {surface} baseline order drift")
            expected_mean = SEALED_CELL_D_WITHIN_MEAN_R2 if surface == WITHIN else SEALED_CELL_D_EXTERNAL_MEAN_R2
            observed_mean = sum(row.r2 for row in rows) / len(rows)
            if observed_mean != expected_mean:
                raise ScoreError(f"sealed Cell-D {surface} governing mean authority drift")
            rows_payload[surface] = [row.payload() for row in rows]
        return {
            "schema": "sealed_cell_d_ordinary_ols_point_last_bin_authority_v1",
            "terminal_relative": SEALED_CELL_D_TERMINAL_RELATIVE,
            "terminal_sha256": self.terminal_sha256,
            "swa_relative": SEALED_CELL_D_SWA_RELATIVE,
            "swa_sha256": self.swa_sha256,
            "baseline_relative": SEALED_CELL_D_BASELINE_RELATIVE,
            "baseline_receipt_sha256": self.baseline_receipt_sha256,
            "metric": dict(METRIC_CONTRACT),
            "inference": INFERENCE_SEMANTICS,
            "posterior_sample_at_inference": False,
            "posterior_mean_at_inference": False,
            "posterior_normalizer_at_inference": False,
            "posterior_credibility_at_inference": False,
            "rows": rows_payload,
        }


def sealed_cell_d_from_payload(value: Mapping[str, object]) -> SealedCellDEvidence:
    """Parse only an already materialized governing Cell-D table mapping."""
    payload = _mapping_copy(value, "sealed Cell-D authority")
    _require(payload.get("schema") == "sealed_cell_d_ordinary_ols_point_last_bin_authority_v1",
             "sealed Cell-D authority schema drift")
    _require(payload.get("inference") == INFERENCE_SEMANTICS
             and payload.get("posterior_sample_at_inference") is False
             and payload.get("posterior_mean_at_inference") is False
             and payload.get("posterior_normalizer_at_inference") is False
             and payload.get("posterior_credibility_at_inference") is False,
             "sealed Cell-D posterior inference boundary drift")
    rows = payload.get("rows")
    _require(isinstance(rows, Mapping) and set(rows) == set(SURFACES), "sealed Cell-D row surface topology drift")

    parsed: dict[str, tuple[BaselineSessionScore, ...]] = {}
    for surface in SURFACES:
        raw = rows[surface]
        _require(isinstance(raw, list), f"sealed Cell-D {surface} rows type drift")
        values: list[BaselineSessionScore] = []
        for item in raw:
            _require(isinstance(item, Mapping), f"sealed Cell-D {surface} row type drift")
            values.append(BaselineSessionScore(
                session=item.get("session"), n_windows=item.get("n_windows"), r2=item.get("r2"),
            ))
        parsed[surface] = tuple(values)
    result = SealedCellDEvidence(
        baseline_receipt_sha256=payload.get("baseline_receipt_sha256"),
        terminal_sha256=payload.get("terminal_sha256"),
        swa_sha256=payload.get("swa_sha256"),
        within=parsed[WITHIN], external=parsed[EXTERNAL],
    )
    _require(result.payload() == payload, "sealed Cell-D authority canonical payload drift")
    return result


@dataclass(frozen=True)
class ScoreIdentity:
    pmc_training: FinalFourSWAProvenance
    sealed_cell_d: SealedCellDEvidence
    closure: ImplementationClosure
    within_roster: tuple[str, ...]
    external_roster: tuple[str, ...]

    def payload(self) -> dict[str, object]:
        within = _ordered_roster(self.within_roster, surface=WITHIN)
        external = _ordered_roster(self.external_roster, surface=EXTERNAL)
        _require(self.sealed_cell_d.baseline_receipt_sha256 == SEALED_CELL_D_BASELINE_SHA256,
                 "sealed Cell-D baseline receipt authority drift")
        _require(self.sealed_cell_d.terminal_sha256 == SEALED_CELL_D_TERMINAL_SHA256
                 and self.sealed_cell_d.swa_sha256 == SEALED_CELL_D_SWA_SHA256,
                 "sealed Cell-D terminal/SWA authority drift")
        _require(tuple(item.session for item in self.sealed_cell_d.within) == within
                 and tuple(item.session for item in self.sealed_cell_d.external) == external,
                 "sealed Cell-D baseline roster differs from score identity")
        closure = self.closure.payload()
        _require(self.pmc_training.payload()["inference"] == INFERENCE_SEMANTICS,
                 "PMC training inference semantics drift")
        return {
            "schema": "posterior_marginalized_cell_d_matched_score_identity_v1",
            "cell": CELL,
            "phase": PHASE,
            "training": self.pmc_training.payload(),
            "sealed_cell_d": self.sealed_cell_d.payload(),
            "rosters": {WITHIN: list(within), EXTERNAL: list(external)},
            "metric": dict(METRIC_CONTRACT),
            "execution_policy": dict(EXECUTION_POLICY),
            "closure": closure,
            "target_free": True,
        }


@dataclass(frozen=True)
class InputRecord:
    surface: str
    session: str
    n_windows: int
    neural_sha256: str
    calibration_m30_sha256: str
    target_sha256: str
    valid_mask_sha256: str
    ordinary_ols_point_carrier_sha256s: Mapping[str, str]

    def payload(self) -> dict[str, object]:
        if self.surface not in SURFACES:
            raise ScoreError("input surface drift")
        _safe_component(self.session, "input session")
        if type(self.n_windows) is not int or self.n_windows <= 0:
            raise ScoreError("input window count drift")
        _sha(self.neural_sha256, "input neural SHA")
        _sha(self.calibration_m30_sha256, "input M30 calibration SHA")
        _sha(self.target_sha256, "input target SHA")
        _sha(self.valid_mask_sha256, "input valid-mask SHA")
        if not isinstance(self.ordinary_ols_point_carrier_sha256s, Mapping) or set(self.ordinary_ols_point_carrier_sha256s) != {
            str(budget) for budget in BUDGETS
        }:
            raise ScoreError("input ordinary OLS point-carrier budget topology drift")
        carriers = {
            key: _sha(self.ordinary_ols_point_carrier_sha256s[key], f"input ordinary OLS M{key} carrier SHA")
            for key in sorted(self.ordinary_ols_point_carrier_sha256s)
        }
        return {
            "surface": self.surface,
            "session": self.session,
            "n_windows": self.n_windows,
            "last_bin_query": "fixed_bin_49_of_each_50_bin_window",
            "neural_sha256": self.neural_sha256,
            "calibration_m30_sha256": self.calibration_m30_sha256,
            "target_sha256": self.target_sha256,
            "valid_mask_sha256": self.valid_mask_sha256,
            "ordinary_ols_point_carrier_sha256s": carriers,
            "same_materialized_input_for_pmc_and_sealed": True,
        }


@dataclass(frozen=True)
class InputAuthority:
    records: tuple[InputRecord, ...]
    shared_materialized_input_pass: bool = True
    cache_read_or_write: bool = False

    def payload(self, *, identity: ScoreIdentity) -> dict[str, object]:
        _require(self.shared_materialized_input_pass is True and self.cache_read_or_write is False,
                 "matched score input pass/cache boundary drift")
        if not isinstance(self.records, tuple) or len(self.records) != 21:
            raise ScoreError("matched score requires exact within-6 plus external-15 inputs")
        rosters = {WITHIN: identity.within_roster, EXTERNAL: identity.external_roster}
        expected_order = [(surface, session) for surface in SURFACES for session in rosters[surface]]
        rows = [record.payload() for record in self.records]
        if [(row["surface"], row["session"]) for row in rows] != expected_order:
            raise ScoreError("matched input authority roster/order drift")
        return {
            "schema": "posterior_marginalized_cell_d_matched_score_input_authority_v1",
            "identity_cell": CELL,
            "records": rows,
            "shared_materialized_input_pass": True,
            "cache_read_or_write": False,
            "within_opened": True,
            "external_opened": True,
            "target_opened": False,
            "formal_opened": False,
            "target_optimizer_steps": 0,
            "target_backward_calls": 0,
            "target_update_calls": 0,
            "normalizer_refit": False,
        }


@dataclass(frozen=True)
class ScoreCell:
    surface: str
    budget: int
    system: str

    def payload(self) -> dict[str, object]:
        if self.surface not in SURFACES or self.budget not in BUDGETS or self.system not in SYSTEMS:
            raise ScoreError("matched score cell topology drift")
        return {
            "surface": self.surface,
            "budget": self.budget,
            "system": self.system,
            "carrier": INFERENCE_SEMANTICS,
        }


def score_matrix() -> tuple[ScoreCell, ...]:
    return tuple(
        ScoreCell(surface, budget, system)
        for surface in SURFACES
        for budget in BUDGETS
        for system in SYSTEMS
    )


@dataclass(frozen=True)
class SessionScore:
    session: str
    n_windows: int
    r2: float
    prediction_sha256: str
    input_record_sha256: str

    def payload(self) -> dict[str, object]:
        _safe_component(self.session, "session score session")
        if type(self.n_windows) is not int or self.n_windows <= 0:
            raise ScoreError("session score window count drift")
        return {
            "session": self.session,
            "n_windows": self.n_windows,
            "r2": _finite(self.r2, "session variance-weighted R2"),
            "prediction_sha256": _sha(self.prediction_sha256, "session prediction SHA"),
            "input_record_sha256": _sha(self.input_record_sha256, "session input-record SHA"),
        }


@dataclass(frozen=True)
class ModeEvidence:
    cell: ScoreCell
    sessions: tuple[SessionScore, ...]
    input_authority_sha256: str
    model_swa_sha256: str
    model_state_before_sha256: str
    model_state_after_sha256: str
    eval_mode: bool = True
    dropout_disabled: bool = True
    gradients_none: bool = True
    finite_outputs: bool = True
    repeated_fixed_batch_bitwise_equal: bool = True
    b3s_m30_recomputed: bool = True
    ordinary_ols_point_t4_used: bool = True
    posterior_sample_used: bool = False
    posterior_mean_used: bool = False
    posterior_normalizer_used: bool = False
    posterior_credibility_used: bool = False
    target_optimizer_steps: int = 0
    target_backward_calls: int = 0
    target_update_calls: int = 0

    def payload(self, *, identity: ScoreIdentity, input_payload: Mapping[str, object]) -> dict[str, object]:
        cell = self.cell.payload()
        expected_roster = identity.within_roster if self.cell.surface == WITHIN else identity.external_roster
        if tuple(item.session for item in self.sessions) != tuple(expected_roster):
            raise ScoreError("mode evidence roster/order drift")
        records = {
            (row["surface"], row["session"]): row
            for row in input_payload.get("records", ())
            if isinstance(row, Mapping)
        }
        rows = [item.payload() for item in self.sessions]
        for row in rows:
            source = records.get((self.cell.surface, row["session"]))
            if source is None or row["n_windows"] != source["n_windows"]:
                raise ScoreError("mode evidence input-record topology drift")
            if row["input_record_sha256"] != _digest(_json(source)):
                raise ScoreError("mode evidence input-record binding drift")
        expected_swa = identity.pmc_training.swa_sha256 if self.cell.system == SYSTEM_PMC else identity.sealed_cell_d.swa_sha256
        if any(flag is not False for flag in (
            self.posterior_sample_used,
            self.posterior_mean_used,
            self.posterior_normalizer_used,
            self.posterior_credibility_used,
        )):
            raise ScoreError("posterior inference leak in matched score evidence")
        if (
            self.input_authority_sha256 != _digest(_json(dict(input_payload)))
            or self.model_swa_sha256 != expected_swa
            or self.model_state_before_sha256 != self.model_state_after_sha256
            or self.eval_mode is not True or self.dropout_disabled is not True
            or self.gradients_none is not True or self.finite_outputs is not True
            or self.repeated_fixed_batch_bitwise_equal is not True
            or self.b3s_m30_recomputed is not True or self.ordinary_ols_point_t4_used is not True
            or self.target_optimizer_steps != 0 or self.target_backward_calls != 0 or self.target_update_calls != 0
        ):
            raise ScoreError("mode evidence inference/state/update boundary drift")
        return {
            "cell": cell,
            "model_swa_sha256": _sha(self.model_swa_sha256, "mode SWA SHA"),
            "sessions": rows,
            "input_authority_sha256": _sha(self.input_authority_sha256, "mode input authority SHA"),
            "model_state_before_sha256": _sha(self.model_state_before_sha256, "mode state-before SHA"),
            "model_state_after_sha256": _sha(self.model_state_after_sha256, "mode state-after SHA"),
            "eval_mode": True,
            "dropout_disabled": True,
            "gradients_none": True,
            "finite_outputs": True,
            "repeated_fixed_batch_bitwise_equal": True,
            "b3s_m30_recomputed": True,
            "ordinary_ols_point_t4_used": True,
            "posterior_sample_used": False,
            "posterior_mean_used": False,
            "posterior_normalizer_used": False,
            "posterior_credibility_used": False,
            "target_optimizer_steps": 0,
            "target_backward_calls": 0,
            "target_update_calls": 0,
        }


@dataclass(frozen=True)
class PairedSessionDelta:
    surface: str
    budget: int
    session: str
    n_windows: int
    pmc_r2: float
    sealed_r2: float
    input_record_sha256: str

    @property
    def delta_r2(self) -> float:
        return float(self.pmc_r2) - float(self.sealed_r2)

    @property
    def sign(self) -> str:
        value = self.delta_r2
        return "+" if value > 0.0 else ("-" if value < 0.0 else "0")

    def payload(self) -> dict[str, object]:
        if self.surface not in SURFACES or self.budget not in BUDGETS:
            raise ScoreError("paired delta cell drift")
        _safe_component(self.session, "paired delta session")
        if type(self.n_windows) is not int or self.n_windows <= 0:
            raise ScoreError("paired delta window count drift")
        pmc = _finite(self.pmc_r2, "paired PMC R2")
        sealed = _finite(self.sealed_r2, "paired sealed R2")
        _sha(self.input_record_sha256, "paired input-record SHA")
        return {
            "surface": self.surface,
            "budget": self.budget,
            "session": self.session,
            "n_windows": self.n_windows,
            "pmc_r2": pmc,
            "sealed_r2": sealed,
            "delta_r2": pmc - sealed,
            "sign": self.sign,
            "input_record_sha256": self.input_record_sha256,
        }


def _reviewed_paired_statistics(deltas: Sequence[float], *, label: str) -> dict[str, object]:
    """Compose the established paired-statistics implementation lazily."""
    if not deltas:
        raise ScoreError("paired statistics requires nonempty deltas")
    try:
        from src.tfpd_lane import matched_scorer
    except Exception as error:  # pragma: no cover - environment/import gate
        raise ScoreError("reviewed matched_scorer is unavailable") from error
    result = matched_scorer.paired_session_stats(
        [float(value) for value in deltas], seed=PAIRED_BOOTSTRAP_SEED, n_boot=PAIRED_BOOTSTRAP_DRAWS,
    )
    required = {
        "mean", "median", "n_positive", "n_total", "min", "max", "all_deltas",
        "bootstrap_95_interval", "exact_sign_pattern", "bootstrap_seed",
    }
    if set(result) < required or result.get("bootstrap_seed") != PAIRED_BOOTSTRAP_SEED:
        raise ScoreError(f"reviewed paired-statistics contract drift: {label}")
    return dict(result)


@dataclass(frozen=True)
class PairedSurfaceBudget:
    surface: str
    budget: int
    rows: tuple[PairedSessionDelta, ...]

    def payload(self, *, expected_roster: Sequence[str]) -> dict[str, object]:
        if tuple(row.session for row in self.rows) != tuple(expected_roster):
            raise ScoreError("paired surface/budget roster/order drift")
        rows = [row.payload() for row in self.rows]
        stats = _reviewed_paired_statistics([row["delta_r2"] for row in rows], label=f"{self.surface}-M{self.budget}")
        return {
            "surface": self.surface,
            "budget": self.budget,
            "rows": rows,
            "equal_session": True,
            "paired_pmc_minus_sealed": stats,
        }


def pair_mode_evidence(
    pmc: ModeEvidence,
    sealed: ModeEvidence,
    *,
    identity: ScoreIdentity,
    input_payload: Mapping[str, object],
) -> PairedSurfaceBudget:
    """Pair exact same-session rows; no window-weighted aggregate is allowed."""
    if pmc.cell.surface != sealed.cell.surface or pmc.cell.budget != sealed.cell.budget:
        raise ScoreError("paired mode surface/budget mismatch")
    if pmc.cell.system != SYSTEM_PMC or sealed.cell.system != SYSTEM_SEALED:
        raise ScoreError("paired mode system mismatch")
    pmc_payload = pmc.payload(identity=identity, input_payload=input_payload)
    sealed_payload = sealed.payload(identity=identity, input_payload=input_payload)
    p_rows = {row["session"]: row for row in pmc_payload["sessions"]}
    s_rows = {row["session"]: row for row in sealed_payload["sessions"]}
    expected = identity.within_roster if pmc.cell.surface == WITHIN else identity.external_roster
    records = {
        (row["surface"], row["session"]): _digest(_json(row))
        for row in input_payload["records"]
    }
    rows = tuple(
        PairedSessionDelta(
            surface=pmc.cell.surface,
            budget=pmc.cell.budget,
            session=session,
            n_windows=p_rows[session]["n_windows"],
            pmc_r2=p_rows[session]["r2"],
            sealed_r2=s_rows[session]["r2"],
            input_record_sha256=records[(pmc.cell.surface, session)],
        )
        for session in expected
    )
    return PairedSurfaceBudget(surface=pmc.cell.surface, budget=pmc.cell.budget, rows=rows)


def decide_verdict(paired: Mapping[str, Mapping[str, Mapping[str, object]]]) -> dict[str, object]:
    """Apply the predeclared within-safety/external-governing gate."""
    try:
        within_m30 = paired[WITHIN][str(SAFETY_BUDGET)]["paired_pmc_minus_sealed"]
        external_m30 = paired[EXTERNAL][str(SAFETY_BUDGET)]["paired_pmc_minus_sealed"]
        external_m4 = paired[EXTERNAL][str(HEADLINE_BUDGET)]["paired_pmc_minus_sealed"]
    except (KeyError, TypeError) as error:
        raise ScoreError("paired verdict budget/surface topology drift") from error
    for label, stats, count in (
        ("within M30", within_m30, EXPECTED_SESSION_COUNTS[WITHIN]),
        ("external M30", external_m30, EXPECTED_SESSION_COUNTS[EXTERNAL]),
        ("external M4", external_m4, EXPECTED_SESSION_COUNTS[EXTERNAL]),
    ):
        if not isinstance(stats, Mapping) or stats.get("n_total") != count:
            raise ScoreError(f"{label} paired-stat count drift")
        if type(stats.get("n_positive")) is not int or not 0 <= stats["n_positive"] <= count:
            raise ScoreError(f"{label} paired-stat positivity drift")
        _finite(stats.get("mean"), f"{label} paired mean")
        _finite(stats.get("median"), f"{label} paired median")
    within_safety = float(within_m30["mean"]) >= M30_SAFETY_MIN
    external_safety = float(external_m30["mean"]) >= M30_SAFETY_MIN
    external_headline = float(external_m4["mean"]) >= M4_HEADLINE_MIN
    external_breadth = int(external_m4["n_positive"]) >= M4_EXTERNAL_POSITIVE_MIN
    if not within_safety or not external_safety:
        verdict = "STOP__M30_SAFETY"
    elif external_safety and external_headline and external_breadth and float(external_m30["mean"]) >= 0.0:
        verdict = "CLEAR_GO"
    elif external_safety and external_headline and external_breadth:
        verdict = "PROMISING__M4_SCREEN"
    else:
        verdict = "HOLD__DESCRIPTIVE_SCREEN"
    return {
        "verdict": verdict,
        "verdict_rule": "STOP_if_within_or_external_M30_safety_fails_else_CLEAR_GO_if_external_M30_nonnegative_and_M4_headline_breadth_pass_else_PROMISING_or_HOLD",
        "within_safety": {
            "budget": SAFETY_BUDGET,
            "threshold": M30_SAFETY_MIN,
            "mean_delta": float(within_m30["mean"]),
            "passed": within_safety,
        },
        "external_governing": {
            "m30_threshold": M30_SAFETY_MIN,
            "m30_mean_delta": float(external_m30["mean"]),
            "m30_passed": external_safety,
            "m4_threshold": M4_HEADLINE_MIN,
            "m4_mean_delta": float(external_m4["mean"]),
            "m4_passed": external_headline,
            "m4_positive_threshold": M4_EXTERNAL_POSITIVE_MIN,
            "m4_positive_sessions": int(external_m4["n_positive"]),
            "m4_total_sessions": EXPECTED_SESSION_COUNTS[EXTERNAL],
            "breadth_passed": external_breadth,
        },
        "formal_verdict": False,
    }


def build_score_payload(
    *,
    identity: ScoreIdentity,
    input_authority: InputAuthority,
    evidence: Sequence[ModeEvidence],
) -> dict[str, object]:
    """Build a canonical no-write score payload from already materialized rows."""
    identity_payload = identity.payload()
    input_payload = input_authority.payload(identity=identity)
    expected = score_matrix()
    if len(evidence) != len(expected) or tuple(item.cell for item in evidence) != expected:
        raise ScoreError("matched score evidence matrix/order drift")
    evidence_payload = [item.payload(identity=identity, input_payload=input_payload) for item in evidence]
    by_key = {(row["cell"]["surface"], row["cell"]["budget"], row["cell"]["system"]): item for row, item in zip(evidence_payload, evidence)}
    paired_payload: dict[str, dict[str, object]] = {surface: {} for surface in SURFACES}
    for surface in SURFACES:
        expected_roster = identity.within_roster if surface == WITHIN else identity.external_roster
        for budget in BUDGETS:
            pmc = by_key[(surface, budget, SYSTEM_PMC)]
            sealed = by_key[(surface, budget, SYSTEM_SEALED)]
            pair = pair_mode_evidence(pmc, sealed, identity=identity, input_payload=input_payload)
            paired_payload[surface][str(budget)] = pair.payload(expected_roster=expected_roster)
    verdict = decide_verdict(paired_payload)
    return {
        "schema": SCHEMA,
        "cell": CELL,
        "phase": PHASE,
        "identity": identity_payload,
        "metric": dict(METRIC_CONTRACT),
        "input_authority": input_payload,
        "matrix": evidence_payload,
        "paired_pmc_minus_sealed": paired_payload,
        "verdict": verdict,
        "boundaries": dict(EXECUTION_POLICY),
        "target_optimizer_steps": 0,
        "target_backward_calls": 0,
        "target_update_calls": 0,
        "formal_opened": False,
    }


def validate_score_payload(value: Mapping[str, object], *, identity: ScoreIdentity | None = None) -> dict[str, object]:
    """Validate durable-shaped score evidence without loading a result path."""
    payload = _mapping_copy(value, "matched score payload")
    required = {
        "schema", "cell", "phase", "identity", "metric", "input_authority", "matrix",
        "paired_pmc_minus_sealed", "verdict", "boundaries", "target_optimizer_steps",
        "target_backward_calls", "target_update_calls", "formal_opened",
    }
    if set(payload) != required or payload["schema"] != SCHEMA or payload["cell"] != CELL or payload["phase"] != PHASE:
        raise ScoreError("matched score payload schema/cell/phase drift")
    if identity is not None and payload["identity"] != identity.payload():
        raise ScoreError("matched score identity drift")
    if payload.get("metric") != METRIC_CONTRACT:
        raise ScoreError("matched score fixed-bin metric contract drift")
    input_authority = payload.get("input_authority")
    if not isinstance(input_authority, Mapping) or not isinstance(input_authority.get("records"), list) \
            or any(
                not isinstance(row, Mapping)
                or row.get("last_bin_query") != "fixed_bin_49_of_each_50_bin_window"
                for row in input_authority["records"]
            ):
        raise ScoreError("matched score input fixed-bin authority drift")
    if payload["boundaries"] != EXECUTION_POLICY or payload["target_optimizer_steps"] != 0 \
            or payload["target_backward_calls"] != 0 or payload["target_update_calls"] != 0 \
            or payload["formal_opened"] is not False:
        raise ScoreError("matched score target/posterior boundary drift")
    matrix = payload["matrix"]
    if not isinstance(matrix, list) or len(matrix) != len(score_matrix()):
        raise ScoreError("matched score matrix cardinality drift")
    expected_cells = [cell.payload() for cell in score_matrix()]
    if [row.get("cell") for row in matrix if isinstance(row, Mapping)] != expected_cells:
        raise ScoreError("matched score matrix order drift")
    matrix_by_key: dict[tuple[str, int, str], Mapping[str, object]] = {}
    for row in matrix:
        if not isinstance(row, Mapping) or row.get("ordinary_ols_point_t4_used") is not True:
            raise ScoreError("matched score inference carrier drift")
        if any(row.get(key) is not False for key in (
            "posterior_sample_used", "posterior_mean_used", "posterior_normalizer_used", "posterior_credibility_used",
        )):
            raise ScoreError("matched score posterior inference leak")
        cell = row.get("cell")
        if not isinstance(cell, Mapping):
            raise ScoreError("matched score matrix cell type drift")
        matrix_by_key[(cell.get("surface"), cell.get("budget"), cell.get("system"))] = row
    paired = payload["paired_pmc_minus_sealed"]
    if not isinstance(paired, Mapping) or set(paired) != set(SURFACES):
        raise ScoreError("matched score paired surface topology drift")
    for surface in SURFACES:
        block = paired[surface]
        if not isinstance(block, Mapping) or set(block) != {str(budget) for budget in BUDGETS}:
            raise ScoreError("matched score paired budget topology drift")
        expected_count = EXPECTED_SESSION_COUNTS[surface]
        for budget in BUDGETS:
            row = block[str(budget)]
            if not isinstance(row, Mapping) or not isinstance(row.get("rows"), list) \
                    or len(row["rows"]) != expected_count:
                raise ScoreError("matched score paired row topology drift")
            expected_roster = (
                tuple(identity.within_roster) if identity is not None and surface == WITHIN else
                tuple(identity.external_roster) if identity is not None else
                tuple(item.get("session") for item in row["rows"])
            )
            if tuple(item.get("session") for item in row["rows"] if isinstance(item, Mapping)) != expected_roster:
                raise ScoreError("matched score paired roster/order drift")
            stats = row.get("paired_pmc_minus_sealed")
            if not isinstance(stats, Mapping):
                raise ScoreError("matched score paired statistics missing")
            recomputed_deltas: list[float] = []
            for item in row["rows"]:
                if not isinstance(item, Mapping):
                    raise ScoreError("matched score paired row type drift")
                pmc = _finite(item.get("pmc_r2"), "paired PMC R2")
                sealed = _finite(item.get("sealed_r2"), "paired sealed R2")
                if item.get("delta_r2") != pmc - sealed:
                    raise ScoreError("matched score paired delta drift")
                expected_sign = "+" if pmc - sealed > 0 else ("-" if pmc - sealed < 0 else "0")
                if item.get("sign") != expected_sign:
                    raise ScoreError("matched score paired sign drift")
                recomputed_deltas.append(pmc - sealed)
                session = item.get("session")
                p_row = matrix_by_key.get((surface, budget, SYSTEM_PMC))
                s_row = matrix_by_key.get((surface, budget, SYSTEM_SEALED))
                if not isinstance(p_row, Mapping) or not isinstance(s_row, Mapping):
                    raise ScoreError("matched score paired matrix endpoint missing")
                p_score = next((candidate for candidate in p_row.get("sessions", ())
                                if isinstance(candidate, Mapping) and candidate.get("session") == session), None)
                s_score = next((candidate for candidate in s_row.get("sessions", ())
                                if isinstance(candidate, Mapping) and candidate.get("session") == session), None)
                if not isinstance(p_score, Mapping) or not isinstance(s_score, Mapping) \
                        or item.get("pmc_r2") != p_score.get("r2") or item.get("sealed_r2") != s_score.get("r2"):
                    raise ScoreError("matched score paired endpoint/value binding drift")
            expected_stats = _reviewed_paired_statistics(recomputed_deltas, label=f"{surface}-M{budget}")
            if dict(stats) != expected_stats:
                raise ScoreError("matched score paired bootstrap/statistics drift")
    verdict = payload.get("verdict")
    if not isinstance(verdict, Mapping) or dict(verdict) != decide_verdict(paired):
        raise ScoreError("matched score verdict/gate drift")
    return payload


def score_last_bin_with_reviewed_metric(predictions: Any, targets: Any, valid_mask: Any) -> float:
    """Score pre-materialized windows through the one reviewed metric seam.

    This helper intentionally imports Torch and the shared scorer only when a
    future authorized backend calls it.  It performs no path or artifact I/O.
    """
    try:
        import torch
        from src.tfpd_lane import matched_scorer
    except Exception as error:  # pragma: no cover - guarded runtime dependency
        raise ScoreError("reviewed Torch/matched-scorer metric is unavailable") from error
    if not all(torch.is_tensor(item) for item in (predictions, targets, valid_mask)):
        raise ScoreError("last-bin metric requires Torch tensors")
    if (predictions.ndim != 3 or predictions.shape[0] <= 0
            or predictions.shape[1] != METRIC_CONTRACT["window_bins"]
            or targets.shape != predictions.shape or predictions.shape[-1] != METRIC_CONTRACT["behavior_coordinates"]):
        raise ScoreError("last-bin prediction/target shape drift")
    if valid_mask.ndim != 2 or valid_mask.shape != predictions.shape[:2] or valid_mask.dtype is not torch.bool:
        raise ScoreError("last-bin valid-mask shape/dtype drift")
    if not bool(torch.isfinite(predictions).all().item()) or not bool(torch.isfinite(targets).all().item()):
        raise ScoreError("last-bin prediction/target nonfinite")
    if not bool(valid_mask[:, 49].all().item()):
        raise ScoreError("fixed governed bin 49 is not valid for every window")
    selected_predictions = predictions[:, 49, :].detach().cpu().contiguous()
    selected_targets = targets[:, 49, :].detach().cpu().contiguous()
    return matched_scorer.session_r2(selected_predictions, selected_targets)


class NoLiveBackend:
    """Public dry sentinel: it cannot resolve terminal, SWA, or evaluation data."""

    def prepare(self) -> None:
        raise ScoreError("PMC-D matched scoring is scaffold-only until a valid terminal/SWA and root capability exist")


def execute_authorized(*_: object, **__: object) -> dict[str, object]:
    """Fail before any root/result/data/CUDA access in this additive stage."""
    raise ScoreError("PMC-D matched score has no live execution capability in this scaffold")


def dry_plan() -> dict[str, object]:
    """Return a standard-library-only no-data scorer plan."""
    return {
        "schema": SCHEMA,
        "cell": CELL,
        "phase": PHASE,
        "status": "DRY_ONLY__NO_TERMINAL_OR_SWA_READ__NO_NWB_NO_CHECKPOINT_TENSOR_NO_CUDA_NO_GPU_NO_WRITE",
        "workorder": {"relative_path": WORKORDER_RELATIVE, "sha256": WORKORDER_SHA256},
        "preconditions": {
            "full_terminal": FULL_TRAIN_TERMINAL_RELATIVE,
            "full_swa": FULL_TRAIN_SWA_RELATIVE,
            "full_swa_manifest": FULL_TRAIN_SWA_MANIFEST_RELATIVE,
            "final_four_checkpoints": list(FULL_TRAIN_CHECKPOINT_RELATIVES),
            "sealed_cell_d_terminal": SEALED_CELL_D_TERMINAL_RELATIVE,
            "sealed_cell_d_swa": SEALED_CELL_D_SWA_RELATIVE,
            "sealed_cell_d_baseline": SEALED_CELL_D_BASELINE_RELATIVE,
            "valid_terminal_and_swa_required_before_execution": True,
        },
        "metric": dict(METRIC_CONTRACT),
        "surfaces": {WITHIN: 6, EXTERNAL: 15},
        "budgets": list(BUDGETS),
        "inference": {
            "carrier": INFERENCE_SEMANTICS,
            "posterior_sample": False,
            "posterior_mean": False,
            "posterior_normalizer": False,
            "posterior_credibility": False,
        },
        "gates": {
            "within_safety_m30_min_delta": M30_SAFETY_MIN,
            "external_governing_m30_min_delta": M30_SAFETY_MIN,
            "external_governing_m4_min_delta": M4_HEADLINE_MIN,
            "external_governing_m4_positive_sessions": M4_EXTERNAL_POSITIVE_MIN,
        },
        "execution_policy": dict(EXECUTION_POLICY),
        "canonical_roots": {
            "authority": AUTHORITY_ROOT_RELATIVE,
            "score": SCORE_ROOT_RELATIVE,
            "creation": "forbidden_until_separate_root_review",
        },
    }
