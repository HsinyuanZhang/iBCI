"""Dry, additive quick-score contract for PIRG.

The source-training cell and the evaluation screen have deliberately separate
lifecycles.  This file is standard-library-only: importing it cannot open an
evaluation asset, load a Cell-D/PIRG artifact, import Torch, or initialise a
device.  A future physical adapter may be attached only after a reviewed
source terminal and a new in-process capability exist.
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
from statistics import median
from typing import Mapping, Protocol, Sequence

from . import plan
from . import train


class PIRGScoreError(RuntimeError):
    """Fail closed before a descriptive PIRG score can become durable."""


RESULT_ROOT_RELATIVE = "tfpd_exploration/results/posterior_identity_residual_gate_score_v1"
WITHIN, EXTERNAL = "within", "external"
SURFACES = (WITHIN, EXTERNAL)
BUDGETS = (30, 10, 4)
BASELINE_MODE = "sealed_cell_d_ols_point"
PIRG_MODE = "posterior_identity_residual_gate"
MODES = (BASELINE_MODE, PIRG_MODE)
M30_SAFETY_MIN = -0.02
M4_PROMISING_MEAN_MIN = 0.03
M4_PROMISING_POSITIVE_MIN = 4

# PIRG reuses V3 only as a reviewed *input/device substrate*.  These are not
# results reused in this score: every one of the twelve Cell-D/PIRG cells is
# newly forwarded.  Binding the completed V3 chain prevents a caller from
# quietly changing the selected 3+3 roster or parser/device lineage.
V3_INPUT_AUTHORITY_SHA256 = plan.V3_QUICK_INPUT_AUTHORITY_SHA256
V3_SCORE_SHA256 = plan.V3_QUICK_SCORE_SHA256
V3_TERMINAL_SHA256 = plan.V3_QUICK_TERMINAL_SHA256
V3_STAGE_ROOT = plan.V3_QUICK_STAGE_ROOT

# This is exactly the fixed V3 engineering selection, in frozen roster order.
FIXED_SESSIONS: Mapping[str, tuple[str, ...]] = {
    WITHIN: (
        "sub-C_ses-CO-20151103",
        "sub-C_ses-CO-20151106",
        "sub-C_ses-CO-20151112",
    ),
    EXTERNAL: (
        "sub-M_ses-CO-20140307",
        "sub-M_ses-CO-20150611",
        "sub-M_ses-CO-20150626",
    ),
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PIRGScoreError(message)


def _json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise PIRGScoreError(f"{label} must be an exact lowercase SHA-256")
    return value


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise PIRGScoreError(f"{label} must be finite")
    return float(value)


@dataclass(frozen=True)
class ScoreClosure:
    """Exact, broader score closure over the source route plus V3 seam."""

    payload_value: Mapping[str, object]

    def payload(self) -> dict[str, object]:
        required = {"paths", "sha256_by_path", "closure_sha256"}
        if not isinstance(self.payload_value, Mapping) or set(self.payload_value) != required:
            raise PIRGScoreError("PIRG score closure schema drift")
        paths = self.payload_value.get("paths")
        hashes = self.payload_value.get("sha256_by_path")
        if (
            not isinstance(paths, list)
            or tuple(paths) != train.SCORE_IMPLEMENTATION_CLOSURE
            or not isinstance(hashes, Mapping)
            or set(hashes) != set(train.SCORE_IMPLEMENTATION_CLOSURE)
        ):
            raise PIRGScoreError("PIRG score closure topology drift")
        body = {
            "paths": list(train.SCORE_IMPLEMENTATION_CLOSURE),
            "sha256_by_path": {
                path: _sha(hashes[path], f"PIRG score closure {path}")
                for path in train.SCORE_IMPLEMENTATION_CLOSURE
            },
        }
        if self.payload_value.get("closure_sha256") != _digest(_json(body)):
            raise PIRGScoreError("PIRG score closure digest drift")
        return {**body, "closure_sha256": str(self.payload_value["closure_sha256"])}


def score_implementation_closure(root: Path) -> ScoreClosure:
    """Descriptor-hash the fixed score closure without touching evaluation data.

    This deliberately has a different topology from the source-training
    closure.  The score consumes the reviewed V3 parser/device composition;
    source training must not stage or import that evaluation machinery.
    """
    base = Path(root).absolute()
    hashes: dict[str, str] = {}
    for relative in train.SCORE_IMPLEMENTATION_CLOSURE:
        path = base / relative
        before = os.lstat(path)
        if not stat.S_ISREG(before.st_mode) or stat.S_ISLNK(before.st_mode):
            raise PIRGScoreError(f"PIRG score closure leaf missing or aliased: {relative}")
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            opened = os.fstat(descriptor)
            if (opened.st_dev, opened.st_ino, opened.st_size) != (before.st_dev, before.st_ino, before.st_size):
                raise PIRGScoreError("PIRG score closure descriptor identity drift")
            pieces: list[bytes] = []
            while True:
                chunk = os.read(descriptor, 1 << 20)
                if not chunk:
                    break
                pieces.append(chunk)
        finally:
            os.close(descriptor)
        after = os.lstat(path)
        if (after.st_dev, after.st_ino, after.st_size) != (before.st_dev, before.st_ino, before.st_size):
            raise PIRGScoreError("PIRG score closure changed during descriptor read")
        hashes[relative] = _digest(b"".join(pieces))
    return ScoreClosure({
        "paths": list(train.SCORE_IMPLEMENTATION_CLOSURE),
        "sha256_by_path": hashes,
        "closure_sha256": _digest(_json({
            "paths": list(train.SCORE_IMPLEMENTATION_CLOSURE),
            "sha256_by_path": hashes,
        })),
    })


@dataclass(frozen=True)
class PIRGScoreIdentity:
    """Immutable source-terminal/model binding for the future quick score."""

    closure: ScoreClosure
    source_terminal_sha256: str
    final_alpha_sha256: str
    source_authority_sha256: str

    def payload(self) -> dict[str, object]:
        return {
            "schema": "posterior_identity_residual_gate_score_identity_v1",
            "cell": plan.CELL,
            "classification": "NON_GOVERNING_PIRG_PERFORMANCE_SCREEN",
            "source_terminal_sha256": _sha(self.source_terminal_sha256, "PIRG score source terminal SHA"),
            "final_alpha_sha256": _sha(self.final_alpha_sha256, "PIRG score final alpha SHA"),
            "source_authority_sha256": _sha(self.source_authority_sha256, "PIRG score source authority SHA"),
            "sealed_cell_d_swa_sha256": plan.SEALED_CELL_D_SWA_SHA256,
            "ordinary_ols_t4_held": True,
            "posterior_mean_used": False,
            "posterior_normalizer_used": False,
            "posterior_sampling_used": False,
            "attention_logit_bias_used": False,
            "gate_scope": "identity_residual_only",
            "credibility_provenance": {
                "same_prefix_conjugate_directional_precision": True,
                "v3_substrate_may_construct_a_posterior_view_to_expose_credibility": True,
                "pirg_forward_reads_only": ["directional_credibility"],
                "pirg_forward_reads_no": ["raw_t4", "normalized_t4", "sample", "attention_logit_bias"],
            },
            "v3_selected_input_substrate": {
                "stage_root": V3_STAGE_ROOT,
                "input_authority_sha256": V3_INPUT_AUTHORITY_SHA256,
                "score_sha256": V3_SCORE_SHA256,
                "terminal_sha256": V3_TERMINAL_SHA256,
                "reused_scientific_cells": False,
                "all_pirg_cells_newly_forwarded": True,
            },
            "closure": self.closure.payload(),
            "boundaries": {
                "target_optimizer_steps": 0,
                "target_backward_calls": 0,
                "target_update_calls": 0,
                "eval_no_grad": True,
                "eval_dropout_disabled": True,
                "formal_opened": False,
                "h1_opened": False,
            },
        }


def validate_score_identity(value: PIRGScoreIdentity | Mapping[str, object]) -> dict[str, object]:
    payload = value.payload() if isinstance(value, PIRGScoreIdentity) else dict(value)
    required = {
        "schema", "cell", "classification", "source_terminal_sha256", "final_alpha_sha256",
        "source_authority_sha256", "sealed_cell_d_swa_sha256", "ordinary_ols_t4_held",
        "posterior_mean_used", "posterior_normalizer_used", "posterior_sampling_used",
        "attention_logit_bias_used", "gate_scope", "credibility_provenance", "v3_selected_input_substrate", "closure", "boundaries",
    }
    if set(payload) != required:
        raise PIRGScoreError("PIRG score identity schema drift")
    closure = ScoreClosure(payload["closure"]).payload()
    rebuilt = PIRGScoreIdentity(
        closure=ScoreClosure(closure),
        source_terminal_sha256=_sha(payload["source_terminal_sha256"], "PIRG score source terminal SHA"),
        final_alpha_sha256=_sha(payload["final_alpha_sha256"], "PIRG score final alpha SHA"),
        source_authority_sha256=_sha(payload["source_authority_sha256"], "PIRG score source authority SHA"),
    ).payload()
    if payload != rebuilt:
        raise PIRGScoreError("PIRG score identity semantic boundary drift")
    return rebuilt


@dataclass(frozen=True)
class ScoreCell:
    surface: str
    mode: str
    budget: int

    def payload(self) -> dict[str, object]:
        if self.surface not in SURFACES or self.mode not in MODES or self.budget not in BUDGETS:
            raise PIRGScoreError("PIRG score-cell topology drift")
        return {"surface": self.surface, "mode": self.mode, "budget": self.budget}


def score_matrix() -> tuple[ScoreCell, ...]:
    return tuple(
        ScoreCell(surface=surface, mode=mode, budget=budget)
        for surface in SURFACES for budget in BUDGETS for mode in MODES
    )


@dataclass(frozen=True)
class SessionScore:
    session: str
    n_windows: int
    r2: float
    prediction_sha256: str
    input_sha256: str

    def payload(self) -> dict[str, object]:
        if not isinstance(self.session, str) or not self.session or type(self.n_windows) is not int or self.n_windows <= 0:
            raise PIRGScoreError("PIRG score session row topology drift")
        return {
            "session": self.session,
            "n_windows": self.n_windows,
            "r2": _finite(self.r2, "PIRG session R2"),
            "prediction_sha256": _sha(self.prediction_sha256, "PIRG prediction SHA"),
            "input_sha256": _sha(self.input_sha256, "PIRG input SHA"),
        }


@dataclass(frozen=True)
class CellEvidence:
    cell: ScoreCell
    model_state_before_sha256: str
    model_state_after_sha256: str
    model_artifact_sha256: str
    base_cell_d_swa_sha256: str
    rows: tuple[SessionScore, ...]
    eval_mode: bool = True
    dropout_disabled: bool = True
    gradients_none: bool = True
    repeated_fixed_batch_bitwise_equal: bool = True
    finite_outputs: bool = True

    def payload(self, *, identity: PIRGScoreIdentity, input_authority_sha256: str) -> dict[str, object]:
        cell = self.cell.payload()
        expected = FIXED_SESSIONS[self.cell.surface]
        if tuple(row.session for row in self.rows) != expected:
            raise PIRGScoreError("PIRG cell evidence session order drift")
        if any(row.n_windows <= 0 for row in self.rows):
            raise PIRGScoreError("PIRG cell evidence empty session")
        expected_artifact = plan.SEALED_CELL_D_SWA_SHA256 if self.cell.mode == BASELINE_MODE else identity.final_alpha_sha256
        if self.model_artifact_sha256 != expected_artifact or self.base_cell_d_swa_sha256 != plan.SEALED_CELL_D_SWA_SHA256:
            raise PIRGScoreError("PIRG cell evidence model artifact drift")
        if (self.model_state_before_sha256 != self.model_state_after_sha256 or not self.eval_mode
                or not self.dropout_disabled or not self.gradients_none
                or not self.repeated_fixed_batch_bitwise_equal or not self.finite_outputs):
            raise PIRGScoreError("PIRG cell forward invariant drift")
        return {
            "cell": cell,
            "model_system": "sealed_cell_d_swa" if self.cell.mode == BASELINE_MODE else "pirg_alpha_only_on_sealed_cell_d_swa",
            "model_artifact_sha256": _sha(self.model_artifact_sha256, "PIRG cell model artifact SHA"),
            "base_cell_d_swa_sha256": _sha(self.base_cell_d_swa_sha256, "PIRG cell base Cell-D SWA SHA"),
            "model_state_before_sha256": _sha(self.model_state_before_sha256, "PIRG cell pre-forward state SHA"),
            "model_state_after_sha256": _sha(self.model_state_after_sha256, "PIRG cell post-forward state SHA"),
            "session_scores": [row.payload() for row in self.rows],
            "mean_r2": sum(float(row.r2) for row in self.rows) / len(self.rows),
            "median_r2": float(median(float(row.r2) for row in self.rows)),
            "input_authority_sha256": _sha(input_authority_sha256, "PIRG cell input authority SHA"),
            "eval_mode": True,
            "dropout_disabled": True,
            "gradients_none": True,
            "repeated_fixed_batch_bitwise_equal": True,
            "finite_outputs": True,
        }


def validate_cell_evidence_payload(
    payload: Mapping[str, object], *, identity: PIRGScoreIdentity, input_authority_sha256: str,
) -> dict[str, object]:
    """Rebuild one cell instead of trusting caller-provided receipt booleans."""
    required = {
        "cell", "model_system", "model_artifact_sha256", "base_cell_d_swa_sha256",
        "model_state_before_sha256", "model_state_after_sha256", "session_scores", "mean_r2", "median_r2",
        "input_authority_sha256", "eval_mode", "dropout_disabled", "gradients_none",
        "repeated_fixed_batch_bitwise_equal", "finite_outputs",
    }
    if not isinstance(payload, Mapping) or set(payload) != required:
        raise PIRGScoreError("PIRG cell evidence schema drift")
    raw_cell = payload.get("cell")
    if not isinstance(raw_cell, Mapping):
        raise PIRGScoreError("PIRG cell evidence cell type drift")
    cell = ScoreCell(
        surface=str(raw_cell.get("surface")), mode=str(raw_cell.get("mode")), budget=raw_cell.get("budget"),
    )
    cell.payload()
    raw_rows = payload.get("session_scores")
    if not isinstance(raw_rows, list):
        raise PIRGScoreError("PIRG cell evidence session-score type drift")
    rows: list[SessionScore] = []
    for row in raw_rows:
        if not isinstance(row, Mapping) or set(row) != {"session", "n_windows", "r2", "prediction_sha256", "input_sha256"}:
            raise PIRGScoreError("PIRG cell evidence session-score schema drift")
        rows.append(SessionScore(
            session=row.get("session"), n_windows=row.get("n_windows"), r2=row.get("r2"),
            prediction_sha256=row.get("prediction_sha256"), input_sha256=row.get("input_sha256"),
        ))
    rebuilt = CellEvidence(
        cell=cell,
        model_state_before_sha256=payload.get("model_state_before_sha256"),
        model_state_after_sha256=payload.get("model_state_after_sha256"),
        model_artifact_sha256=payload.get("model_artifact_sha256"),
        base_cell_d_swa_sha256=payload.get("base_cell_d_swa_sha256"),
        rows=tuple(rows), eval_mode=payload.get("eval_mode"), dropout_disabled=payload.get("dropout_disabled"),
        gradients_none=payload.get("gradients_none"),
        repeated_fixed_batch_bitwise_equal=payload.get("repeated_fixed_batch_bitwise_equal"),
        finite_outputs=payload.get("finite_outputs"),
    ).payload(identity=identity, input_authority_sha256=input_authority_sha256)
    if dict(payload) != rebuilt:
        raise PIRGScoreError("PIRG cell evidence semantic/binding drift")
    return rebuilt


@dataclass(frozen=True)
class InputAuthority:
    """Only opaque, per-session materialization digests enter the receipt."""

    records: Mapping[str, Sequence[Mapping[str, object]]]

    def payload(self, *, identity: PIRGScoreIdentity) -> dict[str, object]:
        rows: dict[str, list[dict[str, object]]] = {}
        for surface in SURFACES:
            original = self.records.get(surface)
            if not isinstance(original, Sequence) or isinstance(original, (str, bytes)) or len(original) != 3:
                raise PIRGScoreError("PIRG input authority surface cardinality drift")
            converted: list[dict[str, object]] = []
            for expected_session, item in zip(FIXED_SESSIONS[surface], original, strict=True):
                if not isinstance(item, Mapping) or set(item) != {
                    "session", "n_windows", "input_sha256", "last_bin_target_sha256", "last_bin_mask_sha256",
                    "point_side_sha256s", "directional_credibility_sha256s", "prefix_row_ids_sha256s",
                }:
                    raise PIRGScoreError("PIRG input authority row schema drift")
                if item.get("session") != expected_session or type(item.get("n_windows")) is not int or item["n_windows"] <= 0:
                    raise PIRGScoreError("PIRG input authority session/order drift")
                bound: dict[str, dict[str, str]] = {}
                for label, key in (
                    ("ordinary OLS point-side", "point_side_sha256s"),
                    ("directional credibility", "directional_credibility_sha256s"),
                    ("same-prefix rows", "prefix_row_ids_sha256s"),
                ):
                    digest_map = item.get(key)
                    expected_keys = {str(budget) for budget in BUDGETS}
                    if not isinstance(digest_map, Mapping) or set(digest_map) != expected_keys:
                        raise PIRGScoreError(f"PIRG {label} M4/M10/M30 authority topology drift")
                    bound[key] = {
                        str(budget): _sha(digest_map[str(budget)], f"PIRG {label} M{budget} SHA")
                        for budget in BUDGETS
                    }
                converted.append({
                    "session": expected_session, "n_windows": int(item["n_windows"]),
                    "input_sha256": _sha(item["input_sha256"], "PIRG input digest"),
                    "last_bin_target_sha256": _sha(item["last_bin_target_sha256"], "PIRG target digest"),
                    "last_bin_mask_sha256": _sha(item["last_bin_mask_sha256"], "PIRG mask digest"),
                    **bound,
                })
            rows[surface] = converted
        return {
            "schema": "posterior_identity_residual_gate_input_authority_v1",
            "identity": validate_score_identity(identity),
            "surfaces": rows,
            "metric": "last_bin_variance_weighted_r2_equal_session",
            "same_inputs_for_cell_d_and_pirg": True,
            "within_opened": True,
            "external_opened": True,
            "target_optimizer_steps": 0,
            "target_backward_calls": 0,
            "target_update_calls": 0,
            "formal_opened": False,
            "h1_opened": False,
        }


def validate_input_authority(payload: Mapping[str, object], *, identity: PIRGScoreIdentity) -> dict[str, object]:
    expected_keys = {
        "schema", "identity", "surfaces", "metric", "same_inputs_for_cell_d_and_pirg",
        "within_opened", "external_opened", "target_optimizer_steps", "target_backward_calls", "target_update_calls",
        "formal_opened", "h1_opened",
    }
    if not isinstance(payload, Mapping) or set(payload) != expected_keys:
        raise PIRGScoreError("PIRG input authority schema drift")
    if payload.get("identity") != validate_score_identity(identity):
        raise PIRGScoreError("PIRG input authority identity drift")
    rebuilt = InputAuthority(payload.get("surfaces", {})).payload(identity=identity)
    if dict(payload) != rebuilt:
        raise PIRGScoreError("PIRG input authority semantic/boundary drift")
    return rebuilt


def _cell_key(cell: Mapping[str, object]) -> tuple[str, str, int]:
    if not isinstance(cell, Mapping) or set(cell) != {"surface", "mode", "budget"}:
        raise PIRGScoreError("PIRG score cell key schema drift")
    return ScoreCell(str(cell["surface"]), str(cell["mode"]), int(cell["budget"])).surface, str(cell["mode"]), int(cell["budget"])


def _paired_summary(*, baseline: Mapping[str, object], pirg: Mapping[str, object]) -> dict[str, object]:
    left, right = baseline.get("session_scores"), pirg.get("session_scores")
    if not isinstance(left, list) or not isinstance(right, list) or len(left) != len(right):
        raise PIRGScoreError("PIRG paired score cardinality drift")
    deltas: list[dict[str, object]] = []
    for b, p in zip(left, right, strict=True):
        if not isinstance(b, Mapping) or not isinstance(p, Mapping) or b.get("session") != p.get("session"):
            raise PIRGScoreError("PIRG paired score session order drift")
        delta = _finite(p.get("r2"), "PIRG paired PIRG R2") - _finite(b.get("r2"), "PIRG paired baseline R2")
        deltas.append({"session": str(b["session"]), "delta_r2": delta})
    values = [float(item["delta_r2"]) for item in deltas]
    return {
        "per_session": deltas,
        "mean_delta_r2": sum(values) / len(values),
        "median_delta_r2": float(median(values)),
        "positive_count": sum(value > 0.0 for value in values),
        "n_sessions": len(values),
    }


def build_score_payload(*, identity: PIRGScoreIdentity, input_payload: Mapping[str, object],
                        evidence: Sequence[Mapping[str, object]]) -> dict[str, object]:
    validate_input_authority(input_payload, identity=identity)
    expected = tuple((item.surface, item.mode, item.budget) for item in score_matrix())
    cells: dict[tuple[str, str, int], dict[str, object]] = {}
    input_sha = _digest(_json(input_payload))
    input_rows = {
        (surface, str(row["session"])): row
        for surface in SURFACES
        for row in input_payload["surfaces"][surface]
    }
    for item in evidence:
        if not isinstance(item, Mapping):
            raise PIRGScoreError("PIRG score evidence type drift")
        checked = validate_cell_evidence_payload(
            item, identity=identity, input_authority_sha256=input_sha,
        )
        key = _cell_key(checked.get("cell"))
        if key in cells:
            raise PIRGScoreError("PIRG duplicate score evidence")
        expected_sessions = FIXED_SESSIONS[key[0]]
        rows = checked.get("session_scores")
        if not isinstance(rows, list) or tuple(row.get("session") for row in rows if isinstance(row, Mapping)) != expected_sessions:
            raise PIRGScoreError("PIRG score evidence selected roster drift")
        for row in rows:
            if not isinstance(row, Mapping):
                raise PIRGScoreError("PIRG score evidence session-score type drift")
            input_row = input_rows.get((key[0], str(row.get("session"))))
            if (
                input_row is None
                or row.get("n_windows") != input_row.get("n_windows")
                or row.get("input_sha256") != _digest(_json(input_row))
            ):
                raise PIRGScoreError("PIRG score evidence per-session input binding drift")
        cells[key] = checked
    if tuple(cells) != expected:
        raise PIRGScoreError("PIRG score matrix/order/topology drift")
    paired: dict[str, dict[str, object]] = {}
    for surface in SURFACES:
        for budget in BUDGETS:
            paired[f"{surface}_m{budget}"] = _paired_summary(
                baseline=cells[(surface, BASELINE_MODE, budget)], pirg=cells[(surface, PIRG_MODE, budget)],
            )
    within_m30 = float(paired["within_m30"]["mean_delta_r2"])
    external_m30 = float(paired["external_m30"]["mean_delta_r2"])
    m4_values = [
        *(float(item["delta_r2"]) for item in paired["within_m4"]["per_session"]),
        *(float(item["delta_r2"]) for item in paired["external_m4"]["per_session"]),
    ]
    m4_mean = sum(m4_values) / len(m4_values)
    m4_positive = sum(value > 0.0 for value in m4_values)
    if within_m30 < M30_SAFETY_MIN or external_m30 < M30_SAFETY_MIN:
        verdict = "STOP__M30_SAFETY"
    elif m4_mean >= M4_PROMISING_MEAN_MIN and m4_positive >= M4_PROMISING_POSITIVE_MIN:
        verdict = "PROMISING__M4_SCREEN"
    else:
        verdict = "HOLD__DESCRIPTIVE_SCREEN"
    return {
        "schema": "posterior_identity_residual_gate_score_v1",
        "classification": "NON_GOVERNING_PIRG_PERFORMANCE_SCREEN",
        "identity": validate_score_identity(identity),
        "input_authority_sha256": input_sha,
        "metric": "last_bin_variance_weighted_r2_equal_session",
        "matrix": [cells[key] for key in expected],
        "paired_pirg_minus_cell_d": paired,
        "screen": {
            "m30_safety_threshold": M30_SAFETY_MIN,
            "m4_promising_mean_threshold": M4_PROMISING_MEAN_MIN,
            "m4_promising_positive_threshold": M4_PROMISING_POSITIVE_MIN,
            "m4_pooled_mean_delta_r2": m4_mean,
            "m4_pooled_positive_count": m4_positive,
            "m4_pooled_n": len(m4_values),
            "verdict": verdict,
            "formal_verdict": False,
            "no_extra_ablation_arms": True,
        },
        "boundaries": {
            "within_opened": True,
            "external_opened": True,
            "target_optimizer_steps": 0,
            "target_backward_calls": 0,
            "target_update_calls": 0,
            "eval_no_grad": True,
            "eval_dropout_disabled": True,
            "formal_opened": False,
            "h1_opened": False,
        },
    }


def validate_score_payload(payload: Mapping[str, object], *, identity: PIRGScoreIdentity,
                           input_payload: Mapping[str, object]) -> dict[str, object]:
    required = {"schema", "classification", "identity", "input_authority_sha256", "metric", "matrix", "paired_pirg_minus_cell_d", "screen", "boundaries"}
    if not isinstance(payload, Mapping) or set(payload) != required:
        raise PIRGScoreError("PIRG score payload schema drift")
    # Rebuild from canonical matrix evidence; this rejects a forged verdict,
    # summary, input digest, or cell ordering rather than trusting booleans.
    rebuilt = build_score_payload(identity=identity, input_payload=input_payload, evidence=payload["matrix"])
    if dict(payload) != rebuilt:
        raise PIRGScoreError("PIRG score payload semantic drift")
    return rebuilt


class _ArtifactRoot:
    """Fresh O_EXCL score writer with owned-pair rollback on group failure."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root).absolute()
        self._parent_fd: int | None = None
        self._fd: int | None = None
        self._identity: tuple[int, int] | None = None

    def reserve(self) -> None:
        self.root.parent.mkdir(parents=True, exist_ok=True)
        self._parent_fd = os.open(self.root.parent, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0))
        try:
            os.mkdir(self.root.name, mode=0o755, dir_fd=self._parent_fd)
        except FileExistsError as error:
            raise PIRGScoreError("PIRG score root already exists") from error
        self._fd = os.open(self.root.name, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0), dir_fd=self._parent_fd)
        info = os.fstat(self._fd)
        self._identity = (int(info.st_dev), int(info.st_ino))

    def _open(self) -> int:
        if self._fd is None or self._identity is None:
            raise PIRGScoreError("PIRG score root is not reserved")
        info = os.fstat(self._fd)
        if (int(info.st_dev), int(info.st_ino)) != self._identity:
            raise PIRGScoreError("PIRG score root identity drift")
        return self._fd

    def _write(self, name: str, body: bytes) -> str:
        _require("/" not in name and name.endswith(".json"), "PIRG score receipt name drift")
        root = self._open()
        digest = _digest(body)
        created: list[str] = []
        try:
            for leaf, payload in ((name, body), (name + ".sha256", f"{digest}  {name}\n".encode("ascii"))):
                fd = os.open(leaf, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444, dir_fd=root)
                created.append(leaf)
                try:
                    view = memoryview(payload)
                    while view:
                        written = os.write(fd, view)
                        if written <= 0:
                            raise PIRGScoreError("PIRG short immutable receipt write")
                        view = view[written:]
                    os.fsync(fd)
                    os.fchmod(fd, 0o444)
                finally:
                    os.close(fd)
            os.fsync(root)
        except BaseException:
            for leaf in reversed(created):
                try:
                    os.unlink(leaf, dir_fd=root)
                except OSError:
                    pass
            raise
        return digest

    def publish_json(self, name: str, payload: Mapping[str, object]) -> str:
        return self._write(name, _json(dict(payload)))

    def publish_group(self, payloads: Mapping[str, Mapping[str, object]]) -> Mapping[str, str]:
        names = tuple(payloads)
        if set(names) != {"score.json", "terminal.json"}:
            raise PIRGScoreError("PIRG score transaction topology drift")
        created: list[str] = []
        try:
            result: dict[str, str] = {}
            for name in ("score.json", "terminal.json"):
                result[name] = self._write(name, _json(dict(payloads[name])))
                created.extend((name, name + ".sha256"))
            return result
        except BaseException:
            if self._fd is not None:
                for name in reversed(created):
                    try:
                        os.unlink(name, dir_fd=self._fd)
                    except OSError:
                        pass
            raise

    def close(self) -> None:
        for descriptor in (self._fd, self._parent_fd):
            if descriptor is not None:
                os.close(descriptor)
        self._fd = self._parent_fd = None


_CAPABILITY_SEAL = object()


@dataclass(frozen=True)
class PIRGScoreCapability:
    identity_sha256: str
    _seal: object = field(default=_CAPABILITY_SEAL, repr=False, compare=False)


def _issue_root_review_capability(identity: PIRGScoreIdentity) -> PIRGScoreCapability:
    return PIRGScoreCapability(_digest(_json(validate_score_identity(identity))))


def _require_capability(value: object, *, identity: PIRGScoreIdentity) -> None:
    if not isinstance(value, PIRGScoreCapability) or value._seal is not _CAPABILITY_SEAL or value.identity_sha256 != _digest(_json(validate_score_identity(identity))):
        raise PIRGScoreError("PIRG score requires an in-process root-reviewed capability")


class PIRGScoreBackend(Protocol):
    def prepare(self, *, identity: PIRGScoreIdentity) -> None: ...
    def resolve_inputs(self, *, identity: PIRGScoreIdentity) -> InputAuthority: ...
    def score_cell(self, *, cell: ScoreCell, input_payload: Mapping[str, object]) -> CellEvidence: ...
    def final_reverify(self, *, identity: PIRGScoreIdentity) -> ScoreClosure: ...
    def progress(self) -> Mapping[str, object]: ...
    def close(self) -> None: ...


def _validated_complete_progress(value: Mapping[str, object]) -> dict[str, object]:
    """Require honest evaluation-open and zero-update facts at terminal time."""
    required = {
        "within_opened", "external_opened", "cuda_initialized", "target_optimizer_steps",
        "target_backward_calls", "target_update_calls", "formal_opened", "h1_opened",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise PIRGScoreError("PIRG score runtime-progress schema drift")
    if (
        value.get("within_opened") is not True or value.get("external_opened") is not True
        or type(value.get("cuda_initialized")) is not bool
        or value.get("target_optimizer_steps") != 0 or value.get("target_backward_calls") != 0
        or value.get("target_update_calls") != 0 or value.get("formal_opened") is not False
        or value.get("h1_opened") is not False
    ):
        raise PIRGScoreError("PIRG score runtime crossed an evaluation/update boundary")
    return dict(value)


def _terminal_payload(*, identity: PIRGScoreIdentity, attempt_sha256: str,
                      input_authority_sha256: str, score_sha256: str,
                      final_closure: Mapping[str, object], progress: Mapping[str, object]) -> dict[str, object]:
    closure = identity.closure.payload()
    if dict(final_closure) != closure:
        raise PIRGScoreError("PIRG score terminal launch/final closure drift")
    return {
        "schema": "posterior_identity_residual_gate_score_terminal_v1",
        "status": "PIRG_QUICK_SCORE_COMPLETE__NON_GOVERNING",
        "identity": validate_score_identity(identity),
        "attempt_sha256": _sha(attempt_sha256, "PIRG score terminal attempt SHA"),
        "input_authority_sha256": _sha(input_authority_sha256, "PIRG score terminal input-authority SHA"),
        "score_sha256": _sha(score_sha256, "PIRG score terminal score SHA"),
        "launch_closure": closure,
        "final_closure": closure,
        "execution_progress": _validated_complete_progress(progress),
        "formal_verdict": False,
        "score_terminal_transactional_group": True,
    }


def validate_score_terminal_payload(
    payload: Mapping[str, object], *, identity: PIRGScoreIdentity, score_sha256: str | None = None,
) -> dict[str, object]:
    required = {
        "schema", "status", "identity", "attempt_sha256", "input_authority_sha256", "score_sha256",
        "launch_closure", "final_closure", "execution_progress", "formal_verdict", "score_terminal_transactional_group",
    }
    if not isinstance(payload, Mapping) or set(payload) != required:
        raise PIRGScoreError("PIRG score terminal schema drift")
    rebuilt = _terminal_payload(
        identity=identity, attempt_sha256=payload.get("attempt_sha256"),
        input_authority_sha256=payload.get("input_authority_sha256"), score_sha256=payload.get("score_sha256"),
        final_closure=payload.get("final_closure"), progress=payload.get("execution_progress"),
    )
    if score_sha256 is not None and rebuilt["score_sha256"] != _sha(score_sha256, "expected PIRG score SHA"):
        raise PIRGScoreError("PIRG score terminal score digest binding drift")
    if dict(payload) != rebuilt:
        raise PIRGScoreError("PIRG score terminal semantic binding drift")
    return rebuilt


def run_score_lifecycle(*, root: Path, identity: PIRGScoreIdentity, backend: PIRGScoreBackend,
                        execution_capability: object) -> Mapping[str, str]:
    """Mockable future score lifecycle.  It never runs from the public CLI."""
    _require_capability(execution_capability, identity=identity)
    artifact = _ArtifactRoot(Path(root).absolute() / RESULT_ROOT_RELATIVE)
    attempt_sha: str | None = None
    input_sha: str | None = None
    terminal_written = False
    stage = "attempt"
    try:
        artifact.reserve()
        attempt = {
            "schema": "posterior_identity_residual_gate_score_attempt_v1",
            "identity": validate_score_identity(identity),
            "status": "ATTEMPT_RESERVED_BEFORE_EVALUATION_INPUTS",
            "within_opened": False, "external_opened": False, "formal_opened": False, "h1_opened": False,
            "target_optimizer_steps": 0, "target_backward_calls": 0, "target_update_calls": 0,
        }
        attempt_sha = artifact.publish_json("attempt.json", attempt)
        stage = "prepare"
        backend.prepare(identity=identity)
        stage = "input_authority"
        input_payload = backend.resolve_inputs(identity=identity).payload(identity=identity)
        validate_input_authority(input_payload, identity=identity)
        input_sha = artifact.publish_json("input_authority.json", input_payload)
        stage = "forwards"
        evidence = []
        for cell in score_matrix():
            item = backend.score_cell(cell=cell, input_payload=input_payload)
            if item.cell != cell:
                raise PIRGScoreError("PIRG scorer returned the wrong cell")
            evidence.append(item.payload(identity=identity, input_authority_sha256=input_sha))
        stage = "final_reverify"
        final = backend.final_reverify(identity=identity).payload()
        if final != identity.closure.payload():
            raise PIRGScoreError("PIRG score launch/final closure drift")
        payload = build_score_payload(identity=identity, input_payload=input_payload, evidence=evidence)
        validate_score_payload(payload, identity=identity, input_payload=input_payload)
        score_sha = _digest(_json(payload))
        terminal = _terminal_payload(
            identity=identity, attempt_sha256=attempt_sha, input_authority_sha256=input_sha,
            score_sha256=score_sha, final_closure=final, progress=backend.progress(),
        )
        validate_score_terminal_payload(terminal, identity=identity, score_sha256=score_sha)
        group = artifact.publish_group({"score.json": payload, "terminal.json": terminal})
        terminal_written = True
        return {"attempt_sha256": attempt_sha, "input_authority_sha256": input_sha, **group}
    except BaseException as error:
        if attempt_sha is not None and not terminal_written:
            failure = {
                "schema": "posterior_identity_residual_gate_score_failure_v1",
                "identity": validate_score_identity(identity), "attempt_sha256": attempt_sha,
                "input_authority_sha256": input_sha, "stage": stage,
                "progress": dict(backend.progress()), "terminal_published": False,
                "error_class": type(error).__name__, "error_sha256": _digest(repr(error).encode("utf-8")),
                "traceback_sha256": _digest("".join(traceback.format_exception(error)).encode("utf-8")),
            }
            try:
                artifact.publish_json("failure.json", failure)
            except BaseException:
                pass
        raise
    finally:
        try:
            backend.close()
        finally:
            artifact.close()


class NoLivePIRGScoreBackend:
    """Public dry guard: fails before any physical input or device action."""

    def prepare(self, *, identity: PIRGScoreIdentity) -> None:
        del identity
        raise PIRGScoreError("PIRG physical scorer is not available from the dry route")

    def resolve_inputs(self, *, identity: PIRGScoreIdentity) -> InputAuthority:
        raise AssertionError("dry PIRG score backend cannot resolve inputs")

    def score_cell(self, *, cell: ScoreCell, input_payload: Mapping[str, object]) -> CellEvidence:
        raise AssertionError("dry PIRG score backend cannot score")

    def final_reverify(self, *, identity: PIRGScoreIdentity) -> ScoreClosure:
        raise AssertionError("dry PIRG score backend cannot reverify")

    def progress(self) -> Mapping[str, object]:
        return {"within_opened": False, "external_opened": False, "cuda_initialized": False}

    def close(self) -> None:
        return None
